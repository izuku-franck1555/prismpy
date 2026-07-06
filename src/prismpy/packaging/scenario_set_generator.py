"""Canonical ISIMIP3b projection scenario-package generator (clone-and-swap).

Given a generated observed-climate BASELINE package and a (gcm × ssp ×
time_slice) matrix, this module produces one canonical PROJECTION package per
matrix cell. Each projection is identical to the baseline EXCEPT for
``weather/``, ``manifest.scenario``, and ``manifest.temporal`` — the
clone-and-swap invariant from :mod:`prismpy.models.scenario`.

Per matrix cell the driver:

1. ``discover_datasets`` + ``cached_cutout`` the four DSSAT-driving variables
   (the cutout cache key is crop-agnostic, so two crops at one AOI share the
   fetch);
2. bridges the cutouts to per-cell ``ClimateTimeSeries`` via
   :func:`prismpy.harmonize.isimip_to_climate.isimip_cutouts_to_climate_timeseries`;
3. clones the baseline package, swaps in the projection weather through the
   translator's public writer, rewrites the config years, and overwrites
   ``manifest.scenario`` with a typed projection block — refreshing the
   ``files[]`` checksums and disclosing the bridge's calendar / dewpoint
   policies under ``manifest.limitations``.

This is the canonical production path; it replaces the ad-hoc ``.local`` POCs
that bypassed the cutout primitive and drove private translator internals.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from prismpy.data_sources.isimip3b import (
    ISIMIP3bClient,
    cached_cutout,
    discover_datasets,
)
from prismpy.harmonize.climate_kind import ClimateKind
from prismpy.harmonize.isimip_to_climate import (
    CALENDAR_LIMITATION_KEY_FIELD,
    CALENDAR_LIMITATION_VALUE_FIELD,
    DEWPOINT_POLICY_FIELD,
    isimip_cutouts_to_climate_timeseries,
)
from prismpy.models.climate import ClimateTimeSeries
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.packaging.manifest import collect_files_with_checksums, save_manifest
from prismpy.packaging.readme_generator import generate_readme
from prismpy.packaging.scenario_helpers import (
    build_projection_scenario_block_for_period,
    rewrite_pythia_config_for_scenario,
)
from prismpy.translators.pythia.translator import PythiaTranslator

# The DSSAT-driving variables plus hurs: the bridge derives TDEW + RHUM from
# humidity, and the observed (AgERA5) baseline carries humidity, so fetching
# hurs keeps the baseline-vs-projection humidity treatment consistent rather
# than emitting RHUM/TDEW=-99 on the projection side. The cutout cache key is
# crop-agnostic, so these are fetched once per (gcm × ssp × slice) AOI.
_PROJECTION_VARIABLES: Tuple[str, ...] = ("tasmax", "tasmin", "pr", "rsds", "hurs")


@dataclass(frozen=True)
class ScenarioSetResult:
    """Outcome of a scenario-set generation run."""

    baseline_package: Path
    projection_packages: List[Path]
    matrix: List[Tuple[str, str, Tuple[int, int]]]


def generate_scenario_set(
    *,
    baseline_package: Union[str, Path],
    baseline_config: Any,
    aoi_bbox: Dict[str, float],
    gcms: Sequence[str],
    ssps: Sequence[str],
    time_slices: Sequence[Tuple[int, int]],
    region_name: str,
    crop_name: str,
    output_dir: Union[str, Path],
    client: Optional[ISIMIP3bClient] = None,
    cache_dir: Optional[Path] = None,
) -> ScenarioSetResult:
    """Generate the projection packages for a scenario set by clone-and-swap.

    Args:
        baseline_package: Path to a generated observed-climate baseline package
            (the clone source). Its ``shapes/sites.shp`` defines the cell grid.
        baseline_config: The baseline's ``ProjectConfig`` — reused to construct
            the translator for the weather writer (no ``__new__`` bypass).
        aoi_bbox: Cutout bbox dict with ``south`` / ``north`` / ``west`` /
            ``east`` keys (degrees, WGS84).
        gcms: ISIMIP3b GCM identifiers.
        ssps: Scenario identifiers (e.g. ``"ssp245"`` / ``"ssp585"``).
        time_slices: ``(start_year, end_year)`` tuples (registered CO₂ periods).
        region_name / crop_name: Carried into the projection scenario label.
        output_dir: Where the ``projection_*`` package dirs are written.
        client: Optional ISIMIP3bClient (constructed if omitted).
        cache_dir: Optional cutout cache root override.

    Returns:
        A :class:`ScenarioSetResult` listing the generated projection packages.
    """
    baseline_package = Path(baseline_package)
    output_dir = Path(output_dir)
    client = client or ISIMIP3bClient()

    grid = _grid_from_baseline_package(baseline_package)
    baseline_reference_label = _baseline_reference_label(baseline_package)
    planting_doy = _baseline_planting_doy(baseline_package)
    baseline_platform = _baseline_platform(baseline_package)

    projections: List[Path] = []
    matrix: List[Tuple[str, str, Tuple[int, int]]] = []
    for gcm in gcms:
        for ssp in ssps:
            for time_slice in time_slices:
                time_slice = (int(time_slice[0]), int(time_slice[1]))
                cutouts: Dict[str, Any] = {}
                for variable in _PROJECTION_VARIABLES:
                    dataset = discover_datasets(
                        client,
                        gcm=gcm,
                        scenario=ssp,
                        variable=variable,
                        time_slice=time_slice,
                    )
                    nc_path = cached_cutout(
                        client, dataset, aoi_bbox, cache_dir=cache_dir
                    )
                    cutouts[variable] = _open_cutout_variable(nc_path, variable)

                climate = isimip_cutouts_to_climate_timeseries(
                    cutouts, grid.cells, gcm_source=gcm
                )
                projection = assemble_projection_package(
                    baseline_package=baseline_package,
                    baseline_config=baseline_config,
                    projection_climate=climate,
                    grid=grid,
                    region_name=region_name,
                    crop_name=crop_name,
                    gcm_source=gcm,
                    rcp_or_ssp=ssp,
                    time_slice=time_slice,
                    baseline_reference_label=baseline_reference_label,
                    output_dir=output_dir,
                    planting_doy=planting_doy,
                )
                # ACEA reads the cloned OBSERVED climate pickles, not the ISIMIP
                # .WTH → an ACEA projection is a forced-CO₂ CO₂-SENSITIVITY, not a
                # GCM future. Reconcile temporal/bias/co2/label to that reality.
                # pythia/craft (.WTH-driven GCM projections) are left untouched.
                if baseline_platform == "acea":
                    finalize_acea_forced_co2_projection(projection, baseline_package)
                projections.append(projection)
                matrix.append((gcm, ssp, time_slice))

    return ScenarioSetResult(
        baseline_package=baseline_package,
        projection_packages=projections,
        matrix=matrix,
    )


def assemble_projection_package(
    *,
    baseline_package: Union[str, Path],
    baseline_config: Any,
    projection_climate: Dict[int, ClimateTimeSeries],
    grid: Optional[SpatialGrid],
    region_name: str,
    crop_name: str,
    gcm_source: str,
    rcp_or_ssp: str,
    time_slice: Tuple[int, int],
    baseline_reference_label: str,
    output_dir: Union[str, Path],
    planting_doy: Optional[int] = None,
) -> Path:
    """Assemble one canonical projection package by clone-and-swap (no network).

    Clones the baseline package and overwrites ONLY the climate-dependent
    surface: ``weather/``, ``manifest.scenario`` (a typed projection block),
    ``manifest.temporal`` (the slice years), the ``data_sources.climate``
    provenance, the refreshed ``files[]`` checksums, and the disclosed
    ``manifest.limitations`` (calendar + dewpoint policy from the bridge).
    Everything else (soil, raster, SNX template, shapefiles, use_case_config)
    is inherited from the baseline clone unchanged.
    """
    baseline_package = Path(baseline_package)
    output_dir = Path(output_dir)
    start_year, end_year = int(time_slice[0]), int(time_slice[1])

    # Crop to exactly the claimed slice: cached_cutout returns the decadal-union
    # (e.g. 2046-2065 resolves to 2041-2070 weather), so the weather must not
    # carry years outside what the manifest + config declare.
    projection_climate = _crop_climate_to_slice(
        projection_climate, start_year, end_year
    )

    projection_dir = output_dir / (
        f"projection_{gcm_source}_{rcp_or_ssp}_{start_year}-{end_year}"
    )
    if projection_dir.exists():
        shutil.rmtree(projection_dir)
    shutil.copytree(baseline_package, projection_dir)

    # Capture the baseline manifest (region / crop / project_name) before it is
    # overwritten, so the projection README can be regenerated from the
    # projection's own params.
    cloned_manifest = json.loads(
        (projection_dir / "manifest.json").read_text(encoding="utf-8")
    )

    # Swap weather: wipe the inherited baseline WTH then write the projection.
    weather_dir = projection_dir / "weather"
    if weather_dir.exists():
        for stale in weather_dir.glob("*.WTH"):
            stale.unlink()
    else:
        weather_dir.mkdir(parents=True)
    translator = PythiaTranslator(config=baseline_config, output_dir=projection_dir)
    translator.write_weather_files(
        projection_climate, climate_kind=ClimateKind.PROJECTION, grid=grid
    )

    # Align the cloned config's year fields to the projection slice so DSSAT
    # requests weather records for the projection years, not the baseline's.
    config_path = projection_dir / "config" / "pythia_config.json"
    if config_path.exists():
        rewrite_pythia_config_for_scenario(
            config_path,
            time_slice_start=start_year,
            time_slice_end=end_year,
            planting_doy=planting_doy,
        )

    scenario_block = build_projection_scenario_block_for_period(
        region_name=region_name,
        crop_name=crop_name,
        gcm_source=gcm_source,
        rcp_or_ssp=rcp_or_ssp,
        time_slice_start=start_year,
        time_slice_end=end_year,
        baseline_reference_label=baseline_reference_label,
    )

    # README is rewritten BEFORE the manifest so the manifest's files[]
    # checksum inventory captures the final (projection) README, not the
    # cloned baseline one.
    _rewrite_projection_readme(
        projection_dir,
        baseline_manifest=cloned_manifest,
        start_year=start_year,
        end_year=end_year,
        gcm_source=gcm_source,
        rcp_or_ssp=rcp_or_ssp,
    )
    _rewrite_projection_manifest(
        projection_dir,
        scenario_block=scenario_block,
        gcm_source=gcm_source,
        rcp_or_ssp=rcp_or_ssp,
        start_year=start_year,
        end_year=end_year,
        projection_climate=projection_climate,
    )
    return projection_dir


def _baseline_platform(baseline_package: Path) -> str:
    """Platform of a baseline package (``manifest.platform``)."""
    manifest = json.loads(
        (Path(baseline_package) / "manifest.json").read_text(encoding="utf-8")
    )
    return str(manifest.get("platform") or "")


def finalize_acea_forced_co2_projection(
    projection_dir: Path, baseline_package: Path
) -> None:
    """Reconcile an ACEA projection into an honest forced-CO₂ CO₂-SENSITIVITY.

    The ISIMIP generator path writes GCM weather as PYTHIA ``.WTH`` and clones
    the WHOLE baseline package. ACEA reads the cloned OBSERVED climate pickles
    (NOT the ``.WTH``), so an ACEA projection is observed-climate-HELD + CO₂
    forced to the scenario level — a pure CO₂-fertilization / WUE sensitivity,
    NOT a realized GCM future. The GCM-framed metadata the generator emits is
    then inconsistent with the ACEA data and breaks the run four ways; this
    reconciles all four (ACEA-only — pythia/craft, whose ``.WTH`` GCM climate
    IS the projection, are untouched):

      (a) ``manifest.temporal`` := baseline observed period. The GCM slice
          (e.g. 2046-2065) ≠ the cloned climate's observed years, so ACEA's
          ``_collect_annual_results`` year-filter (``year not in target_years``)
          drops EVERY result → "No results collected" → physical run FAILS.
      (b) ``scenario.bias_correction_method`` := ``"none"``. Observed climate
          carries no ISIMIP bias correction; a mismatch vs the baseline's
          ``"none"`` fails the UC2 comparison's consistent-method validation.
      (c) the elevated CO₂ is written into the engine's ``co2/`` file (the file
          ACEA actually reads). The generator sets ``manifest.scenario.co2_ppm``
          but never writes it to the co2 file → ACEA runs at the historical
          ~390 ppm → FLAT Δ=0 dose-response (a masquerade).
      (d) honest framing: a ``forcedCO2`` ``scenario_label`` + a disclosure that
          this is a CO₂-sensitivity (climate held), NOT a realized GCM future.
    """
    baseline_package = Path(baseline_package)
    b_manifest = json.loads(
        (baseline_package / "manifest.json").read_text(encoding="utf-8")
    )
    b_temporal = dict(b_manifest.get("temporal", {}))
    b_label = (b_manifest.get("scenario") or {}).get("scenario_label", "OBSERVED")
    b_start, b_end = int(b_temporal["start_year"]), int(b_temporal["end_year"])

    manifest_path = Path(projection_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario = dict(manifest.get("scenario", {}))
    co2 = float(scenario["co2_ppm"])
    ssp = scenario.get("rcp_or_ssp")
    ts_start, ts_end = scenario.get("time_slice_start"), scenario.get("time_slice_end")

    # (a) temporal := observed baseline period (matches the cloned ACEA climate)
    manifest["temporal"] = b_temporal

    # (b) no bias correction on the held observed climate
    scenario["bias_correction_method"] = "none"
    scenario["scenario_bias_correction_provenance"] = None

    # (d) honest CO₂-sensitivity framing (not a GCM future)
    scenario["scenario_label"] = (
        f"{b_label}_forcedCO2-{int(round(co2))}ppm_{ssp}_{ts_start}-{ts_end}"
    )
    scenario["scenario_kind"] = "co2_sensitivity_climate_held"
    scenario["disclosure"] = (
        f"CO2-SENSITIVITY (NOT a realized GCM future): climate HELD at the observed "
        f"{b_start}-{b_end} baseline; atmospheric CO2 forced to {int(round(co2))} ppm "
        f"(the {ssp} level at the {ts_start}-{ts_end} time-slice). Deltas reflect the "
        f"PURE CO2-fertilization / water-use-efficiency response, NOT GCM climate change."
    )
    manifest["scenario"] = scenario

    data_sources = dict(manifest.get("data_sources", {}))
    data_sources["climate"] = (
        f"OBSERVED baseline climate (held) + CO2 forced to {int(round(co2))}ppm "
        f"({ssp} {ts_start}-{ts_end} level) — CO2-sensitivity, not GCM downscaling"
    )
    manifest["data_sources"] = data_sources

    # limitations := baseline acea (observed climate). Drop the ISIMIP GCM-bridge
    # limitations _rewrite_projection_manifest injected (calendar_harmonization /
    # dewpoint_policy describe the .WTH GCM climate ACEA never reads).
    manifest["limitations"] = dict(b_manifest.get("limitations", {}))

    # (c) write the elevated CO₂ into the engine's co2/ file (constant over run years)
    co2_dir = Path(projection_dir) / "co2"
    co2_dir.mkdir(parents=True, exist_ok=True)
    for stale in co2_dir.glob("*.txt"):
        stale.unlink()
    lines = ["Year\tCO2_ppm"] + [f"{y}\t{co2:.2f}" for y in range(1980, b_end + 1)]
    (co2_dir / "GlobalHistoricalCO2_NOAA_1980_2020.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # (d) honest acea-framed README + drop the inherited GCM weather/*.WTH ACEA
    # never reads (a climate-held package must not ship GCM future weather).
    _write_acea_forced_co2_readme(
        Path(projection_dir),
        b_manifest,
        co2=co2,
        ssp=ssp,
        ts_start=ts_start,
        ts_end=ts_end,
        b_start=b_start,
        b_end=b_end,
    )
    weather_dir = Path(projection_dir) / "weather"
    if weather_dir.exists():
        shutil.rmtree(weather_dir)

    # refresh the per-file checksum inventory + summary after the co2 rewrite
    files = collect_files_with_checksums(projection_dir)
    manifest["files"] = files
    total_size = sum(entry["size_bytes"] for entry in files)
    manifest["summary"] = {
        "total_files": len(files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }
    save_manifest(manifest, manifest_path)


def _write_acea_forced_co2_readme(
    projection_dir: Path,
    baseline_manifest: Dict[str, Any],
    *,
    co2: float,
    ssp: Any,
    ts_start: Any,
    ts_end: Any,
    b_start: int,
    b_end: int,
) -> None:
    """Render an honest ACEA forced-CO2 README (not the PYTHIA clone template).

    ``generate_readme(platform="acea")`` emits the AquaCrop package layout (acea
    run cmd, ``climate/*.pckl`` + ``co2/`` inputs; no ``.WTH``/``.SOL``/``.SNX``).
    A CO2-sensitivity disclosure is then injected: climate HELD at the observed
    baseline, CO2 forced to the scenario level, deltas are CO2-fertilization/WUE,
    NOT a GCM future. Package facts (cell count, crop code, climate prefix) are
    read from disk so the README matches the actual on-disk package.
    """
    readme_path = projection_dir / "README.md"
    climate_dir = projection_dir / "climate"
    pickles = sorted(climate_dir.glob("*.pckl")) if climate_dir.exists() else []
    n_climate = len(pickles)
    climate_name = pickles[0].stem.rsplit("_", 1)[0] if pickles else ""
    cell_ids = [
        int(p.stem.rsplit("_", 1)[-1])
        for p in pickles
        if p.stem.rsplit("_", 1)[-1].isdigit()
    ]

    fao_code = ""
    ha_dir = projection_dir / "harvested_areas"
    if ha_dir.exists():
        sub = [p.name for p in ha_dir.iterdir() if p.is_dir()]
        if sub:
            fao_code = sub[0]

    region = baseline_manifest.get("region") or {}
    crop = baseline_manifest.get("crop") or {}
    ds = baseline_manifest.get("data_sources") or {}
    config = {
        "project_name": baseline_manifest.get("project_name", "projection"),
        "package_name": projection_dir.name,
        "region": region,
        "region_name": region.get("name", ""),
        "country": region.get("country", ""),
        "crop": crop,
        "crop_name": crop.get("name", ""),
        "start_year": b_start,
        "end_year": b_end,
        "temporal": {"start_year": b_start, "end_year": b_end},
        "n_cells": len(cell_ids) or n_climate,
        "n_climate_files": n_climate,
        "fao_code": fao_code or crop.get("fao_code", ""),
        "climate_name": climate_name
        or f"{str(region.get('name', '')).lower()}_nasapower",
        "climate_source": f"NASA POWER (observed {b_start}-{b_end}, HELD)",
        "soil_source": ds.get("soil", "iSDA S3 (30m)"),
        "spam_source": ds.get("harvested_areas", "Dummy (placeholder)"),
        "gaez_source": ds.get("crop_suitability", "FAO GAEZ v4 (auto-downloaded)"),
        "gridcells": cell_ids,
        "resolution": 0,
        "scenarios": [1],
        "clock_start": f"{b_start}/01/01",
        "clock_end": f"{b_end}/12/31",
    }
    generate_readme(readme_path, config, platform="acea")

    text = readme_path.read_text(encoding="utf-8")
    banner = (
        "\n> **CO2-SENSITIVITY EXPERIMENT (NOT a realized GCM future).**\n"
        f"> Climate is HELD at the observed {b_start}-{b_end} baseline (NASA POWER);\n"
        f"> atmospheric CO2 is FORCED to {co2:.0f} ppm (the {ssp} level at the\n"
        f"> {ts_start}-{ts_end} time-slice). Reported deltas are the pure\n"
        "> CO2-fertilization / water-use-efficiency response, NOT GCM climate change.\n"
        "> ACEA consumes the observed `climate/*.pckl` + the forced `co2/` file; it\n"
        "> reads NO GCM weather. (scenario_kind: co2_sensitivity_climate_held)\n"
    )
    text = text.replace(
        "**ACEA (AquaCrop) Input Data Package**\n",
        "**ACEA (AquaCrop) Input Data Package**\n" + banner,
        1,
    )
    text = text.replace(
        "| **CO2** | NOAA | Historical atmospheric CO2 levels |",
        f"| **CO2** | NOAA baseline, FORCED to {co2:.0f} ppm "
        f"| {ssp} {ts_start}-{ts_end} level (CO2-sensitivity; climate held) |",
        1,
    )
    readme_path.write_text(text, encoding="utf-8")


def _rewrite_projection_manifest(
    projection_dir: Path,
    *,
    scenario_block: Any,
    gcm_source: str,
    rcp_or_ssp: str,
    start_year: int,
    end_year: int,
    projection_climate: Dict[int, ClimateTimeSeries],
) -> None:
    """Surgically overwrite the cloned manifest's climate-dependent fields."""
    manifest_path = projection_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["scenario"] = scenario_block.model_dump()

    temporal = dict(manifest.get("temporal", {}))
    temporal["start_year"] = start_year
    temporal["end_year"] = end_year
    manifest["temporal"] = temporal

    data_sources = dict(manifest.get("data_sources", {}))
    data_sources["climate"] = (
        f"ISIMIP3b {gcm_source} {rcp_or_ssp} (ISIMIP3BASD bias-corrected)"
    )
    manifest["data_sources"] = data_sources

    # Disclose the bridge's calendar + dewpoint policies (uniform across cells).
    sample_metadata: Dict[str, Any] = {}
    for series in projection_climate.values():
        sample_metadata = getattr(series, "metadata", {}) or {}
        break
    limitations = dict(manifest.get("limitations", {}))
    calendar_key = sample_metadata.get(CALENDAR_LIMITATION_KEY_FIELD)
    if calendar_key:
        limitations[calendar_key] = sample_metadata.get(CALENDAR_LIMITATION_VALUE_FIELD)
    dewpoint_policy = sample_metadata.get(DEWPOINT_POLICY_FIELD)
    if dewpoint_policy:
        limitations["dewpoint_policy"] = dewpoint_policy
    if limitations:
        manifest["limitations"] = limitations

    # Refresh the per-file checksum inventory + summary after the weather swap.
    files = collect_files_with_checksums(projection_dir)
    manifest["files"] = files
    total_size = sum(entry["size_bytes"] for entry in files)
    manifest["summary"] = {
        "total_files": len(files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }

    save_manifest(manifest, manifest_path)


def _crop_climate_to_slice(
    climate: Dict[int, ClimateTimeSeries], start_year: int, end_year: int
) -> Dict[int, ClimateTimeSeries]:
    """Drop per-cell records whose year falls outside [start_year, end_year]."""
    cropped: Dict[int, ClimateTimeSeries] = {}
    for cell_id, series in climate.items():
        kept = [
            record
            for record in series.records
            if start_year <= record.year <= end_year
        ]
        cropped[cell_id] = ClimateTimeSeries(
            location_id=series.location_id,
            lat=series.lat,
            lon=series.lon,
            source=series.source,
            records=kept,
            elevation=series.elevation,
            metadata=series.metadata,
        )
    return cropped


def _rewrite_projection_readme(
    projection_dir: Path,
    *,
    baseline_manifest: Dict[str, Any],
    start_year: int,
    end_year: int,
    gcm_source: str,
    rcp_or_ssp: str,
) -> None:
    """Fully regenerate the projection README from the projection's own params.

    The cloned README advertises the baseline period + observed climate source
    (and a stale year-count). Rather than string-patch baked numbers, rebuild it
    from the projection's region/crop + slice years via the canonical
    ``generate_readme`` (so the period AND derived year-count are correct), and
    pass the projection climate source + citation so the template renders the
    ISIMIP3b provenance NATIVELY — no post-hoc string replacement, so no NASA
    citation key/url survives.
    """
    # ACEA forced-CO2 projections take their honest, acea-framed README from
    # finalize_acea_forced_co2_projection (climate held, CO2 forced): the pythia
    # clone template below would masquerade a CO2-sensitivity as a GCM future.
    if str(baseline_manifest.get("platform") or "") == "acea":
        return
    readme_path = projection_dir / "README.md"
    projection_source = f"ISIMIP3b {gcm_source} {rcp_or_ssp}"
    title_inner = f"ISIMIP3b bias-adjusted GCM climate ({gcm_source} {rcp_or_ssp})"
    climate_citation = (
        "@misc{isimip3b,\n"
        "  title = {" + title_inner + "},\n"
        "  url = {https://www.isimip.org/}\n"
        "}"
    )
    region = baseline_manifest.get("region") or {}
    crop = baseline_manifest.get("crop") or {}

    # Report the REAL package contents (one WTH per cell/site), counted from the
    # projection package on disk — otherwise generate_readme defaults the counts
    # to 0 and the README would falsely claim "0 grid points / 0 weather files".
    n_weather = len(list((projection_dir / "weather").glob("*.WTH")))
    eghr_dir = projection_dir / "eGHR"
    n_sol = len(list(eghr_dir.glob("*.SOL"))) if eghr_dir.exists() else 0
    readme_config = {
        "project_name": baseline_manifest.get("project_name", "projection"),
        "region": region,
        "region_name": region.get("name", ""),
        "country": region.get("country", ""),
        "gadm_level": region.get("gadm_level"),
        "crop": crop,
        "crop_name": crop.get("name", ""),
        "planting_doy": crop.get("planting_doy"),
        "maturity_doy": crop.get("maturity_doy"),
        "temporal": {"start_year": start_year, "end_year": end_year},
        "start_year": start_year,
        "end_year": end_year,
        "data_sources": {"climate": projection_source},
        # Real package contents (not the 0 defaults).
        "package_dir": projection_dir.name,
        "n_sites": n_weather,
        "n_weather_files": n_weather,
        "n_sol_files": n_sol,
        # Read natively by the pythia template: baselines default to NASA,
        # projections carry the ISIMIP3b source + citation.
        "climate_source": projection_source,
        "climate_citation": climate_citation,
    }
    generate_readme(readme_path, readme_config, platform="pythia")


def _open_cutout_variable(nc_path: Path, variable: str) -> Any:
    """Open one variable's DataArray from a cutout netCDF (loaded + closed)."""
    import xarray as xr

    with xr.open_dataset(nc_path) as dataset:
        if variable in dataset:
            return dataset[variable].load()
        data_vars = list(dataset.data_vars)
        if len(data_vars) == 1:
            return dataset[data_vars[0]].load()
        raise KeyError(
            f"Cutout {nc_path} has no variable {variable!r}; "
            f"data_vars={data_vars}."
        )


def _grid_from_baseline_package(baseline_package: Path) -> SpatialGrid:
    """Reconstruct the cell grid from the baseline's ``shapes/sites.shp``.

    The sites shapefile is the authoritative cell roster: its ``ID`` column is
    the sequential WTH-file id and its geometry centroids give each cell's
    ``(lat, lon)`` — the same coordinates the bridge samples the cutout at.
    """
    import geopandas as gpd

    sites_path = baseline_package / "shapes" / "sites.shp"
    if not sites_path.exists():
        raise FileNotFoundError(
            f"Baseline package {baseline_package} has no shapes/sites.shp; "
            "cannot reconstruct the cell grid for projection sampling."
        )
    frame = gpd.read_file(sites_path)
    cells: List[GridCell] = []
    for position, row in enumerate(frame.itertuples(index=False)):
        geometry = getattr(row, "geometry")
        centroid = geometry.centroid
        cell_id = int(getattr(row, "ID", position + 1))
        cells.append(
            GridCell(
                cell_id=cell_id,
                lat=float(centroid.y),
                lon=float(centroid.x),
                row=0,
                col=position,
                resolution="custom",
            )
        )
    return SpatialGrid(resolution="custom", cells=cells)


def _baseline_reference_label(baseline_package: Path) -> str:
    """Read the baseline package's ``manifest.scenario.scenario_label``."""
    manifest = _read_manifest(baseline_package)
    scenario = manifest.get("scenario") or {}
    label = scenario.get("scenario_label")
    if not label:
        raise ValueError(
            f"Baseline package {baseline_package} has no "
            "manifest.scenario.scenario_label; the projection cannot reference "
            "its baseline anchor."
        )
    return str(label)


def _baseline_planting_doy(baseline_package: Path) -> Optional[int]:
    """Read the baseline package's ``manifest.crop.planting_doy`` (if any)."""
    manifest = _read_manifest(baseline_package)
    crop = manifest.get("crop") or {}
    value = crop.get("planting_doy")
    return int(value) if value is not None else None


def _read_manifest(package: Path) -> Dict[str, Any]:
    manifest_path = Path(package) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json under {package}.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


__all__ = [
    "ScenarioSetResult",
    "generate_scenario_set",
    "assemble_projection_package",
]
