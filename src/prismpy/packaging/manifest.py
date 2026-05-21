"""
Manifest Generation for prismpy packages.

Creates manifest.json files with file inventory and SHA256 checksums
for reproducibility and integrity verification.

Manifest content is deterministic: no datetime stamps inside
``manifest.json`` content, no host-specific paths, no build-epoch
fields, no random-seed-dependent values. The same package, generated
twice on the same pinned prismpy code, produces byte-identical
``manifest.json`` files. Filesystem mtimes are still recorded by the
OS on the actual files — that's a filesystem concern, distinct from
manifest CONTENT.

Determinism details:

* ``json.dump`` is called with ``sort_keys=True`` and
  ``ensure_ascii=False``. Keys appear in canonical order; UTF-8 region
  names (e.g., ``"Ménoua"``) round-trip without ``\\uXXXX`` escapes.
* Output is written via ``Path.write_bytes`` so platforms with
  CRLF text-mode translation (Windows) emit LF newlines too. No BOM.
* Per-file entries omit the ``modified`` filesystem mtime — content
  vs. filesystem separation. SHA-256 + relative path + size identify
  the file unambiguously without tying the manifest to wall-clock
  metadata that drifts every regeneration.
* The top-level ``generated_at`` field is omitted. The package's
  reproducibility story rests on the SHA hashes, not on a wall-clock
  stamp.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

from prismpy.cells.canonical_cell_area_km2 import (
    DEG2_TO_KM2_DEFAULT,
    SpatialRef,
    canonical_cell_area_km2,
)


KNOWN_USE_CASE_NAMES: Tuple[str, ...] = (
    "yield_forecast",
    "climate_scenarios",
    "sowing_optimization",
    "drought_management",
    "soil_fertility",
    "livestock_feed",
)


PER_UC_GATES: Dict[str, FrozenSet[str]] = {
    "yield_forecast": frozenset({
        "n_years_gte_3",
        "manifest_cells_populated",
        "manifest_crops_populated",
        "forecast_or_analog_mode_resolved",
    }),
    "climate_scenarios": frozenset({
        "base_package_temporal_complete",
        "at_least_one_scenario_package_present",
        "scenario_packages_temporal_aligned",
    }),
    "sowing_optimization": frozenset({
        "n_years_gte_5",
        "crop_supported_per_platform",
        "manifest_adapter_capability_sowing_rule_default_present",
    }),
    "drought_management": frozenset({
        "n_years_gte_5",
        "n_years_gte_9_for_drought_freq_anomaly",
        "crop_supported_per_platform",
        "manifest_cells_populated",
    }),
    "soil_fertility": frozenset({
        "n_years_gte_3",
        "manifest_cells_populated",
        "crop_supported_per_platform",
        "fertilizer_scenarios_resolvable",
    }),
    "livestock_feed": frozenset({
        "n_years_gte_3",
        "manifest_cells_populated",
        "manifest_cell_areas_populated",
        "manifest_crops_populated",
    }),
}


ADVISORY_GATES: FrozenSet[str] = frozenset({
    "manifest_adapter_capability_sowing_rule_default_present",
    "n_years_gte_9_for_drought_freq_anomaly",
    "scenario_packages_temporal_aligned",
    "fertilizer_scenarios_resolvable",
})


_RESERVED_MANIFEST_KEYS: FrozenSet[str] = frozenset({
    "package_version",
    "generator",
    "generator_version",
    "platform",
    "project_name",
    "region",
    "crop",
    "crops",
    "temporal",
    "data_sources",
    "use_case_config",
    "summary",
    "cells",
    "cell_areas",
    "files",
    "uc_readiness",
    "validation_status",
    "scenario",
})


UC_CONFIG_KEY_TABLE: Dict[str, Tuple[str, ...]] = {
    "yield_forecast": (
        "cores",
        "target_year",
        "forecast_date",
        "max_runs",
        "cultivar_ids",
        "n_analogs",
    ),
    "climate_scenarios": ("cores", "years", "scenario_packages"),
    "sowing_optimization": (
        "cores",
        "sowing_window_start",
        "sowing_window_end",
        "sowing_stride",
        "sowing_rule",
        "subsistence_yield",
    ),
    "drought_management": (
        "cores",
        "risk_metric",
        "critical_window",
        "critical_window_start",
        "critical_window_end",
        "drought_threshold",
        "min_consecutive_days",
        "baseline_start",
        "baseline_end",
        "no_cell_day_output",
        "drought_threshold_grid",
    ),
    "soil_fertility": (
        "cores",
        "scenarios",
        "metric",
        "agg_level",
        "organic_decomp_rate",
        "enable_cost_benefit",
    ),
    "livestock_feed": (
        "cores",
        "harvestable_fraction",
        "rg_ratio",
        "dpi_residue_weight",
        "output_metric",
        "agg_level",
        "no_grid_output",
        "enable_livestock_demand",
        "feed_scenarios",
    ),
}


UC_READINESS_SCHEMA_VERSION = "1.1.1"


ADVISORY_FLAG_UC3_SOWING_RULE_DEFAULT_ABSENT = (
    "sowing_rule_default_absent:falls_back_to_manifest_default"
)
ADVISORY_FLAG_UC5_PYTHIA_PK_SILENT_NO_OP = (
    "pythia_pk_silent_no_op:fertility_stress_unmodeled_v3.1"
)
ADVISORY_FLAG_UC1_SHORTFALL_THRESHOLD_TEMPLATE = (
    "shortfall_threshold:viz_layer_default_{value}_kgha_{crop}_{region}"
)
ADVISORY_FLAG_UC4_SEVERITY_TIER = "severity_tier:viz_layer_thresholds_v1"
ADVISORY_FLAG_UC5_ROI_PRICES = "roi_prices:viz_layer_regional_defaults"
ADVISORY_FLAG_UC6_HERD_DENSITY = "herd_density:GLW_2020_default_supply_side_only"


_PLATFORM_SUPPORTED_CROPS: Dict[str, FrozenSet[str]] = {
    "sarra_py": frozenset({
        "maize", "sorghum", "millet", "cowpea", "rice", "groundnut",
        "Maize", "Sorghum", "Millet", "Cowpea", "Rice", "Groundnut",
    }),
    "pythia": frozenset({
        "maize", "sorghum", "millet", "cowpea", "rice", "groundnut",
        "Maize", "Sorghum", "Millet", "Cowpea", "Rice", "Groundnut",
    }),
    "acea": frozenset({
        "maize", "wheat", "rice", "sorghum", "millet",
        "Maize", "Wheat", "Rice", "Sorghum", "Millet",
    }),
    "craft": frozenset({
        "maize", "sorghum", "millet", "cowpea", "rice", "groundnut",
        "Maize", "Sorghum", "Millet", "Cowpea", "Rice", "Groundnut",
    }),
}




def derive_boundary_label(
    resolved_source: str,
    gadm_level: Optional[int],
) -> Tuple[str, str]:
    """Derive (label, description) for the boundary inclusion field
    on a package manifest and the corresponding README cells.

    The pipeline executor records the RESOLVED boundary source on
    the runtime ``Region`` object after any retrieve-stage fallback
    fires. Manifest writers must read that resolved value and pass
    it here, along with the configured GADM admin level. The level
    is only emitted when the resolved source is GADM; otherwise the
    label / description describe the actual on-disk boundary
    artifact (a manual bounding box, a shapefile, or — when the
    resolved value is the runtime alias ``manual_bounds`` produced
    by a GADM-failed-fallback at retrieve time — the same manual
    label as for an explicit manual configuration).

    Args:
        resolved_source: the runtime-resolved boundary source string.
            Expected values: ``"gadm"``, ``"manual"``,
            ``"manual_bounds"``, ``"shapefile"``. Unknown values
            raise ``ValueError`` so a future ``BoundarySource`` enum
            extension surfaces at sprint-time rather than as a
            silent fallthrough into the manual label.
        gadm_level: the configured GADM admin level. Honored only
            when ``resolved_source == "gadm"``; ignored (and may be
            ``None``) for every other source. ``None`` is also
            tolerated under GADM with a fallback to admin level 2 —
            the same default the BoundaryConfig schema uses.

    Returns:
        A ``(label, description)`` tuple suitable for the manifest's
        ``data_sources.boundaries`` field and the README's boundary
        row.

    Raises:
        ValueError: if ``resolved_source`` is not one of the four
            known values. The message names the offending source so
            the caller can map it to a new branch in this helper.
    """
    if resolved_source == "gadm":
        level = gadm_level if gadm_level is not None else 2
        return (
            f"GADM v4.1 admin level {level}",
            "Official administrative boundaries",
        )
    if resolved_source in ("manual", "manual_bounds"):
        return ("Bounding box", "Manual coordinate bounds")
    if resolved_source == "shapefile":
        return ("Custom shapefile", "User-provided boundary")
    raise ValueError(
        f"Unknown boundary source: {resolved_source!r}. "
        "Update derive_boundary_label() when adding a "
        "BoundarySource enum value."
    )


def compute_sha256(file_path: Union[str, Path]) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        file_path: Path to file

    Returns:
        SHA256 hex digest string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_file_info(
    file_path: Union[str, Path],
    base_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Get file information including size and checksum.

    Args:
        file_path: Path to file
        base_path: Base path for relative path computation

    Returns:
        Dictionary with file info
    """
    file_path = Path(file_path)
    rel_path = str(file_path.relative_to(base_path)) if base_path else str(file_path)

    # Filesystem mtime is intentionally NOT recorded — it's a
    # filesystem concern, not manifest content. Including it would
    # break byte-identical regeneration (mtime advances every write).
    return {
        "path": rel_path,
        "sha256": compute_sha256(file_path),
        "size_bytes": file_path.stat().st_size,
    }


def collect_files_with_checksums(
    directory: Union[str, Path],
    patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Collect all files in directory with their checksums.

    Args:
        directory: Root directory to scan
        patterns: Optional glob patterns to include (default: all files)
        exclude_patterns: Optional patterns to exclude

    Returns:
        List of file info dictionaries
    """
    directory = Path(directory)
    files_info = []

    exclude_patterns = exclude_patterns or [".DS_Store", "*.pyc", "__pycache__"]

    if patterns:
        all_files = []
        for pattern in patterns:
            all_files.extend(directory.rglob(pattern))
    else:
        all_files = [f for f in directory.rglob("*") if f.is_file()]

    for file_path in sorted(all_files):
        skip = False
        for exclude in exclude_patterns:
            if file_path.match(exclude):
                skip = True
                break

        if not skip:
            files_info.append(get_file_info(file_path, directory))

    return files_info


def _eval_gate_n_years_gte_3(project_config: Dict[str, Any]) -> bool:
    start = project_config.get("start_year")
    end = project_config.get("end_year")
    if start is None or end is None:
        return False
    return (int(end) - int(start) + 1) >= 3


def _eval_gate_n_years_gte_5(project_config: Dict[str, Any]) -> bool:
    start = project_config.get("start_year")
    end = project_config.get("end_year")
    if start is None or end is None:
        return False
    return (int(end) - int(start) + 1) >= 5


def _eval_gate_n_years_gte_9_for_drought_freq_anomaly(
    project_config: Dict[str, Any],
) -> bool:
    start = project_config.get("start_year")
    end = project_config.get("end_year")
    if start is None or end is None:
        return False
    return (int(end) - int(start) + 1) >= 9


def _eval_gate_manifest_cells_populated(manifest_so_far: Dict[str, Any]) -> bool:
    cells = manifest_so_far.get("cells")
    return bool(cells)


def _eval_gate_manifest_crops_populated(manifest_so_far: Dict[str, Any]) -> bool:
    crops = manifest_so_far.get("crops")
    if isinstance(crops, list) and len(crops) >= 1:
        return True
    legacy = manifest_so_far.get("crop")
    if isinstance(legacy, dict) and legacy.get("name"):
        return True
    return False


def _eval_gate_manifest_cell_areas_populated(
    manifest_so_far: Dict[str, Any],
) -> bool:
    cell_areas = manifest_so_far.get("cell_areas")
    return bool(cell_areas)


def _eval_gate_forecast_or_analog_mode_resolved(
    uc_config: Dict[str, Any],
) -> bool:
    if uc_config.get("forecast_date") is not None:
        return True
    if uc_config.get("target_year") is not None:
        return True
    return False


def _eval_gate_base_package_temporal_complete(
    project_config: Dict[str, Any],
) -> bool:
    return (
        project_config.get("start_year") is not None
        and project_config.get("end_year") is not None
    )


def _eval_gate_at_least_one_scenario_package_present(
    uc_config: Dict[str, Any],
) -> bool:
    scenario_packages = uc_config.get("scenario_packages") or []
    return isinstance(scenario_packages, list) and len(scenario_packages) >= 1


def _eval_gate_scenario_packages_temporal_aligned(
    uc_config: Dict[str, Any], project_config: Dict[str, Any],
) -> bool:
    # Honest-signal: emitter does not open scenario-package manifests
    # at packaging time to verify temporal alignment with the base
    # package. The gate is classified as ADVISORY (see ADVISORY_GATES);
    # failure surfaces an advisory_flag rather than gates_failed, and
    # downstream consumers (prism-runner UC2 dispatch) perform the
    # authoritative alignment check at execute time.
    return False


def _eval_gate_crop_supported_per_platform(
    manifest_so_far: Dict[str, Any], platform: str,
) -> bool:
    supported = _PLATFORM_SUPPORTED_CROPS.get(platform)
    if supported is None:
        return False
    crop_name = manifest_so_far.get("crop", {}).get("name") if isinstance(
        manifest_so_far.get("crop"), dict,
    ) else None
    if not crop_name:
        crops_list = manifest_so_far.get("crops") or []
        if crops_list and isinstance(crops_list[0], dict):
            crop_name = crops_list[0].get("crop_name") or crops_list[0].get("name")
    if not crop_name:
        return False
    return crop_name in supported


def _eval_gate_manifest_adapter_capability_sowing_rule_default_present(
    manifest_so_far: Dict[str, Any],
) -> bool:
    adapter_capability = manifest_so_far.get("adapter_capability")
    if not isinstance(adapter_capability, dict):
        return False
    return adapter_capability.get("sowing_rule_default") is not None


def _eval_gate_fertilizer_scenarios_resolvable(
    uc_config: Dict[str, Any],
) -> bool:
    # Honest-signal: the 5-priority resolver (preset / CSV / comma-list
    # / package CSV / builtin fallback) is exercised at dispatch time,
    # not at packaging time. The gate is classified as ADVISORY (see
    # ADVISORY_GATES); failure surfaces an advisory_flag rather than
    # gates_failed, and downstream consumers perform the authoritative
    # resolver check.
    return False


def _dispatch_gate(
    gate_name: str,
    project_config: Dict[str, Any],
    uc_config: Dict[str, Any],
    manifest_so_far: Dict[str, Any],
    platform: str,
) -> bool:
    if gate_name == "n_years_gte_3":
        return _eval_gate_n_years_gte_3(project_config)
    if gate_name == "n_years_gte_5":
        return _eval_gate_n_years_gte_5(project_config)
    if gate_name == "n_years_gte_9_for_drought_freq_anomaly":
        return _eval_gate_n_years_gte_9_for_drought_freq_anomaly(project_config)
    if gate_name == "manifest_cells_populated":
        return _eval_gate_manifest_cells_populated(manifest_so_far)
    if gate_name == "manifest_crops_populated":
        return _eval_gate_manifest_crops_populated(manifest_so_far)
    if gate_name == "manifest_cell_areas_populated":
        return _eval_gate_manifest_cell_areas_populated(manifest_so_far)
    if gate_name == "forecast_or_analog_mode_resolved":
        return _eval_gate_forecast_or_analog_mode_resolved(uc_config)
    if gate_name == "base_package_temporal_complete":
        return _eval_gate_base_package_temporal_complete(project_config)
    if gate_name == "at_least_one_scenario_package_present":
        return _eval_gate_at_least_one_scenario_package_present(uc_config)
    if gate_name == "scenario_packages_temporal_aligned":
        return _eval_gate_scenario_packages_temporal_aligned(uc_config, project_config)
    if gate_name == "crop_supported_per_platform":
        return _eval_gate_crop_supported_per_platform(manifest_so_far, platform)
    if gate_name == "manifest_adapter_capability_sowing_rule_default_present":
        return _eval_gate_manifest_adapter_capability_sowing_rule_default_present(manifest_so_far)
    if gate_name == "fertilizer_scenarios_resolvable":
        return _eval_gate_fertilizer_scenarios_resolvable(uc_config)
    return False


def canonical_use_case_config_serializer(
    project_config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Emit closed-world per-UC config dict.

    Iterates EMITTED UCs only (the keys declared in
    ``project_config['use_case_config']``); non-emitted UCs are absent
    from the returned dict. Each per-UC sub-dict uses the closed keyset
    from :data:`UC_CONFIG_KEY_TABLE`.
    """
    uc_config_source = project_config.get("use_case_config") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for uc_name in uc_config_source.keys():
        if uc_name not in KNOWN_USE_CASE_NAMES:
            continue
        allowed_keys = UC_CONFIG_KEY_TABLE[uc_name]
        uc_data = uc_config_source[uc_name] or {}
        out[uc_name] = {k: uc_data[k] for k in allowed_keys if k in uc_data}
    return out


def canonical_crops_emitter(
    project_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Emit widened crops list per §2.7.6.1 MUST-7.

    Always returns ``list[dict]``. When the UC1 sub-config declares
    ``cultivar_ids: list[str]``, emits one entry per cultivar. When
    absent / None / single, emits a 1-element list keyed off the
    primary crop entry in ``project_config``.
    """
    uc_config_source = project_config.get("use_case_config") or {}
    uc1_config = uc_config_source.get("yield_forecast") or {}
    cultivar_ids = uc1_config.get("cultivar_ids")

    crop_name = project_config.get("crop_name", "")
    planting_doy = project_config.get("planting_doy")
    maturity_doy = project_config.get("maturity_doy")

    if (
        isinstance(cultivar_ids, list)
        and len(cultivar_ids) >= 1
        and all(c for c in cultivar_ids)
    ):
        return [
            {
                "crop_name": crop_name,
                "planting_doy": planting_doy,
                "maturity_doy": maturity_doy,
                "cultivar_id": str(cid),
            }
            for cid in cultivar_ids
        ]

    primary_cultivar = project_config.get("cultivar_id", "") or ""
    return [{
        "crop_name": crop_name,
        "planting_doy": planting_doy,
        "maturity_doy": maturity_doy,
        "cultivar_id": str(primary_cultivar),
    }]


def canonical_uc_readiness_emitter(
    project_config: Dict[str, Any],
    platform: str,
    manifest_so_far: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Emit closed-world per-UC readiness dict per §2.7.6 schema.

    Iterates EMITTED UCs only (the keys declared in
    ``project_config['use_case_config']``); non-emitted UCs are absent
    from the returned dict (a UC absent from ``uc_readiness`` is the
    "package not built to serve this UC" signal — prismweb's confirm-
    card renders that UC's tab HIDDEN, not disabled-with-reason).

    For each emitted UC, evaluates the per-UC gates from
    :data:`PER_UC_GATES`, classifies into ``gates_passed`` /
    ``gates_failed`` (HARD) / ``advisory_flags`` (ADVISORY), and
    appends the spec'd contextual advisory_flags (UC3 sowing-rule
    default fallback; UC5 PYTHIA P+K silent no-op when the ACEA
    translator path triggered the flag; 4 display-guide flags per
    §2.7.7).
    """
    uc_config_source = project_config.get("use_case_config") or {}
    additional = project_config.get("_additional_metadata") or {}
    uc5_pythia_pk_triggered = bool(
        project_config.get("_acea_uc5_p_k_silent_no_op_triggered")
        or additional.get("_acea_uc5_p_k_silent_no_op_triggered")
    )

    out: Dict[str, Dict[str, Any]] = {}
    for uc_name in uc_config_source.keys():
        if uc_name not in KNOWN_USE_CASE_NAMES:
            continue

        uc_config = uc_config_source[uc_name] or {}
        applicable_gates = PER_UC_GATES[uc_name]

        gates_passed: List[str] = []
        gates_failed: List[Dict[str, Any]] = []
        advisory_flags: List[str] = []

        for gate_name in sorted(applicable_gates):
            passed = _dispatch_gate(
                gate_name, project_config, uc_config, manifest_so_far, platform,
            )
            if passed:
                gates_passed.append(gate_name)
                continue
            if gate_name in ADVISORY_GATES:
                advisory_flags.append(
                    f"{gate_name}_failed:advisory_fallback"
                )
            else:
                gates_failed.append({
                    "gate_id": gate_name,
                    "reason": f"hard gate {gate_name} did not pass at emit time",
                    "severity": "hard",
                })

        if uc_name == "sowing_optimization":
            advisory_flags.append(
                ADVISORY_FLAG_UC3_SOWING_RULE_DEFAULT_ABSENT,
            )

        # F-BP-5 cycle R1: CRAFT routes via the DSSAT engine (same as
        # PYTHIA per ``adapters/craft.py`` ``run_dssat_site`` reuse) and
        # carries the identical ``@N OPTIONS PHOSP=N POTAS=N`` hard-code
        # in the CRAFT-emitted .SNX template. The disclosure gate widens
        # from ``"pythia"``-only to the DSSAT-engine producer set
        # ``{"pythia", "craft"}``. The joint-flag literal name and the
        # downstream per-element flag names keep their ``pythia`` prefix
        # per Lesson #24 canonical-source-or-pin (stable identifiers);
        # the ``pythia`` prefix is a historical platform-origin label
        # inherited from the cycle-4 first-shipping platform. Engine-axis
        # rename (``*_on_pythia`` → ``*_on_dssat``) is the v3.2 cleanup.
        if (
            uc_name == "soil_fertility"
            and platform in {"pythia", "craft"}
            and uc5_pythia_pk_triggered
        ):
            advisory_flags.append(
                ADVISORY_FLAG_UC5_PYTHIA_PK_SILENT_NO_OP,
            )

        if uc_name == "yield_forecast":
            crop_label = "unknown"
            region_label = "unknown"
            crop_entry = manifest_so_far.get("crop")
            if isinstance(crop_entry, dict) and crop_entry.get("name"):
                crop_label = str(crop_entry["name"])
            region_entry = manifest_so_far.get("region")
            if isinstance(region_entry, dict) and region_entry.get("name"):
                region_label = str(region_entry["name"])
            advisory_flags.append(
                ADVISORY_FLAG_UC1_SHORTFALL_THRESHOLD_TEMPLATE.format(
                    value="default",
                    crop=crop_label,
                    region=region_label,
                )
            )
        elif uc_name == "drought_management":
            advisory_flags.append(ADVISORY_FLAG_UC4_SEVERITY_TIER)
        elif uc_name == "soil_fertility":
            advisory_flags.append(ADVISORY_FLAG_UC5_ROI_PRICES)
        elif uc_name == "livestock_feed":
            advisory_flags.append(ADVISORY_FLAG_UC6_HERD_DENSITY)

        if uc_name == "livestock_feed" and platform == "pythia":
            gates_failed.append({
                "gate_id": "platform_supports_uc",
                "reason": (
                    "UC6 livestock_feed not yet activated on PYTHIA "
                    "platform — UC6 hard-rejects PYTHIA dispatch per "
                    "use_cases/livestock_feed/use_case.py "
                    "supported_platforms list"
                ),
                "severity": "hard",
            })

        entry: Dict[str, Any] = {
            "schema_version": UC_READINESS_SCHEMA_VERSION,
            "gates_passed": sorted(gates_passed),
            "advisory_flags": advisory_flags,
        }
        if gates_failed:
            entry["gates_failed"] = gates_failed

        out[uc_name] = entry

    return out


def _extract_cells_with_centroids(
    package_dir: Path,
) -> Tuple[List[int], Dict[int, float]]:
    """Extract cell IDs and per-cell centroid latitudes from a package.

    Preferred source is ``cell_summary.json``; falls back to
    ``shapes/sites.shp`` via geopandas. Returns
    ``(cells_list, centroid_lat_by_cell_id)`` — ``cells_list`` is the
    ordered list of cell IDs (port of the legacy completion shim's
    extraction logic); ``centroid_lat_by_cell_id`` carries the per-
    cell centroid latitude in degrees when the source provides it
    (cell_summary.json dict entries with ``lat``/``latitude``/
    ``centroid_lat``/``centroid_latitude`` keys, or sites.shp geometry
    centroids). Cells without an available per-cell lat are absent
    from the dict; callers should fall back to a region-level default
    in that case.
    """
    cs_path = package_dir / "cell_summary.json"
    if cs_path.exists():
        try:
            cs = json.loads(cs_path.read_text())
            cells_raw = cs.get("cells") or []
            cells: List[int] = []
            lats: Dict[int, float] = {}
            for c in cells_raw:
                if isinstance(c, dict):
                    cid = c.get("id") or c.get("cell_id")
                    if cid is None:
                        continue
                    cid_int = int(cid)
                    cells.append(cid_int)
                    lat_value = (
                        c.get("lat")
                        or c.get("latitude")
                        or c.get("centroid_lat")
                        or c.get("centroid_latitude")
                    )
                    if lat_value is not None:
                        try:
                            lats[cid_int] = float(lat_value)
                        except (TypeError, ValueError):
                            pass
                else:
                    cells.append(int(c))
            return cells, lats
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    shp = package_dir / "shapes" / "sites.shp"
    if shp.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_file(shp)
            id_col = None
            for candidate in ("ID", "id", "CellID", "cell_id"):
                if candidate in gdf.columns:
                    id_col = candidate
                    break
            if id_col is None:
                return [], {}
            cells = [int(v) for v in gdf[id_col].tolist()]
            lats = {}
            for cid_raw, geom in zip(gdf[id_col], gdf.geometry):
                if geom is None or not hasattr(geom, "centroid"):
                    continue
                try:
                    lats[int(cid_raw)] = float(geom.centroid.y)
                except (TypeError, ValueError):
                    continue
            return cells, lats
        except Exception:
            return [], {}

    return [], {}


def _resolve_resolution_deg(
    project_config: Dict[str, Any], platform: str,
) -> float:
    """Normalize per-translator resolution conventions to ``resolution_deg``.

    Each translator declares package resolution differently:
    - ACEA: integer code at ``project_config['resolution']`` (``1`` =
      5-arcmin, ``0`` = 30-arcmin)
    - PYTHIA, SARRA-Py, CRAFT: string or float; convention varies

    Callers that want full control can populate
    ``project_config['resolution_deg']`` explicitly (always wins).
    Without explicit override + without a platform-specific code, the
    default is 5-arcmin (1/12 deg) — the dominant Sahel-band grid.
    """
    if "resolution_deg" in project_config:
        return float(project_config["resolution_deg"])
    if platform == "acea":
        acea_code = project_config.get("resolution")
        if acea_code == 0 or acea_code == "30arcmin":
            return 30.0 / 60.0
        if acea_code == 1 or acea_code == "5arcmin":
            return 5.0 / 60.0
    raw = project_config.get("resolution")
    if isinstance(raw, (int, float)) and float(raw) > 0:
        return float(raw)
    if isinstance(raw, str):
        if raw == "30arcmin":
            return 30.0 / 60.0
        if raw == "5arcmin":
            return 5.0 / 60.0
        try:
            parsed = float(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return 5.0 / 60.0


def _build_spatial_ref_for_package(
    project_config: Dict[str, Any],
    platform: str,
    centroid_lat_by_cell_id: Dict[int, float],
) -> SpatialRef:
    """Construct a :class:`SpatialRef` for the package.

    Resolution is normalized across the four translator conventions
    via :func:`_resolve_resolution_deg`. The ``cell_centroid_latitude``
    callable returns the per-cell centroid when known (extracted from
    cell_summary.json or sites.shp); otherwise returns a region-level
    fallback lat sourced from ``project_config['region_centroid_lat']``
    (or 0.0 — equator — as last-resort default; callers concerned with
    accuracy should supply per-cell lats via the cell_summary.json
    centroid keys).
    """
    resolution_deg = _resolve_resolution_deg(project_config, platform)
    fallback_lat = float(
        project_config.get("region_centroid_lat", 0.0),
    )

    def _lat_for_cell(cell_id: int) -> float:
        if cell_id in centroid_lat_by_cell_id:
            return centroid_lat_by_cell_id[cell_id]
        return fallback_lat

    return SpatialRef(
        resolution_deg=resolution_deg,
        cell_centroid_latitude=_lat_for_cell,
        deg2_to_km2=DEG2_TO_KM2_DEFAULT,
    )


def _filter_additional_metadata(
    additional_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Sanitize ``additional_metadata`` before the manifest update merge.

    Two protections:

    1. Private (``_``-prefixed) keys are STRIPPED before the merge.
       Callers use this prefix to pass private signals to the manifest
       emitter (e.g., ``_acea_uc5_p_k_silent_no_op_triggered``) without
       intending the keys to land in the published manifest. The strip
       prevents private-signal leak into the on-disk artifact.
    2. Reserved-key collisions RAISE. ``additional_metadata`` keys
       that overlap with the canonical emit set
       (``_RESERVED_MANIFEST_KEYS``) would silently overwrite the
       authoritative emit; the raise forces the caller to rename their
       extension key.
    """
    public: Dict[str, Any] = {}
    collisions: List[str] = []
    for k, v in additional_metadata.items():
        if k.startswith("_"):
            continue
        if k in _RESERVED_MANIFEST_KEYS:
            collisions.append(k)
            continue
        public[k] = v
    if collisions:
        raise ValueError(
            "additional_metadata may not overwrite canonical manifest "
            f"keys; reserved-key collisions: {sorted(collisions)}. "
            "Rename the extension keys."
        )
    return public


def create_manifest(
    package_dir: Union[str, Path],
    project_config: Dict[str, Any],
    platform: str = "sarra_py",
    additional_metadata: Optional[Dict[str, Any]] = None,
    scenario: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Create a complete manifest for a package.

    Args:
        package_dir: Root directory of the package
        project_config: Project configuration dictionary
        platform: Target platform name
        additional_metadata: Optional additional metadata to include
        scenario: Optional ``prismpy.models.scenario.ScenarioBlock``
            instance describing a paired baseline+projection scenario
            package. When provided, the model's serialized fields are
            embedded at the manifest's ``scenario`` key. When omitted,
            no ``scenario`` key is added — existing observed-climate
            manifests without a scenario continue to validate cleanly.

    Returns:
        Complete manifest dictionary
    """
    package_dir = Path(package_dir)

    # Collect all files
    files = collect_files_with_checksums(package_dir)

    # Compute summary statistics
    total_size = sum(f["size_bytes"] for f in files)

    use_case_config_emit = canonical_use_case_config_serializer(project_config)
    crops_list = canonical_crops_emitter(project_config)
    cells_list, centroid_lat_by_cell_id = _extract_cells_with_centroids(package_dir)
    spatial_ref = _build_spatial_ref_for_package(
        project_config, platform, centroid_lat_by_cell_id,
    )
    cell_areas_list = [
        canonical_cell_area_km2(cid, spatial_ref) for cid in cells_list
    ]

    manifest = {
        "package_version": "1.0",
        "generator": "prismpy",
        "generator_version": "1.0.0",
        "platform": platform,
        # ``generated_at`` is intentionally omitted — wall-clock stamps
        # break byte-identical regeneration (every run has a different
        # ``datetime.now()``). The package's reproducibility story
        # rests on the per-file SHA-256 + the manifest's own SHA-256.

        # Project info from config
        "project_name": project_config.get("project_name", "unknown"),

        "region": {
            "name": project_config.get("region_name", ""),
            "country": project_config.get("country", ""),
            # The default applies only when the translator omits
            # the key entirely; an explicit ``None`` from the
            # translator (the resolved-source-discriminator path
            # for non-GADM sources) is preserved by ``dict.get``
            # because the key is present. The default value 2
            # matches the BoundaryConfig schema default for GADM
            # configs, which is the only path that reaches this
            # branch via the omit semantics.
            "gadm_level": project_config.get("gadm_level", 2),
        },

        "crop": {
            "name": project_config.get("crop_name", ""),
            "planting_doy": project_config.get("planting_doy"),
            "maturity_doy": project_config.get("maturity_doy"),
        },

        "crops": crops_list,

        "temporal": {
            "start_year": project_config.get("start_year"),
            "end_year": project_config.get("end_year"),
            "spinup_years": project_config.get("spinup_years", 0),
        },

        "data_sources": project_config.get("data_sources", {}),

        "use_case_config": use_case_config_emit,

        "summary": {
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        },

        "cells": cells_list,

        "cell_areas": cell_areas_list,

        "files": files,

        "uc_readiness": {},

        "validation_status": "PENDING"
    }

    # Add bounds if available
    if "bounds_sarra_py" in project_config:
        manifest["region"]["bounds_sarra_py"] = project_config["bounds_sarra_py"]
    if "bounds_gis" in project_config:
        manifest["region"]["bounds_gis"] = project_config["bounds_gis"]

    if additional_metadata:
        merged_project_config = dict(project_config)
        merged_project_config["_additional_metadata"] = additional_metadata
        manifest["uc_readiness"] = canonical_uc_readiness_emitter(
            merged_project_config, platform, manifest,
        )
        public_extra = _filter_additional_metadata(additional_metadata)
        manifest.update(public_extra)
    else:
        manifest["uc_readiness"] = canonical_uc_readiness_emitter(
            project_config, platform, manifest,
        )

    # Optional scenario block per Sprint G AC-G-3. The block is OPTIONAL
    # outside scenario package contexts (codex H3 absorption); existing
    # observed-climate manifests do not carry a scenario key today and
    # continue to round-trip cleanly. When provided, the ScenarioBlock's
    # serialized form is embedded at the ``scenario`` key.
    if scenario is not None:
        # Late import to avoid pulling pydantic at every manifest call;
        # ``ScenarioBlock`` exposes ``model_dump`` (pydantic v2 API).
        if hasattr(scenario, "model_dump"):
            manifest["scenario"] = scenario.model_dump()
        elif isinstance(scenario, dict):
            # Allow raw dicts for callers that already serialized; the
            # validator will catch any shape errors when validate_manifest
            # runs.
            manifest["scenario"] = dict(scenario)
        else:
            raise TypeError(
                "scenario argument must be a ScenarioBlock instance or "
                "an already-serialized dict; got "
                f"{type(scenario).__name__}"
            )

    return manifest


def save_manifest(
    manifest: Dict[str, Any],
    output_path: Union[str, Path]
) -> Path:
    """
    Save manifest to JSON file with deterministic byte output.

    The serialization is fully canonicalized:

    * Top-level keys and every nested object are sorted lexicographically
      (``sort_keys=True``).
    * Non-ASCII characters round-trip natively (``ensure_ascii=False``).
      Region names like ``"Ménoua"`` appear unescaped, and the bytes are
      stable — no ``\\u00e9`` vs. ``é`` drift between Python versions.
    * Output is written via ``Path.write_bytes`` so platforms with
      automatic newline translation (Windows text mode) still emit LF
      newlines. UTF-8 encoding without BOM.

    Combined with ``create_manifest`` omitting wall-clock stamps and
    filesystem mtimes, the result is byte-identical across re-runs on
    identical inputs and identical pinned prismpy code.

    Args:
        manifest: Manifest dictionary
        output_path: Path to save manifest

    Returns:
        Path to saved manifest
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    body = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    output_path.write_bytes(body)

    return output_path


def validate_manifest(
    manifest_path: Union[str, Path],
    package_dir: Union[str, Path]
) -> Dict[str, Any]:
    """
    Validate a manifest against the actual package contents.

    Args:
        manifest_path: Path to manifest.json
        package_dir: Path to package directory

    Returns:
        Validation results dictionary

    Raises:
        pydantic.ValidationError: when the manifest carries a
            ``scenario`` key that does not match the ``ScenarioBlock``
            schema (Sprint G AC-G-3 + AC-G-10). Manifests without a
            ``scenario`` key validate normally — the schema is OPTIONAL
            outside scenario contexts per codex H3 absorption.
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    package_dir = Path(package_dir)

    # Sprint G AC-G-3 + AC-G-10: when the manifest carries a scenario
    # block, validate it against the ScenarioBlock schema. The model's
    # post-validators enforce time-slice ordering and the AC-G-10
    # co2_ppm/co2_ppm_provenance pairing. Manifests without a scenario
    # key skip this branch — the schema is optional outside scenario
    # contexts.
    scenario_payload = manifest.get("scenario")
    if scenario_payload is not None:
        # Late import keeps pydantic out of the manifest module's
        # import-time path for non-scenario callers.
        from prismpy.models.scenario import ScenarioBlock

        ScenarioBlock.model_validate(scenario_payload)

    results = {
        "valid": True,
        "checked_at": datetime.now().isoformat(),
        "missing_files": [],
        "checksum_mismatches": [],
        "extra_files": [],
    }

    # Track files listed in manifest
    manifest_files = {f["path"] for f in manifest.get("files", [])}

    # Check each file in manifest
    for file_info in manifest.get("files", []):
        file_path = package_dir / file_info["path"]

        if not file_path.exists():
            results["missing_files"].append(file_info["path"])
            results["valid"] = False
        else:
            actual_sha256 = compute_sha256(file_path)
            if actual_sha256 != file_info["sha256"]:
                results["checksum_mismatches"].append({
                    "path": file_info["path"],
                    "expected": file_info["sha256"],
                    "actual": actual_sha256
                })
                results["valid"] = False

    # Check for extra files not in manifest
    actual_files = collect_files_with_checksums(package_dir)
    for file_info in actual_files:
        if file_info["path"] not in manifest_files:
            results["extra_files"].append(file_info["path"])

    return results
