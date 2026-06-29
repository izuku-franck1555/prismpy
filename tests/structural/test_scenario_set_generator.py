"""Network-free tests for the projection scenario-package generator.

Exercises :func:`prismpy.packaging.scenario_set_generator.assemble_projection_package`
(the clone-and-swap core) with a synthetic baseline fixture + synthetic
projection climate — no network, no real cutout, no HDF5. Asserts the
generated package is canonical: a valid projection ``ScenarioBlock``, the 9
fields the prism-runner UC2 processor requires, the golden scenario key set,
swapped weather, rewritten config years, refreshed checksums, and the disclosed
calendar / dewpoint limitations.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Dict

import pytest

from prismpy.config.schema import (
    BoundaryConfig,
    BoundarySource,
    CropCalendarConfig,
    CropConfig,
    ManualBoundsConfig,
    OutputConfig,
    Platform,
    ProjectConfig,
    ProjectInfo,
    RegionConfig,
    TemporalConfig,
)
from prismpy.harmonize.climate_kind import ClimateKind
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.scenario import ScenarioBlock
from prismpy.packaging.scenario_set_generator import assemble_projection_package
from prismpy.translators.pythia.translator import PythiaTranslator

_BASELINE_LABEL = "OBSERVED_KANO_COWPEA_2000-2020"
_SLICE = (2046, 2065)

# The 11 canonical scenario keys (9 UC2-required + 2 provenance) — the golden
# manifest shape a generated projection block must carry.
_GOLDEN_SCENARIO_KEYS = {
    "scenario_label",
    "scenario_role",
    "gcm_source",
    "rcp_or_ssp",
    "time_slice_start",
    "time_slice_end",
    "baseline_reference_label",
    "bias_correction_method",
    "co2_ppm",
    "co2_ppm_provenance",
    "scenario_bias_correction_provenance",
}
_UC2_REQUIRED_FIELDS = _GOLDEN_SCENARIO_KEYS - {
    "co2_ppm_provenance",
    "scenario_bias_correction_provenance",
}

_DEWPOINT_POLICY = "FAO-56 Tetens dewpoint from hurs; propagate-missing on bad RH."
_CALENDAR_KEY = "calendar_noleap_dropped_feb29"


def _project_config(output_dir: Path) -> ProjectConfig:
    return ProjectConfig(
        project=ProjectInfo(name="kano_cowpea", description="orchestrator test"),
        region=RegionConfig(
            name="Kano",
            country="Nigeria",
            country_iso3="NGA",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=8.3, miny=11.9, maxx=8.7, maxy=12.2
                ),
            ),
        ),
        crop=CropConfig(
            name="Cowpea",
            name_short="cwp",
            variety="Medium",
            calendar=CropCalendarConfig(planting_doy=182, maturity_doy=260),
        ),
        temporal=TemporalConfig(start_year=2000, end_year=2020, spinup_years=2),
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir=str(output_dir), structure="by_platform"),
    )


def _baseline_fixture(root: Path) -> Path:
    pkg = root / "baseline"
    (pkg / "weather").mkdir(parents=True)
    (pkg / "config").mkdir()
    (pkg / "weather" / "1.WTH").write_text("$WEATHER : BASELINE PLACEHOLDER\n")
    (pkg / "weather" / "2.WTH").write_text("$WEATHER : BASELINE PLACEHOLDER\n")
    (pkg / "config" / "pythia_config.json").write_text(
        json.dumps(
            {
                "default_setup": {
                    "sdate": "2005-01-01",
                    "pfrst": "2005-06-15",
                    "plast": "2005-07-15",
                },
                "runs": [{"startYear": 2005, "nyers": 1}],
            },
            indent=2,
        )
    )
    manifest = {
        "package_version": "1.0",
        "generator": "prismpy",
        "platform": "pythia",
        "project_name": "Kano_Cowpea",
        "region": {"name": "Kano", "country": "Nigeria", "gadm_level": 1},
        "crop": {"name": "Cowpea", "planting_doy": 182, "maturity_doy": 260},
        "temporal": {"start_year": 2000, "end_year": 2020, "spinup_years": 2},
        "data_sources": {"climate": "AgERA5", "soil": "eGHR"},
        "summary": {"total_files": 2, "total_size_bytes": 0, "total_size_mb": 0.0},
        "cells": [1, 2],
        "files": [],
        "use_case_config": {"climate_scenarios": {"enabled": True}},
        "uc_readiness": {},
        "validation_status": "PENDING",
        "scenario": {
            "scenario_label": _BASELINE_LABEL,
            "scenario_role": "baseline",
            "gcm_source": "observed_AgERA5",
            "rcp_or_ssp": "historical",
            "time_slice_start": 2000,
            "time_slice_end": 2020,
            "baseline_reference_label": _BASELINE_LABEL,
            "bias_correction_method": "none",
            "co2_ppm": 400.0,
            "co2_ppm_provenance": "NOAA Mauna Loa annual mean",
        },
    }
    (pkg / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (pkg / "README.md").write_text(
        "# Kano_Cowpea\n\n"
        "| **Period** | 2000-2020 (21 years) |\n\n"
        "| Weather | NASA POWER | 2000-2020 | Daily SRAD, TMAX, ... |\n"
    )
    return pkg


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wth_years(wth_path: Path) -> set:
    years = set()
    for line in wth_path.read_text().splitlines():
        match = re.match(r"\s*(\d{7})\b", line)
        if match:
            years.add(int(match.group(1)[:4]))
    return years


def _wth_rhum_values(wth_path: Path) -> list:
    values = []
    for line in wth_path.read_text().splitlines():
        match = re.match(r"\s*\d{7}\s+(.*)", line)
        if match:
            cols = match.group(1).split()
            if len(cols) >= 6:
                values.append(float(cols[5]))  # SRAD TMAX TMIN RAIN TDEW RHUM
    return values


def _projection_climate() -> Dict[int, ClimateTimeSeries]:
    climate: Dict[int, ClimateTimeSeries] = {}
    for cell_id in (1, 2):
        records = [
            ClimateRecord(
                date=date(2046, 6, day),
                tmax=32.0,
                tmin=22.0,
                precip=3.0,
                srad=19.0,
                rh=55.0,
                tdew=None,
            )
            for day in (1, 2, 3)
        ]
        climate[cell_id] = ClimateTimeSeries(
            location_id=cell_id,
            lat=12.0,
            lon=8.4,
            source="ISIMIP3b_gfdl-esm4",
            records=records,
            metadata={
                "calendar_limitation_key": _CALENDAR_KEY,
                "calendar_limitation_value": "noleap source; Feb 29 missing.",
                "dewpoint_policy": _DEWPOINT_POLICY,
            },
        )
    return climate


def _assemble(tmp_path: Path, *, time_slice=_SLICE, climate=None) -> Path:
    baseline = _baseline_fixture(tmp_path)
    return assemble_projection_package(
        baseline_package=baseline,
        baseline_config=_project_config(tmp_path / "out"),
        projection_climate=_projection_climate() if climate is None else climate,
        grid=None,
        region_name="Kano",
        crop_name="Cowpea",
        gcm_source="gfdl-esm4",
        rcp_or_ssp="ssp245",
        time_slice=time_slice,
        baseline_reference_label=_BASELINE_LABEL,
        output_dir=tmp_path / "out",
    )


def test_assemble_produces_canonical_projection_manifest(tmp_path: Path) -> None:
    projection_dir = _assemble(tmp_path)
    manifest = json.loads((projection_dir / "manifest.json").read_text())
    scenario = manifest["scenario"]

    # The generated scenario block re-validates through the canonical model.
    block = ScenarioBlock(**scenario)
    assert block.scenario_role == "projection"
    assert block.gcm_source == "gfdl-esm4"
    assert block.rcp_or_ssp == "ssp245"
    assert block.co2_ppm == 478.0
    assert block.baseline_reference_label == _BASELINE_LABEL

    # Golden scenario key set + the 9 UC2-required fields, all non-null.
    assert set(scenario.keys()) == _GOLDEN_SCENARIO_KEYS
    for field in _UC2_REQUIRED_FIELDS:
        assert scenario[field] is not None

    # temporal swapped to the slice; climate provenance updated; checksums refreshed.
    assert manifest["temporal"]["start_year"] == 2046
    assert manifest["temporal"]["end_year"] == 2065
    assert "ISIMIP3b" in manifest["data_sources"]["climate"]
    assert manifest["files"], "files[] checksums were not refreshed after swap"
    assert manifest["summary"]["total_files"] == len(manifest["files"])
    # Every recorded checksum matches the file actually on disk.
    for entry in manifest["files"]:
        on_disk = projection_dir / entry["path"]
        assert on_disk.exists(), f"manifest lists missing file {entry['path']}"
        assert _sha256(on_disk) == entry["sha256"], f"SHA drift for {entry['path']}"

    # Bridge limitations disclosed.
    assert manifest["limitations"][_CALENDAR_KEY]
    assert manifest["limitations"]["dewpoint_policy"] == _DEWPOINT_POLICY


def test_weather_cropped_to_exact_slice(tmp_path: Path) -> None:
    # Climate with records OUTSIDE [2046, 2065] (the decadal-union over-fetch):
    # the projection weather must carry NO years beyond the claimed slice.
    climate: Dict[int, ClimateTimeSeries] = {}
    for cell_id in (1, 2):
        records = [
            ClimateRecord(
                date=date(year, 6, 15),
                tmax=32.0,
                tmin=22.0,
                precip=3.0,
                srad=19.0,
                rh=55.0,
                tdew=None,
            )
            for year in (2044, 2046, 2055, 2065, 2067)
        ]
        climate[cell_id] = ClimateTimeSeries(
            location_id=cell_id,
            lat=12.0,
            lon=8.4,
            source="ISIMIP3b_gfdl-esm4",
            records=records,
            metadata={},
        )
    projection_dir = _assemble(tmp_path, climate=climate)
    for wth in (projection_dir / "weather").glob("*.WTH"):
        years = _wth_years(wth)
        assert years, f"{wth.name} has no data rows"
        assert min(years) >= 2046 and max(years) <= 2065, (
            f"{wth.name} carries out-of-slice years {sorted(years)}"
        )
        assert 2044 not in years and 2067 not in years


def test_projection_readme_regenerated_off_baseline_claims(tmp_path: Path) -> None:
    readme = (_assemble(tmp_path) / "README.md").read_text()
    assert "2046-2065" in readme  # projection slice
    assert "(20 years)" in readme  # derived year-count is the slice's, not 21
    assert "21 years" not in readme  # the stale baseline year-count is gone
    assert "ISIMIP3b gfdl-esm4 ssp245" in readme  # projection source
    assert "2000-2020" not in readme  # stale baseline period gone
    # ZERO NASA anywhere (label, citation key, AND url) — provenance is ISIMIP.
    assert "nasa" not in readme.lower()
    assert "isimip" in readme.lower()  # the ISIMIP citation is present


def test_projection_readme_reports_real_package_counts(tmp_path: Path) -> None:
    # The README must report the ACTUAL package contents, not the 0 defaults.
    projection_dir = _assemble(tmp_path)
    readme = (projection_dir / "README.md").read_text()
    n_weather = len(list((projection_dir / "weather").glob("*.WTH")))
    assert n_weather == 2  # the fixture baseline has 2 cells → 2 projection WTH
    assert "0 grid points" not in readme
    match = re.search(r"(\d+) grid points", readme)
    assert match is not None and int(match.group(1)) == n_weather


def test_humidity_records_emit_real_rhum_not_sentinel(tmp_path: Path) -> None:
    # The synthetic climate carries rh=55; the projection WTH RHUM column must
    # be real humidity, not the MISDAT -99 sentinel (the hurs-fetch consistency).
    wth = sorted((_assemble(tmp_path) / "weather").glob("*.WTH"))[0]
    rhum = _wth_rhum_values(wth)
    assert rhum, "no RHUM data rows parsed"
    assert all(value != -99.0 for value in rhum), f"RHUM carries MISDAT: {rhum}"
    assert all(0.0 < value <= 100.0 for value in rhum)


def test_weather_swapped_not_baseline(tmp_path: Path) -> None:
    projection_dir = _assemble(tmp_path)
    wth_files = sorted((projection_dir / "weather").glob("*.WTH"))
    assert len(wth_files) == 2
    for wth in wth_files:
        text = wth.read_text()
        assert "BASELINE PLACEHOLDER" not in text
        assert "@" in text  # a real DSSAT WTH header, not the placeholder


def test_config_years_rewritten_to_slice(tmp_path: Path) -> None:
    projection_dir = _assemble(tmp_path)
    config = json.loads(
        (projection_dir / "config" / "pythia_config.json").read_text()
    )
    assert config["default_setup"]["sdate"].startswith("2046")
    assert config["runs"][0]["startYear"] == 2046


def test_public_write_weather_files_seam(tmp_path: Path) -> None:
    # The orchestrator uses this PUBLIC seam (not the private writer / __new__).
    (tmp_path / "weather").mkdir()
    translator = PythiaTranslator(
        config=_project_config(tmp_path), output_dir=tmp_path
    )
    written = translator.write_weather_files(
        _projection_climate(), climate_kind=ClimateKind.PROJECTION, grid=None
    )
    assert len(written) == 2
    assert all(p.suffix == ".WTH" and p.exists() for p in written)


def test_unregistered_slice_raises(tmp_path: Path) -> None:
    # An unregistered (ssp, slice) has no canonical CO2 → fail loud, no package.
    with pytest.raises(ValueError):
        _assemble(tmp_path, time_slice=(2030, 2049))
