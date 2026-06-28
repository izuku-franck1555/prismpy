"""LIVE network-gated integration test for the scenario-set generator.

Drives the REAL ISIMIP3b path end-to-end for one (gcm × ssp × slice): real
``discover_datasets`` + ``cached_cutout`` (the hardened #80 primitive) → the
composition bridge → a canonical projection package via clone-and-swap.

Discipline (mirrors ``test_cached_cutout_live_multi_decade_real_fetch`` + the
#80 HDF5/single-process lesson): marked ``slow`` so the default
``-m 'not slow'`` FAST tier (and the bound-gen job) EXCLUDES it; additionally
skippable via ``PRISMPY_SKIP_LIVE_ISIMIP``. Single-process, real HDF5 reads,
under the pinned ``pandas<3``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import pytest

from prismpy.data_sources.isimip3b import (
    ISIMIP3bClient,
    cached_cutout,
    discover_datasets,
)
from prismpy.harmonize.isimip_to_climate import isimip_cutouts_to_climate_timeseries
from prismpy.models.scenario import ScenarioBlock
from prismpy.models.spatial import GridCell
from prismpy.packaging.scenario_set_generator import assemble_projection_package

pytestmark = pytest.mark.slow

_GCM = "gfdl-esm4"
_SSP = "ssp245"
_SLICE = (2046, 2065)
_VARIABLES = ("tasmax", "tasmin", "pr", "rsds", "hurs")
# Grid-ALIGNED bbox: ISIMIP3b is 0.5° (cell centers at k*0.5+0.25). This 1° box
# spans the 4 real cells around Kano (lat 11.75/12.25 × lon 8.25/8.75). A
# sub-grid box that contains no cell center yields an empty (failed) cutout.
_BBOX = {"south": 11.5, "north": 12.5, "west": 8.0, "east": 9.0}
_CELLS = [
    GridCell(cell_id=1, lat=12.25, lon=8.25, row=0, col=0, resolution="custom"),
    GridCell(cell_id=2, lat=11.75, lon=8.75, row=0, col=1, resolution="custom"),
]
_BASELINE_LABEL = "OBSERVED_KANO_COWPEA_2000-2020"


def _open_variable(nc_path: Path, variable: str):
    import xarray as xr

    with xr.open_dataset(nc_path) as dataset:
        if variable in dataset:
            return dataset[variable].load()
        return dataset[list(dataset.data_vars)[0]].load()


def _baseline_fixture(root: Path) -> Path:
    pkg = root / "baseline"
    (pkg / "weather").mkdir(parents=True)
    (pkg / "config").mkdir()
    for seq in (1, 2):
        (pkg / "weather" / f"{seq}.WTH").write_text("$WEATHER : BASELINE\n")
    (pkg / "config" / "pythia_config.json").write_text(
        json.dumps({"default_setup": {"sdate": "2005-01-01"}, "runs": []})
    )
    manifest = {
        "platform": "pythia",
        "project_name": "Kano_Cowpea",
        "region": {"name": "Kano", "country": "Nigeria", "gadm_level": 1},
        "crop": {"name": "Cowpea", "planting_doy": 182, "maturity_doy": 260},
        "temporal": {"start_year": 2000, "end_year": 2020},
        "data_sources": {"climate": "AgERA5"},
        "files": [],
        "cells": [1, 2],
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
    (pkg / "manifest.json").write_text(json.dumps(manifest))
    return pkg


def _build_project_config(output_dir: Path):
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

    return ProjectConfig(
        project=ProjectInfo(name="kano_cowpea", description="live"),
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
        # baseline years (≤2030): the config is the baseline's, reused only for
        # translator construction; projection years flow via the time_slice arg.
        temporal=TemporalConfig(start_year=2000, end_year=2020),
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir=str(output_dir), structure="by_platform"),
    )


@pytest.mark.skipif(
    bool(os.environ.get("PRISMPY_SKIP_LIVE_ISIMIP")),
    reason="Live ISIMIP3b fetch skipped (PRISMPY_SKIP_LIVE_ISIMIP set).",
)
def test_live_real_cutout_to_canonical_projection_package(tmp_path: Path) -> None:
    client = ISIMIP3bClient()

    cutouts: Dict[str, object] = {}
    for variable in _VARIABLES:
        dataset = discover_datasets(
            client, gcm=_GCM, scenario=_SSP, variable=variable, time_slice=_SLICE
        )
        # Persistent default cache (~/.cache/prismpy/isimip3b, 7-day TTL) so
        # re-runs of this slow test reuse the fetch instead of re-downloading.
        nc_path = cached_cutout(client, dataset, _BBOX, cache_dir=None)
        cutouts[variable] = _open_variable(nc_path, variable)

    climate = isimip_cutouts_to_climate_timeseries(
        cutouts, _CELLS, gcm_source=_GCM
    )

    # The bridge yields full multi-decade coverage (#80 integrity at climate level).
    series = climate[1]
    years = {record.year for record in series.records}
    assert _SLICE[0] in years and _SLICE[1] in years
    assert series.n_records > 5000  # ~20 years of daily records
    sample = series.records[0]
    assert -40.0 < sample.tmax < 60.0  # canonical Celsius, not Kelvin
    assert sample.precip >= 0.0
    assert 0.0 <= sample.srad < 40.0  # canonical MJ/m2/day
    # hurs fetched → real humidity + derived dewpoint, not RHUM/TDEW=-99.
    assert sample.rh is not None and 0.0 < sample.rh <= 100.0
    assert sample.tdew is not None and sample.tdew <= sample.tmean

    # Assemble a canonical projection package from the real climate.
    baseline = _baseline_fixture(tmp_path)
    projection_dir = assemble_projection_package(
        baseline_package=baseline,
        baseline_config=_build_project_config(tmp_path / "out"),
        projection_climate=climate,
        grid=None,
        region_name="Kano",
        crop_name="Cowpea",
        gcm_source=_GCM,
        rcp_or_ssp=_SSP,
        time_slice=_SLICE,
        baseline_reference_label=_BASELINE_LABEL,
        output_dir=tmp_path / "out",
    )
    manifest = json.loads((projection_dir / "manifest.json").read_text())
    block = ScenarioBlock(**manifest["scenario"])
    assert block.scenario_role == "projection"
    assert block.gcm_source == _GCM
    assert block.co2_ppm == 478.0
    wth_files = list((projection_dir / "weather").glob("*.WTH"))
    assert len(wth_files) == 2
    # The assembled weather is cropped to EXACTLY the slice — no decadal-union
    # over-fetch leaking years outside [2046, 2065] into the projection WTH.
    import re

    for wth in wth_files:
        years = {
            int(match.group(1)[:4])
            for line in wth.read_text().splitlines()
            if (match := re.match(r"\s*(\d{7})\b", line))
        }
        assert years and min(years) >= _SLICE[0] and max(years) <= _SLICE[1], (
            f"{wth.name} carries out-of-slice years {sorted(years)}"
        )
