"""Grid-resolution provenance correctness.

When the site-grid resolution became configurable (region.grid_resolution),
two provenance/warning sites in the executor still hardcoded "5-arcmin": the
grid decision log and the effective-resolution over-sampling warning. For a
30-arcmin (0.5°) run that mislabels the grid AND emits a false over-sampling
warning (the 0.5° grid matches NASA POWER's 0.5°). These tests pin the fix:
the decision + warning reflect the ACTUAL grid resolution, and the warning
fires only when the grid is genuinely finer than the source.
"""

from __future__ import annotations

import inspect
import logging

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
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.provenance.tracker import ProvenanceTracker

_WARN = "Effective-resolution warning"


def _pipeline() -> TranslationPipeline:
    cfg = ProjectConfig(
        project=ProjectInfo(name="grid_prov_unit", description="warning unit"),
        region=RegionConfig(
            name="Kano",
            country="Nigeria",
            country_iso3="NGA",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=5.75, miny=11.25, maxx=10.25, maxy=12.75
                ),
            ),
        ),
        crop=CropConfig(
            name="Cowpea",
            name_short="cow",
            variety="x",
            calendar=CropCalendarConfig(planting_doy=185, maturity_doy=280),
        ),
        temporal=TemporalConfig(start_year=2000, end_year=2001),
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir="outputs", structure="by_platform"),
    )
    pipe = TranslationPipeline(
        cfg,
        provenance=ProvenanceTracker(enabled=True, project_name="grid_prov_unit"),
    )
    # The effective-resolution warning records a decision against the "grid"
    # artifact, so it must exist first (mirrors the harmonize-stage ordering in
    # the real pipeline). With this started for BOTH tests, the only difference
    # between silent-vs-fires is the resolution argument → warning behaviour.
    pipe.provenance.start_artifact("grid", artifact_id="grid", stage="harmonize")
    return pipe


def test_warning_silent_when_grid_equals_source(caplog):
    # UC2: 30-arcmin (0.5°) grid on 0.5° NASA POWER → NOT finer → no warning.
    pipe = _pipeline()
    with caplog.at_level(logging.WARNING, logger="prismpy.pipeline.executor"):
        pipe._record_effective_resolution_warning(
            0.5, "30arcmin (~56 km)", ["NASA POWER"]
        )
    assert _WARN not in caplog.text


def test_warning_fires_when_grid_finer_than_source(caplog):
    # Existing UCs: 5-arcmin (~0.083°) grid on 0.5° NASA POWER → finer → warning.
    pipe = _pipeline()
    with caplog.at_level(logging.WARNING, logger="prismpy.pipeline.executor"):
        pipe._record_effective_resolution_warning(
            5.0 / 60.0, "5arcmin (~9 km)", ["NASA POWER"]
        )
    assert _WARN in caplog.text


def test_grid_gen_caller_passes_real_resolution_no_hardcode():
    # The grid-gen caller must pass the ACTUAL grid resolution to the decision
    # log + the warning (not a hardcoded 5-arcmin), so a 30-arcmin run is
    # labelled correctly and emits no false over-sampling warning. Red→green
    # vs the pre-fix hardcode.
    import prismpy.pipeline.executor as executor

    src = inspect.getsource(executor)
    assert "target_resolution_deg=grid.increment_deg" in src
    assert 'target_resolution_label=f"{grid.resolution}' in src
    assert 'f"{grid.resolution} uniform grid' in src
    assert "5-arcmin uniform grid" not in src
    assert 'target_resolution_label="5-arcmin (~9 km)"' not in src
