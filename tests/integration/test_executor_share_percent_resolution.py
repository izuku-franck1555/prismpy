"""Executor Stage-3 share_percent must use the grid's resolution-derived
half-cell-edge, not a hardcoded 5-arcmin one.

Drives ``TranslationPipeline._execute_harmonize`` end-to-end (same harness
style as ``test_f_r_executor_behavioral.py``) so the REAL Stage-3 cell-box
+ share_percent + min_share_percent path is exercised.

Bug (pre-fix, executor.py Stage 3): ``halfres = (5.0 / 60.0) / 2.0`` is the
5-arcmin half-cell-edge. It builds the ``cell_box`` used for boundary
inclusion (``cell_box.intersects(admin_geom)``), share_percent
(``intersection.area / cell_box.area``) and the min_share_percent trim. On a
30-arcmin grid (``grid.increment_deg = 0.5``) the box is 6x too small per
side, so inclusion + share + trimming are all wrong for cells straddling an
irregular admin polygon. Fix: ``halfres = grid.increment_deg / 2.0``.

The 30-arcmin test computes the admitted-cell count independently with BOTH
the correct half-width (0.25) and the buggy hardcoded one (0.0417), asserts
they differ (the scenario genuinely discriminates), then asserts the
executor matches the CORRECT count -> RED on the hardcoded value, GREEN once
resolution-derived. The 5-arcmin test pins no-regression (0.0417 is correct
there, so the count is unchanged by the fix).
"""
from __future__ import annotations

from unittest import TestCase

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
from prismpy.models.region import BoundingBox, Region
from prismpy.models.spatial import SpatialGrid
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.provenance.tracker import ProvenanceTracker

# Irregular diamond whose 45-degree edges slice 30-arcmin cells at varied
# fractions, so the full 0.5-deg box and the tiny 5-arcmin box disagree on
# which cells clear the share threshold.
_POLY_WKT = (
    "POLYGON(("
    "-5.0 12.9, "  # north
    "-4.1 12.0, "  # east
    "-5.0 11.1, "  # south
    "-5.9 12.0, "  # west
    "-5.0 12.9"    # close
    "))"
)
_BBOX = BoundingBox(minx=-6.0, miny=11.0, maxx=-4.0, maxy=13.0)
# 8 cells are 32%-covered by the full 0.5-deg box but 0% by the tiny 5-arcmin
# box (their centre sits just outside the diamond) — a 25% threshold admits
# them only when the half-width is resolution-derived: correct 12 vs buggy 4.
_MIN_SHARE = 25.0


def _shapely_or_skip(testcase: TestCase):
    try:
        from shapely import wkt  # noqa: F401
    except ImportError:
        testcase.skipTest("shapely not available — skip executor share probe")


def _make_pipeline(*, resolution: str, min_share_percent: float) -> TranslationPipeline:
    """Minimal pipeline whose BoundaryConfig carries bbox_intersects + the
    share threshold, at the requested grid resolution."""
    cfg = ProjectConfig(
        project=ProjectInfo(
            name='executor_share_percent_resolution',
            description='Stage-3 share_percent resolution-derived half-width',
        ),
        region=RegionConfig(
            name='Koutiala', country='Mali', country_iso3='MLI',
            grid_resolution=resolution,
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=-6.0, miny=11.0, maxx=-4.0, maxy=13.0,
                ),
                inclusion_rule='bbox_intersects',
                min_share_percent=min_share_percent,
            ),
        ),
        crop=CropConfig(
            name='Maize', name_short='mai', variety='Medium-duration',
            calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285),
        ),
        temporal=TemporalConfig(start_year=2015, end_year=2020, spinup_years=2),
        targets=[Platform.CRAFT],
        output=OutputConfig(base_dir='outputs', structure='by_platform'),
    )
    return TranslationPipeline(
        cfg,
        provenance=ProvenanceTracker(
            enabled=True, project_name='executor_share_percent_resolution',
        ),
    )


def _expected_admitted(resolution: str, polygon, min_share: float, halfres: float) -> int:
    """Replicate the executor's Stage-3 admission (cell_box.intersects +
    share = intersection.area/cell_box.area + min_share trim) for a given
    half-cell-edge, so the test can compare the executor against the
    correct and the buggy half-widths independently."""
    from shapely.geometry import box as shp_box
    grid = SpatialGrid.from_bounds(
        _BBOX, resolution=resolution, clip_geometry=None, exclude_cells=None,
    )
    admitted = 0
    for cell in grid.cells:
        cell_box = shp_box(
            cell.lon - halfres, cell.lat - halfres,
            cell.lon + halfres, cell.lat + halfres,
        )
        if not cell_box.intersects(polygon):
            continue
        share = (cell_box.intersection(polygon).area / cell_box.area) * 100.0
        if share < min_share:
            continue
        admitted += 1
    return admitted


def _drive_admitted(pipe: TranslationPipeline) -> int:
    region = Region(
        name='Koutiala', country='Mali', country_iso3='MLI',
        bounds=_BBOX, geometry_wkt=_POLY_WKT,
    )
    # Downstream stages fail (no real data), but Stage 5 fires first and
    # populates provenance.boundary — same pattern as the behavioral tests.
    pipe._execute_harmonize({'region': region})
    boundary = pipe.provenance.record.boundary
    assert boundary is not None, "executor must populate provenance.boundary at Stage 5"
    return boundary['n_cells_admitted']


class TestExecutorSharePercentResolutionHalfWidth(TestCase):
    def setUp(self):
        _shapely_or_skip(self)

    def test_30arcmin_admission_uses_resolution_half_width(self):
        from shapely import wkt
        polygon = wkt.loads(_POLY_WKT)

        correct = _expected_admitted('30arcmin', polygon, _MIN_SHARE, halfres=(30 / 60) / 2)
        buggy = _expected_admitted('30arcmin', polygon, _MIN_SHARE, halfres=(5 / 60) / 2)
        self.assertNotEqual(
            correct, buggy,
            "test scenario is not discriminating — pick a polygon/threshold "
            "where the 0.25-deg box and the 0.0417-deg box admit different "
            "cell counts, else the test cannot catch the bug.",
        )

        admitted = _drive_admitted(_make_pipeline(
            resolution='30arcmin', min_share_percent=_MIN_SHARE,
        ))
        self.assertEqual(
            admitted, correct,
            f"executor admitted {admitted} 30-arcmin cells; the correct "
            f"resolution-derived half-width (0.25) admits {correct}, the "
            f"hardcoded 5-arcmin half-width (0.0417) admits {buggy}. The "
            f"executor must use grid.increment_deg/2.",
        )

    def test_5arcmin_admission_unchanged_no_regression(self):
        from shapely import wkt
        polygon = wkt.loads(_POLY_WKT)

        # At 5-arcmin the correct half-width IS 0.0417 (== the pre-fix
        # hardcode), so the fix leaves this count unchanged.
        correct = _expected_admitted('5arcmin', polygon, _MIN_SHARE, halfres=(5 / 60) / 2)
        admitted = _drive_admitted(_make_pipeline(
            resolution='5arcmin', min_share_percent=_MIN_SHARE,
        ))
        self.assertEqual(
            admitted, correct,
            f"5-arcmin admission regressed: executor {admitted} vs correct "
            f"{correct}. The fix must not change 5-arcmin behavior.",
        )
