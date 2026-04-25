"""V2-22c-PRE.1.9 (D36) — per-cell coverage checks
`coverage_climate_cells` + `coverage_soil_cells`.

The existing global `spatial_temporal_coverage` check stays in the
banner per Appendix H two-zone rendering. PRE.1.9 adds two
NEW per-cell-scoped checks so the cockpit's Layer 1 fill rule and
Layer 3 dimension toggle (Coverage) have per-cell `affected_cells`
lists to consume.

Coverage:
- Climate-coverage failure surfaces every cell with no climate ts
  OR an empty records list.
- Soil-coverage failure surfaces every cell with no soil profile OR
  an empty layers list.
- All-coverage pass yields empty `affected_cells` + result='pass'.
- File-based (SARRA-Py) climate emits `info` with delegation note.
- Both check_ids appear in `_CATEGORY_FROM_PREFIX` allowlist with
  category `'coverage_per_cell'` (PRE.1.2 pivot integration).
- PRE.1.8 `violation_details` carry slim shape (no value / bounds).
"""
from __future__ import annotations

from datetime import date

import pytest

from prismpy.models.region import Region, BoundingBox
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.translators.base import UnifiedData
from prismpy.validators.scientific import (
    _check_coverage_climate_cells,
    _check_coverage_soil_cells,
)


def _make_grid(n_cells=3):
    cells = [
        GridCell(cell_id=i, lat=0.5, lon=0.5,
                 row=0, col=i, resolution="5arcmin")
        for i in range(n_cells)
    ]
    return SpatialGrid(
        bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        resolution="5arcmin", cells=cells,
    )


def _make_ts(records, *, cell_id=0):
    return ClimateTimeSeries(
        records=records, location_id=str(cell_id),
        lat=0.5, lon=0.5, source="test",
    )


def _make_record(*, day=1, tmax=25.0, tmin=15.0, precip=2.0, srad=20.0):
    return ClimateRecord(
        date=date(2020, 1, day),
        tmax=tmax, tmin=tmin, precip=precip, srad=srad,
    )


def _make_unified(*, grid_n=3, climate=None, soil=None):
    return UnifiedData(
        region=Region(
            name="t", country="t", country_iso3="TST",
            bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        ),
        grid=_make_grid(grid_n),
        climate=climate if climate is not None else {},
        soil=soil if soil is not None else {},
    )


class TestCoverageClimateCells:
    """V2-22c-PRE.1.9 — per-cell `coverage_climate_cells` check."""

    def test_full_coverage_yields_pass(self):
        """3 cells, climate present for each → pass."""
        climate = {
            i: _make_ts([_make_record()], cell_id=i)
            for i in range(3)
        }
        unified = _make_unified(climate=climate)
        check = _check_coverage_climate_cells(unified)
        assert check["check"] == "coverage_climate_cells"
        assert check["scope"] == "per_cell"
        assert check["result"] == "pass"
        assert check["details"]["affected_cells"] == []
        assert check["details"]["n_missing"] == 0
        assert check["details"]["n_total"] == 3

    def test_missing_cells_listed_in_affected_cells(self):
        """Cells 0 + 2 have climate; cell 1 doesn't → affected_cells=[1]."""
        climate = {
            0: _make_ts([_make_record()]),
            2: _make_ts([_make_record()]),
        }
        unified = _make_unified(climate=climate)
        check = _check_coverage_climate_cells(unified)
        assert check["result"] == "fail"
        assert check["details"]["affected_cells"] == [1]

    def test_empty_records_count_as_missing(self):
        """A ClimateTimeSeries with `records=[]` is treated as missing
        per D36 spec."""
        climate = {
            0: _make_ts([], cell_id=0),
            1: _make_ts([_make_record()], cell_id=1),
        }
        unified = _make_unified(grid_n=2, climate=climate)
        check = _check_coverage_climate_cells(unified)
        assert check["result"] == "fail"
        assert check["details"]["affected_cells"] == [0]

    def test_affected_cells_sorted_ascending(self):
        """Multiple missing cells → sorted ASC for deterministic
        JSON diffs."""
        climate = {
            5: _make_ts([_make_record()], cell_id=5),
        }
        # Grid is cells 0..5; cells 0..4 missing.
        unified = _make_unified(grid_n=6, climate=climate)
        check = _check_coverage_climate_cells(unified)
        assert check["details"]["affected_cells"] == [0, 1, 2, 3, 4]

    def test_violation_details_carry_slim_shape(self):
        """PRE.1.8 — coverage failures emit per-cell details with
        slim shape (no value/bounds — failure is absence)."""
        climate = {0: _make_ts([_make_record()])}
        unified = _make_unified(grid_n=2, climate=climate)
        check = _check_coverage_climate_cells(unified)
        details = check["details"]["violation_details"]
        assert len(details) == 1
        d = details[0]
        assert d["cell_id"] == 1
        assert d["variable"] == "climate"
        assert d["value"] is None
        assert d["unit"] is None
        assert d["bounds"] is None
        assert d["layer_idx"] is None


class TestCoverageClimateCellsFileBasedDelegation:
    """V2-22c-PRE.1.9 — file-based (SARRA-Py) climate delegates to
    PRE.2 sampling. Check emits `info` (NOT fail) with a clear
    delegation note. SARRA-Py per-cell coverage synthesis is V2-22d
    backlog (#11)."""

    def test_file_based_climate_emits_info_with_delegation_note(self):
        """File-based climate is detected via `_is_file_based_climate`
        which checks for path-dict shape."""
        climate = {
            "rainfall_dir": "/tmp/rainfall",
            "agera5_dir": "/tmp/agera5",
        }
        unified = _make_unified(climate=climate)
        check = _check_coverage_climate_cells(unified)
        assert check["result"] == "info"
        assert "delegated" in check["summary"].lower() or \
               "sarra" in check["summary"].lower()
        assert check["details"]["affected_cells"] == []


class TestCoverageSoilCells:
    """V2-22c-PRE.1.9 — per-cell `coverage_soil_cells` check.
    Same shape as climate-coverage but reads `unified_data.soil`."""

    def test_full_coverage_yields_pass(self):
        soil = {
            i: SoilProfile(
                profile_id=f"p{i}", lat=0.5, lon=0.5, source="iSDA",
                layers=[SoilLayer(
                    depth_top=0, depth_bottom=0.2,
                    sand=40, clay=30, silt=30,
                )],
            )
            for i in range(3)
        }
        unified = _make_unified(soil=soil)
        check = _check_coverage_soil_cells(unified)
        assert check["check"] == "coverage_soil_cells"
        assert check["scope"] == "per_cell"
        assert check["result"] == "pass"
        assert check["details"]["affected_cells"] == []

    def test_no_profile_counts_as_missing(self):
        soil = {
            0: SoilProfile(
                profile_id="p0", lat=0.5, lon=0.5, source="iSDA",
                layers=[SoilLayer(depth_top=0, depth_bottom=0.2,
                                  sand=40, clay=30, silt=30)],
            ),
        }
        # Grid has cells 0..2; cells 1, 2 are missing soil.
        unified = _make_unified(grid_n=3, soil=soil)
        check = _check_coverage_soil_cells(unified)
        assert check["result"] == "fail"
        assert check["details"]["affected_cells"] == [1, 2]

    def test_empty_layers_count_as_missing(self):
        """A SoilProfile with `layers=[]` is treated as missing."""
        soil = {
            0: SoilProfile(
                profile_id="p0", lat=0.5, lon=0.5, source="iSDA",
                layers=[],
            ),
            1: SoilProfile(
                profile_id="p1", lat=0.5, lon=0.5, source="iSDA",
                layers=[SoilLayer(depth_top=0, depth_bottom=0.2,
                                  sand=40, clay=30, silt=30)],
            ),
        }
        unified = _make_unified(grid_n=2, soil=soil)
        check = _check_coverage_soil_cells(unified)
        assert check["result"] == "fail"
        assert check["details"]["affected_cells"] == [0]

    def test_violation_details_use_soil_variable(self):
        """PRE.1.8 — coverage_soil_cells violation_details set
        `variable='soil'` (vs `'climate'` for the climate check)."""
        soil = {0: SoilProfile(
            profile_id="p0", lat=0.5, lon=0.5, source="iSDA",
            layers=[SoilLayer(depth_top=0, depth_bottom=0.2,
                              sand=40, clay=30, silt=30)],
        )}
        unified = _make_unified(grid_n=2, soil=soil)
        check = _check_coverage_soil_cells(unified)
        details = check["details"]["violation_details"]
        assert len(details) == 1
        assert details[0]["variable"] == "soil"
        assert details[0]["cell_id"] == 1


class TestPivotIntegration:
    """V2-22c-PRE.1.9 — confirm the new check_ids flow through the
    PRE.1.2 pivot allowlist into per-cell `failed_checks`. End-to-end
    integration via `_build_cell_summary`."""

    def test_coverage_check_ids_pivot_into_failed_checks(self):
        """Synthesize a validation_report with both new check_ids;
        the cell summary's per-cell `failed_checks` must include
        them under category `coverage_per_cell`."""
        from prismpy.pipeline.executor import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        unified = _make_unified(grid_n=2)
        report = {
            "validation_version": "2.0",
            "checks": [
                {
                    "check": "coverage_climate_cells",
                    "scope": "per_cell",
                    "result": "fail",
                    "details": {"affected_cells": [0]},
                },
                {
                    "check": "coverage_soil_cells",
                    "scope": "per_cell",
                    "result": "fail",
                    "details": {"affected_cells": [1]},
                },
            ],
        }
        out = pipeline._build_cell_summary(unified, report)
        cell_0_categories = {
            e["category"] for e in out["cells"][0]["failed_checks"]
        }
        cell_1_categories = {
            e["category"] for e in out["cells"][1]["failed_checks"]
        }
        assert cell_0_categories == {"coverage_per_cell"}
        assert cell_1_categories == {"coverage_per_cell"}
        # And the check_id is the canonical D36 form, not a synonym.
        assert any(
            e["check_id"] == "coverage_climate_cells"
            for e in out["cells"][0]["failed_checks"]
        )
        assert any(
            e["check_id"] == "coverage_soil_cells"
            for e in out["cells"][1]["failed_checks"]
        )
