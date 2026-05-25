"""Two-tier value-range bounds: PHYSICAL (defect) vs PLAUSIBILITY (warning).

The value-range checks now classify each value against two bands:

* PLAUSIBILITY (``CLIMATE_RANGES`` / ``SOIL_RANGES``) — outside is a
  WARNING (atypical-but-real, acknowledgeable-with-evidence). This is
  the existing band; ``out_of_range_count`` keeps its meaning.
* PHYSICAL (``PHYSICAL_CLIMATE_RANGES`` / ``PHYSICAL_SOIL_RANGES`` /
  ``TEXTURE_SUM_PHYSICAL``) — outside is a DEFECT (physically
  impossible, e.g. a nodata sentinel leaked through scaling). A defect
  is recorded as an additive ``defect=True`` marker, NOT a new
  ``result`` value, and is threaded per-cell so the cockpit ack path
  can refuse it (no-data-cooking: an impossible value is never
  acknowledgeable).

These tests pin the producer honesty floor:
- defect markers fire only for physical violations, never for
  plausibility-only warnings;
- the check ``result`` stays in the existing vocab (defect is a marker,
  not a result enum) — so the rollup never invents ``result="defect"``;
- the per-cell pivot stamps ``defect`` onto every ``failed_checks``
  entry and ``cell_failed_check_details`` row;
- ANY one field's physical violation condemns the whole cell.
"""
from __future__ import annotations

import pytest

from prismpy.models.region import Region, BoundingBox
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.translators.base import UnifiedData
from prismpy.validators.scientific import _check_value_ranges


_ALLOWED_RESULTS = {"pass", "warning", "fail", "info", "unavailable"}


def _region():
    return Region(
        name="t", country="t", country_iso3="TST",
        bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
    )


def _unified_soil(soil_dict):
    return UnifiedData(region=_region(), climate={}, soil=soil_dict)


def _unified_climate(climate_dict):
    return UnifiedData(region=_region(), climate=climate_dict, soil={})


def _by_check(checks, name):
    for c in checks:
        if c["check"] == name:
            return c
    return None


def _one_layer_profile(profile_id, **layer_kwargs):
    return SoilProfile(
        profile_id=profile_id, lat=0.5, lon=0.5, source="iSDA",
        layers=[SoilLayer(depth_top=0.0, depth_bottom=0.2, **layer_kwargs)],
    )


class TestSoilTwoTierDefect:
    """pH / OC / BD carry both tiers; the physical band catches
    impossible values, the plausibility band catches atypical ones."""

    def test_ph_physical_violation_is_defect(self):
        # pH 25.5 is the canonical nodata-leak value (255 x 0.1) —
        # outside the physical [0, 14] band.
        unified = _unified_soil({3: _one_layer_profile("p3", sand=40, clay=30, silt=30, ph=25.5)})
        checks = _check_value_ranges(unified)
        ph = _by_check(checks, "value_range_soil_ph")
        assert ph is not None
        assert ph["details"]["defect"] is True
        assert ph["details"]["defect_count"] == 1
        # Marker, not a new result value.
        assert ph["result"] == "warning"
        vd = ph["details"]["violation_details"]
        assert len(vd) == 1 and vd[0]["defect"] is True

    def test_ph_plausibility_violation_is_warning_not_defect(self):
        # pH 10.0 is inside physical [0, 14] but outside plausibility
        # [3.5, 9.5] — atypical (strongly sodic/calcareous), acknowledgeable.
        unified = _unified_soil({4: _one_layer_profile("p4", sand=40, clay=30, silt=30, ph=10.0)})
        checks = _check_value_ranges(unified)
        ph = _by_check(checks, "value_range_soil_ph")
        assert ph is not None
        assert ph["result"] == "warning"
        assert ph["details"]["defect"] is False
        assert ph["details"]["defect_count"] == 0
        vd = ph["details"]["violation_details"]
        assert len(vd) == 1 and vd[0]["defect"] is False

    def test_ph_in_plausibility_band_passes(self):
        unified = _unified_soil({5: _one_layer_profile("p5", sand=40, clay=30, silt=30, ph=6.5)})
        checks = _check_value_ranges(unified)
        ph = _by_check(checks, "value_range_soil_ph")
        assert ph is not None
        assert ph["result"] == "pass"
        assert ph["details"]["defect"] is False

    def test_single_tier_fraction_outside_100_is_defect(self):
        # sand/clay/silt are single-tier: physical == plausibility == [0,100].
        # A leaked 255 fraction is a defect.
        unified = _unified_soil({6: _one_layer_profile("p6", sand=255, clay=30, silt=30)})
        checks = _check_value_ranges(unified)
        sand = _by_check(checks, "value_range_soil_sand")
        assert sand is not None
        assert sand["details"]["defect"] is True
        assert sand["details"]["defect_count"] == 1

    def test_bulk_density_plausibility_is_warning_not_defect(self):
        # BD 2.5 < physical ceiling 2.65 → warning, not defect (the
        # BD-leak value is subsumed by the per-cell rule, not its own tier).
        unified = _unified_soil({7: _one_layer_profile("p7", sand=40, clay=30, silt=30, bulk_density=2.5)})
        checks = _check_value_ranges(unified)
        bd = _by_check(checks, "value_range_soil_bulk_density")
        assert bd is not None
        assert bd["result"] == "warning"
        assert bd["details"]["defect"] is False


class TestTextureSumThreeTier:
    """texture-sum: pass [95,105] / warning [50,95)∪(105,150] /
    defect <50 ∨ >150."""

    def test_sum_in_band_passes(self):
        unified = _unified_soil({1: _one_layer_profile("p1", sand=40, clay=30, silt=30)})
        ts = _by_check(_check_value_ranges(unified), "value_range_texture_sum")
        assert ts["result"] == "pass"
        assert ts["details"]["defect"] is False
        assert ts["details"]["n_defect_layers"] == 0

    def test_moderate_deviation_is_warning_not_defect(self):
        # 50 + 40 + 30 = 120 — outside [95,105] but inside [50,150].
        unified = _unified_soil({2: _one_layer_profile("p2", sand=50, clay=40, silt=30)})
        ts = _by_check(_check_value_ranges(unified), "value_range_texture_sum")
        assert ts["result"] == "warning"
        assert ts["details"]["defect"] is False
        assert ts["details"]["n_defect_layers"] == 0

    def test_leaked_fraction_sum_is_defect(self):
        # 255 + 30 + 30 = 315 — the primary leak tripwire (> 150).
        unified = _unified_soil({8: _one_layer_profile("p8", sand=255, clay=30, silt=30)})
        ts = _by_check(_check_value_ranges(unified), "value_range_texture_sum")
        assert ts["details"]["defect"] is True
        assert ts["details"]["n_defect_layers"] == 1
        # Uniform key parity with the per-variable value-range checks.
        assert ts["details"]["defect_count"] == 1
        vd = [d for d in ts["details"]["violation_details"]]
        assert vd and vd[0]["defect"] is True


class TestClimateTwoTierDefect:
    """Climate value-range is two-tier: existing CLIMATE_RANGES is the
    plausibility/warning band; PHYSICAL_CLIMATE_RANGES is the defect
    band. (Region-specific bounds remain a separate plausibility check
    and return info for the universal fallback, so the global band must
    stay as the warning tier — not be replaced by physical.)"""

    def _climate(self, tmax):
        from datetime import date
        return _unified_climate({
            0: ClimateTimeSeries(
                location_id=0, lat=0.5, lon=0.5, source="TEST",
                records=[ClimateRecord(
                    date=date(2015, 1, 1), tmax=tmax, tmin=20.0,
                    precip=2.0, srad=20.0,
                )],
            ),
        })

    def test_tmax_plausibility_is_warning_not_defect(self):
        # 62 is outside plausibility [-50, 60] but inside physical [-70, 65].
        ck = _by_check(_check_value_ranges(self._climate(62.0)), "value_range_tmax")
        assert ck is not None
        assert ck["result"] == "warning"
        assert ck["details"]["defect"] is False

    def test_tmax_physical_violation_is_defect(self):
        # 70 is outside physical [-70, 65] — impossible daily max.
        ck = _by_check(_check_value_ranges(self._climate(70.0)), "value_range_tmax")
        assert ck is not None
        assert ck["details"]["defect"] is True
        assert ck["details"]["defect_count"] == 1
        assert ck["result"] == "warning"  # marker, not a result enum


class TestProducerResultVocabulary:
    """Defect is an additive marker — the validator never invents a new
    ``result="defect"`` value (durable #27 two-vocabulary drift)."""

    def test_no_check_emits_defect_as_a_result_value(self):
        unified = _unified_soil({9: _one_layer_profile("p9", sand=255, clay=255, silt=255, ph=25.5)})
        for c in _check_value_ranges(unified):
            assert c["result"] in _ALLOWED_RESULTS, (
                f"{c['check']} emitted result={c['result']!r}; defect must "
                f"ride as a marker, not a result enum"
            )


# --------------------------------------------------------------------------
# Executor per-cell defect threading
# --------------------------------------------------------------------------


def _make_pipeline():
    return TranslationPipeline.__new__(TranslationPipeline)


def _make_grid(n_cells=2):
    cells = [
        GridCell(cell_id=i, lat=0.5 + i * 0.01, lon=0.5 + i * 0.01,
                 row=0, col=i, resolution="5arcmin")
        for i in range(n_cells)
    ]
    return SpatialGrid(
        bounds=BoundingBox(minx=0.0, miny=0.0, maxx=1.0, maxy=1.0),
        resolution="5arcmin", cells=cells,
    )


def _full_unified(n_cells=2):
    from datetime import date
    grid = _make_grid(n_cells)
    climate, soil = {}, {}
    for cid in range(n_cells):
        climate[cid] = ClimateTimeSeries(
            location_id=cid, lat=0.5, lon=0.5, source="TEST",
            records=[ClimateRecord(date=date(2015, 1, 1), tmax=30.0,
                                   tmin=20.0, precip=2.0, srad=20.0)],
        )
        soil[cid] = SoilProfile(
            profile_id=f"p{cid}", lat=0.5, lon=0.5, source="iSDA",
            layers=[SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=40, clay=30)],
        )
    return UnifiedData(region=_region(), grid=grid, climate=climate, soil=soil)


class TestExecutorPerCellDefectThreading:
    """The per-cell pivot reads each violation_details entry's
    ``defect`` flag: a cell with ANY defect violation is stamped
    ``defect=True`` on its failed_checks entry (non-acknowledgeable);
    a warning-only cell is stamped ``defect=False``."""

    def _report(self):
        # Cell 0 carries a physical-defect pH; cell 1 carries only a
        # plausibility warning — same check, different per-cell verdict.
        return {
            "validation_version": "2.1",
            "checks": [{
                "check": "value_range_soil_ph",
                "scope": "per_layer",
                "result": "warning",
                "details": {
                    "defect": True,
                    "affected_cells": [(0, 0), (1, 0)],
                    "violation_details": [
                        {"cell_id": 0, "layer_idx": 0, "variable": "ph",
                         "date": None, "value": 25.5, "unit": "",
                         "bounds": [3.5, 9.5], "defect": True},
                        {"cell_id": 1, "layer_idx": 0, "variable": "ph",
                         "date": None, "value": 10.0, "unit": "",
                         "bounds": [3.5, 9.5], "defect": False},
                    ],
                },
            }],
        }

    def test_defect_cell_failed_check_marked_non_ack(self):
        out = _make_pipeline()._build_cell_summary(_full_unified(2), self._report())
        cells = {c["id"]: c for c in out["cells"]}
        entry0 = cells[0]["failed_checks"][0]
        entry1 = cells[1]["failed_checks"][0]
        assert entry0["check_id"] == "value_range_soil_ph"
        assert entry0["defect"] is True, "defect cell must be non-acknowledgeable"
        assert entry1["defect"] is False, "warning-only cell stays acknowledgeable"

    def test_cell_failed_check_details_carry_defect(self):
        out = _make_pipeline()._build_cell_summary(_full_unified(2), self._report())
        rows = {r["cell_id"]: r for r in out["cell_failed_check_details"]}
        assert rows[0]["defect"] is True
        assert rows[1]["defect"] is False
