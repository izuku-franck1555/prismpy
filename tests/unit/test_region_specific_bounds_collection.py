"""V2-22c-PRE.1.7 — `affected_cells` + `violation_details` collection
on `region_specific_bounds`.

Before V2-22c the per-record violation loop emitted only text-string
diagnostics (`sample_violations[:10]`); cell-level attribution was
impossible from the validator surface. PRE.1.7 dual-tracks: the text
strings are PRESERVED at `[:10]` for human-readable digests, and a
new `affected_cells` list (deduplicated, sorted ASC) plus a structured
`violation_details` array carry the cockpit-needed cell-id lookup +
per-violation context (cell_id, variable, value, unit, bounds).

This is the load-bearing trigger for the V1 clay walkthrough: without
PRE.1.7's `affected_cells`, PRE.1.2's failed_checks pivot reads
`details.affected_cells` as empty and silently omits
`region_specific_bounds` entries from every per-cell `failed_checks`
array.

The text-string `sample_violations[:10]` cap is asserted by the
companion `test_scientific_un_truncation.py` allowlist — that test
locks PRE.1.7's "preserve text-string truncation" rule. This test
locks the new structured fields.
"""
from __future__ import annotations

from datetime import date

import pytest

from prismpy.models.region import Region, BoundingBox
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.translators.base import UnifiedData
from prismpy.validators.scientific import _check_region_bounds


def _make_unified_with_climate(climate_dict, *, lat=15.0, lon=0.0):
    """UnifiedData with a Sahel-centroid region (so `_detect_region`
    picks the Sahel thresholds: tmax [5, 50], tmin [0, 40],
    precip_daily_max 200, srad [5, 35])."""
    return UnifiedData(
        region=Region(
            name="sahel-test", country="t", country_iso3="TST",
            bounds=BoundingBox(
                minx=lon - 0.5, miny=lat - 0.5,
                maxx=lon + 0.5, maxy=lat + 0.5,
            ),
        ),
        climate=climate_dict,
    )


def _make_ts(records, *, cell_id=0):
    """Wrap a list of ClimateRecord objects in a ClimateTimeSeries.
    The location_id / lat / lon / source fields are required by the
    dataclass but irrelevant to `_check_region_bounds` — it only
    reads `ts.records`."""
    return ClimateTimeSeries(
        records=records,
        location_id=str(cell_id),
        lat=15.0, lon=0.0,
        source="test",
    )


def _make_record(*, day=1, tmax=25.0, tmin=15.0, precip=2.0, srad=20.0):
    return ClimateRecord(
        date=date(2020, 1, day),
        tmax=tmax, tmin=tmin, precip=precip, srad=srad,
    )


class TestAffectedCellsList:
    """V2-22c-PRE.1.7 — `details.affected_cells` carries the
    deduplicated cell-id list (sorted ASC). Drives the cockpit
    Layer 1 fill rule and the failed_checks pivot."""

    def test_violating_cells_listed_in_affected_cells(self):
        """Three cells, two with tmax violations (cells 2 + 5);
        affected_cells must contain exactly {2, 5} sorted ASC."""
        climate = {
            2: _make_ts([_make_record(tmax=60.0)]),  # over Sahel max=50
            3: _make_ts([_make_record(tmax=30.0)]),  # in-range
            5: _make_ts([_make_record(tmax=55.0)]),  # over max=50
        }
        unified = _make_unified_with_climate(climate)
        check = _check_region_bounds(unified, config=None)

        assert check["check"] == "region_specific_bounds"
        # Sahel was detected (centroid 15, 0); the threshold path
        # fires (not the "universal" early-return shape).
        assert check["scope"] == "per_record"
        affected = check["details"]["affected_cells"]
        assert affected == [2, 5], (
            f"affected_cells must be deduplicated + sorted ASC; "
            f"got {affected!r}"
        )

    def test_repeated_violations_per_cell_dedupe_in_affected_cells(self):
        """A single cell with multiple violating records on the
        same day OR multiple days should appear ONCE in
        affected_cells (cell-level granularity, not record-level)."""
        climate = {
            7: _make_ts([
                _make_record(day=1, tmax=60.0),  # violates
                _make_record(day=2, tmax=58.0),  # also violates — same cell
                _make_record(day=3, tmin=-10.0),  # violates tmin too
            ]),
        }
        unified = _make_unified_with_climate(climate)
        check = _check_region_bounds(unified, config=None)
        assert check["details"]["affected_cells"] == [7]
        # n_violations is record-level (3), not cell-level.
        assert check["details"]["n_violations"] == 3

    def test_no_violations_yields_empty_affected_cells(self):
        """Pass-result still emits the field — uniform schema for
        the §6.4 schema-bounds-match-strictest-consumer discipline."""
        climate = {
            1: _make_ts([_make_record(tmax=30.0, tmin=15.0,
                                      precip=5.0, srad=20.0)]),
        }
        unified = _make_unified_with_climate(climate)
        check = _check_region_bounds(unified, config=None)
        assert check["result"] == "pass"
        assert check["details"]["affected_cells"] == []


class TestViolationDetails:
    """V2-22c-PRE.1.8 setup — each violation carries cell_id +
    variable + value + unit + bounds context for the cockpit
    drawer's drill-down rendering."""

    def test_tmax_violation_record_has_full_context(self):
        climate = {
            4: _make_ts([_make_record(tmax=60.0)]),
        }
        unified = _make_unified_with_climate(climate)
        check = _check_region_bounds(unified, config=None)
        details = check["details"]["violation_details"]
        assert len(details) == 1
        rec = details[0]
        assert rec["cell_id"] == 4
        assert rec["variable"] == "tmax"
        assert rec["value"] == pytest.approx(60.0)
        assert rec["unit"] == "°C"
        # Sahel tmax threshold is [5, 50]; the 60.0 reading exceeds it.
        assert rec["bounds"] == [5.0, 50.0]

    def test_precip_daily_max_violation_uses_unbounded_lower(self):
        """precip_daily_max is a one-sided threshold (only an upper
        bound). The lower-bound entry must be `None`, not `0` —
        the cockpit's range-display copy reads this verbatim."""
        climate = {
            6: _make_ts([_make_record(precip=300.0)]),  # over Sahel 200 max
        }
        unified = _make_unified_with_climate(climate)
        check = _check_region_bounds(unified, config=None)
        details = check["details"]["violation_details"]
        precip_rec = next(d for d in details if d["variable"] == "precip")
        assert precip_rec["unit"] == "mm/day"
        assert precip_rec["bounds"][0] is None
        assert precip_rec["bounds"][1] == pytest.approx(200.0)

    def test_violation_details_sorted_by_cell_id_then_variable(self):
        """Reproducible JSON diffs across runs — sort by
        (cell_id, variable) so cockpit drill-down cursors are
        stable. Insert violations out of order; emission order
        must be sorted."""
        climate = {
            8: _make_ts([_make_record(srad=40.0)]),  # over [5, 35]
            2: _make_ts([
                _make_record(tmax=60.0),
                _make_record(tmin=-10.0),
            ]),
            5: _make_ts([_make_record(precip=300.0)]),
        }
        unified = _make_unified_with_climate(climate)
        check = _check_region_bounds(unified, config=None)
        details = check["details"]["violation_details"]
        keys = [(d["cell_id"], d["variable"]) for d in details]
        assert keys == sorted(keys), (
            f"violation_details must be sorted by (cell_id, variable); "
            f"got {keys!r}"
        )


class TestSampleViolationsTextStringPreserved:
    """V2-22c-PRE.1.7 explicit preservation rule — the diagnostic
    text-string sample stays capped at 10 (human-readable digest)
    even as the un-truncated cell-id list goes to a NEW
    `affected_cells` field. The companion structural test in
    `test_scientific_un_truncation.py` keeps the cap rule
    enforced at the AST level."""

    def test_sample_violations_capped_at_10(self):
        """Generate 15 violations; sample_violations is the first
        10 text strings."""
        records = [
            _make_record(day=d, tmax=60.0) for d in range(1, 16)
        ]
        climate = {1: _make_ts(records)}
        unified = _make_unified_with_climate(climate)
        check = _check_region_bounds(unified, config=None)
        assert check["details"]["n_violations"] == 15
        assert len(check["details"]["sample_violations"]) == 10
        # Strings (not structured records) — verify shape preserved.
        for s in check["details"]["sample_violations"]:
            assert isinstance(s, str)
            assert "outside" in s or "exceeds" in s

    def test_un_truncated_affected_cells_when_text_sample_capped(self):
        """Backstop: even when sample_violations hits its 10-item
        cap, affected_cells reflects ALL violating cells without
        truncation."""
        # 15 cells, each violating tmax once.
        climate = {
            i: _make_ts([_make_record(tmax=60.0)])
            for i in range(1, 16)
        }
        unified = _make_unified_with_climate(climate)
        check = _check_region_bounds(unified, config=None)
        assert len(check["details"]["sample_violations"]) == 10
        assert len(check["details"]["affected_cells"]) == 15
        assert check["details"]["affected_cells"] == list(range(1, 16))
