"""V2-22c-PRE.1.4 + PRE.1.5 — `affected_cells` collection on the
soil-quality validator family + the composite `value_range_texture_sum`.

Before V2-22c the soil-range validator emitted only aggregate counters
(`out_of_range_count`, `total_values`); the cockpit's per-cell drill-down
had no cell IDs to map onto. PRE.1.4/1.5 is **net-new collection**: the
inner loop now records `(cell_id, layer_idx)` tuples on each violation
and emits them as `details.affected_cells` per check. The companion
`details.violation_details` field carries the per-violation context
(value, unit, bounds) that PRE.1.8 will flatten into the top-level
`cell_failed_check_details` array.

Coverage:
- Texture-sum violation surfaces the offending (cell, layer) tuple.
- Each `value_range_soil_<var>` check carries its own per-variable
  `affected_cells` list — a cell that violates only `clay` does NOT
  get attributed to `sand`'s affected list.
- All seven soil-quality checks (sand / clay / silt / oc / ph /
  bulk_density / texture_sum) emit the new fields.
- Pass-result checks still emit `affected_cells: []` (uniform schema).
"""
from __future__ import annotations

import pytest

from prismpy.models.region import Region, BoundingBox
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.translators.base import UnifiedData
from prismpy.validators.scientific import _check_value_ranges


def _make_unified(soil_dict):
    """Minimal UnifiedData with just soil; climate is set to an empty
    dict because `_check_value_ranges` calls `.items()` on it
    unconditionally."""
    return UnifiedData(
        region=Region(
            name="t", country="t", country_iso3="TST",
            bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        ),
        climate={},
        soil=soil_dict,
    )


def _by_check(checks, name):
    """Return the first check dict matching `name`, or None."""
    for c in checks:
        if c["check"] == name:
            return c
    return None


class TestTextureSumAffectedCells:
    """V2-22c-PRE.1.4 — texture sum (sand+clay+silt) outside
    [95, 105] populates `details.affected_cells` with the violating
    (cell_id, layer_idx) tuples."""

    def test_violating_layer_recorded_with_cell_and_layer_index(self):
        """Layer 1 (idx=1) of cell 7 has sand+clay+silt = 110 → violation."""
        profile = SoilProfile(
            profile_id="p7", lat=0.5, lon=0.5, source="iSDA",
            layers=[
                SoilLayer(depth_top=0, depth_bottom=0.2,
                          sand=40, clay=30, silt=30),  # sums to 100, OK
                SoilLayer(depth_top=0.2, depth_bottom=0.4,
                          sand=50, clay=40, silt=20),  # sums to 110, violates
            ],
        )
        unified = _make_unified({7: profile})
        checks = _check_value_ranges(unified)

        ts = _by_check(checks, "value_range_texture_sum")
        assert ts is not None, "texture_sum check missing entirely"
        assert ts["details"]["violations"] == 1
        assert ts["details"]["affected_cells"] == [(7, 1)]

    def test_no_violation_yields_empty_affected_cells(self):
        """Pass-result check still emits the field — uniform schema
        is the §6.4 schema-bounds-match-strictest-consumer discipline."""
        profile = SoilProfile(
            profile_id="p7", lat=0.5, lon=0.5, source="iSDA",
            layers=[
                SoilLayer(depth_top=0, depth_bottom=0.2,
                          sand=40, clay=30, silt=30),
            ],
        )
        unified = _make_unified({7: profile})
        checks = _check_value_ranges(unified)

        ts = _by_check(checks, "value_range_texture_sum")
        assert ts is not None
        assert ts["details"]["affected_cells"] == []
        assert ts["result"] == "pass"


class TestSoilQualityFamilyAffectedCells:
    """V2-22c-PRE.1.5 — every per-layer soil quality check
    (sand, clay, silt, organic_carbon, ph, bulk_density) carries its
    own `affected_cells` list, independent of every other check.

    The seventh family member, `texture_sum`, is the composite
    (covered by TestTextureSumAffectedCells).
    """

    def test_clay_violation_attributed_only_to_clay_check(self):
        """Only `value_range_soil_clay` should list cell 5 — sand,
        silt, etc. have no violation on this layer."""
        profile = SoilProfile(
            profile_id="p5", lat=0.5, lon=0.5, source="iSDA",
            layers=[
                # Note: clay=120 is impossible IRL; the validator's
                # SoilLayer __post_init__ doesn't reject this — the
                # range check fires on the value-range validator.
                SoilLayer(depth_top=0, depth_bottom=0.2,
                          sand=40, clay=120, silt=-60),
            ],
        )
        unified = _make_unified({5: profile})
        checks = _check_value_ranges(unified)

        clay_check = _by_check(checks, "value_range_soil_clay")
        assert clay_check is not None
        assert (5, 0) in clay_check["details"]["affected_cells"]

        # Sand is in-range (40); the sand check should NOT attribute
        # cell 5 to itself.
        sand_check = _by_check(checks, "value_range_soil_sand")
        assert sand_check is not None
        assert (5, 0) not in sand_check["details"]["affected_cells"]

    def test_violation_details_carry_value_unit_bounds(self):
        """PRE.1.8 setup — the `violation_details` field on each check
        is the upstream feed for the top-level
        `cell_failed_check_details` array. Each entry must carry the
        full (cell, layer, var, value, unit, bounds) tuple so the
        cockpit drawer can render the violation row directly. We use
        `bulk_density` here because its SOIL_RANGES unit is the
        non-empty `g/cm³` — pH is intentionally unitless and would
        false-pass the unit assertion."""
        profile = SoilProfile(
            profile_id="p3", lat=0.5, lon=0.5, source="iSDA",
            layers=[
                SoilLayer(depth_top=0, depth_bottom=0.2,
                          sand=40, clay=15, silt=45,
                          bulk_density=2.5),  # max is 1.9 → out of range
            ],
        )
        unified = _make_unified({3: profile})
        checks = _check_value_ranges(unified)

        bd_check = _by_check(checks, "value_range_soil_bulk_density")
        assert bd_check is not None
        details = bd_check["details"]["violation_details"]
        assert len(details) == 1
        entry = details[0]
        assert entry["cell_id"] == 3
        assert entry["layer_idx"] == 0
        assert entry["variable"] == "bulk_density"
        assert entry["value"] == pytest.approx(2.5)
        assert entry["unit"] == "g/cm³"
        assert entry["bounds"] == [0.5, 1.9]   # SOIL_RANGES["bulk_density"]

    def test_every_soil_check_emits_affected_cells_field(self):
        """Schema discipline — the new `affected_cells` field is
        present on every soil-quality check, even pass results."""
        profile = SoilProfile(
            profile_id="p1", lat=0.5, lon=0.5, source="iSDA",
            layers=[
                SoilLayer(
                    depth_top=0, depth_bottom=0.2,
                    sand=40, clay=30, silt=30,
                    organic_carbon=2.0, ph=6.5, bulk_density=1.4,
                ),
            ],
        )
        unified = _make_unified({1: profile})
        checks = _check_value_ranges(unified)

        for var in ("sand", "clay", "silt", "organic_carbon", "ph", "bulk_density"):
            check = _by_check(checks, f"value_range_soil_{var}")
            assert check is not None, f"missing value_range_soil_{var}"
            assert "affected_cells" in check["details"], (
                f"value_range_soil_{var} missing affected_cells field"
            )
            assert "violation_details" in check["details"], (
                f"value_range_soil_{var} missing violation_details field"
            )

    def test_violation_details_are_per_check_isolated(self):
        """A violation on `clay` must NOT leak into `sand`'s
        violation_details list. Each check's list is filtered by
        variable name."""
        profile = SoilProfile(
            profile_id="p9", lat=0.5, lon=0.5, source="iSDA",
            layers=[
                SoilLayer(depth_top=0, depth_bottom=0.2,
                          sand=40, clay=120, silt=-60),
            ],
        )
        unified = _make_unified({9: profile})
        checks = _check_value_ranges(unified)

        sand_check = _by_check(checks, "value_range_soil_sand")
        clay_check = _by_check(checks, "value_range_soil_clay")
        # sand value is in-range → violation_details empty for sand
        assert sand_check["details"]["violation_details"] == []
        # clay is out-of-range → violation_details has the entry
        assert any(
            d["variable"] == "clay"
            for d in clay_check["details"]["violation_details"]
        )
        # No cross-contamination
        assert all(
            d["variable"] == "clay"
            for d in clay_check["details"]["violation_details"]
        )
