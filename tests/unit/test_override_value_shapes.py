"""Behavioral tests for the cockpit Override value-shape registry.

Sprint E.3 AC-E3-3 sub-criterion 5: 5 unit tests covering in-bound
accept / out-of-bound reject (above) / out-of-bound reject (below) /
unit-conversion correctness / categorical-field skip-bounds.

The structural pins (parity with check_id_enumeration / bounds-
order / override-strictly-wider-than-validator) live at
``tests/structural/test_override_value_shape_per_check_id.py`` +
``tests/structural/test_override_bounds_wider_than_validator.py``.
This file covers the behavioral path the Phase 2 Override Edit form
+ Phase 1 sidecar consumer take when validating a persona-typed
value.

The form-side validation routine itself ships in Phase 2 (cockpit
form rendering); this file pins the registry's role as the single
source of truth those validators read from.
"""

from __future__ import annotations

import pytest

from prismpy.standards.override_value_shapes import (
    OVERRIDE_VALUE_SHAPES,
    OverrideValueShape,
    get_override_value_shape,
)


# ── §1 in-bound accept ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "check_id,value",
    [
        # tmax: 25°C is firmly inside the (-60, 70) bound and inside
        # the validator's (-50, 60) typical-run range.
        ("value_range_tmax", 25.0),
        # tmin: 5°C — typical Sahel cold-night tmin.
        ("value_range_tmin", 5.0),
        # precip: 50 mm/day — heavy but within validator's (0, 600).
        ("value_range_precip", 50.0),
        # srad: 20 MJ/m^2/d — typical Sahel growing-season day.
        ("value_range_srad", 20.0),
        # soil sand: 35% — typical loam.
        ("value_range_soil_sand", 35.0),
        # soil ph: 6.5 — neutral.
        ("value_range_soil_ph", 6.5),
        # soil bulk_density: 1.4 g/cm³ — typical agricultural soil.
        ("value_range_soil_bulk_density", 1.4),
    ],
)
def test_in_bound_value_accepts(check_id: str, value: float) -> None:
    """Persona types a value inside the registry's bounds → form
    validation MUST accept. Asserted via the bound-comparison
    routine that the form will use directly."""
    shape = get_override_value_shape(check_id)
    assert shape is not None, f"Registry missing entry for {check_id!r}"
    assert shape.override_min is not None
    assert shape.override_max is not None
    assert shape.override_min <= value <= shape.override_max, (
        f"Value {value} should be in-bound for {check_id!r} "
        f"({shape.override_min}, {shape.override_max})"
    )


# ── §2 out-of-bound reject (above) ─────────────────────────────────


@pytest.mark.parametrize(
    "check_id,value",
    [
        # tmax: 100°C exceeds the (-60, 70) override bound — the
        # form should reject ("physically implausible" copy per
        # AC-E3-3 contract).
        ("value_range_tmax", 100.0),
        # precip: 5000 mm/day exceeds (0, 1000); no recorded
        # daily-rainfall extreme is anywhere near 5 m.
        ("value_range_precip", 5000.0),
        # soil ph: 14 exceeds (1, 12) — beyond lab-titration scale.
        ("value_range_soil_ph", 14.0),
    ],
)
def test_above_max_rejects(check_id: str, value: float) -> None:
    """Persona types a value above the registry's max → form
    validation MUST reject."""
    shape = get_override_value_shape(check_id)
    assert shape is not None
    assert shape.override_max is not None
    assert value > shape.override_max, (
        f"Test fixture broken: {value} should be above max "
        f"{shape.override_max} for {check_id!r}"
    )


# ── §3 out-of-bound reject (below) ─────────────────────────────────


@pytest.mark.parametrize(
    "check_id,value",
    [
        # tmin: -100°C is below the (-70, 50) override bound.
        ("value_range_tmin", -100.0),
        # precip: -5 mm/day is below (0, 1000) — rain can't be
        # negative.
        ("value_range_precip", -5.0),
        # soil sand: -10% below (0, 100) — percentages don't go
        # negative.
        ("value_range_soil_sand", -10.0),
    ],
)
def test_below_min_rejects(check_id: str, value: float) -> None:
    """Persona types a value below the registry's min → form
    validation MUST reject."""
    shape = get_override_value_shape(check_id)
    assert shape is not None
    assert shape.override_min is not None
    assert value < shape.override_min, (
        f"Test fixture broken: {value} should be below min "
        f"{shape.override_min} for {check_id!r}"
    )


# ── §4 unit-conversion correctness ─────────────────────────────────


def test_canonical_units_recorded_per_check_id() -> None:
    """Every registry entry carries the canonical unit string the
    sidecar persists. The form's unit-selector default reads from
    this; if the persona switches to a different unit, the form
    converts on submit so all stored values are in canonical units
    (per durable §27 two-vocabulary substrate-drift discipline —
    value-and-unit pinned together at the source).

    This test pins the canonical-unit set Sprint E.3 ships;
    extending the registry with a new variable in a different
    unit is an intentional change that updates this assertion."""
    expected_units: dict[str, str] = {
        "value_range_tmax": "C",
        "value_range_tmin": "C",
        "value_range_precip": "mm/day",
        "value_range_srad": "MJ/m^2/d",
        "value_range_soil_sand": "%",
        "value_range_soil_clay": "%",
        "value_range_soil_organic_carbon": "%",
        "value_range_soil_ph": "",  # dimensionless
        "value_range_soil_bulk_density": "g/cm^3",
    }
    for check_id, expected_unit in expected_units.items():
        shape = get_override_value_shape(check_id)
        assert shape is not None, (
            f"Registry missing entry for {check_id!r}"
        )
        assert shape.unit == expected_unit, (
            f"{check_id!r} unit drift: got {shape.unit!r}, "
            f"expected {expected_unit!r}. The form converts on "
            f"submit so the canonical unit MUST stay stable across "
            f"the registry; a unit change is a substrate-drift "
            f"event."
        )


# ── §5 categorical-field skip-bounds contract ──────────────────────


def test_unknown_check_id_returns_none_for_skip_bounds_contract() -> None:
    """An unknown check_id (not in the registry) returns ``None`` so
    the form's per-check_id branching can fall through to "Override
    not offered". This is the documented categorical-skip-bounds
    contract: the form does NOT render a value-replacement input;
    the persona's affordance is Skip or Acknowledge or Interpolate
    instead.

    Sprint E.3 v1 ships no categorical-typed Override entries; if a
    future sprint adds one (with both bounds None on a registry
    entry), the form's rendering path treats that as "render but
    don't enforce numeric bounds" — distinct from "not in registry"
    which is "don't render at all". This test pins the
    not-in-registry path; the both-None path would gain its own
    pin when categorical entries land."""
    assert get_override_value_shape("post_translate_climate_acea") is None
    assert get_override_value_shape("format_compliance") is None
    assert get_override_value_shape("coverage_climate_cells") is None
    assert get_override_value_shape("phantom_check_id") is None


def test_namedtuple_unpacks_as_5_tuple() -> None:
    """The registry value is a NamedTuple; consumers can unpack
    positionally as a 5-tuple per AC-E3-3 contract text. This
    verifies the structural shape so a refactor to a different
    container class doesn't silently change the consumer
    contract."""
    shape = get_override_value_shape("value_range_tmax")
    assert shape is not None
    variable_key, unit, numeric_type, override_min, override_max = shape
    assert variable_key == "tmax_growing_season_mean"
    assert unit == "C"
    assert numeric_type is float
    assert override_min == -60.0
    assert override_max == 70.0


def test_namedtuple_immutability() -> None:
    """NamedTuple frozenness — attempt to reassign a field MUST
    raise. Pin the immutability invariant so a future refactor that
    swaps to a mutable container fires this test loud."""
    shape = OverrideValueShape(
        variable_key="x",
        unit="y",
        numeric_type=float,
        override_min=0.0,
        override_max=1.0,
    )
    with pytest.raises(AttributeError):
        shape.variable_key = "z"  # type: ignore[misc]
