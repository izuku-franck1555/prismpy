"""Structural pin: ``OVERRIDE_VALUE_SHAPES`` registry canonical-source.

Sprint E.3 AC-E3-3 + Stage 1 §9 #4. Three invariants close the
silent-drift class per durable §24 canonical-source-or-pin + durable
§27 two-vocabulary substrate-drift:

§1 Subset-of-enumerated-check-ids — every key in
``OVERRIDE_VALUE_SHAPES`` MUST be a check_id the producer side
emits per ``enumerate_emitted_check_ids()``. A registry entry for a
phantom check_id (typo or stale rename) silently never fires; this
pin catches the drift.

§2 Field type discipline — ``variable_key`` non-empty + ``unit`` is a
string (empty allowed for unitless quantities like pH) + ``numeric_type``
is a recognised numeric class (``float`` / ``int``) + bounds are
``None`` only on the documented categorical-skip-bounds contract
(currently no entries; a future categorical Override target would
flip this).

§3 Bounds-min-≤-max — for every entry with non-None bounds,
``override_min <= override_max``. An inverted pair is a numeric-
formula bug.

The companion ``override-strictly-wider-than-validator`` invariant
lives at ``test_override_bounds_wider_than_validator.py`` per
AC-E3-3 sub-criterion 4.
"""

from __future__ import annotations

from prismpy.cockpit.check_id_enumeration import (
    enumerate_emitted_check_ids,
    matches_known_prefix,
)
from prismpy.standards.override_value_shapes import (
    OVERRIDE_VALUE_SHAPES,
    OverrideValueShape,
    get_override_value_shape,
)


# ── §1 subset of enumerated check_ids ──────────────────────────────


def test_registry_keys_are_subset_of_enumerated_check_ids() -> None:
    """Reject phantom check_ids (typo / stale rename). Every key in
    the registry MUST be a check_id the producer side actually
    emits, either in the static enumeration or matching one of the
    documented prefix families per ``matches_known_prefix``."""
    enumerated = enumerate_emitted_check_ids()
    offenders: list[str] = []
    for check_id in OVERRIDE_VALUE_SHAPES:
        if check_id in enumerated:
            continue
        if matches_known_prefix(check_id):
            continue
        offenders.append(check_id)
    assert not offenders, (
        f"OVERRIDE_VALUE_SHAPES carries phantom check_ids not in the "
        f"producer-side enumeration: {sorted(offenders)}. Per durable "
        f"§24 canonical-source-or-pin: every registry key MUST trace "
        f"to ``enumerate_emitted_check_ids()`` or a documented prefix "
        f"family at ``check_id_enumeration.matches_known_prefix``."
    )


# ── §2 field-type discipline ───────────────────────────────────────


def test_every_entry_has_non_empty_variable_key() -> None:
    """A blank ``variable_key`` would route the override to no
    sidecar slot — silent drop. Pin the bar at non-empty per
    ``feedback_no_data_cooking.md``."""
    for check_id, shape in OVERRIDE_VALUE_SHAPES.items():
        assert shape.variable_key, (
            f"OVERRIDE_VALUE_SHAPES[{check_id!r}].variable_key is "
            f"empty. Override would have no sidecar slot to write to."
        )


def test_unit_is_string() -> None:
    """``unit`` is always a string. Empty allowed (pH, dimensionless).
    Reject ``None`` or non-string values as type-discipline
    violations."""
    for check_id, shape in OVERRIDE_VALUE_SHAPES.items():
        assert isinstance(shape.unit, str), (
            f"OVERRIDE_VALUE_SHAPES[{check_id!r}].unit is not str: "
            f"{type(shape.unit).__name__}. Use empty string for "
            f"unitless quantities."
        )


def test_numeric_type_is_recognised_numeric_class() -> None:
    """``numeric_type`` is ``float`` / ``int`` / ``str``. Sprint E.3
    ships ``float`` for every entry (continuous physical quantities);
    a future categorical Override target would extend the allow
    list."""
    allowed = {float, int, str}
    for check_id, shape in OVERRIDE_VALUE_SHAPES.items():
        assert shape.numeric_type in allowed, (
            f"OVERRIDE_VALUE_SHAPES[{check_id!r}].numeric_type "
            f"{shape.numeric_type!r} not in allow list "
            f"{sorted(t.__name__ for t in allowed)}."
        )


def test_bounds_either_both_present_or_both_none() -> None:
    """``override_min`` and ``override_max`` are paired — one being
    None and the other a float is a half-specified contract that
    breaks form validation. Either both present (numeric bounds
    enforced) or both None (categorical skip-bounds)."""
    for check_id, shape in OVERRIDE_VALUE_SHAPES.items():
        both_present = (
            shape.override_min is not None
            and shape.override_max is not None
        )
        both_absent = (
            shape.override_min is None
            and shape.override_max is None
        )
        assert both_present or both_absent, (
            f"OVERRIDE_VALUE_SHAPES[{check_id!r}] has half-specified "
            f"bounds: min={shape.override_min!r} / "
            f"max={shape.override_max!r}. Either both present or "
            f"both None (categorical skip-bounds)."
        )


# ── §3 bounds order discipline ─────────────────────────────────────


def test_override_min_le_override_max_when_present() -> None:
    """Inverted bounds (min > max) are a numeric-formula bug — every
    value would reject. Pin the order invariant."""
    for check_id, shape in OVERRIDE_VALUE_SHAPES.items():
        if shape.override_min is None:
            continue
        assert shape.override_min <= shape.override_max, (
            f"OVERRIDE_VALUE_SHAPES[{check_id!r}] inverted bounds: "
            f"min={shape.override_min} > max={shape.override_max}."
        )


# ── §4 helper round-trip ───────────────────────────────────────────


def test_get_override_value_shape_round_trips_for_known_check_id() -> None:
    """``get_override_value_shape(check_id)`` returns the same object
    as ``OVERRIDE_VALUE_SHAPES[check_id]`` for every key, and
    returns ``None`` for an unknown check_id (Override not
    surfaced in the form for that affordance)."""
    for check_id, shape in OVERRIDE_VALUE_SHAPES.items():
        got = get_override_value_shape(check_id)
        assert got is shape, (
            f"get_override_value_shape({check_id!r}) returned "
            f"{got!r}; expected {shape!r}"
        )
    # Unknown check_id returns None so the form's per-check_id
    # branching can fall through to "Override not offered".
    assert get_override_value_shape("phantom_check_id_does_not_exist") is None


# ── §5 dunder-all is the canonical export surface ──────────────────


def test_module_exports_canonical_symbols() -> None:
    """The 3 canonical exports: registry dict + NamedTuple + helper.
    Internal constants (none currently) stay private."""
    from prismpy.standards import override_value_shapes
    assert sorted(override_value_shapes.__all__) == [
        "OVERRIDE_VALUE_SHAPES",
        "OverrideValueShape",
        "get_override_value_shape",
    ]


# ── §6 Sprint E.3 v1 coverage scope ────────────────────────────────


def test_v1_coverage_is_nine_check_ids() -> None:
    """Sprint E.3 v1 ships physical-plausibility bounds for the 9
    check_ids enumerated in AC-E3-3 (4 climate + 5 soil). A future
    sprint extending coverage (wind / rh / further soil aspects)
    extends the registry intentionally; this test documents the v1
    scope so the extension is a conscious change, not a silent
    enumeration drift."""
    expected = {
        "value_range_tmax",
        "value_range_tmin",
        "value_range_precip",
        "value_range_srad",
        "value_range_soil_sand",
        "value_range_soil_clay",
        "value_range_soil_organic_carbon",
        "value_range_soil_ph",
        "value_range_soil_bulk_density",
    }
    got = set(OVERRIDE_VALUE_SHAPES.keys())
    assert got == expected, (
        f"Sprint E.3 OVERRIDE_VALUE_SHAPES v1 scope: "
        f"{sorted(expected)}. Got: {sorted(got)}."
    )
