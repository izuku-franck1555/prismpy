"""Structural pin: Override bounds STRICTLY ≥ validator thresholds.

Sprint E.3 AC-E3-3 sub-criterion 4 + Stage 1 §10 phantom-bug pin.
Per AC-E3-3 contract text:

> Override bounds asserted strictly wider than ``CLIMATE_RANGES`` /
> ``SOIL_RANGES`` validator thresholds at
> ``prismpy/validators/scientific.py:61-77`` (Override should always
> allow at least the validator's accepted range, plus margin).

The mental model: the validator's job is to flag a cell as "outside
the typical range we expect for credible runs"; the override's job
is to let a persona document a credible value the validator flagged.
So Override must accept the validator's range AT LEAST — otherwise
the validator would flag a cell whose value the persona cannot
override, leaving them stuck. The bound MAY equal at physical-
endpoint cases (e.g., percentage = 0-100 on both sides; the
override is no wider but no narrower either).

Anti-mutation drill: a future hardening of a validator threshold
that drifted past the override bound would fire this pin loud rather
than silently rejecting personas' documented anomalies.

The mapping from check_id → validator-range key uses the
``variable_key`` slug embedded in the check_id (e.g.,
``value_range_tmax`` → ``CLIMATE_RANGES["tmax"]``,
``value_range_soil_sand`` → ``SOIL_RANGES["sand"]``). A check_id
without a corresponding validator threshold is exempted (the
override has no validator bound to be wider than).
"""

from __future__ import annotations

from prismpy.standards.override_value_shapes import (
    OVERRIDE_VALUE_SHAPES,
)
from prismpy.validators.scientific import (
    CLIMATE_RANGES,
    SOIL_RANGES,
)


# ── helper: derive the validator-range key from a check_id ─────────


def _validator_range_for_check_id(
    check_id: str,
) -> tuple[float, float] | None:
    """Return the (min, max) validator threshold for ``check_id`` or
    ``None`` if the check_id has no corresponding validator range
    (e.g., a categorical / aggregator check_id)."""
    if check_id.startswith("value_range_soil_"):
        soil_key = check_id[len("value_range_soil_"):]
        entry = SOIL_RANGES.get(soil_key)
        if entry is None:
            return None
        return entry[0], entry[1]
    if check_id.startswith("value_range_"):
        climate_key = check_id[len("value_range_"):]
        entry = CLIMATE_RANGES.get(climate_key)
        if entry is None:
            return None
        return entry[0], entry[1]
    return None


# ── §1 override-min ≤ validator-min (lower envelope) ───────────────


def test_override_min_at_or_below_validator_min() -> None:
    """Every entry's ``override_min`` is at or below the validator's
    lower threshold for the same variable. Equality is acceptable on
    physical-bound endpoints (e.g., 0% sand)."""
    offenders: list[str] = []
    for check_id, shape in OVERRIDE_VALUE_SHAPES.items():
        if shape.override_min is None:
            continue
        validator_range = _validator_range_for_check_id(check_id)
        if validator_range is None:
            continue
        validator_min = validator_range[0]
        if shape.override_min > validator_min:
            offenders.append(
                f"{check_id}: override_min={shape.override_min} > "
                f"validator_min={validator_min}"
            )
    assert not offenders, (
        f"Override-bounds-strictly-wider invariant violated on "
        f"lower envelope: {offenders}. The validator flags values "
        f"below its threshold; Override must accept those values "
        f"(at least) so the persona can document the anomaly. Per "
        f"AC-E3-3 sub-4 + durable §24 canonical-source-or-pin: "
        f"hardening a validator threshold past its Override bound "
        f"is a regression class — adjust the Override bound first."
    )


# ── §2 override-max ≥ validator-max (upper envelope) ───────────────


def test_override_max_at_or_above_validator_max() -> None:
    """Every entry's ``override_max`` is at or above the validator's
    upper threshold for the same variable. Equality is acceptable on
    physical-bound endpoints."""
    offenders: list[str] = []
    for check_id, shape in OVERRIDE_VALUE_SHAPES.items():
        if shape.override_max is None:
            continue
        validator_range = _validator_range_for_check_id(check_id)
        if validator_range is None:
            continue
        validator_max = validator_range[1]
        if shape.override_max < validator_max:
            offenders.append(
                f"{check_id}: override_max={shape.override_max} < "
                f"validator_max={validator_max}"
            )
    assert not offenders, (
        f"Override-bounds-strictly-wider invariant violated on "
        f"upper envelope: {offenders}. Same rationale as the lower "
        f"envelope test — Override accepts the validator's range "
        f"at minimum."
    )


# ── §3 documented-anomaly margin on non-physical-bound endpoints ───


def test_climate_overrides_strictly_wider_than_validator_range() -> None:
    """Climate variables aren't bounded by a hard physical limit at
    the validator range edge (Earth surface temps can go to ~57°C in
    Death Valley but the validator stops at 60°C; precipitation can
    hit ~970 mm/day at Cherrapunji but the validator stops at 600).
    The Override bound MUST be strictly wider on at least one side
    so a documented anomaly outside the validator's typical-run
    range is recordable.

    Soil variables (sand/clay = 0-100, organic carbon = 0-30,
    ph = 2.5-10.5, bulk density = 0.5-1.9) are partially physical-
    bounded so the wider-on-at-least-one-side rule applies only to
    climate ranges per the contract scope."""
    expected_strictly_wider_on_some_side: dict[str, tuple[float, float]] = {
        "value_range_tmax": (-50.0, 60.0),
        "value_range_tmin": (-60.0, 50.0),
        "value_range_precip": (0.0, 600.0),
        "value_range_srad": (0.0, 40.0),
    }
    offenders: list[str] = []
    for check_id, validator_range in expected_strictly_wider_on_some_side.items():
        shape = OVERRIDE_VALUE_SHAPES.get(check_id)
        if shape is None:
            offenders.append(f"{check_id}: missing from registry")
            continue
        validator_min, validator_max = validator_range
        wider_below = (
            shape.override_min is not None
            and shape.override_min < validator_min
        )
        wider_above = (
            shape.override_max is not None
            and shape.override_max > validator_max
        )
        if not (wider_below or wider_above):
            offenders.append(
                f"{check_id}: override "
                f"({shape.override_min}, {shape.override_max}) "
                f"not strictly wider than validator "
                f"({validator_min}, {validator_max}) on either side"
            )
    assert not offenders, (
        f"Climate override bounds must be strictly wider than the "
        f"validator's typical-run thresholds on at least one side "
        f"(documented-anomaly margin): {offenders}."
    )
