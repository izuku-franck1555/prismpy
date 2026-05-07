"""FAO-56 Tetens dewpoint derivation for projection climate.

Sprint G AC-G-8: ISIMIP3b projection cutouts carry relative humidity
(``hurs``) but not dewpoint (``tdew``); downstream WTH writers
(CRAFT/PYTHIA per AC-G-7a) need ``tdew`` for the 8-column projection
path. The Tetens / Magnus form gives a clean closed-form derivation.

References:

* FAO Irrigation and Drainage Paper 56 (Allen et al., 1998), §3
  "Atmospheric parameters" — saturation vapour pressure (Eq 11) +
  actual vapour pressure (Eq 17) + dewpoint inversion (Eq 14
  inverted). https://www.fao.org/3/x0490e/x0490e07.htm
* Magnus-Tetens form constants (the historical derivation lineage
  preserved by FAO-56 for irrigation engineering use). The constants
  ``A = 17.27`` and ``B = 237.3 °C`` and ``E_0 = 0.6108 kPa`` are the
  Tetens / Magnus values FAO-56 §3 Eq 11 specifies.

Per durable §24 canonical-source-or-pin: this module IS the canonical
source for Tetens dewpoint. The structural pin
``tests/structural/test_tetens_pin.py`` asserts the three magic
constants (``17.27``, ``237.3``, ``0.6108``) only co-appear in this
module — no call site re-derives the math inline. Downstream consumers
(AC-G-7a/b/c writers) ``from prismpy.harmonize.tetens import derive_tdew``.

Per durable §6.4 schema-layer: inputs are validated at the boundary —
``-90 ≤ tasmean_celsius ≤ 70`` (atmospheric temperature range) and
``0 < hurs_pct ≤ 100`` (relative humidity bounds). ``hurs_pct == 0``
is impossibly dry — the math gives ``ln(0) = -∞`` — and we fail
loud rather than ship NaN downstream. NaN/inf inputs raise
``ValueError`` at the boundary so a silent NaN cannot propagate into
WTH files.
"""

from __future__ import annotations

import math
from typing import Any


# ── FAO-56 Magnus-Tetens constants ───────────────────────────────────


_TETENS_A: float = 17.27
"""FAO-56 §3 Eq 11 numerator coefficient. Dimensionless."""

_TETENS_B: float = 237.3
"""FAO-56 §3 Eq 11 denominator addend. °C."""

_E_0: float = 0.6108
"""FAO-56 §3 Eq 11 saturation vapour pressure at 0 °C. kPa."""


# ── Input validation bounds ──────────────────────────────────────────


_TEMP_MIN_CELSIUS: float = -90.0
"""Lower bound on atmospheric temperature for dewpoint derivation.

The Vostok Antarctica record (-89.2 °C) sets the practical floor;
ISIMIP3b projection scenarios for West/East Africa never approach
this, so any input below -90 indicates upstream corruption or unit
confusion (e.g., the caller passing kelvin without subtracting
273.15)."""


_TEMP_MAX_CELSIUS: float = 70.0
"""Upper bound on atmospheric temperature.

The hottest reliable surface measurement is ~57 °C (Death Valley);
70 °C as an upper bound catches unit confusion (caller passed kelvin
instead of celsius) without flagging legitimate Sahel extremes."""


_HURS_MIN_PERCENT: float = 0.01
"""Lower bound on relative humidity, exclusive of zero.

``hurs == 0`` is mathematically impossible — actual vapour pressure
zero implies infinite negative dewpoint. The 0.01 floor matches
ISIMIP3b's documented saturation-pressure-clipping conventions and
avoids the ``ln(0)`` boundary while allowing legitimately dry-air
inputs."""


_HURS_MAX_PERCENT: float = 100.0
"""Upper bound on relative humidity. ``hurs == 100`` is the
saturation invariant (tdew == temperature)."""


# ── Public API ───────────────────────────────────────────────────────


def derive_tdew(
    tasmean_celsius: float,
    hurs_pct: float,
) -> float:
    """Compute dewpoint temperature from mean temperature + RH.

    Implements FAO-56 §3 Magnus-Tetens dewpoint derivation:

    .. math::

        e_s(T) = 0.6108 \\exp\\!\\left(\\frac{17.27 T}{T + 237.3}\\right)
        \\quad\\text{[kPa, saturation vapour pressure]}

        e_a = \\frac{\\text{hurs}}{100} \\, e_s(T)
        \\quad\\text{[kPa, actual vapour pressure]}

        T_d = \\frac{237.3 \\ln\\!\\left(\\frac{e_a}{0.6108}\\right)}
                  {17.27 - \\ln\\!\\left(\\frac{e_a}{0.6108}\\right)}
        \\quad\\text{[°C]}

    Args:
        tasmean_celsius: Mean atmospheric temperature in °C. Bounds:
            ``-90 ≤ T ≤ 70``. NaN / inf raise ValueError.
        hurs_pct: Relative humidity in percent, ``0 < hurs ≤ 100``.
            ``0`` is rejected (math: ``ln(0) = -∞``); use a small
            positive value (e.g., 0.01) if the source carries a
            saturation-pressure-clipped near-zero. NaN / inf raise
            ValueError.

    Returns:
        Dewpoint temperature in °C. Saturation invariant: when
        ``hurs_pct == 100``, returns ``tasmean_celsius`` exactly to
        within float-precision tolerance.

    Raises:
        ValueError: when inputs are out of bounds, NaN, or inf. The
            boundary-rejection contract per durable §6.4 + the
            ``feedback_no_data_cooking.md`` honest-signal discipline:
            never ship NaN downstream into WTH writers.

    Examples:
        >>> derive_tdew(20.0, 100.0)  # saturation
        20.0
        >>> round(derive_tdew(25.0, 50.0), 2)
        13.86
    """
    _validate_finite(tasmean_celsius, "tasmean_celsius")
    _validate_finite(hurs_pct, "hurs_pct")
    _validate_temperature(tasmean_celsius)
    _validate_relative_humidity(hurs_pct)

    saturation_vapour_pressure = _E_0 * math.exp(
        (_TETENS_A * tasmean_celsius) / (tasmean_celsius + _TETENS_B)
    )
    actual_vapour_pressure = (hurs_pct / 100.0) * saturation_vapour_pressure

    # ``ln(ea / E_0)`` is mathematically equivalent to
    # ``(A * T / (T + B)) + ln(hurs / 100)``; the longer form below
    # mirrors FAO-56's stated arithmetic for audit traceability.
    ln_ratio = math.log(actual_vapour_pressure / _E_0)
    dewpoint = (_TETENS_B * ln_ratio) / (_TETENS_A - ln_ratio)
    return dewpoint


def _validate_finite(value: Any, field_name: str) -> None:
    """Reject NaN / inf early — never propagate into the math."""
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a real number; got {type(value).__name__}"
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            f"{field_name} must be finite; got {value!r} "
            "(NaN / inf is forbidden — ship a fail-loud error rather "
            "than a NaN dewpoint that would silently poison WTH files)."
        )


def _validate_temperature(tasmean_celsius: float) -> None:
    if not (_TEMP_MIN_CELSIUS <= tasmean_celsius <= _TEMP_MAX_CELSIUS):
        raise ValueError(
            f"tasmean_celsius {tasmean_celsius!r} out of bounds "
            f"[{_TEMP_MIN_CELSIUS}, {_TEMP_MAX_CELSIUS}] °C. The likely "
            "cause is unit confusion (caller passed kelvin instead of "
            "celsius — subtract 273.15) or upstream data corruption."
        )


def _validate_relative_humidity(hurs_pct: float) -> None:
    if not (_HURS_MIN_PERCENT <= hurs_pct <= _HURS_MAX_PERCENT):
        raise ValueError(
            f"hurs_pct {hurs_pct!r} out of bounds "
            f"({_HURS_MIN_PERCENT}, {_HURS_MAX_PERCENT}] %. Note the "
            "lower bound is exclusive of zero (the math gives "
            "ln(0) = -inf for actual vapour pressure 0); ISIMIP3b "
            "saturation-pressure-clipping ensures real data stays "
            "above the floor. The likely cause of out-of-bounds is "
            "unit confusion (hurs as fraction 0-1 not percent 0-100)."
        )


def derive_tdew_for_record_or(
    *,
    explicit_tdew: Any,
    tmean_celsius: Any,
    hurs_pct: Any,
    fallback: float,
) -> float:
    """Resolve a single record's TDEW with projection-path fallback chain.

    Order of precedence:

    1. ``explicit_tdew`` — if the source already supplies a non-None
       dewpoint, return it as-is (preserves observed-source TDEW
       directly without re-deriving).
    2. Tetens derivation — if the source supplies non-None
       ``tmean_celsius`` + ``hurs_pct``, derive via :func:`derive_tdew`.
    3. ``fallback`` — when neither resolves; honest-signal "data
       genuinely unavailable" sentinel.

    The Tetens helper validates inputs at the boundary (temperature
    bounds + humidity bounds + finite-check); any out-of-bound input
    (e.g., a kelvin-vs-celsius unit confusion upstream) raises
    ``ValueError`` inside derive_tdew, which this wrapper translates
    into the fallback so a single bad record does not fail the whole
    writer run. The cell-summary / coverage validator records the
    data-availability status downstream.

    Args:
        explicit_tdew: ``record.tdew`` (Optional[float]). When non-
            None, returned directly.
        tmean_celsius: ``record.tmean`` (Optional[float]). Required
            for Tetens derivation.
        hurs_pct: ``record.rh`` (Optional[float]). Required for
            Tetens derivation. Per ClimateRecord docstring rh is
            already %.
        fallback: Platform-specific missing-value sentinel
            (e.g., DSSAT WTH ``-99.0``).

    Returns:
        Float TDEW in °C, or the fallback sentinel.
    """
    if explicit_tdew is not None:
        return float(explicit_tdew)
    if tmean_celsius is None or hurs_pct is None:
        return fallback
    try:
        return derive_tdew(float(tmean_celsius), float(hurs_pct))
    except (ValueError, TypeError):
        # Out-of-bound or non-numeric input — emit the fallback
        # sentinel rather than silently propagating NaN downstream.
        return fallback


__all__ = [
    "derive_tdew",
    "derive_tdew_for_record_or",
]
