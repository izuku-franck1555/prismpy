"""ISIMIP CF → SARRA-Py unit conversions.

Sprint G boundary 7/7 absorption per codex round 2 P1: ISIMIP3b
ships SI / CF-1.x units (kg m⁻² s⁻¹ for precipitation, W m⁻² for
solar radiation, K for temperature). SARRA-Py expects (mm/day for
rainfall, J m⁻²/day for solar radiation, K passthrough for
temperature). This module IS the canonical conversion-math surface;
per durable §24 + the precedent of :mod:`prismpy.harmonize.tetens`
the math is exactly one module so a unit-convention drift surfaces
as one diff line.

The conversion factors are imported from
:mod:`prismpy.standards.isimip_versions` so the pin substrate is
shared with the canonical mapping table.

Public API (minimal):

* :func:`pr_kg_m2_s_to_mm_day` — precipitation conversion
* :func:`rsds_w_m2_to_j_m2_day` — solar radiation conversion
* :func:`temperature_passthrough_k` — temperature passthrough sanity-check
* :func:`convert_to_sarra_py_units` — dispatcher keyed by ISIMIP variable

Per durable §6.4 schema-layer: NaN/inf inputs propagate (the
conversion is mathematical; missing-data discipline lives at the
writer's nodata-replacement step, per codex round 1 boundary 3/7
P2 absorption in the SARRA-Py writer).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from prismpy.standards.isimip_versions import (
    ISIMIP_TO_SARRA_VAR_MAPPING,
    _KELVIN_TO_CELSIUS_OFFSET,
    _KG_PER_M2_PER_SECOND_TO_MM_PER_DAY,
    _W_PER_M2_TO_KJ_PER_M2_PER_DAY,
)


def pr_kg_m2_s_to_mm_day(values: Any) -> Any:
    """Convert precipitation flux from kg m⁻² s⁻¹ → mm/day.

    1 kg of water spread over 1 m² is 1 mm depth; 1 day is 86400 s.
    So the factor is 86400 (per
    :data:`prismpy.standards.isimip_versions._KG_PER_M2_PER_SECOND_TO_MM_PER_DAY`).

    Args:
        values: Numeric (scalar / numpy array / xarray DataArray) in
            kg m⁻² s⁻¹.

    Returns:
        Same shape, in mm/day.
    """
    return values * _KG_PER_M2_PER_SECOND_TO_MM_PER_DAY


def rsds_w_m2_to_kj_m2_day(values: Any) -> Any:
    """Convert downwelling shortwave radiation from W m⁻² →
    kJ m⁻²/day.

    1 W m⁻² × 86400 s/day ÷ 1000 J/kJ = 86.4 kJ m⁻²/day. Matches
    the AgERA5 ``version="SARRA-Py"`` vendored-library output that
    SARRA-Py's ``solar_radiation_flux_daily`` directory expects per
    ``validators/post_translate.py:576`` SARRA_PY_VAR_MAPPING
    (``mul 1e-3`` to scale kJ → MJ at validation time).

    Codex round 2 boundary 7/7 P1 absorption: earlier ``rsds_w_m2_to_j_m2_day``
    shipped J/m²/day which made SARRA-Py read radiation 1000× too
    large.

    Args:
        values: Numeric (scalar / numpy array / xarray DataArray) in
            W m⁻².

    Returns:
        Same shape, in kJ m⁻²/day.
    """
    return values * _W_PER_M2_TO_KJ_PER_M2_PER_DAY


def temperature_kelvin_to_celsius(values: Any) -> Any:
    """Convert ISIMIP temperatures from Kelvin → °C.

    SARRA-Py consumes °C per ``validators/post_translate.py:574-575``
    SARRA_PY_VAR_MAPPING noop ops (comment: "already °C") and
    ``sources/climate/agera5.py:1002`` ``version="SARRA-Py"`` flag
    which triggers K→°C in the vendored library. Codex round 2
    boundary 7/7 P1 absorption: earlier ``temperature_passthrough_k``
    shipped Kelvin which made downstream consumers read 300 K as
    300 °C — scientifically invalid.

    Args:
        values: Numeric (scalar / numpy array / xarray DataArray) in K.

    Returns:
        Same shape, in °C.
    """
    return values - _KELVIN_TO_CELSIUS_OFFSET


def convert_to_sarra_py_units(isimip_variable: str, values: Any) -> Any:
    """Dispatch unit conversion for an ISIMIP variable to its
    SARRA-Py target unit per
    :data:`prismpy.standards.isimip_versions.ISIMIP_TO_SARRA_VAR_MAPPING`.

    Args:
        isimip_variable: ISIMIP CF variable name (e.g., ``"pr"`` /
            ``"tasmax"``). Must be a key in
            :data:`ISIMIP_TO_SARRA_VAR_MAPPING`.
        values: Numeric input (scalar / array / DataArray) in the
            ISIMIP source unit per the mapping's declared
            ``source_unit`` field.

    Returns:
        Same shape, converted to the mapping's declared
        ``target_unit``.

    Raises:
        ValueError: If ``isimip_variable`` is not registered in
            :data:`ISIMIP_TO_SARRA_VAR_MAPPING`. The error message
            enumerates the registered variables so the caller knows
            which to use. Adding a new ISIMIP variable requires
            extending the mapping atomically with this dispatcher.
    """
    if isimip_variable not in ISIMIP_TO_SARRA_VAR_MAPPING:
        raise ValueError(
            f"No SARRA-Py mapping registered for ISIMIP variable "
            f"{isimip_variable!r}. Registered keys: "
            f"{sorted(ISIMIP_TO_SARRA_VAR_MAPPING.keys())}. Extend "
            "prismpy.standards.isimip_versions."
            "ISIMIP_TO_SARRA_VAR_MAPPING + the dispatcher in "
            "prismpy.harmonize.isimip_unit_conversions atomically."
        )
    _, source_unit, target_unit = ISIMIP_TO_SARRA_VAR_MAPPING[
        isimip_variable
    ]

    if source_unit == "kg m-2 s-1" and target_unit == "mm/day":
        return pr_kg_m2_s_to_mm_day(values)
    if source_unit == "W m-2" and target_unit == "kJ m-2 day-1":
        return rsds_w_m2_to_kj_m2_day(values)
    if source_unit == "K" and target_unit == "degC":
        return temperature_kelvin_to_celsius(values)

    raise ValueError(
        f"Unsupported unit conversion: {source_unit!r} → "
        f"{target_unit!r} for ISIMIP variable {isimip_variable!r}. "
        "Extend the dispatcher in prismpy.harmonize.isimip_unit_conversions "
        "with a new conversion helper."
    )


def sarra_py_directory_for_isimip(isimip_variable: str) -> str:
    """Return the SARRA-Py directory name for an ISIMIP CF variable.

    Thin re-export of :data:`ISIMIP_TO_SARRA_VAR_MAPPING` so callers
    that just want the directory name don't traverse the standards
    module. Per durable §24: the mapping table lives once at
    ``isimip_versions``; this helper unwraps the lookup.

    Raises:
        ValueError: If the variable is not registered. Same message
            shape as :func:`convert_to_sarra_py_units` for caller
            diagnostic consistency.
    """
    if isimip_variable not in ISIMIP_TO_SARRA_VAR_MAPPING:
        raise ValueError(
            f"No SARRA-Py directory registered for ISIMIP variable "
            f"{isimip_variable!r}. Registered keys: "
            f"{sorted(ISIMIP_TO_SARRA_VAR_MAPPING.keys())}."
        )
    sarra_dir, _, _ = ISIMIP_TO_SARRA_VAR_MAPPING[isimip_variable]
    return sarra_dir


__all__ = [
    "pr_kg_m2_s_to_mm_day",
    "rsds_w_m2_to_kj_m2_day",
    "temperature_kelvin_to_celsius",
    "convert_to_sarra_py_units",
    "sarra_py_directory_for_isimip",
]
