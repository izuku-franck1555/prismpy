"""Compose ISIMIP3b bbox cutouts into canonical per-cell ClimateTimeSeries.

This module is the bridge between the ISIMIP3b cutout fetch
(:func:`prismpy.data_sources.isimip3b.cached_cutout`) and the platform
translators' ``generate_package`` weather path. It wires three existing
harmonize primitives — calendar conversion, CF→canonical unit conversion,
and FAO-56 dewpoint derivation — with per-cell nearest-neighbour sampling.

There is no new conversion science here: every numeric transform routes
through the single-source-of-truth helpers so a unit or calendar drift
surfaces as one diff line in the primitive module, never duplicated here.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from prismpy.harmonize.calendar_conversion import (
    calendar_for_gcm,
    convert_to_gregorian,
)
from prismpy.harmonize.isimip_unit_conversions import (
    pr_kg_m2_s_to_mm_day,
    rsds_w_m2_to_kj_m2_day,
    temperature_kelvin_to_celsius,
)
from prismpy.harmonize.tetens import derive_tdew
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.spatial import GridCell

# ClimateRecord.srad is MJ/m²/day; the shared unit helper emits kJ/m²/day,
# so apply the documented kJ→MJ factor to land on the canonical agronomic unit.
_KJ_TO_MJ: float = 1e-3

# The four daily DSSAT-driving variables every projection cutout must carry.
_REQUIRED_VARS: Tuple[str, ...] = ("tasmax", "tasmin", "pr", "rsds")

# Optional relative-humidity variable; when present, per-record dewpoint is
# derived via the Tetens helper, otherwise tdew stays None.
_HUMIDITY_VAR: str = "hurs"

# Metadata keys carrying the (optional) calendar-conversion limitation forward
# to the orchestrator's manifest writer so the disclosure is not lost.
CALENDAR_LIMITATION_KEY_FIELD: str = "calendar_limitation_key"
CALENDAR_LIMITATION_VALUE_FIELD: str = "calendar_limitation_value"

# Disclosed dewpoint policy: derive the FAO-56 Tetens dewpoint from a valid
# relative humidity; a record whose RH falls outside (0, 100]% propagates
# missing (rh and tdew left None) rather than fabricating a value. Surfaced in
# series metadata when humidity is supplied so the manifest writer can disclose it.
DEWPOINT_POLICY_FIELD: str = "dewpoint_policy"
_DEWPOINT_POLICY_DESCRIPTION = (
    "FAO-56 Tetens dewpoint from hurs; records with relative humidity outside "
    "(0, 100]% propagate-missing (rh and tdew left None) rather than "
    "substituting a fabricated value."
)


def isimip_cutouts_to_climate_timeseries(
    cutouts_by_var: Mapping[str, Any],
    cells: Iterable[GridCell],
    *,
    gcm_source: str,
    source_label: Optional[str] = None,
) -> Dict[int, ClimateTimeSeries]:
    """Compose per-variable ISIMIP3b cutouts into per-cell ClimateTimeSeries.

    Args:
        cutouts_by_var: Mapping of ISIMIP CF variable name → an xarray
            DataArray (dims ``(time, lat, lon)``) in raw CF units: K for
            temperature, kg m⁻² s⁻¹ for precipitation, W m⁻² for radiation.
            MUST carry the four required variables (``tasmax`` / ``tasmin`` /
            ``pr`` / ``rsds``); ``"hurs"`` (%) is optional and, when present,
            drives per-record FAO-56 dewpoint derivation.
        cells: Iterable of :class:`GridCell`; each is sampled by
            nearest-neighbour at its ``(lat, lon)`` centre.
        gcm_source: ISIMIP3b GCM identifier — selects the source calendar via
            :func:`calendar_for_gcm` (noleap / gregorian / 360_day).
        source_label: Optional ``ClimateTimeSeries.source`` override; defaults
            to ``f"ISIMIP3b_{gcm_source}"``.

    Returns:
        Mapping of ``cell_id`` → :class:`ClimateTimeSeries` in canonical units
        (°C, mm/day, MJ/m²/day). When the source-calendar conversion produced
        a limitation, every series carries it under
        ``metadata[CALENDAR_LIMITATION_KEY_FIELD / _VALUE_FIELD]`` so the
        downstream manifest writer can disclose it.

    Raises:
        ValueError: When a required variable is absent, or when ``gcm_source``
            is not a registered ISIMIP3b GCM (delegated to
            :func:`calendar_for_gcm`).
    """
    missing = [variable for variable in _REQUIRED_VARS if variable not in cutouts_by_var]
    if missing:
        raise ValueError(
            f"isimip_cutouts_to_climate_timeseries missing required "
            f"variable(s) {missing}; got {sorted(cutouts_by_var)}. "
            f"Required: {list(_REQUIRED_VARS)}."
        )

    source_calendar = calendar_for_gcm(gcm_source)

    # Calendar-convert every variable to the standard gregorian calendar;
    # capture the (single, calendar-type-level) limitation for disclosure.
    converted: Dict[str, Any] = {}
    limitation_key: Optional[str] = None
    limitation_value: Optional[str] = None
    for variable, array in cutouts_by_var.items():
        result = convert_to_gregorian(array, source_calendar=source_calendar)
        converted[variable] = result.data
        if result.applies_limitation():
            limitation_key = result.limitation_key
            limitation_value = result.limitation_value

    record_dates = [
        _as_date(value)
        for value in np.asarray(converted["tasmax"]["time"].values)
    ]
    has_humidity = _HUMIDITY_VAR in converted
    source = source_label or f"ISIMIP3b_{gcm_source}"

    series: Dict[int, ClimateTimeSeries] = {}
    for cell in cells:
        tmax_c = temperature_kelvin_to_celsius(_sample_cell(converted["tasmax"], cell))
        tmin_c = temperature_kelvin_to_celsius(_sample_cell(converted["tasmin"], cell))
        precip_mm = pr_kg_m2_s_to_mm_day(_sample_cell(converted["pr"], cell))
        srad_mj = rsds_w_m2_to_kj_m2_day(_sample_cell(converted["rsds"], cell)) * _KJ_TO_MJ
        humidity = _sample_cell(converted[_HUMIDITY_VAR], cell) if has_humidity else None

        records = []
        for index, record_date in enumerate(record_dates):
            tmax_value = float(tmax_c[index])
            tmin_value = float(tmin_c[index])
            tmean_value = (tmax_value + tmin_value) / 2.0
            rh_value: Optional[float] = None
            tdew_value: Optional[float] = None
            if humidity is not None:
                humidity_pct = float(humidity[index])
                # Only a valid RH yields humidity + dewpoint; an out-of-range
                # value (e.g. a 360-day interpolation overshoot) is invalid
                # input and propagates missing rather than fabricating either.
                if 0.0 < humidity_pct <= 100.0:
                    rh_value = humidity_pct
                    try:
                        tdew_value = derive_tdew(tmean_value, humidity_pct)
                    except ValueError:
                        # Corrupt temperature for this record → propagate-missing
                        # the dewpoint; rh stays as the valid observed value.
                        tdew_value = None
            records.append(
                ClimateRecord(
                    date=record_date,
                    tmax=tmax_value,
                    tmin=tmin_value,
                    precip=float(precip_mm[index]),
                    srad=float(srad_mj[index]),
                    rh=rh_value,
                    tdew=tdew_value,
                )
            )

        metadata: Dict[str, Any] = {}
        if limitation_key is not None:
            metadata[CALENDAR_LIMITATION_KEY_FIELD] = limitation_key
            metadata[CALENDAR_LIMITATION_VALUE_FIELD] = limitation_value
        if has_humidity:
            metadata[DEWPOINT_POLICY_FIELD] = _DEWPOINT_POLICY_DESCRIPTION
        series[cell.cell_id] = ClimateTimeSeries(
            location_id=cell.cell_id,
            lat=cell.lat,
            lon=cell.lon,
            source=source,
            records=records,
            metadata=metadata,
        )
    return series


def _sample_cell(array: Any, cell: GridCell) -> np.ndarray:
    """Nearest-neighbour sample a ``(time, lat, lon)`` DataArray at a cell."""
    lat_name, lon_name = _latlon_names(array)
    selected = array.sel({lat_name: cell.lat, lon_name: cell.lon}, method="nearest")
    return np.asarray(selected.values)


def _latlon_names(array: Any) -> Tuple[str, str]:
    """Resolve the latitude / longitude coordinate names on a DataArray."""
    coords = set(getattr(array, "coords", {}))
    lat_name = "lat" if "lat" in coords else "latitude"
    lon_name = "lon" if "lon" in coords else "longitude"
    return lat_name, lon_name


def _as_date(value: Any) -> date:
    """Convert an xarray time scalar (datetime64 or cftime) to a ``date``."""
    if isinstance(value, np.datetime64):
        converted = value.astype("datetime64[D]").astype(object)
        return date(converted.year, converted.month, converted.day)
    # cftime datetimes (and python datetimes) expose year / month / day.
    return date(int(value.year), int(value.month), int(value.day))


__all__ = [
    "isimip_cutouts_to_climate_timeseries",
    "CALENDAR_LIMITATION_KEY_FIELD",
    "CALENDAR_LIMITATION_VALUE_FIELD",
    "DEWPOINT_POLICY_FIELD",
]
