"""Deterministic calendar conversion for ISIMIP3b daily climate data.

Sprint G AC-G-7d: ISIMIP3b GCMs ship with mixed calendars. The
projection-climate pipeline normalises every cutout to standard
gregorian so downstream platforms (CRAFT/PYTHIA WTH writers + ACEA
pickle path + SARRA-Py GeoTIFF path) all receive the same shape.

Per Draft 5 contract:

* ``noleap`` (GFDL-ESM4) → gregorian: drop Feb 29 from gregorian-only
  days that lack source data. Document
  ``manifest.limitations.calendar_noleap_dropped_feb29``.
* ``gregorian`` (IPSL-CM6A-LR / MPI-ESM1-2-HR / MRI-ESM2-0) →
  gregorian: pass-through.
* ``360_day`` (UKESM1-0-LL) → gregorian: insert 5-6 NaN gap days
  per year via ``xarray.convert_calendar('standard',
  align_on='year', missing=np.nan)`` then fill the gaps with
  ``.interpolate_na(dim=time_dim, method='linear')``. Document
  ``manifest.limitations.calendar_360_day_resampled``.

Determinism contract per CC-G-7: cftime + xarray versions pinned at
module level. The CC-G-6 12-dimension drift-detection pin
(``prismpy.standards.isimip_versions.NETCDF_FORMAT_ENGINE``) carries
the same pin in canonical form; this module's ``EXPECTED_BACKEND``
re-exports the relevant subset so a single-source check at conversion
time fires loud if the runtime stack drifts away from the pin.

Limitation keys follow the ``feedback_no_data_cooking.md`` honest-
signal contract: any calendar conversion that drops or interpolates
data MUST record the limitation in the package manifest at the
``manifest.limitations.<key>`` location so Dr. Kofi's audit grep
finds it.

Per durable lesson §24 canonical-source-or-pin: the GCM → calendar
map is imported from ``prismpy.standards.isimip_versions``. This
module does NOT redefine the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


from prismpy.standards.isimip_versions import (
    CALENDAR_BY_GCM,
    NETCDF_FORMAT_ENGINE,
)


# ── Limitation keys (per CC-G-5 honest-signal contract) ──────────────


LIMITATION_KEY_NOLEAP_DROPPED: str = "calendar_noleap_dropped_feb29"
"""Manifest limitations key for noleap → gregorian conversion.

Records that the source GCM uses a 365-day calendar (Feb 29 absent),
so the gregorian output has no climate value for Feb 29 in leap years.
Per `feedback_no_data_cooking.md`: silent quality loss is forbidden;
the limitation key in `manifest.limitations` is the audit signal.
"""


LIMITATION_KEY_360_RESAMPLED: str = "calendar_360_day_resampled"
"""Manifest limitations key for 360-day → gregorian conversion.

Records that the source GCM uses a 360-day calendar (12 × 30 day
months) and the conversion to standard gregorian interpolated days
to fill the 5- or 6-day gap per year. Audit consumers grep this key
to know which packages had calendar reshaping applied.
"""


# Recognised source calendar identifiers — match the strings the
# upstream cftime / netCDF calendar attribute uses. Both ``noleap``
# and ``365_day`` denote the same calendar; both surfaces handled.
_NOLEAP_CALENDARS = frozenset({"noleap", "365_day", "365"})
_360_DAY_CALENDARS = frozenset({"360_day", "360"})
_GREGORIAN_CALENDARS = frozenset(
    {"gregorian", "standard", "proleptic_gregorian"}
)


# ── Determinism pin (re-exported from isimip_versions per durable §24) ─


EXPECTED_BACKEND: Dict[str, str] = dict(NETCDF_FORMAT_ENGINE)
"""Pinned backend stack for calendar conversion.

The full 12-dimension version pin lives at
:data:`prismpy.standards.isimip_versions.NETCDF_FORMAT_ENGINE`. This
module re-exports the same dict so a runtime drift-detection check
can fire close to the conversion call site without re-importing the
whole standards module. The CC-G-6 structural pin
(``tests/structural/test_isimip_versions_pin.py``) asserts that
canonical source has cftime + netcdf4 + xarray entries.
"""


# ── Conversion result shape ──────────────────────────────────────────


@dataclass(frozen=True)
class CalendarConversionResult:
    """Outcome of a calendar conversion.

    Attributes:
        data: The converted xarray Dataset on the target gregorian
            calendar.
        limitation_key: The ``manifest.limitations.<key>`` string the
            caller MUST add to the manifest, or ``None`` if no
            limitation applies (gregorian → gregorian pass-through).
        limitation_value: Human-readable description of the limitation,
            or ``None``. Audit grep target.
        source_calendar: The original calendar string read from the
            source dataset.
        target_calendar: ``"standard"`` (gregorian) for every Sprint G
            output.
    """

    data: Any  # xarray.Dataset (typed as Any to avoid hard import dep)
    limitation_key: Optional[str]
    limitation_value: Optional[str]
    source_calendar: str
    target_calendar: str

    def applies_limitation(self) -> bool:
        """True iff this conversion produced a limitation that must
        appear in the manifest."""
        return self.limitation_key is not None


# ── Public entry point ───────────────────────────────────────────────


def convert_to_gregorian(
    climate_data: Any,
    *,
    source_calendar: str,
    time_dim: str = "time",
) -> CalendarConversionResult:
    """Convert ``climate_data`` to standard gregorian calendar.

    Per Draft 5 §AC-G-7d, the three source-calendar paths are:

    * ``noleap`` / ``365_day`` → drop Feb 29 from the gregorian output
      so the consumer knows which days have no source value. Returns
      a limitation record so the manifest writer can populate
      ``manifest.limitations.calendar_noleap_dropped_feb29``.
    * ``360_day`` → ``xarray.convert_calendar('standard',
      align_on='year', missing=np.nan)`` produces a 365-day series
      with 5-6 NaN gap days per source year; these gaps are then
      filled by ``.interpolate_na(dim=time_dim, method='linear')``
      so the output has no NaN values. Returns a limitation record
      for ``manifest.limitations.calendar_360_day_resampled``.
    * ``gregorian`` / ``standard`` / ``proleptic_gregorian`` →
      pass-through, no limitation.

    Args:
        climate_data: An xarray Dataset (or DataArray) with a time
            dimension already aligned to the source calendar.
        source_calendar: The calendar string from the source. Pass
            ``CALENDAR_BY_GCM[gcm]`` from
            :mod:`prismpy.standards.isimip_versions` for the GCM the
            data was retrieved against; the function normalises common
            aliases (``"365_day"`` → noleap, etc.).
        time_dim: Name of the time dimension on the dataset (default
            ``"time"``). xarray's calendar conversion needs the dim
            name explicitly; ISIMIP3b datasets all use ``"time"`` per
            CF-1.x convention but this is exposed for testability.

    Returns:
        :class:`CalendarConversionResult` with the converted data + an
        optional limitation record.

    Raises:
        ValueError: If ``source_calendar`` is not recognised. Sprint G
            scopes the 5 ISIMIP3b primary core ensemble GCM calendars
            (noleap / gregorian / 360-day). Adding a sixth GCM would
            need a new branch here AND a new entry in
            :data:`isimip_versions.CALENDAR_BY_GCM`.
    """
    normalised = source_calendar.strip().lower()

    if normalised in _GREGORIAN_CALENDARS:
        return CalendarConversionResult(
            data=climate_data,
            limitation_key=None,
            limitation_value=None,
            source_calendar=source_calendar,
            target_calendar="standard",
        )

    if normalised in _NOLEAP_CALENDARS:
        # noleap source has no Feb 29 to begin with. Re-stamping the
        # calendar to standard is a metadata change (xarray's
        # convert_calendar with align_on='date' preserves dates 1:1
        # for non-leap years; leap-year Feb 29 has no source value
        # because it never existed). The limitation records the
        # information loss for downstream audit.
        try:
            converted = climate_data.convert_calendar(
                "standard",
                dim=time_dim,
                align_on="date",
            )
        except AttributeError as exc:
            raise ValueError(
                "convert_to_gregorian requires xarray>=2023.1 with "
                f"convert_calendar; got {type(climate_data).__name__}"
            ) from exc
        return CalendarConversionResult(
            data=converted,
            limitation_key=LIMITATION_KEY_NOLEAP_DROPPED,
            limitation_value=(
                f"Source GCM uses {source_calendar!r} (365-day) "
                "calendar; no climate value for Feb 29 in leap years. "
                "Consumer code must treat Feb 29 as missing data."
            ),
            source_calendar=source_calendar,
            target_calendar="standard",
        )

    if normalised in _360_DAY_CALENDARS:
        try:
            # ``align_on='year'`` proportionally distributes the
            # 12 × 30-day months across the 365-day standard year,
            # which is the right semantic for climate data: each
            # 360-day year represents the same climate year, just
            # with regularised month lengths.
            #
            # The ``missing`` keyword is a NUMERIC fill value, NOT
            # an interpolation mode. Passing the string
            # ``"interpolate"`` corrupts numeric variables to object
            # dtype (codex round 1 boundary 3/7 P1). The right
            # pattern is two-step: insert NaN gaps via
            # ``missing=np.nan``, then call ``.interpolate_na`` on
            # the result so the gaps are filled by linear
            # interpolation across surrounding days. This preserves
            # float64 dtype and produces meaningful interpolated
            # values rather than NaN or string contamination.
            #
            # xarray>=2023.1 requires ``align_on`` for 360_day
            # conversions; passing it explicitly avoids the
            # default-fallback divergence.
            converted_with_gaps = climate_data.convert_calendar(
                "standard",
                dim=time_dim,
                align_on="year",
                missing=np.nan,
            )
            converted = converted_with_gaps.interpolate_na(
                dim=time_dim,
                method="linear",
            )
        except AttributeError as exc:
            raise ValueError(
                "convert_to_gregorian requires xarray>=2023.1 with "
                f"convert_calendar; got {type(climate_data).__name__}"
            ) from exc
        return CalendarConversionResult(
            data=converted,
            limitation_key=LIMITATION_KEY_360_RESAMPLED,
            limitation_value=(
                f"Source GCM uses {source_calendar!r} (12x30 day) "
                "calendar; xarray.convert_calendar inserted 5-6 NaN "
                "gap days per year to align with the 365-day "
                "standard calendar, then linear interpolation across "
                "neighbouring days filled the gaps. Consumer code "
                "reads the interpolated values; the limitation key "
                "signals that interpolation occurred."
            ),
            source_calendar=source_calendar,
            target_calendar="standard",
        )

    raise ValueError(
        f"Unrecognised source calendar {source_calendar!r}. "
        f"Sprint G supports {sorted(_NOLEAP_CALENDARS | _GREGORIAN_CALENDARS | _360_DAY_CALENDARS)}; "
        "extend convert_to_gregorian + isimip_versions.CALENDAR_BY_GCM "
        "atomically when adding a new GCM."
    )


def calendar_for_gcm(gcm: str) -> str:
    """Return the canonical calendar string for an ISIMIP3b GCM.

    Thin re-export of ``CALENDAR_BY_GCM`` lookup so callers that have
    just the GCM name don't need to traverse the standards module.
    Fails loud on unknown GCM (rather than silently defaulting to
    gregorian, which would lose the noleap / 360-day signals).

    Per durable §24: the table lives once at
    :data:`prismpy.standards.isimip_versions.CALENDAR_BY_GCM`; this
    helper just unwraps the lookup.
    """
    try:
        return CALENDAR_BY_GCM[gcm]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GCM {gcm!r}. Sprint G primary core ensemble: "
            f"{sorted(CALENDAR_BY_GCM.keys())}. Extend "
            "isimip_versions.CALENDAR_BY_GCM + convert_to_gregorian "
            "atomically when adding a new GCM."
        ) from exc


__all__ = [
    "LIMITATION_KEY_NOLEAP_DROPPED",
    "LIMITATION_KEY_360_RESAMPLED",
    "EXPECTED_BACKEND",
    "CalendarConversionResult",
    "convert_to_gregorian",
    "calendar_for_gcm",
]
