"""Deterministic calendar conversion for ISIMIP3b daily climate data.

Sprint G AC-G-7d: the projection-climate pipeline emits every cutout
on the standard gregorian calendar so downstream platforms
(CRAFT/PYTHIA WTH writers + ACEA pickle path + SARRA-Py GeoTIFF path)
all receive the same shape. The handling is DATA-DRIVEN — keyed off
the calendar the data ACTUALLY carries, not the GCM's nominal native
calendar:

* The retrieved ISIMIP3b W5E5 bias-adjusted product is delivered on a
  standard ``proleptic_gregorian`` calendar for EVERY GCM (complete
  365/366-day years, no missing steps). So for the UC2 ensemble this
  pipeline performs ZERO calendar conversion: it passes the data
  through unchanged — no Feb-29 drop, no gap insertion, no
  interpolation by us.
* For GCMs whose NATIVE calendar is non-standard (GFDL-ESM4 noleap;
  UKESM1-0-LL 360_day) the calendar was harmonised UPSTREAM by ISIMIP
  (linear time interpolation prior to bias adjustment, then
  quantile-mapped) — NOT by this pipeline. That upstream provenance is
  disclosed honestly via the ``calendar_harmonization`` note
  (:data:`LIMITATION_KEY_CALENDAR_HARMONIZATION`). Gregorian-native
  GCMs (IPSL-CM6A-LR / MPI-ESM1-2-HR / MRI-ESM2-0) need no calendar
  disclosure at all.
* The genuine ``noleap`` → drop-Feb-29 and ``360_day`` →
  insert+interpolate branches are RETAINED for a hypothetical source
  actually delivered on a non-standard calendar; they fire ONLY when
  the DETECTED data calendar is non-gregorian, and their
  ``calendar_noleap_dropped_feb29`` / ``calendar_360_day_resampled``
  limitations then describe a real, local manipulation. They do NOT
  fire for the ISIMIP3b W5E5 product.

Determinism contract per CC-G-7: cftime + xarray versions pinned at
module level. The CC-G-6 12-dimension drift-detection pin
(``prismpy.standards.isimip_versions.NETCDF_FORMAT_ENGINE``) carries
the same pin in canonical form; this module's ``EXPECTED_BACKEND``
re-exports the relevant subset so a single-source check at conversion
time fires loud if the runtime stack drifts away from the pin.

Honest-signal contract (``feedback_no_data_cooking.md``): a calendar
limitation is recorded ONLY when this pipeline actually drops or
interpolates data — never stamped for a pass-through. The
``calendar_harmonization`` note records UPSTREAM (ISIMIP) harmonisation
provenance without falsely claiming a local conversion. Audit grep on
``manifest.limitations.<key>`` therefore reflects what really happened.

Per durable lesson §24 canonical-source-or-pin: the GCM → calendar
map is imported from ``prismpy.standards.isimip_versions``. This
module does NOT redefine the table; it RE-PURPOSES the native-calendar
map as an upstream-harmonisation signal (see ``convert_to_gregorian``).
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
"""Manifest limitations key for a GENUINE noleap → gregorian conversion.

Records that the *retrieved* data was actually on a 365-day (noleap)
calendar (Feb 29 absent), so this pipeline's gregorian output has no
value for Feb 29 in leap years. NOTE: the ISIMIP3b W5E5 bias-adjusted
product is delivered already-gregorian for every GCM — INCLUDING
GFDL-ESM4 (nominally noleap) — so this key does NOT fire for the UC2
ensemble: gfdl-esm4 is gregorian-delivered and gets the
:data:`LIMITATION_KEY_CALENDAR_HARMONIZATION` note, with NO Feb-29
drop. This key is retained only for a hypothetical genuinely-noleap
retrieved source; per `feedback_no_data_cooking.md` it is then the
audit signal for that real Feb-29 drop.
"""


LIMITATION_KEY_360_RESAMPLED: str = "calendar_360_day_resampled"
"""Manifest limitations key for a GENUINE 360-day → gregorian conversion.

Records that the *retrieved* data was actually on a 360-day calendar
and this pipeline interpolated the 5-6 gap days. NOTE: the ISIMIP3b
W5E5 bias-adjusted product is delivered already-gregorian for every
GCM, so this key does NOT fire for the UC2 ensemble — it is retained
only for a hypothetical genuinely-360_day source. Calendar provenance
for the ISIMIP3b product is recorded under
:data:`LIMITATION_KEY_CALENDAR_HARMONIZATION` instead.
"""


LIMITATION_KEY_CALENDAR_HARMONIZATION: str = "calendar_harmonization"
"""Manifest provenance key for ISIMIP3b UPSTREAM calendar harmonisation.

Emitted for GCMs whose NATIVE calendar is non-standard (noleap /
360_day) when the delivered cutout is already standard gregorian —
i.e. ISIMIP harmonised the calendar upstream (before bias adjustment),
not this pipeline. This is a provenance NOTE, not a pipeline
limitation: it credits the upstream harmonisation honestly rather
than silently dropping the fact. Gregorian-native GCMs
(IPSL/MPI/MRI) emit no calendar key at all. Specialist-validated
against the ISIMIP3b bias-adjustment fact sheet (Lange 2019, 2021).
"""


_CALENDAR_HARMONIZATION_NOTE: str = (
    "Climate forcing is ISIMIP3b bias-adjusted and statistically downscaled "
    "CMIP6 output (ISIMIP3BASD v2.5; Lange 2019, 2021), delivered on the "
    "proleptic Gregorian calendar for all GCMs. No calendar conversion was "
    "applied in this pipeline; inputs are complete daily series (no missing "
    "steps, no gap-filling by us). Calendar harmonisation for GCMs with "
    "non-standard native calendars (GFDL-ESM4: 365_day/noleap; UKESM1-0-LL: "
    "360_day) was performed upstream by ISIMIP via linear time interpolation "
    "prior to bias adjustment; harmonised days are therefore quantile-mapped, "
    "not raw interpolations."
)
"""Specialist-validated machine disclosure for ``calendar_harmonization``.

Verbatim per the crop-modeling-specialist's validation against the
ISIMIP3b bias-adjustment fact sheet (Lange 2019, 2021). Do not
paraphrase — the paper-grade variant lives in the methods section.
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
    """Emit ``climate_data`` on the standard gregorian calendar, keyed off
    the calendar the data ACTUALLY carries (not the nominal native one).

    Behaviour by the DETECTED data calendar:

    * Already ``gregorian`` / ``standard`` / ``proleptic_gregorian`` (the
      ISIMIP3b W5E5 delivery for every GCM) → pass through UNCHANGED, no
      conversion. If the declared native ``source_calendar`` is
      non-standard, attach the ``calendar_harmonization`` provenance note
      crediting ISIMIP's UPSTREAM harmonisation; if it is gregorian, no
      limitation at all.
    * Genuinely ``noleap`` / ``365_day`` → drop Feb 29 (no source value)
      and return the ``calendar_noleap_dropped_feb29`` limitation.
    * Genuinely ``360_day`` → ``convert_calendar('standard',
      align_on='year', missing=np.nan)`` + ``.interpolate_na`` to fill
      the 5-6 gap days, returning ``calendar_360_day_resampled``.

    ``source_calendar`` is validated as a known identifier and used to
    decide the upstream-harmonisation note; the conversion path itself is
    chosen by the DETECTED calendar, so a drop/interpolate limitation is
    recorded only when a real, local manipulation actually occurred (the
    ISIMIP3b product never triggers one).

    Args:
        climate_data: An xarray Dataset (or DataArray) with a time
            dimension already aligned to the source calendar.
        source_calendar: The GCM's DECLARED native calendar. Pass
            ``CALENDAR_BY_GCM[gcm]`` from
            :mod:`prismpy.standards.isimip_versions` for the GCM the
            data was retrieved against; the function normalises common
            aliases (``"365_day"`` → noleap, etc.). It is validated and
            used to decide the upstream-harmonisation note — it does NOT
            select the conversion path (the DETECTED data calendar does).
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
    _known = _NOLEAP_CALENDARS | _360_DAY_CALENDARS | _GREGORIAN_CALENDARS
    if normalised not in _known:
        raise ValueError(
            f"Unrecognised source calendar {source_calendar!r}. "
            f"Sprint G supports {sorted(_known)}; extend "
            "convert_to_gregorian + isimip_versions.CALENDAR_BY_GCM "
            "atomically when adding a new GCM."
        )

    # DATA-DRIVEN calendar handling (root-cause fix): branch on the ACTUAL
    # calendar the cutout carries, NOT the GCM's nominal/native calendar. The
    # ISIMIP3b W5E5 bias-adjusted product is delivered on a standard gregorian
    # calendar for ALL GCMs (complete 365/366-day years), so a noleap/360_day
    # NATIVE label does NOT mean the retrieved data is on that calendar.
    # Stamping a "dropped Feb 29" / "interpolated gaps" limitation on data that
    # was never manipulated is a FALSE disclosure; keying off the actual
    # calendar guarantees a limitation is recorded only when a real conversion
    # happened, so the manifest disclosure always matches the data.
    actual = _detect_calendar(climate_data, time_dim)

    if actual in _GREGORIAN_CALENDARS:
        # Already standard-calendar + complete: this pipeline applies ZERO
        # calendar conversion (no drop, no gap-fill). The declared native
        # calendar (CALENDAR_BY_GCM) is RE-PURPOSED here: it no longer means
        # "we convert" — it records whether ISIMIP harmonised the calendar
        # UPSTREAM (before bias adjustment). For non-standard-native GCMs
        # (noleap / 360_day) ISIMIP inserted the missing days upstream via
        # linear time interpolation, then quantile-mapped them — so the
        # delivered leap days are real (quantile-mapped), not raw midpoints.
        # Emit a provenance NOTE crediting that upstream harmonisation
        # (Option c — honesty, not silence). Gregorian-native GCMs
        # (IPSL/MPI/MRI) had nothing harmonised anywhere → no calendar key.
        if normalised in _GREGORIAN_CALENDARS:
            limitation_key: Optional[str] = None
            limitation_value: Optional[str] = None
        else:
            limitation_key = LIMITATION_KEY_CALENDAR_HARMONIZATION
            limitation_value = _CALENDAR_HARMONIZATION_NOTE
        return CalendarConversionResult(
            data=climate_data,
            limitation_key=limitation_key,
            limitation_value=limitation_value,
            source_calendar=source_calendar,
            target_calendar="standard",
        )

    if actual in _NOLEAP_CALENDARS:
        # GENUINE noleap DATA only — the DETECTED calendar is noleap. The
        # ISIMIP3b W5E5 product is delivered gregorian, so this branch does
        # NOT run for the UC2 GCMs; it is the correct path for a hypothetical
        # source actually delivered on a 365-day calendar.
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

    if actual in _360_DAY_CALENDARS:
        # GENUINE 360_day DATA only — the DETECTED calendar is 360_day. The
        # ISIMIP3b W5E5 product is delivered gregorian, so this branch does
        # NOT run for the UC2 GCMs (UKESM1-0-LL included); it is the correct
        # path for a hypothetical source actually delivered on a 360-day
        # calendar.
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
        f"Retrieved data carries an unrecognised calendar {actual!r} "
        f"(declared source {source_calendar!r}). Sprint G supports "
        f"{sorted(_known)}; extend convert_to_gregorian + "
        "isimip_versions.CALENDAR_BY_GCM atomically when adding a new GCM."
    )


def _detect_calendar(climate_data: Any, time_dim: str) -> str:
    """Return the ACTUAL calendar the data carries (lower-cased).

    Reads the CF calendar from the time coordinate's ``.dt.calendar``
    accessor: cftime indices report ``noleap`` / ``360_day`` / etc.,
    while numpy ``datetime64`` indices report ``proleptic_gregorian``.
    This is the calendar the data ACTUALLY uses — which, for the
    ISIMIP3b W5E5 bias-adjusted product, is standard gregorian for
    every GCM regardless of the GCM's nominal native calendar. Falls
    back to ``proleptic_gregorian`` when a plain ``datetime64`` coord
    exposes no calendar attribute.

    Raises:
        ValueError: if ``time_dim`` is absent from ``climate_data``.
    """
    try:
        coord = climate_data[time_dim]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"convert_to_gregorian: time dimension {time_dim!r} not found "
            f"on the input; got dims {getattr(climate_data, 'dims', None)}."
        ) from exc
    cal = getattr(getattr(coord, "dt", None), "calendar", None)
    if cal is None:
        # A datetime64 coordinate without a cftime calendar attribute is
        # proleptic_gregorian by numpy definition.
        cal = "proleptic_gregorian"
    return str(cal).strip().lower()


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
