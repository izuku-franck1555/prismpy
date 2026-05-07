"""ISIMIP3b version pin — canonical source for the 12 substrate dimensions.

Sprint G CC-G-6 closes the byte-identical-reproducibility gap for the
scenario-package generator. Every dimension that could silently change
under our feet between an ISIMIP point release and our next regeneration
gets pinned here in one module. Dr. Kofi audit-grade reproducibility
depends on each value being:

* declared in source (so the diff surfaces upgrade intent), and
* asserted against the live API at sprint Gate B + nightly cron via
  the drift-detection structural pin (AC-G-2 / §2 of
  ``SPRINT-G-VERIFICATION-STRATEGY.md`` Tier 4).

Per durable lesson #24 canonical-source-or-pin: every consumer (the
ISIMIP3b client, the cached_cutout helper, the AC-G-7a/7b/7c/7d
calendar/unit conversion paths, the AC-G-11 bias-correction provenance
string) imports the constants from this module rather than restating
them. A future ISIMIP point release that changes any one of the 12
dimensions surfaces here as one diff line, not as silent drift across
N call sites.

The 12 pinned dimensions:

1. ``SIMULATION_ROUND``  — ``"ISIMIP3b"``
2. ``SCENARIO_PRODUCT_MAP`` — scenario → ISIMIP "product" (ssp585 →
   ``InputData``, ssp245 → ``SecondaryInputData``).
3. ``BIAS_CORRECTION_VERSION`` — ``"ISIMIP3BASD v2.5.0"``.
4. ``REFERENCE_DATASET_VERSION`` — ``"W5E5 v2.0"``.
5. ``CALENDAR_BY_GCM`` — per-GCM calendar convention (5 GCMs).
6. ``CF_VARIABLE_NAMING_CONVENTION`` — ``"CF-1.x"``.
7. ``VARIABLE_UNITS`` — per-variable canonical CF units.
8. ``GRID_RESOLUTION_DEG`` — ``"0.5 deg"`` (ISIMIP3b primary core).
9. ``EXTRACTION_METHOD`` — ``"cutout_bbox"`` (server-side cutout call shape).
10. ``MASK_FILL_CONVENTION`` — ``"_FillValue + sea_mask"``.
11. ``LONGITUDE_CONVENTION`` — ``"-180_to_180"``.
12. ``NETCDF_FORMAT_ENGINE`` — backend stack (xarray + netCDF4 +
    cftime versions).

Plus two derived sets the ISIMIP3b client validates against:

* ``PRIMARY_GCMS`` — the 5 GCMs in the primary core ensemble.
* ``SUPPORTED_VARIABLES`` — the 6 CF-1.x daily variables Sprint G ships.

The drift-detection structural pin lives in
``tests/structural/test_isimip_versions_pin.py`` (AC-G-2 commit). It
runs in two tracks per evaluator MED-Eval-2:

* Mocked-fixture default (every push, no network) — asserts the
  module constants match a captured ``client.datasets()`` fixture.
* Real-API opt-in (``pytest -m network`` / sprint Gate B / nightly
  cron) — asserts the module constants still match the live ISIMIP
  API for all 5 GCMs × 2 SSPs × 6 variables × 2 time-slices.
"""

from __future__ import annotations

from typing import Final, FrozenSet, Mapping, Tuple


# ── 1. simulation_round ──────────────────────────────────────────────

SIMULATION_ROUND: Final[str] = "ISIMIP3b"


# ── 2. scenario → product mapping ────────────────────────────────────

SCENARIO_PRODUCT_MAP: Final[Mapping[str, str]] = {
    "ssp585": "InputData",
    "ssp245": "SecondaryInputData",
}


# ── 3. bias-correction version ───────────────────────────────────────

BIAS_CORRECTION_VERSION: Final[str] = "ISIMIP3BASD v2.5.0"


# ── 4. reference dataset version ─────────────────────────────────────

REFERENCE_DATASET_VERSION: Final[str] = "W5E5 v2.0"


# ── 5. calendar convention per GCM ───────────────────────────────────
#
# Each entry pins the upstream calendar shape so AC-G-7d's calendar
# converter knows ahead of time which conversion semantics apply.
# A drift here means an ISIMIP point release silently switched a GCM
# from one calendar to another — Dr. Kofi's audit trail must catch it.

CALENDAR_BY_GCM: Final[Mapping[str, str]] = {
    "gfdl-esm4": "noleap",
    "ipsl-cm6a-lr": "gregorian",
    "mpi-esm1-2-hr": "gregorian",
    "mri-esm2-0": "gregorian",
    "ukesm1-0-ll": "360_day",
}


# ── 6. CF variable naming convention ─────────────────────────────────

CF_VARIABLE_NAMING_CONVENTION: Final[str] = "CF-1.x"


# ── 7. variable units ────────────────────────────────────────────────
#
# Per-variable canonical CF units. AC-G-7a/7b/7c convert FROM these
# to the platform-specific output units. A drift here means an ISIMIP
# point release changed the upstream unit and AC-G-7's conversion
# math would silently produce wrong values.

VARIABLE_UNITS: Final[Mapping[str, str]] = {
    "rsds": "W m-2",
    "tasmax": "K",
    "tasmin": "K",
    "pr": "kg m-2 s-1",
    "hurs": "%",
    "sfcWind": "m s-1",
}


# ── 8. grid resolution ───────────────────────────────────────────────

GRID_RESOLUTION_DEG: Final[str] = "0.5 deg"


# ── 9. extraction method (call shape) ────────────────────────────────

EXTRACTION_METHOD: Final[str] = "cutout_bbox"


# ── 10. mask + fill convention ───────────────────────────────────────

MASK_FILL_CONVENTION: Final[str] = "_FillValue + sea_mask"


# ── 11. longitude convention ─────────────────────────────────────────
#
# West Africa bboxes use west < east in this convention. A drift to
# 0_to_360 would silently mis-translate every Africa-region bbox at
# the boundary.

LONGITUDE_CONVENTION: Final[str] = "-180_to_180"


# ── 12. netCDF format engine + cftime ────────────────────────────────
#
# The backend library stack that parses + converts the cached netCDF.
# AC-G-7d's calendar conversion semantics are deterministic only if
# this stack is pinned and asserted via the structural pin.

NETCDF_FORMAT_ENGINE: Final[Mapping[str, str]] = {
    "netcdf4": ">=1.6.0,<2.0",  # matches pyproject.toml runtime pin
    "xarray": ">=2023.1,<2025.0",  # matches pyproject.toml runtime pin
    "cftime": ">=1.6.0,<2.0",  # transitive via netCDF4; pin captured here per LOW-Pass4-3
}


# ── Derived: primary GCM ensemble + supported variable allowlist ─────


PRIMARY_GCMS: Final[FrozenSet[str]] = frozenset(CALENDAR_BY_GCM.keys())
"""The 5 GCMs in the ISIMIP3b primary core ensemble."""


SUPPORTED_VARIABLES: Final[FrozenSet[str]] = frozenset(VARIABLE_UNITS.keys())
"""The 6 CF-1.x daily variables Sprint G ships."""


# ── Derived: time-slice ensemble ─────────────────────────────────────


SCENARIO_TIME_SLICES: Final[Tuple[Tuple[int, int], ...]] = (
    (2046, 2065),
    (2086, 2100),
)
"""ISIMIP3b primary scenario time-slice ensemble (mid-century + end-century)."""


# ── ISIMIP → SARRA-Py variable / unit mapping (canonical) ────────────
#
# AC-G-7c boundary 7/7 absorption per codex round 2 P1:
# the SARRA-Py projection writer was emitting raw ISIMIP CF
# variable names (``tasmax`` / ``pr`` / etc) in raw CF units
# (Kelvin / kg m⁻² s⁻¹ / W m⁻²). Existing SARRA-Py consumers
# (``_copy_climate_geotiffs`` + the per-cell sampling +
# validation paths) scan AgERA5-style directory names + expect
# converted units (Kelvin passthrough for temperature; mm/day
# for precipitation; J/m²/day for solar radiation). The
# mapping below IS the canonical ISIMIP → SARRA-Py
# vocabulary + units alignment; per durable §24
# canonical-source-or-pin: every consumer routes through this
# table, no inline restatement.
#
# Coverage: 4 of 6 SARRA-Py expected climate directories —
# ``rainfall`` (← ``pr``), ``2m_temperature_24_hour_maximum``
# (← ``tasmax``), ``2m_temperature_24_hour_minimum``
# (← ``tasmin``), ``solar_radiation_flux_daily`` (← ``rsds``).
# The remaining 2 (``2m_temperature_24_hour_mean``,
# ``ET0Hargeaves``) require derivation (mean = average of
# tmax+tmin; ET0 = Hargreaves-Samani from temperature +
# extraterrestrial radiation) — declared in
# ``manifest.limitations.sarra_py_projection_derivations_pending``
# per Sprint G structural-pin scope; full derivation lands in
# Sprint H+ end-to-end pipeline per team-lead authorization
# 2026-05-07 (no preemptive H+ work in boundary 7/7
# absorption).


_KG_PER_M2_PER_SECOND_TO_MM_PER_DAY: Final[float] = 86400.0
"""Conversion factor: 1 kg m⁻² s⁻¹ × 86400 s/day = 86.4 mm/day.

ISIMIP ``pr`` (precipitation) is shipped in kg m⁻² s⁻¹ per CF-1.x
convention. SARRA-Py's ``rainfall`` directory expects mm/day. 1 kg
of water spread over 1 m² is 1 mm depth, and 1 day = 86400 s, so
the numeric factor is 86400.
"""


_W_PER_M2_TO_J_PER_M2_PER_DAY: Final[float] = 86400.0
"""Conversion factor: 1 W m⁻² × 86400 s/day = 86400 J m⁻²/day.

ISIMIP ``rsds`` (downwelling shortwave at surface) is shipped in
W m⁻² (instantaneous power, daily-averaged). SARRA-Py's
``solar_radiation_flux_daily`` expects accumulated daily energy in
J m⁻²/day, matching the AgERA5 convention. 1 W m⁻² × 86400 s/day =
86400 J m⁻²/day.
"""


ISIMIP_TO_SARRA_VAR_MAPPING: Final[
    Mapping[str, Tuple[str, str, str]]
] = {
    # ISIMIP CF name → (SARRA-Py directory name, source unit, target unit)
    "pr": (
        "rainfall",
        "kg m-2 s-1",
        "mm/day",
    ),
    "tasmax": (
        "2m_temperature_24_hour_maximum",
        "K",
        "K",  # passthrough — AgERA5 + SARRA-Py both consume K
    ),
    "tasmin": (
        "2m_temperature_24_hour_minimum",
        "K",
        "K",
    ),
    "rsds": (
        "solar_radiation_flux_daily",
        "W m-2",
        "J m-2 day-1",
    ),
}
"""Canonical ISIMIP CF variable → SARRA-Py directory name + unit mapping.

Per durable §24 canonical-source-or-pin: every consumer that needs
to translate an ISIMIP3b variable to a SARRA-Py output directory +
unit imports this dict. The structural pin
``tests/structural/test_isimip_to_sarra_mapping.py`` asserts the
SARRA-Py projection writer emits ONLY directory names from this
table's values, never raw ISIMIP CF names directly.

Each tuple value: ``(sarra_directory_name, source_unit, target_unit)``.
The unit fields are self-documenting; the conversion factors live in
``prismpy.harmonize.isimip_unit_conversions`` so the math is in one
place + version-pinned.
"""


SARRA_PY_DERIVED_VARIABLE_DIRECTORIES: Final[FrozenSet[str]] = frozenset(
    {
        # tasmean = (tasmax + tasmin) / 2 — trivial derivation
        "2m_temperature_24_hour_mean",
        # ET0 = Hargreaves-Samani from tasmax + tasmin + Ra(latitude, DOY)
        "ET0Hargeaves",
    }
)
"""SARRA-Py expected directories that require derivation from ISIMIP variables.

These don't appear in :data:`ISIMIP_TO_SARRA_VAR_MAPPING` because
they're not directly downloaded — the values must be computed from
other variables. Sprint G boundary 7/7 absorption defers the
derivation to Sprint H+ end-to-end pipeline; the projection package
declares the deferral via ``manifest.limitations.sarra_py_projection_derivations_pending``.
"""


__all__ = [
    "SIMULATION_ROUND",
    "SCENARIO_PRODUCT_MAP",
    "BIAS_CORRECTION_VERSION",
    "REFERENCE_DATASET_VERSION",
    "CALENDAR_BY_GCM",
    "CF_VARIABLE_NAMING_CONVENTION",
    "VARIABLE_UNITS",
    "GRID_RESOLUTION_DEG",
    "EXTRACTION_METHOD",
    "MASK_FILL_CONVENTION",
    "LONGITUDE_CONVENTION",
    "NETCDF_FORMAT_ENGINE",
    "PRIMARY_GCMS",
    "SUPPORTED_VARIABLES",
    "SCENARIO_TIME_SLICES",
    "ISIMIP_TO_SARRA_VAR_MAPPING",
    "SARRA_PY_DERIVED_VARIABLE_DIRECTORIES",
]
