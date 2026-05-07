"""Structural pin: AC-G-7d calendar conversion semantics.

Sprint G AC-G-7d converts every ISIMIP3b GCM cutout to standard
gregorian calendar. The three source-calendar paths must be
deterministic + version-pinned + signal their information loss
honestly:

* ``noleap`` (GFDL-ESM4) → drop Feb 29 (no source value); record
  ``manifest.limitations.calendar_noleap_dropped_feb29``.
* ``360_day`` (UKESM1-0-LL) → ``xarray.convert_calendar('standard',
  missing='interpolate')``; record
  ``manifest.limitations.calendar_360_day_resampled``.
* ``gregorian`` family (IPSL/MPI/MRI) → pass-through.

Per durable §24 the GCM → calendar table lives ONCE at
:data:`prismpy.standards.isimip_versions.CALENDAR_BY_GCM`. This
module's tests assert the helper imports from that canonical source
rather than redefining it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import cftime
import numpy as np
import pytest
import xarray as xr

from prismpy.harmonize.calendar_conversion import (
    EXPECTED_BACKEND,
    LIMITATION_KEY_360_RESAMPLED,
    LIMITATION_KEY_NOLEAP_DROPPED,
    CalendarConversionResult,
    calendar_for_gcm,
    convert_to_gregorian,
)
from prismpy.standards.isimip_versions import CALENDAR_BY_GCM


# ── Fixture builders ─────────────────────────────────────────────────


def _noleap_dataset(start_year: int = 2046, n_years: int = 1) -> xr.Dataset:
    """Build a synthetic Dataset on the noleap calendar.

    Each year has exactly 365 days (no Feb 29). Variables: a single
    ``tasmax`` field with monotonically increasing values so a drop
    or interpolation is detectable.
    """
    times = []
    for year in range(start_year, start_year + n_years):
        for doy in range(1, 366):  # 1..365
            month, day = _noleap_doy_to_date(doy)
            times.append(cftime.DatetimeNoLeap(year, month, day))
    arr = np.arange(len(times), dtype=float)
    return xr.Dataset(
        {"tasmax": ("time", arr)},
        coords={"time": ("time", times)},
    )


def _noleap_doy_to_date(doy: int) -> tuple[int, int]:
    """Map a 1..365 day-of-year onto a (month, day) tuple in the
    noleap calendar (no Feb 29)."""
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    remaining = doy
    for month_idx, dim in enumerate(days_per_month, start=1):
        if remaining <= dim:
            return month_idx, remaining
        remaining -= dim
    raise ValueError(f"DOY {doy} out of 1..365 range for noleap")


def _three60_day_dataset(start_year: int = 2046, n_years: int = 1) -> xr.Dataset:
    """Build a synthetic Dataset on the 360-day calendar (12 × 30)."""
    times = []
    for year in range(start_year, start_year + n_years):
        for month in range(1, 13):
            for day in range(1, 31):
                times.append(cftime.Datetime360Day(year, month, day))
    arr = np.arange(len(times), dtype=float)
    return xr.Dataset(
        {"tasmax": ("time", arr)},
        coords={"time": ("time", times)},
    )


def _gregorian_dataset(start_year: int = 2046, n_years: int = 1) -> xr.Dataset:
    """Build a synthetic Dataset on the standard gregorian calendar."""
    import pandas as pd

    times = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{start_year + n_years - 1}-12-31",
        freq="D",
    )
    arr = np.arange(len(times), dtype=float)
    return xr.Dataset(
        {"tasmax": ("time", arr)},
        coords={"time": ("time", times.values)},
    )


# ── §1 Limitation key constants ──────────────────────────────────────


def test_noleap_limitation_key_canonical() -> None:
    """The CC-G-5 limitation key must match the canonical name string
    (audit consumers grep on this exact value)."""
    assert LIMITATION_KEY_NOLEAP_DROPPED == "calendar_noleap_dropped_feb29"


def test_360_day_limitation_key_canonical() -> None:
    assert LIMITATION_KEY_360_RESAMPLED == "calendar_360_day_resampled"


# ── §2 Backend pin re-exported from canonical source ─────────────────


def test_expected_backend_sourced_from_isimip_versions() -> None:
    """``EXPECTED_BACKEND`` must equal
    ``isimip_versions.NETCDF_FORMAT_ENGINE``. Per durable §24 we don't
    redefine the version pin in two places; this tests the
    re-export."""
    from prismpy.standards.isimip_versions import NETCDF_FORMAT_ENGINE

    assert EXPECTED_BACKEND == dict(NETCDF_FORMAT_ENGINE)
    for required in ("netcdf4", "xarray", "cftime"):
        assert required in EXPECTED_BACKEND


def test_calendar_module_does_not_redefine_calendar_table() -> None:
    """AST-walk the module: it must import ``CALENDAR_BY_GCM`` from
    isimip_versions, never declare its own copy. Per durable §24."""
    src = (
        Path(__file__).resolve().parents[2]
        / "src/prismpy/harmonize/calendar_conversion.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Confirm the import.
    imports_calendar_by_gcm = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "prismpy.standards.isimip_versions"
        and any(alias.name == "CALENDAR_BY_GCM" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imports_calendar_by_gcm, (
        "calendar_conversion.py must import CALENDAR_BY_GCM from "
        "prismpy.standards.isimip_versions per durable §24."
    )

    # Confirm the name isn't reassigned at module level (which would
    # be a parallel redefinition).
    redefinitions = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CALENDAR_BY_GCM"
            for target in node.targets
        )
    ]
    assert redefinitions == [], (
        "CALENDAR_BY_GCM must not be reassigned at module level in "
        "calendar_conversion.py — that's a canonical-source duplication."
    )


# ── §3 calendar_for_gcm correctness for 5-GCM primary ensemble ───────


@pytest.mark.parametrize(
    "gcm,expected",
    [
        ("gfdl-esm4", "noleap"),
        ("ipsl-cm6a-lr", "gregorian"),
        ("mpi-esm1-2-hr", "gregorian"),
        ("mri-esm2-0", "gregorian"),
        ("ukesm1-0-ll", "360_day"),
    ],
)
def test_calendar_for_gcm_returns_expected_calendar(
    gcm: str, expected: str
) -> None:
    assert calendar_for_gcm(gcm) == expected


def test_calendar_for_gcm_rejects_unknown_gcm() -> None:
    with pytest.raises(ValueError, match="hadgem3-gc31-ll"):
        calendar_for_gcm("hadgem3-gc31-ll")


# ── §4 Gregorian → gregorian pass-through ────────────────────────────


def test_gregorian_pass_through_returns_input_unchanged() -> None:
    """``gregorian`` source → no limitation; data unchanged."""
    ds = _gregorian_dataset()
    result = convert_to_gregorian(ds, source_calendar="gregorian")
    assert result.applies_limitation() is False
    assert result.limitation_key is None
    assert result.limitation_value is None
    assert result.target_calendar == "standard"
    # Data identity preserved (same object reference for pass-through)
    assert result.data is ds


def test_gregorian_aliases_all_pass_through() -> None:
    """``standard`` and ``proleptic_gregorian`` are aliases."""
    ds = _gregorian_dataset()
    for alias in ("standard", "proleptic_gregorian", "Gregorian", "STANDARD"):
        result = convert_to_gregorian(ds, source_calendar=alias)
        assert result.applies_limitation() is False
        assert result.target_calendar == "standard"


# ── §5 noleap → gregorian conversion ─────────────────────────────────


def test_noleap_converts_with_dropped_feb29_limitation() -> None:
    """Source noleap data converts to standard calendar; limitation
    record fires with the canonical key."""
    ds = _noleap_dataset(start_year=2048, n_years=1)  # 2048 is leap year
    result = convert_to_gregorian(ds, source_calendar="noleap")

    assert result.applies_limitation() is True
    assert result.limitation_key == LIMITATION_KEY_NOLEAP_DROPPED
    assert "Feb 29" in (result.limitation_value or "")
    assert result.source_calendar == "noleap"
    assert result.target_calendar == "standard"


def test_noleap_alias_365_day_also_handled() -> None:
    """``365_day`` is a noleap alias."""
    ds = _noleap_dataset(start_year=2048)
    result = convert_to_gregorian(ds, source_calendar="365_day")
    assert result.limitation_key == LIMITATION_KEY_NOLEAP_DROPPED


def test_noleap_conversion_preserves_non_leap_year_data() -> None:
    """In a non-leap year (2047), noleap and gregorian have identical
    365 days — converting should preserve all 365 values."""
    ds = _noleap_dataset(start_year=2047, n_years=1)
    result = convert_to_gregorian(ds, source_calendar="noleap")
    converted = result.data
    # Count days in the converted output
    n_days = converted["tasmax"].size
    # 2047 is non-leap; converted gregorian should have 365 days
    assert n_days == 365


# ── §6 360-day → gregorian conversion ────────────────────────────────


def test_360_day_converts_with_resampled_limitation() -> None:
    ds = _three60_day_dataset(start_year=2046, n_years=1)
    result = convert_to_gregorian(ds, source_calendar="360_day")

    assert result.applies_limitation() is True
    assert result.limitation_key == LIMITATION_KEY_360_RESAMPLED
    assert "interpol" in (result.limitation_value or "").lower()


def test_360_day_conversion_yields_full_year_coverage() -> None:
    """Converting a 1-year 360-day series should yield 365 days
    (xarray.convert_calendar with missing='interpolate' fills the
    5-day gap)."""
    ds = _three60_day_dataset(start_year=2046, n_years=1)
    result = convert_to_gregorian(ds, source_calendar="360_day")
    n_days = result.data["tasmax"].size
    assert n_days == 365  # 2046 is non-leap


# ── §7 Unrecognised calendars fail loud ──────────────────────────────


def test_unknown_calendar_raises_value_error() -> None:
    ds = _gregorian_dataset()
    with pytest.raises(ValueError, match="julian"):
        convert_to_gregorian(ds, source_calendar="julian")


# ── §8 Determinism — same input → identical output ───────────────────


def test_360_day_conversion_is_deterministic_across_two_runs() -> None:
    """Same source data → identical converted bytes across two
    invocations. AC-G-13 deliverable hash pin depends on this."""
    ds = _three60_day_dataset(start_year=2046, n_years=2)
    result_a = convert_to_gregorian(ds, source_calendar="360_day")
    result_b = convert_to_gregorian(ds, source_calendar="360_day")
    np.testing.assert_array_equal(
        result_a.data["tasmax"].values,
        result_b.data["tasmax"].values,
    )


def test_noleap_conversion_is_deterministic_across_two_runs() -> None:
    ds = _noleap_dataset(start_year=2048, n_years=1)
    result_a = convert_to_gregorian(ds, source_calendar="noleap")
    result_b = convert_to_gregorian(ds, source_calendar="noleap")
    np.testing.assert_array_equal(
        result_a.data["tasmax"].values,
        result_b.data["tasmax"].values,
    )


# ── §9 CalendarConversionResult shape ────────────────────────────────


def test_conversion_result_is_frozen_dataclass() -> None:
    """``CalendarConversionResult`` is frozen so callers can't mutate
    the limitation record after the fact."""
    ds = _gregorian_dataset()
    result = convert_to_gregorian(ds, source_calendar="gregorian")
    with pytest.raises(Exception):  # FrozenInstanceError
        result.limitation_key = "tampered"  # type: ignore[misc]


def test_conversion_result_carries_source_and_target_calendars() -> None:
    ds = _three60_day_dataset()
    result = convert_to_gregorian(ds, source_calendar="360_day")
    assert result.source_calendar == "360_day"
    assert result.target_calendar == "standard"


# ── §10 5-GCM round-trip via calendar_for_gcm ────────────────────────


def test_all_primary_gcms_route_to_recognised_calendar() -> None:
    """For every primary-ensemble GCM, ``calendar_for_gcm`` returns a
    calendar string that ``convert_to_gregorian`` accepts."""
    ds = _gregorian_dataset()
    for gcm in CALENDAR_BY_GCM:
        cal = calendar_for_gcm(gcm)
        # Use the gregorian dataset for gregorian GCMs (pass-through);
        # build the right shape for non-gregorian.
        if cal == "noleap":
            test_ds: Any = _noleap_dataset()
        elif cal == "360_day":
            test_ds = _three60_day_dataset()
        else:
            test_ds = ds
        result = convert_to_gregorian(test_ds, source_calendar=cal)
        assert result.target_calendar == "standard"


# ── §11 Codex round 1 P1 absorption — float64 dtype + NaN-free output ─


def test_360_day_output_preserves_float_dtype() -> None:
    """Codex round 1 boundary 3/7 P1: ``missing="interpolate"`` (a
    string) was being passed to ``convert_calendar`` which treats
    ``missing`` as a numeric fill value, NOT an interpolation mode.
    The string was inserted into the added days, contaminating the
    output dtype to ``object``. The fix uses ``missing=np.nan`` plus
    ``.interpolate_na`` so the output preserves float64 dtype."""
    ds = _three60_day_dataset(start_year=2046, n_years=1)
    result = convert_to_gregorian(ds, source_calendar="360_day")
    # The output must be float (NOT object). Object dtype would
    # mean the literal string "interpolate" leaked into the array.
    assert result.data["tasmax"].dtype.kind == "f", (
        f"360-day output dtype is {result.data['tasmax'].dtype} — "
        f"expected float. Codex round 1 P1 regression."
    )


def test_360_day_output_has_no_string_contamination() -> None:
    """No element of the converted output is the literal string
    ``"interpolate"`` (the symptom of the codex round 1 P1 misuse).
    A regression of the bug would surface as object dtype with
    interspersed strings; this asserts the absence."""
    ds = _three60_day_dataset(start_year=2046, n_years=1)
    result = convert_to_gregorian(ds, source_calendar="360_day")
    values = result.data["tasmax"].values
    # All values must be numeric. Convert to float; failure indicates
    # string contamination.
    arr = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(arr)), (
        "360-day output contains non-finite values — interpolate_na "
        "did not fill the gap days"
    )


def test_360_day_output_yields_full_year_with_interpolated_gaps() -> None:
    """365 days for a non-leap year (2046), ALL finite (no NaN)
    after the interpolate_na step fills the 5-6 day gap."""
    ds = _three60_day_dataset(start_year=2046, n_years=1)
    result = convert_to_gregorian(ds, source_calendar="360_day")
    n_days = result.data["tasmax"].size
    # 2046 is non-leap → 365 days; the 5-day gap from 360→365
    # alignment is filled by linear interpolation.
    assert n_days == 365
    # No NaN in the output — every gap must be interpolated.
    n_nan = int(np.isnan(result.data["tasmax"].values).sum())
    assert n_nan == 0, (
        f"360-day output has {n_nan} NaN values — interpolate_na "
        "did not fill the gap days"
    )


def test_360_day_interpolated_values_are_monotonic_with_source() -> None:
    """The 360-day source has strictly increasing values 0..359.
    After conversion + interpolation, the output values should also
    be strictly non-decreasing across the year (linear interpolation
    of a monotonic source preserves monotonicity)."""
    ds = _three60_day_dataset(start_year=2046, n_years=1)
    result = convert_to_gregorian(ds, source_calendar="360_day")
    values = result.data["tasmax"].values.astype(float)
    diffs = np.diff(values)
    # Linear interpolation of a monotonic source must produce
    # non-decreasing differences (allow tiny float epsilon).
    assert np.all(diffs >= -1e-9), (
        "Interpolated output is not monotonic — the gap-fill is "
        "introducing non-monotonic artifacts"
    )


# ── §12 Codex round 1 P3 absorption — time_dim parameter threading ───


def test_time_dim_parameter_threads_to_xarray() -> None:
    """The function exposes ``time_dim`` for testability but the
    internal ``convert_calendar`` call must pass ``dim=time_dim``;
    otherwise xarray defaults to ``"time"`` and the parameter has
    zero effect for any non-``"time"`` coord. Codex round 1
    boundary 3/7 P3.

    Build a dataset whose time coordinate is named ``"DATE"`` (not
    ``"time"``) and assert the conversion succeeds rather than
    raising ``KeyError: 'time'``."""
    times = []
    for year in range(2046, 2047):
        for month in range(1, 13):
            for day in range(1, 31):
                times.append(cftime.Datetime360Day(year, month, day))
    arr = np.arange(len(times), dtype=float)
    ds = xr.Dataset(
        {"tasmax": ("DATE", arr)},
        coords={"DATE": ("DATE", times)},
    )
    result = convert_to_gregorian(
        ds,
        source_calendar="360_day",
        time_dim="DATE",
    )
    assert result.target_calendar == "standard"
    # The time coord should still be named DATE post-conversion.
    assert "DATE" in result.data.dims


def test_time_dim_parameter_threads_for_noleap() -> None:
    """Same as above but for the noleap branch."""
    times = []
    for year in range(2048, 2049):
        for doy in range(1, 366):
            month, day = _noleap_doy_to_date(doy)
            times.append(cftime.DatetimeNoLeap(year, month, day))
    arr = np.arange(len(times), dtype=float)
    ds = xr.Dataset(
        {"tasmax": ("DATE", arr)},
        coords={"DATE": ("DATE", times)},
    )
    result = convert_to_gregorian(
        ds,
        source_calendar="noleap",
        time_dim="DATE",
    )
    assert result.target_calendar == "standard"
    assert "DATE" in result.data.dims
