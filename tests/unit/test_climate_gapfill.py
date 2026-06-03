"""Unit pins for NASA POWER short-gap recovery.

These exercise the recovery functions directly (no network): linear
interpolation of short solar / temperature gaps, the consecutive-day cap,
the rain-never rule, impossible-value normalization, and the single-sourced
cap constant.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from prismpy.models.climate import ClimateRecord
from prismpy.sources.climate._gapfill import (
    MAX_INTERP_GAP_DAYS,
    fill_short_gaps,
    normalize_missing,
    recompute_means,
)


def _rec(day, srad, tmin=10.0, tmax=20.0, precip=0.0):
    return ClimateRecord(date=day, tmax=tmax, tmin=tmin, precip=precip, srad=srad)


def _srad_seq(start, srads):
    return [_rec(start + timedelta(i), s) for i, s in enumerate(srads)]


def test_single_srad_gap_interpolated_strictly_between_brackets():
    recs = _srad_seq(date(2025, 2, 1), [10.0, None, 20.0])
    prov = fill_short_gaps(recs)
    assert recs[1].srad == 15.0
    assert 10.0 < recs[1].srad < 20.0
    assert prov == {"srad": {"n_filled_days": 1, "method": "linear-interp"}}


def test_temperature_gap_interpolated():
    recs = [ClimateRecord(date=date(2025, 2, d), tmax=20.0, tmin=t,
                          precip=0.0, srad=5.0)
            for d, t in ((1, 10.0), (2, None), (3, 14.0))]
    prov = fill_short_gaps(recs)
    assert recs[1].tmin == 12.0
    assert "tmin" in prov


def test_exactly_five_day_gap_filled_six_day_left():
    five = _srad_seq(date(2025, 2, 1), [10.0, None, None, None, None, None, 20.0])
    assert fill_short_gaps(five)["srad"]["n_filled_days"] == 5
    assert all(r.srad is not None for r in five)
    six = _srad_seq(date(2025, 2, 1),
                    [10.0, None, None, None, None, None, None, 20.0])
    assert fill_short_gaps(six) == {}
    assert all(r.srad is None for r in six[1:7])


def test_rain_is_never_interpolated():
    recs = [ClimateRecord(date=date(2025, 2, d), tmax=20.0, tmin=10.0,
                          precip=p, srad=5.0)
            for d, p in ((1, 5.0), (2, None), (3, 3.0))]
    prov = fill_short_gaps(recs)
    assert recs[1].precip is None
    assert "precip" not in prov


def test_boundary_gap_without_right_bracket_left_missing():
    recs = _srad_seq(date(2025, 2, 1), [10.0, 12.0, None])
    assert fill_short_gaps(recs) == {}
    assert recs[2].srad is None


def test_normalize_maps_impossible_values_to_missing():
    r = _rec(date(2025, 2, 1), -5.0, tmin=20.0, tmax=10.0, precip=-2.0)
    normalize_missing([r])
    assert r.srad is None and r.precip is None
    assert r.tmax is None and r.tmin is None


def test_cross_year_gap_interpolated_by_date():
    recs = [_rec(date(2025, 12, 31), 10.0), _rec(date(2026, 1, 1), None),
            _rec(date(2026, 1, 2), 16.0)]
    fill_short_gaps(recs)
    assert recs[1].srad == 13.0


def test_clean_series_yields_empty_provenance():
    assert fill_short_gaps(_srad_seq(date(2025, 2, 1), [10.0, 11.0, 12.0])) == {}


def test_interp_cap_is_single_sourced():
    # The consecutive-day bound is defined once, in the recovery module; the
    # source must delegate, never carry its own copy of the constant.
    assert MAX_INTERP_GAP_DAYS == 5
    climate_dir = (Path(__file__).resolve().parents[2] / "src" / "prismpy"
                   / "sources" / "climate")
    defined = sum(f.read_text(encoding="utf-8").count("MAX_INTERP_GAP_DAYS =")
                  for f in climate_dir.glob("*.py"))
    assert defined == 1, defined


def test_record_with_all_temps_missing_constructs_without_crash():
    # A day with every temperature field missing must not crash on
    # construction; the derived mean is simply left missing.
    r = ClimateRecord(date=date(2025, 2, 1), tmax=None, tmin=None,
                      precip=0.0, srad=5.0)
    assert r.tmean is None


def test_recompute_means_restores_mean_for_recovered_temps():
    r = ClimateRecord(date=date(2025, 2, 1), tmax=20.0, tmin=10.0,
                      precip=0.0, srad=5.0)
    r.tmean = None  # a recovered temperature day whose source mean was missing
    recompute_means([r])
    assert r.tmean == 15.0


def test_real_zero_solar_radiation_is_preserved_not_missing():
    # SRAD of exactly 0.0 is a legitimate high-latitude polar-night value: it
    # must be kept (not nulled, not interpolated). Only negatives are missing.
    recs = _srad_seq(date(2025, 6, 1), [0.0, 0.0, 0.0])
    normalize_missing(recs)
    assert all(r.srad == 0.0 for r in recs)
    assert fill_short_gaps(recs) == {}
    assert all(r.srad == 0.0 for r in recs)
