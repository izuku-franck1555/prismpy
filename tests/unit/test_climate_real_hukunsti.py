"""Real-data pin: the frozen Hukunsti NASA POWER cell.

This is the load-bearing acceptance test — it runs against a FROZEN recording
of the actual NASA POWER point response for the Hukunsti cell, captured while
the solar-radiation gap was still present (NASA backfills the near-real-time
-999 later, so a live fetch would not reproduce it). The cell carries exactly
one interior solar gap (2026-02-17 = -999) bracketed by two real days; the
recovery must load it with an interpolated value, not reject it and not admit
the sentinel.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import pytest

from prismpy.sources.climate import nasa_power as np_mod
from prismpy.sources.climate.nasa_power import NASAPowerConfig, NASAPowerSource

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hukunsti_frozen.json"
_LAT, _LON = -23.291666666666657, 20.041666666666657


def _frozen_parameters():
    doc = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return doc["properties"]["parameter"]


def _fetch_from_frozen(frozen):
    """Serve the frozen recording sliced to each requested window, mirroring
    the per-year fetch the producer issues."""
    def _fetch(self, lat, lon, start_date, end_date, parameters,
               cancel_check=None, on_attempt=None):
        served = {}
        for var, days in frozen.items():
            served[var] = {
                key: value for key, value in days.items()
                if start_date <= datetime.strptime(key, "%Y%m%d").date() <= end_date
            }
        return served
    return _fetch


def test_real_hukunsti_srad_gap_loads_with_correct_interpolation(tmp_path):
    parameters = _frozen_parameters()
    src = NASAPowerSource(config=NASAPowerConfig(request_delay=0.0),
                          cache_dir=tmp_path)
    start, end, latest = date(2025, 1, 1), date(2026, 3, 15), date(2026, 5, 24)
    with mock.patch.object(np_mod, "nasa_power_latest_available_date",
                           lambda *a, **k: latest), \
            mock.patch.object(NASAPowerSource, "_fetch_from_api",
                              _fetch_from_frozen(parameters)):
        result = src.retrieve(lat=_LAT, lon=_LON, start_date=start,
                              end_date=end, use_cache=False)

    # The cell loads (not the 0/646 all-unavailable bug) with one filled day.
    assert result.success is True, result.errors
    assert result.data.metadata.get("gap_fill") == {
        "srad": {"n_filled_days": 1, "method": "linear-interp"}}

    # Value-correctness: the gap is the date-weighted midpoint of its real
    # brackets (16.05 and 28.8), strictly between them — not -999, not a
    # constant, not None.
    gap = [r for r in result.data.records if r.date == date(2026, 2, 17)][0]
    assert gap.srad == pytest.approx((16.05 + 28.8) / 2)
    assert 16.05 < gap.srad < 28.8

    # No sentinel reaches the output, and both endpoints are present.
    assert all(r.srad not in (None, -999, -999.0) for r in result.data.records)
    loaded = {r.date for r in result.data.records}
    assert start in loaded and end in loaded
