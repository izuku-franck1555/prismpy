"""Climate-coverage producer-honesty pins.

The NASA POWER source reports success only when the loaded climate covers the
full requested window — endpoint-bracket plus zero interior missing days — and
it never reaches past the latest published date nor fabricates climate to
pass. These pins lock that contract end to end. Every test mocks the NASA
POWER API, so they run offline and deterministically (no network, no real
clock dependence).
"""
from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pytest

from prismpy.sources.climate import nasa_power as np_mod
from prismpy.sources.climate._availability import (
    DEFAULT_LAG_DAYS,
    MIN_LAG_DAYS,
    nasa_power_latest_available_date,
)
from prismpy.sources.climate._cancel import PipelineCancelled
from prismpy.sources.climate.nasa_power import (
    DEFAULT_PARAMETERS,
    NASAPowerConfig,
    NASAPowerSource,
)

_ROOT = Path(__file__).resolve().parents[2]
_NASA_POWER = _ROOT / "src" / "prismpy" / "sources" / "climate" / "nasa_power.py"
_TRANSLATORS = _ROOT / "src" / "prismpy" / "translators"

_LAT, _LON = 14.5, -13.0


def _make_source(tmp_path, lag_days=DEFAULT_LAG_DAYS):
    return NASAPowerSource(
        config=NASAPowerConfig(climate_lag_days=lag_days, request_delay=0.0),
        cache_dir=tmp_path,
    )


def _fake_api(served_through):
    """Stand-in for ``_fetch_from_api``: serves complete NASA-shaped daily data
    for the requested ``[start, end]`` window, never beyond ``served_through``,
    and records every ``(start, end)`` it was asked to fetch (the fetch-bound
    spy). Returns only real served days — there is no fabrication path."""
    calls = []

    def _fetch(self, lat, lon, start_date, end_date, parameters,
               cancel_check=None, on_attempt=None):
        calls.append((start_date, end_date))
        out = {p: {} for p in DEFAULT_PARAMETERS}
        day = start_date
        while day <= end_date and day <= served_through:
            key = day.strftime("%Y%m%d")
            for p in DEFAULT_PARAMETERS:
                out[p][key] = 5.0
            day += timedelta(days=1)
        return out

    _fetch.calls = calls
    return _fetch


def _run(src, start, end, latest, fetch, *, use_cache=False):
    with mock.patch.object(np_mod, "nasa_power_latest_available_date",
                           lambda *a, **k: latest), \
            mock.patch.object(NASAPowerSource, "_fetch_from_api", fetch):
        return src.retrieve(
            lat=_LAT, lon=_LON, start_date=start, end_date=end,
            use_cache=use_cache,
        )


# ── Reported repro: a wholly-past cross-year season loads its cells ─────────
def test_reported_cross_year_season_loads_cells(tmp_path):
    # Cross-year season entirely in the past; published latest is well past the
    # window end → the cells must load (the all-cells-unavailable bug is fixed).
    start, end, latest = date(2025, 1, 1), date(2026, 3, 15), date(2026, 5, 15)
    result = _run(_make_source(tmp_path), start, end, latest,
                  _fake_api(served_through=latest))
    assert result.success is True
    assert result.data is not None and result.data.records
    days = {r.date for r in result.data.records}
    assert start in days and end in days


# ── Coverage invariant: covered succeeds, uncovered is unavailable ─────────
def test_fully_covered_window_succeeds_without_synthesis(tmp_path):
    # A wholly-past window the API fully serves succeeds, brackets both
    # endpoints, and returns only real served days (no synthesized rows).
    start, end, latest = date(2025, 1, 1), date(2025, 6, 30), date(2026, 1, 1)
    fake = _fake_api(served_through=latest)
    result = _run(_make_source(tmp_path), start, end, latest, fake)
    assert result.success is True
    recs = sorted(result.data.records, key=lambda r: r.date)
    assert recs[0].date <= start and recs[-1].date >= end
    served = {start + timedelta(n) for n in range((end - start).days + 1)}
    assert {r.date for r in recs} <= served


def test_future_end_is_unavailable(tmp_path):
    start, end, latest = date(2025, 1, 1), date(2026, 12, 31), date(2026, 5, 15)
    result = _run(_make_source(tmp_path), start, end, latest,
                  _fake_api(served_through=latest))
    assert result.success is False
    assert any("not yet published" in e for e in result.errors)


def test_transient_failure_is_unavailable(tmp_path):
    start, end, latest = date(2025, 1, 1), date(2026, 3, 15), date(2026, 5, 15)

    def boom(self, lat, lon, start_date, end_date, parameters,
             cancel_check=None, on_attempt=None):
        raise Exception("network down")

    result = _run(_make_source(tmp_path), start, end, latest, boom)
    assert result.success is False
    assert any("failed to load climate" in e for e in result.errors)


def test_cancel_propagates_not_swallowed(tmp_path):
    start, end, latest = date(2025, 1, 1), date(2026, 3, 15), date(2026, 5, 15)

    def cancel(self, lat, lon, start_date, end_date, parameters,
               cancel_check=None, on_attempt=None):
        raise PipelineCancelled("user cancelled")

    with pytest.raises(PipelineCancelled):
        _run(_make_source(tmp_path), start, end, latest, cancel)


def test_interior_gap_is_unavailable_not_silent_partial(tmp_path):
    # A window whose interior has a missing day (a -999 fill -> None) must NOT
    # be reported success=True — the producer cannot emit a silent partial.
    start, end, latest = date(2025, 1, 1), date(2025, 3, 31), date(2026, 1, 1)
    fake = _fake_api(served_through=latest)

    def gapped(self, lat, lon, start_date, end_date, parameters,
               cancel_check=None, on_attempt=None):
        served = fake(self, lat, lon, start_date, end_date, parameters)
        hole = date(2025, 2, 10).strftime("%Y%m%d")
        if hole in served.get("ALLSKY_SFC_SW_DWN", {}):
            served["ALLSKY_SFC_SW_DWN"][hole] = -999
        return served

    result = _run(_make_source(tmp_path), start, end, latest, gapped)
    assert result.success is False
    assert any("incomplete or corrupt" in e for e in result.errors)


# ── Genuinely-future windows: no future request, no fabrication ────────────
def test_no_future_requested_no_fabrication_earlier_cached(tmp_path):
    start, end, latest = date(2025, 1, 1), date(2026, 12, 31), date(2026, 5, 15)
    fake = _fake_api(served_through=latest)
    result = _run(_make_source(tmp_path), start, end, latest, fake,
                  use_cache=True)
    # (a) fetch-bound spy: no requested end_date is past the latest published.
    assert fake.calls, "the fetch must have been attempted"
    assert all(end_req <= latest for (_s, end_req) in fake.calls), fake.calls
    # (b) unavailable with the honest future message.
    assert result.success is False
    assert any("not yet published" in e for e in result.errors)
    # (c) anti-cooking: nothing past the latest published date is ever served.
    served_days = {d for (_s, e) in fake.calls
                   for d in ((_s + timedelta(n)) for n in
                             range((e - _s).days + 1))}
    assert all(d <= latest for d in served_days)
    # (d) earlier full year cached for reuse; the clamped partial year is not.
    cached = {p.name for p in (tmp_path / "climate").glob("*.json")}
    assert any(n.endswith("_2025.json") for n in cached), cached
    assert not any(n.endswith("_2026.json") for n in cached), cached


def test_year_wholly_after_latest_is_not_fetched(tmp_path):
    # A requested window reaching into a calendar year entirely past the latest
    # published date must not issue an inverted (end < start) fetch for that
    # year; the year is skipped and the window reports unavailable.
    start, end, latest = date(2025, 1, 1), date(2027, 6, 30), date(2026, 5, 15)
    fake = _fake_api(served_through=latest)
    result = _run(_make_source(tmp_path), start, end, latest, fake)
    assert result.success is False
    assert any("not yet published" in e for e in result.errors)
    # no fetch is inverted, and the wholly-future year is never requested.
    assert all(s <= e for (s, e) in fake.calls), fake.calls
    assert all(s.year != 2027 for (s, _e) in fake.calls), fake.calls


def test_future_start_within_year_issues_no_fetch(tmp_path):
    # A window starting after the latest published date (within one year) must
    # issue no fetch for that year — not a doomed fetch filtered out of window.
    start, end, latest = date(2026, 10, 1), date(2026, 12, 31), date(2026, 5, 15)
    fake = _fake_api(served_through=latest)
    result = _run(_make_source(tmp_path), start, end, latest, fake)
    assert result.success is False
    assert any("not yet published" in e for e in result.errors)
    assert fake.calls == [], fake.calls


# ── Cache integrity: a clamped partial year is never cached as full ────────
def test_partial_boundary_year_not_cached_and_refetched(tmp_path):
    start, end, latest = date(2025, 1, 1), date(2026, 6, 30), date(2026, 6, 20)
    src = _make_source(tmp_path)
    fake1 = _fake_api(served_through=latest)
    _run(src, start, end, latest, fake1, use_cache=True)
    cached = {p.name for p in (tmp_path / "climate").glob("*.json")}
    assert any(n.endswith("_2025.json") for n in cached), cached
    assert not any(n.endswith("_2026.json") for n in cached), cached
    # second run re-fetches the partial 2026 (does not load a partial as full)
    fake2 = _fake_api(served_through=latest)
    _run(src, start, end, latest, fake2, use_cache=True)
    assert any(s.year == 2026 for (s, _e) in fake2.calls), fake2.calls
    assert all(s.year != 2025 for (s, _e) in fake2.calls), fake2.calls


# ── One published-date source: no current-date arithmetic in the source ────
def test_single_latest_available_source():
    tree = ast.parse(_NASA_POWER.read_text(encoding="utf-8"))
    today_calls = [
        getattr(node, "lineno", "?")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"today", "now"}
    ]
    assert today_calls == [], (
        f"nasa_power.py computes its own current date at lines {today_calls}; "
        "the only published-date arithmetic must live in "
        "nasa_power_latest_available_date."
    )


# ── Published-date helper: conservative, floored, deterministic ────────────
def test_availability_helper_arithmetic_floor_underclaim():
    today = date(2026, 6, 2)
    assert nasa_power_latest_available_date(10, today) == date(2026, 5, 23)
    assert nasa_power_latest_available_date(DEFAULT_LAG_DAYS, today) == (
        today - timedelta(days=DEFAULT_LAG_DAYS)
    )
    # A too-small lag is raised to the floor — never over-claims published data.
    assert nasa_power_latest_available_date(1, today) == (
        today - timedelta(days=MIN_LAG_DAYS)
    )
    assert nasa_power_latest_available_date(None, today) == (
        today - timedelta(days=DEFAULT_LAG_DAYS)
    )
    # The estimate is always strictly in the past (under-claiming).
    assert nasa_power_latest_available_date(10, today) < today


# ── Consumer-routing pin (forward-prevention, behavior-preserving) ─────────
def test_climate_consumer_sites_gate_success_before_data():
    for rel in ("acea/translator.py", "craft/translator.py",
                "pythia/translator.py"):
        text = (_TRANSLATORS / rel).read_text(encoding="utf-8")
        assert "if result.success and result.data" in text, (
            f"{rel} must gate its NASA POWER retrieve on result.success "
            "before attaching result.data."
        )
