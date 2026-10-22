"""Helpers for the NASA POWER ×24 guard tests.

They import no guard symbol, so the same AC-ZERO harness also runs against
the pre-guard code that produced ``zero_goldens.json``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

import requests

from prismpy.sources.climate import _availability
from prismpy.sources.climate import nasa_power as np_mod
from prismpy.sources.climate.nasa_power import NASAPowerConfig, NASAPowerSource

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "nasa_power_x24"
# latest_available 2026-10-28, so REF ends 2026-06-30 (the banked Jan-Jun body).
FROZEN_TODAY = date(2026, 11, 7)
OTHER_VARIABLES = ("T2M_MAX", "T2M_MIN", "T2M", "T2MDEW", "WS2M", "ALLSKY_SFC_SW_DWN", "RH2M")
_OTHER_CONSTANTS = (31.0, 21.0, 26.0, 14.0, 2.5, 20.0, 55.0)

Script = Dict[Tuple[str, str], List[Any]]


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def banked(name: str) -> Dict[str, Dict[str, Any]]:
    return load(f"banked/{name}")["properties"]["parameter"]


def key(day: date) -> str:
    return day.strftime("%Y%m%d")


def window_days(start: date, end: date) -> List[str]:
    return [key(start + timedelta(days=n)) for n in range((end - start).days + 1)]


def body(
    precip: Dict[str, Any],
    others: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """A NASA parameter block: ``precip`` + the 7 other variables
    (constants on the same dates unless ``others`` is given)."""
    if others is None:
        others = {
            name: {day: value for day in precip}
            for name, value in zip(OTHER_VARIABLES, _OTHER_CONSTANTS)
        }
    block = {name: dict(series) for name, series in others.items()}
    block["PRECTOTCORR"] = dict(precip)
    return block


def precip_window(start: date, end: date, wet: Dict[str, float]) -> Dict[str, float]:
    """Every day of [start, end] dry (0.0) except the ``wet`` ones."""
    return {day: wet.get(day, 0.0) for day in window_days(start, end)}


class _Response:
    def __init__(self, parameter: Dict[str, Dict[str, Any]]) -> None:
        self._parameter = parameter

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return {"properties": {"parameter": self._parameter}}


class FakeNasa:
    """Stands in for ``requests.get``. ``script`` maps a requested
    (start, end) window to the items served in order (the last one
    repeats): a parameter block, sliced to the requested window like
    NASA's, or an exception to raise. Requests land in ``calls``."""

    def __init__(self, script: Script) -> None:
        self.script = {window: list(items) for window, items in script.items()}
        self.calls: List[Tuple[str, str]] = []
        self.sleeps: List[float] = []
        self.on_sleep: Optional[Callable[[], None]] = None

    def __call__(self, url, params=None, timeout=None):
        window = (params["start"], params["end"])
        self.calls.append(window)
        queue = self.script.get(window)
        assert queue, f"unexpected NASA POWER request for {window}"
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return _Response({
            name: {day: value for day, value in series.items() if window[0] <= day <= window[1]}
            for name, series in item.items()
        })

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if self.on_sleep is not None:
            self.on_sleep()

    def count(self, window: Tuple[str, str]) -> int:
        return sum(1 for call in self.calls if call == window)


def run(
    fake: FakeNasa,
    *,
    cache_dir: Path,
    lat: float,
    lon: float,
    start: date,
    end: date,
    today: date = FROZEN_TODAY,
    use_cache: bool = False,
    config: Optional[NASAPowerConfig] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    """``retrieve`` with ``requests.get``, ``time.sleep`` and the
    published-date estimate's ``today`` replaced (no network, no wall clock)."""
    source = NASAPowerSource(
        config=config or NASAPowerConfig(request_delay=0.0), cache_dir=cache_dir,
    )

    def latest(lag_days=_availability.DEFAULT_LAG_DAYS, today_=None):
        return _availability.nasa_power_latest_available_date(lag_days, today=today)

    with mock.patch.object(np_mod, "nasa_power_latest_available_date", latest), \
            mock.patch.object(requests, "get", fake), \
            mock.patch("time.sleep", fake.sleep):
        return source.retrieve(
            lat=lat, lon=lon, start_date=start, end_date=end,
            use_cache=use_cache, cancel_check=cancel_check,
        )


def records_digest(records) -> str:
    """sha256 of the records' canonical serialization."""
    rows = []
    for record in records:
        row = dataclasses.asdict(record)
        row["date"] = record.date.isoformat()
        rows.append(row)
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def cache_digests(cache_dir: Path) -> Dict[str, str]:
    folder = cache_dir / "climate"
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.glob("*.json"))
    } if folder.exists() else {}


def outcome(result, cache_dir: Path) -> Dict[str, Any]:
    return {
        "success": result.success,
        "errors": list(result.errors or []),
        "records": records_digest(result.data.records) if result.data else None,
        "cache": cache_digests(cache_dir),
    }


def zero_cases() -> List[Tuple[str, Dict[str, Any], Script]]:
    """Every AC-ZERO retrieval: (id, run kwargs, script)."""
    rate = load("rate_probe_compact.json")
    cases = [
        ("fresh_2025_full", dict(lat=10.75, lon=5.75, start=date(2025, 1, 1), end=date(2025, 12, 31)),
         {("20250101", "20251231"): [banked("ng_10.75_5.75_2025_full_clean.json")]}),
        ("fresh_2020_full", dict(lat=9.75, lon=8.75, start=date(2020, 1, 1), end=date(2020, 12, 31)),
         {("20200101", "20201231"): [banked("ng_9.75_8.75_2020_full_clean.json")]}),
        ("exposed_clean_w1", dict(lat=10.75, lon=5.75, start=date(2026, 1, 1), end=date(2026, 10, 11)),
         {("20260101", "20261011"): [body(rate["clean_precip"], rate["other_variables"])]}),
        ("exposed_clean_fx_a8", dict(lat=10.75, lon=5.75, start=date(2026, 1, 1), end=date(2026, 10, 10)),
         {("20260101", "20261010"): [banked("ng_10.75_5.75_2026-01-01_2026-10-10_clean.json")]}),
    ]
    for file_year in load("past_prefixes_compact.json")["file_years"]:
        year = int(file_year["cache_file"][-9:-5])
        precip = {iso.replace("-", ""): value for iso, value in file_year["precip"].items()}
        for end_day in file_year["endpoints"]:
            end = date.fromisoformat(end_day)
            cases.append((
                f"prefix_{file_year['cache_file'][:-5]}_{end_day}",
                dict(lat=file_year["lat"], lon=file_year["lon"], start=date(year, 1, 1), end=end),
                {(f"{year}0101", key(end)): [body(precip)]},
            ))
    return cases


def zero_cache_outcomes(root: Path) -> Dict[str, Dict[str, Any]]:
    """AC-ZERO's cache paths: a per-year cache hit and an old-style hit."""
    out = {}
    full = banked("ng_10.75_5.75_2025_full_clean.json")
    per_year = root / "per_year"
    run(FakeNasa({("20250101", "20251231"): [full]}), cache_dir=per_year,
        lat=10.75, lon=5.75, start=date(2025, 1, 1), end=date(2025, 12, 31), use_cache=True)
    out["per_year_cache_hit"] = outcome(
        run(FakeNasa({}), cache_dir=per_year, lat=10.75, lon=5.75,
            start=date(2025, 3, 1), end=date(2025, 9, 30), use_cache=True), per_year)
    old_style = root / "old_style"
    source = NASAPowerSource(config=NASAPowerConfig(request_delay=0.0), cache_dir=old_style)
    series = source._convert_to_climate_timeseries(nasa_data=full, lat=10.75, lon=5.75)
    start, end = date(2025, 2, 1), date(2025, 8, 31)
    source._save_climate_cache(series, source._get_climate_cache_path(10.75, 5.75, start, end))
    out["old_style_cache_hit"] = outcome(
        run(FakeNasa({}), cache_dir=old_style, lat=10.75, lon=5.75,
            start=start, end=end, use_cache=True), old_style)
    return out


def zero_outcomes(root: Path) -> Dict[str, Dict[str, Any]]:
    """Run every AC-ZERO retrieval in its own cold cache under ``root``."""
    out = {}
    for index, (case_id, kwargs, script) in enumerate(zero_cases()):
        cache_dir = root / f"c{index}"
        out[case_id] = outcome(run(FakeNasa(script), cache_dir=cache_dir, **kwargs), cache_dir)
    out.update(zero_cache_outcomes(root))
    return out
