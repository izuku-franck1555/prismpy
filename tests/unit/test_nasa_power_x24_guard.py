"""The ×24 guard through ``retrieve``: exposure, the measured de-scale, REF,
the pair, the budget, the refusal, the record, the cache and the log.

``requests.get`` is replaced at the network boundary and ``today`` is frozen,
so every case runs offline and never reads the wall clock.
"""
from __future__ import annotations

import logging
from datetime import date
from unittest import mock

import pytest
import requests

from prismpy.sources.climate import nasa_power as np_mod
from prismpy.sources.climate._cancel import PipelineCancelled
from prismpy.sources.climate.errors import NasaPowerScaledPrecipError
from prismpy.sources.climate.nasa_power import NASAPowerConfig, NASAPowerSource
from tests.unit._nasa_x24_support import (
    FakeNasa,
    banked,
    body,
    cache_digests,
    load,
    outcome,
    precip_window,
    records_digest,
    run,
    window_days,
    zero_cache_outcomes,
    zero_cases,
)

LAT, LON = 10.75, 5.75
GUARD = "NASA POWER rain guard (10.75, 5.75)"
RATE = load("rate_probe_compact.json")
SCALED = body(RATE["scaled_precip"], RATE["other_variables"])
CLEAN = body(RATE["clean_precip"], RATE["other_variables"])
REF_BLOCK = RATE["ref_jan_jun_parameter"]
W, REF = ("20260101", "20261011"), ("20260101", "20260630")
JAN1, END = date(2026, 1, 1), date(2026, 10, 11)

# latest_available 2026-02-19, so REF ([1 Jan, latest - 120 days]) is empty.
EARLY_TODAY, EARLY_END, EARLY_W = date(2026, 3, 1), date(2026, 2, 19), ("20260101", "20260219")
# latest_available 2026-01-15, so 2025 is a complete exposed year with REF to 2025-09-17.
LATE_TODAY, Y25, REF25 = date(2026, 1, 25), ("20250101", "20251231"), ("20250101", "20250917")
FULL25 = banked("ng_10.75_5.75_2025_full_clean.json")


def x24(block):
    precip = {d: round(24 * v, 2) if v > 0 else v for d, v in block["PRECTOTCORR"].items()}
    return dict(block, PRECTOTCORR=precip)


def early(wet):
    return body(precip_window(JAN1, EARLY_END, wet))


EARLY_CLEAN = early({"20260103": 1.37, "20260107": 0.53, "20260111": 2.11, "20260115": 4.07,
                     "20260119": 0.83, "20260123": 3.13, "20260127": 0.71, "20260202": 8.44,
                     "20260206": 1.09, "20260210": 0.37, "20260214": 2.57, "20260218": 0.61})
EARLY_SCALED = x24(EARLY_CLEAN)
UNCONF, UNCONF_X24 = early({"20260110": 0.24}), early({"20260110": 5.76})
FULL25_X24 = x24(FULL25)


def wet_counts(precip, upto="99999999", start="00000000"):
    wet = [v for d, v in precip.items() if start <= d <= upto and v > 0]
    return len(wet), sum(1 for v in wet if abs(v / 0.24 - round(v / 0.24)) <= 1e-6)


def guard_lines(caplog):
    return [m for m in caplog.messages if m.startswith("NASA POWER rain guard")]


def refusal(year, n, evidence, year_end, latest):
    return (f"failed to load climate for {year}: NASA POWER's daily precipitation for "
            f"({LAT}, {LON}) did not pass the ×24 scaling guard in {n} consecutive responses "
            f"({'; '.join(evidence)}); this retrieval checked {year} through {year_end}; "
            f"prismpy's conservative latest-available estimate was {latest}; "
            f"a later retrieval may resolve it")


def precip_of(result):
    return {r.date.strftime("%Y%m%d"): r.precip for r in result.data.records}


# ── unguarded and clean retrievals are unchanged ────────────────────────────
def test_unguarded_and_clean_retrievals_are_byte_identical_to_pre_guard(tmp_path):
    goldens = load("zero_goldens.json")["outcomes"]
    classified = {}
    with mock.patch.object(np_mod, "_classify_precip", side_effect=np_mod._classify_precip) as spy:
        for index, (case_id, kwargs, script) in enumerate(zero_cases()):
            spy.reset_mock()
            cache_dir = tmp_path / f"c{index}"
            got = outcome(run(FakeNasa(script), cache_dir=cache_dir, **kwargs), cache_dir)
            assert got == goldens[case_id], case_id
            classified[case_id] = spy.call_count
        spy.reset_mock()
        cache_outcomes = zero_cache_outcomes(tmp_path / "cache_paths")
        assert spy.call_count == 0
    for case_id, got in cache_outcomes.items():
        assert got == goldens[case_id], case_id
    assert set(goldens) == set(classified) | set(cache_outcomes)
    assert all(n == 0 for c, n in classified.items() if not c.startswith("exposed_"))
    assert all(n >= 1 for c, n in classified.items() if c.startswith("exposed_"))


# ── exposure and the unguarded-window warning ───────────────────────────────
@pytest.mark.parametrize("end, exposed", [
    (date(2026, 6, 30), False), (date(2026, 7, 1), True), (date(2026, 7, 20), True),
])
def test_exposure_boundary_is_latest_available_minus_120_days(tmp_path, caplog, end, exposed):
    caplog.set_level(logging.INFO)
    window = ("20260101", end.strftime("%Y%m%d"))
    fake = FakeNasa({window: [SCALED], **({REF: [REF_BLOCK]} if exposed else {})})
    result = run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=end)
    assert result.success
    upto = window[1]
    if exposed:
        assert fake.calls == [window, REF]
        assert precip_of(result) == {d: v for d, v in CLEAN["PRECTOTCORR"].items() if d <= upto}
    else:
        assert fake.calls == [window]
        assert precip_of(result) == {d: v for d, v in SCALED["PRECTOTCORR"].items() if d <= upto}
        k, m = wet_counts(SCALED["PRECTOTCORR"], upto)
        assert guard_lines(caplog) == [
            f"{GUARD} 2026: precipitation shows the ×24 signature ({m} of {k} wet days on the "
            f"0.24 mm lattice) in a window ending 2026-06-30, before the guarded period; "
            f"used as received"
        ]


def test_unguarded_clean_window_logs_nothing(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    result = run(FakeNasa({Y25: [FULL25]}), cache_dir=tmp_path, lat=LAT, lon=LON,
                 start=date(2025, 1, 1), end=date(2025, 12, 31))
    assert result.success and guard_lines(caplog) == []


@pytest.mark.parametrize("today, windows", [
    (date(2027, 1, 10), [("20260101", "20261231")]),
    (date(2027, 1, 11), [("20260101", "20261231"), ("20270101", "20270101")]),
])
def test_both_sides_of_a_january_boundary_are_guarded(tmp_path, today, windows):
    year_2026 = body(precip_window(JAN1, date(2026, 12, 31), {"20260601": 3.13}))
    year_2027 = body(precip_window(date(2027, 1, 1), date(2027, 1, 1), {}))
    fake = FakeNasa({("20260101", "20261231"): [year_2026], ("20270101", "20270101"): [year_2027]})
    with mock.patch.object(np_mod, "_classify_precip", side_effect=np_mod._classify_precip) as spy:
        run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=date(2027, 1, 31), today=today)
    assert fake.calls == windows
    assert spy.call_count == len(windows)


# ── the measured de-scale and the injected date ─────────────────────────────
def test_banked_scaled_series_is_descaled_to_the_clean_records(tmp_path):
    clean = run(FakeNasa({W: [CLEAN]}), cache_dir=tmp_path / "clean", lat=LAT, lon=LON, start=JAN1, end=END)
    fake = FakeNasa({W: [SCALED], REF: [REF_BLOCK]})
    result = run(fake, cache_dir=tmp_path / "scaled", lat=LAT, lon=LON, start=JAN1, end=END)
    assert fake.calls == [W, REF]
    assert records_digest(result.data.records) == records_digest(clean.data.records)


def test_banked_clean_series_is_used_in_one_request(tmp_path):
    fake = FakeNasa({W: [CLEAN]})
    result = run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=END)
    assert fake.calls == [W] and precip_of(result) == CLEAN["PRECTOTCORR"]


def test_over_wide_payload_is_judged_on_the_requested_window(tmp_path):
    # 12 ×24 wet days in the window, 40 off-lattice days after it: 23% on the lattice overall.
    beyond = body(precip_window(date(2026, 2, 20), date(2026, 3, 31),
                                {key: 3.13 for key in window_days(date(2026, 2, 20), date(2026, 3, 31))}))
    padded = {name: dict(series, **beyond[name]) for name, series in EARLY_SCALED.items()}
    fake = FakeNasa({EARLY_W: [padded, EARLY_CLEAN]}, exact_window=False)
    result = run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=EARLY_END,
                 today=EARLY_TODAY)
    assert fake.calls == [EARLY_W, EARLY_W]
    assert precip_of(result) == EARLY_CLEAN["PRECTOTCORR"]


def test_ref_window_follows_the_injected_today(tmp_path):
    window, ref = ("20260101", "20261011"), ("20260101", "20260724")
    fake = FakeNasa({window: [SCALED], ref: [REF_BLOCK]})
    run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=END, today=date(2026, 12, 1))
    assert fake.calls[:2] == [window, ref]


# ── ×1 admission ────────────────────────────────────────────────────────────
X1_OVERLAP = {"20260105": 0.01, "20260212": 0.02, "20260320": 1.37}
X1_TAIL = dict({f"202607{n:02d}": round(0.24 * n, 2) for n in range(1, 9)}, **{"20260815": 0.37})


def test_x1_response_is_admitted_unchanged_without_record(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    response = body(precip_window(JAN1, END, {**X1_OVERLAP, **X1_TAIL}))
    ref = body(precip_window(JAN1, date(2026, 6, 30), X1_OVERLAP))
    fake = FakeNasa({W: [response], REF: [ref]})
    result = run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=END)
    assert fake.calls == [W, REF]
    assert precip_of(result) == response["PRECTOTCORR"]
    assert "precip_x24_descaled" not in result.data.metadata
    assert guard_lines(caplog) == [
        f"{GUARD} 2026: validation attempt 1 of 6 is scaled: 8 of 12 wet days on the 0.24 mm "
        f"lattice; HTTP attempts so far 1 of 12 (0 transport retries)",
        f"{GUARD} 2026: resolved by measuring ×1 against a clean retrieval to 2026-06-30 after "
        f"1 validation attempts and 2 HTTP attempts (0 transport retries)",
    ]


# ── the de-scale record ─────────────────────────────────────────────────────
def _expected_record(start, end):
    k_ov = sum(1 for v in REF_BLOCK["PRECTOTCORR"].values() if v > 0)
    wet = sorted(d for d, v in RATE["scaled_precip"].items()
                 if start.strftime("%Y%m%d") <= d <= end.strftime("%Y%m%d") and v > 0)
    dates = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in wet]
    return [{
        "year": 2026, "first_date": start.isoformat(), "last_date": end.isoformat(),
        "wet_days_divided": len(dates), "divided_dates": dates, "factor": 24,
        "reason": "nasa_power_x24_response",
        "evidence": f"×24 measured on {k_ov} wet days against a clean retrieval of 2026-01-01 to 2026-06-30",
    }]


def test_descale_record_is_exact(tmp_path):
    result = run(FakeNasa({W: [SCALED], REF: [REF_BLOCK]}), cache_dir=tmp_path,
                 lat=LAT, lon=LON, start=JAN1, end=END)
    assert result.data.metadata["precip_x24_descaled"] == _expected_record(JAN1, END)


def test_descale_record_lists_only_the_returned_dates(tmp_path):
    start, end = date(2026, 3, 1), date(2026, 9, 15)
    fake = FakeNasa({("20260101", "20260915"): [SCALED], REF: [REF_BLOCK]})
    result = run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=start, end=end)
    record = result.data.metadata["precip_x24_descaled"]
    assert record == _expected_record(start, end)
    assert record[0]["divided_dates"] == sorted(record[0]["divided_dates"])


def test_no_record_when_nothing_was_descaled(tmp_path):
    result = run(FakeNasa({W: [CLEAN]}), cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=END)
    assert "precip_x24_descaled" not in result.data.metadata


# ── caching and the REF request ─────────────────────────────────────────────
def test_descaled_complete_year_is_never_cached(tmp_path):
    fake = FakeNasa({Y25: [FULL25_X24], REF25: [FULL25]})
    result = run(fake, cache_dir=tmp_path / "d", lat=LAT, lon=LON,
                 start=date(2025, 1, 1), end=date(2025, 12, 31), today=LATE_TODAY)
    assert result.success and fake.calls == [Y25, REF25]
    assert cache_digests(tmp_path / "d") == {}
    assert result.data.metadata["precip_x24_descaled"][0]["year"] == 2025
    clean = run(FakeNasa({Y25: [FULL25]}), cache_dir=tmp_path / "c", lat=LAT, lon=LON,
                start=date(2025, 1, 1), end=date(2025, 12, 31), today=LATE_TODAY)
    assert records_digest(result.data.records) == records_digest(clean.data.records)
    assert list(cache_digests(tmp_path / "c")) == ["nasapower_10_7500_5_7500_2025.json"]


def test_ref_is_requested_once_and_never_cached(tmp_path):
    unprovable = dict(SCALED, PRECTOTCORR=dict(SCALED["PRECTOTCORR"], **{"20260820": 17.05}))
    fake = FakeNasa({W: [unprovable, unprovable, CLEAN], REF: [REF_BLOCK]})
    result = run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=END)
    assert result.success and precip_of(result) == CLEAN["PRECTOTCORR"]
    assert fake.calls == [W, REF, W, W]
    assert cache_digests(tmp_path) == {}


def test_empty_ref_is_never_requested(tmp_path):
    fake = FakeNasa({EARLY_W: [UNCONF_X24, UNCONF]})
    result = run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=EARLY_END, today=EARLY_TODAY)
    assert result.success and fake.calls == [EARLY_W, EARLY_W]
    assert precip_of(result)["20260110"] == 0.24


# ── re-fetch, the ×24 pair, cancel points ───────────────────────────────────
def _early_run(tmp_path, items, **kwargs):
    fake = FakeNasa({EARLY_W: items})
    result = run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=EARLY_END,
                 today=EARLY_TODAY, **kwargs)
    return fake, result


def test_scaled_then_clean_uses_the_clean_response(tmp_path):
    fake, result = _early_run(tmp_path, [EARLY_SCALED, EARLY_CLEAN])
    assert result.success and len(fake.calls) == 2
    assert precip_of(result) == EARLY_CLEAN["PRECTOTCORR"]


def test_unconfirmed_then_off_lattice_response_resolves(tmp_path):
    off = early({"20260110": 0.47})
    fake, result = _early_run(tmp_path, [UNCONF, off])
    assert result.success and precip_of(result)["20260110"] == 0.47


@pytest.mark.parametrize("items", [[UNCONF_X24, UNCONF], [UNCONF, UNCONF_X24]], ids=["x24_first", "x1_first"])
def test_unconfirmed_x24_pair_uses_the_smaller_in_both_orders(tmp_path, items):
    fake, result = _early_run(tmp_path, items)
    assert result.success and len(fake.calls) == 2
    assert precip_of(result)["20260110"] == 0.24

def test_over_wide_pair_is_judged_on_the_requested_window(tmp_path):
    # Only the ×24 member carries a wet day after the window; within the window the pair holds.
    wide = dict(UNCONF_X24, PRECTOTCORR=dict(UNCONF_X24["PRECTOTCORR"], **{"20260301": 3.13}))
    fake = FakeNasa({EARLY_W: [wide, UNCONF]}, exact_window=False)
    result = run(
        fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=EARLY_END, today=EARLY_TODAY
    )
    assert result.success and fake.calls == [EARLY_W, EARLY_W]
    assert precip_of(result)["20260110"] == 0.24


def test_scaled_3_to_9_pair_resolves(tmp_path):
    four = early({"20260103": 0.24, "20260107": 0.48, "20260111": 0.72, "20260115": 0.96})
    fake, result = _early_run(tmp_path, [x24(four), four])
    assert result.success and precip_of(result)["20260115"] == 0.96


@pytest.mark.parametrize("items", [
    [early({"20260110": 5.76, "20260112": 11.52}), early({"20260110": 0.24})],
    [early({"20260110": 5.76, "20260111": 11.76}), early({"20260110": 0.24, "20260111": 0.48})],
], ids=["unequal_dates", "outside_tolerance"])
def test_non_pairs_are_refused(tmp_path, items):
    fake, result = _early_run(tmp_path, items)
    assert result.success is False and len(fake.calls) == 6
    assert "did not pass the ×24 scaling guard in 6 consecutive responses" in result.errors[0]


def test_cancel_before_the_refetch_delay(tmp_path):
    fake = FakeNasa({EARLY_W: [EARLY_SCALED, EARLY_CLEAN]})
    with pytest.raises(PipelineCancelled) as excinfo:
        run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=EARLY_END,
            today=EARLY_TODAY, cancel_check=lambda: len(fake.calls) >= 1)
    assert excinfo.value.where == "nasa_power.x24.before_refetch_delay=2026"
    assert fake.calls == [EARLY_W] and fake.sleeps == []


def test_cancel_after_the_refetch_delay(tmp_path):
    fake = FakeNasa({EARLY_W: [EARLY_SCALED, EARLY_CLEAN]})
    flag = {"on": False}
    fake.on_sleep = lambda: flag.update(on=True)
    with pytest.raises(PipelineCancelled) as excinfo:
        run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=EARLY_END,
            today=EARLY_TODAY, cancel_check=lambda: flag["on"])
    assert excinfo.value.where == "nasa_power.x24.before_refetch=2026"
    assert fake.calls == [EARLY_W]


def test_cancel_before_ref(tmp_path):
    fake = FakeNasa({W: [SCALED], REF: [REF_BLOCK]})
    with pytest.raises(PipelineCancelled) as excinfo:
        run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=END,
            cancel_check=lambda: len(fake.calls) >= 1)
    assert excinfo.value.where == "nasa_power.x24.before_ref_delay=2026"
    assert fake.calls == [W]


# ── refusal and the retry budget ────────────────────────────────────────────
def test_refusal_on_a_complete_year_is_exact_unconverted_and_uncached(tmp_path):
    fake = FakeNasa({Y25: [FULL25_X24], REF25: [FULL25_X24]})
    with mock.patch.object(NASAPowerSource, "_convert_to_climate_timeseries",
                           side_effect=AssertionError("converted")) as convert:
        result = run(fake, cache_dir=tmp_path, lat=LAT, lon=LON,
                     start=date(2025, 1, 1), end=date(2025, 12, 31), today=LATE_TODAY)
    k, _ = wet_counts(FULL25["PRECTOTCORR"])
    assert result.success is False
    assert result.errors == [refusal(2025, 6, [f"{k} of {k} wet days on the 0.24 mm lattice"] * 6,
                                     "2025-12-31", "2026-01-15")]
    assert convert.call_count == 0 and cache_digests(tmp_path) == {}
    assert fake.calls == [Y25, REF25] + [Y25] * 5


def test_refusal_raises_the_typed_error(tmp_path):
    source = NASAPowerSource(config=NASAPowerConfig(request_delay=0.0), cache_dir=tmp_path)
    fake = FakeNasa({Y25: [FULL25_X24], REF25: [FULL25_X24]})
    with mock.patch.object(requests, "get", fake), mock.patch("time.sleep", fake.sleep):
        with pytest.raises(NasaPowerScaledPrecipError) as excinfo:
            source._fetch_from_api(LAT, LON, date(2025, 1, 1), date(2025, 12, 31),
                                   list(np_mod.DEFAULT_PARAMETERS),
                                   latest_available=date(2026, 1, 15))
    assert type(excinfo.value) is NasaPowerScaledPrecipError and len(excinfo.value.evidence) == 6


def test_six_unconfirmed_responses_give_the_low_wet_evidence(tmp_path):
    fake, result = _early_run(tmp_path, [UNCONF])
    evidence = ["only 1 wet day(s), all on the 0.24 mm lattice"] * 6
    assert result.errors == [refusal(2026, 6, evidence, "2026-02-19", "2026-02-19")]


def test_unconfirmed_then_transport_failures_end_in_the_guard_error(tmp_path):
    fake, result = _early_run(tmp_path, [UNCONF, requests.exceptions.ConnectionError("down")])
    assert len(fake.calls) == 12
    assert result.errors == [refusal(2026, 1, ["only 1 wet day(s), all on the 0.24 mm lattice"],
                                     "2026-02-19", "2026-02-19")]


def test_transport_only_failure_keeps_todays_error(tmp_path):
    fake, result = _early_run(tmp_path, [requests.exceptions.ConnectionError("down")],
                              config=NASAPowerConfig(request_delay=0.0, retry_count=20))
    assert len(fake.calls) == 12
    assert result.errors == ["failed to load climate for 2026: API request failed after retries: down"]


def test_http_budget_counts_ref(tmp_path):
    down = requests.exceptions.ConnectionError("down")
    fake = FakeNasa({W: [SCALED], REF: [down]})
    result = run(fake, cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=END,
                 config=NASAPowerConfig(request_delay=0.0, retry_count=20))
    assert len(fake.calls) == 12 and fake.count(REF) == 11
    k, m = wet_counts(SCALED["PRECTOTCORR"])
    assert result.errors == [refusal(2026, 1, [f"{m} of {k} wet days on the 0.24 mm lattice"],
                                     "2026-10-11", "2026-10-28")]


# ── log lines ───────────────────────────────────────────────────────────────
def test_log_lines_of_the_descale_path(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    run(FakeNasa({W: [SCALED], REF: [REF_BLOCK]}), cache_dir=tmp_path, lat=LAT, lon=LON, start=JAN1, end=END)
    k, m = wet_counts(SCALED["PRECTOTCORR"])
    k_ov = sum(1 for v in REF_BLOCK["PRECTOTCORR"].values() if v > 0)
    assert guard_lines(caplog) == [
        f"{GUARD} 2026: validation attempt 1 of 6 is scaled: {m} of {k} wet days on the 0.24 mm "
        f"lattice; HTTP attempts so far 1 of 12 (0 transport retries)",
        f"{GUARD} 2026: ×24 measured against a clean retrieval to 2026-06-30 ({k_ov} wet days); "
        f"divided by 24",
        f"{GUARD} 2026: resolved by dividing a ×24 response by 24 after 1 validation attempts and "
        f"2 HTTP attempts (0 transport retries)",
    ]


def test_log_lines_of_the_clean_pair_and_refusal_paths(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    k, m = wet_counts(EARLY_SCALED["PRECTOTCORR"])
    _early_run(tmp_path / "a", [EARLY_SCALED, EARLY_CLEAN])
    _early_run(tmp_path / "b", [UNCONF_X24, UNCONF])
    _early_run(tmp_path / "c", [EARLY_CLEAN])
    scaled_line = (f"{GUARD} 2026: validation attempt 1 of 6 is scaled: {m} of {k} wet days on the "
                   f"0.24 mm lattice; HTTP attempts so far 1 of 12 (0 transport retries)")
    assert guard_lines(caplog) == [
        scaled_line,
        f"{GUARD} 2026: resolved by a clean response after 1 validation attempts and 2 HTTP "
        f"attempts (0 transport retries)",
        f"{GUARD} 2026: validation attempt 1 of 6 is unconfirmed: only 1 wet day(s), all on the "
        f"0.24 mm lattice; HTTP attempts so far 1 of 12 (0 transport retries)",
        f"{GUARD} 2026: validation attempt 2 of 6 is unconfirmed: only 1 wet day(s), all on the "
        f"0.24 mm lattice; HTTP attempts so far 2 of 12 (0 transport retries)",
        f"{GUARD} 2026: resolved by the smaller of a ×24 pair after 2 validation attempts and 2 "
        f"HTTP attempts (0 transport retries)",
    ]
    caplog.clear()
    _early_run(tmp_path / "d", [UNCONF])
    assert guard_lines(caplog)[-1] == (f"{GUARD} 2026: refused after 6 validation attempts and 6 "
                                       f"HTTP attempts (0 transport retries)")
