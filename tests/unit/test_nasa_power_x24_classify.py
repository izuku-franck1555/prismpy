"""The ×24 guard's pure seams: classification, the measured factor, the
×24 pair and the de-scale (NASA POWER's PRECTOTCORR ×24 responses)."""
from __future__ import annotations

import math
from datetime import date

import pytest

from prismpy.sources.climate.nasa_power import (
    _classify_precip,
    _descale_precip,
    _measure_factor,
    _x24_pair_smaller,
)
from tests.unit._nasa_x24_support import banked, load, precip_window

SCALED_BANKED = [
    "ng_10.75_5.75_2026-01-01_2026-10-10_x24_a.json",
    "ng_10.75_5.75_2026-01-01_2026-10-10_x24_b.json",
    "ng_10.75_5.75_2026-01-01_2026-10-10_precip_only_x24.json",
    "ng_10.75_5.75_2026-04-01_2026-10-10_x24.json",
    "ng_11.25_3.75_2026-01-01_2026-10-10_x24.json",
]
CLEAN_BANKED = [
    "ng_10.75_5.75_2026-01-01_2026-10-10_precip_only_clean.json",
    "ng_10.75_5.75_2026-01-01_2026-10-10_clean.json",
    "ng_10.75_5.75_2026-01-01_2026-06-30_clean.json",
    "ng_10.75_5.75_2026-07-01_2026-10-10_clean.json",
    "ng_10.75_5.75_2026-05-10_2026-05-25_clean_a.json",
    "ng_10.75_5.75_2026-05-10_2026-05-25_clean_b.json",
    "ng_10.75_5.75_2026-05-10_2026-05-25_clean_c.json",
    "ng_10.75_5.75_2025_full_clean.json",
    "ng_11.25_3.75_2026-05-01_2026-05-31_clean.json",
    "ng_9.75_8.75_2020_full_clean.json",
    "ng_9.75_8.75_2020-04-05_2020-04-18_clean_9wet.json",
    "ng_10.625_8.625_2022-05-18_2022-05-28_clean.json",
    "zm_-15.708_27.208_2026-01-10_2026-01-25_precip_only_clean.json",
]


def _series(*values):
    return {f"202601{n + 1:02d}": v for n, v in enumerate(values)}


# ── AC-CLASS ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", SCALED_BANKED)
def test_banked_scaled_responses_classify_scaled(name):
    assert _classify_precip(banked(name)["PRECTOTCORR"]) == "scaled"


@pytest.mark.parametrize("name", CLEAN_BANKED)
def test_banked_clean_responses_classify_clean(name):
    assert _classify_precip(banked(name)["PRECTOTCORR"]) == "clean"


def test_error_response_is_not_applicable():
    doc = load("banked/error_duplicate_parameter.json")
    precip = doc.get("properties", {}).get("parameter", {}).get("PRECTOTCORR")
    assert _classify_precip(precip) == "n/a"


@pytest.mark.parametrize("values, expected", [
    ([0.24] * 9, "scaled"),
    ([0.24] * 8 + [0.25], "clean"),
    ([0.24] * 5 + [0.25] * 4, "clean"),
    ([0.24] * 5 + [0.25] * 5, "scaled"),
    ([0.24] * 4 + [0.25] * 6, "clean"),
    ([0.24, 0.48, 0.72], "scaled"),
    ([0.24, 0.48, 0.73], "clean"),
    ([0.48], "unconfirmed"),
    ([0.49], "clean"),
    ([0.24, 0.96], "unconfirmed"),
    ([0.24, 0.97], "clean"),
    ([0.24] * 50 + [0.25] * 50, "scaled"),
    ([0.24] * 49 + [0.25] * 51, "clean"),
    ([0.24 * (1 + 1e-7)] * 3, "scaled"),
    ([0.24 * (1 + 1e-5)] * 3, "clean"),
    ([], "clean"),
    ([0.0, 0.0], "clean"),
])
def test_classification_edges(values, expected):
    assert _classify_precip(_series(*values)) == expected


def test_excluded_values_are_never_wet():
    excluded = [-999, -999.0, None, math.nan, math.inf, -math.inf, 0, 0.0, -0.24, True]
    assert _classify_precip(_series(0.24, 0.48, 0.72, *excluded)) == "scaled"
    assert _classify_precip(_series(0.48, *excluded)) == "unconfirmed"
    assert _classify_precip(_series(*excluded)) == "clean"


# ── the measured factor (AC-DESCALE″ / AC-DESCALE‴ / AC-ADMIT) ───────────
REF_END = "20260630"
REF_WET = {"20260105": 1.37, "20260212": 0.53, "20260320": 2.11, "20260410": 4.07,
           "20260515": 12.3}
TAIL_WET = {"20260705": 3.13, "20260812": 0.71, "20260901": 8.44}


def _ref(wet=None):
    return precip_window(date(2026, 1, 1), date(2026, 6, 30), REF_WET if wet is None else wet)


def _response(overlap, tail):
    return precip_window(date(2026, 1, 1), date(2026, 10, 11), {**overlap, **tail})


def _x(values, factor):
    return {day: round(factor * value, 2) for day, value in values.items()}


def test_x24_measured_and_descaled_to_ref():
    response = _response(_x(REF_WET, 24), _x(TAIL_WET, 24))
    measured = _measure_factor(response, _ref(), REF_END)
    assert measured.factor == 24 and measured.k_ov == 5
    assert all(measured.descaled[d] == v for d, v in _ref().items())
    assert all(measured.descaled[d] == v for d, v in TAIL_WET.items())


def test_x1_proven_unscaled():
    response = _response(REF_WET, TAIL_WET)
    assert _measure_factor(response, _ref(), REF_END).factor == 1


@pytest.mark.parametrize("case", ["x48", "ref_on_lattice", "kov_2", "off_by_cent", "descaled_not_clean"])
def test_negatives_are_not_measured(case):
    ref, overlap, tail = _ref(), _x(REF_WET, 24), _x(TAIL_WET, 24)
    if case == "x48":
        overlap, tail = _x(REF_WET, 48), _x(TAIL_WET, 48)
    elif case == "ref_on_lattice":
        lattice = {"20260105": 0.24, "20260212": 0.48, "20260320": 0.72, "20260410": 0.96,
                   "20260515": 1.2}
        ref, overlap = _ref(lattice), _x(lattice, 24)
    elif case == "kov_2":
        two = {"20260105": 1.37, "20260212": 0.53}
        ref, overlap = _ref(two), _x(two, 24)
    elif case == "off_by_cent":
        overlap = dict(overlap, **{"20260320": round(overlap["20260320"] + 0.01, 2)})
    elif case == "descaled_not_clean":
        tail = {f"202607{n:02d}": 5.76 for n in range(1, 11)}
    assert _measure_factor(_response(overlap, tail), ref, REF_END).factor is None


def test_unequal_wet_dates_are_not_measured():
    extra = dict(REF_WET, **{"20260601": 0.37})
    assert _measure_factor(_response(extra, TAIL_WET), _ref(), REF_END).factor is None
    extra_x24 = dict(_x(REF_WET, 24), **{"20260601": 0.48})
    assert _measure_factor(_response(extra_x24, _x(TAIL_WET, 24)), _ref(), REF_END).factor is None


def test_orthogonal_tail_off_lattice_blocks_descale():
    tail = dict(_x(TAIL_WET, 24), **{"20260920": 17.05})
    assert _measure_factor(_response(_x(REF_WET, 24), tail), _ref(), REF_END).factor is None


def test_orthogonal_magnitude_floor_blocks_descale():
    small = {"20260105": 0.01, "20260212": 0.01, "20260320": 0.01}
    response = _response(_x(small, 24), {})
    assert _measure_factor(response, _ref(small), REF_END).factor is None


def test_x24_4_alias_is_not_descaled():
    small = {"20260105": 0.01, "20260212": 0.01, "20260320": 0.01}
    response = _response(_x(small, 24.4), {"20260705": 24.40})
    assert _measure_factor(response, _ref(small), REF_END).factor is None


def test_partial_tail_x24_with_clean_overlap_is_neither():
    response = _response(REF_WET, _x(TAIL_WET, 24))
    assert _measure_factor(response, _ref(), REF_END).factor is None


def test_admit_x1_with_one_tail_value_off_lattice():
    overlap = {"20260105": 0.01, "20260212": 0.02, "20260320": 1.37}
    tail = {f"202607{n:02d}": 0.24 * n for n in range(1, 9)}
    tail["20260815"] = 0.37
    response = _response(overlap, tail)
    assert _classify_precip(response) == "scaled"
    assert _measure_factor(response, _ref(overlap), REF_END).factor == 1


def test_x1_with_tail_entirely_on_lattice_is_not_admitted():
    overlap = {"20260105": 0.01, "20260212": 0.02, "20260320": 1.37}
    tail = {f"202607{n:02d}": 0.24 * n for n in range(1, 10)}
    assert _measure_factor(_response(overlap, tail), _ref(overlap), REF_END).factor is None


# ── the banked scaled bodies at the seam (73) ────────────────────────────
RATE = load("rate_probe_compact.json")
RATE_SCALED = [entry for entry in RATE["manifest"]
               if entry["class"] == "scaled" and entry["window"][1] == "2026-10-11"]


def test_rate_probe_manifest_holds_68_scaled_and_20_ref_bodies():
    assert len(RATE_SCALED) == 68
    assert sum(1 for e in RATE["manifest"] if e["window"] == ["2026-01-01", "2026-06-30"]) == 20


@pytest.mark.parametrize("entry", RATE_SCALED, ids=lambda e: e["body"].rsplit("/", 1)[-1])
def test_rate_probe_scaled_bodies_measure_x24(entry):
    ref = RATE["ref_jan_jun_parameter"]["PRECTOTCORR"]
    measured = _measure_factor(RATE["scaled_precip"], ref, REF_END)
    assert measured.factor == 24
    assert {d: v for d, v in measured.descaled.items()} == RATE["clean_precip"]


@pytest.mark.parametrize("scaled, ref_name, ref_span", [
    ("ng_10.75_5.75_2026-01-01_2026-10-10_x24_a.json", "ng_10.75_5.75_2026-01-01_2026-06-30_clean.json", None),
    ("ng_10.75_5.75_2026-01-01_2026-10-10_x24_b.json", "ng_10.75_5.75_2026-01-01_2026-06-30_clean.json", None),
    ("ng_10.75_5.75_2026-01-01_2026-10-10_precip_only_x24.json", "ng_10.75_5.75_2026-01-01_2026-06-30_clean.json", None),
    ("ng_10.75_5.75_2026-04-01_2026-10-10_x24.json", "ng_10.75_5.75_2026-01-01_2026-06-30_clean.json", "20260401"),
    ("ng_11.25_3.75_2026-01-01_2026-10-10_x24.json", "ng_11.25_3.75_2026-05-01_2026-05-31_clean.json", None),
])
def test_banked_scaled_bodies_measure_x24(scaled, ref_name, ref_span):
    ref = banked(ref_name)["PRECTOTCORR"]
    if ref_span is not None:
        ref = {d: v for d, v in ref.items() if d >= ref_span}
    measured = _measure_factor(banked(scaled)["PRECTOTCORR"], ref, max(ref))
    assert measured.factor == 24


def test_clean_rate_probe_bodies_measure_x1():
    ref = RATE["ref_jan_jun_parameter"]["PRECTOTCORR"]
    assert _measure_factor(RATE["clean_precip"], ref, REF_END).factor == 1


# ── the ×24 pair and the de-scale ────────────────────────────────────────
def test_x24_pair_in_both_orders():
    assert _x24_pair_smaller({"20260110": 5.76}, {"20260110": 0.24}) == "second"
    assert _x24_pair_smaller({"20260110": 0.24}, {"20260110": 5.76}) == "first"


def test_x24_pair_rejected():
    assert _x24_pair_smaller({"20260110": 5.76}, {"20260111": 0.24}) is None
    assert _x24_pair_smaller({"20260110": 5.76, "20260111": 11.52},
                             {"20260110": 0.24, "20260111": 0.49}) is None
    assert _x24_pair_smaller({"20260110": 0.24}, {"20260110": 0.24}) is None
    assert _x24_pair_smaller({"20260110": 0.0}, {"20260110": 0.0}) is None


def test_descale_touches_wet_values_only():
    series = {"a": 5.76, "b": -999, "c": None, "d": math.nan, "e": 0.0, "f": 0, "g": -0.24}
    out = _descale_precip(series)
    assert out["a"] == 0.24
    assert out["b"] == -999 and out["c"] is None and math.isnan(out["d"])
    assert out["e"] == 0.0 and out["f"] == 0 and out["g"] == -0.24
