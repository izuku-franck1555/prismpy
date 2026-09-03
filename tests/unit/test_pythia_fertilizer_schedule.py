"""The PYTHIA translator emits the wizard's per-application fertilizer schedule
into the fertilized run as DSSAT ``@F`` ``{fdap, famn}`` pairs — days after
planting + kg N/ha, threaded straight (no lossy split/fraction round-trip).

Semantic pins (not survival): each checks the VALUE and placement, so a
mis-meaning thread (wrong day, wrong total, unsorted, baseline-contaminating)
fails even though "some N survived". Mutation notes on each assert the revert.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from prismpy.config.schema import ManagementConfig
from tests.unit.test_pythia_plant_mode import _generated_pythia_json


def _runs(tmp_path, mgmt):
    j = _generated_pythia_json(tmp_path, mgmt)
    return j, {
        "baseline": next(r for r in j["runs"] if r["name"].endswith("_baseline")),
        "fertilized": next(r for r in j["runs"] if r["name"].endswith("_fertilized")),
    }


# ── P-value: the exact {fdap, famn} value + DAP meaning ──────────────────────

def test_schedule_emits_exact_fdap_famn_values(tmp_path):
    # timing=15 (days after planting) -> fdap=15; amount=40 kg N/ha -> famn=40.
    # NOT a total, NOT an even split, NOT a day-of-year.
    mgmt = ManagementConfig(planting_density=62500.0,
                            fertilizer_apps=[{"timing": 15, "amount": 40.0}])
    _, runs = _runs(tmp_path, mgmt)
    assert runs["fertilized"]["fertilizers"] == [{"fdap": 15, "famn": 40.0}]


# ── P-emit-target: fertilized run ONLY, never default_setup / baseline ───────

def test_schedule_on_fertilized_run_not_default_setup_or_baseline(tmp_path):
    # The runner merges default_setup into every run; a shared schedule would
    # fertilize the baseline. Revert (emit on default_setup) -> this goes RED.
    mgmt = ManagementConfig(planting_density=62500.0,
                            fertilizer_apps=[{"timing": 0, "amount": 40.0},
                                             {"timing": 30, "amount": 30.0}])
    j, runs = _runs(tmp_path, mgmt)
    assert "fertilizers" not in j["default_setup"]
    assert "fertilizers" not in runs["baseline"]
    assert len(runs["fertilized"]["fertilizers"]) == 2


# ── P-sort: unsorted input renders ascending (DSSAT stops at a future date) ──

def test_schedule_sorted_ascending_by_timing(tmp_path):
    # Revert (drop the sort) -> emits [60, 0, 30] -> RED. DSSAT would skip the
    # 0- and 30-day rows once it passes day 60.
    mgmt = ManagementConfig(planting_density=62500.0,
                            fertilizer_apps=[{"timing": 60, "amount": 50.0},
                                             {"timing": 0, "amount": 40.0},
                                             {"timing": 30, "amount": 30.0}])
    _, runs = _runs(tmp_path, mgmt)
    assert [f["fdap"] for f in runs["fertilized"]["fertilizers"]] == [0, 30, 60]
    assert [f["famn"] for f in runs["fertilized"]["fertilizers"]] == [40.0, 30.0, 50.0]


# ── P-total: fen_tot is the SUM of what is applied, recomputed ───────────────

def test_fen_tot_recomputed_as_sum_of_applications(tmp_path):
    # The recorded total (91) drifts from the per-app sum (90). The emitted total
    # must be the sum actually applied. Revert (keep the recorded total) -> RED.
    mgmt = ManagementConfig(planting_density=62500.0, fertilizer_n_total=91.0,
                            fertilizer_apps=[{"timing": 0, "amount": 30.0},
                                             {"timing": 30, "amount": 30.0},
                                             {"timing": 60, "amount": 30.0}])
    _, runs = _runs(tmp_path, mgmt)
    assert runs["fertilized"]["fen_tot"] == 90.0
    assert runs["fertilized"]["fen_tot"] != 91.0


# ── AC-6 backward-compat: empty schedule -> fen_tot fallback, no @F key ──────

def test_empty_schedule_falls_back_to_fen_tot(tmp_path):
    # No apps -> no fertilizers key (the run is byte-identical to today), and
    # fen_tot keeps the mapped total for the runner's single-pulse fallback.
    mgmt = ManagementConfig(planting_density=62500.0, fertilizer_n_total=80.0)
    _, runs = _runs(tmp_path, mgmt)
    assert "fertilizers" not in runs["fertilized"]
    assert runs["fertilized"]["fen_tot"] == 80.0


# ── AC-6 malformed -> Pydantic raises at the schema boundary (§6.4) ──────────

def test_negative_timing_rejected_at_boundary(tmp_path):
    # DSSAT hard-errors on a negative FDATE; the schema must reject it, not a
    # consumer-layer guard.
    with pytest.raises(ValidationError):
        ManagementConfig(planting_density=62500.0,
                         fertilizer_apps=[{"timing": -5, "amount": 40.0}])


def test_over_nappl_rejected_at_boundary(tmp_path):
    # DSSAT's application-count ceiling (NAPPL = 200).
    with pytest.raises(ValidationError):
        ManagementConfig(planting_density=62500.0,
                         fertilizer_apps=[{"timing": 0, "amount": 1.0}] * 201)


def test_over_range_amount_rejected_at_boundary(tmp_path):
    # FAMN is a 5-char field (<= 999.9).
    with pytest.raises(ValidationError):
        ManagementConfig(planting_density=62500.0,
                         fertilizer_apps=[{"timing": 0, "amount": 1000.0}])
