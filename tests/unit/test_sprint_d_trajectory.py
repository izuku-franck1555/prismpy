"""Sprint D.1 AC-9 — test trajectory pin.

Asserts the prismpy test collection stays within the band the
contract documented for Sprint D.1. Pre-Sprint-D baseline: 790
collected at prismpy main ``4420035``. Post-Sprint-D actual is
~883 driven by harmonize-stage helpers + axis/cause schema +
provenance additions + AC-7 16-case soil-string-pin parametrize.

Anti-mutation drill: a count outside ``[_LOWER, _UPPER]`` fails
this pin. Catches both under-shooting (a contributor
accidentally removed Sprint D tests) and over-shooting (test
churn beyond the documented band).

Sprint D.1 band rationale:

* Lower bound 815 — preserves the contract's lower bound from
  the Draft 3 LOCKED FINAL absorption.
* Upper bound 1080 — successive ratchets:
  - 850 → 900 (Sprint D.1 mid-sprint) to absorb the parametrize-
    fixture spread that came in higher than the contract's
    43-net-new estimate.
  - 900 → 910 (Sprint D.1 commit 10) to absorb codex self-check
    LOW Q3 (3 new remap unit tests in
    ``test_executor_hwsd_remap.py``).
  - 910 → 970 (Sprint E.0) to absorb the WarningCategory enum
    + 5-bucket map + 12-site Sprint D.1 migration tests.
  - 970 → 1080 (Sprint E.0.5 commit 3) to absorb the ECOCROP
    envelope substrate (AC-Q3-A-d + AC-Q3-A-NaN + F28 per-crop
    provenance). Commit 3 added 38 tests + 12 subtests across
    ``test_ecocrop_envelopes.py`` + ``test_envelope_validation.py``.
  - 1080 → 1200 (Sprint E.0.5 commit 6) to absorb the climate-
    envelope verdict logic (AC-Q3-A-a/b/c three-state precip
    + extremes-aware thermal + verdict aggregation + zone
    aggregation helpers).
  - 1200 → 1300 (Sprint E.0.5 commit 9) to absorb the
    walker family (F24 zone-purity + F26 designated-CI-runner
    + F27 Stage 1 scope discipline) + cross-platform numpy.
    quantile reproducibility unit tests + AC-Q2-A1-a public
    constant + AC-Q2-A1-b dual-date-filter negative grep.
    Headroom of ~80 tests reserved for commit 10 bound-gen.
    yml + Gate B anti-mutation drill additions.
  - 1300 → 1500 (Sprint F + bucket-fix train + #227 fix) to
    absorb per-crop physiology bounds + Köppen substrate +
    cell-id companion-file pins + cockpit-dimension-bucket
    map structural pin.
  - 1500 → 1700 (Sprint S) to absorb the per-package eGHR
    substrate builder's new tests (AC-1 byte-pin + AC-2
    substrate-builder integration + headroom reserved for
    AC-3 through AC-13 of Sprint S).
  - 1500 → 1750 (Sprint G AC-G-2.0 cache primitive
    extraction + remaining Sprint G ACs) to absorb the
    sibling-sweep ``test_cache_base.py`` + per-AC pins for
    AC-G-1 ISIMIP3b client (typed exceptions) + AC-G-3
    ScenarioBlock schema bounds + AC-G-7a/7b/7c/7d
    per-translator projection-climate + AC-G-9 3-layer
    canonical CO₂ + AC-G-12 14-fixture mutation-drill matrix
    + AC-G-13 4 paired-set deliverable hash pins. Headroom
    sized for the ~150-225 Sprint G test additions per
    SPRINT-G-VERIFICATION-STRATEGY.md §10.
  - 1750 → 1900 (Sprint G boundary 3/7 close, post-AC-G-7c
    SARRA-Py GeoTIFF projection-climate path) to absorb the
    realised counts from boundary 3/7 sub-checkpoints A–E
    (calendar conversion + Tetens TDEW + CRAFT/PYTHIA WTH
    + ACEA pickle + SARRA-Py GeoTIFF — ~155 tests across the
    5 boundaries already landed; pre-bump 3× zero-flake
    measured 1765 collected) plus headroom for the remaining
    boundary 4/7 (AC-G-9 3-layer canonical CO₂ enforcement)
    + boundary 5/7 (AC-G-10/11 provenance + bias-correction
    provenance string) + boundary 6/7 (AC-G-12 14-fixture
    mutation-drill matrix) + boundary 7/7 (AC-G-13 4
    paired-set deliverables + SHA hash pins).
  - 1900 → 2000 (Sprint G boundary 7/7 absorption,
    post-codex-round-2 P1 ISIMIP→SARRA mapping + unit
    conversions) to absorb 17 absorption tests in
    ``test_isimip_to_sarra_mapping.py`` covering the 4 structural
    pins (mapping completeness + unit conversion correctness
    + two-vocabulary AST walker per
    ``feedback_two_vocabulary_substrate_drift.md`` +
    sibling-sweep over SARRA-Py translator). Pre-bump count 1906.
  - 2000 → 2200 (Sprint G/Sprint S rebase reconciliation) —
    Sprint G branch (anchored at 1894 + 106 headroom = 5.6%
    pre-rebase) replayed onto post-Sprint-S main reveals the
    combined trajectory exceeds the prior Sprint G cap. Empirical
    post-rebase count: 1982 collected. New cap 2200 = 1982 +
    218 headroom = ~11% (matches the historical 5–15% pattern
    documented across Sprint D.1, E.0, E.0.5, F, G stages, and
    Sprint S). Per the pin's contract requirement of "after a
    new sprint deliberately changes the trajectory" — both
    Sprint S (200-unit delta from common base 1500) and Sprint G
    (500-unit delta) are deliberate; the rebased branch reflects
    both.

Re-anchor this pin by updating ``_LOWER`` / ``_UPPER`` after a
new sprint deliberately changes the trajectory.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from unittest import TestCase


_REPO_ROOT = Path(__file__).resolve().parents[2]


_LOWER = 815
# Sprint S re-anchor: 1500 -> 1700 (per-package eGHR substrate).
# Sprint G replay: 1700 -> 1750 (cache-primitive extraction)
# -> 1900 (boundary 3/7 close) -> 2000 (boundary 7/7 absorption).
# Sprint G/S rebase reconciliation: 2000 -> 2200 (empirical
# post-rebase count 1982 + ~11% headroom; matches historical
# 5-15% pattern). Kept in sync with ``test_sprint_c_trajectory.py``.
# Sprint E.2 Phase 1.5 re-anchor: 2200 -> 2300 (foundations +
# AC-E2-25 producer ext + AC-E2-28 writer + 4 structural pins
# add ~15 net tests; new band absorbs ~5% headroom + leaves
# room for AC-E2-25 + AC-E2-28 follow-on probes the contract's
# Drill S + Drill V will surface). Kept in sync with
# ``test_sprint_c_trajectory.py``.
# Sprint E.3 re-anchor: 2300 -> 2400 (per AC-E3-15 +
# `feedback_iteration_test_tiers.md` honest-flag at trajectory
# approach; AC-E3-15 close ships canonical
# ``prismpy/tests/test_trajectory_cap.py`` with the final band;
# this legacy pin retires at AC-E3-15 close per durable §24
# canonical-source-or-pin discipline).
_UPPER = 2400


class TestSprintDTrajectory(TestCase):

    def test_collected_test_count_falls_within_sprint_d_band(self):
        """Run ``pytest --collect-only -q`` in a subprocess so the
        pin reflects the same number a developer (or CI) sees."""
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/',
             '--collect-only', '-q'],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(
            r'(\d+)\s+tests?\s+collected', result.stdout,
        )
        if match is None:
            self.fail(
                f'Could not parse collect-only output: '
                f'{result.stdout[-200:]!r}'
            )
        n_collected = int(match.group(1))
        self.assertGreaterEqual(
            n_collected, _LOWER,
            f'Test collection count {n_collected} is BELOW the '
            f'Sprint D.1 band [{_LOWER}, {_UPPER}]. A drop '
            f'indicates tests were removed without updating this '
            f'pin. Re-anchor _LOWER/_UPPER if intentional.',
        )
        self.assertLessEqual(
            n_collected, _UPPER,
            f'Test collection count {n_collected} is ABOVE the '
            f'Sprint D.1 band [{_LOWER}, {_UPPER}]. Test churn '
            f'beyond the documented band. Re-anchor '
            f'_LOWER/_UPPER if intentional, or split the extra '
            f'tests into a follow-up sprint.',
        )
