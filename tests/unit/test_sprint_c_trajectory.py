"""F-W Sprint C — test trajectory pin (re-anchored across sprints).

Originally anchored at the Sprint C band [772, 792]; re-anchored
in Sprint D.1 to [815, 900] following the pin's own contract
("Re-anchor this pin by updating ``_LOWER`` / ``_UPPER`` after a
new sprint deliberately changes the trajectory"). Sprint D.1's
harmonize-stage helpers + axis/cause schema + provenance
additions add ~90 net-new tests including parametrize spread
across 4 platforms × 4 soil labels and texture/rh boundary
inclusivity cases. Re-anchored to [815, 910] in Sprint D.1
commit 10 to absorb codex self-check LOW Q3 (3 new remap unit
tests in ``test_executor_hwsd_remap.py``). Re-anchored to
[815, 970] in Sprint E.0 to absorb the WarningCategory enum +
5-bucket map + 12-site Sprint D.1 migration. Re-anchored to
[815, 1080] in Sprint E.0.5 commit 3 to absorb the ECOCROP
envelope substrate (AC-Q3-A-d + AC-Q3-A-NaN + F28 per-crop
provenance — 38 tests + 12 subtests across
``test_ecocrop_envelopes.py`` + ``test_envelope_validation.py``).
Re-anchored to [815, 1200] in Sprint E.0.5 commit 6 to absorb
the climate-envelope verdict logic (AC-Q3-A-a/b/c three-state
precip + extremes-aware thermal + verdict aggregation + zone
aggregation helpers — 46 tests in ``test_climate_envelope.py``;
plus 52 KG-classifier + jitter + transitional + antimeridian
tests from commit 5). Re-anchored to [815, 1300] in Sprint
E.0.5 commit 9 to absorb the walker family (F24 + F26 + F27),
cross-platform numpy.quantile reproducibility unit tests,
and the AC-Q2-A1-a public constant + AC-Q2-A1-b negative-
grep structural pins. Re-anchored to [815, 1500] in Sprint
F + bucket-fix train + #227 fix to absorb per-crop physiology
bounds + Köppen substrate + cell-id companion-file pins +
cockpit-dimension-bucket map structural pin. Re-anchored to
[815, 1700] in Sprint S to absorb the per-package eGHR
substrate builder's new tests (AC-1 byte-pin + AC-2
substrate-builder integration + headroom for AC-3 through
AC-13 of Sprint S). Re-anchored to
[815, 1750] in Sprint G AC-G-2.0 cache primitive extraction
+ remaining Sprint G ACs to size headroom for ``test_cache_base.py``
sibling-sweep + per-AC pins (AC-G-1 typed exceptions, AC-G-3
ScenarioBlock schema, AC-G-7a/7b/7c/7d per-translator
projection-climate, AC-G-9 3-layer canonical CO₂, AC-G-12
14-fixture mutation-drill matrix, AC-G-13 4 paired-set
deliverable hash pins) per SPRINT-G-VERIFICATION-STRATEGY.md §10.
Re-anchored to [815, 1900] at Sprint G boundary 3/7 close
(post-AC-G-7c SARRA-Py GeoTIFF projection-climate path) to
absorb the realised counts from sub-checkpoints A–E (calendar
conversion + Tetens TDEW + CRAFT/PYTHIA WTH + ACEA pickle +
SARRA-Py GeoTIFF — ~155 tests across the 5 boundaries already
landed) plus headroom for the remaining boundary 4/7 (AC-G-9
3-layer canonical CO₂ enforcement) + boundary 5/7 (AC-G-10/11
provenance + bias-correction provenance string) + boundary 6/7
(AC-G-12 14-fixture mutation-drill matrix) + boundary 7/7
(AC-G-13 4 paired-set deliverables + SHA hash pins).
Re-anchored to [815, 2000] at Sprint G boundary 7/7 absorption
(post-codex-round-2 P1 ISIMIP→SARRA mapping + unit conversions
+ 4 structural pins via ``test_isimip_to_sarra_mapping.py``).
Re-anchored to [815, 2200] at the Sprint G/Sprint S rebase
reconciliation — Sprint G branch (anchored at 1894 + 5.6%
headroom pre-rebase) replayed onto post-Sprint-S main yields
empirical post-rebase count 1982; new cap 2200 = 1982 + 218
headroom = ~11% (matches the historical 5–15% pattern
documented across Sprint D.1, E.0, E.0.5, F, G stages, and
Sprint S). Per the pin's contract requirement of "after a
new sprint deliberately changes the trajectory" — both
Sprint S (200-unit delta) and Sprint G (500-unit delta) are
deliberate; the rebased branch reflects both.
Kept in sync with ``test_sprint_d_trajectory.py`` since both
pins measure the same ``pytest tests/ --collect-only`` count.

Anti-mutation drill: a count outside ``[_LOWER, _UPPER]`` fails
this pin. Catches both under-shooting (a contributor accidentally
removed Sprint C/D tests) and over-shooting (test churn that
adds tests beyond the budget without re-anchoring this pin).

Re-anchor this pin by updating ``_LOWER`` / ``_UPPER`` after a
new sprint deliberately changes the trajectory.
"""
from __future__ import annotations

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
# 5-15% pattern).
# Sprint E.2 Phase 1.5 re-anchor: 2200 -> 2300 (foundations +
# AC-E2-25 producer ext + AC-E2-28 writer + 4 structural pins).
# Kept in sync with ``test_sprint_d_trajectory.py``.
# Sprint E.3 re-anchor: 2300 -> 2400 (AC-E3-15 close ships
# canonical ``test_trajectory_cap.py``; this legacy pin retires
# at AC-E3-15 close per durable §24 canonical-source-or-pin).
_UPPER = 2400


class TestSprintCTrajectory(TestCase):

    def test_collected_test_count_falls_within_sprint_c_band(self):
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
        # The summary line has the form
        # ``================ N tests collected in X.YYs ================``;
        # extract N via a regex match for ``\d+ tests collected``.
        import re
        match = re.search(r'(\d+)\s+tests?\s+collected', result.stdout)
        self.assertIsNotNone(
            match,
            f'Could not parse collect-only output: '
            f'{result.stdout[-200:]!r}',
        )
        n_collected = int(match.group(1))
        self.assertGreaterEqual(
            n_collected, _LOWER,
            f'Test collection count {n_collected} is BELOW the '
            f'Sprint C band [{_LOWER}, {_UPPER}]. A drop indicates '
            f'tests were removed without updating this pin. '
            f'Re-anchor _LOWER/_UPPER if intentional.',
        )
        self.assertLessEqual(
            n_collected, _UPPER,
            f'Test collection count {n_collected} is ABOVE the '
            f'Sprint C band [{_LOWER}, {_UPPER}]. Test churn beyond '
            f'the documented +30..+50 budget for this sprint. '
            f'Re-anchor _LOWER/_UPPER if intentional, or split the '
            f'extra tests into a follow-up sprint.',
        )
