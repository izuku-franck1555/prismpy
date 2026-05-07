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
grep structural pins.
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
# Sprint S re-anchor: bumped 1500 -> 1700 to absorb the per-package eGHR
# substrate builder's new tests (AC-1 byte-pin + AC-2 substrate-builder
# integration + headroom reserved for AC-3 through AC-13). Pin re-runs
# automatically once Sprint S lands; further bumps required only if
# subsequent sprints exceed the headroom.
_UPPER = 1700


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
