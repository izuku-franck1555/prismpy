"""F-W Sprint C — test trajectory pin (AC-10).

Asserts the prismpy test collection stays within the band the
contract documented for Sprint C: 742 (pre-Sprint-C baseline) +
30 to 50 net-new = 772 to 792.

Anti-mutation drill: a count outside ``[772, 792]`` fails this
pin. Catches both under-shooting (a contributor accidentally
removed Sprint C tests) and over-shooting (test churn that
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


_LOWER = 772
_UPPER = 792


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
