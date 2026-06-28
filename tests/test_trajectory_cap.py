"""Canonical trajectory pin for the prismpy test suite.

Sprint E.3 AC-E3-15 + builder DELTA-CA close. This is the single
canonical trajectory pin per durable §24 canonical-source-or-pin
discipline; the legacy Sprint C / Sprint D pins at
``tests/unit/test_sprint_c_trajectory.py`` +
``tests/unit/test_sprint_d_trajectory.py`` retire at the
Phase 1 ship close.

**Threshold formula** (Sprint G precedent):
``_UPPER = ceil((n_collected + 200) / 50) * 50`` — yields a
50-test-rounded ceiling with 200-test absorption headroom for
the next sprint's net additions. At Phase 1 close
(n_collected ≈ 2357), the formula yields:
``ceil((2357 + 200) / 50) * 50 = ceil(51.14) * 50 = 52 * 50 = 2600``.

The lower bound preserves the historical floor (815) so a
catastrophic test removal (e.g., a refactor that drops a
whole structural-pin batch) fires loud. The narrow band per
``feedback_iteration_test_tiers.md`` ensures sprint-time test
churn surfaces as an intentional pin re-anchor rather than
silent drift.

**Re-anchor cadence**: the pin re-anchors at every sprint's
ship-ritual commit per durable §24. A new sprint that lands
substantial test churn updates `_UPPER` atomically with the
ship-ritual commit; the pin's commit message documents the
recompute.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from unittest import TestCase


_REPO_ROOT = Path(__file__).resolve().parents[1]


# Lower bound preserves the historical Sprint C floor.
_LOWER = 815

# Upper bound per Sprint G threshold formula:
# ceil((n_collected + 200) / 50) * 50.
# At Sprint E.3 Phase 1 ship close (n_collected ≈ 2357):
# ceil((2357 + 200) / 50) * 50 = 2600.
#
# Sprint history:
#   Sprint C / D: 815-2300 (legacy bands; retired at Sprint E.3)
#   Sprint E.3 Phase 1 close: 2600 (this canonical pin)
#   Retry-migration test suites: n_collected ≈ 2658 →
#     ceil((2658 + 200) / 50) * 50 = 2900.
#   isimip3b cutout-fix (feat/isimip3b-cutout-fix): +5 committed tests
#     (root-cause file-path selection, slice-overlap fail-loud, None-guard,
#     multi-decade concat, live integration) → committed n_collected ≈ 2839 →
#     ceil((2839 + 200) / 50) * 50 = 3050. (Committed count measured with the
#     untracked tests/phase3 scratch files excluded — they are NOT on the
#     branch / in CI.)
_UPPER = 3050


class TestTrajectoryCap(TestCase):
    """Pin the test-collection count to a band that catches both
    catastrophic test removals (drops below ``_LOWER``) and
    silent-test-churn drift (rises above ``_UPPER``).

    Re-anchor by updating ``_UPPER`` after a deliberate sprint-
    time addition — the commit message documents the recompute
    via the Sprint G threshold formula.
    """

    def test_collected_test_count_falls_within_band(self) -> None:
        """Run ``pytest --collect-only -q`` in a subprocess so the
        pin reflects the same number a developer (or CI) sees."""
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '--collect-only', '-q'],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r'(\d+)\s+tests?\s+collected', result.stdout)
        assert match is not None, (
            f'Could not parse collect-only output: '
            f'{result.stdout[-200:]!r}'
        )
        n_collected = int(match.group(1))

        assert n_collected >= _LOWER, (
            f'Test collection count {n_collected} is BELOW the '
            f'canonical band [{_LOWER}, {_UPPER}]. A drop indicates '
            f'tests were removed without updating this pin. '
            f'Re-anchor _LOWER if intentional.'
        )
        assert n_collected <= _UPPER, (
            f'Test collection count {n_collected} is ABOVE the '
            f'canonical band [{_LOWER}, {_UPPER}]. Test churn '
            f'beyond the documented band. Re-anchor _UPPER per '
            f'Sprint G threshold formula '
            f'`ceil((n_collected + 200) / 50) * 50` if intentional, '
            f'or split the extra tests into a follow-up sprint.'
        )
