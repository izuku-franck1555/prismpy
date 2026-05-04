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
_UPPER = 1500


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
