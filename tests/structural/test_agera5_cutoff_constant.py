"""Sprint E.0.5 AC-Q2-A1-a + AC-Q2-A1-b — public 180-day cutoff constant.

Pins the public :data:`AGERA5_RECORD_CUTOFF_DAYS = 180`
constant + the negative-grep that the legacy "publication-
date filter alongside accommodation-buffer-days" pattern is
NOT present anywhere in the source tree.

Anti-mutation drills:

- Lower the constant from 180 to anything below →
  ``test_cutoff_pinned_at_180`` fails. The 180-day floor is
  documented as 4× pessimistic AgERA5 lag (~30 days) +
  90-day margin per AC-Q2-A1-Reframe; lowering it shrinks
  that margin.
- Re-introduce the pre-Sprint-E.0.5 dual-date-filter pattern
  ("data_release_date" + "accommodation_buffer_days") in
  any agera5 / bounds source file →
  ``test_no_dual_date_filter`` fails (AC-Q2-A1-b consolidation).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from prismpy.bounds import AGERA5_RECORD_CUTOFF_DAYS


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src" / "prismpy"


class TestAgERA5CutoffConstant(unittest.TestCase):
    """Per AC-Q2-A1-a: the cutoff is pinned at 180 days as a
    public module-level constant in :mod:`prismpy.bounds`."""

    def test_cutoff_pinned_at_180(self):
        self.assertEqual(AGERA5_RECORD_CUTOFF_DAYS, 180)

    def test_cutoff_is_int(self):
        self.assertIsInstance(AGERA5_RECORD_CUTOFF_DAYS, int)

    def test_cutoff_module_path(self):
        # The constant ships in `prismpy.bounds.constants` and
        # is re-exported by `prismpy.bounds.__init__`. Both
        # imports must resolve to the same object.
        from prismpy.bounds.constants import (
            AGERA5_RECORD_CUTOFF_DAYS as direct,
        )
        from prismpy.bounds import (
            AGERA5_RECORD_CUTOFF_DAYS as reexport,
        )
        self.assertEqual(direct, reexport)
        self.assertEqual(direct, 180)


class TestNoDualDateFilter(unittest.TestCase):
    """Per AC-Q2-A1-b: the 180-day cutoff is the single
    inclusion gate; the legacy "publication-date filter +
    accommodation-buffer-days" pattern is consolidated away."""

    def test_no_dual_date_filter_in_source(self):
        # Search the prismpy source tree for the legacy pattern
        # that pairs "data_release_date" with "accommodation"
        # / "buffer_days" — both tokens together flagged the
        # dual-date-filter approach. The 180-day cutoff
        # subsumes that filter chain.
        legacy_pattern = re.compile(
            r"data_release_date.{0,200}accommodation_buffer_days"
            r"|accommodation_buffer_days.{0,200}data_release_date",
            re.DOTALL,
        )
        violations = []
        for py in _SRC_DIR.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if legacy_pattern.search(text):
                violations.append(str(py.relative_to(_REPO_ROOT)))
        self.assertEqual(
            violations, [],
            "AC-Q2-A1-b dual-date-filter consolidation: the "
            "legacy 'data_release_date' + "
            "'accommodation_buffer_days' chain must NOT appear "
            "in source. The 180-day cutoff subsumes it.",
        )


if __name__ == "__main__":
    unittest.main()
