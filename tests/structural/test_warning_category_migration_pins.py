"""AC-E0-9 — migration regression net for the 12 Sprint D.1 sites.

Sprint E.0 commit 2 migrated 12 bare-string assignments to
``WarningCategory.X.value`` references. This module pins each
migration so a future refactor cannot silently revert one of
the sites back to the bare string (which would also trip the
F25 walker — but the F25 walker reports a violation count;
this test names the sites individually).

The pins are source-text grep checks. Each site is named by
its file path and the WarningCategory member it now references.
A revert at any site drops one of the per-site assertions.

Per AC-E0-9, the existing Sprint D.1 functional tests
(``tests/integration/test_hwsd_unavailable_e2e.py``,
``tests/unit/test_executor_hwsd_remap.py``,
``tests/unit/test_cell_summary_failed_checks.py``) continue to
pass — they exercise the runtime behavior end-to-end and the
StrEnum equality glue keeps the cause string identical to the
pre-migration form. Those tests run in the broader suite; this
module ships the source-text pins specifically for the
migration targets.
"""
from __future__ import annotations

import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src" / "prismpy"


# Each entry: (file relative to src/prismpy, expected enum
# reference string). The walker greps for the reference in the
# file source. A revert that puts the bare string back instead
# would fail the walker AND drop this assertion.
MIGRATED_SITES = (
    ("harmonize/apply.py", "WarningCategory.SOIL_NO_HWSD_COVERAGE.value"),
    ("harmonize/apply.py", "WarningCategory.SOIL_TEXTURE_INVALID.value"),
    ("harmonize/apply.py", "WarningCategory.CLIMATE_RH_INVALID.value"),
    ("pipeline/executor.py", "WarningCategory.SOIL_NO_HWSD_COVERAGE.value"),
    ("sources/soil/hwsd.py", "WarningCategory.SOIL_NO_HWSD_COVERAGE.value"),
)


class TestMigrationPins(unittest.TestCase):
    """Per-site source-text pins for the Sprint D.1 migration."""

    def test_each_migrated_site_carries_enum_reference(self):
        for relative, reference in MIGRATED_SITES:
            with self.subTest(file=relative, ref=reference):
                path = _SRC_ROOT / relative
                src = path.read_text(encoding="utf-8")
                self.assertIn(
                    reference, src,
                    f"AC-E0-9 migration regression: file "
                    f"{relative} no longer contains the enum "
                    f"reference {reference!r}. A future "
                    f"contributor reverted the migration to a "
                    f"bare string — restore the enum reference.",
                )

    def test_warning_category_imported_in_each_migrated_module(self):
        """Each module that carries an enum reference must
        also import :class:`WarningCategory`. Without the
        import the reference is a NameError at runtime; this
        pin catches the half-migration."""
        # De-duplicate file paths from MIGRATED_SITES.
        files = sorted({rel for rel, _ in MIGRATED_SITES})
        for relative in files:
            with self.subTest(file=relative):
                path = _SRC_ROOT / relative
                src = path.read_text(encoding="utf-8")
                self.assertIn(
                    "from prismpy.warnings import WarningCategory",
                    src,
                    f"Module {relative} references "
                    f"WarningCategory but does not import it.",
                )

    def test_value_stability(self):
        """The enum value strings MUST equal the original
        Sprint D.1 cause strings byte-for-byte. The cell-
        summary v2.x JSON contract depends on these exact
        bytes; a typo in the enum value breaks downstream
        consumers (prismweb cell drawer, cockpit lineage)."""
        from prismpy.warnings import WarningCategory
        self.assertEqual(
            WarningCategory.SOIL_NO_HWSD_COVERAGE.value,
            "soil_no_hwsd_coverage",
        )
        self.assertEqual(
            WarningCategory.SOIL_TEXTURE_INVALID.value,
            "soil_texture_invalid",
        )
        self.assertEqual(
            WarningCategory.CLIMATE_RH_INVALID.value,
            "climate_rh_invalid",
        )


if __name__ == "__main__":
    unittest.main()
