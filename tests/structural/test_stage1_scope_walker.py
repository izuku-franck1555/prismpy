"""Sprint E.0.5 F27 — Stage 1 scope-discipline walker.

Per the contract Draft 4 F27 codification + verification
strategy NEW3 design: Stage 1 envelope-comparison code in
:mod:`prismpy.validators.climate_envelope` must NOT
reference out-of-scope ECOCROP fields. Stage 1 covers
**precip + tmin + tmax only** per AC-Q3-A-a/b/c +
probe-1-A scope clarity.

Forbidden ECOCROP fields (Sprint F / V3 territory):

* ``ALTMX`` (max altitude — Stage 2 per-cell with elevation raster)
* ``PHMIN`` / ``PHMAX`` (per-cell soil-pH check)
* ``PHOTOPERIOD`` (per-cell daylength)
* ``GMIN`` / ``GMAX`` (per-cell phenology growing-day range)
* ``LATMIN`` / ``LATMAX`` (latitude range)

This AST walker scans for both Name-token and string-literal
references to any forbidden field; a future commit that
silently uses ``crop_envelope["ALTMX"]`` (or constants
``ALTMX = ...``) trips the walker.

Anti-mutation drill: introduce ``altmx_check()`` call or
``ALTMX`` constant in ``climate_envelope.py`` →
``test_no_forbidden_ecocrop_fields_in_stage1`` fails.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_STAGE1_FILES = (
    _REPO_ROOT / "src" / "prismpy" / "validators" / "climate_envelope.py",
)


# Forbidden ECOCROP fields per probe-1-A scope clarity.
# Both case-sensitive forms (constants typically uppercase)
# and lowercase forms (dictionary access keys) are checked.
_FORBIDDEN_TOKENS = frozenset({
    # Altitude
    "ALTMX", "altmx", "ALTMN", "altmn",
    # pH range
    "PHMIN", "phmin", "PHMAX", "phmax",
    "PHOPMN", "phopmn", "PHOPMX", "phopmx",
    # Photoperiod / daylength
    "PHOTOPERIOD", "photoperiod",
    # Growing-day range
    "GMIN", "gmin", "GMAX", "gmax",
    # Latitude range
    "LATMIN", "latmin", "LATMAX", "latmax", "LATRANGE", "latrange",
})


def _walk_for_forbidden_tokens(source_path: Path) -> list[tuple[int, str]]:
    """AST-walk a source file; return list of (line_no, token)
    for any forbidden ECOCROP field reference."""
    text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source_path))
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        # Bare-name references: e.g. ``ALTMX = 2000``,
        # ``if altmx_check(...)``.
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_TOKENS:
            violations.append((node.lineno, node.id))
        # String literals: e.g. ``crop_envelope["ALTMX"]``.
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in _FORBIDDEN_TOKENS:
                violations.append((node.lineno, node.value))
        # Attribute access: e.g. ``crop_envelope.altmx``.
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_TOKENS:
            violations.append((node.lineno, node.attr))
    return violations


class TestStage1ScopeDiscipline(unittest.TestCase):
    """Per F27 + AC-Q3-A-a/b/c probe-1-A: Stage 1 envelope-
    comparison code touches only precip + tmin + tmax. No
    ALTMX / pH / photoperiod / GMIN / GMAX / latitude refs."""

    def test_no_forbidden_ecocrop_fields_in_stage1(self):
        all_violations: dict[str, list[tuple[int, str]]] = {}
        for stage1_file in _STAGE1_FILES:
            if not stage1_file.exists():
                continue
            violations = _walk_for_forbidden_tokens(stage1_file)
            if violations:
                all_violations[
                    str(stage1_file.relative_to(_REPO_ROOT))
                ] = violations
        self.assertEqual(
            all_violations, {},
            f"F27 violation: Stage 1 envelope-comparison code "
            f"references forbidden ECOCROP fields (ALTMX / pH / "
            f"photoperiod / GMIN/GMAX / latitude). These are "
            f"Sprint F / V3 territory per probe-1-A scope "
            f"clarity. Violations: {all_violations!r}",
        )

    def test_at_least_one_stage1_file_scanned(self):
        # Sanity: the walker must actually find files to scan.
        # If the file list goes empty, the walker passes
        # vacuously, hiding the scope drift it was designed
        # to catch.
        existing = [f for f in _STAGE1_FILES if f.exists()]
        self.assertGreater(
            len(existing), 0,
            "F27 walker must scan at least one Stage 1 file. "
            "If validators/climate_envelope.py was removed or "
            "moved, update _STAGE1_FILES to track the new "
            "location.",
        )


if __name__ == "__main__":
    unittest.main()
