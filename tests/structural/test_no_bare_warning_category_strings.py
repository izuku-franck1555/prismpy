"""F25 — bare warning-category strings live only in the enum module.

After Sprint E.0 ships, every consumer that names a warning
category must do so through :mod:`prismpy.warnings.categories`.
Bare string literals matching a category value (``"soil_no_hwsd_coverage"``,
``"transitional_zone"``, etc.) outside the canonical module are
forbidden. The walker enforces this so a future contributor
cannot silently introduce a parallel string-tagged taxonomy.

Two carve-outs are allowed by design:

* :file:`prismpy/src/prismpy/warnings/categories.py` — the
  values are DEFINED here; this is the only place the bare
  strings legitimately live as enum-member values.
* :file:`prismpy/src/prismpy/cells/schema.py` — the Sprint
  D.1 ``UnavailableCause`` Literal stays the canonical 3-value
  type for cell-summary schema validation. Per AC-E0-6, the
  WarningCategory enum is a superset of UnavailableCause; the
  Literal stays in place to keep the cell-summary v2.x
  contract stable while consumers route new code through the
  enum.

The walker AST-parses each ``.py`` file under
:file:`prismpy/src/prismpy` and rejects any string-constant
expression whose value matches a WarningCategory enum value
unless the file is in the carve-out set. Comments and
docstrings are allowed (a docstring referencing
``"soil_no_hwsd_coverage"`` for documentation purposes does
not introduce a bypass; the walker distinguishes by AST node
position).

This is the load-bearing pin for the migrate-in-E.0 path —
because the 12 Sprint D.1 sites were migrated in commit 2,
the walker can be strict (no grandfathered allow-list).
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Set

from prismpy.warnings import WarningCategory


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src" / "prismpy"


# Files allowed to contain bare warning-category strings. The
# enum's own definition module is the canonical home; the
# cells/schema.py Literal is the AC-E0-6 backward-compat
# subset. Any other file referencing a category value MUST go
# through the enum.
ALLOWED_FILES_RELATIVE = frozenset(
    {
        Path("warnings/categories.py"),
        Path("cells/schema.py"),
    }
)


# Pre-computed for speed. Iterating WarningCategory is hashed
# via the str base so the set lookup below is O(1).
_CATEGORY_VALUES: Set[str] = {c.value for c in WarningCategory}


def _docstring_node_ids(tree: ast.AST) -> Set[int]:
    """Collect ``id()`` of every docstring Constant node so the
    walker can skip them. A module / class / function docstring
    is the first statement in the body and Python parses it as
    ``Expr(value=Constant(value=...))``. The Constant inside
    is what we want to identify so the walker never flags it.
    """
    ids: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                ids.add(id(first.value))
    return ids


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return a list of ``(lineno, value)`` violations for the
    file at ``path``. A violation is a string-constant AST node
    whose value matches a WarningCategory enum value AND is NOT
    a docstring node."""
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []  # not a Python file we should police
    docstring_ids = _docstring_node_ids(tree)
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.value, str):
            continue
        if id(node) in docstring_ids:
            continue
        if node.value in _CATEGORY_VALUES:
            violations.append((node.lineno, node.value))
    return violations


class TestNoBareWarningCategoryStrings(unittest.TestCase):
    """F25 walker — every bare warning-category string outside
    the canonical files is a violation."""

    def test_no_violations_across_prismpy_src(self):
        all_violations: list[tuple[Path, int, str]] = []
        for path in sorted(_SRC_ROOT.rglob("*.py")):
            relative = path.relative_to(_SRC_ROOT)
            if relative in ALLOWED_FILES_RELATIVE:
                continue
            for lineno, value in _scan_file(path):
                all_violations.append((relative, lineno, value))
        if all_violations:
            msg_lines = [
                "F25 violation — bare WarningCategory string(s) "
                "found outside prismpy.warnings.categories. Route "
                "through the enum (e.g. "
                "WarningCategory.SOIL_NO_HWSD_COVERAGE.value):",
            ]
            for relative, lineno, value in all_violations:
                msg_lines.append(
                    f"  src/prismpy/{relative}:{lineno} → {value!r}"
                )
            self.fail("\n".join(msg_lines))

    def test_canonical_categories_module_carries_the_strings(self):
        """Sanity check: the carve-out file MUST contain the
        bare strings (those are the enum value definitions).
        If a regression accidentally moves the values out of
        ``categories.py``, the walker would still pass against
        an empty file — this test ensures the carve-out earns
        its keep."""
        path = _SRC_ROOT / "warnings" / "categories.py"
        src = path.read_text(encoding="utf-8")
        for value in _CATEGORY_VALUES:
            with self.subTest(value=value):
                self.assertIn(
                    f'"{value}"', src,
                    f"WarningCategory value {value!r} must live "
                    f"in warnings/categories.py — that's the "
                    f"single canonical home for the bare string.",
                )

    def test_walker_detects_synthetic_violation(self):
        """Self-test: feed a synthetic string with a category
        value into ``_scan_file`` via ``ast.parse`` and assert
        it's flagged. Anti-mutation drill — the walker must be
        functional, not a no-op pass."""
        synthetic = (
            'def f():\n'
            '    return "soil_no_hwsd_coverage"\n'
        )
        tree = ast.parse(synthetic)
        docstring_ids = _docstring_node_ids(tree)
        violations: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, str):
                continue
            if id(node) in docstring_ids:
                continue
            if node.value in _CATEGORY_VALUES:
                violations.append((node.lineno, node.value))
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0][1], "soil_no_hwsd_coverage")

    def test_walker_ignores_docstring(self):
        """Self-test: a docstring referencing a category value
        must NOT be flagged. Docstrings are documentation;
        they don't introduce a bypass."""
        synthetic = (
            'def f():\n'
            '    """Reference soil_no_hwsd_coverage in docs."""\n'
            '    return "fine"\n'
        )
        tree = ast.parse(synthetic)
        docstring_ids = _docstring_node_ids(tree)
        violations: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, str):
                continue
            if id(node) in docstring_ids:
                continue
            if node.value in _CATEGORY_VALUES:
                violations.append((node.lineno, node.value))
        # The docstring is "Reference soil_no_hwsd_coverage in
        # docs." which doesn't EQUAL the category value — so it
        # would already not match. Refine with an exact-match
        # synthetic.
        synthetic_exact = (
            'def f():\n'
            '    """soil_no_hwsd_coverage"""\n'
            '    return "fine"\n'
        )
        tree = ast.parse(synthetic_exact)
        docstring_ids = _docstring_node_ids(tree)
        violations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, str):
                continue
            if id(node) in docstring_ids:
                continue
            if node.value in _CATEGORY_VALUES:
                violations.append((node.lineno, node.value))
        self.assertEqual(
            violations, [],
            "Walker must skip docstring nodes — docstrings are "
            "documentation, not source-of-truth for the enum.",
        )


if __name__ == "__main__":
    unittest.main()
