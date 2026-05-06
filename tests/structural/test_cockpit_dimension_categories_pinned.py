"""Structural pin — every category emitted by
``executor.py::_CATEGORY_FROM_PREFIX`` MUST be a key in
``cockpit/manifest.py::_DIMENSION_BUCKET_MAP``.

Background
==========

Two vocabularies coexist in prismpy:

1. :class:`prismpy.warnings.categories.WarningCategory` — a
   tight 10-member enum produced by Sprint E.0 + Sprint F's
   zone-level paths. Maps to
   :data:`prismpy.warnings.categories.WARNING_BUCKET_MAP`.
2. ``executor.py::_CATEGORY_FROM_PREFIX`` — a parallel 7-entry
   tuple (6 unique RHS values) that the per-cell pivot path at
   ``executor.py:_category_for_check_id`` populates onto
   ``cell_summary.cells[i].failed_checks[j].category``. Coarse
   UI dimension-toggle vocabulary; not enum members.

The cockpit's ``_category_to_bucket_integer`` lookup at
``cockpit/manifest.py`` consults both: dimension-toggle
vocabulary first, :class:`WarningCategory` fallback. Each
dimension-toggle category routes to bucket 3 (TRUE_EXCLUDE)
by default per Sprint E.1's conservative policy; refinement
to bucket 4 (INTERPOLATABLE) for short-gap variants is
Sprint E.2 scope.

Why this pin
============

Without this pin, a future ``executor.py`` edit that adds a
new dimension-toggle prefix → category pair (e.g., new
``coverage_lai_cells`` prefix mapping to ``coverage_per_cell``
or a new ``snowpack_completeness`` category) wouldn't
automatically register in ``_DIMENSION_BUCKET_MAP``. The
producer would emit the new category onto cell records;
the consumer would raise ``UnknownCategoryError`` on the
first real-data project that triggered it.

The pin closes the gap structurally per durable lesson #20
(sibling-sweep at fixup time) + the canonical-source-or-pin
discipline at
``feedback_canonical_source_for_cross_boundary_invariants.md``
(durable #24 candidate): every cross-stage invariant either
has a single canonical source OR a structural pin asserting
consistency across the consumers. Here the two vocabularies
each own their domain — the pin asserts they intersect on
the dimension-toggle subset.

Anti-mutation drill
===================

Add a new prefix → category pair to
``_CATEGORY_FROM_PREFIX`` without adding the new category to
``_DIMENSION_BUCKET_MAP``. The pin fires with a diagnostic
naming the missing entries.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXECUTOR_PATH = (
    REPO_ROOT / "src" / "prismpy" / "pipeline" / "executor.py"
)
COCKPIT_MANIFEST_PATH = (
    REPO_ROOT / "src" / "prismpy" / "cockpit" / "manifest.py"
)


def _extract_category_from_prefix_rhs() -> set[str]:
    """Return the set of right-hand-side category strings from
    ``executor.py::_CATEGORY_FROM_PREFIX``. Walks the AST so
    we don't depend on the executor's import-time side-effects
    (heavy: scientific validators, climate sources, etc.).
    """
    tree = ast.parse(
        EXECUTOR_PATH.read_text(encoding="utf-8"),
        filename=str(EXECUTOR_PATH),
    )
    rhs_values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = node.targets
        if len(targets) != 1:
            continue
        target = targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id != "_CATEGORY_FROM_PREFIX":
            continue
        # Value is a tuple of 2-tuples; the RHS string is the
        # second element of each inner tuple.
        if not isinstance(node.value, ast.Tuple):
            continue
        for inner in node.value.elts:
            if not isinstance(inner, ast.Tuple):
                continue
            if len(inner.elts) != 2:
                continue
            rhs = inner.elts[1]
            if isinstance(rhs, ast.Constant) and isinstance(
                rhs.value, str
            ):
                rhs_values.add(rhs.value)
    return rhs_values


def _extract_dimension_bucket_map_keys() -> set[str]:
    """Return the set of keys declared in
    ``cockpit/manifest.py::_DIMENSION_BUCKET_MAP``. Walks the
    AST so the pin doesn't import the cockpit module (which
    transitively imports the warnings enum + drags the typing
    surface).
    """
    tree = ast.parse(
        COCKPIT_MANIFEST_PATH.read_text(encoding="utf-8"),
        filename=str(COCKPIT_MANIFEST_PATH),
    )
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            # Look for both annotated and bare assignments.
            if not isinstance(node, ast.Assign):
                continue
            target = node.targets[0] if node.targets else None
        else:
            target = node.target
        if not isinstance(target, ast.Name):
            continue
        if target.id != "_DIMENSION_BUCKET_MAP":
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key_node in node.value.keys:
            if isinstance(key_node, ast.Constant) and isinstance(
                key_node.value, str
            ):
                keys.add(key_node.value)
    return keys


class TestCockpitDimensionCategoriesPinned(unittest.TestCase):
    """Pin every ``_CATEGORY_FROM_PREFIX`` RHS to a
    ``_DIMENSION_BUCKET_MAP`` key.
    """

    def test_dimension_categories_pinned_to_bucket_map(self):
        rhs_values = _extract_category_from_prefix_rhs()
        bucket_keys = _extract_dimension_bucket_map_keys()

        self.assertGreater(
            len(rhs_values),
            0,
            "AST walker found zero RHS entries in "
            "_CATEGORY_FROM_PREFIX. Either the tuple was "
            "renamed or the parser drifted. Verify "
            "src/prismpy/pipeline/executor.py.",
        )
        self.assertGreater(
            len(bucket_keys),
            0,
            "AST walker found zero keys in "
            "_DIMENSION_BUCKET_MAP. Either the mapping was "
            "renamed or the parser drifted. Verify "
            "src/prismpy/cockpit/manifest.py.",
        )

        missing = rhs_values - bucket_keys
        self.assertFalse(
            missing,
            "executor.py::_CATEGORY_FROM_PREFIX emits "
            "category value(s) that are NOT keys in "
            "cockpit/manifest.py::_DIMENSION_BUCKET_MAP: "
            f"{sorted(missing)!r}. Every category the "
            "producer emits must have a bucket-routing entry "
            "in the consumer; otherwise the cockpit raises "
            "UnknownCategoryError at manifest-build time on "
            "any real project that emits the missing "
            "category, and the user sees the pre-E.0 "
            "fallback banner on a valid project. "
            "Add the missing key(s) to _DIMENSION_BUCKET_MAP "
            "with the appropriate bucket integer (default 3 "
            "TRUE_EXCLUDE for cell-level disqualifications; "
            "4 INTERPOLATABLE if Sprint E.2 has shipped and "
            "the category supports gap interpolation).",
        )

    def test_bucket_map_keys_route_to_known_buckets(self):
        """Sanity — every value in ``_DIMENSION_BUCKET_MAP``
        must be a recognized bucket integer (0/2/3/4/5)."""
        # Import lazily so the AST-only pin path above stays
        # cheap; this test does pay the import cost.
        from prismpy.cockpit.manifest import _DIMENSION_BUCKET_MAP
        valid_buckets = {0, 2, 3, 4, 5}
        misrouted = {
            cat: bucket
            for cat, bucket in _DIMENSION_BUCKET_MAP.items()
            if bucket not in valid_buckets
        }
        self.assertFalse(
            misrouted,
            "_DIMENSION_BUCKET_MAP carries category(ies) "
            f"routed to non-existent bucket integer(s): "
            f"{misrouted!r}. Valid buckets per "
            f"_BUCKET_INTEGER_MAP: 0 (AUTO_FIXABLE), 2 "
            "(INFORMATIONAL), 3 (TRUE_EXCLUDE), 4 "
            "(INTERPOLATABLE), 5 (MANUAL_OVERRIDE).",
        )


if __name__ == "__main__":
    unittest.main()
