"""V2-22c-PRE.1.3 — assert no `[:N]` slicing on persisted per-cell lists.

Structural test (AST-walk) that catches a regression-by-revert: if any
future change re-introduces `[:N]` slicing on the un-truncated keys, the
cockpit's per-cell drill-down silently loses entries and the user sees a
truncated picture without any error. This test is the load-bearing
backstop for the §6.4 schema-bounds-match-strictest-downstream-consumer
discipline.

Sites un-truncated per PRE.1.3 (5 sites, after PRE.1.7 supersede):
  - validators/scientific.py — `gap_details`, `affected_cells` (×2),
    `sample_missing`
  - validators/post_translate.py — `sample_gaps`, `sample_duplicates`

Site explicitly NOT un-truncated (preserved per PRE.1.7):
  - validators/scientific.py — `sample_violations` text-string list at the
    `region_specific_bounds` emission (the un-truncated cell-id list
    lands on a separate `affected_cells` key per PRE.1.7).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "prismpy" / "validators"
SCIENTIFIC_PY = SRC_ROOT / "scientific.py"
POST_TRANSLATE_PY = SRC_ROOT / "post_translate.py"

# Keys that MUST NOT carry a `[:N]` slice in their assigned value, per
# PRE.1.3. Any new key added here also gets the un-truncation discipline.
UN_TRUNCATED_KEYS = {
    "gap_details",
    "affected_cells",
    "sample_missing",
    "sample_gaps",
    "sample_duplicates",
}

# Keys that ARE allowed to carry a `[:N]` slice — diagnostic text-string
# lists capped per PRE.1.7.
ALLOWED_TRUNCATED_KEYS = {
    "sample_violations",
}


def _is_constant_slice(node: ast.AST) -> bool:
    """Match `<expr>[:N]` where N is an integer constant — the
    truncation pattern PRE.1.3 forbids on the un-truncated keys.

    Returns True for `xs[:10]` / `xs[:5]` / `list(xs)[:20]`. Returns
    False for non-slice subscripts (`xs[0]`, `xs[k]`) and for slices
    bounded only by variables (`xs[:n]`) — the latter is rare and
    arguably out of scope for the static check.
    """
    if not isinstance(node, ast.Subscript):
        return False
    sl = node.slice
    if isinstance(sl, ast.Slice):
        upper = sl.upper
        # Treat any `[:N]` with a Constant upper as a truncation. A
        # Slice with `lower` set (e.g., `xs[5:10]`) is NOT a top-of-list
        # truncation in the PRE.1.3 sense and is left out of this
        # check — un-truncation is about avoiding "drop the tail."
        if sl.lower is None and upper is not None and isinstance(upper, ast.Constant):
            return True
    return False


def _walk_dict_values(tree: ast.AST):
    """Yield (key_name, value_node) pairs from every Dict literal in
    the AST where the key is a Constant string. Catches both
    ``{"affected_cells": list(xs)[:10]}`` and the more exotic
    dict-comprehension form by recursing into all expression nodes."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    yield k.value, v


def _find_truncation_violations(path: Path):
    """Return list of (lineno, key, snippet) for every un-truncation
    violation in the file. Empty list = clean."""
    tree = ast.parse(path.read_text())
    violations = []
    for key, value_node in _walk_dict_values(tree):
        if key not in UN_TRUNCATED_KEYS:
            continue
        if _is_constant_slice(value_node):
            snippet = ast.unparse(value_node)
            violations.append((value_node.lineno, key, snippet))
    return violations


class TestNoConstantSlicingOnUnTruncatedKeys:
    """The structural assertion that defines the un-truncation
    discipline as code, not commit-message convention."""

    def test_scientific_py_clean(self):
        violations = _find_truncation_violations(SCIENTIFIC_PY)
        assert not violations, (
            "PRE.1.3 regression — `[:N]` constant slice surfaced on a "
            f"key in UN_TRUNCATED_KEYS at {SCIENTIFIC_PY.name}: "
            + "; ".join(f"line {ln} {k}={s}" for ln, k, s in violations)
        )

    def test_post_translate_py_clean(self):
        violations = _find_truncation_violations(POST_TRANSLATE_PY)
        assert not violations, (
            "PRE.1.3 regression — `[:N]` constant slice surfaced on a "
            f"key in UN_TRUNCATED_KEYS at {POST_TRANSLATE_PY.name}: "
            + "; ".join(f"line {ln} {k}={s}" for ln, k, s in violations)
        )


class TestAllowedTruncationOnSampleViolations:
    """Backstop — confirm the structural rule does NOT over-enforce
    onto the diagnostic text-string list `sample_violations` that
    PRE.1.7 explicitly preserves at `[:10]`. If a future refactor
    accidentally moves `sample_violations` into `UN_TRUNCATED_KEYS`,
    this test fails and forces the author to read PRE.1.7's rationale
    before committing."""

    def test_sample_violations_is_in_allowed_truncated_set(self):
        assert "sample_violations" in ALLOWED_TRUNCATED_KEYS
        assert "sample_violations" not in UN_TRUNCATED_KEYS

    def test_sample_violations_truncation_still_present_at_emission_site(self):
        """Confirms PRE.1.7's preservation — region_specific_bounds
        keeps the diagnostic text-string sample capped at 10."""
        tree = ast.parse(SCIENTIFIC_PY.read_text())
        found_truncated_sample_violations = False
        for key, value_node in _walk_dict_values(tree):
            if key == "sample_violations" and _is_constant_slice(value_node):
                found_truncated_sample_violations = True
                break
        assert found_truncated_sample_violations, (
            "PRE.1.7 says the text-string `sample_violations[:10]` "
            "diagnostic is PRESERVED at region_specific_bounds. If this "
            "test fails, either the cap was removed (regression — "
            "operators want a 10-line digest, not a wall of text) or "
            "the emission site moved (refactor — re-anchor the "
            "structural test on the new location)."
        )

    def test_sample_violations_truncation_is_single_site(self):
        """Evaluator R5 Gate B addition — the
        ``ALLOWED_TRUNCATED_KEYS = {"sample_violations"}`` allowlist
        permits truncation at exactly ONE site (scientific.py:1498
        per PRE.1.7). A SECOND occurrence represents a discipline
        drift — text-string truncation is a localized diagnostic
        exception, not a pattern to copy. This test catches a
        future "consistency" refactor that bulk-adds ``[:N]`` to
        other text-list keys (e.g., a copy-paste at a sibling
        check that thinks it's matching the PRE.1.7 pattern).

        Scope: both validator files. The expected count is 1 — if
        a future PRE adds a second deliberate diagnostic-text cap
        elsewhere, that's a contract change that should be
        evaluated explicitly (update this expected count + extend
        the rationale comment), not absorbed silently.
        """
        matches = []
        for path in (SCIENTIFIC_PY, POST_TRANSLATE_PY):
            tree = ast.parse(path.read_text())
            for key, value_node in _walk_dict_values(tree):
                if key == "sample_violations" and _is_constant_slice(value_node):
                    matches.append((path.name, value_node.lineno))
        assert len(matches) == 1, (
            f"PRE.1.7 ALLOWED_TRUNCATED_KEYS allowlist expects "
            f"exactly one sample_violations[:N] site across the "
            f"validator files; found {len(matches)}: {matches!r}. "
            "If this is a deliberate addition, extend the "
            "ALLOWED_TRUNCATED_KEYS rationale + bump the expected "
            "count + document the new diagnostic-text discipline. "
            "If accidental (e.g., a 'consistency' refactor that "
            "copy-pasted the cap), un-truncate and route the "
            "structural list to a parallel `affected_cells` field "
            "per the PRE.1.7 dual-track pattern."
        )
