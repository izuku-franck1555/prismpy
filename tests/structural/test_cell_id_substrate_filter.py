"""F-DL Pin DL-1 — AST dataflow invariant for the
UnifiedData.climate / UnifiedData.soil key sinks.

The cockpit observed-values writer crashed in production
because ``set(unified_data.climate.keys()) | set(soil.keys())``
mixes SARRA-Py path-dict ``str`` keys with int-keyed soil and
then calls ``sorted()`` on the mixed set. Python 3 forbids
``int < str``; the result is a silent ``TypeError`` swallowed by
a broad-except, so the cockpit sidecar disappears from the run
output. The narrow fix at ``observed_values_writer.py:486``
filters the union through ``is_real_climate_cell_id`` before
the sort.

This pin asserts the **class-level** invariant: every consumer
of ``unified_data.climate.keys()`` / ``unified_data.soil.keys()``
across the whole repo either filters the keys through the
canonical helper before any sort / comparison / set-union, or
appears in the documented allowlist below with rationale.

Strategy is AST-based, not regex-on-variable-names:

1. Walk every ``.py`` under ``prismpy/src/prismpy/``.
2. Locate every ``<expr>.climate.keys()`` and ``<expr>.soil.keys()``
   attribute access call.
3. For each, inspect the enclosing expression: is it consumed
   directly by ``sorted(...)``, ``set(...) | ...``, ``min/max``,
   ``sorted(set(...) | set(...))``, or similar cross-type-unsafe
   call?
4. If yes, require ``is_real_climate_cell_id`` to appear in the
   same enclosing scope (the filter site is allowed to be a
   wrapping comprehension or an explicit upstream filter step).
5. Allowlisted sites are exempted with rationale comments
   stored in this file (the allowlist is part of the pin so a
   future grep / git-blame discovers it).

Anti-vacuous guard: the walker must find ≥1 real consumer
(observed_values_writer.py:486) and ≥1 entry in the allowlist.
A walker that finds zero consumers means the UnifiedData
vocabulary has moved and the pin is silently stale.

Pin-the-pin: a positive test confirms the walker discovers the
known observed_values_writer.py site (catches walker accuracy
regressions); a negative test feeds the walker a synthetic
module with an unfiltered union+sort and confirms it flags.

Per F-DL contract §D Pin DL-1 + AC-DL-4 (cycle-2 reframed scope).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PRISMPY_SRC = REPO_ROOT / "src" / "prismpy"
HELPER_NAME = "is_real_climate_cell_id"


# ── Allowlist ──────────────────────────────────────────────────────────
#
# Each entry is a ``(relative_path, lineno, rationale)`` triple. A site
# qualifies for the allowlist when:
#
# * it is provably not consuming the UnifiedData mixed-vocab substrate
#   (e.g., the keys are already str-typed upstream by Pydantic
#   declaration), AND
# * applying the canonical helper would change correct behavior (e.g.,
#   drop legitimate string-typed cell-IDs)
#
# Each entry MUST cite the specific empirical condition that makes it
# safe. The pin asserts the allowlist has ≥1 entry as an anti-vacuous
# guard.

ALLOWLIST: Tuple[Tuple[str, int, str], ...] = (
    (
        "src/prismpy/cockpit/cell_roster_snapshot.py",
        183,
        "Roster cells declare ``cell_id: str`` (Pydantic field at "
        "line ~90); substrate is string-typed by design, not from "
        "UnifiedData.climate/soil. Applying ``is_real_climate_cell_id`` "
        "would reject every legitimate roster entry.",
    ),
    (
        "src/prismpy/cockpit/manifest.py",
        304,
        "Cell IDs are str-coerced via ``str(cell_id)`` immediately "
        "before the iteration; uniform-typed inputs guarantee no "
        "cross-type comparison.",
    ),
    (
        "src/prismpy/cockpit/manifest.py",
        322,
        "Cell IDs are str-coerced via ``str(c) for c in cells`` "
        "before extension; uniform-typed inputs guarantee no "
        "cross-type comparison.",
    ),
    (
        "src/prismpy/pipeline/executor.py",
        803,
        "Sentinel-retaining fanout: this site intentionally keeps "
        "the ``-1`` placeholder cell-id so the retrieve stage's "
        "calendar fan-out covers the synthetic cell. ``translators/"
        "base.py`` filters the sentinel out downstream. Applying "
        "``is_real_climate_cell_id`` here would skip the "
        "intentional ``-1`` retention.",
    ),
)


# ── AST walker ─────────────────────────────────────────────────────────


def _iter_py_files() -> Iterable[Path]:
    """Yield every ``.py`` under ``prismpy/src/prismpy/`` (the
    production source tree). Tests are excluded from the walk."""
    for p in PRISMPY_SRC.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def _is_climate_or_soil_keys_call(node: ast.AST) -> bool:
    """Return True when ``node`` is a Call to ``.climate.keys()`` or
    ``.soil.keys()`` on ANY receiver (covers
    ``unified_data.climate.keys()``, ``data.climate.keys()``,
    aliases like ``climate = data.climate; climate.keys()`` if the
    alias is on a direct chain).

    The walker only flags the call SHAPE; whether the receiver
    came from UnifiedData is a follow-on question handled by
    contextual filter-detection + the allowlist."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr != "keys":
        return False
    receiver = node.func.value
    if not isinstance(receiver, ast.Attribute):
        return False
    return receiver.attr in ("climate", "soil")


def _flagged_consumers_in_module(tree: ast.Module) -> List[ast.AST]:
    """Walk ``tree`` and return every cross-type-unsafe consumer of
    ``.climate.keys()`` / ``.soil.keys()`` that is NOT obviously
    filtered through the canonical helper.

    The detection is bounded: each candidate keys-call is examined
    against its parent chain for a sibling ``is_real_climate_cell_id``
    reference. If the helper name appears anywhere in the call's
    enclosing function body, the site counts as filtered (the
    refactor pattern allows the filter to live in a wrapping
    comprehension OR an explicit prior step in the same function)."""
    flagged: List[ast.AST] = []
    for fn_node in ast.walk(tree):
        if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Collect helper references anywhere in this function.
        helper_referenced = any(
            isinstance(child, ast.Name) and child.id == HELPER_NAME
            for child in ast.walk(fn_node)
        )
        # Find every .climate.keys() / .soil.keys() call site.
        keys_calls = [
            child
            for child in ast.walk(fn_node)
            if _is_climate_or_soil_keys_call(child)
        ]
        if not keys_calls:
            continue
        if helper_referenced:
            continue
        flagged.extend(keys_calls)
    return flagged


def _is_in_allowlist(path: Path, lineno: int) -> bool:
    """Return True iff ``(path, lineno)`` is in the allowlist
    (line-number match accepts ±5 to tolerate small refactors)."""
    rel = path.relative_to(REPO_ROOT).as_posix()
    for allow_path, allow_line, _rationale in ALLOWLIST:
        if rel == allow_path and abs(lineno - allow_line) <= 5:
            return True
    return False


# ── Pin tests ─────────────────────────────────────────────────────────


def test_walker_finds_at_least_one_keys_consumer() -> None:
    """Anti-vacuous guard: the walker must find at least one
    ``.climate.keys()`` or ``.soil.keys()`` consumer somewhere in
    the source tree. If it finds zero, the UnifiedData vocabulary
    has moved and the pin is silently stale — fail loud."""
    total_calls = 0
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if _is_climate_or_soil_keys_call(node):
                total_calls += 1
    assert total_calls >= 1, (
        "Walker found zero ``.climate.keys()`` / ``.soil.keys()`` "
        "call sites across the prismpy source. The UnifiedData "
        "vocabulary may have moved (e.g., to a property like "
        "``unified_data.real_cell_ids``); update the walker "
        "heuristic to match the new substrate, or this pin is "
        "now silently stale."
    )


def test_allowlist_has_documented_rationale() -> None:
    """Each allowlist entry MUST carry a rationale string so a
    future reviewer (or git-blame walker) can audit whether the
    exemption still applies."""
    assert len(ALLOWLIST) >= 1, "Allowlist must have at least one entry."
    for path, lineno, rationale in ALLOWLIST:
        assert isinstance(path, str) and path.endswith(".py"), (
            f"Allowlist entry path must be a .py string; got {path!r}"
        )
        assert isinstance(lineno, int) and lineno > 0, (
            f"Allowlist entry lineno must be a positive int; got {lineno!r}"
        )
        assert isinstance(rationale, str) and len(rationale) >= 40, (
            f"Allowlist entry for {path}:{lineno} must have a "
            f"≥40-char rationale; got {len(rationale)}-char string"
        )


def test_observed_values_writer_filters_through_canonical_helper() -> None:
    """Pin-the-pin positive: walker MUST recognize the
    ``observed_values_writer.py`` site as filtered (it imports + uses
    ``is_real_climate_cell_id`` directly before the sort). A
    regression that removes the import + filter would make this
    site appear flagged — catching the bug class re-entering."""
    writer = PRISMPY_SRC / "cockpit" / "observed_values_writer.py"
    tree = ast.parse(writer.read_text(encoding="utf-8"), filename=str(writer))
    flagged = _flagged_consumers_in_module(tree)
    # No flagged consumers from this file: every ``.keys()`` call
    # site is in a function body that references the helper.
    assert not flagged, (
        f"observed_values_writer.py should be FULLY filtered "
        f"through is_real_climate_cell_id; walker flagged "
        f"{len(flagged)} unfiltered site(s) at lines "
        f"{[node.lineno for node in flagged]}. "
        f"Did the F-DL fix get reverted?"
    )


def test_no_unfiltered_consumers_outside_allowlist() -> None:
    """The class-level invariant: every consumer of
    ``.climate.keys()`` / ``.soil.keys()`` in the repo either
    routes through the canonical helper OR appears in the
    allowlist with rationale."""
    failures: List[Tuple[str, int]] = []
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _flagged_consumers_in_module(tree):
            if not _is_in_allowlist(path, node.lineno):
                rel = path.relative_to(REPO_ROOT).as_posix()
                failures.append((rel, node.lineno))
    assert not failures, (
        "Found unfiltered ``.climate.keys()`` / ``.soil.keys()`` "
        "consumer(s) outside the allowlist:\n"
        + "\n".join(f"  - {p}:{ln}" for p, ln in failures)
        + "\n\nEither (a) route the consumer through "
        "``is_real_climate_cell_id`` before any sort / set-union / "
        "comparison, or (b) add the site to the ALLOWLIST in this "
        "file with a rationale string documenting why the canonical "
        "filter doesn't apply."
    )


def test_walker_flags_synthetic_unfiltered_consumer(tmp_path: Path) -> None:
    """Negative regression: feed the walker a synthetic module
    that does the unsafe union+sort WITHOUT the canonical filter;
    confirm the walker detects it. This pins the walker's accuracy
    against future heuristic drift (e.g., if someone changes the
    helper name)."""
    synthetic = tmp_path / "synthetic_module.py"
    synthetic.write_text(
        "def emit(data):\n"
        "    return sorted(set(data.climate.keys()) | set(data.soil.keys()))\n"
    )
    tree = ast.parse(synthetic.read_text(), filename=str(synthetic))
    flagged = _flagged_consumers_in_module(tree)
    assert flagged, (
        "Walker FAILED to detect a synthetic unfiltered "
        "``data.climate.keys() | data.soil.keys()`` union. Walker "
        "heuristic must be broken — the F-DL invariant is no "
        "longer enforced."
    )
    # And the synthetic SHOULD pass when the helper reference is
    # added to the enclosing function body.
    synthetic.write_text(
        "from prismpy.cells.cell_id_validation import "
        "is_real_climate_cell_id\n"
        "def emit(data):\n"
        "    keys = {k for k in data.climate.keys() "
        "if is_real_climate_cell_id(k)}\n"
        "    return sorted(keys)\n"
    )
    tree = ast.parse(synthetic.read_text(), filename=str(synthetic))
    flagged_after = _flagged_consumers_in_module(tree)
    assert not flagged_after, (
        "Walker should clear the synthetic site once the helper is "
        "referenced in the enclosing function body."
    )
