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
   attribute access call AND every ``<alias>.keys()`` call where
   ``<alias>`` is a tracked local alias of ``.climate`` /
   ``.soil`` (cycle-6 amendment — see "Alias tracking" below).
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

Alias tracking (cycle-6 amendment per codex BLOCKING):
production code at ``observed_values_writer.py:462-463`` binds
``climate = unified_data.climate or {}`` and ``soil =
unified_data.soil or {}`` and then writes ``climate.keys()``
/ ``soil.keys()`` at L499. The receiver of these ``.keys()``
calls is an ``ast.Name``, not an ``ast.Attribute``, so a walker
that only matches ``<expr>.climate.keys()`` (cycle-5 shape)
returns zero hits for the actual production site. The cycle-6
walker collects per-scope aliases for any name bound to a
``.climate`` or ``.soil`` attribute (handling the ``or``-fallback
idiom ``foo = bar.climate or {}`` and the ``if-else`` idiom
``foo = bar.climate if bar.climate else {}``) and recognizes
``<alias>.keys()`` as the same flag-shape as
``<expr>.climate.keys()``.

Anti-vacuous guard: the walker must find ≥1 real consumer
(observed_values_writer.py:499) and ≥1 entry in the allowlist.
A walker that finds zero consumers means the UnifiedData
vocabulary has moved and the pin is silently stale.

Pin-the-pin: a positive test confirms the walker discovers the
known observed_values_writer.py site (catches walker accuracy
regressions); a negative test feeds the walker a synthetic
module with an unfiltered union+sort and confirms it flags;
the cycle-6 positive test additionally pins the Name-receiver
pattern (alias `climate.keys()` / `soil.keys()` with the
``or {}`` fallback) so a future walker regression that drops
alias tracking is caught immediately.

Per F-DL contract §D Pin DL-1 + AC-DL-4 (cycle-2 reframed scope;
cycle-6 alias-tracking extension).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Set, Tuple

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


def _unwrap_or_default(value: ast.AST) -> ast.AST:
    """Unwrap the ``<expr> or <fallback>`` idiom (common None-safe
    alias pattern: ``climate = unified_data.climate or {}``) and
    the ``<expr> if <expr> else <fallback>`` alt idiom. Both reduce
    to the first / primary operand. Other expression shapes pass
    through unchanged."""
    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
        # ``a or b`` → take ``a`` (the alias source).
        return value.values[0] if value.values else value
    if isinstance(value, ast.IfExp):
        # ``a if cond else b`` → take ``a`` (the truthy branch).
        return value.body
    return value


def _collect_climate_soil_aliases(
    scope_node: ast.AST,
) -> Tuple[Set[str], Set[str]]:
    """Walk every assignment in ``scope_node`` and return
    ``(climate_aliases, soil_aliases)``: the sets of local names
    that bind directly to a ``.climate`` or ``.soil`` attribute
    of some receiver.

    Handles three idioms observed in production:

    * ``climate = unified_data.climate`` — direct attribute alias.
    * ``climate = unified_data.climate or {}`` — None-safe alias
      (the production pattern at ``observed_values_writer.py:462``).
    * ``climate = unified_data.climate if unified_data.climate
      else {}`` — ternary alt idiom.

    Multi-target assignments (``a = b = ...``) and tuple unpacking
    are NOT tracked (rare; would require dataflow rather than
    pattern match). Augmented assignments (``+=``) are excluded
    because they don't rebind the name to the attribute.

    The function walks the *entire* scope subtree (``ast.walk``)
    so nested ``if`` / ``try`` blocks that rebind the alias are
    captured. This intentionally over-collects: a name that's
    aliased on one branch and reassigned on another still
    qualifies. Over-collection is safe because the walker's job
    is to flag unsafe consumers — extra aliases only make the
    walker stricter, never weaker.

    Per cycle-6 amendment (codex BLOCKING)."""
    climate_aliases: Set[str] = set()
    soil_aliases: Set[str] = set()
    for sub in ast.walk(scope_node):
        if not isinstance(sub, ast.Assign):
            continue
        if len(sub.targets) != 1 or not isinstance(sub.targets[0], ast.Name):
            continue
        target_name = sub.targets[0].id
        value = _unwrap_or_default(sub.value)
        if isinstance(value, ast.Attribute):
            if value.attr == "climate":
                climate_aliases.add(target_name)
            elif value.attr == "soil":
                soil_aliases.add(target_name)
    return climate_aliases, soil_aliases


def _is_climate_or_soil_keys_call(
    node: ast.AST,
    climate_aliases: Set[str] = frozenset(),
    soil_aliases: Set[str] = frozenset(),
) -> bool:
    """Return True when ``node`` is a Call to ``.climate.keys()`` or
    ``.soil.keys()`` on ANY receiver, OR a Call to ``<alias>.keys()``
    where ``<alias>`` is a tracked local alias of a ``.climate`` or
    ``.soil`` attribute.

    Two patterns are recognized:

    1. **Attribute-receiver** (cycle-5 shape):
       ``<expr>.climate.keys()`` / ``<expr>.soil.keys()``. Receiver
       is ``ast.Attribute`` with ``attr in ("climate", "soil")``.
       Example: ``data.climate.keys()`` at
       ``translators/base.py:383``.

    2. **Name-receiver alias** (cycle-6 amendment): ``<alias>.keys()``
       where ``<alias>`` is in the per-scope alias set. Example:
       ``climate.keys()`` at
       ``cockpit/observed_values_writer.py:499`` where
       ``climate = unified_data.climate or {}`` is bound at L462.

    The walker only flags the call SHAPE; whether the receiver
    came from UnifiedData is a follow-on question handled by
    contextual filter-detection + the allowlist."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr != "keys":
        return False
    receiver = node.func.value
    # Pattern 1 — Attribute-receiver: <expr>.climate.keys() / <expr>.soil.keys()
    if isinstance(receiver, ast.Attribute) and receiver.attr in ("climate", "soil"):
        return True
    # Pattern 2 — Name-receiver alias: <alias>.keys() when alias tracked
    if isinstance(receiver, ast.Name):
        return receiver.id in climate_aliases or receiver.id in soil_aliases
    return False


def _flagged_consumers_in_module(tree: ast.Module) -> List[ast.AST]:
    """Walk ``tree`` and return every cross-type-unsafe consumer of
    ``.climate.keys()`` / ``.soil.keys()`` (including alias forms)
    that is NOT obviously filtered through the canonical helper.

    The detection is bounded: each candidate keys-call is examined
    against its parent chain for a sibling ``is_real_climate_cell_id``
    reference. If the helper name appears anywhere in the call's
    enclosing function body, the site counts as filtered (the
    refactor pattern allows the filter to live in a wrapping
    comprehension OR an explicit prior step in the same function).

    Per-scope alias tracking is built before each function's
    keys-call walk (cycle-6 amendment)."""
    flagged: List[ast.AST] = []
    for fn_node in ast.walk(tree):
        if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # Build alias sets first so the walker can recognize
        # Name-receiver patterns local to this function.
        climate_aliases, soil_aliases = _collect_climate_soil_aliases(fn_node)
        # Collect helper references anywhere in this function.
        helper_referenced = any(
            isinstance(child, ast.Name) and child.id == HELPER_NAME
            for child in ast.walk(fn_node)
        )
        # Find every .climate.keys() / .soil.keys() / alias.keys() call.
        keys_calls = [
            child
            for child in ast.walk(fn_node)
            if _is_climate_or_soil_keys_call(child, climate_aliases, soil_aliases)
        ]
        if not keys_calls:
            continue
        if helper_referenced:
            continue
        flagged.extend(keys_calls)
    return flagged


def _all_keys_calls_in_module(tree: ast.Module) -> List[ast.AST]:
    """Return every ``.climate.keys()`` / ``.soil.keys()`` /
    ``<alias>.keys()`` site in ``tree`` regardless of filter
    status. Used by the anti-vacuous walker to count call sites
    across the source tree (it must find at least one)."""
    calls: List[ast.AST] = []
    for fn_node in ast.walk(tree):
        if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        climate_aliases, soil_aliases = _collect_climate_soil_aliases(fn_node)
        for child in ast.walk(fn_node):
            if _is_climate_or_soil_keys_call(child, climate_aliases, soil_aliases):
                calls.append(child)
    return calls


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
    ``.climate.keys()`` / ``.soil.keys()`` / ``<alias>.keys()``
    consumer somewhere in the source tree. If it finds zero, the
    UnifiedData vocabulary has moved and the pin is silently stale
    — fail loud."""
    total_calls = 0
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        total_calls += len(_all_keys_calls_in_module(tree))
    assert total_calls >= 1, (
        "Walker found zero ``.climate.keys()`` / ``.soil.keys()`` / "
        "alias call sites across the prismpy source. The UnifiedData "
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


def test_walker_detects_production_site_via_name_receiver_alias() -> None:
    """Cycle-6 amendment positive pin: the walker MUST detect the
    production site at ``observed_values_writer.py`` where the
    cross-type union runs through Name-receiver aliases
    (``climate.keys()`` / ``soil.keys()`` rather than
    ``unified_data.climate.keys()``).

    Empirically, the cycle-5 walker returned zero hits for this
    file because both ``.keys()`` receivers are ``ast.Name``
    (``id="climate"`` and ``id="soil"``), not ``ast.Attribute``,
    so the walker's Attribute-only matcher rejected them. The
    positive pin `test_observed_values_writer_filters_through_
    canonical_helper` therefore PASSED VACUOUSLY (an empty
    ``flagged`` set has no flagged sites by definition), and a
    future revert of the L498-501 filter would not be caught by
    this pin file at all — only by Pin DL-2's runtime regression.

    This test pins the alias-tracking accuracy: it walks
    ``observed_values_writer.py``, collects all keys-call sites
    (including aliases), and asserts ≥2 sites are found at the
    production line (cycle-6 walker recognizes both
    ``climate.keys()`` and ``soil.keys()``). A future regression
    that drops alias tracking would return zero sites and this
    test would fail loud.

    Per cycle-6 amendment + codex BLOCKING."""
    writer = PRISMPY_SRC / "cockpit" / "observed_values_writer.py"
    tree = ast.parse(writer.read_text(encoding="utf-8"), filename=str(writer))
    all_calls = _all_keys_calls_in_module(tree)
    # The production site at L498-501 contains both climate.keys()
    # and soil.keys() — the walker must detect ≥2 sites from this
    # file (one each for the climate and soil aliases).
    assert len(all_calls) >= 2, (
        f"Walker should detect ≥2 keys-call sites in "
        f"observed_values_writer.py (climate.keys() + soil.keys() "
        f"at L498-501); got {len(all_calls)}. The alias-tracking "
        f"extension may have regressed."
    )
    # Verify the call sites are Name-receivers (the cycle-6
    # pattern), not Attribute-receivers — if they ever become
    # Attribute-receivers (e.g., refactored back to
    # ``unified_data.climate.keys()``), the cycle-5 walker would
    # have detected them and this pin is no longer pinning the
    # right pattern. Fail loud so we revisit the walker's scope.
    name_receiver_calls = [
        call for call in all_calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
    ]
    assert len(name_receiver_calls) >= 2, (
        f"Walker should detect ≥2 Name-receiver keys-call sites "
        f"in observed_values_writer.py (the cycle-6 alias pattern); "
        f"got {len(name_receiver_calls)} Name-receiver site(s). "
        f"If the production site refactored back to Attribute-"
        f"receiver pattern, drop this pin and rely on the cycle-5 "
        f"walker shape; otherwise the alias tracking regressed."
    )
    # Verify the detected sites are inside the writer's function
    # ``write_observed_values_json`` (sanity check on call-site
    # location — guards against a future refactor that moves the
    # union+sort elsewhere and the alias tracking happens to
    # match a different function's locals).
    site_linenos = sorted(call.lineno for call in name_receiver_calls)
    assert all(490 <= ln <= 510 for ln in site_linenos), (
        f"Expected Name-receiver keys-calls between L490-510 "
        f"(observed_values_writer.py write_observed_values_json "
        f"per-cell payload assembly); got linenos {site_linenos}. "
        f"Production site may have moved — verify the walker still "
        f"targets the right function."
    )


def test_no_unfiltered_consumers_outside_allowlist() -> None:
    """The class-level invariant: every consumer of
    ``.climate.keys()`` / ``.soil.keys()`` / ``<alias>.keys()`` in
    the repo either routes through the canonical helper OR
    appears in the allowlist with rationale."""
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


def test_walker_flags_synthetic_name_receiver_alias_pattern(
    tmp_path: Path,
) -> None:
    """Cycle-6 negative regression for the alias pattern: feed the
    walker a synthetic module that uses the production idiom
    (``climate = data.climate or {}; sorted(climate.keys() | ...)``)
    WITHOUT the canonical filter; confirm the walker detects it.
    This pins the alias-tracking heuristic against future
    drift (e.g., if someone refactors ``_collect_climate_soil_aliases``
    and breaks the ``or``-fallback unwrap)."""
    synthetic = tmp_path / "synthetic_alias_module.py"
    synthetic.write_text(
        "def emit(data):\n"
        "    climate = data.climate or {}\n"
        "    soil = data.soil or {}\n"
        "    return sorted(set(climate.keys()) | set(soil.keys()))\n"
    )
    tree = ast.parse(synthetic.read_text(), filename=str(synthetic))
    flagged = _flagged_consumers_in_module(tree)
    assert len(flagged) >= 2, (
        f"Walker FAILED to detect a synthetic Name-receiver alias "
        f"pattern ``climate = data.climate or {{}}; climate.keys()"
        f"``); got {len(flagged)} flagged site(s). Alias tracking "
        f"must be broken — the F-DL cycle-6 amendment is no "
        f"longer enforced and production-site regressions would "
        f"slip through."
    )
    # And the synthetic SHOULD clear when the helper is referenced.
    synthetic.write_text(
        "from prismpy.cells.cell_id_validation import "
        "is_real_climate_cell_id\n"
        "def emit(data):\n"
        "    climate = data.climate or {}\n"
        "    soil = data.soil or {}\n"
        "    keys = {k for k in (set(climate.keys()) | set(soil.keys())) "
        "if is_real_climate_cell_id(k)}\n"
        "    return sorted(keys)\n"
    )
    tree = ast.parse(synthetic.read_text(), filename=str(synthetic))
    flagged_after = _flagged_consumers_in_module(tree)
    assert not flagged_after, (
        "Walker should clear the synthetic alias site once the "
        "helper is referenced in the enclosing function body."
    )
