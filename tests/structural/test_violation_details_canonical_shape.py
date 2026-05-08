"""Sprint E.2 AC-E2-25 + Codex round 1 MEDIUM absorption —
every ``violation_details`` producer in scientific.py emits
the canonical 7-key entry shape.

The cockpit's ``cell_failed_check_details`` flattener at
``executor.py:3887-3905`` reads each producer's
``details["violation_details"]`` list and projects the canonical
keys into the per-cell summary file. The 7-key shape
``{cell_id, layer_idx, variable, date, value, unit, bounds}``
is the contract every per-cell-scoped check honors so the
flattener doesn't have to special-case any producer:

* ``cell_id`` — string cell ID (always populated)
* ``layer_idx`` — int (soil layered) or None (climate / coverage / temporal)
* ``variable`` — string (e.g., ``"tmax"``, ``"sand"``, ``"climate"``)
* ``date`` — ISO date string (per-day climate failures) or None
* ``value`` — numeric / None (None for absence-of-data signal)
* ``unit`` — string / None
* ``bounds`` — list[low, high] / None

Per durable §27 two-vocabulary substrate-drift: a producer that
emits 6 keys (omitting any of the canonical 7) violates the
shape contract — the flattener's tolerant ``vd.get("date")`` is
defensive, but partial pin coverage leaves the schema drift
silent. This pin AST-walks every literal ``violation_details``
producer + asserts the dict literal carries all 7 keys.

Pre-fix (Codex round 1 HIGH-1.5 / MEDIUM): the two coverage
producers (``_check_coverage_climate_cells`` +
``_check_coverage_soil_cells``) emitted 6-key dicts (no
``date``). The Codex MEDIUM finding called this out + the
fixup adds ``"date": None`` to both. This pin locks the close.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_FILE = (
    REPO_ROOT / "src" / "prismpy" / "validators" / "scientific.py"
)


CANONICAL_VIOLATION_DETAILS_KEYS: frozenset[str] = frozenset({
    "cell_id",
    "layer_idx",
    "variable",
    "date",
    "value",
    "unit",
    "bounds",
})


def _collect_violation_details_dict_literals(
    source: str,
) -> list[tuple[int, ast.Dict]]:
    """Walk the AST + return ``(line_no, ast.Dict)`` for every
    dict literal that appears in a context where it could be
    a ``violation_details`` entry.

    Heuristic: dict literals whose key set includes both
    ``cell_id`` AND ``variable`` are violation_details entries.
    The two keys appear together exclusively in this shape
    across scientific.py (verified empirically; soil_layer
    schemas + climate record models live in models/, not
    inside dict literals here).
    """
    tree = ast.parse(source)
    out: list[tuple[int, ast.Dict]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        # Collect literal key names from ast.Constant nodes.
        key_names: set[str] = set()
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                key_names.add(key.value)
        # Heuristic match: both ``cell_id`` and ``variable`` keys
        # are present. The conservative match avoids false
        # positives on unrelated dicts (e.g., a config dict that
        # happens to have ``cell_id`` alone).
        if {"cell_id", "variable"}.issubset(key_names):
            out.append((node.lineno, node))
    return out


@pytest.fixture(scope="module")
def violation_details_dicts() -> list[tuple[int, ast.Dict]]:
    """Discover every violation_details dict literal in
    scientific.py."""
    text = SCIENTIFIC_FILE.read_text(encoding="utf-8")
    return _collect_violation_details_dict_literals(text)


def test_at_least_six_violation_details_producers_discovered(
    violation_details_dicts,
) -> None:
    """Sanity floor — Sprint E.2 AC-E2-25 sub-4 ships per-cell
    violation_details across at least 6 producer sites: 2 soil
    (range + texture-sum) + 4 climate (range + region_bounds 4
    variables) + 1 temporal pivot (list-comp inside
    ``violation_details=[...]``) + 2 coverage (climate + soil
    cells). The pin asserts at least 6 are reachable; if the
    count drops, a producer was deleted without updating this
    pin."""
    assert len(violation_details_dicts) >= 6, (
        f"Expected >= 6 violation_details dict literals; got "
        f"{len(violation_details_dicts)}. Did a producer get "
        f"deleted without updating this pin? Or did the "
        f"discovery heuristic break?"
    )


def test_every_producer_emits_canonical_7_key_shape(
    violation_details_dicts,
) -> None:
    """Every literal ``violation_details`` entry across
    scientific.py producers MUST carry all 7 canonical keys.

    Drift detection: a future producer that emits 6 keys
    (omits ``date``, ``layer_idx``, ``unit``, etc.) silently
    drops information the cockpit drawer needs. The flattener's
    ``vd.get(<key>)`` tolerance hides the gap at runtime; this
    pin catches it at CI time.

    Pre-fix Codex round 1 MEDIUM finding: the two coverage
    producers emitted 6-key dicts (missing ``date``). This pin
    locks the close.
    """
    offenders: list[str] = []
    for line_no, dict_node in violation_details_dicts:
        actual_keys: set[str] = set()
        for key in dict_node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                actual_keys.add(key.value)
        missing = CANONICAL_VIOLATION_DETAILS_KEYS - actual_keys
        if missing:
            offenders.append(
                f"scientific.py:{line_no}: missing keys "
                f"{sorted(missing)}"
            )
    assert not offenders, (
        "violation_details producers missing canonical keys "
        "(7-key contract: cell_id, layer_idx, variable, date, "
        "value, unit, bounds — per Codex Gate A HIGH A2 + "
        "round 1 MEDIUM):\n  " + "\n  ".join(offenders)
    )
