"""Structural pin: action-vocabulary parity between AffordanceType
producer + AFFORDANCE_TO_ACTION_MAP + CellDecisionRecord.action consumer.

Sprint E.2 §0.2 canonical-source #1 + Drill Q. Asserts:

* ``set(typing.get_args(AffordanceType))`` covers exactly the
  ``AFFORDANCE_TO_ACTION_MAP`` keys.
* ``set(AFFORDANCE_TO_ACTION_MAP.values()) - {None}`` ⊆
  ``set(typing.get_args(DecisionAction))``.
* The ``"rerun_full_sources"`` exit-boundary maps to ``None`` (not
  to a CellDecisionRecord action).

Drift in either direction fails the pin loud per durable §24
canonical-source-or-pin discipline.

Drill Q (per §5) — mutation drills are evaluator's responsibility;
this pin enforces the structural invariant the drill exercises.
"""

from __future__ import annotations

import typing

from prismpy.models.decision_log import DecisionAction
from prismpy.validators.affordance_routing import (
    AFFORDANCE_TO_ACTION_MAP,
    AffordanceType,
)


def test_affordance_type_args_match_map_keys() -> None:
    """Every AffordanceType Literal value MUST have exactly one
    entry in AFFORDANCE_TO_ACTION_MAP, and every map key MUST be
    a Literal value."""
    affordance_args = set(typing.get_args(AffordanceType))
    map_keys = set(AFFORDANCE_TO_ACTION_MAP.keys())

    missing_in_map = affordance_args - map_keys
    orphan_in_map = map_keys - affordance_args

    assert not missing_in_map, (
        f"AffordanceType Literal members without a map entry: "
        f"{sorted(missing_in_map)}. Every AffordanceType MUST map "
        f"to a CellDecisionRecord action OR explicitly to None "
        f"(exit boundary)."
    )
    assert not orphan_in_map, (
        f"AFFORDANCE_TO_ACTION_MAP keys not in AffordanceType: "
        f"{sorted(orphan_in_map)}. Either remove the orphan keys "
        f"or extend the Literal."
    )


def test_map_values_are_subset_of_decision_action() -> None:
    """Every non-None map value MUST be a valid DecisionAction
    Literal member. The None values represent the exit-boundary
    affordances (no decision-log entry)."""
    decision_action_args = set(typing.get_args(DecisionAction))
    non_none_values = {
        v for v in AFFORDANCE_TO_ACTION_MAP.values() if v is not None
    }
    invalid = non_none_values - decision_action_args
    assert not invalid, (
        f"AFFORDANCE_TO_ACTION_MAP values that are NOT valid "
        f"CellDecisionRecord.action Literal members: {sorted(invalid)}. "
        f"Either fix the map OR extend DecisionAction."
    )


def test_rerun_full_sources_maps_to_none_exit_boundary() -> None:
    """Per §0.2 #1: ``"rerun_full_sources"`` is the EXIT BOUNDARY —
    spawns a new pipeline run rather than writing a decision-log
    entry. Drill Q (d) verifies consumer behavior on this case."""
    assert AFFORDANCE_TO_ACTION_MAP["rerun_full_sources"] is None


def test_four_per_cell_actions_map_to_decision_actions() -> None:
    """The four per-cell affordances each map to one of the four
    canonical CellDecisionRecord.action values."""
    expected = {
        "interpolate": "apply_interpolation",
        "skip": "skip_from_analysis",
        "override": "document_override",
        "acknowledge": "acknowledge",
    }
    for affordance, expected_action in expected.items():
        actual = AFFORDANCE_TO_ACTION_MAP[affordance]
        assert actual == expected_action, (
            f"AFFORDANCE_TO_ACTION_MAP[{affordance!r}] should map "
            f"to {expected_action!r}; got {actual!r}"
        )


def test_no_other_module_exports_affordance_type_alias() -> None:
    """Defensive scan: ``AffordanceType`` lives canonically in
    ``prismpy.validators.affordance_routing``. A future contributor
    re-defining the Literal in another module is a durable §24
    violation. This pin walks ``src/prismpy/`` for the literal
    string ``AffordanceType = Literal[`` and asserts only the
    canonical module declares it."""
    import re
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "src" / "prismpy"
    pattern = re.compile(r"\bAffordanceType\s*=\s*Literal\[")
    declarations: list[str] = []
    for py_file in src.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if pattern.search(text):
            declarations.append(str(py_file.relative_to(project_root)))
    assert declarations == ["src/prismpy/validators/affordance_routing.py"], (
        f"AffordanceType Literal must only be declared in "
        f"validators/affordance_routing.py; got: {declarations}"
    )
