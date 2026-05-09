"""Structural pin: ``CellDecisionRecord`` schema + ``current_decisions``
canonical reader.

Sprint E.2 AC-E2-4 + AC-E2-21 + §0.2 canonical-layer #1-#3 + #6.

Asserts:

* The Pydantic model has the contracted fields with the right types
  + ``extra="forbid"`` + ``validate_assignment=True``.
* The combined ``model_validator`` enforces:
  - ``action == "apply_interpolation"`` ⇒ interpolation_record present
  - ``action != "apply_interpolation"`` ⇒ interpolation_record absent
  - When present, self-link integrity (record.applied_at_decision_id
    == self.decision_id).
* ``current_decisions()`` correctly handles:
  - Single decision per cell.
  - Two decisions on same cell — most-recent wins.
  - Apply-then-revert — both filtered; cell maps to None.
  - Apply-revert-apply (write A, revert A, write C) — C is current.
  - Same-timestamp tie-break by sequence_number.
  - Shuffled-input invariance (caller need not pre-sort).
  - Empty input.
  - Multi-cell input.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from prismpy.models.decision_log import CellDecisionRecord, current_decisions
from prismpy.models.interpolated_cell import InterpolatedCellRecord


# ── Fixture builders ─────────────────────────────────────────────────


def _ts(seconds: int = 0) -> datetime:
    """Deterministic timestamps for ordering tests."""
    return datetime(2026, 5, 7, 12, 0, seconds, tzinfo=timezone.utc)


def _interp_record(decision_id: UUID) -> InterpolatedCellRecord:
    return InterpolatedCellRecord(
        interpolation_method="idw_k4_r15km_w_inverse_dist_sq",
        source_cells=["c001", "c002"],
        uncertainty_ci_lower=410.0,
        uncertainty_ci_upper=414.0,
        method_doi="10.1145/800186.810616",
        applied_at_decision_id=decision_id,
        affected_zone_code="BSh",
        caveat_codes=[],
    )


def _decision(
    *,
    cell_id: str = "c001",
    action: str = "skip_from_analysis",
    sequence_number: int = 1,
    timestamp: datetime | None = None,
    revert_of: UUID | None = None,
    decision_id: UUID | None = None,
    bulk_operation_id: UUID | None = None,
    with_interp: bool = False,
) -> CellDecisionRecord:
    decision_id = decision_id or uuid4()
    interp = _interp_record(decision_id) if with_interp else None
    return CellDecisionRecord(
        decision_id=decision_id,
        cell_id=cell_id,
        action=action,
        check_id="value_range_precip",
        timestamp=timestamp or _ts(sequence_number),
        user_identity="test-user",
        method_or_rationale="test rationale",
        sequence_number=sequence_number,
        interpolation_record=interp,
        revert_of=revert_of,
        bulk_operation_id=bulk_operation_id,
    )


# ── §1 schema invariants ────────────────────────────────────────────


def test_minimal_valid_decision_constructs() -> None:
    record = _decision()
    assert record.action == "skip_from_analysis"
    assert record.interpolation_record is None
    assert record.revert_of is None
    assert record.bulk_operation_id is None


def test_apply_interpolation_with_record_constructs() -> None:
    record = _decision(action="apply_interpolation", with_interp=True)
    assert record.interpolation_record is not None
    assert record.interpolation_record.applied_at_decision_id == record.decision_id


def test_apply_interpolation_without_record_rejected() -> None:
    with pytest.raises(ValidationError, match="MUST carry an interpolation_record"):
        _decision(action="apply_interpolation", with_interp=False)


def test_non_apply_action_with_record_rejected() -> None:
    """Symmetric absence: skip / override / acknowledge MUST NOT
    carry an interpolation_record per WA CA-2 sub-criterion."""
    decision_id = uuid4()
    interp = _interp_record(decision_id)
    with pytest.raises(ValidationError, match="MUST NOT carry"):
        CellDecisionRecord(
            decision_id=decision_id,
            cell_id="c1",
            action="skip_from_analysis",  # non-apply action
            check_id="value_range_precip",
            timestamp=_ts(),
            user_identity="u",
            method_or_rationale="r",
            sequence_number=1,
            interpolation_record=interp,  # SHOULD NOT be present
        )


def test_self_link_integrity_violation_rejected() -> None:
    """When interpolation_record is present, its applied_at_decision_id
    MUST equal the enclosing decision_id. Per §0.2 #6 + CA-H7."""
    enclosing_id = uuid4()
    different_id = uuid4()
    interp = _interp_record(different_id)  # mismatched UUID
    with pytest.raises(ValidationError, match="self-link integrity violation"):
        CellDecisionRecord(
            decision_id=enclosing_id,
            cell_id="c1",
            action="apply_interpolation",
            check_id="value_range_precip",
            timestamp=_ts(),
            user_identity="u",
            method_or_rationale="r",
            sequence_number=1,
            interpolation_record=interp,
        )


def test_unknown_action_rejected() -> None:
    """The DecisionAction Literal locks the four canonical actions."""
    with pytest.raises(ValidationError):
        _decision(action="rerun_full_sources")


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        CellDecisionRecord(
            decision_id=uuid4(),
            cell_id="c1",
            action="acknowledge",
            check_id="value_range_precip",
            timestamp=_ts(),
            user_identity="u",
            method_or_rationale="r",
            sequence_number=1,
            unknown_field="x",
        )


def test_validate_assignment_config_enabled() -> None:
    config = CellDecisionRecord.model_config
    assert config.get("validate_assignment") is True
    assert config.get("extra") == "forbid"


def test_bulk_operation_id_optional_uuid() -> None:
    bulk_uuid = uuid4()
    record = _decision(bulk_operation_id=bulk_uuid)
    assert record.bulk_operation_id == bulk_uuid


# ── §2 current_decisions reader semantics ───────────────────────────


def test_empty_input_returns_empty_dict() -> None:
    assert current_decisions([]) == {}


def test_single_decision_returns_that_decision() -> None:
    """Sprint E.3 AC-E3-6 reshape: keys are now ``(cell_id,
    check_id)`` tuples. The fixture's ``check_id`` is
    ``"value_range_precip"`` (per ``_decision`` factory)."""
    a = _decision(cell_id="c1", sequence_number=1)
    result = current_decisions([a])
    assert result == {("c1", "value_range_precip"): a}


def test_two_decisions_same_cell_most_recent_wins() -> None:
    """When two decisions land on the same ``(cell_id, check_id)``
    pair, the most-recent (later timestamp) wins per
    (timestamp, sequence_number) ordering."""
    a = _decision(cell_id="c1", sequence_number=1, timestamp=_ts(1))
    b = _decision(cell_id="c1", sequence_number=2, timestamp=_ts(2))
    result = current_decisions([a, b])
    assert result == {("c1", "value_range_precip"): b}


def test_apply_then_revert_filters_both() -> None:
    """Drill I (a): write A → revert A → both filtered;
    pair maps to None."""
    a = _decision(cell_id="c1", sequence_number=1, timestamp=_ts(1))
    revert_b = _decision(
        cell_id="c1",
        sequence_number=2,
        timestamp=_ts(2),
        revert_of=a.decision_id,
    )
    result = current_decisions([a, revert_b])
    assert result == {("c1", "value_range_precip"): None}


def test_apply_revert_apply_chain_returns_third_decision() -> None:
    """Drill I (a): write A, revert A, write C → C is current."""
    a = _decision(cell_id="c1", sequence_number=1, timestamp=_ts(1))
    revert_b = _decision(
        cell_id="c1",
        sequence_number=2,
        timestamp=_ts(2),
        revert_of=a.decision_id,
    )
    c = _decision(cell_id="c1", sequence_number=3, timestamp=_ts(3))
    result = current_decisions([a, revert_b, c])
    assert result == {("c1", "value_range_precip"): c}


def test_shuffled_input_produces_same_output_as_sorted(
) -> None:
    """Drill I (b) — shuffled-input invariance per WA Draft-2 CA-17.
    Caller need not pre-sort; the canonical reader sorts internally
    by (timestamp, sequence_number)."""
    a = _decision(cell_id="c1", sequence_number=1, timestamp=_ts(1))
    revert_b = _decision(
        cell_id="c1",
        sequence_number=2,
        timestamp=_ts(2),
        revert_of=a.decision_id,
    )
    c = _decision(cell_id="c1", sequence_number=3, timestamp=_ts(3))
    sorted_records = [a, revert_b, c]
    expected = current_decisions(sorted_records)
    rng = random.Random(42)
    for _ in range(6):
        shuffled = sorted_records[:]
        rng.shuffle(shuffled)
        assert current_decisions(shuffled) == expected, (
            f"Shuffled input produced different result; "
            f"shuffled order: {[r.decision_id for r in shuffled]}"
        )


def test_same_timestamp_resolves_by_sequence_number() -> None:
    """Drill I (c) — same-timestamp deterministic resolution per
    §0.2 canonical-source #2. A and B at SAME timestamp T with
    different sequence_numbers; B reverts A; assert (A, B both
    filtered → cell maps to None)."""
    same_t = _ts(1)
    a = _decision(cell_id="c1", sequence_number=1, timestamp=same_t)
    revert_b = _decision(
        cell_id="c1",
        sequence_number=2,
        timestamp=same_t,
        revert_of=a.decision_id,
    )
    # Both forward and reverse input order should produce same result.
    assert current_decisions([a, revert_b]) == {("c1", "value_range_precip"): None}
    assert current_decisions([revert_b, a]) == {("c1", "value_range_precip"): None}


def test_multi_cell_independent_state() -> None:
    """Different cells have independent decision histories. The
    tuple-keyed reshape per AC-E3-6 surfaces them as distinct
    ``(cell_id, check_id)`` pairs."""
    a1 = _decision(cell_id="c1", sequence_number=1, timestamp=_ts(1))
    a2 = _decision(cell_id="c2", sequence_number=2, timestamp=_ts(2))
    revert_a1 = _decision(
        cell_id="c1",
        sequence_number=3,
        timestamp=_ts(3),
        revert_of=a1.decision_id,
    )
    result = current_decisions([a1, a2, revert_a1])
    assert result == {
        ("c1", "value_range_precip"): None,
        ("c2", "value_range_precip"): a2,
    }


def test_apply_revert_remaining_counter_pattern() -> None:
    """Drill I (d) — apply N decisions on N distinct cells; revert
    one; assert N-1 pairs have current decisions + 1 pair maps to
    None. Models AC-E2-9 remaining-counter math + Sprint E.3
    AC-E3-6 tuple-keyed counter LOCK."""
    decisions = [
        _decision(cell_id=f"c{i}", sequence_number=i, timestamp=_ts(i))
        for i in range(1, 11)
    ]
    # Revert decision on c5.
    revert = _decision(
        cell_id="c5",
        sequence_number=11,
        timestamp=_ts(11),
        revert_of=decisions[4].decision_id,
    )
    result = current_decisions(decisions + [revert])
    # 10 pairs total; c5 maps to None; others map to their decisions.
    active_pairs = [key for key, decision in result.items() if decision is not None]
    inactive_pairs = [key for key, decision in result.items() if decision is None]
    assert len(active_pairs) == 9
    assert inactive_pairs == [("c5", "value_range_precip")]


def test_multi_check_coexistence_per_cell() -> None:
    """Sprint E.3 AC-E3-6 NEW: a single cell can carry multiple
    active decisions on different ``check_id``s. Override on
    ``value_range_tmax`` AND Acknowledge on ``value_range_tmin``
    on the same cell coexist as separate tuple-keyed entries.

    This is the core motivation for the reshape — Drill-E3-K-style
    multi-check coexistence per cell that the Sprint E.2 P2
    cell-only-keyed reader silently collapsed."""
    a_tmax = _decision(
        cell_id="c1",
        sequence_number=1,
        timestamp=_ts(1),
        action="acknowledge",
    )
    # Manually override the fixture's check_id to a different value.
    b_tmin = CellDecisionRecord(
        decision_id=uuid4(),
        cell_id="c1",
        action="acknowledge",
        check_id="value_range_tmin",
        timestamp=_ts(2),
        user_identity="test-user",
        method_or_rationale="separate check on same cell",
        sequence_number=2,
    )
    result = current_decisions([a_tmax, b_tmin])
    assert result == {
        ("c1", "value_range_precip"): a_tmax,
        ("c1", "value_range_tmin"): b_tmin,
    }
    # The user-visible counter (decided_pair_count per AC-E3-6
    # counter semantic LOCK) is the length of the dict.
    assert len(result) == 2


def test_dunder_all_lists_canonical_exports() -> None:
    from prismpy.models import decision_log
    assert sorted(decision_log.__all__) == [
        "CellDecisionRecord",
        "DecisionAction",
        "current_decisions",
    ]
