"""Behavioral tests for the IDW orchestrator.

Sprint E.3 AC-E3-11 + Drill-E3-D (empty-observed-values honest-
signal) + builder CA-2 location-LOCK validation.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from prismpy.cockpit.idw_orchestrator import (
    IdwOrchestrationResult,
    IdwOrchestrationTask,
    build_spatial_index,
    run_idw_orchestrator,
)
from prismpy.config.schema import Platform
from prismpy.harmonize.idw_interpolation import Cell


# ── Fixture builders ───────────────────────────────────────────────


def _task(
    *,
    target_id: str = "c001",
    n_candidates: int = 4,
    platform_value: str = "sarra_py",
    radius_buffer_km: float = 0.5,  # candidates within ~5 km
) -> IdwOrchestrationTask:
    """Build a task with N candidates in a tight cluster around the
    target so all 4 platforms' radii find ≥4 neighbours."""
    target = Cell(cell_id=target_id, lat=8.5, lon=13.2, value=0.0)
    candidates = tuple(
        Cell(
            cell_id=f"cand_{i:03d}",
            lat=8.5 + 0.005 * (i + 1),
            lon=13.2 + 0.005 * (i + 1),
            value=400.0 + i * 5.0,
        )
        for i in range(n_candidates)
    )
    return IdwOrchestrationTask(
        target_cell=target,
        candidate_cells=candidates,
        affected_zone_code="BSh",
        decision_id=uuid4(),
        caveat_codes=(),
        platform_value=platform_value,
    )


# ── §1 happy path ──────────────────────────────────────────────────


def test_single_task_returns_record() -> None:
    """Standard case: 4 candidates within radius → record built
    with imputed value + CI + source_cells."""
    task = _task(n_candidates=4)
    results = run_idw_orchestrator([task])
    assert len(results) == 1
    result = results[0]
    assert result.error is None
    assert result.record is not None
    assert result.decision_id == task.decision_id
    assert len(result.record.source_cells) == 4


def test_multiple_tasks_return_results_in_order() -> None:
    """Batch dispatch preserves task order in the result list so
    callers can index by position."""
    tasks = [_task(target_id=f"t{i:03d}") for i in range(3)]
    results = run_idw_orchestrator(tasks)
    assert len(results) == 3
    for task, result in zip(tasks, results):
        assert result.decision_id == task.decision_id


# ── §2 platform radius dispatch ────────────────────────────────────


def test_sarra_py_uses_15km_radius() -> None:
    task = _task(platform_value=Platform.SARRA_PY.value)
    results = run_idw_orchestrator([task])
    assert results[0].record is not None
    assert results[0].record.radius_km == 15.0


def test_acea_uses_100km_radius() -> None:
    """CMS CA-1 BLOCKING — ACEA cells dispatched with 100 km."""
    task = _task(platform_value=Platform.ACEA.value)
    results = run_idw_orchestrator([task])
    assert results[0].record is not None
    assert results[0].record.radius_km == 100.0


# ── §3 Drill-E3-D empty-observed-values ────────────────────────────


def test_empty_task_list_returns_empty_results() -> None:
    """Drill-E3-D: orchestrator no-ops with honest-signal warning
    when invoked with no tasks. Returns empty list (not None) so
    the caller doesn't crash on the no-tasks path."""
    results = run_idw_orchestrator([])
    assert results == []


def test_empty_task_list_emits_warning(caplog) -> None:
    """The honest-signal floor per ``feedback_no_data_cooking.md``:
    an empty task list MUST emit a warning rather than silently
    returning empty results. Callers that mis-thread the input
    should see the warning in logs."""
    import logging

    caplog.set_level(logging.WARNING)
    run_idw_orchestrator([])
    assert any(
        "empty task list" in record.message.lower()
        for record in caplog.records
    )


# ── §4 InsufficientNeighborsError defensive fallback ───────────────


def test_zero_candidates_returns_error_result() -> None:
    """When zero candidates fall within the radius (a state the
    affordance routing should prevent at AC-E2-3 but the
    orchestrator catches defensively), the result carries an
    error string rather than raising; the batch continues."""
    # Target with no nearby candidates (candidates 1000 km away)
    target = Cell(cell_id="c001", lat=8.5, lon=13.2, value=0.0)
    far_candidates = (
        Cell(cell_id="far_001", lat=18.5, lon=23.2, value=400.0),
    )
    task = IdwOrchestrationTask(
        target_cell=target,
        candidate_cells=far_candidates,
        affected_zone_code="BSh",
        decision_id=uuid4(),
        caveat_codes=(),
        platform_value=Platform.SARRA_PY.value,
    )
    results = run_idw_orchestrator([task])
    assert len(results) == 1
    assert results[0].record is None
    assert results[0].error is not None
    assert "InsufficientNeighborsError" in results[0].error or "neighbours" in results[0].error.lower()


# ── §5 build_spatial_index helper ──────────────────────────────────


def test_build_spatial_index_returns_index_for_non_empty_roster() -> None:
    cells = [Cell(cell_id=f"c{i}", lat=8.5, lon=13.2, value=0.0) for i in range(3)]
    index = build_spatial_index(cells)
    assert index is not None
    assert len(index) == 3


def test_build_spatial_index_returns_none_for_empty_roster() -> None:
    """Empty observed_values roster → None rather than crashing on
    the SpatialIndex constructor's empty-input ValueError. Closes
    the AC-E3-11 sub-3 honest-signal-on-empty path."""
    index = build_spatial_index([])
    assert index is None


# ── §6 read-only contract ──────────────────────────────────────────


def test_orchestrator_does_not_mutate_input_tasks() -> None:
    """The orchestrator's input task list MUST stay byte-equivalent
    before+after invocation. Pin via dataclass-frozen + structural
    inspection."""
    task = _task()
    target_before = task.target_cell
    candidates_before = task.candidate_cells
    decision_id_before = task.decision_id

    run_idw_orchestrator([task])

    assert task.target_cell is target_before
    assert task.candidate_cells is candidates_before
    assert task.decision_id == decision_id_before


def test_input_dataclass_is_frozen() -> None:
    """``IdwOrchestrationTask`` is frozen — direct attribute
    reassignment raises FrozenInstanceError. Pin the immutability
    invariant."""
    import dataclasses

    task = _task()
    with pytest.raises(dataclasses.FrozenInstanceError):
        task.platform_value = "different"  # type: ignore[misc]
