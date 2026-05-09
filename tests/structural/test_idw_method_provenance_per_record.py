"""Structural pin: per-record IDW method provenance fields.

Sprint E.3 AC-E3-11 sub-6 + post-Draft 3 codex HIGH-2 absorbed
+ AC-E3-12 #11 (canonical pin registry). Three invariants close
the false-method-identity drift class per durable §24:

§1 InterpolatedCellRecord schema carries the 3 per-record
parameter fields (``radius_km`` / ``k`` / ``weight_power``).
A future schema change that drops them silently re-introduces
the false-method-identity-via-literal pattern.

§2 The orchestrator persists ``radius_km == IDW_RADIUS_BY_PLATFORM
[platform.value]`` per record. ACEA cells get radius_km=100.0,
NOT 15.0; PYTHIA cells get 25.0; SARRA-Py / CRAFT get 15.0.

§3 The canonical method literal (``"idw"``) is preferred for new
records; the legacy literal (``"idw_k4_r15km_w_inverse_dist_sq"``)
is accepted during the migration window for legacy rows.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from prismpy.cockpit.idw_orchestrator import (
    IdwOrchestrationTask,
    run_idw_orchestrator,
)
from prismpy.config.schema import Platform
from prismpy.harmonize.idw_interpolation import Cell
from prismpy.models.interpolated_cell import InterpolatedCellRecord
from prismpy.standards.idw_methods import (
    IDW_CANONICAL_METHOD_LITERAL,
    IDW_DEFAULT_K,
    IDW_DEFAULT_W,
)


# ── §1 schema carries the 3 parameter fields ───────────────────────


def test_record_schema_carries_radius_km_field() -> None:
    """The ``radius_km`` field MUST be on the schema. A future
    refactor that drops it silently re-introduces the
    false-method-identity-via-literal pattern."""
    assert "radius_km" in InterpolatedCellRecord.model_fields


def test_record_schema_carries_k_field() -> None:
    assert "k" in InterpolatedCellRecord.model_fields


def test_record_schema_carries_weight_power_field() -> None:
    assert "weight_power" in InterpolatedCellRecord.model_fields


def test_record_field_types_match_contract_spec() -> None:
    """Per AC-E3-11 sub-2: radius_km is float, k is int,
    weight_power is float. A type drift fires loud."""
    fields = InterpolatedCellRecord.model_fields
    assert fields["radius_km"].annotation is float
    assert fields["k"].annotation is int
    assert fields["weight_power"].annotation is float


# ── §2 orchestrator persists per-platform radius per record ────────


def _make_task(
    *,
    platform_value: str,
    target_id: str = "c001",
) -> IdwOrchestrationTask:
    """Build a task with 5 candidate cells in a tight cluster around
    the target, far enough from each other that all 4 platforms'
    radii (15 / 25 / 100 km) yield ≥4 candidates."""
    target = Cell(cell_id=target_id, lat=8.5, lon=13.2, value=0.0)
    candidates = tuple(
        Cell(
            cell_id=f"c{i:03d}",
            lat=8.5 + 0.005 * i,  # ~0.55 km north per increment
            lon=13.2 + 0.005 * i,
            value=400.0 + i * 5.0,
        )
        for i in range(2, 7)  # c002..c006, all within ~5 km
    )
    return IdwOrchestrationTask(
        target_cell=target,
        candidate_cells=candidates,
        affected_zone_code="BSh",
        decision_id=uuid4(),
        caveat_codes=(),
        platform_value=platform_value,
    )


def test_orchestrator_persists_acea_radius_100km() -> None:
    """CMS CA-1 BLOCKING + Drill-E3-P-ACEA: ACEA cells persist
    ``radius_km=100.0`` per record. A drift to 15 km would
    silently mismatch the actual dispatched radius."""
    task = _make_task(platform_value=Platform.ACEA.value)
    results = run_idw_orchestrator([task])
    assert len(results) == 1
    record = results[0].record
    assert record is not None
    assert record.radius_km == 100.0


def test_orchestrator_persists_pythia_radius_25km() -> None:
    task = _make_task(platform_value=Platform.PYTHIA.value)
    results = run_idw_orchestrator([task])
    record = results[0].record
    assert record is not None
    assert record.radius_km == 25.0


def test_orchestrator_persists_sarra_py_radius_15km() -> None:
    task = _make_task(platform_value=Platform.SARRA_PY.value)
    results = run_idw_orchestrator([task])
    record = results[0].record
    assert record is not None
    assert record.radius_km == 15.0


def test_orchestrator_persists_craft_radius_15km() -> None:
    task = _make_task(platform_value=Platform.CRAFT.value)
    results = run_idw_orchestrator([task])
    record = results[0].record
    assert record is not None
    assert record.radius_km == 15.0


def test_orchestrator_persists_canonical_k_and_weight_power() -> None:
    """Per-record k and weight_power match the canonical defaults
    (4 and 2.0) — Sprint E.3 ships a single kernel; future kernels
    extend the literal + the per-record numeric values together."""
    task = _make_task(platform_value=Platform.SARRA_PY.value)
    results = run_idw_orchestrator([task])
    record = results[0].record
    assert record is not None
    assert record.k == IDW_DEFAULT_K
    assert record.weight_power == IDW_DEFAULT_W


# ── §3 method literal is the canonical post-E.3 form ───────────────


def test_orchestrator_persists_canonical_method_literal() -> None:
    """Records produced by the orchestrator use the canonical
    ``"idw"`` literal, NOT the legacy parameter-encoded form. Per
    codex HIGH-2 absorption: per-record parameters carry the
    actual numeric values; the literal is the kernel-family
    handle."""
    task = _make_task(platform_value=Platform.SARRA_PY.value)
    results = run_idw_orchestrator([task])
    record = results[0].record
    assert record is not None
    assert record.interpolation_method == IDW_CANONICAL_METHOD_LITERAL


def test_legacy_literal_still_accepted_during_migration_window() -> None:
    """A pre-E.3 record constructed with the legacy literal still
    validates per the migration-window 2-arg Union. Post-migration
    tightening drops this acceptance (V3+ task)."""
    record = InterpolatedCellRecord(
        interpolation_method="idw_k4_r15km_w_inverse_dist_sq",
        source_cells=["c001", "c002"],
        uncertainty_ci_lower=410.0,
        uncertainty_ci_upper=414.0,
        method_doi="10.1145/800186.810616",
        applied_at_decision_id=uuid4(),
        affected_zone_code="BSh",
        caveat_codes=[],
    )
    assert record.interpolation_method == "idw_k4_r15km_w_inverse_dist_sq"
    # Default values fill the new fields for legacy construction.
    assert record.radius_km == 15.0
    assert record.k == 4
    assert record.weight_power == 2.0
