"""Behavioral tests for the cockpit_overrides.json sidecar writer.

Sprint E.3 AC-E3-7 sub-criteria 1-7 + Drill-E3-Q (all-reverted-bulk)
+ Drill-E3-R (partial-bulk-revert) + WA CA-20 atomicity drill.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from prismpy.cockpit.cockpit_overrides_writer import (
    CockpitOverrideSidecar,
    OverrideSidecarEntry,
    SCHEMA_VERSION,
    write_cockpit_overrides_json,
)


# ── Fixture builders ───────────────────────────────────────────────


def _value_replacement_record(
    *,
    cell_id: str = "c001",
    check_id: str = "value_range_tmax",
    decision_id: str = None,
    climate_values: dict = None,
    soil_values: dict = None,
    evidence_type: str = "field_observation",
    documentary_basis: str = None,
):
    """Mock record_dict (post-model_dump shape) for a value-
    replacement override."""
    decision_id = decision_id or str(uuid4())
    return {
        "decision_id": decision_id,
        "cell_id": cell_id,
        "action": "document_override",
        "check_id": check_id,
        "timestamp": datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
        "user_identity": "test-user",
        "method_or_rationale": "test rationale",
        "sequence_number": 1,
        "interpolation_record": None,
        "revert_of": None,
        "bulk_operation_id": None,
        "override_record": {
            "override_climate_values": climate_values
            or {"tmax_growing_season_mean": 32.5},
            "override_soil_values": soil_values,
            "evidence_type": evidence_type,
            "evidence_type_other_specify": None,
            "documentary_basis_other_specify": None,
            "evidence_detail": "Sahel hot-day extreme documented at field station 2024-07-15.",
            "applied_at_decision_id": decision_id,
            "applied_to_scope": "single_cell",
            "applied_to_zone_id": None,
            "applied_to_snapshot": [cell_id],
            "check_id": check_id,
            "category_d_documentary_basis": documentary_basis,
        },
    }


def _cat_d_record(
    *,
    cell_id: str = "c002",
    check_id: str = "crop_region_mismatch",
    decision_id: str = None,
):
    """Mock record_dict for a Cat D documentary override (filtered
    by writer per codex CA-3 absorption)."""
    decision_id = decision_id or str(uuid4())
    return {
        "decision_id": decision_id,
        "cell_id": cell_id,
        "action": "document_override",
        "check_id": check_id,
        "timestamp": datetime(2026, 5, 9, 12, 0, 1, tzinfo=timezone.utc).isoformat(),
        "user_identity": "test-user",
        "method_or_rationale": "irrigation",
        "sequence_number": 2,
        "interpolation_record": None,
        "revert_of": None,
        "bulk_operation_id": None,
        "override_record": {
            "override_climate_values": None,
            "override_soil_values": None,
            "evidence_type": "irrigation",
            "evidence_type_other_specify": None,
            "documentary_basis_other_specify": None,
            "evidence_detail": "Sub-watershed irrigated by the Lagdo dam since 1982.",
            "applied_at_decision_id": decision_id,
            "applied_to_scope": "zone",
            "applied_to_zone_id": "BSh",
            "applied_to_snapshot": [cell_id],
            "check_id": check_id,
            "category_d_documentary_basis": "irrigation_infrastructure",
        },
    }


# ── §1 sidecar round-trip + extra-forbid ───────────────────────────


def test_sidecar_pydantic_round_trip(tmp_path: Path) -> None:
    """AC-E3-7 sub-1: writes valid JSON conforming to extended
    schema; round-trips through CockpitOverrideSidecar with
    extra='forbid'."""
    record = _value_replacement_record()
    block = {"c001": {"value_range_tmax": record}}
    output_path = tmp_path / "cockpit_overrides.json"

    write_cockpit_overrides_json(
        cockpit_overrides_at_launch=block,
        output_path=output_path,
    )

    raw = json.loads(output_path.read_text())
    sidecar = CockpitOverrideSidecar(**raw)
    assert sidecar.schema_version == SCHEMA_VERSION
    assert len(sidecar.overrides) == 1
    entry = sidecar.overrides[0]
    assert entry.cell_id == "c001"
    assert entry.check_id == "value_range_tmax"
    assert entry.variable_key == "tmax_growing_season_mean"
    assert entry.value == 32.5
    assert entry.unit == "C"
    assert entry.evidence_type == "field_observation"


def test_sidecar_extra_forbid_rejects_typo(tmp_path: Path) -> None:
    """A typo'd field name on the sidecar payload rejects at
    construction. Pin the extra='forbid' contract."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="typo_field"):
        CockpitOverrideSidecar(
            schema_version="1.0",
            produced_at=datetime.now(timezone.utc),
            overrides=[],
            typo_field="should fail",  # type: ignore[call-arg]
        )


# ── §2 Cat D filtering (codex CA-3) ────────────────────────────────


def test_cat_d_documentary_row_filtered_from_sidecar(tmp_path: Path) -> None:
    """AC-E3-7 sub-2: Cat D documentary rows correctly filtered.

    Drill: a Cat D row in the snapshot block does NOT appear in
    the sidecar overrides[] array. Per codex CA-3 absorbed: Cat
    D rows are audit/methods-only — the documentary basis IS the
    override; no value to dispatch to translator."""
    block = {
        "c001": {"value_range_tmax": _value_replacement_record()},
        "c002": {"crop_region_mismatch": _cat_d_record()},
    }
    output_path = tmp_path / "cockpit_overrides.json"
    write_cockpit_overrides_json(
        cockpit_overrides_at_launch=block,
        output_path=output_path,
    )
    sidecar = CockpitOverrideSidecar(**json.loads(output_path.read_text()))
    cell_ids = [entry.cell_id for entry in sidecar.overrides]
    assert "c001" in cell_ids
    assert "c002" not in cell_ids, (
        "Cat D documentary row leaked into sidecar — per codex "
        "CA-3 + AC-E3-7 sub-2, Cat D rows MUST stay audit-only"
    )


# ── §3 all-reverted-bulk semantic (Drill-E3-Q + codex MED-2) ───────


def test_empty_overrides_block_writes_empty_list(tmp_path: Path) -> None:
    """Drill-E3-Q + codex MED-2: empty cockpit_overrides_at_launch
    block writes sidecar with overrides: [] empty array. The empty-
    array file SHOULD still emit so consumers can distinguish
    'writer fired with no overrides' from 'writer never fired'
    (silent-skip class violation per
    feedback_no_data_cooking.md)."""
    output_path = tmp_path / "cockpit_overrides.json"
    write_cockpit_overrides_json(
        cockpit_overrides_at_launch={},
        output_path=output_path,
    )
    raw = json.loads(output_path.read_text())
    assert raw["overrides"] == []
    assert raw["schema_version"] == "1.0"


def test_all_reverted_bulk_writes_empty_list(tmp_path: Path) -> None:
    """When every value-replacement entry has been reverted, the
    block carries None values; sidecar emits empty overrides[]."""
    block = {
        "c001": {"value_range_tmax": None},
        "c002": {"value_range_tmin": None},
    }
    output_path = tmp_path / "cockpit_overrides.json"
    write_cockpit_overrides_json(
        cockpit_overrides_at_launch=block,
        output_path=output_path,
    )
    raw = json.loads(output_path.read_text())
    assert raw["overrides"] == []


# ── §4 partial-bulk-revert (Drill-E3-R) ────────────────────────────


def test_partial_bulk_revert_preserves_non_reverted(tmp_path: Path) -> None:
    """Drill-E3-R: when some overrides are reverted (None) but
    others remain active, sidecar carries exactly the N-M
    non-reverted entries."""
    block = {
        "c001": {"value_range_tmax": _value_replacement_record(cell_id="c001")},
        "c002": {"value_range_tmin": None},  # reverted
        "c003": {
            "value_range_precip": _value_replacement_record(
                cell_id="c003",
                check_id="value_range_precip",
                climate_values={"precip_growing_season_total": 850.0},
            )
        },
    }
    output_path = tmp_path / "cockpit_overrides.json"
    write_cockpit_overrides_json(
        cockpit_overrides_at_launch=block,
        output_path=output_path,
    )
    sidecar = CockpitOverrideSidecar(**json.loads(output_path.read_text()))
    assert len(sidecar.overrides) == 2
    cell_ids = {entry.cell_id for entry in sidecar.overrides}
    assert cell_ids == {"c001", "c003"}


# ── §5 multi-variable override emits multiple entries ──────────────


def test_multi_variable_override_emits_per_variable_entries(
    tmp_path: Path,
) -> None:
    """A single OverrideRecord with multiple variable_keys in
    override_climate_values yields multiple sidecar entries
    (one per (cell_id, variable_key) pair) — the translator-
    side consumer dispatches per-variable_key directly."""
    record = _value_replacement_record(
        climate_values={
            "tmax_growing_season_mean": 32.5,
            "tmin_growing_season_mean": 18.0,
        }
    )
    block = {"c001": {"value_range_tmax": record}}
    output_path = tmp_path / "cockpit_overrides.json"
    write_cockpit_overrides_json(
        cockpit_overrides_at_launch=block,
        output_path=output_path,
    )
    sidecar = CockpitOverrideSidecar(**json.loads(output_path.read_text()))
    variable_keys = {entry.variable_key for entry in sidecar.overrides}
    assert variable_keys == {
        "tmax_growing_season_mean",
        "tmin_growing_season_mean",
    }


# ── §6 atomicity drill (WA CA-20) ──────────────────────────────────


def test_no_torn_artifact_on_simulated_mid_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """AC-E3-7 sub-4 + WA CA-20 absorbed: partial-write failure
    mode (interrupt mid-write) → torn artifacts NOT visible to
    consumers (write to temp file + rename pattern).

    Simulate the failure by patching ``os.replace`` to raise
    BEFORE the rename commits. The output_path MUST NOT exist
    after the failure (no half-written file visible)."""
    import os as os_module

    record = _value_replacement_record()
    block = {"c001": {"value_range_tmax": record}}
    output_path = tmp_path / "cockpit_overrides.json"

    def _replace_raises(*args, **kwargs):
        raise OSError("simulated mid-write failure")

    monkeypatch.setattr(os_module, "replace", _replace_raises)

    with pytest.raises(OSError, match="simulated mid-write failure"):
        write_cockpit_overrides_json(
            cockpit_overrides_at_launch=block,
            output_path=output_path,
        )

    # The output path MUST NOT exist — the rename was the commit
    # point and it never fired. Temp files may or may not exist
    # depending on the failure path; the consumer-visible
    # artifact MUST be absent.
    assert not output_path.exists(), (
        "Torn artifact at output_path after mid-write failure — "
        "atomicity contract violated per WA CA-20 + AC-E3-7 sub-4"
    )


def test_temp_file_cleaned_up_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """Companion to atomicity drill: the temp file should NOT
    persist in the output directory after a write failure (we
    explicitly unlink it on the exception path)."""
    import os as os_module

    record = _value_replacement_record()
    block = {"c001": {"value_range_tmax": record}}
    output_path = tmp_path / "cockpit_overrides.json"

    def _replace_raises(*args, **kwargs):
        raise OSError("simulated mid-write failure")

    monkeypatch.setattr(os_module, "replace", _replace_raises)

    with pytest.raises(OSError):
        write_cockpit_overrides_json(
            cockpit_overrides_at_launch=block,
            output_path=output_path,
        )

    # No leftover temp files (.cockpit_overrides.*.tmp).
    leftover = list(tmp_path.glob(".cockpit_overrides.*.tmp"))
    assert not leftover, (
        f"Leftover temp files after failure: {leftover}"
    )


# ── §7 byte-stable output ──────────────────────────────────────────


def test_writer_emits_byte_stable_output(tmp_path: Path) -> None:
    """Two writes of the same input produce byte-equivalent JSON
    files (modulo timestamp). Pin sorted-keys + sorted-overrides
    for determinism."""
    record = _value_replacement_record()
    block = {"c001": {"value_range_tmax": record}}
    fixed_ts = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)

    write_cockpit_overrides_json(
        cockpit_overrides_at_launch=block,
        output_path=tmp_path / "a.json",
        produced_at=fixed_ts,
    )
    write_cockpit_overrides_json(
        cockpit_overrides_at_launch=block,
        output_path=tmp_path / "b.json",
        produced_at=fixed_ts,
    )
    a_text = (tmp_path / "a.json").read_text()
    b_text = (tmp_path / "b.json").read_text()
    assert a_text == b_text, "Writer output is not byte-stable"
