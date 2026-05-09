"""Phantom-bug pin: bulk-override revert preserves shape invariants.

Sprint E.3 AC-E3-13 #4 + codex CA-9 absorbed. The bulk-revert
shape invariants (Drill-E3-Q + Drill-E3-R per AC-E3-7):

§1 All-reverted-bulk → empty sidecar entries + manifest flag
False + methods text omits override paragraph (Drill-E3-Q).

§2 Partial-bulk-revert preserves exactly the non-reverted entries
(Drill-E3-R). N overrides committed, M reverted → sidecar
carries exactly N-M entries; manifest flag stays True.

The pins exercise the writer + Cat D filtering at
:mod:`prismpy.cockpit.cockpit_overrides_writer` end-to-end via
synthetic fixtures.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from prismpy.cockpit.cockpit_overrides_writer import (
    CockpitOverrideSidecar,
    write_cockpit_overrides_json,
)


def _record_dict(
    *,
    cell_id: str,
    check_id: str,
    decision_id: str = None,
    documentary_basis: str = None,
) -> dict:
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
            "override_climate_values": (
                {"tmax_growing_season_mean": 32.5}
                if documentary_basis is None
                else None
            ),
            "override_soil_values": None,
            "evidence_type": "field_observation",
            "evidence_type_other_specify": None,
            "documentary_basis_other_specify": None,
            "evidence_detail": "Sahel hot-day extreme documented at field station 2024.",
            "applied_at_decision_id": decision_id,
            "applied_to_scope": "single_cell",
            "applied_to_zone_id": None,
            "applied_to_snapshot": [cell_id],
            "check_id": check_id,
            "category_d_documentary_basis": documentary_basis,
        },
    }


# ── §1 all-reverted-bulk (Drill-E3-Q) ──────────────────────────────


def test_all_reverted_bulk_emits_empty_overrides_array(
    tmp_path: Path,
) -> None:
    """Drill-E3-Q: when every value-replacement override has been
    reverted, the snapshot block carries None values; the writer
    emits sidecar with overrides: [] empty array per codex MED-2
    absorption."""
    block = {
        "c001": {"value_range_tmax": None},
        "c002": {"value_range_tmin": None},
        "c003": {"value_range_precip": None},
    }
    output_path = tmp_path / "cockpit_overrides.json"
    write_cockpit_overrides_json(
        cockpit_overrides_at_launch=block,
        output_path=output_path,
    )
    sidecar = CockpitOverrideSidecar(**json.loads(output_path.read_text()))
    assert sidecar.overrides == [], (
        f"All-reverted-bulk MUST yield empty overrides[]; got "
        f"{len(sidecar.overrides)} entries: {sidecar.overrides}"
    )


def test_all_reverted_bulk_keeps_canonical_schema_version(
    tmp_path: Path,
) -> None:
    """Even in the all-reverted case the writer emits a complete
    sidecar with schema_version + produced_at populated. Closes
    the silent-skip class — consumers can distinguish 'writer
    fired with no entries' from 'writer never fired'."""
    output_path = tmp_path / "cockpit_overrides.json"
    write_cockpit_overrides_json(
        cockpit_overrides_at_launch={},
        output_path=output_path,
    )
    raw = json.loads(output_path.read_text())
    assert raw["schema_version"] == "1.0"
    assert "produced_at" in raw
    assert raw["overrides"] == []


# ── §2 partial-bulk-revert (Drill-E3-R) ────────────────────────────


def test_partial_bulk_revert_preserves_exactly_n_minus_m_entries(
    tmp_path: Path,
) -> None:
    """Drill-E3-R: N=5 overrides committed, M=2 reverted. Sidecar
    MUST carry exactly N-M=3 non-reverted entries."""
    block = {
        "c001": {"value_range_tmax": _record_dict(
            cell_id="c001", check_id="value_range_tmax",
        )},
        "c002": {"value_range_tmin": None},  # reverted
        "c003": {"value_range_precip": _record_dict(
            cell_id="c003", check_id="value_range_precip",
        )},
        "c004": {"value_range_srad": _record_dict(
            cell_id="c004", check_id="value_range_srad",
        )},
        "c005": {"value_range_soil_ph": None},  # reverted
    }
    output_path = tmp_path / "cockpit_overrides.json"
    write_cockpit_overrides_json(
        cockpit_overrides_at_launch=block,
        output_path=output_path,
    )
    sidecar = CockpitOverrideSidecar(**json.loads(output_path.read_text()))
    assert len(sidecar.overrides) == 3, (
        f"Partial-bulk-revert: expected 3 non-reverted entries; "
        f"got {len(sidecar.overrides)}"
    )
    cell_ids = {entry.cell_id for entry in sidecar.overrides}
    assert cell_ids == {"c001", "c003", "c004"}


# ── §3 Cat D rows do not contribute to non-empty count ────────────


def test_cat_d_documentary_rows_do_not_create_orphan_entries(
    tmp_path: Path,
) -> None:
    """Cat D rows are filtered from sidecar emission per codex CA-3.
    A run with only Cat D rows + zero value-replacement entries
    yields the all-reverted-equivalent sidecar shape (empty
    overrides[]) — same as if no overrides were committed at all.
    This pins the no-orphan invariant: documentary-only rows
    don't leak into the sidecar."""
    block = {
        "c001": {"crop_region_mismatch": _record_dict(
            cell_id="c001",
            check_id="crop_region_mismatch",
            documentary_basis="irrigation_infrastructure",
        )},
        "c002": {"crop_region_mismatch": _record_dict(
            cell_id="c002",
            check_id="crop_region_mismatch",
            documentary_basis="documented_microclimate",
        )},
    }
    output_path = tmp_path / "cockpit_overrides.json"
    write_cockpit_overrides_json(
        cockpit_overrides_at_launch=block,
        output_path=output_path,
    )
    sidecar = CockpitOverrideSidecar(**json.loads(output_path.read_text()))
    assert sidecar.overrides == [], (
        f"Cat D documentary rows leaked into sidecar: "
        f"{sidecar.overrides}. Per codex CA-3 + AC-E3-7 sub-2: "
        f"Cat D rows MUST stay audit-only — the documentary "
        f"basis IS the override."
    )


def test_mixed_value_and_cat_d_yields_only_value_entries(
    tmp_path: Path,
) -> None:
    """A mixed scenario — 2 value-replacement + 1 Cat D — yields
    exactly 2 sidecar entries (the Cat D row filtered out)."""
    block = {
        "c001": {"value_range_tmax": _record_dict(
            cell_id="c001", check_id="value_range_tmax",
        )},
        "c002": {"crop_region_mismatch": _record_dict(
            cell_id="c002",
            check_id="crop_region_mismatch",
            documentary_basis="irrigation_infrastructure",
        )},
        "c003": {"value_range_precip": _record_dict(
            cell_id="c003", check_id="value_range_precip",
        )},
    }
    output_path = tmp_path / "cockpit_overrides.json"
    write_cockpit_overrides_json(
        cockpit_overrides_at_launch=block,
        output_path=output_path,
    )
    sidecar = CockpitOverrideSidecar(**json.loads(output_path.read_text()))
    cell_ids = {entry.cell_id for entry in sidecar.overrides}
    assert cell_ids == {"c001", "c003"}
    assert "c002" not in cell_ids
