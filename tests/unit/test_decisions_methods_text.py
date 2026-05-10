"""Unit pin for the cockpit decisions methods-text renderer.

Sprint E.3 fixup +15 (F-BO Boundary 4). The renderer is the
unified consumer point for documentary-text emission — every
Cat D Override + Acknowledge decision in
``cockpit_decisions_at_launch`` surfaces in a single text block
the prismweb post-execute hook writes into every platform's
package directory.

Covers:
* Empty snapshot → 4 section headers + "(no entries)" markers
* Cat A/B/C value-replacement Override emits value pairs +
  "(Value applied to canonical per-cell files...)" suffix per
  the AC #1 user-snippet bar (durable §25)
* Cat D documentary Override emits documentary_basis +
  evidence_type + evidence_detail
* Acknowledge emits method_or_rationale + "(Cell INCLUDED...)"
* Skip emits rationale + "(Cell EXCLUDED...)"
* Section ordering is stable (Value-Replacement → Documented →
  Acknowledged → Skipped)
"""

from __future__ import annotations

from prismpy.cockpit.decisions_methods_text import (
    render_cockpit_decisions_text,
)


def _value_replacement_record(cell_id: str, check_id: str) -> dict:
    return {
        "decision_id": "11111111-1111-1111-1111-111111111111",
        "cell_id": cell_id,
        "action": "document_override",
        "check_id": check_id,
        "timestamp": "2026-05-10T12:00:00+00:00",
        "user_identity": "test_user",
        "method_or_rationale": "Persona note: high tmax based on field data",
        "sequence_number": 1,
        "override_record": {
            "category_d_documentary_basis": None,
            "override_climate_values": {"tmax_growing_season_mean": 38.5},
            "override_soil_values": {},
            "evidence_type": "field_observation",
            "evidence_type_other_specify": None,
            "evidence_detail": (
                "Mathon et al. 2002 documented a Sahel MCS hot-day extreme "
                "consistent with the imputed value"
            ),
        },
    }


def _cat_d_record(cell_id: str, check_id: str, evidence_detail: str) -> dict:
    return {
        "decision_id": "22222222-2222-2222-2222-222222222222",
        "cell_id": cell_id,
        "action": "document_override",
        "check_id": check_id,
        "timestamp": "2026-05-10T13:00:00+00:00",
        "user_identity": "test_user",
        "method_or_rationale": "Cat D documentary basis",
        "sequence_number": 2,
        "override_record": {
            "category_d_documentary_basis": "agricultural_extension_report",
            "category_d_documentary_basis_other_specify": None,
            "override_climate_values": None,
            "override_soil_values": None,
            "evidence_type": "extension_report",
            "evidence_type_other_specify": None,
            "evidence_detail": evidence_detail,
        },
    }


def _acknowledge_record(cell_id: str, check_id: str, note: str) -> dict:
    return {
        "decision_id": "33333333-3333-3333-3333-333333333333",
        "cell_id": cell_id,
        "action": "acknowledge",
        "check_id": check_id,
        "timestamp": "2026-05-10T14:00:00+00:00",
        "user_identity": "test_user",
        "method_or_rationale": note,
        "sequence_number": 3,
    }


def _skip_record(cell_id: str, check_id: str, rationale: str) -> dict:
    return {
        "decision_id": "44444444-4444-4444-4444-444444444444",
        "cell_id": cell_id,
        "action": "skip_from_analysis",
        "check_id": check_id,
        "timestamp": "2026-05-10T15:00:00+00:00",
        "user_identity": "test_user",
        "method_or_rationale": rationale,
        "sequence_number": 4,
    }


def test_empty_snapshot_renders_all_four_sections() -> None:
    text = render_cockpit_decisions_text(
        cockpit_decisions_at_launch={},
        derived_run_id="run-uuid-123",
        derived_run_number=2,
    )
    assert "Derived run #2 (run-uuid-123)" in text
    assert "## Value-Replacement Overrides (Cat A / B / C)" in text
    assert "## Documented Overrides (Cat D — documentary basis)" in text
    assert "## Acknowledged Warnings (Bucket 2 informational)" in text
    assert "## Skipped Cells (excluded from analysis)" in text
    assert text.count("(no entries)") == 4


def test_value_replacement_override_renders_value_pairs_and_apply_suffix() -> None:
    snapshot = {
        "4374122": {
            "value_range_tmax": _value_replacement_record(
                "4374122", "value_range_tmax",
            ),
        },
    }
    text = render_cockpit_decisions_text(
        cockpit_decisions_at_launch=snapshot,
    )
    assert "Cell 4374122 (value_range_tmax)" in text
    assert "tmax_growing_season_mean=38.5" in text
    assert "Value applied to canonical per-cell files" in text
    assert "apply_override" in text
    assert "evidence_type = field_observation" in text


def test_cat_d_documentary_override_surfaces_evidence_detail() -> None:
    """AC #3 acceptance — evidence_detail from a Cat D Override
    must appear in the rendered methods text so a journal
    reviewer can audit the persona's documentary basis."""
    snapshot = {
        "4365486": {
            "transitional_zone_acknowledge": _cat_d_record(
                "4365486",
                "transitional_zone_acknowledge",
                "TEST_DOC_REF_for_AC3_acceptance",
            ),
        },
    }
    text = render_cockpit_decisions_text(
        cockpit_decisions_at_launch=snapshot,
    )
    assert "Cell 4365486 (transitional_zone_acknowledge)" in text
    assert "documentary basis = agricultural_extension_report" in text
    assert "TEST_DOC_REF_for_AC3_acceptance" in text


def test_acknowledge_decision_surfaces_method_or_rationale() -> None:
    """AC #7 acceptance — Acknowledge cell methods.txt contains
    the persona's evidence text + decision_id and indicates the
    cell IS included (acknowledge is non-blocking)."""
    snapshot = {
        "4369806": {
            "climate_envelope_tail_acknowledge": _acknowledge_record(
                "4369806",
                "climate_envelope_tail_acknowledge",
                "TEST_ACK_NOTE: Documented Sahel transitional climate",
            ),
        },
    }
    text = render_cockpit_decisions_text(
        cockpit_decisions_at_launch=snapshot,
    )
    assert "Cell 4369806 (climate_envelope_tail_acknowledge)" in text
    assert "TEST_ACK_NOTE: Documented Sahel transitional climate" in text
    assert "Decision: 33333333-3333-3333-3333-333333333333" in text
    assert "Cell INCLUDED in canonical files" in text


def test_skip_decision_renders_with_excluded_marker() -> None:
    """Skipped cells render under their dedicated section + flag
    the cell as EXCLUDED so the persona / reviewer can immediately
    tell skip ≠ acknowledge."""
    snapshot = {
        "4400000": {
            "outside_crop_envelope": _skip_record(
                "4400000",
                "outside_crop_envelope",
                "Beyond canonical sorghum range",
            ),
        },
    }
    text = render_cockpit_decisions_text(
        cockpit_decisions_at_launch=snapshot,
    )
    assert "Cell 4400000 (outside_crop_envelope)" in text
    assert "Beyond canonical sorghum range" in text
    assert "Cell EXCLUDED from canonical files" in text


def test_mixed_decision_types_partition_into_correct_sections() -> None:
    """Each decision type lands under its expected section — no
    cross-contamination of the partition."""
    snapshot = {
        "100": {
            "value_range_tmax": _value_replacement_record("100", "value_range_tmax"),
        },
        "200": {
            "transitional_zone_acknowledge": _cat_d_record(
                "200", "transitional_zone_acknowledge", "cat-d evidence text",
            ),
        },
        "300": {
            "climate_envelope_tail_acknowledge": _acknowledge_record(
                "300", "climate_envelope_tail_acknowledge", "ack note for cell 300",
            ),
        },
        "400": {
            "outside_crop_envelope": _skip_record(
                "400", "outside_crop_envelope", "skip rationale for cell 400",
            ),
        },
    }
    text = render_cockpit_decisions_text(
        cockpit_decisions_at_launch=snapshot,
    )
    # All four sections populated.
    assert text.count("(no entries)") == 0
    assert "Cell 100" in text
    assert "Cell 200" in text
    assert "Cell 300" in text
    assert "Cell 400" in text
    # Section header ordering: Value → Documented → Ack → Skip.
    pos_val = text.index("## Value-Replacement Overrides")
    pos_doc = text.index("## Documented Overrides")
    pos_ack = text.index("## Acknowledged Warnings")
    pos_skip = text.index("## Skipped Cells")
    assert pos_val < pos_doc < pos_ack < pos_skip


def test_none_snapshot_renders_empty_stub_without_crashing() -> None:
    """Defensive — a derived run with no decisions at all should
    still emit a "no decisions" stub so the persona sees the
    file came from a real but intervention-free run."""
    text = render_cockpit_decisions_text(
        cockpit_decisions_at_launch=None,
    )
    assert "# Cockpit Decisions — User Audit Trail" in text
    assert text.count("(no entries)") == 4
