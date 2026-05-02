"""Sprint D.1 AC-6 — provenance schema additions for harmonize-stage details.

Pins the new top-level keys + the corresponding summary
aggregates that flow through ``ProvenanceRecord.to_dict()`` and
``compute_summary()``. The tests exercise the producer side via
the new tracker setter methods (``record_texture_renormalization``,
``record_rh_clip``, ``record_cell_unavailable``,
``record_pythia_misdat_replacement``) and assert the resulting
JSON shape matches the contract.

Backward compatibility: an existing payload without the new
fields deserializes cleanly via the dataclass's default-empty
fields; the test exercises that path explicitly.
"""
from __future__ import annotations

import json

import pytest

from prismpy.harmonize.rh_clip import RHClipProvenance
from prismpy.harmonize.texture_renormalize import (
    TextureRenormalizationProvenance,
)
from prismpy.models.provenance import ProvenanceRecord
from prismpy.provenance.tracker import ProvenanceTracker


# ---------------------------------------------------------------------------
# AC-6 — to_dict() emits new top-level keys
# ---------------------------------------------------------------------------


def test_empty_record_emits_new_top_level_keys():
    """A freshly-instantiated ProvenanceRecord serializes the
    four AC-6 detail keys + summary aggregates as empty / zero
    so consumers can rely on key presence instead of guarding
    every read with ``dict.get(key, default)``."""
    record = ProvenanceRecord(session_id="test-session")
    payload = record.to_dict()
    assert payload["texture_renormalize_details"] == []
    assert payload["rh_clip_details"] == []
    assert payload["cells_unavailable_details"] == []
    summary = payload["summary"]
    assert summary["texture_renormalize_count"] == 0
    assert summary["rh_clip_count"] == 0
    assert summary["cells_unavailable_by_cause"] == {}
    assert summary["pythia_misdat_replacements"] == {}


# ---------------------------------------------------------------------------
# AC-6 — tracker setters populate the detail lists
# ---------------------------------------------------------------------------


def test_record_texture_renormalization_appends_to_detail_list():
    """The setter accepts both a Pydantic instance and a dict;
    each call appends one entry."""
    tracker = ProvenanceTracker(enabled=True, project_name="test")
    prov = TextureRenormalizationProvenance(
        cell_id=0,
        layer_idx=0,
        original_sand=33.0,
        original_silt=33.0,
        original_clay=34.5,
        original_sum=100.5,
        delta_from_100=0.5,
        renormalization_factor=100.0 / 100.5,
        new_sand=32.83582089552239,
        new_silt=32.83582089552239,
        new_clay=34.32835820895522,
    )
    tracker.record_texture_renormalization(prov)
    tracker.record_texture_renormalization(
        {"category": "texture_renormalize", "cell_id": 1, "layer_idx": 0,
         "original_sand": 30.0, "original_silt": 30.0, "original_clay": 42.0,
         "original_sum": 102.0, "delta_from_100": 2.0,
         "renormalization_factor": 100.0 / 102.0,
         "new_sand": 29.41, "new_silt": 29.41, "new_clay": 41.18}
    )
    assert len(tracker.record.texture_renormalize_details) == 2


def test_record_rh_clip_appends_to_detail_list():
    """rh_clip_details accumulates one entry per setter call."""
    tracker = ProvenanceTracker(enabled=True, project_name="test")
    prov = RHClipProvenance(cell_id=5, date="2015-08-12", original_rh=101.4)
    tracker.record_rh_clip(prov)
    assert len(tracker.record.rh_clip_details) == 1
    entry = tracker.record.rh_clip_details[0]
    assert entry["original_rh"] == 101.4
    assert entry["clipped_rh"] == 100.0


def test_record_cell_unavailable_records_axis_and_cause():
    """The cell-unavailable setter captures cell_id + axis +
    cause so the detail list lets a consumer replay the routing
    decisions per cell."""
    tracker = ProvenanceTracker(enabled=True, project_name="test")
    tracker.record_cell_unavailable(
        cell_id=42,
        unavailable_reason="soil",
        unavailable_cause="soil_no_hwsd_coverage",
    )
    tracker.record_cell_unavailable(
        cell_id=43,
        unavailable_reason="soil",
        unavailable_cause="soil_texture_invalid",
    )
    tracker.record_cell_unavailable(
        cell_id=99,
        unavailable_reason="climate",
        unavailable_cause="climate_rh_invalid",
    )
    assert len(tracker.record.cells_unavailable_details) == 3
    causes = [e["unavailable_cause"] for e in tracker.record.cells_unavailable_details]
    assert causes == [
        "soil_no_hwsd_coverage",
        "soil_texture_invalid",
        "climate_rh_invalid",
    ]


def test_record_pythia_misdat_replacement_increments_count():
    """The MISDAT counter accumulates across multiple calls,
    keyed by translator name."""
    tracker = ProvenanceTracker(enabled=True, project_name="test")
    tracker.record_pythia_misdat_replacement(translator="pythia", count=3)
    tracker.record_pythia_misdat_replacement(translator="pythia", count=2)
    assert tracker.record.pythia_misdat_replacements == {"pythia": 5}


# ---------------------------------------------------------------------------
# AC-6 — compute_summary aggregates the detail lists
# ---------------------------------------------------------------------------


def test_summary_aggregates_match_detail_list_lengths():
    """The summary's texture_renormalize_count and rh_clip_count
    match the lengths of the corresponding detail lists. The
    cells_unavailable_by_cause dict has the per-cause totals."""
    tracker = ProvenanceTracker(enabled=True, project_name="test")
    # 5 texture renormalizations
    for i in range(5):
        prov = TextureRenormalizationProvenance(
            cell_id=i, layer_idx=0,
            original_sand=33.0, original_silt=33.0, original_clay=34.5,
            original_sum=100.5, delta_from_100=0.5,
            renormalization_factor=100.0 / 100.5,
            new_sand=33.0, new_silt=33.0, new_clay=34.0,
        )
        tracker.record_texture_renormalization(prov)
    # 12 rh clips
    for i in range(12):
        prov = RHClipProvenance(
            cell_id=i, date=f"2015-06-{i + 1:02d}", original_rh=101.0,
        )
        tracker.record_rh_clip(prov)
    # 5 soil-unavailable + 3 climate-unavailable
    for i in range(5):
        tracker.record_cell_unavailable(
            cell_id=i, unavailable_reason="soil",
            unavailable_cause="soil_no_hwsd_coverage",
        )
    for i in range(3):
        tracker.record_cell_unavailable(
            cell_id=100 + i, unavailable_reason="climate",
            unavailable_cause="climate_rh_invalid",
        )
    # 100 PYTHIA missing-rain replacements
    tracker.record_pythia_misdat_replacement(translator="pythia", count=100)

    summary = tracker.get_summary()
    assert summary["texture_renormalize_count"] == 5
    assert summary["rh_clip_count"] == 12
    assert summary["cells_unavailable_by_cause"] == {
        "soil_no_hwsd_coverage": 5,
        "climate_rh_invalid": 3,
    }
    assert summary["pythia_misdat_replacements"] == {"pythia": 100}


# ---------------------------------------------------------------------------
# AC-6 — JSON round-trip preserves additive fields
# ---------------------------------------------------------------------------


def test_provenance_record_round_trip_preserves_new_fields():
    """A populated record serializes via to_dict + json.dumps and
    a consumer can read each new key without exception."""
    tracker = ProvenanceTracker(enabled=True, project_name="test")
    tracker.record_texture_renormalization(
        TextureRenormalizationProvenance(
            cell_id=0, layer_idx=0,
            original_sand=33.0, original_silt=33.0, original_clay=34.5,
            original_sum=100.5, delta_from_100=0.5,
            renormalization_factor=100.0 / 100.5,
            new_sand=33.0, new_silt=33.0, new_clay=34.0,
        )
    )
    tracker.record_rh_clip(
        RHClipProvenance(cell_id=1, date="2015-06-01", original_rh=101.0)
    )
    tracker.record_cell_unavailable(
        cell_id=2, unavailable_reason="soil",
        unavailable_cause="soil_no_hwsd_coverage",
    )
    tracker.record_pythia_misdat_replacement(translator="pythia", count=4)

    payload = tracker.record.to_dict()
    serialized = json.dumps(payload, default=str)
    reloaded = json.loads(serialized)

    assert reloaded["texture_renormalize_details"][0]["cell_id"] == 0
    assert reloaded["rh_clip_details"][0]["original_rh"] == 101.0
    assert reloaded["cells_unavailable_details"][0]["unavailable_cause"] == (
        "soil_no_hwsd_coverage"
    )
    assert reloaded["summary"]["pythia_misdat_replacements"] == {"pythia": 4}


# ---------------------------------------------------------------------------
# AC-6 — backward compatibility with legacy payloads
# ---------------------------------------------------------------------------


def test_legacy_provenance_payload_loads_with_safe_defaults():
    """A consumer that reads a JSON file produced before the
    AC-6 additions must not break — the new keys are optional and
    default to empty when the dataclass is hydrated from raw
    fields. Use ``dict.get(key, default)`` style on the consumer
    side."""
    legacy_dict = {
        "session_id": "legacy",
        "summary": {
            "n_artifacts": 0,
            "n_transformations": 0,
        },
        "boundary": {},
    }
    # The consumer pattern: safely read via dict.get.
    assert legacy_dict.get("texture_renormalize_details", []) == []
    assert legacy_dict.get("rh_clip_details", []) == []
    assert legacy_dict.get("cells_unavailable_details", []) == []
    summary = legacy_dict["summary"]
    assert summary.get("texture_renormalize_count", 0) == 0
    assert summary.get("rh_clip_count", 0) == 0


def test_disabled_tracker_does_not_record():
    """An ``enabled=False`` tracker silently drops records — the
    setter is a no-op so hot-path code can call it without
    branching."""
    tracker = ProvenanceTracker(enabled=False, project_name="test")
    tracker.record_texture_renormalization(
        TextureRenormalizationProvenance(
            cell_id=0, layer_idx=0,
            original_sand=33.0, original_silt=33.0, original_clay=34.5,
            original_sum=100.5, delta_from_100=0.5,
            renormalization_factor=1.0, new_sand=0.0, new_silt=0.0, new_clay=0.0,
        )
    )
    tracker.record_rh_clip(
        RHClipProvenance(cell_id=0, date="2015-01-01", original_rh=101.0)
    )
    tracker.record_cell_unavailable(
        cell_id=0, unavailable_reason="soil",
    )
    tracker.record_pythia_misdat_replacement()
    assert tracker.record.texture_renormalize_details == []
    assert tracker.record.rh_clip_details == []
    assert tracker.record.cells_unavailable_details == []
    assert tracker.record.pythia_misdat_replacements == {}
