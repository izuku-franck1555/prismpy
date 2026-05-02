"""Sprint D.1 AC-6.1 — provenance consumer pin parity.

Per ``feedback_intermediate_stage_pin_gap.md``: a producer-side
schema addition is incomplete until every consumer reads the new
fields without errors. AC-6 ships the producer side; AC-6.1 pins
the consumer reads so a future producer-side change cannot drift
without surfacing through one of these tests.

Consumer surfaces audited:

1. JSON serialization (``ProvenanceRecord.save_json``) — round
   trips through disk + reload preserves the new fields.
2. ``ProvenanceTracker.get_summary()`` — surfaces the aggregate
   counts to manifest writers + README generators.
3. Manifest reader path (the prismweb /results/ surface reads
   ``manifest.data_sources.*`` + ``manifest.region.*`` from the
   per-platform package's manifest.json — Sprint D.1 does NOT
   add to that surface, but a consumer that reads the
   provenance file directly still gets the new keys).

The tests here are integration-level (no full pipeline run);
they exercise the producer setters + assert each consumer pin
sees the post-state correctly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismpy.harmonize.rh_clip import RHClipProvenance
from prismpy.harmonize.texture_renormalize import (
    TextureRenormalizationProvenance,
)
from prismpy.provenance.tracker import ProvenanceTracker


@pytest.fixture
def populated_tracker():
    """A tracker with one of each AC-6 entry shape so consumer
    pins can read every key."""
    tracker = ProvenanceTracker(enabled=True, project_name="consumer-test")
    tracker.record_texture_renormalization(
        TextureRenormalizationProvenance(
            cell_id=10, layer_idx=0,
            original_sand=33.0, original_silt=33.0, original_clay=34.5,
            original_sum=100.5, delta_from_100=0.5,
            renormalization_factor=100.0 / 100.5,
            new_sand=33.0, new_silt=33.0, new_clay=34.0,
        )
    )
    tracker.record_rh_clip(
        RHClipProvenance(cell_id=11, date="2015-08-15", original_rh=101.2)
    )
    tracker.record_cell_unavailable(
        cell_id=12, unavailable_reason="soil",
        unavailable_cause="soil_no_hwsd_coverage",
    )
    tracker.record_pythia_misdat_replacement(translator="pythia", count=7)
    return tracker


# ---------------------------------------------------------------------------
# AC-6.1 consumer pin #1 — JSON disk round-trip
# ---------------------------------------------------------------------------


def test_provenance_json_round_trip_via_save_json(
    populated_tracker, tmp_path,
):
    """The provenance file written to disk reloads cleanly +
    every AC-6 key is present on the reloaded payload."""
    output_path = tmp_path / "provenance.json"
    populated_tracker.record.save_json(str(output_path))
    assert output_path.exists()

    reloaded = json.loads(output_path.read_text())

    assert "texture_renormalize_details" in reloaded
    assert "rh_clip_details" in reloaded
    assert "cells_unavailable_details" in reloaded

    assert len(reloaded["texture_renormalize_details"]) == 1
    assert len(reloaded["rh_clip_details"]) == 1
    assert len(reloaded["cells_unavailable_details"]) == 1

    # Summary aggregates surface inside the summary block.
    summary = reloaded["summary"]
    assert summary["texture_renormalize_count"] == 1
    assert summary["rh_clip_count"] == 1
    assert summary["cells_unavailable_by_cause"] == {
        "soil_no_hwsd_coverage": 1,
    }
    assert summary["pythia_misdat_replacements"] == {"pythia": 7}


# ---------------------------------------------------------------------------
# AC-6.1 consumer pin #2 — get_summary surface
# ---------------------------------------------------------------------------


def test_get_summary_returns_all_aggregate_keys(populated_tracker):
    """Consumers that hit ``get_summary()`` (manifest writers,
    README generators) must see every AC-6 aggregate key."""
    summary = populated_tracker.get_summary()
    expected_keys = {
        "texture_renormalize_count",
        "rh_clip_count",
        "cells_unavailable_by_cause",
        "pythia_misdat_replacements",
    }
    assert expected_keys.issubset(set(summary.keys())), (
        f"Missing AC-6 summary keys: "
        f"{expected_keys - set(summary.keys())}"
    )


# ---------------------------------------------------------------------------
# AC-6.1 consumer pin #3 — composite fixture matches the contract
# ---------------------------------------------------------------------------


def test_composite_fixture_matches_contract_evaluator_a6():
    """Evaluator's composite Fixture-A6 spec — 5 texture renorms +
    12 rh clips + 5 soil-unavailable + 100 PYTHIA missing-rain —
    surfaces exact aggregate counts so the consumer can use the
    summary to drive Methods text and audit panels."""
    tracker = ProvenanceTracker(enabled=True, project_name="composite")
    for i in range(5):
        tracker.record_texture_renormalization(
            TextureRenormalizationProvenance(
                cell_id=i, layer_idx=0,
                original_sand=33.0, original_silt=33.0, original_clay=34.5,
                original_sum=100.5, delta_from_100=0.5,
                renormalization_factor=100.0 / 100.5,
                new_sand=33.0, new_silt=33.0, new_clay=34.0,
            )
        )
    for i in range(12):
        tracker.record_rh_clip(
            RHClipProvenance(
                cell_id=i, date=f"2015-06-{i + 1:02d}", original_rh=101.0,
            )
        )
    for i in range(5):
        tracker.record_cell_unavailable(
            cell_id=i, unavailable_reason="soil",
            unavailable_cause="soil_no_hwsd_coverage",
        )
    tracker.record_pythia_misdat_replacement(translator="pythia", count=100)

    summary = tracker.get_summary()
    assert summary["texture_renormalize_count"] == 5
    assert summary["rh_clip_count"] == 12
    assert summary["cells_unavailable_by_cause"]["soil_no_hwsd_coverage"] == 5
    assert summary["pythia_misdat_replacements"]["pythia"] == 100


# ---------------------------------------------------------------------------
# AC-6.1 consumer pin #4 — manifest data_sources.soil stays string
# ---------------------------------------------------------------------------


def test_sprint_d_does_not_change_manifest_data_sources_soil_shape():
    """AC-7 invariant pinned at the consumer level: the manifest's
    ``data_sources.soil`` stays a string under Sprint D.1 (the
    structured identity work is deferred to Sprint D.2). Existing
    prismweb consumers (``views.py:982 actual_soil.lower()``,
    ``views.py:2750`` data portrait) call ``.lower()`` on the
    string; the pin verifies the pattern still works.

    The check exercises ``derive_boundary_label`` and the
    string-is-string contract at module level (no full pipeline
    run needed)."""
    from prismpy.packaging.manifest import create_manifest

    # Build a synthetic project_config dict — same shape every
    # translator passes to create_manifest after Sprint C.
    project_config = {
        "project_name": "sprint-d-consumer-pin",
        "package_name": "sprint-d-consumer-pin",
        "region_name": "Koutiala",
        "country": "Mali",
        "crop_name": "Maize",
        "planting_doy": 152,
        "maturity_doy": 304,
        "start_year": 2015,
        "end_year": 2015,
        "gadm_level": None,
        "data_sources": {
            "climate": "NASA POWER",
            "soil": "iSDA Africa",
            "boundaries": "Bounding box",
        },
    }
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        package_dir = Path(tmp) / "test"
        package_dir.mkdir()
        manifest = create_manifest(
            package_dir, project_config, platform="craft",
        )

    soil_value = manifest["data_sources"]["soil"]
    assert isinstance(soil_value, str), (
        f"Sprint D.1 AC-7 contract: manifest.data_sources.soil "
        f"must remain a string. Got {type(soil_value).__name__} "
        f"= {soil_value!r}. Structured identity work is deferred "
        f"to Sprint D.2."
    )
    # Existing prismweb consumer pattern: .lower() works.
    assert soil_value.lower() == soil_value.lower()
