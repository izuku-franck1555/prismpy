"""Structural pin: methods-text generator + cockpit snapshot helper +
manifest consistency validator.

Sprint E.2 AC-E2-7 + AC-E2-8 + AC-E2-22 + §0.2 #4 (cockpit_snapshot
prismpy half).

Asserts:
* methods-text generator emits opening cell-count + Shepard 1968
  citation + IDW parameters + INTERPOLATION-PRESENT flag note.
* Phrase 1 (degraded k=2/3) and Phrase 2 (k=1 zero-width) appear
  verbatim when applicable.
* Per-zone caveat phrases are emitted once per unique code.
* Empty record list returns empty string (no orphan paragraph).
* serialize_decisions_to_config produces JSON-stable output with
  sorted keys.
* validate_manifest_cell_summary_consistency raises on drift in
  either direction; passes on consistent state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from prismpy.config.schema import Platform
from prismpy.models.decision_log import CellDecisionRecord
from prismpy.models.interpolated_cell import InterpolatedCellRecord
from prismpy.packaging.cockpit_snapshot import serialize_decisions_to_config
from prismpy.packaging.methods_text import generate_interpolation_methods_paragraph
from prismpy.validators.manifest_consistency import (
    ManifestConsistencyError,
    validate_manifest_cell_summary_consistency,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _interp_record(
    *, source_cells: list[str], caveats: list[str] | None = None
) -> InterpolatedCellRecord:
    decision_id = uuid4()
    return InterpolatedCellRecord(
        interpolation_method="idw_k4_r15km_w_inverse_dist_sq",
        source_cells=source_cells,
        uncertainty_ci_lower=410.0,
        uncertainty_ci_upper=414.0,
        method_doi="10.1145/800186.810616",
        applied_at_decision_id=decision_id,
        affected_zone_code="BSh",
        caveat_codes=caveats or [],
    )


# ── §1 methods-text generator ───────────────────────────────────────


def test_empty_record_list_returns_empty_string() -> None:
    assert generate_interpolation_methods_paragraph([], Platform.PYTHIA) == ""


def test_full_path_paragraph_contains_idw_anchor() -> None:
    records = [_interp_record(source_cells=["c1", "c2", "c3", "c4"])]
    text = generate_interpolation_methods_paragraph(records, Platform.PYTHIA)
    assert "Shepard 1968" in text
    assert "inverse-distance-weighted" in text
    assert "4 nearest neighbors" in text or "up to 4 nearest" in text
    assert "15 km" in text
    assert "INTERPOLATION-PRESENT" in text


def test_phrase_1_appears_for_degraded_k2_or_k3() -> None:
    """Drill F + AC-E2-7 sub-criterion: when ANY record has
    1 < len(source_cells) < 4, Phrase 1 verbatim appears."""
    records = [_interp_record(source_cells=["c1", "c2"])]
    text = generate_interpolation_methods_paragraph(records, Platform.PYTHIA)
    assert (
        "imputed with fewer than k=4 neighbors; uncertainty bounds for "
        "these cells are conservative under the normality assumption"
    ) in text


def test_phrase_2_appears_for_k1_single_neighbour() -> None:
    """Drill F2 + AC-E2-7 sub-criterion: when ANY record has
    len(source_cells) == 1, Phrase 2 verbatim appears."""
    records = [_interp_record(source_cells=["c_only"])]
    text = generate_interpolation_methods_paragraph(records, Platform.PYTHIA)
    assert (
        "imputed from a single neighbor within R=15km; uncertainty bounds "
        "for these cells are uninformative (zero-width by construction)"
    ) in text


def test_per_zone_caveat_phrase_appears_once_per_unique_code() -> None:
    """Two records with the SAME caveat code → caveat phrase appears
    once (deduped)."""
    records = [
        _interp_record(
            source_cells=["c1", "c2", "c3", "c4"],
            caveats=["sahel-precip-convective"],
        ),
        _interp_record(
            source_cells=["c5", "c6", "c7", "c8"],
            caveats=["sahel-precip-convective"],
        ),
    ]
    text = generate_interpolation_methods_paragraph(records, Platform.PYTHIA)
    # The phrase should appear exactly once.
    sahel_phrase = "Sahel-zone precipitation interpolation"
    assert text.count(sahel_phrase) == 1


def test_multi_platform_paragraph_names_platform() -> None:
    records = [_interp_record(source_cells=["c1", "c2", "c3", "c4"])]
    pythia_text = generate_interpolation_methods_paragraph(records, Platform.PYTHIA)
    sarra_text = generate_interpolation_methods_paragraph(records, Platform.SARRA_PY)
    craft_text = generate_interpolation_methods_paragraph(records, Platform.CRAFT)
    assert "PYTHIA" in pythia_text
    assert "SARRA-Py" in sarra_text
    assert "CRAFT" in craft_text


# ── §2 cockpit_snapshot serialize_decisions_to_config ────────────────


def test_empty_dict_serializes_to_empty_snapshot() -> None:
    """Sprint E.3 AC-E3-14 Extension 1: serializer emits BOTH the
    decisions block + the overrides block. Empty input → both
    blocks empty."""
    snapshot = serialize_decisions_to_config({})
    assert snapshot == {
        "cockpit_decisions_at_launch": {},
        "cockpit_overrides_at_launch": {},
    }


def test_single_decision_serializes_with_cell_id_key() -> None:
    """Sprint E.3 AC-E3-14 reshape: nested
    ``cockpit_decisions_at_launch[<cell_id>][<check_id>] = record``
    JSON shape preserves multi-check coexistence per cell."""
    decision_id = uuid4()
    record = CellDecisionRecord(
        decision_id=decision_id,
        cell_id="c1",
        action="acknowledge",
        check_id="value_range_precip",
        timestamp=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
        user_identity="u",
        method_or_rationale="r",
        sequence_number=1,
    )
    snapshot = serialize_decisions_to_config(
        {("c1", "value_range_precip"): record}
    )
    assert "cockpit_decisions_at_launch" in snapshot
    assert "c1" in snapshot["cockpit_decisions_at_launch"]
    assert "value_range_precip" in snapshot["cockpit_decisions_at_launch"]["c1"]
    assert (
        snapshot["cockpit_decisions_at_launch"]["c1"]["value_range_precip"]["cell_id"]
        == "c1"
    )


def test_serialization_is_json_dumpable() -> None:
    """The output must be JSON-serializable so PipelineRun.config_snapshot
    can persist it via Django JSONField."""
    decision_id = uuid4()
    record = CellDecisionRecord(
        decision_id=decision_id,
        cell_id="c1",
        action="acknowledge",
        check_id="value_range_precip",
        timestamp=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
        user_identity="u",
        method_or_rationale="r",
        sequence_number=1,
    )
    snapshot = serialize_decisions_to_config(
        {("c1", "value_range_precip"): record}
    )
    # Should not raise.
    json.dumps(snapshot)


def test_keys_sorted_for_deterministic_output() -> None:
    """Sorted outer + inner keys ensure byte-stable output across runs."""
    records = {
        (f"c{i}", "value_range_precip"): CellDecisionRecord(
            decision_id=uuid4(),
            cell_id=f"c{i}",
            action="acknowledge",
            check_id="value_range_precip",
            timestamp=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
            user_identity="u",
            method_or_rationale="r",
            sequence_number=i,
        )
        for i in range(5, 0, -1)
    }
    snapshot = serialize_decisions_to_config(records)
    outer_keys = list(snapshot["cockpit_decisions_at_launch"].keys())
    assert outer_keys == sorted(outer_keys)


def test_multi_check_per_cell_serializes_under_nested_keys() -> None:
    """Sprint E.3 AC-E3-14 NEW: a single cell with two active
    decisions on different check_ids serializes to two separate
    inner-dict entries under the same outer cell_id key."""
    record_tmax = CellDecisionRecord(
        decision_id=uuid4(),
        cell_id="c1",
        action="acknowledge",
        check_id="value_range_tmax",
        timestamp=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
        user_identity="u",
        method_or_rationale="r",
        sequence_number=1,
    )
    record_tmin = CellDecisionRecord(
        decision_id=uuid4(),
        cell_id="c1",
        action="acknowledge",
        check_id="value_range_tmin",
        timestamp=datetime(2026, 5, 7, 12, 0, 1, tzinfo=timezone.utc),
        user_identity="u",
        method_or_rationale="r",
        sequence_number=2,
    )
    snapshot = serialize_decisions_to_config(
        {
            ("c1", "value_range_tmax"): record_tmax,
            ("c1", "value_range_tmin"): record_tmin,
        }
    )
    inner = snapshot["cockpit_decisions_at_launch"]["c1"]
    assert "value_range_tmax" in inner
    assert "value_range_tmin" in inner
    # Inner keys also sorted alphabetically.
    assert list(inner.keys()) == sorted(inner.keys())


def test_none_value_passes_through_unchanged() -> None:
    """When current_decisions returns None for a pair (all decisions
    reverted), the snapshot preserves the None marker under the
    nested key."""
    snapshot = serialize_decisions_to_config(
        {("c1", "value_range_precip"): None}
    )
    assert (
        snapshot["cockpit_decisions_at_launch"]["c1"]["value_range_precip"]
        is None
    )


# ── §3 manifest consistency validator ───────────────────────────────


def test_validator_passes_on_consistent_state() -> None:
    """Both flag=True + at-least-one cell with decision_id."""
    manifest = {"flags": {"interpolation_present": True}}
    cell_summary = {
        "cells": [{"cell_id": "c1", "interpolation_decision_id": "uuid-here"}]
    }
    # Should not raise.
    validate_manifest_cell_summary_consistency(manifest, cell_summary)


def test_validator_passes_on_clean_package() -> None:
    """Flag=False + no decision IDs is also consistent."""
    manifest = {"flags": {"interpolation_present": False}}
    cell_summary = {"cells": [{"cell_id": "c1"}]}
    validate_manifest_cell_summary_consistency(manifest, cell_summary)


def test_validator_raises_when_flag_false_but_cells_imputed() -> None:
    """Drill H (a) — the under-claim case: cells have interpolations
    but the manifest doesn't say so. Honest-signal violation."""
    manifest = {"flags": {"interpolation_present": False}}
    cell_summary = {
        "cells": [{"cell_id": "c1", "interpolation_decision_id": "uuid-here"}]
    }
    with pytest.raises(ManifestConsistencyError, match="MUST be True"):
        validate_manifest_cell_summary_consistency(manifest, cell_summary)


def test_validator_raises_when_flag_true_but_no_cells_imputed() -> None:
    """Drill H (b) — the over-claim case: manifest claims
    interpolation but no cell carries a decision_id (e.g., post-revert
    state)."""
    manifest = {"flags": {"interpolation_present": True}}
    cell_summary = {"cells": [{"cell_id": "c1"}]}
    with pytest.raises(ManifestConsistencyError, match="over-claimed"):
        validate_manifest_cell_summary_consistency(manifest, cell_summary)


def test_validator_handles_missing_flags_dict() -> None:
    """A manifest without a flags dict is treated as flag=False
    (defensive default)."""
    manifest: dict = {}
    cell_summary = {"cells": []}
    # Should not raise (both effectively absent).
    validate_manifest_cell_summary_consistency(manifest, cell_summary)


# ── §4 dunder-all surfaces ───────────────────────────────────────────


def test_methods_text_module_exports() -> None:
    from prismpy.packaging import methods_text
    assert methods_text.__all__ == ["generate_interpolation_methods_paragraph"]


def test_cockpit_snapshot_module_exports() -> None:
    from prismpy.packaging import cockpit_snapshot
    # Sprint E.3 AC-E3-14 sub-6 absorbed — dual-shape loader
    # ``deserialize_decisions_from_config`` joins the canonical
    # exports alongside the writer.
    assert sorted(cockpit_snapshot.__all__) == [
        "deserialize_decisions_from_config",
        "serialize_decisions_to_config",
    ]


def test_manifest_consistency_module_exports() -> None:
    from prismpy.validators import manifest_consistency
    assert sorted(manifest_consistency.__all__) == [
        "ManifestConsistencyError",
        "validate_manifest_cell_summary_consistency",
    ]
