"""Sprint D.1 — apply_harmonize_transformations integration helper.

Pins the harmonize-stage orchestrator that the pipeline executor
calls between cascade resolution and unified-data assembly. The
helper iterates soil profiles + climate time series, invokes the
per-event helpers (texture_action_for / rh_action_for), and
writes provenance via the tracker setters. Cells routed to
``unavailable`` accumulate on the returned stats so the
executor surfaces them through the cell-summary build path.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pytest

from prismpy.harmonize.apply import (
    HarmonizeStats,
    apply_harmonize_transformations,
)
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.provenance.tracker import ProvenanceTracker


def _layer(sand: float, clay: float, silt: float) -> SoilLayer:
    return SoilLayer(
        depth_top=0.0,
        depth_bottom=0.2,
        sand=sand,
        clay=clay,
        silt=silt,
        organic_carbon=0.5,
        bulk_density=1.4,
        ph=6.5,
    )


def _profile(layers: list, cell_id: int = 0) -> SoilProfile:
    return SoilProfile(
        profile_id=f"TEST_{cell_id:06d}",
        lat=12.4,
        lon=-5.4,
        source="TEST",
        layers=layers,
        total_depth=0.2,
        metadata={"source": "TEST", "cascade_rank": 1},
    )


def _record(d: date, rh: Optional[float]) -> ClimateRecord:
    return ClimateRecord(
        date=d, tmax=30.0, tmin=20.0, precip=2.0, srad=20.0, wind=2.0,
        rh=rh, tdew=18.0,
    )


def _ts(records: list, cell_id: int = 0) -> ClimateTimeSeries:
    return ClimateTimeSeries(
        location_id=cell_id, lat=12.4, lon=-5.4, source="TEST",
        records=records, elevation=300.0,
    )


# ---------------------------------------------------------------------------
# Empty-input smoke
# ---------------------------------------------------------------------------


def test_empty_inputs_returns_zero_stats():
    """The helper handles None / empty inputs without raising."""
    stats = apply_harmonize_transformations(
        soil_data=None, climate_data=None, provenance_tracker=None,
    )
    assert stats == HarmonizeStats()


# ---------------------------------------------------------------------------
# AC-1 — texture renormalization in soil_data
# ---------------------------------------------------------------------------


def test_renormalizes_layer_in_place_and_records_provenance():
    """A layer with delta=0.5% renormalizes; the profile's layer
    list is replaced with the renormalized copy + provenance is
    written."""
    layer = _layer(sand=33.0, silt=33.0, clay=34.5)  # sum=100.5
    profile = _profile([layer])
    soil_data = {0: profile}
    tracker = ProvenanceTracker(enabled=True, project_name="t")

    stats = apply_harmonize_transformations(
        soil_data=soil_data, climate_data=None,
        provenance_tracker=tracker,
    )

    assert stats.n_texture_renormalized == 1
    assert stats.n_texture_warned == 0
    assert stats.cells_unavailable == []
    new_layer = soil_data[0].layers[0]
    assert abs(new_layer.sand + new_layer.silt + new_layer.clay - 100.0) < 1e-9
    assert len(tracker.record.texture_renormalize_details) == 1


def test_warn_retain_layer_unchanged_no_provenance():
    """A layer with delta=4% sits in the warn band — keeps values
    unchanged, no provenance written, no cell-unavailable."""
    layer = _layer(sand=33.0, silt=33.0, clay=38.0)  # sum=104
    profile = _profile([layer])
    soil_data = {0: profile}
    tracker = ProvenanceTracker(enabled=True, project_name="t")

    stats = apply_harmonize_transformations(
        soil_data=soil_data, climate_data=None,
        provenance_tracker=tracker,
    )

    assert stats.n_texture_renormalized == 0
    assert stats.n_texture_warned == 1
    assert stats.cells_unavailable == []
    assert len(tracker.record.texture_renormalize_details) == 0


def test_exclude_layer_routes_cell_to_unavailable():
    """A layer with delta=10% exceeds the warn band — the cell
    routes to ``unavailable`` with cause ``soil_texture_invalid``."""
    layer = _layer(sand=40.0, silt=30.0, clay=40.0)  # sum=110
    profile = _profile([layer], cell_id=42)
    soil_data = {42: profile}
    tracker = ProvenanceTracker(enabled=True, project_name="t")

    stats = apply_harmonize_transformations(
        soil_data=soil_data, climate_data=None,
        provenance_tracker=tracker,
    )

    assert len(stats.cells_unavailable) == 1
    entry = stats.cells_unavailable[0]
    assert entry["cell_id"] == 42
    assert entry["unavailable_reason"] == "soil"
    assert entry["unavailable_cause"] == "soil_texture_invalid"
    assert len(tracker.record.cells_unavailable_details) == 1


# ---------------------------------------------------------------------------
# AC-3 — rh clipping in climate_data
# ---------------------------------------------------------------------------


def test_clips_rh_in_climate_record_and_records_provenance():
    """A record with rh=101 clips to 100 + records provenance."""
    rec = _record(date(2015, 6, 15), rh=101.0)
    ts = _ts([rec])
    climate_data = {0: ts}
    tracker = ProvenanceTracker(enabled=True, project_name="t")

    stats = apply_harmonize_transformations(
        soil_data=None, climate_data=climate_data,
        provenance_tracker=tracker,
    )

    assert stats.n_rh_clipped == 1
    assert stats.cells_unavailable == []
    assert climate_data[0].records[0].rh == 100.0
    assert len(tracker.record.rh_clip_details) == 1


def test_excludes_cell_when_rh_above_clip_threshold():
    """A record with rh=110 exceeds the clip window — the cell
    routes to ``unavailable`` with axis=climate +
    cause=climate_rh_invalid."""
    rec = _record(date(2015, 6, 15), rh=110.0)
    ts = _ts([rec], cell_id=99)
    climate_data = {99: ts}
    tracker = ProvenanceTracker(enabled=True, project_name="t")

    stats = apply_harmonize_transformations(
        soil_data=None, climate_data=climate_data,
        provenance_tracker=tracker,
    )

    assert len(stats.cells_unavailable) == 1
    entry = stats.cells_unavailable[0]
    assert entry["cell_id"] == 99
    assert entry["unavailable_reason"] == "climate"
    assert entry["unavailable_cause"] == "climate_rh_invalid"


def test_passes_rh_below_100_unchanged():
    """rh values <= 100 don't trigger any action."""
    records = [_record(date(2015, 6, i + 1), rh=85.0) for i in range(5)]
    ts = _ts(records)
    climate_data = {0: ts}
    tracker = ProvenanceTracker(enabled=True, project_name="t")

    stats = apply_harmonize_transformations(
        soil_data=None, climate_data=climate_data,
        provenance_tracker=tracker,
    )

    assert stats.n_rh_clipped == 0
    assert stats.cells_unavailable == []


# ---------------------------------------------------------------------------
# AC-4 — HWSD unavailable_cells propagated as cell-unavailable
# ---------------------------------------------------------------------------


def test_hwsd_unavailable_cells_surfaced_as_cell_unavailable():
    """The loader-level unavailable list (HWSD coverage misses)
    propagates through to the cell-unavailable stats."""
    hwsd_list = [
        {"cell_id": 1, "cause": "soil_no_hwsd_coverage"},
        {"cell_id": 2, "cause": "soil_no_hwsd_coverage"},
    ]
    tracker = ProvenanceTracker(enabled=True, project_name="t")

    stats = apply_harmonize_transformations(
        soil_data=None, climate_data=None,
        provenance_tracker=tracker,
        hwsd_unavailable_cells=hwsd_list,
    )

    assert len(stats.cells_unavailable) == 2
    causes = [e["unavailable_cause"] for e in stats.cells_unavailable]
    assert all(c == "soil_no_hwsd_coverage" for c in causes)
    assert all(
        e["unavailable_reason"] == "soil"
        for e in stats.cells_unavailable
    )


def test_hwsd_unavailable_default_cause_when_missing():
    """An entry without a cause string defaults to
    ``soil_no_hwsd_coverage``."""
    hwsd_list = [{"cell_id": 5}]
    stats = apply_harmonize_transformations(
        soil_data=None, climate_data=None,
        hwsd_unavailable_cells=hwsd_list,
    )
    assert len(stats.cells_unavailable) == 1
    assert (
        stats.cells_unavailable[0]["unavailable_cause"]
        == "soil_no_hwsd_coverage"
    )


# ---------------------------------------------------------------------------
# Composite — texture + rh + HWSD all in one run
# ---------------------------------------------------------------------------


def test_composite_run_aggregates_all_actions():
    """A composite fixture with one of each action surfaces all
    counts + cell-unavailable entries correctly."""
    soil_data = {
        0: _profile([_layer(33.0, 34.5, 33.0)]),  # sum=100.5 → renorm
        1: _profile([_layer(33.0, 38.0, 33.0)]),  # sum=104 → warn
        2: _profile([_layer(40.0, 40.0, 30.0)], cell_id=2),  # sum=110 → exclude
    }
    climate_data = {
        10: _ts([_record(date(2015, 6, 15), rh=101.0)], cell_id=10),  # clip
        11: _ts([_record(date(2015, 6, 15), rh=110.0)], cell_id=11),  # excl
    }
    hwsd_list = [{"cell_id": 100, "cause": "soil_no_hwsd_coverage"}]
    tracker = ProvenanceTracker(enabled=True, project_name="t")

    stats = apply_harmonize_transformations(
        soil_data=soil_data, climate_data=climate_data,
        provenance_tracker=tracker,
        hwsd_unavailable_cells=hwsd_list,
    )

    assert stats.n_texture_renormalized == 1
    assert stats.n_texture_warned == 1
    assert stats.n_rh_clipped == 1
    # 1 hwsd-coverage + 1 texture-invalid + 1 rh-invalid = 3
    assert len(stats.cells_unavailable) == 3
    causes = sorted(
        e["unavailable_cause"] for e in stats.cells_unavailable
    )
    assert causes == [
        "climate_rh_invalid",
        "soil_no_hwsd_coverage",
        "soil_texture_invalid",
    ]


def test_disabled_provenance_silent_pass_through():
    """When provenance_tracker is None the helper still applies
    the transformations; only the provenance writes are skipped."""
    layer = _layer(sand=33.0, silt=33.0, clay=34.5)
    profile = _profile([layer])
    soil_data = {0: profile}

    stats = apply_harmonize_transformations(
        soil_data=soil_data, climate_data=None,
        provenance_tracker=None,
    )

    assert stats.n_texture_renormalized == 1
    new_layer = soil_data[0].layers[0]
    assert abs(new_layer.sand + new_layer.silt + new_layer.clay - 100.0) < 1e-9
