"""Unit pin for ``_apply_soil_overrides_to_assignment`` helper.

Sprint E.3 fixup +15 (F-BN Boundary 3). The helper synthesizes
per-cell soil profiles for cells with cockpit-recorded overrides
so the canonical .SOL writer downstream emits the persona's
documented values without silently mutating profiles shared with
non-overridden cells.

Covers:
* No-op on empty sidecar (returns identical references)
* No-op on sidecar entries that don't match a known soil
  variable_key (climate-only overrides skip this helper)
* Single-cell sand override → new profile id, top-layer sand
  updated, silt re-computed, hydraulic properties cleared
* Multi-key per-cell override (sand + clay + ph) → all applied
  to top layer atomically
* Multi-cell parallel overrides → each gets a distinct new
  profile id
* Original profiles unchanged (purity invariant — durable §24
  canonical-source-or-pin)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict
from uuid import UUID

from prismpy.cockpit.cockpit_overrides_writer import (
    CockpitOverrideSidecar,
    OverrideSidecarEntry,
)
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.translators._shared.eghr_substrate import (
    _apply_soil_overrides_to_assignment,
)


_FIXED_DECISION_ID = UUID("00000000-0000-0000-0000-000000000001")
_FIXED_PRODUCED_AT = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)


def _make_profile(sand: float, clay: float) -> SoilProfile:
    return SoilProfile(
        profile_id=f"P_{int(sand)}_{int(clay)}",
        lat=12.0,
        lon=2.0,
        source="hwsd2",
        layers=[
            SoilLayer(
                depth_top=0.0,
                depth_bottom=0.3,
                sand=sand,
                clay=clay,
                silt=100.0 - sand - clay,
                organic_carbon=0.5,
                bulk_density=1.4,
                ph=6.5,
            ),
        ],
    )


def _make_sidecar(entries) -> CockpitOverrideSidecar:
    return CockpitOverrideSidecar(
        schema_version="1.0",
        produced_at=_FIXED_PRODUCED_AT,
        overrides=list(entries),
    )


def _sand_override_entry(cell_id: str, value: float) -> OverrideSidecarEntry:
    return OverrideSidecarEntry(
        cell_id=cell_id,
        check_id="value_range_soil_sand",
        variable_key="soil_sand_pct",
        value=value,
        unit="%",
        decision_id=_FIXED_DECISION_ID,
        evidence_type="field_observation",
    )


def test_helper_is_noop_when_sidecar_has_no_soil_entries() -> None:
    """A sidecar that carries only climate entries (no soil
    variable_keys) should return the inputs unchanged. The
    ``apply_override`` climate path lives in the WTH writer; the
    eGHR substrate's per-cell synthesis is a soil-only concern."""
    cell_to_profile_id = {0: 1, 1: 1, 2: 2}
    profiles = {1: _make_profile(60.0, 18.0), 2: _make_profile(40.0, 30.0)}
    sidecar = _make_sidecar([
        OverrideSidecarEntry(
            cell_id="0",
            check_id="value_range_tmax",
            variable_key="tmax_growing_season_mean",
            value=999.0,
            unit="C",
            decision_id=_FIXED_DECISION_ID,
            evidence_type="field_observation",
        ),
    ])

    new_cell_to_profile_id, new_profiles = _apply_soil_overrides_to_assignment(
        cell_to_profile_id=cell_to_profile_id,
        profiles_by_id=profiles,
        sidecar=sidecar,
    )

    assert new_cell_to_profile_id == cell_to_profile_id
    assert new_profiles == profiles


def test_single_cell_sand_override_synthesizes_new_profile() -> None:
    """An override on cell 0's sand creates a new profile id,
    re-points cell 0's assignment, and leaves other cells'
    references unchanged."""
    cell_to_profile_id = {0: 1, 1: 1, 2: 2}
    profiles: Dict[int, SoilProfile] = {
        1: _make_profile(60.0, 18.0),
        2: _make_profile(40.0, 30.0),
    }
    sidecar = _make_sidecar([_sand_override_entry("0", 88.0)])

    new_cell_to_profile_id, new_profiles = _apply_soil_overrides_to_assignment(
        cell_to_profile_id=cell_to_profile_id,
        profiles_by_id=profiles,
        sidecar=sidecar,
    )

    # Cell 0 re-pointed; cells 1, 2 unchanged.
    assert new_cell_to_profile_id[0] != 1
    assert new_cell_to_profile_id[1] == 1
    assert new_cell_to_profile_id[2] == 2
    assert new_cell_to_profile_id[0] == max(new_profiles)

    # New profile carries override; silt recomputed.
    new_profile = new_profiles[new_cell_to_profile_id[0]]
    assert new_profile.layers[0].sand == 88.0
    # silt = 100 - sand - clay = 100 - 88 - 18 = -6 → clamped to 0.0.
    assert new_profile.layers[0].silt == 0.0

    # Hydraulic properties cleared so downstream recomputes.
    assert new_profile.layers[0].wilting_point is None
    assert new_profile.layers[0].field_capacity is None
    assert new_profile.layers[0].saturated_wc is None


def test_original_profile_unchanged_after_synthesis() -> None:
    """Purity invariant — the helper must NOT mutate the base
    profile in-place. The original cell 0 sand value stays
    intact even though cell 0 now points at a synthesized
    profile carrying the override."""
    cell_to_profile_id = {0: 1, 1: 1}
    profiles: Dict[int, SoilProfile] = {1: _make_profile(60.0, 18.0)}
    sidecar = _make_sidecar([_sand_override_entry("0", 88.0)])

    _apply_soil_overrides_to_assignment(
        cell_to_profile_id=cell_to_profile_id,
        profiles_by_id=profiles,
        sidecar=sidecar,
    )

    # Base profile retains its original sand value — cell 1 (which
    # still points at it) sees the original soil.
    assert profiles[1].layers[0].sand == 60.0


def test_multi_key_override_applies_all_on_top_layer() -> None:
    """A persona overriding sand + clay + ph on the same cell sees
    all three on the synthesized profile's top layer; silt
    recomputes from the new sand + clay."""
    cell_to_profile_id = {0: 1}
    profiles: Dict[int, SoilProfile] = {1: _make_profile(60.0, 18.0)}
    sidecar = _make_sidecar([
        _sand_override_entry("0", 50.0),
        OverrideSidecarEntry(
            cell_id="0",
            check_id="value_range_soil_clay",
            variable_key="soil_clay_pct",
            value=25.0,
            unit="%",
            decision_id=_FIXED_DECISION_ID,
            evidence_type="field_observation",
        ),
        OverrideSidecarEntry(
            cell_id="0",
            check_id="value_range_soil_ph",
            variable_key="soil_ph",
            value=7.2,
            unit="",
            decision_id=_FIXED_DECISION_ID,
            evidence_type="field_observation",
        ),
    ])

    new_cell_to_profile_id, new_profiles = _apply_soil_overrides_to_assignment(
        cell_to_profile_id=cell_to_profile_id,
        profiles_by_id=profiles,
        sidecar=sidecar,
    )

    new_profile_id = new_cell_to_profile_id[0]
    new_profile = new_profiles[new_profile_id]
    assert new_profile.layers[0].sand == 50.0
    assert new_profile.layers[0].clay == 25.0
    assert new_profile.layers[0].ph == 7.2
    assert new_profile.layers[0].silt == 25.0  # 100 - 50 - 25


def test_multi_cell_parallel_overrides_get_distinct_ids() -> None:
    """Two cells with independent overrides each get their own
    new profile id — no collision."""
    cell_to_profile_id = {0: 1, 1: 1, 2: 2}
    profiles: Dict[int, SoilProfile] = {
        1: _make_profile(60.0, 18.0),
        2: _make_profile(40.0, 30.0),
    }
    sidecar = _make_sidecar([
        _sand_override_entry("0", 88.0),
        _sand_override_entry("2", 30.0),
    ])

    new_cell_to_profile_id, new_profiles = _apply_soil_overrides_to_assignment(
        cell_to_profile_id=cell_to_profile_id,
        profiles_by_id=profiles,
        sidecar=sidecar,
    )

    new_id_cell_0 = new_cell_to_profile_id[0]
    new_id_cell_2 = new_cell_to_profile_id[2]
    # New ids distinct from each other AND from the original ids.
    assert new_id_cell_0 != new_id_cell_2
    assert new_id_cell_0 not in {1, 2}
    assert new_id_cell_2 not in {1, 2}
    # Each cell's new profile carries its own override.
    assert new_profiles[new_id_cell_0].layers[0].sand == 88.0
    assert new_profiles[new_id_cell_2].layers[0].sand == 30.0


def test_override_on_unknown_cell_id_skips() -> None:
    """A sidecar entry whose cell_id isn't in the grid roster
    (e.g., persona overrode a cell that's been since excluded by
    a skip-from-analysis decision) does NOT crash + does NOT
    affect any existing assignment."""
    cell_to_profile_id = {0: 1, 1: 1}
    profiles: Dict[int, SoilProfile] = {1: _make_profile(60.0, 18.0)}
    sidecar = _make_sidecar([_sand_override_entry("99", 88.0)])

    new_cell_to_profile_id, new_profiles = _apply_soil_overrides_to_assignment(
        cell_to_profile_id=cell_to_profile_id,
        profiles_by_id=profiles,
        sidecar=sidecar,
    )

    assert new_cell_to_profile_id == cell_to_profile_id
    assert new_profiles == profiles
