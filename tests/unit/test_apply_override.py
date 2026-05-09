"""Behavioral tests for the ``apply_override`` 3-source precedence helper.

Sprint E.3 AC-E3-8 sub-criteria + Drill-E3-K (multi-check
coexistence: Override on tmax + Interpolate on coverage_climate
imputing tmax → translator output uses Override).

The helper's purity invariant (sidecar / interpolation_record /
unified_data byte-equivalent before+after) is exercised via
direct Pydantic frozen=True checks here; the cross-translator
36-cell drill at AC-E3-9 covers the per-platform invariant.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from prismpy.cockpit.cockpit_overrides_writer import (
    CockpitOverrideSidecar,
    OverrideSidecarEntry,
)
from prismpy.models.interpolated_cell import InterpolatedCellRecord
from prismpy.translators._shared.cockpit_overrides import apply_override


# ── Fixture builders ───────────────────────────────────────────────


def _sidecar(*entries: OverrideSidecarEntry) -> CockpitOverrideSidecar:
    return CockpitOverrideSidecar(
        schema_version="1.0",
        produced_at=datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc),
        overrides=list(entries),
    )


def _entry(
    *,
    cell_id: str = "c001",
    variable_key: str = "tmax_growing_season_mean",
    value: float = 32.5,
    unit: str = "C",
    decision_id: UUID = None,
    evidence_type: str = "field_observation",
) -> OverrideSidecarEntry:
    return OverrideSidecarEntry(
        cell_id=cell_id,
        check_id="value_range_tmax",
        variable_key=variable_key,
        value=value,
        unit=unit,
        decision_id=decision_id or uuid4(),
        evidence_type=evidence_type,
    )


def _interpolation_record(*, decision_id: UUID = None) -> InterpolatedCellRecord:
    return InterpolatedCellRecord(
        interpolation_method="idw_k4_r15km_w_inverse_dist_sq",
        source_cells=["c002", "c003"],
        uncertainty_ci_lower=30.0,
        uncertainty_ci_upper=34.0,
        method_doi="10.1145/800186.810616",
        applied_at_decision_id=decision_id or uuid4(),
        affected_zone_code="BSh",
        caveat_codes=[],
    )


# ── §1 override match wins ─────────────────────────────────────────


def test_override_match_returns_override_value() -> None:
    """A matching (cell_id, variable_key) entry in sidecar →
    helper returns the override value, not raw_value."""
    sidecar = _sidecar(_entry(cell_id="c001", value=32.5))
    raw = 26.0  # raw value the translator would otherwise write
    result = apply_override(
        cell_id="c001",
        variable_key="tmax_growing_season_mean",
        raw_value=raw,
        sidecar=sidecar,
    )
    assert result == 32.5


def test_override_wins_over_interpolation_record() -> None:
    """Drill-E3-K: when sidecar has Override AND
    interpolation_record is non-None for the same cell,
    Override wins per the precedence chain."""
    sidecar = _sidecar(_entry(cell_id="c001", value=32.5))
    raw = 28.0  # already IDW-imputed upstream
    interp = _interpolation_record()
    result = apply_override(
        cell_id="c001",
        variable_key="tmax_growing_season_mean",
        raw_value=raw,
        sidecar=sidecar,
        interpolation_record=interp,
    )
    assert result == 32.5, (
        "Override MUST win over Interpolation per AC-E3-8 + "
        "Drill-E3-K: the user's documented decision dominates "
        "mechanical imputation"
    )


# ── §2 no override match → raw_value ───────────────────────────────


def test_no_override_match_returns_raw_value() -> None:
    """When the sidecar has no matching entry for (cell_id,
    variable_key), the helper returns raw_value unchanged."""
    sidecar = _sidecar(_entry(cell_id="c099", value=99.9))
    raw = 26.0
    result = apply_override(
        cell_id="c001",
        variable_key="tmax_growing_season_mean",
        raw_value=raw,
        sidecar=sidecar,
    )
    assert result == raw


def test_wrong_variable_key_returns_raw_value() -> None:
    """Same cell_id but different variable_key → raw_value."""
    sidecar = _sidecar(
        _entry(cell_id="c001", variable_key="tmax_growing_season_mean"),
    )
    raw = 4.5
    result = apply_override(
        cell_id="c001",
        variable_key="tmin_growing_season_mean",
        raw_value=raw,
        sidecar=sidecar,
    )
    assert result == raw


# ── §3 sidecar None → raw_value ────────────────────────────────────


def test_none_sidecar_returns_raw_value() -> None:
    """A run with no overrides → sidecar is None → helper short-
    circuits to raw_value."""
    raw = 26.0
    result = apply_override(
        cell_id="c001",
        variable_key="tmax_growing_season_mean",
        raw_value=raw,
        sidecar=None,
    )
    assert result == raw


def test_empty_sidecar_returns_raw_value() -> None:
    """An empty sidecar (all-reverted-bulk path) → raw_value."""
    sidecar = _sidecar()  # empty overrides list
    raw = 26.0
    result = apply_override(
        cell_id="c001",
        variable_key="tmax_growing_season_mean",
        raw_value=raw,
        sidecar=sidecar,
    )
    assert result == raw


# ── §4 interpolation_record alone (no override) → raw_value ────────


def test_interpolation_record_alone_returns_raw_value() -> None:
    """Cell IS imputed but no override → raw_value (which is
    already the IDW-imputed value upstream)."""
    sidecar = _sidecar()
    raw = 28.0  # IDW-imputed
    interp = _interpolation_record()
    result = apply_override(
        cell_id="c001",
        variable_key="tmax_growing_season_mean",
        raw_value=raw,
        sidecar=sidecar,
        interpolation_record=interp,
    )
    assert result == raw


# ── §5 purity drill — sidecar / interpolation_record unmodified ────


def test_helper_does_not_mutate_sidecar() -> None:
    """Pin: the helper is PURE per AC-E3-8 contract. Pydantic
    frozen=True enforces immutability, but we assert it
    behaviorally too."""
    entry = _entry(cell_id="c001", value=32.5)
    sidecar = _sidecar(entry)
    sidecar_dump_before = sidecar.model_dump(mode="json")

    apply_override(
        cell_id="c001",
        variable_key="tmax_growing_season_mean",
        raw_value=26.0,
        sidecar=sidecar,
    )
    sidecar_dump_after = sidecar.model_dump(mode="json")
    assert sidecar_dump_before == sidecar_dump_after


def test_helper_does_not_mutate_interpolation_record() -> None:
    """Same purity invariant for interpolation_record."""
    sidecar = _sidecar(_entry(cell_id="c001", value=32.5))
    interp = _interpolation_record()
    interp_dump_before = interp.model_dump(mode="json")

    apply_override(
        cell_id="c001",
        variable_key="tmax_growing_season_mean",
        raw_value=26.0,
        sidecar=sidecar,
        interpolation_record=interp,
    )
    interp_dump_after = interp.model_dump(mode="json")
    assert interp_dump_before == interp_dump_after


# ── §6 multiple entries per cell ───────────────────────────────────


def test_multiple_entries_dispatch_per_variable_key() -> None:
    """A cell with override entries on tmax AND tmin: helper
    returns the matching variable_key's value for each."""
    sidecar = _sidecar(
        _entry(cell_id="c001", variable_key="tmax_growing_season_mean", value=32.5),
        _entry(cell_id="c001", variable_key="tmin_growing_season_mean", value=18.0),
    )
    tmax_result = apply_override(
        cell_id="c001",
        variable_key="tmax_growing_season_mean",
        raw_value=26.0,
        sidecar=sidecar,
    )
    tmin_result = apply_override(
        cell_id="c001",
        variable_key="tmin_growing_season_mean",
        raw_value=15.0,
        sidecar=sidecar,
    )
    assert tmax_result == 32.5
    assert tmin_result == 18.0
