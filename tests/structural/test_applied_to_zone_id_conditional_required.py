"""Structural pin: ``applied_to_zone_id`` validator fires for both
error shapes per AC-E3-2 sub-criterion 5.

Sprint E.3 AC-E3-2 sub-5 + builder grounding-pass CA-7. The
``OverrideRecord`` cross-validator at ``prismpy/models/override.py``
pairs ``applied_to_scope`` with the optional companion
``applied_to_zone_id`` field — the actual Köppen zone identifier
lives in the companion when scope is "zone", and MUST be absent
otherwise. Two error shapes are pinned here:

§1 ``scope=zone`` with ``zone_id=None`` MUST reject — a "zone"
scope override that doesn't name a zone has nothing to apply.

§2 ``scope=single_cell`` with ``zone_id="Cwa"`` MUST reject — a
single-cell override carrying a zone id is structurally
inconsistent (which scope wins?). Same rejection for
``enumerated_cells``.

The behavioral pin lives at ``tests/unit/test_override_record.py``;
this structural pin is intentionally minimal — it asserts the
AC-E3-2 sub-5 contract by re-exercising the two error shapes. A
future refactor that drops one direction of the validator would
fire here loud.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prismpy.models.override import OverrideRecord
from uuid import uuid4


def _base_kwargs() -> dict:
    """Minimal valid OverrideRecord kwargs (Cat A/B/C value-
    replacement) — caller overrides ``applied_to_scope`` /
    ``applied_to_zone_id`` to exercise the two error shapes."""
    return dict(
        override_climate_values={"tmax_growing_season_mean": 32.5},
        override_soil_values=None,
        evidence_type="field_observation",
        evidence_type_other_specify=None,
        documentary_basis_other_specify=None,
        evidence_detail="Field-measured at the Bénoué station 2024-07-15.",
        applied_at_decision_id=uuid4(),
        applied_to_scope="single_cell",
        applied_to_zone_id=None,
        applied_to_snapshot=("c001",),
        check_id="value_range_tmax",
        category_d_documentary_basis=None,
    )


# ── §1 zone scope without zone_id ──────────────────────────────────


def test_zone_scope_with_null_zone_id_rejects() -> None:
    """AC-E3-2 sub-5 first error shape — pinned per the contract
    text: ``scope=zone, zone_id=None`` MUST reject."""
    kwargs = _base_kwargs()
    kwargs["applied_to_scope"] = "zone"
    kwargs["applied_to_zone_id"] = None
    with pytest.raises(ValidationError, match="applied_to_zone_id"):
        OverrideRecord(**kwargs)


def test_zone_scope_with_blank_zone_id_rejects() -> None:
    """Whitespace-only zone_id counts as empty per the strip()
    guard inside the validator."""
    kwargs = _base_kwargs()
    kwargs["applied_to_scope"] = "zone"
    kwargs["applied_to_zone_id"] = "   "
    with pytest.raises(ValidationError, match="applied_to_zone_id"):
        OverrideRecord(**kwargs)


# ── §2 non-zone scope with non-null zone_id ────────────────────────


def test_single_cell_scope_with_zone_id_rejects() -> None:
    """AC-E3-2 sub-5 second error shape — pinned per the contract
    text: ``scope=single_cell, zone_id="Cwa"`` MUST reject."""
    kwargs = _base_kwargs()
    kwargs["applied_to_scope"] = "single_cell"
    kwargs["applied_to_zone_id"] = "Cwa"
    with pytest.raises(ValidationError, match="applied_to_zone_id"):
        OverrideRecord(**kwargs)


def test_enumerated_cells_scope_with_zone_id_rejects() -> None:
    """The third scope discriminator follows the same invariant —
    enumerated cells don't share a zone."""
    kwargs = _base_kwargs()
    kwargs["applied_to_scope"] = "enumerated_cells"
    kwargs["applied_to_zone_id"] = "BSh"
    with pytest.raises(ValidationError, match="applied_to_zone_id"):
        OverrideRecord(**kwargs)


# ── §3 happy paths ─────────────────────────────────────────────────


def test_zone_scope_with_valid_zone_id_accepts() -> None:
    """Happy path: scope=zone + zone_id non-empty."""
    kwargs = _base_kwargs()
    kwargs["applied_to_scope"] = "zone"
    kwargs["applied_to_zone_id"] = "Cwa"
    record = OverrideRecord(**kwargs)
    assert record.applied_to_scope == "zone"
    assert record.applied_to_zone_id == "Cwa"


def test_single_cell_scope_with_null_zone_id_accepts() -> None:
    """Happy path: scope=single_cell + zone_id None."""
    record = OverrideRecord(**_base_kwargs())
    assert record.applied_to_scope == "single_cell"
    assert record.applied_to_zone_id is None
