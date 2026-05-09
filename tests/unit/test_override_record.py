"""Behavioral tests for ``OverrideRecord`` Pydantic schema.

Sprint E.3 AC-E3-4 sub-1 + Drill-E3-AC + Drill-E3-G. Covers all 5
model_validator branches + extra-forbid drill + Tuple-immutability
drill + frozen-immutability drill + Cat D documentary-basis enum-
coverage drill.

The schema lives at ``prismpy/models/override.py``; this file
exercises the validators behaviorally. Structural pins
(canonical-source AST walkers for the EvidenceType + AppliedToScope
+ CategoryDDocumentaryBasis Literals) live at
``tests/structural/`` per Stage 1 §9 vocabulary parity discipline.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from prismpy.models.override import (
    CategoryDDocumentaryBasis,
    OverrideRecord,
)


# ── Fixture builders ───────────────────────────────────────────────


def _decision_id():
    return uuid4()


def _cat_ab_kwargs(**overrides):
    """Default kwargs for a Cat A/B/C value-replacement override."""
    base = dict(
        override_climate_values={"tmax_growing_season_mean": 32.5},
        override_soil_values=None,
        evidence_type="field_observation",
        evidence_type_other_specify=None,
        documentary_basis_other_specify=None,
        evidence_detail="Field-measured at the Bénoué station 2024-07-15.",
        applied_at_decision_id=_decision_id(),
        applied_to_scope="single_cell",
        applied_to_zone_id=None,
        applied_to_snapshot=("c001",),
        check_id="value_range_tmax",
        category_d_documentary_basis=None,
    )
    base.update(overrides)
    return base


def _cat_d_kwargs(**overrides):
    """Default kwargs for a Cat D documentary-basis override."""
    base = dict(
        override_climate_values=None,
        override_soil_values=None,
        evidence_type="irrigation",
        evidence_type_other_specify=None,
        documentary_basis_other_specify=None,
        evidence_detail="Sub-watershed irrigated by the Lagdo dam since 1982.",
        applied_at_decision_id=_decision_id(),
        applied_to_scope="zone",
        applied_to_zone_id="BSh",
        applied_to_snapshot=("c001", "c002", "c003"),
        check_id="crop_region_mismatch",
        category_d_documentary_basis="irrigation_infrastructure",
    )
    base.update(overrides)
    return base


# ── §1 minimal valid records construct ─────────────────────────────


def test_cat_ab_value_replacement_record_constructs() -> None:
    """A Cat A/B/C value-replacement override with non-null climate
    values + no documentary basis → constructs cleanly."""
    record = OverrideRecord(**_cat_ab_kwargs())
    assert record.override_climate_values == {"tmax_growing_season_mean": 32.5}
    assert record.category_d_documentary_basis is None
    assert record.applied_to_snapshot == ("c001",)


def test_cat_d_documentary_record_constructs() -> None:
    """A Cat D documentary override with null override values + non-
    null category_d_documentary_basis → constructs cleanly."""
    record = OverrideRecord(**_cat_d_kwargs())
    assert record.override_climate_values is None
    assert record.override_soil_values is None
    assert record.category_d_documentary_basis == "irrigation_infrastructure"
    assert record.applied_to_zone_id == "BSh"


# ── §2 validator 1 — evidence_type_other_specify conditional ───────


def test_evidence_type_other_requires_specify_text() -> None:
    """``evidence_type == "other"`` MUST be paired with non-empty
    ``evidence_type_other_specify``."""
    with pytest.raises(ValidationError, match="evidence_type_other_specify"):
        OverrideRecord(
            **_cat_ab_kwargs(
                evidence_type="other",
                evidence_type_other_specify=None,
            )
        )


def test_evidence_type_other_rejects_blank_specify() -> None:
    """Whitespace-only specify counts as empty per the strip()
    guard."""
    with pytest.raises(ValidationError, match="evidence_type_other_specify"):
        OverrideRecord(
            **_cat_ab_kwargs(
                evidence_type="other",
                evidence_type_other_specify="   ",
            )
        )


def test_evidence_type_named_rejects_specify_text() -> None:
    """When ``evidence_type`` is one of the 5 named buckets,
    ``evidence_type_other_specify`` MUST be None — typo'd UI
    can't smuggle a free-form payload alongside a categorical pick."""
    with pytest.raises(ValidationError, match="evidence_type_other_specify"):
        OverrideRecord(
            **_cat_ab_kwargs(
                evidence_type="citation",
                evidence_type_other_specify="not allowed here",
            )
        )


def test_evidence_type_other_with_specify_text_accepts() -> None:
    """The happy path for the 'other' evidence_type."""
    record = OverrideRecord(
        **_cat_ab_kwargs(
            evidence_type="other",
            evidence_type_other_specify="Cultural-knowledge basis from local agronomist",
        )
    )
    assert record.evidence_type == "other"
    assert (
        record.evidence_type_other_specify
        == "Cultural-knowledge basis from local agronomist"
    )


# ── §3 validator 2+3 — value-replacement vs documentary discriminator ──


def test_cat_ab_record_without_values_rejects() -> None:
    """A non-Cat-D row (``category_d_documentary_basis is None``)
    MUST carry non-null ``override_climate_values`` OR
    ``override_soil_values``."""
    with pytest.raises(ValidationError, match="value-replacement"):
        OverrideRecord(
            **_cat_ab_kwargs(
                override_climate_values=None,
                override_soil_values=None,
            )
        )


def test_cat_d_record_with_climate_values_rejects() -> None:
    """A Cat D documentary row MUST have null override values; the
    documentary basis IS the override."""
    with pytest.raises(ValidationError, match="Cat D documentary row"):
        OverrideRecord(
            **_cat_d_kwargs(
                override_climate_values={"tmax_growing_season_mean": 30.0},
            )
        )


def test_cat_d_record_with_soil_values_rejects() -> None:
    """Same constraint on the soil side."""
    with pytest.raises(ValidationError, match="Cat D documentary row"):
        OverrideRecord(
            **_cat_d_kwargs(
                override_soil_values={"soil_ph": 6.5},
            )
        )


def test_cat_ab_with_soil_values_only_accepts() -> None:
    """A Cat A/B/C row with ONLY soil values (no climate) is valid."""
    record = OverrideRecord(
        **_cat_ab_kwargs(
            override_climate_values=None,
            override_soil_values={"soil_ph": 6.5},
            check_id="value_range_soil_ph",
        )
    )
    assert record.override_soil_values == {"soil_ph": 6.5}
    assert record.override_climate_values is None


# ── §4 validator 4 — documentary_basis_other_specify conditional ───


def test_documentary_basis_other_requires_specify_text() -> None:
    """``category_d_documentary_basis == "other"`` MUST be paired
    with non-empty ``documentary_basis_other_specify``."""
    with pytest.raises(ValidationError, match="documentary_basis_other_specify"):
        OverrideRecord(
            **_cat_d_kwargs(
                category_d_documentary_basis="other",
                documentary_basis_other_specify=None,
            )
        )


def test_documentary_basis_named_rejects_specify_text() -> None:
    """When ``category_d_documentary_basis`` is a named basis,
    ``documentary_basis_other_specify`` MUST be None."""
    with pytest.raises(ValidationError, match="documentary_basis_other_specify"):
        OverrideRecord(
            **_cat_d_kwargs(
                category_d_documentary_basis="irrigation_infrastructure",
                documentary_basis_other_specify="should not be here",
            )
        )


def test_documentary_basis_other_with_text_accepts() -> None:
    """Happy path for the 'other' Cat D branch."""
    record = OverrideRecord(
        **_cat_d_kwargs(
            category_d_documentary_basis="other",
            documentary_basis_other_specify="Tile drainage system installed 2019",
        )
    )
    assert record.category_d_documentary_basis == "other"
    assert (
        record.documentary_basis_other_specify
        == "Tile drainage system installed 2019"
    )


def test_evidence_type_other_independent_of_documentary_basis_other() -> None:
    """Codex MEDIUM-4 absorption — a Cat D row whose evidence_type
    is 'other' AND whose documentary_basis is also 'other' carries
    TWO separate free-form payloads. Conflating them would lose
    half the persona's audit trail."""
    record = OverrideRecord(
        **_cat_d_kwargs(
            evidence_type="other",
            evidence_type_other_specify="Local agronomist field visit notes",
            category_d_documentary_basis="other",
            documentary_basis_other_specify="Sub-surface drainage anomaly",
        )
    )
    assert record.evidence_type_other_specify == "Local agronomist field visit notes"
    assert (
        record.documentary_basis_other_specify == "Sub-surface drainage anomaly"
    )


# ── §5 validator 5 — applied_to_zone_id conditional ────────────────


def test_zone_scope_requires_zone_id() -> None:
    """``applied_to_scope == "zone"`` MUST be paired with non-empty
    ``applied_to_zone_id``."""
    with pytest.raises(ValidationError, match="applied_to_zone_id"):
        OverrideRecord(
            **_cat_ab_kwargs(
                applied_to_scope="zone",
                applied_to_zone_id=None,
            )
        )


def test_single_cell_scope_rejects_zone_id() -> None:
    """``applied_to_scope == "single_cell"`` MUST have null
    ``applied_to_zone_id``."""
    with pytest.raises(ValidationError, match="applied_to_zone_id"):
        OverrideRecord(
            **_cat_ab_kwargs(
                applied_to_scope="single_cell",
                applied_to_zone_id="Cwa",
            )
        )


def test_enumerated_cells_scope_rejects_zone_id() -> None:
    """``applied_to_scope == "enumerated_cells"`` MUST have null
    ``applied_to_zone_id``."""
    with pytest.raises(ValidationError, match="applied_to_zone_id"):
        OverrideRecord(
            **_cat_ab_kwargs(
                applied_to_scope="enumerated_cells",
                applied_to_zone_id="BSh",
            )
        )


# ── §6 extra-forbid drill ──────────────────────────────────────────


def test_extra_field_rejects() -> None:
    """``extra="forbid"`` rejects typo'd field names at construction."""
    kwargs = _cat_ab_kwargs()
    kwargs["typo_field_name"] = "should fail"
    with pytest.raises(ValidationError, match="typo_field_name"):
        OverrideRecord(**kwargs)


# ── §7 immutability drills (frozen=True + Tuple) ───────────────────


def test_frozen_assignment_rejects() -> None:
    """``frozen=True`` rejects top-level field reassignment."""
    record = OverrideRecord(**_cat_ab_kwargs())
    with pytest.raises(ValidationError):
        record.evidence_detail = "some new text"


def test_tuple_snapshot_immutability_drill() -> None:
    """Drill-E3-AC: ``applied_to_snapshot`` is a Tuple, not a List —
    attempts to ``append`` raise ``AttributeError`` (Tuple has no
    append method) per WA CA-19 absorbed.

    A future refactor that drifts to ``List[CellID]`` would preserve
    Pydantic frozen=True (top-level) but lose the structural
    immutability of the embedded sequence; this drill catches it."""
    record = OverrideRecord(**_cat_ab_kwargs())
    assert isinstance(record.applied_to_snapshot, tuple)
    with pytest.raises(AttributeError):
        record.applied_to_snapshot.append("c999")  # type: ignore[attr-defined]


def test_evidence_detail_min_length_floor() -> None:
    """The 20-char minimum on evidence_detail rejects placeholder
    strings without requiring full-paragraph essays for legitimate
    one-line citations."""
    with pytest.raises(ValidationError, match="evidence_detail"):
        OverrideRecord(
            **_cat_ab_kwargs(
                evidence_detail="too short",
            )
        )


# ── §8 Cat D documentary-basis enum-coverage drill ─────────────────


def test_all_four_documentary_bases_accept() -> None:
    """Drill-E3-G partial: each of the 4 named Cat D bases
    constructs cleanly when paired with appropriate companion
    fields."""
    bases: list[str] = list(typing.get_args(CategoryDDocumentaryBasis))
    assert set(bases) == {
        "irrigation_infrastructure",
        "documented_microclimate",
        "shallow_rooted_crop_variety",
        "other",
    }
    # Construct each named base. 'other' needs the companion specify
    # field; the three named bases reject it.
    for basis in bases:
        if basis == "other":
            record = OverrideRecord(
                **_cat_d_kwargs(
                    category_d_documentary_basis=basis,
                    documentary_basis_other_specify="A custom rationale",
                )
            )
        else:
            record = OverrideRecord(
                **_cat_d_kwargs(
                    category_d_documentary_basis=basis,
                )
            )
        assert record.category_d_documentary_basis == basis


def test_invalid_documentary_basis_rejects() -> None:
    """A non-canonical documentary basis fails at the Literal type
    boundary."""
    with pytest.raises(ValidationError):
        OverrideRecord(
            **_cat_d_kwargs(
                category_d_documentary_basis="not_a_real_basis",  # type: ignore[arg-type]
            )
        )


import typing  # imported here to keep §8 self-contained
