"""``OverrideRecord`` — canonical schema for a per-cell cockpit Override.

Sprint E.3 AC-E3-4 + Stage 1 §2 (Q1 OverrideRecord shape) + Stage 0 §5
(Override-with-values mental model). Parallels
``InterpolatedCellRecord`` at ``prismpy/models/interpolated_cell.py``.
The cockpit's Override decision-workflow semantics are split across
two halves per the I-DN-1 boundary:

* This module owns the **Pydantic schema** for an Override decision
  payload — PURE, no Django, no I/O. Consumed by:
  - The translator-side ``apply_override`` helper at
    ``prismpy/translators/_shared/cockpit_overrides.py`` (AC-E3-8 +
    sidecar reader).
  - The cockpit Override Edit form (Phase 2; prismweb).
  - The audit-log surfacer + methods-text generator.

* The prismweb side at ``core/models.py::PipelineRunDecision`` adds
  an ``override_record: JSONField`` per AC-E3-16.5 prismweb
  persistence bridge; the service-layer ``_row_to_record``
  re-hydrates the JSON back into this Pydantic model on read.

**Mental model — Override-with-values vs documentary-basis** (Stage 0
§5 + §6 Override scope LOCK):

* **Cat A/B/C (value-replacement)** — the persona documents a real
  value the validator flagged as out-of-typical-range. The override
  carries a non-null ``override_climate_values`` OR
  ``override_soil_values`` payload that the translator reads at
  per-cell write time via ``apply_override``. Examples: a documented
  Sahel hot-day extreme tmax of 46 °C; a calibrated soil
  bulk-density of 1.7 g/cm³ from a recent profile pit.

* **Cat D (documentary-basis)** — the persona documents a structural
  reason the cell's flagged warning doesn't actually apply, WITHOUT
  carrying a value-replacement. ``category_d_documentary_basis``
  records the structural reason (irrigation_infrastructure,
  documented_microclimate, shallow_rooted_crop_variety, other);
  ``override_*_values`` MUST be null for Cat D. Cat D rows are
  audit / methods-only — the translator-side sidecar writer at
  AC-E3-7 explicitly filters Cat D out of sidecar emission per
  codex CA-3 absorbed.

**Five validators close the schema-layer invariants** (per AC-E3-4 +
Sprint E.3 contract sub-criteria):

1. ``evidence_type_other_specify`` non-empty iff ``evidence_type ==
   "other"``; mirrors ``WizardOverrideRecord`` pattern at
   ``prismpy/provenance/wizard_decisions.py:176``.
2. Cat A/B/C (``category_d_documentary_basis is None``) requires
   non-null ``override_climate_values`` OR ``override_soil_values``.
3. Cat D (``category_d_documentary_basis is not None``) requires
   null ``override_climate_values`` AND null
   ``override_soil_values``; the documentary basis IS the override.
4. ``documentary_basis_other_specify`` non-empty iff
   ``category_d_documentary_basis == "other"`` (NEW post-Draft 3
   codex MEDIUM-4 — separate companion field; does NOT conflate with
   ``evidence_type_other_specify``).
5. ``applied_to_zone_id`` non-empty iff ``applied_to_scope == "zone"``
   (per AC-E3-2 sub-2 + builder grounding-pass CA-7).

Self-link integrity (``applied_at_decision_id == owning
CellDecisionRecord.decision_id``) is enforced at the **enclosing
``CellDecisionRecord``** layer per ``prismpy/models/interpolated_cell
.py:21-24`` precedent (codex LOW-2 absorbed: embedded records cannot
see their owner; the assertion lives where it can read both sides).

**Tuple immutability** (WA CA-19 absorbed): ``applied_to_snapshot``
is a ``Tuple[CellID, ...]``, NOT ``List[CellID]``. Pydantic
``frozen=True`` only freezes top-level field reassignment; a mutable
``List`` reference can still ``append`` after construction.
``Tuple`` is structurally immutable, mirroring the precedent at
``prismpy/cockpit/manifest.py:153``
``CockpitManifestEntry.affected_cells: Tuple[str, ...]``.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prismpy.models.interpolated_cell import CellID
from prismpy.provenance.wizard_decisions import EvidenceType
from prismpy.standards.applied_to_scope import AppliedToScope


# ── Cat D documentary-basis vocabulary (4-value Literal) ────────────


# Per WA CA-17 absorbed: Cat D rows MUST carry a structural reason
# the cell's flagged warning doesn't actually apply. The four named
# bases cover the dominant Sprint F + Sprint E.3 audit-data class:
#
# * ``irrigation_infrastructure`` — a flagged precip-coverage cell
#   in an irrigated zone (the rainfed assumption is wrong; irrigation
#   supplies the water).
# * ``documented_microclimate`` — a flagged tmax extreme that's
#   accurate per local station data the broader climatology missed.
# * ``shallow_rooted_crop_variety`` — a flagged soil-depth /
#   profile-aggregation issue that's irrelevant to a shallow-rooted
#   variety (millet, cowpea) where only the topsoil layer matters.
# * ``other`` — escape hatch for documentary bases that don't fit the
#   three named categories. The companion field
#   ``documentary_basis_other_specify`` carries the persona's free-
#   form description (validator below enforces conditional-required
#   pairing).
CategoryDDocumentaryBasis = Literal[
    "irrigation_infrastructure",
    "documented_microclimate",
    "shallow_rooted_crop_variety",
    "other",
]


# Minimum free-form-text floors shared across multiple validators.
# A 20-character minimum on ``evidence_detail`` rejects single-token
# placeholders (``"ok"`` / ``"yes"`` / ``"see notes"``) without
# requiring full-paragraph essays for legitimate one-line citations.
_MIN_EVIDENCE_DETAIL_CHARS = 20


class OverrideRecord(BaseModel):
    """Canonical schema for a per-cell cockpit Override decision.

    Pure Pydantic — no Django, no I/O. The prismweb side at
    ``core/services/pipeline_run_decisions.py::_row_to_record`` re-
    hydrates this from the Django ``PipelineRunDecision.override_record``
    JSONField on read.

    Schema-layer guarantees per durable §6.4:

    * ``extra="forbid"`` — typo'd field names raise at construction.
    * ``frozen=True`` — top-level field reassignment raises.
    * ``validate_assignment=True`` — for the model_validator pass.
    * ``Tuple[CellID, ...]`` on ``applied_to_snapshot`` — structural
      immutability that survives ``frozen=True``'s top-level-only
      reach (WA CA-19).
    * 5 model_validators enforce the value/documentary discriminator
      + conditional-required companions per AC-E3-4 contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )

    override_climate_values: Optional[Dict[str, float]] = None
    """Per-variable climate overrides. Keys are canonical
    ``variable_key`` strings (e.g., ``tmax_growing_season_mean``)
    matching the registry at
    ``prismpy/standards/override_value_shapes.py``. Values are
    floats in the canonical unit per the registry's ``unit`` field.

    None for Cat D documentary rows. Cat A/B/C rows MUST carry
    either this field or ``override_soil_values`` non-null
    (validator 2 below)."""

    override_soil_values: Optional[Dict[str, Any]] = None
    """Per-variable soil overrides. Keys match canonical soil
    variable_keys; values may be ``float`` (sand %, ph) or
    ``str`` (categorical) per the registry. ``Any`` annotation
    accommodates the wider categorical surface; the form-side
    validation enforces per-key types.

    None for Cat D documentary rows."""

    evidence_type: EvidenceType = Field(
        ...,
        description=(
            "Categorical evidence basis per AC-E3-1 + canonical "
            "Literal at wizard_decisions.py:119 (6 values: "
            "local_trial / irrigation / cultivar_specific / "
            "citation / field_observation / other). Companion "
            "evidence_type_other_specify carries free-form text "
            "when 'other' is selected."
        ),
    )

    evidence_type_other_specify: Optional[str] = Field(
        default=None,
        max_length=200,
        description=(
            "Free-form text required only when "
            "``evidence_type == 'other'``. Empty / whitespace-only "
            "submissions reject in the model validator below. "
            "When ``evidence_type`` is one of the 5 named buckets "
            "this field MUST be ``None`` so a typo'd UI doesn't "
            "smuggle a free-form payload alongside a categorical "
            "pick (per AC-E3-1 conditional-required pattern)."
        ),
    )

    documentary_basis_other_specify: Optional[str] = Field(
        default=None,
        max_length=200,
        description=(
            "Free-form text required only when "
            "``category_d_documentary_basis == 'other'``. Distinct "
            "from ``evidence_type_other_specify`` per codex "
            "MEDIUM-4 absorbed: a Cat D row whose evidence_type "
            "is 'other' AND whose documentary_basis is also "
            "'other' carries TWO separate free-form payloads — "
            "one explaining the evidence type, one explaining the "
            "structural-reason category. Conflating them would "
            "lose half the persona's audit trail."
        ),
    )

    evidence_detail: str = Field(
        ...,
        min_length=_MIN_EVIDENCE_DETAIL_CHARS,
        description=(
            "Persona-supplied evidence text (citation, field-"
            "observation note, irrigation-system documentation, "
            "etc.). Minimum 20 chars rejects placeholder strings; "
            "the form-side filler-rejection heuristic per "
            "AC-F-6 + warning-auditor LOW-4 closes single-char "
            "repeats that would clear the length floor without "
            "carrying real content."
        ),
    )

    applied_at_decision_id: UUID = Field(
        ...,
        description=(
            "UUID of the enclosing ``CellDecisionRecord``. Self-"
            "link integrity ('applied_at_decision_id == enclosing "
            "decision_id') is enforced at the enclosing layer per "
            "interpolated_cell.py:21-24 precedent (codex LOW-2 "
            "absorbed: embedded records cannot see their owner)."
        ),
    )

    applied_to_scope: AppliedToScope = Field(
        ...,
        description=(
            "Scope discriminator per AC-E3-2 canonical Literal "
            "(3 values: single_cell / zone / enumerated_cells). "
            "When 'zone' is selected, the companion "
            "``applied_to_zone_id`` carries the zone identifier; "
            "validator 5 below enforces the conditional-required "
            "pairing."
        ),
    )

    applied_to_zone_id: Optional[str] = Field(
        default=None,
        max_length=20,
        description=(
            "Köppen-zone identifier (e.g., 'Cwa', 'BWh') populated "
            "iff ``applied_to_scope == 'zone'`` per AC-E3-2 sub-2. "
            "Validator 5 enforces both directions: scope=zone with "
            "zone_id=None rejects, scope=single_cell with non-null "
            "zone_id rejects."
        ),
    )

    applied_to_snapshot: Tuple[CellID, ...] = Field(
        ...,
        description=(
            "Snapshot of cell-ids the override applied to at "
            "decision time. ``Tuple`` not ``List`` per WA CA-19: "
            "Pydantic frozen=True only freezes top-level "
            "reassignment; a mutable List could be appended-to "
            "after construction. ``Tuple`` is structurally "
            "immutable, mirroring CockpitManifestEntry."
            "affected_cells precedent at manifest.py:153. "
            "Snapshot semantics: if zone X gains new cells in a "
            "later run, the prior OverrideRecord's snapshot does "
            "NOT silently extend — the State C-newcells variant "
            "(per AC-E3-19) surfaces the new cells for explicit "
            "extension or new-decision creation."
        ),
    )

    check_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Per-cell warning identifier the override answers "
            "(e.g., 'value_range_tmax', 'crop_region_mismatch'). "
            "Stored as a string here; the canonical-enumeration "
            "membership validation lives at the enclosing "
            "``CellDecisionRecord`` layer per AC-E3-5 since the "
            "enumeration is dynamic (computed from "
            "validators/scientific.py expansions)."
        ),
    )

    category_d_documentary_basis: Optional[CategoryDDocumentaryBasis] = Field(
        default=None,
        description=(
            "Cat D structural-reason discriminator (4-value "
            "Literal per WA CA-17). None for Cat A/B/C value-"
            "replacement rows; non-None marks the override as "
            "documentary-only (no override values; not emitted "
            "to translator sidecar per AC-E3-7 codex CA-3 "
            "absorbed)."
        ),
    )

    @model_validator(mode="after")
    def _validate_override_invariants(self) -> "OverrideRecord":
        """Five schema-level invariants per AC-E3-4 contract.

        1. ``evidence_type_other_specify`` non-empty iff
           ``evidence_type == "other"``.
        2. Cat A/B/C (``category_d_documentary_basis is None``)
           requires non-null ``override_climate_values`` OR
           ``override_soil_values``.
        3. Cat D (``category_d_documentary_basis is not None``)
           requires null ``override_climate_values`` AND null
           ``override_soil_values``.
        4. ``documentary_basis_other_specify`` non-empty iff
           ``category_d_documentary_basis == "other"``.
        5. ``applied_to_zone_id`` non-empty iff
           ``applied_to_scope == "zone"``.
        """
        # ── §1 evidence_type_other_specify conditional-required ──
        if self.evidence_type == "other":
            if (
                self.evidence_type_other_specify is None
                or not self.evidence_type_other_specify.strip()
            ):
                raise ValueError(
                    "OverrideRecord.evidence_type_other_specify MUST "
                    "be a non-empty string when evidence_type == "
                    "'other'; got "
                    f"{self.evidence_type_other_specify!r}."
                )
        else:
            if self.evidence_type_other_specify is not None:
                raise ValueError(
                    f"OverrideRecord.evidence_type_other_specify MUST "
                    f"be None when evidence_type == "
                    f"{self.evidence_type!r}; got "
                    f"{self.evidence_type_other_specify!r}. The "
                    f"free-form text field is reserved for the "
                    f"'other' branch per AC-E3-1 sub-2."
                )

        # ── §2 + §3 value-replacement vs documentary discriminator ─
        is_documentary = self.category_d_documentary_basis is not None
        has_climate_values = self.override_climate_values is not None
        has_soil_values = self.override_soil_values is not None

        if is_documentary:
            # §3 Cat D — null override values + non-null documentary
            # basis (the basis discriminator already enforces non-null
            # via the type system; this validator enforces the
            # NULL override-values pair).
            if has_climate_values or has_soil_values:
                raise ValueError(
                    f"OverrideRecord with category_d_documentary_basis="
                    f"{self.category_d_documentary_basis!r} (Cat D "
                    f"documentary row) MUST have null "
                    f"override_climate_values AND null "
                    f"override_soil_values; got climate="
                    f"{self.override_climate_values!r}, soil="
                    f"{self.override_soil_values!r}. Cat D rows are "
                    f"audit/methods-only — the documentary basis IS "
                    f"the override (per codex CA-3 + AC-E3-7 sidecar "
                    f"filtering)."
                )
        else:
            # §2 Cat A/B/C — non-null override values required.
            if not (has_climate_values or has_soil_values):
                raise ValueError(
                    "OverrideRecord with category_d_documentary_basis="
                    "None (Cat A/B/C value-replacement row) MUST have "
                    "non-null override_climate_values OR non-null "
                    "override_soil_values; both are None. A value-"
                    "replacement Override that doesn't carry a value "
                    "has nothing to apply at translator-write time."
                )

        # ── §4 documentary_basis_other_specify conditional-required ─
        if self.category_d_documentary_basis == "other":
            if (
                self.documentary_basis_other_specify is None
                or not self.documentary_basis_other_specify.strip()
            ):
                raise ValueError(
                    "OverrideRecord.documentary_basis_other_specify "
                    "MUST be a non-empty string when "
                    "category_d_documentary_basis == 'other'; got "
                    f"{self.documentary_basis_other_specify!r}."
                )
        else:
            if self.documentary_basis_other_specify is not None:
                raise ValueError(
                    f"OverrideRecord.documentary_basis_other_specify "
                    f"MUST be None when "
                    f"category_d_documentary_basis == "
                    f"{self.category_d_documentary_basis!r}; got "
                    f"{self.documentary_basis_other_specify!r}. The "
                    f"free-form text field is reserved for the "
                    f"'other' Cat D branch per codex MEDIUM-4."
                )

        # ── §5 applied_to_zone_id conditional-required ─────────────
        if self.applied_to_scope == "zone":
            if (
                self.applied_to_zone_id is None
                or not self.applied_to_zone_id.strip()
            ):
                raise ValueError(
                    "OverrideRecord.applied_to_zone_id MUST be a "
                    "non-empty string when applied_to_scope == "
                    "'zone'; got "
                    f"{self.applied_to_zone_id!r}. The companion zone "
                    "identifier carries the actual Köppen code (e.g., "
                    "'Cwa') per AC-E3-2 sub-2."
                )
        else:
            if self.applied_to_zone_id is not None:
                raise ValueError(
                    f"OverrideRecord.applied_to_zone_id MUST be None "
                    f"when applied_to_scope == {self.applied_to_scope!r}; "
                    f"got {self.applied_to_zone_id!r}. The companion "
                    f"zone identifier is reserved for the 'zone' "
                    f"scope discriminator per AC-E3-2 sub-2."
                )

        return self


__all__ = [
    "CategoryDDocumentaryBasis",
    "OverrideRecord",
]
