"""``InterpolatedCellRecord`` — canonical schema for an imputed cell.

Sprint E.2 AC-E2-1. Every cell whose value the cockpit's IDW engine
imputes carries one of these records on the per-cell decision-log
entry (``CellDecisionRecord.interpolation_record`` per AC-E2-4 +
§0.2 canonical-source #6 self-link integrity).

The record captures four kinds of information:

* **Method identity** — ``interpolation_method`` (canonical Literal
  mirroring ``IDW_DEFAULT_METHOD_LITERAL``) + ``method_doi`` (DOI of
  the foundational paper for the method).
* **Sources** — ``source_cells`` (the k ≤ 4 neighbour cells whose
  values were combined into the imputed value).
* **Uncertainty** — ``uncertainty_ci_lower`` / ``uncertainty_ci_upper``
  (95 % confidence interval bounds for the imputed value).
* **Domain context** — ``affected_zone_code`` (Köppen zone the cell
  falls in) + ``caveat_codes`` (zone-specific caveats applicable to
  the imputation; cf. ``prismpy.standards.interpolation_caveats``).

Plus a self-link: ``applied_at_decision_id`` references the enclosing
``CellDecisionRecord.decision_id``. The cross-record self-link is
validated at the ``CellDecisionRecord`` layer per §0.2 #6 (this
record can't see the decision-log context to validate it itself).

The Literal at ``interpolation_method`` is hardcoded VERBATIM in this
module rather than imported from ``IDW_DEFAULT_METHOD_LITERAL`` —
``typing.Literal`` arguments must be compile-time string constants
for the type checker. A mirror pin at
``tests/structural/test_idw_method_literal_mirrors_constant.py``
asserts the schema's spelling matches the canonical constant. Drift
between the two fires loud at CI time (durable §24
canonical-source-or-pin discipline).
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prismpy.koppen.zones import KoppenZone
from prismpy.standards.caveat_codes import CaveatCode


# CellID is a thin alias for str (matches the existing
# ``prismpy.cells.schema`` representation at line 185 — `cell_id:
# Optional[str]`). A future canonical CellID type would extend
# this; for now an explicit alias keeps the field self-documenting.
CellID = str


# DOI regex (per AC-E2-1 sub-criterion + Builder CA-1 absorption):
# Python regex syntax with explicit lowercase + uppercase character
# class — NO trailing /i flag (Python regex doesn't terminate with
# JS-style flags).
_DOI_PATTERN = r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$"


class InterpolatedCellRecord(BaseModel):
    """Canonical schema for an imputed cell record.

    Attached to the enclosing ``CellDecisionRecord`` when its
    ``action == "apply_interpolation"`` (per AC-E2-4 +
    §0.2 canonical-source #6 self-link integrity). The record's
    ``applied_at_decision_id`` MUST equal the enclosing decision's
    ``decision_id``; the cross-record assertion lives at the
    ``CellDecisionRecord`` layer.

    Schema-layer guarantees per durable §6.4:

    * ``extra="forbid"`` — a typo in a field name raises at
      construction (silent data loss prevented).
    * ``validate_assignment=True`` — assignment-time validation
      catches drift introduced after construction.
    * ``source_cells`` non-empty (a 0-neighbour case routes to skip
      per AC-E2-3 BEFORE the engine runs; an InterpolatedCellRecord
      with no source_cells is a substrate bug).
    * ``uncertainty_ci_lower <= uncertainty_ci_upper`` enforced; an
      inverted CI is a numeric-formula bug.
    * ``method_doi`` matches the DOI regex; freeform strings risk
      drifting into bibliographic-text rather than identifiers.
    * ``affected_zone_code`` validated against the canonical
      ``KoppenZone`` Literal (mirror-of-registry per AC-E2-20).
    * ``caveat_codes`` items validated against the canonical
      ``CaveatCode`` Literal (per §0.2 canonical-source #7).
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    interpolation_method: Literal["idw", "idw_k4_r15km_w_inverse_dist_sq"]
    """Canonical kernel-family identifier. Sprint E.3 AC-E3-11
    extends the Literal to a 2-arg union covering BOTH the post-E.3
    canonical ``"idw"`` (kernel-family handle) AND the legacy pre-E.3
    parameter-encoded ``"idw_k4_r15km_w_inverse_dist_sq"`` (accepted
    during the migration window). Per-record parameters
    (``radius_km`` / ``k`` / ``weight_power`` below) carry the
    actual numeric values so a future kernel-family extension
    (e.g., kriging) can land here without restating the parameter
    surface.

    The Django migration ``0024_interpolated_cell_record_schema_extension.py``
    (ships at AC-E3-16 prismweb-side per builder DELTA-CA-2)
    rewrites legacy rows to ``"idw"`` post-deployment; the post-
    migration tightening drops the legacy literal from this Union
    (V3+ task).
    """

    source_cells: list[CellID] = Field(min_length=1)
    """Cell IDs of the neighbours whose values were combined for
    the imputation. ``min_length=1`` rejects the empty case at
    construction; a 0-neighbour cell routes to skip BEFORE the
    engine runs (AC-E2-3 routing rule + ``InsufficientNeighborsError``
    in AC-E2-2).
    """

    uncertainty_ci_lower: float
    """95 % CI lower bound for the imputed value. ``model_validator``
    below enforces ``ci_lower <= ci_upper`` (an inverted CI signals
    a numeric-formula bug)."""

    uncertainty_ci_upper: float
    """95 % CI upper bound for the imputed value."""

    method_doi: str = Field(pattern=_DOI_PATTERN)
    """DOI for the foundational reference of the interpolation
    method. MVP fixed to ``"10.1145/800186.810616"`` (Shepard 1968,
    *Proceedings of the 1968 ACM National Conference*: 517–524 — the
    foundational IDW paper).
    """

    applied_at_decision_id: UUID
    """UUID of the enclosing ``CellDecisionRecord``. Self-link
    integrity (``== enclosing decision_id``) is enforced at the
    ``CellDecisionRecord`` layer per §0.2 #6."""

    affected_zone_code: KoppenZone
    """Köppen-Geiger zone code the cell falls in. Validated
    against the canonical ``KoppenZone`` Literal (mirror-of-
    registry per AC-E2-20)."""

    caveat_codes: list[CaveatCode] = Field(default_factory=list)
    """Domain-specific caveats applicable to the imputation. Empty
    list is valid (no caveats apply). Each code is validated
    against the canonical ``CaveatCode`` Literal (per §0.2 #7).
    """

    radius_km: float = Field(default=15.0, gt=0.0)
    """Per-record IDW search radius (km). Sprint E.3 AC-E3-11
    canonical-source — the methods-text generator reads this
    field rather than parsing the legacy literal pattern. Per
    :data:`prismpy.standards.idw_methods.IDW_RADIUS_BY_PLATFORM`:
    SARRA-Py / CRAFT = 15 km, PYTHIA = 25 km, ACEA = 100 km
    (CMS CA-1 BLOCKING absorbed). Default 15.0 preserves backward
    compatibility for legacy rows + pre-E.3 fixtures during the
    migration window; the prismweb-side migration ``0024``
    rewrites legacy rows with the platform-correct radius before
    the post-migration tightening fires."""

    k: int = Field(default=4, ge=1)
    """Per-record nearest-neighbour count. Sprint E.3 AC-E3-11
    canonical-source. Default 4 matches Sprint E.2 era
    :data:`prismpy.standards.idw_methods.IDW_DEFAULT_K`."""

    weight_power: float = Field(default=2.0, gt=0.0)
    """Per-record inverse-distance weighting exponent. Sprint E.3
    AC-E3-11 canonical-source. Default 2.0 (Shepard's original
    formulation) matches Sprint E.2 era
    :data:`prismpy.standards.idw_methods.IDW_DEFAULT_W`."""

    @model_validator(mode="after")
    def _validate_ci_ordering(self) -> "InterpolatedCellRecord":
        """Reject inverted CI bounds at construction. An imputed
        value must satisfy ``ci_lower <= mean <= ci_upper``; the
        mean isn't on this record (it's on the enclosing decision's
        cell context), but the lower<=upper invariant catches the
        most common numeric-formula bug class without that mean."""
        if self.uncertainty_ci_lower > self.uncertainty_ci_upper:
            raise ValueError(
                f"InterpolatedCellRecord uncertainty bounds inverted: "
                f"ci_lower={self.uncertainty_ci_lower} > "
                f"ci_upper={self.uncertainty_ci_upper}. The 95% CI "
                f"must satisfy lower <= upper."
            )
        return self


__all__ = [
    "CellID",
    "InterpolatedCellRecord",
]
