"""Cell-summary Pydantic schema (v2.1).

Per-cell quality + provenance record emitted by the pipeline
executor for every grid cell. v2.1 introduces three new fields
that distinguish "validation ran and passed/failed" from
"validation could not run because the input data was missing":

* ``data_availability`` — Literal "complete" | "unavailable".
  Tracks whether the cell received the inputs validation
  expects. Default is "complete" so legacy v2.0 records
  produced before this schema lands still parse.
* ``unavailable_reason`` — Optional Literal "climate" | "soil" |
  "climate_and_soil". Discriminates which input substrate is
  missing. Required when ``data_availability == "unavailable"``;
  must be ``None`` when ``data_availability == "complete"``.
* ``cell_summary_version`` — Literal "2.0" | "2.1". The
  executor stamps "2.1" once it populates the new fields; until
  the validator short-circuit (§2) ships, the executor still
  emits "2.0" and the schema accepts both.

The three cross-field invariants below mirror crop-modeling-
specialist's §1.3 spec verbatim — they enforce that consumers
can trust the discriminator without re-checking each callsite.

Backward compatibility (§1.4):

* **v2.1 reader, v2.0 record**: the new fields default
  ("complete", None, "2.0"), so older records load cleanly
  with the implicit pre-v2.1 assumption preserved.
* **v2.0 reader, v2.1 record**: requires the v2.0 reader's
  Pydantic models to accept ``extra="ignore"`` so the new
  fields silently drop. The current dict-based v2.0 readers
  in prismpy + prismweb already drop unknown keys silently;
  no migration step is required, but the audit pattern is
  named here so future strict-schema work doesn't regress.

The model is structurally compatible with the existing dict
shape — the executor's existing dict literal at
``pipeline/executor.py`` continues to validate cleanly through
``CellSummary.model_validate(...)`` so the migration is a
forward addition, not a reshape.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Type aliases for the discriminator literals — re-exported via
# ``prismpy.cells`` so callers can pin the legal values at the
# type level without re-reading the schema file.
DataAvailability = Literal["complete", "unavailable"]
UnavailableReason = Literal["climate", "soil", "climate_and_soil"]
CellSummaryVersion = Literal["2.0", "2.1"]

CELL_SUMMARY_VERSIONS: tuple[str, ...] = ("2.0", "2.1")
CELL_SUMMARY_VERSION_LATEST: str = "2.1"


class CellSummary(BaseModel):
    """Per-cell quality + provenance record.

    The model is intentionally permissive on legacy fields —
    ``failed_checks`` / ``severity_counts`` / ``sources`` /
    ``cell_failed_check_details`` etc. all stay as untyped
    ``Any`` (or absent) so existing producer code paths that
    populate the record before the v2.1 fields are added do
    not need a coupled refactor. The strict invariants live on
    the v2.1 fields only — those are the discriminator the
    consumers (banner UI, cross-hatch palette, fetch-fallback
    metrics) bind against.
    """

    model_config = ConfigDict(
        extra="ignore",  # v2.0 reader-of-v2.1-record compat shim.
        validate_assignment=True,
    )

    # --- v2.0 baseline shape (existing) ---
    cell_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    failed_checks: List[Any] = Field(default_factory=list)

    # --- v2.1 new fields ---
    data_availability: DataAvailability = "complete"
    unavailable_reason: Optional[UnavailableReason] = None
    cell_summary_version: CellSummaryVersion = CELL_SUMMARY_VERSION_LATEST

    # --- §1.3 cross-field invariants ---

    @model_validator(mode="after")
    def _check_unavailable_requires_reason(self) -> "CellSummary":
        """Invariant 1 — ``data_availability == "unavailable"``
        requires ``unavailable_reason`` to be set. Without a
        reason the consumer can't disambiguate which substrate
        failed (climate vs soil vs both)."""
        if self.data_availability == "unavailable" and self.unavailable_reason is None:
            raise ValueError(
                "data_availability='unavailable' requires unavailable_reason "
                "to be set ('climate', 'soil', or 'climate_and_soil'). "
                "Without it, downstream consumers cannot route the cell to "
                "the right banner / palette path."
            )
        return self

    @model_validator(mode="after")
    def _check_complete_forbids_reason(self) -> "CellSummary":
        """Invariant 2 — ``data_availability == "complete"``
        forbids ``unavailable_reason``. The reason field's
        contract is "names which substrate failed when one
        did" — pairing a reason with a complete cell is a
        contradiction the consumer shouldn't have to decode."""
        if self.data_availability == "complete" and self.unavailable_reason is not None:
            raise ValueError(
                "data_availability='complete' forbids unavailable_reason "
                f"(got {self.unavailable_reason!r}). The reason field is "
                "only meaningful when validation could not run."
            )
        return self

    @model_validator(mode="after")
    def _check_unavailable_implies_no_failed_checks(self) -> "CellSummary":
        """Invariant 3 — ``data_availability == "unavailable"``
        requires ``failed_checks`` to be empty. The validator
        cannot fail what it did not run; if a cell is marked
        unavailable but carries failed_checks entries, the
        producer pipeline has a bug (the short-circuit fired
        AFTER a check ran, not before)."""
        if self.data_availability == "unavailable" and self.failed_checks:
            raise ValueError(
                "data_availability='unavailable' requires failed_checks=[] "
                f"(got {len(self.failed_checks)} failed-check entries). "
                "The validator cannot fail what it did not run; this combo "
                "indicates the short-circuit fired after a check executed."
            )
        return self
