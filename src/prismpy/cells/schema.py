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

from typing import Any, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Type aliases for the discriminator literals — re-exported via
# ``prismpy.cells`` so callers can pin the legal values at the
# type level without re-reading the schema file.
DataAvailability = Literal["complete", "unavailable"]
UnavailableReason = Literal["climate", "soil", "climate_and_soil"]
CellSummaryVersion = Literal["2.0", "2.1"]

CELL_SUMMARY_VERSIONS: tuple[str, ...] = ("2.0", "2.1")
CELL_SUMMARY_VERSION_LATEST: str = "2.1"


# Climate / soil check-id taxonomies used by the per-axis
# invariant below. The lists mirror the V2-22c-PRE failed_check
# `category` field (which is the preferred discriminator); the
# name-based fallback handles older records that pre-date the
# explicit category field.
_CLIMATE_CHECK_NAMES: frozenset[str] = frozenset({
    "temporal_completeness",
    "cross_variable_consistency",
})
_CLIMATE_VALUE_RANGE_VARS: frozenset[str] = frozenset({
    "tmax", "tmin", "precip", "srad", "rh", "wind",
})


def _failed_check_category(failed_check: Any) -> Optional[str]:
    """Return ``"climate"`` / ``"soil"`` / ``None`` for a single
    ``failed_checks`` entry. Prefers the explicit ``category``
    field (V2-22c-PRE schema lock); falls back to a check-id
    name-prefix match for legacy records.

    Returns ``None`` for non-dict entries or entries whose
    check-id does not match either taxonomy — those entries
    are NOT enforced by the per-axis invariant (they could be
    coverage / post-translate / region-scope checks that fall
    outside the per-cell climate/soil split)."""
    if not isinstance(failed_check, dict):
        return None
    explicit = failed_check.get("category")
    if explicit in ("climate", "soil"):
        return explicit
    raw_id = failed_check.get("check_id") or failed_check.get("check")
    if not isinstance(raw_id, str):
        return None
    if raw_id in _CLIMATE_CHECK_NAMES:
        return "climate"
    if raw_id.startswith("value_range_"):
        suffix = raw_id[len("value_range_"):]
        if suffix in _CLIMATE_VALUE_RANGE_VARS:
            return "climate"
        if suffix == "texture_sum" or suffix.startswith("soil_"):
            return "soil"
    if raw_id.startswith("soil_completeness_"):
        return "soil"
    return None


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
    def _check_unavailable_failed_checks_per_axis(self) -> "CellSummary":
        """Invariant 3 (per-axis, refined per crop-modeling-
        specialist's revised §1.3) — when a cell is unavailable
        the failed_check entries must respect the unavailable
        AXIS, not the cell as a whole. Mixed-availability cells
        (climate-available + soil-unavailable, or vice versa)
        legitimately retain fails on the AVAILABLE axis; only
        the unavailable-axis checks must be absent.

        Per-axis predicate:

        * ``unavailable_reason == "climate"`` forbids climate
          checks in ``failed_checks``; soil checks are allowed
          (the validator did run on the available soil axis).
        * ``unavailable_reason == "soil"`` forbids soil checks;
          climate checks are allowed.
        * ``unavailable_reason == "climate_and_soil"`` forbids
          both — neither axis ran, so failed_checks must be
          empty.

        The category discriminator uses the failed-check entry's
        explicit ``category`` field (V2-22c-PRE lock), with a
        check-id name-prefix fallback for legacy records.
        Coverage / post-translate / region-scope checks that
        carry no climate/soil category fall outside this
        invariant — they are not enforced here because they
        operate at a region scope where the per-cell axis
        distinction does not apply."""
        if self.data_availability != "unavailable":
            return self
        reason = self.unavailable_reason
        if reason == "climate":
            forbidden: Tuple[str, ...] = ("climate",)
        elif reason == "soil":
            forbidden = ("soil",)
        elif reason == "climate_and_soil":
            forbidden = ("climate", "soil")
        else:
            # The earlier invariant guarantees reason is set
            # when data_availability is unavailable, so this is
            # defense-in-depth only.
            return self
        for entry in self.failed_checks:
            category = _failed_check_category(entry)
            if category is None or category not in forbidden:
                continue
            check_id = (
                entry.get("check_id")
                or entry.get("check")
                or "<unknown>"
            ) if isinstance(entry, dict) else "<non-dict>"
            raise ValueError(
                f"data_availability='unavailable' with "
                f"unavailable_reason={reason!r} forbids {category} "
                f"checks in failed_checks (got check_id={check_id!r}). "
                f"The validator cannot fail an axis that did not run."
            )
        return self
