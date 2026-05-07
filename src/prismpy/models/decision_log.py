"""``CellDecisionRecord`` — canonical schema for a per-cell cockpit
decision + the ``current_decisions`` canonical reader.

Sprint E.2 AC-E2-4 + AC-E2-21 + §0.2 canonical-layer #1-#3 + #6.

The cockpit's decision-workflow semantics live in two halves:

* This module owns the **Pydantic schema** + the **canonical reader
  for current state per cell**. PURE — no Django, no I/O.

* The prismweb side at ``core/models.py::PipelineRunDecision`` +
  ``core/views/cockpit.py::cockpit_prepare()`` owns the **Django
  persistence** + **HTMX endpoint orchestration**. The split honors
  the I-DN-1 canonical-source-vs-Django-persistence boundary: prismweb
  consumes Pydantic via ``model_validate`` on read/write boundaries.

Each ``CellDecisionRecord`` captures one user decision on one cell:

* **Identity** — ``decision_id`` (UUID); ``cell_id`` (canonical
  cell-id reference per ``prismpy.cells.schema``).

* **Action** — one of ``acknowledge`` (Bucket-2) / ``skip_from_analysis``
  (Bucket-3) / ``apply_interpolation`` (Bucket-4) /
  ``document_override`` (Bucket-5). The ``"rerun_full_sources"``
  exit-boundary affordance does NOT map to a decision (it spawns a
  new pipeline run via the existing rerun mechanism per §0.2 #1).

* **Provenance** — ``check_id`` (the per-cell warning identifier
  the decision answers); ``timestamp`` (server-side wall-clock,
  ISO 8601 UTC); ``user_identity`` (who made the decision);
  ``method_or_rationale`` (free-text — IDW description for
  interpolate; skip rationale for skip; override note for
  override; acknowledgement note for acknowledge).

* **Ordering tuple** — ``(timestamp, sequence_number)`` per §0.2
  canonical-source #2. Service-layer auto-increments
  ``sequence_number`` per ``pipeline_run_id`` on POST. Same-
  millisecond opposing decisions resolve deterministically by
  ``sequence_number`` ascending.

* **Bulk grouping** — ``bulk_operation_id`` (Optional UUID; per
  AC-E2-23 + CA-H2/CA-22) groups all CellDecisionRecords belonging
  to one bulk-interpolation action. ``None`` for per-cell decisions.

* **Interpolation context** — ``interpolation_record`` is non-None
  iff ``action == "apply_interpolation"``. Schema-level
  model_validator enforces both directions (presence + symmetric
  absence) per WA CA-2. Self-link integrity is also enforced:
  ``interpolation_record.applied_at_decision_id == self.decision_id``
  (per §0.2 #6 + CA-H7).

* **Revert chain** — ``revert_of`` is non-None when this record
  reverts a prior decision. Service-layer (prismweb) validates the
  referenced UUID exists (Pydantic can't see cross-record context).
  ``current_decisions()`` below reads the chain and returns the
  most-recent non-reverted state per cell — write A → revert A →
  both filtered; write C → C is current.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from prismpy.models.interpolated_cell import CellID, InterpolatedCellRecord


# ── Action taxonomy ─────────────────────────────────────────────────


# The four per-cell actions the cockpit creates a decision-log entry
# for. ``"rerun_full_sources"`` (the fifth ``AffordanceType`` value
# from AC-E2-3) is an exit-boundary that spawns a new pipeline run
# rather than writing a decision-log entry — the rerun itself is
# the audit trail. Per §0.2 canonical-source #1 + AC-E2-3
# ``AFFORDANCE_TO_ACTION_MAP`` parity.
DecisionAction = Literal[
    "apply_interpolation",
    "skip_from_analysis",
    "document_override",
    "acknowledge",
]


class CellDecisionRecord(BaseModel):
    """Canonical schema for a per-cell cockpit decision.

    Pure Pydantic — no Django, no I/O. The prismweb side at
    ``core/models.py::PipelineRunDecision`` consumes this via
    ``model_validate`` on read/write boundaries (per I-DN-1 split).

    Schema-layer guarantees per durable §6.4:

    * ``extra="forbid"`` — typo'd field names raise at construction.
    * ``validate_assignment=True`` — assignment-time validation.
    * ``model_validator(mode="after")`` enforces three invariants:
      - ``action == "apply_interpolation"`` ⇔ ``interpolation_record
        is not None`` (per WA CA-2 — both directions).
      - When ``interpolation_record is not None``, the embedded
        record's ``applied_at_decision_id`` == this record's
        ``decision_id`` (per §0.2 #6 + CA-H7 self-link integrity).
    * Cross-record validation (revert_of references an existing
      decision) is at the service-layer in prismweb (Pydantic scope
      stops at single-record validation).
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    decision_id: UUID
    """UUID of this decision. Becomes the canonical handle for
    revert-of references and self-link integrity on the embedded
    InterpolatedCellRecord."""

    cell_id: CellID
    """Canonical cell-id reference. Two decisions on the same cell
    are valid — the most-recent-non-reverted entry per cell wins
    per ``current_decisions()`` reader semantics."""

    action: DecisionAction
    """One of the four per-cell decision actions. The
    ``"rerun_full_sources"`` exit-boundary affordance is NOT in this
    Literal — it spawns a new pipeline run via the existing rerun
    mechanism rather than writing a decision-log entry."""

    check_id: str
    """Per-cell warning identifier the decision answers (e.g.,
    ``value_range_precip``, ``crop_region_mismatch``). Sprint F
    shipped these as raw strings; an enum tightening is out-of-
    scope for E.2."""

    timestamp: datetime
    """Server-side wall-clock at decision creation. Combined with
    ``sequence_number`` per §0.2 canonical-source #2 for the
    deterministic ordering tuple."""

    user_identity: str
    """Who made the decision. May be a placeholder
    ``"legacy"`` string for migrated E.1 entries (per
    AC-E2-13 migration reader). Single-user workflows in V2 use the
    Django session user; multi-user attribution is V3+."""

    method_or_rationale: str
    """Free-text rationale for the decision:
    - ``apply_interpolation``: IDW description (auto-generated).
    - ``skip_from_analysis``: skip rationale (user input or
      preflight-derived for bulk).
    - ``document_override``: user's override note.
    - ``acknowledge``: optional acknowledgement note."""

    sequence_number: int
    """Service-layer auto-incremented per ``pipeline_run_id``.
    Combined with ``timestamp`` for the deterministic ordering
    tuple per §0.2 canonical-source #2; same-millisecond decisions
    resolve in sequence_number ascending order."""

    interpolation_record: Optional[InterpolatedCellRecord] = None
    """The embedded interpolation record when
    ``action == "apply_interpolation"``; ``None`` for all other
    actions. Schema-level model_validator enforces presence symmetry
    AND self-link integrity (record.applied_at_decision_id ==
    self.decision_id)."""

    revert_of: Optional[UUID] = None
    """If non-None, this record reverts the prior decision with the
    matching UUID. Service-layer (prismweb) validates the reference
    exists; Pydantic scope stops at single-record validation."""

    bulk_operation_id: Optional[UUID] = None
    """If non-None, all CellDecisionRecords sharing this UUID
    belong to one bulk-interpolation action (per AC-E2-23 +
    CA-H2/CA-22). ``None`` for per-cell decisions. Audit log
    groups records by this field."""

    @model_validator(mode="after")
    def _validate_interpolation_record_invariants(self) -> "CellDecisionRecord":
        """Three schema-level invariants per WA CA-2 + §0.2 #6:

        1. ``action == "apply_interpolation"`` ⇒ ``interpolation_record
           is not None``.
        2. ``action != "apply_interpolation"`` ⇒ ``interpolation_record
           is None``.
        3. When ``interpolation_record is not None``, the embedded
           record's ``applied_at_decision_id`` == this record's
           ``decision_id`` (self-link integrity per CA-H7).
        """
        if self.action == "apply_interpolation":
            if self.interpolation_record is None:
                raise ValueError(
                    "CellDecisionRecord with action='apply_interpolation' "
                    "MUST carry an interpolation_record; got None."
                )
            if self.interpolation_record.applied_at_decision_id != self.decision_id:
                raise ValueError(
                    f"CellDecisionRecord self-link integrity violation: "
                    f"interpolation_record.applied_at_decision_id="
                    f"{self.interpolation_record.applied_at_decision_id} "
                    f"!= decision_id={self.decision_id}. The embedded "
                    f"record's applied_at_decision_id MUST match the "
                    f"enclosing decision_id (per §0.2 canonical-source "
                    f"#6)."
                )
        else:
            if self.interpolation_record is not None:
                raise ValueError(
                    f"CellDecisionRecord with action={self.action!r} MUST "
                    f"NOT carry an interpolation_record; got non-None "
                    f"record. interpolation_record is reserved for "
                    f"action='apply_interpolation'."
                )
        return self


# ── Canonical reader: most-recent-non-reverted per cell ──────────────


def current_decisions(
    records: list[CellDecisionRecord],
) -> dict[CellID, Optional[CellDecisionRecord]]:
    """Return the active (non-reverted) decision per cell.

    AC-E2-21 + §0.2 canonical-source #3. The reader handles the full
    revert-of-revert chain semantics:

    * Write A → ``current_decisions == {cell_a: A}``.
    * Write A → write B with ``revert_of=A.decision_id`` →
      ``current_decisions == {cell_a: None}`` (both A and the
      revert filtered; no current decision active for the cell).
    * Write A → revert A → write C (no revert_of) →
      ``current_decisions == {cell_a: C}`` (C is the current
      decision; A and the revert are inert history).

    The reader sorts internally by ``(timestamp, sequence_number)``
    ascending per §0.2 canonical-source #2 — caller need NOT
    pre-sort; passing a shuffled list produces the same output as
    passing a sorted list (per WA Draft-2 CA-17 shuffled-input
    invariance).

    Cells with NO decisions in the input list do not appear in the
    output dict. The dict's keys are exactly the set of cells with
    at least one decision in ``records``.

    Per durable §24 canonical-source-or-pin: this is the only place
    in prismpy where revert-chain semantics are computed; consumers
    in prismweb (top-strip remaining-counter at AC-E2-9, audit-log
    expand at State D, manifest flag invariant at AC-E2-8) MUST
    route through this helper rather than re-implementing.

    Args:
        records: list of CellDecisionRecord — order doesn't matter.

    Returns:
        Dict mapping each cell_id present in ``records`` to its
        current active decision (or ``None`` if all decisions on
        the cell were reverted).
    """
    if not records:
        return {}

    # Sort by canonical ordering tuple per §0.2 #2. Caller need
    # not pre-sort. Tie-break by decision_id (UUID stringification)
    # for determinism on duplicate (timestamp, sequence_number)
    # tuples per codex MEDIUM-3 absorption.
    sorted_records = sorted(
        records,
        key=lambda r: (r.timestamp, r.sequence_number, str(r.decision_id)),
    )

    # Per-cell stack semantics (codex HIGH #1 absorption). Walk
    # records in chronological order; for each cell, maintain a
    # stack of "active" non-reverted decisions. Encountering a
    # ``revert_of=X`` record pops the matching entry from the
    # stack (a no-op if ``X`` isn't on the stack — defensive).
    # The top of each cell's stack at end-of-walk is the current
    # active decision; an empty stack means no current decision
    # (all reverted).
    #
    # This semantic correctly handles the Drill I revert-of-revert
    # chain (write A → revert A → write C → C is current) AND the
    # codex-flagged write-D → write-A → revert-A → D-restored case
    # (D remains current; the revert undoes A and unmasks D).
    stacks: dict[CellID, list[CellDecisionRecord]] = {}
    for record in sorted_records:
        cell_stack = stacks.setdefault(record.cell_id, [])
        if record.revert_of is not None:
            # Revert action: pop the matching decision off the
            # stack. We walk from top (most-recent) so a revert
            # that targets a non-current entry still un-masks
            # the entry below it on the next walk.
            for idx in range(len(cell_stack) - 1, -1, -1):
                if cell_stack[idx].decision_id == record.revert_of:
                    cell_stack.pop(idx)
                    break
            # The revert record itself does NOT push onto the
            # stack — it's a reversal, not a new decision.
        else:
            # Regular decision: push.
            cell_stack.append(record)

    return {
        cell_id: (stack[-1] if stack else None)
        for cell_id, stack in stacks.items()
    }


__all__ = [
    "CellDecisionRecord",
    "DecisionAction",
    "current_decisions",
]
