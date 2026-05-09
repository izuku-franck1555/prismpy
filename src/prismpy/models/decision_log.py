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
from typing import Literal, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prismpy.models.interpolated_cell import CellID, InterpolatedCellRecord
from prismpy.models.override import OverrideRecord

# NOTE: ``check_id_enumeration`` imports are deferred to the
# validator body below to break a circular import chain
# (decision_log → cockpit.check_id_enumeration → cockpit.routing_decision
# → validators.affordance_routing → decision_log.DecisionAction).
# The lazy-import pattern keeps the module-load graph acyclic while
# still letting the validator read the canonical enumeration at
# construction time.


# Sentinel check_id assigned by the prismweb-side migration backfill
# at AC-E3-16 to legacy ``PipelineRunDecision`` rows that pre-date the
# Sprint F check_id-threading work and have no recoverable check_id
# from manifest / ``Project.cockpit_decisions``. Per AC-E3-6 sub-
# criterion 6 backfill priority: cockpit_decisions JSON → manifest's
# first failed_checks entry → this sentinel. Validator below accepts
# the sentinel even though it's not in
# ``enumerate_emitted_check_ids()``.
_UNKNOWN_LEGACY_CHECK_ID = "unknown_legacy"


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

    check_id: str = Field(..., min_length=1)
    """Per-cell warning identifier the decision answers (e.g.,
    ``value_range_precip``, ``crop_region_mismatch``). Sprint E.3
    AC-E3-5 hardens validation: the model_validator below asserts
    ``check_id`` belongs to ``enumerate_emitted_check_ids()`` (the
    canonical producer-side enumeration at
    ``prismpy/cockpit/check_id_enumeration.py``) OR matches one of
    the documented prefix families OR equals the
    ``"unknown_legacy"`` sentinel that the prismweb-side migration
    backfill assigns to pre-Sprint-F rows with no recoverable
    check_id (per AC-E3-6 sub-6 backfill priority)."""

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

    override_record: Optional[OverrideRecord] = None
    """The embedded override record when
    ``action == "document_override"``; ``None`` for all other
    actions. Sprint E.3 AC-E3-5 schema-level model_validator
    enforces presence symmetry AND self-link integrity
    (``record.applied_at_decision_id == self.decision_id``) per
    codex LOW-2 absorbed (the integrity assertion lives at the
    enclosing layer because the embedded record cannot see its
    owner's UUID)."""

    @model_validator(mode="after")
    def _validate_check_id_canonical_membership(self) -> "CellDecisionRecord":
        """Sprint E.3 AC-E3-5 hardening: ``check_id`` MUST be a
        canonical producer-side enumeration member, a documented-
        prefix-family member, or the ``"unknown_legacy"`` sentinel
        the migration backfill assigns to pre-Sprint-F rows.

        The enumeration is dynamic (computed from
        ``validators/scientific.py::CLIMATE_RANGES`` /
        ``SOIL_RANGES`` expansions) so the validator runs at
        construction time rather than as a static Literal.
        """
        if self.check_id == _UNKNOWN_LEGACY_CHECK_ID:
            return self
        # Lazy import per module-level note above (breaks circular
        # import via cockpit.check_id_enumeration → cockpit.routing_decision
        # → validators.affordance_routing → decision_log).
        from prismpy.cockpit.check_id_enumeration import (
            enumerate_emitted_check_ids,
            matches_known_prefix,
        )
        if self.check_id in enumerate_emitted_check_ids():
            return self
        if matches_known_prefix(self.check_id):
            return self
        raise ValueError(
            f"CellDecisionRecord.check_id={self.check_id!r} is not a "
            f"canonical enumeration member. Allowed: members of "
            f"``enumerate_emitted_check_ids()`` "
            f"(validators/scientific.py + post_translate fan-out), "
            f"members matching a known prefix family per "
            f"``check_id_enumeration.matches_known_prefix``, or the "
            f"``{_UNKNOWN_LEGACY_CHECK_ID!r}`` sentinel for legacy "
            f"rows backfilled by the migration at AC-E3-16."
        )

    @model_validator(mode="after")
    def _validate_override_record_invariants(self) -> "CellDecisionRecord":
        """Sprint E.3 AC-E3-5 + codex LOW-2 absorbed: 4th invariant
        branch (parallels existing 3 invariants for
        ``interpolation_record``):

        1. ``action == "document_override"`` ⇒ ``override_record is
           not None``.
        2. ``action != "document_override"`` ⇒ ``override_record is
           None``.
        3. When ``override_record is not None``, the embedded
           record's ``applied_at_decision_id == self.decision_id``
           (self-link integrity per codex LOW-2 absorbed; lives at
           the enclosing layer because OverrideRecord cannot see its
           owner's UUID).
        """
        if self.action == "document_override":
            if self.override_record is None:
                raise ValueError(
                    "CellDecisionRecord with action='document_override' "
                    "MUST carry an override_record; got None."
                )
            if self.override_record.applied_at_decision_id != self.decision_id:
                raise ValueError(
                    f"CellDecisionRecord self-link integrity violation: "
                    f"override_record.applied_at_decision_id="
                    f"{self.override_record.applied_at_decision_id} "
                    f"!= decision_id={self.decision_id}. The embedded "
                    f"record's applied_at_decision_id MUST match the "
                    f"enclosing decision_id (per codex LOW-2 + §0.2 "
                    f"canonical-source #6)."
                )
        else:
            if self.override_record is not None:
                raise ValueError(
                    f"CellDecisionRecord with action={self.action!r} MUST "
                    f"NOT carry an override_record; got non-None "
                    f"record. override_record is reserved for "
                    f"action='document_override'."
                )
        return self

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
) -> dict[Tuple[CellID, str], Optional[CellDecisionRecord]]:
    """Return the active (non-reverted) decision per ``(cell_id, check_id)``
    pair.

    Sprint E.3 AC-E3-6 reshape (post-Draft 2 builder CA-1 + CA-3
    absorbed) — the reader keys output by ``(cell_id, check_id)``
    tuples to support multi-check coexistence per cell. A single
    cell may now carry multiple active decisions (e.g., Override on
    ``value_range_tmax`` AND Acknowledge on
    ``coverage_climate_cells``); the cockpit's audit panel and
    methods-text generator surface them independently.

    Sprint E.2 AC-E2-21 + §0.2 canonical-source #3 revert-chain
    semantics carry forward unchanged at the per-tuple-key level:

    * Write A → ``current_decisions == {(cell_a, check_id_X): A}``.
    * Write A → write B with ``revert_of=A.decision_id`` →
      ``current_decisions == {(cell_a, check_id_X): None}`` (both A
      and the revert filtered; no current decision active for the
      pair).
    * Write A → revert A → write C (no revert_of) →
      ``current_decisions == {(cell_a, check_id_X): C}`` (C is the
      current decision; A and the revert are inert history).

    Revert records carry the same ``check_id`` as their target
    decision — the prismweb-side service-layer
    ``record_decision`` enforces this match (per Sprint E.2 P2
    cross-source pin "Revert d2 — same cell + same check"). The
    reader trusts that contract: a revert pops from the stack
    keyed on the revert record's own ``(cell_id, check_id)`` pair.

    The reader sorts internally by ``(timestamp, sequence_number)``
    ascending per §0.2 canonical-source #2 — caller need NOT
    pre-sort; passing a shuffled list produces the same output as
    passing a sorted list (per WA Draft-2 CA-17 shuffled-input
    invariance).

    ``(cell_id, check_id)`` pairs with NO decisions in the input
    list do not appear in the output dict. The dict's keys are
    exactly the set of pairs with at least one decision in
    ``records``.

    Per durable §24 canonical-source-or-pin: this is the only
    place in prismpy where revert-chain semantics are computed;
    consumers in prismweb (top-strip remaining-counter at
    AC-E2-9, audit-log expand at State D, manifest flag invariant
    at AC-E2-8) MUST route through this helper rather than
    re-implementing.

    **Counter semantic LOCK** (builder CA-1 absorbed): the cockpit
    UI's "Decisions made: N" counter shows the **tuple-keyed pair
    count** (length of this dict's non-None values). Both metrics
    surface in the prismweb cockpit_e2 view context for UI
    flexibility:

    * ``decided_pair_count`` — count of ``(cell_id, check_id)``
      pairs with non-null active decision (canonical user-visible
      counter).
    * ``decided_cell_count`` — count of unique ``cell_id``s with
      ≥1 active decision (informational; available if UI surfaces
      both).

    Args:
        records: list of CellDecisionRecord — order doesn't matter.

    Returns:
        Dict mapping each ``(cell_id, check_id)`` pair present in
        ``records`` to its current active decision (or ``None`` if
        every decision on the pair was reverted).
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

    # Per-(cell_id, check_id) stack semantics. Walk records in
    # chronological order; for each pair, maintain a stack of
    # "active" non-reverted decisions. Encountering a
    # ``revert_of=X`` record pops the matching entry from the pair-
    # specific stack (a no-op if ``X`` isn't on the stack —
    # defensive). The top of each pair's stack at end-of-walk is
    # the current active decision; an empty stack means no current
    # decision (all reverted).
    stacks: dict[Tuple[CellID, str], list[CellDecisionRecord]] = {}
    for record in sorted_records:
        key = (record.cell_id, record.check_id)
        pair_stack = stacks.setdefault(key, [])
        if record.revert_of is not None:
            # Revert action: pop the matching decision off the
            # pair-specific stack. We walk from top (most-recent)
            # so a revert that targets a non-current entry still
            # un-masks the entry below it on the next walk.
            for idx in range(len(pair_stack) - 1, -1, -1):
                if pair_stack[idx].decision_id == record.revert_of:
                    pair_stack.pop(idx)
                    break
            # The revert record itself does NOT push onto the
            # stack — it's a reversal, not a new decision.
        else:
            # Regular decision: push.
            pair_stack.append(record)

    return {
        key: (stack[-1] if stack else None)
        for key, stack in stacks.items()
    }


__all__ = [
    "CellDecisionRecord",
    "DecisionAction",
    "current_decisions",
]
