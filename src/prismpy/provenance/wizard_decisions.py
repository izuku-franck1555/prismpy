"""Wizard-time decision records (Sprint F AC-F-6).

The wizard-time override flow has a different lifecycle than
the in-memory :class:`prismpy.provenance.tracker.ProvenanceTracker`:

* The user declares an override BEFORE the pipeline runs (no
  tracker instance exists yet) — typically at the prismweb
  ``create_project`` view-handler when the Stage 1 verdict
  surfaces a Bucket 3 ``CROP_REGION_MISMATCH``.
* The decision is persisted (prismweb stores the canonical
  payload in ``Project.wizard_decisions`` JSONField).
* When the pipeline starts, the saved overrides are replayed
  into a fresh :class:`ProvenanceTracker` via
  :meth:`ProvenanceTracker.record_decision` with
  ``decision_type=DecisionType.USER_OVERRIDE``.

This module ships the prismpy side of that contract:

* :class:`WizardOverrideRecord` — frozen Pydantic model that
  validates the structured fields (rationale ≥50 chars,
  ``evidence_type`` Literal of 5 values, ``affected_zones``
  non-empty list, ``verdict_hash`` SHA-256 hex string for
  stale-override rejection per codex Gate A #13).
* :func:`build_wizard_override_payload` — converts a record
  to the JSON-serializable dict prismweb persists.
* :func:`compute_verdict_hash` — canonical hashing of a
  Stage 1 verdict snapshot so a future cache-key recompute
  detects an override that was applied to a now-stale
  verdict.

prismweb wires:

* Forms validate via :class:`WizardOverrideRecord`.
* Persists serialized dict in ``Project.wizard_decisions``.
* At pipeline start, deserializes + calls
  ``tracker.record_decision`` per saved override.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Sprint F AC-F-6 + warning-auditor LOW-4: rationale floor
# bumped from 20 chars (Draft 1) to 50 chars (Draft 2) to
# reject obviously-trivial overrides ("ok", "yes please",
# "test"). The 50-char floor still admits a one-sentence
# justification while excluding hand-wavers.
_MIN_RATIONALE_CHARS: int = 50

# Banned filler patterns — repeated single chars (e.g.,
# "aaaaaaaaaa..." or "..............") would clear the
# 50-char floor without carrying real content. Reject them
# heuristically per warning-auditor LOW-4.
_FILLER_PATTERN_PARAMETERS = (
    # (description, predicate-on-rationale)
    (
        "single-char repeats",
        lambda r: len(set(r.strip())) <= 2,
    ),
)


# Sprint F AC-F-6 + codex Gate A #15: evidence_type Literal
# fixed at five categories. Future expansion lands as
# additive enum-extension under V2-23 polish.
EvidenceType = Literal[
    "local_trial",
    "irrigation",
    "cultivar_specific",
    "citation",
    "field_observation",
]


class WizardOverrideRecord(BaseModel):
    """Canonical structured payload for a wizard-time
    Bucket 3 override decision.

    Pydantic-frozen + extra-forbid so a typo'd field on the
    prismweb side fails loud at form-submit time rather than
    drifting through to provenance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_type: Literal["user_override"] = Field(
        default="user_override",
        description=(
            "Pinned to the canonical USER_OVERRIDE value so "
            "the field round-trips into "
            "``DecisionType.USER_OVERRIDE`` at pipeline start."
        ),
    )

    rationale: str = Field(
        ..., min_length=_MIN_RATIONALE_CHARS,
        description=(
            "User-provided text explaining the override. Must "
            "be at least 50 characters per AC-F-6 + warning-"
            "auditor LOW-4 (filler-rejection heuristic blocks "
            "single-char repeats that would clear the length "
            "floor without carrying real content)."
        ),
    )

    evidence_type: EvidenceType = Field(
        ...,
        description=(
            "Categorical evidence basis per AC-F-6 + codex "
            "Gate A #15: local_trial / irrigation / "
            "cultivar_specific / citation / field_observation. "
            "Pins the evidence shape so the cockpit drawer + "
            "audit log can render per-category copy."
        ),
    )

    affected_zones: List[str] = Field(
        ..., min_length=1,
        description=(
            "Non-empty list of KG zone codes that the "
            "override applies to. AC-F-10's per-zone "
            "checkbox UI populates this list from the "
            "user's selection; compatible zones in the "
            "region don't appear here."
        ),
    )

    verdict_hash: str = Field(
        ..., min_length=64, max_length=64,
        description=(
            "SHA-256 hex of the Stage 1 verdict snapshot at "
            "override time. AC-F-5 cache-key recompute "
            "compares the current verdict hash against this "
            "value to detect overrides that reference a "
            "stale substrate version per codex Gate A #13."
        ),
    )

    evidence_url: Optional[str] = Field(
        default=None,
        description=(
            "Optional citation URL (per AC-F-6 + evaluator "
            "#6). When provided, must be https://."
        ),
    )

    methodology_paper_doi: Optional[str] = Field(
        default=None,
        description=(
            "Optional DOI for a methodology paper supporting "
            "the override (per AC-F-6 + evaluator #6)."
        ),
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description=(
            "UTC timestamp at override-record creation. "
            "Pinned at construction so the saved payload "
            "carries deterministic ordering for the audit "
            "log."
        ),
    )

    @field_validator("rationale")
    @classmethod
    def _reject_filler(cls, value: str) -> str:
        for description, predicate in _FILLER_PATTERN_PARAMETERS:
            if predicate(value):
                raise ValueError(
                    f"Rationale rejected as filler "
                    f"({description}); please describe the "
                    f"specific basis for the override."
                )
        return value

    @field_validator("verdict_hash")
    @classmethod
    def _verdict_hash_is_hex(cls, value: str) -> str:
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(
                "verdict_hash must be a 64-char SHA-256 hex "
                "string."
            ) from exc
        return value

    @field_validator("evidence_url")
    @classmethod
    def _evidence_url_https(
        cls, value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None
        if not value.startswith("https://"):
            raise ValueError(
                "evidence_url must use https://; got "
                f"{value!r}."
            )
        return value


def build_wizard_override_payload(
    record: WizardOverrideRecord,
) -> Dict[str, Any]:
    """Convert a :class:`WizardOverrideRecord` into the
    JSON-serializable dict prismweb persists in
    ``Project.wizard_decisions``.

    The dict is what the pipeline-start replay reads via
    :meth:`ProvenanceTracker.record_decision` to thread the
    override into the run's provenance trail.
    """
    payload = record.model_dump(mode="json")
    # Pydantic's JSON-mode dump produces a string for the
    # datetime; we leave it that way so the JSONField round-
    # trips cleanly without an additional serializer.
    return payload


def compute_verdict_hash(
    stage_1_verdicts: Dict[str, Any],
) -> str:
    """Compute a canonical SHA-256 hex over a Stage 1 verdict
    snapshot for stale-override rejection.

    The hash is over a sorted-key JSON dump so two semantically
    equal snapshots produce the same hash regardless of dict
    insertion order. Floats are formatted via JSON's default
    repr.

    Args:
        stage_1_verdicts: The dict shape from AC-F-5 cache:
            ``{"schema_version": ..., "cache_key": ...,
              "entries": [...]}``. Pass the FULL cached snap-
            shot, not just the entries list, so a schema or
            cache-key change also forces a hash mismatch.

    Returns:
        64-char SHA-256 hex string.
    """
    canonical = json.dumps(
        stage_1_verdicts, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
