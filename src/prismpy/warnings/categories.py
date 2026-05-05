"""Canonical warning category enum + 5-bucket response map.

Defines the chokepoint that every prismweb consumer routes
through when classifying or rendering a cockpit warning.

Sprint E.0 (V2-22c-RESTART Phase 0) ships this foundation;
Sprint E.0.5 lights up the new INFORMATIONAL category
(``TRANSITIONAL_ZONE``, ``INSUFFICIENTLY_SAMPLED``,
``CLIMATE_ENVELOPE_TAIL``); Sprint F lights up
``CROP_REGION_MISMATCH`` (Bucket 5 — wizard-time documented
override) and reserves ``CROP_PHYSIOLOGY_VIOLATION`` for the
Stage 2 per-cell detector (Bucket 3); Sprint E.2 lights up
``SHORT_GAP_INTERPOLATABLE`` for Bucket 4 expansion;
``MANUAL_OVERRIDE`` is reserved for V3.

**Reservation policy** (per warning-auditor probe-1-A):

* **Currently in production**: ``SOIL_NO_HWSD_COVERAGE``,
  ``SOIL_TEXTURE_INVALID``, ``CLIMATE_RH_INVALID`` — emitted
  by Sprint D.1 producers since prismpy main `966e21f`.
* **Near-term (E.0.5)**: ``TRANSITIONAL_ZONE``,
  ``INSUFFICIENTLY_SAMPLED``, ``CLIMATE_ENVELOPE_TAIL`` —
  Bucket 2 INFORMATIONAL.
* **Sprint F (Bucket 5)**: ``CROP_REGION_MISMATCH`` —
  wizard-time documented-override path with rationale +
  evidence_type + verdict_hash.
* **Sprint F (Bucket 3, Stage 2 deferred)**:
  ``CROP_PHYSIOLOGY_VIOLATION`` — per-cell ECOCROP tolerance.
* **Near-term (E.2)**: ``SHORT_GAP_INTERPOLATABLE``.
* **Reserved-only (V3)**: ``MANUAL_OVERRIDE``.

The empty Bucket 1 (``AUTO_FIXABLE``) is intentional — Sprint
D.1 auto-fixes (e.g., texture renormalization within the 5%
deviation tolerance) are silent and provenance-only; they do
not surface to the cockpit. A future sprint scoping "auto-
fixes that surface to cockpit" will add the corresponding
enum value at that time per YAGNI.
"""
from __future__ import annotations

from enum import Enum
from typing import Tuple


class WarningCategory(str, Enum):
    """Canonical warning category strings (Pydantic-compatible
    StrEnum).

    The ``(str, Enum)`` base preserves Python 3.10 compatibility
    (``enum.StrEnum`` is 3.11+) AND grants string-equality
    semantics: ``WarningCategory.SOIL_NO_HWSD_COVERAGE ==
    "soil_no_hwsd_coverage"`` evaluates True. Sprint D.1's
    existing producers, which assign the bare string today,
    keep working unmodified — Sprint E.0's site migrations
    are about chokepoint discipline, not behavioral change.

    Adding a new value MUST be paired with a
    :data:`WARNING_BUCKET_MAP` entry; the
    ``test_no_orphans_no_duplicates`` regression catches a
    miss.
    """

    # ── Currently shipping (Sprint D.1) — Bucket 3 TRUE_EXCLUDE ──
    SOIL_NO_HWSD_COVERAGE = "soil_no_hwsd_coverage"
    SOIL_TEXTURE_INVALID = "soil_texture_invalid"
    CLIMATE_RH_INVALID = "climate_rh_invalid"

    # ── Reserved for Sprint E.0.5 — Bucket 2 INFORMATIONAL ──
    TRANSITIONAL_ZONE = "transitional_zone"
    INSUFFICIENTLY_SAMPLED = "insufficiently_sampled"
    CLIMATE_ENVELOPE_TAIL = "climate_envelope_tail"

    # ── Sprint F — Bucket 5 MANUAL_OVERRIDE_WITH_EVIDENCE ──
    # CROP_REGION_MISMATCH is a Stage 1 wizard-time emit that the
    # user can override with documented evidence (rationale +
    # evidence_type + verdict_hash) per Sprint F AC-F-6 +
    # AC-F-10. Promoted from Bucket 3 (TRUE_EXCLUDE) to Bucket 5
    # so the data + UI classification matches the implementation
    # behavior — the cockpit and wizard banner both offer the
    # documented-override flow, which is the Bucket 5 contract.
    CROP_REGION_MISMATCH = "crop_region_mismatch"
    # ── Reserved for Sprint F (V2-23) — Bucket 3 TRUE_EXCLUDE ──
    # Stage 2 per-cell ECOCROP tolerance violations stay in
    # Bucket 3; the cockpit cannot meaningfully accept an
    # override on every individual cell, only at the wizard /
    # zone level (Stage 1 → Bucket 5).
    CROP_PHYSIOLOGY_VIOLATION = "crop_physiology_violation"

    # ── Reserved for Sprint E.2 — Bucket 4 INTERPOLATABLE ──
    SHORT_GAP_INTERPOLATABLE = "short_gap_interpolatable"

    # ── Reserved for V3 — Bucket 5 MANUAL_OVERRIDE_WITH_EVIDENCE ──
    # The user-driven manual-override category beyond Stage 1
    # crop-region (which already uses Bucket 5 per Sprint F).
    MANUAL_OVERRIDE = "manual_override"


class WarningBucket(str, Enum):
    """The five cockpit response buckets a warning can land in.

    The bucket determines how the cockpit renders the warning
    and what user action (if any) it permits:

    * ``AUTO_FIXABLE`` — silent auto-fix, no cockpit surface.
      Reserved; currently empty.
    * ``INFORMATIONAL`` — show the warning; no action required.
    * ``TRUE_EXCLUDE`` — cell is excluded from the data
      package; user can review per-cell explanation.
    * ``INTERPOLATABLE`` — short data gap; gap-fill is
      offered as a remediation option.
    * ``MANUAL_OVERRIDE_WITH_EVIDENCE`` — user can override
      with documented justification (V3 only).
    """

    AUTO_FIXABLE = "auto_fixable"
    INFORMATIONAL = "informational"
    TRUE_EXCLUDE = "true_exclude"
    INTERPOLATABLE = "interpolatable"
    MANUAL_OVERRIDE_WITH_EVIDENCE = "manual_override_with_evidence"


# Every :class:`WarningCategory` value MUST appear here exactly
# once. The :func:`categories_in_bucket` helper returns
# deterministically sorted tuples so byte-identity serialization
# tests stay stable across hash-randomized Python sessions.
#
# Insertion order matches the declaration order in
# :class:`WarningCategory` so the
# ``test_bucket_map_ordering_matches_enum`` regression catches
# accidental reorderings during refactors.
WARNING_BUCKET_MAP: dict[WarningCategory, WarningBucket] = {
    # Sprint D.1 — TRUE_EXCLUDE for missing-substrate causes
    WarningCategory.SOIL_NO_HWSD_COVERAGE: WarningBucket.TRUE_EXCLUDE,
    WarningCategory.SOIL_TEXTURE_INVALID: WarningBucket.TRUE_EXCLUDE,
    WarningCategory.CLIMATE_RH_INVALID: WarningBucket.TRUE_EXCLUDE,
    # Sprint E.0.5 — INFORMATIONAL for legitimate climate
    # variability + sample-quality + zone-boundary signals
    WarningCategory.TRANSITIONAL_ZONE: WarningBucket.INFORMATIONAL,
    WarningCategory.INSUFFICIENTLY_SAMPLED: WarningBucket.INFORMATIONAL,
    WarningCategory.CLIMATE_ENVELOPE_TAIL: WarningBucket.INFORMATIONAL,
    # Sprint F — Stage 1 wizard-time crop-region mismatch is
    # MANUAL_OVERRIDE_WITH_EVIDENCE: the wizard offers a
    # documented-override path (rationale + evidence_type +
    # verdict_hash). Promoted from TRUE_EXCLUDE per ux-expert
    # verdict + honest-signal review (data + UI must match).
    WarningCategory.CROP_REGION_MISMATCH: WarningBucket.MANUAL_OVERRIDE_WITH_EVIDENCE,
    # Sprint F (V2-23) — per-cell crop-physiological violations
    # remain TRUE_EXCLUDE: the cockpit cannot meaningfully accept
    # an override on every individual cell.
    WarningCategory.CROP_PHYSIOLOGY_VIOLATION: WarningBucket.TRUE_EXCLUDE,
    # Sprint E.2 — INTERPOLATABLE for short data gaps
    WarningCategory.SHORT_GAP_INTERPOLATABLE: WarningBucket.INTERPOLATABLE,
    # V3 — MANUAL_OVERRIDE_WITH_EVIDENCE
    WarningCategory.MANUAL_OVERRIDE: WarningBucket.MANUAL_OVERRIDE_WITH_EVIDENCE,
}


def bucket_for(category: WarningCategory) -> WarningBucket:
    """Return the bucket assigned to ``category``.

    Raises :class:`KeyError` if the category has no bucket
    assignment — but ``test_no_orphans_no_duplicates`` rules
    that out at module import time, so production callers
    never see a KeyError on a known enum value.
    """
    return WARNING_BUCKET_MAP[category]


def categories_in_bucket(
    bucket: WarningBucket,
) -> Tuple[WarningCategory, ...]:
    """Return all categories assigned to ``bucket`` as a tuple.

    Sorted by the category's string value so the result is
    deterministic across Python sessions (Python's hash
    randomization perturbs dict iteration order in older
    versions; sorting here insulates downstream byte-identity
    serialization tests from that). Returns an empty tuple
    when no categories belong to the bucket — currently the
    case for :attr:`WarningBucket.AUTO_FIXABLE`.
    """
    return tuple(
        sorted(
            (cat for cat, b in WARNING_BUCKET_MAP.items() if b == bucket),
            key=lambda c: c.value,
        )
    )
