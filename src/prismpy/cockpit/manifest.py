"""Cockpit warning manifest builder (Sprint E.1 AC-E1-1).

The cockpit groups flagged warnings under three buckets per
:data:`prismpy.warnings.WARNING_BUCKET_MAP` semantics:

* Bucket 2 INFORMATIONAL — "Acknowledge"
* Bucket 3 TRUE_EXCLUDE — "Skip from analysis"
* Bucket 5 MANUAL_OVERRIDE_WITH_EVIDENCE — "Document and continue"

Per-AC discipline:

* :class:`CockpitManifestEntry` dataclass ships the canonical
  shape both producer (pipeline finish) and consumer (cockpit
  template) read off. Frozen + slotted so a typo'd field on
  either side fails-loud at attribute-access time.
* :func:`build_cockpit_warning_manifest` is the single entry-
  point. Bridges the per-cell summary the validator emits +
  the per-zone Stage 1 verdict snapshot Sprint F shipped into
  one ordered list of manifest entries. Each entry carries a
  stable ``entry_id`` (deterministic over
  ``parent_run_id + bucket + category + sorted(affected)``)
  so the cockpit's match-disjoint set-cover algorithm at
  AC-E1-7 can dedupe entries across re-renders without
  drifting on every read.
* :func:`compute_manifest_hash` produces the byte-stable JSON
  roundtrip hash the cockpit uses to detect drift between
  manifest-as-rendered and manifest-as-launched (per the
  per-run snapshot helper at AC-E1-2).

The manifest is read-only — the cockpit consumes it; user
decisions (acknowledge / skip / override) write into the
provenance tracker via :meth:`record_cockpit_decision` per
AC-E1-0; the manifest itself never mutates per CC-30.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from prismpy.warnings.categories import (
    WARNING_BUCKET_MAP,
    WarningBucket,
    WarningCategory,
)


# Cockpit's three blocking-decision buckets — Bucket 2
# INFORMATIONAL is non-blocking but rendered for acknowledge.
# Bucket 0 AUTO_FIXABLE is silent (no cockpit row); Bucket 4
# INTERPOLATABLE is deferred to Sprint E.2 per OOS.
_BUCKET_INTEGER_MAP: Dict[WarningBucket, int] = {
    WarningBucket.AUTO_FIXABLE: 0,
    WarningBucket.INFORMATIONAL: 2,
    WarningBucket.TRUE_EXCLUDE: 3,
    WarningBucket.INTERPOLATABLE: 4,
    WarningBucket.MANUAL_OVERRIDE_WITH_EVIDENCE: 5,
}


class UnknownCategoryError(ValueError):
    """A category emitted by the producer is not in
    :data:`WARNING_BUCKET_MAP`. Indicates substrate drift
    between Sprint E.0's enum declaration and the validator
    emit; surfaces here at manifest-build time rather than
    silently dropping the entry from the cockpit."""


@dataclass(frozen=True)
class CockpitManifestEntry:
    """One row in the cockpit's flagged-warnings manifest.

    Frozen so a downstream consumer cannot mutate the payload
    in-place; sorted-iteration over the manifest stays stable.

    Attributes:
        entry_id: Deterministic hash over
            ``parent_run_id + bucket + category + sorted(affected_cells)
            + sorted(affected_zones)``. Stable across re-renders so
            the match-disjoint set-cover at AC-E1-7 dedupes
            cleanly. Hex-encoded SHA-256.
        bucket: Integer 0/2/3/4/5 per :data:`_BUCKET_INTEGER_MAP`.
            Cockpit template's per-bucket grouping reads this.
        category: Lowercase :class:`WarningCategory` enum value
            (e.g., ``"climate_envelope_tail"``,
            ``"crop_region_mismatch"``).
        affected_cells: Sorted list of cell ids (per-cell
            warnings, Bucket 2 / Bucket 3 from pipeline finish).
        affected_zones: Sorted list of KG zone codes (per-zone
            wizard-time warnings, Bucket 5 from Sprint F's
            ``Project.stage_1_verdicts.entries[].zone``).
        count: Cardinality — for cell-level entries this is
            ``len(affected_cells)``; for zone-level entries
            this is ``len(affected_zones)`` so the cockpit's
            "{count} cells" / "{count} zones" rendering picks
            the right denominator.
    """
    entry_id: str
    bucket: int
    category: str
    affected_cells: Tuple[str, ...] = field(default_factory=tuple)
    affected_zones: Tuple[str, ...] = field(default_factory=tuple)
    count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable projection used by
        :func:`compute_manifest_hash` + persistence."""
        return {
            "entry_id": self.entry_id,
            "bucket": self.bucket,
            "category": self.category,
            "affected_cells": list(self.affected_cells),
            "affected_zones": list(self.affected_zones),
            "count": self.count,
        }


def _category_to_bucket_integer(category_value: str) -> int:
    """Resolve the bucket integer for a category value.

    Raises :class:`UnknownCategoryError` when the category
    is not declared in :data:`WARNING_BUCKET_MAP`. Caller
    decides whether to skip the entry or fail the build.
    """
    try:
        category_enum = WarningCategory(category_value)
    except ValueError as exc:
        raise UnknownCategoryError(
            f"Category {category_value!r} is not a "
            f"WarningCategory enum value; the producer side "
            f"emitted an unrecognized warning. Reconcile with "
            f"prismpy.warnings.categories.WarningCategory + "
            f"WARNING_BUCKET_MAP."
        ) from exc
    bucket = WARNING_BUCKET_MAP.get(category_enum)
    if bucket is None:  # pragma: no cover — guarded by enum
        raise UnknownCategoryError(
            f"Category {category_value!r} is in WarningCategory "
            f"but missing from WARNING_BUCKET_MAP."
        )
    return _BUCKET_INTEGER_MAP[bucket]


def _entry_id(
    parent_run_id: str,
    bucket: int,
    category: str,
    affected_cells: Tuple[str, ...],
    affected_zones: Tuple[str, ...],
) -> str:
    """Compute the deterministic entry_id over the join key.

    The hash binds parent_run_id (so a refinement run's
    manifest carries fresh ids) + bucket + category + the
    sorted affected_* tuples. The cockpit's set-cover
    algorithm reads this to dedupe entries across re-renders;
    a hash collision would only trigger on byte-identical
    inputs, in which case dedupe is the correct behavior.
    """
    payload = json.dumps(
        {
            "parent_run_id": parent_run_id,
            "bucket": bucket,
            "category": category,
            "affected_cells": list(affected_cells),
            "affected_zones": list(affected_zones),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_cockpit_warning_manifest(
    parent_run_id: str,
    cell_summary: Optional[Dict[str, Any]] = None,
    stage_1_verdicts: Optional[Dict[str, Any]] = None,
) -> List[CockpitManifestEntry]:
    """Build the canonical cockpit warning manifest.

    Aggregates per-cell flagged warnings (Bucket 2 / 3 from
    pipeline finish) + per-zone Stage 1 verdict entries
    (Bucket 5 from Sprint F wizard-time emit) into one ordered
    list the cockpit groups for display.

    Args:
        parent_run_id: The :class:`PipelineRun.id` whose
            warnings are being surfaced. Bound into every
            entry's ``entry_id`` so a refinement run's manifest
            carries fresh ids without drifting against the
            parent's.
        cell_summary: Optional dict with per-category cell-id
            lists, shape::

                {"category_value": {"cells": ["c1", ...]}}

            One bucket per category per cockpit's match-
            disjoint discipline; multi-cell entries collapse
            to one manifest row with the full ``affected_cells``
            tuple.
        stage_1_verdicts: Optional Sprint F snapshot dict per
            :func:`prismweb.core.services.wizard_validation
            .build_stage_1_verdicts_snapshot`. The
            ``entries[]`` list provides per-zone Bucket 5
            CROP_REGION_MISMATCH rows. The cockpit groups by
            (category, sorted_zones); a multi-zone Bucket 5
            entry collapses to one manifest row with the full
            ``affected_zones`` tuple.

    Returns:
        Sorted list of :class:`CockpitManifestEntry` ordered by
        ``(bucket, category, entry_id)`` so the cockpit's
        per-bucket grouping renders in deterministic order
        across re-loads + the byte-stable JSON roundtrip
        (per AC-E1-2 manifest_hash) holds.

    Raises:
        UnknownCategoryError: When the producer side emitted a
            category not in :class:`WarningCategory` /
            :data:`WARNING_BUCKET_MAP`. Per F25 + AC-E1-1
            substrate-drift discipline.
    """
    entries: List[CockpitManifestEntry] = []
    if cell_summary:
        for category_value, payload in cell_summary.items():
            cells = payload.get("cells") if isinstance(payload, dict) else None
            if not cells:
                continue
            sorted_cells = tuple(sorted(str(c) for c in cells))
            bucket = _category_to_bucket_integer(category_value)
            entry_id = _entry_id(
                parent_run_id, bucket, category_value,
                sorted_cells, tuple(),
            )
            entries.append(CockpitManifestEntry(
                entry_id=entry_id,
                bucket=bucket,
                category=category_value,
                affected_cells=sorted_cells,
                affected_zones=tuple(),
                count=len(sorted_cells),
            ))

    if stage_1_verdicts:
        # Group Stage 1 entries by category. Sprint F's snapshot
        # carries one entry per zone with the same category;
        # the cockpit collapses to one row with the full zone
        # tuple. Per CC-30 the manifest never re-derives —
        # consumes the producer-side snapshot verbatim.
        zones_by_category: Dict[str, List[str]] = {}
        for entry in stage_1_verdicts.get("entries", []) or []:
            if not isinstance(entry, dict):
                continue
            cat = entry.get("category")
            zone = entry.get("zone")
            if not cat or not zone:
                continue
            zones_by_category.setdefault(cat, []).append(str(zone))
        for category_value, zone_list in zones_by_category.items():
            sorted_zones = tuple(sorted(set(zone_list)))
            if not sorted_zones:
                continue
            bucket = _category_to_bucket_integer(category_value)
            entry_id = _entry_id(
                parent_run_id, bucket, category_value,
                tuple(), sorted_zones,
            )
            entries.append(CockpitManifestEntry(
                entry_id=entry_id,
                bucket=bucket,
                category=category_value,
                affected_cells=tuple(),
                affected_zones=sorted_zones,
                count=len(sorted_zones),
            ))

    entries.sort(key=lambda e: (e.bucket, e.category, e.entry_id))
    return entries


def compute_manifest_hash(
    entries: List[CockpitManifestEntry],
) -> str:
    """Compute the byte-stable manifest hash for drift
    detection between manifest-rendered and manifest-launched.

    JSON-roundtrip with sorted keys + canonical separators so
    the hash is stable across hash-randomized Python sessions
    + serialization round-trips through the prismweb
    :class:`PipelineRun.cell_validation_status` JSONField.
    """
    payload = json.dumps(
        [entry.to_dict() for entry in entries],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
