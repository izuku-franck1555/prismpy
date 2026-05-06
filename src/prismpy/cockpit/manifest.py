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


# The pipeline's per-cell warning emit at
# ``executor.py:_category_for_check_id`` populates
# ``cell_summary.cells[i].failed_checks[j].category`` with one of
# six dimension-toggle category strings declared in
# ``executor.py::_CATEGORY_FROM_PREFIX``. Those strings are NOT
# members of :class:`WarningCategory` (which carries a
# tighter-semantics 10-member taxonomy used at the zone-level +
# Sprint F+ paths) — they're a parallel, looser vocabulary
# intended for the cockpit's left-rail dimension filter.
#
# Both vocabularies coexist intentionally: the dimension-toggle
# names are coarse UI buckets ("value_range" covers tmax-tmin
# inversions and unit-of-measure outliers under one umbrella);
# :class:`WarningCategory` enum values are precise causes
# ("climate_envelope_tail" vs "physiological_bound_violation").
# The cockpit consumes both — per-cell rows arrive via the
# dimension-toggle vocabulary, zone-level rows via the
# enum vocabulary.
#
# Each dimension-toggle category routes to bucket 3
# (TRUE_EXCLUDE) by default — every per-cell dimension-toggle
# category indicates the cell carries a defect that disqualifies
# it from analysis. Refinement to bucket 4 (INTERPOLATABLE) for
# short-gap variants of ``temporal`` / ``coverage_per_cell`` is
# Sprint E.2 scope; until then, the conservative default is to
# exclude (the no-data-cooking honest-signal posture per
# ``feedback_no_data_cooking.md`` — silent reclassification to
# a less-severe bucket would surface fewer warnings to the user
# at the cost of trust).
#
# A structural pin at
# ``tests/structural/test_cockpit_dimension_categories_pinned.py``
# AST-walks ``_CATEGORY_FROM_PREFIX`` + asserts every emitted
# category value is a key here, so a future executor edit that
# adds a new dimension-toggle category without updating this
# map fails loud at structural-test time rather than silently
# falling back to ``UnknownCategoryError`` on the first
# real-data project that emits the new category.
_DIMENSION_BUCKET_MAP: Dict[str, int] = {
    "value_range": 3,
    "cross_variable": 3,
    "temporal": 3,
    "soil_completeness": 3,
    "region_specific_bounds": 3,
    "coverage_per_cell": 3,
}


class UnknownCategoryError(ValueError):
    """A category emitted by the producer is not in
    :data:`WARNING_BUCKET_MAP` AND not in
    :data:`_DIMENSION_BUCKET_MAP`. Indicates substrate drift
    between Sprint E.0's enum declaration, the executor's
    dimension-toggle vocabulary at ``_CATEGORY_FROM_PREFIX``,
    and the validator emit; surfaces here at manifest-build
    time rather than silently dropping the entry from the
    cockpit."""


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

    Two vocabularies coexist (intentionally — see
    :data:`_DIMENSION_BUCKET_MAP` docstring for why):

    1. The dimension-toggle vocabulary at
       ``executor.py::_CATEGORY_FROM_PREFIX`` — six coarse UI
       categories the cockpit's left-rail dimension filter
       reads. Per-cell warnings carry these.
    2. :class:`WarningCategory` enum — the tighter
       Sprint E.0 taxonomy. Zone-level + Sprint F+ warnings
       carry these.

    Look up the dimension-toggle vocabulary first (it covers
    the per-cell pivot path that produces the bulk of cockpit
    rows on real projects — pre-fix, every per-cell warning
    from a fresh project tripped the
    :class:`UnknownCategoryError` path because the
    dimension-toggle category strings are not enum members,
    triggering the cockpit's pre-E.0 fallback banner on
    valid projects).

    Fall back to :class:`WarningCategory` for the zone-level
    path. Raise :class:`UnknownCategoryError` only when
    neither vocabulary recognizes the category — surfaces
    substrate drift at manifest-build time rather than
    silently dropping the entry.
    """
    if category_value in _DIMENSION_BUCKET_MAP:
        return _DIMENSION_BUCKET_MAP[category_value]
    try:
        category_enum = WarningCategory(category_value)
    except ValueError as exc:
        raise UnknownCategoryError(
            f"Category {category_value!r} is neither a "
            f"WarningCategory enum value nor a per-cell "
            f"dimension-toggle category. Reconcile with "
            f"prismpy.warnings.categories.WarningCategory "
            f"and prismpy.pipeline.executor."
            f"_CATEGORY_FROM_PREFIX (the structural pin at "
            f"tests/structural/"
            f"test_cockpit_dimension_categories_pinned.py "
            f"keeps the dimension vocabulary in sync with "
            f"_DIMENSION_BUCKET_MAP)."
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


def _cells_by_category(
    cell_summary: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Pivot the producer-shape cell summary into a
    {category_value: [cell_id, ...]} dict.

    Two input shapes accepted:

    1. **Producer shape** — the
       :func:`prismpy.pipeline.executor._build_cell_summary`
       output: ``{"cells": [{cell_id, failed_checks:
       [{check_id, category, ...}, ...]}, ...]}``. Walks each
       cell's ``failed_checks`` array; every check carries a
       ``category`` field per V2-22c-PRE.1.2 D34.
    2. **Pre-pivoted shape** — ``{category_value: {"cells":
       [...]}}`` (legacy / test fixture form). The fast path
       for callers that already have the per-category cell-id
       lists.

    Mixing the two shapes within one call is undefined; the
    builder picks ``"cells"`` top-level (producer shape) over
    a per-category dict by inspecting whether ``cell_summary``
    has a top-level ``cells`` list.
    """
    by_category: Dict[str, List[str]] = {}
    cells_field = (
        cell_summary.get("cells")
        if isinstance(cell_summary, dict)
        else None
    )
    if isinstance(cells_field, list):
        # Producer shape — per-cell failed_checks pivot. Each
        # cell may carry multiple failed_checks; surface each
        # category × cell pair into the by_category map.
        for cell in cells_field:
            if not isinstance(cell, dict):
                continue
            cell_id = cell.get("cell_id") or cell.get("id")
            if not cell_id:
                continue
            failed = cell.get("failed_checks") or []
            for check in failed:
                if not isinstance(check, dict):
                    continue
                category_value = check.get("category")
                if not category_value:
                    continue
                by_category.setdefault(
                    category_value, [],
                ).append(str(cell_id))
        return by_category

    # Pre-pivoted shape — used by tests + by callers who have
    # already grouped the cells by category.
    if isinstance(cell_summary, dict):
        for category_value, payload in cell_summary.items():
            if category_value == "cells":
                continue
            cells = (
                payload.get("cells")
                if isinstance(payload, dict)
                else None
            )
            if not cells:
                continue
            by_category.setdefault(
                category_value, [],
            ).extend(str(c) for c in cells)
    return by_category


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
        cell_summary: Optional dict from the producer at
            :func:`prismpy.pipeline.executor._build_cell_summary`,
            shape::

                {"cells": [{cell_id, failed_checks: [{check_id,
                            category, result}, ...]}, ...]}

            Helper :func:`_cells_by_category` pivots into a
            per-category map; legacy pre-pivoted
            ``{category_value: {"cells": [...]}}`` shape also
            accepted for fixture-driven tests.
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
        cells_by_category = _cells_by_category(cell_summary)
        for category_value, cell_ids in cells_by_category.items():
            if not cell_ids:
                continue
            # De-dupe cell_ids that appear in multiple
            # failed_checks for the same cell+category.
            sorted_cells = tuple(sorted(set(cell_ids)))
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
