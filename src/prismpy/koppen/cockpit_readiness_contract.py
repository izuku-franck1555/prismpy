"""Sprint E.1 AC-E1-4 — cockpit-readiness schema reference.

The Stage 1 verdicts snapshot is written by prismweb's
:func:`build_stage_1_verdicts_snapshot` helper at wizard-time
and read by the cockpit (consumer) at run-time. The shape is
declared here so:

* Producer (prismweb wizard write-path) and consumer (prismweb
  cockpit read-path + cockpit-readiness contract test) read
  the same authoritative reference, not duplicated frozensets
  on each side.
* The cross-repo consumer test pin (Sprint F shipped at
  ``prismweb/tests/test_stage_1_cockpit_readiness_contract.py``)
  reads the reference here rather than copy-pasting the field
  set into every consumer test.

What's pinned:

* :data:`SCHEMA_VERSION_CURRENT` — the canonical
  ``stage_1_v1`` string. Bumping this requires co-evolving
  prismpy + prismweb (the cockpit reader rejects unknown
  schema versions); a future ``stage_1_v2`` ratchet lands
  here first.
* :data:`TOP_LEVEL_FIELDS` — every top-level field the
  snapshot dict carries. Drift on either side surfaces here.
* :data:`ENTRY_FIELDS` — every per-entry field a
  ``crop_region_mismatch`` Bucket 5 row carries. Sprint F
  Path β step 4 added ``zone_label`` (F-Path-β-1
  human-readable resolution) and ``explanation`` (the
  plain-language paragraph) and ``verbatim_source_url`` (the
  per-crop FAO ECOCROP datasheet) on top of Sprint F's
  initial 10-key set.

The reference is a substrate-tier authority — Sprint F's
prismweb-side snapshot builder + cockpit-readiness contract
test both read these constants rather than embedding the field
sets inline.
"""
from __future__ import annotations

from typing import FrozenSet


# Schema version pinning. The wizard banner + cockpit reader
# reject snapshots with a different ``schema_version`` string;
# a co-evolution sprint bumps this constant + ratchets prismweb
# to the new shape.
SCHEMA_VERSION_CURRENT: str = "stage_1_v1"


# Top-level keys the snapshot carries. Every cockpit consumer
# (drawer Bucket 5 panel + run-lineage rail + cockpit-readiness
# contract test) reads from this canonical reference.
TOP_LEVEL_FIELDS: FrozenSet[str] = frozenset({
    "schema_version",
    "cache_key",
    "region_cache_key",
    "crop",
    "created_at",
    "substrate_versions",
    "entries",
})


# Operational-failure snapshot extension (codex review #DIM-1
# from Sprint F). When the validator could not run cleanly,
# the snapshot carries these two extra keys; the cockpit reader
# distinguishes "ran cleanly with no Bucket 3" from "couldn't
# run" via the explicit unavailable flag.
UNAVAILABLE_TOP_LEVEL_FIELDS: FrozenSet[str] = frozenset({
    "stage_1_unavailable",
    "unavailable_reason",
})


# Per-entry fields on each ``crop_region_mismatch`` Bucket 5
# row. The first 10 are the Sprint F base set; the last 3 are
# the Path β step 4 + bucket-5 cockpit-readiness extension.
ENTRY_FIELDS: FrozenSet[str] = frozenset({
    # Sprint F base set (10) — pinned in
    # _STAGE_1_SNAPSHOT_FIXTURE at
    # tests/unit/test_stage_1_provenance.py:31.
    "category",
    "zone",
    "zone_label",
    "crop",
    "verdict",
    "reason",
    "n_cell_days_in_zone",
    "stage_1_verdict_id",
    "override_decision_id",
    "override_status",
    # Bucket 5 / Path β step 4 extension — required for the
    # wizard banner + cockpit drawer to render the persona-
    # facing UX (plain-language explanation + per-crop FAO
    # ECOCROP record link).
    "explanation",
    "verbatim_source_url",
    # Cockpit drawer audit trail — flat-list iteration of the
    # validator's per-zone details payload. Optional for
    # consumers that only need the top-level keys above.
    "details",
})


# Required per-entry fields the cockpit-readiness contract
# test pins. Subset of :data:`ENTRY_FIELDS`; the remaining
# fields are optional (e.g., ``stage_1_verdict_id`` is None
# until a verdict gets persisted to a future verdict-table).
REQUIRED_ENTRY_FIELDS: FrozenSet[str] = frozenset({
    "category",
    "zone",
    "zone_label",
    "crop",
    "verdict",
    "reason",
    "explanation",
    "verbatim_source_url",
})


# Per-substrate-version field set the snapshot's
# ``substrate_versions`` block carries. Drift on either side
# fails the contract test.
SUBSTRATE_VERSION_FIELDS: FrozenSet[str] = frozenset({
    "bounds_version",
    "zone_classifier_version",
    "ecocrop_envelope_version",
    "zone_aggregates_version",
})


__all__ = [
    "SCHEMA_VERSION_CURRENT",
    "TOP_LEVEL_FIELDS",
    "UNAVAILABLE_TOP_LEVEL_FIELDS",
    "ENTRY_FIELDS",
    "REQUIRED_ENTRY_FIELDS",
    "SUBSTRATE_VERSION_FIELDS",
]
