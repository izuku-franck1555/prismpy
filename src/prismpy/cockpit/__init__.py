"""Cockpit-side helpers for Sprint E.1 (V2-22c-RESTART Phase 4).

The cockpit's flagged-warnings panel needs a deterministic
manifest of every entry the user can act on (acknowledge / skip
/ override) before launching the next pipeline run. This
package ships the canonical builder + dataclass shape so the
producer (pipeline finish in prismpy) and the consumer (cockpit
template + Alpine state in prismweb) read the same payload
across the cross-repo boundary.

Substrate-tier — the manifest is what the cockpit's per-bucket
grouping reads off; the wizard banner consumes the same
prismpy substrate at a different surface (per-zone Bucket 5
entries from ``Project.stage_1_verdicts``).
"""
from prismpy.cockpit.manifest import (
    CockpitManifestEntry,
    UnknownCategoryError,
    build_cockpit_warning_manifest,
    compute_manifest_hash,
)


__all__ = [
    "CockpitManifestEntry",
    "UnknownCategoryError",
    "build_cockpit_warning_manifest",
    "compute_manifest_hash",
]
