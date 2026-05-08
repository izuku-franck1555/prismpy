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
from prismpy.cockpit.bucket_thresholds import (
    COVERAGE_PER_CELL_BUCKET_4_MIN_PCT,
    PROFILE_DEPTH_BUCKET_3_MIN_M,
    TEMPORAL_GAP_BUCKET_4_MAX_DAYS,
)
from prismpy.cockpit.cell_failure_context import (
    CellFailureContext,
)
from prismpy.cockpit.check_id_enumeration import (
    POST_TRANSLATE_CHECK_IDS,
    VALIDATOR_CHECK_IDS,
    VALUE_RANGE_PREFIX_FAMILIES,
    enumerate_emitted_check_ids,
    matches_known_prefix,
)
from prismpy.cockpit.diagnostic_variant import (
    DIAGNOSTIC_VARIANT_VALUES,
    DiagnosticVariant,
)
from prismpy.cockpit.manifest import (
    CockpitManifestEntry,
    UnknownCategoryError,
    build_cockpit_warning_manifest,
    compute_manifest_hash,
)
from prismpy.cockpit.observed_values_writer import (
    AGGREGATION_METHOD,
    AGGREGATION_UNITS,
    OBSERVED_VALUES_CLIMATE_KEYS,
    OBSERVED_VALUES_SOIL_KEYS,
    SCHEMA_VERSION as OBSERVED_VALUES_SCHEMA_VERSION,
    SOIL_AGGREGATION_EGHR_SKIP,
    SOIL_AGGREGATION_IN_MEMORY,
    compute_climate_aggregates,
    compute_soil_aggregates,
    write_observed_values_json,
)
from prismpy.cockpit.per_run_snapshot import (
    compute_snapshot_at_launch_hash,
    is_snapshot_unavailable,
    snapshot_for_pipeline_run,
)
from prismpy.cockpit.routing_decision import (
    RoutingDecision,
    bucket_for,
)


__all__ = [
    # Sprint E.1 — manifest builder + per-run snapshot.
    "CockpitManifestEntry",
    "UnknownCategoryError",
    "build_cockpit_warning_manifest",
    "compute_manifest_hash",
    "compute_snapshot_at_launch_hash",
    "is_snapshot_unavailable",
    "snapshot_for_pipeline_run",
    # Sprint E.2 — per-cell routing canonical triple +
    # vocabulary canonical sources (Draft 6.2 absorptions).
    "COVERAGE_PER_CELL_BUCKET_4_MIN_PCT",
    "CellFailureContext",
    "DIAGNOSTIC_VARIANT_VALUES",
    "DiagnosticVariant",
    "POST_TRANSLATE_CHECK_IDS",
    "PROFILE_DEPTH_BUCKET_3_MIN_M",
    "RoutingDecision",
    "TEMPORAL_GAP_BUCKET_4_MAX_DAYS",
    "VALIDATOR_CHECK_IDS",
    "VALUE_RANGE_PREFIX_FAMILIES",
    "bucket_for",
    "enumerate_emitted_check_ids",
    "matches_known_prefix",
    # Sprint E.2 AC-E2-28 — HARMONIZE-stage cockpit observed-
    # values writer + canonical 17-key Hybrid A schema.
    "AGGREGATION_METHOD",
    "AGGREGATION_UNITS",
    "OBSERVED_VALUES_CLIMATE_KEYS",
    "OBSERVED_VALUES_SCHEMA_VERSION",
    "OBSERVED_VALUES_SOIL_KEYS",
    "SOIL_AGGREGATION_EGHR_SKIP",
    "SOIL_AGGREGATION_IN_MEMORY",
    "compute_climate_aggregates",
    "compute_soil_aggregates",
    "write_observed_values_json",
]
