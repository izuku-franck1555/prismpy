"""Canonical TypedDict for the per-cell failure context.

Sprint E.2 AC-E2-3 ext + Codex round 1 LOW finding (Builder
Sub-CA #1) absorption — replaces the bare ``Dict[str, Any]``
boundary at the entry points of
:func:`prismpy.cockpit.routing_decision.bucket_for` +
:func:`prismpy.validators.affordance_routing.route_affordance`
with a single canonical :class:`typing.TypedDict` so type
checkers + structural review can spot key drift instead of
silently letting an unknown key flow through.

Per durable §24 canonical-source-or-pin: this is the single
canonical home for the per-cell failure context schema.
Every callsite that constructs the dict OR types its
parameter as the context schema imports
:class:`CellFailureContext` from this module — no parallel
TypedDict declarations across producer / consumer call
boundaries.

Per durable §27 two-vocabulary substrate-drift: this Python
TypedDict is the producer-side boundary; cross-language
consumers (the future prismweb cockpit JS getter at Phase 2)
read the same key vocabulary via runtime serialization +
the WA CA-8 vocab parity pin. Until Phase 2 ships, the
TypedDict catches Python-side drift; a future symmetric
cross-language pin closes the JS-side gap.
"""
from __future__ import annotations

from typing import Optional, TypedDict


class CellFailureContext(TypedDict, total=False):
    """Per-cell metric dict the cockpit routing engine reads
    alongside ``check_id`` + ``routed_affordance`` to compute
    the canonical :class:`prismpy.cockpit.routing_decision.RoutingDecision`
    triple.

    All keys are optional (``total=False``). The empty dict
    ``{}`` is the canonical "no per-cell context" sentinel and
    routes the cell through the conservative-fallback path
    in :func:`bucket_for`. Each individual key carries one
    routing signal:

    * ``gap_count``: temporal-completeness gap-day count.
      Branch threshold is
      :data:`prismpy.cockpit.bucket_thresholds.TEMPORAL_GAP_BUCKET_4_MAX_DAYS`
      (≤14 → bucket 4 INTERPOLATABLE; >14 → bucket 3 TRUE_EXCLUDE).
    * ``coverage_pct``: per-cell coverage percentage for
      coverage_per_cell failures. Threshold
      :data:`prismpy.cockpit.bucket_thresholds.COVERAGE_PER_CELL_BUCKET_4_MIN_PCT`
      (≥80 → bucket 4 + ``interpolate``; <80 → bucket 3 +
      ``rerun_full_sources``).
    * ``layer_idx``: failing soil layer index for soil-layered
      diagnostic-variant dispatch. ``None`` routes to the
      cell-level-scalar variant.
    * ``daily_failure_count``: per-day climate-range failure
      count. ``> 0`` routes to the climate-dual-scale
      variant; otherwise cell-level-scalar.
    * ``has_existing_override``: cell carries a Bucket-5
      researcher-documented override. ``True`` routes to the
      documented-override variant short-circuit.
    * ``is_highland_precip``: cell falls in a highland zone
      with elevation above the orographic-exclusion
      threshold (``Cwa`` zone × elevation > 1500m per
      Decision 2 caveat 2). Routes ``value_range_precip``
      cells to the highland-excluded variant.
    * ``profile_depth_m``: soil profile depth in meters
      (NOT cm). Below
      :data:`prismpy.cockpit.bucket_thresholds.PROFILE_DEPTH_BUCKET_3_MIN_M`
      keeps soil_profile_depth checks at bucket 3
      unconditionally (profile depth varies physically; not
      interpolable).

    Unknown keys flow through silently — TypedDict's
    ``total=False`` allows extra keys at runtime; a future
    routing-engine extension can read them without changing
    the schema. Adding a new RECOGNIZED key requires editing
    THIS module + the consuming routing branch in lockstep
    so the type checker tracks the addition.
    """

    gap_count: int
    coverage_pct: float
    layer_idx: Optional[int]
    daily_failure_count: int
    has_existing_override: bool
    is_highland_precip: bool
    profile_depth_m: Optional[float]


__all__ = [
    "CellFailureContext",
]
