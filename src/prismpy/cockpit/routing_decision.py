"""Per-cell cockpit routing — atomic ``(bucket, affordance,
diagnostic_variant)`` triple.

Sprint E.2 AC-E2-3 ext + Codex Gate A MEDIUM A1 + Builder
Sub-CA #6 + Codex Gate A MEDIUM A4/B1 (DiagnosticVariant
Literal canonical home).

The cockpit's per-cell routing was per-category in Sprint E.1
(:data:`prismpy.cockpit.manifest._DIMENSION_BUCKET_MAP`):
every dimension-toggle category mapped to bucket 3
TRUE_EXCLUDE unconditionally. AC-E2-3 ext refines this to
PER-CELL routing — the same category can route to bucket 4
INTERPOLATABLE for short-gap variants (e.g.,
``temporal_completeness`` with ``gap_count <= 14``) or stay
at bucket 3 for long-gap variants.

The atomic triple :class:`RoutingDecision` is the single
output shape per cell — bucket-int + affordance-string +
diagnostic-variant-string. Three consumers read this triple:

* The manifest builder (``manifest.py::build_manifest``)
  reads ``bucket`` for the per-bucket grouping the cockpit's
  AT A GLANCE counters render.
* The cockpit affordance UI reads ``affordance`` to decide
  which CTA to render per cell ("Skip from analysis" /
  "Apply interpolation" / "Acknowledge" / "Document
  override").
* The cell-detail drawer reads ``diagnostic_variant`` to
  dispatch to the right State C″ template (variant A
  cell-level-scalar / variant B climate-dual-scale / variant
  C soil-layered).

Per durable §24 canonical-source-or-pin: returning the
atomic triple from ONE function eliminates the producer-
consumer drift class where bucket + affordance + variant
each got computed at a different callsite with subtly
different inputs (Codex MEDIUM A1 + Builder Sub-CA #6).

Per durable §27 two-vocabulary substrate-drift: this module
+ :mod:`prismpy.cockpit.diagnostic_variant` + the JS
``cVariant`` getter at
``static/js/cockpit-redesign/cockpit-state.js`` form a
producer-consumer triangle. Structural pins enforce that
every value the producer can emit is handled by every
consumer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from prismpy.cockpit.bucket_thresholds import (
    COVERAGE_PER_CELL_BUCKET_4_MIN_PCT,
    PROFILE_DEPTH_BUCKET_3_MIN_M,
    TEMPORAL_GAP_BUCKET_4_MAX_DAYS,
)
from prismpy.cockpit.diagnostic_variant import DiagnosticVariant
from prismpy.validators.affordance_routing import AffordanceType
from prismpy.warnings.categories import WarningCategory


# ── Atomic triple ────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoutingDecision:
    """Per-cell cockpit routing triple — bucket + affordance +
    diagnostic variant.

    Frozen so a downstream consumer can't mutate the triple in-
    place (every callsite reads the canonical decision the
    routing engine produced); slotted via the dataclass default
    on Python 3.10+ for attribute-typo failure-loud behaviour.

    Attributes:
        bucket: Integer bucket assignment per
            :data:`prismpy.cockpit.manifest._BUCKET_INTEGER_MAP`
            (0 = AUTO_FIXABLE / 2 = INFORMATIONAL / 3 =
            TRUE_EXCLUDE / 4 = INTERPOLATABLE / 5 =
            MANUAL_OVERRIDE_WITH_EVIDENCE).
        affordance: The cockpit affordance the cell should
            surface — one of :data:`AffordanceType` Literal
            members. Re-emitted from :func:`route_affordance`'s
            output so the triple carries the per-cell routing
            atomically.
        diagnostic_variant: The State C drawer variant the
            cell-detail surface should render — one of
            :data:`DiagnosticVariant` Literal members.
    """

    bucket: int
    affordance: AffordanceType
    diagnostic_variant: DiagnosticVariant


# ── Routing engine ──────────────────────────────────────────────────


def bucket_for(
    check_id: str,
    cell_failure_context: Dict[str, Any],
    routed_affordance: AffordanceType,
) -> RoutingDecision:
    """Compute the per-cell ``RoutingDecision`` atomic triple.

    The bucket assignment is per-cell-aware: short-gap variants
    of ``temporal_completeness`` + ``coverage_per_cell`` route
    to bucket 4 INTERPOLATABLE when their per-cell metric is
    inside the canonical thresholds (per
    :mod:`prismpy.cockpit.bucket_thresholds`); long-gap
    variants stay at bucket 3 TRUE_EXCLUDE.

    The diagnostic variant is determined by validator emit
    shape (cell-level scalar vs climate dual-scale vs soil
    layered) plus zone signals (highland-excluded) plus
    decision-log signals (documented-override).

    Args:
        check_id: The validator-emitted check identifier (e.g.
            ``"value_range_tmax"`` / ``"temporal_completeness"``
            / ``"value_range_soil_clay"`` / ``"crop_region_mismatch"``).
        cell_failure_context: Per-cell metric dict carrying the
            signals needed for refined bucket routing + variant
            classification. Recognized keys:

            * ``gap_count``: int — gap-day count for
              temporal_completeness failures.
            * ``coverage_pct``: float — per-cell coverage
              percentage for coverage_per_cell failures.
            * ``layer_idx``: int | None — failing soil layer
              index for soil-layered diagnostic-variant
              dispatch.
            * ``daily_failure_count``: int — per-day failure
              count for climate value-range checks (>0 routes
              to climate-dual-scale variant).
            * ``has_existing_override``: bool — cell carries
              a Bucket-5 documented override.
            * ``is_highland_precip``: bool — cell falls in a
              highland zone with elevation > threshold.
            * ``profile_depth_m``: float | None — soil profile
              depth (for soil_profile_depth check_id; below
              :data:`PROFILE_DEPTH_BUCKET_3_MIN_M` stays
              bucket 3 unconditionally).

            Unknown keys are ignored; missing keys default to
            no-refinement (cell stays in the conservative
            bucket per Sprint E.1 contract). Empty dict +
            unknown ``check_id`` falls through to the default
            bucket-2 ACKNOWLEDGE path.
        routed_affordance: The affordance returned by
            :func:`prismpy.validators.affordance_routing.route_affordance`
            for this cell. Passed through to the triple so the
            three consumers (manifest / UI / drawer) read the
            same atomic decision.

    Returns:
        :class:`RoutingDecision` atomic triple.
    """
    ctx = cell_failure_context or {}

    # 1) Documented-override cells (Bucket-5) — wizard-time
    #    override decision already on file. Renders State C‴.
    if ctx.get("has_existing_override") is True:
        return RoutingDecision(
            bucket=5,
            affordance=routed_affordance,
            diagnostic_variant="documented-override",
        )

    # 2) Crop-region mismatch — wizard-time documented-override
    #    territory per ``route_affordance``. Bucket 5 surfaces
    #    the existing override (if any) OR awaits user evidence.
    #    Routes through :class:`WarningCategory` per F25 pin
    #    (no bare WarningCategory strings outside the enum
    #    module; mirrors ``route_affordance`` at
    #    ``affordance_routing.py:197``).
    if check_id == WarningCategory.CROP_REGION_MISMATCH.value:
        return RoutingDecision(
            bucket=5,
            affordance=routed_affordance,
            diagnostic_variant="documented-override",
        )

    # 3) Highland-precip orographic exclusion (Decision 2
    #    caveat 2). Bucket 3; renders State C′.
    if check_id == "value_range_precip" and ctx.get("is_highland_precip") is True:
        return RoutingDecision(
            bucket=3,
            affordance=routed_affordance,
            diagnostic_variant="highland-excluded",
        )

    # 4) Soil profile depth — physical constraint, not
    #    interpolable; stays bucket 3 unconditionally below
    #    DSSAT minimum. Renders State C″ variant A.
    if check_id == "soil_profile_depth":
        depth_m = ctx.get("profile_depth_m")
        if depth_m is not None and depth_m < PROFILE_DEPTH_BUCKET_3_MIN_M:
            return RoutingDecision(
                bucket=3,
                affordance=routed_affordance,
                diagnostic_variant="cell-level-scalar",
            )

    # 5) Soil value-range checks — never interpolable per
    #    ``route_affordance`` (soil profile gaps don't follow
    #    climate gradients). Bucket 3.
    #
    #    Variant: layer_idx present → soil-layered (renders C″
    #    Variant C with per-layer + rootzone-context); absent
    #    → cell-level-scalar.
    if check_id.startswith("value_range_soil_"):
        layer_idx = ctx.get("layer_idx")
        variant: DiagnosticVariant
        if layer_idx is not None:
            variant = "soil-layered"
        else:
            variant = "cell-level-scalar"
        return RoutingDecision(
            bucket=3,
            affordance=routed_affordance,
            diagnostic_variant=variant,
        )

    # 6) Climate value-range — bucket 4 INTERPOLATABLE when the
    #    affordance routed to interpolate; bucket 3 otherwise
    #    (e.g., 0-neighbour fallback). Renders State C″ variant
    #    B (climate-dual-scale) when daily failures are present.
    if check_id.startswith("value_range_") and not check_id.startswith(
        "value_range_soil_"
    ):
        bucket = 4 if routed_affordance == "interpolate" else 3
        daily_failure_count = ctx.get("daily_failure_count", 0)
        variant = (
            "climate-dual-scale"
            if isinstance(daily_failure_count, int) and daily_failure_count > 0
            else "cell-level-scalar"
        )
        return RoutingDecision(
            bucket=bucket,
            affordance=routed_affordance,
            diagnostic_variant=variant,
        )

    # 7) Region-specific bounds — same shape as climate
    #    value_range; interpolable when neighbours exist.
    if check_id == "region_specific_bounds":
        bucket = 4 if routed_affordance == "interpolate" else 3
        return RoutingDecision(
            bucket=bucket,
            affordance=routed_affordance,
            diagnostic_variant="cell-level-scalar",
        )

    # 8) Temporal completeness — short-gap routes to bucket 4
    #    INTERPOLATABLE; long-gap stays bucket 3 TRUE_EXCLUDE.
    #    Threshold canonical-sourced from
    #    :data:`TEMPORAL_GAP_BUCKET_4_MAX_DAYS` per durable §24.
    if check_id == "temporal_completeness":
        gap_count = ctx.get("gap_count")
        if (
            isinstance(gap_count, int)
            and gap_count <= TEMPORAL_GAP_BUCKET_4_MAX_DAYS
            and routed_affordance != "rerun_full_sources"
        ):
            return RoutingDecision(
                bucket=4,
                affordance=routed_affordance,
                diagnostic_variant="cell-level-scalar",
            )
        # Long gap or rerun-routed: bucket 3.
        return RoutingDecision(
            bucket=3,
            affordance=routed_affordance,
            diagnostic_variant="cell-level-scalar",
        )

    # 9) Coverage-per-cell — high-coverage routes to bucket 4;
    #    low-coverage stays bucket 3. Threshold canonical-
    #    sourced from
    #    :data:`COVERAGE_PER_CELL_BUCKET_4_MIN_PCT`.
    if check_id == "coverage_per_cell":
        coverage_pct = ctx.get("coverage_pct")
        if (
            isinstance(coverage_pct, (int, float))
            and coverage_pct >= COVERAGE_PER_CELL_BUCKET_4_MIN_PCT
            and routed_affordance != "rerun_full_sources"
        ):
            return RoutingDecision(
                bucket=4,
                affordance=routed_affordance,
                diagnostic_variant="cell-level-scalar",
            )
        return RoutingDecision(
            bucket=3,
            affordance=routed_affordance,
            diagnostic_variant="cell-level-scalar",
        )

    # 10) Cross-variable consistency — too multi-dimensional
    #     to interpolate per ``route_affordance``. Bucket 3.
    if check_id == "cross_variable":
        return RoutingDecision(
            bucket=3,
            affordance=routed_affordance,
            diagnostic_variant="cell-level-scalar",
        )

    # 11) Default — informational acknowledgement (Bucket 2).
    #     Falls through here for ``climate_provider_notice``
    #     + ``soil_dataset_vintage`` + future Bucket-2
    #     check_ids that the producer enumerates without
    #     exception.
    return RoutingDecision(
        bucket=2,
        affordance=routed_affordance,
        diagnostic_variant="cell-level-scalar",
    )


__all__ = [
    "RoutingDecision",
    "bucket_for",
]
