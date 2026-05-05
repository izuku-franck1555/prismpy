"""CropPhysiologicalValidator — Stage 1 zone-level crop-region match.

Sprint F (V2-22c-RESTART Phase 3) populates :meth:`validate`
with the zone-level Stage 1 verdict. The validator iterates the
zones in :class:`InputValidationContext.zone_aggregates` for the
single crop carried on the context, calls the substrate verdict
functions :func:`compare_precip_iqr` and
:func:`compare_thermal_extremes`, and emits
:data:`WarningCategory.CROP_REGION_MISMATCH` (Bucket 5
MANUAL_OVERRIDE_WITH_EVIDENCE — wizard-time documented-override
contract) on either INCOMPATIBLE branch.

What this validator does NOT emit:

* MARGINAL_HETEROGENEOUS / MARGINAL_THERMAL_SEASONAL verdicts
  are already surfaced by :class:`ClimateEnvelopeValidator`'s
  :data:`WarningCategory.CLIMATE_ENVELOPE_TAIL` (Bucket 2 INFO)
  emit in Sprint E.0.5. Re-emitting them here would produce
  noisy double-banners on the same zone.
* :data:`WarningCategory.CROP_PHYSIOLOGY_VIOLATION` is the
  Stage 2 per-cell ECOCROP-tolerance signal, deferred to V2-23
  (task #157). The category is declared in :attr:`EMITS` for
  forward-compat per the F30 walker's declared-superset-of-
  runtime semantics; Sprint F never emits it.
* :data:`WarningCategory.INSUFFICIENTLY_SAMPLED` is the
  unsupported-crop signal (no envelope in the bundled v1
  ECOCROP JSON). It is also declared in :attr:`EMITS` for
  forward-compat; the wizard caller emits it via
  :meth:`unsupported_crop_result` when no envelope can be
  loaded for the user-selected crop.

Anti-mutation context:

* If a future change weakens the substrate's INCOMPATIBLE
  predicate to fire on MARGINAL_*, this validator would emit
  CROP_REGION_MISMATCH for marginal cases and conflict with
  ClimateEnvelopeValidator's Bucket 2 emit on the same zone.
  Anti-mutation drill 1 in AC-F-1 catches that double-emit.
* If the unsupported-crop path silently falls through (no
  INSUFFICIENTLY_SAMPLED emit), an unsupported crop would
  pass silent through wizard pre-flight. Anti-mutation drill
  2 in AC-F-1 catches that fall-through via the
  :meth:`unsupported_crop_result` factory.
"""
from __future__ import annotations

from typing import Optional

from prismpy.validators.base import ValidationIssue
from prismpy.validators.climate_envelope import (
    CompatibilityVerdict,
    compare_precip_iqr,
    compare_thermal_extremes,
    precip_verdict_explanation,
    precip_verdict_reason,
    thermal_verdict_explanation,
    thermal_verdict_reason,
)
from prismpy.validators.input_base import (
    InputValidationContext,
    InputValidationResult,
    InputValidator,
    ZoneAggregate,
)
from prismpy.warnings.categories import WarningCategory


class CropPhysiologicalValidator(InputValidator):
    """Stage 1 zone-level crop-region compatibility validator.

    For each zone in :attr:`InputValidationContext.zone_aggregates`,
    runs the precip + thermal verdict functions against the
    crop's ECOCROP envelope and emits CROP_REGION_MISMATCH on
    INCOMPATIBLE. MARGINAL_* verdicts route to
    :class:`ClimateEnvelopeValidator`'s Bucket 2 INFO emit
    instead so a marginal zone surfaces once with the
    informational severity that fits its data — not twice with
    a Bucket 3 EXCLUDE that would block-list a workable region.

    The :attr:`EMITS` frozenset declares the full Stage 1 +
    Stage 2 + unsupported-crop emission surface so the F30
    walker's declared-superset-of-runtime semantics remain
    stable across the V2-23 expansion. Sprint F runtime emits
    only :data:`WarningCategory.CROP_REGION_MISMATCH`.
    """

    EMITS = frozenset({
        WarningCategory.CROP_REGION_MISMATCH,
        # Declared but never emitted in Sprint F. Reserved for
        # Stage 2 V2-23 per-cell ECOCROP tolerance check.
        WarningCategory.CROP_PHYSIOLOGY_VIOLATION,
        # Declared but emitted only via :meth:`unsupported_crop_result`
        # — when the wizard cannot load an envelope for the
        # user-selected crop, the wizard calls the factory
        # which produces an :class:`InputValidationResult` with
        # this neutral signal.
        WarningCategory.INSUFFICIENTLY_SAMPLED,
    })

    def validate(
        self, input_state: InputValidationContext,
    ) -> InputValidationResult:
        """Run Stage 1 zone-level crop-region match per AC-F-1.

        Iterates :attr:`InputValidationContext.zone_aggregates`
        for the single crop carried on the context; emits
        CROP_REGION_MISMATCH for each zone whose precip or
        thermal verdict is INCOMPATIBLE.

        The ``InputValidationContext`` is constructed by the
        caller (wizard view-handler) with a valid envelope
        already in hand. Unsupported-crop handling lives in
        :meth:`unsupported_crop_result`; callers that cannot
        load an envelope for the user's crop must route through
        that factory instead of constructing a context that
        would fail Pydantic validation.
        """
        crop_envelope = input_state.crop_envelope
        crop_name = input_state.crop_name
        crop_tmin = crop_envelope.TMIN
        crop_tmax = crop_envelope.TMAX
        crop_rmin = crop_envelope.RMIN
        crop_rmax = crop_envelope.RMAX

        issues: list[ValidationIssue] = []
        per_zone_verdicts: dict[str, dict[str, str]] = {}

        for zone, aggs in input_state.zone_aggregates.items():
            # Sample-quality skip mirrors ClimateEnvelopeValidator's
            # path: zones with too few cell-days have no
            # statistically-stable IQR / extreme percentile, so
            # a Stage 1 verdict on them would be substrate-unsafe.
            # The skip is silent here because the same zone has
            # already produced an INSUFFICIENTLY_SAMPLED issue in
            # ClimateEnvelopeValidator's pass per AC-F-1; emitting
            # twice would be noisy.
            if aggs.n_cell_days < input_state.min_cell_days_per_zone:
                per_zone_verdicts[zone] = {
                    "precip": "skipped_insufficient_sample",
                    "thermal": "skipped_insufficient_sample",
                }
                continue

            precip_verdict = compare_precip_iqr(
                p25=aggs.p25, p50=aggs.p50, p75=aggs.p75,
                rmin=crop_rmin, rmax=crop_rmax,
            )
            thermal_verdict = compare_thermal_extremes(
                zone_p10_extreme_tmin=aggs.p10_extreme_tmin,
                zone_p90_extreme_tmax=aggs.p90_extreme_tmax,
                crop_tmin=crop_tmin,
                crop_tmax=crop_tmax,
            )
            per_zone_verdicts[zone] = {
                "precip": precip_verdict.value,
                "thermal": thermal_verdict.value,
            }

            # Per AC-F-1 + AC-F-5 cache schema: emit ONE
            # CROP_REGION_MISMATCH per zone regardless of how
            # many variables crossed the envelope. Codex pre-
            # commit review flagged a one-issue-vs-two-issues
            # ambiguity in Draft 2 wording; the cache schema's
            # ``entries[]`` shape (singular ``verdict``,
            # singular ``reason`` per zone-crop pair) drives the
            # one-row-per-zone resolution. Multi-variable cases
            # combine the per-variable reason strings into a
            # single message; ``details["variables"]`` carries
            # the full per-variable breakdown so the cockpit
            # drawer can still render the structured fields.
            precip_incompat = (
                precip_verdict is CompatibilityVerdict.INCOMPATIBLE
            )
            thermal_incompat = (
                thermal_verdict is CompatibilityVerdict.INCOMPATIBLE
            )
            if precip_incompat or thermal_incompat:
                reasons: list[str] = []
                explanations: list[str] = []
                variables_in_emit: list[str] = []
                # Capitalize crop name for the plain-language
                # explanation copy ("Rice typically needs..." vs
                # the validator's lowercased ``crop_name``).
                pretty_crop = crop_name[:1].upper() + crop_name[1:]
                # Zone label is informational; the validator does
                # not yet thread the human-readable label through
                # zone_aggregates, so the explanation falls back
                # to "this region" via the helper's default.
                if precip_incompat:
                    reasons.append(precip_verdict_reason(
                        verdict=precip_verdict,
                        p25=aggs.p25, p50=aggs.p50, p75=aggs.p75,
                        rmin=crop_rmin, rmax=crop_rmax,
                    ) or "precip INCOMPATIBLE")
                    explanation = precip_verdict_explanation(
                        precip_verdict,
                        aggs.p25, aggs.p50, aggs.p75,
                        crop_rmin, crop_rmax,
                        pretty_crop,
                        zone_label=zone,
                    )
                    if explanation:
                        explanations.append(explanation)
                    variables_in_emit.append("precip")
                if thermal_incompat:
                    reasons.append(thermal_verdict_reason(
                        verdict=thermal_verdict,
                        zone_p10_extreme_tmin=aggs.p10_extreme_tmin,
                        zone_p90_extreme_tmax=aggs.p90_extreme_tmax,
                        crop_tmin=crop_tmin,
                        crop_tmax=crop_tmax,
                    ) or "thermal INCOMPATIBLE")
                    explanation = thermal_verdict_explanation(
                        thermal_verdict,
                        aggs.p10_extreme_tmin,
                        aggs.p90_extreme_tmax,
                        crop_tmin, crop_tmax,
                        pretty_crop,
                        zone_label=zone,
                    )
                    if explanation:
                        explanations.append(explanation)
                    variables_in_emit.append("thermal")
                issues.append(self._build_mismatch_issue(
                    crop_name=crop_name,
                    zone=zone,
                    variables=variables_in_emit,
                    reason="; ".join(reasons),
                    explanation=" ".join(explanations),
                    aggs=aggs,
                    envelope_tmin=crop_tmin,
                    envelope_tmax=crop_tmax,
                    envelope_rmin=crop_rmin,
                    envelope_rmax=crop_rmax,
                ))
            # COMPATIBLE / MARGINAL_* → no issue. Marginal
            # signals are already covered by
            # ClimateEnvelopeValidator's Bucket 2 INFO emit.

        return InputValidationResult(
            valid=all(i.severity != "error" for i in issues),
            issues=issues,
            metadata={
                "validator": "CropPhysiologicalValidator",
                "sprint": "F",
                "stage": 1,
                "per_zone_verdicts": per_zone_verdicts,
                "crop": crop_name,
            },
        )

    # Banner-copy budget: AC-F-2 caps the rendered wizard
    # banner at ≤120 chars. The validator's ``message`` is the
    # data-only stem the wizard wraps with crop name + zone
    # label + ECOCROP URL, so the stem itself stays under a
    # tighter budget. 100 chars leaves comfortable headroom
    # for "...; ..." composition when both precip + thermal
    # fire on the same zone.
    _MAX_REASON_CHARS = 100

    @staticmethod
    def _build_mismatch_issue(
        crop_name: str,
        zone: str,
        variables: list[str],
        reason: str,
        aggs: ZoneAggregate,
        envelope_tmin: float,
        envelope_tmax: float,
        envelope_rmin: float,
        envelope_rmax: float,
        explanation: str = "",
    ) -> ValidationIssue:
        """Build a single CROP_REGION_MISMATCH issue per zone.

        ``message`` carries the data-only reason composed from
        the per-variable reason helpers; the wizard banner
        composes crop name + zone label + ECOCROP source URL
        on top of this when rendering the persona-readable copy
        per AC-F-10. ``details`` carries structured fields for
        the cockpit drawer + override audit log, including the
        list of ``variables`` (precip / thermal) that fired so
        the drawer can break the row down per variable when the
        user opens it.

        Per ux-expert verdict: the wizard banner now surfaces a
        plain-language ``explanation`` field (visible by default)
        in addition to the technical ``reason`` (collapsed in a
        disclosure). Both come from the substrate so the copy +
        data stay tightly coupled.

        ``message`` is truncated to ``_MAX_REASON_CHARS`` chars
        with ``...`` to defend the AC-F-2 ≤120-char banner-copy
        budget against unrealistic substrate values (e.g., a
        unit-corruption upstream that pushed ``p50`` past the
        ``.0f`` formatter). The full untruncated reason stays
        in ``details["reason_full"]`` for the audit log.
        """
        truncated = reason
        if len(truncated) > CropPhysiologicalValidator._MAX_REASON_CHARS:
            cap = CropPhysiologicalValidator._MAX_REASON_CHARS - 3
            truncated = truncated[:cap] + "..."
        return ValidationIssue(
            severity="error",
            category=WarningCategory.CROP_REGION_MISMATCH.value,
            message=truncated,
            details={
                "zone": zone,
                "variables": list(variables),
                "crop": crop_name,
                "verdict": CompatibilityVerdict.INCOMPATIBLE.value,
                "reason_full": reason,
                "explanation": explanation,
                "n_cell_days_in_zone": aggs.n_cell_days,
                "p25": aggs.p25,
                "p50": aggs.p50,
                "p75": aggs.p75,
                "p10_extreme_tmin": aggs.p10_extreme_tmin,
                "p90_extreme_tmax": aggs.p90_extreme_tmax,
                "envelope_tmin": envelope_tmin,
                "envelope_tmax": envelope_tmax,
                "envelope_rmin": envelope_rmin,
                "envelope_rmax": envelope_rmax,
            },
        )

    @staticmethod
    def unsupported_crop_result(
        crop_name: str,
        *,
        region_name: Optional[str] = None,
        zones: Optional[list] = None,
    ) -> InputValidationResult:
        """Return the canonical neutral signal for an unsupported
        crop (no envelope in the bundled v1 ECOCROP JSON).

        The wizard caller reaches this factory when it cannot
        load a :class:`CropEnvelope` for the user-selected crop
        — constructing an :class:`InputValidationContext`
        without an envelope is impossible because the field is
        required. Per AC-F-1, the wizard emits
        :data:`WarningCategory.INSUFFICIENTLY_SAMPLED` (Bucket
        2 INFO) so the user sees "compatibility check
        unavailable for this crop" rather than a silent green
        light.

        Optional ``region_name`` + ``zones`` are kept keyword-
        only so a future Sprint that wants to thread region or
        zone context through the unsupported-crop signal can
        add fields without breaking the existing call sites.
        """
        details = {
            "crop": crop_name,
            "reason": "no_envelope_in_substrate",
        }
        if region_name is not None:
            details["region"] = region_name
        if zones is not None:
            details["zones"] = list(zones)
        return InputValidationResult(
            valid=True,
            issues=[
                ValidationIssue(
                    severity="warning",
                    category=WarningCategory.INSUFFICIENTLY_SAMPLED.value,
                    message=(
                        f"compatibility check unavailable for "
                        f"{crop_name!r} (no ECOCROP envelope in v1)"
                    ),
                    details=details,
                ),
            ],
            metadata={
                "validator": "CropPhysiologicalValidator",
                "sprint": "F",
                "stage": 1,
                "unsupported_crop": True,
                "crop": crop_name,
            },
        )
