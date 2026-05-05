"""Sprint F AC-F-1 — CropPhysiologicalValidator Stage 1 body.

Pins the class shape + EMITS frozenset (3 categories incl.
forward-compat declarations per F30 declared-superset semantics)
+ the Sprint F INCOMPATIBLE-only emit logic + the unsupported-
crop factory.

Anti-mutation drills (per AC-F-1):

* If the validator emits CROP_REGION_MISMATCH for MARGINAL_*
  verdicts → ``test_marginal_zones_emit_no_issue`` fails. The
  Bucket 2 informational signal already lives at
  ``ClimateEnvelopeValidator``; double-emit would surface the
  same zone twice with conflicting severities.
* If the unsupported-crop path silently falls through →
  ``test_unsupported_crop_factory_emits_insufficiently_sampled``
  fails (asserts the neutral signal fires).
* If the substrate's INCOMPATIBLE predicate is weakened to
  fire on MARGINAL_* → the marginal-only fixture suddenly
  starts producing CROP_REGION_MISMATCH issues, conflicting
  with ClimateEnvelopeValidator's CLIMATE_ENVELOPE_TAIL
  Bucket 2 emission on the same zone.
* If a category is added to EMITS without a matching emit
  site, the F30 walker (AC-F-8) catches it — declared-
  superset-of-runtime is allowed, but the walker enforces
  that runtime emits stay within EMITS.
"""
from __future__ import annotations

import unittest

from prismpy.validators.base import ValidationIssue
from prismpy.validators.crop_physiological import (
    CropPhysiologicalValidator,
)
from prismpy.validators.input_base import (
    CropEnvelope,
    InputValidationContext,
    InputValidationResult,
    InputValidator,
    ZoneAggregate,
)
from prismpy.warnings.categories import WarningCategory


# Fixture envelopes anchored against the Sprint F AC-F-0 ECOCROP
# expansion. Values match prismpy/src/prismpy/koppen/ecocrop_envelopes.json.
MAIZE_ENVELOPE = CropEnvelope(
    TMIN=10.0, TMAX=47.0, RMIN=400.0, RMAX=1800.0,
)
RICE_ENVELOPE = CropEnvelope(
    TMIN=10.0, TMAX=36.0, RMIN=1000.0, RMAX=4000.0,
)
SORGHUM_ENVELOPE = CropEnvelope(
    TMIN=8.0, TMAX=40.0, RMIN=300.0, RMAX=700.0,
)


# Synthetic zone aggregates anchored against the AC-F-3 fixture
# canonical zones. P50 values chosen so each fixture exercises a
# known precip verdict branch; thermal extremes are pinned inside
# the broadest crop envelope (rice TMIN=10 / TMAX=36) so the
# precip-branch tests do NOT also trigger thermal INCOMPATIBLE
# emits and shadow the AC-F-1 emit-isolation invariant. A
# separate ``HIGHLAND_COLD_KILL`` / ``TROPICAL_HEAT_KILL``
# fixture pair exercises the thermal branches deliberately.
SAHEL_BSh_DRY = ZoneAggregate(
    p25=200.0, p50=280.0, p75=350.0,
    p10_extreme_tmin=15.0, p90_extreme_tmax=35.0,
    n_cell_days=2_000_000,
)
SAHEL_BSh_MARGINAL_PRECIP = ZoneAggregate(
    p25=200.0, p50=350.0, p75=600.0,
    p10_extreme_tmin=15.0, p90_extreme_tmax=35.0,
    n_cell_days=2_000_000,
)
SAHEL_BSh_COMPATIBLE = ZoneAggregate(
    p25=600.0, p50=900.0, p75=1200.0,
    p10_extreme_tmin=15.0, p90_extreme_tmax=35.0,
    n_cell_days=2_000_000,
)
HIGHLAND_COLD_KILL = ZoneAggregate(
    p25=600.0, p50=900.0, p75=1200.0,
    p10_extreme_tmin=-5.0,  # below maize TMIN=10
    p90_extreme_tmax=30.0,
    n_cell_days=2_000_000,
)
TROPICAL_HEAT_KILL = ZoneAggregate(
    # Rice (TMAX=36) under heat stress.
    p25=2000.0, p50=2500.0, p75=3000.0,
    p10_extreme_tmin=20.0,
    p90_extreme_tmax=48.0,  # above rice TMAX=36
    n_cell_days=2_000_000,
)
INSUFFICIENT_ZONE = ZoneAggregate(
    p25=200.0, p50=280.0, p75=350.0,
    p10_extreme_tmin=15.0, p90_extreme_tmax=42.0,
    n_cell_days=100,  # below MIN_CELL_DAYS_PER_ZONE
)


def _ctx(
    crop_name: str,
    crop_envelope: CropEnvelope,
    zones: dict,
) -> InputValidationContext:
    return InputValidationContext(
        crop_name=crop_name,
        crop_envelope=crop_envelope,
        zone_aggregates=zones,
    )


class TestCropPhysiologicalValidatorShape(unittest.TestCase):
    """Sprint F AC-F-1: pin the class hierarchy + the EMITS
    frozenset (3-category declared-superset per F30 walker
    declared-superset-of-runtime semantics)."""

    def test_is_input_validator_subclass(self):
        self.assertTrue(
            issubclass(CropPhysiologicalValidator, InputValidator),
        )

    def test_emits_includes_three_categories(self):
        # AC-F-1 EMITS = three categories. CROP_REGION_MISMATCH
        # is the runtime emit; CROP_PHYSIOLOGY_VIOLATION is
        # forward-compat for Stage 2 V2-23; INSUFFICIENTLY_SAMPLED
        # is forward-compat for the unsupported-crop factory.
        self.assertEqual(
            CropPhysiologicalValidator.EMITS,
            frozenset({
                WarningCategory.CROP_REGION_MISMATCH,
                WarningCategory.CROP_PHYSIOLOGY_VIOLATION,
                WarningCategory.INSUFFICIENTLY_SAMPLED,
            }),
        )

    def test_emits_does_not_include_climate_envelope_tail(self):
        # CLIMATE_ENVELOPE_TAIL is ClimateEnvelopeValidator's
        # domain. CC-22 enforces single-canonical-emitter
        # discipline at runtime; F30 walker enforces it
        # structurally.
        self.assertNotIn(
            WarningCategory.CLIMATE_ENVELOPE_TAIL,
            CropPhysiologicalValidator.EMITS,
        )

    def test_can_instantiate(self):
        v = CropPhysiologicalValidator()
        self.assertIsInstance(v, InputValidator)


class TestCropPhysiologicalValidatorIncompatibleEmit(unittest.TestCase):
    """Sprint F AC-F-1: per-zone INCOMPATIBLE → CROP_REGION_MISMATCH
    emit. Covers the canonical G-c-4 rice-in-Sahel inverse case
    plus heat-kill / cold-kill thermal cases."""

    def setUp(self):
        self.validator = CropPhysiologicalValidator()

    def test_rice_dry_zone_emits_crop_region_mismatch(self):
        # G-c-4 canonical case: rice (RMIN=1000) × Sahel BSh
        # (P50=280). Substrate routes precip to INCOMPATIBLE;
        # validator emits ONE CROP_REGION_MISMATCH per zone
        # (the one-issue-per-zone shape matches AC-F-5 cache
        # entries[]).
        ctx = _ctx("rice", RICE_ENVELOPE, {"BSh": SAHEL_BSh_DRY})
        result = self.validator.validate(ctx)
        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        self.assertEqual(
            issue.category,
            WarningCategory.CROP_REGION_MISMATCH.value,
        )
        self.assertEqual(issue.severity, "error")
        self.assertIn("280", issue.message)  # P50 reported
        self.assertIn("1000", issue.message)  # RMIN reported
        self.assertIn("below", issue.message)
        # Details carry the structured fields the cockpit reads.
        self.assertEqual(issue.details["zone"], "BSh")
        self.assertEqual(issue.details["variables"], ["precip"])
        self.assertEqual(issue.details["crop"], "rice")
        self.assertEqual(issue.details["verdict"], "incompatible")

    def test_cold_kill_emits_thermal_mismatch(self):
        # Maize at highland zone with extreme tmin -5°C →
        # cold-kill below crop TMIN=10°C → INCOMPATIBLE thermal
        # → CROP_REGION_MISMATCH (single issue, variables=[thermal]).
        ctx = _ctx("maize", MAIZE_ENVELOPE, {"highland": HIGHLAND_COLD_KILL})
        result = self.validator.validate(ctx)
        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        self.assertEqual(issue.details["variables"], ["thermal"])
        self.assertIn("tmin", issue.message)

    def test_heat_kill_emits_thermal_mismatch(self):
        # Rice TMAX=36 vs zone P90 extreme tmax=48 →
        # heat-kill alone → INCOMPATIBLE thermal.
        ctx = _ctx("rice", RICE_ENVELOPE, {"tropical": TROPICAL_HEAT_KILL})
        result = self.validator.validate(ctx)
        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        self.assertEqual(issue.details["variables"], ["thermal"])
        self.assertIn("tmax", issue.message)

    def test_marginal_zones_emit_no_issue(self):
        # AC-F-1 BLOCKER 1 resolution: MARGINAL_* never drives
        # a Sprint F emission. Sahel BSh × sorghum produces a
        # marginal precip verdict (P50=350 in envelope but
        # P25=200 below RMIN=300); CropPhysiologicalValidator
        # emits NOTHING — ClimateEnvelopeValidator already
        # surfaces the CLIMATE_ENVELOPE_TAIL Bucket 2 INFO.
        ctx = _ctx("sorghum", SORGHUM_ENVELOPE, {"BSh": SAHEL_BSh_MARGINAL_PRECIP})
        result = self.validator.validate(ctx)
        self.assertEqual(result.issues, [])

    def test_compatible_zone_emits_no_issue(self):
        ctx = _ctx("maize", MAIZE_ENVELOPE, {"BSh": SAHEL_BSh_COMPATIBLE})
        result = self.validator.validate(ctx)
        self.assertEqual(result.issues, [])

    def test_insufficient_zone_silent_skip(self):
        # Zones below MIN_CELL_DAYS_PER_ZONE skip silently
        # because ClimateEnvelopeValidator's pass already
        # emits INSUFFICIENTLY_SAMPLED for the same zone;
        # Sprint F does not double-emit.
        ctx = _ctx("rice", RICE_ENVELOPE, {"BSh": INSUFFICIENT_ZONE})
        result = self.validator.validate(ctx)
        self.assertEqual(result.issues, [])
        # Per-zone metadata records the skip reason for audit.
        self.assertEqual(
            result.metadata["per_zone_verdicts"]["BSh"]["precip"],
            "skipped_insufficient_sample",
        )

    def test_emit_carries_plain_language_explanation_in_details(self):
        # Per ux-expert verdict, the wizard banner surfaces a
        # plain-language explanation (visible by default)
        # alongside the technical reason (collapsed in a
        # disclosure). The validator must thread the
        # explanation through ``details["explanation"]`` so the
        # wizard banner reads it without recomputing.
        ctx = _ctx("rice", RICE_ENVELOPE, {"BSh": SAHEL_BSh_DRY})
        result = self.validator.validate(ctx)
        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        self.assertIn("explanation", issue.details)
        explanation = issue.details["explanation"]
        # Plain-language vocabulary, not the technical reason.
        self.assertIn("Rice", explanation)
        self.assertIn("too dry", explanation.lower())
        # Names the substrate values for honest-signal trust.
        self.assertIn("1000mm", explanation)
        self.assertIn("280mm", explanation)
        # Sentence ends cleanly without the "the this region"
        # double-mention bug codex review #DIM-2 surfaced.
        self.assertNotIn("the this region", explanation.lower())
        # F-Path-β-1 — validator now resolves the human-readable
        # zone label via ``koppen.zone_aggregates.label_for`` so
        # the persona reads "Hot semi-arid" rather than the raw
        # Köppen code "BSh" leaking into the plain-language copy.
        self.assertIn("Hot semi-arid", explanation)
        self.assertNotIn("BSh", explanation)

    def test_emit_carries_human_readable_zone_label_in_details(self):
        # F-Path-β-1 — alongside the explanation copy, the issue
        # surfaces the human-readable zone label as a structured
        # field so the cockpit drawer + per-zone technical detail
        # block can render "Hot semi-arid:" rather than "BSh:".
        # The Köppen code stays on ``zone`` for the audit trail
        # + cockpit filter; ``zone_label`` is the user-facing
        # name resolved from the zone-aggregates substrate.
        ctx = _ctx("rice", RICE_ENVELOPE, {"BSh": SAHEL_BSh_DRY})
        result = self.validator.validate(ctx)
        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        self.assertEqual(issue.details.get("zone"), "BSh")
        self.assertEqual(issue.details.get("zone_label"), "Hot semi-arid")

    def test_thermal_only_emit_threads_zone_label_through_explanation(self):
        # F-Path-β-1 codex follow-up — the precip-path test pins
        # absence of the Köppen code in the explanation, but a
        # regression that drops ``zone_label=pretty_zone_label``
        # from ``thermal_verdict_explanation`` alone would slip
        # past it (the precip explanation would still pass). This
        # case fires thermal INCOMPAT against a substrate-real
        # zone (``BSh``) so the explanation pulls "Hot semi-arid"
        # via ``label_for`` and the assertion surface is the
        # thermal helper specifically.
        thermal_only_envelope = CropEnvelope(
            RMIN=200.0, RMAX=600.0,    # BSh P50=280 → COMPATIBLE precip
            TMIN=20.0, TMAX=40.0,       # BSh P10=15 < TMIN=20 → cold-kill
        )
        ctx = _ctx(
            "thermal_only_test_crop",
            thermal_only_envelope,
            {"BSh": SAHEL_BSh_DRY},
        )
        result = self.validator.validate(ctx)
        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        self.assertEqual(issue.details["variables"], ["thermal"])
        explanation = issue.details["explanation"]
        self.assertIn(
            "Hot semi-arid", explanation,
            "Thermal explanation must carry the human-readable "
            "zone label per F-Path-β-1; a regression that drops "
            "``zone_label=pretty_zone_label`` from "
            "``thermal_verdict_explanation`` would surface here.",
        )
        self.assertNotIn(
            "BSh", explanation,
            "Thermal explanation must NOT leak the Köppen code "
            "into persona-facing copy.",
        )

    def test_combined_precip_and_thermal_emit_one_issue(self):
        # When both precip AND thermal fire INCOMPATIBLE on
        # the same zone, Sprint F emits ONE
        # CROP_REGION_MISMATCH issue with combined reason +
        # ``details["variables"] = ["precip", "thermal"]``.
        # The one-issue-per-zone shape matches AC-F-5 cache
        # ``entries[]`` (singular ``verdict`` + ``reason``).
        zone = ZoneAggregate(
            p25=200.0, p50=280.0, p75=350.0,
            p10_extreme_tmin=20.0,
            p90_extreme_tmax=48.0,  # rice TMAX=36 → heat-kill
            n_cell_days=2_000_000,
        )
        ctx = _ctx("rice", RICE_ENVELOPE, {"BSh": zone})
        result = self.validator.validate(ctx)
        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        self.assertEqual(
            issue.details["variables"], ["precip", "thermal"],
        )
        # Combined reason carries both halves; the joiner is
        # "; " so the cockpit can split on it for per-variable
        # rendering inside the drawer.
        self.assertIn("280", issue.message)  # precip P50
        self.assertIn("48", issue.message)   # thermal P90
        self.assertIn(";", issue.message)
        # Untruncated reason mirrored in details for audit log.
        self.assertEqual(
            issue.details["reason_full"], issue.message,
        )

    def test_message_truncated_to_budget_on_extreme_input(self):
        # AC-F-2 banner-copy budget defense: if a unit-corrupted
        # upstream pushes ``p50`` past the helper's ``.0f``
        # formatter, the validator truncates ``message`` so the
        # ≤120-char banner-copy budget still holds. Full reason
        # stays in ``details["reason_full"]`` for audit trail.
        # Synthetic over-budget zone: 16-digit values force the
        # combined precip + thermal reason past the validator's
        # 100-char internal budget.
        zone = ZoneAggregate(
            p25=9_999_999_999_999_999.0,
            p50=9_999_999_999_999_999.0,
            p75=9_999_999_999_999_999.0,
            p10_extreme_tmin=20.0,
            p90_extreme_tmax=9_999_999_999_999_999.0,
            n_cell_days=2_000_000,
        )
        ctx = _ctx("rice", RICE_ENVELOPE, {"BSh": zone})
        result = self.validator.validate(ctx)
        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        # ``message`` truncated to ≤100 chars (validator's
        # internal budget; banner adds crop+zone+url on top
        # within the AC-F-2 ≤120-char ceiling).
        self.assertLessEqual(len(issue.message), 100)
        # ``...`` truncation marker present.
        self.assertTrue(
            issue.message.endswith("..."),
            f"expected truncation marker at end of "
            f"{issue.message!r}",
        )
        # Full reason length exceeds the truncated message.
        self.assertGreater(
            len(issue.details["reason_full"]),
            len(issue.message),
        )


class TestCropPhysiologicalValidatorMetadata(unittest.TestCase):
    """Pin per-run metadata + idempotency."""

    def setUp(self):
        self.validator = CropPhysiologicalValidator()

    def test_metadata_carries_per_zone_verdicts(self):
        ctx = _ctx(
            "maize", MAIZE_ENVELOPE,
            {
                "BSh_dry": SAHEL_BSh_DRY,
                "BSh_compatible": SAHEL_BSh_COMPATIBLE,
            },
        )
        result = self.validator.validate(ctx)
        self.assertIn("per_zone_verdicts", result.metadata)
        self.assertIn("BSh_dry", result.metadata["per_zone_verdicts"])
        self.assertIn("BSh_compatible", result.metadata["per_zone_verdicts"])

    def test_metadata_records_sprint_f_stage_1(self):
        ctx = _ctx("maize", MAIZE_ENVELOPE, {"BSh": SAHEL_BSh_COMPATIBLE})
        result = self.validator.validate(ctx)
        self.assertEqual(result.metadata["sprint"], "F")
        self.assertEqual(result.metadata["stage"], 1)
        self.assertEqual(result.metadata["crop"], "maize")

    def test_validate_idempotent(self):
        ctx = _ctx("rice", RICE_ENVELOPE, {"BSh": SAHEL_BSh_DRY})
        a = self.validator.validate(ctx)
        b = self.validator.validate(ctx)
        # Same input → same number of issues + same per-zone
        # verdict map (substrate is deterministic).
        self.assertEqual(len(a.issues), len(b.issues))
        self.assertEqual(
            a.metadata["per_zone_verdicts"],
            b.metadata["per_zone_verdicts"],
        )


class TestUnsupportedCropFactory(unittest.TestCase):
    """Sprint F AC-F-1 BLOCKER 1 anti-mutation drill 2: the
    unsupported-crop path must emit a neutral
    INSUFFICIENTLY_SAMPLED signal, not pass silent."""

    def test_unsupported_crop_factory_emits_insufficiently_sampled(self):
        result = CropPhysiologicalValidator.unsupported_crop_result(
            "cassava",
        )
        self.assertEqual(len(result.issues), 1)
        issue = result.issues[0]
        self.assertEqual(
            issue.category,
            WarningCategory.INSUFFICIENTLY_SAMPLED.value,
        )
        self.assertEqual(issue.severity, "warning")
        self.assertIn("cassava", issue.message)
        self.assertIn("ECOCROP", issue.message)
        # ``valid=True`` because the wizard isn't blocking the
        # user — they just don't get a Stage 1 verdict.
        self.assertTrue(result.valid)

    def test_unsupported_crop_metadata_flags_unsupported(self):
        result = CropPhysiologicalValidator.unsupported_crop_result(
            "cassava",
        )
        self.assertTrue(result.metadata["unsupported_crop"])
        self.assertEqual(result.metadata["crop"], "cassava")


class TestEmitsDiscipline(unittest.TestCase):
    """Pin runtime emissions stay within EMITS — every issue
    category from any test fixture must be a member of the
    declared frozenset. Walks across multiple fixture shapes
    so a regression that adds an out-of-EMITS emit is caught
    even if a single fixture happens not to exercise it."""

    def test_runtime_emits_subset_of_declared(self):
        validator = CropPhysiologicalValidator()
        emits_values = {c.value for c in validator.EMITS}

        fixtures = [
            ("rice", RICE_ENVELOPE, {"BSh": SAHEL_BSh_DRY}),
            ("maize", MAIZE_ENVELOPE, {"highland": HIGHLAND_COLD_KILL}),
            ("rice", RICE_ENVELOPE, {"tropical": TROPICAL_HEAT_KILL}),
            ("sorghum", SORGHUM_ENVELOPE, {"BSh": SAHEL_BSh_MARGINAL_PRECIP}),
            ("maize", MAIZE_ENVELOPE, {"BSh": SAHEL_BSh_COMPATIBLE}),
            ("rice", RICE_ENVELOPE, {"BSh": INSUFFICIENT_ZONE}),
        ]
        for crop, envelope, zones in fixtures:
            with self.subTest(crop=crop, zones=list(zones)):
                ctx = _ctx(crop, envelope, zones)
                result = validator.validate(ctx)
                for issue in result.issues:
                    self.assertIn(issue.category, emits_values)

        # Also covers the unsupported-crop factory.
        unsupported = CropPhysiologicalValidator.unsupported_crop_result(
            "cassava",
        )
        for issue in unsupported.issues:
            self.assertIn(issue.category, emits_values)


if __name__ == "__main__":
    unittest.main()
