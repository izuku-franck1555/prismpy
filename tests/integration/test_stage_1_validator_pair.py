"""Sprint F AC-F-1 BLOCKER 1 integration drill.

Pins the bucket-discipline contract that splits the Stage 1
emit surface across two validators:

* :class:`ClimateEnvelopeValidator` owns the marginal /
  insufficient-sample signals (Bucket 2 INFO).
* :class:`CropPhysiologicalValidator` owns the incompatible
  signals (Bucket 3 EXCLUDE).

Codex commit-2/3 review flagged two MEDIUM gaps that this
integration drill closes:

* (Drill 1, AC-F-1 anti-mutation) — a marginal-only fixture
  must surface CLIMATE_ENVELOPE_TAIL exactly once and produce
  zero CROP_REGION_MISMATCH emissions. A future substrate
  refactor that weakens ``compare_precip_iqr`` to fire
  INCOMPATIBLE on marginal-heterogeneous would route the same
  zone to both buckets at once; the unit-level tests for each
  validator can't catch that interaction.
* (Drill 2, codex MED #5) — an insufficient-sample fixture
  must surface INSUFFICIENTLY_SAMPLED via the climate-envelope
  pass and produce zero emissions from the crop-physiological
  pass. Without the climate-envelope pass running first, the
  silent skip in CropPhysiologicalValidator would hide the
  zone entirely from the cockpit.
"""
from __future__ import annotations

import unittest

from prismpy.validators.climate_envelope import ClimateEnvelopeValidator
from prismpy.validators.crop_physiological import (
    CropPhysiologicalValidator,
)
from prismpy.validators.input_base import (
    CropEnvelope,
    InputValidationContext,
    ZoneAggregate,
)
from prismpy.warnings.categories import WarningCategory


SORGHUM_ENVELOPE = CropEnvelope(
    TMIN=8.0, TMAX=40.0, RMIN=300.0, RMAX=700.0,
)
RICE_ENVELOPE = CropEnvelope(
    TMIN=10.0, TMAX=36.0, RMIN=1000.0, RMAX=4000.0,
)
MAIZE_ENVELOPE = CropEnvelope(
    TMIN=10.0, TMAX=47.0, RMIN=400.0, RMAX=1800.0,
)


class TestStage1ValidatorPair(unittest.TestCase):
    """Pair-wise bucket-discipline checks for the Stage 1
    surface."""

    def setUp(self):
        self.climate = ClimateEnvelopeValidator()
        self.crop = CropPhysiologicalValidator()

    def _run_pair(
        self,
        crop_name: str,
        crop_envelope: CropEnvelope,
        zones: dict,
    ):
        ctx = InputValidationContext(
            crop_name=crop_name,
            crop_envelope=crop_envelope,
            zone_aggregates=zones,
        )
        return self.climate.validate(ctx), self.crop.validate(ctx)

    def test_marginal_zone_routes_to_bucket2_only(self):
        # Sahel BSh × sorghum: P25=200 below RMIN=300 but
        # P50=350 in envelope → MARGINAL_HETEROGENEOUS. The
        # bucket-discipline contract says CLIMATE_ENVELOPE_TAIL
        # (Bucket 2) fires exactly once and CROP_REGION_MISMATCH
        # (Bucket 3) NEVER fires on this fixture.
        marginal = ZoneAggregate(
            p25=200.0, p50=350.0, p75=600.0,
            p10_extreme_tmin=15.0,
            p90_extreme_tmax=35.0,  # well within sorghum TMAX=40
            n_cell_days=2_000_000,
        )
        climate_result, crop_result = self._run_pair(
            "sorghum", SORGHUM_ENVELOPE, {"BSh": marginal},
        )

        # Climate-envelope pass: exactly one Bucket 2 INFO emit.
        climate_categories = [i.category for i in climate_result.issues]
        self.assertEqual(
            climate_categories.count(
                WarningCategory.CLIMATE_ENVELOPE_TAIL.value,
            ),
            1,
        )
        # Crop-physiological pass: zero issues. If a substrate
        # refactor weakens INCOMPATIBLE to fire on marginal,
        # this assertion catches the double-emit immediately.
        self.assertEqual(crop_result.issues, [])

    def test_incompatible_zone_routes_to_bucket3_only(self):
        # Rice (RMIN=1000) × Sahel BSh (P50=280) → INCOMPATIBLE
        # precip. Bucket 3 EXCLUDE fires once on the crop
        # validator; the climate validator stays silent on
        # INCOMPATIBLE per its contract (CROP_REGION_MISMATCH
        # is NOT in ClimateEnvelopeValidator.EMITS).
        incompatible = ZoneAggregate(
            p25=200.0, p50=280.0, p75=350.0,
            p10_extreme_tmin=15.0, p90_extreme_tmax=35.0,
            n_cell_days=2_000_000,
        )
        climate_result, crop_result = self._run_pair(
            "rice", RICE_ENVELOPE, {"BSh": incompatible},
        )

        # Climate-envelope pass: zero CROP_REGION_MISMATCH
        # emits (and zero CLIMATE_ENVELOPE_TAIL — INCOMPATIBLE
        # on precip is not a tail signal).
        climate_categories = [i.category for i in climate_result.issues]
        self.assertNotIn(
            WarningCategory.CROP_REGION_MISMATCH.value,
            climate_categories,
        )
        # Crop-physiological pass: exactly one Bucket 3 emit.
        crop_categories = [i.category for i in crop_result.issues]
        self.assertEqual(
            crop_categories.count(
                WarningCategory.CROP_REGION_MISMATCH.value,
            ),
            1,
        )

    def test_insufficient_sample_routes_to_bucket2_only(self):
        # Codex MED #5: insufficient zones rely on the climate
        # validator running first. Drill: insufficient zone
        # surfaces INSUFFICIENTLY_SAMPLED via the climate pass
        # and stays silent in the crop pass.
        insufficient = ZoneAggregate(
            p25=200.0, p50=280.0, p75=350.0,
            p10_extreme_tmin=15.0, p90_extreme_tmax=35.0,
            n_cell_days=100,
        )
        climate_result, crop_result = self._run_pair(
            "rice", RICE_ENVELOPE, {"BSh": insufficient},
        )

        # Climate-envelope pass: exactly one INSUFFICIENTLY_SAMPLED
        # emit.
        climate_categories = [i.category for i in climate_result.issues]
        self.assertEqual(
            climate_categories.count(
                WarningCategory.INSUFFICIENTLY_SAMPLED.value,
            ),
            1,
        )
        # Crop-physiological pass: zero issues on insufficient
        # zones (silent skip per AC-F-1; the climate pass owns
        # the signal).
        self.assertEqual(crop_result.issues, [])

    def test_compatible_zone_silent_in_both_validators(self):
        compatible = ZoneAggregate(
            p25=600.0, p50=900.0, p75=1200.0,
            p10_extreme_tmin=15.0, p90_extreme_tmax=35.0,
            n_cell_days=2_000_000,
        )
        climate_result, crop_result = self._run_pair(
            "maize", MAIZE_ENVELOPE, {"BSh": compatible},
        )
        self.assertEqual(climate_result.issues, [])
        self.assertEqual(crop_result.issues, [])

    def test_no_category_double_emit_across_validators(self):
        # CC-22 single-canonical-emitter discipline: every
        # category emitted by ClimateEnvelopeValidator MUST
        # NOT also be emittable by CropPhysiologicalValidator.
        # The intersection of EMITS frozensets must be empty
        # at the runtime-emit level. (Static EMITS frozensets
        # may overlap on forward-compat declarations like
        # INSUFFICIENTLY_SAMPLED; this test pins the runtime
        # contract.)
        # A mixed fixture exercising marginal + incompatible
        # in the same context.
        zones = {
            "BSh_marginal": ZoneAggregate(
                p25=200.0, p50=500.0, p75=900.0,
                p10_extreme_tmin=15.0, p90_extreme_tmax=35.0,
                n_cell_days=2_000_000,
            ),
            "BSh_dry": ZoneAggregate(
                p25=200.0, p50=280.0, p75=350.0,
                p10_extreme_tmin=15.0, p90_extreme_tmax=35.0,
                n_cell_days=2_000_000,
            ),
        }
        climate_result, crop_result = self._run_pair(
            "rice", RICE_ENVELOPE, zones,
        )
        climate_categories = {i.category for i in climate_result.issues}
        crop_categories = {i.category for i in crop_result.issues}
        # No category should land in both validators on the
        # same context.
        self.assertEqual(
            climate_categories & crop_categories, set(),
            f"Bucket-discipline violation: same category emitted "
            f"by both validators. climate={climate_categories} "
            f"crop={crop_categories}",
        )


if __name__ == "__main__":
    unittest.main()
