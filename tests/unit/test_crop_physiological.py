"""Sprint E.0.5 commit 8 — CropPhysiologicalValidator skeleton.

Pins the class shape + EMITS frozenset metadata + the
Sprint F deferral (validate returns no issues in E.0.5).

Anti-mutation drills:

- Drop CROP_PHYSIOLOGY_VIOLATION from EMITS →
  ``test_emits_includes_both_crop_categories`` fails.
- Populate validate() with non-empty issues for E.0.5 →
  ``test_validate_returns_empty_for_e_0_5`` fails (Sprint F
  populates; E.0.5 ships skeleton only).
- Drop the InputValidator subclass relationship →
  ``test_is_input_validator_subclass`` fails.
"""
from __future__ import annotations

import unittest

from prismpy.validators.crop_physiological import (
    CropPhysiologicalValidator,
)
from prismpy.validators.input_base import (
    InputValidationContext,
    InputValidationResult,
    InputValidator,
)
from prismpy.warnings.categories import WarningCategory


def _ctx() -> InputValidationContext:
    return InputValidationContext(
        crop_name="maize",
        crop_envelope={
            "TMIN": 10.0, "TMAX": 47.0,
            "RMIN": 400.0, "RMAX": 1800.0,
        },
        zone_aggregates={
            "BSh": {
                "p25": 200.0, "p50": 300.0, "p75": 380.0,
                "p10_extreme_tmin": 5.0,
                "p90_extreme_tmax": 50.0,
                "n_cell_days": 2_000_000,
            },
        },
    )


class TestCropPhysiologicalValidatorShape(unittest.TestCase):
    """Pin the class hierarchy + the EMITS frozenset + the
    Sprint F deferral (skeleton in E.0.5)."""

    def test_is_input_validator_subclass(self):
        self.assertTrue(
            issubclass(CropPhysiologicalValidator, InputValidator),
        )

    def test_emits_includes_both_crop_categories(self):
        self.assertEqual(
            CropPhysiologicalValidator.EMITS,
            frozenset({
                WarningCategory.CROP_PHYSIOLOGY_VIOLATION,
                WarningCategory.CROP_REGION_MISMATCH,
            }),
        )

    def test_emits_does_not_include_climate_categories(self):
        # CLIMATE_ENVELOPE_TAIL + INSUFFICIENTLY_SAMPLED are
        # ClimateEnvelopeValidator's domain. Per F25-shape walker
        # discipline, EMITS frozensets must not overlap across
        # validators (each warning category has one canonical
        # emitter).
        self.assertNotIn(
            WarningCategory.CLIMATE_ENVELOPE_TAIL,
            CropPhysiologicalValidator.EMITS,
        )
        self.assertNotIn(
            WarningCategory.INSUFFICIENTLY_SAMPLED,
            CropPhysiologicalValidator.EMITS,
        )

    def test_can_instantiate(self):
        v = CropPhysiologicalValidator()
        self.assertIsInstance(v, InputValidator)


class TestCropPhysiologicalValidatorSkeleton(unittest.TestCase):
    """Pin the Sprint F deferral: validate returns no issues
    in E.0.5; Sprint F populates the body."""

    @classmethod
    def setUpClass(cls):
        cls.validator = CropPhysiologicalValidator()
        cls.context = _ctx()

    def test_validate_returns_empty_for_e_0_5(self):
        result = self.validator.validate(self.context)
        self.assertIsInstance(result, InputValidationResult)
        self.assertTrue(result.valid)
        self.assertEqual(result.issues, [])

    def test_metadata_documents_skeleton_status(self):
        result = self.validator.validate(self.context)
        self.assertIn("status", result.metadata)
        self.assertIn("skeleton", result.metadata["status"])

    def test_validate_idempotent(self):
        a = self.validator.validate(self.context)
        b = self.validator.validate(self.context)
        # Empty issues both times.
        self.assertEqual(a.issues, [])
        self.assertEqual(b.issues, [])


if __name__ == "__main__":
    unittest.main()
