"""Sprint E.0.5 commit 8 — InputValidator ABC + Context + Result.

Pins the parallel-input-validator architecture (distinct
from the existing output-validator BaseValidator) so future
Sprint F + V2-23 input-validator extractions can rely on
the shape.

Anti-mutation drills:

- Drop the EMITS class attribute from ``InputValidator`` →
  ``test_emits_classvar_present`` fails.
- Make the ABC's ``validate`` non-abstract → an instantiation
  test passes when it shouldn't.
- Mutate :class:`InputValidationContext` after construction
  → ``test_context_is_frozen`` fails.
- Add an unexpected field to a context →
  ``test_context_extra_fields_forbidden`` fails.
"""
from __future__ import annotations

import unittest
from typing import FrozenSet

from prismpy.validators.base import ValidationIssue
from prismpy.validators.input_base import (
    InputValidationContext,
    InputValidationResult,
    InputValidator,
)
from prismpy.warnings.categories import WarningCategory


class TestInputValidationResult(unittest.TestCase):
    """Pin the dataclass shape — distinct from
    :class:`prismpy.validators.base.ValidationResult` (no
    platform / output_dir for input-time validators)."""

    def test_default_empty_issues(self):
        result = InputValidationResult(valid=True)
        self.assertEqual(result.issues, [])
        self.assertEqual(result.metadata, {})

    def test_n_errors_n_warnings_n_info(self):
        result = InputValidationResult(
            valid=False,
            issues=[
                ValidationIssue(severity="error", category="x", message="e1"),
                ValidationIssue(severity="error", category="x", message="e2"),
                ValidationIssue(severity="warning", category="x", message="w1"),
                ValidationIssue(severity="info", category="x", message="i1"),
            ],
        )
        self.assertEqual(result.n_errors, 2)
        self.assertEqual(result.n_warnings, 1)
        self.assertEqual(result.n_info, 1)


class TestInputValidationContext(unittest.TestCase):
    """Pin the Pydantic context model: frozen + extra-forbid +
    required fields."""

    def _ctx(self, **overrides):
        base = dict(
            crop_name="maize",
            crop_envelope={"TMIN": 10.0, "TMAX": 47.0, "RMIN": 400.0, "RMAX": 1800.0},
            zone_aggregates={
                "BSh": {
                    "p25": 600.0, "p50": 900.0, "p75": 1200.0,
                    "p10_extreme_tmin": 12.0,
                    "p90_extreme_tmax": 40.0,
                    "n_cell_days": 2_000_000,
                },
            },
        )
        base.update(overrides)
        return InputValidationContext(**base)

    def test_constructs_with_required_fields(self):
        ctx = self._ctx()
        self.assertEqual(ctx.crop_name, "maize")
        self.assertEqual(ctx.min_cell_days_per_zone, 1_000_000)

    def test_context_is_frozen(self):
        ctx = self._ctx()
        with self.assertRaises(Exception):
            ctx.crop_name = "rice"

    def test_context_extra_fields_forbidden(self):
        with self.assertRaises(Exception):
            self._ctx(extra_field="should not pass")

    def test_empty_crop_name_rejected(self):
        with self.assertRaises(Exception):
            self._ctx(crop_name="")

    def test_negative_min_cell_days_rejected(self):
        with self.assertRaises(Exception):
            self._ctx(min_cell_days_per_zone=-1)


class TestInputValidatorABC(unittest.TestCase):
    """The :class:`InputValidator` ABC enforces ``validate``
    abstract + ``EMITS`` class metadata."""

    def test_abstract_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            InputValidator()

    def test_emits_classvar_present(self):
        # The base class declares EMITS as an empty frozenset
        # by default; subclasses override.
        self.assertTrue(hasattr(InputValidator, "EMITS"))
        self.assertIsInstance(
            InputValidator.EMITS, frozenset,
        )

    def test_subclass_can_override_emits_and_validate(self):
        class _ToyValidator(InputValidator):
            EMITS: FrozenSet[WarningCategory] = frozenset({
                WarningCategory.CLIMATE_ENVELOPE_TAIL,
            })

            def validate(self, input_state):
                return InputValidationResult(valid=True)

        toy = _ToyValidator()
        self.assertEqual(
            toy.EMITS, frozenset({WarningCategory.CLIMATE_ENVELOPE_TAIL}),
        )

    def test_subclass_missing_validate_cannot_instantiate(self):
        class _Incomplete(InputValidator):
            EMITS: FrozenSet[WarningCategory] = frozenset()

        with self.assertRaises(TypeError):
            _Incomplete()


if __name__ == "__main__":
    unittest.main()
