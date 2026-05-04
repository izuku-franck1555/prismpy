"""CropPhysiologicalValidator skeleton for Sprint F per-cell checks.

Per Sprint E.0.5 §Scope summary item #3 + warning-auditor
probe-C-1 LOW counter-add. Empty skeleton in this sprint;
Sprint F populates :meth:`validate` with the per-cell ECOCROP
TMIN/TMAX/RMIN/RMAX tolerance check that emits:

* :data:`WarningCategory.CROP_PHYSIOLOGY_VIOLATION` (Bucket 3
  EXCLUDE) when a per-cell daily value violates ECOCROP
  tolerance — Stage 2 per-cell granularity.
* :data:`WarningCategory.CROP_REGION_MISMATCH` (Bucket 3
  EXCLUDE) when the zone-level (Stage 1) ECOCROP envelope
  comparison says incompatible. (Stage 1 logic lives in
  :mod:`prismpy.validators.climate_envelope` per
  AC-Q3-A-a/b/c; the INCOMPATIBLE verdict gets surfaced
  through this validator's Sprint F implementation rather
  than ClimateEnvelopeValidator's EMITS.)

For Sprint E.0.5 the validator is callable but emits no
issues. The ``EMITS`` frozenset is the static metadata pin
that the Sprint F walker can extend to enforce per-validator-
EMITS-discipline.
"""
from __future__ import annotations

from prismpy.validators.input_base import (
    InputValidationContext,
    InputValidationResult,
    InputValidator,
)
from prismpy.warnings.categories import WarningCategory


class CropPhysiologicalValidator(InputValidator):
    """Sprint F per-cell ECOCROP tolerance + Stage 1 zone-mismatch
    surface (skeleton in Sprint E.0.5).

    Sprint F populates :meth:`validate` with the per-cell
    physiology check; for E.0.5 the validator's ``validate``
    returns no issues so a wizard pre-validation pipeline
    can wire it in without breaking. The class shape +
    ``EMITS`` frozenset are the contract pins that survive
    into Sprint F.
    """

    EMITS = frozenset({
        WarningCategory.CROP_PHYSIOLOGY_VIOLATION,
        WarningCategory.CROP_REGION_MISMATCH,
    })

    def validate(
        self, input_state: InputValidationContext,
    ) -> InputValidationResult:
        """Sprint E.0.5 skeleton: returns no issues.

        Sprint F populates the per-cell ECOCROP check + the
        Stage 1 INCOMPATIBLE-verdict mapping into
        :data:`WarningCategory.CROP_REGION_MISMATCH`.
        """
        return InputValidationResult(
            valid=True,
            issues=[],
            metadata={
                "validator": "CropPhysiologicalValidator",
                "sprint": "E.0.5",
                "status": "skeleton; Sprint F populates body",
            },
        )
