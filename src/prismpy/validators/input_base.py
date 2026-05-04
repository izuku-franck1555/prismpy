"""Input-validator base class for wizard-time pre-pipeline checks.

Per Sprint E.0.5 §Scope summary item #3 + builder Adj 6 +
team-lead Decision 2 (commit 8). Distinct from the existing
:class:`prismpy.validators.base.BaseValidator` output-
validator shape (which validates files written by translators
post-pipeline) — this ABC handles input-time predicate checks
at wizard launch: project config, region, crop, climate
envelope, KG zone.

Architectural note (mental-model split per durable lesson
#21 workflow-first): output validators check artifacts on
disk; input validators check pre-pipeline state. The two are
distinct surfaces with different concerns; subclassing them
under a single base would conflate two mental models. We
keep them parallel: :class:`BaseValidator` for outputs,
:class:`InputValidator` for inputs. The canonical
:class:`prismpy.validators.base.ValidationIssue` schema is
shared between both per CC-14 schema-layer convergence
(domain separation at validator-input layer; canonical
schema convergence at validator-output layer).

Subclasses declare ``EMITS`` as a frozenset of
:class:`prismpy.warnings.categories.WarningCategory` values
the validator can emit. F25-shape walker discipline
(future Sprint F walker can extend): every WarningCategory
emitted by ``validate()`` MUST appear in ``EMITS``. For
Sprint E.0.5 this is doc-only static metadata.

Sprint F + V2-23 work extracts the input-time checks now
in ``validators/scientific.py`` into class-based
:class:`InputValidator` subclasses; Sprint E.0.5 ships the
parallel base + the first two subclasses
(:class:`ClimateEnvelopeValidator` +
:class:`CropPhysiologicalValidator` skeleton).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, FrozenSet, List

from pydantic import BaseModel, ConfigDict, Field

from prismpy.validators.base import ValidationIssue
from prismpy.warnings.categories import WarningCategory


@dataclass
class InputValidationResult:
    """Result of an input-time validator's check.

    Distinct from :class:`prismpy.validators.base.ValidationResult`
    (output-validator scope: requires platform + output_dir +
    files_checked; those don't apply at wizard-launch time).
    The canonical :class:`ValidationIssue` schema is preserved
    per CC-14 so the cockpit can consume issues from both
    output- and input-validators uniformly.
    """
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def n_info(self) -> int:
        return sum(1 for i in self.issues if i.severity == "info")


class InputValidationContext(BaseModel):
    """Pre-pipeline state passed to :meth:`InputValidator.validate`.

    Holds the crop selection, the crop's ECOCROP envelope,
    and the per-zone climate aggregates that bound-gen
    produced (or that the wizard pre-fetched). Frozen after
    construction so a downstream validator can't mutate the
    input state mid-check; ``extra='forbid'`` rejects
    typo'd / unrecognized fields fail-loud per the substrate's
    honest-signal contract.

    Sprint F adds per-cell daily series + per-cell soil data;
    extending the model is straightforward (Pydantic v2 model
    inheritance) without breaking the E.0.5 skeleton consumers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    crop_name: str = Field(
        ...,
        min_length=1,
        description=(
            "Crop key for ECOCROP envelope lookup (e.g. "
            "'maize', 'rice'). Matches keys in the bundled "
            "``ecocrop_envelopes.json``."
        ),
    )
    crop_envelope: Dict[str, float] = Field(
        ...,
        description=(
            "ECOCROP envelope for the crop: TMIN/TMAX/RMIN/RMAX "
            "(°C / mm). Loaded via "
            ":func:`prismpy.koppen.load_ecocrop_envelopes`."
        ),
    )
    zone_aggregates: Dict[str, Dict[str, float]] = Field(
        ...,
        description=(
            "Per-zone climate aggregates keyed by Köppen-Geiger "
            "zone code (e.g. 'BSh', 'Aw'). Each value carries "
            "p25/p50/p75 (precip mm/yr), "
            "p10_extreme_tmin/p90_extreme_tmax (°C), and "
            "n_cell_days for the AC-Q2-E sample-quality check."
        ),
    )
    min_cell_days_per_zone: int = Field(
        default=1_000_000,
        ge=0,
        description=(
            "Sample-quality threshold (defaults to "
            ":data:`prismpy.bounds.MIN_CELL_DAYS_PER_ZONE`)."
        ),
    )


class InputValidator(ABC):
    """Abstract base for input-time wizard-pre-pipeline validators.

    Subclasses declare:

    * ``EMITS`` — class-level :class:`frozenset` of
      :class:`WarningCategory` values the validator can emit.
      F25-shape walker discipline: every category that
      :meth:`validate` produces MUST appear in ``EMITS``.
      Static metadata; readable from the class without
      instantiation.
    * ``validate(input_state)`` — main check entry point;
      returns an :class:`InputValidationResult` carrying any
      :class:`ValidationIssue` instances found.
    """

    EMITS: ClassVar[FrozenSet[WarningCategory]] = frozenset()

    @abstractmethod
    def validate(
        self, input_state: InputValidationContext,
    ) -> InputValidationResult:
        """Run the validator's checks on the given input state.

        Returns an :class:`InputValidationResult` carrying any
        issues found. Subclasses MUST emit only WarningCategory
        values listed in ``EMITS``.
        """
        raise NotImplementedError
