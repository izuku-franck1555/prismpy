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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prismpy.validators.base import ValidationIssue
from prismpy.warnings.categories import WarningCategory


class CropEnvelope(BaseModel):
    """Frozen ECOCROP envelope for a single crop.

    Replaces the earlier ``Dict[str, float]`` shape so a
    downstream validator cannot mutate the envelope mid-check
    and so typo'd / extra fields fail-loud rather than being
    silently accepted. Strict-ordering validators mirror the
    AC-Q3-A-NaN contract on the bundled JSON loader.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    TMIN: float = Field(
        ...,
        description="Absolute minimum tolerable annual temperature (°C).",
    )
    TMAX: float = Field(
        ...,
        description="Absolute maximum tolerable annual temperature (°C).",
    )
    RMIN: float = Field(
        ..., ge=0,
        description="Minimum required annual rainfall (mm).",
    )
    RMAX: float = Field(
        ..., ge=0,
        description="Maximum tolerable annual rainfall (mm).",
    )

    @model_validator(mode="after")
    def _validate_ordering(self) -> "CropEnvelope":
        if not self.TMIN < self.TMAX:
            raise ValueError(
                f"CropEnvelope: TMIN ({self.TMIN}) must be "
                f"strictly less than TMAX ({self.TMAX})."
            )
        if not self.RMIN < self.RMAX:
            raise ValueError(
                f"CropEnvelope: RMIN ({self.RMIN}) must be "
                f"strictly less than RMAX ({self.RMAX})."
            )
        return self


class ZoneAggregate(BaseModel):
    """Frozen per-zone climate aggregates consumed by the
    Stage 1 wizard-time envelope check.

    Replaces the earlier ``Dict[str, float]`` shape so missing
    keys (notably ``n_cell_days``) cannot silently default to
    zero and silence the precipitation / thermal verdict
    behind a sample-quality warning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    p25: float = Field(
        ..., ge=0,
        description="Zone P25 of per-cell annual mean precip (mm/yr).",
    )
    p50: float = Field(
        ..., ge=0,
        description="Zone P50 of per-cell annual mean precip (mm/yr).",
    )
    p75: float = Field(
        ..., ge=0,
        description="Zone P75 of per-cell annual mean precip (mm/yr).",
    )
    p10_extreme_tmin: float = Field(
        ...,
        description=(
            "Zone P10 of per-cell extreme tmin (the 30-yr "
            "single coldest day per cell), in °C."
        ),
    )
    p90_extreme_tmax: float = Field(
        ...,
        description=(
            "Zone P90 of per-cell extreme tmax (the 30-yr "
            "single hottest day per cell), in °C."
        ),
    )
    n_cell_days: int = Field(
        ..., ge=0,
        description=(
            "Total cell-days contributing to the zone aggregate. "
            "Compared to the AC-Q2-E threshold by the validator. "
            "Required: a missing value here would silence the "
            "envelope verdict behind a sample-quality warning."
        ),
    )

    @model_validator(mode="after")
    def _validate_ordering(self) -> "ZoneAggregate":
        if not (self.p25 <= self.p50 <= self.p75):
            raise ValueError(
                f"ZoneAggregate: precip percentiles must satisfy "
                f"P25 <= P50 <= P75; got P25={self.p25}, "
                f"P50={self.p50}, P75={self.p75}."
            )
        if self.p10_extreme_tmin > self.p90_extreme_tmax:
            raise ValueError(
                f"ZoneAggregate: P10 extreme tmin "
                f"({self.p10_extreme_tmin}) must not exceed "
                f"P90 extreme tmax ({self.p90_extreme_tmax}); an "
                f"inverted aggregate indicates swapped variables "
                f"or unit corruption upstream."
            )
        return self


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
    crop_envelope: CropEnvelope = Field(
        ...,
        description=(
            "ECOCROP envelope for the crop. Frozen Pydantic "
            "model; see :class:`CropEnvelope`."
        ),
    )
    zone_aggregates: Dict[str, ZoneAggregate] = Field(
        ...,
        min_length=1,
        description=(
            "Per-zone climate aggregates keyed by Köppen-Geiger "
            "zone code (e.g. 'BSh', 'Aw'). At least one zone "
            "is required so an empty input does not return a "
            "false-green preflight; an empty region or failed "
            "KG lookup must surface as an error before reaching "
            "the validator."
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
