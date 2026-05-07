"""Scenario package metadata schema (Sprint G AC-G-3 + AC-G-4).

Climate scenario packages pair a baseline (observed climate) with one
or more projections (GCM-derived bias-corrected daily climate). Within
a scenario set, every package is identical to the baseline EXCEPT for
``weather/``, ``manifest.scenario``, and ``manifest.temporal``. The
``ScenarioBlock`` schema below captures the projection-specific
metadata that lives at ``manifest.scenario`` so the validator,
provenance audit, and downstream consumers all read the same shape.

Key invariants:

* The block is OPTIONAL outside scenario contexts (codex H3 absorption
  per Draft 5). Existing observed-climate manifests do not carry a
  ``scenario`` key today and continue to validate cleanly — the
  validator only enforces the schema when the key is present.
* ``ScenarioRole`` and ``BiasCorrectionMethod`` are CLOSED enums.
  Out-of-domain values raise ``ValidationError`` at the boundary, not
  at the consumer (durable §6.4 schema-layer discipline).
* Numeric bounds match the strictest downstream consumer per durable
  §6.4: ``time_slice_start/end`` ∈ [1900, 2200]; ``co2_ppm`` ∈
  [200.0, 2000.0].
* AC-G-10 is enforced as a post-validator on ``ScenarioBlock`` itself
  (per Builder MED-1 absorption — prismpy's ``Manifest`` is a dict
  produced by ``create_manifest()``, not a Pydantic class). When
  ``co2_ppm`` is set, ``co2_ppm_provenance`` MUST be a non-empty,
  non-whitespace string. Empty / whitespace / None raises
  ``MissingProvenanceError`` at construction time.

The ``BASE = "baseline"`` enum value (NOT ``"base"``) follows codex
LOW-1 absorption — the longer form is more durable for future UI/API
consumers that surface the role to researchers.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prismpy.standards.co2_ppm import (
    CO2_PPM_BY_SCENARIO_PERIOD,
    CO2ProvenanceMismatchError,
    co2_ppm_matches_canonical,
    get_co2_ppm_with_provenance,
)


# ── AC-G-4 closed enums ──────────────────────────────────────────────


class ScenarioRole(str, Enum):
    """Closed enum for the scenario block's ``scenario_role`` field.

    ``BASE`` ("baseline") marks the observed-climate package; every
    projection in the same set carries ``baseline_reference_label``
    pointing back to the baseline's ``scenario_label``. ``SCENARIO``
    is reserved for non-GCM projections (e.g., user-defined alternate
    climate paths) — Sprint G ships GCM-derived projections only;
    Sprint H+ may activate ``SCENARIO``. ``PROJECTION`` is the GCM-
    derived projection role.
    """

    BASE = "baseline"
    SCENARIO = "scenario"
    PROJECTION = "projection"


class BiasCorrectionMethod(str, Enum):
    """Closed enum for the scenario block's ``bias_correction_method``
    field.

    ``NONE`` is reserved for observed-climate baselines (no bias
    correction applied because the data is already observed).
    ``DELTA_METHOD`` / ``QUANTILE_MAPPING`` / ``TREND_PRESERVING`` are
    the three ISIMIP3b production bias-correction algorithms.
    ``UNKNOWN`` is a null sentinel for legacy / external-source
    packages — F-G-3 forbids it in shipped scenario packages
    (``validate_scenario_set(mode=SHIP)`` rejects). Per AC-G-6
    + Draft 5 mode-disambiguation, ``mode=LEGACY`` honors AC-G-6's
    ``unknown`` exclusion from the conflict-rule check.
    """

    NONE = "none"
    DELTA_METHOD = "delta_method"
    QUANTILE_MAPPING = "quantile_mapping"
    TREND_PRESERVING = "trend_preserving"
    UNKNOWN = "unknown"


# ── AC-G-3 ScenarioBlock Pydantic model ──────────────────────────────


_TIME_SLICE_MIN_YEAR = 1900
_TIME_SLICE_MAX_YEAR = 2200
_CO2_PPM_MIN = 200.0
_CO2_PPM_MAX = 2000.0


class MissingProvenanceError(ValueError):
    """Raised when a numeric scenario field lacks its provenance partner.

    Pydantic validation errors wrap this in ``ValidationError``; the
    typed exception is preserved as the ``__cause__`` so callers that
    discriminate on exception type can detect the AC-G-10 violation
    (``co2_ppm`` set + ``co2_ppm_provenance`` empty).
    """


class ScenarioBlock(BaseModel):
    """Manifest-level scenario metadata for paired baseline+projection sets.

    Schema bounds match the strictest downstream consumer per durable
    §6.4. Out-of-domain values are rejected at the boundary (Pydantic
    validation), not at the consumer's branch — a consumer-layer
    ``if not 200 <= ppm <= 2000`` guard duplicating these bounds is a
    schema-layer-discipline violation.
    """

    model_config = ConfigDict(
        # Forbid extra fields so a typo at the producer side surfaces
        # as ValidationError instead of being silently dropped at
        # consumer read time.
        extra="forbid",
        # Validate every field on assignment so a post-construction
        # mutation cannot bypass the bounds.
        validate_assignment=True,
        # ScenarioRole / BiasCorrectionMethod are str enums so the
        # serialized form stays readable (``"baseline"`` not
        # ``ScenarioRole.BASE``).
        use_enum_values=True,
    )

    scenario_label: str = Field(
        ..., min_length=1, description="Unique identifier for this package."
    )
    scenario_role: ScenarioRole = Field(
        ..., description="Role within a scenario set."
    )
    gcm_source: str = Field(
        ...,
        min_length=1,
        description=(
            "GCM identifier (e.g., 'gfdl-esm4', 'ipsl-cm6a-lr'). "
            "Sprint G primary core ensemble enforced at the validator "
            "boundary, not at the schema, so external-source packages "
            "with non-primary GCMs can still pass schema validation in "
            "ValidationMode.LEGACY."
        ),
    )
    rcp_or_ssp: str = Field(
        ...,
        min_length=1,
        description=(
            "Climate scenario identifier (e.g., 'ssp245', 'ssp585'). "
            "Closed-enum tightening deferred to Sprint H+ wizard sprint "
            "per codex LOW-2 absorption."
        ),
    )
    time_slice_start: int = Field(
        ...,
        ge=_TIME_SLICE_MIN_YEAR,
        le=_TIME_SLICE_MAX_YEAR,
        description="Inclusive start year of the projection window.",
    )
    time_slice_end: int = Field(
        ...,
        ge=_TIME_SLICE_MIN_YEAR,
        le=_TIME_SLICE_MAX_YEAR,
        description=(
            "Inclusive end year of the projection window. Cross-field "
            "validator asserts >= time_slice_start."
        ),
    )
    baseline_reference_label: str = Field(
        ...,
        min_length=1,
        description=(
            "scenario_label of the paired baseline package. The "
            "validator enforces this matches the baseline's label."
        ),
    )
    bias_correction_method: BiasCorrectionMethod = Field(
        ..., description="Bias-correction algorithm applied."
    )
    co2_ppm: float = Field(
        ...,
        ge=_CO2_PPM_MIN,
        le=_CO2_PPM_MAX,
        description=(
            "Atmospheric CO2 concentration in parts per million. "
            "AC-G-9 Layer 2 cross-checks this value against the "
            "canonical lookup for (rcp_or_ssp, time_slice)."
        ),
    )
    co2_ppm_provenance: Optional[str] = Field(
        default=None,
        description=(
            "Citation string for co2_ppm. Mandatory in shipped scenario "
            "packages (AC-G-10). Empty / whitespace / None raises "
            "MissingProvenanceError at construction."
        ),
    )
    scenario_bias_correction_provenance: Optional[str] = Field(
        default=None,
        description=(
            "Version-pinned bias-correction provenance string sourced "
            "from ISIMIP dataset metadata at fetch time. Format: "
            "'<method_name> v<method_version> against <reference_dataset> "
            "v<reference_version>' "
            "(e.g., 'ISIMIP3BASD v2.5.0 quantile-mapping against W5E5 "
            "v2.0'). Mandatory when bias_correction_method != 'none' per "
            "AC-G-11 honest-signal contract; empty / whitespace / None "
            "with a non-none method at ship time raises "
            "MissingProvenanceError. NONE method (observed baseline) "
            "is exempt."
        ),
    )

    @field_validator(
        "co2_ppm_provenance",
        "scenario_bias_correction_provenance",
        mode="before",
    )
    @classmethod
    def _strip_provenance_whitespace(cls, value: Optional[str]) -> Optional[str]:
        """Normalize whitespace-only provenance to None so the
        post-validator below treats it the same as a missing field
        (per AC-G-10 §10.3 acceptance — ' ' counts as empty)."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @model_validator(mode="after")
    def _check_time_slice_ordering(self) -> "ScenarioBlock":
        """Cross-field invariant: time_slice_end >= time_slice_start."""
        if self.time_slice_end < self.time_slice_start:
            raise ValueError(
                f"time_slice_end ({self.time_slice_end}) must be >= "
                f"time_slice_start ({self.time_slice_start})"
            )
        return self

    @model_validator(mode="after")
    def _check_co2_provenance_pairing(self) -> "ScenarioBlock":
        """AC-G-10 enforcement: when ``co2_ppm`` is set (always set on
        a constructed ScenarioBlock since the field is required),
        ``co2_ppm_provenance`` MUST be a non-empty, non-whitespace
        string. Empty / whitespace / None raises
        ``MissingProvenanceError`` at construction.

        The honest-signal contract per ``feedback_no_data_cooking.md``
        + Draft 5 CC-G-3: every numeric scenario field has a provenance
        partner. No bare numbers without provenance attribution for
        Dr. Kofi's audit trail.
        """
        if not self.co2_ppm_provenance:
            raise MissingProvenanceError(
                "co2_ppm_provenance is mandatory when co2_ppm is set. "
                "Per AC-G-10 / CC-G-3, every numeric scenario field "
                "needs a provenance string for audit traceability."
            )
        return self

    @model_validator(mode="after")
    def _check_bias_correction_provenance_pairing(self) -> "ScenarioBlock":
        """AC-G-11 enforcement: when ``bias_correction_method`` is not
        ``NONE``, ``scenario_bias_correction_provenance`` MUST be a
        non-empty, non-whitespace string. Empty / whitespace / None
        raises ``MissingProvenanceError``.

        ``BiasCorrectionMethod.NONE`` (observed-climate baseline) is
        exempt because there is no bias-correction algorithm applied —
        the observed data is observed; provenance is the climate-source
        provenance, not a bias-correction citation.

        Honest-signal contract per ``feedback_no_data_cooking.md``:
        every projection that applied a bias-correction algorithm must
        carry the version-pinned citation so Dr. Kofi can audit the
        full transformation chain (which version of which algorithm
        against which reference dataset).
        """
        # ``use_enum_values=True`` config means
        # ``self.bias_correction_method`` is the str value, not the
        # enum instance. Compare against the enum's string value.
        if self.bias_correction_method == BiasCorrectionMethod.NONE.value:
            return self
        if not self.scenario_bias_correction_provenance:
            raise MissingProvenanceError(
                "scenario_bias_correction_provenance is mandatory when "
                f"bias_correction_method={self.bias_correction_method!r} "
                "(non-NONE). Per AC-G-11 honest-signal contract, every "
                "bias-correction algorithm application must carry a "
                "version-pinned provenance citation. Format: "
                "'<method_name> v<method_version> against "
                "<reference_dataset> v<reference_version>'."
            )
        return self

    @model_validator(mode="after")
    def _check_co2_canonical_agreement(self) -> "ScenarioBlock":
        """AC-G-9 Layer 2 — semantic check against the canonical
        lookup at :data:`prismpy.standards.co2_ppm.CO2_PPM_BY_SCENARIO_PERIOD`.

        For a registered ``(rcp_or_ssp, time_slice)`` tuple, asserts:

        * ``math.isclose(co2_ppm, canonical_value, rel_tol=1e-9)`` —
          per pass-2 MEDIUM-Rebase-3 the tolerance catches deliberate
          cooking while absorbing JSON-roundtrip rounding noise.
        * ``co2_ppm_provenance == canonical_provenance`` — exact
          string match. Paraphrased citations fail loud.

        The lookup key is built from ``rcp_or_ssp.upper()`` so
        ``"ssp245"`` (the field description's lowercase example) maps
        to ``"SSP245"`` (the canonical table's uppercase key) without
        the user-input case becoming a silent Layer 2 bypass.

        When ``(rcp_or_ssp, time_slice)`` is NOT in the canonical
        table (e.g., a non-primary-core-ensemble scenario shipped via
        ``ValidationMode.LEGACY``), Layer 2 skips silently — Layer 1
        substrate enforcement + AC-G-10 provenance-pairing still apply,
        and the validator's ``mode=SHIP`` rejects unregistered
        scenarios at the ship-validation boundary.
        """
        scenario_key = self.rcp_or_ssp.upper()
        time_slice_key = (self.time_slice_start, self.time_slice_end)
        lookup_key = (scenario_key, time_slice_key)
        if lookup_key not in CO2_PPM_BY_SCENARIO_PERIOD:
            # Not a primary-core-ensemble scenario — Layer 2 does not
            # apply. Layer 1 + AC-G-10 still enforce the substrate-
            # and-pairing invariants.
            return self
        expected_ppm, expected_provenance = CO2_PPM_BY_SCENARIO_PERIOD[
            lookup_key
        ]
        if not co2_ppm_matches_canonical(self.co2_ppm, expected_ppm):
            raise CO2ProvenanceMismatchError(
                f"co2_ppm={self.co2_ppm} disagrees with canonical "
                f"value {expected_ppm} for scenario {scenario_key} "
                f"time_slice {time_slice_key}. Per AC-G-9 Layer 2, "
                "the value MUST match the canonical lookup at "
                "prismpy.standards.co2_ppm within rel_tol=1e-9.",
                scenario=scenario_key,
                time_slice=time_slice_key,
                observed_co2_ppm=self.co2_ppm,
                expected_co2_ppm=expected_ppm,
                observed_provenance=self.co2_ppm_provenance,
                expected_provenance=expected_provenance,
            )
        if self.co2_ppm_provenance != expected_provenance:
            raise CO2ProvenanceMismatchError(
                f"co2_ppm_provenance does not match canonical "
                f"citation for scenario {scenario_key} time_slice "
                f"{time_slice_key}. Paraphrased / approximate "
                "provenance fails loud per AC-G-9 Layer 2.",
                scenario=scenario_key,
                time_slice=time_slice_key,
                observed_co2_ppm=self.co2_ppm,
                expected_co2_ppm=expected_ppm,
                observed_provenance=self.co2_ppm_provenance,
                expected_provenance=expected_provenance,
            )
        return self


# ── AC-G-7b ProjectionClimateMeta (per-cell sidecar) ─────────────────


class ProjectionClimateMeta(BaseModel):
    """Per-cell sidecar metadata for the ACEA / SARRA-Py projection path.

    Sprint G AC-G-7b emits a ``.meta.json`` file alongside each per-cell
    climate artifact (ACEA pickle or SARRA-Py per-variable directory).
    The sidecar carries the projection-source provenance fields a
    consumer needs to read off-disk without re-loading the parent
    package's ``manifest.scenario`` block.

    Per Draft 5 line 174 + LOW-Pass4-6 in
    ``SPRINT-G-IMPLEMENTATION-DISCIPLINE-NOTES.md``: the schema lives
    in ``prismpy.models.scenario`` parallel to ``ScenarioBlock``; both
    writer + reader route through this model so a typo at the producer
    side surfaces as ``ValidationError`` rather than silently dropping
    at consumer read time.

    Cell vs variable identifier:

    * AC-G-7b ACEA writer populates ``cell_id`` (one sidecar per
      30-arcmin cell pickle).
    * AC-G-7c SARRA-Py writer populates ``variable`` (one sidecar per
      per-variable GeoTIFF directory).

    Both are optional; at least one MUST be set for the sidecar to be
    semantically meaningful, but the schema permits both empty for
    edge-case constructor flexibility (the writers themselves enforce
    the AC-specific constraint at emission time).
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )

    gcm_source: str = Field(
        ..., min_length=1, description="GCM identifier (e.g., 'gfdl-esm4')."
    )
    bias_correction_method: BiasCorrectionMethod = Field(
        ..., description="Bias-correction algorithm applied."
    )
    time_slice_start: int = Field(
        ..., ge=_TIME_SLICE_MIN_YEAR, le=_TIME_SLICE_MAX_YEAR
    )
    time_slice_end: int = Field(
        ..., ge=_TIME_SLICE_MIN_YEAR, le=_TIME_SLICE_MAX_YEAR
    )
    cell_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="ACEA per-cell sidecar — 30-arcmin cell ID.",
    )
    variable: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "SARRA-Py per-variable sidecar — variable name "
            "(e.g., 'tasmax', 'pr')."
        ),
    )
    scenario_label: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "Optional pairing reference back to the parent scenario "
            "block. When set, must equal the parent manifest's "
            "scenario.scenario_label."
        ),
    )

    @model_validator(mode="after")
    def _check_time_slice_ordering(self) -> "ProjectionClimateMeta":
        if self.time_slice_end < self.time_slice_start:
            raise ValueError(
                f"time_slice_end ({self.time_slice_end}) must be >= "
                f"time_slice_start ({self.time_slice_start})"
            )
        return self


__all__ = [
    "ScenarioRole",
    "BiasCorrectionMethod",
    "ScenarioBlock",
    "ProjectionClimateMeta",
    "MissingProvenanceError",
    "CO2ProvenanceMismatchError",
]
