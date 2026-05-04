"""Per-zone sample-quality threshold + assessment.

Per Sprint E.0.5 AC-Q2-E. Replaces the bound-gen "minimum
cell count" threshold with a "minimum cell-days" threshold
that decouples cell count from temporal window. Future 60-yr
WMO climatological-normal windows admit zones with fewer
cells without code changes.

Threshold rationale (per AC-Q2-E-Justification): at
``n = 1,000,000`` cell-days the empirical 95% confidence-
interval half-width on P99 is ≈ 0.036 °C — conservative
~10× over the standard literature floor (n ≥ 10,000) and
defensible as (a) a round number, (b) future-proofing
against shorter windows + smaller zones, and (c) an
asymmetric false-negative bias toward honest signal (a
zone we mark insufficient that turns out to have been
fine is recoverable; the converse is not).

Used by:

* Bound-generation: at the per-zone aggregation step,
  insufficient zones return ``None`` from the bound
  generator (no entry in the bounds JSON).
* Cockpit: a Bucket 2 INFORMATIONAL warning surfaces
  zones with ``sample_quality='insufficient'`` via the
  ``WarningCategory.INSUFFICIENTLY_SAMPLED`` slot
  (shipped in Sprint E.0).

Per AC-Q2-E-b, every per-zone provenance record carries
the threshold value alongside the verdict so a reader of
the bounds JSON can audit "what threshold was applied" at
the time the bounds were generated, independent of any
later threshold ratchet.

Likely affected zones at the v1 threshold (per the research
doc): Csc, Dsc, Dsd, Dwd, EF — all small global areal
extent.
"""
from __future__ import annotations

from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Minimum cell-days required for a zone aggregate to count
# as sufficiently sampled per AC-Q2-E-a. Anti-mutation: a
# bound-gen change that re-introduces a "minimum cell count"
# threshold (instead of cell-days) breaks the AC-Q2-E
# decoupling claim; the structural test pins the constant
# at exactly 1,000,000.
MIN_CELL_DAYS_PER_ZONE: Final[int] = 1_000_000


class SampleQuality(str, Enum):
    """Two-state per-zone sample-quality verdict.

    ``sufficient`` — zone passes :data:`MIN_CELL_DAYS_PER_ZONE`;
    bound-gen produces a per-zone aggregate.

    ``insufficient`` — zone fails the threshold; bound-gen
    returns ``None`` for this zone (sentinel: no entry in
    the bounds JSON), and the cockpit surfaces a Bucket 2
    INFORMATIONAL warning at run-time when a project lands
    in an insufficient zone.
    """
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class ZoneSampleQuality(BaseModel):
    """Per-zone sample-quality provenance record.

    Per AC-Q2-E-b: every per-zone bound-gen record carries
    the cell-count + cell-day count + verdict + reason +
    threshold-applied so a bounds-file reader can audit the
    sufficiency decision against the threshold in force at
    bound-gen time (independent of any later threshold
    ratchet).

    The model is round-trip-safe via
    ``.model_dump_json() / .model_validate_json()``;
    embeddable inside the bounds JSON or as a sidecar
    fragment.
    """

    # frozen=True makes the record immutable after
    # construction so a downstream caller cannot mutate
    # ``sample_quality`` or ``n_cell_days`` after the
    # model_validator has run; without this, an enriched-
    # then-serialized record could emit a "sufficient"
    # verdict with sub-threshold cell-days and hide the
    # Bucket 2 INSUFFICIENTLY_SAMPLED warning. extra='forbid'
    # rejects typo'd / extra fields at construction.
    model_config = ConfigDict(extra="forbid", frozen=True)

    n_cells: int = Field(
        ..., ge=0,
        description=(
            "Number of cells contributing to the zone aggregate."
        ),
    )
    n_cell_days: int = Field(
        ..., ge=0,
        description=(
            "Total cell-days contributing to the zone aggregate "
            "(= sum over cells of the number of valid AgERA5 "
            "daily records per cell). The sufficiency threshold "
            "is applied to this count, not to ``n_cells`` alone."
        ),
    )
    sample_quality: SampleQuality = Field(
        ...,
        description=(
            "Sufficiency verdict per :data:`MIN_CELL_DAYS_PER_ZONE`. "
            "Caller (bound-gen) sets this consistently with "
            "n_cell_days; the model validator pins the relationship."
        ),
    )
    sample_quality_reason: str = Field(
        ...,
        description=(
            "Human-readable reason explaining the verdict. "
            "Anti-mutation: empty-string or missing reason "
            "breaks the cockpit Bucket 2 informational rendering."
        ),
        min_length=1,
    )
    threshold: int = Field(
        ..., ge=0,
        description=(
            "The threshold value applied at this assessment. "
            "Recorded so a bounds-file reader can audit "
            "sufficiency against the threshold in force at "
            "bound-gen time, even after a future ratchet."
        ),
    )

    @model_validator(mode="after")
    def _validate_verdict_matches_count(self) -> "ZoneSampleQuality":
        """Caller-supplied ``sample_quality`` must agree with the
        threshold check applied to ``n_cell_days``. A record
        with n_cell_days < threshold but sample_quality='sufficient'
        is incoherent and rejected fail-loud."""
        passes = self.n_cell_days >= self.threshold
        verdict_says_sufficient = (
            self.sample_quality == SampleQuality.SUFFICIENT
        )
        if passes != verdict_says_sufficient:
            raise ValueError(
                f"ZoneSampleQuality: sample_quality verdict "
                f"({self.sample_quality.value}) inconsistent with "
                f"n_cell_days ({self.n_cell_days}) vs threshold "
                f"({self.threshold}). The verdict must mirror "
                f"the threshold check fail-loud."
            )
        return self


def assess_zone_sample_quality(
    n_cells: int,
    n_cell_days: int,
    threshold: int = MIN_CELL_DAYS_PER_ZONE,
) -> ZoneSampleQuality:
    """Build a :class:`ZoneSampleQuality` record from cell counts.

    Bound-gen calls this once per zone after computing the
    cell + cell-day count; the returned record both carries
    the verdict and serializes into the per-zone provenance
    block of the bounds file. The threshold defaults to
    :data:`MIN_CELL_DAYS_PER_ZONE` (the AC-Q2-E-a constant)
    but the caller can override for tests or future ratchet
    scenarios.

    Raises :class:`ValueError` for negative inputs (Pydantic
    ``ge=0`` constraint).
    """
    if n_cell_days >= threshold:
        verdict = SampleQuality.SUFFICIENT
        reason = (
            f"Zone passes the {threshold:,} cell-days "
            f"threshold (n_cell_days={n_cell_days:,}, "
            f"n_cells={n_cells:,})."
        )
    else:
        verdict = SampleQuality.INSUFFICIENT
        reason = (
            f"Zone fails the {threshold:,} cell-days threshold "
            f"(n_cell_days={n_cell_days:,} < {threshold:,}, "
            f"n_cells={n_cells:,}); bound-gen will skip this "
            f"zone and the cockpit will surface a Bucket 2 "
            f"INSUFFICIENTLY_SAMPLED warning at run-time when "
            f"a project lands here."
        )
    return ZoneSampleQuality(
        n_cells=n_cells,
        n_cell_days=n_cell_days,
        sample_quality=verdict,
        sample_quality_reason=reason,
        threshold=threshold,
    )
