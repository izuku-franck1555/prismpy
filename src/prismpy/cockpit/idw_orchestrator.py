"""IDW orchestrator — drives per-cell interpolation for the cockpit.

Sprint E.3 AC-E3-11 + Draft 2 codex CA-5 + builder CA-2 (location
LOCK to ``prismpy/cockpit/idw_orchestrator.py`` per sibling-pattern
rationale: cockpit/observed_values_writer + per_run_snapshot
precedent; single Phase 1 ship cadence; ``idw_interpolation``
consumer + sidecar writer + cell_roster comparator all run in
same import scope) + CMS CA-1 BLOCKING (per-platform IDW radius
registry).

The orchestrator's job is to compose the IDW substrate (the engine
at :mod:`prismpy.harmonize.idw_interpolation` + the spatial index
at :mod:`prismpy.spatial_index`) with the cockpit's per-cell
decision context (target cell + decision_id + Köppen zone) and
emit :class:`InterpolatedCellRecord` instances ready for
persistence on a :class:`CellDecisionRecord`.

**Platform-aware radius dispatch** (CMS CA-1 BLOCKING absorbed):
the orchestrator reads
:func:`prismpy.standards.idw_methods.get_idw_radius_for_platform`
to pick the per-platform search radius rather than using the
universal :data:`IDW_DEFAULT_R = 15.0`. ACEA cells at 50 km grid
need 100 km radius (≈2× cell size) to capture decorrelation
envelope; with the prior universal default ACEA cells silently
failed IDW (zero candidates within 15 km at 50 km grid spacing).

**Per-record method-provenance** (post-Draft 4 codex HIGH-2
absorbed): the InterpolatedCellRecord schema extension at
``prismpy/models/interpolated_cell.py`` carries the per-record
``radius_km`` / ``k`` / ``weight_power`` so the methods-text
generator reads the actual numeric values rather than parsing the
legacy literal pattern. The orchestrator persists the platform's
canonical radius on each record so a 4-platform package mix
records the right radius per cell.

**Read-only contract** (AC-E3-11 sub-1 + 4): the orchestrator does
NOT mutate ``cockpit_observed_values.json`` or the affordance
routing context. Per durable §6.4 schema-layer discipline + the
behavioral pin at
``tests/structural/test_idw_orchestrator_consumer_reads_only.py``,
the inputs are read-only.

**Empty-observed-values honest signal** (AC-E3-11 sub-3 +
``feedback_no_data_cooking.md``): if the observed-values fixture
is empty (no climate / soil aggregates), the orchestrator returns
an empty result list AND surfaces a warning rather than silently
producing degraded interpolations. The downstream affordance-
routing rule then treats the empty observed-values case as
"impute via skip" rather than "impute via zero-neighbour
interpolate".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from uuid import UUID

from prismpy.harmonize.idw_interpolation import (
    Cell,
    InsufficientNeighborsError,
    InterpolationResult,
    interpolate_idw,
)
from prismpy.koppen.zones import KoppenZone
from prismpy.models.interpolated_cell import (
    CellID,
    InterpolatedCellRecord,
)
from prismpy.spatial_index import SpatialIndex
from prismpy.standards.caveat_codes import CaveatCode
from prismpy.standards.idw_methods import (
    IDW_CANONICAL_METHOD_LITERAL,
    IDW_DEFAULT_K,
    IDW_DEFAULT_W,
    get_idw_radius_for_platform,
)


_LOGGER = logging.getLogger(__name__)


# Shepard 1968 — the foundational IDW reference. Pinned here for
# the InterpolatedCellRecord.method_doi field; consumers read this
# constant rather than restating the DOI inline.
_SHEPARD_1968_DOI = "10.1145/800186.810616"


@dataclass(frozen=True)
class IdwOrchestrationTask:
    """One per-cell interpolation task fed to the orchestrator.

    The orchestrator's caller (the prismweb-side commit-decision
    handler at AC-E3-19, or the evaluator harness for Drill-E3-D)
    composes a list of these tasks from the affordance-routing
    output + observed_values + per-cell decision IDs. Each task
    fully specifies one IDW invocation; the orchestrator's job is
    purely to execute the dispatch + build the record.

    Attributes:
        target_cell: The cell whose value is being imputed.
        candidate_cells: Pool of candidate neighbours. Caller is
            responsible for filtering to cells with non-null
            observed values for the variable being interpolated;
            the orchestrator does NOT do the variable-specific
            filtering (the engine just operates on Cell.value).
        affected_zone_code: Köppen zone code the target cell falls
            in. Carried through to the record's
            ``affected_zone_code`` field.
        decision_id: The enclosing CellDecisionRecord's UUID — the
            record's ``applied_at_decision_id`` mirrors this.
        caveat_codes: Caveat codes applicable per the
            :mod:`prismpy.standards.interpolation_caveats` rule
            (caller resolves zone × check_id → caveats).
        platform_value: The canonical ``Platform.*.value`` string.
            Drives the radius dispatch via
            :func:`get_idw_radius_for_platform`.
    """

    target_cell: Cell
    candidate_cells: Tuple[Cell, ...]
    affected_zone_code: KoppenZone
    decision_id: UUID
    caveat_codes: Tuple[CaveatCode, ...]
    platform_value: str


@dataclass(frozen=True)
class IdwOrchestrationResult:
    """One per-cell interpolation outcome.

    ``record`` is non-None on the success path; ``error`` is non-
    None on the InsufficientNeighborsError path (zero candidates
    within radius — should be routed to skip pre-orchestrator per
    AC-E2-3, but the orchestrator catches it as a defensive
    fallback). Exactly one of ``record`` / ``error`` is populated
    (XOR invariant per codex round 1 MEDIUM CA absorbed); the
    ``__post_init__`` validator enforces this so a future caller
    can't ignore the error path by reading ``record`` without an
    error check.

    ``decision_id`` echoes the input task's ``decision_id`` so
    callers can correlate results back to decisions without re-
    walking the task list.
    """

    decision_id: UUID
    record: Optional[InterpolatedCellRecord]
    error: Optional[str]

    def __post_init__(self) -> None:
        """Enforce XOR — exactly one of ``record`` / ``error`` is
        populated. Both-None and both-populated are bug shapes
        per codex round 1 MEDIUM CA absorbed."""
        record_present = self.record is not None
        error_present = self.error is not None
        if record_present == error_present:
            raise ValueError(
                f"IdwOrchestrationResult XOR invariant violated: "
                f"exactly one of record / error must be populated. "
                f"Got record_present={record_present}, "
                f"error_present={error_present}. "
                f"decision_id={self.decision_id}"
            )

    @property
    def is_success(self) -> bool:
        """True iff the task produced a record (no error). Lets
        callers branch on ``result.is_success`` rather than
        re-checking the XOR pair."""
        return self.record is not None


def run_idw_orchestrator(
    tasks: List[IdwOrchestrationTask],
) -> List[IdwOrchestrationResult]:
    """Execute IDW interpolation across a batch of per-cell tasks.

    For each task, the orchestrator:

    1. Resolves the platform-specific radius via
       :func:`get_idw_radius_for_platform`.
    2. Dispatches :func:`interpolate_idw` with the platform radius
       + canonical k / weight_power.
    3. Builds the :class:`InterpolatedCellRecord` using the
       canonical post-E.3 ``"idw"`` method literal + the actual
       per-record numeric parameters (radius_km / k /
       weight_power) per AC-E3-11 sub-2 + codex HIGH-2 absorbed.

    Args:
        tasks: List of per-cell interpolation tasks.

    Returns:
        List of result records — one per task. Order matches the
        input order so callers can index by position.

    The orchestrator does NOT mutate input tasks or candidate
    cells; the dataclasses are frozen and the engine's API is
    read-only. Empty input → empty output (with a warning).
    """
    if not tasks:
        _LOGGER.warning(
            "IDW orchestrator invoked with empty task list — "
            "no interpolations dispatched. The honest-signal floor "
            "per feedback_no_data_cooking.md surfaces this as a "
            "warning rather than a silent zero-interpolation case."
        )
        return []

    results: List[IdwOrchestrationResult] = []
    for task in tasks:
        result = _run_single_task(task)
        results.append(result)
    return results


def _run_single_task(
    task: IdwOrchestrationTask,
) -> IdwOrchestrationResult:
    """Execute one task — dispatch IDW + build the record."""
    radius_km = get_idw_radius_for_platform(task.platform_value)

    try:
        interp_result: InterpolationResult = interpolate_idw(
            task.target_cell,
            list(task.candidate_cells),
            k=IDW_DEFAULT_K,
            radius_km=radius_km,
            weight_power=IDW_DEFAULT_W,
        )
    except InsufficientNeighborsError as exc:
        # Defensive fallback per AC-E2-3 routing: this case should
        # have been routed to skip before reaching the orchestrator,
        # but if it gets here we surface it as a result-level error
        # rather than crashing the entire batch.
        return IdwOrchestrationResult(
            decision_id=task.decision_id,
            record=None,
            error=str(exc),
        )

    record = InterpolatedCellRecord(
        interpolation_method=IDW_CANONICAL_METHOD_LITERAL,
        source_cells=interp_result.source_cells,
        uncertainty_ci_lower=interp_result.ci_lower,
        uncertainty_ci_upper=interp_result.ci_upper,
        method_doi=_SHEPARD_1968_DOI,
        applied_at_decision_id=task.decision_id,
        affected_zone_code=task.affected_zone_code,
        caveat_codes=list(task.caveat_codes),
        radius_km=radius_km,
        k=IDW_DEFAULT_K,
        weight_power=IDW_DEFAULT_W,
    )
    return IdwOrchestrationResult(
        decision_id=task.decision_id,
        record=record,
        error=None,
    )


def build_spatial_index(cells: List[Cell]) -> Optional[SpatialIndex]:
    """Convenience wrapper — return a SpatialIndex for ``cells`` or
    None when the roster is empty.

    Centralises the empty-roster guard so callers don't all
    re-implement the empty-check + None-return pattern. The
    SpatialIndex constructor itself raises on empty input per the
    Sprint E.2 substrate contract; the orchestrator's empty-
    observed-values honest-signal path needs the empty case to
    return None instead of crashing.
    """
    if not cells:
        return None
    return SpatialIndex(cells)


__all__ = [
    "IdwOrchestrationResult",
    "IdwOrchestrationTask",
    "build_spatial_index",
    "run_idw_orchestrator",
]
