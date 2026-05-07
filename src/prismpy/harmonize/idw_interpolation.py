"""Inverse-distance-weighted spatial interpolation for the cockpit.

Sprint E.2 AC-E2-2. The cockpit's INTERPOLATABLE bucket calls
``interpolate_idw`` to compute an imputed value + 95 % confidence
interval for each cell flagged with a value-range or region-bound
warning. The function is pure (no I/O, no global state) so it can
be called from the affordance-routing preflight at modal-open time
or from the bulk-commit transaction interchangeably.

Method (per AC-E2-19 canonical constants):

* ``k = 4`` nearest neighbours within ``R = 15 km``.
* Inverse-distance weighting with ``w = 1 / d²``.
* 95 % CI half-width ``= 1.96 × s_w / sqrt(n)`` where ``s_w`` is the
  weighted standard deviation across the contributing neighbours.

Edge cases (per AC-E2-2 sub-criteria + Drills E / F / F2 / R):

* **Self-match (d = 0)** — when ``target_cell.cell_id`` appears in
  ``candidate_cells``, the matching candidate is skipped before the
  distance calculation. Semantic: we interpolate from NEIGHBOURS, not
  from the target itself; a hot self-match would also blow up the
  ``1 / d²`` weight.

* **Zero candidates within R** — raises ``InsufficientNeighborsError``.
  Per AC-E2-3 the affordance-routing rule catches this BEFORE the
  engine runs (when ``n_candidates_in_radius == 0`` the cell is
  routed to ``"skip"``); the engine raises only as a defensive
  guard for misuse.

* **k = 1 degraded path** — single neighbour produces zero-width CI
  (``ci_lower == ci_upper == value``). Honest signal: the
  ``degraded_due_to_insufficient_neighbours`` flag is True, and the
  AC-E2-7 methods-text generator emits the Phrase 2 caveat
  ("uninformative; zero-width by construction").

* **k = 2 / k = 3 degraded path** — ``degraded_due_to_insufficient_neighbours``
  is True; the CI formula is the same as k=4 with the smaller n,
  producing wider intervals reflecting fewer contributing samples.
  AC-E2-7 emits the Phrase 1 caveat.

* **k = 4 full path** — ``degraded_due_to_insufficient_neighbours``
  is False; the CI formula uses n=4.

Distance: great-circle via ``haversine_distance`` from
``prismpy.utils.gis_utils``. The cockpit's grid spacing is on the
order of 9 km; haversine accuracy across 15 km is well within the
~0.5 % bound that matters for IDW weights at this resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from prismpy.models.interpolated_cell import CellID
from prismpy.standards.idw_methods import (
    IDW_DEFAULT_K,
    IDW_DEFAULT_R,
    IDW_DEFAULT_W,
)
from prismpy.utils.gis_utils import haversine_distance


# ── Public types ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cell:
    """Minimal frozen cell representation for IDW.

    The IDW engine only needs (cell_id, lat, lon, value); fuller
    cell schemas (``prismpy.cells.schema``) carry far more fields
    that aren't relevant to the interpolation math. Using a
    purpose-specific frozen dataclass keeps the engine signature
    light + makes the candidate-list builder's responsibility
    explicit.
    """

    cell_id: CellID
    lat: float
    lon: float
    value: float


@dataclass(frozen=True)
class InterpolationResult:
    """Output of one IDW interpolation call.

    Attributes:
        value: Imputed value (weighted mean of contributing
            neighbours).
        ci_lower / ci_upper: 95 % CI bounds for the imputed value.
            ``ci_lower == ci_upper == value`` for the k=1 degraded
            path (zero-width by construction).
        source_cells: Cell IDs of the contributing neighbours
            (length == n; ordered by distance ascending).
        n_neighbors_in_radius: Count of candidates within R after
            self-filter; equals ``len(source_cells)`` after k-cap.
        degraded_due_to_insufficient_neighbors: True when fewer
            than k neighbours were available.
    """

    value: float
    ci_lower: float
    ci_upper: float
    source_cells: list[CellID] = field(default_factory=list)
    n_neighbors_in_radius: int = 0
    degraded_due_to_insufficient_neighbors: bool = False


class InsufficientNeighborsError(Exception):
    """Raised when ``interpolate_idw`` finds zero candidates within R.

    Per AC-E2-3 the affordance-routing rule catches the 0-neighbour
    case BEFORE the engine runs (cell is routed to ``"skip"``); this
    exception fires only as a defensive guard against misuse.
    """


# ── Engine ───────────────────────────────────────────────────────────


# 1.96 is the standard normal critical value for a two-sided 95 %
# confidence interval. Sprint S precedent: prismpy uses 1.96 for
# similar normal-CI computations elsewhere.
_Z_95_TWO_SIDED = 1.96


def interpolate_idw(
    target_cell: Cell,
    candidate_cells: list[Cell] | tuple[Cell, ...],
    *,
    k: int = IDW_DEFAULT_K,
    radius_km: float = IDW_DEFAULT_R,
    weight_power: float = IDW_DEFAULT_W,
) -> InterpolationResult:
    """Compute the IDW-imputed value for ``target_cell`` from
    ``candidate_cells``.

    Args:
        target_cell: Cell whose value is being imputed.
        candidate_cells: Pool of candidate neighbour cells. The
            engine filters to those within ``radius_km`` AND
            excludes any candidate whose ``cell_id`` matches
            ``target_cell.cell_id`` (self-match guard). The pool
            need not be pre-filtered geographically; the engine
            does the radius scan internally.
        k: Number of nearest neighbours to combine. Default 4.
        radius_km: Search radius (km). Default 15.
        weight_power: Inverse-distance exponent. Default 2.

    Returns:
        ``InterpolationResult`` carrying the imputed value, 95 % CI
        bounds, contributing source cells, and the degraded flag.

    Raises:
        InsufficientNeighborsError: when zero candidates fall within
            ``radius_km`` after self-filter. Per AC-E2-3 routing
            this case shouldn't reach the engine in production.
    """
    # Self-filter + radius scan in one pass. We collect (distance,
    # candidate) tuples and sort.
    target_lat = float(target_cell.lat)
    target_lon = float(target_cell.lon)

    in_radius: list[tuple[float, Cell]] = []
    for candidate in candidate_cells:
        if candidate.cell_id == target_cell.cell_id:
            continue  # self-match guard (AC-E2-2 sub-criterion)
        dist_km = haversine_distance(
            target_lon, target_lat, float(candidate.lon), float(candidate.lat)
        )
        if dist_km <= radius_km:
            in_radius.append((dist_km, candidate))

    n_in_radius = len(in_radius)
    if n_in_radius == 0:
        raise InsufficientNeighborsError(
            f"interpolate_idw: target_cell {target_cell.cell_id!r} has "
            f"zero candidate neighbours within R={radius_km} km. Per "
            f"AC-E2-3 routing this case should be routed to 'skip' "
            f"before reaching the engine."
        )

    # Sort by distance ascending; cap at k.
    in_radius.sort(key=lambda dc: dc[0])
    contributors = in_radius[:k]
    n = len(contributors)
    degraded = n < k

    # Weighted mean: w_i = 1 / d_i^weight_power.
    weights: list[float] = []
    values: list[float] = []
    source_ids: list[CellID] = []
    for dist_km, candidate in contributors:
        # Defensive: distances are positive (haversine is non-negative;
        # the self-match guard above catches the d=0 self case).
        # Hostile candidate-pool with co-located non-self cells (two
        # cells at identical lat/lon) would still yield d=0 here. Use
        # a small epsilon floor so 1/d^w doesn't blow up, while
        # preserving the dominant-neighbour effect.
        d = max(dist_km, 1e-9)
        weight = 1.0 / (d ** weight_power)
        weights.append(weight)
        values.append(float(candidate.value))
        source_ids.append(candidate.cell_id)

    sum_w = sum(weights)
    weighted_mean = sum(w * v for w, v in zip(weights, values)) / sum_w

    if n == 1:
        # k=1 zero-width path: single-sample variance is zero by
        # construction. The CI half-width is 1.96 * 0 / sqrt(1) = 0.
        # Per AC-E2-2 sub-criterion + AC-E2-7 Phrase 2 caveat.
        return InterpolationResult(
            value=weighted_mean,
            ci_lower=weighted_mean,
            ci_upper=weighted_mean,
            source_cells=source_ids,
            n_neighbors_in_radius=n_in_radius,
            degraded_due_to_insufficient_neighbors=True,
        )

    # k>=2: compute weighted standard deviation across contributors.
    # s_w = sqrt(sum(w * (v - mean)^2) / sum(w)).
    weighted_variance = (
        sum(w * (v - weighted_mean) ** 2 for w, v in zip(weights, values))
        / sum_w
    )
    s_w = math.sqrt(weighted_variance)
    ci_half_width = _Z_95_TWO_SIDED * s_w / math.sqrt(n)

    return InterpolationResult(
        value=weighted_mean,
        ci_lower=weighted_mean - ci_half_width,
        ci_upper=weighted_mean + ci_half_width,
        source_cells=source_ids,
        n_neighbors_in_radius=n_in_radius,
        degraded_due_to_insufficient_neighbors=degraded,
    )


__all__ = [
    "Cell",
    "InsufficientNeighborsError",
    "InterpolationResult",
    "interpolate_idw",
]
