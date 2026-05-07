"""Cell-roster spatial index for cockpit IDW preflight.

Sprint E.2 AC-E2-2 + Builder Q4 grounding-pass discretion. The
cockpit's bulk-interpolation preflight (AC-E2-23) needs to ask
"how many of N candidate cells have ≥1 neighbour within R km?"
quickly enough that the modal opens without UI lag.

Phase 1 substrate uses a stdlib-only naive scan so the index is
available without dependency expansion. The implementation does an
O(N) haversine-distance scan per query; for typical bucket sizes
(50 candidates × 9 000-cell pipeline package) this is well within
the modal-open UX budget. A future scipy-cKDTree backend can land
in Phase 2 as a perf optimization (declared substrate-extension)
if real-data probes show the naive scan is the bottleneck.

Per durable §24 canonical-source-or-pin: this is the only place in
prismpy that exposes the cell-neighbour query surface; consumers
route through the wrapper rather than re-implementing the
haversine-loop pattern.
"""

from __future__ import annotations

from prismpy.harmonize.idw_interpolation import Cell
from prismpy.utils.gis_utils import haversine_distance


class SpatialIndex:
    """Linear-scan neighbour lookup over a cell roster.

    Build once per pipeline run; query per target cell. Calls to
    ``query_neighbours_within_radius_km`` return cells within the
    requested radius (great-circle, exact via ``haversine_distance``),
    excluding the target cell by ``cell_id`` equality.

    The index is stdlib-only by design — Sprint E.2 Phase 1
    substrate ships without a scipy dependency expansion. A future
    sprint may swap the backend for ``scipy.spatial.cKDTree`` if
    real-data probes show the naive scan is a bottleneck; the public
    interface stays unchanged so the swap is transparent to
    consumers.
    """

    def __init__(self, cells: list[Cell] | tuple[Cell, ...]) -> None:
        if not cells:
            raise ValueError(
                "SpatialIndex requires at least one cell; got empty roster."
            )
        self._cells: list[Cell] = list(cells)

    def __len__(self) -> int:
        return len(self._cells)

    def query_neighbours_within_radius_km(
        self,
        target: Cell,
        radius_km: float,
    ) -> list[Cell]:
        """Return cells in the index within ``radius_km`` of
        ``target`` (great-circle), excluding the target cell by
        ``cell_id`` equality. Returned cells are NOT sorted; the
        IDW engine sorts internally.

        Args:
            target: Cell whose neighbours we want.
            radius_km: Maximum great-circle distance (km).

        Returns:
            List of cells within ``radius_km`` (target excluded);
            empty list if no neighbours within range.
        """
        results: list[Cell] = []
        for candidate in self._cells:
            if candidate.cell_id == target.cell_id:
                continue
            exact_km = haversine_distance(
                target.lon, target.lat, candidate.lon, candidate.lat
            )
            if exact_km <= radius_km:
                results.append(candidate)
        return results


__all__ = [
    "SpatialIndex",
]
