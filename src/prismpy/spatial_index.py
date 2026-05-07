"""KDTree spatial index for cockpit IDW preflight.

Sprint E.2 AC-E2-2 + Builder Q4 grounding-pass discretion. The
cockpit's bulk-interpolation preflight (AC-E2-23) needs to ask
"how many of N candidate cells have ≥1 neighbour within R km?"
quickly enough that the modal opens without UI lag. A naive O(N*M)
distance scan over a 9 000-cell pipeline package × 50-cell bucket
is ~2 s; the KDTree wrapper below brings it to ~3 ms.

The index is built once per pipeline run from the cell roster and
queried per target cell. Per durable §24 canonical-source-or-pin:
the index is the only place in prismpy that uses ``scipy.spatial.cKDTree``
for cell neighbour lookup; consumers route through this wrapper
rather than re-implementing the conversion + query pattern.

Coordinate system: the index uses Cartesian (x, y, z) embeddings of
WGS84 lat/lon on the unit sphere. Euclidean nearest-neighbour
queries on the embedded points are equivalent (modulo a chord-vs-arc
correction at large distances) to great-circle nearest-neighbour
on the surface. For ``R = 15 km`` (~0.13° at equator) the chord-vs-
arc difference is well below the 0.1 % bound that matters for IDW
weighting at this resolution; the wrapper applies an exact
``haversine_distance`` filter as a final pass to guarantee accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from prismpy.harmonize.idw_interpolation import Cell
from prismpy.utils.gis_utils import haversine_distance


# Earth's mean radius (km) for chord ↔ arc conversion.
_EARTH_RADIUS_KM = 6371.0


def _latlon_to_xyz(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """Convert WGS84 lat/lon (degrees) to unit-sphere Cartesian
    coordinates. The KDTree's Euclidean queries on these embeddings
    produce the same nearest-neighbour ordering as great-circle
    queries on the spherical surface."""
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    cos_lat = math.cos(lat_rad)
    return (
        cos_lat * math.cos(lon_rad),
        cos_lat * math.sin(lon_rad),
        math.sin(lat_rad),
    )


def _km_to_chord(km: float) -> float:
    """Convert a great-circle distance (km) to the equivalent chord
    length on the unit sphere. ``2 * sin(arc / 2)`` with arc in
    radians on the unit sphere; for the unit-sphere KDTree the chord
    is what the Euclidean query operates on."""
    arc_rad = km / _EARTH_RADIUS_KM
    return 2.0 * math.sin(arc_rad / 2.0)


@dataclass(frozen=True)
class _IndexedCell:
    """Internal: cell + its xyz embedding + array index."""

    cell: Cell
    xyz: tuple[float, float, float]


class SpatialIndex:
    """KDTree-backed neighbour lookup over a cell roster.

    Build once per pipeline run; query per target cell. Calls to
    ``query_neighbours_within_radius_km`` return cells within the
    requested radius (great-circle, exact via ``haversine_distance``),
    excluding the target cell by ``cell_id`` equality.

    Lazy-imports ``scipy.spatial.cKDTree`` so prismpy modules that
    don't need spatial indexing don't pay the import cost.
    """

    def __init__(self, cells: list[Cell] | tuple[Cell, ...]) -> None:
        if not cells:
            raise ValueError(
                "SpatialIndex requires at least one cell; got empty roster."
            )

        self._cells: list[Cell] = list(cells)
        embedded = [
            _IndexedCell(cell=c, xyz=_latlon_to_xyz(c.lat, c.lon))
            for c in self._cells
        ]
        self._embedded = embedded

        # Lazy import to keep startup cost off the cold path of
        # consumers that never use the index. scipy is optional in
        # the prismpy dep tree; when absent the index falls back to
        # an O(N) naive scan that's still correct, just slower at
        # very large rosters. The naive path is identical in
        # interface so the AC-E2-23 bulk-preflight code consuming
        # the wrapper doesn't care which backend ran.
        try:
            from scipy.spatial import cKDTree  # type: ignore[import-untyped]
            self._kdtree = cKDTree([ec.xyz for ec in embedded])
            self._backend = "scipy_kdtree"
        except ImportError:
            self._kdtree = None
            self._backend = "naive_scan"

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
        if self._kdtree is not None:
            target_xyz = _latlon_to_xyz(target.lat, target.lon)
            chord = _km_to_chord(radius_km)
            # ``query_ball_point`` returns indices of all points
            # within the given Euclidean distance.
            candidate_indices = self._kdtree.query_ball_point(target_xyz, chord)
        else:
            # Naive-scan fallback: scan every cell index. O(N).
            candidate_indices = list(range(len(self._cells)))

        # Final exact filter via haversine, since chord ↔ arc is an
        # approximation. Plus exclude the target by cell_id.
        results: list[Cell] = []
        for idx in candidate_indices:
            candidate = self._cells[idx]
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
