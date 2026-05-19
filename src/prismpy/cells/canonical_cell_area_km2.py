"""Canonical per-cell geodesic area helper.

Returns full-cell area in km² from a cell identifier and a
:class:`SpatialRef` descriptor of the grid. The implementation factors
the spherical-first-order formula previously inlined at
``prismpy/src/prismpy/translators/craft/translator.py:1394`` into a
reusable canonical helper that downstream consumers
(``prismpy.packaging.manifest.create_manifest``; future translator
emitters) call by name instead of duplicating.

Semantics: ``manifest.cell_areas[]`` carries FULL-cell geodesic area.
Admin-intersected area (cell area × administrative-boundary share) is a
separate consumer-side derivation OR a separate v3.2 manifest field;
this helper does NOT take an admin boundary as input.

Formula::

    area_km2 = resolution_deg² × DEG2_TO_KM2 × cos(latitude_radians)

``DEG2_TO_KM2 = 12364.0`` is the spherical-mean first-order constant
``(π · R_earth / 180)²`` with ``R_earth = 6371 km`` (mean Earth radius).
The constant is bound to the :class:`SpatialRef` so future translators
can override it (e.g., the CRAFT translator's slightly different
``111.32²`` legacy constant — kept for backward compatibility until
the v3.2 cleanup).

Magnitudes (specialist 2026-05-19 sanity check):
- 5-arcmin (resolution_deg = 1/12) cell at equator: ~85.9 km²
- 5-arcmin cell at 25°N: ~77.8 km² (declining via cos(lat))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


DEG2_TO_KM2_DEFAULT: float = 12364.0


@dataclass(frozen=True)
class SpatialRef:
    """Grid descriptor consumed by :func:`canonical_cell_area_km2`.

    Three fields per spec:

    - ``resolution_deg``: cell grid resolution in degrees (e.g.
      ``1.0 / 12.0`` for 5-arcmin).
    - ``cell_centroid_latitude``: callable mapping ``cell_id`` to
      centroid latitude in DEGREES (the helper converts to radians
      internally). The callable form lets callers plug in a per-package
      lookup (cell_summary.json centroids, sites.shp geometry centroids,
      a uniform region-level fallback, etc.) without binding the helper
      to a specific persistence format.
    - ``deg2_to_km2``: deg²-to-km² conversion constant; defaults to the
      spherical-first-order ``DEG2_TO_KM2_DEFAULT`` (12364.0). Override
      only when reproducing a legacy formula.
    """

    resolution_deg: float
    cell_centroid_latitude: Callable[[int], float] = field()
    deg2_to_km2: float = DEG2_TO_KM2_DEFAULT


def canonical_cell_area_km2(cell_id: int, spatial_ref: SpatialRef) -> float:
    """Return full-cell geodesic area in km² for ``cell_id``.

    Spherical first-order approximation per the module docstring
    formula. Latitude resolution is delegated to the
    :class:`SpatialRef`'s ``cell_centroid_latitude`` callable so
    callers can vary the per-cell lat by lookup source.
    """
    latitude_deg = float(spatial_ref.cell_centroid_latitude(cell_id))
    cell_area_deg2 = float(spatial_ref.resolution_deg) ** 2
    return (
        cell_area_deg2
        * float(spatial_ref.deg2_to_km2)
        * math.cos(math.radians(latitude_deg))
    )


__all__ = [
    "DEG2_TO_KM2_DEFAULT",
    "SpatialRef",
    "canonical_cell_area_km2",
]
