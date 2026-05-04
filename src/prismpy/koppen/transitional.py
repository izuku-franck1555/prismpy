"""TRANSITIONAL_ZONE detection at native 1km resolution.

Per AC-Q1-A and the research doc §Q1.3: a cell is
TRANSITIONAL iff any of the 8 immediately adjacent native-
1km Beck cells has a different (non-zero) KG class than the
center cell. This is the honest-fail signal at the
resolution of the substrate; the raster cannot resolve
finer than 1km.

Use cases:

* **Bound generation** (AC-Q2-B1): exclude transitional cells
  from per-zone percentile pools (the cell's climate is
  ambiguous between two zones).
* **Wizard-time validation**: surface transitional cells via
  ``WarningCategory.TRANSITIONAL_ZONE`` (Sprint E.0).
* **Stage 1 compatibility**: route transitional cells to a
  Bucket 2 informational signal rather than a percentile-
  based bound check.

Ocean cells (where the center sample is nodata) are NOT
transitional — they are ocean. An ocean center returns
``False`` from :func:`is_transitional_cell`. Ocean
neighbors are treated per the "non-zero" rule from the
research doc: an ocean neighbor does not flag a non-ocean
cell as transitional (a coastal land cell is interior with
respect to its land neighbors).
"""
from __future__ import annotations

from typing import Optional

from prismpy.koppen.kg_classifier import KGClassifier, KGZone
from prismpy.koppen.raster_loader import NATIVE_CELL_DEG


# 8-neighbor offsets in (delta_lat, delta_lon) at native
# cell size. Includes 4 cardinal + 4 diagonal neighbors.
NEIGHBOR_OFFSETS: tuple[tuple[float, float], ...] = (
    (-NATIVE_CELL_DEG, -NATIVE_CELL_DEG),
    (-NATIVE_CELL_DEG, 0.0),
    (-NATIVE_CELL_DEG, +NATIVE_CELL_DEG),
    (0.0, -NATIVE_CELL_DEG),
    (0.0, +NATIVE_CELL_DEG),
    (+NATIVE_CELL_DEG, -NATIVE_CELL_DEG),
    (+NATIVE_CELL_DEG, 0.0),
    (+NATIVE_CELL_DEG, +NATIVE_CELL_DEG),
)


def is_transitional_cell(
    lat: float, lon: float, classifier: KGClassifier,
) -> bool:
    """Return ``True`` if the cell at (lat, lon) is transitional.

    A cell is TRANSITIONAL iff any of the 8 native-1km
    neighbors has a different (non-zero) KG zone than the
    center. Ocean cells (center=nodata) are NOT transitional.
    Ocean neighbors do not flag a non-ocean center.

    Per AC-Q1-A. Per the research doc §Q1.3 and CC-13.
    """
    center = classifier.classify(lat, lon)
    if center is None:
        # Ocean / nodata center — not transitional, just ocean.
        return False
    for dlat, dlon in NEIGHBOR_OFFSETS:
        neighbor = classifier.classify(lat + dlat, lon + dlon)
        if neighbor is None:
            # Ocean neighbor: skip per the "non-zero" rule
            # (research doc §Q1.3).
            continue
        if neighbor != center:
            return True
    return False


def classify_with_transitional_flag(
    lat: float, lon: float, classifier: KGClassifier,
) -> tuple[Optional[KGZone], bool]:
    """Classify a cell + flag whether it is transitional in one pass.

    Returns ``(center_zone, is_transitional)``. Saves a
    redundant center-sample call when both pieces of info
    are needed (e.g. bound-gen pre-filtering, wizard banner
    rendering).

    Ocean cells return ``(None, False)``.
    """
    center = classifier.classify(lat, lon)
    if center is None:
        return None, False
    transitional = False
    for dlat, dlon in NEIGHBOR_OFFSETS:
        neighbor = classifier.classify(lat + dlat, lon + dlon)
        if neighbor is None:
            continue
        if neighbor != center:
            transitional = True
            break
    return center, transitional
