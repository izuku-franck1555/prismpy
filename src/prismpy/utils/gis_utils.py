"""
GIS utilities for prismpy.

These utilities handle geospatial operations including bounding box
manipulations, coordinate transformations, and grid calculations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import math


def bounds_to_polygon(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> List[Tuple[float, float]]:
    """Convert bounding box to polygon coordinates.

    Returns coordinates in counter-clockwise order (GIS convention).

    Args:
        minx: Minimum longitude (western edge)
        miny: Minimum latitude (southern edge)
        maxx: Maximum longitude (eastern edge)
        maxy: Maximum latitude (northern edge)

    Returns:
        List of (lon, lat) tuples forming a closed polygon
    """
    return [
        (minx, miny),  # SW corner
        (maxx, miny),  # SE corner
        (maxx, maxy),  # NE corner
        (minx, maxy),  # NW corner
        (minx, miny),  # Close the polygon
    ]


def polygon_to_bounds(
    polygon: List[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """Extract bounding box from polygon coordinates.

    Args:
        polygon: List of (lon, lat) tuples

    Returns:
        Tuple of (minx, miny, maxx, maxy)
    """
    if not polygon:
        raise ValueError("Polygon must have at least one point")

    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]

    return (min(lons), min(lats), max(lons), max(lats))


def expand_bounds(
    bounds: Tuple[float, float, float, float],
    buffer_deg: float,
) -> Tuple[float, float, float, float]:
    """Expand bounding box by a buffer.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy)
        buffer_deg: Buffer distance in degrees

    Returns:
        Expanded bounds tuple
    """
    minx, miny, maxx, maxy = bounds
    return (
        minx - buffer_deg,
        miny - buffer_deg,
        maxx + buffer_deg,
        maxy + buffer_deg,
    )


# AgMIP 5-arcmin simulation grid. The bbox_intersects inclusion rule admits a
# cell whenever its EXTENT touches the raw bbox, so an admitted cell CENTER can
# sit up to HALF a sim-cell outside the raw bbox. The climate fetch must reach
# at least that far past the bbox to enclose every admitted cell center.
AGMIP_SIM_CELL_DEG: float = 5.0 / 60.0                       # 0.0833333 (5')
AOI_CELL_OVERFLOW_MARGIN_DEG: float = AGMIP_SIM_CELL_DEG / 2.0  # 0.0416667
# Defeats floating-point ties at native-pixel boundaries: the AgERA5 native
# resolution round-trips as 0.09999999999999432, so an exact-edge request can
# land a hair inside the half-open keep-rule and silently drop a boundary
# pixel. The nudge is far smaller than half a native pixel (never pulls in an
# extra pixel ring) and far larger than the floating-point noise it cancels.
GRID_SNAP_TOLERANCE_DEG: float = 1e-6


def snap_bounds_outward_to_grid(
    bounds: Tuple[float, float, float, float],
    resolution: float,
    *,
    margin: float = AOI_CELL_OVERFLOW_MARGIN_DEG,
    tolerance: float = GRID_SNAP_TOLERANCE_DEG,
    origin: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Widen ``bounds`` outward to enclosing native-grid pixel edges.

    Used only on the climate FETCH path (AgERA5 / TAMSAT) so the fetched
    native pixels fully enclose the AgMIP outward-snapped cell roster. The
    cell roster is derived elsewhere from the raw bounds and is NOT changed by
    this function — only the fetch extent (and the bbox-keyed cache key) widen.

    Two steps, reusing :func:`expand_bounds` for the pad:

      1. Pad every edge by ``margin`` (default: half a sim-cell, the greatest
         distance an admitted cell center can sit past the raw bbox) so the
         padded box encloses every admitted center regardless of how the bbox
         aligns to the sim grid.
      2. Snap each padded edge OUTWARD to the enclosing native-grid pixel edge
         (pixel centers at ``origin + resolution*k``; edges halfway between),
         then nudge ``tolerance`` further out. Snapping to whole pixels means
         any point inside the padded box also lies inside a fetched pixel; the
         nudge defeats floating-point ties at the half-open request boundary.

    Args:
        bounds: ``(minx, miny, maxx, maxy)`` raw extent.
        resolution: Native pixel size in degrees (AgERA5 0.1, TAMSAT 0.0375).
        margin: Outward pad applied before snapping (degrees).
        tolerance: Outward nudge past the snapped pixel edge (degrees).
        origin: Native-grid phase — pixel centers at ``origin + resolution*k``.

    Returns:
        Widened ``(minx, miny, maxx, maxy)`` aligned to native pixel edges.
    """
    minx, miny, maxx, maxy = expand_bounds(bounds, margin)

    def _edge_down(value: float) -> float:
        """Largest native pixel edge <= ``value``."""
        k = math.floor((value - origin) / resolution - 0.5)
        return origin + (k + 0.5) * resolution

    def _edge_up(value: float) -> float:
        """Smallest native pixel edge >= ``value``."""
        k = math.ceil((value - origin) / resolution - 0.5)
        return origin + (k + 0.5) * resolution

    return (
        _edge_down(minx) - tolerance,
        _edge_down(miny) - tolerance,
        _edge_up(maxx) + tolerance,
        _edge_up(maxy) + tolerance,
    )


def bounds_contain_point(
    bounds: Tuple[float, float, float, float],
    lon: float,
    lat: float,
) -> bool:
    """Check if a point is within bounds.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy)
        lon: Longitude of point
        lat: Latitude of point

    Returns:
        True if point is within bounds
    """
    minx, miny, maxx, maxy = bounds
    return minx <= lon <= maxx and miny <= lat <= maxy


def bounds_overlap(
    bounds1: Tuple[float, float, float, float],
    bounds2: Tuple[float, float, float, float],
) -> bool:
    """Check if two bounding boxes overlap.

    Args:
        bounds1: First bounds tuple
        bounds2: Second bounds tuple

    Returns:
        True if bounds overlap
    """
    minx1, miny1, maxx1, maxy1 = bounds1
    minx2, miny2, maxx2, maxy2 = bounds2

    return not (
        maxx1 < minx2 or  # 1 is left of 2
        minx1 > maxx2 or  # 1 is right of 2
        maxy1 < miny2 or  # 1 is below 2
        miny1 > maxy2     # 1 is above 2
    )


def haversine_distance(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    """Calculate great-circle distance between two points.

    Uses the Haversine formula.

    Args:
        lon1, lat1: First point (degrees)
        lon2, lat2: Second point (degrees)

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def degrees_to_km(degrees: float, lat: float = 0) -> Tuple[float, float]:
    """Convert degrees to approximate kilometers.

    Note: This is approximate as the conversion varies with latitude.

    Args:
        degrees: Distance in degrees
        lat: Latitude for longitude correction

    Returns:
        Tuple of (km_lon, km_lat) - distance in km for longitude and latitude
    """
    # 1 degree latitude is approximately 111 km everywhere
    km_lat = degrees * 111

    # 1 degree longitude varies with latitude
    km_lon = degrees * 111 * math.cos(math.radians(lat))

    return km_lon, km_lat


def km_to_degrees(km: float, lat: float = 0) -> Tuple[float, float]:
    """Convert kilometers to approximate degrees.

    Args:
        km: Distance in kilometers
        lat: Latitude for longitude correction

    Returns:
        Tuple of (deg_lon, deg_lat) - distance in degrees
    """
    deg_lat = km / 111

    # Longitude degrees vary with latitude
    cos_lat = math.cos(math.radians(lat))
    deg_lon = km / (111 * cos_lat) if cos_lat > 0 else km / 111

    return deg_lon, deg_lat


def latlon_to_rowcol(
    lat: float,
    lon: float,
    resolution_deg: float,
    origin: str = "top_left",
) -> Tuple[int, int]:
    """Convert lat/lon to row/column indices.

    Args:
        lat: Latitude
        lon: Longitude
        resolution_deg: Cell size in degrees
        origin: Grid origin ("top_left" or "bottom_left")

    Returns:
        Tuple of (row, col)
    """
    col = int((lon + 180) / resolution_deg)

    if origin == "top_left":
        row = int((90 - lat) / resolution_deg)
    else:
        row = int((lat + 90) / resolution_deg)

    return row, col


def rowcol_to_latlon(
    row: int,
    col: int,
    resolution_deg: float,
    origin: str = "top_left",
    center: bool = True,
) -> Tuple[float, float]:
    """Convert row/column to lat/lon.

    Args:
        row: Row index
        col: Column index
        resolution_deg: Cell size in degrees
        origin: Grid origin ("top_left" or "bottom_left")
        center: Return cell center (True) or corner (False)

    Returns:
        Tuple of (lat, lon)
    """
    offset = 0.5 if center else 0

    lon = -180 + (col + offset) * resolution_deg

    if origin == "top_left":
        lat = 90 - (row + offset) * resolution_deg
    else:
        lat = -90 + (row + offset) * resolution_deg

    return lat, lon


def compute_cell_id_global(
    row: int,
    col: int,
    n_cols: int = 4320,
) -> int:
    """Compute global cell ID from row and column.

    Default is 5-arcmin grid (4320 columns, 2160 rows).

    Args:
        row: Row index (0-indexed)
        col: Column index (0-indexed)
        n_cols: Number of columns in the global grid

    Returns:
        Cell ID
    """
    return row * n_cols + col


def cell_id_to_rowcol(
    cell_id: int,
    n_cols: int = 4320,
) -> Tuple[int, int]:
    """Convert cell ID to row and column.

    Args:
        cell_id: Cell ID
        n_cols: Number of columns in the global grid

    Returns:
        Tuple of (row, col)
    """
    row = cell_id // n_cols
    col = cell_id % n_cols
    return row, col


def generate_grid_points(
    bounds: Tuple[float, float, float, float],
    resolution_deg: float,
    include_boundary: bool = True,
) -> List[Tuple[float, float]]:
    """Generate grid points within a bounding box.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy)
        resolution_deg: Grid spacing in degrees
        include_boundary: Include points on the boundary

    Returns:
        List of (lon, lat) tuples for grid cell centers
    """
    minx, miny, maxx, maxy = bounds
    points = []

    # Start at cell centers
    start_lon = minx + resolution_deg / 2
    start_lat = miny + resolution_deg / 2

    lon = start_lon
    while lon <= maxx:
        lat = start_lat
        while lat <= maxy:
            if include_boundary or bounds_contain_point(bounds, lon, lat):
                points.append((lon, lat))
            lat += resolution_deg
        lon += resolution_deg

    return points


def calculate_area_km2(
    bounds: Tuple[float, float, float, float],
) -> float:
    """Calculate approximate area of a bounding box in km².

    Uses spherical approximation.

    Args:
        bounds: Tuple of (minx, miny, maxx, maxy) in degrees

    Returns:
        Area in square kilometers
    """
    minx, miny, maxx, maxy = bounds

    # Average latitude for calculating longitude distance
    avg_lat = (miny + maxy) / 2

    # Width in km (longitude varies with latitude)
    width_km = (maxx - minx) * 111 * math.cos(math.radians(avg_lat))

    # Height in km (latitude is constant)
    height_km = (maxy - miny) * 111

    return width_km * height_km
