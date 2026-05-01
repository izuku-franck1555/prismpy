"""
Spatial grid models for prismpy.

These models represent the grid cell structures used by different platforms.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

from prismpy.models.region import BoundingBox


@dataclass
class GridCell:
    """A single grid cell in a spatial grid.

    Attributes:
        cell_id: Unique cell identifier
        lat: Latitude of cell center
        lon: Longitude of cell center
        row: Row index in grid (0-indexed, top to bottom)
        col: Column index in grid (0-indexed, left to right)
        resolution: Grid resolution identifier
        share_percent: Fraction (0-100) of the cell's area that
            falls inside the admin polygon. Pre-computed at the
            HARMONIZE stage by the F-R AC-2 helper; downstream
            translators read this single value instead of each
            re-running the polygon-intersection arithmetic. Stays
            ``None`` when the boundary has no admin polygon
            (e.g., manual-bbox without GADM lookup).
    """
    cell_id: int
    lat: float
    lon: float
    row: int
    col: int
    resolution: str = "5arcmin"
    # F-R AC-3.5: pre-computed at AC-2 harmonize helper; read by
    # AC-3 CRAFT translator (4 paths). Optional[float] (NOT
    # float-with-default-0.0) so ``None`` distinguishes
    # "polygon-less boundary; SP not computable" from
    # "polygon present; SP is exactly 0.0".
    share_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cell_id": self.cell_id,
            "lat": self.lat,
            "lon": self.lon,
            "row": self.row,
            "col": self.col,
            "resolution": self.resolution,
            "share_percent": self.share_percent,
        }


@dataclass
class SpatialGrid:
    """Unified grid representation supporting multiple resolutions.

    This class handles the different grid systems used by various platforms:
    - CRAFT: 5 arc-minute cells (4320 columns x 2160 rows globally)
    - ACEA: 5 or 30 arc-minute cells
    - PYTHIA: Point-based (variable resolution)
    - SARRA-Py: Continuous bounding box (no discrete cells)

    Attributes:
        resolution: Resolution identifier ("5arcmin", "30arcmin", "custom")
        cells: List of grid cells
        increment_deg: Cell size in degrees
        bounds: Bounding box of the grid
    """
    resolution: Literal["5arcmin", "30arcmin", "custom"]
    cells: List[GridCell] = field(default_factory=list)
    increment_deg: float = 0.0833333  # 5 arcmin default
    bounds: Optional[BoundingBox] = None

    # Global grid dimensions
    GLOBAL_COLS_5ARCMIN = 4320
    GLOBAL_ROWS_5ARCMIN = 2160
    GLOBAL_COLS_30ARCMIN = 720
    GLOBAL_ROWS_30ARCMIN = 360

    def __post_init__(self):
        """Set increment based on resolution."""
        if self.resolution == "5arcmin":
            self.increment_deg = 5 / 60  # 0.0833...
        elif self.resolution == "30arcmin":
            self.increment_deg = 30 / 60  # 0.5

    @property
    def n_cells(self) -> int:
        """Number of cells in the grid."""
        return len(self.cells)

    @property
    def n_rows(self) -> int:
        """Number of rows in the grid."""
        if not self.cells:
            return 0
        return max(c.row for c in self.cells) - min(c.row for c in self.cells) + 1

    @property
    def n_cols(self) -> int:
        """Number of columns in the grid."""
        if not self.cells:
            return 0
        return max(c.col for c in self.cells) - min(c.col for c in self.cells) + 1

    def get_cell_ids(self) -> List[int]:
        """Get list of all cell IDs."""
        return [c.cell_id for c in self.cells]

    def get_cell_by_id(self, cell_id: int) -> Optional[GridCell]:
        """Get cell by ID."""
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        return None

    def iter_cells(self) -> Iterator[GridCell]:
        """Iterate over all cells."""
        return iter(self.cells)

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([c.to_dict() for c in self.cells])

    @classmethod
    def compute_cell_id_5arcmin(cls, row: int, col: int) -> int:
        """Compute 5-arcmin cell ID from row and column.

        The global 5-arcmin grid has 4320 columns and 2160 rows.
        Cell ID = row * 4320 + col (0-indexed, origin at top-left)
        """
        return row * cls.GLOBAL_COLS_5ARCMIN + col

    @classmethod
    def compute_cell_id_30arcmin(cls, row: int, col: int) -> int:
        """Compute 30-arcmin cell ID from row and column.

        The global 30-arcmin grid has 720 columns and 360 rows.
        Cell ID = row * 720 + col (0-indexed, origin at top-left)
        """
        return row * cls.GLOBAL_COLS_30ARCMIN + col

    @classmethod
    def rowcol_from_latlon_5arcmin(cls, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to row/col for 5-arcmin grid.

        Origin is at top-left (90N, -180E).
        """
        increment = 5 / 60  # 0.0833...
        row = int((90 - lat) / increment)
        col = int((lon + 180) / increment)
        return row, col

    @classmethod
    def rowcol_from_latlon_30arcmin(cls, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to row/col for 30-arcmin grid.

        Origin is at top-left (90N, -180E).
        """
        increment = 0.5
        row = int((90 - lat) / increment)
        col = int((lon + 180) / increment)
        return row, col

    @classmethod
    def latlon_from_rowcol_5arcmin(cls, row: int, col: int) -> Tuple[float, float]:
        """Convert row/col to lat/lon for 5-arcmin grid (cell center)."""
        increment = 5 / 60
        lat = 90 - (row + 0.5) * increment
        lon = -180 + (col + 0.5) * increment
        return lat, lon

    @classmethod
    def latlon_from_rowcol_30arcmin(cls, row: int, col: int) -> Tuple[float, float]:
        """Convert row/col to lat/lon for 30-arcmin grid (cell center)."""
        increment = 0.5
        lat = 90 - (row + 0.5) * increment
        lon = -180 + (col + 0.5) * increment
        return lat, lon

    @classmethod
    def convert_5arcmin_to_30arcmin(cls, cell_id_5: int) -> int:
        """Convert 5-arcmin cell ID to corresponding 30-arcmin cell ID.

        Each 30-arcmin cell contains 6x6 = 36 5-arcmin cells.
        """
        row_5 = cell_id_5 // cls.GLOBAL_COLS_5ARCMIN
        col_5 = cell_id_5 % cls.GLOBAL_COLS_5ARCMIN
        row_30 = row_5 // 6
        col_30 = col_5 // 6
        return cls.compute_cell_id_30arcmin(row_30, col_30)

    @classmethod
    def from_bounds(
        cls,
        bounds: BoundingBox,
        resolution: Literal["5arcmin", "30arcmin"] = "5arcmin",
        clip_geometry: Optional[Any] = None,
        exclude_cells: Optional[List[int]] = None,
    ) -> "SpatialGrid":
        """Create a grid from a bounding box, optionally clipped to a polygon.

        Args:
            bounds: Bounding box to create grid for
            resolution: Grid resolution
            clip_geometry: Optional shapely geometry or GeoDataFrame to clip points.
                          Only points inside this geometry will be included.
            exclude_cells: Optional list of cell IDs to drop from the
                grid (V2-22c-PRE.3.2 / D15). Filter applies INSIDE
                the inner construction loop so the excluded cells
                never enter the cells list — translators that
                iterate `grid.cells` automatically inherit the
                pruning, no per-translator edit required.

        Returns:
            SpatialGrid with cells covering the bounding box (or clipped to geometry)
        """
        if resolution == "5arcmin":
            increment = 5 / 60
            compute_id = cls.compute_cell_id_5arcmin
            rowcol_func = cls.rowcol_from_latlon_5arcmin
            latlon_func = cls.latlon_from_rowcol_5arcmin
        else:
            increment = 0.5
            compute_id = cls.compute_cell_id_30arcmin
            rowcol_func = cls.rowcol_from_latlon_30arcmin
            latlon_func = cls.latlon_from_rowcol_30arcmin

        # Get row/col range for bounds
        row_min, col_min = rowcol_func(bounds.maxy, bounds.minx)
        row_max, col_max = rowcol_func(bounds.miny, bounds.maxx)

        # Prepare geometry for point-in-polygon test
        geometry_union = None
        if clip_geometry is not None:
            try:
                # Handle GeoDataFrame
                if hasattr(clip_geometry, 'geometry'):
                    from shapely.ops import unary_union
                    geometry_union = unary_union(clip_geometry.geometry)
                # Handle shapely geometry directly
                elif hasattr(clip_geometry, 'contains'):
                    geometry_union = clip_geometry
            except Exception:
                # If geometry processing fails, fall back to bounding box
                geometry_union = None

        # V2-22c-PRE.3.2 (D15) — set lookup is O(1) per cell;
        # construction-time filter avoids materializing the
        # excluded cells then dropping them. Per §6.4
        # schema-bounds-match-strictest-downstream-consumer, the
        # filter lives at the factory locus so every translator
        # iterating `grid.cells` automatically inherits the prune.
        exclude_set = set(exclude_cells) if exclude_cells else set()

        cells = []
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                lat, lon = latlon_func(row, col)

                # Check if point is inside geometry (if provided)
                if geometry_union is not None:
                    from shapely.geometry import Point
                    point = Point(lon, lat)
                    if not geometry_union.contains(point):
                        continue  # Skip points outside the polygon

                cell_id = compute_id(row, col)
                if cell_id in exclude_set:
                    # V2-22c-PRE.3.2 (D15) — cell explicitly
                    # excluded (e.g., post-remediation re-run).
                    continue
                cells.append(GridCell(
                    cell_id=cell_id,
                    lat=lat,
                    lon=lon,
                    row=row,
                    col=col,
                    resolution=resolution
                ))

        return cls(
            resolution=resolution,
            cells=cells,
            increment_deg=increment,
            bounds=bounds
        )
