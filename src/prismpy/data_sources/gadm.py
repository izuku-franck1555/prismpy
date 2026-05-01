"""
GADM (Global Administrative Areas) Data Source for prismpy.

This module provides dynamic access to GADM administrative boundary data
for generating CRAFT schemas at any admin level for any country.

GADM Structure:
- Level 0: Country boundary (COUNTRY attribute)
- Level 1: State/Province/Region (NAME_1 attribute)
- Level 2: District/County/Cercle (NAME_2 attribute)
- Level 3: Sub-district/Commune (NAME_3 attribute)

File naming convention: gadm41_{ISO3}_{level}.shp
Example: gadm41_MLI_0.shp, gadm41_MLI_1.shp, gadm41_MLI_2.shp

CRAFT Schema Level Mapping:
- CRAFT Level 1 (Country) → uses GADM Level 0 shapefile
- CRAFT Level 2 (State)   → uses GADM Level 1 shapefile
- CRAFT Level 3 (District)→ uses GADM Level 2 shapefile

Author: prismpy team
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# GADM attribute column names by level
GADM_ATTRIBUTES = {
    0: 'COUNTRY',   # Country name
    1: 'NAME_1',    # State/Province name
    2: 'NAME_2',    # District name
    3: 'NAME_3',    # Sub-district name
}

# Conversion factor: degree² to km² at equator
# 1 degree ≈ 111.32 km, so 1 deg² ≈ 12392 km²
DEG2_TO_KM2 = 12364  # Legacy CRAFT uses this value


class GADMDataSource:
    """Data source for GADM administrative boundary data.

    Provides methods to:
    - Load GADM shapefiles by country and level
    - Filter to specific admin regions by name
    - Calculate grid cell intersections with admin boundaries
    - Generate CRAFT-compatible schema data

    Example usage:
        gadm = GADMDataSource(gadm_path="/path/to/gadm/files")

        # Load Mali Level 1 (states/regions)
        gdf = gadm.load_shapefile("MLI", gadm_level=1)

        # Filter to Koutiala region
        koutiala_gdf = gadm.filter_by_name(gdf, gadm_level=1, name="Koutiala")

        # Calculate intersection areas for grid cells
        intersections = gadm.calculate_intersections(koutiala_gdf, grid_cells)
    """

    def __init__(self, gadm_path: Optional[str] = None):
        """Initialize GADM data source.

        Args:
            gadm_path: Path to directory containing GADM shapefiles.
                       Expected files: gadm41_{ISO3}_{level}.shp
        """
        self.gadm_path = Path(gadm_path) if gadm_path else None
        self._geopandas_available = None

    def _check_geopandas(self) -> bool:
        """Check if geopandas is available."""
        if self._geopandas_available is None:
            try:
                import geopandas
                self._geopandas_available = True
            except ImportError:
                self._geopandas_available = False
                logger.warning("geopandas not installed. GADM features require: pip install geopandas")
        return self._geopandas_available

    def get_shapefile_path(self, country_iso3: str, gadm_level: int) -> Optional[Path]:
        """Get path to GADM shapefile for a country and level.

        Args:
            country_iso3: ISO 3-letter country code (e.g., "MLI", "NGA", "KEN")
            gadm_level: GADM level (0=country, 1=state, 2=district, 3=sub-district)

        Returns:
            Path to shapefile if found, None otherwise
        """
        if not self.gadm_path:
            logger.warning("GADM path not configured")
            return None

        # Standard GADM naming convention
        filename = f"gadm41_{country_iso3.upper()}_{gadm_level}.shp"
        shapefile_path = self.gadm_path / filename

        if shapefile_path.exists():
            return shapefile_path

        # Try alternative locations
        alt_paths = [
            self.gadm_path / country_iso3.upper() / filename,
            self.gadm_path / "GADM" / filename,
        ]

        for alt_path in alt_paths:
            if alt_path.exists():
                return alt_path

        logger.warning(f"GADM shapefile not found: {shapefile_path}")
        return None

    def load_shapefile(self, country_iso3: str, gadm_level: int) -> Optional[Any]:
        """Load GADM shapefile for a country and level.

        Args:
            country_iso3: ISO 3-letter country code
            gadm_level: GADM level (0-3)

        Returns:
            GeoDataFrame with admin boundaries, or None if not found
        """
        if not self._check_geopandas():
            return None

        import geopandas as gpd

        shapefile_path = self.get_shapefile_path(country_iso3, gadm_level)
        if not shapefile_path:
            return None

        logger.info(f"Loading GADM shapefile: {shapefile_path}")

        try:
            gdf = gpd.read_file(shapefile_path)

            # Ensure WGS84 CRS
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            # Validate geometry
            invalid_count = (~gdf.geometry.is_valid).sum()
            if invalid_count > 0:
                logger.warning(f"Fixing {invalid_count} invalid geometries")
                gdf.geometry = gdf.geometry.buffer(0)

            logger.info(f"Loaded {len(gdf)} polygons from GADM level {gadm_level}")
            return gdf

        except Exception as e:
            logger.error(f"Error loading GADM shapefile: {e}")
            return None

    def filter_by_name(
        self,
        gdf: Any,
        gadm_level: int,
        name: str,
        case_sensitive: bool = False
    ) -> Optional[Any]:
        """Filter GeoDataFrame to specific admin region by name.

        Args:
            gdf: GeoDataFrame from load_shapefile()
            gadm_level: GADM level of the data
            name: Admin name to filter by
            case_sensitive: Whether to match case exactly

        Returns:
            Filtered GeoDataFrame, or None if name not found
        """
        if gdf is None or len(gdf) == 0:
            return None

        # Get the attribute column for this level
        attr_col = GADM_ATTRIBUTES.get(gadm_level)
        if not attr_col or attr_col not in gdf.columns:
            # Try alternative column names
            alt_cols = [f'Level{gadm_level}Name', f'NAME_{gadm_level}', 'name', 'NAME']
            for col in alt_cols:
                if col in gdf.columns:
                    attr_col = col
                    break
            else:
                logger.error(f"Cannot find admin name column for GADM level {gadm_level}")
                logger.info(f"Available columns: {gdf.columns.tolist()}")
                return None

        # Filter by name
        if case_sensitive:
            filtered = gdf[gdf[attr_col] == name]
        else:
            filtered = gdf[gdf[attr_col].str.lower() == name.lower()]

        if len(filtered) == 0:
            # Show available names for debugging
            available = gdf[attr_col].unique().tolist()
            logger.warning(f"Admin region '{name}' not found in {attr_col}")
            logger.info(f"Available regions: {available[:20]}{'...' if len(available) > 20 else ''}")
            return None

        logger.info(f"Filtered to '{name}': {len(filtered)} polygon(s)")
        return filtered

    def calculate_intersections(
        self,
        gdf: Any,
        cells: List[Tuple[int, float, float]],  # [(cell_id, lat, lon), ...]
        resolution_deg: float = 5/60,
    ) -> Dict[int, Dict[str, float]]:
        """Calculate intersection between grid cells and admin boundary.

        Returns both SharePercent (% of cell covered) and Area (km² of intersection).

        Args:
            gdf: GeoDataFrame with admin boundary polygon(s)
            cells: List of (cell_id, lat, lon) tuples for grid cells
            resolution_deg: Grid cell size in degrees (default 5 arcmin)

        Returns:
            Dictionary mapping cell_id to {
                'share_percent': float,  # 0-100
                'area_km2': float,       # intersection area in km²
                'lat': float,
                'lon': float
            }
        """
        if not self._check_geopandas():
            return {}

        from shapely.geometry import box

        if gdf is None or len(gdf) == 0:
            logger.warning("No geometry provided for intersection calculation")
            return {}

        # Union all polygons to get single admin boundary
        admin_geom = gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union

        half_res = resolution_deg / 2
        cell_area_deg2 = resolution_deg * resolution_deg

        results = {}

        for cell_id, lat, lon in cells:
            # Create cell bounding box
            cell_box = box(
                lon - half_res,
                lat - half_res,
                lon + half_res,
                lat + half_res
            )

            if cell_box.intersects(admin_geom):
                intersection = cell_box.intersection(admin_geom)
                intersection_area_deg2 = intersection.area

                # SharePercent: percentage of cell covered by admin boundary
                share_percent = (intersection_area_deg2 / cell_area_deg2) * 100

                # Area: intersection area in km² (with latitude correction)
                area_km2 = intersection_area_deg2 * DEG2_TO_KM2 * math.cos(math.radians(lat))

                results[cell_id] = {
                    'share_percent': round(share_percent, 2),
                    'area_km2': area_km2,
                    'lat': lat,
                    'lon': lon,
                }
            else:
                # Cell doesn't intersect admin boundary
                results[cell_id] = {
                    'share_percent': 0.0,
                    'area_km2': 0.0,
                    'lat': lat,
                    'lon': lon,
                }

        # Log statistics
        non_zero = sum(1 for r in results.values() if r['share_percent'] > 0)
        full_coverage = sum(1 for r in results.values() if r['share_percent'] >= 99.9)
        partial = non_zero - full_coverage

        logger.info(f"Intersection calculation: {len(results)} cells total, "
                   f"{non_zero} intersecting ({full_coverage} full, {partial} partial)")

        return results

    def get_craft_level_from_gadm(self, gadm_level: int) -> int:
        """Convert GADM level to CRAFT schema level.

        GADM Level 0 (country boundary) → CRAFT Level 1
        GADM Level 1 (states)           → CRAFT Level 2
        GADM Level 2 (districts)        → CRAFT Level 3

        Args:
            gadm_level: GADM level (0-2)

        Returns:
            CRAFT schema level (1-3)
        """
        return gadm_level + 1

    def get_gadm_level_from_craft(self, craft_level: int) -> int:
        """Convert CRAFT schema level to GADM level.

        CRAFT Level 1 (country) → GADM Level 0
        CRAFT Level 2 (state)   → GADM Level 1
        CRAFT Level 3 (district)→ GADM Level 2

        Args:
            craft_level: CRAFT schema level (1-3)

        Returns:
            GADM level (0-2)
        """
        return craft_level - 1


    def generate_schema_data(
        self,
        gdf: Any,
        resolution_deg: float = 5/60,
        decimal_places: int = 2,
        admin_info: Optional[Dict[str, str]] = None,
        *,
        threshold: float = 0.0,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate CRAFT schema data from admin boundary polygon.

        This is the core algorithm matching legacy CRAFT schema generation:
        1. Create fishnet grid covering polygon bounds
        2. Filter to cells that intersect with polygon
        3. Calculate intersection area (NOT full cell area)
        4. (F-R AC-3) Drop cells whose SharePercent is below ``threshold``
        5. Return data for both CRAFT_Schema and Python_Schemas

        Args:
            gdf: GeoDataFrame with admin boundary polygon(s)
            resolution_deg: Grid resolution in degrees (default 5 arcmin = 5/60)
            decimal_places: Decimal places for SharePercent
            admin_info: Dict with admin names {'level1': 'Mali', 'level2': 'Koutiala', ...}
            threshold: F-R AC-3 SharePercent threshold (0.0-100.0). Cells
                with share_percent < threshold are excluded from BOTH
                returned row lists. Default 0.0 admits every intersecting
                cell (AgMIP-canonical baseline). The 4 CRAFT-translator
                callsites (paths 1, 1b, 1c, 1d) thread
                ``BoundaryConfig.min_share_percent`` here so the canonical
                grid produced at HARMONIZE (AC-2) and the CRAFT schema
                rows agree on the same cell set.

        Returns:
            Tuple of (craft_schema_rows, python_schema_rows) where:
            - craft_schema_rows: List of {'cellid': int, 'share_percent': float}
            - python_schema_rows: List of {'cellid': int, 'lat': float, 'lon': float,
                                           'elevation': float, 'area': float, ...}
        """
        if not self._check_geopandas():
            return [], []

        import geopandas as gpd
        from shapely.geometry import box
        import numpy as np

        if gdf is None or len(gdf) == 0:
            logger.warning("No geometry provided for schema generation")
            return [], []

        # Union all polygons to get single admin boundary
        admin_geom = gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union

        # Get bounds and create fishnet
        bounds = admin_geom.bounds  # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = bounds

        # Align bounds to global grid
        minx_aligned = math.floor(minx / resolution_deg) * resolution_deg
        miny_aligned = math.floor(miny / resolution_deg) * resolution_deg
        maxx_aligned = math.ceil(maxx / resolution_deg) * resolution_deg
        maxy_aligned = math.ceil(maxy / resolution_deg) * resolution_deg

        # Generate grid cells
        lons = np.arange(minx_aligned, maxx_aligned, resolution_deg)
        lats = np.arange(miny_aligned, maxy_aligned, resolution_deg)

        cell_area_deg2 = resolution_deg * resolution_deg
        half_res = resolution_deg / 2

        craft_schema_rows = []
        python_schema_rows = []

        cells_checked = 0
        cells_intersecting = 0

        for lon in lons:
            for lat in lats:
                cells_checked += 1
                cell_center_lon = lon + half_res
                cell_center_lat = lat + half_res

                # Create cell box
                cell_box = box(lon, lat, lon + resolution_deg, lat + resolution_deg)

                # Check intersection
                if not cell_box.intersects(admin_geom):
                    continue

                cells_intersecting += 1

                # Calculate intersection
                intersection = cell_box.intersection(admin_geom)
                intersection_area_deg2 = intersection.area

                # F-R AC-3 + codex Gate B fix #4: compare the threshold
                # against the UNROUNDED SharePercent, then round for
                # display. Rounding-then-comparing diverges from the
                # canonical-grid filter at AC-2 Stage 3 (which uses
                # the unrounded percentage); cells with true SP just
                # below the threshold but rounding up would be admitted
                # here yet excluded from the canonical grid, breaking
                # the cell-set agreement this kwarg exists to enforce.
                share_percent_raw = (intersection_area_deg2 / cell_area_deg2) * 100
                if share_percent_raw < threshold:
                    continue
                # Round AFTER threshold-pass for display precision.
                share_percent = round(share_percent_raw, decimal_places)

                # Area: intersection area in km² (with latitude correction)
                # Formula from legacy: area_km² = area_deg² * 12364 * cos(lat)
                area_km2 = intersection_area_deg2 * DEG2_TO_KM2 * math.cos(math.radians(cell_center_lat))

                # Calculate CellID (CRAFT formula: 1-indexed)
                cellid = self._calculate_cellid(cell_center_lon, cell_center_lat, resolution_deg)

                # CRAFT schema row
                craft_schema_rows.append({
                    'cellid': cellid,
                    'share_percent': share_percent,
                })

                # Python schema row
                python_row = {
                    'cellid': cellid,
                    'lat': cell_center_lat,
                    'lon': cell_center_lon,
                    'elevation': -99.0,  # CRAFT default
                    'area': area_km2,
                }

                # Add admin info if provided
                if admin_info:
                    for key, value in admin_info.items():
                        python_row[key] = value

                python_schema_rows.append(python_row)

        # Sort by CellID ascending (matching CRAFT's actual working schema files)
        craft_schema_rows.sort(key=lambda x: x['cellid'])
        python_schema_rows.sort(key=lambda x: x['cellid'])

        logger.info(f"Schema generation: {cells_checked} cells checked, "
                   f"{cells_intersecting} intersecting with boundary")

        return craft_schema_rows, python_schema_rows

    def _calculate_cellid(self, lon: float, lat: float, resolution_deg: float) -> int:
        """Calculate CRAFT CellID for a grid cell at given coordinates.

        Matches CRAFT/R raster convention (top-left origin, 1-indexed).

        Formula: CellID = (row_from_top * ncols) + col + 1

        Args:
            lon: Longitude of cell center
            lat: Latitude of cell center
            resolution_deg: Grid resolution in degrees

        Returns:
            1-indexed CellID
        """
        ncols = int(360 / resolution_deg)
        nrows = int(180 / resolution_deg)

        col = int((lon + 180) / resolution_deg)
        row_from_bottom = int((lat + 90) / resolution_deg)
        row_from_top = nrows - 1 - row_from_bottom

        cellid = (row_from_top * ncols) + col + 1
        return cellid

    def get_cell_center_from_cellid(self, cellid: int, resolution_deg: float = 5/60) -> Tuple[float, float]:
        """Calculate cell center lat/lon from CellID (reverse of _calculate_cellid).

        Args:
            cellid: Grid cell ID (1-indexed)
            resolution_deg: Grid resolution in degrees

        Returns:
            Tuple of (latitude, longitude) of cell center
        """
        ncols = int(360 / resolution_deg)
        nrows = int(180 / resolution_deg)

        # Reverse calculation (convert to 0-indexed)
        cellid_0 = cellid - 1
        row_from_top = cellid_0 // ncols
        col = cellid_0 % ncols

        # Convert row from top to row from bottom
        row_from_bottom = nrows - 1 - row_from_top

        # Calculate cell boundaries (southwest corner)
        lon_min = -180 + (col * resolution_deg)
        lat_min = -90 + (row_from_bottom * resolution_deg)

        # Cell center
        lon_center = lon_min + (resolution_deg / 2)
        lat_center = lat_min + (resolution_deg / 2)

        return lat_center, lon_center


def create_gadm_source(config: Optional[Dict] = None) -> GADMDataSource:
    """Factory function to create GADM data source from config.

    Args:
        config: Configuration dictionary with optional 'gadm_path' key

    Returns:
        Configured GADMDataSource instance
    """
    gadm_path = None
    if config:
        gadm_path = config.get('gadm_path') or config.get('gadm', {}).get('data_path')
    return GADMDataSource(gadm_path=gadm_path)
