"""
GADM boundary data source retriever.

This module provides functionality to extract region boundaries from GADM
(Global Administrative Areas) shapefiles.

Reference: SARRA-Py/01-REGION-DEFINITION/ implementation patterns.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
from shapely.geometry import box

from prismpy.models.region import BoundingBox, Region
from prismpy.provenance.tracker import ProvenanceTracker, DecisionType
from prismpy.sources.base import DataSource, RetrievalResult
from prismpy.utils.sanitization import normalize_region_name


logger = logging.getLogger(__name__)


class GADMSource(DataSource):
    """Data source for GADM administrative boundary shapefiles.

    GADM (Global Administrative Areas) provides administrative boundary
    data at multiple levels:
    - Level 0: Country boundaries
    - Level 1: First-level subdivisions (states, provinces, regions)
    - Level 2: Second-level subdivisions (districts, counties)
    - Level 3+: Lower administrative levels (where available)

    The retriever extracts bounding boxes and optionally full geometries
    from GADM shapefiles, filtering by administrative name.

    Attributes:
        NAME: Data source identifier
        GADM_VERSION: Default GADM version
        GADM_LEVELS: Valid administrative levels
    """

    NAME = "gadm"
    GADM_VERSION = "4.1"
    GADM_LEVELS = (0, 1, 2, 3, 4, 5)

    # Column naming patterns in GADM shapefiles
    GADM_NAME_COLUMNS = {
        0: "COUNTRY",
        1: "NAME_1",
        2: "NAME_2",
        3: "NAME_3",
        4: "NAME_4",
        5: "NAME_5",
    }

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the GADM data source.

        Args:
            base_path: Base directory containing GADM shapefiles
            cache_dir: Directory for caching extracted bounds
            provenance: Provenance tracker for audit trail
        """
        super().__init__(cache_dir=cache_dir, provenance=provenance)
        self.base_path = Path(base_path) if base_path else Path("data/gadm")

    def retrieve(
        self,
        region: Region = None,
        shapefile_path: Optional[Union[str, Path]] = None,
        country_iso3: Optional[str] = None,
        gadm_level: int = 2,
        filter_field: Optional[str] = None,
        filter_value: Optional[str] = None,
        include_geometry: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve region bounds from GADM shapefile.

        Can be called in two modes:
        1. With a Region object that contains boundary configuration
        2. With explicit parameters (shapefile_path, filter_field, filter_value)

        Args:
            region: Region object with boundary configuration (optional)
            shapefile_path: Direct path to GADM shapefile
            country_iso3: ISO3 country code for constructing default path
            gadm_level: GADM administrative level (0-5)
            filter_field: Column name to filter by (e.g., "NAME_2")
            filter_value: Value to filter for (e.g., "Koutiala")
            include_geometry: Include full WKT geometry in result
            use_cache: Use cached results if available
            **kwargs: Additional parameters

        Returns:
            RetrievalResult containing Region object with extracted bounds
        """
        errors = []
        warnings = []
        metadata = {"source": self.NAME, "gadm_level": gadm_level}

        # Extract parameters from region config if provided
        if region is not None:
            country_iso3 = region.country_iso3
            filter_value = filter_value or region.name
            # Additional config may come from kwargs

        # Determine shapefile path
        if shapefile_path is None:
            if country_iso3 is None:
                return self.create_result(
                    success=False,
                    errors=["Either shapefile_path or country_iso3 must be provided"],
                )
            shapefile_path = self._get_default_shapefile_path(country_iso3, gadm_level)

        shapefile_path = Path(shapefile_path)
        metadata["shapefile_path"] = str(shapefile_path)

        # Check if shapefile exists
        if not shapefile_path.exists():
            return self.create_result(
                success=False,
                errors=[f"Shapefile not found: {shapefile_path}"],
                metadata=metadata,
            )

        # Determine filter field if not specified
        if filter_field is None:
            filter_field = self.GADM_NAME_COLUMNS.get(gadm_level, f"NAME_{gadm_level}")

        metadata["filter_field"] = filter_field
        metadata["filter_value"] = filter_value

        # Check cache (skip cache if geometry is needed since cache doesn't store geometry)
        if use_cache and filter_value and not include_geometry:
            cache_path = self._get_bounds_cache_path(country_iso3 or "unknown", filter_value)
            if cache_path.exists():
                try:
                    cached_region = self._load_cached_bounds(cache_path)
                    self.logger.info(f"Loaded cached bounds for {filter_value}")
                    metadata["from_cache"] = True
                    return self.create_result(
                        success=True,
                        data=cached_region,
                        output_path=cache_path,
                        metadata=metadata,
                    )
                except Exception as e:
                    warnings.append(f"Cache read failed, fetching fresh: {e}")
        elif use_cache and include_geometry:
            self.logger.debug("Skipping cache - geometry requested but cache doesn't store geometry")

        # Load and process shapefile
        try:
            result_region, geometry_wkt = self._extract_bounds(
                shapefile_path=shapefile_path,
                filter_field=filter_field,
                filter_value=filter_value,
                country_iso3=country_iso3,
                gadm_level=gadm_level,
                include_geometry=include_geometry,
            )
        except ValueError as e:
            return self.create_result(
                success=False,
                errors=[str(e)],
                metadata=metadata,
            )
        except Exception as e:
            return self.create_result(
                success=False,
                errors=[f"Failed to extract bounds: {e}"],
                metadata=metadata,
            )

        # Record provenance
        if self.provenance:
            self.provenance.record_decision(
                decision_type=DecisionType.SOURCE_SELECTION,
                description=f"Extracted bounds from GADM level {gadm_level}",
                rationale="GADM provides standardized administrative boundaries",
                alternatives=["Manual bounds", "Custom shapefile"],
                reference=f"GADM v{self.GADM_VERSION}",
            )

        # Cache the result
        if filter_value:
            cache_path = self._get_bounds_cache_path(
                result_region.country_iso3, filter_value
            )
            try:
                self._save_bounds_cache(result_region, cache_path)
                metadata["cache_path"] = str(cache_path)
            except Exception as e:
                warnings.append(f"Failed to cache bounds: {e}")

        # Add bounds formats to metadata
        metadata["bounds_gis"] = result_region.bounds.to_gis_format()
        metadata["bounds_sarra_py"] = result_region.bounds.to_sarra_py_format()
        metadata["from_cache"] = False

        return self.create_result(
            success=True,
            data=result_region,
            warnings=warnings,
            metadata=metadata,
        )

    def validate(self, data: Any) -> List[str]:
        """Validate extracted region data.

        Args:
            data: Region object to validate

        Returns:
            List of validation error messages
        """
        errors = []

        if not isinstance(data, Region):
            errors.append(f"Expected Region object, got {type(data)}")
            return errors

        # Validate bounds
        bounds = data.bounds
        if bounds.minx >= bounds.maxx:
            errors.append(f"Invalid bounds: minx ({bounds.minx}) >= maxx ({bounds.maxx})")
        if bounds.miny >= bounds.maxy:
            errors.append(f"Invalid bounds: miny ({bounds.miny}) >= maxy ({bounds.maxy})")

        # Validate coordinate ranges
        if not (-180 <= bounds.minx <= 180):
            errors.append(f"minx out of range: {bounds.minx}")
        if not (-180 <= bounds.maxx <= 180):
            errors.append(f"maxx out of range: {bounds.maxx}")
        if not (-90 <= bounds.miny <= 90):
            errors.append(f"miny out of range: {bounds.miny}")
        if not (-90 <= bounds.maxy <= 90):
            errors.append(f"maxy out of range: {bounds.maxy}")

        # Validate SARRA-Py format relationships
        sarra_py_bounds = bounds.to_sarra_py_format()
        lat_nw, lon_nw, lat_se, lon_se = sarra_py_bounds
        if lat_nw <= lat_se:
            errors.append(f"SARRA-Py format: lat_NW ({lat_nw}) should be > lat_SE ({lat_se})")
        if lon_se <= lon_nw:
            errors.append(f"SARRA-Py format: lon_SE ({lon_se}) should be > lon_NW ({lon_nw})")

        return errors

    def _extract_bounds(
        self,
        shapefile_path: Path,
        filter_field: Optional[str],
        filter_value: Optional[str],
        country_iso3: Optional[str],
        gadm_level: int,
        include_geometry: bool,
    ) -> Tuple[Region, Optional[str]]:
        """Extract bounding box from shapefile.

        Args:
            shapefile_path: Path to GADM shapefile
            filter_field: Column to filter by
            filter_value: Value to filter for
            country_iso3: ISO3 country code
            gadm_level: GADM administrative level
            include_geometry: Include WKT geometry

        Returns:
            Tuple of (Region, geometry_wkt or None)

        Raises:
            ValueError: If filter column not found or no features match
        """
        self.logger.info(f"Reading shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)

        # Store original CRS
        original_crs = str(gdf.crs) if gdf.crs else "EPSG:4326"

        # Filter if requested
        if filter_field and filter_value:
            self.logger.info(f"Filtering: {filter_field} = '{filter_value}'")

            if filter_field not in gdf.columns:
                available_cols = [c for c in gdf.columns if c.startswith("NAME")]
                raise ValueError(
                    f"Column '{filter_field}' not found. "
                    f"Available name columns: {available_cols}"
                )

            # Try exact match first
            filtered = gdf[gdf[filter_field] == filter_value]

            # If no match, try case-insensitive match
            if len(filtered) == 0:
                filtered = gdf[gdf[filter_field].str.lower() == filter_value.lower()]

            if len(filtered) == 0:
                # Show available values for debugging
                unique_values = sorted(gdf[filter_field].dropna().unique())[:20]
                raise ValueError(
                    f"No features found where {filter_field} = '{filter_value}'. "
                    f"Sample available values: {unique_values}"
                )

            gdf = filtered
            self.logger.info(f"Found {len(gdf)} matching feature(s)")

        # Extract bounds using total_bounds (union of all geometries)
        # Returns [minx, miny, maxx, maxy]
        bounds_array = gdf.total_bounds
        minx, miny, maxx, maxy = bounds_array

        # Create BoundingBox
        bounds = BoundingBox(
            minx=float(minx),
            miny=float(miny),
            maxx=float(maxx),
            maxy=float(maxy),
            crs=original_crs,
        )

        # Extract geometry WKT if requested
        geometry_wkt = None
        if include_geometry:
            # Dissolve all geometries into one
            dissolved = gdf.dissolve()
            geometry_wkt = dissolved.geometry.iloc[0].wkt

        # Infer country name if not available
        country_name = None
        if "COUNTRY" in gdf.columns:
            country_name = gdf["COUNTRY"].iloc[0]
        elif "NAME_0" in gdf.columns:
            country_name = gdf["NAME_0"].iloc[0]
        else:
            country_name = country_iso3 or "Unknown"

        # Create Region object
        region = Region(
            name=filter_value or "Full Region",
            country=country_name,
            country_iso3=country_iso3 or self._infer_iso3(country_name),
            bounds=bounds,
            gadm_level=gadm_level,
            geometry_wkt=geometry_wkt,
            crs=original_crs,
            metadata={
                "shapefile": str(shapefile_path),
                "filter_field": filter_field,
                "filter_value": filter_value,
                "feature_count": len(gdf),
            },
        )

        return region, geometry_wkt

    def _get_default_shapefile_path(self, country_iso3: str, gadm_level: int) -> Path:
        """Get default shapefile path for a country and level.

        GADM shapefile naming convention:
        gadm41_{ISO3}_{level}.shp

        Args:
            country_iso3: ISO3 country code (e.g., "MLI")
            gadm_level: Administrative level (0-5)

        Returns:
            Path to shapefile
        """
        filename = f"gadm41_{country_iso3.upper()}_{gadm_level}.shp"
        return self.base_path / country_iso3.upper() / filename

    def _get_bounds_cache_path(self, country_iso3: str, region_name: str) -> Path:
        """Get cache file path for bounds.

        Args:
            country_iso3: ISO3 country code
            region_name: Region name

        Returns:
            Path to cache file
        """
        safe_name = normalize_region_name(region_name)
        filename = f"bounds_{country_iso3.lower()}_{safe_name}.json"
        return self.cache_dir / "boundaries" / filename

    def _save_bounds_cache(self, region: Region, cache_path: Path) -> None:
        """Save region bounds to cache file.

        Args:
            region: Region object
            cache_path: Path to cache file
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "region": region.name,
            "country": region.country,
            "country_iso3": region.country_iso3,
            "crs": region.crs,
            "gadm_level": region.gadm_level,
            "bounds_gis": region.bounds.to_gis_format(),
            "bounds_gis_description": "[minx, miny, maxx, maxy]",
            "bounds_sarra_py": region.bounds.to_sarra_py_format(),
            "bounds_sarra_py_description": "[lat_NW, lon_NW, lat_SE, lon_SE]",
            "coordinates": {
                "minx": region.bounds.minx,
                "miny": region.bounds.miny,
                "maxx": region.bounds.maxx,
                "maxy": region.bounds.maxy,
            },
            "metadata": region.metadata,
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        self.logger.debug(f"Cached bounds to {cache_path}")

    def _load_cached_bounds(self, cache_path: Path) -> Region:
        """Load region from cache file.

        Args:
            cache_path: Path to cache file

        Returns:
            Region object
        """
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        coords = data.get("coordinates", {})
        if coords:
            bounds = BoundingBox(
                minx=coords["minx"],
                miny=coords["miny"],
                maxx=coords["maxx"],
                maxy=coords["maxy"],
                crs=data.get("crs", "EPSG:4326"),
            )
        else:
            # Fallback to bounds_gis format
            gis_bounds = data["bounds_gis"]
            bounds = BoundingBox.from_gis_format(gis_bounds, data.get("crs", "EPSG:4326"))

        return Region(
            name=data["region"],
            country=data["country"],
            country_iso3=data["country_iso3"],
            bounds=bounds,
            gadm_level=data.get("gadm_level", 2),
            crs=data.get("crs", "EPSG:4326"),
            metadata=data.get("metadata", {}),
        )

    def _infer_iso3(self, country_name: str) -> str:
        """Infer ISO3 code from country name.

        Args:
            country_name: Country name

        Returns:
            ISO3 code (or "UNK" if unknown)
        """
        # Common mappings for SSA countries
        iso3_map = {
            "mali": "MLI",
            "senegal": "SEN",
            "burkina faso": "BFA",
            "niger": "NER",
            "nigeria": "NGA",
            "ghana": "GHA",
            "côte d'ivoire": "CIV",
            "ivory coast": "CIV",
            "cameroon": "CMR",
            "ethiopia": "ETH",
            "kenya": "KEN",
            "tanzania": "TZA",
            "uganda": "UGA",
            "zambia": "ZMB",
            "zimbabwe": "ZWE",
            "mozambique": "MOZ",
            "malawi": "MWI",
            "south africa": "ZAF",
        }
        return iso3_map.get(country_name.lower(), "UNK")

    def list_available_regions(
        self,
        shapefile_path: Union[str, Path],
        name_field: Optional[str] = None,
    ) -> List[str]:
        """List all available region names in a shapefile.

        Useful for discovering what regions are available before filtering.

        Args:
            shapefile_path: Path to GADM shapefile
            name_field: Column containing region names (auto-detected if None)

        Returns:
            Sorted list of unique region names
        """
        gdf = gpd.read_file(shapefile_path)

        if name_field is None:
            # Try to find the appropriate name column
            for level in range(5, -1, -1):
                col = f"NAME_{level}" if level > 0 else "COUNTRY"
                if col in gdf.columns:
                    name_field = col
                    break

        if name_field is None or name_field not in gdf.columns:
            raise ValueError(f"Could not find name column in {shapefile_path}")

        return sorted(gdf[name_field].dropna().unique().tolist())

    def get_shapefile_info(
        self,
        shapefile_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Get information about a GADM shapefile.

        Args:
            shapefile_path: Path to shapefile

        Returns:
            Dictionary with shapefile metadata
        """
        gdf = gpd.read_file(shapefile_path)

        info = {
            "path": str(shapefile_path),
            "crs": str(gdf.crs) if gdf.crs else None,
            "feature_count": len(gdf),
            "columns": list(gdf.columns),
            "total_bounds": list(gdf.total_bounds),
            "name_columns": [c for c in gdf.columns if c.startswith("NAME") or c == "COUNTRY"],
        }

        # Add sample values for name columns
        info["sample_values"] = {}
        for col in info["name_columns"]:
            unique_vals = gdf[col].dropna().unique()[:5]
            info["sample_values"][col] = list(unique_vals)

        return info
