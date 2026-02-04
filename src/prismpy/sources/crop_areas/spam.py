"""
SPAM (Spatial Production Allocation Model) crop area data source.

This module provides functionality to access SPAM crop harvested area data,
which provides global gridded crop statistics at 5 arc-minute resolution.

SPAM data is used for crop masks and harvested area weighting.

Reference: ACEA/06-HARVESTED-AREAS-PREPARATION/ implementation patterns.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from prismpy.models.region import Region
from prismpy.provenance.tracker import ProvenanceTracker
from prismpy.sources.base import DataSource, RetrievalResult


logger = logging.getLogger(__name__)


# SPAM crop codes
SPAM_CROPS = {
    "maize": "MAIZ",
    "wheat": "WHEA",
    "rice": "RICE",
    "sorghum": "SORG",
    "millet": "PMIL",  # Pearl millet
    "barley": "BARL",
    "cassava": "CASS",
    "groundnut": "GROU",
    "soybean": "SOYB",
    "cotton": "COTT",
}

# SPAM technology levels
SPAM_TECH_LEVELS = ["A", "H", "I", "L", "R", "S"]  # All, High, Irrigated, Low, Rainfed, Subsistence


@dataclass
class SPAMConfig:
    """Configuration for SPAM data access."""

    data_dir: Optional[Path] = None
    version: str = "2020"  # SPAM version year
    resolution: str = "5min"  # 5 arc-minutes
    variable: str = "H"  # H=Harvested area, P=Physical area, Y=Yield, V=Value


@dataclass
class SPAMData:
    """Container for SPAM crop area data.

    Attributes:
        region_name: Name of the region
        bounds: Bounding box
        crop: Crop name
        crop_code: SPAM crop code
        tech_level: Technology level
        data_path: Path to data file
        values: Dictionary mapping cell ID to harvested area (hectares)
        total_area: Total harvested area in region
        statistics: Summary statistics
    """

    region_name: str
    bounds: List[float]
    crop: str
    crop_code: str
    tech_level: str
    data_path: Optional[Path]
    values: Dict[int, float]
    total_area: float
    statistics: Dict[str, Any]


class SPAMSource(DataSource):
    """Data source for SPAM crop harvested area data.

    SPAM provides global crop production statistics allocated to grid cells.
    Data is available at 5 arc-minute (~10km) resolution.

    File naming convention:
    spam{year}V{tech}{variable}_{crop}.tif

    Example: spam2020VHA_MAIZ.tif (2020, All tech, Harvested area, Maize)

    Attributes:
        NAME: Data source identifier
        CROPS: Available crop mappings
        VERSION: Default SPAM version
    """

    NAME = "spam"
    CROPS = SPAM_CROPS
    VERSION = "2020"
    RESOLUTION = 5 / 60  # 5 arc-minutes in degrees
    CRS = "EPSG:4326"

    def __init__(
        self,
        config: Optional[SPAMConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the SPAM data source.

        Args:
            config: SPAM configuration
            cache_dir: Directory for caching
            provenance: Provenance tracker
        """
        super().__init__(cache_dir=cache_dir, provenance=provenance)
        self.config = config or SPAMConfig()

    def retrieve(
        self,
        region: Region,
        crop: str,
        tech_level: str = "A",  # All technologies
        cell_coords: Optional[List[Tuple[float, float]]] = None,
        raster_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve SPAM crop area data for a region.

        Args:
            region: Region with bounding box
            crop: Crop name (e.g., "maize") or SPAM code (e.g., "MAIZ")
            tech_level: Technology level (A=All, H=High, etc.)
            cell_coords: List of (lat, lon) coordinates to sample
            raster_path: Direct path to SPAM raster file
            **kwargs: Additional parameters

        Returns:
            RetrievalResult containing SPAMData object
        """
        errors = []
        warnings = []
        metadata = {"source": self.NAME, "version": self.config.version}

        # Resolve crop code
        crop_code = self.CROPS.get(crop.lower(), crop.upper())
        metadata["crop"] = crop
        metadata["crop_code"] = crop_code
        metadata["tech_level"] = tech_level

        # Find raster file
        if raster_path:
            raster_path = Path(raster_path)
        else:
            raster_path = self._find_spam_file(crop_code, tech_level)

        if not raster_path or not raster_path.exists():
            return self.create_result(
                success=False,
                errors=[f"SPAM file not found for {crop_code}"],
                metadata=metadata,
            )

        metadata["raster_path"] = str(raster_path)

        # Extract values
        values = {}
        if cell_coords:
            try:
                raw_values = self._sample_raster(raster_path, cell_coords)
                values = {i: v for i, v in enumerate(raw_values)}
            except Exception as e:
                return self.create_result(
                    success=False,
                    errors=[f"Raster sampling failed: {e}"],
                    metadata=metadata,
                )
        else:
            # Extract from bounds
            try:
                values = self._extract_from_bounds(raster_path, region.bounds)
            except Exception as e:
                warnings.append(f"Bounds extraction failed: {e}")

        # Calculate statistics
        area_values = [v for v in values.values() if v > 0]
        total_area = sum(area_values)
        stats = {
            "total_area_ha": total_area,
            "cells_with_crop": len(area_values),
            "mean_area_ha": np.mean(area_values) if area_values else 0,
            "max_area_ha": max(area_values) if area_values else 0,
        }

        spam_data = SPAMData(
            region_name=region.name,
            bounds=region.bounds.to_gis_format(),
            crop=crop,
            crop_code=crop_code,
            tech_level=tech_level,
            data_path=raster_path,
            values=values,
            total_area=total_area,
            statistics=stats,
        )

        metadata["statistics"] = stats

        if total_area == 0:
            warnings.append(f"No {crop} harvested area found in region")

        return self.create_result(
            success=True,
            data=spam_data,
            warnings=warnings,
            metadata=metadata,
        )

    def validate(self, data: Any) -> List[str]:
        """Validate SPAM data."""
        warnings = []

        if not isinstance(data, SPAMData):
            return [f"Expected SPAMData, got {type(data)}"]

        if data.total_area == 0:
            warnings.append("No crop area found in region")

        if data.statistics["cells_with_crop"] == 0:
            warnings.append("No cells with non-zero crop area")

        return warnings

    def _find_spam_file(
        self,
        crop_code: str,
        tech_level: str,
    ) -> Optional[Path]:
        """Find SPAM raster file.

        Args:
            crop_code: SPAM crop code
            tech_level: Technology level

        Returns:
            Path to file or None
        """
        if not self.config.data_dir:
            return None

        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            return None

        # Try different naming patterns
        patterns = [
            f"spam{self.config.version}V{tech_level}{self.config.variable}_{crop_code}.tif",
            f"spam{self.config.version}v{tech_level.lower()}{self.config.variable.lower()}_{crop_code.lower()}.tif",
            f"*{crop_code}*.tif",
            f"*{crop_code.lower()}*.tif",
        ]

        for pattern in patterns:
            matches = list(data_dir.glob(pattern))
            if matches:
                return matches[0]

            # Try in subdirectories
            matches = list(data_dir.glob(f"**/{pattern}"))
            if matches:
                return matches[0]

        return None

    def _sample_raster(
        self,
        raster_path: Path,
        coords: List[Tuple[float, float]],
    ) -> List[float]:
        """Sample raster values at coordinates.

        Args:
            raster_path: Path to GeoTIFF
            coords: List of (lat, lon) tuples

        Returns:
            List of harvested area values (hectares)
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required for SPAM data extraction")

        with rasterio.open(raster_path) as src:
            # Convert (lat, lon) to (lon, lat)
            xy_coords = [(lon, lat) for lat, lon in coords]

            values = []
            for val in src.sample(xy_coords):
                v = val[0]
                # Handle nodata
                if src.nodata is not None and v == src.nodata:
                    v = 0.0
                elif v < 0:
                    v = 0.0
                values.append(float(v))

        return values

    def _extract_from_bounds(
        self,
        raster_path: Path,
        bounds,
    ) -> Dict[int, float]:
        """Extract values within bounding box.

        Args:
            raster_path: Path to GeoTIFF
            bounds: BoundingBox object

        Returns:
            Dictionary mapping cell index to area value
        """
        try:
            import rasterio
            from rasterio.windows import from_bounds
        except ImportError:
            raise ImportError("rasterio required")

        with rasterio.open(raster_path) as src:
            window = from_bounds(
                bounds.minx, bounds.miny, bounds.maxx, bounds.maxy,
                src.transform
            )
            data = src.read(1, window=window)

            # Create values dictionary
            values = {}
            idx = 0
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    v = data[i, j]
                    if src.nodata is None or v != src.nodata:
                        if v > 0:
                            values[idx] = float(v)
                        else:
                            values[idx] = 0.0
                    else:
                        values[idx] = 0.0
                    idx += 1

        return values

    def generate_crop_mask(
        self,
        spam_data: SPAMData,
        threshold: float = 0.0,
    ) -> Dict[int, bool]:
        """Generate binary crop mask from harvested area data.

        Args:
            spam_data: SPAMData object
            threshold: Minimum area threshold (hectares)

        Returns:
            Dictionary mapping cell ID to presence (True/False)
        """
        return {
            cell_id: area > threshold
            for cell_id, area in spam_data.values.items()
        }

    def list_available_crops(self) -> List[str]:
        """List available crop names."""
        return list(self.CROPS.keys())

    def clip_to_file(
        self,
        input_path: Path,
        output_path: Path,
        bounds: "BoundingBox",
    ) -> Optional[Path]:
        """Clip a global SPAM raster to region bounds and save.

        Args:
            input_path: Path to global SPAM GeoTIFF
            output_path: Path for output clipped GeoTIFF
            bounds: BoundingBox with minx, miny, maxx, maxy

        Returns:
            Path to clipped file, or None if failed
        """
        try:
            import rasterio
            from rasterio.windows import from_bounds
        except ImportError:
            logger.error("rasterio required for SPAM clipping")
            return None

        if not input_path.exists():
            logger.error(f"Input SPAM file not found: {input_path}")
            return None

        try:
            with rasterio.open(input_path) as src:
                # Calculate window from bounds
                window = from_bounds(
                    bounds.minx, bounds.miny, bounds.maxx, bounds.maxy,
                    src.transform
                )

                # Read clipped data
                clipped_data = src.read(1, window=window)

                # Calculate new transform for clipped area
                new_transform = src.window_transform(window)

                # Create output profile
                profile = src.profile.copy()
                profile.update({
                    'height': clipped_data.shape[0],
                    'width': clipped_data.shape[1],
                    'transform': new_transform,
                })

                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Write clipped raster
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(clipped_data, 1)

                logger.info(f"Clipped SPAM raster to: {output_path}")
                return output_path

        except Exception as e:
            logger.error(f"Failed to clip SPAM raster: {e}")
            return None
