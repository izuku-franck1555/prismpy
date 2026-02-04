"""
Spatial harmonization utilities for prismpy.

This module provides functionality for:
- Resampling rasters to different resolutions
- Reprojecting between coordinate reference systems
- Aligning grids across different data sources
- Spatial aggregation and interpolation

Reference: Platform-specific grid requirements from SARRA-Py, CRAFT, PYTHIA, ACEA.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from prismpy.models.region import BoundingBox, Region
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.provenance.tracker import DecisionType, ProvenanceTracker


logger = logging.getLogger(__name__)


class ResampleMethod(str, Enum):
    """Resampling methods for raster data."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    AVERAGE = "average"
    MODE = "mode"  # For categorical data
    MIN = "min"
    MAX = "max"


class AggregationMethod(str, Enum):
    """Methods for spatial aggregation."""
    MEAN = "mean"
    SUM = "sum"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    MAJORITY = "majority"  # For categorical


# Platform-specific grid specifications
PLATFORM_GRIDS = {
    "sarra_py": {
        "resolution": 0.0375,  # ~4km (TAMSAT native)
        "crs": "EPSG:4326",
    },
    "craft": {
        "resolution": 5 / 60,  # 5 arc-minutes
        "crs": "EPSG:4326",
        "global_cols": 4320,
        "global_rows": 2160,
    },
    "pythia": {
        "resolution": 5 / 60,  # 5 arc-minutes
        "crs": "EPSG:4326",
    },
    "acea": {
        "resolution_5arcmin": 5 / 60,
        "resolution_30arcmin": 30 / 60,
        "crs": "EPSG:4326",
    },
}


@dataclass
class HarmonizationResult:
    """Result of a spatial harmonization operation.

    Attributes:
        success: Whether harmonization succeeded
        data: Harmonized data (array or grid)
        source_resolution: Original resolution
        target_resolution: Target resolution
        method: Method used
        metadata: Additional metadata
        warnings: List of warnings
    """
    success: bool
    data: Any
    source_resolution: float
    target_resolution: float
    method: str
    metadata: Dict[str, Any]
    warnings: List[str]


class SpatialHarmonizer:
    """Handles spatial harmonization between different data sources and platforms.

    Key responsibilities:
    1. Resample data to target resolution
    2. Align grids to platform-specific specifications
    3. Handle CRS transformations
    4. Aggregate or interpolate as needed
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the spatial harmonizer.

        Args:
            provenance: Provenance tracker for recording decisions
        """
        self.provenance = provenance

    def resample_to_resolution(
        self,
        data: np.ndarray,
        source_resolution: float,
        target_resolution: float,
        method: ResampleMethod = ResampleMethod.BILINEAR,
        bounds: Optional[BoundingBox] = None,
    ) -> HarmonizationResult:
        """Resample data array to a different resolution.

        Args:
            data: 2D numpy array
            source_resolution: Source resolution in degrees
            target_resolution: Target resolution in degrees
            method: Resampling method
            bounds: Optional bounding box for georeferencing

        Returns:
            HarmonizationResult with resampled data
        """
        warnings = []

        if source_resolution == target_resolution:
            return HarmonizationResult(
                success=True,
                data=data,
                source_resolution=source_resolution,
                target_resolution=target_resolution,
                method="no_change",
                metadata={"reason": "resolutions match"},
                warnings=[],
            )

        # Calculate scale factor
        scale_factor = source_resolution / target_resolution

        # Determine new dimensions
        new_height = int(data.shape[0] * scale_factor)
        new_width = int(data.shape[1] * scale_factor)

        if new_height < 1 or new_width < 1:
            return HarmonizationResult(
                success=False,
                data=None,
                source_resolution=source_resolution,
                target_resolution=target_resolution,
                method=method.value,
                metadata={"error": "Invalid target dimensions"},
                warnings=["Target resolution too coarse for input data"],
            )

        try:
            # Use scipy for resampling
            from scipy import ndimage

            if method == ResampleMethod.NEAREST:
                order = 0
            elif method == ResampleMethod.BILINEAR:
                order = 1
            elif method == ResampleMethod.CUBIC:
                order = 3
            else:
                order = 1  # Default to bilinear

            # Handle NaN values
            mask = np.isnan(data)
            if mask.any():
                # Fill NaN temporarily for resampling
                data_filled = np.where(mask, np.nanmean(data), data)
                resampled = ndimage.zoom(data_filled, scale_factor, order=order)
                # Resample mask and reapply
                mask_resampled = ndimage.zoom(mask.astype(float), scale_factor, order=0) > 0.5
                resampled = np.where(mask_resampled, np.nan, resampled)
            else:
                resampled = ndimage.zoom(data, scale_factor, order=order)

        except ImportError:
            # Fallback to simple resampling
            warnings.append("scipy not available, using simple resampling")
            resampled = self._simple_resample(data, new_height, new_width, method)

        # Record provenance
        if self.provenance:
            self.provenance.record_decision(
                decision_type=DecisionType.SPATIAL_ALIGNMENT,
                description=f"Resampled from {source_resolution}° to {target_resolution}°",
                rationale=f"Platform requires {target_resolution}° resolution",
                alternatives=["nearest", "bilinear", "cubic"],
                reference=f"Method: {method.value}",
            )

        return HarmonizationResult(
            success=True,
            data=resampled,
            source_resolution=source_resolution,
            target_resolution=target_resolution,
            method=method.value,
            metadata={
                "original_shape": data.shape,
                "resampled_shape": resampled.shape,
                "scale_factor": scale_factor,
            },
            warnings=warnings,
        )

    def align_to_platform_grid(
        self,
        data: np.ndarray,
        bounds: BoundingBox,
        platform: str,
        source_resolution: float,
    ) -> HarmonizationResult:
        """Align data to a platform-specific grid.

        Args:
            data: 2D numpy array
            bounds: Bounding box of the data
            platform: Target platform (sarra_py, craft, pythia, acea)
            source_resolution: Source data resolution

        Returns:
            HarmonizationResult with aligned data
        """
        warnings = []

        if platform not in PLATFORM_GRIDS:
            return HarmonizationResult(
                success=False,
                data=None,
                source_resolution=source_resolution,
                target_resolution=0,
                method="align",
                metadata={"error": f"Unknown platform: {platform}"},
                warnings=[f"Platform {platform} not recognized"],
            )

        grid_spec = PLATFORM_GRIDS[platform]

        # Handle ACEA dual resolution
        if platform == "acea":
            target_resolution = grid_spec.get("resolution_5arcmin", 5 / 60)
        else:
            target_resolution = grid_spec["resolution"]

        # Resample if needed
        if abs(source_resolution - target_resolution) > 1e-6:
            result = self.resample_to_resolution(
                data=data,
                source_resolution=source_resolution,
                target_resolution=target_resolution,
                method=ResampleMethod.BILINEAR,
                bounds=bounds,
            )
            if not result.success:
                return result
            aligned_data = result.data
            warnings.extend(result.warnings)
        else:
            aligned_data = data

        # For CRAFT, compute cell IDs
        metadata = {
            "platform": platform,
            "target_resolution": target_resolution,
        }

        if platform == "craft":
            # CRAFT uses global 5-arcmin cell IDs
            metadata["cell_id_formula"] = "row * 4320 + col"

        return HarmonizationResult(
            success=True,
            data=aligned_data,
            source_resolution=source_resolution,
            target_resolution=target_resolution,
            method="platform_alignment",
            metadata=metadata,
            warnings=warnings,
        )

    def create_unified_grid(
        self,
        region: Region,
        resolution: float,
    ) -> SpatialGrid:
        """Create a unified spatial grid for a region.

        Args:
            region: Region with bounding box
            resolution: Grid resolution in degrees

        Returns:
            SpatialGrid object
        """
        return SpatialGrid.from_bounds(
            bounds=region.bounds,
            resolution=resolution,
        )

    def aggregate_to_coarser(
        self,
        data: np.ndarray,
        factor: int,
        method: AggregationMethod = AggregationMethod.MEAN,
    ) -> np.ndarray:
        """Aggregate data to a coarser resolution.

        Args:
            data: 2D numpy array
            factor: Aggregation factor (e.g., 6 for 5arcmin -> 30arcmin)
            method: Aggregation method

        Returns:
            Aggregated array
        """
        # Ensure dimensions are divisible
        h, w = data.shape
        new_h = h // factor
        new_w = w // factor
        trimmed = data[:new_h * factor, :new_w * factor]

        # Reshape for aggregation
        reshaped = trimmed.reshape(new_h, factor, new_w, factor)

        if method == AggregationMethod.MEAN:
            return np.nanmean(reshaped, axis=(1, 3))
        elif method == AggregationMethod.SUM:
            return np.nansum(reshaped, axis=(1, 3))
        elif method == AggregationMethod.MEDIAN:
            return np.nanmedian(reshaped, axis=(1, 3))
        elif method == AggregationMethod.MIN:
            return np.nanmin(reshaped, axis=(1, 3))
        elif method == AggregationMethod.MAX:
            return np.nanmax(reshaped, axis=(1, 3))
        elif method == AggregationMethod.MAJORITY:
            # For categorical data
            from scipy import stats
            result = np.zeros((new_h, new_w))
            for i in range(new_h):
                for j in range(new_w):
                    block = reshaped[i, :, j, :].flatten()
                    block = block[~np.isnan(block)]
                    if len(block) > 0:
                        result[i, j] = stats.mode(block, keepdims=False)[0]
                    else:
                        result[i, j] = np.nan
            return result
        else:
            return np.nanmean(reshaped, axis=(1, 3))

    def interpolate_to_points(
        self,
        data: np.ndarray,
        bounds: BoundingBox,
        points: List[Tuple[float, float]],
        method: str = "bilinear",
    ) -> List[float]:
        """Interpolate gridded data to specific points.

        Args:
            data: 2D numpy array
            bounds: Bounding box of the data
            points: List of (lat, lon) tuples
            method: Interpolation method ('nearest' or 'bilinear')

        Returns:
            List of interpolated values
        """
        from scipy import interpolate

        # Create coordinate arrays
        lats = np.linspace(bounds.maxy, bounds.miny, data.shape[0])
        lons = np.linspace(bounds.minx, bounds.maxx, data.shape[1])

        if method == "nearest":
            # Nearest neighbor
            values = []
            for lat, lon in points:
                i = np.argmin(np.abs(lats - lat))
                j = np.argmin(np.abs(lons - lon))
                values.append(float(data[i, j]))
        else:
            # Bilinear interpolation
            interp_func = interpolate.RegularGridInterpolator(
                (lats[::-1], lons),  # Flip lats for ascending order
                data[::-1, :],
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            values = [float(interp_func((lat, lon))) for lat, lon in points]

        return values

    def reproject_bounds(
        self,
        bounds: BoundingBox,
        target_crs: str,
    ) -> BoundingBox:
        """Reproject bounding box to a different CRS.

        Args:
            bounds: Source bounding box
            target_crs: Target CRS (e.g., "EPSG:32632")

        Returns:
            Reprojected bounding box
        """
        if bounds.crs == target_crs:
            return bounds

        try:
            from pyproj import Transformer

            transformer = Transformer.from_crs(
                bounds.crs, target_crs, always_xy=True
            )

            # Transform corners
            minx, miny = transformer.transform(bounds.minx, bounds.miny)
            maxx, maxy = transformer.transform(bounds.maxx, bounds.maxy)

            return BoundingBox(
                minx=minx,
                miny=miny,
                maxx=maxx,
                maxy=maxy,
                crs=target_crs,
            )

        except ImportError:
            logger.warning("pyproj not available for reprojection")
            return bounds

    def _simple_resample(
        self,
        data: np.ndarray,
        new_height: int,
        new_width: int,
        method: ResampleMethod,
    ) -> np.ndarray:
        """Simple resampling without scipy.

        Args:
            data: Input array
            new_height: Target height
            new_width: Target width
            method: Resampling method

        Returns:
            Resampled array
        """
        # Simple nearest-neighbor resampling
        row_indices = np.linspace(0, data.shape[0] - 1, new_height).astype(int)
        col_indices = np.linspace(0, data.shape[1] - 1, new_width).astype(int)

        return data[np.ix_(row_indices, col_indices)]

    def compute_cell_ids_craft(
        self,
        bounds: BoundingBox,
        resolution: float = 5 / 60,
    ) -> List[int]:
        """Compute CRAFT cell IDs for a region.

        CRAFT uses a global 5-arcmin grid with cell ID = row * 4320 + col.

        Args:
            bounds: Region bounding box
            resolution: Grid resolution (default 5 arcmin)

        Returns:
            List of cell IDs within the region
        """
        GLOBAL_COLS = 4320
        GLOBAL_ROWS = 2160

        # Calculate row/col ranges
        # Row 0 is at lat 90, col 0 is at lon -180
        min_row = int((90 - bounds.maxy) / resolution)
        max_row = int((90 - bounds.miny) / resolution)
        min_col = int((bounds.minx + 180) / resolution)
        max_col = int((bounds.maxx + 180) / resolution)

        # Clamp to valid range
        min_row = max(0, min(min_row, GLOBAL_ROWS - 1))
        max_row = max(0, min(max_row, GLOBAL_ROWS - 1))
        min_col = max(0, min(min_col, GLOBAL_COLS - 1))
        max_col = max(0, min(max_col, GLOBAL_COLS - 1))

        cell_ids = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                cell_id = row * GLOBAL_COLS + col
                cell_ids.append(cell_id)

        return cell_ids

    def get_cell_coordinates(
        self,
        cell_id: int,
        resolution: float = 5 / 60,
    ) -> Tuple[float, float]:
        """Get center coordinates for a CRAFT cell ID.

        Args:
            cell_id: CRAFT cell ID
            resolution: Grid resolution

        Returns:
            Tuple of (lat, lon) for cell center
        """
        GLOBAL_COLS = 4320

        row = cell_id // GLOBAL_COLS
        col = cell_id % GLOBAL_COLS

        lat = 90 - (row + 0.5) * resolution
        lon = -180 + (col + 0.5) * resolution

        return (lat, lon)
