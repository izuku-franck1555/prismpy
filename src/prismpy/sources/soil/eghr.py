"""
eGHR (Global Harmonized Soils Reference) soil data source retriever.

This module provides functionality to access eGHR/GGCMI soil data,
which provides soil profile IDs mapped to DSSAT-compatible profiles.

eGHR is used by PYTHIA for linking grid cells to soil profiles.

Reference: PYTHIA/03-SOIL-PREPARATION/ implementation patterns.
"""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from prismpy.models.region import Region
from prismpy.models.soil import SoilProfile
from prismpy.provenance.tracker import ProvenanceTracker
from prismpy.sources.base import DataSource, RetrievalResult


logger = logging.getLogger(__name__)


@dataclass
class eGHRConfig:
    """Configuration for eGHR data access."""

    raster_path: Optional[Path] = None  # ggcmi_soils_2.tif
    database_path: Optional[Path] = None  # GHR.db
    sol_dir: Optional[Path] = None  # Directory with ML.SOL, NG.SOL, etc.


@dataclass
class eGHRData:
    """Container for eGHR soil data.

    Attributes:
        region_name: Name of the region
        bounds: Bounding box
        pixel_ids: Dictionary mapping cell ID to pixel ID
        profile_names: Dictionary mapping cell ID to profile name
        profile_coverage: Number of cells with valid profiles
    """

    region_name: str
    bounds: List[float]
    pixel_ids: Dict[int, int]
    profile_names: Dict[int, str]
    profile_coverage: float  # Percentage of cells with valid profiles


class eGHRSource(DataSource):
    """Data source for eGHR (Global Harmonized Soils Reference) data.

    eGHR provides a mapping from global grid pixels to DSSAT-compatible
    soil profile names. The workflow:
    1. Sample pixel IDs from GGCMI raster
    2. Look up profile names in GHR.db SQLite database
    3. Profiles are stored in country-specific .SOL files

    Attributes:
        NAME: Data source identifier
        NODATA: Raster nodata value
    """

    NAME = "eghr"
    NODATA = -9999
    CRS = "EPSG:4326"

    def __init__(
        self,
        config: Optional[eGHRConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the eGHR data source.

        Args:
            config: eGHR configuration
            cache_dir: Directory for caching
            provenance: Provenance tracker
        """
        super().__init__(cache_dir=cache_dir, provenance=provenance)
        self.config = config or eGHRConfig()

    def retrieve(
        self,
        region: Region,
        cell_coords: Optional[List[Tuple[float, float]]] = None,
        clip_raster: bool = False,
        output_raster: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve eGHR soil data for a region.

        Args:
            region: Region with bounding box
            cell_coords: List of (lat, lon) coordinates to sample
            clip_raster: Whether to clip raster to region bounds
            output_raster: Path for clipped raster output
            **kwargs: Additional parameters

        Returns:
            RetrievalResult containing eGHRData object
        """
        errors = []
        warnings = []
        metadata = {"source": self.NAME}

        # Check required files
        if not self.config.raster_path or not self.config.raster_path.exists():
            return self.create_result(
                success=False,
                errors=[f"eGHR raster not found: {self.config.raster_path}"],
                metadata=metadata,
            )

        if not self.config.database_path or not self.config.database_path.exists():
            return self.create_result(
                success=False,
                errors=[f"GHR database not found: {self.config.database_path}"],
                metadata=metadata,
            )

        # Clip raster if requested
        if clip_raster and output_raster:
            try:
                self._clip_raster_to_bounds(
                    input_path=self.config.raster_path,
                    output_path=output_raster,
                    bounds=region.bounds.to_gis_format(),
                )
                metadata["clipped_raster"] = str(output_raster)
            except Exception as e:
                warnings.append(f"Raster clipping failed: {e}")

        # Sample pixel IDs at coordinates
        pixel_ids = {}
        if cell_coords:
            try:
                pixel_ids = self._sample_pixel_ids(cell_coords)
                metadata["cells_sampled"] = len(cell_coords)
            except Exception as e:
                return self.create_result(
                    success=False,
                    errors=[f"Pixel sampling failed: {e}"],
                    metadata=metadata,
                )

        # Look up profile names from database
        profile_names = {}
        if pixel_ids:
            try:
                profile_names = self._lookup_profiles(pixel_ids)
            except Exception as e:
                warnings.append(f"Profile lookup failed: {e}")

        # Calculate coverage
        valid_profiles = sum(1 for p in profile_names.values() if p is not None)
        coverage = 100 * valid_profiles / len(cell_coords) if cell_coords else 0

        eghr_data = eGHRData(
            region_name=region.name,
            bounds=region.bounds.to_gis_format(),
            pixel_ids=pixel_ids,
            profile_names=profile_names,
            profile_coverage=coverage,
        )

        metadata["profile_coverage"] = coverage
        metadata["valid_profiles"] = valid_profiles

        if coverage < 90:
            warnings.append(f"Only {coverage:.1f}% of cells have valid soil profiles")

        return self.create_result(
            success=True,
            data=eghr_data,
            warnings=warnings,
            metadata=metadata,
        )

    def validate(self, data: Any) -> List[str]:
        """Validate eGHR data."""
        warnings = []

        if not isinstance(data, eGHRData):
            return [f"Expected eGHRData, got {type(data)}"]

        if data.profile_coverage < 80:
            warnings.append(
                f"Low profile coverage: {data.profile_coverage:.1f}% "
                "(less than 80% of cells have valid profiles)"
            )

        # Check for nodata pixels
        nodata_count = sum(1 for pid in data.pixel_ids.values() if pid == self.NODATA)
        if nodata_count > 0:
            warnings.append(f"{nodata_count} cells have nodata pixel IDs")

        return warnings

    def _sample_pixel_ids(
        self,
        coords: List[Tuple[float, float]],
    ) -> Dict[int, int]:
        """Sample pixel IDs from GGCMI raster at coordinates.

        Args:
            coords: List of (lat, lon) tuples

        Returns:
            Dictionary mapping cell index to pixel ID
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required for eGHR pixel sampling")

        with rasterio.open(self.config.raster_path) as src:
            # Convert (lat, lon) to (lon, lat) for rasterio
            xy_coords = [(lon, lat) for lat, lon in coords]

            pixel_ids = {}
            for i, val in enumerate(src.sample(xy_coords)):
                pixel_id = int(val[0])
                pixel_ids[i] = pixel_id

        return pixel_ids

    def _lookup_profiles(
        self,
        pixel_ids: Dict[int, int],
    ) -> Dict[int, Optional[str]]:
        """Look up profile names from GHR database.

        Args:
            pixel_ids: Dictionary mapping cell index to pixel ID

        Returns:
            Dictionary mapping cell index to profile name
        """
        profile_names = {}

        conn = sqlite3.connect(self.config.database_path)
        cursor = conn.cursor()

        # Check table structure
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Determine correct table and column names
        if "profile_map" in tables:
            table = "profile_map"
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]

            pid_col = "pixel_id" if "pixel_id" in columns else "PIXEL_ID"
            name_col = "profile_name" if "profile_name" in columns else "PROFILE_NAME"

            # Build lookup
            cursor.execute(f"SELECT {pid_col}, {name_col} FROM {table}")
            lookup = {row[0]: row[1] for row in cursor.fetchall()}

            for cell_id, pixel_id in pixel_ids.items():
                if pixel_id != self.NODATA:
                    profile_names[cell_id] = lookup.get(pixel_id)
                else:
                    profile_names[cell_id] = None
        else:
            self.logger.warning(f"profile_map table not found. Available: {tables}")
            for cell_id in pixel_ids:
                profile_names[cell_id] = None

        conn.close()
        return profile_names

    def _clip_raster_to_bounds(
        self,
        input_path: Path,
        output_path: Union[str, Path],
        bounds: List[float],
    ) -> None:
        """Clip raster to bounding box.

        Args:
            input_path: Input raster path
            output_path: Output raster path
            bounds: [minx, miny, maxx, maxy]
        """
        try:
            import rasterio
            from rasterio.windows import from_bounds
        except ImportError:
            raise ImportError("rasterio required for raster clipping")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(input_path) as src:
            window = from_bounds(*bounds, src.transform)
            data = src.read(window=window)
            transform = src.window_transform(window)

            profile = src.profile.copy()
            profile.update({
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": transform,
            })

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data)

        self.logger.info(f"Clipped raster saved to {output_path}")

    def get_sol_file_path(
        self,
        profile_name: str,
    ) -> Optional[Path]:
        """Get path to .SOL file for a profile.

        Profile names are typically country-prefixed (e.g., ML03371689).
        The country code determines which .SOL file to use.

        Args:
            profile_name: Profile name (e.g., "ML03371689")

        Returns:
            Path to .SOL file or None
        """
        if not profile_name or not self.config.sol_dir:
            return None

        # Extract country code (first 2 characters)
        country_code = profile_name[:2].upper()
        sol_file = self.config.sol_dir / f"{country_code}.SOL"

        if sol_file.exists():
            return sol_file

        # Try alternative naming
        alt_file = self.config.sol_dir / f"ML.SOL"  # DSSAT default
        if alt_file.exists():
            return alt_file

        return None

    def list_available_profiles(
        self,
        country_code: Optional[str] = None,
    ) -> List[str]:
        """List available profile names from database.

        Args:
            country_code: Optional 2-letter country code to filter

        Returns:
            List of profile names
        """
        conn = sqlite3.connect(self.config.database_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        if "profile_map" not in tables:
            conn.close()
            return []

        cursor.execute("PRAGMA table_info(profile_map)")
        columns = [row[1] for row in cursor.fetchall()]
        name_col = "profile_name" if "profile_name" in columns else "PROFILE_NAME"

        if country_code:
            cursor.execute(
                f"SELECT DISTINCT {name_col} FROM profile_map WHERE {name_col} LIKE ?",
                (f"{country_code}%",)
            )
        else:
            cursor.execute(f"SELECT DISTINCT {name_col} FROM profile_map")

        profiles = [row[0] for row in cursor.fetchall() if row[0]]

        conn.close()
        return sorted(profiles)
