"""
iSDA (iSDAsoil) soil data source retriever for Africa.

This module provides functionality to access iSDAsoil raster data,
which provides soil property predictions for Africa at ~30m resolution.

iSDA is primarily used by SARRA-Py for root zone depth and soil properties.

Reference: SARRA-Py/03-SOIL-PREPARATION/ implementation patterns.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from prismpy.models.region import Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.provenance.tracker import ProvenanceTracker
from prismpy.sources.base import DataSource, RetrievalResult


logger = logging.getLogger(__name__)


# iSDA variable mappings
ISDA_VARIABLES = {
    "rzd": "gyga_af_erzd__m_1km.tif",  # Root zone depth
    "sand": "sand_tot_psa_0-20cm_mean.tif",
    "clay": "clay_tot_psa_0-20cm_mean.tif",
    "soc": "oc_0-20cm_mean.tif",  # Organic carbon
    "ph": "ph_h2o_0-20cm_mean.tif",
    "bd": "bdod_0-20cm_mean.tif",  # Bulk density
}


@dataclass
class iSDAConfig:
    """Configuration for iSDA data access."""

    data_dir: Optional[Path] = None
    sarra_py_assets_dir: Optional[Path] = None  # SARRA-Py/data/assets/
    resolution: float = 0.00833  # ~1km resampled


@dataclass
class iSDAData:
    """Container for iSDA soil data.

    Attributes:
        region_name: Name of the region
        bounds: Bounding box
        data_dir: Directory containing iSDA files
        variables: Dictionary of available variables and their file paths
        profiles: Dictionary of SoilProfile objects by location ID
    """

    region_name: str
    bounds: List[float]
    data_dir: Path
    variables: Dict[str, Path]
    profiles: Optional[Dict[int, SoilProfile]] = None


class iSDASource(DataSource):
    """Data source for iSDA (iSDAsoil) African soil data.

    iSDAsoil provides machine learning-based predictions of soil properties
    across Africa at high resolution (~30m native, often resampled to ~1km).

    Data can be loaded from:
    1. SARRA-Py installation assets directory
    2. Downloaded iSDA GeoTIFF files

    Attributes:
        NAME: Data source identifier
        VARIABLES: Available soil variables
        CRS: Coordinate reference system
    """

    NAME = "isda"
    VARIABLES = ISDA_VARIABLES
    CRS = "EPSG:4326"

    def __init__(
        self,
        config: Optional[iSDAConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the iSDA data source.

        Args:
            config: iSDA configuration
            cache_dir: Directory for caching
            provenance: Provenance tracker
        """
        super().__init__(cache_dir=cache_dir, provenance=provenance)
        self.config = config or iSDAConfig()

    def retrieve(
        self,
        region: Region,
        data_dir: Optional[Union[str, Path]] = None,
        variables: Optional[List[str]] = None,
        extract_profiles: bool = False,
        grid_points: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve iSDA soil data for a region.

        Args:
            region: Region with bounding box
            data_dir: Directory containing iSDA files
            variables: Variables to retrieve (default: all available)
            extract_profiles: Whether to extract soil profiles at grid points
            grid_points: List of (lat, lon) tuples for profile extraction
            **kwargs: Additional parameters

        Returns:
            RetrievalResult containing iSDAData object
        """
        errors = []
        warnings = []
        metadata = {"source": self.NAME}

        # Determine data directory
        if data_dir:
            data_dir = Path(data_dir)
        elif self.config.data_dir:
            data_dir = self.config.data_dir
        elif self.config.sarra_py_assets_dir:
            data_dir = self.config.sarra_py_assets_dir
        else:
            data_dir = self.cache_dir / "isda"

        metadata["data_dir"] = str(data_dir)

        # Check available variables
        available_vars = {}
        if data_dir.exists():
            for var_name, filename in self.VARIABLES.items():
                file_path = data_dir / filename
                if file_path.exists():
                    available_vars[var_name] = file_path
                else:
                    # Try without subdirectory
                    alt_patterns = [
                        data_dir / f"*{var_name}*.tif",
                        data_dir / "**" / filename,
                    ]
                    for pattern in alt_patterns:
                        matches = list(data_dir.glob(str(pattern).replace(str(data_dir) + "/", "")))
                        if matches:
                            available_vars[var_name] = matches[0]
                            break

        if not available_vars:
            return self.create_result(
                success=False,
                errors=[f"No iSDA files found in {data_dir}"],
                metadata=metadata,
            )

        metadata["available_variables"] = list(available_vars.keys())

        # Filter to requested variables
        if variables:
            available_vars = {k: v for k, v in available_vars.items() if k in variables}

        # Create iSDAData object
        isda_data = iSDAData(
            region_name=region.name,
            bounds=region.bounds.to_gis_format(),
            data_dir=data_dir,
            variables=available_vars,
        )

        # Extract profiles if requested
        if extract_profiles and grid_points:
            try:
                profiles = self._extract_profiles(
                    available_vars=available_vars,
                    grid_points=grid_points,
                    region=region,
                    progress_callback=kwargs.get('progress_callback'),
                )
                isda_data.profiles = profiles
                metadata["profile_count"] = len(profiles)
            except Exception as e:
                warnings.append(f"Profile extraction failed: {e}")

        return self.create_result(
            success=True,
            data=isda_data,
            output_path=data_dir,
            warnings=warnings,
            metadata=metadata,
        )

    def validate(self, data: Any) -> List[str]:
        """Validate iSDA data.

        Args:
            data: iSDAData object

        Returns:
            List of validation messages
        """
        warnings = []

        if not isinstance(data, iSDAData):
            return [f"Expected iSDAData, got {type(data)}"]

        if not data.data_dir.exists():
            warnings.append(f"Data directory does not exist: {data.data_dir}")

        if not data.variables:
            warnings.append("No variables available")

        return warnings

    def load_variable(
        self,
        file_path: Union[str, Path],
        bounds: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a single variable from GeoTIFF.

        Args:
            file_path: Path to GeoTIFF file
            bounds: Optional bounding box for clipping [minx, miny, maxx, maxy]

        Returns:
            Tuple of (data array, metadata dict)
        """
        try:
            import rasterio
            from rasterio.windows import from_bounds
        except ImportError:
            raise ImportError("rasterio required for loading iSDA files")

        with rasterio.open(file_path) as src:
            if bounds:
                window = from_bounds(*bounds, src.transform)
                data = src.read(1, window=window)
                transform = src.window_transform(window)
            else:
                data = src.read(1)
                transform = src.transform

            meta = {
                "crs": str(src.crs),
                "transform": transform,
                "nodata": src.nodata,
                "shape": data.shape,
            }

        # Handle nodata
        if meta["nodata"] is not None:
            data = np.where(data == meta["nodata"], np.nan, data)

        return data, meta

    def sample_at_points(
        self,
        file_path: Union[str, Path],
        points: List[Tuple[float, float]],
    ) -> List[Optional[float]]:
        """Sample raster values at specific points.

        Args:
            file_path: Path to GeoTIFF file
            points: List of (lat, lon) tuples

        Returns:
            List of values (None for nodata/invalid points)
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required for sampling iSDA files")

        with rasterio.open(file_path) as src:
            # Convert points to (lon, lat) for rasterio
            coords = [(lon, lat) for lat, lon in points]

            values = []
            for val in src.sample(coords):
                v = val[0]
                if src.nodata is not None and v == src.nodata:
                    values.append(None)
                else:
                    values.append(float(v))

        return values

    def _extract_profiles(
        self,
        available_vars: Dict[str, Path],
        grid_points: List[Tuple[float, float]],
        region: Region,
        progress_callback=None,
    ) -> Dict[int, SoilProfile]:
        """Extract soil profiles at grid points.

        Args:
            available_vars: Dictionary of variable paths
            grid_points: List of (lat, lon) tuples
            region: Region object

        Returns:
            Dictionary mapping location ID to SoilProfile
        """
        profiles = {}

        # Sample each variable at all points
        var_values = {}
        var_names = list(available_vars.keys())
        for idx, var_name in enumerate(var_names):
            if progress_callback:
                progress_callback(idx + 1, len(var_names))
            var_values[var_name] = self.sample_at_points(available_vars[var_name], grid_points)

        # Create profiles
        for i, (lat, lon) in enumerate(grid_points):
            sand = var_values.get("sand", [None] * len(grid_points))[i]
            clay = var_values.get("clay", [None] * len(grid_points))[i]
            soc = var_values.get("soc", [None] * len(grid_points))[i]
            ph = var_values.get("ph", [None] * len(grid_points))[i]
            bd = var_values.get("bd", [None] * len(grid_points))[i]
            rzd = var_values.get("rzd", [None] * len(grid_points))[i]

            # Calculate silt if sand and clay available
            silt = None
            if sand is not None and clay is not None:
                silt = max(0, 100 - sand - clay)

            # Create single layer (iSDA is typically topsoil 0-20cm)
            layer = SoilLayer(
                depth_top=0.0,
                depth_bottom=0.2,  # 20cm
                sand=sand,
                clay=clay,
                silt=silt,
                organic_carbon=soc,
                bulk_density=bd / 100 if bd else None,  # Convert from cg/cm³
                ph=ph / 10 if ph else None,  # Convert from 10x scale
            )

            profile = SoilProfile(
                profile_id=f"iSDA_{region.country_iso3}_{i:06d}",
                lat=lat,
                lon=lon,
                source=self.NAME,
                layers=[layer],
                total_depth=rzd / 100 if rzd else 0.2,  # Convert from cm
                metadata={
                    "root_zone_depth": rzd,
                    "region": region.name,
                    # V2-22c-PRE.1.10 (D37) — cascade-provenance
                    # metadata fields. iSDA Africa S3 release; the
                    # source loader populates `cascade_rank=1`
                    # (primary success) by default. The cascade
                    # orchestrator at executor.py overrides this
                    # to rank=2 + fallback_attempts when iSDA
                    # returned no profile and HWSD took over.
                    "source": "iSDA Africa",
                    "version": "S3",
                    "cascade_rank": 1,
                    "fallback_attempts": [],
                },
            )
            profiles[i] = profile

        return profiles

    def get_grid_for_region(
        self,
        region: Region,
        resolution: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Generate grid points for a region.

        Args:
            region: Region with bounding box
            resolution: Grid resolution in degrees (default: config value)

        Returns:
            List of (lat, lon) tuples
        """
        resolution = resolution or self.config.resolution or 0.00833

        bounds = region.bounds
        points = []

        lat = bounds.miny + resolution / 2
        while lat < bounds.maxy:
            lon = bounds.minx + resolution / 2
            while lon < bounds.maxx:
                points.append((lat, lon))
                lon += resolution
            lat += resolution

        return points
