"""
HWSD (Harmonized World Soil Database) soil data source retriever.

This module provides functionality to access HWSD v2.0 soil data,
which provides global soil properties at 30 arc-second (~1km) resolution.

HWSD is used by CRAFT and ACEA for soil properties.

Reference: ACEA/03-SOIL-PREPARATION/ implementation patterns.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from prismpy.models.region import Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.provenance.tracker import ProvenanceTracker
from prismpy.sources.base import DataSource, RetrievalResult


logger = logging.getLogger(__name__)


# HWSD variable mappings
HWSD_VARIABLES = {
    "sand": "T_SAND",  # Top-soil sand %
    "clay": "T_CLAY",  # Top-soil clay %
    "silt": "T_SILT",  # Top-soil silt %
    "soc": "T_OC",     # Top-soil organic carbon %
    "ph": "T_PH_H2O",  # Top-soil pH
    "bd": "T_BULK_DENSITY",  # Top-soil bulk density
    "gravel": "T_GRAVEL",  # Top-soil gravel %
}

# Default soil values (typical Sahel)
DEFAULT_SOIL = {
    "sand": 60.0,
    "clay": 18.0,
    "silt": 22.0,
    "soc": 0.5,
    "ph": 6.5,
    "bd": 1.4,
}


@dataclass
class HWSDConfig:
    """Configuration for HWSD data access."""

    bil_path: Optional[Path] = None  # HWSD2.bil raster
    mdb_path: Optional[Path] = None  # HWSD2.mdb database
    nc_path: Optional[Path] = None   # HWSD2.nc or .nc4
    use_defaults: bool = True        # Use defaults for missing data
    layer: str = "D1"                # D1=topsoil, D2=subsoil


@dataclass
class HWSDData:
    """Container for HWSD soil data.

    Attributes:
        region_name: Name of the region
        bounds: Bounding box
        source_type: Type of source used (bil_mdb, netcdf, csv)
        profiles: Dictionary of SoilProfile objects by cell ID
        statistics: Summary statistics
    """

    region_name: str
    bounds: List[float]
    source_type: str
    profiles: Dict[int, SoilProfile]
    statistics: Dict[str, Any]


class HWSDSource(DataSource):
    """Data source for HWSD (Harmonized World Soil Database) v2.0.

    HWSD provides global soil properties and can be accessed via:
    1. BIL raster + MDB database (most complete)
    2. NetCDF file
    3. Pre-processed CSV/raster files

    Attributes:
        NAME: Data source identifier
        VERSION: HWSD version
        VARIABLES: Available soil variables
    """

    NAME = "hwsd"
    VERSION = "2.0"
    VARIABLES = HWSD_VARIABLES
    CRS = "EPSG:4326"

    def __init__(
        self,
        config: Optional[HWSDConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the HWSD data source.

        Args:
            config: HWSD configuration
            cache_dir: Directory for caching
            provenance: Provenance tracker
        """
        super().__init__(cache_dir=cache_dir, provenance=provenance)
        self.config = config or HWSDConfig()

    def retrieve(
        self,
        region: Region,
        cell_ids: Optional[List[int]] = None,
        cell_coords: Optional[List[Tuple[float, float]]] = None,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve HWSD soil data for a region.

        Args:
            region: Region with bounding box
            cell_ids: Optional list of cell IDs for direct lookup
            cell_coords: Optional list of (lat, lon) coordinates
            **kwargs: Additional parameters

        Returns:
            RetrievalResult containing HWSDData object
        """
        errors = []
        warnings = []
        metadata = {
            "source": self.NAME,
            "version": self.VERSION,
            "layer": self.config.layer,
        }

        profiles = {}
        source_type = None

        # Try BIL + MDB method first
        if self.config.bil_path and self.config.mdb_path:
            try:
                profiles = self._extract_from_bil_mdb(
                    region=region,
                    cell_coords=cell_coords,
                )
                source_type = "bil_mdb"
                self.logger.info(f"Extracted {len(profiles)} profiles from BIL+MDB")
            except Exception as e:
                warnings.append(f"BIL+MDB extraction failed: {e}")

        # Try NetCDF method
        if not profiles and self.config.nc_path:
            try:
                profiles = self._extract_from_netcdf(
                    region=region,
                    cell_coords=cell_coords,
                )
                source_type = "netcdf"
                self.logger.info(f"Extracted {len(profiles)} profiles from NetCDF")
            except Exception as e:
                warnings.append(f"NetCDF extraction failed: {e}")

        # Apply defaults for missing data
        if self.config.use_defaults and not profiles:
            self.logger.warning("Using default soil values")
            warnings.append("No HWSD data found, using default values")

            if cell_coords:
                for i, (lat, lon) in enumerate(cell_coords):
                    profiles[i] = self._create_default_profile(i, lat, lon, region)
            source_type = "defaults"

        if not profiles:
            return self.create_result(
                success=False,
                errors=["Failed to extract HWSD data from any source"],
                warnings=warnings,
                metadata=metadata,
            )

        # Calculate statistics
        stats = self._calculate_statistics(profiles)

        hwsd_data = HWSDData(
            region_name=region.name,
            bounds=region.bounds.to_gis_format(),
            source_type=source_type,
            profiles=profiles,
            statistics=stats,
        )

        metadata["source_type"] = source_type
        metadata["profile_count"] = len(profiles)
        metadata["statistics"] = stats

        return self.create_result(
            success=True,
            data=hwsd_data,
            warnings=warnings,
            metadata=metadata,
        )

    def validate(self, data: Any) -> List[str]:
        """Validate HWSD data."""
        warnings = []

        if not isinstance(data, HWSDData):
            return [f"Expected HWSDData, got {type(data)}"]

        if not data.profiles:
            warnings.append("No soil profiles extracted")

        # Check for default-only profiles
        if data.source_type == "defaults":
            warnings.append("All profiles are defaults - no actual HWSD data used")

        # Validate individual profiles
        for profile_id, profile in data.profiles.items():
            if profile.layers:
                layer = profile.layers[0]
                if layer.sand is not None and layer.clay is not None:
                    total = layer.sand + layer.clay + (layer.silt or 0)
                    if abs(total - 100) > 5:
                        warnings.append(
                            f"Profile {profile_id}: texture fractions sum to {total:.1f}%"
                        )

        return warnings

    def _extract_from_bil_mdb(
        self,
        region: Region,
        cell_coords: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[int, SoilProfile]:
        """Extract soil data from BIL raster + MDB database.

        This is the most robust method for HWSD v2.0.

        Args:
            region: Region with bounding box
            cell_coords: Optional coordinates to sample

        Returns:
            Dictionary of soil profiles
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required for HWSD BIL extraction")

        profiles = {}

        # Step 1: Extract SMU IDs from BIL raster
        smu_ids = self._sample_bil_raster(cell_coords)

        # Step 2: Export HWSD2_LAYERS table from MDB
        layers_df = self._export_mdb_table()

        # Step 3: Create lookup and build profiles
        if layers_df is not None and smu_ids:
            # Validate required columns exist (HWSD v2.0 format)
            required_cols = ["LAYER", "SEQUENCE"]
            missing_cols = [c for c in required_cols if c not in layers_df.columns]
            if missing_cols:
                logger.error(f"HWSD MDB missing required columns: {missing_cols}")
                logger.error("Expected HWSD v2.0 format with LAYER, SEQUENCE columns.")
                logger.warning("Falling back to default soil profiles for all cells.")
                if self.config.use_defaults:
                    for i, (lat, lon) in enumerate(cell_coords or []):
                        profiles[i] = self._create_default_profile(i, lat, lon, region)
                return profiles

            # Filter to selected layer and sequence
            layer_filter = (layers_df["LAYER"] == self.config.layer) & (layers_df["SEQUENCE"] == 1)
            layers_df = layers_df[layer_filter]

            # Create lookup by SMU ID
            # Note: HWSD2 uses HWSD2_SMU_ID as the key, not 'ID'
            smu_col = "HWSD2_SMU_ID" if "HWSD2_SMU_ID" in layers_df.columns else "ID"
            soil_lookup = layers_df.set_index(smu_col).to_dict("index")

            for i, (lat, lon) in enumerate(cell_coords or []):
                smu_id = smu_ids[i] if i < len(smu_ids) else None

                if smu_id and smu_id in soil_lookup:
                    props = soil_lookup[smu_id]
                    profiles[i] = self._create_profile_from_hwsd(
                        cell_id=i, lat=lat, lon=lon, props=props, region=region
                    )
                elif self.config.use_defaults:
                    profiles[i] = self._create_default_profile(i, lat, lon, region)

        return profiles

    def _sample_bil_raster(
        self,
        coords: Optional[List[Tuple[float, float]]],
    ) -> List[Optional[int]]:
        """Sample SMU IDs from BIL raster at coordinates."""
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required")

        if not coords:
            return []

        with rasterio.open(self.config.bil_path) as src:
            # Convert (lat, lon) to (lon, lat) for rasterio
            xy_coords = [(lon, lat) for lat, lon in coords]
            values = list(src.sample(xy_coords))
            return [int(v[0]) if v[0] != src.nodata else None for v in values]

    def _export_mdb_table(self) -> Optional[pd.DataFrame]:
        """Export HWSD2_LAYERS table from MDB using mdbtools."""
        try:
            # Use mdb-export (from mdbtools package)
            result = subprocess.run(
                ["mdb-export", str(self.config.mdb_path), "HWSD2_LAYERS"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                self.logger.warning(f"mdb-export failed: {result.stderr}")
                return None

            # Parse CSV output
            from io import StringIO
            return pd.read_csv(StringIO(result.stdout))

        except FileNotFoundError:
            self.logger.warning("mdbtools not installed. Install with: brew install mdbtools")
            return None
        except Exception as e:
            self.logger.warning(f"MDB export failed: {e}")
            return None

    def _extract_from_netcdf(
        self,
        region: Region,
        cell_coords: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[int, SoilProfile]:
        """Extract soil data from NetCDF file."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray required for NetCDF extraction")

        profiles = {}

        ds = xr.open_dataset(self.config.nc_path)

        # Determine variable names (may vary by file)
        var_map = {}
        for internal, hwsd_name in self.VARIABLES.items():
            if hwsd_name in ds:
                var_map[internal] = hwsd_name
            elif hwsd_name.lower() in ds:
                var_map[internal] = hwsd_name.lower()

        if not var_map:
            self.logger.warning("No recognized variables in NetCDF")
            return profiles

        # Sample at coordinates
        if cell_coords:
            for i, (lat, lon) in enumerate(cell_coords):
                props = {}
                for internal, nc_var in var_map.items():
                    try:
                        val = float(ds[nc_var].sel(lat=lat, lon=lon, method="nearest"))
                        props[internal] = val if not np.isnan(val) else None
                    except Exception:
                        props[internal] = None

                if any(v is not None for v in props.values()):
                    profiles[i] = self._create_profile_from_dict(
                        cell_id=i, lat=lat, lon=lon, props=props, region=region
                    )
                elif self.config.use_defaults:
                    profiles[i] = self._create_default_profile(i, lat, lon, region)

        ds.close()
        return profiles

    def _create_profile_from_hwsd(
        self,
        cell_id: int,
        lat: float,
        lon: float,
        props: Dict[str, Any],
        region: Region,
    ) -> SoilProfile:
        """Create SoilProfile from HWSD database row."""
        # Extract values with None handling
        sand = props.get("T_SAND") or props.get("SAND")
        clay = props.get("T_CLAY") or props.get("CLAY")
        silt = props.get("T_SILT") or props.get("SILT")
        soc = props.get("T_OC") or props.get("OC")
        ph = props.get("T_PH_H2O") or props.get("PH")
        bd = props.get("T_BULK_DENSITY") or props.get("BULK_DENSITY")

        # Calculate silt if missing
        if silt is None and sand is not None and clay is not None:
            silt = max(0, 100 - sand - clay)

        layer = SoilLayer(
            depth_top=0.0,
            depth_bottom=0.2,  # D1 = 0-20cm
            sand=sand,
            clay=clay,
            silt=silt,
            organic_carbon=soc,
            bulk_density=bd,
            ph=ph,
        )

        return SoilProfile(
            profile_id=f"HWSD_{region.country_iso3}_{cell_id:06d}",
            lat=lat,
            lon=lon,
            source=self.NAME,
            layers=[layer],
            total_depth=0.2,
            metadata={"hwsd_smu_id": props.get("HWSD2_SMU_ID") or props.get("ID")},
        )

    def _create_profile_from_dict(
        self,
        cell_id: int,
        lat: float,
        lon: float,
        props: Dict[str, Any],
        region: Region,
    ) -> SoilProfile:
        """Create SoilProfile from dictionary of properties."""
        sand = props.get("sand")
        clay = props.get("clay")
        silt = props.get("silt")

        if silt is None and sand is not None and clay is not None:
            silt = max(0, 100 - sand - clay)

        layer = SoilLayer(
            depth_top=0.0,
            depth_bottom=0.2,
            sand=sand,
            clay=clay,
            silt=silt,
            organic_carbon=props.get("soc"),
            bulk_density=props.get("bd"),
            ph=props.get("ph"),
        )

        return SoilProfile(
            profile_id=f"HWSD_{region.country_iso3}_{cell_id:06d}",
            lat=lat,
            lon=lon,
            source=self.NAME,
            layers=[layer],
            total_depth=0.2,
            metadata={},
        )

    def _create_default_profile(
        self,
        cell_id: int,
        lat: float,
        lon: float,
        region: Region,
    ) -> SoilProfile:
        """Create profile with default values."""
        layer = SoilLayer(
            depth_top=0.0,
            depth_bottom=0.2,
            sand=DEFAULT_SOIL["sand"],
            clay=DEFAULT_SOIL["clay"],
            silt=DEFAULT_SOIL["silt"],
            organic_carbon=DEFAULT_SOIL["soc"],
            bulk_density=DEFAULT_SOIL["bd"],
            ph=DEFAULT_SOIL["ph"],
        )

        return SoilProfile(
            profile_id=f"HWSD_DEFAULT_{cell_id:06d}",
            lat=lat,
            lon=lon,
            source=f"{self.NAME}_default",
            layers=[layer],
            total_depth=0.2,
            metadata={"is_default": True},
        )

    def _calculate_statistics(
        self,
        profiles: Dict[int, SoilProfile],
    ) -> Dict[str, Any]:
        """Calculate summary statistics for profiles."""
        if not profiles:
            return {}

        sand_vals = []
        clay_vals = []

        for profile in profiles.values():
            if profile.layers:
                layer = profile.layers[0]
                if layer.sand is not None:
                    sand_vals.append(layer.sand)
                if layer.clay is not None:
                    clay_vals.append(layer.clay)

        return {
            "count": len(profiles),
            "sand_mean": np.mean(sand_vals) if sand_vals else None,
            "sand_std": np.std(sand_vals) if sand_vals else None,
            "clay_mean": np.mean(clay_vals) if clay_vals else None,
            "clay_std": np.std(clay_vals) if clay_vals else None,
        }
