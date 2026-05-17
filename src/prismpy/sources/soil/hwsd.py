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
from prismpy.warnings import WarningCategory


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
        # Sprint D.1 AC-4 — cells the loader could not extract a
        # real profile for. Each entry is a dict with at least
        # ``cell_id`` and ``cause`` (currently always
        # ``"soil_no_hwsd_coverage"``). The executor consumes
        # this list at HARMONIZE end + routes the cells to
        # ``data_availability='unavailable'`` with axis ``soil``
        # and the recorded cause. Replaces the prior
        # DEFAULT_SOIL silent-fill that used to inject a Sahel-
        # typical synthetic profile and pretend the cell was
        # covered.
        self.unavailable_cells: List[Dict[str, Any]] = []

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

        # Sprint D.1 AC-4 — when HWSD returns nothing, the cells are
        # genuinely without coverage; flag them as unavailable
        # rather than silently inject a Sahel-typical synthetic
        # profile. The executor reads ``self.unavailable_cells`` at
        # HARMONIZE end and routes each cell accordingly.
        if self.config.use_defaults and not profiles:
            self.logger.info(
                "HWSD retrieval returned no profiles; flagging cells unavailable"
            )
            warnings.append(
                "No HWSD data found; cells flagged data_availability='unavailable'"
            )
            if cell_coords:
                for i, _ in enumerate(cell_coords):
                    self._record_unavailable(
                        i,
                        cause=WarningCategory.SOIL_NO_HWSD_COVERAGE.value,
                    )
            source_type = "no_coverage"

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
        # Sprint D.1 AC-4 — surface the unavailable-cells list to
        # the executor so HARMONIZE can route cells to the
        # ``data_availability='unavailable'`` path with the
        # recorded cause.
        if self.unavailable_cells:
            metadata["unavailable_cells"] = list(self.unavailable_cells)

        return self.create_result(
            success=True,
            data=hwsd_data,
            warnings=warnings,
            metadata=metadata,
        )

    def validate(self, data: Any) -> List[str]:
        """Validate HWSD data.

        Sprint D.1 AC-1.1: the source-level texture-sum warning
        previously emitted from this method is retired.
        Texture-fraction validation now happens once at the
        harmonize stage via
        :func:`prismpy.harmonize.texture_renormalize.texture_action_for`,
        which routes layers to renormalize / warn-retain / exclude
        with provenance per layer. Keeping the duplicate
        source-level emit would surface the same warning twice
        (once here, once at harmonize) and let the two layers
        drift if the thresholds ever changed independently.
        """
        warnings = []

        if not isinstance(data, HWSDData):
            return [f"Expected HWSDData, got {type(data)}"]

        if not data.profiles:
            warnings.append("No soil profiles extracted")

        # Sprint D.1 AC-4: ``data.source_type == "defaults"`` no
        # longer occurs (DEFAULT_SOIL injection is retired); the
        # equivalent signal is ``data.source_type == "no_coverage"``
        # plus a non-empty ``unavailable_cells`` list propagated
        # through the retrieval-result metadata.
        if data.source_type == "no_coverage":
            warnings.append(
                "HWSD coverage missing; cells routed to "
                "data_availability='unavailable'"
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
                logger.warning("MDB columns missing; flagging cells unavailable.")
                if self.config.use_defaults:
                    for i, _ in enumerate(cell_coords or []):
                        self._record_unavailable(
                        i,
                        cause=WarningCategory.SOIL_NO_HWSD_COVERAGE.value,
                    )
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
                    self._record_unavailable(
                        i,
                        cause=WarningCategory.SOIL_NO_HWSD_COVERAGE.value,
                    )

        # Sprint D.1 AC-4 \u2014 record the SMU-lookup-miss outcome
        # honestly. The previous FALLBACK_SUBSTITUTION decision
        # used to claim cells received a Sahel-typical synthetic
        # profile (DEFAULT_SOIL injection); that fallback is
        # retired. Cells that miss the SMU lookup are now flagged
        # for the executor to route to data_availability=
        # 'unavailable' with cause soil_no_hwsd_coverage.
        n_unavailable = len(self.unavailable_cells)
        if n_unavailable > 0 and self.provenance:
            self.provenance.record_decision(
                decision_type=DecisionType.FALLBACK_SUBSTITUTION,
                description=(
                    f"HWSD SMU-lookup miss: {n_unavailable} cells flagged "
                    f"data_availability='unavailable' (cause: "
                    f"soil_no_hwsd_coverage)"
                ),
                rationale=(
                    "The HWSD BIL raster returned Soil Mapping Unit IDs for "
                    "all queried coordinates, but some SMU IDs were absent "
                    "from the MDB soil-properties table (HWSD2_LAYERS). "
                    "This occurs for water bodies, bare rock, urban areas, "
                    "or HWSD mapping gaps. The cells are routed to "
                    "data_availability='unavailable' so downstream "
                    "consumers see an honest missing-soil signal rather "
                    "than a synthetic placeholder profile."
                ),
                alternatives=[
                    "iSDA 30m (higher resolution, fewer gaps in Africa)",
                    "Spatial interpolation from neighbouring HWSD cells",
                    "Synthetic DEFAULT_SOIL profile (retired in Sprint D.1 "
                    "as data cooking)",
                ],
                reference="prismpy.sources.soil.hwsd.HWSDSource._extract_from_bil_mdb",
                artifact_id="soil",
            )

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

        # F-DP: wrap the dataset in a `with` block so the HDF5 file
        # handle is released even when the function returns early
        # (the `if not var_map` branch below) OR when an exception
        # escapes the inner sel/property loop. The pre-F-DP shape
        # had an explicit `ds.close()` at the bottom + a bare early
        # `return profiles` that bypassed it, plus no exception
        # handling around the close at all — exception-unsafe per
        # the §27 producer-consumer pattern (the producer here is
        # the resource-acquisition; consumers are every subsequent
        # statement). `with` makes the close happen on every exit
        # path.
        with xr.open_dataset(self.config.nc_path) as ds:
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
                        self._record_unavailable(
                            i,
                            cause=WarningCategory.SOIL_NO_HWSD_COVERAGE.value,
                        )

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
            metadata={
                "hwsd_smu_id": props.get("HWSD2_SMU_ID") or props.get("ID"),
                # V2-22c-PRE.1.10 (D37) — cascade-provenance defaults.
                # Loader-side cascade_rank=1 means "this loader
                # produced the profile"; the cascade orchestrator at
                # executor.py overrides to rank=2 + fallback_attempts
                # when HWSD ran as the iSDA fallback path.
                "source": "HWSD",
                "version": "v2.0",
                "cascade_rank": 1,
                "fallback_attempts": [],
            },
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
            metadata={
                # V2-22c-PRE.1.10 (D37) — cascade defaults.
                "source": "HWSD",
                "version": "v2.0",
                "cascade_rank": 1,
                "fallback_attempts": [],
            },
        )

    def _create_default_profile(
        self,
        cell_id: int,
        lat: float,
        lon: float,
        region: Region,
    ) -> None:
        """Sprint D.1 AC-4: returns None.

        The synthetic Sahel-typical DEFAULT_SOIL profile is retired.
        Callers must consult the loader's ``unavailable_cells``
        list (or ``RetrievalResult.metadata['unavailable_cells']``)
        to discover which cells lack HWSD coverage and route them
        to ``data_availability='unavailable'`` with axis ``soil``
        and cause ``soil_no_hwsd_coverage``.

        The DEFAULT_SOIL dict at :data:`DEFAULT_SOIL` (module top
        of file) is preserved as historical documentation of the
        retired synthetic values; no caller in active code paths
        constructs a profile from it.
        """
        return None

    def _record_unavailable(
        self,
        cell_id: int,
        cause: str = WarningCategory.SOIL_NO_HWSD_COVERAGE.value,
    ) -> None:
        """Record that the loader could not produce a real profile
        for this cell. The executor reads
        ``self.unavailable_cells`` at HARMONIZE stage end and
        routes each cell to ``data_availability='unavailable'``
        with axis ``soil`` and the recorded cause.
        """
        self.unavailable_cells.append(
            {"cell_id": cell_id, "cause": cause}
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
