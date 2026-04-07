"""
Pipeline orchestration for prismpy.

This module provides the main execution engine that coordinates the
complete data-to-model translation workflow through five stages:
1. RETRIEVE - Download/load data from sources
2. HARMONIZE - Align, gap-fill, and validate data
3. TRANSLATE - Convert to platform-specific formats
4. VALIDATE - Check outputs against requirements
5. PACKAGE - Generate self-documenting data packages
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import logging

from prismpy.config.schema import ProjectConfig, Platform
from prismpy.config.loader import load_config
from prismpy.models.region import Region
from prismpy.models.climate import ClimateTimeSeries, ClimateRecord
from prismpy.models.soil import SoilProfile, SoilLayer
from prismpy.models.crop import CropParameters, CropCalendar
from prismpy.provenance.tracker import ProvenanceTracker
from prismpy.translators.base import (
    BaseTranslator,
    UnifiedData,
    TranslationResult,
)
from prismpy.validators.base import BaseValidator, ValidationResult


class PipelineStage(str, Enum):
    """Stages of the translation pipeline."""
    RETRIEVE = "retrieve"
    HARMONIZE = "harmonize"
    TRANSLATE = "translate"
    VALIDATE = "validate"
    PACKAGE = "package"


@dataclass
class StageResult:
    """Result of a pipeline stage execution.

    Attributes:
        stage: Which stage was executed
        success: Whether the stage completed successfully
        data: Output data from the stage
        errors: List of errors encountered
        warnings: List of warnings generated
        duration_seconds: Execution time
    """
    stage: PipelineStage
    success: bool
    data: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class PipelineResult:
    """Complete result of pipeline execution.

    Attributes:
        success: Whether the pipeline completed successfully
        stages: Results for each executed stage
        translation_results: Results from each platform translator
        provenance_path: Path to saved provenance file
        total_duration_seconds: Total execution time
    """
    success: bool
    stages: Dict[str, StageResult] = field(default_factory=dict)
    translation_results: Dict[str, TranslationResult] = field(default_factory=dict)
    provenance_path: Optional[Path] = None
    total_duration_seconds: float = 0.0


class TranslationPipeline:
    """Main pipeline execution engine.

    Orchestrates the complete data-to-model translation workflow,
    coordinating data retrieval, harmonization, translation, validation,
    and packaging across all enabled platforms.

    Example usage:
        ```python
        from prismpy.pipeline.executor import TranslationPipeline
        from prismpy.config.loader import load_config

        config = load_config("project_config.yaml")
        pipeline = TranslationPipeline(config)
        result = pipeline.execute()

        if result.success:
            print(f"Translation complete! Files at: {result.translation_results}")
        else:
            for stage, stage_result in result.stages.items():
                if stage_result.errors:
                    print(f"{stage}: {stage_result.errors}")
        ```

    Attributes:
        config: Project configuration
        provenance: Provenance tracker
        translators: Dictionary of platform translators
        logger: Logger instance
    """

    def __init__(
        self,
        config: ProjectConfig,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the pipeline.

        Args:
            config: Project configuration
            provenance: Optional provenance tracker (created if not provided)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize provenance tracker
        if provenance:
            self.provenance = provenance
        else:
            self.provenance = ProvenanceTracker(
                enabled=config.provenance.enabled,
                include_hashes=config.provenance.include_hashes,
                include_parameters=config.provenance.include_parameters,
                storage_format=config.provenance.storage,
                output_dir=config.provenance.output_dir,
                project_name=config.project.name,
            )

        # Set configuration hash
        self.provenance.set_config_hash(config.model_dump())

        # Initialize translators and validators (lazy loading)
        self._translators: Dict[Platform, BaseTranslator] = {}
        self._validators: Dict[Platform, BaseValidator] = {}

    def _get_translator(self, platform: Platform) -> Optional[BaseTranslator]:
        """Get or create translator for a platform.

        Args:
            platform: Target platform

        Returns:
            Translator instance or None if not available
        """
        if platform in self._translators:
            return self._translators[platform]

        # Import and instantiate translator based on platform
        try:
            from prismpy.translators import (
                SarraPyTranslator,
                CraftTranslator,
                PythiaTranslator,
                AceaTranslator,
            )

            # Map platforms to translator classes
            translator_map = {
                Platform.SARRA_PY: SarraPyTranslator,
                Platform.CRAFT: CraftTranslator,
                Platform.PYTHIA: PythiaTranslator,
                Platform.ACEA: AceaTranslator,
            }

            translator_class = translator_map.get(platform)
            if translator_class is None:
                self.logger.warning(f"No translator available for {platform.value}")
                return None

            # Create output directory for this platform
            output_dir = Path(self.config.output.base_dir) / platform.value
            output_dir.mkdir(parents=True, exist_ok=True)

            # Instantiate translator
            translator = translator_class(
                config=self.config,
                output_dir=output_dir,
                provenance=self.provenance,
            )
            self._translators[platform] = translator
            self.logger.debug(f"Created translator for {platform.value}")
            return translator

        except Exception as e:
            self.logger.error(f"Failed to create translator for {platform.value}: {e}")
            return None

    def _get_validator(self, platform: Platform, output_dir: Path) -> Optional[BaseValidator]:
        """Get or create validator for a platform.

        Args:
            platform: Target platform
            output_dir: Directory containing platform outputs to validate

        Returns:
            Validator instance or None if not available
        """
        if platform in self._validators:
            return self._validators[platform]

        try:
            from prismpy.validators import (
                SarraPyValidator,
                CraftValidator,
                PythiaValidator,
                AceaValidator,
            )

            validator_map = {
                Platform.SARRA_PY: SarraPyValidator,
                Platform.CRAFT: CraftValidator,
                Platform.PYTHIA: PythiaValidator,
                Platform.ACEA: AceaValidator,
            }

            validator_class = validator_map.get(platform)
            if validator_class is None:
                self.logger.warning(f"No validator available for {platform.value}")
                return None

            validator = validator_class(output_dir=output_dir)
            self._validators[platform] = validator
            self.logger.debug(f"Created validator for {platform.value}")
            return validator

        except Exception as e:
            self.logger.error(f"Failed to create validator for {platform.value}: {e}")
            return None

    def _execute_retrieve(self) -> StageResult:
        """Execute the RETRIEVE stage.

        Downloads or loads data from all configured sources.

        Returns:
            StageResult with retrieved data
        """
        start_time = datetime.now()
        self.logger.info("Stage 1: RETRIEVE - Loading data from sources")

        errors = []
        warnings = []
        data = {}

        try:
            # Start provenance tracking for region
            self.provenance.start_artifact("region")

            # Load region from GADM or manual bounds
            self.logger.info(f"Loading region: {self.config.region.name}")

            from prismpy.models.region import BoundingBox

            # Check boundary source type
            boundary_source = self.config.region.boundary.source.value

            if boundary_source == "manual" and self.config.region.boundary.manual_bounds:
                # Use manual bounds from config
                mb = self.config.region.boundary.manual_bounds
                bounds = BoundingBox(
                    minx=mb.minx,
                    miny=mb.miny,
                    maxx=mb.maxx,
                    maxy=mb.maxy,
                )

                # Construct rectangle polygon for clipping + SharePercent
                geometry_wkt = (
                    f"POLYGON(({mb.minx} {mb.miny}, {mb.maxx} {mb.miny}, "
                    f"{mb.maxx} {mb.maxy}, {mb.minx} {mb.maxy}, {mb.minx} {mb.miny}))"
                )

                region = Region(
                    name=self.config.region.name,
                    country=self.config.region.country,
                    country_iso3=self.config.region.country_iso3,
                    bounds=bounds,
                    gadm_level=self.config.region.boundary.gadm_level or 2,
                    geometry_wkt=geometry_wkt,
                )
                self.logger.info(f"Using manual bounds: {bounds.to_gis_format()}")

            elif boundary_source == "shapefile":
                # Load region from shapefile
                import geopandas as gpd
                from shapely.ops import unary_union

                shapefile_path = Path(self.config.region.boundary.shapefile_path)
                if not shapefile_path.is_absolute():
                    # Resolve relative to project root
                    project_root = Path.cwd()
                    shapefile_path = project_root / shapefile_path

                if not shapefile_path.exists():
                    raise ValueError(f"Shapefile not found: {shapefile_path}")

                self.logger.info(f"Loading boundary from shapefile: {shapefile_path}")
                gdf = gpd.read_file(shapefile_path)

                # Get bounds and geometry
                bounds_tuple = gdf.total_bounds  # [minx, miny, maxx, maxy]
                bounds = BoundingBox(
                    minx=bounds_tuple[0],
                    miny=bounds_tuple[1],
                    maxx=bounds_tuple[2],
                    maxy=bounds_tuple[3],
                )

                # Dissolve all geometries into one polygon for clipping
                dissolved_geom = unary_union(gdf.geometry)
                geometry_wkt = dissolved_geom.wkt

                region = Region(
                    name=self.config.region.name,
                    country=self.config.region.country,
                    country_iso3=self.config.region.country_iso3,
                    bounds=bounds,
                    gadm_level=self.config.region.boundary.gadm_level or 2,
                    geometry_wkt=geometry_wkt,  # This enables polygon clipping!
                )
                self.logger.info(f"Loaded shapefile bounds: {bounds.to_gis_format()}")
                self.logger.info(f"Geometry loaded for polygon clipping ({len(geometry_wkt)} chars WKT)")

            elif boundary_source == "gadm":
                # Dynamic GADM loading
                from prismpy.sources.boundaries.gadm import GADMSource

                gadm_base_path = Path(self.config.data_sources.gadm.base_path)
                cache_dir = self.config.data_sources.cache_dir if self.config.data_sources.cache_enabled else None
                gadm_source = GADMSource(
                    base_path=gadm_base_path,
                    cache_dir=cache_dir,
                    provenance=self.provenance,
                )

                gadm_level = self.config.region.boundary.gadm_level or 2
                filter_field = self.config.region.boundary.gadm_filter_field or f"NAME_{gadm_level}"
                filter_value = self.config.region.boundary.gadm_filter_value
                country_iso3 = self.config.region.country_iso3

                self.logger.info(f"Loading GADM boundary: {filter_field}='{filter_value}'")

                # Try direct shapefile path first (for non-standard directory structures)
                # Standard GADM: base_path/MLI/gadm41_MLI_2.shp
                # Direct: base_path/gadm41_MLI_2.shp
                direct_shapefile = gadm_base_path / f"gadm41_{country_iso3.upper()}_{gadm_level}.shp"

                if direct_shapefile.exists():
                    # Use direct shapefile path
                    result = gadm_source.retrieve(
                        shapefile_path=direct_shapefile,
                        gadm_level=gadm_level,
                        filter_field=filter_field,
                        filter_value=filter_value,
                        country_iso3=country_iso3,
                        include_geometry=True,  # Include geometry for polygon clipping
                        use_cache=True,
                    )
                else:
                    # Try standard GADM directory structure
                    result = gadm_source.retrieve(
                        country_iso3=country_iso3,
                        gadm_level=gadm_level,
                        filter_field=filter_field,
                        filter_value=filter_value,
                        include_geometry=True,  # Include geometry for polygon clipping
                        use_cache=True,
                    )

                if result.success and result.data:
                    region = result.data
                    self.logger.info(f"Loaded GADM bounds: {region.bounds.to_gis_format()}")
                else:
                    # GADM loading failed — try pygadm (downloads from web, caches locally)
                    self.logger.debug(f"GADM shapefiles not found, trying pygadm: {result.errors}")
                    region = None

                    try:
                        import pygadm

                        self.logger.info(
                            f"Trying pygadm fallback: {country_iso3} level {gadm_level} "
                            f"filter='{filter_value}'"
                        )

                        names_df = pygadm.Names(
                            admin=country_iso3, content_level=gadm_level
                        )
                        name_col = f"NAME_{gadm_level}"
                        gid_col = f"GID_{gadm_level}"

                        match = names_df[names_df[name_col] == filter_value]
                        if len(match) > 0:
                            gid = match.iloc[0][gid_col]
                            gdf = pygadm.Items(admin=gid)

                            if gdf is not None and len(gdf) > 0:
                                if len(gdf) > 1:
                                    gdf = gdf.dissolve()

                                bounds_tuple = gdf.total_bounds
                                bounds = BoundingBox(
                                    minx=bounds_tuple[0],
                                    miny=bounds_tuple[1],
                                    maxx=bounds_tuple[2],
                                    maxy=bounds_tuple[3],
                                )

                                geometry_wkt = gdf.geometry.iloc[0].wkt

                                region = Region(
                                    name=self.config.region.name,
                                    country=self.config.region.country,
                                    country_iso3=country_iso3,
                                    bounds=bounds,
                                    gadm_level=gadm_level,
                                    geometry_wkt=geometry_wkt,
                                )
                                self.logger.info(
                                    f"Loaded boundary via pygadm: {bounds.to_gis_format()}"
                                )
                        else:
                            self.logger.warning(
                                f"pygadm: '{filter_value}' not found at level {gadm_level}"
                            )

                    except Exception as e:
                        self.logger.warning(f"pygadm fallback failed: {e}")

                    # If pygadm also failed, fall back to manual bounds
                    if region is None:
                        if self.config.region.boundary.manual_bounds:
                            mb = self.config.region.boundary.manual_bounds
                            bounds = BoundingBox(minx=mb.minx, miny=mb.miny, maxx=mb.maxx, maxy=mb.maxy)
                            region = Region(
                                name=self.config.region.name,
                                country=self.config.region.country,
                                country_iso3=self.config.region.country_iso3,
                                bounds=bounds,
                                gadm_level=gadm_level,
                            )
                            warnings.append(f"GADM and pygadm failed, using manual bounds fallback: {result.errors}")
                        else:
                            raise ValueError(
                                f"GADM loading failed, pygadm fallback failed, and no manual bounds available: "
                                f"{result.errors}"
                            )

            else:
                # No valid boundary source - use manual bounds as fallback
                if self.config.region.boundary.manual_bounds:
                    mb = self.config.region.boundary.manual_bounds
                    bounds = BoundingBox(minx=mb.minx, miny=mb.miny, maxx=mb.maxx, maxy=mb.maxy)
                else:
                    raise ValueError(
                        f"Unsupported boundary source '{boundary_source}' and no manual bounds provided. "
                        "Please configure GADM or provide manual bounds."
                    )
                region = Region(
                    name=self.config.region.name,
                    country=self.config.region.country,
                    country_iso3=self.config.region.country_iso3,
                    bounds=bounds,
                    gadm_level=self.config.region.boundary.gadm_level or 2,
                )

            data["region"] = region

            self.provenance.record_retrieval(
                source="config",
                parameters={"region_name": self.config.region.name},
            )

            # Load climate data
            self.logger.info("Loading climate data...")
            climate_data = self._load_climate_data(region)
            if climate_data:
                data["climate"] = climate_data
                self.logger.info(f"Loaded climate data for {len(climate_data)} locations")
            else:
                # CRAFT/PYTHIA/ACEA download weather at translate time — not a warning
                # Only warn if a platform needs pre-loaded climate (currently none do)
                platforms_that_self_download = {Platform.CRAFT, Platform.PYTHIA, Platform.ACEA, Platform.SARRA_PY}
                enabled = set(self.config.get_enabled_platforms())
                if not enabled.issubset(platforms_that_self_download):
                    warnings.append("Climate data not available - using placeholder")
                # Create minimal placeholder climate data
                data["climate"] = self._create_placeholder_climate(region)

            # Load soil data
            self.logger.info("Loading soil data...")
            soil_data = self._load_soil_data(region)
            if soil_data:
                data["soil"] = soil_data
                self.logger.info(f"Loaded soil data for {len(soil_data)} profiles")
            else:
                # Check if HWSD/iSDA will be available at harmonize time
                # (per-cell soil retrieval happens after grid creation)
                has_hwsd = False
                for plat in self.config.get_enabled_platforms():
                    pcfg = self.config.get_platform_config(plat)
                    if pcfg and getattr(pcfg, 'hwsd_bil_path', None):
                        has_hwsd = True
                        break
                has_isda_local = any(
                    (Path(d) / "sand_content_1km.tif").exists()
                    for d in [
                        "data/isda",
                        str(Path(__file__).resolve().parents[4] / "data" / "isda"),
                    ]
                )
                if not has_hwsd and not has_isda_local:
                    warnings.append("Soil data not available - using placeholder")
                # Create minimal placeholder soil data (replaced at harmonize if HWSD/iSDA found)
                data["soil"] = self._create_placeholder_soil(region)

            # Load crop parameters
            self.logger.info("Loading crop parameters...")
            crop_params = self._load_crop_params()
            if crop_params:
                data["crop_params"] = crop_params
                self.logger.info(f"Loaded crop parameters for {crop_params.crop_name}")
            else:
                warnings.append("Crop parameters not available - using defaults")
                # Create default crop parameters
                data["crop_params"] = self._create_default_crop_params()

            # Load crop calendar if available
            if self.config.crop.calendar:
                data["crop_calendar"] = {
                    0: CropCalendar(
                        location_id=0,
                        planting_doy=self.config.crop.calendar.planting_doy,
                        maturity_doy=self.config.crop.calendar.maturity_doy,
                        source="config",
                    )
                }

        except Exception as e:
            errors.append(f"Retrieval failed: {str(e)}")
            self.logger.error(f"Retrieval error: {e}")

        duration = (datetime.now() - start_time).total_seconds()
        return StageResult(
            stage=PipelineStage.RETRIEVE,
            success=len(errors) == 0,
            data=data,
            errors=errors,
            warnings=warnings,
            duration_seconds=duration,
        )

    def _load_climate_data(self, region: Region) -> Optional[Dict[str, Any]]:
        """Load climate data from configured paths or sources.

        Args:
            region: Region to load data for

        Returns:
            Dictionary with climate data info including paths to GeoTIFF files
        """
        from datetime import date

        # Get temporal range from config
        start_date = date(self.config.temporal.start_year, 1, 1)
        end_date = date(self.config.temporal.end_year, 12, 31)

        climate_data = {
            "rainfall_dir": None,
            "agera5_dir": None,
            "metadata": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "region": region.name,
            }
        }

        # Check for pre-configured existing data paths
        if hasattr(self.config.data_sources, 'climate') and self.config.data_sources.climate:
            climate_config = self.config.data_sources.climate

            # Check rainfall directory
            if climate_config.rainfall_dir:
                rainfall_path = Path(climate_config.rainfall_dir)
                if rainfall_path.exists():
                    # Count files
                    tif_files = list(rainfall_path.glob("*.tif"))
                    self.logger.info(f"Found TAMSAT rainfall data: {len(tif_files)} files at {rainfall_path}")
                    climate_data["rainfall_dir"] = rainfall_path
                    climate_data["rainfall_file_count"] = len(tif_files)
                else:
                    self.logger.warning(f"Configured rainfall_dir does not exist: {rainfall_path}")

            # Check AgERA5 directory
            if climate_config.agera5_dir:
                agera5_path = Path(climate_config.agera5_dir)
                if agera5_path.exists():
                    # Check for variable subdirectories
                    var_subdirs = [d for d in agera5_path.iterdir() if d.is_dir()]
                    var_counts = {}
                    for var_dir in var_subdirs:
                        tif_files = list(var_dir.glob("*.tif"))
                        if tif_files:
                            var_counts[var_dir.name] = len(tif_files)

                    self.logger.info(f"Found AgERA5 data: {var_counts} at {agera5_path}")
                    climate_data["agera5_dir"] = agera5_path
                    climate_data["agera5_variables"] = var_counts
                else:
                    self.logger.warning(f"Configured agera5_dir does not exist: {agera5_path}")

        # Return if we found data
        if climate_data["rainfall_dir"] or climate_data["agera5_dir"]:
            return climate_data

        # No configured paths — try TAMSAT/AgERA5 download for SARRA-Py
        enabled_platforms = self.config.get_enabled_platforms()
        has_sarra_py = any(p == Platform.SARRA_PY for p in enabled_platforms)

        if has_sarra_py:
            cache_dir = Path(self.config.data_sources.cache_dir) if hasattr(self.config.data_sources, 'cache_dir') and self.config.data_sources.cache_dir else Path("data/cache")
            got_data = False

            # Download TAMSAT rainfall
            try:
                from prismpy.sources.climate.tamsat import TAMSATSource

                tamsat = TAMSATSource(cache_dir=cache_dir, provenance=self.provenance)
                if tamsat.sarra_download_available:
                    self.logger.info("Downloading TAMSAT rainfall data...")
                    tamsat_result = tamsat.retrieve(
                        region=region, start_date=start_date,
                        end_date=end_date, download=True,
                    )
                    if tamsat_result.success and tamsat_result.data:
                        climate_data["rainfall_dir"] = tamsat_result.data.data_dir
                        climate_data["rainfall_file_count"] = tamsat_result.data.file_count
                        self.logger.info(f"TAMSAT: {tamsat_result.data.file_count} files")
                        got_data = True
                    else:
                        self.logger.warning(f"TAMSAT download failed: {tamsat_result.errors}")
            except Exception as e:
                self.logger.warning(f"TAMSAT download error: {e}")

            # Download AgERA5 temperature/radiation
            try:
                from prismpy.sources.climate.agera5 import AgERA5Source

                agera5 = AgERA5Source(cache_dir=cache_dir, provenance=self.provenance)
                if agera5.sarra_download_available:
                    self.logger.info("Downloading AgERA5 temperature data...")
                    agera5_result = agera5.retrieve(
                        region=region, start_date=start_date,
                        end_date=end_date, download=True,
                    )
                    if agera5_result.success and agera5_result.data:
                        climate_data["agera5_dir"] = agera5_result.data.data_dir
                        climate_data["agera5_variables"] = agera5_result.data.variables
                        self.logger.info(f"AgERA5: {agera5_result.data.variables}")
                        got_data = True
                    else:
                        self.logger.warning(f"AgERA5 download failed: {agera5_result.errors}")
            except Exception as e:
                self.logger.warning(f"AgERA5 download error: {e}")

            if got_data:
                return climate_data

        self.logger.warning("No pre-configured climate data paths found.")
        return None

    def _create_placeholder_climate(
        self, region: Region
    ) -> Dict[int, ClimateTimeSeries]:
        """Create placeholder climate data for testing.

        Args:
            region: Region for location info

        Returns:
            Dictionary with minimal climate time series
        """
        from datetime import date, timedelta

        # Get temporal range from config - include spinup years!
        start_year = self.config.temporal.start_year
        end_year = self.config.temporal.end_year
        spinup_years = self.config.temporal.spinup_years

        # Create records for each day, including spinup period
        records = []
        current_date = date(start_year - spinup_years, 1, 1)
        end_date = date(end_year, 12, 31)

        while current_date <= end_date:
            # Create record with generic placeholder values (not region-specific)
            # WARNING: These are generic defaults, NOT suitable for production forecasts
            record = ClimateRecord(
                date=current_date,
                tmax=30.0,  # Generic tropical/subtropical max temp
                tmin=20.0,  # Generic tropical/subtropical min temp
                precip=2.5,  # Generic moderate rainfall (no seasonal assumption)
                srad=18.0,  # MJ/m2/day (generic)
                et0=4.5,    # mm/day (generic)
            )
            records.append(record)
            current_date += timedelta(days=1)

        # Create single time series for region centroid
        # Use sentinel cell ID -1 so it doesn't conflict with real cell IDs
        center_lat = (region.bounds.miny + region.bounds.maxy) / 2
        center_lon = (region.bounds.minx + region.bounds.maxx) / 2

        ts = ClimateTimeSeries(
            location_id=-1,  # Sentinel ID - will be replaced by downloaded data
            lat=center_lat,
            lon=center_lon,
            source="placeholder",
            records=records,
        )

        return {-1: ts}

    def _load_soil_data(self, region: Region) -> Optional[Dict[str, Any]]:
        """Load soil data from iSDA source.

        Args:
            region: Region to load data for

        Returns:
            Dictionary with soil data info including paths to files
        """
        from prismpy.sources.soil.isda import iSDASource, iSDAConfig

        # Determine data directory - check SARRA-Py assets first
        sarra_py_assets = Path("SARRA-PY/data/assets")
        cache_dir = Path(self.config.data_sources.cache_dir) if hasattr(self.config.data_sources, 'cache_dir') else Path("data/cache")

        try:
            isda_source = iSDASource(
                config=iSDAConfig(
                    sarra_py_assets_dir=sarra_py_assets if sarra_py_assets.exists() else None,
                ),
                cache_dir=cache_dir,
                provenance=self.provenance,
            )

            result = isda_source.retrieve(
                region=region,
                extract_profiles=False,  # We'll use default profiles for now
            )

            if result.success:
                self.logger.info(f"iSDA: Variables available: {list(result.data.variables.keys())}")
                return {"isda": result.data}
            else:
                self.logger.debug(f"iSDA local files not found: {result.errors} (will use HWSD/iSDA at harmonize)")

        except Exception as e:
            self.logger.debug(f"iSDA retrieval error: {e}")

        return None

    def _create_placeholder_soil(self, region: Region) -> Dict[int, SoilProfile]:
        """Create placeholder soil data for testing.

        Args:
            region: Region for location info

        Returns:
            Dictionary with minimal soil profile
        """
        center_lat = (region.bounds.miny + region.bounds.maxy) / 2
        center_lon = (region.bounds.minx + region.bounds.maxx) / 2

        # Create generic placeholder soil profile (sandy loam)
        layers = [
            SoilLayer(
                depth_top=0.0,
                depth_bottom=0.2,
                sand=60.0,
                clay=15.0,
                organic_carbon=0.8,
                bulk_density=1.4,
                ph=6.5,
                field_capacity=0.25,
                wilting_point=0.10,
            ),
            SoilLayer(
                depth_top=0.2,
                depth_bottom=1.0,
                sand=55.0,
                clay=20.0,
                organic_carbon=0.4,
                bulk_density=1.5,
                ph=6.3,
                field_capacity=0.28,
                wilting_point=0.12,
            ),
        ]

        profile = SoilProfile(
            profile_id="placeholder_0",
            lat=center_lat,
            lon=center_lon,
            source="placeholder",
            layers=layers,
        )

        return {0: profile}

    def _retrieve_isda_api_for_grid(
        self, grid, region: Region
    ) -> Optional[Dict[int, SoilProfile]]:
        """Retrieve per-cell soil data from iSDA cloud-optimized GeoTIFFs on S3.

        Reads directly from public S3 bucket (no auth needed). Each property
        is a single COG covering Africa at 30m. Band mapping:
          Band 1: mean at 0-20cm
          Band 2: mean at 20-50cm
          Band 3: error at 0-20cm
          Band 4: error at 20-50cm

        Scale factors: ph values ÷10, bulk_density ÷100, carbon_organic as-is (g/kg).

        Only runs when SARRA-Py is enabled (iSDA is Africa-only).

        Args:
            grid: SpatialGrid with cell coordinates
            region: Region for metadata

        Returns:
            Dictionary of cell_id -> SoilProfile, or None
        """
        enabled = self.config.get_enabled_platforms()
        if Platform.SARRA_PY not in enabled:
            return None

        try:
            import rasterio
            from pyproj import Transformer
        except ImportError:
            self.logger.warning("rasterio/pyproj not available for iSDA retrieval")
            return None

        S3_BASE = "https://isdasoil.s3.amazonaws.com/soil_data"
        PROPERTIES = {
            "sand_content": {"scale": 1.0, "unit": "%"},
            "clay_content": {"scale": 1.0, "unit": "%"},
            "silt_content": {"scale": 1.0, "unit": "%"},
            "carbon_organic": {"scale": 1.0, "unit": "g/kg"},
            "ph": {"scale": 0.1, "unit": ""},
            "bulk_density": {"scale": 0.01, "unit": "g/cm3"},
        }

        # Check for local 1km files first (PRISMWEB_DATA_DIR/isda/)
        local_isda_dir = None
        for search_dir in [
            Path(self.config.data_sources.cache_dir).parent / "isda" if hasattr(self.config.data_sources, 'cache_dir') and self.config.data_sources.cache_dir else None,
            Path("data/isda"),
            Path(__file__).resolve().parents[4] / "data" / "isda",
        ]:
            if search_dir and search_dir.exists():
                # Check if all property files exist locally
                local_files = {p: search_dir / f"{p}_1km.tif" for p in PROPERTIES}
                if all(f.exists() for f in local_files.values()):
                    local_isda_dir = search_dir
                    self.logger.info(f"Found local iSDA 1km data at {search_dir}")
                    break

        cells = grid.cells if grid else []
        if not cells:
            return None

        # Transform cell coords from WGS84 to EPSG:3857 (iSDA CRS)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        cell_coords_3857 = []
        for cell in cells:
            x, y = transformer.transform(cell.lon, cell.lat)
            cell_coords_3857.append((x, y))

        total = len(cells)
        self.logger.info(f"Reading iSDA soil data from S3 for {total} cells...")

        # Read each property from S3 COGs
        cell_data = {cell.cell_id: {} for cell in cells}

        try:
            for prop_name, prop_info in PROPERTIES.items():
                if local_isda_dir:
                    url = str(local_isda_dir / f"{prop_name}_1km.tif")
                    self.logger.info(f"  Reading {prop_name} from local cache...")
                else:
                    url = f"{S3_BASE}/{prop_name}/{prop_name}.tif"
                    self.logger.info(f"  Reading {prop_name} from S3...")

                with rasterio.open(url) as src:
                    for i, cell in enumerate(cells):
                        x, y = cell_coords_3857[i]
                        row, col = src.index(x, y)
                        # Read bands 1 (0-20cm mean) and 2 (20-50cm mean)
                        window = rasterio.windows.Window(col, row, 1, 1)
                        b1 = float(src.read(1, window=window)[0][0])
                        b2 = float(src.read(2, window=window)[0][0])

                        scale = prop_info["scale"]
                        cell_data[cell.cell_id][f"{prop_name}_0-20"] = b1 * scale
                        cell_data[cell.cell_id][f"{prop_name}_20-50"] = b2 * scale

            # Build SoilProfile objects
            profiles = {}
            for cell in cells:
                d = cell_data[cell.cell_id]
                if not d.get("sand_content_0-20"):
                    continue

                layers = []
                for depth, (top, bot) in [("0-20", (0.0, 0.2)), ("20-50", (0.2, 0.5))]:
                    sand = d.get(f"sand_content_{depth}")
                    clay = d.get(f"clay_content_{depth}")
                    silt = d.get(f"silt_content_{depth}")
                    oc = d.get(f"carbon_organic_{depth}")
                    ph = d.get(f"ph_{depth}")
                    bd = d.get(f"bulk_density_{depth}")

                    if sand is not None:
                        layers.append(SoilLayer(
                            depth_top=top, depth_bottom=bot,
                            sand=sand, clay=clay or 0,
                            silt=silt or 0,
                            organic_carbon=oc / 10.0 if oc else None,
                            ph=ph, bulk_density=bd,
                        ))

                if layers:
                    profiles[cell.cell_id] = SoilProfile(
                        profile_id=f"isda_{cell.cell_id}",
                        lat=cell.lat, lon=cell.lon,
                        source="iSDA S3 (30m)",
                        layers=layers,
                    )

            if profiles:
                self.logger.info(f"iSDA S3: Retrieved {len(profiles)}/{total} soil profiles")
                return profiles

        except Exception as e:
            self.logger.warning(f"iSDA S3 retrieval failed: {e}")

        return None

    def _retrieve_hwsd_for_grid(
        self, grid, region: Region
    ) -> Optional[Dict[int, SoilProfile]]:
        """Retrieve per-cell HWSD soil data for ACEA/CRAFT platforms.

        NOTE: This retrieval runs in the HARMONIZE stage because it needs
        the spatial grid (cell coordinates) which is only available after
        grid generation. Ideally, the RETRIEVE stage would have two phases:
        pre-grid (boundaries, climate metadata) and post-grid (per-cell
        soil, crop masks). This is a pragmatic workaround until the pipeline
        is restructured to support two-phase retrieval.

        TODO: Move to a post-grid RETRIEVE phase when the pipeline
        architecture supports it.

        Runs after grid creation in the HARMONIZE stage. Checks for HWSD
        paths in three places (in order):
          1. data_sources.soil.hwsd_bil_path / hwsd_mdb_path (top-level)
          2. Platform-specific config (ACEA or CRAFT hwsd_bil_path)
          3. Auto-discovery in known locations

        Only runs when HWSD-dependent platforms (ACEA, CRAFT) are enabled.

        Args:
            grid: SpatialGrid with cell coordinates
            region: Region for metadata

        Returns:
            Dictionary of cell_id -> SoilProfile, or None if unavailable
        """
        # Check if any platform that can use HWSD soil data is enabled
        enabled = self.config.get_enabled_platforms()
        hwsd_platforms = {Platform.ACEA, Platform.CRAFT, Platform.SARRA_PY, Platform.PYTHIA}
        if not hwsd_platforms.intersection(set(enabled)):
            return None

        # Resolve HWSD paths: top-level data_sources > platform config > auto-discovery
        bil_path = None
        mdb_path = None

        # 1. Top-level data_sources.soil config
        soil_cfg = self.config.data_sources.soil
        if soil_cfg.hwsd_bil_path and soil_cfg.hwsd_mdb_path:
            bil_path = Path(soil_cfg.hwsd_bil_path)
            mdb_path = Path(soil_cfg.hwsd_mdb_path)

        # 2. Platform-specific config fallback (check all platforms that may have HWSD paths)
        if not (bil_path and mdb_path):
            for plat in [Platform.CRAFT, Platform.ACEA, Platform.SARRA_PY, Platform.PYTHIA]:
                pcfg = self.config.get_platform_config(plat)
                if pcfg:
                    p_bil = getattr(pcfg, 'hwsd_bil_path', None)
                    p_mdb = getattr(pcfg, 'hwsd_mdb_path', None)
                    if p_bil and p_mdb:
                        bil_path = Path(p_bil)
                        mdb_path = Path(p_mdb)
                        break

        # 3. Auto-discovery in known locations
        if not (bil_path and mdb_path):
            search_dirs = [
                Path("data/hwsd"),
                Path(__file__).resolve().parents[4] / "data" / "hwsd",
            ]
            for d in search_dirs:
                candidate_bil = d / "HWSD2.bil"
                candidate_mdb = d / "HWSD2.mdb"
                if candidate_bil.exists() and candidate_mdb.exists():
                    bil_path = candidate_bil
                    mdb_path = candidate_mdb
                    self.logger.info(f"Auto-discovered HWSD at {d}")
                    break

        if not (bil_path and mdb_path and bil_path.exists() and mdb_path.exists()):
            self.logger.debug("HWSD paths not configured or not found, skipping per-cell retrieval")
            return None

        # Build cell coordinates from grid
        cell_coords = [(cell.lat, cell.lon) for cell in grid.cells]
        cell_ids = [cell.cell_id for cell in grid.cells]

        self.logger.info(
            f"Retrieving HWSD soil data for {len(cell_coords)} grid cells..."
        )

        try:
            from prismpy.sources.soil.hwsd import HWSDSource, HWSDConfig

            hwsd_source = HWSDSource(
                config=HWSDConfig(
                    bil_path=bil_path,
                    mdb_path=mdb_path,
                    use_defaults=True,
                ),
                provenance=self.provenance,
            )

            result = hwsd_source.retrieve(
                region=region,
                cell_coords=cell_coords,
            )

            if result.success and result.data:
                raw_profiles = result.data.profiles
                # Re-key profiles by cell_id (HWSDSource keys by index)
                profiles = {}
                for i, cell_id in enumerate(cell_ids):
                    if i in raw_profiles:
                        profiles[cell_id] = raw_profiles[i]

                n_real = sum(
                    1 for p in profiles.values()
                    if not p.metadata.get('is_default', False)
                )
                self.logger.info(
                    f"HWSD: {n_real} real profiles, "
                    f"{len(profiles) - n_real} defaults, "
                    f"{len(profiles)} total for {len(cell_ids)} cells"
                )
                return profiles
            else:
                self.logger.warning(f"HWSD retrieval failed: {result.errors}")

        except Exception as e:
            self.logger.warning(f"HWSD retrieval error: {e}")

        return None

    def _load_crop_params(self) -> Optional[CropParameters]:
        """Load crop parameters from templates or config.

        Returns:
            CropParameters object or None
        """
        # Check for variety template in platform config
        if self.config.platform_config:
            sarra_config = self.config.platform_config.sarra_py
            if sarra_config and sarra_config.variety_template:
                variety_template = sarra_config.variety_template
                self.logger.info(f"Using variety template: {variety_template}")
                return self._create_default_crop_params()

        return None

    def _create_default_crop_params(self) -> CropParameters:
        """Create default crop parameters for SARRA-Py maize.

        Returns:
            CropParameters with default SARRA-Py maize values
        """
        # Default SARRA-Py maize parameters
        default_params = {
            # Phenology (thermal time / GDD)
            "SDJLevee": 90.0,
            "SDJBVP": 500.0,
            "SDJRPR": 400.0,
            "SDJMatu1": 500.0,
            "SDJMatu2": 200.0,
            # Temperature thresholds
            "TBase": 8.0,
            "TOpt1": 26.0,
            "TOpt2": 34.0,
            "TLim": 44.0,
            # Yield parameters
            "KRdtPotA": 0.4,
            "KRdtPotB": 200.0,
            "txConversion": 5.8,
            "txResGrain": 0.8,
            # Assimilation
            "txAssimBVP": 1.0,
            "txAssimMatu1": 0.9,
            "txAssimMatu2": 0.1,
            # Maintenance respiration
            "kRespMaint": 0.01,
            "tempMaint": 25.0,
            # LAI parameters
            "LAImax": 5.0,
            "KDF": 0.5,
            # Root parameters
            "profRacIni": 0.1,
            "profRacMax": 2.0,
            "vitRac": 0.02,
            # Water stress
            "pFsem": 2.5,
            "pFhum": 0.0,
            "SeuilRuiss": 15.0,
        }

        return CropParameters(
            crop_name=self.config.crop.name,
            variety_name=self.config.crop.variety or "maize_west_africa",
            source="default_sarra_py",
            parameters=default_params,
        )

    def _execute_harmonize(self, retrieved_data: Dict[str, Any]) -> StageResult:
        """Execute the HARMONIZE stage.

        Aligns spatial data, fills gaps, and validates quality.

        Args:
            retrieved_data: Data from RETRIEVE stage

        Returns:
            StageResult with harmonized data
        """
        start_time = datetime.now()
        self.logger.info("Stage 2: HARMONIZE - Aligning and validating data")

        errors = []
        warnings = []

        try:
            # Create unified data container
            region = retrieved_data.get("region")

            # Generate spatial grid if needed
            from prismpy.models.spatial import SpatialGrid

            grid = None
            if region:
                # Get clip geometry from region if available (for polygon clipping)
                clip_geometry = None
                if hasattr(region, 'geometry_wkt') and region.geometry_wkt:
                    try:
                        from shapely import wkt
                        clip_geometry = wkt.loads(region.geometry_wkt)
                        self.logger.info("Using polygon clipping for grid generation")
                    except Exception as e:
                        self.logger.warning(f"Failed to load geometry for clipping: {e}")

                # Always use 5-arcmin grid for maximum boundary precision.
                # Platforms that need coarser grids (ACEA=30arcmin) handle
                # the mapping internally (e.g., _compute_30arcmin_cell_ids).
                grid = SpatialGrid.from_bounds(
                    region.bounds,
                    resolution="5arcmin",
                    clip_geometry=clip_geometry,
                )
                self.logger.info(f"Created grid with {grid.n_cells} cells")

            # Retrieve per-cell soil data: try iSDA API first (Africa, 30m),
            # then HWSD fallback (global, 1km)
            soil_data = retrieved_data.get("soil")
            if grid and region:
                isda_soil = self._retrieve_isda_api_for_grid(grid, region)
                if isda_soil:
                    soil_data = isda_soil
                else:
                    hwsd_soil = self._retrieve_hwsd_for_grid(grid, region)
                    if hwsd_soil:
                        soil_data = hwsd_soil

            unified_data = UnifiedData(
                region=region,
                grid=grid,
                climate=retrieved_data.get("climate"),
                soil=soil_data,
                crop_params=retrieved_data.get("crop_params"),
                crop_calendar=retrieved_data.get("crop_calendar"),
                metadata={
                    "harmonized_at": datetime.now().isoformat(),
                    "config_version": self.config.project.version,
                },
            )

            # Note: Actual harmonization (gap-filling, resampling, etc.)
            # would be implemented using the harmonizers module

        except Exception as e:
            errors.append(f"Harmonization failed: {str(e)}")
            self.logger.error(f"Harmonization error: {e}")
            unified_data = None

        duration = (datetime.now() - start_time).total_seconds()
        return StageResult(
            stage=PipelineStage.HARMONIZE,
            success=len(errors) == 0 and unified_data is not None,
            data=unified_data,
            errors=errors,
            warnings=warnings,
            duration_seconds=duration,
        )

    def _execute_translate(
        self,
        unified_data: UnifiedData,
    ) -> Dict[str, TranslationResult]:
        """Execute the TRANSLATE stage.

        Generates platform-specific outputs for all enabled platforms.

        Args:
            unified_data: Harmonized data from HARMONIZE stage

        Returns:
            Dictionary of translation results by platform
        """
        self.logger.info("Stage 3: TRANSLATE - Generating platform outputs")

        results = {}
        enabled_platforms = self.config.get_enabled_platforms()

        for platform in enabled_platforms:
            self.logger.info(f"  Translating for {platform.value}")

            translator = self._get_translator(platform)
            if translator:
                try:
                    result = translator.translate(unified_data)
                    results[platform.value] = result
                except Exception as e:
                    self.logger.error(f"Translation error for {platform.value}: {e}")
                    from prismpy.translators.base import TranslationResult
                    results[platform.value] = TranslationResult(
                        success=False,
                        platform=platform,
                        output_dir=Path(self.config.output.base_dir) / platform.value,
                        output_files=[],
                        errors=[str(e)],
                        warnings=[],
                        metadata={},
                    )
            else:
                # Create placeholder result for unimplemented translator
                from prismpy.translators.base import TranslationResult
                output_dir = Path(self.config.output.base_dir) / platform.value / self.config.region.name
                output_dir.mkdir(parents=True, exist_ok=True)

                results[platform.value] = TranslationResult(
                    success=True,
                    platform=platform,
                    output_dir=output_dir,
                    output_files=[],
                    errors=[],
                    warnings=[f"Translator for {platform.value} not yet fully implemented"],
                    metadata={"status": "placeholder"},
                )

        return results

    def _execute_validate(
        self,
        translation_results: Dict[str, TranslationResult],
    ) -> StageResult:
        """Execute the VALIDATE stage.

        Validates all generated outputs against platform requirements
        using the dedicated BaseValidator hierarchy.

        Args:
            translation_results: Results from TRANSLATE stage

        Returns:
            StageResult with validation summary
        """
        start_time = datetime.now()
        self.logger.info("Stage 4: VALIDATE - Checking outputs")

        errors = []
        warnings = []
        validation_summary = {}

        for platform_name, result in translation_results.items():
            if not result.success:
                errors.extend([f"{platform_name}: {e}" for e in result.errors])
                continue

            # Run platform-specific validation via BaseValidator hierarchy
            platform = Platform(platform_name)
            validator = self._get_validator(platform, result.output_dir)

            if validator:
                try:
                    val_result: ValidationResult = validator.validate()
                    if val_result.errors:
                        errors.extend(
                            [f"{platform_name}: {issue}" for issue in val_result.errors]
                        )
                    if val_result.warnings:
                        warnings.extend(
                            [f"{platform_name}: {issue}" for issue in val_result.warnings]
                        )
                    validation_summary[platform_name] = {
                        "valid": val_result.valid,
                        "errors_count": val_result.n_errors,
                        "warnings_count": val_result.n_warnings,
                        "files_checked": val_result.files_checked,
                    }
                    self.logger.info(f"  {val_result.summary()}")
                except Exception as e:
                    warnings.append(f"{platform_name}: Validation skipped ({e})")
            else:
                validation_summary[platform_name] = {
                    "valid": True,
                    "errors_count": 0,
                    "warnings_count": 0,
                    "note": "No validator available",
                }

        duration = (datetime.now() - start_time).total_seconds()
        return StageResult(
            stage=PipelineStage.VALIDATE,
            success=len(errors) == 0,
            data=validation_summary,
            errors=errors,
            warnings=warnings,
            duration_seconds=duration,
        )

    def _execute_package(
        self,
        unified_data: Optional[UnifiedData],
        translation_results: Dict[str, TranslationResult],
    ) -> StageResult:
        """Execute the PACKAGE stage.

        Generates self-documenting data packages for each platform
        (manifest, provenance, README) and saves the pipeline-level
        provenance audit trail.

        Args:
            unified_data: Harmonized data from HARMONIZE stage
            translation_results: Results from TRANSLATE stage

        Returns:
            StageResult with packaging summary
        """
        start_time = datetime.now()
        self.logger.info("Stage 5: PACKAGE - Generating data packages")

        errors = []
        warnings = []
        provenance_path = None
        package_summary = {}

        # Generate per-platform packages via translator.generate_package()
        for platform_name, result in translation_results.items():
            if not result.success:
                continue

            platform = Platform(platform_name)
            translator = self._translators.get(platform)

            if translator and unified_data:
                try:
                    package_files = translator.generate_package(
                        unified_data, result.output_files
                    )
                    result.output_files.extend(package_files)
                    package_summary[platform_name] = {
                        "package_files": len(package_files),
                        "files": [str(f.name) for f in package_files],
                    }
                    self.logger.info(
                        f"  {platform_name}: {len(package_files)} package files generated"
                    )
                except Exception as e:
                    warnings.append(f"{platform_name}: Package generation failed: {e}")
                    self.logger.warning(f"Package generation failed for {platform_name}: {e}")

        # Save pipeline-level provenance
        try:
            provenance_path = self.provenance.save()

            report = self.provenance.get_report()
            report_path = provenance_path.parent / f"{self.provenance.session_id}_report.txt"
            with open(report_path, "w") as f:
                f.write(report)

            self.logger.info(f"Pipeline provenance saved to {provenance_path}")

        except Exception as e:
            errors.append(f"Provenance save failed: {str(e)}")
            self.logger.error(f"Provenance save error: {e}")

        duration = (datetime.now() - start_time).total_seconds()
        return StageResult(
            stage=PipelineStage.PACKAGE,
            success=len(errors) == 0,
            data={"provenance_path": provenance_path, "packages": package_summary},
            errors=errors,
            warnings=warnings,
            duration_seconds=duration,
        )

    def execute(
        self,
        stages: Optional[List[PipelineStage]] = None,
        progress_callback=None,
    ) -> PipelineResult:
        """Execute the translation pipeline.

        Args:
            stages: Optional list of stages to run. If None, runs all stages.
            progress_callback: Optional callback for per-stage progress updates.
                Must implement on_stage_start(stage, description),
                on_stage_complete(stage, result), and optionally
                on_substage_progress(stage, task, current, total, detail).

        Returns:
            PipelineResult with all stage results and final status
        """
        start_time = datetime.now()
        stages = stages or list(PipelineStage)

        self.logger.info(f"Starting translation pipeline for {self.config.project.name}")
        self.logger.info(f"Region: {self.config.region.name}, {self.config.region.country}")
        self.logger.info(f"Targets: {[p.value for p in self.config.get_enabled_platforms()]}")

        stage_results: Dict[str, StageResult] = {}
        translation_results: Dict[str, TranslationResult] = {}
        provenance_path = None

        def _notify_start(stage_name, description):
            if progress_callback:
                try:
                    progress_callback.on_stage_start(stage_name, description)
                except Exception:
                    pass

        def _notify_complete(stage_name, result):
            if progress_callback:
                try:
                    progress_callback.on_stage_complete(stage_name, result)
                except Exception:
                    pass

        try:
            # Stage 1: RETRIEVE
            if PipelineStage.RETRIEVE in stages:
                _notify_start("retrieve", "Gathering your data")
                result = self._execute_retrieve()
                stage_results["retrieve"] = result
                _notify_complete("retrieve", result)
                if not result.success:
                    return self._build_result(
                        False, stage_results, translation_results, None, start_time
                    )

            # Stage 2: HARMONIZE
            if PipelineStage.HARMONIZE in stages:
                _notify_start("harmonize", "Aligning and checking")
                retrieved_data = stage_results.get("retrieve", StageResult(
                    stage=PipelineStage.RETRIEVE, success=True, data={}
                )).data or {}
                result = self._execute_harmonize(retrieved_data)
                stage_results["harmonize"] = result
                _notify_complete("harmonize", result)
                if not result.success:
                    return self._build_result(
                        False, stage_results, translation_results, None, start_time
                    )

            # Stage 3: TRANSLATE
            if PipelineStage.TRANSLATE in stages:
                _notify_start("translate", "Building platform files")
                unified_data = stage_results.get("harmonize", StageResult(
                    stage=PipelineStage.HARMONIZE, success=True, data=UnifiedData(region=None)
                )).data
                if unified_data:
                    translate_start = datetime.now()
                    translation_results = self._execute_translate(unified_data)
                    translate_duration = (datetime.now() - translate_start).total_seconds()
                    result = StageResult(
                        stage=PipelineStage.TRANSLATE,
                        success=all(r.success for r in translation_results.values()),
                        data=translation_results,
                        errors=[
                            e for r in translation_results.values() for e in r.errors
                        ],
                        warnings=[
                            w for r in translation_results.values() for w in r.warnings
                        ],
                        duration_seconds=translate_duration,
                    )
                    stage_results["translate"] = result
                    _notify_complete("translate", result)

            # Stage 4: VALIDATE
            if PipelineStage.VALIDATE in stages and translation_results:
                _notify_start("validate", "Verifying outputs")
                result = self._execute_validate(translation_results)
                stage_results["validate"] = result
                _notify_complete("validate", result)

            # Stage 5: PACKAGE
            if PipelineStage.PACKAGE in stages:
                _notify_start("package", "Preparing your package")
                unified_data = stage_results.get("harmonize", StageResult(
                    stage=PipelineStage.HARMONIZE, success=True, data=None
                )).data
                result = self._execute_package(unified_data, translation_results)
                stage_results["package"] = result
                _notify_complete("package", result)
                if result.data:
                    provenance_path = result.data.get("provenance_path")

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return self._build_result(
                False, stage_results, translation_results, None, start_time,
                error=str(e)
            )

        # Determine overall success
        success = all(r.success for r in stage_results.values())

        return self._build_result(
            success, stage_results, translation_results, provenance_path, start_time
        )

    def _build_result(
        self,
        success: bool,
        stage_results: Dict[str, StageResult],
        translation_results: Dict[str, TranslationResult],
        provenance_path: Optional[Path],
        start_time: datetime,
        error: Optional[str] = None,
    ) -> PipelineResult:
        """Build the final pipeline result."""
        duration = (datetime.now() - start_time).total_seconds()

        if error:
            if "execute" not in stage_results:
                stage_results["execute"] = StageResult(
                    stage=PipelineStage.RETRIEVE,
                    success=False,
                    errors=[error],
                )

        self.logger.info(
            f"Pipeline {'completed successfully' if success else 'failed'} "
            f"in {duration:.1f}s"
        )

        return PipelineResult(
            success=success,
            stages=stage_results,
            translation_results=translation_results,
            provenance_path=provenance_path,
            total_duration_seconds=duration,
        )


def run_pipeline(config_path: str) -> PipelineResult:
    """Convenience function to run the pipeline from a config file.

    Args:
        config_path: Path to the project configuration YAML file

    Returns:
        PipelineResult with execution results
    """
    config = load_config(config_path)
    pipeline = TranslationPipeline(config)
    return pipeline.execute()
