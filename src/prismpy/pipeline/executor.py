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
from typing import Any, Dict, List, Optional, Tuple, Type
import logging

from prismpy.config.schema import ProjectConfig, Platform
from prismpy.config.loader import load_config
from prismpy.models.region import Region
from prismpy.models.climate import ClimateTimeSeries, ClimateRecord
from prismpy.models.soil import SoilProfile, SoilLayer
from prismpy.models.crop import CropParameters, CropCalendar
from prismpy.models.provenance import DecisionType, OperationType
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
    # V2-22c-PRE.4.1 (D25) — REMEDIATION runs between TRANSLATE and
    # VALIDATE so post-remediation validation is honest. Most runs
    # (originals, retries) take the no-op path inside
    # _execute_remediation; only re-runs derived from a cockpit
    # bulk-fix submission carry a remediation_spec to apply.
    REMEDIATION = "remediation"
    VALIDATE = "validate"
    PACKAGE = "package"


from prismpy.sources.climate._cancel import PipelineCancelled, raise_if_cancelled
# V2-22c-PRE.4 — RemediationBlocked is the structured exception for
# Veto #4 server enforcement. Imported at module level so the narrow
# `except RemediationBlocked` catch in `execute()` (evaluator §12.5
# binding) sees the symbol without a per-stage local import. The
# remediation helpers (`_veto_4_tier`, `_block_message`) stay local
# to `_execute_remediation` to keep import overhead off the cancel
# fast path.
from prismpy.pipeline._remediation import RemediationBlocked


def _extract_run_id(callback) -> Optional[str]:
    """Best-effort run-identifier extraction for cache-manifest provenance.

    V2-22a B2 stage 9 — climate sources accept an optional `run_id`
    kwarg that is recorded in the cache manifest's provenance field. The
    executor reads it from the progress callback in two defensive ways:

      1. `callback.run_id` if exposed directly (preferred shape — clean
         protocol, decouples the source from any specific callback class).
      2. `callback.run.id` for the current prismweb shape, where the
         WebProgressCallback exposes a Django PipelineRun model on `.run`.

    Returns None when no identifier is discoverable; manifest writers
    record None as an empty string with no semantic loss — the manifest's
    correctness checks (bbox, file_count, marker) do not depend on run_id.
    """
    if callback is None:
        return None
    explicit = getattr(callback, 'run_id', None)
    if explicit is not None:
        return str(explicit)
    run_obj = getattr(callback, 'run', None)
    if run_obj is None:
        return None
    inner_id = getattr(run_obj, 'id', None)
    return str(inner_id) if inner_id is not None else None


def _gate_value_range_climate_delegation(
    sci: Dict[str, Any],
    sarra_py_enabled: bool,
) -> None:
    """Escalate a `value_range_climate` info to warning post-merge.

    Rationale: the scientific validator emits the `value_range_climate`
    info record unconditionally when it detects SARRA-Py-style
    file-based climate data, delegating actual range verification to
    the platform validator (`post_translate_range_sarra_py_<var>`
    records). The info's "When available, the per-variable ranges
    appear below" hedge is honest ONLY when those delegated records
    actually land. If translation failed, the platform validator
    skipped, or rasterio errored, the info is misleading — the user
    looks for ranges that aren't there.

    Called unconditionally at the post-merge surfacing point in the
    executor so every exit path — successful merge, post-translate
    skipped entirely, or post-translate raised — observes the same
    gating logic. Idempotent: if the delegated records are present,
    the check is a no-op; if the check has already been escalated,
    re-running this helper is a safe no-op as well. Presence of AT
    LEAST ONE delegated record proves the validator ran; no
    escalation fires in the partial-variable case.

    `sarra_py_enabled` gates the entire escalation so a non-SARRA-Py
    run (e.g., ACEA-only with file-based climate config populated)
    does NOT manufacture a bogus SARRA-Py warning. The scientific
    validator's `_is_file_based_climate()` detector keys on dict
    shape (`rainfall_dir` / `agera5_dir`) and would emit the
    delegation info even when SARRA-Py wasn't in the pipeline; the
    gate must reconcile that against the pipeline's actual enabled
    platforms before escalating.

    On escalation, both `result` ("info" → "warning") AND `passed`
    (True → False) are updated together so downstream consumers
    that key off either field (UI renderers, audit tooling, JSON
    report serializers) agree on the check's outcome. `check`,
    `scope`, `manuscript_claim`, and `details.coverage_kind` are
    preserved so cross-run diffing still matches the same record.
    """
    if not sarra_py_enabled:
        return
    checks = sci.get("checks", [])
    delegated = [
        c for c in checks
        if c.get("check", "").startswith(
            "post_translate_range_sarra_py_"
        )
    ]
    info_idx = next(
        (
            i for i, c in enumerate(checks)
            if c.get("check") == "value_range_climate"
            and c.get("details", {}).get("coverage_kind") == "delegated"
        ),
        None,
    )
    if info_idx is not None and not delegated:
        checks[info_idx]["result"] = "warning"
        checks[info_idx]["passed"] = False
        checks[info_idx]["summary"] = (
            "SARRA-Py sampled climate range check did not run — "
            "platform validator produced no per-variable records."
        )


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
            self.provenance.start_artifact("region", stage="retrieve")

            # Load region from GADM or manual bounds
            self.logger.info(f"Loading region: {self.config.region.name}")

            from prismpy.models.region import BoundingBox

            # Check boundary source type
            boundary_source = self.config.region.boundary.source.value

            # Codex Path A — track the RESOLVED boundary source
            # independently of config.source, since the GADM failure
            # path falls back to manual_bounds while config.source
            # still reads "gadm". `region.boundary_source` must
            # reflect what the pipeline actually used so cache /
            # lock / ETA paths route correctly.
            resolved_boundary_source = boundary_source

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
                                # V2-19b-fix Finding 3 + terminology fix:
                                # record pygadm as the boundary source.
                                # In prismweb, pygadm IS the standard
                                # boundary path (no local GADM files
                                # shipped). Not a fallback.
                                if self.provenance and self.provenance.enabled:
                                    self.provenance.record_decision(
                                        decision_type=DecisionType.SOURCE_SELECTION,
                                        description=(
                                            f"Boundary source: pygadm "
                                            f"({country_iso3} level {gadm_level} "
                                            f"'{filter_value}')"
                                        ),
                                        rationale=(
                                            "pygadm provides programmatic access to "
                                            "GADM v4.1 administrative boundaries "
                                            "without requiring local data files. "
                                            "Used as the standard boundary source "
                                            "in prismweb. Boundaries are downloaded "
                                            "on demand from the GADM web service "
                                            "and cached locally for reuse."
                                        ),
                                        alternatives=[
                                            "Local GADM shapefiles (when pre-downloaded)",
                                            "Manual bounding box from config",
                                        ],
                                        reference="prismpy.pipeline.executor._execute_retrieve (pygadm branch)",
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
                            # Resolved source is manual on this fallback
                            # path even though config.source was gadm.
                            resolved_boundary_source = 'manual'
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
                    # Unsupported-source branch also resolves to manual.
                    resolved_boundary_source = 'manual'
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

            # V2-22b/P.1 — record the RESOLVED boundary source, not
            # the requested config source. Fallback branches (GADM
            # failed → manual_bounds) set `resolved_boundary_source
            # = 'manual'` above so cache / lock / ETA paths route by
            # bbox, not by the stale config.source value.
            region.boundary_source = resolved_boundary_source

            data["region"] = region

            self.provenance.record_retrieval(
                source="config",
                parameters={"region_name": self.config.region.name},
            )

            # Load climate data
            self.logger.info("Loading climate data...")
            # V2-19: dedicated artifact for climate lineage
            if self.provenance.enabled:
                self.provenance.start_artifact("climate", artifact_id="climate", stage="retrieve")
                # V2-19 site #6: per-platform climate SOURCE_SELECTION decisions
                self._record_climate_source_decisions()
            climate_data = self._load_climate_data(region)
            # V2-19: record the climate retrieval transformation so any
            # pending decisions recorded during _load_climate_data flush
            if self.provenance.enabled:
                self.provenance.record_retrieval(
                    source="climate_sources",
                    parameters={
                        "n_locations": len(climate_data) if climate_data else 0,
                        "enabled_platforms": [
                            p.value for p in self.config.get_enabled_platforms()
                        ],
                    },
                    artifact_id="climate",
                )
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
            # V2-19: dedicated artifact for soil lineage
            if self.provenance.enabled:
                self.provenance.start_artifact("soil", artifact_id="soil", stage="retrieve")
            soil_data = self._load_soil_data(region)
            # V2-19: record the soil retrieval transformation to flush
            # any pending decisions recorded during _load_soil_data
            if self.provenance.enabled:
                self.provenance.record_retrieval(
                    source="soil_sources",
                    parameters={
                        "n_profiles": len(soil_data) if soil_data else 0,
                    },
                    artifact_id="soil",
                )
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

        except PipelineCancelled:
            # V2-22b L Gate B round 3 (F-9): whole-stage wrapper was
            # catching PipelineCancelled and rewriting as
            # StageResult(success=False). The new `except PipelineCancelled`
            # handler at tasks.py:646 was dead code on real pipelines.
            # Re-raise so the boundary catcher at pipeline.execute()
            # + the prismweb cleanup helper run correctly. Preserves
            # original ``exc.where`` for operator diagnostics.
            raise
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

    def _record_climate_source_decisions(self) -> None:
        """V2-19 site #6: record per-platform climate SOURCE_SELECTION.

        For each enabled platform, emit a decision documenting which
        climate source(s) will be used. This surfaces the platform-
        climate mapping that is otherwise buried in platform config.

        The mapping is:
        - SARRA-Py → TAMSAT v3.1 (rainfall) + AgERA5 (temperature, radiation)
        - CRAFT, PYTHIA, ACEA → NASA POWER v9 (all variables)

        Mappings come from Platform Defaults documented in the README
        and enforced in the respective translators.
        """
        if not self.provenance or not self.provenance.enabled:
            return

        from prismpy.config.schema import Platform

        # V2-19 C7 (CD-14): Platform climate source cascade with
        # AC11-compliant rationales (source + positive/negative domain
        # + reviewer pre-answer).
        climate_map = {
            Platform.SARRA_PY: (
                "TAMSAT v3.1 + AgERA5",
                "SARRA-Py requires high-resolution rainfall (TAMSAT v3.1 "
                "at 4 km, Maidment et al. 2017) and standard met variables "
                "(AgERA5 at 0.1\u00b0, Boogaard et al. 2020). Valid for "
                "West African Sahel where TAMSAT is calibrated against "
                "dense rain-gauge networks. NOT valid outside TAMSAT's "
                "Africa domain or in regions where TAMSAT has known "
                "biases (montane East Africa, coastal Guinean zone). "
                "NASA POWER (0.5\u00b0) is too coarse for SARRA-Py's "
                "field-to-district scale simulations.",
                ["NASA POWER (rejected: 0.5\u00b0 too coarse for field scale)"],
            ),
            Platform.CRAFT: (
                "NASA POWER v9",
                "NASA POWER (Stackhouse et al. 2018) provides a unified "
                "source for Tmax, Tmin, precipitation, solar radiation at "
                "0.5\u00b0 (~56 km). Valid for CRAFT's admin-unit scale "
                "(5-arcmin grid, typical study areas 5,000-50,000 km\u00b2). "
                "NOT valid where sub-grid climate heterogeneity matters "
                "(mountainous terrain, coastal gradients, urban heat "
                "islands). A reviewer would ask: 'Why not use higher-"
                "resolution gridded products?' Answer: CRAFT aggregates "
                "to admin-unit level anyway, and NASA POWER's global "
                "coverage eliminates multi-source fusion artifacts.",
                ["TAMSAT + AgERA5 (higher resolution but Africa-only, "
                 "multi-source fusion complexity)"],
            ),
            Platform.PYTHIA: (
                "NASA POWER v9",
                "PYTHIA downloads per-site weather via NASA POWER /point "
                "endpoint. Each grid cell centroid is queried individually. "
                "Valid for site-level DSSAT simulations. NOT valid for "
                "sub-daily or hourly weather (POWER is daily only).",
                ["TAMSAT + AgERA5 (Africa-only)"],
            ),
            Platform.ACEA: (
                "NASA POWER v9",
                "ACEA uses NASA POWER at the 30-arcmin grid level. Valid "
                "for AquaCrop's coarse resolution. NOT valid for "
                "sub-grid irrigation scheduling.",
                ["TAMSAT + AgERA5 (Africa-only)"],
            ),
        }

        for platform in self.config.get_enabled_platforms():
            if platform not in climate_map:
                continue
            source, rationale, alternatives = climate_map[platform]
            self.provenance.record_decision(
                decision_type=DecisionType.SOURCE_SELECTION,
                description=f"Climate source for {platform.value}: {source}",
                rationale=rationale,
                alternatives=alternatives,
                reference=f"prismpy.pipeline.executor._record_climate_source_decisions",
                artifact_id="climate",
            )

    # V2-19 B1: Source native resolutions (for effective-resolution warning).
    # Values are in decimal degrees at the equator. Where a source is
    # documented in km rather than degrees, the km→degrees conversion uses
    # 1° ≈ 111 km.
    _SOURCE_RESOLUTIONS_DEG: Dict[str, float] = {
        "NASA POWER": 0.5,            # ~55 km
        "AgERA5": 0.1,                # ~10 km
        "TAMSAT": 0.0375,             # ~4 km
        "HWSD": 1.0 / 111.0,          # 1 km → ~0.009°
        "iSDA": 30.0 / 111000.0,      # 30 m → ~0.00027°
    }

    def _record_effective_resolution_warning(
        self,
        target_resolution_deg: float,
        target_resolution_label: str,
        active_sources: List[str],
    ) -> None:
        """V2-19 B1: emit effective-resolution warning when target grid is finer than source.

        When the target grid resolution is finer than a source's native
        resolution, neighbouring cells will share identical values —
        the effective resolution is the SOURCE's native resolution, not
        the target grid. Surfacing this prevents users from interpreting
        results at a finer spatial scale than the data actually supports.

        Per crop-modeling-specialist: this is "the single highest-value
        feature PRISM could ship." The warning being noisy is the feature,
        not the bug — it fires whenever source ≠ target native resolution.

        Args:
            target_resolution_deg: Target grid resolution in decimal degrees
            target_resolution_label: Human label (e.g., "5-arcmin (~9 km)")
            active_sources: Names of sources that contributed data for this run
        """
        if not self.provenance or not self.provenance.enabled:
            return

        # For each active source, check if target is finer than its native
        coarser_sources: List[Tuple[str, float]] = []
        for source in active_sources:
            src_res = self._SOURCE_RESOLUTIONS_DEG.get(source)
            if src_res is None:
                continue
            if target_resolution_deg < src_res:
                coarser_sources.append((source, src_res))

        if not coarser_sources:
            return  # target is coarser or equal to all sources — no warning

        # Build the warning message and rationale
        lines = [
            f"Effective-resolution warning: target grid {target_resolution_label} "
            f"is finer than source native resolution for:"
        ]
        for src, res in coarser_sources:
            native_km = res * 111.0
            lines.append(
                f"  - {src}: native {res:.4f}° (~{native_km:.0f} km) — "
                f"neighbouring cells share identical values; effective "
                f"resolution is ~{native_km:.0f} km, NOT "
                f"~{target_resolution_deg * 111.0:.0f} km"
            )
        lines.append(
            "Consider aggregating results to the native resolution of the "
            "coarsest source before interpreting per-cell values."
        )
        message = "\n".join(lines)

        self.logger.warning(message)
        self.provenance.record_decision(
            decision_type=DecisionType.SOURCE_SELECTION,
            description=(
                f"Effective resolution WARNING: target {target_resolution_label} "
                f"finer than {len(coarser_sources)} source(s)"
            ),
            rationale=message,
            alternatives=[
                "Aggregate results to coarsest source native resolution",
                "Use only sources with native resolution ≥ target",
                "Accept apparent-vs-effective mismatch and document explicitly",
            ],
            reference="prismpy.pipeline.executor._record_effective_resolution_warning",
            artifact_id="grid",
            severity="warning",
            label="Resolution mismatch: source coarser than target grid",
        )

    def _load_climate_data(self, region: Region) -> Optional[Dict[str, Any]]:
        """Load climate data from configured paths or sources.

        Args:
            region: Region to load data for

        Returns:
            Dictionary with climate data info including paths to GeoTIFF files
        """
        from datetime import date

        # Get temporal range from config (cross-year-aware)
        start_date = date(self.config.temporal.start_year, 1, 1)
        crop_cal = self.config.crop.calendar if self.config.crop else None
        end_date = self.config.temporal.get_climate_end_date(crop_cal)

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
                    import time as _time
                    _tamsat_last_report = [0.0]
                    def _tamsat_progress(current, total, detail=''):
                        now = _time.monotonic()
                        is_final = current >= total
                        if self._progress_callback and (is_final or now - _tamsat_last_report[0] >= 10):
                            _tamsat_last_report[0] = now
                            label = detail or f'TAMSAT rainfall: file {current} of {total}'
                            self._progress_callback.on_substage_progress(
                                'retrieve', 'Downloading TAMSAT rainfall',
                                current, total, label)
                    tamsat_result = tamsat.retrieve(
                        region=region, start_date=start_date,
                        end_date=end_date, download=True,
                        progress_callback=_tamsat_progress,
                        # V2-22b L: thread cancel_check so Phase 1 +
                        # Phase 2 observe user cancel.
                        cancel_check=getattr(self, '_cancel_check', None),
                        run_id=_extract_run_id(self._progress_callback),
                    )
                    if tamsat_result.success and tamsat_result.data:
                        climate_data["rainfall_dir"] = tamsat_result.data.data_dir
                        climate_data["rainfall_file_count"] = tamsat_result.data.file_count
                        self.logger.info(f"TAMSAT: {tamsat_result.data.file_count} files")
                        got_data = True
                    else:
                        self.logger.warning(f"TAMSAT download failed: {tamsat_result.errors}")
            except PipelineCancelled:
                # V2-22b L: cancel unwinds past the broad except so the
                # pipeline.execute boundary can run handler-local cleanup.
                raise
            except Exception as e:
                self.logger.warning(f"TAMSAT download error: {e}")

            # Download AgERA5 temperature/radiation
            try:
                from prismpy.sources.climate.agera5 import AgERA5Source

                agera5 = AgERA5Source(cache_dir=cache_dir, provenance=self.provenance)
                if agera5.sarra_download_available:
                    self.logger.info("Downloading AgERA5 temperature data...")
                    # Force immediate substage update so the label
                    # transitions from "TAMSAT: 1096/1096" to AgERA5
                    if self._progress_callback:
                        self._progress_callback.on_substage_progress(
                            'retrieve', 'Downloading AgERA5 temperature',
                            0, 1, 'AgERA5 temperature: starting CDS download...')
                    _agera5_last_report = [0.0]
                    def _agera5_progress(current, total, detail=''):
                        now = _time.monotonic()
                        is_final = current >= total
                        if self._progress_callback and (is_final or now - _agera5_last_report[0] >= 10):
                            _agera5_last_report[0] = now
                            # V2-22a 1.5 — W3 fallback deleted. After W4
                            # removal in agera5.py, _phase_monitor is the
                            # sole caller and always passes a non-empty
                            # detail. An empty detail here would indicate
                            # a regression worth surfacing, not masking.
                            self._progress_callback.on_substage_progress(
                                'retrieve', 'Downloading AgERA5 temperature',
                                current, total, detail)
                    agera5_result = agera5.retrieve(
                        region=region, start_date=start_date,
                        end_date=end_date, download=True,
                        progress_callback=_agera5_progress,
                        # V2-22b L: use the explicit _cancel_check stored
                        # on self by execute() — decouples prismpy from
                        # prismweb's private `callback._is_cancelled`
                        # method. V2-22c will formalize this.
                        cancel_check=getattr(self, '_cancel_check', None),
                        run_id=_extract_run_id(self._progress_callback),
                    )
                    if agera5_result.success and agera5_result.data:
                        climate_data["agera5_dir"] = agera5_result.data.data_dir
                        climate_data["agera5_variables"] = agera5_result.data.variables
                        self.logger.info(f"AgERA5: {agera5_result.data.variables}")
                        got_data = True
                    else:
                        self.logger.warning(f"AgERA5 download failed: {agera5_result.errors}")
                        # Scan cache for partial files — report what IS on disk
                        # so validation can flag incomplete data honestly.
                        self._report_partial_agera5(cache_dir, region, climate_data)
            except PipelineCancelled:
                # V2-22b L F-4: Gate A's HIGH 1 list cited :924 (TAMSAT
                # branch) but missed this AgERA5-branch site. Without
                # the carve-out, user cancel inside AgERA5 gets logged
                # as a generic "download error" and the pipeline
                # continues into translate. Propagate.
                raise
            except Exception as e:
                self.logger.warning(f"AgERA5 download error: {e}")
                self._report_partial_agera5(cache_dir, region, climate_data)

            if got_data:
                return climate_data

        self.logger.warning("No pre-configured climate data paths found.")
        return None

    def _report_partial_agera5(
        self, cache_dir: Path, region, climate_data: Dict,
    ) -> None:
        """Scan AgERA5 cache for partial files after a failed download.

        Even when download fails (timeout, network error), some files
        may already exist from partial downloads or previous runs.
        Report them so validation can flag incomplete data honestly
        instead of silently ignoring the gap.

        Uses `region_cache_key_from_region(region)` so manual
        regions with bbox-unique cache paths are looked up
        correctly — the previous `normalize_region_name(region.name)`
        key would collide across different manual projects that
        share the `"Unnamed study area"` display name.
        """
        from prismpy.utils.sanitization import region_cache_key_from_region
        safe_name = region_cache_key_from_region(region)
        agera5_cache = cache_dir / "agera5" / f"AgERA5_{safe_name}"
        if not agera5_cache.exists():
            # Mark that AgERA5 was expected but has zero files
            climate_data["agera5_variables"] = {}
            climate_data["agera5_expected"] = True
            return
        var_counts = {}
        for var_dir in agera5_cache.iterdir():
            if var_dir.is_dir():
                count = len(list(var_dir.glob("*.tif")))
                if count:
                    var_counts[var_dir.name] = count
        if var_counts:
            climate_data["agera5_dir"] = agera5_cache
            climate_data["agera5_variables"] = var_counts
            self.logger.warning(
                f"Using partial AgERA5 cache: {var_counts}"
            )
        else:
            climate_data["agera5_variables"] = {}
        climate_data["agera5_expected"] = True

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
        crop_cal = self.config.crop.calendar if self.config.crop else None
        end_date = self.config.temporal.get_climate_end_date(crop_cal)

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

    # V2-19 B3: iSDA Africa coverage bounding box.
    # Wider-with-fallback policy (per crop-modeling-specialist recommendation):
    # includes continental Africa + Madagascar + offshore island states.
    # Regions outside the bbox skip iSDA and fall through to HWSD.
    # Canary Islands (~-17°W, 28°N) are inside the bbox but iSDA has no data
    # there — the per-cell NoData skip at sample_at_points handles that
    # edge case gracefully (empty profiles → HWSD fallback via the outer cascade).
    ISDA_AFRICA_BBOX = (-20.0, -40.0, 55.0, 40.0)  # (minx, miny, maxx, maxy)

    def _region_in_isda_coverage(self, region: Region) -> bool:
        """Check if a region's bounding box intersects iSDA's Africa coverage.

        Returns True if any part of the region bbox overlaps the continental
        Africa bbox (-20°W to 55°E, -40°S to 40°N). Uses axis-aligned bbox
        intersection — a cheap, correct check for convex rectangular regions.
        """
        b = region.bounds
        minx, miny, maxx, maxy = self.ISDA_AFRICA_BBOX
        return (
            b.maxx > minx and b.minx < maxx
            and b.maxy > miny and b.miny < maxy
        )

    def _ensure_isda_1km_cache(
        self,
        prop_name: str,
        target_dir: Path,
    ) -> Optional[Path]:
        """Ensure an iSDA 1km-resampled COG is available locally for ``prop_name``.

        Implements the Tier 1 / Tier 2 portion of the iSDA 3-tier cascade:

          * **Tier 1** — return the existing cache file if it is present and
            non-empty.
          * **Tier 2** — open the S3 COG and read the data at ~1 km via the
            COG overview pyramid, then save it locally with an atomic
            ``tmp → rename`` write.
          * **Tier 3 fallback** — return ``None`` on any failure. The caller
            falls through to direct 30 m per-pixel reads from the same S3 COG.

        The atomic ``tmp → rename`` prevents corrupted cache files if the
        process dies mid-download. Concurrent pipeline runs hitting the same
        cache directory are safe because POSIX rename is atomic — at worst
        the download work is duplicated, never the resulting file.

        The target resolution is computed CRS-aware: geographic CRS use
        ``1/111 ≈ 0.00903°`` (1 km at the equator; at Africa's maximum
        latitude of ±37° one degree shrinks to ~89 km, producing a slightly
        finer-than-1 km grid near the Atlas and Cape regions — acceptable
        for average-resampled soil variables). Projected CRS use 1000 m.

        ``Resampling.average`` is the correct choice for continuous soil
        variables (sand/clay/silt/pH/bulk density/organic carbon). Categorical
        rasters (e.g. SPAM crop masks) would require ``Resampling.nearest``,
        which is out of scope for this helper.

        Args:
            prop_name: iSDA property name (sand_content, clay_content, etc.).
            target_dir: Cache directory — will be created if missing.

        Returns:
            ``Path`` to the 1 km cache file if Tier 1 or Tier 2 succeeded,
            ``None`` if both failed (caller should fall back to Tier 3).
        """
        cache_file = target_dir / f"{prop_name}_1km.tif"

        # Tier 1: existing non-empty cache
        if cache_file.exists() and cache_file.stat().st_size > 0:
            return cache_file

        # Tier 2: download via COG overview pyramid
        try:
            import rasterio
            from rasterio.enums import Resampling
        except ImportError:
            self.logger.warning(
                "rasterio unavailable for iSDA 1km cache download"
            )
            return None

        s3_url = (
            f"https://isdasoil.s3.amazonaws.com/soil_data/"
            f"{prop_name}/{prop_name}.tif"
        )

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.logger.warning(
                f"iSDA cache: cannot create target dir {target_dir}: {exc}"
            )
            return None

        # V2-19b-fix Finding 5: per-process tmp suffix prevents concurrent
        # pipelines from colliding on the same tmp file. Without this,
        # both pipelines download to the same .tif.tmp, and the loser's
        # rename fails with FileNotFoundError when the winner's rename
        # has already moved the tmp file. The per-PID suffix ensures
        # each pipeline writes to its own tmp path.
        import os as _os
        tmp_file = cache_file.with_suffix(f".tif.tmp.{_os.getpid()}")

        try:
            self.logger.info(
                f"Downloading iSDA {prop_name} at ~1km via COG overview from S3..."
            )
            # Set GDAL HTTP timeout for rasterio's network calls.
            # Without this, rasterio.open on an S3 URL can block
            # indefinitely if the server is unresponsive.
            import os as _os2
            _os2.environ.setdefault('GDAL_HTTP_TIMEOUT', '120')
            _os2.environ.setdefault('GDAL_HTTP_CONNECTTIMEOUT', '30')

            with rasterio.open(s3_url) as src:
                # CRS-aware target resolution selection.
                # Geographic (degrees): 1/111 ≈ 0.00903° (~1km at equator;
                # ~89km per degree at Africa's max latitude ±37°, so slightly
                # finer than 1km there — acceptable for average resampling).
                # Projected (meters), e.g. iSDA's native EPSG:3857: 1000 m.
                if src.crs and src.crs.is_geographic:
                    target_res = 1.0 / 111.0
                else:
                    target_res = 1000.0

                native_res = abs(src.transform.a)
                if native_res <= 0:
                    self.logger.warning(
                        f"iSDA {prop_name}: invalid native resolution "
                        f"{native_res}, aborting cache write"
                    )
                    return None

                scale = native_res / target_res
                out_height = max(1, int(src.height * scale))
                out_width = max(1, int(src.width * scale))
                if out_height < 10 or out_width < 10:
                    self.logger.warning(
                        f"iSDA {prop_name}: computed output "
                        f"{out_height}x{out_width} too small, "
                        f"aborting cache write"
                    )
                    return None

                # Resampling.average is correct for continuous soil variables
                # (sand/clay/silt/pH/bulk density/organic carbon).
                data = src.read(
                    out_shape=(src.count, out_height, out_width),
                    resampling=Resampling.average,
                )

                # New transform for the downsampled grid.
                new_transform = src.transform * src.transform.scale(
                    src.width / out_width,
                    src.height / out_height,
                )

                # Atomic write: tmp file then os.replace (atomic on POSIX).
                with rasterio.open(
                    tmp_file, "w",
                    driver="GTiff",
                    height=out_height,
                    width=out_width,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=new_transform,
                    compress="LZW",
                    nodata=src.nodata,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                ) as dst:
                    dst.write(data)

            # os.replace is atomic on POSIX — if another process races us
            # and replaces the cache file first, our replace overwrites
            # theirs (both produced the same data, so either outcome is
            # correct). We prefer os.replace over Path.rename because
            # replace is unconditional on POSIX and handles the race
            # where cache_file already exists from another process.
            _os.replace(str(tmp_file), str(cache_file))
            size_mb = cache_file.stat().st_size // (1024 * 1024)
            self.logger.info(
                f"  Cached iSDA {prop_name} 1km at {cache_file} (~{size_mb} MB)"
            )
            return cache_file

        except Exception as exc:  # noqa: BLE001
            # V2-19b-fix Finding 5: race-loser detection. If our tmp file
            # was cleaned up or renamed out from under us, but the cache
            # file now exists (a concurrent pipeline wrote it), treat
            # this as a Tier 1 hit — the cache is ready, just created by
            # someone else. No phantom FALLBACK_SUBSTITUTION.
            if cache_file.exists() and cache_file.stat().st_size > 0:
                self.logger.info(
                    f"  iSDA {prop_name}: cache file appeared during download "
                    f"(concurrent pipeline won the race), using it"
                )
                # Clean up our own stale tmp file if it survived
                try:
                    tmp_file.unlink(missing_ok=True)
                except OSError:
                    pass
                return cache_file

            self.logger.warning(
                f"iSDA 1km cache unavailable for {prop_name} "
                f"({exc.__class__.__name__}: {exc}), "
                f"falling back to 30m direct reads"
            )
            # Clean up partial tmp file on failure
            try:
                tmp_file.unlink(missing_ok=True)
            except OSError:
                pass
            return None

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

        V2-19 B3: runs whenever the study region's bounding box intersects
        iSDA's Africa coverage (wider-with-fallback bbox). The previous
        platform-based gate (SARRA-Py-only) was a misuse of geography —
        iSDA's limitation is geographic (Africa-only), not platform-specific.
        CRAFT/PYTHIA/ACEA users in Africa now also receive iSDA as the primary
        soil source.

        Args:
            grid: SpatialGrid with cell coordinates
            region: Region for metadata

        Returns:
            Dictionary of cell_id -> SoilProfile, or None
        """
        # V2-19 B3: geographic bbox gate replaces the platform-based gate.
        # Previous code: `if Platform.SARRA_PY not in enabled: return None`
        # was a misuse — iSDA's limitation is geographic, not platform-specific.
        if not self._region_in_isda_coverage(region):
            self.logger.info(
                "Region %s outside iSDA Africa coverage (%s), skipping iSDA retrieval",
                region.name,
                region.bounds.to_gis_format()
                if hasattr(region.bounds, "to_gis_format")
                else "bbox unavailable",
            )
            # V2-19 C12 (ISDA-GATE): geographic gate rationale with AC11
            # positive/negative domain and reviewer pre-answer.
            if self.provenance and self.provenance.enabled:
                self.provenance.record_decision(
                    decision_type=DecisionType.SOURCE_SELECTION,
                    description=(
                        f"iSDA skipped: region {region.name} outside Africa bbox"
                    ),
                    rationale=(
                        "iSDA provides 30-metre soil data for continental Africa "
                        "only (Hengl et al. 2021). The region's bounding box "
                        "does not intersect iSDA's coverage bbox (-20\u00b0W to "
                        "55\u00b0E, -40\u00b0S to 40\u00b0N), so the cascade "
                        "falls through to HWSD v2.0 (1 km global, FAO/IIASA "
                        "2023). Valid gate logic for any non-African study "
                        "region. Wider-with-fallback policy: the bbox includes "
                        "Madagascar and Canary Islands; edge-case NoData cells "
                        "are handled by the per-cell skip in the sample loop. "
                        "NOT applicable for regions that straddle the Africa "
                        "boundary (e.g., Sinai, Canary Islands) — currently "
                        "handled by the same per-cell NoData skip, but a "
                        "mixed iSDA/HWSD cascade per cell is deferred to V2-20."
                    ),
                    alternatives=[
                        "HWSD 1 km (geographic fallback, next in cascade)",
                        "Mixed per-cell iSDA/HWSD cascade (V2-20)",
                    ],
                    reference="prismpy.pipeline.executor._retrieve_isda_api_for_grid",
                    artifact_id="soil",
                )
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

        # V2-19 B3 v2: 3-tier cascade for iSDA data access.
        #   Tier 1 — existing local 1km cache file (fast, no network)
        #   Tier 2 — download 1km via COG overview pyramid + atomic cache write
        #   Tier 3 — direct 30m per-pixel reads from S3 (slow fallback)
        #
        # Each property is resolved independently, so a single property that
        # fails Tier 2 doesn't block the others. Partial failures are recorded
        # as FALLBACK_SUBSTITUTION decisions after the loop.
        if hasattr(self.config.data_sources, 'cache_dir') and self.config.data_sources.cache_dir:
            cache_parent = Path(self.config.data_sources.cache_dir).parent
            cache_target_dir = cache_parent / "isda"
        else:
            cache_target_dir = Path(__file__).resolve().parents[4] / "data" / "isda"

        prop_cache_files: Dict[str, Optional[Path]] = {}
        for prop_name in PROPERTIES:
            prop_cache_files[prop_name] = self._ensure_isda_1km_cache(
                prop_name, cache_target_dir
            )

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

        # V2-19 B3 v2: record cascade outcome as provenance decisions before
        # the per-cell read loop so the decisions land regardless of per-cell
        # success (which only affects profile-count metrics, not source choice).
        if self.provenance and self.provenance.enabled:
            # V2-19 B0 finding #1a: iSDA per-cell sampling uses a
            # window-based read at the exact transformed (x, y) pixel,
            # which is effectively nearest-neighbour (single 1×1 window,
            # no interpolation). Record the implicit choice explicitly.
            self.provenance.record_decision(
                decision_type=DecisionType.RESAMPLING_METHOD,
                description="iSDA point sampling: nearest-neighbour (1x1 window)",
                rationale=(
                    "Cell coordinates are transformed to the raster CRS, "
                    "then src.index(x, y) + a 1x1 Window selects exactly "
                    "one pixel — no bilinear/cubic interpolation. This is "
                    "appropriate for continuous soil variables sampled at "
                    "~1km or 30m where target cell size is comparable to "
                    "native pixel size."
                ),
                alternatives=[
                    "Bilinear interpolation (smoother but not native rasterio)",
                    "Average over a larger window (requires grid cell extent)",
                ],
                reference=(
                    "prismpy.pipeline.executor._retrieve_isda_api_for_grid "
                    "lines ~1373-1380 (rasterio.windows.Window(col, row, 1, 1))"
                ),
                artifact_id="soil",
            )
            missing = [p for p, cf in prop_cache_files.items() if cf is None]
            if not missing:
                self.provenance.record_decision(
                    decision_type=DecisionType.SOURCE_SELECTION,
                    description="Soil source: iSDA 1km resampled from 30m COG",
                    rationale=(
                        "iSDA provides Africa-wide 30m native resolution. "
                        "Resampled to ~1km via rasterio COG overview reads "
                        "(bandwidth-efficient, no full 30m download). "
                        f"Cache location: local iSDA 1km cache"
                    ),
                    alternatives=[
                        "iSDA 30m direct reads (tier 3 fallback, slower)",
                        "HWSD 1km (lower resolution, outer-cascade fallback)",
                    ],
                    reference="prismpy.pipeline.executor._retrieve_isda_api_for_grid",
                    artifact_id="soil",
                )
            else:
                self.provenance.record_decision(
                    decision_type=DecisionType.FALLBACK_SUBSTITUTION,
                    description=(
                        f"iSDA 1km cache failed for "
                        f"{len(missing)}/{len(PROPERTIES)} properties"
                    ),
                    rationale=(
                        f"Network or S3 issue prevented 1km cache download for: "
                        f"{missing}. Falling back to direct 30m per-pixel reads "
                        f"for those properties (slower but scientifically "
                        f"equivalent — same underlying source)."
                    ),
                    alternatives=[
                        "HWSD 1km fallback (outer-cascade, lower resolution)",
                        "Skip properties (unacceptable — missing data)",
                    ],
                    reference="prismpy.pipeline.executor._retrieve_isda_api_for_grid",
                    artifact_id="soil",
                )

        # Read each property from its resolved source (1km cache or S3 30m).
        cell_data = {cell.cell_id: {} for cell in cells}

        try:
            for prop_name, prop_info in PROPERTIES.items():
                cache_file = prop_cache_files.get(prop_name)
                if cache_file is not None:
                    url = str(cache_file)
                    self.logger.info(
                        f"  Reading {prop_name} from 1km cache ({cache_file.name})..."
                    )
                else:
                    url = f"{S3_BASE}/{prop_name}/{prop_name}.tif"
                    self.logger.info(
                        f"  Reading {prop_name} from S3 at 30m (tier 3 fallback)..."
                    )

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

            # V2-19 B0 finding #1b: HWSD per-cell sampling uses
            # rasterio src.sample() inside HWSDSource._sample_bil_raster,
            # which is nearest-neighbour by default (no interpolation
            # keyword argument passed). Record the implicit choice.
            if self.provenance and self.provenance.enabled:
                self.provenance.record_decision(
                    decision_type=DecisionType.RESAMPLING_METHOD,
                    description="HWSD point sampling: nearest-neighbour (rasterio default)",
                    rationale=(
                        "HWSDSource._sample_bil_raster calls "
                        "rasterio.src.sample(xy_coords) without a resampling "
                        "keyword, so rasterio uses its nearest-neighbour "
                        "default. Appropriate for SMU ID lookups (categorical) "
                        "but also used for the per-cell SMU classification "
                        "that drives soil property selection — soil-property "
                        "smoothing between cells is therefore absent."
                    ),
                    alternatives=[
                        "Bilinear on numeric properties (inappropriate for SMU IDs)",
                        "Mode resampling over a window (more robust but slower)",
                    ],
                    reference=(
                        "prismpy.sources.soil.hwsd.HWSDSource._sample_bil_raster "
                        "line ~318 (src.sample(xy_coords))"
                    ),
                    artifact_id="soil",
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
                n_total = len(profiles)
                n_fallback = n_total - n_real
                self.logger.info(
                    f"HWSD: {n_real} real profiles, "
                    f"{n_fallback} defaults, "
                    f"{n_total} total for {len(cell_ids)} cells"
                )

                # V2-19 site #5 + C3 (RH-01): HWSD DEFAULT_SOIL fallback
                # with AC11-compliant rationale (source + domains + pre-answer).
                if self.provenance and self.provenance.enabled and n_fallback > 0:
                    proportion = n_fallback / max(n_total, 1)
                    above_threshold = proportion > 0.05
                    rationale = (
                        f"HWSD DEFAULT_SOIL Sahel-typical profile was "
                        f"substituted for {n_fallback} of {n_total} cells "
                        f"({proportion * 100:.1f}%) where the HWSD BIL raster "
                        f"returned no Soil Mapping Unit (SMU). Fixed values: "
                        f"sand=60%, clay=18%, silt=22%, SOC=0.5%, pH=6.5, "
                        f"BD=1.4 g/cm\u00b3. No source citation in code "
                        f"(comment says 'typical Sahel'). Literature sanity "
                        f"check: Sahelian sandy soils range sand 40-80%, "
                        f"clay 5-25%, SOC 0.1-1.5%, pH 5.5-7.5, BD 1.3-1.7 "
                        f"g/cm\u00b3 (Bationo et al. 2007, Vanlauwe et al. "
                        f"2015). Current defaults sit mid-range for sandy "
                        f"Sahelian soils. Valid for sandy Ferric Lixisols and "
                        f"Arenosols typical of the Sahel (Niger, Mali, "
                        f"Burkina Faso). NOT valid for Sudano-Sahelian "
                        f"clay-loam Vertisols (clay >40%), Guinea savanna "
                        f"Ferralsols (SOC >2%), East African volcanic "
                        f"Andosols (BD <1.0), or Miombo woodland soils "
                        f"(pH <5.0). A reviewer would ask: 'Are these "
                        f"defaults appropriate for my study region?' For "
                        f"regions outside the Sahel sandy belt, override via "
                        f"iSDA 30m source or provide site-specific profiles."
                    )
                    if above_threshold:
                        rationale += (
                            f" \u26a0\ufe0f Above 5% reviewer threshold "
                            f"({proportion * 100:.1f}% cells used defaults) "
                            f"-- document fallback locations explicitly in "
                            f"methods and verify against local soil surveys."
                        )
                    self.provenance.record_decision(
                        decision_type=DecisionType.FALLBACK_SUBSTITUTION,
                        description=(
                            f"HWSD DEFAULT_SOIL fallback: {n_fallback}/{n_total} "
                            f"cells ({proportion * 100:.1f}%)"
                        ),
                        rationale=rationale,
                        alternatives=[
                            "iSDA 30m (higher resolution, preferred for Africa)",
                            "Regional soil-default library by agro-ecological zone (V2-20)",
                            "Interpolation from neighbouring cells (not implemented)",
                        ],
                        reference="prismpy.sources.soil.hwsd.DEFAULT_SOIL (hwsd.py:42-49)",
                        artifact_id="soil",
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
                # V2-19: start grid artifact so grid-creation decisions bind here
                if self.provenance.enabled:
                    self.provenance.start_artifact("grid", artifact_id="grid", stage="harmonize")

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
                # V2-22c-PRE.3.3 (D15) — thread the operator's
                # `region.exclude_cells` through to the SpatialGrid
                # factory. Translators iterate `grid.cells` directly
                # so the prune propagates without per-translator
                # edits per the §6.4 schema-bounds discipline.
                grid = SpatialGrid.from_bounds(
                    region.bounds,
                    resolution="5arcmin",
                    clip_geometry=clip_geometry,
                    exclude_cells=getattr(
                        self.config.region, 'exclude_cells', None,
                    ),
                )
                self.logger.info(f"Created grid with {grid.n_cells} cells")

                # V2-19 site #3: record AGGREGATION_METHOD decision for grid
                # creation. The 5-arcmin resolution is hardcoded (not
                # per-platform configurable) because all downstream
                # platforms either use 5-arcmin directly or map to coarser
                # grids internally at translate time.
                if self.provenance.enabled:
                    self.provenance.record_decision(
                        decision_type=DecisionType.AGGREGATION_METHOD,
                        description=(
                            f"5-arcmin uniform grid ({grid.n_cells} cells) "
                            f"within region bounds"
                        ),
                        rationale=(
                            "5-arcmin is the hardcoded canonical grid resolution "
                            "in prismpy. It maximises boundary precision for "
                            "small regions and is compatible with all target "
                            "platforms. Platforms requiring coarser grids "
                            "(e.g., ACEA 30-arcmin) handle resolution mapping "
                            "internally at translate time."
                        ),
                        alternatives=[
                            "30-arcmin (ACEA native)",
                            "0.5-degree (NASA POWER native)",
                            "Per-platform native resolution (more complex)",
                        ],
                        reference="prismpy.pipeline.executor._execute_harmonize",
                    )
                    self.provenance.record_transformation(
                        operation=OperationType.BUILD_GRID,
                        parameters={
                            "resolution": "5arcmin",
                            "n_cells": grid.n_cells,
                            "clipped": clip_geometry is not None,
                            "bounds": region.bounds.to_gis_format()
                            if hasattr(region.bounds, "to_gis_format") else None,
                        },
                        artifact_id="grid",
                    )

                    # V2-19 B1: effective-resolution warning. Determine which
                    # sources are active for this run from platform defaults,
                    # then check if target 5-arcmin is finer than any native.
                    active_sources: List[str] = []
                    from prismpy.config.schema import Platform
                    enabled_platforms = self.config.get_enabled_platforms()
                    if Platform.SARRA_PY in enabled_platforms:
                        active_sources.extend(["TAMSAT", "AgERA5"])
                    if any(
                        p in enabled_platforms
                        for p in (Platform.CRAFT, Platform.PYTHIA, Platform.ACEA)
                    ):
                        active_sources.append("NASA POWER")
                    # Soil sources — iSDA tried first for African regions,
                    # HWSD is the fallback for non-Africa or when iSDA empty.
                    if region and self._region_in_isda_coverage(region):
                        active_sources.append("iSDA")
                    active_sources.append("HWSD")
                    # De-duplicate while preserving order
                    seen = set()
                    active_sources = [
                        s for s in active_sources
                        if not (s in seen or seen.add(s))
                    ]
                    self._record_effective_resolution_warning(
                        target_resolution_deg=5.0 / 60.0,  # 5 arc-minutes
                        target_resolution_label="5-arcmin (~9 km)",
                        active_sources=active_sources,
                    )

            # Retrieve per-cell soil data: try iSDA API first (Africa, 30m),
            # then HWSD fallback (global, 1km)
            soil_data = retrieved_data.get("soil")
            if grid and region:
                # V2-19b-fix: deleted symptom-suppression hack that manually
                # set `self.provenance._current_artifact_id = "soil"`. All
                # iSDA/HWSD cascade decisions now pass explicit artifact_id=
                # "soil", and the Finding 4 direct-attach fix in tracker.py
                # properly honors explicit artifact_id regardless of the
                # current-artifact pointer's state. Manipulating tracker
                # internals from caller code is an antipattern that masks
                # API contract violations.
                isda_soil = self._retrieve_isda_api_for_grid(grid, region)
                if isda_soil:
                    soil_data = isda_soil
                    # V2-19 site #4: SOURCE_SELECTION for soil cascade
                    if self.provenance.enabled:
                        self.provenance.record_decision(
                            decision_type=DecisionType.SOURCE_SELECTION,
                            description=(
                                f"Soil source: iSDA Africa (30m) — {len(isda_soil)} cells"
                            ),
                            rationale=(
                                "iSDA provides 30-metre soil properties for continental "
                                "Africa, the highest resolution available for the region. "
                                "Used as the primary source when available."
                            ),
                            alternatives=["HWSD v2.0 (1km, global fallback)"],
                            reference="prismpy.sources.soil.isda",
                            artifact_id="soil",
                        )
                else:
                    hwsd_soil = self._retrieve_hwsd_for_grid(grid, region)
                    if hwsd_soil:
                        soil_data = hwsd_soil
                        # V2-22c-PRE.1.10 (D37) cascade-rank update —
                        # HWSD ran as the iSDA fallback path. Each
                        # HWSD-served cell's metadata bumps to
                        # cascade_rank=2 + records the iSDA failure
                        # in fallback_attempts. The cockpit drawer
                        # reads this and renders "iSDA failed, HWSD
                        # fallback used" verbatim per AC-14.3.
                        for cell_id, profile in hwsd_soil.items():
                            if not hasattr(profile, 'metadata'):
                                continue
                            if profile.metadata is None:
                                profile.metadata = {}
                            profile.metadata["cascade_rank"] = 2
                            profile.metadata["fallback_attempts"] = [
                                {
                                    "source": "iSDA Africa",
                                    "reason": "no_data_at_centroid",
                                },
                            ]
                        # V2-19 site #4: SOURCE_SELECTION falling back to HWSD
                        if self.provenance.enabled:
                            self.provenance.record_decision(
                                decision_type=DecisionType.SOURCE_SELECTION,
                                description=(
                                    f"Soil source: HWSD v2.0 (1km, fallback) — "
                                    f"{len(hwsd_soil)} cells"
                                ),
                                rationale=(
                                    "HWSD v2.0 was used as the soil source because "
                                    "iSDA was not available (iSDA retrieval returned "
                                    "no profiles for this region)."
                                ),
                                alternatives=["iSDA Africa (30m, preferred when available)"],
                                reference="prismpy.sources.soil.hwsd",
                                artifact_id="soil",
                            )

                # V2-19: record the soil cascade transformation to flush
                # any pending decisions (including fallback records from
                # HWSD per-cell lookups)
                if self.provenance.enabled:
                    self.provenance.record_transformation(
                        operation=OperationType.RETRIEVE,
                        parameters={
                            "cascade": "iSDA→HWSD",
                            "final_source": (
                                "iSDA" if isda_soil else
                                "HWSD" if soil_data else "none"
                            ),
                            "n_profiles": len(soil_data) if soil_data else 0,
                        },
                        artifact_id="soil",
                    )

            # V2-22c-PRE.1.10 (D37) — backstop default cascade
            # metadata for in-memory climate time series. Source
            # loaders that haven't been ported yet emit ts objects
            # without `metadata.cascade_rank`; the cell-summary
            # read path defaults rank=1 if absent, but populating
            # it here makes the schema explicit on the wire +
            # gives the cockpit a uniform shape.
            climate_data_for_unified = retrieved_data.get("climate")
            if isinstance(climate_data_for_unified, dict):
                for cell_id, ts in climate_data_for_unified.items():
                    if not hasattr(ts, 'metadata'):
                        continue
                    if ts.metadata is None:
                        ts.metadata = {}
                    ts.metadata.setdefault(
                        "source", getattr(ts, 'source', None),
                    )
                    ts.metadata.setdefault("cascade_rank", 1)
                    ts.metadata.setdefault("fallback_attempts", [])
                    # Version is best-effort — loaders that have
                    # been ported populate it; others leave it
                    # as None and the cockpit drawer renders
                    # "version: unknown".
                    ts.metadata.setdefault("version", None)

            unified_data = UnifiedData(
                region=region,
                grid=grid,
                climate=climate_data_for_unified,
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
                # Pass progress callback to translator for substage reporting
                translator.progress_callback = getattr(self, '_progress_callback', None)
                # V2-22b L: thread the pipeline-level cancel_check to
                # the translator so its per-cell NASA POWER loops can
                # cooperatively cancel. Attribute-assignment mirrors
                # the progress_callback pattern above and avoids
                # invalidating BaseTranslator.translate()'s signature.
                translator.cancel_check = getattr(self, '_cancel_check', None)

                # V2-19: start a dedicated artifact for this platform's translation
                # output. This gives the FORMAT_CHOICE decision emitted by the
                # translator (translators/*/translator.py:record_decision) a home,
                # and the record_transformation(TRANSLATE) flush below drains it.
                output_artifact = f"output_{platform.value}"
                if self.provenance.enabled:
                    self.provenance.start_artifact(
                        artifact_type=output_artifact,
                        artifact_id=output_artifact,
                        stage="translate",
                    )

                try:
                    result = translator.translate(unified_data)
                    results[platform.value] = result

                    # V2-19: explicit TRANSLATE transformation flushes pending
                    # decisions (including the translator's FORMAT_CHOICE call)
                    # into this output artifact's lineage.
                    if self.provenance.enabled:
                        self.provenance.record_transformation(
                            operation=OperationType.TRANSLATE,
                            parameters={
                                "platform": platform.value,
                                "region": self.config.region.name,
                                "output_files": len(result.output_files)
                                if hasattr(result, "output_files") else 0,
                                "success": result.success,
                            },
                            artifact_id=output_artifact,
                        )
                except PipelineCancelled:
                    # V2-22b L: per-platform translate broad except must
                    # not rewrite cancel as a translation error result;
                    # let the pipeline.execute boundary handle it.
                    raise
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
                    # V2-19: still flush pending decisions on failure
                    if self.provenance.enabled:
                        try:
                            self.provenance.record_transformation(
                                operation=OperationType.TRANSLATE,
                                parameters={
                                    "platform": platform.value,
                                    "region": self.config.region.name,
                                    "error": str(e),
                                    "success": False,
                                },
                                artifact_id=output_artifact,
                            )
                        except Exception:
                            pass  # never crash on provenance errors
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

    def _execute_remediation(
        self,
        translation_results: Dict[str, "TranslationResult"],
        unified_data,
    ) -> StageResult:
        """V2-22c-PRE.4.2 — apply ``remediation_spec`` from config to
        translation outputs.

        No-op when ``self.config`` does not carry a
        ``remediation_spec`` (most runs: originals, retries). Only
        remediation re-runs (where the prismweb layer wrote
        ``remediation_spec`` into ``PipelineRun.remediation_spec`` and
        threaded it into the prismpy config) perform work.

        PRE.4 ships scaffolding + Veto #4 server enforcement (D29
        Layer 2). The full applier for impute / substitute / override
        classes is V2-22c R5 builder work; this method only enforces
        the structural invariants and stubs the day-level exclusion
        / substitution / override paths with deferred-to-R5 warnings.

        Critical: the ``if tier == 'block': raise`` clause runs
        UNCONDITIONALLY before any ``veto_4_acknowledged`` flag check
        per evaluator §12.4 — adversarial bypass with
        ``veto_4_acknowledged: true`` for a BLOCK case CANNOT defeat
        server enforcement. Tests #5/#6/#7/#10 in the 10-test matrix
        guard this invariant.
        """
        from prismpy.pipeline._remediation import (
            RemediationBlocked,
            _veto_4_tier,
            _block_message,
        )

        start_time = datetime.now()

        spec = getattr(self.config, 'remediation_spec', None)
        if spec is None or not isinstance(spec, dict):
            # No-op path — most common case (originals, retries).
            return StageResult(
                stage=PipelineStage.REMEDIATION,
                success=True,
                data={},
                errors=[],
                warnings=[],
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

        errors: List[str] = []
        warnings: List[str] = []

        # Day-level exclusion stub — full implementation R5. Cell-
        # level exclusions are already handled at PRE.3
        # ``SpatialGrid.from_bounds`` via ``RegionConfig.exclude_cells``;
        # this branch covers the day-level slice within the spec.
        exclusions = spec.get('exclusions', {}) or {}
        excluded_days = exclusions.get('days', []) or []
        if excluded_days:
            warnings.append(
                f"day-level exclusion of {len(excluded_days)} days "
                "deferred to R5"
            )

        # Veto #4 server enforcement on the imputations array (D29
        # Layer 2 + D33 conservative thresholds). This is the
        # load-bearing audit-locus per the contract.
        imputations = spec.get('imputations', []) or []
        for imp in imputations:
            tier, block_reason = _veto_4_tier(imp, unified_data)

            # BLOCK is non-acknowledgeable per D33 — adversarial
            # bypass via veto_4_acknowledged: true MUST NOT defeat
            # server enforcement. Tests #5/#6/#7 catch the regression
            # where a refactor inverts this clause and the
            # acknowledgment check.
            if tier == 'block':
                raise RemediationBlocked(
                    cell_id=imp.get('cell_id'),
                    reason=block_reason,
                    message=_block_message(imp, block_reason),
                )

            # WARN is acknowledgeable per D33 — user-acknowledgment
            # trail honored. Test #4 covers the unacknowledged WARN
            # path; test #3 covers the acknowledged WARN pass-through.
            if tier == 'warn' and not imp.get('veto_4_acknowledged', False):
                raise RemediationBlocked(
                    cell_id=imp.get('cell_id'),
                    reason=RemediationBlocked.REASON_WARN_UNACKED,
                    message=(
                        f"Imputation crosses some soil-class boundaries "
                        f"(cell={imp.get('cell_id')}); user did not "
                        f"acknowledge Veto #4 client-side; server blocks "
                        f"per D29 Layer 2."
                    ),
                )
            # silent OR (warn + acknowledged): allow through.

        # Substitutions + overrides scaffolding only — full applier R5.
        substitutions = spec.get('substitutions', []) or []
        if substitutions:
            warnings.append(
                f"{len(substitutions)} substitutions deferred to R5"
            )
        overrides = spec.get('overrides', []) or []
        if overrides:
            warnings.append(f"{len(overrides)} overrides deferred to R5")

        return StageResult(
            stage=PipelineStage.REMEDIATION,
            success=True,
            data={
                'spec_applied_summary': {
                    'exclusions_cells': len(exclusions.get('cells', []) or []),
                    'exclusions_days': len(excluded_days),
                    'imputations_validated': len(imputations),
                    'substitutions_deferred': len(substitutions),
                    'overrides_deferred': len(overrides),
                },
            },
            errors=errors,
            warnings=warnings,
            duration_seconds=(datetime.now() - start_time).total_seconds(),
        )

    def _execute_validate(
        self,
        translation_results: Dict[str, TranslationResult],
        unified_data: Optional["UnifiedData"] = None,
    ) -> StageResult:
        """Execute the VALIDATE stage.

        Runs two validation layers:
        1. Platform-specific format validation (existing BaseValidator hierarchy)
        2. Scientific data quality checks (V2-19 Phase 2a — 6 Tier 1 checks
           from manuscript Section 2.5)

        Args:
            translation_results: Results from TRANSLATE stage
            unified_data: Harmonized data for scientific validation

        Returns:
            StageResult with validation summary including scientific checks
        """
        start_time = datetime.now()
        self.logger.info("Stage 4: VALIDATE - Checking outputs")

        errors = []
        warnings = []
        validation_summary = {}

        # Layer 1: Platform-specific format validation
        for platform_name, result in translation_results.items():
            if not result.success:
                errors.extend([f"{platform_name}: {e}" for e in result.errors])
                continue

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

        # Layer 2: Scientific data quality checks (V2-19 Phase 2a).
        # Runs the scientific validator and stores the raw report.
        # Surfacing (CLI warnings, logger.info, provenance) is
        # deferred to AFTER Layer 3's post-translate merge + gate, so
        # every output channel observes the final escalated state —
        # not the pre-merge snapshot. (V2-22b/P.2 codex self-check
        # MEDIUM: emitting warnings/log/provenance before the
        # delegation gate runs caused operators to see a clean
        # scientific summary in CLI + provenance while the final
        # JSON report was a warning.)
        scientific_report = None
        post_translate_report = None
        # V2-22c-PRE.2.1 + 2.3 + 2.4 — when SARRA-Py is enabled,
        # sample per-cell climate values from the translated
        # GeoTIFFs once at validate-stage entry. The sampled dict
        # feeds (a) the value-range synthesis in
        # ``run_scientific_validation`` and (b) the cell-summary
        # ``has_climate`` computation in the packaging stage.
        # Stash on ``unified_data.metadata`` so the packaging stage
        # can read it without re-sampling.
        sarra_climate_per_cell: Optional[Dict[int, Dict[str, list]]] = None
        if unified_data:
            try:
                enabled_platforms = [
                    p.value for p in self.config.get_enabled_platforms()
                ]
                # V2-22c-PRE codex P2 #4 — gate the sampler on
                # SARRA-Py translation SUCCESS, not just enabled.
                # When SARRA-Py is enabled but its translation failed,
                # `base_dir/sarra_py` may still exist and contain
                # stale or partial GeoTIFFs from a previous successful
                # run. Reading those would feed false-positive per-cell
                # climate values into validation + the cockpit's
                # `has_climate` rendering. R6-class regression risk.
                # Source the directory from the actual translation
                # result so a failed run can't shadow the sampler.
                sarra_translation_result = (
                    translation_results.get('sarra_py')
                    if isinstance(translation_results, dict)
                    else None
                )
                if (
                    "sarra_py" in enabled_platforms
                    and unified_data.grid is not None
                    and getattr(unified_data.grid, 'cells', None)
                    and sarra_translation_result is not None
                    and getattr(sarra_translation_result, 'success', False)
                ):
                    sarra_dir = Path(getattr(
                        sarra_translation_result, 'output_dir',
                        Path(self.config.output.base_dir) / 'sarra_py',
                    ))
                    if sarra_dir.is_dir():
                        from prismpy.validators.post_translate import (
                            sample_sarra_py_per_cell,
                        )
                        sarra_climate_per_cell = sample_sarra_py_per_cell(
                            sarra_dir, unified_data.grid.cells,
                        )
                        if sarra_climate_per_cell:
                            # Stash on metadata so the packaging
                            # stage's _build_cell_summary call can
                            # read it without re-sampling.
                            if unified_data.metadata is None:
                                unified_data.metadata = {}
                            unified_data.metadata[
                                "_sarra_climate_per_cell"
                            ] = sarra_climate_per_cell
            except Exception as e:
                self.logger.warning(
                    f"PRE.2 SARRA-Py sampling failed (continuing without "
                    f"per-cell climate): {e}"
                )
                sarra_climate_per_cell = None

            try:
                from prismpy.validators.scientific import run_scientific_validation

                scientific_report = run_scientific_validation(
                    unified_data, self.config, enabled_platforms,
                    sarra_climate_per_cell=sarra_climate_per_cell,
                )
                validation_summary["scientific"] = scientific_report
            except Exception as e:
                self.logger.warning(f"Scientific validation failed: {e}")
                warnings.append(f"Scientific validation skipped: {e}")

        # Layer 3: Post-translate climate validation (V2-20).
        # Validates ACTUAL per-cell weather data from translator output
        # files (.WTH for PYTHIA, .pckl for ACEA). Replaces the V2-19
        # placeholder-based checks with real-data validation.
        try:
            from prismpy.validators.post_translate import (
                run_post_translate_validation,
            )

            base_dir = Path(self.config.output.base_dir)
            post_translate_report = run_post_translate_validation(
                translation_results, base_dir
            )

            # Merge post-translate checks into the scientific report's
            # 5-category structure (UX-expert item 5). The scientific
            # vs post-translate split is an implementation concern;
            # the user sees one unified validation report.
            if "scientific" in validation_summary:
                sci = validation_summary["scientific"]
                from prismpy.validators.scientific import _get_check_category
                for check in post_translate_report.get("checks", []):
                    # Apply items 6-8 to post-translate checks
                    result = check.get("result", "pass")
                    check["passed"] = result in ("pass", "info")
                    cat = _get_check_category(check.get("check", ""))
                    check["category"] = cat
                    details = check.get("details", {})
                    if "unit" in details:
                        check["unit"] = details["unit"]
                    # Merge into category
                    if "categories" in sci and cat in sci["categories"]:
                        sci["categories"][cat]["checks"].append(check)
                        if result == "fail":
                            sci["categories"][cat]["passed"] = False
                    # Also add to flat checks list
                    sci.setdefault("checks", []).append(check)
                # Update counts
                all_checks = sci.get("checks", [])
                sci["n_checks"] = len(all_checks)
                results_all = [c.get("result", "pass") for c in all_checks]
                sci["n_pass"] = sum(1 for r in results_all if r == "pass")
                sci["n_warning"] = sum(1 for r in results_all if r == "warning")
                sci["n_fail"] = sum(1 for r in results_all if r == "fail")
                if "fail" in results_all:
                    sci["overall_result"] = "fail"
                    sci["passed"] = False
                elif "warning" in results_all:
                    sci["overall_result"] = "warning"
                cats = sci.get("categories", {})
                sci["categories_passed"] = sum(
                    1 for c in cats.values() if c.get("passed", True)
                )
            else:
                validation_summary["post_translate"] = post_translate_report
        except Exception as e:
            self.logger.warning(f"Post-translate validation failed: {e}")
            warnings.append(f"Post-translate validation skipped: {e}")
            post_translate_report = None

        # Gate runs unconditionally — covers three paths equally:
        # (1) post-translate succeeded + merged: gate either no-ops
        #     (delegated records present) or escalates (they aren't).
        # (2) post-translate raised BEFORE merging: gate sees no
        #     delegated records and escalates the info. This is the
        #     exact failure path the gate is meant to expose, and
        #     was previously bypassed by the try/except envelope.
        # (3) scientific validator failed: `final_sci` is None and
        #     the gate no-ops against its `sci.get("checks", [])`
        #     defensive read.
        # After escalation, totals in `sci` may drift from the
        # pre-gate counts (one info → warning shifts n_warning up
        # and the overall_result). Recompute here so every channel
        # surfaces the gated state.
        final_sci = validation_summary.get("scientific")
        if final_sci is not None:
            sarra_py_enabled = any(
                p == Platform.SARRA_PY
                for p in self.config.get_enabled_platforms()
            )
            _gate_value_range_climate_delegation(
                final_sci, sarra_py_enabled=sarra_py_enabled,
            )
            all_checks = final_sci.get("checks", [])
            final_sci["n_checks"] = len(all_checks)
            results_all = [c.get("result", "pass") for c in all_checks]
            final_sci["n_pass"] = sum(
                1 for r in results_all if r == "pass"
            )
            final_sci["n_warning"] = sum(
                1 for r in results_all if r == "warning"
            )
            final_sci["n_fail"] = sum(
                1 for r in results_all if r == "fail"
            )
            if "fail" in results_all:
                final_sci["overall_result"] = "fail"
                final_sci["passed"] = False
            elif "warning" in results_all:
                final_sci["overall_result"] = "warning"

        # Post-merge surfacing: emit CLI warnings, logger info, and
        # provenance from the FINAL merged+gated state so every
        # output channel agrees. When the merge landed, iterate
        # over `sci["checks"]` (includes post-translate, escalated
        # value_range_climate, etc.). When the merge didn't land
        # (scientific validation failed, or post-translate had no
        # scientific to merge into), fall back to the raw reports.
        if final_sci is not None:
            # Surface fail/warning-level checks as pipeline WARNINGS
            # (not errors). Validation is a REPORTING mechanism, not
            # a GATE — users receive their package alongside the
            # validation report showing what failed. Pipeline errors
            # are reserved for actual pipeline failures (data
            # retrieval crash, translation failure).
            for check in final_sci.get("checks", []):
                label = (
                    "Post-translate validation"
                    if check.get("check", "").startswith("post_translate_")
                    else "Scientific validation"
                )
                result = check.get("result")
                if result == "fail":
                    warnings.append(f"{label} FAIL: {check['summary']}")
                elif result == "warning":
                    warnings.append(f"{label}: {check['summary']}")

            self.logger.info(
                f"  Scientific validation: "
                f"{final_sci.get('overall_result', 'unknown')} "
                f"({final_sci.get('n_pass', 0)} pass, "
                f"{final_sci.get('n_warning', 0)} warn, "
                f"{final_sci.get('n_fail', 0)} fail)"
            )

            # Wire the FINAL merged result into provenance so the
            # artifact record matches the JSON report an auditor
            # downloads. Recording pre-merge state here would leave
            # provenance claiming "PASS" while the report says
            # "WARNING" — exactly the inconsistency codex flagged.
            if self.provenance and self.provenance.enabled:
                self.provenance.start_artifact(
                    "validation", artifact_id="validation",
                    stage="validate",
                )
                self.provenance.record_transformation(
                    operation=OperationType.VALIDATE,
                    parameters={
                        "n_checks": final_sci.get("n_checks", 0),
                        "overall_result": final_sci.get(
                            "overall_result", "unknown"
                        ),
                        "n_pass": final_sci.get("n_pass", 0),
                        "n_warning": final_sci.get("n_warning", 0),
                        "n_fail": final_sci.get("n_fail", 0),
                    },
                    artifact_id="validation",
                )
                # Record a QUALITY_CHECK decision summarizing the outcome
                self.provenance.record_decision(
                    decision_type=DecisionType.QUALITY_CHECK,
                    description=(
                        f"Scientific validation: "
                        f"{final_sci.get('overall_result', 'unknown').upper()} "
                        f"({final_sci.get('n_checks', 0)} checks)"
                    ),
                    rationale=(
                        f"6 automated quality checks covering temporal "
                        f"completeness, cross-variable consistency, value "
                        f"ranges, soil completeness, format compliance, "
                        f"and spatial/temporal coverage. "
                        f"Result: {final_sci.get('n_pass', 0)} pass, "
                        f"{final_sci.get('n_warning', 0)} warning, "
                        f"{final_sci.get('n_fail', 0)} fail."
                    ),
                    alternatives=[
                        "Skip validation (not recommended for research use)",
                        "Region-specific thresholds (planned enhancement)",
                    ],
                    reference="prismpy.validators.scientific.run_scientific_validation",
                    artifact_id="validation",
                )
        elif post_translate_report is not None:
            # No scientific report (validator failed earlier) — but
            # post-translate ran solo. Surface its checks on their
            # own so the user still sees the signal.
            for check in post_translate_report.get("checks", []):
                if check["result"] == "fail":
                    warnings.append(
                        f"Post-translate validation FAIL: {check['summary']}"
                    )
                elif check["result"] == "warning":
                    warnings.append(
                        f"Post-translate validation: {check['summary']}"
                    )

        if post_translate_report is not None:
            self.logger.info(
                f"  Post-translate validation: "
                f"{post_translate_report['overall_result']} "
                f"({post_translate_report['n_checks']} checks)"
            )

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
        validate_result: Optional[StageResult] = None,
    ) -> StageResult:
        """Execute the PACKAGE stage.

        Generates self-documenting data packages for each platform
        (manifest, provenance, README, validation_report) and saves
        the pipeline-level provenance audit trail.

        Args:
            unified_data: Harmonized data from HARMONIZE stage
            translation_results: Results from TRANSLATE stage
            validate_result: Results from VALIDATE stage (includes
                scientific validation report for ZIP inclusion)

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

        # V2-19b-fix Finding 1 (HYBRID): save canonical provenance at the
        # run-level output path, then copy to each platform's subdir.
        # Single JSON serialization + N cheap file copies.
        import shutil
        try:
            # Canonical run-level save — NOT cwd-relative
            base_dir = Path(self.config.output.base_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            canonical_rich = base_dir / "_pipeline_provenance.json"
            provenance_path = self.provenance.save(output_path=canonical_rich)
            # save() writes both rich + stages files side-by-side:
            #   _pipeline_provenance.json  (rich System A)
            #   _pipeline_provenance_stages.json  (auto-derived System B compat)
            canonical_stages = canonical_rich.with_name(
                canonical_rich.stem + "_stages" + canonical_rich.suffix
            )

            # Per-platform distribution — copy provenance + validation
            # into each enabled platform's output directory. These files
            # end up in the user-downloadable ZIP, making the package
            # self-contained for data quality auditing.
            for platform_name in translation_results:
                platform_dir = base_dir / platform_name
                if platform_dir.is_dir():
                    # provenance.json in the platform dir is now System A
                    # (replaces the legacy System B template that
                    # packaging/provenance.py wrote pre-V2-19). V2-20
                    # Day 1 (Task #56) will remove the legacy writer.
                    shutil.copy2(
                        canonical_rich,
                        platform_dir / "provenance.json",
                    )
                    shutil.copy2(
                        canonical_stages,
                        platform_dir / "provenance_stages.json",
                    )

                    # V2-19 Phase 2b: save validation_report.json into
                    # each platform dir so the ZIP is self-contained.
                    # Researchers can audit data quality without visiting
                    # the webapp.
                    if validate_result and validate_result.data:
                        import json as _json
                        val_data = validate_result.data
                        val_report_path = platform_dir / "validation_report.json"
                        try:
                            with open(val_report_path, "w", encoding="utf-8") as vf:
                                _json.dump(val_data, vf, indent=2, default=str, ensure_ascii=False)
                            self.logger.info(
                                f"  {platform_name}: validation_report.json saved"
                            )
                        except Exception as ve:
                            self.logger.warning(
                                f"  {platform_name}: validation_report.json "
                                f"save failed: {ve}"
                            )

                    # V2-21 C-group: generate cell_summary.json for the
                    # interactive map. Per-cell metadata enables Leaflet
                    # to color cells by validation status + show tooltips.
                    # V2-22c-PRE.1.2 / 1.8 — thread the validation_report
                    # through so _build_cell_summary can pivot the
                    # per-check `affected_cells` lists into per-cell
                    # `failed_checks` arrays + flatten the per-violation
                    # context into the top-level `cell_failed_check_details`.
                    if unified_data and hasattr(unified_data, 'grid') and unified_data.grid:
                        try:
                            # V2-22c-PRE codex P1 #1 — `validate_result.data`
                            # is the validation_summary envelope
                            # (`{'scientific': {...}, 'post_translate': {...}}`),
                            # NOT the raw report. `_build_cell_summary` reads
                            # `validation_report.get('checks', [])` so passing
                            # the envelope produces empty `failed_checks` even
                            # when the scientific report has failing checks.
                            # Extract the merged scientific report (the
                            # validate stage merges post_translate's per-cell
                            # checks into this) so the pivot fires correctly.
                            _val_data = (
                                validate_result.data
                                if validate_result and validate_result.data
                                else None
                            )
                            _val_report_for_cells = (
                                _val_data.get('scientific')
                                if isinstance(_val_data, dict)
                                else None
                            )
                            # V2-22c-PRE.2.3 — when the validate stage
                            # ran the SARRA-Py per-cell sampler, the
                            # sampled values are stashed on
                            # ``unified_data.metadata`` so packaging
                            # can read them without re-sampling.
                            _sarra_per_cell = None
                            if (
                                unified_data.metadata
                                and isinstance(unified_data.metadata, dict)
                            ):
                                _sarra_per_cell = unified_data.metadata.get(
                                    "_sarra_climate_per_cell",
                                )
                            cell_summary = self._build_cell_summary(
                                unified_data, _val_report_for_cells,
                                sarra_climate_per_cell=_sarra_per_cell,
                            )
                            cs_path = platform_dir / "cell_summary.json"
                            with open(cs_path, "w", encoding="utf-8") as csf:
                                _json.dump(cell_summary, csf, indent=2,
                                           default=str, ensure_ascii=False)
                            self.logger.info(
                                f"  {platform_name}: cell_summary.json saved "
                                f"({len(cell_summary.get('cells', []))} cells)"
                            )
                        except Exception as cse:
                            self.logger.warning(
                                f"  {platform_name}: cell_summary.json "
                                f"save failed: {cse}"
                            )

                    self.logger.info(
                        f"  {platform_name}: package files distributed"
                    )

            report = self.provenance.get_report()
            report_path = canonical_rich.parent / f"{self.provenance.session_id}_report.txt"
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

    # V2-22c-PRE.1.2 (D34) — per-prefix → category mapping for the
    # per-cell `failed_checks` pivot. Keys are check_id prefixes;
    # values are the cockpit's left-rail dimension-toggle category
    # enum. Tuple-of-pairs preserves match order (longest-prefix-first
    # discipline isn't needed today since the prefixes don't nest, but
    # the ordered shape is robust to future additions).
    #
    # The cockpit's left-rail dimension toggle (D21) reads `category`
    # to project per-dimension status; bare `check_id` strings would
    # require the cockpit to recompute the prefix→category mapping
    # client-side, duplicating the schema discipline. Per §6.4
    # schema-bounds-match-strictest-downstream-consumer, the
    # validator-side projection is canonical.
    _CATEGORY_FROM_PREFIX = (
        ("value_range_", "value_range"),
        ("cross_variable_consistency", "cross_variable"),
        ("temporal_completeness", "temporal"),
        ("soil_completeness_", "soil_completeness"),
        ("region_specific_bounds", "region_specific_bounds"),
        ("coverage_climate_cells", "coverage_per_cell"),
        ("coverage_soil_cells", "coverage_per_cell"),
    )

    # Scopes that the validator emits with per-cell semantics. Region-
    # level checks (`format_compliance`, `spatial_temporal_coverage`,
    # `region_specific_bounds` when scope=global) carry scope='global'
    # and are excluded from the per-cell pivot — the cockpit's
    # Region-Level Status Banner consumes them via validation_report
    # directly per Appendix H two-zone rendering.
    _PER_CELL_SCOPES = frozenset({"per_cell", "per_record", "per_layer"})

    @classmethod
    def _category_for_check_id(cls, check_id: str) -> Optional[str]:
        """Return the per-cell dimension-toggle category for a
        check_id, or None if the check is not per-cell-scoped."""
        for prefix, category in cls._CATEGORY_FROM_PREFIX:
            if check_id.startswith(prefix):
                return category
        return None

    @staticmethod
    def _affected_cell_ids(affected) -> set:
        """Coerce a heterogeneous `details.affected_cells` list into a
        flat set of cell_ids. PRE.1.4/1.5 emit `(cell_id, layer_idx)`
        tuples; PRE.1.7 emits bare cell_ids; the per-cell pivot is
        cell-id only, so layer_idx is discarded here. Duck-typed so a
        future check that emits a different tuple shape still surfaces
        the cell_id correctly."""
        out = set()
        for entry in affected or []:
            if isinstance(entry, (list, tuple)):
                if len(entry) >= 1:
                    out.add(entry[0])
            else:
                out.add(entry)
        return out

    def _build_cell_summary(
        self,
        unified_data,
        validation_report: Optional[Dict[str, Any]] = None,
        sarra_climate_per_cell: Optional[Dict[int, Dict[str, list]]] = None,
    ) -> Dict[str, Any]:
        """Build per-cell summary for the interactive map.

        Gathers validation-relevant metadata per grid cell:
        - Coordinates from the spatial grid
        - Soil source from SoilProfile.source
        - Climate ranges from per-cell time series (if available)
        - Validation status derived from soil source + data availability
        - V2-22c-PRE.1.2: per-cell `failed_checks` array pivoted from
          ``validation_report.checks`` (when provided)
        - V2-22c-PRE.1.8: top-level `cell_failed_check_details` array
          flattened from ``validation_report.checks[i].details.
          violation_details`` (when provided)

        ``validation_report`` is optional so callers that don't have a
        post-validate report (legacy CLI smoke, mid-pipeline previews,
        unit tests bypassing __init__) still get a well-formed dict.
        When omitted, every cell carries `failed_checks: []` and the
        top-level `cell_failed_check_details` list is empty.

        Returns:
            Dict with 'cells' list + 'resolution' + 'n_cells' +
            'cell_summary_version' + 'cell_failed_check_details'
        """
        grid = unified_data.grid
        soil = unified_data.soil or {}
        climate = unified_data.climate or {}

        cells = []
        for cell in grid.cells:
            cid = cell.cell_id
            cell_data = {
                "id": cid,
                "lat": round(cell.lat, 6),
                "lon": round(cell.lon, 6),
                # V2-22c-PRE.1.2 — initialize empty list per evaluator
                # §12.2 binding: cells with no failures emit
                # `failed_checks: []` (empty list, NOT absent key).
                # The pivot below appends entries for every per-cell-
                # scoped check this cell shows up on.
                "failed_checks": [],
            }

            # Soil source
            profile = soil.get(cid)
            if profile and hasattr(profile, 'source'):
                cell_data["soil_source"] = profile.source
                # Flag DEFAULT_SOIL as warning
                is_default = (
                    profile.metadata.get('is_default', False)
                    if hasattr(profile, 'metadata') and profile.metadata
                    else False
                )
                cell_data["soil_default"] = is_default
                # V2-22c-PRE.1.6 (D26) — emit per-cell soil texture class
                # via the existing SoilProfile.surface_texture USDA-triangle
                # classifier. Reuses the property at models/soil.py:150-154
                # — no new classifier.
                #
                # Evaluator §2 / §12 numeric criterion: the key MUST exist
                # on every cell, with `None` (JSON null) for the no-layers
                # / DEFAULT_SOIL edge cases. The cockpit's Veto #4 client
                # preflight reads `cellSummary.cells[X].soil_class` and
                # depends on a deterministic null vs string — `undefined`
                # from key elision would force the JS into a tri-state
                # (string | null | undefined) and silently disable the
                # cross-class block on no-soil cells.
                cell_data["soil_class"] = getattr(profile, 'surface_texture', None)
            else:
                cell_data["soil_source"] = "none"
                cell_data["soil_default"] = False
                # Same null-not-elide discipline when no profile exists at
                # all — uniform consumer interface across all 3 paths
                # (valid profile / empty layers / no profile).
                cell_data["soil_class"] = None

            # V2-22c-PRE.1.10 (D37) — per-cell cascade-provenance
            # projection. Reads each source's metadata field and emits
            # the cockpit-consumable shape under `cell.sources.{climate,
            # soil}.{name, version, cascade_rank, fallback_attempts}`.
            #
            # Source loaders + cascade orchestrator populate these
            # metadata keys; this read path defaults to `cascade_rank=1`
            # and `fallback_attempts=[]` when the orchestrator hasn't
            # threaded through (e.g., legacy fixtures, mid-pipeline
            # previews). The cockpit drawer (AC-14.3) renders the
            # cascade as "Climate: AgERA5 v2.0 (rank 1 of 2 — iSDA
            # failed, HWSD fallback used)" — needs all four fields
            # available, with deterministic defaults so the rank-1
            # primary-success case still surfaces a sensible string.
            #
            # Per D37 contract: the field is elided per cascade-class
            # when no source emitted profile/ts for the cell — the
            # cell's missing-data state is surfaced via PRE.1.9
            # coverage checks, not via a half-populated `sources`.
            sources_block: Dict[str, Any] = {}
            if profile is not None and hasattr(profile, 'source'):
                meta = getattr(profile, 'metadata', None) or {}
                sources_block["soil"] = {
                    "name": meta.get("source", profile.source),
                    "version": meta.get("version"),
                    "cascade_rank": meta.get("cascade_rank", 1),
                    "fallback_attempts": meta.get("fallback_attempts", []),
                }

            # Climate data (per-cell time series if available)
            ts = climate.get(cid)
            if ts and hasattr(ts, 'metadata'):
                ts_meta = ts.metadata or {}
                sources_block["climate"] = {
                    "name": ts_meta.get("source", getattr(ts, 'source', None)),
                    "version": ts_meta.get("version"),
                    "cascade_rank": ts_meta.get("cascade_rank", 1),
                    "fallback_attempts": ts_meta.get("fallback_attempts", []),
                }
            if sources_block:
                cell_data["sources"] = sources_block

            # Climate data (per-cell time series if available)
            if ts and hasattr(ts, 'records') and ts.records:
                tmax_vals = [r.tmax for r in ts.records if r.tmax is not None]
                tmin_vals = [r.tmin for r in ts.records if r.tmin is not None]
                if tmax_vals:
                    cell_data["tmax_range"] = [
                        round(min(tmax_vals), 1),
                        round(max(tmax_vals), 1),
                    ]
                if tmin_vals:
                    cell_data["tmin_range"] = [
                        round(min(tmin_vals), 1),
                        round(max(tmin_vals), 1),
                    ]
                cell_data["n_days"] = len(ts.records)
                cell_data["has_climate"] = True
            elif (
                sarra_climate_per_cell
                and cid in sarra_climate_per_cell
                and sarra_climate_per_cell[cid]
            ):
                # V2-22c-PRE.2.3 (D14) — SARRA-Py per-cell `has_climate`
                # derives from sampled-values presence, not from
                # `unified_data.climate` (which is path-dict shaped for
                # SARRA-Py, so the existing `hasattr(ts, 'records')`
                # check returns False for every cell — the universal-
                # fail bug per V2-22c contract §1.1 sample run
                # `766c6907-...`).
                #
                # Bucket shape: {var_canonical: [pixel_value_per_day]}.
                # Treat as has_climate=True when at least one variable
                # has at least one sampled value across the 4 mapped
                # vars (rain / tmax / tmin / srad).
                bucket = sarra_climate_per_cell[cid]
                tmax_vals = bucket.get("tmax", [])
                tmin_vals = bucket.get("tmin", [])
                if tmax_vals:
                    cell_data["tmax_range"] = [
                        round(min(tmax_vals), 1),
                        round(max(tmax_vals), 1),
                    ]
                if tmin_vals:
                    cell_data["tmin_range"] = [
                        round(min(tmin_vals), 1),
                        round(max(tmin_vals), 1),
                    ]
                # Aggregate sample count across all 4 vars — matches
                # the existing CRAFT/PYTHIA path where n_days is the
                # number of records (here it's number of sampled
                # pixel reads).
                total_samples = sum(len(v) for v in bucket.values())
                cell_data["n_days"] = total_samples
                cell_data["has_climate"] = bool(total_samples > 0)
            else:
                cell_data["has_climate"] = False

            # Validation status: pass / warning / fail
            warnings = []
            if cell_data.get("soil_default"):
                warnings.append("default_soil")
            if cell_data.get("soil_source") == "none":
                warnings.append("no_soil")
            if not cell_data.get("has_climate"):
                warnings.append("no_climate")

            if "no_soil" in warnings or "no_climate" in warnings:
                cell_data["validation_status"] = "fail"
            elif warnings:
                cell_data["validation_status"] = "warning"
            else:
                cell_data["validation_status"] = "pass"

            cells.append(cell_data)

        # V2-22c-PRE.1.2 (D34) — pivot per-check `affected_cells` lists
        # into per-cell `failed_checks` structured arrays. Each entry on
        # a cell is a 3-key object {check_id, result, category}
        # (evaluator §12.2 Pydantic-style binding). Region-scoped
        # checks (scope='global') are excluded from the per-cell pivot
        # — they live in the banner per Appendix H.
        #
        # V2-22c-PRE.1.8 (D35) — flatten per-violation context into the
        # top-level `cell_failed_check_details` array. Each entry
        # carries the full (cell_id, check_id, result, layer_idx,
        # variable, value, unit, bounds) tuple the cockpit drawer
        # renders directly.
        cell_failed_check_details: List[Dict[str, Any]] = []
        cells_by_id = {c["id"]: c for c in cells}
        if isinstance(validation_report, dict):
            for check in validation_report.get("checks", []) or []:
                result = check.get("result")
                if result not in ("fail", "warning"):
                    continue
                if check.get("scope") not in self._PER_CELL_SCOPES:
                    continue
                check_id = check.get("check")
                if not isinstance(check_id, str):
                    continue
                category = self._category_for_check_id(check_id)
                if category is None:
                    # Per-cell-scoped check whose check_id doesn't
                    # match a known prefix — skip rather than emit a
                    # category=None entry that would fail the
                    # downstream Pydantic-style validation. Surfaces
                    # as a sibling-sweep finding at evaluator §12.2.
                    continue
                details = check.get("details") or {}
                affected_ids = self._affected_cell_ids(
                    details.get("affected_cells"),
                )
                entry_template = {
                    "check_id": check_id,
                    "result": result,
                    "category": category,
                }
                for cell_id in affected_ids:
                    cell = cells_by_id.get(cell_id)
                    if cell is None:
                        # affected_cells lists a cell_id that the grid
                        # doesn't know about — defensive skip; the
                        # validator should never emit IDs outside the
                        # grid, but a stale validation_report against
                        # a re-built grid (PRE.3 exclude_cells path)
                        # could surface this.
                        continue
                    cell["failed_checks"].append(dict(entry_template))

                # PRE.1.8 flatten — per-violation detail rows.
                for vd in details.get("violation_details", []) or []:
                    if not isinstance(vd, dict):
                        continue
                    cell_id = vd.get("cell_id")
                    if cell_id is None or cell_id not in cells_by_id:
                        continue
                    cell_failed_check_details.append({
                        "cell_id": cell_id,
                        "check_id": check_id,
                        "result": result,
                        "category": category,
                        "layer_idx": vd.get("layer_idx"),
                        "variable": vd.get("variable"),
                        "value": vd.get("value"),
                        "unit": vd.get("unit"),
                        "bounds": vd.get("bounds"),
                    })

        # Stable ordering for both the per-cell `failed_checks` arrays
        # and the top-level `cell_failed_check_details` list — sorted
        # by (check_id, result) so reproducible JSON diffs survive
        # any future change in validator iteration order. Evaluator §2
        # binding for cockpit-cursor stability.
        for cell in cells:
            cell["failed_checks"].sort(
                key=lambda e: (e["check_id"], e["result"]),
            )
        cell_failed_check_details.sort(
            key=lambda e: (
                e["cell_id"], e["check_id"],
                e.get("layer_idx") if e.get("layer_idx") is not None else -1,
            ),
        )

        return {
            # V2-22c-PRE.1.1 (D5/D7) — `cell_summary_version: "2.0"` matches
            # the existing `validation_version: "2.0"` precedent at
            # validators/scientific.py:155 and `post_translate_version: "1.0"`
            # at validators/post_translate.py:114. The cockpit's loader-
            # fallback at prismweb/core/views.py:_load_cell_summary uses this
            # field to detect pre-PRE.1 fixtures and synthesize the empty
            # `failed_checks: []` shape per V2-22c contract D11/D19.
            "cell_summary_version": "2.0",
            "n_cells": len(cells),
            "resolution": getattr(grid, 'resolution', '5arcmin'),
            "cells": cells,
            # V2-22c-PRE.1.8 — top-level array; cockpit drawer reads
            # this directly so a single read renders the full
            # violation context without joining back to cells.
            "cell_failed_check_details": cell_failed_check_details,
        }

    def execute(
        self,
        stages: Optional[List[PipelineStage]] = None,
        progress_callback=None,
        cancel_check=None,
    ) -> PipelineResult:
        """Execute the translation pipeline.

        Args:
            stages: Optional list of stages to run. If None, runs all stages.
            progress_callback: Optional callback for per-stage progress updates.
                Must implement on_stage_start(stage, description),
                on_stage_complete(stage, result), and optionally
                on_substage_progress(stage, task, current, total, detail).
            cancel_check: Optional callable returning True when the user
                has requested cancellation. V2-22b/L: threaded through
                to every climate download loop + translator per-cell
                loop. None disables cancellation (CLI / unit-test usage).
                Passing ``callback._is_cancelled`` keeps backward compat
                with prismweb's old reach-into-private-method pattern;
                V2-22c will formalize this as a public ``ProgressCallback``
                protocol member.

        Returns:
            PipelineResult with all stage results and final status
        """
        self._progress_callback = progress_callback
        self._cancel_check = cancel_check
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
            # V2-22b L F-6 (AC L.12 / CA-7): inter-stage cancel hook.
            # Cancel observed between stages — e.g., during HARMONIZE
            # post-retrieve but before TRANSLATE — must not wait for
            # the next stage's natural cancel-observation point.
            # Checked at the top of each stage iteration so cancel
            # latency between stages is sub-second rather than whole-
            # stage-duration.
            def _check_cancel_before_stage(stage_name: str) -> None:
                raise_if_cancelled(
                    getattr(self, '_cancel_check', None),
                    f"executor.stage.{stage_name}",
                )

            # Stage 1: RETRIEVE
            if PipelineStage.RETRIEVE in stages:
                _check_cancel_before_stage("retrieve")
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
                _check_cancel_before_stage("harmonize")
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
                _check_cancel_before_stage("translate")
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

            # Stage 3.5: REMEDIATION (V2-22c-PRE.4 / D25)
            # Runs between TRANSLATE and VALIDATE so post-remediation
            # validation is honest (the user sees validation against
            # the corrected outputs, not the originals). Most runs
            # take the no-op path inside _execute_remediation; only
            # cockpit-bulk-fix re-runs carry a remediation_spec.
            #
            # Evaluator §12.5 binding — narrow ``except RemediationBlocked``
            # catch (NOT bare ``except Exception``). A broad catch
            # would swallow the structured `reason` payload and erase
            # the cockpit's AC-9.3 BLOCK copy specificity. Test
            # `tests/unit/test_remediation_stage.py` AST-walks this
            # block to assert the narrow shape.
            if PipelineStage.REMEDIATION in stages:
                _check_cancel_before_stage("remediation")
                _notify_start("remediation", "Applying corrections")
                harmonize_data_rem = stage_results.get(
                    "harmonize",
                    StageResult(
                        stage=PipelineStage.HARMONIZE,
                        success=True, data=None,
                    ),
                ).data
                try:
                    result = self._execute_remediation(
                        translation_results, harmonize_data_rem,
                    )
                except RemediationBlocked as block:
                    # Failure does NOT short-circuit the pipeline —
                    # VALIDATE still runs against the (un-remediated)
                    # outputs so the user sees post-failure state
                    # honestly. Cockpit reads
                    # `stage_results['remediation'].errors[0]` and
                    # renders the BLOCK reason verbatim.
                    result = StageResult(
                        stage=PipelineStage.REMEDIATION,
                        success=False,
                        errors=[str(block)],
                        warnings=[],
                        duration_seconds=0.0,
                    )
                stage_results["remediation"] = result
                _notify_complete("remediation", result)

            # Stage 4: VALIDATE
            if PipelineStage.VALIDATE in stages and translation_results:
                _check_cancel_before_stage("validate")
                _notify_start("validate", "Verifying outputs")
                # Pass unified_data for scientific validation (Phase 2a)
                harmonize_data = stage_results.get("harmonize", StageResult(
                    stage=PipelineStage.HARMONIZE, success=True, data=None
                )).data
                result = self._execute_validate(
                    translation_results, unified_data=harmonize_data
                )
                stage_results["validate"] = result
                _notify_complete("validate", result)

            # Stage 5: PACKAGE
            if PipelineStage.PACKAGE in stages:
                _check_cancel_before_stage("package")
                _notify_start("package", "Preparing your package")
                unified_data = stage_results.get("harmonize", StageResult(
                    stage=PipelineStage.HARMONIZE, success=True, data=None
                )).data
                validate_result = stage_results.get("validate")
                result = self._execute_package(
                    unified_data, translation_results, validate_result
                )
                stage_results["package"] = result
                _notify_complete("package", result)
                if result.data:
                    provenance_path = result.data.get("provenance_path")

        except PipelineCancelled:
            # V2-22b L: pipeline.execute boundary — propagate cancel so
            # the prismweb caller's _execute_pipeline_cancelled_cleanup
            # runs. Do NOT convert to an error-state PipelineResult; the
            # run.status is already set to 'error' by the cancel writer
            # and the handler-local cleanup coerces project.status.
            raise
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return self._build_result(
                False, stage_results, translation_results, None, start_time,
                error=str(e)
            )
        finally:
            # V2-19 improvement 3 + 4: guarantee pending decisions flush
            # on all exit paths (success, failure, exception). This creates
            # a synthetic "pipeline" artifact for any straggler decisions
            # recorded after the last transformation completed.
            try:
                if self.provenance and self.provenance.enabled:
                    self.provenance.finalize()
            except Exception as finalize_exc:
                self.logger.warning(
                    "Provenance finalize() failed (non-fatal): %s",
                    finalize_exc,
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
