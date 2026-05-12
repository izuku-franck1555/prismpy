"""
SARRA-Py translator for prismpy.

This module translates unified data to SARRA-Py model input format:
- YAML configuration file with region, crop, and simulation parameters
- GeoTIFF climate files (from TAMSAT/AgERA5) organized by variable
- YAML parameter files (variety.yaml, itk.yaml, soil.yaml)
- Standardized package with manifest.json, provenance.json, README.md

SARRA-Py Quirks (from analysis):
1. Bounding box format: [lat_NW, lon_NW, lat_SE, lon_SE] (NOT standard GIS)
2. iSDA path issue: Relative paths require running from specific directory
3. YAML dates: Must be converted to datetime.date objects
4. Climate sources: Separate files for rainfall (TAMSAT) and temperature (AgERA5)

Reference: SARRA-Py configuration structure and run_sarra_py.py
"""

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


def _to_native(obj):
    """Convert numpy types to Python native for YAML serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_native(v) for v in obj)
    return obj

from prismpy.config.schema import (
    Platform,
    PhenologyConfig,
    PhysiologyConfig,
    ManagementConfig,
    GenericSoilConfig,
)
from prismpy.models.climate import ClimateTimeSeries
from prismpy.models.crop import CropParameters, CropCalendar
from prismpy.models.region import Region
from prismpy.models.soil import SoilProfile
from prismpy.provenance.tracker import DecisionType, ProvenanceTracker
# Sprint E.3 AC-E3-9 — cockpit override dispatch helper. The
# climate / soil / management per-cell write sites in this
# translator route raw values through ``apply_override`` before
# writing per-cell platform files; with no sidecar the helper is
# a no-op short-circuit on the raw value, preserving Sprint E.2
# era output byte-equivalent for runs without overrides.
from prismpy.translators._shared.cockpit_overrides import apply_override
from prismpy.translators.base import (
    BaseTranslator,
    SarraPyTranslatorBase,
    TranslationResult,
    UnifiedData,
)


logger = logging.getLogger(__name__)


@dataclass
class SarraPyConfig:
    """SARRA-Py configuration structure.

    This mirrors the YAML structure that SARRA-Py expects.
    """
    project_name: str
    region_name: str
    country: str
    bounds: List[float]  # [lat_NW, lon_NW, lat_SE, lon_SE]
    start_date: date
    end_date: date
    crop_name: str
    planting_doy: int
    harvest_doy: int

    # Climate sources
    rainfall_source: str = "tamsat"
    temperature_source: str = "agera5"

    # Soil source
    soil_source: str = "isda"

    # Simulation parameters
    spinup_days: int = 30
    output_daily: bool = True


class SarraPyTranslator(SarraPyTranslatorBase):
    """Translator for SARRA-Py crop model.

    Generates a standardized package matching legacy SARRA-Py modules:
    1. README.md - Usage instructions
    2. manifest.json - File inventory with SHA256 checksums
    3. provenance.json - Data sources and processing decisions
    4. config/ - Project configuration
    5. data/boundaries/ - bounds.json with region bounding box
    6. data/climate/ - TAMSAT rainfall and AgERA5 variables by subdirectory
    7. data/soil/ - Soil parameters
    8. parameters/ - variety.yaml, itk.yaml, soil.yaml
    9. validation/ - Validation report

    Output structure:
        output_dir/
        ├── README.md
        ├── manifest.json
        ├── provenance.json
        ├── config/
        │   └── project_config.yaml
        ├── data/
        │   ├── boundaries/
        │   │   └── bounds.json
        │   ├── climate/
        │   │   ├── rainfall/
        │   │   ├── 2m_temperature_24_hour_maximum/
        │   │   ├── 2m_temperature_24_hour_mean/
        │   │   ├── 2m_temperature_24_hour_minimum/
        │   │   ├── ET0Hargeaves/
        │   │   └── solar_radiation_flux_daily/
        │   └── soil/
        │       └── soil_params.yaml
        ├── parameters/
        │   ├── variety.yaml
        │   ├── itk.yaml
        │   └── soil.yaml
        └── validation/
            └── validation_report.json
    """

    def translate(self, data: UnifiedData) -> TranslationResult:
        """Translate unified data to SARRA-Py format.

        Args:
            data: UnifiedData container with region, climate, soil, crop data

        Returns:
            TranslationResult with output files and status
        """
        self.log_translation_start(data)
        errors = []
        warnings = []
        output_files = []

        # Sprint E.3 fixup +15 (F-BN Boundary 3) — SARRA-Py translator
        # is in the PYTHIA-only ship's deferred-translator set per the
        # team-lead's scope guard. Surface a runtime warning whenever
        # the cockpit override sidecar is non-empty so the persona
        # gets an honest signal that their override decisions are NOT
        # being applied to SARRA-Py canonical GeoTIFF files; full
        # wiring lands in Phase 4.6.
        cockpit_sidecar = getattr(self, "cockpit_override_sidecar", None)
        if cockpit_sidecar is not None and cockpit_sidecar.overrides:
            warnings.append(
                f"Cockpit override sidecar carries {len(cockpit_sidecar.overrides)} "
                "entries but SARRA-Py value-replacement wiring is not yet "
                "implemented; values will NOT be applied to canonical SARRA-Py "
                "files (per-variable GeoTIFFs). Ship in Phase 4.6 per Sprint "
                "E.3 fixup +15 scope guard."
            )
            self.logger.warning(
                "SARRA-Py translator: cockpit override sidecar present (%d "
                "entries) but value-replacement wiring deferred to Phase 4.6 — "
                "persona's documented overrides will NOT surface in this "
                "SARRA-Py package.",
                len(cockpit_sidecar.overrides),
            )

        # Validate input data
        input_errors = self.validate_input_data(data)
        if input_errors:
            return self.create_result(
                success=False,
                output_files=[],
                errors=input_errors,
            )

        # Create standardized output subdirectories
        standard_subdirs = [
            "config",
            "data/boundaries",
            "data/climate/rainfall",
            "data/climate/2m_temperature_24_hour_maximum",
            "data/climate/2m_temperature_24_hour_mean",
            "data/climate/2m_temperature_24_hour_minimum",
            "data/climate/ET0Hargeaves",
            "data/climate/solar_radiation_flux_daily",
            "data/soil",
            "parameters",
            "validation",
        ]
        for subdir in standard_subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        try:
            # 1. Generate bounds.json
            bounds_file = self._generate_bounds_json(data.region)
            output_files.append(bounds_file)

            # 2. Generate main config.yaml
            config_file = self._generate_config_yaml(data)
            output_files.append(config_file)

            # 3. Generate standalone parameter files
            variety_file = self._generate_variety_yaml(data.crop_params)
            output_files.append(variety_file)

            itk_file = self._generate_itk_yaml(data)
            output_files.append(itk_file)

            if data.soil:
                soil_file = self._generate_soil_yaml(data.soil)
                output_files.append(soil_file)

            # 4. Generate climate data files (organized by variable)
            if data.climate:
                climate_files = self._generate_climate_files(
                    data.climate, data.region
                )
                output_files.extend(climate_files)

            # 5. Generate validation report
            validation_file = self._generate_validation_report(data)
            output_files.append(validation_file)

            # 6. Validate outputs
            validation_errors = self.validate_outputs()
            if validation_errors:
                warnings.extend(validation_errors)

        except Exception as e:
            logger.error(f"SARRA-Py translation failed: {e}")
            errors.append(str(e))
            return self.create_result(
                success=False,
                output_files=output_files,
                errors=errors,
            )

        # Record provenance
        if self.provenance:
            self.provenance.record_decision(
                decision_type=DecisionType.FORMAT_CHOICE,
                description=f"Generated SARRA-Py configuration for {data.region.name}",
                rationale="SARRA-Py requires YAML config with NetCDF climate data",
                alternatives=["manual configuration"],
                reference="prismpy.translators.sarra_py.translator.translate",
            )
            # V2-19 C8 (TS-03): spinup period rationale
            spinup_yrs = self.config.temporal.spinup_years
            self.provenance.record_decision(
                decision_type=DecisionType.DEFAULT_VALUE,
                description=(
                    f"Spinup period: {spinup_yrs} year(s) before "
                    f"simulation start"
                ),
                rationale=(
                    f"SARRA-Py requires a spinup period to initialize "
                    f"soil water balance from arbitrary initial conditions. "
                    f"The {spinup_yrs}-year spinup allows the model's "
                    f"internal soil moisture to equilibrate with the "
                    f"climate forcing before the analysis period begins. "
                    f"Literature recommends 1-3 years for Sahelian crop "
                    f"models depending on soil depth and initial moisture "
                    f"assumptions (Sultan et al. 2005). Valid for shallow-"
                    f"rooted annual crops in semi-arid regions (Sahel, "
                    f"Sudan savanna). NOT valid for deep-rooted perennials "
                    f"or irrigated systems where longer spinup (3-5 years) "
                    f"may be needed for deep soil layers to equilibrate."
                ),
                alternatives=[
                    "1-year spinup (faster, less stable for dry years)",
                    "3-year spinup (more conservative, recommended for "
                    "deep soils or long-term trend analysis)",
                ],
                reference=(
                    "prismpy.config.schema.TemporalConfig.spinup_years "
                    f"(default=2, current={spinup_yrs})"
                ),
            )

        # AC-F-CP-13.5: n_climate_locations counts REAL cells, not the
        # sentinel placeholder. The non-int safety of is_real_climate_cell_id
        # also keeps the SARRA-Py path-dict variants (string keys like
        # ``rainfall_dir`` / ``agera5_dir``) from inflating the count.
        from prismpy.sources.climate import is_real_climate_cell_id
        _real_climate_locations = sum(
            1 for k in (data.climate or {}).keys()
            if is_real_climate_cell_id(k)
        )
        result = self.create_result(
            success=True,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "region": data.region.name,
                "n_climate_locations": _real_climate_locations,
                "n_soil_profiles": len(data.soil) if data.soil else 0,
            },
        )

        self.log_translation_complete(result)
        return result

    def validate_outputs(self) -> List[str]:
        """Validate generated SARRA-Py outputs.

        Returns:
            List of validation error messages
        """
        errors = []

        # Check config/project_config.yaml exists (standardized path)
        config_path = self.output_dir / "config" / "project_config.yaml"
        if not config_path.exists():
            errors.append(f"Missing project_config.yaml at {config_path}")
        else:
            # Validate YAML structure
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                required_keys = ["project", "region", "temporal", "crop"]
                for key in required_keys:
                    if key not in config:
                        errors.append(f"Missing required section '{key}' in config.yaml")

                # Validate bounding box format
                if "region" in config and "bounds" in config["region"]:
                    bounds = config["region"]["bounds"]
                    if len(bounds) != 4:
                        errors.append(f"Bounding box must have 4 values, got {len(bounds)}")

            except yaml.YAMLError as e:
                errors.append(f"Invalid YAML in config.yaml: {e}")

        # Check output subdirectories
        for subdir in self.OUTPUT_SUBDIRS:
            subdir_path = self.output_dir / subdir
            if not subdir_path.exists():
                errors.append(f"Missing output subdirectory: {subdir}")

        # Check translated data files (package files checked in PACKAGE stage)
        data_files = [
            "data/boundaries/bounds.json",
            "parameters/variety.yaml",
            "parameters/itk.yaml",
            "parameters/soil.yaml",
            "validation/validation_report.json",
        ]
        for data_file in data_files:
            file_path = self.output_dir / data_file
            if not file_path.exists():
                errors.append(f"Missing data file: {data_file}")

        return errors

    def generate_package(
        self, data: UnifiedData, output_files: List[Path]
    ) -> List[Path]:
        """Generate package metadata files (manifest, provenance, README).

        Called by the pipeline's PACKAGE stage after translation and
        validation are complete.

        Args:
            data: Unified data container
            output_files: List of files generated during translation

        Returns:
            List of generated package file paths
        """
        return self._generate_package_files(data, output_files)

    def _generate_config_yaml(self, data: UnifiedData) -> Path:
        """Generate the main SARRA-Py configuration YAML file.

        Args:
            data: UnifiedData with region, temporal, and crop info

        Returns:
            Path to generated config.yaml
        """
        # Get platform-specific config
        platform_config = self.get_platform_config()

        # Build SARRA-Py bounding box format [lat_NW, lon_NW, lat_SE, lon_SE]
        sarra_bounds = data.region.bounds.to_sarra_py_format()

        # Get date range from config or climate data (cross-year-aware)
        start_date = date(self.config.temporal.start_year, 1, 1)
        crop_cal = self.config.crop.calendar if self.config.crop else None
        end_date = self.config.temporal.get_climate_end_date(crop_cal)

        if data.climate:
            # Check if it's path-based format or in-memory format
            if isinstance(data.climate, dict):
                if 'metadata' in data.climate and 'start_date' in data.climate['metadata']:
                    # Path-based format with metadata
                    start_date = date.fromisoformat(data.climate['metadata']['start_date'])
                    end_date = date.fromisoformat(data.climate['metadata']['end_date'])
                elif 'rainfall_dir' not in data.climate and 'agera5_dir' not in data.climate:
                    # In-memory ClimateTimeSeries format
                    first_ts = next(iter(data.climate.values()), None)
                    if first_ts and hasattr(first_ts, 'date_range'):
                        date_range = first_ts.date_range
                        if date_range:
                            start_date = date_range[0]
                            end_date = date_range[1]

        # Build configuration dictionary
        config = {
            "project": {
                "name": self.config.project.name,
                "description": f"SARRA-Py simulation for {data.region.name}",
                "generated_by": "prismpy",
            },
            "region": {
                "name": data.region.name,
                "country": data.region.country,
                "bounds": sarra_bounds,  # [lat_NW, lon_NW, lat_SE, lon_SE]
                "crs": data.region.bounds.crs,
            },
            "temporal": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "spinup_years": self.config.temporal.spinup_years,
                # V2-19 C8 (TS-03): spinup period rationale — recorded
                # in the output config dict; the decision is wired below.
            },
            "crop": {
                "name": self.config.crop.name,
                "variety": self.config.crop.variety or "default",
            },
            "climate": {
                "rainfall": {
                    "source": "tamsat",
                    "directory": str(self.output_dir / "rainfall"),
                },
                "temperature": {
                    "source": "agera5",
                    "directory": str(self.output_dir / "climate"),
                },
            },
            "soil": {
                "source": "isda",
                "directory": str(self.output_dir / "params"),
            },
            "output": {
                "directory": str(self.output_dir),
                "daily_output": True,
            },
        }

        # Add crop calendar if available
        if data.crop_calendar:
            first_calendar = next(iter(data.crop_calendar.values()))
            config["crop"]["calendar"] = {
                "planting_doy": first_calendar.planting_doy,
                "harvest_doy": first_calendar.harvest_doy,
            }
        elif self.config.crop.calendar:
            config["crop"]["calendar"] = {
                "planting_doy": self.config.crop.calendar.planting_doy,
                "harvest_doy": self.config.crop.calendar.maturity_doy,  # Config uses maturity_doy
            }

        # Write YAML file to config/project_config.yaml (standardized path)
        config_path = self.output_dir / "config" / "project_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(_to_native(config), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated SARRA-Py config: {config_path}")
        return config_path

    def _generate_crop_params(self, crop_params: CropParameters) -> Path:
        """Generate crop parameters YAML for SARRA-Py.

        Args:
            crop_params: CropParameters object

        Returns:
            Path to generated crop_params.yaml
        """
        params_dict = {
            "crop": {
                "name": crop_params.name,
                "variety": crop_params.variety or "default",
            },
            "phenology": {
                "base_temp": crop_params.base_temp,
                "optimal_temp": crop_params.optimal_temp,
                "max_temp": crop_params.max_temp,
            },
        }

        # Add thermal time requirements if available
        if crop_params.gdd_emergence:
            params_dict["thermal_time"] = {
                "emergence": crop_params.gdd_emergence,
                "flowering": crop_params.gdd_flowering,
                "maturity": crop_params.gdd_maturity,
            }

        # Add water stress parameters if available
        if crop_params.kc_initial is not None:
            params_dict["water"] = {
                "kc_initial": crop_params.kc_initial,
                "kc_mid": crop_params.kc_mid,
                "kc_end": crop_params.kc_end,
            }

        # Use SARRA-Py format method if available
        if hasattr(crop_params, 'to_sarra_py_format'):
            params_dict = crop_params.to_sarra_py_format()

        # Write YAML
        params_path = self.output_dir / "params" / "crop_params.yaml"
        with open(params_path, 'w') as f:
            yaml.dump(params_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated crop params: {params_path}")
        return params_path

    def _generate_soil_params(
        self,
        soil_profiles: Dict[int, SoilProfile],
        region: Region,
    ) -> Path:
        """Generate soil parameters YAML for SARRA-Py.

        SARRA-Py uses iSDA soil data. This generates a summary of soil
        properties for the region.

        Args:
            soil_profiles: Dictionary of location_id to SoilProfile
            region: Region for spatial reference

        Returns:
            Path to generated soil_params.yaml
        """
        # Aggregate soil properties across profiles
        sand_values = []
        clay_values = []
        silt_values = []
        depths = []

        for profile in soil_profiles.values():
            if profile.layers:
                # Get surface layer properties
                surface = profile.layers[0]
                sand_values.append(surface.sand)
                clay_values.append(surface.clay)
                if surface.silt:
                    silt_values.append(surface.silt)
                depths.append(profile.total_depth or 1.5)

        # V2-19 B0 finding #5: SARRA-Py aggregates per-cell soil
        # properties across the region using an unweighted arithmetic
        # mean (np.mean). This collapses spatial variability to a
        # single regional value without area weighting or profile-count
        # weighting. Record the implicit choice explicitly.
        if sand_values and self.provenance:
            self.provenance.record_decision(
                decision_type=DecisionType.AGGREGATION_METHOD,
                description=(
                    "SARRA-Py regional soil: unweighted arithmetic mean "
                    f"of {len(sand_values)} surface-layer profiles"
                ),
                rationale=(
                    "Surface sand/clay/silt values across the region are "
                    "reduced with np.mean (no area weighting, no depth "
                    "weighting, no category-aware grouping). SARRA-Py "
                    "consumes a single region-level soil parameter set, "
                    "so all spatial variability is collapsed to the mean "
                    "here — the regional model's spatial resolution is "
                    "coarser than this aggregation step."
                ),
                alternatives=[
                    "Area-weighted mean (requires cell-area computation)",
                    "Per-cell soil parameters (not supported by SARRA-Py config)",
                    "Dominant soil class by majority vote (categorical only)",
                ],
                reference=(
                    "prismpy.translators.sarra_py.translator "
                    "line ~504 (np.mean(sand_values/clay_values/silt_values))"
                ),
            )

        # Calculate regional averages
        soil_params = {
            "source": "isda",
            "region": {
                "name": region.name,
                "n_profiles": len(soil_profiles),
            },
            "surface_layer": {
                "sand_mean": float(np.mean(sand_values)) if sand_values else None,
                "clay_mean": float(np.mean(clay_values)) if clay_values else None,
                "silt_mean": float(np.mean(silt_values)) if silt_values else None,
            },
            "profile": {
                "depth_mean": float(np.mean(depths)) if depths else 1.5,
                "depth_range": [float(min(depths)), float(max(depths))] if depths else [1.5, 1.5],
            },
        }

        # Add individual profile data
        profiles_list = []
        for loc_id, profile in soil_profiles.items():
            profile_dict = {
                "location_id": loc_id,
                "lat": profile.lat,
                "lon": profile.lon,
                "source": profile.source,
                "texture": profile.surface_texture,
                "layers": [],
            }
            for layer in profile.layers:
                layer_dict = {
                    "depth_top": layer.depth_top,
                    "depth_bottom": layer.depth_bottom,
                    "sand": layer.sand,
                    "clay": layer.clay,
                }
                if layer.field_capacity:
                    layer_dict["field_capacity"] = layer.field_capacity
                if layer.wilting_point:
                    layer_dict["wilting_point"] = layer.wilting_point
                profile_dict["layers"].append(layer_dict)
            profiles_list.append(profile_dict)

        soil_params["profiles"] = profiles_list

        # Write YAML
        soil_path = self.output_dir / "params" / "soil_params.yaml"
        with open(soil_path, 'w') as f:
            yaml.dump(soil_params, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated soil params: {soil_path}")
        return soil_path

    def _generate_climate_files(
        self,
        climate_data: Any,
        region: Region,
        *,
        climate_kind: "ClimateKind" = None,
        projection_meta: "ProjectionClimateMeta" = None,
    ) -> List[Path]:
        """Generate or copy climate data files for SARRA-Py.

        SARRA-Py expects:
        - Rainfall from TAMSAT as GeoTIFF files
        - Temperature/radiation from AgERA5 as GeoTIFF files

        Handles three input formats:
        1. Dict with 'rainfall_dir' and 'agera5_dir' paths — copies existing
           observed-climate GeoTIFFs (default OBSERVED path).
        2. Dict[int, ClimateTimeSeries] — generates NetCDF from in-memory
           observed-climate data (default OBSERVED path).
        3. ``climate_kind=ClimateKind.PROJECTION`` + Dict[str, xr.DataArray] —
           generates per-variable GeoTIFF directories from ISIMIP3b xarray
           data (Sprint G AC-G-7c).

        Per Sprint G AC-G-7c, the projection path additionally writes
        a sidecar ``<variable>/.meta.json`` per variable directory
        capturing ``gcm_source`` / ``bias_correction_method`` /
        ``time_slice`` so downstream consumers can introspect
        provenance without re-reading the parent manifest. The same
        ``ProjectionClimateMeta`` schema as ACEA's per-cell sidecar
        (per durable §24 canonical-source).

        Args:
            climate_data: Either path dict, ClimateTimeSeries dict, or
                (when ``climate_kind=PROJECTION``) Dict[variable_name, xr.DataArray].
            region: Region for metadata.
            climate_kind: Source provenance discriminator. Defaults to
                ``ClimateKind.OBSERVED`` (backward-compat).
            projection_meta: Required when ``climate_kind=PROJECTION``;
                emitted into the per-variable sidecars.

        Returns:
            List of generated/copied file paths.
        """
        # Late import — keeps harmonize/ + models.scenario off the
        # translator's import-time path for callers that don't touch
        # projection climate.
        from prismpy.harmonize.climate_kind import ClimateKind as _ClimateKind

        if climate_kind is None:
            climate_kind = _ClimateKind.OBSERVED

        if climate_kind == _ClimateKind.PROJECTION:
            if projection_meta is None:
                raise ValueError(
                    "climate_kind=PROJECTION requires projection_meta. The "
                    "per-variable sidecar .meta.json files cannot be "
                    "written without gcm_source / bias_correction_method "
                    "/ time_slice fields. Pass a ProjectionClimateMeta "
                    "instance."
                )
            return self._generate_projection_climate_geotiffs(
                climate_data, projection_meta
            )

        import shutil
        output_files = []

        # Check if climate_data contains file paths (new format)
        if isinstance(climate_data, dict) and ('rainfall_dir' in climate_data or 'agera5_dir' in climate_data):
            return self._copy_climate_geotiffs(climate_data)

        # Otherwise, assume in-memory ClimateTimeSeries data (original format)
        if not isinstance(climate_data, dict):
            logger.warning("Climate data format not recognized")
            return output_files

        # Try to use xarray/netCDF4 for NetCDF output
        try:
            import xarray as xr
            use_netcdf = True
        except ImportError:
            logger.warning("xarray not available, using CSV for climate data")
            use_netcdf = False

        # Separate rainfall and temperature data
        for loc_id, ts in climate_data.items():
            if not hasattr(ts, 'records'):  # Skip if not a ClimateTimeSeries
                continue
            if use_netcdf:
                # Create NetCDF files
                rainfall_file = self._create_rainfall_netcdf(ts, loc_id)
                if rainfall_file:
                    output_files.append(rainfall_file)

                temp_file = self._create_temperature_netcdf(ts, loc_id)
                if temp_file:
                    output_files.append(temp_file)
            else:
                # Fallback to CSV
                csv_file = self._create_climate_csv(ts, loc_id)
                output_files.append(csv_file)

        return output_files

    def _generate_projection_climate_geotiffs(
        self,
        climate_by_variable: Dict[str, Any],
        projection_meta: "ProjectionClimateMeta",
    ) -> List[Path]:
        """Emit projection-climate per-variable GeoTIFF directories (AC-G-7c).

        Per Sprint G AC-G-7c: SARRA-Py's projection-climate path takes a
        ``Dict[variable_name, xarray.DataArray]`` produced by AC-G-7d
        calendar conversion + AC-G-2 cached_cutout, and writes one
        directory per variable under
        ``<output_dir>/data/climate/<variable>/`` containing one GeoTIFF
        per day plus a sidecar ``.meta.json`` per directory carrying
        ``ProjectionClimateMeta``.

        Determinism contract per CC-G-7 + the AC-G-13 deliverable hash
        precondition. GeoTIFF byte-identity requires explicit pinning of:

        * ``crs``: explicitly EPSG:4326 (WGS84) so default-from-env can't
          drift.
        * ``dtype``: explicitly ``float32`` (matches ISIMIP3b atmospheric
          data; smaller than float64 without precision loss for the
          variables in scope).
        * ``nodata``: explicitly ``-9999.0`` (the SARRA-Py canonical
          missing-value sentinel).
        * ``compress='lzw'``: lossless, deterministic across rasterio
          versions.
        * ``tiled=True`` + ``blockxsize=256`` + ``blockysize=256``:
          tiled output with explicit block size so internal layout
          doesn't drift on different GDAL builds.
        * GDAL ``.aux.xml`` sidecars disabled via ``GDAL_PAM_ENABLED=NO``
          process-env lock during the write.

        The helper does NOT modify the global GDAL_PAM_ENABLED setting
        for the whole process — it scopes the override to the write
        block via ``rasterio.Env``.

        Args:
            climate_by_variable: Dict of variable name (e.g., ``"tasmax"``,
                ``"pr"``) to xarray DataArray. Each DataArray must have
                ``time``, ``lat``, ``lon`` dims and a CRS attached via
                ``rio.write_crs``.
            projection_meta: Per-package projection metadata. Each
                variable's sidecar inherits this; the sidecar's
                ``variable`` field is set to the per-variable name.

        Returns:
            List of paths emitted (every per-day GeoTIFF + every
            per-variable sidecar).
        """
        import json

        import rasterio
        from rasterio.transform import from_bounds

        from prismpy.harmonize.isimip_unit_conversions import (
            convert_to_sarra_py_units,
            sarra_py_directory_for_isimip,
        )
        from prismpy.models.scenario import ProjectionClimateMeta as _Meta
        from prismpy.standards.isimip_versions import (
            ISIMIP_TO_SARRA_VAR_MAPPING,
        )

        output_files: List[Path] = []
        climate_root = self.output_dir / "data" / "climate"

        # Pin GDAL options for byte-identical output.
        gdal_env = {
            "GDAL_PAM_ENABLED": "NO",  # No .aux.xml sidecars
        }

        for variable, data_array in climate_by_variable.items():
            # Codex round 2 P1 absorption — route every emitted variable
            # through the canonical ISIMIP → SARRA-Py mapping per durable
            # §24. Variables not in the mapping (e.g., ``hurs``,
            # ``sfcWind``, ``tasmean``) have no SARRA-Py consumer and
            # are skipped with a log warning rather than emitted as raw
            # CF names that downstream consumers would ignore.
            if variable not in ISIMIP_TO_SARRA_VAR_MAPPING:
                logger.warning(
                    f"Skipping ISIMIP variable {variable!r} — no SARRA-Py "
                    "directory mapping registered in "
                    "prismpy.standards.isimip_versions."
                    "ISIMIP_TO_SARRA_VAR_MAPPING. Add an entry to extend "
                    "consumer coverage."
                )
                continue
            sarra_dir_name = sarra_py_directory_for_isimip(variable)
            var_dir = climate_root / sarra_dir_name
            var_dir.mkdir(parents=True, exist_ok=True)

            # Resolve geospatial layout. The DataArray must carry a CRS
            # (caller's responsibility); we read the lat/lon edges to
            # compute the affine transform explicitly so rasterio can't
            # drift on auto-derivation.
            try:
                lat_values = data_array["lat"].values
                lon_values = data_array["lon"].values
                time_values = data_array["time"].values
            except (KeyError, AttributeError) as exc:
                raise ValueError(
                    f"Projection climate DataArray for {variable!r} must have "
                    "'time', 'lat', 'lon' dims; missing one or more"
                ) from exc

            # Pixel size (assumes regular grid; ISIMIP3b is 0.5°)
            lat_pixel = abs(float(lat_values[1]) - float(lat_values[0])) if len(lat_values) > 1 else 0.5
            lon_pixel = abs(float(lon_values[1]) - float(lon_values[0])) if len(lon_values) > 1 else 0.5

            west = float(lon_values.min()) - lon_pixel / 2
            east = float(lon_values.max()) + lon_pixel / 2
            south = float(lat_values.min()) - lat_pixel / 2
            north = float(lat_values.max()) + lat_pixel / 2
            height, width = len(lat_values), len(lon_values)
            transform = from_bounds(west, south, east, north, width, height)

            with rasterio.Env(**gdal_env):
                for time_idx, time_val in enumerate(time_values):
                    # Format the date stable as YYYY-MM-DD for filename
                    if hasattr(time_val, "strftime"):
                        date_str = time_val.strftime("%Y-%m-%d")
                    else:
                        # numpy datetime64 → str
                        date_str = str(time_val).split("T")[0][:10]

                    raw_band = data_array.isel(time=time_idx).values.astype(
                        "float32"
                    )
                    # Codex round 2 P1 absorption — convert raw ISIMIP
                    # CF units to SARRA-Py target units BEFORE writing
                    # (kg m⁻² s⁻¹ → mm/day for ``pr``; W m⁻² →
                    # J m⁻²/day for ``rsds``; K passthrough for
                    # ``tasmax`` / ``tasmin``). Per durable §24 the
                    # math lives in ``harmonize.isimip_unit_conversions``.
                    converted = convert_to_sarra_py_units(variable, raw_band)
                    band = np.asarray(converted, dtype="float32")
                    # NaN-as-valid-value-leakage guard (codex round 1
                    # boundary 3/7 P2): any NaN that survived from
                    # upstream ``_FillValue`` decoding or a sea-mask
                    # cell would be written into the GeoTIFF band as a
                    # NaN float32 — which GDAL and rasterio do NOT mask
                    # on read because the stored value does not match
                    # the declared ``nodata=-9999.0`` tag. The result is
                    # silent NaN propagation through downstream SARRA-Py
                    # ingestion. Replace NaN with the declared sentinel
                    # before write so the nodata tag actually applies.
                    band = np.where(np.isnan(band), -9999.0, band).astype("float32")
                    # Lat axis is typically descending in geospatial convention
                    # (north → south); flip if the source is ascending so
                    # the GeoTIFF's row 0 is the northernmost row.
                    if len(lat_values) > 1 and lat_values[0] < lat_values[-1]:
                        band = band[::-1, :]

                    tif_path = var_dir / f"{date_str}.tif"
                    # GeoTIFF tiled-output requires block dims to be
                    # multiples of 16; small ISIMIP3b cutouts (Niamey
                    # area is ~6 × 6 pixels at 0.5°) fall below that
                    # threshold and fall back to stripped output.
                    write_kwargs = {
                        "driver": "GTiff",
                        "height": height,
                        "width": width,
                        "count": 1,
                        "dtype": "float32",
                        "crs": "EPSG:4326",
                        "transform": transform,
                        "nodata": -9999.0,
                        "compress": "lzw",
                    }
                    if width >= 16 and height >= 16:
                        # Tiled output requires block dims as multiples
                        # of 16. Always use 16x16 blocks regardless of
                        # image size — small ISIMIP3b cutouts (~6 × 6
                        # pixels at 0.5° over Niamey) skip this branch
                        # entirely; larger images get 16x16 tiles
                        # consistently. Performance-optimal would be
                        # 256x256 for large global outputs, but that
                        # would require a separate "single-tile-vs-
                        # multi-tile" branch and the deterministic-
                        # output contract is the priority.
                        write_kwargs["tiled"] = True
                        write_kwargs["blockxsize"] = 16
                        write_kwargs["blockysize"] = 16
                    with rasterio.open(tif_path, "w", **write_kwargs) as dst:
                        dst.write(band, 1)

                    output_files.append(tif_path)

            # Per-variable sidecar — one .meta.json per directory.
            per_var_meta = _Meta(
                gcm_source=projection_meta.gcm_source,
                bias_correction_method=projection_meta.bias_correction_method,
                time_slice_start=projection_meta.time_slice_start,
                time_slice_end=projection_meta.time_slice_end,
                variable=str(variable),
                scenario_label=projection_meta.scenario_label,
            )
            sidecar_path = var_dir / ".meta.json"
            sidecar_payload = json.dumps(
                per_var_meta.model_dump(exclude_none=True),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
            sidecar_path.write_bytes(sidecar_payload)
            output_files.append(sidecar_path)

        logger.info(
            f"Generated SARRA-Py projection climate: "
            f"{len(climate_by_variable)} variables, "
            f"{len(output_files)} files total"
        )
        return output_files

    def _copy_climate_geotiffs(self, climate_data: Dict[str, Any]) -> List[Path]:
        """Copy existing GeoTIFF climate files to package structure.

        Args:
            climate_data: Dict with 'rainfall_dir' and/or 'agera5_dir' paths

        Returns:
            List of copied file paths
        """
        import shutil
        output_files = []

        # Copy TAMSAT rainfall files
        rainfall_dir = climate_data.get('rainfall_dir')
        if rainfall_dir and rainfall_dir.exists():
            dest_dir = self.output_dir / "data" / "climate" / "rainfall"
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy only .tif files — SARRA-Py consumes daily GeoTIFFs only.
            # Raw .nc files (TAMSAT pentad downloads) must NOT be included.
            climate_files = list(rainfall_dir.glob("*.tif"))
            logger.info(f"Copying {len(climate_files)} rainfall GeoTIFF files...")

            for src_file in climate_files:
                dest_file = dest_dir / src_file.name
                shutil.copy2(src_file, dest_file)
                output_files.append(dest_file)

            logger.info(f"Copied rainfall data: {len(climate_files)} files")

        # Copy AgERA5 variable files
        agera5_dir = climate_data.get('agera5_dir')
        if agera5_dir and agera5_dir.exists():
            # AgERA5 has subdirectories for each variable
            var_dirs = [d for d in agera5_dir.iterdir() if d.is_dir()]

            for var_dir in var_dirs:
                var_name = var_dir.name
                dest_dir = self.output_dir / "data" / "climate" / var_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                climate_files = list(var_dir.glob("*.tif"))
                if climate_files:
                    logger.info(f"Copying {len(climate_files)} {var_name} GeoTIFF files...")

                    for src_file in climate_files:
                        dest_file = dest_dir / src_file.name
                        shutil.copy2(src_file, dest_file)
                        output_files.append(dest_file)

            total_vars = len([d for d in agera5_dir.iterdir() if d.is_dir()])
            logger.info(f"Copied AgERA5 data: {total_vars} variables")

        return output_files

    def _create_rainfall_netcdf(
        self,
        ts: ClimateTimeSeries,
        loc_id: int,
    ) -> Optional[Path]:
        """Create TAMSAT-style rainfall NetCDF file.

        Args:
            ts: ClimateTimeSeries with precipitation data
            loc_id: Location identifier

        Returns:
            Path to NetCDF file or None if creation failed
        """
        try:
            import xarray as xr
            import pandas as pd

            # Extract precipitation
            dates = [r.date for r in ts.records]
            precip = [r.precip for r in ts.records]

            # Create xarray Dataset
            ds = xr.Dataset(
                {
                    "precip": (["time"], precip, {
                        "units": "mm/day",
                        "long_name": "Daily precipitation",
                        "source": ts.source,
                    }),
                },
                coords={
                    "time": pd.to_datetime(dates),
                    "lat": ts.lat,
                    "lon": ts.lon,
                },
                attrs={
                    "location_id": loc_id,
                    "source": "TAMSAT via prismpy",
                    "created_by": "prismpy.translators.sarra_py",
                },
            )

            # Save to NetCDF (in data/climate/rainfall/ per package structure)
            output_path = self.output_dir / "data" / "climate" / "rainfall" / f"rainfall_{loc_id}.nc"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(output_path)

            logger.debug(f"Created rainfall NetCDF: {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"Failed to create rainfall NetCDF for loc {loc_id}: {e}")
            return None

    def _create_temperature_netcdf(
        self,
        ts: ClimateTimeSeries,
        loc_id: int,
    ) -> Optional[Path]:
        """Create AgERA5-style temperature NetCDF file.

        Args:
            ts: ClimateTimeSeries with temperature data
            loc_id: Location identifier

        Returns:
            Path to NetCDF file or None if creation failed
        """
        try:
            import xarray as xr
            import pandas as pd

            # Extract temperature and radiation data
            dates = [r.date for r in ts.records]
            tmax = [r.tmax for r in ts.records]
            tmin = [r.tmin for r in ts.records]
            tmean = [r.tmean for r in ts.records]
            srad = [r.srad for r in ts.records]

            # Create xarray Dataset
            ds = xr.Dataset(
                {
                    "tmax": (["time"], tmax, {
                        "units": "degC",
                        "long_name": "Maximum temperature",
                    }),
                    "tmin": (["time"], tmin, {
                        "units": "degC",
                        "long_name": "Minimum temperature",
                    }),
                    "tmean": (["time"], tmean, {
                        "units": "degC",
                        "long_name": "Mean temperature",
                    }),
                    "srad": (["time"], srad, {
                        "units": "MJ/m2/day",
                        "long_name": "Solar radiation",
                    }),
                },
                coords={
                    "time": pd.to_datetime(dates),
                    "lat": ts.lat,
                    "lon": ts.lon,
                },
                attrs={
                    "location_id": loc_id,
                    "source": "AgERA5 via prismpy",
                    "created_by": "prismpy.translators.sarra_py",
                },
            )

            # Add ET0 if available
            if any(r.et0 is not None for r in ts.records):
                et0 = [r.et0 if r.et0 is not None else np.nan for r in ts.records]
                ds["et0"] = (["time"], et0, {
                    "units": "mm/day",
                    "long_name": "Reference evapotranspiration",
                })

            # Save to NetCDF (in data/climate/ per package structure)
            output_path = self.output_dir / "data" / "climate" / f"climate_{loc_id}.nc"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(output_path)

            logger.debug(f"Created climate NetCDF: {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"Failed to create temperature NetCDF for loc {loc_id}: {e}")
            return None

    def _create_climate_csv(
        self,
        ts: ClimateTimeSeries,
        loc_id: int,
    ) -> Path:
        """Create CSV climate file as NetCDF fallback.

        Args:
            ts: ClimateTimeSeries
            loc_id: Location identifier

        Returns:
            Path to CSV file
        """
        import pandas as pd

        # Convert to DataFrame
        df = ts.to_dataframe()

        # Add location metadata
        df["location_id"] = loc_id
        df["lat"] = ts.lat
        df["lon"] = ts.lon

        # Save to CSV (in data/climate/ per package structure)
        output_path = self.output_dir / "data" / "climate" / f"climate_{loc_id}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.debug(f"Created climate CSV: {output_path}")
        return output_path

    # =========================================================================
    # NEW METHODS FOR STANDARDIZED PACKAGE GENERATION
    # =========================================================================

    def _generate_bounds_json(self, region: Region) -> Path:
        """Generate bounds.json with SARRA-Py and GIS format bounds.

        Args:
            region: Region object

        Returns:
            Path to bounds.json
        """
        sarra_bounds = region.bounds.to_sarra_py_format()
        gis_bounds = [
            region.bounds.minx,
            region.bounds.miny,
            region.bounds.maxx,
            region.bounds.maxy,
        ]

        bounds_data = {
            "region": region.name,
            "country": region.country,
            "crs": region.bounds.crs,
            "bounds_sarra_py": sarra_bounds,
            "bounds_sarra_py_description": "[lat_NW, lon_NW, lat_SE, lon_SE]",
            "bounds_gis": gis_bounds,
            "bounds_gis_description": "[minx, miny, maxx, maxy]",
            "coordinates": {
                "lat_NW": sarra_bounds[0],
                "lon_NW": sarra_bounds[1],
                "lat_SE": sarra_bounds[2],
                "lon_SE": sarra_bounds[3],
            }
        }

        output_path = self.output_dir / "data" / "boundaries" / "bounds.json"
        with open(output_path, 'w') as f:
            json.dump(bounds_data, f, indent=2)

        logger.info(f"Generated bounds.json: {output_path}")
        return output_path

    def _load_template(self, template_type: str) -> Optional[Dict[str, Any]]:
        """Load a parameter template from file.

        Args:
            template_type: One of 'variety', 'itk', 'soil'

        Returns:
            Template parameters as dict, or None if not found
        """
        sarra_config = self.config.platform_config.sarra_py if self.config.platform_config else None
        if not sarra_config:
            return None

        # Priority 1: Explicit template file path
        explicit_file = getattr(sarra_config, f'{template_type}_template_file', None)
        if explicit_file:
            template_path = Path(explicit_file)
            if template_path.exists():
                with open(template_path, 'r') as f:
                    params = yaml.safe_load(f)
                logger.info(f"Loaded {template_type} template from: {template_path}")
                return params
            else:
                logger.warning(f"Template file not found: {template_path}")

        # Priority 2: templates_dir + template name
        if sarra_config.templates_dir:
            template_name = getattr(sarra_config, f'{template_type}_template', None)
            if template_name:
                template_path = Path(sarra_config.templates_dir) / template_type / f"{template_name}.yaml"
                if template_path.exists():
                    with open(template_path, 'r') as f:
                        params = yaml.safe_load(f)
                    logger.info(f"Loaded {template_type} template from: {template_path}")
                    return params
                else:
                    logger.warning(f"Template not found: {template_path}")

        return None

    # =========================================================================
    # GENERIC PARAMETER MAPPING METHODS
    # =========================================================================

    def _map_generic_to_sarra_py_variety(self) -> Dict[str, Any]:
        """Map generic crop parameters to SARRA-Py variety.yaml format.

        Uses config.crop.phenology and config.crop.physiology if available,
        otherwise uses maize defaults from the PhenologyConfig/PhysiologyConfig classes.

        Returns:
            Dict with all 52 SARRA-Py variety parameters
        """
        pheno = self.config.crop.phenology
        phys = self.config.crop.physiology

        # Use defaults if not provided
        if pheno is None:
            pheno = PhenologyConfig()  # Uses class defaults
        if phys is None:
            phys = PhysiologyConfig()  # Uses class defaults

        # Split grain filling into two phases (SARRA-Py specific: 71.4/28.6 split)
        # Based on calibrated maize parameters where SDJMatu1=500, SDJMatu2=200 (total 700)
        matu1 = round(pheno.grain_filling_gdd * 0.714, 1)
        matu2 = round(pheno.grain_filling_gdd * 0.286, 1)

        return {
            # Phenology (thermal time in degree-days)
            "SDJLevee": pheno.emergence_gdd,
            "SDJBVP": pheno.vegetative_phase_gdd,
            "SDJRPR": pheno.reproductive_phase_gdd,
            "SDJMatu1": matu1,
            "SDJMatu2": matu2,

            # Temperature thresholds
            "TBase": phys.base_temperature,
            "TOpt1": phys.optimal_temperature,
            "TOpt2": phys.optimal_temperature + 8.0,  # SARRA-Py convention
            "TLim": phys.max_temperature,

            # Biomass/yield parameters
            "txConversion": phys.radiation_use_efficiency,
            "KRdtPotA": phys.harvest_index,
            "KRdtPotB": 200.0,

            # Photoperiod sensitivity
            # PPExp=0 disables photoperiod effect, PPsens is kept at default 5.0
            "PPExp": 0.0 if pheno.photoperiod_sensitivity < 0.1 else pheno.photoperiod_sensitivity,
            "SeuilPP": 13.6,
            "PPsens": 5.0,  # Default value from calibration
            "PPCrit": 11.0,

            # SARRA-Py specific defaults (from maize West Africa calibration)
            "pcReallocFeuille": 0.7,
            "txAssimBVP": 1.0,
            "txAssimMatu1": 0.9,
            "txAssimMatu2": 0.1,
            "kRespMaint": 0.01,
            "aeroTotBase": 0.6,
            "aeroTotPente": 3.5e-05,
            "feuilAeroBase": 0.60,
            "feuilAeroPente": -1.4e-04,
            "txRealloc": 0.4,
            "tempMaint": 25.0,
            "kcMax": 1.25,
            "PFactor": 0.45,
            "seuilCstrMortality": 3.0,
            "kdf": 0.4,
            "txResGrain": 0.55,
            "VRacLevee": 30.0,
            "VRacBVP": 15.0,
            "VRacPSP": 15.0,
            "VRacRPR": 15.0,
            "VRacMatu1": 12.0,
            "VRacMatu2": 12.0,
            "slaMin": 0.002,
            "slaMax": 0.006,
            "slaPente": 0.4,
            "densiteA": 0.7,
            "densiteP": 4.5,
            "densOpti": 65000.0,
            "AGauss": 1.0,
            "KRdtBiom": 0.0,
            "LGauss": 1.0,
            "NIYo": 1.0,
            "NIp": 0.0,
            "phaseDevVeg": 0,
            "poidsSecGrain": 0.38,
            "senCO2": 10.0,
        }

    def _map_generic_to_sarra_py_itk(self) -> Dict[str, Any]:
        """Map generic management parameters to SARRA-Py itk.yaml format.

        Uses config.management if available, otherwise uses maize defaults.

        Returns:
            Dict with all 24 SARRA-Py ITK parameters
        """
        mgmt = self.config.management

        # Use defaults if not provided
        if mgmt is None:
            mgmt = ManagementConfig(planting_density=62500.0)
            # V2-19 C1 (TP-04): same rationale as PYTHIA — see
            # pythia/translator.py for the full AC11-compliant text.
            # Recorded here too because SARRA-Py consumes the density
            # independently (as "densite" in the ITK config).
            if self.provenance:
                self.provenance.record_decision(
                    decision_type=DecisionType.DEFAULT_VALUE,
                    description=(
                        "Planting density fallback: 62,500 plants/ha "
                        "(no user-provided management config)"
                    ),
                    rationale=(
                        "No source in code. Literature range for tropical "
                        "maize: 25,000 (Sahel rainfed) to 80,000 (irrigated "
                        "high-input). Default is moderate-to-high density "
                        "(~6.25 plants/m²), appropriate for Sudan-Savanna "
                        "improved OPV maize. NOT valid for Sahelian millet/"
                        "sorghum (<30,000) or high-input commercial (>70,000). "
                        "Override via config.management.planting_density."
                    ),
                    alternatives=[
                        "User-provided planting density from config (preferred)",
                        "Crop-specific defaults (V2-20)",
                    ],
                    reference=(
                        "prismpy.translators.sarra_py.translator "
                        "ManagementConfig(planting_density=62500.0)"
                    ),
                )

        # Compute opportunistic sowing date (year before start)
        sarra_config = self.config.platform_config.sarra_py if self.config.platform_config else None
        sow_month = sarra_config.sowing_search_month if sarra_config else 5
        sow_day = sarra_config.sowing_search_day if sarra_config else 1
        opportunistic_year = self.config.temporal.start_year - 1
        date_semis = f"{opportunistic_year}-{sow_month}-{sow_day}"

        return {
            "DateSemis": date_semis,
            "seuilEauSemis": mgmt.sowing_threshold_mm,
            "nbjTestSemis": 0,
            "profRacIni": 0.0,
            "densite": mgmt.planting_density,
            "NI": float('nan'),
            "irrigAuto": mgmt.irrigation,
            "irrigAutoTarget": mgmt.irrigation_target,
            "maxIrrig": 0.0 if not mgmt.irrigation else 50.0,
            "coefMc": 0.0,
            "surfMc": 1.0,
            "biomIniMc": 0.0,
            "humSatMc": 0.0,
            "mulch": 1.0,
            "KI": 0.0,
            "KNLit": 0.0,
            "KNUp": 0.0,
            "KT": 0.0,
            "DisMc": 0,
            "TxRecolte": 0.0,
            "TxaTerre": 0.0,
            "NbUBT": 10.0,
            "dateFin": 300.0,
            "precision": 0.0,
        }

    def _map_generic_to_sarra_py_soil(self) -> Dict[str, Any]:
        """Map generic soil configuration to SARRA-Py soil.yaml format.

        Uses config.soil_config if available, otherwise uses West Africa defaults.

        Returns:
            Dict with all 13 SARRA-Py soil parameters
        """
        soil_cfg = self.config.soil_config

        # Default depth
        depth_m = soil_cfg.default_depth_m if soil_cfg else 1.5
        depth_mm = depth_m * 1000

        # SARRA-Py uses surface and deep layers
        surf_thick = min(200.0, depth_mm * 0.15)
        prof_thick = depth_mm - surf_thick

        return {
            "epaisseurSurf": surf_thick,
            "epaisseurProf": prof_thick,
            "stockIniSurf": 30.0,
            "stockIniProf": 170.0,
            "seuilRuiss": 20.0,
            "pourcRuiss": 0.3,
            "ru": 132.0,
            "HumCR": 0.32,
            "HumFC": 0.32,
            "HumPF": 0.18,
            "HumSat": 0.48,
            "Pevap": 0.2,
            "PercolationMax": 5.0,
        }

    # =========================================================================
    # PARAMETER FILE GENERATION METHODS
    # =========================================================================

    def _generate_variety_yaml(self, crop_params: Optional[CropParameters]) -> Path:
        """Generate variety.yaml from calibrated database, template, or config.

        Priority order:
        1. Calibrated variety database (prismpy/data/sarra_py_varieties/{crop}.yaml)
        2. Explicit template file (variety_template_file or templates_dir)
        3. Generic config mapping (crop.phenology/physiology)
        4. crop_params.parameters (from data loading)
        5. Error - no variety source available

        Args:
            crop_params: CropParameters object or None

        Returns:
            Path to variety.yaml
        """
        variety_params = None

        # Priority 1: Load from calibrated variety database (field-validated params)
        if variety_params is None:
            crop_lower = self.config.crop.name.lower()
            variety_db_dirs = [
                Path(__file__).resolve().parents[4] / "data" / "sarra_py_varieties",
                Path("data/sarra_py_varieties"),
            ]
            for db_dir in variety_db_dirs:
                variety_file = db_dir / f"{crop_lower}.yaml"
                if variety_file.exists():
                    with open(variety_file, 'r') as f:
                        variety_params = yaml.safe_load(f)
                    logger.info(f"Loaded calibrated variety params from {variety_file}")
                    break

        # Priority 2: Try to load from explicit template
        if variety_params is None:
            variety_params = self._load_template('variety')
            if variety_params:
                logger.info("Loaded variety params from template")

        # Priority 3: Use generic mapping if phenology/physiology available
        if variety_params is None:
            if self.config.crop.phenology is not None or self.config.crop.physiology is not None:
                variety_params = self._map_generic_to_sarra_py_variety()
                logger.info("Generated variety params from generic config (phenology/physiology)")

        # Priority 4: Fall back to crop_params
        if variety_params is None and crop_params:
            if hasattr(crop_params, 'parameters') and crop_params.parameters:
                variety_params = crop_params.parameters
                logger.info("Using variety parameters from crop_params")
            elif hasattr(crop_params, 'to_sarra_py_format'):
                variety_params = crop_params.to_sarra_py_format()
                logger.info("Using variety parameters from crop_params.to_sarra_py_format()")

        # Error if no source available
        if variety_params is None:
            raise ValueError(
                "No variety parameters available. Please specify either:\n"
                "  - A calibrated YAML in prismpy/data/sarra_py_varieties/\n"
                "  - variety_template_file: explicit path to YAML file\n"
                "  - crop.phenology/physiology: generic parameters in config\n"
                "  - Ensure crop_params has parameters"
            )

        output_path = self.output_dir / "parameters" / "variety.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(variety_params, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated variety.yaml: {output_path}")
        return output_path

    def _generate_itk_yaml(self, data: UnifiedData) -> Path:
        """Generate itk.yaml from generic config or template.

        Priority order:
        1. Generic config (management) - platform agnostic
        2. Template file (itk_template_file or templates_dir)
        3. Error - no ITK source available

        Args:
            data: UnifiedData with temporal and crop info

        Returns:
            Path to itk.yaml
        """
        itk_params = None

        # Priority 1: Use generic mapping if management config available
        if self.config.management is not None:
            itk_params = self._map_generic_to_sarra_py_itk()
            logger.info("Generated ITK params from generic config (management)")

        # Priority 2: Try to load from template
        if itk_params is None:
            itk_params = self._load_template('itk')
            if itk_params:
                logger.info("Loaded ITK params from template")

                # Update DateSemis for opportunistic mode
                sarra_config = self.config.platform_config.sarra_py if self.config.platform_config else None
                sow_month = sarra_config.sowing_search_month if sarra_config else 5
                sow_day = sarra_config.sowing_search_day if sarra_config else 1
                opportunistic_year = self.config.temporal.start_year - 1
                itk_params["DateSemis"] = f"{opportunistic_year}-{sow_month}-{sow_day}"

        # Error if no source available
        if itk_params is None:
            raise ValueError(
                "No ITK parameters available. Please specify either:\n"
                "  - management: generic management config\n"
                "  - itk_template_file: explicit path to YAML file\n"
                "  - templates_dir + itk_template: template directory and name"
            )

        # Handle NaN values (YAML loaded as string '.nan')
        if itk_params.get("NI") == ".nan":
            itk_params["NI"] = float('nan')

        output_path = self.output_dir / "parameters" / "itk.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(itk_params, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated itk.yaml: {output_path}")
        return output_path

    def _generate_soil_yaml(self, soil_data: Any) -> Path:
        """Generate soil.yaml from generic config or template.

        Priority order:
        1. Generic config (soil_config) - platform agnostic
        2. Template file (soil_template_file or templates_dir)
        3. Error - no soil source available

        Soil data from SoilProfile objects can override computed values.

        Args:
            soil_data: Either Dict[int, SoilProfile] or dict with 'isda' data

        Returns:
            Path to soil.yaml
        """
        soil_params = None

        # Priority 1: Use generic mapping if soil_config available
        if self.config.soil_config is not None:
            soil_params = self._map_generic_to_sarra_py_soil()
            logger.info("Generated soil params from generic config (soil_config)")

        # Priority 2: Try to load from template
        if soil_params is None:
            soil_params = self._load_template('soil')
            if soil_params:
                logger.info("Loaded soil params from template")

        # Error if no source available
        if soil_params is None:
            raise ValueError(
                "No soil parameters available. Please specify either:\n"
                "  - soil_config: generic soil configuration\n"
                "  - soil_template_file: explicit path to YAML file\n"
                "  - templates_dir + soil_template: template directory and name"
            )

        # Override with computed values from SoilProfile objects if available
        if isinstance(soil_data, dict) and 'isda' not in soil_data:
            depths = []
            fc_values = []
            wp_values = []

            for key, value in soil_data.items():
                if hasattr(value, 'total_depth') and hasattr(value, 'layers'):
                    # This is a SoilProfile
                    default_depth = 1.5
                    if self.config.soil_config:
                        default_depth = self.config.soil_config.default_depth_m
                    elif self.config.platform_config and self.config.platform_config.sarra_py:
                        default_depth = self.config.platform_config.sarra_py.default_soil_depth_m

                    if value.total_depth:
                        depths.append(value.total_depth * 1000)  # m to mm
                    else:
                        depths.append(default_depth * 1000)
                    for layer in value.layers:
                        if layer.field_capacity:
                            fc_values.append(layer.field_capacity)
                        if layer.wilting_point:
                            wp_values.append(layer.wilting_point)

            # Override with computed values from soil profiles
            if depths:
                total_depth = float(np.mean(depths))
                soil_params["epaisseurSurf"] = float(min(200.0, total_depth * 0.15))
                soil_params["epaisseurProf"] = float(total_depth - soil_params["epaisseurSurf"])

            if fc_values:
                soil_params["HumFC"] = float(np.mean(fc_values))
            if wp_values:
                soil_params["HumPF"] = float(np.mean(wp_values))

            # Compute RU (plant available water)
            if fc_values and wp_values:
                paw = float(np.mean(fc_values) - np.mean(wp_values))
                total_depth_m = (soil_params["epaisseurSurf"] + soil_params["epaisseurProf"]) / 1000
                soil_params["ru"] = float(paw * total_depth_m * 1000)  # mm

        # Ensure all values are Python floats (not numpy)
        soil_params = {k: float(v) for k, v in soil_params.items()}

        output_path = self.output_dir / "parameters" / "soil.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(soil_params, f, default_flow_style=False, sort_keys=False)

        # Also save to data/soil/
        soil_data_path = self.output_dir / "data" / "soil" / "soil_params.yaml"
        with open(soil_data_path, 'w') as f:
            yaml.dump(soil_params, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated soil.yaml: {output_path}")
        return output_path

    def _generate_package_files(
        self,
        data: UnifiedData,
        output_files: List[Path]
    ) -> List[Path]:
        """Generate manifest.json, provenance.json, and README.md.

        Args:
            data: UnifiedData
            output_files: List of already generated output files

        Returns:
            List of generated package file paths
        """
        from prismpy.packaging import (
            create_manifest,
            save_manifest,
            generate_readme,
        )

        package_files = []

        # Build project config for manifest
        sarra_bounds = data.region.bounds.to_sarra_py_format()
        gis_bounds = [
            data.region.bounds.minx,
            data.region.bounds.miny,
            data.region.bounds.maxx,
            data.region.bounds.maxy,
        ]

        # Resolved-source discriminator: read the runtime boundary
        # source recorded on the Region (post-fallback at retrieve)
        # and honor the configured GADM admin level only under GADM.
        # Reading from the runtime ``Region.gadm_level`` here (the
        # previous behavior) bypassed the resolved-source path —
        # the runtime ``gadm_level`` is coerced to an integer for
        # cache-path string formatting and therefore lies under
        # MANUAL or SHAPEFILE configurations. Manifest derivation
        # uses the discriminated value below; cache paths continue
        # to read the runtime field unchanged.
        from prismpy.packaging.manifest import derive_boundary_label
        boundary_config = self.config.region.boundary
        resolved_boundary_source = (
            getattr(data.region, 'boundary_source', None)
            or boundary_config.source.value
        )
        manifest_gadm_level = (
            boundary_config.gadm_level
            if resolved_boundary_source == 'gadm' else None
        )
        boundary_label, _ = derive_boundary_label(
            resolved_boundary_source, manifest_gadm_level,
        )

        project_config = {
            "project_name": self.config.project.name,
            "region_name": data.region.name,
            "country": data.region.country,
            "gadm_level": manifest_gadm_level,
            "crop_name": self.config.crop.name,
            "planting_doy": self.config.crop.calendar.planting_doy if self.config.crop.calendar else None,
            "maturity_doy": self.config.crop.calendar.maturity_doy if self.config.crop.calendar else None,
            "start_year": self.config.temporal.start_year,
            "end_year": self.config.temporal.end_year,
            "spinup_years": self.config.temporal.spinup_years,
            "bounds_sarra_py": sarra_bounds,
            "bounds_gis": gis_bounds,
            "data_sources": {
                "boundaries": boundary_label,
                "rainfall": "TAMSAT v3.1",
                "temperature": "AgERA5",
                "soil": "iSDA",
                "crop_parameters": "SARRA-Py defaults",
            },
            "package_name": f"{data.region.name.lower()}_{self.config.crop.name.lower()}_sarra_py_package",
        }

        # V2-20: Legacy System B provenance.json generation deleted.
        # Provenance is now handled by System A (prismpy.provenance.tracker)
        # and distributed via executor._execute_package.

        # 2. Generate README.md
        readme_path = self.output_dir / "README.md"
        generate_readme(readme_path, project_config, platform="sarra_py")
        package_files.append(readme_path)
        logger.info(f"Generated README.md: {readme_path}")

        # 3. Generate manifest.json (must be last - needs all files)
        manifest = create_manifest(self.output_dir, project_config, platform="sarra_py")
        manifest_path = self.output_dir / "manifest.json"
        save_manifest(manifest, manifest_path)
        package_files.append(manifest_path)
        logger.info(f"Generated manifest.json: {manifest_path}")

        return package_files

    def _generate_validation_report(self, data: UnifiedData) -> Path:
        """Generate validation/validation_report.json.

        Args:
            data: UnifiedData

        Returns:
            Path to validation_report.json
        """
        validation_results = {
            "valid": True,
            "checked_at": datetime.now().isoformat(),
            "package_structure": {
                "readme": (self.output_dir / "README.md").exists(),
                "manifest": (self.output_dir / "manifest.json").exists(),
                "provenance": (self.output_dir / "provenance.json").exists(),
                "bounds": (self.output_dir / "data" / "boundaries" / "bounds.json").exists(),
                "variety": (self.output_dir / "parameters" / "variety.yaml").exists(),
                "itk": (self.output_dir / "parameters" / "itk.yaml").exists(),
                "soil": (self.output_dir / "parameters" / "soil.yaml").exists(),
            },
            "climate_directories": {},
            "file_counts": {},
        }

        # Check climate subdirectories
        climate_vars = [
            "rainfall",
            "2m_temperature_24_hour_maximum",
            "2m_temperature_24_hour_mean",
            "2m_temperature_24_hour_minimum",
            "ET0Hargeaves",
            "solar_radiation_flux_daily",
        ]

        for var in climate_vars:
            var_path = self.output_dir / "data" / "climate" / var
            if var_path.exists():
                files = list(var_path.glob("*"))
                validation_results["climate_directories"][var] = len(files)
            else:
                validation_results["climate_directories"][var] = 0
                validation_results["valid"] = False

        # Count total files
        total_files = sum(1 for _ in self.output_dir.rglob("*") if _.is_file())
        validation_results["file_counts"]["total"] = total_files

        output_path = self.output_dir / "validation" / "validation_report.json"
        with open(output_path, 'w') as f:
            json.dump(validation_results, f, indent=2)

        logger.info(f"Generated validation_report.json: {output_path}")
        return output_path
