"""
ACEA translator for prismpy.

This module translates unified data to ACEA (AquaCrop) model input format:
- Python project configuration class (project_conf)
- Climate pickle files: (tmax, tmin, prec, et0) tuples
- Soil data as CSV
- Crop calendar and parameters

ACEA Quirks (from analysis):
1. Class name MUST be 'project_conf' (NOT 'ACEA_Config')
2. All attributes MUST be class-level (NOT in __init__)
3. crop_model MUST be exactly 'AquaCrop' (NOT 'AquaCrop_v6')
4. gridcells MUST contain 30-arcmin cell IDs (NOT 5-arcmin)
5. Climate pickle format: tuple of (tmax, tmin, prec, et0) numpy arrays
6. Resolution: 0=30arcmin, 1=5arcmin (confusing!)

Reference: ACEA/09-PROJECT-CONFIG-ASSEMBLY/assemble_config.py
Reference: ACEA/02-CLIMATE-PREPARATION/create_climate_pickles.py
"""

import json
import logging
import pickle
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prismpy.config.schema import Platform
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.crop import CropParameters, CropCalendar
from prismpy.models.region import Region
from prismpy.models.soil import SoilProfile
from prismpy.models.spatial import SpatialGrid, GridCell
from prismpy.provenance.tracker import DecisionType, ProvenanceTracker
from prismpy.sources.climate._cancel import PipelineCancelled, raise_if_cancelled
from prismpy.translators.base import (
    BaseTranslator,
    AceaTranslatorBase,
    TranslationResult,
    UnifiedData,
)


logger = logging.getLogger(__name__)


# Maximum valid 30-arcmin cell ID
MAX_30ARCMIN_ID = 360 * 720 - 1  # 259,199


# ACEA internal crop codes
# Source: ACEA/modules/acea/acea_crops.py (crop_code == XX patterns)
# NOTE: These are ACEA's internal codes, NOT standard FAO codes!
# Standard FAO codes differ (e.g., Wheat is 41 in FAO but 15 in ACEA)
ACEA_FAO_CODE_MAP = {
    'Maize': 56, 'Corn': 56,
    'Wheat': 15,  # ACEA internal (not FAO 41)
    'Rice': 27,
    'Barley': 44,
    'Oats': 75,
    'Millet': 79, 'Pearl Millet': 79, 'Finger Millet': 79,  # ACEA internal (not FAO 33)
    'Sorghum': 83,  # ACEA internal (not FAO 52)
    'Cereals': 108,
    'Potato': 116,
    'Sweet Potato': 122,
    'Cassava': 125,
    'Taro': 136,
    'Yams': 137,
    'Sugarcane': 156,
    'Sugar Beet': 157,
    'Bean': 176, 'Beans': 176,
    'Pea': 187,
    'Chickpea': 191,
    'Cowpea': 195,
    'Pigeon Pea': 197,
    'Lentil': 201,
    'Pulses': 211,
    'Soybean': 236, 'Soya': 236,  # ACEA internal (not FAO 79)
    'Groundnut': 242, 'Peanut': 242,  # ACEA internal (not FAO 74)
    'Sunflower': 267,  # ACEA internal (not FAO 80)
    'Rapeseed': 270, 'Canola': 270,
    'Sesame': 289,
    'Cotton': 328,  # ACEA internal (not FAO 89)
    'Oilseeds': 339,
    'Tomato': 388,
    'Pepper': 401,
    'Onion': 403,
    'Vegetables': 463,
    'Banana': 486,
    'Plantains': 489,
}


# Crop name to SPAM code mapping (for harvested area data)
# SPAM codes are 4-letter uppercase identifiers
SPAM_CODE_MAP = {
    'Maize': 'MAIZ', 'Corn': 'MAIZ',
    'Wheat': 'WHEA',
    'Rice': 'RICE',
    'Sorghum': 'SORG',
    'Millet': 'PMIL', 'Pearl Millet': 'PMIL', 'Finger Millet': 'SMIL',
    'Barley': 'BARL',
    'Cassava': 'CASS',
    'Potato': 'POTA',
    'Sweet Potato': 'SWPO',
    'Soybean': 'SOYB', 'Soya': 'SOYB',
    'Groundnut': 'GROU', 'Peanut': 'GROU',
    'Cotton': 'COTT',
    'Sugarcane': 'SUGC',
    'Sunflower': 'SUNF',
    'Bean': 'BEAN', 'Beans': 'BEAN',
    'Cowpea': 'COWP',
    'Chickpea': 'CHIC',
    'Lentil': 'LENT',
    'Pigeon Pea': 'PIGE',
}


# Crop-specific GDD defaults for ACEA
# Values derived from AquaCrop documentation and crop literature
CROP_GDD_DEFAULTS = {
    'Maize': {
        'gdd_emergence': 90, 'gdd_max_root': 700, 'gdd_senescence': 1200,
        'gdd_maturity': 1600, 'gdd_yield_form': 900,
        'gdd_duration_flowering': 200, 'gdd_duration_yield_form': 500,
        'cgc': 0.012, 'cdc': 0.004, 'base_temp': 8.0,
    },
    'Corn': {  # Alias for Maize
        'gdd_emergence': 90, 'gdd_max_root': 700, 'gdd_senescence': 1200,
        'gdd_maturity': 1600, 'gdd_yield_form': 900,
        'gdd_duration_flowering': 200, 'gdd_duration_yield_form': 500,
        'cgc': 0.012, 'cdc': 0.004, 'base_temp': 8.0,
    },
    'Wheat': {
        'gdd_emergence': 80, 'gdd_max_root': 500, 'gdd_senescence': 900,
        'gdd_maturity': 1100, 'gdd_yield_form': 700,
        'gdd_duration_flowering': 150, 'gdd_duration_yield_form': 400,
        'cgc': 0.010, 'cdc': 0.005, 'base_temp': 0.0,
    },
    'Rice': {
        'gdd_emergence': 100, 'gdd_max_root': 600, 'gdd_senescence': 1000,
        'gdd_maturity': 1300, 'gdd_yield_form': 800,
        'gdd_duration_flowering': 180, 'gdd_duration_yield_form': 450,
        'cgc': 0.011, 'cdc': 0.004, 'base_temp': 10.0,
    },
    'Sorghum': {
        'gdd_emergence': 85, 'gdd_max_root': 650, 'gdd_senescence': 1100,
        'gdd_maturity': 1500, 'gdd_yield_form': 850,
        'gdd_duration_flowering': 180, 'gdd_duration_yield_form': 480,
        'cgc': 0.011, 'cdc': 0.004, 'base_temp': 10.0,
    },
    'Millet': {
        'gdd_emergence': 70, 'gdd_max_root': 500, 'gdd_senescence': 900,
        'gdd_maturity': 1200, 'gdd_yield_form': 700,
        'gdd_duration_flowering': 150, 'gdd_duration_yield_form': 400,
        'cgc': 0.010, 'cdc': 0.004, 'base_temp': 10.0,
    },
    'Pearl Millet': {
        'gdd_emergence': 70, 'gdd_max_root': 500, 'gdd_senescence': 900,
        'gdd_maturity': 1200, 'gdd_yield_form': 700,
        'gdd_duration_flowering': 150, 'gdd_duration_yield_form': 400,
        'cgc': 0.010, 'cdc': 0.004, 'base_temp': 10.0,
    },
    'Barley': {
        'gdd_emergence': 75, 'gdd_max_root': 480, 'gdd_senescence': 850,
        'gdd_maturity': 1050, 'gdd_yield_form': 680,
        'gdd_duration_flowering': 140, 'gdd_duration_yield_form': 380,
        'cgc': 0.010, 'cdc': 0.005, 'base_temp': 0.0,
    },
    'Soybean': {
        'gdd_emergence': 80, 'gdd_max_root': 600, 'gdd_senescence': 1000,
        'gdd_maturity': 1350, 'gdd_yield_form': 750,
        'gdd_duration_flowering': 200, 'gdd_duration_yield_form': 450,
        'cgc': 0.011, 'cdc': 0.004, 'base_temp': 10.0,
    },
    'Cotton': {
        'gdd_emergence': 100, 'gdd_max_root': 800, 'gdd_senescence': 1400,
        'gdd_maturity': 1800, 'gdd_yield_form': 1000,
        'gdd_duration_flowering': 250, 'gdd_duration_yield_form': 600,
        'cgc': 0.010, 'cdc': 0.003, 'base_temp': 12.0,
    },
    'Groundnut': {
        'gdd_emergence': 75, 'gdd_max_root': 550, 'gdd_senescence': 950,
        'gdd_maturity': 1250, 'gdd_yield_form': 700,
        'gdd_duration_flowering': 180, 'gdd_duration_yield_form': 420,
        'cgc': 0.011, 'cdc': 0.004, 'base_temp': 10.0,
    },
    'Cassava': {
        'gdd_emergence': 150, 'gdd_max_root': 1500, 'gdd_senescence': 3500,
        'gdd_maturity': 4500, 'gdd_yield_form': 2000,
        'gdd_duration_flowering': 0, 'gdd_duration_yield_form': 2000,
        'cgc': 0.008, 'cdc': 0.002, 'base_temp': 15.0,
    },
    'Potato': {
        'gdd_emergence': 60, 'gdd_max_root': 400, 'gdd_senescence': 800,
        'gdd_maturity': 1000, 'gdd_yield_form': 500,
        'gdd_duration_flowering': 150, 'gdd_duration_yield_form': 400,
        'cgc': 0.012, 'cdc': 0.005, 'base_temp': 5.0,
    },
    'Sugarcane': {
        'gdd_emergence': 200, 'gdd_max_root': 2000, 'gdd_senescence': 4000,
        'gdd_maturity': 5000, 'gdd_yield_form': 2500,
        'gdd_duration_flowering': 0, 'gdd_duration_yield_form': 2000,
        'cgc': 0.006, 'cdc': 0.002, 'base_temp': 12.0,
    },
}


# ACEA project_conf template
ACEA_CONFIG_TEMPLATE = '''# -*- coding: utf-8 -*-
"""
ACEA Project Configuration: {project_name}

Generated: {timestamp}
Generator: prismpy

Region: {region_name}, {country}
Crop: {crop_name}
Period: {start_year}-{end_year}

DO NOT change the class name 'project_conf' - ACEA expects this exact name!
"""


class project_conf:
    # ==================================================================
    # GENERAL SETTINGS
    # ==================================================================
    project_name = '{project_name}'
    crop_model = 'AquaCrop'
    scenarios = {scenarios}
    multi_core = {multi_core}
    CPUs = {cpus}

    # ==================================================================
    # SCOPE - SPATIAL AND TEMPORAL
    # ==================================================================
    resolution = {resolution}
    gridcells = {gridcells}

    # Temporal scope (ACEA requires date strings)
    clock_start = '{clock_start}'
    clock_end = '{clock_end}'
    spinup = {spinup}

    # Self-contained mode: run all cells without SPAM dependency
    real_cropland = False
    landuse = '{landuse}'

    # ==================================================================
    # CROP SETTINGS
    # ==================================================================
    crop_name = '{crop_name}'
    crop_name_short = '{crop_name_short}'
    crop_fao = {crop_fao}
    crop_gs_increase = 15
    crop_perennial = False
    crop_phenology = False

    # GAEZ settings for soil fertility calibration
    crop_name_4code = '{crop_name_4code}'
    crop_gaez = {crop_gaez}
    LAI_CC_a = 1.005
    LAI_CC_b = -0.6
    LAI_CC_c = 1.2
    get_GAEZ_ccx_hi = True

    # ==================================================================
    # CLIMATE SETTINGS
    # ==================================================================
    climate_name = '{climate_name}'
    climate_start = {climate_start}
    climate_end = {climate_end}
    co2_name = '{co2_name}'

    # ==================================================================
    # SOIL SETTINGS
    # ==================================================================
    soil_dz = [.1, .1, .1, .3, .4, .6, .7, .7]
    init_wc = 50
    use_soilgrids = 0  # 0 = HWSD, 1 = SoilGrids
    gw_max_level = -1.
    gw_min_level = -3.

    # ==================================================================
    # FIELD MANAGEMENT SETTINGS
    # ==================================================================
    off_season = True
    irr_thresholds = {irr_thresholds}
    bunds = {bunds}
    bunds_dz = {bunds_dz}
    mulching = False
    mulching_area = 0.9
    mulching_factor = 0.8

    # ==================================================================
    # RESULT MANAGEMENT
    # ==================================================================
    project_save_annual = True
    project_save_daily = False

    # ==================================================================
    # ADDITIONAL SETTINGS
    # ==================================================================
    rerun_simulated_cells = True
    virtual_irrigation = '{virtual_irrigation}'
    soil_fertility = {soil_fertility}
    tuned = 0
'''


class AceaTranslator(AceaTranslatorBase):
    """Translator for ACEA (AquaCrop Environment Analysis) system.

    ACEA runs AquaCrop simulations across spatial grids using
    preprocessed climate, soil, and crop calendar data.

    Generates:
    1. climate/*.pckl - Climate pickle files per cell
    2. soil/soil_data.csv - Soil properties
    3. crop_calendar/calendar.csv - Crop calendar per cell
    4. crop_params/params.yaml - Crop parameters
    5. harvested_areas/areas.csv - Harvested area per cell
    6. config/{project}_config.py - Main ACEA configuration class

    Output structure:
        output_dir/
        ├── climate/
        │   └── {climate_name}_{cell_id}.pckl
        ├── soil/
        │   └── soil_data.csv
        ├── crop_calendar/
        │   └── calendar.csv
        ├── crop_params/
        │   └── params.yaml
        ├── harvested_areas/
        │   └── areas.csv
        └── config/
            └── {project}_config.py
    """

    def translate(self, data: UnifiedData) -> TranslationResult:
        """Translate unified data to ACEA format.

        Args:
            data: UnifiedData container with all required data

        Returns:
            TranslationResult with output files and status
        """
        self.log_translation_start(data)
        errors = []
        warnings = []
        output_files = []

        # Validate input data
        input_errors = self.validate_input_data(data)
        if input_errors:
            return self.create_result(
                success=False,
                output_files=[],
                errors=input_errors,
            )

        # Create output subdirectories
        for subdir in self.OUTPUT_SUBDIRS:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        try:
            # Compute 30-arcmin cell IDs (stored for PACKAGE stage)
            cell_ids_30arcmin = self._compute_30arcmin_cell_ids(data.grid)
            self._cell_ids_30arcmin = cell_ids_30arcmin

            # Validate cell IDs are 30-arcmin
            if not self._validate_30arcmin_ids(cell_ids_30arcmin):
                warnings.append("Some cell IDs exceed 30-arcmin maximum. "
                              "Verify grid resolution.")

            # Generate climate_name early (stored for PACKAGE stage)
            region_name = data.region.name if data.region else "region"
            climate_name = f"{region_name.lower().replace(' ', '_')}_nasapower"
            self._climate_name = climate_name

            # 1. Generate climate pickle files
            # Check if we have enough climate data for all cells
            n_climate = len(data.climate) if data.climate else 0
            n_cells = len(cell_ids_30arcmin)

            climate_data = data.climate or {}

            # Check if climate download is enabled
            platform_config = self.get_platform_config()
            download_enabled = True
            download_delay = 2.0
            if platform_config:
                download_enabled = getattr(platform_config, 'download_climate', True)
                download_delay = getattr(platform_config, 'climate_download_delay', 2.0)

            if n_climate < n_cells and data.grid and download_enabled:
                # Download NASA POWER climate data for missing cells
                logger.info(f"Insufficient climate data ({n_climate}/{n_cells} cells), "
                           f"downloading from NASA POWER...")

                # Get date range from config (cross-year-aware)
                start_year = self.config.temporal.start_year
                end_year = self.config.temporal.end_year
                spinup = self.config.temporal.spinup_years
                start_date = date(start_year - spinup, 1, 1)
                crop_cal = self.config.crop.calendar if self.config.crop else None
                end_date = self.config.temporal.get_climate_end_date(crop_cal)

                # Download climate for unique 30-arcmin cells only (not all 5-arcmin).
                # NASA POWER's native resolution is ~0.5° (30-arcmin), so multiple
                # 5-arcmin cells within the same 30-arcmin tile get identical data.
                # This reduces API calls from ~244 to ~17 for Saint-Louis.
                # Wire progress callback for substage reporting
                def _acea_progress(current, total):
                    cb = getattr(self, 'progress_callback', None)
                    if cb and hasattr(cb, 'on_substage_progress'):
                        cb.on_substage_progress(
                            'translate',
                            'Downloading climate from NASA POWER',
                            current, total,
                            f'cell {current} of {total}',
                        )
                downloaded_climate = self._download_climate_30arcmin(
                    data.grid, cell_ids_30arcmin, start_date, end_date,
                    request_delay=download_delay,
                    progress_callback=_acea_progress,
                )

                # Merge with existing climate data
                if downloaded_climate:
                    climate_data = {**climate_data, **downloaded_climate}
                    logger.info(f"Climate data now has {len(climate_data)} locations")
            elif n_climate < n_cells:
                logger.warning(f"Insufficient climate data ({n_climate}/{n_cells} cells) "
                             f"and download_climate is disabled")

            if climate_data:
                climate_files = self._generate_climate_pickles(
                    climate_data, cell_ids_30arcmin, climate_name
                )
                output_files.extend(climate_files)

                # F13 — surface per-cell climate onto the shared
                # UnifiedData so the cell-summary writer and per-cell
                # coverage validators observe the actual climate-loaded
                # state. ``_download_climate_30arcmin`` already maps
                # the 30-arcmin tile downloads back to 5-arcmin
                # ``cell.cell_id`` keys (see the post-loop fanout at
                # the bottom of that method), so ``climate_data`` is
                # already 5-arcmin keyed by the time we get here. Pass
                # it straight to the helper — re-fanning out via tile
                # IDs would double-map and look up tile_ids in a
                # 5-arcmin-keyed dict, returning None for every cell.
                self._surface_per_cell_climate(data, climate_data)

            # 2. Generate soil data (ACEA-compatible NetCDF)
            # ACEA requires soil data in NetCDF format (HWSD_soil_data_on_cropland_v2.3.nc)
            # We generate this from HWSD to make packages work on any ACEA installation
            soil_nc_file = None
            hwsd_bil = None
            hwsd_mdb = None
            include_soil = True

            if platform_config:
                hwsd_bil = getattr(platform_config, 'hwsd_bil_path', None)
                hwsd_mdb = getattr(platform_config, 'hwsd_mdb_path', None)
                include_soil = getattr(platform_config, 'include_soil_in_package', True)

            if hwsd_bil and hwsd_mdb and include_soil:
                logger.info("Generating ACEA-compatible soil NetCDF from HWSD...")
                soil_nc_file = self._generate_acea_soil_netcdf(
                    cell_ids_30arcmin,
                    Path(hwsd_bil), Path(hwsd_mdb)
                )
                if soil_nc_file:
                    output_files.append(soil_nc_file)
            elif include_soil and data.soil:
                # Fallback: generate NetCDF from already-retrieved soil profiles
                # This covers cases where HWSD BIL/MDB paths aren't configured
                logger.info("Generating ACEA soil NetCDF from retrieved soil profiles...")
                soil_nc_file = self._generate_soil_netcdf_from_profiles(
                    data.soil, cell_ids_30arcmin
                )
                if soil_nc_file:
                    output_files.append(soil_nc_file)

            # Also generate CSV for reference/debugging (optional)
            if data.soil:
                soil_csv = self._generate_soil_csv(data.soil, cell_ids_30arcmin)
                output_files.append(soil_csv)

            # 3. Generate crop calendar (NetCDF format for ACEA)
            crop_short = self.config.crop.name_short
            if data.crop_calendar:
                # Generate NetCDF crop calendar (ACEA's expected format)
                calendar_nc_files = self._generate_crop_calendar_nc(
                    data.crop_calendar, cell_ids_30arcmin, crop_short
                )
                output_files.extend(calendar_nc_files)

                # Also generate CSV for reference/debugging
                calendar_csv = self._generate_crop_calendar(
                    data.crop_calendar, cell_ids_30arcmin
                )
                output_files.append(calendar_csv)

            # 4. Generate crop parameters (NetCDF format for ACEA)
            if data.crop_params:
                # Generate NetCDF crop params (ACEA's expected format)
                params_nc_files = self._generate_crop_params_nc(
                    data.crop_params, cell_ids_30arcmin, crop_short
                )
                output_files.extend(params_nc_files)

                # Also generate YAML for reference/debugging
                params_yaml = self._generate_crop_params(data.crop_params)
                output_files.append(params_yaml)

            # 5. Generate CO2 data file (embedded historical data)
            co2_file = self._generate_co2_data()
            output_files.append(co2_file)

            # 6. Generate harvested areas (SPAM data) - REQUIRED by ACEA
            # ACEA always loads SPAM files during initialization, even with real_cropland=False
            spam_files = []
            spam_dir = None
            spam_required = False
            include_spam = True

            if platform_config:
                spam_dir = getattr(platform_config, 'spam_data_dir', None)
                spam_required = getattr(platform_config, 'spam_required', False)
                include_spam = getattr(platform_config, 'include_spam_in_package', True)

            if include_spam and data.region:
                if spam_dir:
                    # User provided SPAM directory - clip the files
                    logger.info("Clipping SPAM harvested area data...")
                    spam_files = self._clip_spam_data(
                        data.region,
                        self.config.crop.name,
                        Path(spam_dir),
                    )

                    # Check if clipping succeeded
                    if not spam_files and spam_required:
                        raise ValueError(
                            f"SPAM files not found in {spam_dir} for crop {self.config.crop.name}. "
                            f"ACEA requires SPAM harvested area files. Either:\n"
                            f"  1. Provide correct spam_data_dir path with SPAM files\n"
                            f"  2. Set spam_required=False to generate dummy files"
                        )
                    elif not spam_files:
                        # Clipping failed but not required - generate dummy files
                        logger.warning(f"SPAM files not found in {spam_dir}, generating dummy files")
                        spam_files = self._generate_dummy_spam_files(data.region, self.config.crop.name)

                elif spam_required:
                    # SPAM required but no directory provided
                    raise ValueError(
                        f"SPAM data is required (spam_required=True) but spam_data_dir not provided. "
                        f"ACEA requires SPAM harvested area files. Either:\n"
                        f"  1. Provide spam_data_dir path with SPAM files\n"
                        f"  2. Set spam_required=False to generate dummy files"
                    )
                else:
                    # Not required and not provided - generate dummy files for self-contained package
                    logger.info("Generating dummy SPAM files for self-contained package...")
                    spam_files = self._generate_dummy_spam_files(data.region, self.config.crop.name)

                output_files.extend(spam_files)

            # 7. Handle GAEZ data (download/copy) if configured
            if platform_config:
                gaez_dir = getattr(platform_config, 'gaez_data_dir', None)
                gaez_auto = getattr(platform_config, 'gaez_auto_download', True)
                include_gaez = getattr(platform_config, 'include_gaez_in_package', True)

                if include_gaez and (gaez_dir or gaez_auto):
                    logger.info("Handling GAEZ crop suitability data...")
                    gaez_files = self._handle_gaez_data(
                        crop_name=self.config.crop.name,
                        gaez_data_dir=Path(gaez_dir) if gaez_dir else None,
                        auto_download=gaez_auto,
                    )
                    output_files.extend(gaez_files)

            # 8. Generate ACEA configuration Python file
            config_file = self._generate_acea_config(data, cell_ids_30arcmin, climate_name)
            output_files.append(config_file)

            # 9. Generate install script for package
            install_script = self._generate_install_script(data)
            output_files.append(install_script)

            # 10. Validate outputs
            validation_errors = self.validate_outputs()
            if validation_errors:
                warnings.extend(validation_errors)

        except PipelineCancelled:
            # V2-22b L Gate B round 3: translate() outer-try carve-out.
            raise
        except Exception as e:
            logger.error(f"ACEA translation failed: {e}")
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
                description=f"Generated ACEA inputs for {data.region.name}",
                rationale="ACEA requires project_conf class with pickle climate files",
                alternatives=["manual configuration"],
                reference="prismpy.translators.acea.translator.translate",
            )

        result = self.create_result(
            success=True,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "region": data.region.name,
                "n_cells": len(cell_ids_30arcmin),
                "n_climate_pickles": len(data.climate) if data.climate else 0,
                "cell_id_range": [min(cell_ids_30arcmin), max(cell_ids_30arcmin)]
                    if cell_ids_30arcmin else [0, 0],
            },
        )

        self.log_translation_complete(result)
        return result

    def validate_outputs(self) -> List[str]:
        """Validate generated ACEA outputs.

        Returns:
            List of validation error messages
        """
        errors = []

        # Check config file exists
        config_dir = self.output_dir / "config"
        config_files = list(config_dir.glob("*_config.py"))
        if not config_files:
            errors.append("No configuration file generated in config/")
        else:
            # Validate Python config structure
            config_path = config_files[0]
            try:
                with open(config_path, 'r') as f:
                    content = f.read()

                if "class project_conf:" not in content:
                    errors.append("Config file missing 'class project_conf:'")
                if "crop_model = 'AquaCrop'" not in content:
                    errors.append("Config file missing correct crop_model")
                if "gridcells = " not in content:
                    errors.append("Config file missing gridcells attribute")

            except Exception as e:
                errors.append(f"Error reading config file: {e}")

        # Check climate directory has pickle files
        climate_dir = self.output_dir / "climate"
        if climate_dir.exists():
            pickle_files = list(climate_dir.glob("*.pckl"))
            if not pickle_files:
                errors.append("No climate pickle files generated")

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
        cell_ids = getattr(self, '_cell_ids_30arcmin', [])
        climate_name = getattr(self, '_climate_name', 'region_nasapower')
        return self._generate_package_metadata(
            data, cell_ids, climate_name, output_files
        )

    def _compute_30arcmin_cell_ids(self, grid: Optional[SpatialGrid]) -> List[int]:
        """Compute UNIQUE 30-arcmin cell IDs from grid.

        ACEA uses 30-arcmin global grid:
        - Row 0 at lat 90, Col 0 at lon -180
        - Cell ID = row * 720 + col

        Multiple 5-arcmin cells may map to the same 30-arcmin cell,
        so we return unique IDs only (sorted for consistency).

        Args:
            grid: SpatialGrid with cells

        Returns:
            List of unique 30-arcmin cell IDs (sorted)
        """
        if not grid or not grid.cells:
            return []

        cell_ids_set = set()
        resolution = 30 / 60  # 30 arcmin in degrees

        for cell in grid.cells:
            row = int((90 - cell.lat) / resolution)
            col = int((cell.lon + 180) / resolution)

            # Clamp to valid range
            row = max(0, min(row, self.GRID_ROWS_30ARCMIN - 1))
            col = max(0, min(col, self.GRID_COLS_30ARCMIN - 1))

            cell_id = row * self.GRID_COLS_30ARCMIN + col
            cell_ids_set.add(cell_id)

        # Return sorted list of unique IDs
        return sorted(cell_ids_set)

    def _validate_30arcmin_ids(self, cell_ids: List[int]) -> bool:
        """Validate that all cell IDs are valid 30-arcmin IDs.

        Args:
            cell_ids: List of cell IDs

        Returns:
            True if all IDs are valid
        """
        if not cell_ids:
            return True

        max_id = max(cell_ids)
        return max_id <= MAX_30ARCMIN_ID

    def _cell_id_to_row_col(self, cell_id: int) -> Tuple[int, int]:
        """Convert 30-arcmin cell ID to row/col indices for NetCDF.

        ACEA uses 30-arcmin global grid:
        - 360 rows (lat from 90°N to -90°S)
        - 720 cols (lon from -180° to 180°)
        - Cell ID = row * 720 + col

        Args:
            cell_id: 30-arcmin cell ID

        Returns:
            Tuple of (row, col) for NetCDF indexing
        """
        row = cell_id // self.GRID_COLS_30ARCMIN
        col = cell_id % self.GRID_COLS_30ARCMIN
        return (row, col)

    def _generate_climate_pickles(
        self,
        climate_data: Dict[int, ClimateTimeSeries],
        cell_ids_30arcmin: List[int],
        climate_name: str = "climate",
    ) -> List[Path]:
        """Generate ACEA climate pickle files.

        ACEA format: pickle.dump((tmax, tmin, prec, et0), f)
        Each element is a numpy float32 array of daily values.

        Args:
            climate_data: Dictionary of location_id to ClimateTimeSeries
            cell_ids_30arcmin: List of 30-arcmin cell IDs
            climate_name: Name prefix for climate pickle files

        Returns:
            List of generated pickle file paths
        """
        output_files = []
        climate_dir = self.output_dir / "climate"

        # Map 5-arcmin to 30-arcmin if needed
        id_mapping = self._create_id_mapping(climate_data.keys(), cell_ids_30arcmin)

        for loc_id, ts in climate_data.items():
            # Get 30-arcmin cell ID
            cell_id_30 = id_mapping.get(loc_id, loc_id)

            # Skip placeholder cells (negative IDs)
            if loc_id < 0:
                logger.debug(f"Skipping placeholder climate data (ID={loc_id})")
                continue

            # Extract arrays
            tmax = np.array([r.tmax for r in ts.records], dtype=np.float32)
            tmin = np.array([r.tmin for r in ts.records], dtype=np.float32)
            prec = np.array([r.precip for r in ts.records], dtype=np.float32)

            # ET0 - use provided or compute simple estimate
            et0_values = []
            for r in ts.records:
                if r.et0 is not None:
                    et0_values.append(r.et0)
                else:
                    # Simple Hargreaves-Samani estimate
                    et0 = self._estimate_et0_hargreaves(
                        r.tmax, r.tmin, ts.lat, r.doy
                    )
                    et0_values.append(et0)
            et0 = np.array(et0_values, dtype=np.float32)

            # Save pickle
            pickle_path = climate_dir / f"{climate_name}_{cell_id_30}.pckl"
            with open(pickle_path, 'wb') as f:
                pickle.dump((tmax, tmin, prec, et0), f)

            output_files.append(pickle_path)
            logger.debug(f"Generated climate pickle: {pickle_path}")

        logger.info(f"Generated {len(output_files)} ACEA climate pickles")
        return output_files

    def _estimate_et0_hargreaves(
        self,
        tmax: float,
        tmin: float,
        lat: float,
        doy: int,
    ) -> float:
        """Estimate ET0 using Hargreaves-Samani equation.

        ET0 = 0.0023 * Ra * (Tmean + 17.8) * (Tmax - Tmin)^0.5

        Args:
            tmax: Maximum temperature (°C)
            tmin: Minimum temperature (°C)
            lat: Latitude
            doy: Day of year

        Returns:
            Estimated ET0 (mm/day)
        """
        tmean = (tmax + tmin) / 2
        tdelta = max(0, tmax - tmin)

        # Extraterrestrial radiation (simplified)
        lat_rad = np.radians(lat)
        dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
        delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))

        Ra = (24 * 60 / np.pi) * 0.082 * dr * (
            ws * np.sin(lat_rad) * np.sin(delta) +
            np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )

        # Hargreaves equation
        et0 = 0.0023 * Ra * (tmean + 17.8) * np.sqrt(tdelta)

        return max(0, et0)

    def _create_id_mapping(
        self,
        source_ids: Any,
        target_ids: List[int],
    ) -> Dict[int, int]:
        """Create mapping from source IDs to 30-arcmin IDs.

        Maps 5-arcmin cell IDs to their parent 30-arcmin cell IDs based on
        spatial relationship. Each 30-arcmin cell contains a 6x6 grid of
        5-arcmin cells.

        Args:
            source_ids: Source location IDs (may be 5-arcmin cell IDs)
            target_ids: Target 30-arcmin cell IDs

        Returns:
            Dictionary mapping source_id to 30-arcmin cell_id
        """
        mapping = {}
        target_set = set(target_ids)

        # Grid constants
        COLS_5ARCMIN = 4320  # 360 degrees / (5/60) degrees
        COLS_30ARCMIN = 720  # 360 degrees / 0.5 degrees

        for src_id in source_ids:
            if src_id < 0:
                # Skip placeholder IDs
                mapping[src_id] = src_id
                continue

            # Check if source ID is in 5-arcmin range (larger numbers)
            if src_id >= 1000000:  # 5-arcmin IDs are typically in millions
                # Convert 5-arcmin cell to 30-arcmin cell
                row_5 = src_id // COLS_5ARCMIN
                col_5 = src_id % COLS_5ARCMIN

                # Each 30-arcmin cell contains 6x6 5-arcmin cells
                row_30 = row_5 // 6
                col_30 = col_5 // 6

                cell_id_30 = row_30 * COLS_30ARCMIN + col_30
            else:
                # Assume already 30-arcmin or small ID
                cell_id_30 = src_id

            # Only map if target is in our list
            if cell_id_30 in target_set:
                mapping[src_id] = cell_id_30
            else:
                # Keep original if not in target set
                mapping[src_id] = src_id

        return mapping

    def _download_nasa_power_climate(
        self,
        grid: SpatialGrid,
        start_date: date,
        end_date: date,
        request_delay: float = 2.0,
    ) -> Dict[int, ClimateTimeSeries]:
        """Download NASA POWER climate data for all grid cells.

        This method fetches daily climate data from NASA POWER API
        for each cell in the grid, making the package self-contained.

        Args:
            grid: SpatialGrid with cells (must have lat/lon)
            start_date: Start date for climate data
            end_date: End date for climate data
            request_delay: Delay between API requests (seconds)

        Returns:
            Dictionary mapping cell_id to ClimateTimeSeries
        """
        try:
            from prismpy.sources.climate.nasa_power import (
                NASAPowerSource, NASAPowerConfig
            )
        except ImportError:
            logger.error("NASAPowerSource not available")
            return {}

        logger.info(f"Downloading NASA POWER climate data for {len(grid.cells)} cells")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Configure NASA POWER source
        config = NASAPowerConfig(
            request_delay=request_delay,
            parameters=[
                "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"
            ]
        )
        source = NASAPowerSource(config)

        climate_data = {}
        total_cells = len(grid.cells)

        for i, cell in enumerate(grid.cells):
            # V2-22b L (AC L.3): per-cell cancel for the 5-arcmin loop.
            raise_if_cancelled(
                getattr(self, 'cancel_check', None),
                f"acea.5arcmin.cell={i + 1}/{total_cells}",
            )
            cell_id = cell.cell_id
            logger.info(f"Downloading climate for cell {i+1}/{total_cells} "
                       f"(lat={cell.lat:.2f}, lon={cell.lon:.2f})")

            try:
                # Retrieve climate data for this location
                result = source.retrieve(
                    lat=cell.lat,
                    lon=cell.lon,
                    start_date=start_date,
                    end_date=end_date,
                    cancel_check=getattr(self, 'cancel_check', None),
                )

                if result.success and result.data:
                    climate_data[cell_id] = result.data
                    logger.debug(f"Downloaded {len(result.data.records)} days for cell {cell_id}")
                else:
                    logger.warning(f"Failed to download climate for cell {cell_id}: "
                                 f"{result.errors}")

            except PipelineCancelled:
                # V2-22b L: 5-arcmin per-cell broad except must not swallow cancel.
                raise
            except Exception as e:
                logger.error(f"Error downloading climate for cell {cell_id}: {e}")

        logger.info(f"Downloaded climate data for {len(climate_data)}/{total_cells} cells")
        return climate_data

    def _download_climate_30arcmin(
        self,
        grid: SpatialGrid,
        cell_ids_30arcmin: List[int],
        start_date: date,
        end_date: date,
        request_delay: float = 2.0,
        progress_callback=None,
    ) -> Dict[int, 'ClimateTimeSeries']:
        """Download NASA POWER climate for unique 30-arcmin cells only.

        Maps the 5-arcmin grid to unique 30-arcmin cell centers, downloads
        once per unique cell, then assigns results to all child 5-arcmin IDs.
        This matches the reference package approach (17 downloads for
        Saint-Louis instead of 244).

        Args:
            grid: 5-arcmin SpatialGrid
            cell_ids_30arcmin: Unique 30-arcmin cell IDs
            start_date: Start date
            end_date: End date
            request_delay: Delay between API calls

        Returns:
            Climate data keyed by 5-arcmin cell IDs (for compatibility)
        """
        try:
            from prismpy.sources.climate.nasa_power import (
                NASAPowerSource, NASAPowerConfig
            )
        except ImportError:
            logger.error("NASAPowerSource not available")
            return {}

        resolution = 30 / 60  # 0.5 degrees

        # Compute 30-arcmin cell centers
        centers_30 = {}
        for cell_id in cell_ids_30arcmin:
            row = cell_id // self.GRID_COLS_30ARCMIN
            col = cell_id % self.GRID_COLS_30ARCMIN
            lat = 90 - (row + 0.5) * resolution
            lon = -180 + (col + 0.5) * resolution
            centers_30[cell_id] = (lat, lon)

        logger.info(f"Downloading NASA POWER for {len(centers_30)} unique 30-arcmin cells "
                    f"(mapped from {len(grid.cells)} 5-arcmin cells)")

        config = NASAPowerConfig(
            request_delay=request_delay,
            parameters=["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"]
        )
        source = NASAPowerSource(config)

        # Download per unique 30-arcmin cell
        climate_30 = {}
        for i, (cell_id, (lat, lon)) in enumerate(centers_30.items()):
            # V2-22b L (AC L.3): per-cell cancel for the 30-arcmin loop.
            raise_if_cancelled(
                getattr(self, 'cancel_check', None),
                f"acea.30arcmin.cell={i + 1}/{len(centers_30)}",
            )
            if progress_callback:
                progress_callback(i + 1, len(centers_30))
            logger.info(f"Downloading climate for 30arcmin cell {i+1}/{len(centers_30)} "
                       f"(ID={cell_id}, lat={lat:.2f}, lon={lon:.2f})")
            try:
                result = source.retrieve(
                    lat=lat, lon=lon, start_date=start_date, end_date=end_date,
                    cancel_check=getattr(self, 'cancel_check', None),
                )
                if result.success and result.data:
                    climate_30[cell_id] = result.data
            except PipelineCancelled:
                # V2-22b L: 30-arcmin per-cell broad except must not swallow cancel.
                raise
            except Exception as e:
                logger.error(f"Error downloading climate for 30arcmin cell {cell_id}: {e}")

        logger.info(f"Downloaded {len(climate_30)}/{len(centers_30)} unique 30-arcmin cells")

        # Map back to 5-arcmin cell IDs for compatibility with the rest of the translator
        climate_data = {}
        COLS_5 = 4320
        for cell in grid.cells:
            row_5 = cell.cell_id // COLS_5
            col_5 = cell.cell_id % COLS_5
            row_30 = row_5 // 6
            col_30 = col_5 // 6
            parent_30 = row_30 * self.GRID_COLS_30ARCMIN + col_30
            if parent_30 in climate_30:
                climate_data[cell.cell_id] = climate_30[parent_30]

        return climate_data

    def _generate_soil_csv(
        self,
        soil_profiles: Dict[int, SoilProfile],
        cell_ids_30arcmin: List[int],
    ) -> Path:
        """Generate ACEA soil data CSV.

        Args:
            soil_profiles: Dictionary of location_id to SoilProfile
            cell_ids_30arcmin: 30-arcmin cell IDs

        Returns:
            Path to generated CSV
        """
        csv_path = self.output_dir / "soil" / "soil_data.csv"

        with open(csv_path, 'w') as f:
            # Header
            f.write("cell_id,lat,lon,sand,clay,silt,oc,ph,bulk_density,")
            f.write("wilting_point,field_capacity,saturated_wc\n")

            for i, (loc_id, profile) in enumerate(soil_profiles.items()):
                cell_id = cell_ids_30arcmin[i] if i < len(cell_ids_30arcmin) else loc_id

                if profile.layers:
                    layer = profile.layers[0]  # Surface layer
                    f.write(f"{cell_id},{profile.lat},{profile.lon},")
                    f.write(f"{layer.sand},{layer.clay},{layer.silt or 0},")
                    f.write(f"{layer.organic_carbon or 0},{layer.ph or 6.5},")
                    f.write(f"{layer.bulk_density or 1.4},")
                    f.write(f"{layer.wilting_point or 0.1},")
                    f.write(f"{layer.field_capacity or 0.25},")
                    f.write(f"{layer.saturated_wc or 0.45}\n")

        logger.info(f"Generated ACEA soil CSV: {csv_path}")
        return csv_path

    def _generate_soil_netcdf_from_profiles(
        self,
        soil_profiles: Dict[int, SoilProfile],
        cell_ids_30arcmin: List[int],
    ) -> Optional[Path]:
        """Generate ACEA-compatible soil NetCDF from already-retrieved soil profiles.

        Fallback for when HWSD BIL/MDB source files are not available.
        Uses the sand/clay values that prismpy already retrieved via its
        soil data source (HWSD, iSDA, etc.).

        Args:
            soil_profiles: Dictionary of location_id to SoilProfile
            cell_ids_30arcmin: 30-arcmin cell IDs

        Returns:
            Path to generated NetCDF file, or None on failure
        """
        try:
            import netCDF4 as nc
        except ImportError:
            logger.error("netCDF4 not available, cannot generate soil NetCDF")
            return None

        nc_path = self.output_dir / "soil" / "HWSD_soil_data_on_cropland_v2.3.nc"
        nc_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize masked arrays for 30-arcmin global grid
        sand_data = np.ma.masked_all((360, 720), dtype=np.float32)
        clay_data = np.ma.masked_all((360, 720), dtype=np.float32)

        lats = np.array([89.75 - i * 0.5 for i in range(360)], dtype=np.float64)
        lons = np.array([-179.75 + i * 0.5 for i in range(720)], dtype=np.float64)

        # Build lookup from soil profiles (keyed by cell ID)
        profile_lookup = {}
        for i, (loc_id, profile) in enumerate(soil_profiles.items()):
            cell_id = cell_ids_30arcmin[i] if i < len(cell_ids_30arcmin) else loc_id
            if profile.layers:
                layer = profile.layers[0]
                profile_lookup[cell_id] = (
                    layer.sand if layer.sand is not None else 40.0,
                    layer.clay if layer.clay is not None else 25.0,
                )

        # Populate ALL cells in the package — use profile data if available,
        # otherwise use defaults (ACEA needs valid data for every simulated cell)
        n_from_profile = 0
        n_from_default = 0
        for cell_id in cell_ids_30arcmin:
            row = cell_id // 720
            col = cell_id % 720
            if not (0 <= row < 360 and 0 <= col < 720):
                continue
            if cell_id in profile_lookup:
                sand_data[row, col], clay_data[row, col] = profile_lookup[cell_id]
                n_from_profile += 1
            else:
                sand_data[row, col] = 40.0  # Sandy loam default
                clay_data[row, col] = 25.0
                n_from_default += 1

        n_populated = n_from_profile + n_from_default
        if n_populated == 0:
            logger.warning("No soil data could be mapped to grid cells")
            return None

        if n_from_default > 0:
            logger.warning(
                f"Used default soil values for {n_from_default}/{n_populated} cells "
                f"(no profile data available)"
            )

        # Write NetCDF
        ds = nc.Dataset(nc_path, 'w', format='NETCDF4')
        ds.createDimension('lat', 360)
        ds.createDimension('lon', 720)

        lat_var = ds.createVariable('lat', 'f8', ('lat',))
        lat_var[:] = lats

        lon_var = ds.createVariable('lon', 'f8', ('lon',))
        lon_var[:] = lons

        sand_var = ds.createVariable('sand', 'f4', ('lat', 'lon'), fill_value=-9999.0)
        sand_var[:] = sand_data

        clay_var = ds.createVariable('clay', 'f4', ('lat', 'lon'), fill_value=-9999.0)
        clay_var[:] = clay_data

        ds.title = "HWSD soil data for ACEA simulation"
        ds.source = "Generated by prismpy from retrieved soil profiles"
        ds.history = f"Created {datetime.now().isoformat()}"
        ds.close()

        logger.info(f"Generated ACEA soil NetCDF from profiles: {nc_path} "
                     f"({n_populated} cells)")
        return nc_path

    def _extract_hwsd_soil(
        self,
        grid: SpatialGrid,
        cell_ids_30arcmin: List[int],
        bil_path: Path,
        mdb_path: Path,
    ) -> Optional[Path]:
        """Extract HWSD soil data and generate soil CSV.

        Uses the ACEA/utils/hwsd_extraction module to extract soil properties
        from HWSD BIL+MDB files for the grid cells.

        Args:
            grid: Spatial grid with cell coordinates
            cell_ids_30arcmin: 30-arcmin cell IDs
            bil_path: Path to HWSD2.bil raster
            mdb_path: Path to HWSD2.mdb database

        Returns:
            Path to generated soil CSV, or None if extraction fails
        """
        # The HWSD v2.0 BIL+MDB extraction helper used to live at
        # ../../../../ACEA/utils/hwsd_extraction.py and was reached via
        # a runtime ``sys.path.insert`` shim. The shim silently fell
        # through to ``return None`` whenever the ACEA toolkit was not
        # co-located with prismpy on disk. The helper now lives under
        # ``prismpy.vendor.hwsd_extraction`` so the import resolves
        # deterministically across environments. The broad-except
        # carve-out at the executor / translator layer surfaces a
        # vendor-build-broken state as fail-loud
        # ``ModuleNotFoundError`` rather than as a silent skip.
        from prismpy.vendor.hwsd_extraction import extract_hwsd_soil_data

        if not bil_path.exists():
            logger.warning(f"HWSD BIL file not found: {bil_path}")
            return None

        if not mdb_path.exists():
            logger.warning(f"HWSD MDB file not found: {mdb_path}")
            return None

        # Build cells DataFrame with lat, lon, cell_id
        cells_data = []
        for i, cell in enumerate(grid.cells):
            cell_id = cell_ids_30arcmin[i] if i < len(cell_ids_30arcmin) else cell.id
            cells_data.append({
                'cell_id': cell_id,
                'lat': cell.lat,
                'lon': cell.lon,
            })

        cells_df = pd.DataFrame(cells_data)

        logger.info(f"Extracting HWSD soil data for {len(cells_df)} cells...")

        try:
            soil_df = extract_hwsd_soil_data(
                bil_path=str(bil_path),
                mdb_path=str(mdb_path),
                cells_df=cells_df,
                layer="D1",  # Topsoil
                verbose=True,
            )

            # Generate soil CSV from extracted data
            csv_path = self.output_dir / "soil" / "soil_data.csv"

            with open(csv_path, 'w') as f:
                f.write("cell_id,lat,lon,sand,clay,silt,oc,ph,bulk_density,")
                f.write("wilting_point,field_capacity,saturated_wc\n")

                for _, row in soil_df.iterrows():
                    # Use defaults for properties not in HWSD
                    f.write(f"{int(row['cell_id'])},{row['lat']},{row['lon']},")
                    f.write(f"{row['sand']},{row['clay']},{row['silt']},")
                    f.write(f"0.5,6.5,1.4,")  # oc, ph, bulk_density defaults
                    f.write(f"0.1,0.25,0.45\n")  # wp, fc, sat defaults

            logger.info(f"Generated ACEA soil CSV from HWSD: {csv_path}")
            return csv_path

        except Exception as e:
            logger.error(f"HWSD extraction failed: {e}")
            return None

    def _generate_acea_soil_netcdf(
        self,
        cell_ids_30arcmin: List[int],
        bil_path: Path,
        mdb_path: Path,
    ) -> Optional[Path]:
        """Generate ACEA-compatible soil NetCDF file.

        Creates a NetCDF file in the exact format ACEA expects:
        - 360x720 global grid (30-arcmin)
        - Variables: sand, clay, lat, lon
        - Only populates cells we need (rest is masked)

        This replaces ACEA's internal HWSD_soil_data_on_cropland_v2.3.nc
        to enable simulations in any region.

        Args:
            cell_ids_30arcmin: List of 30-arcmin cell IDs to populate
            bil_path: Path to HWSD2.bil raster
            mdb_path: Path to HWSD2.mdb database

        Returns:
            Path to generated NetCDF file, or None if extraction fails
        """
        try:
            import netCDF4 as nc
            import rasterio
        except ImportError as e:
            logger.error(f"Required package not available: {e}")
            return None

        if not bil_path.exists():
            logger.warning(f"HWSD BIL file not found: {bil_path}")
            return None

        if not mdb_path.exists():
            logger.warning(f"HWSD MDB file not found: {mdb_path}")
            return None

        logger.info(f"Generating ACEA soil NetCDF for {len(cell_ids_30arcmin)} cells...")

        # Create output path
        nc_path = self.output_dir / "soil" / "HWSD_soil_data_on_cropland_v2.3.nc"
        nc_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize masked arrays for 30-arcmin global grid
        sand_data = np.ma.masked_all((360, 720), dtype=np.float32)
        clay_data = np.ma.masked_all((360, 720), dtype=np.float32)

        # Create lat/lon arrays (cell centers)
        lats = np.array([89.75 - i * 0.5 for i in range(360)], dtype=np.float64)
        lons = np.array([-179.75 + i * 0.5 for i in range(720)], dtype=np.float64)

        # Extract soil properties from HWSD
        # First, load HWSD2_LAYERS table using mdb-export (works on Mac/Linux)
        hwsd_lookup = {}
        try:
            import subprocess
            result = subprocess.run(
                ['mdb-export', str(mdb_path), 'HWSD2_LAYERS'],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                import csv
                import io
                reader = csv.DictReader(io.StringIO(result.stdout))
                for row in reader:
                    smu_id = int(row['HWSD2_SMU_ID'])
                    layer = row['LAYER']
                    # Use D1 (topsoil 0-20cm) or D2 (20-40cm)
                    if layer in ('D1', 'D2') and smu_id not in hwsd_lookup:
                        try:
                            sand = float(row['SAND']) if row['SAND'] else None
                            clay = float(row['CLAY']) if row['CLAY'] else None
                            if sand is not None and clay is not None:
                                hwsd_lookup[smu_id] = (sand, clay)
                        except (ValueError, TypeError):
                            pass
                logger.info(f"Loaded {len(hwsd_lookup)} soil mapping units from HWSD2")
            else:
                logger.warning(f"mdb-export failed: {result.stderr}")
        except FileNotFoundError:
            logger.warning("mdb-export not found, trying pyodbc...")
            # Fallback to pyodbc. ``pyodbc`` is an opt-in extras-group
            # dependency declared in pyproject [project.optional-
            # dependencies] under ``acea-mdb``; it is platform-specific
            # (Windows + Microsoft Access ODBC driver, or Linux with
            # MDBTools system driver). The previous silent-skip path
            # caught ``Exception as e`` and logged a warning, which
            # left the user with no actionable next step. The
            # current pattern raises ``ModuleNotFoundError`` when the
            # extras-group is not installed so the install remediation
            # surfaces fail-loud at first call. Genuine ODBC
            # connection / driver errors still log a warning because
            # those are runtime data-availability issues, not
            # configuration ones.
            try:
                import pyodbc
            except ImportError as e:
                raise ModuleNotFoundError(
                    "pyodbc is a required prismpy dependency but did "
                    "not import. The Python wheel ships with prismpy; "
                    "the matching ODBC system package must also be "
                    "available — install ``brew install unixodbc`` on "
                    "macOS, ``apt-get install -y unixodbc-dev`` on "
                    "Linux. The ACEA HWSD MDB fallback also accepts "
                    "the ``mdbtools`` system package as the preferred "
                    "primary path (``brew install mdbtools`` / "
                    "``apt-get install -y mdbtools``)."
                ) from e

            try:
                conn_str = (
                    "DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
                    f"DBQ={mdb_path}"
                )
                try:
                    conn = pyodbc.connect(conn_str)
                except pyodbc.Error:
                    conn_str = f"DRIVER={{MDBTools}};DBQ={mdb_path}"
                    conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT HWSD2_SMU_ID, SAND, CLAY FROM HWSD2_LAYERS "
                    "WHERE LAYER = 'D1'"
                )
                for row in cursor.fetchall():
                    if row[1] is not None and row[2] is not None:
                        hwsd_lookup[int(row[0])] = (
                            float(row[1]),
                            float(row[2]),
                        )
                conn.close()
                logger.info(
                    f"Loaded {len(hwsd_lookup)} soil mapping units "
                    "via pyodbc"
                )
            except pyodbc.Error as e:
                logger.warning(
                    f"pyodbc connection / query failed: {e}"
                )
            except Exception as e:  # noqa: BLE001
                # Row coercion / cleanup / unexpected runtime errors
                # (float parse failures, malformed cursor rows, etc.).
                # The original silent-skip caught these under a single
                # ``except Exception`` and continued with default soil
                # values; preserve that behavior so a row-parse failure
                # does not abort the whole HWSD extraction. The
                # narrower ``pyodbc.Error`` carve-out above still
                # captures connection / driver issues with a more
                # specific log line.
                logger.warning(f"pyodbc fallback runtime error: {e}")
        except Exception as e:
            logger.warning(f"HWSD database loading failed: {e}")

        # Now extract soil for each cell
        if hwsd_lookup:
            with rasterio.open(bil_path) as src:
                for cell_id in cell_ids_30arcmin:
                    row = cell_id // 720
                    col = cell_id % 720
                    lat = lats[row]
                    lon = lons[col]

                    try:
                        hwsd_row, hwsd_col = src.index(lon, lat)
                        window = rasterio.windows.Window(hwsd_col, hwsd_row, 1, 1)
                        smu_id = int(src.read(1, window=window)[0, 0])

                        if smu_id > 0 and smu_id in hwsd_lookup:
                            sand, clay = hwsd_lookup[smu_id]
                            sand_data[row, col] = sand
                            clay_data[row, col] = clay
                            logger.debug(f"Cell {cell_id}: SMU={smu_id}, sand={sand}, clay={clay}")
                        elif smu_id > 0:
                            logger.debug(f"Cell {cell_id}: SMU={smu_id} not in lookup, using default")
                            sand_data[row, col] = 40.0
                            clay_data[row, col] = 25.0
                    except Exception as e:
                        logger.debug(f"Cell {cell_id} extraction failed: {e}")
                        sand_data[row, col] = 40.0
                        clay_data[row, col] = 25.0
        else:
            logger.warning("No HWSD lookup available, using default soil values")
            # Fallback: use reasonable defaults for cells
            for cell_id in cell_ids_30arcmin:
                row = cell_id // 720
                col = cell_id % 720
                sand_data[row, col] = 40.0  # Sandy loam default
                clay_data[row, col] = 25.0

        # Count populated cells
        n_valid = np.sum(~sand_data.mask)
        logger.info(f"Populated {n_valid} cells with soil data")

        # Write NetCDF file
        ds = nc.Dataset(nc_path, 'w', format='NETCDF4')

        # Create dimensions
        ds.createDimension('lat', 360)
        ds.createDimension('lon', 720)

        # Create variables
        lat_var = ds.createVariable('lat', 'f8', ('lat',))
        lat_var[:] = lats

        lon_var = ds.createVariable('lon', 'f8', ('lon',))
        lon_var[:] = lons

        sand_var = ds.createVariable('sand', 'f4', ('lat', 'lon'), fill_value=-9999.0)
        sand_var[:] = sand_data

        clay_var = ds.createVariable('clay', 'f4', ('lat', 'lon'), fill_value=-9999.0)
        clay_var[:] = clay_data

        # Add metadata
        ds.title = "HWSD soil data for ACEA simulation"
        ds.source = "Generated by prismpy from HWSD2"
        ds.history = f"Created {datetime.now().isoformat()}"

        ds.close()

        logger.info(f"Generated ACEA soil NetCDF: {nc_path}")
        return nc_path

    def _clip_spam_data(
        self,
        region: Region,
        crop_name: str,
        spam_data_dir: Path,
    ) -> List[Path]:
        """Clip SPAM harvested area rasters to region bounds.

        Clips global SPAM rasters to the region and saves them in the
        package output directory for self-contained packages.

        Args:
            region: Region with bounding box
            crop_name: Crop name (e.g., 'Wheat')
            spam_data_dir: Path to directory containing global SPAM rasters

        Returns:
            List of paths to clipped SPAM files
        """
        from prismpy.sources.crop_areas.spam import SPAMSource, SPAMConfig

        clipped_files = []

        # Get SPAM code for crop
        spam_code = SPAM_CODE_MAP.get(crop_name)
        if not spam_code:
            logger.warning(f"No SPAM code mapping for crop: {crop_name}")
            return clipped_files

        # Get ACEA FAO code for output directory
        acea_fao_code = ACEA_FAO_CODE_MAP.get(crop_name, 999)

        # Output directory for harvested areas
        output_dir = self.output_dir / "harvested_areas" / str(acea_fao_code)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SPAM source
        spam_source = SPAMSource(
            config=SPAMConfig(data_dir=spam_data_dir)
        )

        # Technology levels to clip (R=Rainfed, I=Irrigated, A=All)
        tech_levels = ['R', 'I', 'A']

        for tech in tech_levels:
            # Try different file naming patterns
            patterns = [
                f"spam2020V2r0_global_H_{spam_code}_{tech}.tif",      # SPAM code naming
                f"spam2020_V2r0_global_H_{spam_code}_{tech}.tif",     # Original ZIP naming (underscore after year)
                f"spam2020V2r0_global_H_{acea_fao_code}_{tech}.tif",  # FAO code naming (ACEA convention)
                f"spam2020v2r0_global_H_{spam_code}_{tech}.tif",      # Lowercase version
                f"spam2010V1r0_global_H_{spam_code}_{tech}.tif",      # Older SPAM 2010
                f"*{spam_code}*_{tech}.tif",                           # Wildcard fallback
            ]

            input_path = None
            for pattern in patterns:
                matches = list(spam_data_dir.glob(pattern))
                if matches:
                    input_path = matches[0]
                    break

            if not input_path:
                logger.debug(f"SPAM file not found for {spam_code}_{tech}")
                continue

            # Output file maintains similar naming
            output_filename = f"spam2020V2r0_global_H_{acea_fao_code}_{tech}.tif"
            output_path = output_dir / output_filename

            # Clip to region bounds
            result = spam_source.clip_to_file(
                input_path=input_path,
                output_path=output_path,
                bounds=region.bounds,
            )

            if result:
                clipped_files.append(result)

        if clipped_files:
            logger.info(f"Clipped {len(clipped_files)} SPAM files to {output_dir}")
        else:
            logger.warning(f"No SPAM files found for {crop_name} in {spam_data_dir}")

        return clipped_files

    def _generate_dummy_spam_files(
        self,
        region: Region,
        crop_name: str,
    ) -> List[Path]:
        """Generate GLOBAL dummy SPAM files for self-contained packages.

        Creates GLOBAL 5-arcmin SPAM GeoTIFF files (2160x4320 pixels) that
        ACEA expects. Only populates cells within the region with positive
        values. This allows ACEA to initialize and run without real SPAM data.

        IMPORTANT: ACEA indexes SPAM rasters using global 5-arcmin coordinates:
            y5 = (30-arcmin row) * 6
            x5 = (30-arcmin col) * 6
        So the raster MUST be global (2160x4320) for indexing to work.

        Args:
            region: Region with bounding box
            crop_name: Crop name (e.g., 'Wheat')

        Returns:
            List of paths to generated dummy SPAM files
        """
        try:
            import rasterio
            from rasterio.transform import Affine
        except ImportError:
            logger.error("rasterio required for dummy SPAM generation")
            return []

        generated_files = []

        # Get ACEA FAO code for output directory
        acea_fao_code = ACEA_FAO_CODE_MAP.get(crop_name, 999)

        # Output directory for harvested areas
        output_dir = self.output_dir / "harvested_areas" / str(acea_fao_code)
        output_dir.mkdir(parents=True, exist_ok=True)

        # SPAM is GLOBAL 5-arcmin grid (ACEA expects this exact size)
        # 360 degrees / (5/60) = 4320 columns
        # 180 degrees / (5/60) = 2160 rows
        resolution = 5 / 60  # 5 arc-minutes in degrees
        width = 4320   # Global columns
        height = 2160  # Global rows

        # Transform: top-left is (-180, 90), pixel size is resolution
        # Note: y resolution is negative (north to south)
        transform = Affine(resolution, 0, -180, 0, -resolution, 90)

        # Create global grid initialized with nodata
        # Use 0 for no harvested area (nodata=-1 would mark as invalid)
        dummy_data = np.zeros((height, width), dtype=np.float32)

        # Calculate which 5-arcmin pixels fall within the region
        # and set them to 1.0 hectare
        min_col = int((region.bounds.minx + 180) / resolution)
        max_col = int((region.bounds.maxx + 180) / resolution) + 1
        min_row = int((90 - region.bounds.maxy) / resolution)
        max_row = int((90 - region.bounds.miny) / resolution) + 1

        # Clamp to valid range
        min_col = max(0, min(min_col, width - 1))
        max_col = max(0, min(max_col, width))
        min_row = max(0, min(min_row, height - 1))
        max_row = max(0, min(max_row, height))

        # Set region cells to 1.0 hectare
        dummy_data[min_row:max_row, min_col:max_col] = 1.0

        n_cells = (max_row - min_row) * (max_col - min_col)
        logger.info(f"Creating global SPAM raster (2160x4320) with {n_cells} region cells")

        # Raster profile matching SPAM format
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': width,
            'height': height,
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': transform,
            'nodata': -1,
            'compress': 'lzw',
        }

        # Generate files for each technology level (R, I, A)
        tech_levels = ['R', 'I', 'A']

        for tech in tech_levels:
            output_filename = f"spam2020V2r0_global_H_{acea_fao_code}_{tech}.tif"
            output_path = output_dir / output_filename

            try:
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(dummy_data, 1)

                generated_files.append(output_path)
                logger.debug(f"Generated dummy SPAM file: {output_path}")

            except Exception as e:
                logger.error(f"Failed to generate dummy SPAM file {output_path}: {e}")

        if generated_files:
            logger.info(f"Generated {len(generated_files)} dummy SPAM files for self-contained package")
            logger.info(f"  Note: Using placeholder data since real_cropland=False")

        return generated_files

    def _handle_gaez_data(
        self,
        crop_name: str,
        gaez_data_dir: Optional[Path] = None,
        auto_download: bool = True,
    ) -> List[Path]:
        """Handle GAEZ data - download if needed, copy to output.

        Downloads GAEZ crop suitability data from FAO S3 bucket if not
        cached, then copies to the package output directory.

        Args:
            crop_name: Crop name (e.g., 'Wheat')
            gaez_data_dir: User-provided GAEZ data directory (optional)
            auto_download: Whether to auto-download from FAO

        Returns:
            List of output file paths
        """
        from prismpy.sources.gaez.downloader import GAEZDownloader

        output_dir = self.output_dir / "gaez"
        output_files = []

        # Use user-provided directory if available
        if gaez_data_dir and gaez_data_dir.exists():
            logger.info(f"Using user-provided GAEZ data from: {gaez_data_dir}")
            # Copy relevant files to output
            import shutil

            for subdir in ['LUT', 'PotentialYield', 'F3']:
                src_dir = gaez_data_dir / subdir
                if src_dir.exists():
                    dst_dir = output_dir / subdir
                    dst_dir.mkdir(parents=True, exist_ok=True)

                    for f in src_dir.glob('*.tif'):
                        # Only copy files matching the crop
                        crop_lower = crop_name.lower()
                        fname_lower = f.name.lower()
                        if crop_lower in fname_lower or any(
                            c.lower().replace('_', '') in fname_lower
                            for c in self._get_gaez_cultivars(crop_name)
                        ):
                            dst_path = dst_dir / f.name
                            shutil.copy2(f, dst_path)
                            output_files.append(dst_path)

            if output_files:
                logger.info(f"Copied {len(output_files)} GAEZ files to package")
            return output_files

        # Auto-download if enabled
        if auto_download:
            logger.info(f"Auto-downloading GAEZ data for {crop_name}...")
            downloader = GAEZDownloader()

            try:
                output_files = downloader.copy_to_output(
                    crop_name=crop_name,
                    output_dir=output_dir,
                    water_supplies=['irr', 'rf'],
                    input_levels=['High', 'Low'],
                )
                logger.info(f"Downloaded and copied {len(output_files)} GAEZ files")
            except (ImportError, ModuleNotFoundError):
                # Mirror the climate-source carve-outs (executor.py +
                # tamsat.py + agera5.py): an undeclared / vendor-build-
                # broken transitive dep is a configuration error, not
                # a runtime data error. Letting it surface as ``GAEZ
                # download failed: {e}`` would re-create the silent-
                # skip class the F-AL substrate-hardening sweep
                # closed at the GAEZDownloader.._download_with_retry
                # entry. Per durable lesson #6 + #20, propagate so
                # pip / CI / startup surfaces the missing dep loudly.
                raise
            except Exception as e:
                logger.warning(f"GAEZ download failed: {e}")

        return output_files

    def _get_gaez_cultivars(self, crop_name: str) -> List[str]:
        """Get GAEZ cultivar names for a crop."""
        cultivar_map = {
            'Maize': ['Highland_maize', 'Lowland_maize', 'Temperate_maize', 'Maize'],
            'Corn': ['Highland_maize', 'Lowland_maize', 'Temperate_maize', 'Maize'],
            'Wheat': ['Spring_wheat', 'Winter_wheat'],
            'Rice': ['Wetland_rice', 'Dryland_rice'],
            'Sorghum': ['Highland_sorghum', 'Lowland_sorghum'],
            'Millet': ['Pearl_millet', 'Foxtail_millet'],
        }
        return cultivar_map.get(crop_name, [crop_name])

    def _generate_install_script(self, data: UnifiedData) -> Path:
        """Generate install.py script for the package.

        Creates a Python script that copies package data to an ACEA installation.

        Args:
            data: Unified data containing region and crop info

        Returns:
            Path to generated install.py
        """
        script_path = self.output_dir / "install.py"

        region_name = data.region.name if data.region else "Unknown"
        crop_name = self.config.crop.name

        script_content = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ACEA Package Installer

Installs the {region_name} {crop_name} package data to an ACEA installation.

Usage:
    python install.py /path/to/ACEA/data/acea

This will copy:
    - Climate pickles to climate/
    - Crop calendar to crop_calendar/
    - Crop parameters to crop_parameters/
    - Soil data to soil/
    - Harvested areas to harvested_areas/
    - GAEZ data to gaez/
    - CO2 data to CO2/
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def merge_soil_netcdf(src_nc: Path, dst_nc: Path, dry_run: bool = False):
    """Merge soil data from source NetCDF into destination NetCDF.

    This preserves existing cells and adds new cells from the package.

    Args:
        src_nc: Source NetCDF from package
        dst_nc: Destination NetCDF in ACEA installation
        dry_run: If True, only show what would be done
    """
    try:
        import netCDF4 as nc
        import numpy as np
    except ImportError:
        print("  Warning: netCDF4 not available, copying file instead of merging")
        if not dry_run:
            shutil.copy2(src_nc, dst_nc)
        return

    # Read source data
    src_ds = nc.Dataset(src_nc, 'r')
    src_sand = src_ds.variables['sand'][:]
    src_clay = src_ds.variables['clay'][:]

    # Count new cells
    new_cells = np.sum(~np.ma.getmask(src_sand))

    if dst_nc.exists():
        # Load existing data
        dst_ds = nc.Dataset(dst_nc, 'r')
        dst_sand = dst_ds.variables['sand'][:].copy()
        dst_clay = dst_ds.variables['clay'][:].copy()
        existing_cells = np.sum(~np.ma.getmask(dst_sand))
        dst_ds.close()

        # Merge: add new cells where destination is masked
        for i in range(360):
            for j in range(720):
                if not np.ma.is_masked(src_sand[i, j]):
                    dst_sand[i, j] = src_sand[i, j]
                    dst_clay[i, j] = src_clay[i, j]

        merged_cells = np.sum(~np.ma.getmask(dst_sand))
        print(f"  Merging soil data: {{existing_cells}} existing + {{new_cells}} new = {{merged_cells}} total cells")
    else:
        dst_sand = src_sand
        dst_clay = src_clay
        print(f"  Creating soil data with {{new_cells}} cells")

    src_ds.close()

    if dry_run:
        print(f"  [MERGE] {{src_nc}} -> {{dst_nc}}")
        return

    # Write merged data
    dst_nc.parent.mkdir(parents=True, exist_ok=True)
    out_ds = nc.Dataset(dst_nc, 'w', format='NETCDF4')

    out_ds.createDimension('lat', 360)
    out_ds.createDimension('lon', 720)

    lat_var = out_ds.createVariable('lat', 'f8', ('lat',))
    lat_var[:] = [89.75 - i * 0.5 for i in range(360)]

    lon_var = out_ds.createVariable('lon', 'f8', ('lon',))
    lon_var[:] = [-179.75 + i * 0.5 for i in range(720)]

    sand_var = out_ds.createVariable('sand', 'f4', ('lat', 'lon'), fill_value=-9999.0)
    sand_var[:] = dst_sand

    clay_var = out_ds.createVariable('clay', 'f4', ('lat', 'lon'), fill_value=-9999.0)
    clay_var[:] = dst_clay

    out_ds.close()


def install(acea_data_dir: Path, dry_run: bool = False):
    """Install package data to ACEA data directory.

    Args:
        acea_data_dir: Path to ACEA data directory (e.g., ACEA/data/acea)
        dry_run: If True, only show what would be done
    """
    pkg_dir = Path(__file__).parent

    # Mapping of package directories to ACEA data directories
    dir_mappings = {{
        "climate": "climate",
        "crop_calendar": "crop_calendar",
        "crop_params": "crop_parameters",
        "harvested_areas": "harvested_areas",
        "gaez": "gaez",
        "co2": "CO2",
        "config": "config",  # Project config
    }}

    copied = []
    skipped = []

    # Handle soil NetCDF specially - merge instead of overwrite
    soil_nc = pkg_dir / "soil" / "HWSD_soil_data_on_cropland_v2.3.nc"
    if soil_nc.exists():
        dst_soil_nc = acea_data_dir / "soil" / "HWSD_soil_data_on_cropland_v2.3.nc"
        merge_soil_netcdf(soil_nc, dst_soil_nc, dry_run)
        copied.append(str(dst_soil_nc))

    # Copy soil_data.csv (reference file, OK to overwrite)
    soil_csv = pkg_dir / "soil" / "soil_data.csv"
    if soil_csv.exists():
        dst_csv = acea_data_dir / "soil" / "soil_data.csv"
        if not dry_run:
            dst_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(soil_csv, dst_csv)
        copied.append(str(dst_csv))

    for pkg_subdir, acea_subdir in dir_mappings.items():
        src_dir = pkg_dir / pkg_subdir
        if not src_dir.exists():
            continue

        dst_dir = acea_data_dir / acea_subdir

        for src_file in src_dir.rglob("*"):
            if not src_file.is_file():
                continue

            # Preserve subdirectory structure
            rel_path = src_file.relative_to(src_dir)
            dst_file = dst_dir / rel_path

            if dry_run:
                print(f"  [COPY] {{src_file}} -> {{dst_file}}")
                copied.append(str(dst_file))
            else:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                copied.append(str(dst_file))
                print(f"  Copied: {{rel_path}}")

    print(f"\\n{{len(copied)}} files copied to {{acea_data_dir}}")

    # Also copy the project config to projects/ directory
    config_dir = pkg_dir / "config"
    if config_dir.exists():
        projects_dir = acea_data_dir.parent.parent / "projects"
        if projects_dir.exists():
            for config_file in config_dir.glob("*_config.py"):
                dst_config = projects_dir / config_file.name
                if not dry_run:
                    shutil.copy2(config_file, dst_config)
                print(f"  Config copied to: {{dst_config}}")


def main():
    parser = argparse.ArgumentParser(
        description="Install ACEA package data to ACEA installation"
    )
    parser.add_argument(
        "acea_data_dir",
        type=Path,
        help="Path to ACEA data directory (e.g., /path/to/ACEA/data/acea)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without copying"
    )

    args = parser.parse_args()

    if not args.acea_data_dir.exists():
        print(f"ERROR: ACEA data directory not found: {{args.acea_data_dir}}")
        sys.exit(1)

    print(f"Installing {region_name} {crop_name} package to {{args.acea_data_dir}}")
    print()

    install(args.acea_data_dir, dry_run=args.dry_run)

    print()
    print("Installation complete!")
    print()
    print("To run ACEA simulation:")
    print(f"  cd {{args.acea_data_dir.parent.parent}}")
    print("  python -c \\"from projects.{{config_name}}_config import project_conf; "
          "from acea.main import run_acea; run_acea(project_conf)\\"")


if __name__ == "__main__":
    main()
'''

        # Get config name from output dir
        config_name = self.output_dir.name.replace('-', '_').replace(' ', '_')
        script_content = script_content.replace("{config_name}", config_name)

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make executable
        script_path.chmod(0o755)

        logger.info(f"Generated install script: {script_path}")
        return script_path

    def _generate_package_metadata(
        self,
        data: UnifiedData,
        cell_ids_30arcmin: List[int],
        climate_name: str,
        output_files: List[Path],
    ) -> List[Path]:
        """Generate package metadata files (manifest, README, provenance).

        Creates self-documenting package with:
        - manifest.json: File inventory with SHA256 checksums
        - README.md: Usage instructions and data source documentation
        - provenance.json: Data lineage and processing decisions

        Args:
            data: UnifiedData container with translation inputs
            cell_ids_30arcmin: List of 30-arcmin cell IDs
            climate_name: Climate file prefix (e.g., 'oromia_nasapower')
            output_files: List of files already generated

        Returns:
            List of metadata file paths generated
        """
        from prismpy.packaging.manifest import create_manifest, save_manifest
        from prismpy.packaging.readme_generator import generate_readme
        from datetime import datetime

        logger.info("Generating ACEA package metadata...")
        metadata_files = []

        # Get platform config
        platform_config = self.get_platform_config()

        # Count files by type
        climate_files = list((self.output_dir / "climate").glob("*.pckl"))
        n_climate_files = len(climate_files)

        # Get ACEA FAO code
        crop_name = self.config.crop.name
        fao_code = ACEA_FAO_CODE_MAP.get(crop_name, 999)

        # Determine data sources
        climate_source = "NASA POWER"
        # V2-19b-fix Finding 7: read actual soil source from data, not
        # config. The config always has hwsd_bil_path injected by prismweb
        # regardless of what the pipeline actually used.
        soil_source = "source unavailable"
        if data.soil:
            first_source = next(
                (p.source for p in data.soil.values() if hasattr(p, 'source') and p.source),
                None,
            )
            if first_source:
                soil_source = first_source
            elif not data.soil:
                soil_source = "Placeholder (pipeline default)"

        spam_source = "Dummy (placeholder)"
        if platform_config:
            spam_dir = getattr(platform_config, 'spam_data_dir', None)
            if spam_dir:
                spam_source = "SPAM 2020"

        gaez_source = "Not available"
        gaez_dir = self.output_dir / "gaez"
        if gaez_dir.exists() and any(gaez_dir.rglob("*.tif")):
            gaez_source = "FAO GAEZ v4 (auto-downloaded)"

        # Temporal settings
        start_year = self.config.temporal.start_year
        end_year = self.config.temporal.end_year
        spinup_years = self.config.temporal.spinup_years
        climate_start = start_year - spinup_years

        # Resolved-source discriminator for manifest derivation:
        # read the runtime boundary source (post-fallback) and
        # honor the configured GADM admin level only under GADM.
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

        # Build config dict for manifest/README
        package_config = {
            # Project info
            'project_name': self.config.project.name if hasattr(self.config, 'project') and self.config.project else 'ACEA Package',
            'package_name': 'acea',

            # Region info
            'region_name': data.region.name,
            'country': data.region.country,

            # Crop info
            'crop_name': crop_name,
            'fao_code': fao_code,
            'planting_doy': (
                self.config.crop.calendar.planting_doy
                if (self.config.crop
                    and getattr(self.config.crop, 'calendar', None))
                else None
            ),
            'maturity_doy': (
                self.config.crop.calendar.maturity_doy
                if (self.config.crop
                    and getattr(self.config.crop, 'calendar', None))
                else None
            ),

            # Boundary metadata. ``gadm_level`` is the configured
            # admin level only when the resolved source is GADM;
            # ``None`` for manual / shapefile / GADM-failed-fallback
            # so the manifest tracks the actual on-disk artifact.
            'gadm_level': manifest_gadm_level,

            # Temporal info
            'start_year': start_year,
            'end_year': end_year,
            'spinup_years': spinup_years,

            # ACEA-specific
            'n_cells': len(cell_ids_30arcmin),
            'n_climate_files': n_climate_files,
            'climate_name': climate_name,
            'gridcells': cell_ids_30arcmin,

            # Config parameters
            'clock_start': f"{climate_start}/01/01",
            'clock_end': self.config.temporal.get_climate_end_date(
                self.config.crop.calendar if self.config.crop else None
            ).strftime('%Y/%m/%d'),
            'resolution': 1,  # 5arcmin
            'scenarios': [1],  # rainfed

            # Data sources
            'climate_source': climate_source,
            'soil_source': soil_source,
            'spam_source': spam_source,
            'gaez_source': gaez_source,

            'data_sources': {
                'climate': climate_source,
                'soil': soil_source,
                'harvested_areas': spam_source,
                'crop_suitability': gaez_source,
                'boundaries': boundary_label,
            },
        }

        # 1. Generate manifest
        try:
            manifest = create_manifest(self.output_dir, package_config, platform='acea')
            manifest_path = save_manifest(manifest, self.output_dir / 'manifest.json')
            metadata_files.append(manifest_path)
            logger.info(f"Generated manifest: {manifest_path}")
        except Exception as e:
            logger.warning(f"Failed to generate manifest: {e}")

        # 2. Generate README
        try:
            readme_path = generate_readme(
                self.output_dir / 'README.md',
                package_config,
                platform='acea'
            )
            metadata_files.append(readme_path)
            logger.info(f"Generated README: {readme_path}")
        except Exception as e:
            logger.warning(f"Failed to generate README: {e}")

        # V2-20: Legacy System B provenance.json generation deleted.
        # Provenance is now handled by System A (prismpy.provenance.tracker)
        # and distributed via executor._execute_package.

        return metadata_files

    def _generate_crop_calendar(
        self,
        crop_calendar: Dict[int, CropCalendar],
        cell_ids_30arcmin: List[int],
    ) -> Path:
        """Generate ACEA crop calendar CSV.

        Args:
            crop_calendar: Dictionary of location_id to CropCalendar
            cell_ids_30arcmin: 30-arcmin cell IDs

        Returns:
            Path to generated CSV
        """
        csv_path = self.output_dir / "crop_calendar" / "calendar.csv"

        with open(csv_path, 'w') as f:
            f.write("cell_id,planting_doy,harvest_doy\n")

            for i, (loc_id, calendar) in enumerate(crop_calendar.items()):
                cell_id = cell_ids_30arcmin[i] if i < len(cell_ids_30arcmin) else loc_id
                f.write(f"{cell_id},{calendar.planting_doy},{calendar.harvest_doy}\n")

        logger.info(f"Generated ACEA crop calendar: {csv_path}")
        return csv_path

    def _generate_crop_calendar_nc(
        self,
        crop_calendar: Dict[int, CropCalendar],
        cell_ids_30arcmin: List[int],
        crop_short: str,
    ) -> List[Path]:
        """Generate ACEA crop calendar NetCDF files.

        ACEA expects NetCDF files with:
        - Dimensions: lat (360), lon (720) - global 30-arcmin grid
        - Variables: planting_day, maturity_day (int16)
        - Fill value: -9999 for cells with no data

        Generates both rainfed (rf) and irrigated (ir) versions.

        Args:
            crop_calendar: Dictionary of cell_id to CropCalendar
            cell_ids_30arcmin: List of 30-arcmin cell IDs
            crop_short: Short crop name (e.g., 'mai' for Maize)

        Returns:
            List of generated NetCDF file paths
        """
        try:
            from netCDF4 import Dataset
        except ImportError:
            logger.warning("netCDF4 not installed, skipping NetCDF calendar generation")
            return []

        output_files = []
        calendar_dir = self.output_dir / "crop_calendar"

        # Build mapping from cell_id to calendar data
        # If crop_calendar has only one entry, apply it to all cells
        calendar_items = list(crop_calendar.values())

        if len(calendar_items) == 1:
            # Single calendar entry - apply to all cells
            default_cal = calendar_items[0]
            calendar_by_cell = {cell_id: default_cal for cell_id in cell_ids_30arcmin}
            logger.debug(f"Using single calendar for all {len(cell_ids_30arcmin)} cells")
        elif len(calendar_items) == len(cell_ids_30arcmin):
            # One calendar per cell - map by index
            calendar_by_cell = {}
            for i, (loc_id, cal) in enumerate(crop_calendar.items()):
                cell_id = cell_ids_30arcmin[i]
                calendar_by_cell[cell_id] = cal
        else:
            # Try to map by location ID
            calendar_by_cell = {}
            for i, (loc_id, cal) in enumerate(crop_calendar.items()):
                cell_id = cell_ids_30arcmin[i] if i < len(cell_ids_30arcmin) else loc_id
                calendar_by_cell[cell_id] = cal
            logger.warning(f"Calendar count ({len(crop_calendar)}) differs from cell count ({len(cell_ids_30arcmin)})")

        # Generate both rainfed and irrigated versions (identical for now)
        for irr_type in ['rf', 'ir']:
            nc_path = calendar_dir / f"{crop_short}_{irr_type}_crop_calendar.nc"

            nc = Dataset(str(nc_path), 'w', format='NETCDF4')

            # Create dimensions
            nc.createDimension('lat', self.GRID_ROWS_30ARCMIN)
            nc.createDimension('lon', self.GRID_COLS_30ARCMIN)

            # Create coordinate variables
            lat_var = nc.createVariable('lat', 'f8', ('lat',))
            lat_var.long_name = 'latitude'
            lat_var.units = 'degrees_north'
            lat_var[:] = np.arange(89.75, -90, -0.5)

            lon_var = nc.createVariable('lon', 'f8', ('lon',))
            lon_var.long_name = 'longitude'
            lon_var.units = 'degrees_east'
            lon_var[:] = np.arange(-179.75, 180, 0.5)

            # Create data variables with fill value
            planting_var = nc.createVariable(
                'planting_day', 'i2', ('lat', 'lon'),
                fill_value=-9999, zlib=True, complevel=6
            )
            planting_var.long_name = 'Planting day of year'
            planting_var.units = 'day'
            planting_var.valid_range = np.array([1, 366], dtype='i2')

            maturity_var = nc.createVariable(
                'maturity_day', 'i2', ('lat', 'lon'),
                fill_value=-9999, zlib=True, complevel=6
            )
            maturity_var.long_name = 'Maturity day of year'
            maturity_var.units = 'day'
            maturity_var.valid_range = np.array([1, 366], dtype='i2')

            # Initialize with fill values
            planting_data = np.full((self.GRID_ROWS_30ARCMIN, self.GRID_COLS_30ARCMIN),
                                    -9999, dtype='i2')
            maturity_data = np.full((self.GRID_ROWS_30ARCMIN, self.GRID_COLS_30ARCMIN),
                                    -9999, dtype='i2')

            # Fill data for cells in our grid
            for cell_id, cal in calendar_by_cell.items():
                row, col = self._cell_id_to_row_col(cell_id)
                if 0 <= row < self.GRID_ROWS_30ARCMIN and 0 <= col < self.GRID_COLS_30ARCMIN:
                    planting_data[row, col] = cal.planting_doy
                    # Use harvest_doy if available, otherwise maturity_doy
                    harvest = cal.harvest_doy if cal.harvest_doy else cal.maturity_doy
                    maturity_data[row, col] = harvest

            planting_var[:, :] = planting_data
            maturity_var[:, :] = maturity_data

            # Global attributes
            nc.title = f"{crop_short.upper()} {'Rainfed' if irr_type == 'rf' else 'Irrigated'} Crop Calendar"
            nc.source = "prismpy framework"
            nc.conventions = "CF-1.6"
            nc.calendar_type = "rainfed" if irr_type == "rf" else "irrigated"
            nc.grid_resolution = "30-arcmin (0.5 degree)"
            nc.grid_dimensions = "360 x 720 (global)"
            nc.cells_with_data = len(calendar_by_cell)
            nc.institution = "prismpy"

            nc.close()
            output_files.append(nc_path)
            logger.debug(f"Generated crop calendar NetCDF: {nc_path}")

        logger.info(f"Generated {len(output_files)} ACEA crop calendar NetCDF files")
        return output_files

    def _generate_crop_params(self, crop_params: CropParameters) -> Path:
        """Generate ACEA crop parameters YAML.

        Args:
            crop_params: CropParameters object

        Returns:
            Path to generated YAML
        """
        import yaml

        params_path = self.output_dir / "crop_params" / "params.yaml"

        # Use ACEA crop-specific defaults from CROP_GDD_DEFAULTS
        # This ensures proper GDD values for each crop instead of inheriting
        # wrong values from generic pipeline defaults
        params_dict = self._get_default_acea_params(crop_params)

        with open(params_path, 'w') as f:
            yaml.dump(params_dict, f, default_flow_style=False)

        logger.info(f"Generated ACEA crop params: {params_path}")
        return params_path

    def _generate_crop_params_nc(
        self,
        crop_params: CropParameters,
        cell_ids_30arcmin: List[int],
        crop_short: str,
    ) -> List[Path]:
        """Generate ACEA crop parameters NetCDF files.

        ACEA expects NetCDF files with:
        - Dimensions: lat (360), lon (720) - global 30-arcmin grid
        - Variables: 11 GDD and canopy parameters (float32)
        - Fill value: NaN for cells with no data

        Generates both rainfed (rf) and irrigated (ir) versions.

        Args:
            crop_params: CropParameters object
            cell_ids_30arcmin: List of 30-arcmin cell IDs
            crop_short: Short crop name (e.g., 'mai' for Maize)

        Returns:
            List of generated NetCDF file paths
        """
        try:
            from netCDF4 import Dataset
        except ImportError:
            logger.warning("netCDF4 not installed, skipping NetCDF params generation")
            return []

        output_files = []
        params_dir = self.output_dir / "crop_params"

        # Get ACEA-format parameters using crop-specific defaults
        # Always use _get_default_acea_params() which has CROP_GDD_DEFAULTS
        # instead of to_acea_format() which may inherit wrong values from
        # pipeline's generic defaults (e.g., SARRA-Py maize SDJMatu1=500)
        params_dict = self._get_default_acea_params(crop_params)

        # Define the 11 required ACEA parameters with descriptions and units
        param_metadata = {
            'gdd_emergence': ('GDD from sowing to emergence', 'degree-days'),
            'gdd_max_root': ('GDD from sowing to maximum rooting depth', 'degree-days'),
            'gdd_senescence': ('GDD from sowing to senescence', 'degree-days'),
            'gdd_maturity': ('GDD from sowing to maturity', 'degree-days'),
            'gdd_yield_form': ('GDD from sowing to yield formation', 'degree-days'),
            'gdd_duration_flowering': ('Duration of flowering stage', 'degree-days'),
            'gdd_duration_yield_form': ('Duration of yield formation', 'degree-days'),
            'cgc': ('Canopy growth coefficient', 'dimensionless'),
            'cdc': ('Canopy decline coefficient', 'dimensionless'),
            'temp_max_avg': ('Average maximum temperature', 'degC'),
            'temp_min_avg': ('Average minimum temperature', 'degC'),
        }

        # Generate both rainfed and irrigated versions
        for irr_type in ['rf', 'ir']:
            nc_path = params_dir / f"{crop_short}_{irr_type}_crop_parameters.nc"

            nc = Dataset(str(nc_path), 'w', format='NETCDF4')

            # Create dimensions
            nc.createDimension('lat', self.GRID_ROWS_30ARCMIN)
            nc.createDimension('lon', self.GRID_COLS_30ARCMIN)

            # Create coordinate variables
            lat_var = nc.createVariable('lat', 'f8', ('lat',))
            lat_var.long_name = 'latitude'
            lat_var.units = 'degrees_north'
            lat_var[:] = np.arange(89.75, -90, -0.5)

            lon_var = nc.createVariable('lon', 'f8', ('lon',))
            lon_var.long_name = 'longitude'
            lon_var.units = 'degrees_east'
            lon_var[:] = np.arange(-179.75, 180, 0.5)

            # Create data variables for each parameter
            for param_name, (long_name, units) in param_metadata.items():
                param_var = nc.createVariable(
                    param_name, 'f4', ('lat', 'lon'),
                    fill_value=np.nan, zlib=True, complevel=6
                )
                param_var.long_name = long_name
                param_var.units = units

                # Initialize with fill values
                param_data = np.full((self.GRID_ROWS_30ARCMIN, self.GRID_COLS_30ARCMIN),
                                     np.nan, dtype='f4')

                # Fill data for cells in our grid with uniform value
                value = params_dict.get(param_name, np.nan)
                for cell_id in cell_ids_30arcmin:
                    row, col = self._cell_id_to_row_col(cell_id)
                    if 0 <= row < self.GRID_ROWS_30ARCMIN and 0 <= col < self.GRID_COLS_30ARCMIN:
                        param_data[row, col] = value

                param_var[:, :] = param_data

            # Global attributes
            nc.title = f"{crop_short.upper()} {'Rainfed' if irr_type == 'rf' else 'Irrigated'} Crop Parameters"
            nc.source = "prismpy framework"
            nc.conventions = "CF-1.6"
            nc.param_type = "rainfed" if irr_type == "rf" else "irrigated"
            nc.grid_resolution = "30-arcmin (0.5 degree)"
            nc.grid_dimensions = "360 x 720 (global)"
            nc.cells_with_data = len(cell_ids_30arcmin)
            nc.institution = "prismpy"

            nc.close()
            output_files.append(nc_path)
            logger.debug(f"Generated crop params NetCDF: {nc_path}")

        logger.info(f"Generated {len(output_files)} ACEA crop parameters NetCDF files")
        return output_files

    def _get_default_acea_params(self, crop_params: CropParameters) -> Dict[str, float]:
        """Get default ACEA parameters with crop-specific values.

        Uses CROP_GDD_DEFAULTS lookup table with fallback to Maize defaults
        for unknown crops. Config values take precedence when provided.

        Args:
            crop_params: CropParameters object

        Returns:
            Dictionary with 11 ACEA parameters
        """
        # Get crop name for lookup
        crop_name = self.config.crop.name

        # Get crop-specific defaults with fallback to Maize
        defaults = CROP_GDD_DEFAULTS.get(crop_name, CROP_GDD_DEFAULTS['Maize'])

        # Log if using fallback
        if crop_name not in CROP_GDD_DEFAULTS:
            logger.warning(f"No GDD defaults for crop '{crop_name}', using Maize defaults. "
                          f"Known crops: {', '.join(sorted(CROP_GDD_DEFAULTS.keys()))}")
            # V2-19 C5 (TA-05): ACEA Maize silent fallback — parallel
            # to C2/TP-06 (PYTHIA CROPGRO→CERES). Phenology is wrong for
            # any non-maize crop that falls through to Maize defaults.
            if self.provenance:
                self.provenance.record_decision(
                    decision_type=DecisionType.FALLBACK_SUBSTITUTION,
                    description=(
                        f"ACEA Maize phenology fallback: crop "
                        f"'{crop_name}' not in CROP_GDD_DEFAULTS"
                    ),
                    rationale=(
                        f"Crop '{crop_name}' has no entry in "
                        f"CROP_GDD_DEFAULTS (known: Maize, Wheat, Rice, "
                        f"Sorghum, Millet, etc.). Falling back to Maize "
                        f"phenology (gdd_maturity=1600, base_temp=8\u00b0C). "
                        f"This is scientifically inappropriate — different "
                        f"crops have fundamentally different thermal time "
                        f"requirements, canopy dynamics (CGC/CDC), and "
                        f"harvest indices. Results using Maize phenology "
                        f"for a non-maize crop should be treated as "
                        f"indicative only. V2-20 will add strict_mode "
                        f"enforcement parallel to PYTHIA TP-06."
                    ),
                    alternatives=[
                        "Add crop-specific CROP_GDD_DEFAULTS entry",
                        "User-provided phenology overrides in config",
                        "Raise error instead of silent fallback (V2-20)",
                    ],
                    reference=(
                        "prismpy.translators.acea.translator "
                        "CROP_GDD_DEFAULTS.get(crop_name, "
                        "CROP_GDD_DEFAULTS['Maize'])"
                    ),
                    severity="warning",
                    label=f"ACEA: '{crop_name}' using Maize phenology",
                )

        # Use config values if provided, otherwise use crop-specific defaults
        # Access base_temp through get_param() method since it's not a direct attribute
        base_temp = (crop_params.get_param('base_temp') if hasattr(crop_params, 'get_param')
                     else getattr(crop_params, 'base_temp', None)) or defaults.get('base_temp', 8.0)

        # Access optimal_temp through get_param() method
        optimal_temp = (crop_params.get_param('optimal_temp') if hasattr(crop_params, 'get_param')
                        else getattr(crop_params, 'optimal_temp', None)) or 30.0

        # Check AceaConfig for GDD overrides (e.g., from literature calibration)
        platform_config = self.get_platform_config()
        gdd_maturity_override = getattr(platform_config, 'gdd_maturity', None) if platform_config else None
        gdd_senescence_override = getattr(platform_config, 'gdd_senescence', None) if platform_config else None
        gdd_max_root_override = getattr(platform_config, 'gdd_max_root', None) if platform_config else None

        return {
            'gdd_emergence': getattr(crop_params, 'emergence_gdd', None) or defaults['gdd_emergence'],
            'gdd_max_root': gdd_max_root_override or defaults['gdd_max_root'],
            'gdd_senescence': gdd_senescence_override or defaults['gdd_senescence'],
            'gdd_maturity': gdd_maturity_override or defaults['gdd_maturity'],
            'gdd_yield_form': defaults['gdd_yield_form'],
            'gdd_duration_flowering': defaults['gdd_duration_flowering'],
            'gdd_duration_yield_form': defaults['gdd_duration_yield_form'],
            'cgc': defaults['cgc'],
            'cdc': defaults['cdc'],
            'temp_max_avg': optimal_temp,
            'temp_min_avg': base_temp + 5.0,  # Slightly above base
        }

    def _generate_harvested_areas(
        self,
        grid: SpatialGrid,
        cell_ids_30arcmin: List[int],
    ) -> Path:
        """Generate ACEA harvested areas CSV.

        Args:
            grid: SpatialGrid with cells
            cell_ids_30arcmin: 30-arcmin cell IDs

        Returns:
            Path to generated CSV
        """
        csv_path = self.output_dir / "harvested_areas" / "areas.csv"

        with open(csv_path, 'w') as f:
            f.write("cell_id,lat,lon,harvested_area_ha\n")

            for i, cell in enumerate(grid.cells):
                cell_id = cell_ids_30arcmin[i] if i < len(cell_ids_30arcmin) else cell.cell_id
                # Get harvested area from metadata if available, otherwise default to 0
                metadata = getattr(cell, 'metadata', {}) or {}
                area = metadata.get("harvested_area", 0.0) if isinstance(metadata, dict) else 0.0
                f.write(f"{cell_id},{cell.lat},{cell.lon},{area}\n")

        logger.info(f"Generated ACEA harvested areas: {csv_path}")
        return csv_path

    def _generate_co2_data(self) -> Path:
        """Generate CO2 concentration file with embedded historical data.

        ACEA expects a tab-separated file with columns: Year, CO2_ppm
        Contains NOAA historical CO2 data from 1980-2030.

        Returns:
            Path to generated CO2 file
        """
        co2_dir = self.output_dir / "co2"
        co2_dir.mkdir(parents=True, exist_ok=True)

        # Embedded historical CO2 data (NOAA observations + projections)
        # Source: NOAA Global Monitoring Laboratory
        CO2_DATA = {
            1980: 338.91, 1981: 340.11, 1982: 341.45, 1983: 343.05,
            1984: 344.65, 1985: 346.35, 1986: 347.61, 1987: 349.31,
            1988: 351.69, 1989: 353.20, 1990: 354.45, 1991: 355.70,
            1992: 356.54, 1993: 357.21, 1994: 358.96, 1995: 360.97,
            1996: 362.74, 1997: 363.88, 1998: 366.84, 1999: 368.54,
            2000: 369.71, 2001: 371.32, 2002: 373.45, 2003: 375.98,
            2004: 377.70, 2005: 380.12, 2006: 382.00, 2007: 384.04,
            2008: 385.83, 2009: 387.64, 2010: 390.10, 2011: 391.85,
            2012: 394.06, 2013: 396.74, 2014: 398.87, 2015: 401.01,
            2016: 404.41, 2017: 406.76, 2018: 408.72, 2019: 411.66,
            2020: 414.24, 2021: 416.45, 2022: 418.56, 2023: 421.08,
            2024: 423.50, 2025: 426.00, 2026: 428.50, 2027: 431.00,
            2028: 433.50, 2029: 436.00, 2030: 438.50,  # Projections
        }

        co2_path = co2_dir / "GlobalHistoricalCO2_NOAA_1980_2020.txt"
        with open(co2_path, 'w') as f:
            f.write("Year\tCO2_ppm\n")
            for year, co2 in sorted(CO2_DATA.items()):
                f.write(f"{year}\t{co2:.2f}\n")

        logger.info(f"Generated ACEA CO2 data: {co2_path}")
        return co2_path

    def _generate_acea_config(
        self,
        data: UnifiedData,
        cell_ids_30arcmin: List[int],
        climate_name: str,
    ) -> Path:
        """Generate ACEA project_conf Python configuration file.

        Args:
            data: UnifiedData with all configuration info
            cell_ids_30arcmin: 30-arcmin cell IDs
            climate_name: Name prefix for climate files (e.g., 'oromia_nasapower')

        Returns:
            Path to generated Python config file
        """
        # Get configuration values
        project_name = self.config.project.name
        region_name = data.region.name
        country = data.region.country
        crop_name = self.config.crop.name
        # Use name_short (e.g., 'mai' for maize) - NOT variety
        crop_short = self.config.crop.name_short

        # Temporal settings
        start_year = self.config.temporal.start_year
        end_year = self.config.temporal.end_year
        spinup_years = self.config.temporal.spinup_years
        climate_start = start_year - spinup_years
        climate_end = end_year

        clock_start = f"{climate_start}/01/01"
        crop_cal = self.config.crop.calendar if self.config.crop else None
        clock_end = self.config.temporal.get_climate_end_date(crop_cal).strftime('%Y/%m/%d')

        # Format grid cells list
        gridcells_str = self._format_gridcells(cell_ids_30arcmin)

        # Crop mappings
        gaez_cultivar_map = {
            'Maize': ['Highland_maize', 'Lowland_maize', 'Maize', 'Temperate_maize'],
            'Wheat': ['Spring_wheat', 'Winter_wheat'],
            'Rice': ['Wetland_rice', 'Dryland_rice'],
            'Sorghum': ['Highland_sorghum', 'Lowland_sorghum'],
            'Millet': ['Pearl_millet', 'Foxtail_millet'],
        }
        crop_gaez = gaez_cultivar_map.get(crop_name, [crop_name])

        spam_code_map = {
            'Maize': 'maiz', 'Wheat': 'whea', 'Rice': 'rice',
            'Sorghum': 'sorg', 'Millet': 'mill',
        }
        crop_name_4code = spam_code_map.get(crop_name, crop_short[:4].lower())

        # Platform config
        platform_config = self.get_platform_config()
        resolution = 1  # ACEA: 1 = 5arcmin, 0 = 30arcmin (integer!)
        co2_name = "GlobalHistoricalCO2_NOAA_1980_2020"
        scenarios = [1]  # ACEA: 1=rainfed, 2=irrigated (integers!)
        soil_fertility = 0

        # Irrigation and field management defaults
        irr_thresholds = [50] * 4
        bunds = False
        bunds_dz = 0.3
        virtual_irrigation = 'Lowinput'

        if platform_config:
            if hasattr(platform_config, 'resolution'):
                # Convert resolution string to ACEA integer
                res_val = platform_config.resolution
                if res_val == "30arcmin" or res_val == 0:
                    resolution = 0
                else:
                    resolution = 1  # Default to 5arcmin
            if hasattr(platform_config, 'scenarios'):
                scenarios = platform_config.scenarios
            # Irrigation config from AceaConfig
            if hasattr(platform_config, 'bunds'):
                bunds = platform_config.bunds
            if hasattr(platform_config, 'bunds_dz'):
                bunds_dz = platform_config.bunds_dz
            if hasattr(platform_config, 'irr_thresholds') and platform_config.irr_thresholds is not None:
                irr_thresholds = platform_config.irr_thresholds
            if hasattr(platform_config, 'virtual_irrigation'):
                virtual_irrigation = platform_config.virtual_irrigation
            if hasattr(platform_config, 'soil_fertility') and platform_config.soil_fertility is not None:
                soil_fertility = platform_config.soil_fertility

        # Auto-detect irrigation from management config if not explicitly set in platform config
        management = self.config.management if hasattr(self.config, 'management') else None
        if management and getattr(management, 'irrigation', False):
            # If management.irrigation is True and platform config didn't explicitly set values,
            # upgrade to irrigated defaults
            if not (platform_config and getattr(platform_config, 'bunds', False)):
                # Only auto-set if not already configured
                if virtual_irrigation == 'Lowinput':
                    virtual_irrigation = 'Highinput'
            if scenarios == [1]:
                # Scenario 3 = surface irrigation (no groundwater data needed)
                # Scenario 2 = groundwater irrigation (requires GW_monthly_5arcmin_50m.nc)
                scenarios = [1, 3]

        # Generate config content
        config_content = ACEA_CONFIG_TEMPLATE.format(
            project_name=project_name,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            region_name=region_name,
            country=country,
            crop_name=crop_name,
            crop_name_short=crop_short,
            crop_fao=ACEA_FAO_CODE_MAP.get(crop_name, 999),  # ACEA internal code lookup
            crop_name_4code=crop_name_4code,
            crop_gaez=crop_gaez,
            start_year=start_year,
            end_year=end_year,
            climate_start=climate_start,
            climate_end=climate_end,
            clock_start=clock_start,
            clock_end=clock_end,
            spinup=spinup_years,
            landuse='spam2020',
            resolution=resolution,
            gridcells=gridcells_str,
            climate_name=climate_name,
            co2_name=co2_name,
            scenarios=scenarios,
            virtual_irrigation=virtual_irrigation,
            irr_thresholds=irr_thresholds,
            bunds=bunds,
            bunds_dz=bunds_dz,
            soil_fertility=soil_fertility,
            multi_core='True',
            cpus=4,
        )

        # Write Python config file
        config_filename = f"{project_name}_config.py"
        config_path = self.output_dir / "config" / config_filename
        with open(config_path, 'w') as f:
            f.write(config_content)

        logger.info(f"Generated ACEA config: {config_path}")

        # Also generate JSON summary
        json_path = self.output_dir / "config" / f"{project_name}_config.json"
        summary = {
            'project_name': project_name,
            'region': region_name,
            'country': country,
            'crop': crop_name,
            'start_year': start_year,
            'end_year': end_year,
            'resolution': resolution,
            'n_cells': len(cell_ids_30arcmin),
            'cell_id_range': [min(cell_ids_30arcmin), max(cell_ids_30arcmin)]
                if cell_ids_30arcmin else [0, 0],
            'cell_id_type': '30-arcmin',
            'scenarios': scenarios,
            'generated': datetime.now().isoformat()
        }
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return config_path

    def _format_gridcells(self, cell_ids: List[int]) -> str:
        """Format grid cells list for Python config file.

        Args:
            cell_ids: List of 30-arcmin cell IDs

        Returns:
            Formatted string for Python list
        """
        if len(cell_ids) <= 5:
            return str(cell_ids)

        # Multi-line format for readability
        lines = ["["]
        for i in range(0, len(cell_ids), 10):
            chunk = cell_ids[i:i+10]
            lines.append("        " + ", ".join(str(c) for c in chunk) + ",")
        lines[-1] = lines[-1].rstrip(',')  # Remove trailing comma
        lines.append("    ]")
        return "\n".join(lines)
