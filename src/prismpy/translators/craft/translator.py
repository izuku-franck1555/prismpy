"""
CRAFT translator for prismpy.

This module translates unified data to CRAFT model input format:
- Tab-separated schema files for grid cells
- Tab-separated weather files
- DSSAT ML.SOL format soil files
- Tab-separated crop mask and management files

CRAFT Quirks (from analysis):
1. CellID formula: row * 4320 + col (5-arcmin global grid)
2. Admin name sanitization: Windows-forbidden chars + accents removed
3. Tab-separated output: Specific column order required
4. Uses NASA POWER for climate, HWSD for soil

Reference: CRAFT-Notes-Python/07-OUTPUT-GENERATION/
"""

import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from prismpy.config.schema import Platform
from prismpy.data_sources.gadm import GADMDataSource
from prismpy.models.climate import ClimateTimeSeries, ClimateRecord
from prismpy.models.crop import CropParameters, CropCalendar
from prismpy.models.region import Region
from prismpy.models.soil import SoilProfile, SoilLayer
from prismpy.models.spatial import SpatialGrid
from prismpy.provenance.tracker import DecisionType, ProvenanceTracker
from prismpy.translators.base import (
    BaseTranslator,
    CraftTranslatorBase,
    TranslationResult,
    UnifiedData,
)
from prismpy.utils.sanitization import sanitize_admin_name
from prismpy.utils.date_utils import date_to_yrdoy, doy_to_date
from prismpy.utils.zones import get_management_for_cell, get_zone_summary
from prismpy.data_sources.gadm import GADMDataSource
from prismpy.sources.soil.hwsd import HWSDSource, HWSDConfig
from prismpy.config.defaults import DEFAULT_CLIMATE_START_YEAR
from prismpy.packaging.manifest import create_manifest, save_manifest
from prismpy.packaging.readme_generator import generate_readme


logger = logging.getLogger(__name__)


# Type alias for soil mapping
SoilMapping = Dict[int, str]  # cell_id -> profile_name


class CraftTranslator(CraftTranslatorBase):
    """Translator for CRAFT crop forecast system.

    CRAFT (Crop Risk Assessment Forecasting Tool) uses DSSAT for simulations
    with a specific data organization structure.

    All parameters are configurable via the config file - no hardcoded
    region/crop-specific values. Country codes, cultivar IDs, fertilizer
    settings, and planting dates all come from user configuration.

    Generates:
    1. schema/CRAFT_Schema/Level{N}/Schema/5m_{Admin}.txt - CRAFT database schema
    2. schema/Python_Schemas/Level{N}/Schema_{Admin}.txt - Python scripts schema
    3. weather/input.csv - NASA POWER download input file
    4. soil/{CC}.SOL - DSSAT soil profile database (CC = country code)
    5. soil/soil_mask.txt - CellID to SoilProfile mapping
    6. crop_mask/mask.txt - Crop presence indicators (CellId, Percent)
    7. management/cultivar_data.txt - Cultivar assignments per cell
    8. management/planting_data.txt - Planting parameters per cell (MMDD format)
    9. management/fertilizer_data.txt - Split fertilizer applications per cell
    10. management/organic_fertilizer_data.txt - Organic residue applications

    Output structure:
        output_dir/
        ├── schema/
        │   ├── CRAFT_Schema/Level{N}/Schema/5m_{Admin}.txt
        │   └── Python_Schemas/Level{N}/Schema_{Admin}.txt
        ├── weather/
        │   └── input.csv
        ├── soil/
        │   ├── {CC}.SOL
        │   └── soil_mask.txt
        ├── crop_mask/
        │   └── mask.txt
        └── management/
            ├── cultivar_data.txt
            ├── planting_data.txt
            ├── fertilizer_data.txt
            └── organic_fertilizer_data.txt
    """

    def translate(self, data: UnifiedData) -> TranslationResult:
        """Translate unified data to CRAFT format.

        Args:
            data: UnifiedData container with region, grid, climate, soil data

        Returns:
            TranslationResult with output files and status
        """
        self.log_translation_start(data)
        errors = []
        warnings = []
        output_files = []

        # Initialize GADM-filtered cell tracking
        # This will be populated by _generate_craft_schema() if GADM is used
        # All management files will use these cells for consistency
        self._valid_cellids: Optional[set] = None
        # Store full GADM cell data (cellid, lat, lon) for management files
        # This ensures all GADM cells are used even if not in grid bounding box
        self._gadm_cells: Optional[List[Dict]] = None

        # Validate input data (base validation: region, grid, climate, soil)
        input_errors = self.validate_input_data(data)
        if input_errors:
            return self.create_result(
                success=False,
                output_files=[],
                errors=input_errors,
            )

        # CRAFT-specific validation: require country and crop calendar
        craft_errors = []
        if data.region and not data.region.country:
            craft_errors.append("Region.country is required (e.g., 'Mali', 'Kenya') - needed for soil file naming and profile IDs")
        if not self.config.crop.calendar:
            craft_errors.append("crop.calendar with planting_doy is required - CRAFT cannot use hardcoded defaults")
        elif not self.config.crop.calendar.planting_doy:
            craft_errors.append("crop.calendar.planting_doy is required for planting and management data")

        if craft_errors:
            return self.create_result(
                success=False,
                output_files=[],
                errors=craft_errors,
            )

        # Create output subdirectories
        for subdir in self.OUTPUT_SUBDIRS:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        try:
            # Get platform config for feature flags
            platform_config = self.get_platform_config()

            # 1. Generate CRAFT_Schema (proper hierarchical format for CRAFT database)
            craft_schema_files = self._generate_craft_schema(data.grid, data.region)
            output_files.extend(craft_schema_files)

            # 1b. Generate Python_Schemas (for internal Python scripts 02-08) if enabled
            generate_python_schema = True
            if platform_config:
                generate_python_schema = getattr(platform_config, 'generate_python_schema', True)
            if generate_python_schema:
                python_schema_files = self._generate_python_schema(data.grid, data.region)
                output_files.extend(python_schema_files)

            # 2. Generate weather input CSV for NASA Power download via CRAFT GUI
            # This is the standard CRAFT workflow - download weather through GUI
            if data.grid:
                weather_input_csv = self._generate_weather_input_csv(data.grid)
                output_files.append(weather_input_csv)

            # 2b. If actual climate data is provided, also generate WTH files
            if data.climate and len(data.climate) > 0:
                # Check if we have real data (not just placeholder with cell_id=-1)
                has_real_data = any(
                    cell_id >= 0 and ts.records and len(ts.records) > 1
                    for cell_id, ts in data.climate.items()
                )
                if has_real_data:
                    # Filter out placeholder data
                    real_climate = {
                        cell_id: ts for cell_id, ts in data.climate.items()
                        if cell_id >= 0 and ts.records and len(ts.records) > 1
                    }
                    if real_climate:
                        weather_files = self._generate_weather_files(real_climate)
                        output_files.extend(weather_files)

            # 3. Generate soil package (.SOL file + soil_mask.txt)
            # Uses HWSD data if configured, otherwise creates default profile
            cell_to_profile: Optional[SoilMapping] = None
            if data.grid:
                soil_file, cell_to_profile = self._generate_soil_package(
                    grid=data.grid,
                    region=data.region,
                    existing_soil_data=data.soil,
                )
                output_files.append(soil_file)

            # 4. Generate soil mask (uses mapping from soil package)
            include_soil_mask = True
            if platform_config:
                include_soil_mask = getattr(platform_config, 'include_soil_mask', True)
            if include_soil_mask and data.grid:
                soil_mask_file = self._generate_soil_mask(
                    grid=data.grid,
                    region=data.region,
                    cell_to_profile=cell_to_profile,
                )
                output_files.append(soil_mask_file)

            # 5. Generate crop mask
            if data.grid:
                mask_file = self._generate_crop_mask(data.grid, data.crop_calendar)
                output_files.append(mask_file)

            # 6. Generate cultivar data (NEW)
            if data.grid:
                cultivar_file = self._generate_cultivar_data(data.grid)
                output_files.append(cultivar_file)

            # 7. Generate planting data (NEW - with MMDD format)
            if data.grid:
                planting_file = self._generate_planting_data(data.grid, data.crop_calendar)
                output_files.append(planting_file)

            # 8. Generate fertilizer data (NEW - 2 applications per cell)
            if data.grid:
                fertilizer_file = self._generate_fertilizer_data(data.grid)
                output_files.append(fertilizer_file)

            # 8b. Generate organic fertilizer data (residue)
            if data.grid:
                organic_fert_file = self._generate_organic_fertilizer_data(data.grid)
                output_files.append(organic_fert_file)

            # 9. Generate legacy management file (for backward compatibility)
            management_file = self._generate_management(data.crop_params, data.crop_calendar)
            output_files.append(management_file)

            # Validate outputs
            validation_errors = self.validate_outputs()
            if validation_errors:
                warnings.extend(validation_errors)

            # 10. Generate package metadata files (manifest, README, provenance)
            try:
                package_files = self._generate_package_metadata(data, output_files)
                output_files.extend(package_files)
            except Exception as pkg_error:
                logger.warning(f"Failed to generate package metadata: {pkg_error}")
                warnings.append(f"Package metadata generation failed: {pkg_error}")

        except Exception as e:
            logger.error(f"CRAFT translation failed: {e}")
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
                description=f"Generated CRAFT inputs for {data.region.name}",
                rationale="CRAFT requires tab-separated files with DSSAT soil format",
                alternatives=["manual configuration"],
                reference=f"Output: {self.output_dir}",
            )

        result = self.create_result(
            success=True,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "region": data.region.name,
                "n_cells": len(data.grid.cells) if data.grid else 0,
                "n_weather_files": len(data.climate) if data.climate else 0,
                "n_soil_profiles": len(data.soil) if data.soil else 0,
            },
        )

        self.log_translation_complete(result)
        return result

    def validate_outputs(self) -> List[str]:
        """Validate generated CRAFT outputs.

        Returns:
            List of validation error messages
        """
        errors = []

        # Check CRAFT_Schema exists (hierarchical schema for CRAFT database)
        craft_schema_dir = self.output_dir / "schema" / "CRAFT_Schema"
        if not craft_schema_dir.exists():
            errors.append(f"Missing CRAFT_Schema directory at {craft_schema_dir}")
        else:
            # Check for at least one schema file
            schema_files = list(craft_schema_dir.glob("**/5m_*.txt"))
            if not schema_files:
                errors.append("No CRAFT schema files (5m_*.txt) found in CRAFT_Schema/")
            else:
                # Validate first schema file structure
                try:
                    with open(schema_files[0], 'r') as f:
                        header = f.readline().strip()
                        # CRAFT schema format: CELLID\tSHAREPERCENT
                        required_cols = ["CELLID", "SHAREPERCENT"]
                        for col in required_cols:
                            if col not in header:
                                errors.append(f"Missing column '{col}' in CRAFT schema")
                except Exception as e:
                    errors.append(f"Error reading CRAFT schema: {e}")

        # Check weather directory has input.csv (for NASA Power download)
        weather_dir = self.output_dir / "weather"
        if weather_dir.exists():
            input_csv = weather_dir / "input.csv"
            if not input_csv.exists():
                errors.append("No weather input.csv generated in weather/")

        # Check soil file exists (any .SOL file)
        soil_dir = self.output_dir / "soil"
        if soil_dir.exists():
            sol_files = list(soil_dir.glob("*.SOL"))
            if not sol_files:
                errors.append(f"No .SOL soil file found in {soil_dir}")
        else:
            errors.append(f"Missing soil directory at {soil_dir}")

        return errors

    def _to_craft_cellid(self, cell_id_0: int) -> int:
        """Convert 0-indexed cell ID to CRAFT 1-indexed format.

        CRAFT uses 1-indexed CellIDs: row * 4320 + col + 1
        Our internal representation is 0-indexed: row * 4320 + col

        Args:
            cell_id_0: 0-indexed cell ID from SpatialGrid

        Returns:
            1-indexed CellID for CRAFT format
        """
        return cell_id_0 + 1

    def _get_filtered_cells(self, grid: SpatialGrid) -> List:
        """Get cells filtered by GADM boundary if available.

        If GADM filtering was used during schema generation, returns the
        authoritative GADM cell list (with lat/lon). This ensures ALL GADM
        cells are used, even those outside the grid bounding box.

        This ensures consistency between schema and management files:
        - Schema: 149 cells (GADM-filtered)
        - Management files: 149 cells (same GADM-filtered cells)

        Args:
            grid: SpatialGrid with all cells

        Returns:
            List of cell-like objects to use (GADM cells if available, grid cells if not)
        """
        if self._gadm_cells is not None:
            # Use authoritative GADM cells (includes cells outside grid bounding box)
            # Convert dict format to cell-like objects for consistency
            class GadmCell:
                def __init__(self, data):
                    self.cell_id = data['cellid'] - 1  # Convert back to 0-indexed
                    self.lat = data['lat']
                    self.lon = data['lon']

            return [GadmCell(cell_data) for cell_data in self._gadm_cells]
        elif self._valid_cellids is not None:
            # Fallback: Filter grid cells by valid IDs (may miss some cells)
            filtered = [
                cell for cell in grid.cells
                if self._to_craft_cellid(cell.cell_id) in self._valid_cellids
            ]
            return filtered
        else:
            # No GADM filtering - use all grid cells
            return list(grid.cells)

    def _generate_craft_schema(self, grid: SpatialGrid, region: Region) -> List[Path]:
        """Generate CRAFT schema files in proper hierarchical structure.

        Produces the standard CRAFT_Schema structure for database upload:
        - schema/CRAFT_Schema/Level{N}/Schema/5m_{AdminNames}.txt

        File format: CELLID\tSHAREPERCENT (tab-separated)

        SharePercent calculation priority:
        1. GADM: Uses GADM shapefile for accurate boundary-based calculation (recommended)
        2. Manual shapefile: Uses admin_shapefile_path if provided
        3. Bounding box: 100% for all cells (simplified mode with warning)

        GADM vs bounding box difference:
        - GADM: Only cells that INTERSECT the actual boundary are included
        - Bounding box: ALL cells in the rectangular bounds are included

        Args:
            grid: SpatialGrid with cells
            region: Region for metadata

        Returns:
            List of paths to generated schema files
        """
        output_files = []
        platform_config = self.get_platform_config()

        # Get schema configuration from platform config
        schema_level = 2  # Default to Level 2 (state/region)
        admin_level1_name = None
        admin_level2_name = None
        admin_level3_name = None
        decimal_places = 2

        # GADM configuration
        gadm_data_path = None
        gadm_country_iso3 = None
        gadm_admin_name = None

        # Legacy shapefile path
        shapefile_path = None

        if platform_config:
            schema_level = getattr(platform_config, 'schema_level', 2)
            admin_level1_name = getattr(platform_config, 'admin_level1_name', None)
            admin_level2_name = getattr(platform_config, 'admin_level2_name', None)
            admin_level3_name = getattr(platform_config, 'admin_level3_name', None)
            decimal_places = getattr(platform_config, 'schema_decimal_places', 2)

            # GADM configuration
            gadm_data_path = getattr(platform_config, 'gadm_data_path', None)
            gadm_country_iso3 = getattr(platform_config, 'gadm_country_iso3', None)
            gadm_admin_name = getattr(platform_config, 'gadm_admin_name', None)

            # Legacy shapefile path
            shapefile_path = getattr(platform_config, 'admin_shapefile_path', None)

        # Derive admin names from region if not explicitly set
        if not admin_level1_name:
            admin_level1_name = sanitize_admin_name(region.country or "Unknown")
        if not admin_level2_name:
            admin_level2_name = sanitize_admin_name(region.name or "Unknown")
        if not gadm_admin_name:
            gadm_admin_name = region.name  # Use region name for GADM filtering

        # Create CRAFT_Schema directory structure (inside schema/)
        level_dir = self.output_dir / "schema" / "CRAFT_Schema" / f"Level{schema_level}" / "Schema"
        level_dir.mkdir(parents=True, exist_ok=True)

        # Build filename based on level
        if schema_level == 1:
            filename = f"5m_{admin_level1_name}.txt"
        elif schema_level == 2:
            filename = f"5m_{admin_level1_name}_{admin_level2_name}.txt"
        else:  # Level 3
            if not admin_level3_name:
                logger.warning("Level 3 requires admin_level3_name - using region name")
                admin_level3_name = admin_level2_name
            filename = f"5m_{admin_level1_name}_{admin_level2_name}_{admin_level3_name}.txt"

        schema_path = level_dir / filename

        # =====================================================================
        # OPTION 1: Use GADM for accurate boundary-based schema (RECOMMENDED)
        # =====================================================================
        if gadm_data_path and gadm_country_iso3:
            craft_schema_rows, python_schema_rows = self._generate_schema_from_gadm(
                gadm_data_path=gadm_data_path,
                country_iso3=gadm_country_iso3,
                schema_level=schema_level,
                admin_name=gadm_admin_name,
                decimal_places=decimal_places,
            )

            if craft_schema_rows:
                # Store valid cell IDs for use by management files
                # This ensures all output files use the same GADM-filtered cells
                self._valid_cellids = {row['cellid'] for row in craft_schema_rows}

                # Store full GADM cell data (cellid, lat, lon) for management files
                # This ensures management files use ALL GADM cells, even those
                # outside the grid bounding box
                self._gadm_cells = python_schema_rows

                logger.info(f"GADM filtering: {len(self._valid_cellids)} valid cells "
                           f"(management files will use same cells)")

                # Write CRAFT schema file from GADM data
                with open(schema_path, 'w') as f:
                    f.write("CELLID\tSHAREPERCENT\n")
                    for row in craft_schema_rows:
                        f.write(f"{row['cellid']}\t{row['share_percent']}\n")

                output_files.append(schema_path)
                logger.info(f"Generated CRAFT schema from GADM: {schema_path} "
                           f"({len(craft_schema_rows)} cells, level {schema_level})")
                return output_files
            else:
                logger.warning("GADM schema generation returned no cells - falling back to grid-based method")

        # =====================================================================
        # OPTION 2: Use grid cells + shapefile/bounding box (fallback)
        # =====================================================================
        # Calculate SharePercent for each cell
        share_percents = self._calculate_share_percent(grid, shapefile_path, decimal_places)

        # Write CRAFT schema file
        with open(schema_path, 'w') as f:
            f.write("CELLID\tSHAREPERCENT\n")

            # Write cells sorted by CELLID descending (CRAFT convention)
            cells_with_share = []
            for cell in grid.cells:
                craft_cellid = self._to_craft_cellid(cell.cell_id)
                share_pct = share_percents.get(cell.cell_id, 100.0)
                # Only include cells with non-zero share (intersection with boundary)
                if share_pct > 0:
                    cells_with_share.append((craft_cellid, share_pct))

            # Sort descending by CellID
            cells_with_share.sort(key=lambda x: x[0], reverse=True)

            for cellid, share_pct in cells_with_share:
                f.write(f"{cellid}\t{share_pct}\n")

        output_files.append(schema_path)
        logger.info(f"Generated CRAFT schema: {schema_path} ({len(cells_with_share)} cells, level {schema_level})")

        return output_files

    def _generate_schema_from_gadm(
        self,
        gadm_data_path: Path,
        country_iso3: str,
        schema_level: int,
        admin_name: Optional[str] = None,
        decimal_places: int = 2,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Generate schema data directly from GADM boundaries.

        This is the recommended method for accurate schema generation.
        It creates a fishnet grid covering the exact boundary polygon and
        calculates accurate SharePercent and Area values.

        Args:
            gadm_data_path: Path to GADM shapefiles directory
            country_iso3: ISO3 country code (e.g., "MLI", "NGA")
            schema_level: CRAFT schema level (1=country, 2=state, 3=district)
            admin_name: Admin region name to filter to
            decimal_places: Decimal places for SharePercent

        Returns:
            Tuple of (craft_schema_rows, python_schema_rows)
        """
        # Get GADM level from config, or default to schema_level - 1
        # Override needed when country admin structure differs from standard
        # (e.g., Mali cercles are at GADM L2, but CRAFT uses them as Level 2)
        platform_config = self.get_platform_config()
        config_gadm_level = None
        if platform_config:
            config_gadm_level = getattr(platform_config, 'gadm_level', None)

        if config_gadm_level is not None:
            gadm_level = config_gadm_level
            logger.info(f"Using configured GADM level: {gadm_level} (from gadm_level config)")
        else:
            # Default: GADM level = CRAFT level - 1
            gadm_level = schema_level - 1
            logger.info(f"Using default GADM level: {gadm_level} (schema_level - 1)")

        logger.info(f"Loading GADM data: {country_iso3} level {gadm_level} (CRAFT level {schema_level})")

        # Initialize GADM data source
        gadm = GADMDataSource(gadm_path=str(gadm_data_path))

        # Load shapefile
        gdf = gadm.load_shapefile(country_iso3, gadm_level)
        if gdf is None:
            logger.warning(f"Could not load GADM shapefile for {country_iso3} level {gadm_level}")
            return [], []

        # Filter to specific admin region if name provided
        if admin_name:
            gdf = gadm.filter_by_name(gdf, gadm_level, admin_name)
            if gdf is None or len(gdf) == 0:
                logger.warning(f"Admin region '{admin_name}' not found in GADM level {gadm_level}")
                return [], []

        # Generate schema data from GADM boundary
        craft_rows, python_rows = gadm.generate_schema_data(
            gdf=gdf,
            resolution_deg=5/60,  # 5 arcmin
            decimal_places=decimal_places,
        )

        logger.info(f"GADM schema generation: {len(craft_rows)} cells from boundary")

        return craft_rows, python_rows

    def _generate_python_schema(self, grid: SpatialGrid, region: Region) -> List[Path]:
        """Generate Python schema files for internal use (scripts 02-08).

        Format: CellID\tLatitude\tLongitude\tElevation\tArea\tLevel{N}Name (tab-separated)

        IMPORTANT: Area represents the INTERSECTION area between each cell and the
        admin boundary, NOT the full cell area.

        Area calculation priority:
        1. GADM: Uses GADM shapefile for accurate boundary-based calculation
        2. Manual shapefile: Uses admin_shapefile_path if provided
        3. Bounding box: Uses full cell area (inaccurate, with warning)

        Args:
            grid: SpatialGrid with cells
            region: Region for metadata

        Returns:
            List of paths to generated Python schema files
        """
        import math

        output_files = []
        platform_config = self.get_platform_config()

        # Get schema configuration
        schema_level = 2
        admin_level1_name = None
        admin_level2_name = None
        admin_level3_name = None
        decimal_places = 2

        # GADM configuration
        gadm_data_path = None
        gadm_country_iso3 = None
        gadm_admin_name = None

        # Legacy shapefile path
        shapefile_path = None

        if platform_config:
            schema_level = getattr(platform_config, 'schema_level', 2)
            admin_level1_name = getattr(platform_config, 'admin_level1_name', None)
            admin_level2_name = getattr(platform_config, 'admin_level2_name', None)
            admin_level3_name = getattr(platform_config, 'admin_level3_name', None)
            decimal_places = getattr(platform_config, 'schema_decimal_places', 2)

            # GADM configuration
            gadm_data_path = getattr(platform_config, 'gadm_data_path', None)
            gadm_country_iso3 = getattr(platform_config, 'gadm_country_iso3', None)
            gadm_admin_name = getattr(platform_config, 'gadm_admin_name', None)

            # Legacy shapefile path
            shapefile_path = getattr(platform_config, 'admin_shapefile_path', None)

        # Derive admin names from region if not explicitly set
        if not admin_level1_name:
            admin_level1_name = sanitize_admin_name(region.country or "Unknown")
        if not admin_level2_name:
            admin_level2_name = sanitize_admin_name(region.name or "Unknown")
        if not gadm_admin_name:
            gadm_admin_name = region.name

        # Create Python_Schemas directory (inside schema/)
        python_schema_dir = self.output_dir / "schema" / "Python_Schemas" / f"Level{schema_level}"
        python_schema_dir.mkdir(parents=True, exist_ok=True)

        # Build filename
        if schema_level == 1:
            filename = f"Schema_{admin_level1_name}.txt"
        elif schema_level == 2:
            filename = f"Schema_{admin_level1_name}_{admin_level2_name}.txt"
        else:  # Level 3
            if not admin_level3_name:
                admin_level3_name = admin_level2_name
            filename = f"Schema_{admin_level1_name}_{admin_level2_name}_{admin_level3_name}.txt"

        python_schema_path = python_schema_dir / filename

        # =====================================================================
        # OPTION 1: Use GADM for accurate boundary-based schema (RECOMMENDED)
        # =====================================================================
        if gadm_data_path and gadm_country_iso3:
            _, python_schema_rows = self._generate_schema_from_gadm(
                gadm_data_path=gadm_data_path,
                country_iso3=gadm_country_iso3,
                schema_level=schema_level,
                admin_name=gadm_admin_name,
                decimal_places=decimal_places,
            )

            if python_schema_rows:
                # Write Python schema file from GADM data
                with open(python_schema_path, 'w') as f:
                    # Build header based on level
                    header_cols = ["CellID", "Latitude", "Longitude", "Elevation", "Area"]
                    if schema_level >= 1:
                        header_cols.append("Level1Name")
                    if schema_level >= 2:
                        header_cols.append("Level2Name")
                    if schema_level >= 3:
                        header_cols.append("Level3Name")
                    f.write("\t".join(header_cols) + "\n")

                    for row_data in python_schema_rows:
                        row = [
                            str(row_data['cellid']),
                            f"{row_data['lat']:.8f}",
                            f"{row_data['lon']:.8f}",
                            f"{row_data['elevation']:.2f}",
                            f"{row_data['area']:.12f}",
                        ]

                        # Add admin names based on level
                        if schema_level >= 1:
                            row.append(admin_level1_name)
                        if schema_level >= 2:
                            row.append(admin_level2_name)
                        if schema_level >= 3:
                            row.append(admin_level3_name or admin_level2_name)

                        f.write("\t".join(row) + "\n")

                output_files.append(python_schema_path)
                logger.info(f"Generated Python schema from GADM: {python_schema_path} ({len(python_schema_rows)} cells)")
                return output_files
            else:
                logger.warning("GADM schema generation returned no cells - falling back to grid-based method")

        # =====================================================================
        # OPTION 2: Use grid cells + shapefile/bounding box (fallback)
        # =====================================================================
        # Calculate intersection areas for each cell
        intersection_areas = self._calculate_intersection_areas(grid, shapefile_path)

        # Write Python schema file
        with open(python_schema_path, 'w') as f:
            # Build header based on level
            header_cols = ["CellID", "Latitude", "Longitude", "Elevation", "Area"]
            if schema_level >= 1:
                header_cols.append("Level1Name")
            if schema_level >= 2:
                header_cols.append("Level2Name")
            if schema_level >= 3:
                header_cols.append("Level3Name")
            f.write("\t".join(header_cols) + "\n")

            # Write cells sorted by CellID descending
            cells_sorted = sorted(grid.cells, key=lambda c: c.cell_id, reverse=True)

            for cell in cells_sorted:
                craft_cellid = self._to_craft_cellid(cell.cell_id)

                # Get intersection area (or full cell area as fallback)
                area_km2 = intersection_areas.get(cell.cell_id, 0.0)

                # Only include cells with non-zero area (actual intersection)
                if area_km2 <= 0:
                    continue

                elevation = -99.0  # Placeholder (CRAFT default)

                row = [
                    str(craft_cellid),
                    f"{cell.lat:.8f}",
                    f"{cell.lon:.8f}",
                    f"{elevation:.2f}",
                    f"{area_km2:.12f}",
                ]

                # Add admin names based on level
                if schema_level >= 1:
                    row.append(admin_level1_name)
                if schema_level >= 2:
                    row.append(admin_level2_name)
                if schema_level >= 3:
                    row.append(admin_level3_name or admin_level2_name)

                f.write("\t".join(row) + "\n")

        output_files.append(python_schema_path)
        logger.info(f"Generated Python schema: {python_schema_path} ({len(grid.cells)} cells)")

        return output_files

    def _calculate_intersection_areas(
        self,
        grid: SpatialGrid,
        shapefile_path: Optional[Path],
    ) -> Dict[int, float]:
        """Calculate intersection area between each cell and admin boundary.

        The Area field in Python schemas represents the portion of each cell
        that falls within the admin boundary polygon, NOT the full cell area.

        - Cells fully inside boundary: Area ≈ 84 km² (full cell)
        - Cells on boundary edge: Area < 84 km² (partial coverage)

        Formula: Area_km² = intersection_area_deg² * 12364 * cos(latitude)
        where 12364 = (111.32 km/deg)² ≈ conversion factor from deg² to km²

        Args:
            grid: SpatialGrid with cells
            shapefile_path: Optional path to admin boundary shapefile

        Returns:
            Dictionary mapping cell_id (0-indexed) to intersection area in km²
        """
        import math

        intersection_areas = {}
        resolution = 5 / 60  # 5 arcmin in degrees
        half_res = resolution / 2

        # Conversion factor: deg² to km² at equator
        # 1 degree ≈ 111.32 km, so 1 deg² ≈ 12392 km²
        DEG2_TO_KM2 = 12364  # Legacy CRAFT uses 12364

        if shapefile_path and Path(shapefile_path).exists():
            try:
                import geopandas as gpd
                from shapely.geometry import box

                logger.info(f"Computing intersection areas from shapefile: {shapefile_path}")

                gdf = gpd.read_file(shapefile_path)
                if gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")

                # Union all polygons to get single admin boundary
                admin_geom = gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union

                for cell in grid.cells:
                    # Create cell bounding box
                    cell_box = box(
                        cell.lon - half_res,
                        cell.lat - half_res,
                        cell.lon + half_res,
                        cell.lat + half_res
                    )

                    # Calculate intersection
                    if cell_box.intersects(admin_geom):
                        intersection = cell_box.intersection(admin_geom)
                        # Area in deg² -> km² with latitude correction
                        intersection_area_deg2 = intersection.area
                        area_km2 = intersection_area_deg2 * DEG2_TO_KM2 * math.cos(math.radians(cell.lat))
                        intersection_areas[cell.cell_id] = area_km2
                    else:
                        intersection_areas[cell.cell_id] = 0.0

                logger.info(f"Computed accurate intersection areas for {len(intersection_areas)} cells")

            except ImportError:
                logger.warning("geopandas not available - using full cell area (inaccurate)")
                self._fill_full_cell_areas(grid, intersection_areas, DEG2_TO_KM2, resolution)

            except Exception as e:
                logger.warning(f"Error computing intersection areas: {e}")
                logger.warning("Falling back to full cell area (inaccurate)")
                self._fill_full_cell_areas(grid, intersection_areas, DEG2_TO_KM2, resolution)
        else:
            if shapefile_path:
                logger.warning(f"Shapefile not found: {shapefile_path}")
            logger.warning("Using full cell area - for accurate Area values, provide admin_shapefile_path")
            self._fill_full_cell_areas(grid, intersection_areas, DEG2_TO_KM2, resolution)

        return intersection_areas

    def _fill_full_cell_areas(
        self,
        grid: SpatialGrid,
        areas_dict: Dict[int, float],
        deg2_to_km2: float,
        resolution: float,
    ) -> None:
        """Fill areas dictionary with full cell areas (fallback when no shapefile).

        Args:
            grid: SpatialGrid with cells
            areas_dict: Dictionary to fill with areas
            deg2_to_km2: Conversion factor from deg² to km²
            resolution: Cell resolution in degrees
        """
        import math

        cell_area_deg2 = resolution * resolution  # Full cell area in deg²

        for cell in grid.cells:
            area_km2 = cell_area_deg2 * deg2_to_km2 * math.cos(math.radians(cell.lat))
            areas_dict[cell.cell_id] = area_km2

    def _calculate_share_percent(
        self,
        grid: SpatialGrid,
        shapefile_path: Optional[Path],
        decimal_places: int = 2
    ) -> Dict[int, float]:
        """Calculate SharePercent for each cell.

        SharePercent = percentage of cell area covered by admin boundary (0-100)

        Args:
            grid: SpatialGrid with cells
            shapefile_path: Optional path to admin boundary shapefile
            decimal_places: Decimal places for rounding

        Returns:
            Dictionary mapping cell_id (0-indexed) to SharePercent
        """
        share_percents = {}

        if shapefile_path and Path(shapefile_path).exists():
            # Try to use geopandas for accurate intersection-based calculation
            try:
                import geopandas as gpd
                from shapely.geometry import box

                logger.info(f"Computing SharePercent from shapefile: {shapefile_path}")

                gdf = gpd.read_file(shapefile_path)
                if gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")

                # Union all polygons
                admin_geom = gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union

                resolution = 5 / 60  # 5 arcmin in degrees
                half_res = resolution / 2

                for cell in grid.cells:
                    # Create cell bounding box
                    cell_box = box(
                        cell.lon - half_res,
                        cell.lat - half_res,
                        cell.lon + half_res,
                        cell.lat + half_res
                    )

                    # Calculate intersection
                    if cell_box.intersects(admin_geom):
                        intersection = cell_box.intersection(admin_geom)
                        share_pct = (intersection.area / cell_box.area) * 100
                        share_percents[cell.cell_id] = round(share_pct, decimal_places)
                    else:
                        share_percents[cell.cell_id] = 0.0

                logger.info(f"Computed accurate SharePercent for {len(share_percents)} cells")

            except ImportError:
                logger.warning("geopandas not available - using simplified SharePercent (100% for all cells)")
                for cell in grid.cells:
                    share_percents[cell.cell_id] = 100.0

            except Exception as e:
                logger.warning(f"Error computing SharePercent from shapefile: {e}")
                logger.warning("Falling back to simplified SharePercent (100% for all cells)")
                for cell in grid.cells:
                    share_percents[cell.cell_id] = 100.0
        else:
            # Simplified mode: all cells get 100% share
            if shapefile_path:
                logger.warning(f"Shapefile not found: {shapefile_path}")
            logger.warning("Using simplified SharePercent (100% for all cells in bounding box)")
            logger.warning("For accurate SharePercent values, configure gadm_data_path in platform_config.craft")
            for cell in grid.cells:
                share_percents[cell.cell_id] = 100.0

        return share_percents

    def _generate_weather_input_csv(self, grid: SpatialGrid) -> Path:
        """Generate CRAFT weather input CSV for NASA Power download.

        This CSV file is used by the CRAFT GUI to download weather data from
        NASA Power. It contains the cell coordinates needed for the API queries.

        Output format (comma-separated):
        CellID,Latitude,Longitude
        4054256,11.79166259,-5.37500699

        IMPORTANT: After generating this file, use CRAFT GUI:
        1. CRAFT -> Data -> Upload Data -> Weather Data
        2. Select: Download from NASA
        3. Start Date: Use your temporal.start_year (1984+ recommended for SRAD data)
        4. End Date: Use your temporal.end_year (or latest available)
        5. Input file: Browse to this input.csv
        6. Download duration: ~45-75 minutes for 150 cells

        Note: NASA POWER SRAD (solar radiation) data starts 1984-01-01.
        For simulations requiring SRAD, use start year >= 1984.

        Args:
            grid: SpatialGrid with cells

        Returns:
            Path to generated input.csv file
        """
        weather_dir = self.output_dir / "weather"
        weather_dir.mkdir(parents=True, exist_ok=True)
        input_csv_path = weather_dir / "input.csv"

        # Get GADM-filtered cells if available (for consistency with schema)
        filtered_cells = self._get_filtered_cells(grid)

        with open(input_csv_path, 'w') as f:
            # Header (comma-separated, matching CRAFT expected format)
            f.write("CellID,Latitude,Longitude\n")

            # Sort cells by CellID descending (CRAFT convention)
            cells_sorted = sorted(filtered_cells, key=lambda c: self._to_craft_cellid(c.cell_id), reverse=True)

            for cell in cells_sorted:
                # Use 1-indexed CRAFT CellID
                craft_cellid = self._to_craft_cellid(cell.cell_id)
                # 8 decimal places for coordinates (matching legacy format)
                f.write(f"{craft_cellid},{cell.lat:.8f},{cell.lon:.8f}\n")

        logger.info(f"Generated CRAFT weather input CSV: {input_csv_path} ({len(filtered_cells)} cells)")
        logger.info("  Next: Use CRAFT GUI -> Data -> Weather Data -> Download from NASA")
        # Get start year from config if available, otherwise use default
        start_year = getattr(self.config.temporal, 'start_year', DEFAULT_CLIMATE_START_YEAR) if self.config.temporal else DEFAULT_CLIMATE_START_YEAR
        if start_year < DEFAULT_CLIMATE_START_YEAR:
            logger.warning(f"  NOTE: Start year {start_year} is before {DEFAULT_CLIMATE_START_YEAR}. NASA POWER SRAD data starts {DEFAULT_CLIMATE_START_YEAR}-01-01.")
        else:
            logger.info(f"  Recommended start date: {start_year}0101 (NASA POWER SRAD available from {DEFAULT_CLIMATE_START_YEAR})")

        return input_csv_path

    def _generate_weather_files(
        self,
        climate_data: Dict[int, ClimateTimeSeries],
    ) -> List[Path]:
        """Generate CRAFT tab-separated weather files.

        NOTE: This method generates WTH-format files if climate data is provided.
        For most use cases, use _generate_weather_input_csv() instead to create
        input.csv for NASA Power download via CRAFT GUI.

        CRAFT weather format:
        YRDOY   SRAD    TMAX    TMIN    RAIN

        Args:
            climate_data: Dictionary of cell_id to ClimateTimeSeries

        Returns:
            List of generated weather file paths
        """
        output_files = []
        weather_dir = self.output_dir / "weather"

        for cell_id, ts in climate_data.items():
            weather_file = weather_dir / f"cell_{cell_id}.txt"

            with open(weather_file, 'w') as f:
                # Header
                f.write("YRDOY\tSRAD\tTMAX\tTMIN\tRAIN\n")

                # Data rows
                for record in ts.records:
                    yrdoy = date_to_yrdoy(record.date)
                    srad = record.srad if record.srad is not None else -99.0
                    tmax = record.tmax if record.tmax is not None else -99.0
                    tmin = record.tmin if record.tmin is not None else -99.0
                    rain = record.precip if record.precip is not None else -99.0

                    f.write(f"{yrdoy}\t{srad:.1f}\t{tmax:.1f}\t{tmin:.1f}\t{rain:.1f}\n")

            output_files.append(weather_file)
            logger.debug(f"Generated weather file: {weather_file}")

        logger.info(f"Generated {len(output_files)} CRAFT weather files")
        return output_files

    def _generate_soil_file(
        self,
        soil_profiles: Dict[int, SoilProfile],
        region: Region,
    ) -> Path:
        """Generate DSSAT .SOL format soil file.

        Filename is dynamic based on country code: {CC}.SOL
        (e.g., ML.SOL for Mali, KE.SOL for Kenya, NG.SOL for Nigeria)

        .SOL format:
        *{CC}{profile_id}
        @SITE        COUNTRY          LAT     LONG ...
        ...
        @  SLB  SLLL  SDUL  SSAT ...
        layer data...

        Args:
            soil_profiles: Dictionary of cell_id to SoilProfile
            region: Region for country code derivation

        Returns:
            Path to generated {CC}.SOL file
        """
        # Get country code dynamically from region config
        country_code = self._get_country_code(region)
        logger.info(f"Using country code: {country_code} (from {region.country or 'unknown'})")

        # Dynamic soil filename based on country code (e.g., ML.SOL, KE.SOL, NG.SOL)
        soil_filename = f"{country_code}.SOL"
        soil_path = self.output_dir / "soil" / soil_filename

        with open(soil_path, 'w') as f:
            # File header
            f.write(f"*SOILS: {region.name} - Generated by prismpy\n\n")

            for cell_id, profile in soil_profiles.items():
                # Profile header (10-character ID starting with *)
                profile_id = f"{country_code}{cell_id:08d}"[:10]
                f.write(f"*{profile_id}\n")

                # Site line
                f.write(f"@SITE        COUNTRY          LAT     LONG ")
                f.write("SCS FAMILY\n")

                site_name = sanitize_admin_name(region.name)[:12]
                f.write(f" {site_name:<12} {region.country:<15} ")
                f.write(f"{profile.lat:7.3f} {profile.lon:8.3f} ")
                texture = profile.surface_texture or "Unknown"
                f.write(f" {texture}\n")

                # Layer header
                f.write("@  SLB  SLLL  SDUL  SSAT  SSKS  SBDM  SLOC ")
                f.write(" SLCL  SLSI  SLCF  SLNI  SLHW  SLHB\n")

                # Layer data
                for layer in profile.layers:
                    # Ensure hydraulic properties are computed
                    if layer.wilting_point is None:
                        layer.estimate_hydraulic_properties()
                        logger.debug(f"  Estimated hydraulic properties for layer at {layer.depth_bottom}m depth")

                    slb = int(layer.depth_bottom * 100)  # cm
                    slll = layer.wilting_point or 0.10
                    sdul = layer.field_capacity or 0.25
                    ssat = layer.saturated_wc or 0.45
                    ssks = 10.0  # Saturated hydraulic conductivity (estimated)
                    sbdm = layer.bulk_density or 1.4
                    sloc = layer.organic_carbon or 0.5
                    slcl = layer.clay
                    slsi = layer.silt or (100 - layer.sand - layer.clay)
                    slcf = 0.0  # Coarse fragment (estimated)
                    slni = 0.0  # Nitrogen (estimated)
                    slhw = layer.ph or 6.5
                    slhb = layer.ph or 6.5

                    # Log when using estimated defaults instead of measured values
                    if layer.wilting_point is None or layer.field_capacity is None:
                        logger.debug(f"  Cell {cell_id}, layer {slb}cm: using estimated WP={slll:.3f}, FC={sdul:.3f}, SAT={ssat:.3f}")
                    if layer.ph is None:
                        logger.debug(f"  Cell {cell_id}, layer {slb}cm: using estimated pH={slhw:.1f}")

                    f.write(f"  {slb:3d} {slll:5.3f} {sdul:5.3f} {ssat:5.3f} ")
                    f.write(f"{ssks:5.1f} {sbdm:5.2f} {sloc:5.2f} ")
                    f.write(f"{slcl:5.1f} {slsi:5.1f} {slcf:5.1f} ")
                    f.write(f"{slni:5.2f} {slhw:5.1f} {slhb:5.1f}\n")

                f.write("\n")  # Blank line between profiles

        logger.info(f"Generated CRAFT soil file: {soil_path} ({len(soil_profiles)} profiles)")
        return soil_path

    def _generate_soil_package(
        self,
        grid: SpatialGrid,
        region: Region,
        existing_soil_data: Optional[Dict[int, SoilProfile]] = None,
    ) -> Tuple[Path, SoilMapping]:
        """Generate CRAFT soil package: .SOL file + cell-to-profile mapping.

        This implements Option B: Per-Soil-Type Profiles
        - Queries HWSD to get SMU (Soil Mapping Unit) ID for each cell
        - Groups cells by SMU ID to identify unique soil types
        - Generates one profile per unique SMU
        - Returns mapping for soil_mask.txt generation

        The result is a self-contained soil package where:
        - .SOL contains unique soil profiles (fewer than cells)
        - soil_mask.txt maps each CellID to its profile

        Args:
            grid: SpatialGrid with cells
            region: Region for metadata
            existing_soil_data: Optional pre-loaded soil profiles

        Returns:
            Tuple of (path to .SOL file, dict mapping cell_id -> profile_name)
        """
        platform_config = self.get_platform_config()
        country_code = self._get_country_code(region)

        # Get HWSD configuration from platform config
        hwsd_bil_path = None
        hwsd_mdb_path = None
        if platform_config:
            hwsd_bil_path = getattr(platform_config, 'hwsd_bil_path', None)
            hwsd_mdb_path = getattr(platform_config, 'hwsd_mdb_path', None)

        # Get filtered cells (GADM boundary if available)
        filtered_cells = self._get_filtered_cells(grid)

        # Prepare cell coordinates for HWSD query
        cell_coords = [(cell.lat, cell.lon) for cell in filtered_cells]
        cell_ids = [cell.cell_id for cell in filtered_cells]

        # =========================================================================
        # STEP 1: Query HWSD for soil data (if paths configured)
        # =========================================================================
        smu_to_profile: Dict[int, SoilProfile] = {}  # SMU ID -> profile
        cell_to_smu: Dict[int, int] = {}  # cell_id -> SMU ID

        if hwsd_bil_path and hwsd_mdb_path:
            logger.info(f"Querying HWSD for {len(filtered_cells)} cells...")
            try:
                hwsd_source = HWSDSource(
                    config=HWSDConfig(
                        bil_path=Path(hwsd_bil_path),
                        mdb_path=Path(hwsd_mdb_path),
                        use_defaults=True,
                    )
                )

                result = hwsd_source.retrieve(
                    region=region,
                    cell_coords=cell_coords,
                )

                if result.success and result.data:
                    hwsd_profiles = result.data.profiles

                    # Build SMU-based mapping
                    for i, cell_id in enumerate(cell_ids):
                        if i in hwsd_profiles:
                            profile = hwsd_profiles[i]
                            # Get SMU ID from profile metadata
                            smu_id = profile.metadata.get('hwsd_smu_id', i)
                            if smu_id is None:
                                smu_id = i  # Fallback to cell index

                            cell_to_smu[cell_id] = smu_id

                            # Store unique profiles by SMU
                            if smu_id not in smu_to_profile:
                                smu_to_profile[smu_id] = profile

                    logger.info(f"HWSD: {len(smu_to_profile)} unique soil types for {len(cell_to_smu)} cells")
                else:
                    logger.warning(f"HWSD query failed: {result.errors}")

            except Exception as e:
                logger.warning(f"HWSD query error: {e}")

        # =========================================================================
        # STEP 2: Use existing soil data if available (from pipeline)
        # =========================================================================
        if not smu_to_profile and existing_soil_data:
            logger.info("Using existing soil data from pipeline")
            for cell_id, profile in existing_soil_data.items():
                smu_id = hash(f"{profile.lat:.4f}_{profile.lon:.4f}") % 100000
                smu_to_profile[smu_id] = profile
                # Map all cells to this profile if only one exists
                if len(existing_soil_data) == 1:
                    for cid in cell_ids:
                        cell_to_smu[cid] = smu_id
                else:
                    cell_to_smu[cell_id] = smu_id

        # =========================================================================
        # STEP 3: Create default profile if no soil data available
        # =========================================================================
        if not smu_to_profile:
            logger.warning("No HWSD data available - creating default soil profile")
            logger.warning("For accurate soil data, configure hwsd_bil_path and hwsd_mdb_path")

            # Create a single default profile
            center_lat = (region.bounds.miny + region.bounds.maxy) / 2
            center_lon = (region.bounds.minx + region.bounds.maxx) / 2

            default_layers = [
                SoilLayer(
                    depth_top=0.0,
                    depth_bottom=0.2,
                    sand=60.0,
                    clay=18.0,
                    silt=22.0,
                    organic_carbon=0.5,
                    bulk_density=1.4,
                    ph=6.5,
                    field_capacity=0.25,
                    wilting_point=0.10,
                ),
                SoilLayer(
                    depth_top=0.2,
                    depth_bottom=1.0,
                    sand=55.0,
                    clay=22.0,
                    silt=23.0,
                    organic_carbon=0.3,
                    bulk_density=1.5,
                    ph=6.3,
                    field_capacity=0.28,
                    wilting_point=0.12,
                ),
            ]

            default_smu_id = 0
            smu_to_profile[default_smu_id] = SoilProfile(
                profile_id=f"{country_code}_DEFAULT",
                lat=center_lat,
                lon=center_lon,
                source="default",
                layers=default_layers,
            )

            # All cells map to default profile
            for cell_id in cell_ids:
                cell_to_smu[cell_id] = default_smu_id

        # =========================================================================
        # STEP 4: Generate .SOL file with unique profiles
        # =========================================================================
        soil_filename = f"{country_code}.SOL"
        soil_path = self.output_dir / "soil" / soil_filename

        # Create profile name mapping: SMU ID -> profile name in .SOL
        smu_to_profile_name: Dict[int, str] = {}

        with open(soil_path, 'w') as f:
            f.write(f"*SOILS: {region.name} - Generated by prismpy (HWSD-based)\n\n")

            for smu_id, profile in sorted(smu_to_profile.items()):
                # Profile ID: {CC}{SMU_ID:08d} (10 chars max)
                profile_name = f"{country_code}{smu_id:08d}"[:10]
                smu_to_profile_name[smu_id] = profile_name

                f.write(f"*{profile_name}\n")

                # Site line
                f.write("@SITE        COUNTRY          LAT     LONG SCS FAMILY\n")
                site_name = sanitize_admin_name(region.name)[:12]
                texture = profile.surface_texture or "Unknown"
                f.write(f" {site_name:<12} {region.country or 'Unknown':<15} ")
                f.write(f"{profile.lat:7.3f} {profile.lon:8.3f}  {texture}\n")

                # Layer header
                f.write("@  SLB  SLLL  SDUL  SSAT  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB\n")

                # Layer data
                for layer in profile.layers:
                    if layer.wilting_point is None:
                        layer.estimate_hydraulic_properties()

                    slb = int(layer.depth_bottom * 100)
                    slll = layer.wilting_point or 0.10
                    sdul = layer.field_capacity or 0.25
                    ssat = layer.saturated_wc or 0.45
                    ssks = 10.0
                    sbdm = layer.bulk_density or 1.4
                    sloc = layer.organic_carbon or 0.5
                    slcl = layer.clay or 18.0
                    slsi = layer.silt or (100 - (layer.sand or 60) - slcl)
                    slcf = 0.0
                    slni = 0.0
                    slhw = layer.ph or 6.5
                    slhb = layer.ph or 6.5

                    f.write(f"  {slb:3d} {slll:5.3f} {sdul:5.3f} {ssat:5.3f} ")
                    f.write(f"{ssks:5.1f} {sbdm:5.2f} {sloc:5.2f} ")
                    f.write(f"{slcl:5.1f} {slsi:5.1f} {slcf:5.1f} ")
                    f.write(f"{slni:5.2f} {slhw:5.1f} {slhb:5.1f}\n")

                f.write("\n")

        logger.info(f"Generated CRAFT soil file: {soil_path} ({len(smu_to_profile)} unique profiles)")

        # =========================================================================
        # STEP 5: Build cell_id -> profile_name mapping for soil_mask.txt
        # =========================================================================
        cell_to_profile_name: SoilMapping = {}
        for cell_id in cell_ids:
            smu_id = cell_to_smu.get(cell_id, 0)
            profile_name = smu_to_profile_name.get(smu_id, f"{country_code}00000000")
            cell_to_profile_name[cell_id] = profile_name

        return soil_path, cell_to_profile_name

    def _generate_crop_mask(
        self,
        grid: SpatialGrid,
        crop_calendar: Optional[Dict[int, CropCalendar]] = None,
    ) -> Path:
        """Generate CRAFT crop mask file.

        The crop mask indicates what fraction of each cell has the target crop.

        Output format (tab-separated, sorted by CellId descending):
        CellId    Percent
        4054256   1.0
        4054255   0.717050

        Where Percent is 0-1 range (1.0 = 100% of cell has crop).

        Modes:
        - uniform: All cells get same percentage (default, uses crop_mask_percent)
        - raster: Extract from SPAM GeoTIFF (uses spam_raster_path)
          Formula: Percent = SPAM_hectares / cell_area_ha

        Args:
            grid: SpatialGrid with cells
            crop_calendar: Optional calendar data per cell

        Returns:
            Path to generated mask.txt
        """
        mask_path = self.output_dir / "crop_mask" / "mask.txt"
        platform_config = self.get_platform_config()

        # Get config options
        default_percent = 1.0
        spam_path = None
        cap_at_100 = True
        na_to_zero = True

        if platform_config:
            default_percent = getattr(platform_config, 'crop_mask_percent', 1.0)
            spam_path = getattr(platform_config, 'spam_raster_path', None)
            cap_at_100 = getattr(platform_config, 'spam_cap_at_100_percent', True)
            na_to_zero = getattr(platform_config, 'spam_na_to_zero', True)

        # Get GADM-filtered cells if available (for consistency with schema)
        filtered_cells = self._get_filtered_cells(grid)

        # Determine mode based on config
        use_raster = spam_path is not None and Path(spam_path).exists()

        if use_raster:
            # RASTER MODE: Extract from SPAM GeoTIFF
            logger.info(f"Crop mask mode: SPAM raster ({spam_path})")
            cell_percents = self._extract_crop_mask_from_spam(
                filtered_cells, spam_path, cap_at_100, na_to_zero
            )
        else:
            # UNIFORM MODE: All cells get same percentage
            if spam_path and not Path(spam_path).exists():
                logger.warning(f"SPAM raster not found: {spam_path}, using uniform mode")
            logger.info(f"Crop mask mode: uniform ({default_percent:.0%} coverage)")
            cell_percents = {cell.cell_id: default_percent for cell in filtered_cells}

        with open(mask_path, 'w') as f:
            # Header (lowercase 'd' in CellId per CRAFT legacy format)
            f.write("CellId\tPercent\n")

            # Sort cells by CellId descending (CRAFT convention)
            cells_sorted = sorted(filtered_cells, key=lambda c: self._to_craft_cellid(c.cell_id), reverse=True)

            for cell in cells_sorted:
                # Use 1-indexed CRAFT CellID
                craft_cellid = self._to_craft_cellid(cell.cell_id)
                percent = cell_percents.get(cell.cell_id, default_percent)

                # Write percent as float (0-1 range)
                f.write(f"{craft_cellid}\t{percent}\n")

        # Log statistics
        percents = list(cell_percents.values())
        if use_raster:
            min_p = min(percents)
            max_p = max(percents)
            mean_p = sum(percents) / len(percents)
            with_crop = sum(1 for p in percents if p > 0)
            logger.info(f"Generated CRAFT crop mask: {mask_path} ({len(filtered_cells)} cells)")
            logger.info(f"  Percent range: {min_p:.4f} - {max_p:.4f}, mean: {mean_p:.4f}")
            logger.info(f"  Cells with crop: {with_crop}/{len(filtered_cells)} ({100*with_crop/len(filtered_cells):.1f}%)")
        else:
            logger.info(f"Generated CRAFT crop mask: {mask_path} ({len(filtered_cells)} cells, {default_percent:.0%} coverage)")

        return mask_path

    def _extract_crop_mask_from_spam(
        self,
        cells: List,
        spam_path: Path,
        cap_at_100: bool = True,
        na_to_zero: bool = True,
    ) -> Dict[int, float]:
        """Extract crop percentages from SPAM raster.

        Formula: Percent = SPAM_hectares / cell_area_ha

        IMPORTANT: Uses the Area from Python schema (actual intersection area
        with admin boundary), NOT the full cell area. This is critical for
        edge cells where the intersection area can be much smaller.

        Args:
            cells: List of GridCell objects
            spam_path: Path to SPAM GeoTIFF
            cap_at_100: Cap values > 1.0
            na_to_zero: Replace NA with 0

        Returns:
            Dict mapping cell_id to percent (0-1)
        """
        try:
            import rasterio
        except ImportError:
            logger.error("rasterio not installed. Run: pip install rasterio")
            logger.warning("Falling back to uniform mode")
            return {}

        import pandas as pd

        cell_percents = {}

        # Load Python schema to get actual intersection areas
        # This is critical: edge cells have smaller areas than full cells
        schema_areas = self._load_schema_areas()
        if schema_areas:
            logger.info(f"  Loaded {len(schema_areas)} cell areas from Python schema")
        else:
            logger.warning("  Could not load schema areas - using estimated full cell areas")

        with rasterio.open(spam_path) as src:
            logger.info(f"  SPAM raster: {src.height}x{src.width}, res={src.res[0]:.4f} deg")

            # Extract values at cell centroids
            coords = [(cell.lon, cell.lat) for cell in cells]
            values = list(src.sample(coords))

            na_count = 0
            over_100_count = 0
            estimated_area_count = 0

            for cell, value in zip(cells, values):
                spam_ha = value[0]  # Harvested area in hectares
                craft_cellid = self._to_craft_cellid(cell.cell_id)

                # Handle NA
                if np.isnan(spam_ha):
                    na_count += 1
                    if na_to_zero:
                        spam_ha = 0.0
                    else:
                        spam_ha = 0.0  # Still need a value

                # Get cell area from schema (actual intersection area)
                # This is CRITICAL for matching legacy behavior
                cell_area_km2 = schema_areas.get(craft_cellid) if schema_areas else None

                if cell_area_km2 is None:
                    # Fallback: estimate full cell area (less accurate for edge cells)
                    estimated_area_count += 1
                    import math
                    dx = 5 / 60 * 111.32 * math.cos(math.radians(cell.lat))  # km
                    dy = 5 / 60 * 110.57  # km (roughly constant)
                    cell_area_km2 = dx * dy

                cell_area_ha = cell_area_km2 * 100  # km² to ha

                # Calculate percentage
                percent = spam_ha / cell_area_ha if cell_area_ha > 0 else 0

                # Cap at 100%
                if percent > 1.0:
                    over_100_count += 1
                    if cap_at_100:
                        percent = 1.0

                cell_percents[cell.cell_id] = percent

            if na_count > 0:
                logger.info(f"  NA cells (set to 0): {na_count}")
            if over_100_count > 0 and cap_at_100:
                logger.info(f"  Cells capped at 100%: {over_100_count}")
            if estimated_area_count > 0:
                logger.warning(f"  Cells using estimated area (not in schema): {estimated_area_count}")

        return cell_percents

    def _load_schema_areas(self) -> Optional[Dict[int, float]]:
        """Load cell areas from Python schema.

        The Python schema contains the actual intersection area of each cell
        with the admin boundary. This is crucial for accurate SPAM percentage
        calculations, especially for edge cells.

        Returns:
            Dict mapping CRAFT CellID to Area in km², or None if not available
        """
        import pandas as pd

        # Try to find the Python schema file (inside schema/)
        python_schema_dir = self.output_dir / "schema" / "Python_Schemas"
        if not python_schema_dir.exists():
            return None

        # Look for schema file (Level2 is most common)
        schema_files = list(python_schema_dir.glob("**/Schema_*.txt"))
        if not schema_files:
            return None

        # Use the most recently modified one
        schema_file = max(schema_files, key=lambda f: f.stat().st_mtime)
        logger.debug(f"Loading schema areas from: {schema_file}")

        try:
            df = pd.read_csv(schema_file, sep='\t')

            # Schema has columns: CellID, Latitude, Longitude, Elevation, Area, Level1Name, Level2Name, ...
            if 'CellID' not in df.columns or 'Area' not in df.columns:
                logger.warning(f"Schema missing CellID or Area columns: {df.columns.tolist()}")
                return None

            # Build mapping: CellID -> Area (km²)
            return dict(zip(df['CellID'], df['Area']))

        except Exception as e:
            logger.warning(f"Failed to load schema areas: {e}")
            return None

    def _generate_management(
        self,
        crop_params: Optional[CropParameters] = None,
        crop_calendar: Optional[Dict[int, CropCalendar]] = None,
    ) -> Path:
        """Generate CRAFT management file.

        Management file contains planting dates, fertilizer, and other practices.

        Args:
            crop_params: CropParameters for crop-specific info
            crop_calendar: Crop calendar per cell

        Returns:
            Path to generated management.txt
        """
        management_path = self.output_dir / "management" / "management.txt"

        with open(management_path, 'w') as f:
            # Header
            f.write("CellID\tPlantingDOY\tHarvestDOY\tFertilizerN\tIrrigation\n")

            if crop_calendar:
                for cell_id, calendar in crop_calendar.items():
                    planting_doy = calendar.planting_doy
                    harvest_doy = calendar.harvest_doy
                    fert_n = calendar.fertilizer_n if hasattr(calendar, 'fertilizer_n') else 0
                    irrigation = 0  # Default rainfed

                    f.write(f"{cell_id}\t{planting_doy}\t{harvest_doy}\t")
                    f.write(f"{fert_n}\t{irrigation}\n")
            else:
                # Require calendar from config - no hardcoded region-specific defaults
                if self.config.crop.calendar:
                    planting_doy = self.config.crop.calendar.planting_doy
                    harvest_doy = self.config.crop.calendar.harvest_doy
                    f.write(f"0\t{planting_doy}\t{harvest_doy}\t0\t0\n")
                else:
                    logger.error("No crop calendar specified - cannot generate management file with planting/harvest dates")
                    raise ValueError("crop.calendar with planting_doy and harvest_doy is required for CRAFT management")

        logger.info(f"Generated CRAFT management file: {management_path}")
        return management_path

    def _latlon_to_rowcol(self, lat: float, lon: float, resolution: float = 5/60) -> Tuple[int, int]:
        """Convert lat/lon to row/col in 5-arcmin global grid.

        CRAFT uses a global 5-arcmin grid:
        - Row 0 is at lat 90
        - Col 0 is at lon -180

        Args:
            lat: Latitude
            lon: Longitude
            resolution: Grid resolution in degrees (default: 5 arcmin)

        Returns:
            Tuple of (row, col)
        """
        row = int((90 - lat) / resolution)
        col = int((lon + 180) / resolution)

        # Clamp to valid range
        row = max(0, min(row, self.GLOBAL_ROWS - 1))
        col = max(0, min(col, self.GLOBAL_COLS - 1))

        return row, col

    def _doy_to_mmdd(self, doy: int) -> str:
        """Convert day-of-year to MMDD format for CRAFT/DSSAT.

        CRAFT requires planting dates in MMDD format (e.g., "0604" for June 4),
        NOT Julian day-of-year format.

        Args:
            doy: Day of year (1-366)

        Returns:
            MMDD string (e.g., "0604" for June 4)
        """
        # Days in each month (non-leap year)
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        cumulative = [sum(days_in_month[:i+1]) for i in range(12)]

        # Clamp doy to valid range
        doy = max(1, min(doy, 365))

        # Find month
        month = 1
        for i, cum in enumerate(cumulative):
            if doy <= cum:
                month = i + 1
                break

        # Calculate day within month
        if month == 1:
            day = doy
        else:
            day = doy - cumulative[month - 2]

        return f"{month:02d}{day:02d}"

    def _get_country_code(self, region: Region) -> str:
        """Get 2-letter country code from region data.

        Uses ISO3 code first (taking first 2 chars), then country name.
        Falls back to 'XX' placeholder if no country is specified.

        Args:
            region: Region with country info

        Returns:
            2-letter country code string
        """
        if hasattr(region, 'country_iso3') and region.country_iso3:
            return region.country_iso3[:2].upper()
        elif region.country:
            return region.country[:2].upper()
        else:
            logger.warning("No country specified - using 'XX' as placeholder country code")
            return "XX"

    def _generate_soil_mask(
        self,
        grid: SpatialGrid,
        region: Region,
        cell_to_profile: Optional[SoilMapping] = None,
    ) -> Path:
        """Generate CRAFT soil mask file.

        Maps each grid cell to a DSSAT soil profile from the .SOL file.

        When cell_to_profile mapping is provided (from HWSD-based generation),
        uses actual profile names. Otherwise falls back to formula-based names.

        Output format (tab-separated, sorted by CellID descending):
        CellID    SoilProfile    SharePCT
        4054256   ML00012345     1

        Args:
            grid: SpatialGrid with cells
            region: Region for country code
            cell_to_profile: Optional mapping from cell_id to profile name

        Returns:
            Path to generated soil mask file
        """
        soil_mask_path = self.output_dir / "soil" / "soil_mask.txt"

        # Get country code dynamically from region config
        country_code = self._get_country_code(region)

        # Get GADM-filtered cells if available (for consistency with schema)
        filtered_cells = self._get_filtered_cells(grid)

        with open(soil_mask_path, 'w') as f:
            # Header
            f.write("CellID\tSoilProfile\tSharePCT\n")

            # Sort cells by CellID descending (CRAFT convention)
            cells_sorted = sorted(filtered_cells, key=lambda c: self._to_craft_cellid(c.cell_id), reverse=True)

            for cell in cells_sorted:
                # Use 1-indexed CRAFT CellID
                craft_cellid = self._to_craft_cellid(cell.cell_id)

                # Get profile name from mapping or use fallback formula
                if cell_to_profile and cell.cell_id in cell_to_profile:
                    soil_profile = cell_to_profile[cell.cell_id]
                else:
                    # Fallback: formula-based (for backward compatibility)
                    soil_profile = f"{country_code}0{craft_cellid - 1}"

                share_pct = 1  # 100% coverage

                f.write(f"{craft_cellid}\t{soil_profile}\t{share_pct}\n")

        logger.info(f"Generated CRAFT soil mask: {soil_mask_path} ({len(filtered_cells)} cells)")
        return soil_mask_path

    def _generate_cultivar_data(self, grid: SpatialGrid) -> Path:
        """Generate CRAFT cultivar data file.

        Supports spatial variability through management zones. If zones are
        configured with cultivar overrides, each cell gets zone-specific
        cultivar. Otherwise, all cells use the same default (uniform mode).

        Output format (tab-separated, sorted by CellID descending):
        CellID    CultivarID    CultivarPercentage
        4054256   GH0010        1

        Args:
            grid: SpatialGrid with cells

        Returns:
            Path to generated cultivar data file
        """
        cultivar_path = self.output_dir / "management" / "cultivar_data.txt"
        platform_config = self.get_platform_config()

        # Get management config and zones (framework-agnostic)
        management = getattr(self.config, 'management', None)
        zones = getattr(self.config, 'management_zones', [])

        # Determine default cultivar: check management first, then platform_config
        default_cultivar = None
        if management:
            default_cultivar = getattr(management, 'default_cultivar', None)
        if not default_cultivar and platform_config:
            default_cultivar = getattr(platform_config, 'default_cultivar', None)

        if not default_cultivar:
            logger.error("No cultivar specified. Set management.default_cultivar or platform_config.craft.default_cultivar")
            raise ValueError("Cultivar is required - DSSAT cultivar codes are crop/region-specific")

        logger.info(f"Default cultivar: {default_cultivar}")

        # Check if any zones have cultivar overrides
        zones_with_cultivar = [z for z in zones if z.overrides.cultivar is not None]
        if zones_with_cultivar:
            logger.info(f"Cultivar zones configured: {[z.zone_name for z in zones_with_cultivar]}")

        # Get GADM-filtered cells if available (for consistency with schema)
        filtered_cells = self._get_filtered_cells(grid)

        # Build a synthetic management config for zone utilities if needed
        if management is None:
            from prismpy.config.schema import ManagementConfig
            # Check platform config for planting density (convert plants/m² to plants/ha)
            density = 55000  # fallback
            if platform_config and getattr(platform_config, 'plant_population', None):
                density = int(platform_config.plant_population * 10000)
            management = ManagementConfig(
                planting_density=density,
                default_cultivar=default_cultivar,
            )

        # Track cultivar assignments for logging
        cultivar_counts: Dict[str, int] = {}

        with open(cultivar_path, 'w') as f:
            # Header
            f.write("CellID\tCultivarID\tCultivarPercentage\n")

            # Sort cells by CellID descending (CRAFT convention)
            cells_sorted = sorted(filtered_cells, key=lambda c: self._to_craft_cellid(c.cell_id), reverse=True)

            for cell in cells_sorted:
                # Use 1-indexed CRAFT CellID
                craft_cellid = self._to_craft_cellid(cell.cell_id)

                # Get zone-specific cultivar (or default if no zones)
                params = get_management_for_cell(
                    lat=cell.lat,
                    lon=cell.lon,
                    cell_id=craft_cellid,
                    management=management,
                    zones=zones,
                    schema_df=None,
                )

                # Use zone-specific cultivar or fall back to default
                cultivar = params.get('cultivar') or default_cultivar

                # Track cultivar assignment
                cultivar_counts[cultivar] = cultivar_counts.get(cultivar, 0) + 1

                f.write(f"{craft_cellid}\t{cultivar}\t1\n")

        # Log cultivar distribution
        if len(cultivar_counts) > 1:
            logger.info(f"Cultivar distribution: {cultivar_counts}")
        else:
            logger.info(f"Using cultivar: {default_cultivar} (uniform)")

        logger.info(f"Generated CRAFT cultivar data: {cultivar_path} ({len(filtered_cells)} cells)")
        return cultivar_path

    def _generate_planting_data(
        self,
        grid: SpatialGrid,
        crop_calendar: Optional[Dict[int, CropCalendar]] = None,
    ) -> Path:
        """Generate CRAFT planting data file with full DSSAT format.

        CRITICAL: PDATE uses MMDD format (e.g., "0604" = June 4), NOT Julian DOY!

        Supports spatial variability through management zones. If zones are
        configured with planting_date_mmdd, each cell gets zone-specific dates.
        Otherwise, all cells use the same default (uniform mode - backward compatible).

        Output format (tab-separated, 15 columns):
        CellID  PDATE  EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  Percent_Share

        Parameters:
        - PDATE: Planting date (MMDD format)
        - EDATE: Emergence date (-99 = not used)
        - PPOP: Plant population (plants/m²)
        - PPOE: Emergence percentage (-99 = not used)
        - PLME: Planting method (S=seed, T=transplant)
        - PLDS: Plant distribution (R=rows, B=broadcast)
        - PLRS: Row spacing (cm)
        - PLRD: Row direction (degrees, 0=N-S)
        - PLDP: Planting depth (cm)
        - PLWT: Plant weight (-99 = not used)
        - PAGE: Plant age (-99 = not used)
        - PENV: Plant environment (-99 = not used)
        - PLPH: Plant height (-99 = not used)
        - Percent_Share: Always 1 (100%)

        Args:
            grid: SpatialGrid with cells
            crop_calendar: Optional crop calendar per cell

        Returns:
            Path to generated planting data file
        """
        planting_path = self.output_dir / "management" / "planting_data.txt"
        platform_config = self.get_platform_config()

        # Get agronomic parameters from config (no hardcoded values)
        ppop = 5.5
        plme = "S"
        plds = "R"
        plrs = 75
        plrd = 0
        pldp = 5

        if platform_config:
            ppop = getattr(platform_config, 'plant_population', 5.5)
            plme = getattr(platform_config, 'planting_method', 'S')
            plds = getattr(platform_config, 'plant_distribution', 'R')
            plrs = int(getattr(platform_config, 'row_spacing_cm', 75))  # Integer for CRAFT format
            plrd = int(getattr(platform_config, 'planting_row_direction', 0))  # Integer for CRAFT format
            pldp = int(getattr(platform_config, 'planting_depth_cm', 5))  # Integer for CRAFT format

        # Log planting settings for transparency
        logger.info(f"Planting settings: population={ppop} plants/m², spacing={plrs}cm, depth={pldp}cm")

        # Get default planting date - REQUIRE from config, no hardcoded fallback
        default_pdate = None

        # Priority 1: Direct MMDD override in platform config
        if platform_config and getattr(platform_config, 'planting_date_mmdd', None):
            default_pdate = platform_config.planting_date_mmdd
        # Priority 2: Convert from crop calendar DOY
        elif self.config.crop.calendar and self.config.crop.calendar.planting_doy:
            default_pdate = self._doy_to_mmdd(self.config.crop.calendar.planting_doy)

        # Fail if no planting date specified (don't use hardcoded default)
        if not default_pdate:
            logger.error("No planting date specified. Set crop.calendar.planting_doy or platform_config.craft.planting_date_mmdd")
            raise ValueError("Planting date is required - specify in crop.calendar or platform_config.craft.planting_date_mmdd")

        # Get management config and zones (framework-agnostic)
        management = getattr(self.config, 'management', None)
        zones = getattr(self.config, 'management_zones', [])

        # Check if any zones have planting date overrides
        zones_with_pdate = [z for z in zones if z.overrides.planting_date_mmdd is not None]
        if zones_with_pdate:
            logger.info(f"Planting date zones configured: {len(zones_with_pdate)} zones with date overrides")
        else:
            logger.info(f"No planting date zones - using uniform date: {default_pdate}")

        # Get GADM-filtered cells if available (for consistency with schema)
        filtered_cells = self._get_filtered_cells(grid)

        # Track zone assignments for logging
        zone_counts: Dict[str, int] = {}
        pdate_counts: Dict[str, int] = {}

        with open(planting_path, 'w') as f:
            # Header (15 columns)
            cols = ["CellID", "PDATE", "EDATE", "PPOP", "PPOE", "PLME", "PLDS",
                    "PLRS", "PLRD", "PLDP", "PLWT", "PAGE", "PENV", "PLPH", "Percent_Share"]
            f.write("\t".join(cols) + "\n")

            # Sort cells by CellID descending (CRAFT convention)
            cells_sorted = sorted(filtered_cells, key=lambda c: self._to_craft_cellid(c.cell_id), reverse=True)

            for cell in cells_sorted:
                # Use 1-indexed CRAFT CellID
                craft_cellid = self._to_craft_cellid(cell.cell_id)

                # Determine planting date with priority:
                # 1. crop_calendar (cell-specific from external source)
                # 2. Zone override (management_zones with planting_date_mmdd)
                # 3. Default from config
                if crop_calendar and cell.cell_id in crop_calendar:
                    pdate = self._doy_to_mmdd(crop_calendar[cell.cell_id].planting_doy)
                    zone_name = "crop_calendar"
                elif management and zones:
                    # Get zone-specific params (includes planting_date_mmdd if set)
                    params = get_management_for_cell(
                        lat=cell.lat,
                        lon=cell.lon,
                        cell_id=craft_cellid,
                        management=management,
                        zones=zones,
                        schema_df=None,  # Admin filtering not implemented yet
                    )
                    zone_name = params.get('matched_zone')
                    # Use zone's planting date if set, otherwise default
                    pdate = params.get('planting_date_mmdd') or default_pdate
                else:
                    pdate = default_pdate
                    zone_name = None

                # Track zone assignment
                zone_key = zone_name or 'default'
                zone_counts[zone_key] = zone_counts.get(zone_key, 0) + 1
                pdate_counts[pdate] = pdate_counts.get(pdate, 0) + 1

                # Write row with all 15 columns (integers for PLRS, PLRD, PLDP)
                row = [
                    str(craft_cellid),
                    pdate,           # PDATE (MMDD format!)
                    "-99",           # EDATE
                    str(ppop),       # PPOP
                    "-99",           # PPOE
                    plme,            # PLME
                    plds,            # PLDS
                    str(plrs),       # PLRS (integer)
                    str(plrd),       # PLRD
                    str(pldp),       # PLDP (integer)
                    "-99",           # PLWT
                    "-99",           # PAGE
                    "-99",           # PENV
                    "-99",           # PLPH
                    "1"              # Percent_Share
                ]
                f.write("\t".join(row) + "\n")

        # Log zone distribution
        if zones_with_pdate:
            logger.info(f"Planting date zone distribution: {zone_counts}")
            logger.info(f"Planting date distribution: {pdate_counts}")
        logger.info(f"Generated CRAFT planting data: {planting_path} ({len(filtered_cells)} cells)")
        return planting_path

    def _generate_fertilizer_data(self, grid: SpatialGrid) -> Path:
        """Generate CRAFT fertilizer data file.

        Supports spatial variability through management zones. If zones are
        configured, each cell gets zone-specific N values. Otherwise, all
        cells use the same defaults (uniform mode - backward compatible).

        Generates 2 rows per cell for split fertilizer applications:
        - Application 1: Basal (at planting)
        - Application 2: Top-dressing (later)

        Output format (tab-separated, 14 columns):
        CellID  Level  FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD  Name  Percent_Fraction

        DSSAT codes:
        - FMCD: Fertilizer material (FE005 = Urea/NPK)
        - FACD: Application method (AP002 = Broadcast incorporated)
        - FDATE: Days after planting (DAP)

        Args:
            grid: SpatialGrid with cells

        Returns:
            Path to generated fertilizer data file
        """
        fert_path = self.output_dir / "management" / "fertilizer_data.txt"
        platform_config = self.get_platform_config()

        # Get DSSAT-specific settings from platform config (not zone-variable)
        fert_material = "FE005"
        fert_application = "AP002"
        fert_depth = 5
        default_p = 10.0
        default_k = 10.0

        if platform_config:
            fert_material = getattr(platform_config, 'fertilizer_material_code', 'FE005')
            fert_application = getattr(platform_config, 'fertilizer_application_code', 'AP002')
            fert_depth = getattr(platform_config, 'fertilizer_depth_cm', 5)
            default_p = getattr(platform_config, 'default_fertilizer_p', 10.0)
            default_k = getattr(platform_config, 'default_fertilizer_k', 10.0)

        # Get management config and zones (framework-agnostic)
        management = getattr(self.config, 'management', None)
        zones = getattr(self.config, 'management_zones', [])

        # Log zone configuration
        if zones:
            logger.info(get_zone_summary(zones))
        else:
            logger.info("No management zones - using uniform fertilizer settings")

        # Create fallback management if not configured
        if management is None:
            # Build a minimal management config from platform config or defaults
            from prismpy.config.schema import ManagementConfig
            default_n = 40.0
            split_ratio = 0.25
            app1_dap = 24
            app2_dap = 34
            density = 55000  # fallback
            if platform_config:
                default_n = getattr(platform_config, 'default_fertilizer_n', 40.0)
                split_ratio = getattr(platform_config, 'fertilizer_split_ratio', 0.25)
                app1_dap = getattr(platform_config, 'fertilizer_app1_dap', 24)
                app2_dap = getattr(platform_config, 'fertilizer_app2_dap', 34)
                # Convert plants/m² to plants/ha if available
                if getattr(platform_config, 'plant_population', None):
                    density = int(platform_config.plant_population * 10000)

            # Convert CRAFT-style split ratio to lists
            fractions = [split_ratio, 1.0 - split_ratio]
            splits = [app1_dap, app2_dap]

            management = ManagementConfig(
                planting_density=density,
                fertilizer_n_total=default_n,
                fertilizer_n_splits=splits,
                fertilizer_n_fractions=fractions,
            )
            logger.debug(f"Using synthesized management config: N={default_n}, splits={splits}, density={density}")

        # Log default fertilizer settings
        logger.info(f"Default fertilizer: N={management.fertilizer_n_total:.1f} kg/ha")
        logger.info(f"Default splits: DAP {management.fertilizer_n_splits}, fractions {management.fertilizer_n_fractions}")
        logger.info(f"DSSAT codes: material={fert_material}, application={fert_application}")

        # Get GADM-filtered cells if available (for consistency with schema)
        filtered_cells = self._get_filtered_cells(grid)

        # Track zone assignments for logging
        zone_counts: Dict[str, int] = {}

        with open(fert_path, 'w') as f:
            # Header (14 columns - matching legacy CRAFT format)
            cols = ["CellID", "Level", "FDATE", "FMCD", "FACD", "FDEP",
                    "FAMN", "FAMP", "FAMK", "FAMC", "FAMO", "FOCD", "Name", "Percent_Fraction"]
            f.write("\t".join(cols) + "\n")

            # Sort cells by CellID descending (CRAFT convention)
            cells_sorted = sorted(filtered_cells, key=lambda c: self._to_craft_cellid(c.cell_id), reverse=True)

            for cell in cells_sorted:
                # Use 1-indexed CRAFT CellID
                craft_cellid = self._to_craft_cellid(cell.cell_id)

                # Get zone-specific management params (or defaults if no zones)
                params = get_management_for_cell(
                    lat=cell.lat,
                    lon=cell.lon,
                    cell_id=craft_cellid,
                    management=management,
                    zones=zones,
                    schema_df=None,  # Admin filtering not implemented yet
                )

                # Track zone assignment
                zone_name = params.get('matched_zone', 'default')
                zone_counts[zone_name] = zone_counts.get(zone_name, 0) + 1

                # Get N params (potentially zone-specific)
                total_n = params['fertilizer_n_total']
                splits = params['fertilizer_n_splits']
                fractions = params['fertilizer_n_fractions']

                # Write row for each application
                for i, (dap, frac) in enumerate(zip(splits, fractions)):
                    n_amount = total_n * frac
                    # P and K only on first application
                    p_amount = default_p if i == 0 else 0
                    k_amount = default_k if i == 0 else 0

                    row = [
                        str(craft_cellid),
                        "1",                      # Level
                        str(dap),                 # FDATE
                        fert_material,            # FMCD
                        fert_application,         # FACD
                        str(fert_depth),          # FDEP
                        f"{n_amount:.0f}",        # FAMN
                        f"{p_amount:.0f}",        # FAMP
                        f"{k_amount:.0f}",        # FAMK
                        "0",                      # FAMC (Ca)
                        "0",                      # FAMO (Other)
                        "-99",                    # FOCD
                        "-99",                    # Name
                        "1"                       # Percent_Fraction (100%)
                    ]
                    f.write("\t".join(row) + "\n")

        # Log zone distribution
        total_rows = len(filtered_cells) * len(management.fertilizer_n_splits)
        if zones:
            logger.info(f"Zone distribution: {zone_counts}")
        logger.info(f"Generated CRAFT fertilizer data: {fert_path} ({total_rows} rows)")
        return fert_path

    def _generate_organic_fertilizer_data(self, grid: SpatialGrid) -> Path:
        """Generate CRAFT organic fertilizer (residue) data file.

        Output format (tab-separated, 13 columns, sorted by CellID descending):
        CellID  Level  RDATE  RCOD  RAMT  RESN  RESP  RESK  RINP  RDEP  RMET  RENAME  Percent_Fraction

        DSSAT residue codes:
        - RCOD: Residue type (RE001=crop residue, RE002=green manure, RE003=manure)
        - RMET: Residue method code

        By default, generates placeholder data with zeros (organic typically disabled).

        Args:
            grid: SpatialGrid with cells

        Returns:
            Path to generated organic fertilizer data file
        """
        organic_path = self.output_dir / "management" / "organic_fertilizer_data.txt"
        platform_config = self.get_platform_config()

        # Get organic fertilizer configuration
        organic_enabled = False
        rcod = "RE001"  # Default: crop residue
        ramt = 0        # Amount (kg/ha dry weight)
        rdate = 0       # Days after planting
        resn = 0        # N content (%)
        resp = 0        # P content (%)
        resk = 0        # K content (%)
        rinp = 0        # Incorporation %
        rdep = 0        # Depth (cm)
        rmet = -99      # Method code
        rename = -99    # Name

        if platform_config:
            organic_enabled = getattr(platform_config, 'organic_fertilizer_enabled', False)
            if organic_enabled:
                rcod = getattr(platform_config, 'organic_residue_code', 'RE001')
                ramt = getattr(platform_config, 'organic_amount', 0)
                rdate = getattr(platform_config, 'organic_dap', 0)

        # Get GADM-filtered cells if available (for consistency with schema)
        filtered_cells = self._get_filtered_cells(grid)

        with open(organic_path, 'w') as f:
            # Header (13 columns - matching legacy CRAFT format)
            cols = ["CellID", "Level", "RDATE", "RCOD", "RAMT", "RESN", "RESP",
                    "RESK", "RINP", "RDEP", "RMET", "RENAME", "Percent_Fraction"]
            f.write("\t".join(cols) + "\n")

            # Sort cells by CellID descending (CRAFT convention)
            cells_sorted = sorted(filtered_cells, key=lambda c: self._to_craft_cellid(c.cell_id), reverse=True)

            for cell in cells_sorted:
                # Use 1-indexed CRAFT CellID
                craft_cellid = self._to_craft_cellid(cell.cell_id)

                row = [
                    str(craft_cellid),
                    "1",              # Level
                    str(rdate),       # RDATE
                    rcod,             # RCOD
                    str(ramt),        # RAMT
                    str(resn),        # RESN
                    str(resp),        # RESP
                    str(resk),        # RESK
                    str(rinp),        # RINP
                    str(rdep),        # RDEP
                    str(rmet),        # RMET
                    str(rename),      # RENAME
                    "1"               # Percent_Fraction (100%)
                ]
                f.write("\t".join(row) + "\n")

        logger.info(f"Generated CRAFT organic fertilizer data: {organic_path} ({len(filtered_cells)} rows)")
        return organic_path

    def _generate_package_metadata(
        self,
        data: UnifiedData,
        output_files: List[Path]
    ) -> List[Path]:
        """Generate package metadata files (manifest, README, provenance).

        Creates self-documenting package with:
        - manifest.json: File inventory with SHA256 checksums
        - README.md: Usage instructions and data source documentation
        - provenance.json: Data lineage and processing decisions

        Args:
            data: UnifiedData container with translation inputs
            output_files: List of files already generated

        Returns:
            List of metadata file paths generated
        """
        logger.info("Generating CRAFT package metadata...")
        metadata_files = []

        # Get platform config
        platform_config = None
        if hasattr(self.config, 'platform_config') and self.config.platform_config:
            platform_config = getattr(self.config.platform_config, 'craft', None)

        # Get management config
        management = getattr(self.config, 'management', None)

        # Count cells and soil profiles
        n_cells = len(self._valid_cellids) if self._valid_cellids else (
            len(data.grid.cells) if data.grid else 0
        )
        n_soil_profiles = len(data.soil) if data.soil else 0

        # Determine data sources from config
        soil_source = "Default profile"
        soil_description = "Generic soil profile for simulation"
        if platform_config:
            hwsd_bil = getattr(platform_config, 'hwsd_bil_path', None)
            hwsd_mdb = getattr(platform_config, 'hwsd_mdb_path', None)
            if hwsd_bil and hwsd_mdb:
                soil_source = "HWSD v2.0"
                soil_description = "Per-SMU profiles from Harmonized World Soil Database"

        crop_mask_source = "Uniform (100%)"
        crop_mask_description = "All cells assumed to have target crop"
        if platform_config:
            spam_path = getattr(platform_config, 'spam_raster_path', None)
            if spam_path:
                crop_mask_source = "SPAM 2020"
                crop_mask_description = "Harvested area fractions from MapSPAM"

        boundary_source = "Bounding box"
        boundary_description = "Manual coordinate bounds"
        if platform_config:
            gadm_path = getattr(platform_config, 'gadm_data_path', None)
            if gadm_path:
                boundary_source = "GADM v4.1"
                boundary_description = "Official administrative boundaries"

        # Get admin names for schema
        admin_level1 = getattr(platform_config, 'admin_level1_name', 'Unknown') if platform_config else 'Unknown'
        admin_level2 = getattr(platform_config, 'admin_level2_name', data.region.name) if platform_config else data.region.name
        admin_names = f"{admin_level1}_{admin_level2}"

        # Get management parameters
        cultivar = "GH0010"
        plant_pop = 5.5
        row_spacing = 75
        total_n = 40.0
        planting_date = "MMDD"
        n_split_ratio = "0.25/0.75"

        if platform_config:
            cultivar = getattr(platform_config, 'default_cultivar', cultivar)
            plant_pop = getattr(platform_config, 'plant_population', plant_pop)
            row_spacing = getattr(platform_config, 'row_spacing_cm', row_spacing)
            total_n = getattr(platform_config, 'default_fertilizer_n', total_n)
            planting_date = getattr(platform_config, 'planting_date_mmdd', planting_date)
            split_ratio = getattr(platform_config, 'fertilizer_split_ratio', 0.25)
            n_split_ratio = f"{split_ratio}/{1-split_ratio}"

        if management:
            cultivar = getattr(management, 'default_cultivar', cultivar)
            total_n = getattr(management, 'fertilizer_n_total', total_n)

        # Build config dict for manifest/README
        country_code = getattr(
            self.config.region, 'country_iso3', 'ML'
        )[:2] if hasattr(self.config, 'region') else 'ML'

        package_config = {
            # Project info
            'project_name': self.config.project.name if hasattr(self.config, 'project') and self.config.project else 'CRAFT Package',
            'package_name': 'craft',

            # Region info
            'region_name': data.region.name,
            'country': data.region.country,
            'country_code': country_code,

            # Crop info
            'crop_name': self.config.crop.name if hasattr(self.config, 'crop') and self.config.crop else 'Unknown',

            # Temporal info
            'start_year': self.config.temporal.start_year if hasattr(self.config, 'temporal') and self.config.temporal else 2010,
            'end_year': self.config.temporal.end_year if hasattr(self.config, 'temporal') and self.config.temporal else 2020,

            # CRAFT-specific
            'n_cells': n_cells,
            'n_soil_profiles': n_soil_profiles,
            'schema_level': getattr(platform_config, 'schema_level', 2) if platform_config else 2,
            'admin_names': admin_names,

            # Data sources
            'soil_source': soil_source,
            'soil_description': soil_description,
            'crop_mask_source': crop_mask_source,
            'crop_mask_description': crop_mask_description,
            'boundary_source': boundary_source,
            'boundary_description': boundary_description,

            # Management parameters
            'cultivar': cultivar,
            'plant_pop': plant_pop,
            'row_spacing': row_spacing,
            'total_n': total_n,
            'planting_date': planting_date,
            'n_split_ratio': n_split_ratio,

            # Data sources dict for manifest
            'data_sources': {
                'soil': soil_source,
                'crop_mask': crop_mask_source,
                'boundaries': boundary_source,
                'climate': 'NASA POWER (to be downloaded)',
            },

            # Additional manifest metadata
            'gadm_level': getattr(platform_config, 'gadm_level', 2) if platform_config else 2,
        }

        # 1. Generate manifest
        try:
            manifest = create_manifest(self.output_dir, package_config, platform='craft')
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
                platform='craft'
            )
            metadata_files.append(readme_path)
            logger.info(f"Generated README: {readme_path}")
        except Exception as e:
            logger.warning(f"Failed to generate README: {e}")

        # 3. Generate provenance.json using stage-based format (like SARRA-Py)
        try:
            from prismpy.packaging.provenance import (
                ProvenanceTracker as PackageProvenance,
                create_decision,
            )
            from datetime import datetime

            # Create stage-based provenance tracker
            tracker = PackageProvenance(
                session_id=f"ct_{data.region.name.lower()}_{datetime.now().strftime('%Y%m%d')}",
                workflow="prismpy"
            )

            # Get bounds for RETRIEVE stage
            bounds = None
            if data.region and data.region.bounds:
                bounds = [
                    data.region.bounds.minx,
                    data.region.bounds.miny,
                    data.region.bounds.maxx,
                    data.region.bounds.maxy,
                ]

            # RETRIEVE stage: Boundary/schema source
            tracker.add_stage(
                "RETRIEVE",
                inputs={
                    "region": data.region.name,
                    "country": data.region.country,
                    "bounds": bounds,
                    "n_cells": n_cells,
                },
                outputs=[
                    f"schema/CRAFT_Schema/Level{package_config['schema_level']}/Schema/5m_{admin_names}.txt",
                    f"schema/Python_Schemas/Level{package_config['schema_level']}/Schema_{admin_names}.txt",
                ],
                decisions=[
                    create_decision(
                        "BOUNDARY_SOURCE",
                        boundary_source,
                        boundary_description,
                        alternatives=["Bounding box", "GADM v4.1"] if boundary_source != "Bounding box" else None
                    )
                ]
            )

            # HARMONIZE stage: Soil and crop mask sources
            tracker.add_stage(
                "HARMONIZE",
                inputs={
                    "soil_source": soil_source,
                    "crop_mask_source": crop_mask_source,
                    "climate_source": "NASA POWER",
                },
                outputs=[
                    f"soil/{country_code}.SOL",
                    "soil/soil_mask.txt",
                    "crop_mask/mask.txt",
                    "weather/input.csv",
                ],
                decisions=[
                    create_decision(
                        "SOIL_SOURCE",
                        soil_source,
                        soil_description,
                        alternatives=["Default profile", "HWSD v2.0"] if soil_source != "Default profile" else None
                    ),
                    create_decision(
                        "CROP_MASK_SOURCE",
                        crop_mask_source,
                        crop_mask_description,
                        alternatives=["Uniform (100%)", "SPAM 2020"] if crop_mask_source != "Uniform (100%)" else None
                    ),
                    create_decision(
                        "CLIMATE_SOURCE",
                        "NASA POWER",
                        "Daily gridded climate data (SRAD, TMAX, TMIN, RAIN) to be downloaded",
                        alternatives=["AgERA5", "CHIRPS"]
                    ),
                ]
            )

            # TRANSLATE stage: Management configuration
            tracker.add_stage(
                "TRANSLATE",
                inputs={
                    "platform": "craft",
                    "cultivar": cultivar,
                    "plant_population": plant_pop,
                    "row_spacing_cm": row_spacing,
                    "total_n_kg_ha": total_n,
                    "planting_date_mmdd": planting_date,
                },
                outputs=[
                    "management/cultivar_data.txt",
                    "management/planting_data.txt",
                    "management/fertilizer_data.txt",
                    "management/organic_fertilizer_data.txt",
                ],
                decisions=[
                    create_decision(
                        "CULTIVAR",
                        cultivar,
                        f"DSSAT cultivar code for {package_config.get('crop_name', 'crop')}"
                    ),
                    create_decision(
                        "FERTILIZER_STRATEGY",
                        f"{total_n} kg N/ha split {n_split_ratio}",
                        "Total nitrogen applied in two applications based on regional practice"
                    ),
                    create_decision(
                        "PLANTING_DATE",
                        planting_date,
                        "Planting date in MMDD format based on regional onset"
                    ),
                ]
            )

            provenance_path = self.output_dir / "provenance.json"
            tracker.save(provenance_path)
            metadata_files.append(provenance_path)
            logger.info(f"Generated provenance: {provenance_path}")
        except Exception as e:
            logger.warning(f"Failed to generate provenance: {e}")

        logger.info(f"Generated {len(metadata_files)} package metadata files")
        return metadata_files
