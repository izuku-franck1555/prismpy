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

from prismpy.cells.admission import canonical_climate_for_grid
from prismpy.config.schema import Platform
from prismpy.data_sources.gadm import GADMDataSource
from prismpy.models.climate import ClimateTimeSeries, ClimateRecord
from prismpy.models.crop import CropParameters, CropCalendar
from prismpy.models.region import Region
from prismpy.models.soil import SoilProfile, SoilLayer
from prismpy.models.spatial import SpatialGrid
from prismpy.provenance.tracker import DecisionType, ProvenanceTracker
from prismpy.sources.climate._cancel import PipelineCancelled, raise_if_cancelled
from prismpy.translators._shared.dssat_sol_writer import write_dssat_sol
# Sprint E.3 AC-E3-9 — cockpit override dispatch helper. The
# climate / soil / management per-cell write sites in this
# translator route raw values through ``apply_override`` before
# writing per-cell platform files; with no sidecar the helper is
# a no-op short-circuit on the raw value, preserving Sprint E.2
# era output byte-equivalent for runs without overrides.
from prismpy.translators._shared.cockpit_overrides import apply_override
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

        # Sprint E.3 fixup +15 (F-BN Boundary 3) — CRAFT translator is
        # in the PYTHIA-only ship's deferred-translator set per the
        # team-lead's scope guard. Surface a runtime warning whenever
        # the cockpit override sidecar is non-empty so the persona
        # gets an honest signal that their override decisions are NOT
        # being applied to CRAFT canonical files; full wiring lands in
        # Phase 4.6 once the PYTHIA path is empirically green.
        cockpit_sidecar = getattr(self, "cockpit_override_sidecar", None)
        if cockpit_sidecar is not None and cockpit_sidecar.overrides:
            warnings.append(
                f"Cockpit override sidecar carries {len(cockpit_sidecar.overrides)} "
                "entries but CRAFT value-replacement wiring is not yet "
                "implemented; values will NOT be applied to canonical CRAFT "
                "files (WTH / SOL). Ship in Phase 4.6 per Sprint E.3 fixup "
                "+15 scope guard."
            )
            self.logger.warning(
                "CRAFT translator: cockpit override sidecar present (%d entries) "
                "but value-replacement wiring deferred to Phase 4.6 — persona's "
                "documented overrides will NOT surface in this CRAFT package.",
                len(cockpit_sidecar.overrides),
            )

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

            # 2b. Download real NASA POWER weather for all cells and
            # generate per-cell weather files (V2-20: self-contained
            # packages — no external CRAFT GUI download needed).
            #
            # Route the climate dict through the canonical admission
            # helper so admission discipline lives in one trusted code
            # path. The helper drops sentinel + foreign + degenerate
            # entries; the count compares the admitted view against
            # n_cells to decide whether to fall back on NASA POWER.
            from prismpy.sources.climate import is_real_climate_cell_id

            if data.grid:
                climate_data = data.climate
                n_cells = len(data.grid.cells)
                gate_canonical = canonical_climate_for_grid(
                    climate_data, data.grid
                )
                n_climate = len(gate_canonical.per_cell)

                if n_climate < n_cells:
                    logger.info(
                        f"Downloading NASA POWER weather for {n_cells} cells "
                        f"({n_climate} already available)..."
                    )
                    def _craft_progress(current, total):
                        cb = getattr(self, 'progress_callback', None)
                        if cb and hasattr(cb, 'on_substage_progress'):
                            cb.on_substage_progress(
                                'translate',
                                'Downloading weather from NASA POWER',
                                current, total,
                                f'cell {current} of {total}',
                            )
                    climate_data = self._download_cell_weather(
                        data, progress_callback=_craft_progress
                    )

                if climate_data:
                    real_climate = canonical_climate_for_grid(
                        climate_data, data.grid
                    ).per_cell
                    if real_climate:
                        weather_files = self._generate_weather_files(real_climate)
                        output_files.extend(weather_files)
                        # F13 — surface real per-cell climate onto the
                        # shared UnifiedData so the cell-summary writer
                        # and per-cell coverage validators observe the
                        # actual loaded state. Without this surfacing
                        # the placeholder sentinel at -1 stays as the
                        # only entry and every real cell renders as
                        # has_climate=False after the §1+§2 v2.1
                        # consumer reads.
                        self._surface_per_cell_climate(data, real_climate)
                    else:
                        warnings.append(
                            "NASA POWER download returned no valid data — "
                            "weather files not generated"
                        )

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

            # 8b. Generate organic fertilizer data (residue) — only if enabled
            if data.grid:
                platform_config = self.get_platform_config()
                organic_enabled = platform_config and getattr(platform_config, 'organic_fertilizer_enabled', False)
                if organic_enabled:
                    organic_fert_file = self._generate_organic_fertilizer_data(data.grid)
                    output_files.append(organic_fert_file)

            # 9. Generate legacy management file (for backward compatibility)
            management_file = self._generate_management(data.crop_params, data.crop_calendar)
            output_files.append(management_file)

            # Validate outputs
            validation_errors = self.validate_outputs()
            if validation_errors:
                warnings.extend(validation_errors)

        except PipelineCancelled:
            # V2-22b L Gate B round 3: translate() outer-try carve-out
            # — NASA POWER per-cell loop raises PipelineCancelled that
            # would otherwise be rewritten as "CRAFT translation failed".
            raise
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
                reference="prismpy.translators.craft.translator.translate",
            )

        # AC-F-CP-13.5: n_weather_files counts REAL cells, not the
        # sentinel placeholder.
        from prismpy.sources.climate import is_real_climate_cell_id
        _real_weather_count = sum(
            1 for k in (data.climate or {}).keys()
            if is_real_climate_cell_id(k)
        )
        result = self.create_result(
            success=True,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "region": data.region.name,
                "n_cells": len(data.grid.cells) if data.grid else 0,
                "n_weather_files": _real_weather_count,
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
        try:
            return self._generate_package_metadata(data, output_files)
        except Exception as e:
            logger.warning(f"Failed to generate package metadata: {e}")
            return []

    def _to_craft_cellid(self, cell_id_0: int) -> int:
        """Return the canonical 0-indexed cell ID for CRAFT companion files.

        Earlier versions of this helper added ``+1`` to match a presumed
        1-indexed CRAFT CellID convention. An empirical smoke test with
        the CRAFT operator team confirmed that CRAFT treats CellID as
        an opaque string key — either convention runs as long as it stays
        consistent within one package. The ``+1`` offset introduced a
        cross-file drift between ``cell_summary.json`` (0-indexed, written
        by the executor) and the CRAFT companions (1-indexed via this
        helper); the drift surfaced at every multi-row bbox row boundary
        because each row's start id collides with the previous row's
        end+1, producing a small but persistent xor mismatch when
        reconciling cell_summary against the companions.

        Aligning to the 0-indexed canonical roster eliminates the drift
        at the source: cell_summary.json is the canonical cell-id source
        for the package, and every CRAFT companion writer inherits the
        same id without offset. The package-internal consistency the
        engine actually checks is preserved.

        Args:
            cell_id_0: 0-indexed cell ID from SpatialGrid (canonical).

        Returns:
            The same 0-indexed cell id; the CRAFT companions now match
            cell_summary.json entry-for-entry.
        """
        return cell_id_0

    def _get_filtered_cells(self, grid: SpatialGrid) -> List:
        """Get the canonical cell roster every CRAFT companion file
        must iterate.

        Returns ``grid.cells`` directly. ``grid.cells`` is the
        canonical post-harmonize roster — the same roster
        ``executor._build_cell_summary`` writes into
        ``cell_summary.json``. By routing every CRAFT writer (schema,
        soil mask, crop mask, cultivar data, planting data) through
        this single source, every companion file in the package
        carries the same set of cell IDs the cockpit, the per-cell
        coverage validators, and the ``cell_summary.json`` consumer
        all agree on.

        Earlier versions of this helper preferred a GADM-resolved
        cell list stored at ``self._gadm_cells`` and would include
        admin-boundary cells that fell outside the grid bounding
        box. That deliberately introduced cells which had no real
        climate or soil data attached — a silent data-cooking
        pattern that the post-Sprint-F audit surfaced empirically:
        ``cell_summary.json`` and ``soil/soil_mask.txt`` could
        differ by 13-28% of the cell count on the same package
        because the soil mask was iterating
        ``self._gadm_cells`` while the summary was iterating
        ``grid.cells``. CRAFT and DSSAT's downstream JOIN across
        these companion files would then either drop the
        mismatched cells silently (best case) or substitute
        placeholder values (worst case). Per
        ``feedback_no_data_cooking.md``: every cell in the package
        is a real cell with real data; admin-region completeness
        is the cockpit's surface, not a synthetic-cell pad in the
        package.

        ``self._valid_cellids`` is also retired here — it was a
        secondary GADM-filter index that pointed at the same
        roster, and routing through ``grid.cells`` makes it
        redundant. The attribute remains assigned during schema
        generation for any external caller that still reads it
        for diagnostics; it does not influence the canonical
        roster the writers iterate.

        Args:
            grid: SpatialGrid with all cells

        Returns:
            ``list(grid.cells)`` — the canonical post-harmonize
            roster matching ``cell_summary.json``.
        """
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
        craft_level = schema_level  # Default, overridden if platform_config exists
        admin_level1_name = None
        admin_level2_name = None
        admin_level3_name = None
        decimal_places = 2

        # GADM configuration
        gadm_data_path = None
        gadm_country_iso3 = None
        gadm_admin_name = None
        gadm_level = None

        # Legacy shapefile path
        shapefile_path = None

        if platform_config:
            schema_level = getattr(platform_config, 'schema_level', 2)
            # craft_level: controls WHICH CRAFT folder to write to (project-relative)
            # Falls back to schema_level for backward compatibility
            craft_level = getattr(platform_config, 'craft_level', None) or schema_level
            admin_level1_name = getattr(platform_config, 'admin_level1_name', None)
            admin_level2_name = getattr(platform_config, 'admin_level2_name', None)
            admin_level3_name = getattr(platform_config, 'admin_level3_name', None)
            decimal_places = getattr(platform_config, 'schema_decimal_places', 2)

            # GADM configuration
            gadm_data_path = getattr(platform_config, 'gadm_data_path', None)
            gadm_country_iso3 = getattr(platform_config, 'gadm_country_iso3', None)
            gadm_admin_name = getattr(platform_config, 'gadm_admin_name', None)
            gadm_level = getattr(platform_config, 'gadm_level', None)

            # Legacy shapefile path
            shapefile_path = getattr(platform_config, 'admin_shapefile_path', None)

        # Derive admin names from region if not explicitly set
        if not admin_level1_name:
            admin_level1_name = sanitize_admin_name(region.country or "Unknown")
        if not admin_level2_name:
            admin_level2_name = sanitize_admin_name(region.name or "Unknown")
        if not gadm_admin_name:
            gadm_admin_name = region.name  # Use region name for GADM filtering

        # Create CRAFT_Schema directory structure using craft_level (project-relative)
        level_dir = self.output_dir / "schema" / "CRAFT_Schema" / f"Level{craft_level}" / "Schema"
        level_dir.mkdir(parents=True, exist_ok=True)
        shape_dir = self.output_dir / "schema" / "CRAFT_Schema" / f"Level{craft_level}" / "Shape"
        shape_dir.mkdir(parents=True, exist_ok=True)

        # Build filename based on craft_level (project-relative hierarchy)
        if craft_level == 1:
            filename = f"5m_{admin_level1_name}.txt"
        elif craft_level == 2:
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
            # F-R AC-3 Path 1: thread BoundaryConfig.min_share_percent
            # so schema rows match the canonical filtered grid built at
            # AC-2 HARMONIZE.
            craft_schema_rows, python_schema_rows = self._generate_schema_from_gadm(
                gadm_data_path=gadm_data_path,
                country_iso3=gadm_country_iso3,
                schema_level=schema_level,
                admin_name=gadm_admin_name,
                decimal_places=decimal_places,
                threshold=self.config.region.boundary.min_share_percent,
            )

            if craft_schema_rows:
                # F-AK substrate fix: intersect the GADM-resolved schema
                # rows with the canonical post-harmonize grid.cells
                # roster so the schema file emits exactly the same cell
                # set every CRAFT companion writer uses (and every
                # cell_summary.json reader downstream observes). The
                # GADM resolver legitimately surfaces admin-boundary
                # cells that fall outside the grid bounding box, but
                # those cells carry no real climate or soil data; the
                # F-AK audit found they leaked into the schema file
                # while the management files (now correctly routed
                # through grid.cells) carried the smaller set, and a
                # CRAFT load that JOIN'ed schema × management would
                # silently drop the 13-28% per-package mismatched
                # cells or substitute placeholders for them. Filtering
                # the schema rows here keeps every companion file in
                # the package in lockstep with cell_summary.json.
                canonical_craft_ids = {
                    self._to_craft_cellid(cell.cell_id)
                    for cell in grid.cells
                }
                rows_before_intersect = len(craft_schema_rows)
                craft_schema_rows = [
                    row for row in craft_schema_rows
                    if row['cellid'] in canonical_craft_ids
                ]
                python_schema_rows = [
                    row for row in python_schema_rows
                    if row['cellid'] in canonical_craft_ids
                ]
                rows_dropped = rows_before_intersect - len(craft_schema_rows)
                if rows_dropped:
                    logger.info(
                        f"GADM resolver returned {rows_before_intersect} "
                        f"rows; {rows_dropped} cells lay outside the "
                        f"canonical grid.cells roster (admin-boundary "
                        f"cells with no real data) and were dropped "
                        f"from the schema file to keep it in lockstep "
                        f"with the management files."
                    )

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
                # Use CRLF line endings for Windows/CRAFT compatibility
                with open(schema_path, 'w', newline='\r\n') as f:
                    f.write("CELLID\tSHAREPERCENT\n")
                    for row in craft_schema_rows:
                        sp = row['share_percent']
                        # Format integer values without decimal (100 not 100.0)
                        if sp == int(sp):
                            f.write(f"{row['cellid']}\t{int(sp)}\n")
                        else:
                            f.write(f"{row['cellid']}\t{sp}\n")

                output_files.append(schema_path)
                logger.info(f"Generated CRAFT schema from GADM: {schema_path} "
                           f"({len(craft_schema_rows)} cells, craft_level {craft_level})")

                # Copy GADM shapefile to Shape/ directory within the package
                # (CRAFT requires shapefiles alongside schema for basemap rendering)
                effective_gadm_level = gadm_level if gadm_level is not None else schema_level - 1
                self._copy_gadm_shapefile_to_shape_dir(
                    gadm_data_path, gadm_country_iso3,
                    effective_gadm_level,
                    shape_dir, admin_level1_name, craft_level
                )

                # Generate 5m_area.txt within the package
                # (user must copy this to C:/CCAFSToolkit/ for CRAFT to find it)
                if python_schema_rows:
                    self._generate_area_file(python_schema_rows)

                return output_files
            else:
                logger.warning("GADM schema generation returned no cells - falling back to grid-based method")

        # =====================================================================
        # OPTION 1b: Use region geometry (from pygadm) for accurate schema
        # =====================================================================
        if hasattr(region, 'geometry_wkt') and region.geometry_wkt:
            try:
                import geopandas as gpd
                from shapely import wkt as shapely_wkt

                geom = shapely_wkt.loads(region.geometry_wkt)
                gdf_pygadm = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

                gadm = GADMDataSource(gadm_path=None)
                # F-R AC-3 Path 1b: thread BoundaryConfig.min_share_percent
                # so the pygadm-geometry schema rows match the canonical
                # filtered grid built at AC-2 HARMONIZE.
                craft_schema_rows, python_schema_rows = gadm.generate_schema_data(
                    gdf=gdf_pygadm,
                    resolution_deg=5 / 60,
                    decimal_places=decimal_places,
                    threshold=self.config.region.boundary.min_share_percent,
                )

                if craft_schema_rows:
                    # F-AK substrate fix (Option 1b path mirror):
                    # intersect the pygadm-geometry schema rows with
                    # the canonical grid.cells roster — same rationale
                    # as the GADM-shapefile path above. Without this
                    # filter the pygadm-geometry path could re-introduce
                    # the same drift class on regions resolved via the
                    # WKT geometry fallback.
                    canonical_craft_ids = {
                        self._to_craft_cellid(cell.cell_id)
                        for cell in grid.cells
                    }
                    rows_before_intersect = len(craft_schema_rows)
                    craft_schema_rows = [
                        row for row in craft_schema_rows
                        if row['cellid'] in canonical_craft_ids
                    ]
                    python_schema_rows = [
                        row for row in python_schema_rows
                        if row['cellid'] in canonical_craft_ids
                    ]
                    rows_dropped = rows_before_intersect - len(craft_schema_rows)
                    if rows_dropped:
                        logger.info(
                            f"pygadm geometry returned {rows_before_intersect} "
                            f"rows; {rows_dropped} cells lay outside the "
                            f"canonical grid.cells roster and were dropped "
                            f"to keep the schema in lockstep with the "
                            f"management files."
                        )

                    self._valid_cellids = {row['cellid'] for row in craft_schema_rows}
                    self._gadm_cells = python_schema_rows

                    logger.info(f"pygadm geometry schema: {len(self._valid_cellids)} cells")

                    with open(schema_path, 'w', newline='\r\n') as f:
                        f.write("CELLID\tSHAREPERCENT\n")
                        for row in craft_schema_rows:
                            sp = row['share_percent']
                            if sp == int(sp):
                                f.write(f"{row['cellid']}\t{int(sp)}\n")
                            else:
                                f.write(f"{row['cellid']}\t{sp}\n")

                    output_files.append(schema_path)
                    logger.info(
                        f"Generated CRAFT schema from pygadm geometry: {schema_path} "
                        f"({len(craft_schema_rows)} cells, craft_level {craft_level})"
                    )

                    if python_schema_rows:
                        self._generate_area_file(python_schema_rows)

                    # Generate GADM shapefiles from pygadm geometry
                    # (CRAFT requires shapefiles for basemap rendering)
                    self._generate_shapefile_from_geometry(
                        gdf_pygadm, shape_dir, region, craft_level
                    )

                    return output_files

            except Exception as e:
                logger.warning(f"pygadm geometry schema generation failed: {e}")

        # =====================================================================
        # OPTION 2: Use grid cells + shapefile/bounding box (fallback)
        # =====================================================================
        # Calculate SharePercent for each cell
        share_percents = self._calculate_share_percent(grid, shapefile_path, decimal_places)

        # Write CRAFT schema file
        # Use CRLF line endings for Windows/CRAFT compatibility
        with open(schema_path, 'w', newline='\r\n') as f:
            f.write("CELLID\tSHAREPERCENT\n")

            cells_with_share = []
            for cell in grid.cells:
                craft_cellid = self._to_craft_cellid(cell.cell_id)
                share_pct = share_percents.get(cell.cell_id, 100.0)
                # Only include cells with non-zero share (intersection with boundary)
                if share_pct > 0:
                    cells_with_share.append((craft_cellid, share_pct))

            # Sort ascending by CellID (matching CRAFT's working schema)
            cells_with_share.sort(key=lambda x: x[0])

            for cellid, share_pct in cells_with_share:
                # Format integer values without decimal (100 not 100.0)
                if share_pct == int(share_pct):
                    f.write(f"{cellid}\t{int(share_pct)}\n")
                else:
                    f.write(f"{cellid}\t{share_pct}\n")

        output_files.append(schema_path)
        logger.info(f"Generated CRAFT schema: {schema_path} ({len(cells_with_share)} cells, craft_level {craft_level})")

        return output_files

    def _generate_schema_from_gadm(
        self,
        gadm_data_path: Path,
        country_iso3: str,
        schema_level: int,
        admin_name: Optional[str] = None,
        decimal_places: int = 2,
        *,
        threshold: float = 0.0,
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
            threshold: F-R AC-3 SharePercent threshold (forwarded to
                ``gadm.generate_schema_data``). Default 0.0 admits all
                intersecting cells (AgMIP-canonical baseline).

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
            threshold=threshold,  # F-R AC-3 SP threshold filter
        )

        logger.info(f"GADM schema generation: {len(craft_rows)} cells from boundary")

        return craft_rows, python_rows

    def _copy_gadm_shapefile_to_shape_dir(
        self,
        gadm_data_path: str,
        country_iso3: str,
        gadm_level: int,
        shape_dir: Path,
        level1_name: str,
        craft_level: int,
    ) -> None:
        """Copy GADM shapefile components to the package's Shape/ directory.

        CRAFT requires shapefiles alongside schema files for basemap rendering
        and ODBC validation. Files are placed WITHIN the output package — the
        user is responsible for copying the package to the CRAFT machine.
        """
        import shutil

        gadm_path = Path(gadm_data_path)
        shp_name = f"gadm41_{country_iso3.upper()}_{gadm_level}"

        # Search for the shapefile in common locations
        search_paths = [
            gadm_path / f"{shp_name}.shp",
            gadm_path / country_iso3.upper() / f"{shp_name}.shp",
            gadm_path / "GADM" / f"{shp_name}.shp",
        ]

        shp_source = None
        for sp in search_paths:
            if sp.exists():
                shp_source = sp
                break

        if not shp_source:
            logger.warning(f"GADM shapefile not found for copying to Shape/. "
                          f"Searched: {[str(p) for p in search_paths]}. "
                          f"CRAFT basemap may not display correctly.")
            return

        # Copy all shapefile components
        shp_stem = shp_source.stem
        shp_dir = shp_source.parent
        extensions = [".shp", ".dbf", ".prj", ".shx", ".cpg", ".sbn", ".sbx", ".shp.xml"]
        copied = 0

        for ext in extensions:
            src = shp_dir / f"{shp_stem}{ext}"
            if src.exists():
                shutil.copy2(src, shape_dir / f"{shp_stem}{ext}")
                copied += 1

        logger.info(f"Copied {copied} shapefile components to {shape_dir}")

        # Ensure Level1Name attribute exists in the copied shapefile
        # (CRAFT uses this column for basemap labeling)
        copied_shp = shape_dir / f"{shp_stem}.shp"
        if copied_shp.exists():
            try:
                import geopandas as _gpd
                gdf = _gpd.read_file(copied_shp)
                if "Level1Name" not in gdf.columns:
                    gdf["Level1Name"] = level1_name
                    if "ObjectId" not in gdf.columns:
                        gdf["ObjectId"] = range(1, len(gdf) + 1)
                    gdf.to_file(copied_shp)
                    logger.info(f"Added Level1Name='{level1_name}' attribute to shapefile")
            except Exception as e:
                logger.warning(f"Could not add Level1Name attribute: {e}")

    def _generate_shapefile_from_geometry(
        self,
        gdf,
        shape_dir: Path,
        region,
        craft_level: int,
    ) -> None:
        """Generate GADM shapefiles from pygadm geometry for CRAFT basemap.

        When local GADM shapefiles aren't available, this creates them from
        the pygadm-downloaded geometry so the package is self-contained.

        Args:
            gdf: GeoDataFrame with the boundary geometry
            shape_dir: Output Shape/ directory in the package
            region: Region object with name/country info
            craft_level: CRAFT level for naming
        """
        try:
            import geopandas as _gpd

            shape_dir.mkdir(parents=True, exist_ok=True)
            country_iso3 = region.country_iso3 if hasattr(region, 'country_iso3') else 'UNK'
            gadm_level = region.gadm_level if hasattr(region, 'gadm_level') else 2
            shp_name = f"gadm41_{country_iso3.upper()}_{gadm_level}"

            # Ensure CRS is set
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")

            # Add attributes CRAFT expects
            if "Level1Name" not in gdf.columns:
                gdf["Level1Name"] = region.name
            if "ObjectId" not in gdf.columns:
                gdf["ObjectId"] = range(1, len(gdf) + 1)
            if f"NAME_{gadm_level}" not in gdf.columns:
                gdf[f"NAME_{gadm_level}"] = region.name

            output_path = shape_dir / f"{shp_name}.shp"
            gdf.to_file(output_path)

            n_files = len([f for f in shape_dir.iterdir() if f.is_file()])
            logger.info(f"Generated {n_files} shapefile components from pygadm geometry at {shape_dir}")

        except Exception as e:
            logger.warning(f"Could not generate shapefile from pygadm geometry: {e}")

    def _generate_area_file(self, python_schema_rows: list) -> None:
        """Generate 5m_area.txt file within the output package.

        CRAFT checks for this file at C:/CCAFSToolkit/5m_area.txt after schema
        generation. We place it in the package's schema/ directory — the user
        must copy it to the CRAFT root directory on the target machine.

        Format: CELLID\\tArea (tab-separated, area in km², 6 decimal places)
        """
        area_path = self.output_dir / "schema" / "5m_area.txt"

        # Sort ascending by CellID
        sorted_rows = sorted(python_schema_rows, key=lambda x: x['cellid'])

        with open(area_path, 'w', newline='\r\n') as f:
            f.write("CELLID\tArea\n")
            for row in sorted_rows:
                area = row.get('area', 0.0)
                f.write(f"{row['cellid']}\t{area:.6f}\n")

        logger.info(f"Generated 5m_area.txt: {area_path} ({len(sorted_rows)} cells)")

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
        craft_level = schema_level  # Default, overridden if platform_config exists
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
            craft_level = getattr(platform_config, 'craft_level', None) or schema_level
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

        # Create Python_Schemas directory using craft_level
        python_schema_dir = self.output_dir / "schema" / "Python_Schemas" / f"Level{craft_level}"
        python_schema_dir.mkdir(parents=True, exist_ok=True)

        # Build filename using craft_level
        if craft_level == 1:
            filename = f"Schema_{admin_level1_name}.txt"
        elif craft_level == 2:
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
            # F-R AC-3 Path 1c (audit MISSED; surfaced at builder Phase 0):
            # mirror Path 1 in the Python-schema-generation method.
            _, python_schema_rows = self._generate_schema_from_gadm(
                gadm_data_path=gadm_data_path,
                country_iso3=gadm_country_iso3,
                schema_level=schema_level,
                admin_name=gadm_admin_name,
                decimal_places=decimal_places,
                threshold=self.config.region.boundary.min_share_percent,
            )

            if python_schema_rows:
                # Write Python schema file from GADM data
                with open(python_schema_path, 'w', newline='\r\n') as f:
                    # Build header based on craft_level (project-relative hierarchy)
                    header_cols = ["CellID", "Latitude", "Longitude", "Elevation", "Area"]
                    if craft_level >= 1:
                        header_cols.append("Level1Name")
                    if craft_level >= 2:
                        header_cols.append("Level2Name")
                    if craft_level >= 3:
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

                        # Add admin names based on craft_level
                        if craft_level >= 1:
                            row.append(admin_level1_name)
                        if craft_level >= 2:
                            row.append(admin_level2_name)
                        if craft_level >= 3:
                            row.append(admin_level3_name or admin_level2_name)

                        f.write("\t".join(row) + "\n")

                output_files.append(python_schema_path)
                logger.info(f"Generated Python schema from GADM: {python_schema_path} ({len(python_schema_rows)} cells)")
                return output_files
            else:
                logger.warning("GADM schema generation returned no cells - falling back to grid-based method")

        # =====================================================================
        # OPTION 1b: Use region geometry (from pygadm) for Python schema
        # =====================================================================
        if not output_files and hasattr(region, 'geometry_wkt') and region.geometry_wkt:
            try:
                import geopandas as gpd
                from shapely import wkt as shapely_wkt

                geom = shapely_wkt.loads(region.geometry_wkt)
                gdf_pygadm = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

                gadm = GADMDataSource(gadm_path=None)
                # F-R AC-3 Path 1d (audit MISSED; surfaced at builder Phase 0):
                # mirror Path 1b in the Python-schema-generation method.
                _, python_schema_rows = gadm.generate_schema_data(
                    gdf=gdf_pygadm,
                    resolution_deg=5 / 60,
                    decimal_places=decimal_places,
                    threshold=self.config.region.boundary.min_share_percent,
                )

                if python_schema_rows:
                    with open(python_schema_path, 'w', newline='\r\n') as f:
                        header_cols = ["CellID", "Latitude", "Longitude", "Elevation", "Area"]
                        if craft_level >= 1:
                            header_cols.append("Level1Name")
                        if craft_level >= 2:
                            header_cols.append("Level2Name")
                        if craft_level >= 3:
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
                            if craft_level >= 1:
                                row.append(admin_level1_name)
                            if craft_level >= 2:
                                row.append(admin_level2_name)
                            if craft_level >= 3:
                                row.append(admin_level3_name or admin_level2_name)
                            f.write("\t".join(row) + "\n")

                    output_files.append(python_schema_path)
                    logger.info(f"Generated Python schema from pygadm geometry: {python_schema_path} ({len(python_schema_rows)} cells)")
                    return output_files

            except Exception as e:
                logger.warning(f"pygadm geometry Python schema generation failed: {e}")

        # =====================================================================
        # OPTION 2: Use grid cells + shapefile/bounding box (fallback)
        # =====================================================================
        # Calculate intersection areas for each cell
        intersection_areas = self._calculate_intersection_areas(grid, shapefile_path)

        # Write Python schema file
        with open(python_schema_path, 'w', newline='\r\n') as f:
            # Build header based on craft_level (project-relative hierarchy)
            header_cols = ["CellID", "Latitude", "Longitude", "Elevation", "Area"]
            if craft_level >= 1:
                header_cols.append("Level1Name")
            if craft_level >= 2:
                header_cols.append("Level2Name")
            if craft_level >= 3:
                header_cols.append("Level3Name")
            f.write("\t".join(header_cols) + "\n")

            # Write cells sorted by CellID ascending (matching CRAFT working schema)
            cells_sorted = sorted(grid.cells, key=lambda c: c.cell_id)

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

                # Add admin names based on craft_level
                if craft_level >= 1:
                    row.append(admin_level1_name)
                if craft_level >= 2:
                    row.append(admin_level2_name)
                if craft_level >= 3:
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

        with open(input_csv_path, 'w', newline='\r\n') as f:
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

    def _download_cell_weather(
        self,
        data,
        progress_callback=None,
    ) -> Dict[int, "ClimateTimeSeries"]:
        """Download NASA POWER weather data for all grid cells.

        V2-20: makes CRAFT packages self-contained by downloading real
        per-cell weather during translate, same pattern as PYTHIA.
        Leverages NASA POWER caching so repeated runs (or runs after
        a PYTHIA run on the same region) don't re-download.

        Args:
            data: UnifiedData with grid info
            progress_callback: Optional callback(current, total)

        Returns:
            Dictionary mapping cell_id to ClimateTimeSeries
        """
        import time
        from prismpy.sources.climate.nasa_power import (
            NASAPowerSource, NASAPowerConfig,
        )

        # Determine date range from config (cross-year-aware)
        if self.config.temporal:
            start_date = f"{self.config.temporal.start_year}-01-01"
            crop_cal = self.config.crop.calendar if self.config.crop else None
            end_date = self.config.temporal.get_climate_end_date(crop_cal).isoformat()
        else:
            raise ValueError(
                "temporal.start_year and end_year are required for "
                "CRAFT weather download"
            )

        # Initialize NASA POWER source with rate limiting + caching
        nasa_config = NASAPowerConfig(
            request_delay=2.0,
            retry_count=3,
            timeout=120,
        )
        # Use cache dir from data_sources config if available
        cache_dir = None
        if (hasattr(self.config, 'data_sources')
                and hasattr(self.config.data_sources, 'cache_dir')
                and self.config.data_sources.cache_dir):
            cache_dir = Path(self.config.data_sources.cache_dir)

        source = NASAPowerSource(
            config=nasa_config,
            cache_dir=cache_dir,
        )

        climate_data = {}
        cells = data.grid.cells if data.grid else []
        total = len(cells)
        failed = []

        logger.info(
            f"Downloading NASA POWER weather for {total} CRAFT cells "
            f"({start_date} to {end_date})"
        )

        for i, cell in enumerate(cells):
            # V2-22b L (AC L.3): per-cell cancel — granularity is one
            # NASA POWER call plus the rate-limiting sleep. Cancel
            # observed here raises before the source.retrieve on the
            # current iteration.
            raise_if_cancelled(
                getattr(self, 'cancel_check', None),
                f"craft.cell={i + 1}/{total}",
            )
            cell_id = cell.cell_id

            if progress_callback:
                progress_callback(i + 1, total)

            try:
                result = source.retrieve(
                    lat=cell.lat,
                    lon=cell.lon,
                    start_date=start_date,
                    end_date=end_date,
                    location_id=cell_id,
                    use_cache=True,
                    cancel_check=getattr(self, 'cancel_check', None),
                )

                if result.success and result.data:
                    result.data.location_id = cell_id
                    climate_data[cell_id] = result.data
                else:
                    failed.append(cell_id)
                    logger.warning(
                        f"NASA POWER failed for cell {cell_id}: "
                        f"{result.errors}"
                    )

            except PipelineCancelled:
                # V2-22b L: per-cell broad except must not swallow
                # cancel; propagate so the translate() boundary can
                # unwind through pipeline.execute.
                raise
            except Exception as e:
                failed.append(cell_id)
                logger.error(
                    f"Exception downloading weather for cell {cell_id}: {e}"
                )

            # V2-22b L: pre-sleep cancel observed on the next
            # iteration rather than after a full 2-s wait.
            raise_if_cancelled(
                getattr(self, 'cancel_check', None),
                f"craft.before_sleep={i + 1}/{total}",
            )
            # Rate limiting between requests
            if i < total - 1:
                time.sleep(2.0)

        logger.info(
            f"Downloaded weather for {len(climate_data)}/{total} cells"
            + (f" ({len(failed)} failed)" if failed else "")
        )

        if failed:
            logger.warning(
                f"Failed cells: {failed[:10]}"
                + (f"... and {len(failed) - 10} more" if len(failed) > 10 else "")
            )

        return climate_data

    def _generate_weather_files(
        self,
        climate_data: Dict[int, ClimateTimeSeries],
        *,
        climate_kind: "ClimateKind" = None,
    ) -> List[Path]:
        """Generate CRAFT tab-separated weather files.

        NOTE: This method generates WTH-format files if climate data is provided.
        For most use cases, use _generate_weather_input_csv() instead to create
        input.csv for NASA Power download via CRAFT GUI.

        CRAFT weather format:
            * ``ClimateKind.OBSERVED`` (default): 5-column legacy path
              ``YRDOY\\tSRAD\\tTMAX\\tTMIN\\tRAIN`` — preserves the existing
              TAMSAT/AgERA5-source emission contract.
            * ``ClimateKind.PROJECTION`` (Sprint G AC-G-7a): 8-column
              projection path ``YRDOY\\tSRAD\\tTMAX\\tTMIN\\tRAIN\\tTDEW\\tRHUM\\tWIND``.
              TDEW is derived via FAO-56 Tetens
              (:func:`prismpy.harmonize.tetens.derive_tdew`) when the
              source supplies hurs + tmean but not tdew (the ISIMIP3b
              standard); honest-signal MISDAT (-99.0) is emitted only
              when the source is genuinely missing both tdew and the
              tmean+hurs pair needed to derive it.

        Args:
            climate_data: Dictionary of cell_id to ClimateTimeSeries
            climate_kind: Source provenance discriminator. Defaults to
                ``ClimateKind.OBSERVED`` (backward-compat with all
                existing callers); pass ``ClimateKind.PROJECTION`` to
                emit the 8-column WTH for ISIMIP3b-derived projection
                packages.

        Returns:
            List of generated weather file paths
        """
        # Late import to keep harmonize/ off the translator's import-
        # time path for callers that don't touch projection climate.
        from prismpy.harmonize.climate_kind import ClimateKind as _ClimateKind
        from prismpy.harmonize.tetens import derive_tdew_for_record_or

        if climate_kind is None:
            climate_kind = _ClimateKind.OBSERVED

        output_files = []
        weather_dir = self.output_dir / "weather"
        is_projection = climate_kind == _ClimateKind.PROJECTION
        MISDAT = -99.0

        for cell_id, ts in climate_data.items():
            # Use CRAFT 1-indexed CellID for self-documenting filenames
            # that match schema/soil file naming (e.g., 4054256.txt)
            craft_cellid = self._to_craft_cellid(cell_id)
            weather_file = weather_dir / f"{craft_cellid}.txt"

            with open(weather_file, 'w', newline='\r\n') as f:
                # Header — column count varies by climate_kind
                if is_projection:
                    f.write(
                        "YRDOY\tSRAD\tTMAX\tTMIN\tRAIN\tTDEW\tRHUM\tWIND\n"
                    )
                else:
                    f.write("YRDOY\tSRAD\tTMAX\tTMIN\tRAIN\n")

                # Data rows
                for record in ts.records:
                    yrdoy = date_to_yrdoy(record.date)
                    srad = record.srad if record.srad is not None else MISDAT
                    tmax = record.tmax if record.tmax is not None else MISDAT
                    tmin = record.tmin if record.tmin is not None else MISDAT
                    rain = record.precip if record.precip is not None else MISDAT

                    if is_projection:
                        # Projection-path: derive TDEW from Tetens when
                        # the source omits it but supplies hurs +
                        # tmean (AC-G-8 + AC-G-7a). Honest-signal
                        # MISDAT only when neither path resolves.
                        rhum = record.rh if record.rh is not None else MISDAT
                        wind = record.wind if record.wind is not None else MISDAT
                        tdew = derive_tdew_for_record_or(
                            explicit_tdew=record.tdew,
                            tmean_celsius=record.tmean,
                            hurs_pct=record.rh,
                            fallback=MISDAT,
                        )
                        f.write(
                            f"{yrdoy}\t{srad:.1f}\t{tmax:.1f}\t{tmin:.1f}\t"
                            f"{rain:.1f}\t{tdew:.1f}\t{rhum:.1f}\t{wind:.1f}\n"
                        )
                    else:
                        f.write(
                            f"{yrdoy}\t{srad:.1f}\t{tmax:.1f}\t{tmin:.1f}\t"
                            f"{rain:.1f}\n"
                        )

            output_files.append(weather_file)
            logger.debug(f"Generated weather file: {weather_file}")

        logger.info(
            f"Generated {len(output_files)} CRAFT weather files "
            f"(climate_kind={climate_kind.value})"
        )
        return output_files

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

        # Skip BIL/MDB query if executor already provided per-cell HWSD profiles
        executor_has_soil = (
            existing_soil_data
            and len(existing_soil_data) > 1
            and any(
                p.source == "hwsd" for p in existing_soil_data.values()
                if hasattr(p, 'source')
            )
        )
        if executor_has_soil:
            logger.info(
                f"Using {len(existing_soil_data)} HWSD profiles from executor "
                f"(skipping redundant BIL/MDB query)"
            )
        elif hwsd_bil_path and hwsd_mdb_path:
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

        # Delegate the DSSAT v4.8 .SOL byte-level layout to the shared writer.
        # The HWSD orchestration above (steps 1-3) lives in the CRAFT
        # translator because it is CRAFT-specific; the file format itself
        # is shared with PYTHIA's eGHR substrate builder, so a single
        # implementation prevents drift between the two emitters.
        smu_to_profile_name: Dict[int, str] = write_dssat_sol(
            soil_path=soil_path,
            profiles_by_id=smu_to_profile,
            country_code=country_code,
            region=region,
        )

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

        with open(mask_path, 'w', newline='\r\n') as f:
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

        with open(management_path, 'w', newline='\r\n') as f:
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

        with open(soil_mask_path, 'w', newline='\r\n') as f:
            # Header
            f.write("CellID\tSoilProfile\tSharePCT\n")

            # Sort cells by CellID descending (CRAFT convention)
            cells_sorted = sorted(filtered_cells, key=lambda c: self._to_craft_cellid(c.cell_id), reverse=True)

            for cell in cells_sorted:
                # Use the canonical 0-indexed cell id for the
                # CellID column so the soil-mask file matches
                # cell_summary.json + every other CRAFT companion
                # writer in the package.
                craft_cellid = self._to_craft_cellid(cell.cell_id)

                # Get profile name from mapping or use fallback formula
                if cell_to_profile and cell.cell_id in cell_to_profile:
                    soil_profile = cell_to_profile[cell.cell_id]
                else:
                    # Fallback: formula-based profile name keyed off
                    # the canonical 0-indexed cell id. The earlier
                    # ``craft_cellid - 1`` form was a back-shift that
                    # paired with the prior +1 ``_to_craft_cellid``
                    # to recover the canonical id; with the helper
                    # now identity, the back-shift would land on
                    # the wrong cell.
                    soil_profile = f"{country_code}0{cell.cell_id}"

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
            # Use a generic DSSAT cultivar code as fallback so the pipeline can complete.
            # The researcher should set a region-specific cultivar for actual simulations.
            crop_name = (self.config.crop.name or "").lower()
            # Each entry pairs the prismpy crop key with a cultivar
            # code that is verified-present in DSSAT's column-1 of the
            # corresponding ``.CUL`` file. The choices favor Sahel
            # personas (Aminata in Niamey, Salif in Burkina/Mali) when
            # there is a defensible regional variant in the registry;
            # otherwise the regional fallback variety is used. The
            # structural test ``test_cultivar_defaults_against_registry``
            # pins each (crop, code) tuple against the ``.CUL`` file so
            # a future contributor cannot silently drift to an
            # unregistered code.
            generic_cultivars = {
                # MZCER048.CUL: "990002 MEDIUM SEASON" — DSSAT regional fallback.
                "maize": "990002",
                # SGCER048.CUL: "IB0001 RIO" — generic sorghum cultivar.
                "sorghum": "IB0001",
                # MLCER048.CUL: "IB0149 Sadore-Local" — ICRISAT Sadore-station
                # local variety, native to the Niamey region. Replaces the
                # prior IB0001 default which is an ECOTYPE pointer in
                # MLCER048.CUL, not a column-1 cultivar code; CRAFT and
                # PYTHIA both reject loads against unregistered cultivars.
                # Cross-repo drift note: prismweb's crop_mappings.py
                # currently returns millet → "990002" from its
                # wizard-side default mapping, so a fall-through that
                # bypasses both prismweb's wizard cultivar_ingeno selection
                # AND its crop_mappings default lands here on IB0149. The
                # resulting prismweb-vs-prismpy split is intentional for
                # the rare double-fallthrough case (Sahel-fit > legacy
                # parity) and is tracked for a companion prismweb PR that
                # will reconcile the wizard default to IB0149 once the
                # cross-repo cultivar contract is locked.
                "millet": "IB0149",
                # RICER048.CUL: "IB0001 IR 8" — IRRI generic rice cultivar.
                "rice": "IB0001",
                # CPGRO048.CUL: "II0003 IT90K-277-2" — IITA Nigeria cowpea.
                # Replaces the prior IB0001 default (not in CPGRO048.CUL
                # column 1) and aligns with prismweb's wizard-side default
                # so the cross-repo cultivar contract stays consistent.
                # II0003 was chosen over CP0005 (TVU3644 - Nigeria, a
                # similarly-Sahel-relevant TVU-series cowpea) because
                # II0003 is the IITA-released IT90K-277-2 line — modern,
                # widely-cultivated, and the prismweb wizard already
                # serves this code as its default. Picking CP0005 would
                # break the cross-repo wizard-vs-fallback parity for a
                # marginal varietal-vintage gain.
                "cowpea": "II0003",
                # PNGRO048.CUL: "IB0001 STARR, v tamnut" — generic groundnut.
                "groundnut": "IB0001",
            }
            default_cultivar = generic_cultivars.get(crop_name, "990002")
            logger.warning(
                f"No cultivar specified — using generic default '{default_cultivar}'. "
                f"Set management.default_cultivar for region-specific results."
            )

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

        with open(cultivar_path, 'w', newline='\r\n') as f:
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
            ppop = getattr(platform_config, 'plant_population', None) or 5.5
            plme = getattr(platform_config, 'planting_method', None) or 'S'
            plds = getattr(platform_config, 'plant_distribution', None) or 'R'
            plrs = int(getattr(platform_config, 'row_spacing_cm', None) or 75)
            plrd = int(getattr(platform_config, 'planting_row_direction', None) or 0)
            pldp = int(getattr(platform_config, 'planting_depth_cm', None) or 5)

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

        with open(planting_path, 'w', newline='\r\n') as f:
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

        with open(fert_path, 'w', newline='\r\n') as f:
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
        # When disabled, ALL numeric fields must be -99 (DSSAT "not applicable").
        # CRAFT validates each field against its valid range and rejects 0.
        organic_enabled = False
        rcod = "RE001"  # Default: crop residue
        ramt = -99      # Amount (kg/ha dry weight)
        rdate = -99     # Days after planting
        resn = -99      # N content (%)
        resp = -99      # P content (%)
        resk = -99      # K content (%)
        rinp = -99      # Incorporation %
        rdep = -99      # Depth (cm)
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

        with open(organic_path, 'w', newline='\r\n') as f:
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

        # Count cells and soil profiles. F-AK: read from data.grid.cells
        # (the canonical post-harmonize roster the schema, soil mask,
        # crop mask, cultivar, and cell_summary writers all iterate
        # post-fix) rather than from self._valid_cellids. The latter
        # was populated from the GADM resolver's pre-intersect output
        # and could over-report by the same 13-28% the F-AK audit
        # surfaced — the README would advertise the GADM-padded count
        # while the package itself shipped the smaller intersected
        # set. Routing through data.grid.cells keeps the README's
        # "Grid Cells" tile in lockstep with the actual file count.
        n_cells = len(data.grid.cells) if data.grid else 0
        n_soil_profiles = len(data.soil) if data.soil else 0

        # V2-19b-fix Finding 7: determine soil source from ACTUAL data, not
        # config inference. The config always has hwsd paths injected by
        # prismweb regardless of whether the pipeline actually used HWSD.
        # Reading SoilProfile.source reflects what the pipeline did, not
        # what was available to it.
        soil_source = "source unavailable"
        soil_description = "Soil source could not be determined"
        if data.soil:
            first_source = next(
                (p.source for p in data.soil.values() if hasattr(p, 'source') and p.source),
                None,
            )
            if first_source:
                soil_source = first_source
                if "iSDA" in first_source or "isda" in first_source:
                    soil_description = "Per-cell profiles from iSDA Africa (30m native)"
                elif "HWSD" in first_source or "hwsd" in first_source:
                    soil_description = "Per-SMU profiles from Harmonized World Soil Database"
                elif "eGHR" in first_source or "eghr" in first_source:
                    soil_description = "Per-cell profiles from eGHR global soil database"
                elif "placeholder" in first_source.lower():
                    soil_source = "Default profile"
                    soil_description = "Generic soil profile for simulation"
                else:
                    soil_description = f"Soil data from {first_source}"

        crop_mask_source = "Uniform (100%)"
        crop_mask_description = "All cells assumed to have target crop"
        if platform_config:
            spam_path = getattr(platform_config, 'spam_raster_path', None)
            if spam_path:
                crop_mask_source = "SPAM 2020"
                crop_mask_description = "Harvested area fractions from MapSPAM"

        # Manifest derivation reads the RESOLVED runtime boundary
        # source so the package label tracks what actually landed
        # on disk (the executor sets ``data.region.boundary_source``
        # after any retrieve-stage fallback fires; a config that
        # requested GADM but fell back to manual bounds resolves to
        # ``manual``). The configured GADM admin level is honored
        # only when the resolved source is GADM, so non-GADM
        # packages emit a null admin level rather than a fake
        # default.
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
        boundary_source, boundary_description = derive_boundary_label(
            resolved_boundary_source, manifest_gadm_level,
        )

        # Get admin names for schema
        admin_level1 = getattr(platform_config, 'admin_level1_name', 'Unknown') if platform_config else 'Unknown'
        admin_level2 = getattr(platform_config, 'admin_level2_name', data.region.name) if platform_config else data.region.name
        admin_names = f"{admin_level1}_{admin_level2}"

        # Get management parameters. Per F-AF-v2 absorption:
        # ``getattr(obj, name, default)`` returns ``None`` when
        # the attribute exists but is None (the wizard now pins
        # at platform_config.craft and leaves
        # management.default_cultivar=None). The README
        # generator's resolver chain recovers, but hardening
        # here too keeps the package_config dict semantically
        # clean and stops the None values from leaking into
        # any future consumer that reads the dict directly.
        def _attr_or(obj, name, fallback):
            """Return ``getattr(obj, name)`` only when the
            attribute exists AND its value is not None / not
            an empty string. Any None or empty result falls
            through to ``fallback``."""
            if obj is None:
                return fallback
            value = getattr(obj, name, None)
            if value is None:
                return fallback
            if isinstance(value, str) and not value.strip():
                return fallback
            return value

        cultivar = "GH0010"
        plant_pop = 5.5
        row_spacing = 75
        total_n = 40.0
        planting_date = "MMDD"
        n_split_ratio = "0.25/0.75"

        if platform_config:
            cultivar = _attr_or(platform_config, 'default_cultivar', cultivar)
            plant_pop = _attr_or(platform_config, 'plant_population', plant_pop)
            row_spacing = _attr_or(platform_config, 'row_spacing_cm', row_spacing)
            total_n = _attr_or(platform_config, 'default_fertilizer_n', total_n)
            planting_date = _attr_or(platform_config, 'planting_date_mmdd', planting_date)
            split_ratio = _attr_or(platform_config, 'fertilizer_split_ratio', 0.25)
            n_split_ratio = f"{split_ratio}/{1-split_ratio}"

        if management:
            cultivar = _attr_or(management, 'default_cultivar', cultivar)
            total_n = _attr_or(management, 'fertilizer_n_total', total_n)

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
            'planting_doy': (
                self.config.crop.calendar.planting_doy
                if (hasattr(self.config, 'crop') and self.config.crop
                    and getattr(self.config.crop, 'calendar', None))
                else None
            ),
            'maturity_doy': (
                self.config.crop.calendar.maturity_doy
                if (hasattr(self.config, 'crop') and self.config.crop
                    and getattr(self.config.crop, 'calendar', None))
                else None
            ),

            # Temporal info
            'start_year': self.config.temporal.start_year if hasattr(self.config, 'temporal') and self.config.temporal else 2010,
            'end_year': self.config.temporal.end_year if hasattr(self.config, 'temporal') and self.config.temporal else 2020,

            # CRAFT-specific
            'n_cells': n_cells,
            'n_soil_profiles': n_soil_profiles,
            'schema_level': getattr(platform_config, 'schema_level', 2) if platform_config else 2,
            'craft_level': (getattr(platform_config, 'craft_level', None) or getattr(platform_config, 'schema_level', 2)) if platform_config else 2,
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

            # Additional manifest metadata. ``gadm_level`` is the
            # configured admin level only when the resolved source
            # is GADM; otherwise it is ``None`` so the manifest
            # signals "no admin level applied" instead of leaking
            # a fake default. Computed once via the resolved-source
            # discriminator above.
            'gadm_level': manifest_gadm_level,

            # Closed-world UC declaration: CRAFT packages serve UC1
            # (yield_forecast), UC3 (sowing_optimization), UC4
            # (drought_management), UC5 (soil_fertility), and UC6
            # (livestock_feed) — the Sahelian-six crop coverage matches
            # all five operationally-active UCs. Empty per-UC dicts
            # signal "use UC defaults at dispatch time".
            'use_case_config': {
                'yield_forecast': {},
                'sowing_optimization': {},
                'drought_management': {},
                'soil_fertility': {},
                'livestock_feed': {},
            },
        }

        # 1. Generate manifest. CRAFT routes through DSSAT (same engine
        # as PYTHIA per ``adapters/craft.py`` ``run_dssat_site`` reuse;
        # F-BP-5 cycle R1 specialist analysis). The CRAFT-emitted .SNX
        # templates ship with PHOSP=N + POTAS=N hard-coded in the
        # ``@N OPTIONS`` row (empirically verified across all 6 source
        # templates), so the soil-fertility P+K silent-no-op disclosure
        # applies unconditionally to every CRAFT package. Trigger key
        # reused from the ACEA-as-PYTHIA-target original site per Lesson
        # #24 canonical-source-or-pin; same single SSOT trigger for all
        # DSSAT-engine producer paths (ACEA, PYTHIA direct, CRAFT). The
        # downstream prismpy emit gate at ``packaging/manifest.py:644-
        # 651`` filters per UC name + per-platform so the joint flag
        # lands only on ``soil_fertility`` for ``platform in {"pythia",
        # "craft"}``.
        manifest_extra: Dict[str, Any] = {
            '_acea_uc5_p_k_silent_no_op_triggered': True,
        }
        try:
            manifest = create_manifest(
                self.output_dir, package_config, platform='craft',
                additional_metadata=manifest_extra,
            )
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

        # V2-20: Legacy System B provenance.json generation deleted.
        # Provenance is now handled by prismpy.provenance.tracker (System A)
        # and distributed to platform dirs via the hybrid save in
        # executor._execute_package.

        logger.info(f"Generated {len(metadata_files)} package metadata files")
        return metadata_files
