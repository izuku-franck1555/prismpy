"""
PYTHIA translator for prismpy.

This module translates unified data to PYTHIA model input format:
- JSON configuration file with PYTHIA-specific syntax
- DSSAT .WTH weather files per site
- Site shapefile with point locations
- GeoTIFF raster files for management and soil data
- SNX DSSAT experiment template

PYTHIA Quirks (from analysis):
1. YRDOY date format: 2015001 = Jan 1, 2015
2. JSON function syntax: lookup_wth::MLCP::vector::...
3. Point shapefile: Requires ID, Latitude, Longitude fields
4. eGHR soil profiles referenced via lookup_ghr::raster::

Reference: PYTHIA/07-JSON-CONFIG-ASSEMBLY/assemble_config.py
Reference: PYTHIA/utils/wth_utils.py
"""

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from prismpy.cells.admission import canonical_climate_for_grid
from prismpy.config.schema import (
    Platform,
    PhenologyConfig,
    PhysiologyConfig,
    ManagementConfig,
    GenericSoilConfig,
    ProjectConfig,
)
from prismpy.models.climate import ClimateTimeSeries, ClimateRecord
from prismpy.models.crop import CropParameters, CropCalendar
from prismpy.models.region import Region
from prismpy.models.soil import SoilProfile
from prismpy.models.spatial import SpatialGrid
from prismpy.provenance.tracker import DecisionType, ProvenanceTracker
from prismpy.sources.climate._cancel import PipelineCancelled, raise_if_cancelled
from prismpy.translators._shared import build_eghr_substrate
# Sprint E.3 AC-E3-9 — cockpit override dispatch helper. The
# climate / soil / management per-cell write sites in this
# translator route raw values through ``apply_override`` before
# writing per-cell platform files; with no sidecar the helper is
# a no-op short-circuit on the raw value, preserving Sprint E.2
# era output byte-equivalent for runs without overrides.
from prismpy.translators._shared.cockpit_overrides import apply_override
from prismpy.translators.base import (
    BaseTranslator,
    PythiaTranslatorBase,
    TranslationResult,
    UnifiedData,
)
from prismpy.utils.date_utils import date_to_yrdoy, doy_to_date


logger = logging.getLogger(__name__)


class BuildEghrSubstrateError(RuntimeError):
    """Raised when the per-package eGHR substrate cannot be built.

    Fires under the canonical substrate path
    (``prefer_canonical_substrate=True``) when ``build_eghr_substrate``
    cannot complete — for example because ``data.grid`` is missing,
    ``data.soil`` is empty, or one of the writers raises a filesystem
    or SQLite error. The legacy bundled-file flow is only entered when
    an operator explicitly opts in via ``prefer_canonical_substrate=
    False``; under the canonical path the helper fails loud rather
    than silently dropping back to the bundled global database, which
    matches the project's honest-signal contract for substrate
    failures.
    """


class PythiaTranslator(PythiaTranslatorBase):
    """Translator for PYTHIA spatial DSSAT system.

    PYTHIA (Python Thinkpiece for Intelligent Agriculture) runs DSSAT
    simulations across spatial grids using weather files and soil rasters.

    Generates:
    1. shapes/sites.shp - Site point shapefile
    2. weather/*.WTH - DSSAT weather files per site
    3. raster/*.tif - Management rasters (fertilizer, planting, cultivar)
    4. templates/*.SNX - DSSAT experiment template
    5. config/pythia_config.json - Main PYTHIA JSON configuration

    Output structure:
        output_dir/
        ├── shapes/
        │   └── sites.shp (+ .shx, .dbf, .prj)
        ├── weather/
        │   └── *.WTH
        ├── raster/
        │   └── *.tif
        ├── templates/
        │   └── template.SNX
        └── config/
            └── pythia_config.json
    """

    def __init__(
        self,
        config: ProjectConfig,
        output_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
        *,
        prefer_canonical_substrate: bool = True,
    ):
        """Initialize the PYTHIA translator.

        Args:
            config: Project configuration.
            output_dir: Base output directory (overrides config).
            provenance: Provenance tracker for audit trail.
            prefer_canonical_substrate: When ``True`` (default), the
                eGHR triple (raster/soil.tif, eGHR/GHR.db, eGHR/{CC}.SOL)
                is synthesized per package via
                :func:`prismpy.translators._shared.build_eghr_substrate`
                from the upstream-resolved per-cell soil profiles. When
                ``False``, the legacy path runs: clip the bundled
                ``pythia.eghr_raster_path`` global raster, copy the
                bundled ``pythia.eghr_database_path`` GHR.db, and copy
                only the country-specific ``.SOL`` files matched by
                ``_get_required_country_codes``. The legacy path is
                slated for removal once a future acceptance criterion
                confirms substrate-only is the canonical happy path
                across every supported country.
        """
        super().__init__(
            config=config,
            output_dir=output_dir,
            provenance=provenance,
        )
        self.prefer_canonical_substrate = prefer_canonical_substrate
        # Sprint S Gate-B-FIX — record the dispatch flag at __init__
        # exit. When the runtime is on stale modules (e.g., dev server
        # still on a pre-Sprint-S sys.modules cache) this log line is
        # silent; presence of the line therefore doubles as a deploy
        # signal for any operator reviewing the run log. Pair with the
        # provenance['eghr_substrate_decision'] field below for the
        # full canonical-source-or-pin (durable §24) auditing chain.
        logger.info(
            "PythiaTranslator initialized: prefer_canonical_substrate=%s "
            "(Sprint S canonical-substrate dispatch %s)",
            self.prefer_canonical_substrate,
            "ENABLED" if self.prefer_canonical_substrate else "DISABLED",
        )

    # DSSAT weather file format constants
    WTH_HEADER = "$WEATHER DATA: Generated by prismpy"
    WTH_COLUMNS = ["DATE", "SRAD", "TMAX", "TMIN", "RAIN", "TDEW", "RHUM", "WIND"]

    def translate(self, data: UnifiedData) -> TranslationResult:
        """Translate unified data to PYTHIA format.

        Generates a complete PYTHIA package including:
        1. Site shapefile with grid points
        2. Weather files (.WTH) - downloaded from NASA POWER if not provided
        3. Soil raster - clipped from eGHR global raster
        4. Crop mask raster - clipped from SPAM 2020
        5. Management rasters (fertilizer, planting DOY, cultivar)
        6. SNX experiment template with Jinja2 variables
        7. PYTHIA JSON configuration
        8. Package metadata (manifest, provenance, README)

        Args:
            data: UnifiedData container with region, grid, climate, soil, calendar

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
            # 1. Generate site shapefile
            if data.grid:
                logger.info("Step 1/8: Generating sites shapefile...")
                shape_file = self._generate_sites_shapefile(data.grid, data.region)
                output_files.append(shape_file)

            # 2. Generate weather files (.WTH)
            # Gate on canonical admission so the missing-sites set
            # reflects which grid cells truly lack a valid series.
            # Foreign keys whose records would otherwise count as
            # coverage are dropped at the helper's admission boundary;
            # the writer-input filter below shares the same helper so
            # the gate's view and the writer's view cannot drift apart.
            logger.info("Step 2/8: Generating weather files...")
            climate_data = data.climate or {}
            all_site_keys = (
                {c.cell_id for c in data.grid.cells} if data.grid else set()
            )
            gate_canonical = canonical_climate_for_grid(
                climate_data, data.grid
            )
            missing_sites = all_site_keys - gate_canonical.per_cell.keys()
            if missing_sites and data.grid:
                logger.info(
                    f"Climate coverage incomplete "
                    f"({len(gate_canonical.per_cell)}/{len(all_site_keys)} sites) — "
                    f"downloading from NASA POWER for {len(missing_sites)} missing sites..."
                )
                # Wire progress callback for substage reporting
                def _pythia_progress(current, total):
                    cb = getattr(self, 'progress_callback', None)
                    if cb and hasattr(cb, 'on_substage_progress'):
                        cb.on_substage_progress(
                            'translate',
                            'Downloading weather from NASA POWER',
                            current, total,
                            f'site {current} of {total}',
                        )
                downloaded = self._download_site_weather(
                    data,
                    subset_site_ids=sorted(missing_sites),
                    progress_callback=_pythia_progress,
                )
                # Merge with existing real climate (preserves partial
                # pre-retrieve coverage; doesn't discard).
                if downloaded:
                    climate_data = {**climate_data, **downloaded}

            # Canonical admission at the producer boundary so the
            # writer never sees the sentinel placeholder, foreign
            # 30-arcmin keys, or degenerate empty / one-record series.
            # The helper iterates ``data.grid.cells`` so weather file
            # IDs map 1:1 to shapefile ``ID`` values and a partial
            # pre-retrieve state survives intact through the merge.
            real_climate_data = canonical_climate_for_grid(
                climate_data, data.grid
            ).per_cell
            if real_climate_data:
                # Pass ``data.grid`` so the writer's sequential IDs
                # match the sites shapefile's ``ID`` column. Missing-
                # climate cells emit a sentinel WTH preserving the
                # numbering instead of renumbering surviving sites.
                weather_files = self._generate_weather_files(
                    real_climate_data, grid=data.grid
                )
                output_files.extend(weather_files)
                # F13 — surface per-cell climate back onto the shared
                # UnifiedData so the cell-summary writer, per-cell
                # coverage validators, and the manifest's len(climate)
                # reader observe the actual climate-loaded state. The
                # download path returns one ClimateTimeSeries per grid
                # cell keyed by cell.cell_id; without this surfacing
                # the placeholder at -1 stays as the only entry and
                # every real cell ends up has_climate=False even
                # though .WTH files exist on disk.
                self._surface_per_cell_climate(data, real_climate_data)
            else:
                warnings.append("No climate data available - weather files not generated")

            # 3. Generate soil raster (clip from eGHR)
            # Skip when the canonical substrate path will produce
            # ``raster/soil.tif`` later in step 7 — otherwise the
            # legacy clip step issues a stale "Soil raster not
            # generated" warning before the canonical builder writes
            # the file, leaving the TranslationResult under-counting
            # the actual artifacts on disk (the codex-flagged
            # confusing-warning class).
            canonical_substrate_will_run = self._canonical_substrate_will_run(data)
            if canonical_substrate_will_run:
                logger.info(
                    "Step 3/8: Skipping legacy soil-raster clip; "
                    "canonical eGHR substrate will produce raster/soil.tif."
                )
            else:
                logger.info("Step 3/8: Generating soil raster...")
                soil_raster = self._generate_soil_raster(data)
                if soil_raster:
                    output_files.append(soil_raster)
                else:
                    warnings.append("Soil raster not generated (eGHR not configured)")

            # 4. Generate crop mask raster (clip from SPAM)
            logger.info("Step 4/8: Generating crop mask raster...")
            crop_mask = self._generate_crop_mask_raster(data)
            if crop_mask:
                output_files.append(crop_mask)
            else:
                warnings.append("Crop mask not generated (SPAM not configured)")

            # 5. Generate management rasters
            logger.info("Step 5/8: Generating management rasters...")
            if data.crop_calendar or data.grid:
                raster_files = self._generate_management_rasters(
                    data.crop_calendar or {}, data.grid
                )
                output_files.extend(raster_files)

            # 6. Generate SNX template
            logger.info("Step 6/9: Generating SNX template...")
            snx_file = self._generate_snx_template(data)
            output_files.append(snx_file)

            # 7. Include eGHR data in package (for self-contained package)
            logger.info("Step 7/9: Including eGHR soil data in package...")
            eghr_dir = self._include_eghr_data(data)
            if eghr_dir:
                # Count files in eGHR directory
                eghr_files = list(eghr_dir.glob("*"))
                logger.info(f"Included {len(eghr_files)} eGHR files in package")
                # Surface the canonically-built soil raster on the
                # output_files list when step 3 deferred to the
                # canonical path. The raster lives at output_dir/
                # raster/soil.tif; without this surfacing the
                # TranslationResult's output count under-reports the
                # canonical artifacts (the codex-flagged P3 class).
                if canonical_substrate_will_run:
                    canonical_raster = self.output_dir / "raster" / "soil.tif"
                    if canonical_raster.exists():
                        output_files.append(canonical_raster)

            # 8. Generate PYTHIA JSON configuration
            logger.info("Step 8/9: Generating PYTHIA JSON config...")
            config_file = self._generate_pythia_json(data)
            output_files.append(config_file)

            # 9. Validate outputs
            validation_errors = self.validate_outputs()
            if validation_errors:
                warnings.extend(validation_errors)

        except PipelineCancelled:
            # V2-22b L Gate B round 3: translate() outer-try carve-out.
            raise
        except Exception as e:
            logger.error(f"PYTHIA translation failed: {e}")
            import traceback
            traceback.print_exc()
            errors.append(str(e))
            from prismpy.errors import classify_to_event_dict
            return self.create_result(
                success=False,
                output_files=output_files,
                errors=errors,
                error_events=[classify_to_event_dict(e)],
            )

        # Record provenance
        if self.provenance:
            self.provenance.record_decision(
                decision_type=DecisionType.FORMAT_CHOICE,
                description=f"Generated PYTHIA inputs for {data.region.name}",
                rationale="PYTHIA requires JSON config with DSSAT .WTH weather files",
                alternatives=["manual configuration"],
                reference="prismpy.translators.pythia.translator.translate",
            )

        # Count output files by type
        weather_count = len([f for f in output_files if str(f).endswith('.WTH')])
        raster_count = len([f for f in output_files if str(f).endswith('.tif')])

        result = self.create_result(
            success=True,
            output_files=output_files,
            warnings=warnings,
            metadata={
                "region": data.region.name,
                "n_sites": len(data.grid.cells) if data.grid else 0,
                "n_weather_files": weather_count,
                "n_rasters": raster_count,
                "total_files": len(output_files),
            },
        )

        self.log_translation_complete(result)
        logger.info(f"PYTHIA package complete: {len(output_files)} files generated")
        return result

    def validate_outputs(self) -> List[str]:
        """Validate generated PYTHIA outputs.

        Returns:
            List of validation error messages
        """
        errors = []

        # Check JSON config exists
        config_path = self.output_dir / "config" / "pythia_config.json"
        if not config_path.exists():
            errors.append(f"Missing pythia_config.json at {config_path}")
        else:
            # Validate JSON structure
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                required_keys = ["name", "default_setup", "runs"]
                for key in required_keys:
                    if key not in config:
                        errors.append(f"Missing required key '{key}' in pythia_config.json")

            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in pythia_config.json: {e}")

        # Check weather directory
        weather_dir = self.output_dir / "weather"
        if weather_dir.exists():
            wth_files = list(weather_dir.glob("*.WTH"))
            if not wth_files:
                errors.append("No .WTH files generated in weather/")

        # Check shapes directory
        shapes_dir = self.output_dir / "shapes"
        if shapes_dir.exists():
            shp_files = list(shapes_dir.glob("*.shp"))
            if not shp_files:
                errors.append("No shapefile generated in shapes/")

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
        # §7 — copy the observed-trials CSV (if supplied) to data/n_trials.csv
        # BEFORE the manifest is built, so create_manifest's n_trials_present gate
        # sees the real artifact. Fail-loud (see BaseTranslator._copy_observed_trials).
        trials = self._copy_observed_trials()
        package_files = self._generate_package_files(data)
        if trials is not None:
            package_files.append(trials)
        return package_files

    def _generate_sites_shapefile(
        self,
        grid: SpatialGrid,
        region: Region,
    ) -> Path:
        """Generate PYTHIA sites point shapefile.

        PYTHIA requires a shapefile with:
        - ID field (unique site identifier)
        - Latitude field
        - Longitude field

        Args:
            grid: SpatialGrid with cells/sites
            region: Region for metadata

        Returns:
            Path to generated shapefile
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            # Create point features with SEQUENTIAL IDs (1, 2, 3, ...)
            # CRITICAL: ID must match weather file naming (1.WTH, 2.WTH, ...)
            # The lookup_wth function uses: lookup_wth::<prefix>::vector::<shapefile>::ID
            # PYTHIA looks up the ID value and constructs the weather filename from it
            features = []
            for seq_id, cell in enumerate(grid.cells, start=1):
                features.append({
                    "geometry": Point(cell.lon, cell.lat),
                    "ID": seq_id,  # Sequential ID matching weather file names
                    "CellID": cell.cell_id,  # Original grid cell ID for reference
                    "Latitude": cell.lat,
                    "Longitude": cell.lon,
                    "Region": region.name,
                })

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")

            # Save shapefile
            shp_path = self.output_dir / "shapes" / "sites.shp"
            gdf.to_file(shp_path)

            logger.info(f"Generated PYTHIA sites shapefile: {shp_path} ({len(features)} sites)")
            return shp_path

        except ImportError:
            logger.warning("geopandas not available, creating CSV instead")
            return self._generate_sites_csv(grid, region)

    def _generate_sites_csv(
        self,
        grid: SpatialGrid,
        region: Region,
    ) -> Path:
        """Fallback: Generate sites as CSV when geopandas unavailable.

        Args:
            grid: SpatialGrid with cells/sites
            region: Region for metadata

        Returns:
            Path to generated CSV
        """
        csv_path = self.output_dir / "shapes" / "sites.csv"

        with open(csv_path, 'w') as f:
            # Use sequential IDs to match weather file naming (1.WTH, 2.WTH, ...)
            f.write("ID,CellID,Latitude,Longitude,Region\n")
            for seq_id, cell in enumerate(grid.cells, start=1):
                f.write(f"{seq_id},{cell.cell_id},{cell.lat},{cell.lon},{region.name}\n")

        logger.info(f"Generated PYTHIA sites CSV: {csv_path}")
        return csv_path

    def write_weather_files(
        self,
        climate_data: Dict[int, ClimateTimeSeries],
        *,
        climate_kind,
        grid: Optional[SpatialGrid] = None,
    ) -> List[Path]:
        """Write DSSAT ``.WTH`` weather files into this package's ``weather/``.

        Public seam over the internal writer for clone-and-swap orchestration:
        a scenario-set generator constructs the translator normally (via
        ``__init__``), points ``output_dir`` at a cloned baseline package, and
        calls this to write the projection weather without re-running the full
        package pipeline. ``climate_kind=ClimateKind.PROJECTION`` selects the
        projection WTH path (FAO-56 Tetens dewpoint from ``record.rh`` /
        ``record.tmean`` when ``record.tdew`` is None). When ``grid`` is
        provided, sequential WTH IDs match the ``sites.shp`` ID column.

        Args:
            climate_data: Mapping of cell_id to ClimateTimeSeries.
            climate_kind: Source-provenance discriminator (OBSERVED / PROJECTION).
            grid: Optional SpatialGrid for sites.shp-aligned sequential IDs.

        Returns:
            List of generated ``.WTH`` file paths.
        """
        return self._generate_weather_files(
            climate_data, climate_kind=climate_kind, grid=grid
        )

    def _generate_weather_files(
        self,
        climate_data: Dict[int, ClimateTimeSeries],
        *,
        climate_kind: "ClimateKind" = None,
        grid: Optional[SpatialGrid] = None,
    ) -> List[Path]:
        """Generate DSSAT .WTH weather files.

        .WTH format:
        $WEATHER DATA: ...
        @ INSI       LAT      LONG    ELEV   TAV   AMP REFHT WNDHT
          XXXX  latitude  longitude  elev   tav   amp   2.0   2.0

        @  DATE  SRAD  TMAX  TMIN  RAIN  TDEW  RHUM  WIND
        YRDOY  srad  tmax  tmin  rain  tdew  rhum  wind

        Per Sprint G AC-G-7a, the writer accepts a ``climate_kind``
        discriminator so the projection path derives TDEW via FAO-56
        Tetens (:func:`prismpy.harmonize.tetens.derive_tdew_for_record_or`)
        when source ``record.tdew`` is None but ``record.rh`` and
        ``record.tmean`` are present (the ISIMIP3b standard). The
        observed-climate path keeps the legacy MISDAT (-99.0) fallback
        because TAMSAT/AgERA5 sources don't reliably supply hurs to
        derive TDEW from. Both paths emit the 8-column WTH; the only
        behavior difference is whether tdew is computed or sentinel-
        replaced.

        Args:
            climate_data: Dictionary of site_id to ClimateTimeSeries
            climate_kind: Source provenance discriminator. Defaults to
                ``ClimateKind.OBSERVED`` for backward-compat with all
                existing callers.
            grid: Optional SpatialGrid. When provided, .WTH sequential
                IDs match ``sites.shp`` (``enumerate(grid.cells,
                start=1)``) so runtime lookups via the sites
                shapefile's ``ID`` column resolve to the correct
                weather file. Cells without a valid climate series
                emit a sentinel WTH with the header only, no data
                rows — the Coverage validator surfaces those as
                missing-coverage honestly instead of silently
                renumbering the surviving sites. When ``grid`` is
                None, the writer falls back to ``sorted(climate_data
                .keys())`` ordering for back-compat with callers that
                pre-date the parity requirement.

        Returns:
            List of generated .WTH file paths
        """
        # Late import — see CRAFT translator for the rationale.
        from prismpy.harmonize.climate_kind import ClimateKind as _ClimateKind
        from prismpy.harmonize.tetens import derive_tdew_for_record_or
        from prismpy.translators._shared.cockpit_overrides import apply_override

        if climate_kind is None:
            climate_kind = _ClimateKind.OBSERVED
        is_projection = climate_kind == _ClimateKind.PROJECTION
        output_files = []
        weather_dir = self.output_dir / "weather"

        # Sprint E.3 fixup +15 (F-BN Boundary 3) — read the loaded
        # cockpit override sidecar from the translator attribute the
        # executor sets at ``_execute_translate`` entry. ``None`` is
        # the universal short-circuit; ``apply_override`` below
        # returns the raw record-field value unchanged in that case.
        # v1 semantic per team-lead disposition 3: a season-aggregate
        # override (e.g.,
        # ``tmax_growing_season_mean=999.0``) is applied as a
        # per-day constant — every daily record's tmax becomes the
        # override value. Persona-documented-anomaly maps to
        # season-aggregate by design; daily-variance preservation
        # deferred to Phase 4.6 crop-modeling-specialist review.
        cockpit_sidecar = getattr(self, "cockpit_override_sidecar", None)

        # Build the emission roster. When ``grid`` is provided, the
        # sequential IDs come from ``enumerate(grid.cells, start=1)``
        # — the same numbering ``_generate_sites_shapefile`` uses for
        # the shapefile ``ID`` column. PYTHIA's runtime expectation
        # (``lookup_wth::<prefix>::vector::<shapefile>::ID``) is that
        # ``<ID>.WTH`` exists for every shapefile row; missing-climate
        # cells therefore get a sentinel WTH that preserves the
        # numbering and lets the Coverage validator surface the gap.
        # When ``grid`` is None, the legacy ``sorted(climate_data
        # .keys())`` ordering is preserved for back-compat with any
        # caller that pre-dates the parity contract.
        emission_roster: List[Tuple[int, int, Optional[ClimateTimeSeries]]] = []
        if grid is not None and grid.cells:
            for seq_num, cell in enumerate(grid.cells, start=1):
                emission_roster.append(
                    (seq_num, cell.cell_id, climate_data.get(cell.cell_id))
                )
        else:
            site_ids = sorted(climate_data.keys())
            for seq_num, site_id in enumerate(site_ids, start=1):
                emission_roster.append(
                    (seq_num, site_id, climate_data[site_id])
                )

        for seq_num, site_id, ts in emission_roster:
            wth_path = weather_dir / f"{seq_num}.WTH"

            # Sentinel emit for cells without a valid series: header
            # only, no data rows. The Coverage validator reads the
            # empty data section and surfaces missing-coverage
            # honestly. Filename slot is preserved so the sites.shp
            # ID → WTH lookup keeps working for every other cell.
            if ts is None or not (
                hasattr(ts, "records")
                and ts.records
                and len(ts.records) > 1
            ):
                with open(wth_path, "w") as f:
                    f.write(f"{self.WTH_HEADER}\n\n")
                    f.write(
                        "@ INSI       LAT      LONG    ELEV   TAV   "
                        "AMP REFHT WNDHT\n"
                    )
                    f.write(
                        "@  DATE  SRAD  TMAX  TMIN  RAIN  TDEW  RHUM  WIND\n"
                    )
                output_files.append(wth_path)
                continue

            station_code = f"{seq_num}"  # Use just the number

            # Calculate TAV (annual average temp) and AMP (amplitude)
            tav, amp = self._calculate_tav_amp(ts)

            # Determine elevation
            elev = ts.elevation if hasattr(ts, 'elevation') and ts.elevation else -99

            # Generate .WTH filename (matching legacy format: 1.WTH, 2.WTH, etc.)
            wth_path = weather_dir / f"{seq_num}.WTH"

            with open(wth_path, 'w') as f:
                # Header
                f.write(f"{self.WTH_HEADER}\n\n")

                # Station info line (4-character station code: Sxxx format)
                insi = f"S{seq_num:03d}"[:4]  # S001, S002, etc.
                f.write("@ INSI       LAT      LONG    ELEV   TAV   AMP REFHT WNDHT\n")
                f.write(f"  {insi:4s} {ts.lat:9.5f} {ts.lon:9.5f} ")
                f.write(f"{elev:7.2f} {tav:5.1f} {amp:5.1f}   2.0   2.0\n\n")

                # Data header
                f.write("@  DATE  SRAD  TMAX  TMIN  RAIN  TDEW  RHUM  WIND\n")

                # Data rows
                for record in ts.records:
                    yrdoy = date_to_yrdoy(record.date)

                    # Sprint D.1 AC-2 — every climate field uses the
                    # DSSAT-canonical -99.0 missing-value sentinel
                    # (Jones 2003 DSSAT v4.7 Wth.WTH spec). Rain
                    # previously defaulted to 0.0 on missing and
                    # was then re-clamped from -99 to 0.0 on the
                    # next line; both paths silently filled missing
                    # rain with phantom zero-rain days. Papers using
                    # PYTHIA-derived Sahel rainy-season rain
                    # analyses would attribute yield variability to
                    # zero-rain days that never existed.
                    MISDAT = -99.0
                    srad = record.srad if record.srad is not None else MISDAT
                    tmax = record.tmax if record.tmax is not None else MISDAT
                    tmin = record.tmin if record.tmin is not None else MISDAT
                    rain = record.precip if record.precip is not None else MISDAT

                    # Sprint E.3 fixup +15 (F-BN Boundary 3) — per-cell
                    # value-replacement override. v1 semantic: a
                    # season-aggregate override on
                    # ``<var>_growing_season_mean`` /
                    # ``precip_growing_season_total`` applies as a
                    # per-day constant — every record's field becomes
                    # the override value. The new season-mean exactly
                    # matches the override, so the validator no longer
                    # flags the cell. Per team-lead disposition 3 +
                    # AC #1 literal user-snippet reading per durable
                    # §25. ``apply_override`` is a pure helper; the
                    # sidecar=None short-circuit on the universal
                    # no-override path makes the call free for runs
                    # without overrides. ``str(site_id)`` matches the
                    # sidecar entry's ``cell_id`` field per durable
                    # §27 producer-consumer parity (sidecar carries
                    # cell_id as str; site_id keys are ints).
                    cell_id_str = str(site_id)
                    tmax = apply_override(
                        cell_id_str,
                        "tmax_growing_season_mean",
                        tmax,
                        cockpit_sidecar,
                    )
                    tmin = apply_override(
                        cell_id_str,
                        "tmin_growing_season_mean",
                        tmin,
                        cockpit_sidecar,
                    )
                    rain = apply_override(
                        cell_id_str,
                        "precip_growing_season_total",
                        rain,
                        cockpit_sidecar,
                    )
                    srad = apply_override(
                        cell_id_str,
                        "srad_growing_season_mean",
                        srad,
                        cockpit_sidecar,
                    )

                    if is_projection:
                        # Projection-source TDEW comes from Tetens
                        # (record.tdew preferred when supplied; rh +
                        # tmean derive otherwise; MISDAT only when
                        # neither resolves). Per AC-G-7a + AC-G-8.
                        tdew = derive_tdew_for_record_or(
                            explicit_tdew=record.tdew,
                            tmean_celsius=record.tmean,
                            hurs_pct=record.rh,
                            fallback=MISDAT,
                        )
                    else:
                        # Observed-climate path keeps the legacy MISDAT
                        # fallback for missing TDEW. TAMSAT/AgERA5
                        # don't reliably supply hurs to derive TDEW
                        # from, and emitting -99.0 is the documented
                        # DSSAT "data not available" semantic per the
                        # Jones 2003 v4.7 Wth.WTH spec.
                        tdew = record.tdew if record.tdew is not None else MISDAT
                    rhum = record.rh if record.rh is not None else MISDAT
                    wind = record.wind if record.wind is not None else MISDAT

                    # Clamp negative real-rain values to 0 (data
                    # error correction), but never the missing
                    # sentinel — let MISDAT propagate honestly.
                    if rain != MISDAT and rain < 0.0:
                        rain = 0.0

                    f.write(f"{yrdoy:7d} {srad:5.1f} {tmax:5.1f} {tmin:5.1f} ")
                    f.write(f"{rain:5.1f} {tdew:5.1f} {rhum:5.1f} {wind:5.1f}\n")

            output_files.append(wth_path)
            logger.debug(f"Generated weather file: {wth_path}")

        logger.info(
            f"Generated {len(output_files)} PYTHIA .WTH files "
            f"(climate_kind={climate_kind.value})"
        )
        return output_files

    def _calculate_tav_amp(self, ts: ClimateTimeSeries) -> Tuple[float, float]:
        """Calculate TAV (annual average temperature) and AMP (amplitude).

        TAV = average of all mean temperatures
        AMP = (max monthly average - min monthly average) / 2

        Args:
            ts: ClimateTimeSeries

        Returns:
            Tuple of (TAV, AMP)
        """
        if not ts.records:
            return -99.0, -99.0

        # Calculate mean temperatures
        tmean_fallback_used = False
        tmeans = []
        for record in ts.records:
            if record.tmean is not None:
                tmeans.append(record.tmean)
            elif record.tmax is not None and record.tmin is not None:
                tmeans.append((record.tmax + record.tmin) / 2)
                tmean_fallback_used = True

        if not tmeans:
            return -99.0, -99.0

        # V2-19 B0 finding #3: record tmean fallback (tmax+tmin)/2 as
        # an explicit DEFAULT_VALUE decision whenever the fallback was
        # actually exercised. This distinguishes source-provided tmean
        # from our derived approximation.
        # F15 sibling-sweep: the prior description leaked the
        # ``tmean`` variable name + the slash-formula notation into a
        # researcher-facing surface (Methods tab via the prismweb
        # provenance reader). Plain-language description now spells
        # out what happens conceptually; the technical method
        # (variable names, formula) stays in the rationale field
        # below where Dr. Kofi's audit-grep finds it.
        if tmean_fallback_used and self.provenance:
            self.provenance.record_decision(
                decision_type=DecisionType.DEFAULT_VALUE,
                description=(
                    "Daily mean temperature filled from the average of "
                    "the day's min and max where the source did not "
                    "provide a mean directly"
                ),
                rationale=(
                    "When the climate source returns no mean temperature "
                    "for a given day, the arithmetic mean of tmax and tmin "
                    "(``(tmax + tmin) / 2``) is used as a best-effort "
                    "approximation. This is a standard practice but "
                    "slightly biases the annual average because the true "
                    "daytime-weighted mean is closer to 0.5 * (tmax + "
                    "tmin + diurnal-shape correction)."
                ),
                alternatives=[
                    "Drop records with missing tmean (reduces sample size)",
                    "Use diurnal-cycle correction from sunrise/sunset times",
                ],
                reference=(
                    "prismpy.translators.pythia.translator._calculate_tav_amp "
                    "line ~484 ((tmax + tmin) / 2 fallback)"
                ),
            )

        tav = float(np.mean(tmeans))

        # V2-19 B0 finding #4: TAV is an unweighted arithmetic mean over
        # all records in the series — no seasonal or monthly weighting.
        # F15 (2026-04-28): the human-readable ``description`` field
        # surfaces in the Methods tab via the provenance reader.
        # The earlier "PYTHIA TAV: unweighted arithmetic mean of daily
        # mean temperatures" wording leaked the CLI parameter name +
        # statistical method into a researcher-facing surface (durable
        # lesson #7 CLI-artifact-leak). Plain-language phrasing now
        # describes the user-visible meaning ("what this number does
        # for the simulation"); the technical method stays in the
        # rationale field below where Dr. Kofi's audit grep finds it.
        if self.provenance:
            self.provenance.record_decision(
                decision_type=DecisionType.AGGREGATION_METHOD,
                description=(
                    "Average annual temperature used for soil thermal "
                    "layer calibration"
                ),
                rationale=(
                    "Computed as the unweighted arithmetic mean of "
                    "daily mean temperatures across the climate series "
                    "(``np.mean(tmeans)``). Every day weighs equally — "
                    "a year with more cool-dry-season days reads as the "
                    "same TAV as a year with more warm-wet-season days. "
                    "DSSAT's TAV parameter expects 'annual average "
                    "temperature' and this matches the traditional "
                    "unweighted definition."
                ),
                alternatives=[
                    "Monthly average then average-of-averages (more stable)",
                    "Area-weighted when aggregating across cells (N/A here)",
                ],
                reference=(
                    "prismpy.translators.pythia.translator._calculate_tav_amp "
                    "line ~489 (np.mean(tmeans))"
                ),
            )

        # Calculate monthly averages for AMP
        monthly_temps = {}
        for record in ts.records:
            month = record.date.month
            tmean = record.tmean if record.tmean else (record.tmax + record.tmin) / 2
            if month not in monthly_temps:
                monthly_temps[month] = []
            monthly_temps[month].append(tmean)

        if len(monthly_temps) < 2:
            amp = 5.0  # Default
        else:
            monthly_avgs = [np.mean(temps) for temps in monthly_temps.values()]
            amp = (max(monthly_avgs) - min(monthly_avgs)) / 2

        return round(tav, 1), round(amp, 1)

    def _generate_management_rasters(
        self,
        crop_calendar: Dict[int, CropCalendar],
        grid: Optional[SpatialGrid] = None,
    ) -> List[Path]:
        """Generate management GeoTIFF rasters for PYTHIA.

        Creates rasters for:
        - Planting DOY (planting_doy.tif)
        - Fertilizer application (fertilizer.tif)
        - Cultivar zone (cultivar.tif)

        Args:
            crop_calendar: Crop calendar per cell
            grid: SpatialGrid for spatial reference

        Returns:
            List of generated raster file paths
        """
        output_files = []
        raster_dir = self.output_dir / "raster"
        raster_dir.mkdir(parents=True, exist_ok=True)

        # Check if we have grid info for proper raster generation
        if grid is None or not grid.cells:
            logger.warning("No grid data available, generating minimal rasters")
            # Fall back to CSV if no grid
            return self._generate_management_csv_fallback(crop_calendar, raster_dir)

        try:
            import rasterio
            from rasterio.transform import from_bounds

            # Calculate bounds from grid cells
            lats = [cell.lat for cell in grid.cells]
            lons = [cell.lon for cell in grid.cells]
            minx, maxx = min(lons), max(lons)
            miny, maxy = min(lats), max(lats)

            # Add buffer
            buffer = 0.1
            minx -= buffer
            miny -= buffer
            maxx += buffer
            maxy += buffer

            # Calculate raster dimensions based on grid resolution
            pythia_config = None
            if self.config.platform_config and hasattr(self.config.platform_config, 'pythia'):
                pythia_config = self.config.platform_config.pythia

            resolution = pythia_config.grid_resolution if pythia_config else 0.0833
            width = max(1, int((maxx - minx) / resolution))
            height = max(1, int((maxy - miny) / resolution))

            # Create transform
            transform = from_bounds(minx, miny, maxx, maxy, width, height)

            # Get values from generic config mapping
            fertilizer_params = self._map_generic_to_fertilizer()
            cultivar_params = self._map_generic_to_cultivar()

            # Get planting DOY from config - REQUIRED
            if not self.config.crop.calendar or not self.config.crop.calendar.planting_doy:
                raise ValueError(
                    "crop.calendar.planting_doy is required for management raster generation. "
                    "Please specify the planting day of year in your config file."
                )
            planting_doy = self.config.crop.calendar.planting_doy

            # Raster metadata
            meta = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': 'float32',
                'crs': 'EPSG:4326',
                'transform': transform,
                'nodata': -9999,
            }

            # 1. Fertilizer raster (uniform value)
            fert_value = fertilizer_params.get("fen_tot", 60)
            fert_data = np.full((height, width), fert_value, dtype=np.float32)
            fert_path = raster_dir / "fertilizer.tif"

            with rasterio.open(fert_path, 'w', **meta) as dst:
                dst.write(fert_data, 1)
                dst.update_tags(units="kg_N_per_ha", description="Fertilizer N application")

            output_files.append(fert_path)
            logger.info(f"Generated fertilizer raster: {fert_path} ({fert_value} kg N/ha)")

            # 2. Planting DOY raster (uniform value)
            plant_data = np.full((height, width), planting_doy, dtype=np.float32)
            plant_path = raster_dir / "planting_doy.tif"

            with rasterio.open(plant_path, 'w', **meta) as dst:
                dst.write(plant_data, 1)
                dst.update_tags(units="day_of_year", description="Planting day of year")

            output_files.append(plant_path)
            logger.info(f"Generated planting DOY raster: {plant_path} (DOY {planting_doy})")

            # 3. Cultivar zone raster (uniform value = zone 1)
            cultivar_zone = 1  # Single cultivar zone
            cult_data = np.full((height, width), cultivar_zone, dtype=np.float32)
            cult_path = raster_dir / "cultivar.tif"

            with rasterio.open(cult_path, 'w', **meta) as dst:
                dst.write(cult_data, 1)
                dst.update_tags(
                    units="zone_id",
                    description="Cultivar zone",
                    cultivar_code=cultivar_params.get("ingeno", self._get_default_cultivar_ingeno()),
                    cultivar_name=cultivar_params.get("cname", self._get_default_cultivar_cname()),
                )

            output_files.append(cult_path)
            logger.info(f"Generated cultivar raster: {cult_path} (zone {cultivar_zone})")

            logger.info(f"Generated {len(output_files)} PYTHIA management GeoTIFFs")
            return output_files

        except ImportError:
            logger.warning("Rasterio not available, falling back to CSV")
            return self._generate_management_csv_fallback(crop_calendar, raster_dir)

    def _generate_management_csv_fallback(
        self,
        crop_calendar: Dict[int, CropCalendar],
        raster_dir: Path,
    ) -> List[Path]:
        """Fallback: Generate management data as CSV when rasterio unavailable."""
        output_files = []

        # Planting data
        planting_path = raster_dir / "planting_doy.csv"
        with open(planting_path, 'w') as f:
            f.write("CellID,PlantingDOY,HarvestDOY\n")
            for cell_id, calendar in crop_calendar.items():
                f.write(f"{cell_id},{calendar.planting_doy},{calendar.harvest_doy}\n")
        output_files.append(planting_path)

        # Fertilizer data
        fert_path = raster_dir / "fertilizer.csv"
        with open(fert_path, 'w') as f:
            f.write("CellID,FertilizerN\n")
            for cell_id, calendar in crop_calendar.items():
                fert_n = getattr(calendar, 'fertilizer_n', 0)
                f.write(f"{cell_id},{fert_n}\n")
        output_files.append(fert_path)

        logger.info(f"Generated {len(output_files)} PYTHIA management CSVs (fallback)")
        return output_files

    def _get_dssat_executable(self) -> str:
        """Get DSSAT executable path from config.

        IMPORTANT: Default to Windows path since DSSAT is primarily used on Windows.
        Users on other platforms should set dssat_executable explicitly in config.

        Returns:
            Path to DSSAT executable
        """
        # Check if explicitly configured
        pythia_config = None
        if self.config.platform_config and hasattr(self.config.platform_config, 'pythia'):
            pythia_config = self.config.platform_config.pythia

        if pythia_config and pythia_config.dssat_executable:
            return pythia_config.dssat_executable

        # Default to Windows path (most common use case for DSSAT)
        # Users on Mac/Linux should set dssat_executable in their config
        dssat_version = pythia_config.dssat_version if pythia_config else "4.8"
        version_num = dssat_version.replace(".", "")

        # Use standard Windows DSSAT installation path
        return f"C:/DSSAT{version_num}/DSCSM0{version_num}.EXE"

    def _doy_to_calendar_date(self, doy: int, year: int) -> str:
        """Convert day of year to calendar date string (YYYY-MM-DD).

        Args:
            doy: Day of year (1-365/366)
            year: Year for the date

        Returns:
            Date string in YYYY-MM-DD format
        """
        from datetime import timedelta
        dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def _plant_mode_from_sowing(sowing_mode: str) -> str:
        """sowing_mode -> DSSAT SNX PLANT method: opportunistic -> "A" (reads the
        PFRST/PLAST window), fixed_date -> "R" (on PDATE). "F" is non-standard/never
        emitted; unknown raises. Schema normalizes the "fixed" alias to "fixed_date"."""
        mapping = {"opportunistic": "A", "fixed_date": "R"}
        if sowing_mode not in mapping:
            raise ValueError(
                f"Unknown sowing_mode {sowing_mode!r}: expected one of "
                f"{sorted(mapping)} (schema normalizes 'fixed'->'fixed_date')."
            )
        return mapping[sowing_mode]

    def _map_generic_to_pythia_config(self) -> Dict[str, Any]:
        """Map generic config to PYTHIA JSON default_setup parameters.

        Uses config.crop.phenology, config.crop.physiology, and config.management
        if available, otherwise uses defaults from PythiaConfig.

        Returns:
            Dictionary with PYTHIA-compatible parameter values
        """
        # Get PYTHIA-specific config
        pythia_config = None
        if self.config.platform_config and hasattr(self.config.platform_config, 'pythia'):
            pythia_config = self.config.platform_config.pythia

        # Get calendar from config
        calendar = self.config.crop.calendar
        if not calendar:
            raise ValueError("crop.calendar is required - planting_doy and maturity_doy must be specified")

        planting_doy = calendar.planting_doy

        # Get planting window from config (with fallback)
        planting_window = 30  # Default
        if pythia_config and hasattr(pythia_config, 'planting_window_days'):
            planting_window = pythia_config.planting_window_days

        # Get management config
        mgmt = self.config.management
        if mgmt is None:
            mgmt = ManagementConfig(planting_density=62500.0)
            # V2-19 C1 (TP-04): planting density fallback rationale
            if self.provenance:
                self.provenance.record_decision(
                    decision_type=DecisionType.DEFAULT_VALUE,
                    description=(
                        "Planting density fallback: 62,500 plants/ha "
                        "(no user-provided management config)"
                    ),
                    rationale=(
                        "No source in code for the 62,500 plants/ha default. "
                        "Literature range for tropical maize: 25,000 (Sahel "
                        "rainfed smallholder, Traore et al. 2013) to 80,000 "
                        "(irrigated high-input commercial, Sime et al. 2022). "
                        "The default sits at the high end of rainfed practice "
                        "and low end of irrigated practice (~6.25 plants/m²). "
                        "Valid for moderate-input to high-input rainfed maize "
                        "in the Sudan-Savanna zone of West Africa (row spacing "
                        "70-80 cm, intra-row 20-25 cm). NOT valid for "
                        "low-input Sahelian millet/sorghum systems (typically "
                        "10,000-30,000 plants/ha) or irrigated commercial "
                        "maize in southern Africa (>70,000). Users should "
                        "override via config.management.planting_density for "
                        "their specific agro-ecological context."
                    ),
                    alternatives=[
                        "User-provided planting density from config (preferred)",
                        "Region-specific density lookup from literature (V2-20)",
                        "Crop-specific defaults (millet 30k, sorghum 50k, etc.)",
                    ],
                    reference=(
                        "prismpy.translators.pythia.translator "
                        "ManagementConfig(planting_density=62500.0)"
                    ),
                )

        # Get temporal config - REQUIRED
        if self.config.temporal:
            start_year = self.config.temporal.start_year
            end_year = self.config.temporal.end_year
            nyers = end_year - start_year + 1
        else:
            raise ValueError("temporal config is required - start_year and end_year must be specified")

        # Map irrigation setting
        if mgmt.irrigation:
            irrig = "A"  # Automatic irrigation
        else:
            irrig = "N"  # No irrigation (rainfed)

        # Get initial conditions from config (with fallbacks)
        ph2ol = 40.0
        icin = 10.0
        icsw_pct = 25.0
        icren = 0.8

        if pythia_config:
            if hasattr(pythia_config, 'soil_water_threshold'):
                ph2ol = pythia_config.soil_water_threshold
            if hasattr(pythia_config, 'initial_inorganic_n'):
                icin = pythia_config.initial_inorganic_n
            if hasattr(pythia_config, 'initial_soil_water_pct'):
                icsw_pct = pythia_config.initial_soil_water_pct
            if hasattr(pythia_config, 'residue_n_concentration'):
                icren = pythia_config.residue_n_concentration

        # Convert DOY to calendar dates (YYYY-MM-DD format for PYTHIA compatibility)
        pfrst_date = self._doy_to_calendar_date(planting_doy, start_year)
        plast_doy = min(planting_doy + planting_window, 365)
        plast_date = self._doy_to_calendar_date(plast_doy, start_year)

        # sowing_mode -> DSSAT PLANT method (shared helper); PDATE = window start.
        plant_mode = self._plant_mode_from_sowing(getattr(mgmt, "sowing_mode", "opportunistic"))

        return {
            # Temporal settings
            "nyers": nyers,
            "sdate": f"{start_year}-01-01",

            # Planting window (calendar date format: YYYY-MM-DD)
            "pfrst": pfrst_date,
            "plast": plast_date,

            # DSSAT PLANT method + PDATE; consumed by _generate_pythia_json default_setup.
            "plant_mode": plant_mode,
            "pdate": pfrst_date,

            # Soil moisture threshold for planting (%)
            "ph2ol": ph2ol,

            # Initial conditions
            "icin": icin,  # Initial inorganic N (kg/ha)
            "icsw%": icsw_pct,  # Initial soil water (%)
            "icren": icren,  # Residue N concentration

            # Flooding history (standard DSSAT defaults)
            "flhst": "FH302",
            "fhdur": 10,

            # Irrigation
            "irrig": irrig,
        }

    # DSSAT crop model families for auto-detection
    _LEGUME_CROPS = frozenset({
        "cowpea", "soybean", "soya bean", "bean", "beans", "dry bean",
        "chickpea", "groundnut", "peanut", "pigeon pea", "pigeonpea",
        "lentil", "faba bean", "velvet bean", "mung bean",
    })

    # V2-19: default CROPGRO cultivar codes for supported legumes.
    # These are standard DSSAT generic cultivars that exist in the
    # respective crop cultivar files. When a legume has a known
    # default, _map_generic_to_cultivar uses it instead of falling
    # through to CERES-Maize GDD codes (990001-990003).
    _LEGUME_DEFAULT_CULTIVARS: Dict[str, Tuple[str, str]] = {
        "cowpea": ("II0003", "IT90K-277-2"),
        "groundnut": ("IB0001", "FLEUR_11"),
        "peanut": ("IB0001", "FLEUR_11"),
        "soybean": ("IB0001", "MEDIUM_GRO"),
        "soya bean": ("IB0001", "MEDIUM_GRO"),
        "chickpea": ("IB0001", "MEDIUM_GRO"),
        "bean": ("IB0001", "MEDIUM_GRO"),
        "beans": ("IB0001", "MEDIUM_GRO"),
    }

    def _get_pythia_config(self):
        """Get PythiaConfig from platform_config, or None."""
        if self.config.platform_config and hasattr(self.config.platform_config, 'pythia'):
            return self.config.platform_config.pythia
        return None

    def _get_dssat_smodel(self) -> str:
        """Get the DSSAT simulation model code (SMODEL) for the configured crop.

        Resolution order:
        1. Explicit override via PythiaConfig.dssat_smodel (e.g., 'CPGRO')
        2. Auto-detect from crop name: legumes → 'CROPGRO', cereals → '{crop_code}CER'

        Returns:
            6-character DSSAT SMODEL string (e.g., 'MZCER ', 'CROPGRO')
        """
        pythia_cfg = self._get_pythia_config()

        # Priority 1: explicit override
        if pythia_cfg and pythia_cfg.dssat_smodel:
            smodel = pythia_cfg.dssat_smodel.strip()
            logger.info(f"Using explicit DSSAT SMODEL: {smodel}")
            return smodel

        # Priority 2: auto-detect from crop name
        crop_name = self.config.crop.name.lower().strip()
        crop_code = self._get_dssat_crop_code()

        if crop_name in self._LEGUME_CROPS:
            logger.info(
                f"Auto-detected legume crop '{crop_name}' → SMODEL=CROPGRO"
            )
            return "CROPGRO"

        # Default: CERES family ({crop_code}CER)
        smodel = f"{crop_code}CER"
        logger.info(f"Using CERES SMODEL for '{crop_name}': {smodel}")
        return smodel

    def _get_dssat_symbiosis(self) -> str:
        """Get the DSSAT symbiotic N fixation switch for the configured crop.

        Resolution order:
        1. Explicit override via PythiaConfig.dssat_symbiosis
        2. Auto-detect: legumes → 'Y', others → 'N'

        Returns:
            'Y' or 'N'
        """
        pythia_cfg = self._get_pythia_config()

        # Priority 1: explicit override
        if pythia_cfg and pythia_cfg.dssat_symbiosis:
            symbi = pythia_cfg.dssat_symbiosis.strip().upper()
            if symbi in ('Y', 'N'):
                return symbi
            logger.warning(
                f"Invalid dssat_symbiosis value '{symbi}', expected Y or N. "
                f"Falling back to auto-detection."
            )

        # Priority 2: auto-detect from crop name
        crop_name = self.config.crop.name.lower().strip()
        if crop_name in self._LEGUME_CROPS:
            return "Y"
        return "N"

    def _get_default_cultivar_ingeno(self) -> str:
        """Get default cultivar INGENO, checking config override first."""
        pythia_cfg = self._get_pythia_config()
        if pythia_cfg and pythia_cfg.dssat_cultivar_ingeno:
            return pythia_cfg.dssat_cultivar_ingeno
        return "990002"

    def _get_default_cultivar_cname(self) -> str:
        """Get default cultivar CNAME, checking config override first."""
        pythia_cfg = self._get_pythia_config()
        if pythia_cfg and pythia_cfg.dssat_cultivar_cname:
            return pythia_cfg.dssat_cultivar_cname
        return "MEDIUM_SEASON"

    def _map_generic_to_cultivar(self) -> Dict[str, Any]:
        """Map generic phenology to DSSAT cultivar selection.

        Resolution order:
        1. Explicit cultivar override via PythiaConfig.dssat_cultivar_ingeno
           (for CROPGRO crops or when a specific cultivar is needed)
        2. GDD-based maturity class mapping using CERES-Maize generic codes
           (990001/990002/990003) — only appropriate for CERES crops

        GDD thresholds are configurable via PythiaConfig.short_season_gdd_max
        and PythiaConfig.medium_season_gdd_max.

        Returns:
            Dictionary with cultivar parameters
        """
        pythia_config = self._get_pythia_config()

        # Priority 1: Explicit cultivar override (required for CROPGRO crops)
        if pythia_config and pythia_config.dssat_cultivar_ingeno:
            ingeno = pythia_config.dssat_cultivar_ingeno
            cname = pythia_config.dssat_cultivar_cname or "USER_DEFINED"
            logger.info(
                f"Using explicit DSSAT cultivar: INGENO={ingeno}, CNAME={cname}"
            )
            return {
                "ingeno": ingeno,
                "cname": cname,
                "maturity_class": "user_defined",
                "total_gdd": None,
            }

        # Priority 2: CROPGRO legume default cultivar (V2-19 fix for TP-06)
        crop_name = self.config.crop.name.lower().strip()
        if crop_name in self._LEGUME_DEFAULT_CULTIVARS:
            ingeno, cname = self._LEGUME_DEFAULT_CULTIVARS[crop_name]
            logger.info(
                f"Using CROPGRO default cultivar for '{crop_name}': "
                f"INGENO={ingeno}, CNAME={cname}"
            )
            return {
                "ingeno": ingeno,
                "cname": cname,
                "maturity_class": "cropgro_default",
                "total_gdd": None,
            }

        # Priority 3: GDD-based maturity class mapping (CERES crops only)
        pheno = self.config.crop.phenology

        # Use defaults if not provided
        if pheno is None:
            pheno = PhenologyConfig()
            # V2-19 C4 (TA-01): GDD defaults rationale
            if self.provenance:
                self.provenance.record_decision(
                    decision_type=DecisionType.DEFAULT_VALUE,
                    description=(
                        "Crop GDD defaults: emergence=90, veg=500, "
                        "repro=400, grain_fill=700 (total=1690 GDD)"
                    ),
                    rationale=(
                        "No user-provided phenology config. Generic "
                        "PhenologyConfig defaults (base_temp=8\u00b0C, "
                        "total 1690 GDD) approximate a medium-season "
                        "tropical maize (110-120 day). Literature range "
                        "for maize: 1200-2000 GDD total depending on "
                        "maturity group (Kiniry et al. 1991). Default "
                        "is mid-range. Valid for medium-duration improved "
                        "OPV maize in West Africa. NOT valid for "
                        "short-season pearl millet (~1000 GDD), long-"
                        "season sorghum (~1800 GDD), or temperate cereals "
                        "(wheat base_temp=0\u00b0C). Users should set "
                        "config.crop.phenology for their specific cultivar."
                    ),
                    alternatives=[
                        "User-provided phenology from config (preferred)",
                        "Crop-specific GDD lookup table (V2-20)",
                        "DSSAT ECOTYPE/CULTIVAR file defaults",
                    ],
                    reference=(
                        "prismpy.config.schema.PhenologyConfig defaults "
                        "(emergence_gdd=90, vegetative_phase_gdd=500, "
                        "reproductive_phase_gdd=400, grain_filling_gdd=700)"
                    ),
                )

        # Calculate total thermal time requirement
        total_gdd = (
            pheno.emergence_gdd +
            pheno.vegetative_phase_gdd +
            pheno.reproductive_phase_gdd +
            pheno.grain_filling_gdd
        )

        # Warn if using CERES cultivar codes for a non-CERES crop
        crop_name = self.config.crop.name.lower().strip()
        if crop_name in self._LEGUME_CROPS:
            logger.warning(
                f"Crop '{crop_name}' is a legume but no dssat_cultivar_ingeno "
                f"was provided. The GDD-based cultivar codes (990001-990003) "
                f"are CERES-Maize specific and will NOT work with CROPGRO. "
                f"Set platform_config.pythia.dssat_cultivar_ingeno in your config."
            )
            # V2-19 C2 (TP-06): CROPGRO → CERES-Maize silent fallback.
            # Uses crop-modeling-specialist's verbatim rationale per AC11.
            if self.provenance:
                self.provenance.record_decision(
                    decision_type=DecisionType.FALLBACK_SUBSTITUTION,
                    description=(
                        f"CROPGRO→CERES-Maize silent fallback: legume "
                        f"'{crop_name}' assigned CERES cultivar codes "
                        f"(990001-990003)"
                    ),
                    rationale=(
                        "Current behavior is scientifically unacceptable. "
                        "CROPGRO models C3 legumes with symbiotic N\u2082 "
                        "fixation; CERES-Maize models a C4 cereal with no "
                        "N\u2082 fixation. Yields, biomass partitioning, N "
                        "dynamics, and water-use efficiency are not "
                        "comparable. Users who select a CROPGRO crop and "
                        "receive CERES-Maize output should discard those "
                        "results. V2-20 will enforce this via (1) warning "
                        "log before fallback, (2) opt-in strict_mode flag "
                        "that raises instead, (3) user-facing badge in "
                        "the UI."
                    ),
                    alternatives=[
                        "Explicit dssat_cultivar_ingeno override (required for CROPGRO)",
                        "Raise error instead of silent fallback (V2-20 strict_mode)",
                        "CROPGRO-specific cultivar database (not yet available)",
                    ],
                    reference=(
                        "prismpy.translators.pythia.translator."
                        "_map_generic_to_cultivar (legume warning branch)"
                    ),
                    severity="error",
                    label="CROPGRO→CERES-Maize: invalid cultivar codes",
                )

        # Get GDD thresholds from config (with defaults)
        short_gdd_max = 1400.0
        medium_gdd_max = 1700.0

        if pythia_config:
            if hasattr(pythia_config, 'short_season_gdd_max'):
                short_gdd_max = pythia_config.short_season_gdd_max
            if hasattr(pythia_config, 'medium_season_gdd_max'):
                medium_gdd_max = pythia_config.medium_season_gdd_max

        # Map to cultivar based on maturity (thermal time requirement)
        if total_gdd < short_gdd_max:
            # Short season / early maturing
            return {
                "ingeno": "990001",
                "cname": "SHORT_SEASON",
                "maturity_class": "early",
                "total_gdd": total_gdd,
            }
        elif total_gdd < medium_gdd_max:
            # Medium season / intermediate
            return {
                "ingeno": "990002",
                "cname": "MEDIUM_SEASON",
                "maturity_class": "medium",
                "total_gdd": total_gdd,
            }
        else:
            # Long season / late maturing
            return {
                "ingeno": "990003",
                "cname": "LONG_SEASON",
                "maturity_class": "late",
                "total_gdd": total_gdd,
            }

    def _map_generic_to_fertilizer(self) -> Dict[str, Any]:
        """Map generic management to fertilizer configuration.

        Uses ManagementConfig.fertilizer_n_total, fertilizer_n_splits,
        and fertilizer_n_fractions for full control over fertilization.

        Returns:
            Dictionary with fertilizer settings for PYTHIA
        """
        mgmt = self.config.management

        # Default values
        fen_tot = 60.0
        fer_dap = [0, 30]
        fer_pct = [50, 50]

        if mgmt is not None:
            # Get fertilizer total
            if hasattr(mgmt, 'fertilizer_n_total'):
                fen_tot = mgmt.fertilizer_n_total

            # Get application splits (days after planting)
            if hasattr(mgmt, 'fertilizer_n_splits'):
                fer_dap = mgmt.fertilizer_n_splits

            # Get application fractions
            if hasattr(mgmt, 'fertilizer_n_fractions'):
                # Convert fractions to percentages
                fer_pct = [int(f * 100) for f in mgmt.fertilizer_n_fractions]

        return {
            "fen_tot": fen_tot,
            "fer_dap": fer_dap,
            "fer_pct": fer_pct,
        }

    # DSSAT 2-character crop codes (experiment filename convention)
    DSSAT_CROP_CODES = {
        'maize': 'MZ', 'corn': 'MZ',
        'sorghum': 'SG', 'millet': 'ML',
        'rice': 'RI', 'cowpea': 'CP',
        'groundnut': 'PN', 'peanut': 'PN',
        'soybean': 'SB', 'wheat': 'WH',
        'barley': 'BA', 'bean': 'BN',
        'cotton': 'CO', 'sunflower': 'SU',
        'potato': 'PT', 'cassava': 'CS',
    }

    # Per-crop DSSAT @P planting defaults (West African smallholder rainfed) — the fallback used
    # ONLY when a real planting value is not recorded (a value the wizard supplies is threaded
    # instead). ppop = plants/m² (PPOP is DSSAT-native plants/m²); plrs/pldp = cm. Sourced from a
    # crop-modeling review of published West-African DSSAT calibration; keyed on crop.name.lower().
    PLANTING_DEFAULTS = {
        'maize':     {'ppop': 5.3, 'plrs': 75.0, 'pldp': 5.0},
        'corn':      {'ppop': 5.3, 'plrs': 75.0, 'pldp': 5.0},
        'sorghum':   {'ppop': 9.0, 'plrs': 75.0, 'pldp': 3.0},
        'millet':    {'ppop': 3.0, 'plrs': 90.0, 'pldp': 2.0},
        'rice':      {'ppop': 25.0, 'plrs': 20.0, 'pldp': 3.0},
        'cowpea':    {'ppop': 13.0, 'plrs': 75.0, 'pldp': 4.0},
        'groundnut': {'ppop': 15.0, 'plrs': 50.0, 'pldp': 5.0},
        'peanut':    {'ppop': 15.0, 'plrs': 50.0, 'pldp': 5.0},
    }
    # An unmapped crop falls back to the wizard-generic maize density (plants/m²) — never -99.
    PLANTING_DEFAULT_FALLBACK = {'ppop': 6.25, 'plrs': 70.0, 'pldp': 5.0}

    @staticmethod
    def _ascii_fold_for_dssat(value: str) -> str:
        """Strip diacritics and non-ASCII chars from a DSSAT identifier.

        DSSAT v4.8 reads experiment-file names + weather-station codes
        as fixed-width byte strings. Multi-byte UTF-8 (e.g., ``É`` =
        ``0xC3 0x89``, 2 bytes per glyph) shifts the byte boundaries
        and truncates the read at the wrong offset — a project named
        ``"Bénoué"`` produces ``"BÉSG8001.SNX"`` whose 12-char prefix
        gets truncated mid-byte to ``"BÉSG8001.SN"``, and DSSAT logs
        ``WARNING.OUT: File not found: BÉSG8001.SN``.

        The fix is to ASCII-fold the source string at the boundary
        before any slicing or formatting downstream. NFKD decomposes
        the diacritic into base+combining-mark; the ``encode("ASCII",
        "ignore")`` step drops the non-ASCII combining mark; the result
        is single-byte-per-char ASCII that DSSAT's fixed-width byte
        reads handle correctly.

        Apply this ONLY to DSSAT-byte-consumed strings (SNX filename,
        wsta prefix, INSI, batch labels). Display strings on
        ``manifest.region.name`` keep their diacritics — only the
        DSSAT-consumed surface needs ASCII.

        Args:
            value: Source string (may contain non-ASCII characters).

        Returns:
            ASCII-only equivalent. Empty input returns empty string.
        """
        import unicodedata

        if not value:
            return value
        return unicodedata.normalize("NFKD", value).encode(
            "ASCII", "ignore"
        ).decode("ASCII")

    def _get_dssat_crop_code(self) -> str:
        """Get 2-character DSSAT crop code for experiment filenames."""
        crop_lower = self.config.crop.name.lower()
        return self.DSSAT_CROP_CODES.get(crop_lower, self.config.crop.name_short[:2].upper())

    def _crop_planting_default(self) -> Dict[str, float]:
        """The per-crop @P fallback for this run's crop (or the generic fallback)."""
        return self.PLANTING_DEFAULTS.get(
            self.config.crop.name.lower(), self.PLANTING_DEFAULT_FALLBACK)

    def _resolve_planting_params(self) -> Dict[str, float]:
        """The DSSAT @P planting values, converted ONCE — the single home of the plants/ha ->
        plants/m² conversion (no downstream re-division). ``ppop``/``ppoe`` are plants/m² (DSSAT
        PPOP is native plants/m²): ``management.planting_density`` is plants/ha and is divided by
        10000 here. ``plrs`` is ``management.row_spacing_cm`` (cm, direct). ``pldp`` has no PYTHIA
        config source, so it takes the per-crop default. Each falls back to the per-crop default
        when unrecorded — never -99. (A plants/m² ``plant_population`` override is a CRAFT-config
        field handled in the CRAFT translator; the PYTHIA path carries no such override.)"""
        mgmt = self.config.management
        crop_default = self._crop_planting_default()

        density_ha = getattr(mgmt, 'planting_density', None) if mgmt else None
        ppop = (float(density_ha) / 10000.0 if density_ha is not None    # plants/ha -> plants/m²
                else float(crop_default['ppop']))

        row = getattr(mgmt, 'row_spacing_cm', None) if mgmt else None
        plrs = float(row) if row is not None else float(crop_default['plrs'])

        pldp = float(crop_default['pldp'])              # no PYTHIA depth field -> per-crop default

        return {'ppop': ppop, 'ppoe': ppop, 'plrs': plrs, 'pldp': pldp}

    def _get_template_filename(self) -> str:
        """Get the actual template filename based on region and crop.

        DSSAT convention: exactly 8 characters (e.g., KACP8001.SNX).
        ASCII-folded so non-ASCII region names (e.g., ``Bénoué``) do
        not produce multi-byte SNX filenames that DSSAT truncates
        mid-byte (see :meth:`_ascii_fold_for_dssat` for the byte-
        boundary failure mode).

        Returns:
            Template filename
        """
        region_ascii = self._ascii_fold_for_dssat(self.config.region.name)
        region_code = region_ascii[:2].upper()
        crop_code = self._get_dssat_crop_code()
        return f"{region_code}{crop_code}8001.SNX"

    # ISO3 to ISO2 country code mapping - comprehensive global coverage
    # Used for weather station prefixes and soil file lookups
    ISO3_TO_ISO2 = {
        # Africa
        "MLI": "ML", "BEN": "BJ", "NER": "NE", "BFA": "BF",
        "GHA": "GH", "TGO": "TG", "CIV": "CI", "SEN": "SN",
        "NGA": "NG", "CMR": "CM", "ETH": "ET", "KEN": "KE",
        "TZA": "TZ", "UGA": "UG", "ZAF": "ZA", "ZMB": "ZM",
        "ZWE": "ZW", "MOZ": "MZ", "MWI": "MW", "AGO": "AO",
        "DZA": "DZ", "EGY": "EG", "MAR": "MA", "TUN": "TN",
        "SDN": "SD", "SSD": "SS", "COD": "CD", "COG": "CG",
        "RWA": "RW", "BDI": "BI", "SOM": "SO", "ERI": "ER",
        "NAM": "NA", "BWA": "BW", "SWZ": "SZ", "LSO": "LS",
        "MDG": "MG", "MUS": "MU", "GMB": "GM", "GNB": "GW",
        "GIN": "GN", "SLE": "SL", "LBR": "LR", "CPV": "CV",
        "GAB": "GA", "GNQ": "GQ", "CAF": "CF", "TCD": "TD",
        # Asia
        "CHN": "CN", "IND": "IN", "IDN": "ID", "PAK": "PK",
        "BGD": "BD", "JPN": "JP", "PHL": "PH", "VNM": "VN",
        "THA": "TH", "MMR": "MM", "MYS": "MY", "KOR": "KR",
        "PRK": "KP", "NPL": "NP", "LKA": "LK", "KHM": "KH",
        "LAO": "LA", "AFG": "AF", "IRN": "IR", "IRQ": "IQ",
        "SAU": "SA", "YEM": "YE", "OMN": "OM", "ARE": "AE",
        "KAZ": "KZ", "UZB": "UZ", "TKM": "TM", "TJK": "TJ",
        "KGZ": "KG", "MNG": "MN", "TWN": "TW",
        # Americas
        "USA": "US", "CAN": "CA", "MEX": "MX", "BRA": "BR",
        "ARG": "AR", "COL": "CO", "PER": "PE", "VEN": "VE",
        "CHL": "CL", "ECU": "EC", "BOL": "BO", "PRY": "PY",
        "URY": "UY", "GUY": "GY", "SUR": "SR", "GTM": "GT",
        "HND": "HN", "NIC": "NI", "SLV": "SV", "CRI": "CR",
        "PAN": "PA", "CUB": "CU", "DOM": "DO", "HTI": "HT",
        "JAM": "JM",
        # Europe
        "DEU": "DE", "FRA": "FR", "GBR": "GB", "ITA": "IT",
        "ESP": "ES", "POL": "PL", "ROU": "RO", "NLD": "NL",
        "BEL": "BE", "GRC": "GR", "CZE": "CZ", "PRT": "PT",
        "HUN": "HU", "SWE": "SE", "AUT": "AT", "BGR": "BG",
        "DNK": "DK", "FIN": "FI", "SVK": "SK", "NOR": "NO",
        "IRL": "IE", "HRV": "HR", "LTU": "LT", "SVN": "SI",
        "LVA": "LV", "EST": "EE", "UKR": "UA", "BLR": "BY",
        "SRB": "RS", "CHE": "CH", "RUS": "RU", "TUR": "TR",
        # Oceania
        "AUS": "AU", "NZL": "NZ", "PNG": "PG", "FJI": "FJ",
    }

    def _iso3_to_iso2(self, iso3: str) -> str:
        """Convert ISO3 country code to ISO2.

        Args:
            iso3: 3-letter ISO country code (e.g., "MLI", "USA")

        Returns:
            2-letter ISO country code (e.g., "ML", "US")
        """
        if not iso3:
            return "XX"
        return self.ISO3_TO_ISO2.get(iso3, iso3[:2])

    def _get_wsta_prefix(self) -> str:
        """Get the weather station prefix for PYTHIA lookup.

        PYTHIA's lookup_wth function uses this prefix for output symlink naming.
        Format: 4-character code like "MLKO" (country + region). The region
        portion is ASCII-folded so non-ASCII region names (e.g., ``Bénoué``)
        produce single-byte-per-char prefixes; DSSAT consumes these as
        fixed-width byte strings and truncates multi-byte glyphs mid-byte
        otherwise (see :meth:`_ascii_fold_for_dssat`).

        Returns:
            4-character weather station prefix (e.g., MLKO for Mali Koutiala,
            CMBE for Cameroon Bénoué).
        """
        country_iso3 = self.config.region.country_iso3 or ""
        region_name = self.config.region.name or ""

        country_code = self._iso3_to_iso2(country_iso3)

        # Build 4-char prefix: 2 chars country + 2 chars region.
        # ASCII-fold the region BEFORE slicing so non-ASCII glyphs do not
        # land in the DSSAT-byte-consumed prefix.
        region_ascii = self._ascii_fold_for_dssat(region_name)
        region_code = region_ascii[:2].upper() if region_ascii else "XX"
        prefix = f"{country_code}{region_code}"[:4]

        return prefix

    def _generate_pythia_json(self, data: UnifiedData) -> Path:
        """Generate PYTHIA JSON configuration file.

        The JSON uses PYTHIA-specific function syntax:
        - xy_from_vector::path/to/shapefile
        - lookup_wth::SITE::vector::path::id_field
        - lookup_ghr::raster::path/to/soil_raster
        - raster::path/to/raster

        Uses generic config mapping when available (phenology, physiology, management).
        All paths are relative to package root (./) for portability.

        Args:
            data: UnifiedData with all configuration info

        Returns:
            Path to generated pythia_config.json
        """
        # Build RELATIVE file paths for portability
        # All paths start with ./ so package can be moved anywhere
        sites_file = "./shapes/sites.shp"
        weather_dir = "./weather/"
        soil_raster = "./raster/soil.tif"
        harvest_area = "./raster/harvest_area.tif"

        # Get parameters from generic config mapping
        use_generic_mapping = (
            self.config.crop.phenology is not None or
            self.config.crop.physiology is not None or
            self.config.management is not None
        )

        if use_generic_mapping:
            logger.info("Using generic config mapping for PYTHIA JSON")
            config_params = self._map_generic_to_pythia_config()
            cultivar_params = self._map_generic_to_cultivar()
            fertilizer_params = self._map_generic_to_fertilizer()

            nyers = config_params["nyers"]
            sdate = config_params["sdate"]
            pfrst = config_params["pfrst"]
            plast = config_params["plast"]
            irrig = config_params["irrig"]
            plant_mode = config_params["plant_mode"]
            pdate = config_params["pdate"]
            fen_tot = fertilizer_params["fen_tot"]
            ingeno = cultivar_params["ingeno"]
            cname = cultivar_params["cname"]

            logger.info(f"  Cultivar: {ingeno} ({cname}) - {cultivar_params['total_gdd']} GDD")
            logger.info(f"  Planting: {pfrst} to {plast}")
            logger.info(f"  Fertilizer: {fen_tot} kg N/ha")
            logger.info(f"  Irrigation: {irrig}")
        else:
            # Fallback - still use config values where available
            logger.info("No generic phenology/physiology config - using crop calendar and defaults")

            # Get date range - REQUIRED
            if self.config.temporal:
                start_year = self.config.temporal.start_year
                end_year = self.config.temporal.end_year
                nyers = end_year - start_year + 1
                sdate = f"{start_year}-01-01"
            else:
                raise ValueError(
                    "temporal config is required - please specify start_year and end_year in your config file"
                )

            # Get crop calendar - REQUIRED
            if self.config.crop.calendar:
                planting_doy = self.config.crop.calendar.planting_doy
                # Get planting window from pythia config
                pythia_cfg = self.config.platform_config.pythia if self.config.platform_config else None
                planting_window = 30
                if pythia_cfg and hasattr(pythia_cfg, 'planting_window_days'):
                    planting_window = pythia_cfg.planting_window_days
                # Convert DOY to calendar dates (YYYY-MM-DD format)
                pfrst = self._doy_to_calendar_date(planting_doy, start_year)
                plast_doy = min(planting_doy + planting_window, 365)
                plast = self._doy_to_calendar_date(plast_doy, start_year)
            else:
                raise ValueError(
                    "crop.calendar is required - please specify planting_doy and maturity_doy in your config file"
                )

            # Get management settings
            mgmt = self.config.management
            irrig = "A" if mgmt and mgmt.irrigation else "N"
            plant_mode = self._plant_mode_from_sowing(getattr(mgmt, "sowing_mode", "opportunistic"))
            pdate = pfrst
            fen_tot = mgmt.fertilizer_n_total if mgmt and hasattr(mgmt, 'fertilizer_n_total') else 60

            # Use cultivar from config override or default mapping
            pythia_cfg = self._get_pythia_config()
            if pythia_cfg and pythia_cfg.dssat_cultivar_ingeno:
                ingeno = pythia_cfg.dssat_cultivar_ingeno
                cname = pythia_cfg.dssat_cultivar_cname or "USER_DEFINED"
            else:
                ingeno = "990002"
                cname = "MEDIUM_SEASON"

        # Get start year for runs
        start_year = int(sdate.split("-")[0])

        # Get threading settings from config
        pythia_config = None
        if self.config.platform_config and hasattr(self.config.platform_config, 'pythia'):
            pythia_config = self.config.platform_config.pythia

        threads = pythia_config.threads if pythia_config and hasattr(pythia_config, 'threads') else 2
        cores = pythia_config.cores if pythia_config and hasattr(pythia_config, 'cores') else 4

        # DSSAT-native planting values the runner reads directly (unit-aware; the ha->m² conversion
        # lives once in the resolver). Absent-edge falls back to the per-crop default, never -99.
        planting = self._resolve_planting_params()

        # Build PYTHIA JSON structure
        # All paths are relative (./) for portability - package can be moved anywhere
        pythia_json = {
            "name": self.config.project.name,
            "workDir": "./",
            "templateDir": "./templates/",
            "weatherDir": weather_dir,
            "threads": threads,
            "cores": cores,
            "ghr_root": "./eGHR/",  # eGHR data included in package

            "default_setup": {
                "template": self._get_template_filename(),
                "sites": f"xy_from_vector::{sites_file}",
                "nyers": nyers,
                "sdate": sdate,
                "pfrst": pfrst,
                "plast": plast,
                "ph2ol": 40,
                "icin": 10,
                "icsw%": 25,
                "icren": 0.8,
                "flhst": "FH302",
                "fhdur": 10,
                "id_soil": f"lookup_ghr::raster::{soil_raster}",
                "wsta": f"lookup_wth::{self._get_wsta_prefix()}::vector::{sites_file}::ID",
                "ic_layers": "generate_ic_layers::$id_soil",
                "irrig": irrig,
                "plant_mode": plant_mode,
                "pdate": pdate,
                "ppop": planting['ppop'],
                "ppoe": planting['ppoe'],
                "plrs": planting['plrs'],
                "pldp": planting['pldp'],
                "ingeno": ingeno,
                "cname": cname,
            },

            "dssat": {
                "executable": self._get_dssat_executable()
            },

            "analytics_setup": {
                "per_pixel_prefix": "output",
                "singleOutput": True
            },

            "runs": []
        }

        # Add runs/scenarios
        crop_name = self.config.crop.name

        # Baseline run (no fertilizer)
        run_baseline = {
            "name": f"{crop_name}_baseline",
            "harvestArea": f"raster::{harvest_area}",
            "startYear": start_year,
            "fen_tot": 0,
            "irrig": irrig,
            "plant_mode": plant_mode,
            "pdate": pdate,
            "ingeno": ingeno,
            "cname": f"{cname}_BASELINE",
        }
        pythia_json["runs"].append(run_baseline)

        # Fertilized scenario (using mapped fen_tot)
        run_fert = {
            "name": f"{crop_name}_fertilized",
            "harvestArea": f"raster::{harvest_area}",
            "startYear": start_year,
            "fen_tot": fen_tot,
            "irrig": irrig,
            "plant_mode": plant_mode,
            "pdate": pdate,
            "ingeno": ingeno,
            "cname": f"{cname}_FERTILIZED",
        }
        pythia_json["runs"].append(run_fert)

        # Write JSON file
        config_path = self.output_dir / "config" / "pythia_config.json"
        with open(config_path, 'w') as f:
            json.dump(pythia_json, f, indent='\t')

        logger.info(f"Generated PYTHIA JSON config: {config_path}")
        return config_path

    # =========================================================================
    # Raster Clipping Methods
    # =========================================================================

    def _clip_global_raster(
        self,
        input_path: Path,
        output_path: Path,
        bounds: Tuple[float, float, float, float],
        buffer: float = 0.1,
    ) -> Path:
        """Clip a global raster to region bounds.

        Reuses the pattern from eGHR source.

        Args:
            input_path: Path to input global raster
            output_path: Path to save clipped raster
            bounds: Tuple of (minx, miny, maxx, maxy)
            buffer: Buffer to add to bounds in degrees

        Returns:
            Path to clipped raster
        """
        import rasterio
        from rasterio.windows import from_bounds

        minx, miny, maxx, maxy = bounds
        minx -= buffer
        miny -= buffer
        maxx += buffer
        maxy += buffer

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(input_path) as src:
            # Create window from bounds
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            window = window.round_offsets().round_lengths()

            # Read data in window
            data = src.read(1, window=window)
            transform = src.window_transform(window)

            # Prepare output metadata
            meta = src.meta.copy()
            meta.update({
                'height': data.shape[0],
                'width': data.shape[1],
                'transform': transform,
            })

            # Write clipped raster
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(data, 1)

        logger.info(f"Clipped raster to {output_path} ({data.shape[1]}x{data.shape[0]} pixels)")
        return output_path

    def _generate_soil_raster(self, data: UnifiedData) -> Optional[Path]:
        """Clip eGHR soil raster to region bounds.

        Args:
            data: UnifiedData with grid info

        Returns:
            Path to soil raster or None if not configured
        """
        # Get eGHR raster path from config
        pythia_config = None
        if self.config.platform_config and hasattr(self.config.platform_config, 'pythia'):
            pythia_config = self.config.platform_config.pythia

        if not pythia_config or not pythia_config.eghr_raster_path:
            logger.warning("eGHR raster path not configured, skipping soil raster generation")
            return None

        # Resolve path relative to project root
        eghr_path = Path(pythia_config.eghr_raster_path)
        if not eghr_path.is_absolute():
            # Resolve relative to current working directory (matches CRAFT behavior)
            eghr_path = Path.cwd() / eghr_path

        if not eghr_path.exists():
            logger.error(f"eGHR raster not found: {eghr_path}")
            return None

        # Calculate bounds from grid
        if data.grid and data.grid.cells:
            lats = [cell.lat for cell in data.grid.cells]
            lons = [cell.lon for cell in data.grid.cells]
            bounds = (min(lons), min(lats), max(lons), max(lats))
        else:
            logger.warning("No grid data, using region bounds")
            if hasattr(data.region.bounds, 'to_gis_format'):
                bounds = tuple(data.region.bounds.to_gis_format())
            else:
                # Fallback to manual bounds from config
                mb = self.config.region.boundary.manual_bounds
                if mb:
                    bounds = (mb.minx, mb.miny, mb.maxx, mb.maxy)
                else:
                    logger.error("Cannot determine region bounds")
                    return None

        # Output path
        output_path = self.output_dir / "raster" / "soil.tif"

        try:
            return self._clip_global_raster(eghr_path, output_path, bounds)
        except Exception as e:
            logger.error(f"Failed to clip soil raster: {e}")
            return None

    def _generate_crop_mask_raster(self, data: UnifiedData) -> Optional[Path]:
        """Clip SPAM 2020 harvest area raster to region bounds.

        Args:
            data: UnifiedData with grid info

        Returns:
            Path to crop mask raster or None if not configured
        """
        # Get SPAM raster directory from config
        pythia_config = None
        if self.config.platform_config and hasattr(self.config.platform_config, 'pythia'):
            pythia_config = self.config.platform_config.pythia

        if not pythia_config or not pythia_config.spam_raster_dir:
            logger.warning("SPAM raster directory not configured, skipping crop mask generation")
            return None

        # Get SPAM crop code from crop name
        crop_code = self._get_spam_crop_code()
        spam_version = pythia_config.spam_version or "2020"

        # Build SPAM raster filename
        if spam_version == "2020":
            spam_filename = f"spam2020_V2r0_global_H_{crop_code}_A.tif"
        else:
            spam_filename = f"spam2010V2r0_global_H_{crop_code}_A.tif"

        # Resolve path
        spam_dir = Path(pythia_config.spam_raster_dir)
        if not spam_dir.is_absolute():
            spam_dir = Path.cwd() / spam_dir

        spam_path = spam_dir / spam_filename

        if not spam_path.exists():
            # Try simplified naming convention (e.g., spam2020_cowpea.tif)
            crop_lower = self.config.crop.name.lower()
            alt_filename = f"spam{spam_version}_{crop_lower}.tif"
            alt_path = spam_dir / alt_filename
            if alt_path.exists():
                spam_path = alt_path
                logger.info(f"Using simplified SPAM filename: {alt_filename}")
            else:
                logger.error(f"SPAM raster not found: {spam_path} or {alt_path}")
                return None

        # Calculate bounds from grid
        if data.grid and data.grid.cells:
            lats = [cell.lat for cell in data.grid.cells]
            lons = [cell.lon for cell in data.grid.cells]
            bounds = (min(lons), min(lats), max(lons), max(lats))
        else:
            logger.warning("No grid data, using region bounds")
            if hasattr(data.region.bounds, 'to_gis_format'):
                bounds = tuple(data.region.bounds.to_gis_format())
            else:
                mb = self.config.region.boundary.manual_bounds
                if mb:
                    bounds = (mb.minx, mb.miny, mb.maxx, mb.maxy)
                else:
                    logger.error("Cannot determine region bounds")
                    return None

        # Output path
        output_path = self.output_dir / "raster" / "harvest_area.tif"

        try:
            return self._clip_global_raster(spam_path, output_path, bounds)
        except Exception as e:
            logger.error(f"Failed to clip SPAM raster: {e}")
            return None

    def _get_spam_crop_code(self) -> str:
        """Get SPAM crop code from config or auto-detect from crop name.

        Priority:
        1. Explicit spam_crop_code in PythiaConfig
        2. Auto-detect from crop name using lookup table
        3. Warning if unknown crop (falls back to crop name uppercase)

        Returns:
            4-letter SPAM crop code (e.g., MAIZ, RICE)
        """
        # Check for explicit override in config
        pythia_config = None
        if self.config.platform_config and hasattr(self.config.platform_config, 'pythia'):
            pythia_config = self.config.platform_config.pythia

        if pythia_config and pythia_config.spam_crop_code:
            return pythia_config.spam_crop_code

        # Auto-detect from crop name
        crop_name = self.config.crop.name.lower()

        # Mapping of crop names to SPAM codes
        crop_map = {
            "maize": "MAIZ",
            "corn": "MAIZ",
            "rice": "RICE",
            "wheat": "WHEA",
            "sorghum": "SORG",
            "millet": "PMIL",
            "pearl millet": "PMIL",
            "barley": "BARL",
            "cassava": "CASS",
            "groundnut": "GROU",
            "peanut": "GROU",
            "soybean": "SOYB",
            "soy": "SOYB",
            "cotton": "COTT",
            "sugarcane": "SUGC",
            "sugar cane": "SUGC",
            "potato": "POTA",
            "sweet potato": "SWPO",
            "bean": "BEAN",
            "beans": "BEAN",
            "chickpea": "CHIC",
            "cowpea": "COWP",
            "lentil": "LENT",
            "pigeon pea": "PIGE",
            "sunflower": "SUNF",
            "rapeseed": "RAPE",
            "canola": "RAPE",
        }

        spam_code = crop_map.get(crop_name)

        if spam_code is None:
            # Warn user and try to construct a code
            logger.warning(
                f"Unknown crop '{crop_name}' for SPAM lookup. "
                f"Consider setting pythia.spam_crop_code in config. "
                f"Known crops: {', '.join(sorted(set(crop_map.values())))}"
            )
            # Try to use the crop short name as fallback
            spam_code = self.config.crop.name_short.upper()[:4]
            logger.warning(f"Using '{spam_code}' as SPAM code fallback")

        return spam_code

    # =========================================================================
    # eGHR Data Inclusion
    # =========================================================================

    def _get_required_country_codes(self) -> set:
        """Resolve the country codes covered by the per-package eGHR substrate.

        The package's ``raster/soil.tif`` carries an integer pixel id at
        each cell; the package's ``eGHR/GHR.db`` maps each pixel id to a
        ten-character profile name whose first two characters are the
        ISO 3166-1 alpha-2 country code. This helper reads both files
        from the local per-package paths the substrate builder produces
        (or that the legacy copy path lays down on top of a bundled
        eGHR archive) and returns the set of country codes the package
        actually covers.

        On the happy path — a per-package eGHR substrate is present at
        ``output_dir/raster/soil.tif`` and ``output_dir/eGHR/GHR.db`` —
        no fallback fires and the result is the exact country roster
        the substrate registered. The fallback branch only applies to
        edge cases where the substrate has not been written yet (early
        partial runs, tests that exercise this helper without a built
        substrate, or future code paths that defer the substrate build).

        Returns:
            Set of 2-letter country codes covered by the package.
        """
        soil_raster = self.output_dir / "raster" / "soil.tif"
        db_path = self.output_dir / "eGHR" / "GHR.db"

        if soil_raster.exists() and db_path.exists():
            return self._enumerate_countries_from_local_substrate(
                soil_raster=soil_raster,
                db_path=db_path,
            )

        # Substrate not yet present: derive from the region's ISO3 code.
        # The happy path never reaches this branch once the substrate
        # builder has run; this is the partial-run / early-test edge.
        logger.warning(
            "Per-package eGHR substrate missing (raster=%s db=%s); "
            "falling back to region country code.",
            soil_raster.exists(),
            db_path.exists(),
        )
        country_codes: set = set()
        iso3 = self.config.region.country_iso3
        if iso3:
            country_codes.add(self._iso3_to_iso2(iso3))
        return country_codes

    def _enumerate_countries_from_local_substrate(
        self,
        soil_raster: Path,
        db_path: Path,
    ) -> set:
        """Enumerate country codes from the per-package substrate triple.

        Reads unique non-nodata pixel ids from ``soil_raster``, queries
        ``profile_map.profile`` for the matching rows in ``db_path``,
        and extracts the 2-letter country prefix from each profile
        name. Idempotent and side-effect free.
        """
        import sqlite3

        import numpy as np
        import rasterio

        country_codes: set = set()

        with rasterio.open(soil_raster) as src:
            data = src.read(1)
            unique_pixels = np.unique(data[data > 0])

        if len(unique_pixels) == 0:
            logger.warning(
                "Per-package eGHR substrate has no non-nodata pixel ids in "
                "soil.tif; package covers no cells."
            )
            return country_codes

        logger.info(
            "Found %d unique soil pixels in modeling area",
            len(unique_pixels),
        )

        try:
            conn = sqlite3.connect(str(db_path))
            try:
                cursor = conn.cursor()
                batch_size = 1000
                for start in range(0, len(unique_pixels), batch_size):
                    batch = unique_pixels[start : start + batch_size]
                    placeholders = ",".join("?" * len(batch))
                    cursor.execute(
                        "SELECT DISTINCT profile FROM profile_map "
                        f"WHERE id IN ({placeholders})",
                        batch.tolist(),
                    )
                    for row in cursor.fetchall():
                        profile = row[0]
                        if profile and len(profile) >= 2:
                            country_codes.add(profile[:2].upper())
            finally:
                conn.close()
        except sqlite3.Error as exc:
            # The local GHR.db is malformed: log and fall back to region.
            logger.warning(
                "Failed to read country codes from per-package GHR.db at %s: %s",
                db_path,
                exc,
            )
            iso3 = self.config.region.country_iso3
            if iso3:
                country_codes.add(self._iso3_to_iso2(iso3))
            return country_codes

        logger.info(
            "Countries covered by modeling area: %s",
            sorted(country_codes),
        )
        return country_codes

    def _include_eghr_data(
        self,
        data: Optional[UnifiedData] = None,
    ) -> Optional[Path]:
        """Stage the per-package eGHR substrate triple in ``output_dir``.

        Dispatches on :attr:`prefer_canonical_substrate` between two
        paths that produce the same on-disk artifacts:

        - **canonical substrate** (default; ``prefer_canonical_substrate=True``):
          synthesize the triple via :func:`build_eghr_substrate` from
          the per-cell soil profiles already resolved by the upstream
          pipeline (``data.soil``) and the project grid (``data.grid``).
          A substrate-build failure raises :class:`BuildEghrSubstrateError`
          loud rather than silently dropping back to the bundled
          global database.
        - **legacy bundled-file** (``prefer_canonical_substrate=False``):
          clip ``pythia.eghr_raster_path`` to the project bbox, copy
          the bundled ``pythia.eghr_database_path`` into the package,
          and copy only the country-specific ``.SOL`` files matched
          by ``_get_required_country_codes``. Preserved for projects
          whose ingestion has not yet adopted per-cell soil resolution.

        Args:
            data: Optional :class:`UnifiedData` carrying the grid and
                per-cell soil profiles needed for the canonical path.
                Required when ``prefer_canonical_substrate=True``.

        Returns:
            Path to the package's ``eGHR/`` directory if at least one
            artifact was written, otherwise ``None``.
        """
        # Sprint S Gate-B-FIX — record the dispatch decision in
        # provenance.json so downstream consumers (Dr. Kofi grep, the
        # AC-8 reproduction snippet, the evaluator's Gate B verifier)
        # have an unambiguous binary signal for which substrate path
        # ran. Per durable §24 canonical-source-or-pin: the field is
        # the source of truth for "did canonical fire on this run"
        # and replaces the inferring-from-side-effects detective work
        # the evaluator did on b5fb6538-3f98-491d-8d09-d5be4d35074b.
        import os as _os
        env_disabled = _os.environ.get("PRISMPY_DISABLE_CANONICAL_EGHR") == "1"

        if not self.prefer_canonical_substrate:
            self._record_substrate_decision(
                "legacy_bundled", "disabled_via_flag",
                "Operator constructed PythiaTranslator with "
                "prefer_canonical_substrate=False; legacy bundled-file "
                "flow runs by explicit opt-out.",
            )
            return self._include_eghr_data_legacy()
        if env_disabled:
            self._record_substrate_decision(
                "legacy_bundled", "disabled_via_env",
                "PRISMPY_DISABLE_CANONICAL_EGHR=1 set in the runtime "
                "environment; legacy bundled-file flow runs by operator "
                "escape hatch.",
            )
            return self._include_eghr_data_legacy()
        if not self._canonical_substrate_will_run(data):
            grid_present = bool(data and data.grid)
            soil_dict_nonempty = bool(
                data and isinstance(data.soil, dict) and data.soil
            )
            country_iso3 = self.config.region.country_iso3
            reason_text = (
                f"Canonical eGHR substrate inputs unavailable "
                f"(grid={'yes' if grid_present else 'no'}, "
                f"soil_dict_nonempty={'yes' if soil_dict_nonempty else 'no'}, "
                f"country_iso3={country_iso3!r}); fell back to legacy "
                f"bundled-file flow."
            )
            logger.info(reason_text)
            self._record_substrate_decision(
                "legacy_bundled", "inputs_unavailable", reason_text,
            )
            return self._include_eghr_data_legacy()
        self._record_substrate_decision(
            "canonical", "ok",
            "Per-package eGHR substrate synthesized via "
            "prismpy.translators._shared.build_eghr_substrate from "
            "upstream-resolved per-cell soil profiles.",
        )
        return self._include_eghr_data_canonical(data)

    def _record_substrate_decision(
        self,
        decision: str,
        reason: str,
        rationale: str,
    ) -> None:
        """Record the eGHR substrate dispatch decision in provenance.

        Sprint S Gate-B-FIX (durable §24 canonical-source-or-pin):
        downstream consumers — AC-8 reproduction snippet, evaluator
        Gate B verifier, Dr. Kofi grep, Sprint S regression net —
        read ``provenance.json["eghr_substrate_decision"]`` for the
        unambiguous binary signal of which substrate path ran. The
        absence of this key on a delivered package is itself a
        signal that the package was built with pre-Sprint-S code.

        Args:
            decision: ``"canonical"`` or ``"legacy_bundled"``.
            reason: Short machine-readable reason code; one of
                ``"ok"``, ``"disabled_via_flag"``, ``"disabled_via_env"``,
                or ``"inputs_unavailable"``.
            rationale: Human-readable rationale for Dr. Kofi /
                evaluator audit trail; should explain WHY the
                dispatch chose this branch.
        """
        if not self.provenance:
            return
        # Surface the dispatch decision via the dedicated tracker
        # setter (see :meth:`prismpy.provenance.tracker.ProvenanceTracker.set_eghr_substrate_decision`).
        # The setter writes to dedicated top-level fields on the
        # ProvenanceRecord so the AC-8 reproduction snippet + the
        # evaluator's Gate B verifier read ``provenance.json
        # ["eghr_substrate_decision"]`` directly without inferring
        # from secondary signals (durable §24 canonical-source-or-pin).
        self.provenance.set_eghr_substrate_decision(
            decision=decision,
            reason=reason,
        )
        # Also record a DecisionRecord (implicit pending-list path,
        # matching the existing translator pattern at line 322+) for
        # the human-readable audit trail (rationale + alternatives
        # + reference for Dr. Kofi's "why was this chosen?" review).
        # The DecisionRecord is auxiliary; the top-level field above
        # is the load-bearing source-of-truth that consumers grep.
        # The pending decision flushes when the next transformation
        # is recorded; in PYTHIA's translate() flow the FORMAT_CHOICE
        # decision at line 322 flushes the pending queue.
        self.provenance.record_decision(
            decision_type=DecisionType.FORMAT_CHOICE,
            description=(
                f"PYTHIA eGHR substrate dispatch: {decision} "
                f"(reason={reason})"
            ),
            rationale=rationale,
            alternatives=(
                ["legacy_bundled"]
                if decision == "canonical"
                else ["canonical"]
            ),
            reference=(
                "prismpy.translators.pythia.translator."
                "_include_eghr_data dispatch (Sprint S Gate-B-FIX)"
            ),
        )

    def _canonical_substrate_will_run(
        self,
        data: Optional[UnifiedData],
    ) -> bool:
        """Return True if the canonical substrate builder is the dispatch target.

        The dispatch only chooses the canonical path when:

        1. The ``PRISMPY_DISABLE_CANONICAL_EGHR`` environment variable
           is NOT set to ``"1"``. Operators set this escape hatch
           when running on legacy bundled-eGHR assets — the env-var
           takes precedence over the constructor parameter so a
           process can opt out of canonical mode without rebuilding
           the translator (e.g., from a CI harness that wraps the
           prismpy pipeline). When the var is set, the dispatcher
           emits a WARNING per the project's no-data-cooking
           contract so the legacy path is never chosen silently.
        2. The operator has not forced legacy mode via
           ``prefer_canonical_substrate=False``.
        3. Every canonical-path input is present: ``data.grid``,
           a non-empty per-cell ``data.soil`` dict, and
           ``self.config.region.country_iso3``.

        When any condition is unmet, the dispatcher routes to the
        legacy bundled flow. Operators who want loud-fail instead
        can pass ``prefer_canonical_substrate=False`` to force
        legacy mode (explicit opt-out via the constructor) or set
        ``PRISMPY_DISABLE_CANONICAL_EGHR=1`` (operator escape hatch
        from the runtime environment).
        """
        import os

        if os.environ.get("PRISMPY_DISABLE_CANONICAL_EGHR") == "1":
            logger.warning(
                "Canonical eGHR substrate disabled via "
                "PRISMPY_DISABLE_CANONICAL_EGHR=1; using legacy "
                "global GHR.db path."
            )
            return False
        if not self.prefer_canonical_substrate:
            return False
        if data is None or data.grid is None:
            return False
        if not isinstance(data.soil, dict) or not data.soil:
            return False
        if not self.config.region.country_iso3:
            return False
        return True

    def _include_eghr_data_canonical(
        self,
        data: Optional[UnifiedData],
    ) -> Optional[Path]:
        """Synthesize the per-package eGHR substrate via the shared builder.

        Caller (``_include_eghr_data``) guarantees the inputs are
        complete via :meth:`_canonical_substrate_will_run`; this
        method only raises :class:`BuildEghrSubstrateError` when the
        underlying writers themselves fail (rasterio, sqlite3, or
        filesystem errors during artifact generation). Input-shape
        problems are absorbed by the dispatcher's fallback to legacy
        mode, not raised here.
        """
        # The dispatcher guarantees these are populated; assert defensively
        # so a future caller that bypasses the dispatcher fails loud rather
        # than racing on a None dereference.
        assert data is not None and data.grid is not None
        assert isinstance(data.soil, dict) and data.soil
        country_iso3 = self.config.region.country_iso3
        assert country_iso3
        country_code = self._iso3_to_iso2(country_iso3)

        region = data.region if data.region is not None else self._region_from_config()

        # Sprint E.3 fixup +15 (F-BN Boundary 3) — thread the cockpit
        # override sidecar through to the substrate builder so
        # per-cell soil overrides synthesize fresh profiles before
        # the canonical .SOL writer fires. ``None`` (the universal
        # no-override case) preserves the pre-fixup byte-equivalent
        # output. Per-cell synthetic profile generation per
        # ``_apply_soil_overrides_to_assignment`` preserves the
        # honest-signal floor — non-overridden cells sharing the
        # base profile stay unaffected.
        cockpit_sidecar = getattr(self, "cockpit_override_sidecar", None)

        try:
            result = build_eghr_substrate(
                grid=data.grid,
                profiles_by_cell=data.soil,
                country_code=country_code,
                region=region,
                output_dir=self.output_dir,
                cockpit_override_sidecar=cockpit_sidecar,
            )
        except Exception as exc:  # raised by writers / rasterio / sqlite3
            raise BuildEghrSubstrateError(
                f"build_eghr_substrate failed: {exc}"
            ) from exc

        logger.info(
            "Canonical eGHR substrate built: %d cells -> %d unique profiles "
            "(soil_raster=%s ghr_db=%s sol=%s)",
            result.cell_count,
            result.profile_count,
            result.soil_raster_path,
            result.ghr_db_path,
            result.sol_path,
        )
        return result.ghr_db_path.parent

    def _region_from_config(self) -> Region:
        """Build a :class:`Region` from ``self.config`` when ``data.region`` is unset."""
        from prismpy.models.region import BoundingBox

        cfg_region = self.config.region
        manual = cfg_region.boundary.manual_bounds if cfg_region.boundary else None
        if manual is not None:
            bounds = BoundingBox(
                minx=manual.minx,
                miny=manual.miny,
                maxx=manual.maxx,
                maxy=manual.maxy,
            )
        else:
            bounds = BoundingBox(minx=0.0, miny=0.0, maxx=0.0, maxy=0.0)
        return Region(
            name=cfg_region.name,
            country=cfg_region.country,
            country_iso3=cfg_region.country_iso3 or "XXX",
            bounds=bounds,
        )

    def _include_eghr_data_legacy(self) -> Optional[Path]:
        """Legacy bundled-file flow.

        Copies the bundled global GHR.db into the package and copies
        country-specific ``.SOL`` files matched by
        ``_get_required_country_codes``. Preserved for projects whose
        ingestion has not yet adopted per-cell soil resolution; the
        canonical substrate path is the default for new projects.
        """
        import shutil

        # Get pythia config
        pythia_config = None
        if self.config.platform_config and hasattr(self.config.platform_config, 'pythia'):
            pythia_config = self.config.platform_config.pythia

        if not pythia_config:
            logger.warning("No PYTHIA config available, skipping eGHR data inclusion")
            return None

        # Create eGHR directory and stage the GHR.db FIRST so the
        # country-code resolver below can read it from the local
        # per-package path. The resolver no longer consults the
        # bundled global eghr_database_path directly; it only reads
        # output_dir/eGHR/GHR.db. Copying the source database (or,
        # in a future revision, building it via build_eghr_substrate)
        # before resolving the country codes keeps cross-border
        # bounding boxes from falling back to a single-country
        # region default.
        eghr_output = self.output_dir / "eGHR"
        eghr_output.mkdir(parents=True, exist_ok=True)

        files_copied = 0

        # Copy GHR.db
        if pythia_config.eghr_database_path:
            src_db = Path(pythia_config.eghr_database_path)
            # Resolve relative path
            if not src_db.is_absolute():
                src_db = Path.cwd() / src_db

            if src_db.exists():
                dst_db = eghr_output / "GHR.db"
                shutil.copy2(src_db, dst_db)
                files_copied += 1
                logger.info(f"Copied GHR.db to package ({src_db.stat().st_size / 1024 / 1024:.1f} MB)")
            else:
                logger.warning(f"GHR.db not found at {src_db}")

        # Determine which countries are needed (reads the local
        # GHR.db that was just staged, or — once build_eghr_substrate
        # wires up — the per-package synthesized one).
        required_countries = self._get_required_country_codes()
        if not required_countries:
            logger.warning("No country codes determined, skipping eGHR data inclusion")
            return None

        # Copy only required .SOL files (filtered by country)
        if pythia_config.eghr_sol_dir:
            src_sol_dir = Path(pythia_config.eghr_sol_dir)
            # Resolve relative path
            if not src_sol_dir.is_absolute():
                src_sol_dir = Path.cwd() / src_sol_dir

            if src_sol_dir.exists():
                sol_files_copied = 0
                total_size = 0

                for country_code in required_countries:
                    sol_file = src_sol_dir / f"{country_code}.SOL"
                    if sol_file.exists():
                        dst_sol = eghr_output / sol_file.name
                        shutil.copy2(sol_file, dst_sol)
                        sol_files_copied += 1
                        files_copied += 1
                        total_size += sol_file.stat().st_size
                        logger.debug(f"Copied {sol_file.name} ({sol_file.stat().st_size / 1024 / 1024:.1f} MB)")
                    else:
                        logger.warning(f"SOL file not found for country {country_code}: {sol_file}")

                logger.info(f"Copied {sol_files_copied} .SOL files for countries: {sorted(required_countries)} "
                           f"({total_size / 1024 / 1024:.1f} MB total)")
            else:
                logger.warning(f"eGHR SOL directory not found at {src_sol_dir}")

        if files_copied > 0:
            logger.info(f"eGHR data included in package: {eghr_output} ({files_copied} files)")
            return eghr_output
        else:
            logger.warning("No eGHR data files were copied")
            return None

    # =========================================================================
    # Weather Download Methods
    # =========================================================================

    def _download_site_weather(
        self,
        data: UnifiedData,
        progress_callback: Optional[callable] = None,
        subset_site_ids: Optional[List[int]] = None,
    ) -> Dict[int, 'ClimateTimeSeries']:
        """Download NASA POWER weather data for grid sites.

        Args:
            data: UnifiedData with grid info
            progress_callback: Optional callback(current, total) for progress
            subset_site_ids: Optional list of cell IDs to fetch. If provided,
                only those sites are downloaded (used by the AC-F-CP-13 gate
                to fetch the missing-sites subset rather than re-downloading
                every site). If ``None`` (legacy callers), every grid cell
                is fetched.

        Returns:
            Dictionary mapping site_id to ClimateTimeSeries
        """
        from prismpy.sources.climate.nasa_power import NASAPowerSource, NASAPowerConfig

        # Get config settings
        pythia_config = None
        if self.config.platform_config and hasattr(self.config.platform_config, 'pythia'):
            pythia_config = self.config.platform_config.pythia

        # Determine date range - REQUIRED from config
        if pythia_config and pythia_config.climate_start_date:
            start_date = pythia_config.climate_start_date
        elif self.config.temporal:
            start_date = f"{self.config.temporal.start_year}-01-01"
        else:
            raise ValueError(
                "temporal.start_year or platform_config.pythia.climate_start_date is required. "
                "Please specify the simulation period in your config file."
            )

        if pythia_config and pythia_config.climate_end_date:
            end_date = pythia_config.climate_end_date
        elif self.config.temporal:
            crop_cal = self.config.crop.calendar if self.config.crop else None
            end_date = self.config.temporal.get_climate_end_date(crop_cal).isoformat()
        else:
            raise ValueError(
                "temporal.end_year or platform_config.pythia.climate_end_date is required. "
                "Please specify the simulation period in your config file."
            )

        # Get delay setting
        delay = 2.0
        if pythia_config and hasattr(pythia_config, 'weather_download_delay'):
            delay = pythia_config.weather_download_delay

        # Initialize NASA POWER source with rate limiting
        nasa_config = NASAPowerConfig(
            request_delay=delay,
            retry_count=3,
            timeout=120,
        )
        source = NASAPowerSource(config=nasa_config)

        # Download for each grid cell. When ``subset_site_ids`` is provided
        # (AC-F-CP-13 gate path), restrict to that subset so the caller can
        # skip already-fetched sites.
        climate_data = {}
        all_cells = data.grid.cells if data.grid else []
        if subset_site_ids is not None:
            wanted = set(subset_site_ids)
            cells = [c for c in all_cells if c.cell_id in wanted]
        else:
            cells = all_cells
        total = len(cells)

        logger.info(f"Downloading NASA POWER weather for {total} sites ({start_date} to {end_date})")

        for i, cell in enumerate(cells):
            # V2-22b L (AC L.3): per-cell cancel.
            raise_if_cancelled(
                getattr(self, 'cancel_check', None),
                f"pythia.cell={i + 1}/{total}",
            )
            site_id = cell.cell_id

            if progress_callback:
                progress_callback(i + 1, total)

            logger.info(f"Downloading weather for site {site_id} ({i+1}/{total}): "
                        f"({cell.lat:.4f}, {cell.lon:.4f})")

            try:
                result = source.retrieve(
                    lat=cell.lat,
                    lon=cell.lon,
                    start_date=start_date,
                    end_date=end_date,
                    location_id=site_id,
                    use_cache=True,  # Use caching to speed up repeat runs
                    cancel_check=getattr(self, 'cancel_check', None),
                )

                if result.success and result.data:
                    # Update location ID in the ClimateTimeSeries
                    result.data.location_id = site_id
                    climate_data[site_id] = result.data
                else:
                    logger.warning(f"Failed to download weather for site {site_id}: {result.errors}")

            except PipelineCancelled:
                # V2-22b L: per-cell broad except must not swallow cancel.
                raise
            except Exception as e:
                logger.error(f"Exception downloading weather for site {site_id}: {e}")

            # V2-22b L: pre-sleep cancel check.
            raise_if_cancelled(
                getattr(self, 'cancel_check', None),
                f"pythia.before_sleep={i + 1}/{total}",
            )
            # Rate limiting (already built into NASAPowerSource, but add explicit delay)
            if i < total - 1:
                time.sleep(delay)

        logger.info(f"Downloaded weather for {len(climate_data)}/{total} sites")
        return climate_data

    # =========================================================================
    # SNX Template Generation
    # =========================================================================

    def _generate_snx_template(self, data: UnifiedData) -> Path:
        """Generate DSSAT SNX experiment template with Jinja2 variables.

        Both ``exp_id`` and the on-disk write path derive from
        :meth:`_get_template_filename` so the SNX file written here
        agrees byte-for-byte with the template name referenced by
        ``pythia_config["default_setup"]["template"]`` written by
        :meth:`_generate_pythia_json`. This prevents the cross-
        write-site drift class where one site folds a non-ASCII
        region name while the other writes the raw multi-byte form
        — DSSAT then opens the folded path, finds nothing, and
        logs ``WARNING.OUT: File not found``.

        Args:
            data: UnifiedData with region and config info

        Returns:
            Path to generated SNX template
        """
        # Single source of truth for the SNX filename. The helper
        # ASCII-folds DSSAT-byte-consumed identifiers so non-ASCII
        # region names (e.g., "Bénoué") write byte-clean filenames
        # ("BESG8001.SNX") that DSSAT's fixed-width Fortran reads
        # handle correctly.
        template_filename = self._get_template_filename()
        exp_id = template_filename[:-4]  # strip ".SNX"

        # Get mapped parameters
        config_params = self._map_generic_to_pythia_config()
        cultivar_params = self._map_generic_to_cultivar()
        fertilizer_params = self._map_generic_to_fertilizer()

        # Build SNX content. The @P planting values are jinja placeholders the runner fills from the
        # discrete pythia_config fields — no longer computed/baked here (the ha->m² conversion lives
        # once, in _resolve_planting_params, which feeds pythia_config's default_setup).
        template_content = self._build_snx_content(
            exp_id=exp_id,
            region_name=data.region.name,
            country=data.region.country,
            crop_name=self.config.crop.name,
            cultivar=cultivar_params,
            fertilizer=fertilizer_params,
            config=config_params,
        )

        # Create templates directory
        templates_dir = self.output_dir / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)

        # Write template using the SAME filename the config references,
        # not a re-derived raw region.name slice. Cross-write-site pin
        # at tests/unit/test_pythia_dssat_ascii_fold.py asserts these
        # two surfaces never drift.
        template_path = templates_dir / template_filename
        with open(template_path, 'w') as f:
            f.write(template_content)

        logger.info(f"Generated SNX template: {template_path}")
        return template_path

    def _build_snx_content(
        self,
        exp_id: str,
        region_name: str,
        country: str,
        crop_name: str,
        cultivar: Dict[str, Any],
        fertilizer: Dict[str, Any],
        config: Dict[str, Any],
        ppop: float = 5.0,
        row_spacing: float = 70.0,
    ) -> str:
        """Build SNX file content with Jinja2 placeholders.

        This creates a DSSAT experiment file template that PYTHIA will
        fill in with site-specific values at runtime.

        DSSAT SNX format requirements:
        - TNAME field must be exactly 25 characters
        - MF (fertilizer level) must be 0 if no fertilizers applied
        - RAMT (residue amount) must be a valid value (not -99)

        Args:
            exp_id: Experiment ID (e.g., MLMZ8001)
            region_name: Region name
            country: Country name
            crop_name: Crop name
            cultivar: Cultivar parameters
            fertilizer: Fertilizer parameters
            config: PYTHIA config parameters
            ppop: Plant population (plants/m2)
            row_spacing: Row spacing (cm)

        Returns:
            SNX file content as string
        """
        # Get DSSAT crop code (from centralized mapping)
        crop_code = self._get_dssat_crop_code()

        # Get DSSAT SMODEL (CERES vs CROPGRO) and SYMBI switch
        smodel = self._get_dssat_smodel()
        symbi = self._get_dssat_symbiosis()

        # Build treatment name - must be exactly 25 characters
        # Format: "{crop_name} rainfed" padded to 25 chars
        base_tname = f"{crop_name} rainfed"
        tname = base_tname[:25].ljust(25)  # Truncate if too long, pad if too short

        # Get residue amount from config or use sensible default
        # DSSAT requires a valid value (not -99)
        # Default: 250 kg/ha (typical crop residue incorporation)
        mgmt = self.config.management
        residue_amount = 250  # Default: standard residue amount
        if mgmt and hasattr(mgmt, 'residue_amount_kg_ha'):
            residue_amount = int(mgmt.residue_amount_kg_ha)

        # Per-crop @P fallback baked into each placeholder's ``default()`` — the runner fills
        # ppop/ppoe/plrs/pldp from the discrete pythia_config fields; the default is the honest
        # per-crop literal for the absent/malformed edge (never -99). Preserve the 1-space DSSAT
        # separator + fixed width so the @P columns stay 6-wide.
        pdef = self._crop_planting_default()

        content = f"""*EXP.DETAILS: {exp_id}{crop_code} {country} {region_name}, {crop_name} management scenarios

*GENERAL
@PEOPLE
 prismpy automated
@ADDRESS
 Generated by prismpy
@SITE
 {region_name} Region, {country}

*TREATMENTS                        -------------FACTOR LEVELS------------
@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM
 1 1 1 0 {tname} 1  1  0  1  1  0 {{% if fertilizers %}}1{{% else %}}0{{% endif %}}  1  0  0  {{% if eco2_override_active|default(false) %}}1{{% else %}}0{{% endif %}}  0  1

*CULTIVARS
@C CR INGENO CNAME
 1 {crop_code} {{{{ ingeno }}}} {{{{ cname }}}}

*FIELDS
@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL    FLNAME
 1 GENERIC2 {{{{ wsta }}}}   -99   -99   -99   -99   -99   -99 -99    -99  {{{{ id_soil }}}} {country} {crop_name} rainfed
@L ...........XCRD ...........YCRD .....ELEV .............AREA .SLEN .FLWR .SLAS FLHST FHDUR
 1 {{{{ xcrd }}}} {{{{ ycrd }}}}       -99               -99   -99   -99   -99 {{{{ flhst }}}} {{{{ fhdur }}}}

*INITIAL CONDITIONS
@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME
 1    {crop_code} {{{{ sdate }}}}{{{{ icrt }}}}   -99     1     1   -99{{{{ icres }}}}{{{{ icren }}}}   -99   100     5 rf
@C  ICBL  SH2O  SNH4  SNO3
{{% for layer in ic_layers %}}
 1{{{{ layer.icbl }}}}{{{{ layer.sh2o }}}}{{{{layer.snh4 }}}}{{{{ layer.sno3 }}}}
{{% endfor %}}

*PLANTING DETAILS
@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  SPRL                        PLNAME
 1 {{{{ pdate | default(-99) }}}}   -99 {{{{ "%5.1f"|format(ppop|default({pdef['ppop']})) }}}} {{{{ "%5.1f"|format(ppoe|default({pdef['ppop']})) }}}}     S     R {{{{ "%5.0f"|format(plrs|default({pdef['plrs']})) }}}}   -99 {{{{ "%5.0f"|format(pldp|default({pdef['pldp']})) }}}}   -99   -99   -99   -99   -99                        auto

*FERTILIZERS (INORGANIC)
@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME
{{% for fert in fertilizers %}}
 1 {{{{ fert.fdap }}}} FE005 AP002     5 {{{{ fert.famn }}}}     0   -99   -99   -99   -99 Urea
{{% endfor %}}

*RESIDUES AND ORGANIC FERTILIZER
@R RDATE  RCOD  RAMT  RESN  RESP  RESK  RINP  RDEP  RMET RENAME
 1     0 RE003 {residue_amount:4d}   -99   -99   -99   -99   -99   -99 Residue application

*ENVIRONMENT MODIFICATIONS
@E ODATE EDAY  ERAD  EMAX  EMIN  ERAIN ECO2  EDEW  EWIND ENVNAME
 1 {{{{ sdate }}}} A   0 A   0 A   0 A   0 A   0 {{% if eco2_override_active|default(false) %}}R{{{{ "%4d"|format(co2_ppm) }}}}{{% else %}}A   0{{% endif %}} A   0 A   0 ENV modify

*SIMULATION CONTROLS
@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL
 1 GE          {{{{ nyers }}}}     1     S {{{{ sdate }}}}  2150 Rainfed                   {smodel}
@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2
 1 OP              Y     Y     {symbi}     N     N     N     N     N     M
@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL
 1 ME              M     M     E     R     S     L     R     1     P     S     2
@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS
 1 MA              {{{{ plant_mode | default("R") }}}} {{{{ irrig }}}}     D     D     M
@N OUTPUTS     FNAME OVVEW SUMRY FROPT GROUT CAOUT WAOUT NIOUT MIOUT DIOUT VBOSE CHOUT OPOUT FMOPT
 1 OU              N     N     Y    14     N     N     N     N     N     N     0     N     N     C

@  AUTOMATIC MANAGEMENT
@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN
 1 PL          {{{{ pfrst }}}} {{{{ plast }}}} {{{{ ph2ol }}}}   100    30    40    10
@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF
 1 IR             30    50   100 GS000 IR001    10   1.0
@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF
 1 NI             30    10    50 FE001 GS000
@N RESIDUES    RIPCN RTIME RIDEP
 1 RE            100     1    20
@N HARVEST     HFRST HLAST HPCNP HPCNR
 1 HA              0 01096   100     0
"""
        return content

    # =========================================================================
    # Package Assembly Methods
    # =========================================================================

    def _generate_package_files(self, data: UnifiedData) -> List[Path]:
        """Generate package metadata files (manifest, provenance, README).

        Args:
            data: UnifiedData with region and config info

        Returns:
            List of generated file paths
        """
        from prismpy.packaging.manifest import create_manifest, save_manifest

        output_files = []

        # V2-20: Legacy System B provenance.json generation deleted.
        # Provenance is now handled by System A and distributed via
        # executor._execute_package.

        # 2. Generate manifest.json
        manifest_path = self._generate_manifest(data)
        if manifest_path:
            output_files.append(manifest_path)

        # 3. Generate README.md
        readme_path = self._generate_readme(data)
        if readme_path:
            output_files.append(readme_path)

        return output_files

    # V2-20: _generate_provenance deleted (was System B legacy).
    # Provenance is now handled by System A (prismpy.provenance.tracker)
    # and distributed via executor._execute_package.

    def _generate_manifest(self, data: UnifiedData) -> Path:
        """Generate manifest.json with file inventory and checksums.

        The manifest carries a non-null ``scenario`` block populated
        from the project's temporal + region + crop config via
        :func:`prismpy.packaging.scenario_helpers.build_baseline_scenario_block_for_period`.
        Every PYTHIA baseline package emits the block — UC2 climate-
        scenarios consumers (and any future scenario-set workflow)
        read ``manifest.scenario.scenario_role`` from EVERY package
        in the comparison set, including the baseline anchor.

        Args:
            data: UnifiedData with region info

        Returns:
            Path to manifest.json
        """
        from prismpy.packaging.manifest import (
            create_manifest, derive_boundary_label, save_manifest,
        )
        from prismpy.packaging.scenario_helpers import (
            build_baseline_scenario_block_for_period,
        )

        # Resolved-source discriminator: read the runtime boundary
        # source recorded on the Region (post-fallback at retrieve)
        # and honor the configured GADM admin level only under GADM.
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

        # Build project config for manifest. ``use_case_config`` declares
        # the UCs this package was built to serve — the manifest emitter
        # uses this keyset for closed-world ``uc_readiness`` emit;
        # downstream dispatch may pass UC-specific config overrides at
        # run time but the emit-side declaration here gates which UCs
        # the prismweb confirm-card surfaces.
        from prismpy.packaging.manifest import use_case_config_for
        project_config = {
            "project_name": self.config.project.name,
            "region_name": data.region.name,
            "country": data.region.country,
            "gadm_level": manifest_gadm_level,
            "crop_name": self.config.crop.name,
            "planting_doy": self.config.crop.calendar.planting_doy if self.config.crop.calendar else None,
            "maturity_doy": self.config.crop.calendar.maturity_doy if self.config.crop.calendar else None,
            "start_year": self.config.temporal.start_year if self.config.temporal else None,
            "end_year": self.config.temporal.end_year if self.config.temporal else None,
            "spinup_years": self.config.temporal.spinup_years if self.config.temporal else 0,
            "data_sources": {
                "climate": "NASA POWER",
                "soil": "eGHR",
                "crop_mask": "SPAM 2020",
                "boundaries": boundary_label,
            },
            # F-BP-18: config-driven from the platform→UC SSOT (was a hardcoded
            # literal that drifted — it OMITTED livestock_feed though the
            # consumer supports pythia UC6). pythia now gains livestock_feed.
            "use_case_config": use_case_config_for("pythia"),
        }

        # Build the baseline scenario block. Every PYTHIA package
        # emits a non-null ``manifest.scenario`` so paired-set
        # consumers (UC2 climate scenarios) read the block from EVERY
        # package — baseline-anchor and projection siblings alike.
        # When temporal config is missing we skip emission to avoid
        # injecting a malformed block; downstream consumers handle
        # the legacy missing-block case via ``.get("scenario")``.
        scenario_block = None
        if (
            self.config.temporal is not None
            and self.config.temporal.start_year is not None
            and self.config.temporal.end_year is not None
        ):
            try:
                scenario_block = build_baseline_scenario_block_for_period(
                    region_name=data.region.name,
                    crop_name=self.config.crop.name,
                    time_slice_start=self.config.temporal.start_year,
                    time_slice_end=self.config.temporal.end_year,
                )
            except Exception as exc:  # noqa: BLE001 — emission is best-effort
                # A schema-bound failure (e.g., end_year < start_year)
                # falls back to scenario-null rather than crashing the
                # whole manifest emission. Existing pre-CA-1 behaviour
                # for the malformed-config case.
                logger.warning(
                    "Skipping baseline scenario block emission: %s", exc,
                )

        # PYTHIA always sets the soil-fertility P+K silent-no-op trigger
        # for PYTHIA packages. ``use_case_config`` above unconditionally
        # includes ``soil_fertility`` (the downstream emit gate at
        # ``packaging/manifest.py:644-650`` filters per UC name, so the
        # joint advisory_flag lands only on the soil_fertility entry —
        # never on yield_forecast / sowing_optimization / drought_
        # management). Mirrors the ACEA pattern at
        # ``acea/translator.py:2408-2417`` (ACEA platform manifests
        # carry the trigger too, but the same per-platform gate filters
        # ACEA out so only PYTHIA packages emit the flag).
        # Trigger semantic per parent contract v1.1.7 section 2.7.6.1:
        # "PYTHIA UC5 silently no-ops P+K" — structurally true because
        # PYTHIA templates hardcode PHOSP=N + POTAS=N (DSSAT @N OPTIONS
        # row), so per-element fertility stress is unmodeled by design.
        additional_metadata = {
            "_acea_uc5_p_k_silent_no_op_triggered": True,
        }

        # Create manifest with the (optional) scenario block plumbed
        # through so the on-disk JSON carries it at top level.
        manifest = create_manifest(
            package_dir=self.output_dir,
            project_config=project_config,
            platform="pythia",
            scenario=scenario_block,
            additional_metadata=additional_metadata,
        )

        # Save
        manifest_path = self.output_dir / "manifest.json"
        save_manifest(manifest, manifest_path)

        logger.info(f"Generated manifest: {manifest_path}")
        return manifest_path

    def _generate_readme(self, data: UnifiedData) -> Path:
        """Generate README.md using the centralized template.

        Args:
            data: UnifiedData with region info

        Returns:
            Path to README.md
        """
        from prismpy.packaging.manifest import derive_boundary_label
        from prismpy.packaging.readme_generator import generate_readme

        # Resolved-source discriminator (mirrors the manifest path)
        # so the README boundary cell tracks the runtime outcome.
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

        # Count files by type
        weather_count = len(list((self.output_dir / "weather").glob("*.WTH"))) if (self.output_dir / "weather").exists() else 0
        sol_count = len(list((self.output_dir / "eGHR").glob("*.SOL"))) if (self.output_dir / "eGHR").exists() else 0
        n_sites = len(data.grid.cells) if data.grid else 0

        # Get parameters
        cultivar = self._map_generic_to_cultivar()
        fertilizer = self._map_generic_to_fertilizer()
        config_params = self._map_generic_to_pythia_config()

        # Get management config
        mgmt = self.config.management
        # Planting values via the single resolver (the one ha->m² conversion + per-crop fallback),
        # so the README reports the same ppop/plrs the SNX + pythia_config carry (no 2nd conversion).
        _planting = self._resolve_planting_params()
        plant_pop = _planting['ppop']
        row_spacing = _planting['plrs']

        # Build config dictionary for template
        readme_config = {
            # Project info
            'project_name': self.config.project.name,
            'package_dir': self.output_dir.name,
            'region_name': data.region.name,
            'country': data.region.country,
            'crop_name': self.config.crop.name,
            'gadm_level': manifest_gadm_level,

            # Temporal
            'start_year': self.config.temporal.start_year if self.config.temporal else 2010,
            'end_year': self.config.temporal.end_year if self.config.temporal else 2019,
            'n_years': (self.config.temporal.end_year - self.config.temporal.start_year + 1) if self.config.temporal else 10,

            # Package contents
            'n_sites': n_sites,
            'n_weather_files': weather_count,
            'n_sol_files': sol_count,
            'wsta_prefix': self._get_wsta_prefix(),
            'template_name': self._get_template_filename(),

            # Crop parameters
            'cultivar_code': cultivar.get('ingeno', '990002'),
            'cultivar_name': cultivar.get('cname', 'MEDIUM_SEASON'),
            'total_gdd': cultivar.get('total_gdd', 'N/A'),
            'pfrst': config_params.get('pfrst', 'N/A'),
            'plast': config_params.get('plast', 'N/A'),

            # Management settings
            'fen_tot': fertilizer.get('fen_tot', 60),
            'plant_pop': plant_pop,
            'row_spacing': row_spacing,
            'irrigation': 'Enabled' if mgmt and mgmt.irrigation else 'Rainfed',

            # Data sources
            'data_sources': {
                'climate': 'NASA POWER',
                'soil': 'eGHR (GGCMI)',
                'crop_mask': 'SPAM 2020',
                'boundaries': boundary_label,
            }
        }

        # Generate README using centralized template
        readme_path = self.output_dir / "README.md"
        generate_readme(readme_path, readme_config, platform="pythia")

        logger.info(f"Generated README: {readme_path}")
        return readme_path
