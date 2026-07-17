"""
Abstract base class for platform-specific translators.

This module defines the interface that all platform translators must implement,
ensuring consistent behavior across SARRA-Py, CRAFT, PYTHIA, and ACEA.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import shutil
import os
import tempfile

from prismpy.cells.cell_id_validation import is_real_climate_cell_id
from prismpy.config.schema import ProjectConfig, Platform
from prismpy.models.region import Region
from prismpy.models.spatial import SpatialGrid
from prismpy.models.climate import ClimateTimeSeries
from prismpy.models.soil import SoilProfile
from prismpy.models.crop import CropParameters, CropCalendar
from prismpy.provenance.tracker import ProvenanceTracker


@dataclass
class UnifiedData:
    """Container for all unified data passed to translators.

    This is the canonical data representation that translators consume.
    All data has been harmonized and validated before reaching translators.

    Attributes:
        region: Region definition with bounds
        grid: Spatial grid (for cell-based platforms)
        climate: Climate data - either Dict[int, ClimateTimeSeries] for in-memory
                 or Dict with 'rainfall_dir', 'agera5_dir' paths for existing files
        soil: Soil data - either Dict[int, SoilProfile] for in-memory
              or Dict with 'isda' data object for existing files
        crop_params: Crop parameters
        crop_calendar: Crop calendar for each location
        metadata: Additional metadata
    """
    region: Region
    grid: Optional[SpatialGrid] = None
    climate: Optional[Any] = None  # Dict[int, ClimateTimeSeries] or path dict
    soil: Optional[Any] = None  # Dict[int, SoilProfile] or path dict
    crop_params: Optional[CropParameters] = None
    crop_calendar: Optional[Dict[int, CropCalendar]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TranslationResult:
    """Result of a translation operation.

    Attributes:
        success: Whether translation completed successfully
        platform: Target platform
        output_dir: Directory containing output files
        output_files: List of generated output file paths
        errors: List of errors encountered
        warnings: List of warnings generated
        metadata: Additional result metadata
        error_events: Structured error payloads built by the producer at
            catch sites so consumers can dispatch on error class instead
            of pattern-matching ``errors`` strings. Additive — every
            existing reader of ``errors`` keeps working unchanged.
    """
    success: bool
    platform: Platform
    output_dir: Path
    output_files: List[Path]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    error_events: List[Dict[str, Any]] = field(default_factory=list)


class ObservedTrialsCopyError(RuntimeError):
    """§7: the modeler-supplied observed N-trials CSV (n_trials_source_path)
    could not be copied into the package (missing / not a file / OS copy
    error). Typed so the PACKAGE stage treats a trials-copy failure as FATAL
    (a config error, identical across platforms) while genuinely tolerable
    per-platform generate_package failures stay non-fatal warnings.
    """


class BaseTranslator(ABC):
    """Abstract base class for platform-specific translators.

    All platform translators must inherit from this class and implement
    the required abstract methods. This ensures a consistent interface
    across all translators.

    The translation process follows these steps:
    1. Validate that required data is available
    2. Prepare output directory structure
    3. Generate platform-specific output files
    4. Validate outputs against platform requirements
    5. Return translation result with file list and any errors

    Attributes:
        config: Project configuration
        platform: Target platform identifier
        output_dir: Base output directory
        provenance: Provenance tracker (optional)
        logger: Logger instance
    """

    # Platform identifier (must be set by subclasses)
    PLATFORM: Platform = None

    # Required data types for this translator
    REQUIRED_DATA: List[str] = ["region"]

    def __init__(
        self,
        config: ProjectConfig,
        output_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the translator.

        Args:
            config: Project configuration
            output_dir: Base output directory (overrides config)
            provenance: Provenance tracker for audit trail
        """
        self.config = config
        self.provenance = provenance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            base_dir = Path(config.output.base_dir)
            if config.output.structure == "by_platform":
                self.output_dir = base_dir / self.PLATFORM.value / config.region.name
            else:
                self.output_dir = base_dir / config.region.name / self.PLATFORM.value

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _copy_observed_trials(self) -> Optional[Path]:
        """§7: place the modeler-supplied observed N-trials CSV at the package's
        data/n_trials.csv (the by-convention path prism-runner's n_response_skill
        UC reads). Returns the dest Path when a source is supplied, or None when
        none is (n_response_skill is then honestly not-ready via n_trials_present).

        POST-CONDITION INVARIANT (fail-closed, closing the placement-bug class):
        on return, either (source supplied) dest is a REGULAR FILE, or (no source)
        dest does NOT exist. Guarded three ways: (1) the whole body runs under a
        catch-all boundary, so ANY exception (OSError, RuntimeError from a
        resolve() symlink loop, RecursionError from rmtree, ...) becomes a typed
        ObservedTrialsCopyError - only our own typed error passes through un-
        rewrapped; (2) dest is enforced INSIDE output_dir (no symlinked parent /
        path escape) before any FS op; (3) the copy is atomic (copy to a temp in
        dest.parent, then os.replace) so a hard-linked external file is never
        mutated in place and a partial copy is never the visible dest. ANY
        deviation raises so the PACKAGE stage fails loud (executor -> errors ->
        success=False) instead of shipping a package that falsely reads as
        trials-present.
        """
        src = getattr(self.config, "n_trials_source_path", None)
        dest = self.output_dir / "data" / "n_trials.csv"
        try:
            # Containment: dest must live inside output_dir. A symlinked parent
            # (data/ -> /external) or any path escape would let mkdir/copy write
            # off-package and let the reconcile removal destroy an EXTERNAL target.
            # Enforce BEFORE any filesystem op (removal / mkdir / copy).
            out_root = self.output_dir.resolve()
            if dest.parent.is_symlink() or out_root not in dest.resolve().parents:
                raise ObservedTrialsCopyError(
                    f"observed-trials dest {dest} is not contained in the package "
                    f"root {self.output_dir} (symlinked parent or path escape): "
                    "refusing to place trials off-package"
                )
            if not src:
                # No source: dest MUST NOT exist. Reconcile any stale artifact
                # (regular file OR directory) left in a reused output dir, then
                # assert the post-condition.
                self._remove_stale_trials_dest(dest)
                if dest.exists() or dest.is_symlink():
                    raise ObservedTrialsCopyError(
                        f"observed-trials reconcile failed: {dest} still present "
                        "after removal"
                    )
                return None
            src_path = Path(src)
            if not src_path.is_file():
                raise ObservedTrialsCopyError(
                    f"n_trials_source_path {src_path} is not a file: cannot copy "
                    "the observed-trials CSV for n_response_skill (UC7)."
                )
            # Refuse a non-regular-file already at the EXACT dest (a directory /
            # symlink would swallow or misdirect the copy). A stale regular file is
            # fine - the atomic replace below swaps it without mutating its inode.
            if dest.is_symlink() or (dest.exists() and not dest.is_file()):
                raise ObservedTrialsCopyError(
                    f"observed-trials dest {dest} exists but is not a regular file: "
                    "refusing to copy (a directory/symlink would swallow the CSV)"
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: copy to a temp in the SAME dir, then os.replace onto
            # dest. os.replace breaks any existing link (a stale dest HARD-LINKED
            # to an external file is NOT mutated in place) and makes a partial copy
            # never the visible dest.
            fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=".n_trials-", suffix=".tmp")
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                shutil.copy2(src_path, tmp)
                os.replace(tmp, dest)
            finally:
                tmp.unlink(missing_ok=True)
            # Post-condition: the EXACT dest path is now a regular file.
            if dest.is_symlink() or not dest.is_file():
                raise ObservedTrialsCopyError(
                    f"post-copy invariant violated: {dest} is not a regular file"
                )
            self.logger.info("Copied observed N-trials CSV -> %s", dest)
            return dest
        except ObservedTrialsCopyError:
            raise
        except Exception as exc:
            # Catch-all fail-closed boundary: every placement failure - OSError,
            # RuntimeError (a resolve() symlink loop on py3.11), RecursionError
            # (rmtree on a deep dir), shutil.SameFileError, ... - is typed so
            # nothing escapes to a silent PACKAGE success. Our own
            # ObservedTrialsCopyError is re-raised above un-rewrapped;
            # BaseException (KeyboardInterrupt/SystemExit) is intentionally NOT
            # caught.
            raise ObservedTrialsCopyError(
                f"observed-trials placement failed for {dest}: {exc}"
            ) from exc

    def _remove_stale_trials_dest(self, dest: Path) -> None:
        """Remove a stale observed-trials artifact at ``dest`` (a regular file or
        a directory) from a reused output dir. Called only from within
        ``_copy_observed_trials``'s catch-all boundary, which types any removal
        failure - so a reconcile can never fail silently."""
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        elif dest.exists() or dest.is_symlink():
            dest.unlink()

    @abstractmethod
    def translate(self, data: UnifiedData) -> TranslationResult:
        """Translate unified data to platform-specific format.

        This is the main entry point for translation. Subclasses must
        implement this method to generate all required output files.

        Args:
            data: Unified data container with all input data

        Returns:
            TranslationResult with success status and output files
        """
        pass

    @abstractmethod
    def validate_outputs(self) -> List[str]:
        """Validate generated outputs against platform requirements.

        Returns:
            List of validation error messages (empty if valid)
        """
        pass

    @abstractmethod
    def get_required_data(self) -> List[str]:
        """Get list of required data types for this translator.

        Returns:
            List of required data type names (e.g., ["region", "climate", "soil"])
        """
        pass

    def validate_input_data(self, data: UnifiedData) -> List[str]:
        """Validate that required input data is available.

        Args:
            data: Unified data container

        Returns:
            List of validation error messages
        """
        errors = []
        required = self.get_required_data()

        if "region" in required and data.region is None:
            errors.append("Region data is required")

        if "grid" in required and data.grid is None:
            errors.append("Spatial grid is required")

        if "climate" in required and not data.climate:
            errors.append("Climate data is required")

        if "soil" in required and not data.soil:
            errors.append("Soil data is required")

        if "crop_params" in required and data.crop_params is None:
            errors.append("Crop parameters are required")

        if "crop_calendar" in required and not data.crop_calendar:
            errors.append("Crop calendar is required")

        return errors

    def get_platform_config(self) -> Any:
        """Get the platform-specific configuration.

        Returns:
            Platform configuration object or None
        """
        return self.config.get_platform_config(self.PLATFORM)

    def create_result(
        self,
        success: bool,
        output_files: List[Path],
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error_events: Optional[List[Dict[str, Any]]] = None,
    ) -> TranslationResult:
        """Create a translation result object.

        Args:
            success: Whether translation succeeded
            output_files: List of output file paths
            errors: List of errors
            warnings: List of warnings
            metadata: Additional metadata
            error_events: Structured error payloads (see TranslationResult).
                Optional; defaults to an empty list so existing callers
                stay backward-compatible.

        Returns:
            TranslationResult object
        """
        return TranslationResult(
            success=success,
            platform=self.PLATFORM,
            output_dir=self.output_dir,
            output_files=output_files,
            errors=errors or [],
            warnings=warnings or [],
            metadata=metadata or {},
            error_events=error_events or [],
        )

    def generate_package(
        self, data: UnifiedData, output_files: List[Path]
    ) -> List[Path]:
        """Generate package metadata files (manifest, provenance, README).

        Called by the pipeline's PACKAGE stage after translation and
        validation are complete. Subclasses should override this to
        provide platform-specific package generation.

        Args:
            data: Unified data container
            output_files: List of files generated during translation

        Returns:
            List of generated package file paths
        """
        return []

    def log_translation_start(self, data: UnifiedData) -> None:
        """Log the start of translation."""
        self.logger.info(
            f"Starting {self.PLATFORM.value} translation for {data.region.name}"
        )

    def log_translation_complete(self, result: TranslationResult) -> None:
        """Log the completion of translation."""
        if result.success:
            self.logger.info(
                f"Completed {self.PLATFORM.value} translation: "
                f"{len(result.output_files)} files generated"
            )
        else:
            self.logger.error(
                f"Failed {self.PLATFORM.value} translation: "
                f"{len(result.errors)} errors"
            )

    def _surface_per_cell_climate(
        self,
        data: UnifiedData,
        climate_by_cell_id: Optional[Dict[int, ClimateTimeSeries]],
    ) -> None:
        """Surface translator-downloaded per-cell climate back onto the
        shared ``UnifiedData.climate`` dict so downstream readers see the
        actual loaded state instead of the harmonize-stage placeholder.

        The retrieve stage emits ``{-1: ClimateTimeSeries(source="placeholder")}``
        for platforms that self-download weather at translate time
        (CRAFT / PYTHIA / ACEA). Without this surfacing, the cell-summary
        builder, the per-cell coverage validators, and the manifest's
        ``len(climate)`` reader all see the placeholder and report every
        real cell as unavailable — even when the translator successfully
        wrote per-cell weather files to disk.

        ``climate_by_cell_id`` keys are 5-arcmin grid cell IDs (the same
        ``cell.cell_id`` the cell-summary builder iterates). Negative
        keys (sentinel placeholders, scratch IDs) are skipped so the
        surfaced state stays clean. The placeholder entry at ``-1`` is
        dropped after the merge so the validator's per-cell loop does
        not double-count it.

        Helper-side filter: ``ts.records`` truthy (at least one
        record) is the minimum bar for a real entry. CRAFT applies a
        stricter pre-filter (``len(ts.records) > 1``) before passing
        in, so a single-record entry never reaches the helper from
        that translator. PYTHIA / ACEA pass their downloaded dicts
        directly, so any non-empty records list surfaces. ACEA in
        particular relies on
        ``_download_climate_30arcmin`` having ALREADY mapped the
        per-tile downloads back to ``cell.cell_id`` keys before the
        surfacing call — re-fanning out via tile_ids would
        double-map and produce an empty result.

        Mutates ``data.climate`` in place; safe to call multiple times
        (later calls overwrite earlier entries for the same cell, which
        matches the in-process re-translate semantics).
        """
        if not climate_by_cell_id or not isinstance(data.climate, dict):
            return
        real = {
            cid: ts
            for cid, ts in climate_by_cell_id.items()
            if is_real_climate_cell_id(cid)
            and hasattr(ts, "records")
            and ts.records
        }
        if not real:
            return
        data.climate.update(real)
        # Drop the harmonize-stage sentinel placeholder so downstream
        # consumers (validators, cell-summary writer) do not iterate
        # the synthetic ``-1`` cell alongside the real grid IDs.
        data.climate.pop(-1, None)

        # F-CK hot-fix +17 — re-fan ``data.crop_calendar`` across the
        # post-surfacing climate roster.
        #
        # CRAFT / PYTHIA / ACEA self-download per-cell weather at
        # translate time. The retrieve stage's calendar fan-out at
        # ``pipeline/executor.py`` produces a calendar keyed by the
        # retrieve-time climate roster (the ``{-1: placeholder}``
        # sentinel for those translators). After this helper drops
        # the ``-1`` sentinel and adds the real cell IDs, the
        # consumer at ``cockpit/observed_values_writer.py:260`` then
        # queries ``cell_id not in crop_calendar`` for those real
        # cells — and raises ``ValueError`` unless we re-fan here.
        # Without the re-fan, the producer-consumer vocabulary
        # drifts (durable §27) and every package build emits an
        # empty ``cockpit_observed_values.json`` (CMS §9.4 violation).
        #
        # Source of truth: ``self.config.crop.calendar`` carries the
        # canonical wizard-supplied planting / maturity doys. Reading
        # the config directly (rather than copying from an existing
        # calendar entry) keeps the helper independent of the
        # retrieve stage's emit shape — if the upstream changes the
        # placeholder key or skips the retrieve-stage fan-out
        # entirely, the helper still produces the right calendar.
        crop_calendar_config = getattr(
            getattr(getattr(self, "config", None), "crop", None),
            "calendar",
            None,
        )
        if crop_calendar_config is not None:
            # F-DL AC-DL-3 site 2 — same canonical helper drives the
            # ``crop_calendar`` re-fan. The cell-id vocabulary
            # invariant is identical to site 1 above; keeping both
            # call sites on the canonical predicate avoids future
            # drift where the per-cell filter widens but the fanout
            # filter doesn't (or vice versa).
            data.crop_calendar = {
                cid: CropCalendar(
                    location_id=cid,
                    planting_doy=crop_calendar_config.planting_doy,
                    maturity_doy=crop_calendar_config.maturity_doy,
                    source="config",
                )
                for cid in data.climate.keys()
                if is_real_climate_cell_id(cid)
            }


class SarraPyTranslatorBase(BaseTranslator):
    """Base class for SARRA-Py translator with platform-specific constants."""

    PLATFORM = Platform.SARRA_PY
    REQUIRED_DATA = ["region", "climate", "soil", "crop_params"]

    # SARRA-Py specific output subdirectories (matching standardized package structure)
    OUTPUT_SUBDIRS = ["config", "data/boundaries", "data/climate", "data/soil", "parameters", "validation"]

    def get_required_data(self) -> List[str]:
        return self.REQUIRED_DATA


class CraftTranslatorBase(BaseTranslator):
    """Base class for CRAFT translator with platform-specific constants."""

    PLATFORM = Platform.CRAFT
    REQUIRED_DATA = ["region", "grid", "climate", "soil"]

    # CRAFT specific output subdirectories
    OUTPUT_SUBDIRS = ["schema", "weather", "soil", "crop_mask", "management"]

    # Global grid dimensions
    GLOBAL_COLS = 4320
    GLOBAL_ROWS = 2160

    def __init__(self, *args, **kwargs):
        # Root choke point + side-effect-free: check BEFORE super().__init__ creates
        # the output dir. Any CRAFT instantiation produces CRAFT output regardless
        # of config.targets, so force the CRAFT resolution check here.
        config = args[0] if args else kwargs.get("config")
        if config is not None:
            config.assert_craft_resolution_compatible(targets=[Platform.CRAFT])
        super().__init__(*args, **kwargs)

    def get_required_data(self) -> List[str]:
        return self.REQUIRED_DATA


class PythiaTranslatorBase(BaseTranslator):
    """Base class for PYTHIA translator with platform-specific constants."""

    PLATFORM = Platform.PYTHIA
    REQUIRED_DATA = ["region", "grid", "climate", "soil", "crop_calendar"]

    # PYTHIA specific output subdirectories
    OUTPUT_SUBDIRS = ["shapes", "weather", "raster", "templates", "config"]

    def get_required_data(self) -> List[str]:
        return self.REQUIRED_DATA


class AceaTranslatorBase(BaseTranslator):
    """Base class for ACEA translator with platform-specific constants."""

    PLATFORM = Platform.ACEA
    REQUIRED_DATA = ["region", "grid", "climate", "soil", "crop_params", "crop_calendar"]

    # ACEA specific output subdirectories
    OUTPUT_SUBDIRS = ["climate", "soil", "crop_calendar", "crop_params", "co2", "config"]

    # ACEA grid dimensions
    GRID_ROWS_30ARCMIN = 360
    GRID_COLS_30ARCMIN = 720
    GRID_ROWS_5ARCMIN = 2160
    GRID_COLS_5ARCMIN = 4320

    def get_required_data(self) -> List[str]:
        return self.REQUIRED_DATA
