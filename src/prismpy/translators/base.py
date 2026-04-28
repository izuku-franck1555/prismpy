"""
Abstract base class for platform-specific translators.

This module defines the interface that all platform translators must implement,
ensuring consistent behavior across SARRA-Py, CRAFT, PYTHIA, and ACEA.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

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
    """
    success: bool
    platform: Platform
    output_dir: Path
    output_files: List[Path]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


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
    ) -> TranslationResult:
        """Create a translation result object.

        Args:
            success: Whether translation succeeded
            output_files: List of output file paths
            errors: List of errors
            warnings: List of warnings
            metadata: Additional metadata

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

        Mutates ``data.climate`` in place; safe to call multiple times
        (later calls overwrite earlier entries for the same cell, which
        matches the in-process re-translate semantics).
        """
        if not climate_by_cell_id or not isinstance(data.climate, dict):
            return
        real = {
            cid: ts
            for cid, ts in climate_by_cell_id.items()
            if cid >= 0
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
