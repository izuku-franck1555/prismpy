"""
Abstract base class for data sources.

All data source retrievers must inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

from prismpy.models.region import Region
from prismpy.provenance.tracker import ProvenanceTracker


@dataclass
class RetrievalResult:
    """Result of a data retrieval operation.

    Attributes:
        success: Whether retrieval succeeded
        data: Retrieved data (type varies by source)
        output_path: Path where data was saved (if applicable)
        errors: List of errors encountered
        warnings: List of warnings generated
        metadata: Additional metadata about the retrieval
    """
    success: bool
    data: Any = None
    output_path: Optional[Path] = None
    errors: list = None
    warnings: list = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.errors = self.errors or []
        self.warnings = self.warnings or []
        self.metadata = self.metadata or {}


class DataSource(ABC):
    """Abstract base class for data source retrievers.

    All data sources (GADM, NASA POWER, TAMSAT, etc.) must inherit
    from this class and implement the required methods.

    Attributes:
        name: Data source identifier
        provenance: Provenance tracker for audit trail
        cache_dir: Directory for caching retrieved data
        logger: Logger instance
    """

    # Source identifier (must be set by subclasses)
    NAME: str = "base"

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the data source.

        Args:
            cache_dir: Directory for caching data
            provenance: Provenance tracker
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.provenance = provenance
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def retrieve(
        self,
        region: Region,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve data for a region.

        Args:
            region: Region to retrieve data for
            **kwargs: Additional retrieval parameters

        Returns:
            RetrievalResult with retrieved data
        """
        pass

    @abstractmethod
    def validate(self, data: Any) -> list:
        """Validate retrieved data.

        Args:
            data: Data to validate

        Returns:
            List of validation error messages
        """
        pass

    def get_cache_path(self, region: Region, suffix: str = "") -> Path:
        """Get cache file path for a region.

        Args:
            region: Region
            suffix: File suffix/extension

        Returns:
            Path for cache file
        """
        from prismpy.utils.sanitization import normalize_region_name
        region_name = normalize_region_name(region.name)
        filename = f"{self.NAME}_{region_name}{suffix}"
        return self.cache_dir / filename

    def is_cached(self, region: Region, suffix: str = "") -> bool:
        """Check if data is cached for a region.

        Args:
            region: Region
            suffix: File suffix

        Returns:
            True if cache file exists
        """
        return self.get_cache_path(region, suffix).exists()

    def create_result(
        self,
        success: bool,
        data: Any = None,
        output_path: Optional[Path] = None,
        errors: Optional[list] = None,
        warnings: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """Create a retrieval result object.

        Args:
            success: Whether retrieval succeeded
            data: Retrieved data
            output_path: Path where data was saved
            errors: List of errors
            warnings: List of warnings
            metadata: Additional metadata

        Returns:
            RetrievalResult object
        """
        return RetrievalResult(
            success=success,
            data=data,
            output_path=output_path,
            errors=errors or [],
            warnings=warnings or [],
            metadata=metadata or {},
        )
