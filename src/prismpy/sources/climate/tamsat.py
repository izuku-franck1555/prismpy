"""
TAMSAT rainfall data source retriever.

This module provides functionality to access TAMSAT (Tropical Applications of
Meteorology using SATellite data) rainfall estimates for Africa.

TAMSAT provides daily rainfall estimates at ~4km (0.0375°) resolution,
available from 1983-present, primarily used by SARRA-Py.

Reference: SARRA-Py/02-WEATHER-PREPARATION/ implementation patterns.
"""

import glob
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from prismpy.models.region import BoundingBox, Region
from prismpy.provenance.tracker import DecisionType, ProvenanceTracker
from prismpy.sources.base import DataSource, RetrievalResult


logger = logging.getLogger(__name__)


@dataclass
class TAMSATConfig:
    """Configuration for TAMSAT data access."""

    base_url: str = "https://www.tamsat.org.uk/"
    version: str = "v3.1"
    resolution: float = 0.0375  # ~4km
    data_dir: Optional[Path] = None  # Local data directory
    use_sarra_download: bool = True  # Try to use SARRA_data_download library
    timeout: int = 300  # Download timeout


@dataclass
class TAMSATData:
    """Container for TAMSAT rainfall data.

    Attributes:
        region_name: Name of the region
        bounds: Bounding box (GIS format)
        bounds_sarra_py: Bounding box (SARRA-Py format)
        start_date: Start of data coverage
        end_date: End of data coverage
        resolution: Spatial resolution in degrees
        data_dir: Directory containing the GeoTIFF files
        file_count: Number of daily files available
        variables: List of available variables (typically just 'rain')
    """

    region_name: str
    bounds: List[float]
    bounds_sarra_py: List[float]
    start_date: date
    end_date: date
    resolution: float
    data_dir: Path
    file_count: int
    variables: List[str]


class TAMSATSource(DataSource):
    """Data source for TAMSAT rainfall estimates.

    TAMSAT (Tropical Applications of Meteorology using SATellite data)
    provides daily rainfall estimates for Africa at ~4km resolution.

    The data can be accessed in two ways:
    1. Using the SARRA_data_download library (if available)
    2. Loading from pre-downloaded GeoTIFF files

    Attributes:
        NAME: Data source identifier
        EARLIEST_DATE: Earliest available date
        RESOLUTION: Native resolution in degrees
        CRS: Coordinate reference system
    """

    NAME = "tamsat"
    EARLIEST_DATE = date(1983, 1, 1)
    RESOLUTION = 0.0375  # ~4km
    CRS = "EPSG:4326"

    # File naming pattern
    FILE_PATTERN = "TAMSAT_{version}_{region}_rfe_filled_{year}_{month:02d}_{day:02d}.tif"

    def __init__(
        self,
        config: Optional[TAMSATConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the TAMSAT data source.

        Args:
            config: TAMSAT configuration
            cache_dir: Directory for caching data
            provenance: Provenance tracker
        """
        super().__init__(cache_dir=cache_dir, provenance=provenance)
        self.config = config or TAMSATConfig()
        self._sarra_download_available = None

    @property
    def sarra_download_available(self) -> bool:
        """Check if SARRA_data_download library is available."""
        if self._sarra_download_available is None:
            try:
                from SARRA_data_download.get_satellite_rainfall_estimates import (
                    download_TAMSAT_year_parallel,
                )
                self._sarra_download_available = True
            except ImportError:
                self._sarra_download_available = False
        return self._sarra_download_available

    def retrieve(
        self,
        region: Region,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        download: bool = False,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve TAMSAT rainfall data for a region.

        Args:
            region: Region with bounding box
            start_date: Start date (YYYY-MM-DD or date object)
            end_date: End date (YYYY-MM-DD or date object)
            data_dir: Directory containing/for TAMSAT data
            download: Whether to download data if not available locally
            **kwargs: Additional parameters

        Returns:
            RetrievalResult containing TAMSATData object
        """
        errors = []
        warnings = []
        metadata = {
            "source": self.NAME,
            "version": self.config.version,
            "resolution": self.config.resolution,
        }

        # Parse dates
        start_date = self._parse_date(start_date) if start_date else self.EARLIEST_DATE
        end_date = self._parse_date(end_date) if end_date else date.today() - timedelta(days=1)

        metadata["start_date"] = start_date.isoformat()
        metadata["end_date"] = end_date.isoformat()

        # Determine data directory
        if data_dir:
            data_dir = Path(data_dir)
        elif self.config.data_dir:
            data_dir = self.config.data_dir
        else:
            # Default: cache_dir/tamsat/{region_name}/
            from prismpy.utils.sanitization import normalize_region_name
            safe_name = normalize_region_name(region.name)
            data_dir = self.cache_dir / "tamsat" / safe_name

        # Get bounds in both formats
        bounds_gis = region.bounds.to_gis_format()
        bounds_sarra_py = region.bounds.to_sarra_py_format()

        metadata["bounds_gis"] = bounds_gis
        metadata["bounds_sarra_py"] = bounds_sarra_py
        metadata["data_dir"] = str(data_dir)

        # Check if data exists locally
        if data_dir.exists():
            file_info = self._validate_local_files(
                data_dir=data_dir,
                region_name=region.name,
                start_date=start_date,
                end_date=end_date,
            )

            if file_info["complete"]:
                self.logger.info(
                    f"Found {file_info['file_count']} TAMSAT files for {region.name}"
                )

                tamsat_data = TAMSATData(
                    region_name=region.name,
                    bounds=bounds_gis,
                    bounds_sarra_py=bounds_sarra_py,
                    start_date=start_date,
                    end_date=end_date,
                    resolution=self.config.resolution,
                    data_dir=data_dir,
                    file_count=file_info["file_count"],
                    variables=["rain"],
                )

                metadata["from_local"] = True
                metadata["file_count"] = file_info["file_count"]
                metadata["missing_dates"] = [d.isoformat() for d in file_info["missing_dates"][:10]]

                if file_info["missing_dates"]:
                    warnings.append(
                        f"{len(file_info['missing_dates'])} dates missing from local files"
                    )

                return self.create_result(
                    success=True,
                    data=tamsat_data,
                    output_path=data_dir,
                    warnings=warnings,
                    metadata=metadata,
                )
            else:
                warnings.append(
                    f"Local data incomplete: {file_info['file_count']} files found, "
                    f"{len(file_info['missing_dates'])} dates missing"
                )

        # Download if requested and library available
        if download:
            if not self.sarra_download_available:
                return self.create_result(
                    success=False,
                    errors=[
                        "SARRA_data_download library not available. "
                        "Install with: pip install SARRA-data-download"
                    ],
                    warnings=warnings,
                    metadata=metadata,
                )

            try:
                self._download_tamsat(
                    bounds=bounds_sarra_py,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=data_dir,
                    region_name=region.name,
                )

                # Re-validate after download
                file_info = self._validate_local_files(
                    data_dir=data_dir,
                    region_name=region.name,
                    start_date=start_date,
                    end_date=end_date,
                )

                tamsat_data = TAMSATData(
                    region_name=region.name,
                    bounds=bounds_gis,
                    bounds_sarra_py=bounds_sarra_py,
                    start_date=start_date,
                    end_date=end_date,
                    resolution=self.config.resolution,
                    data_dir=data_dir,
                    file_count=file_info["file_count"],
                    variables=["rain"],
                )

                metadata["downloaded"] = True
                metadata["file_count"] = file_info["file_count"]

                # Record provenance
                if self.provenance:
                    self.provenance.record_retrieval(
                        source=self.NAME,
                        parameters={
                            "region": region.name,
                            "bounds": bounds_sarra_py,
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                        },
                        output_path=data_dir,
                        decisions=[],
                    )

                return self.create_result(
                    success=True,
                    data=tamsat_data,
                    output_path=data_dir,
                    warnings=warnings,
                    metadata=metadata,
                )

            except Exception as e:
                return self.create_result(
                    success=False,
                    errors=[f"Download failed: {e}"],
                    warnings=warnings,
                    metadata=metadata,
                )

        # Data not available and download not requested
        return self.create_result(
            success=False,
            errors=[
                f"TAMSAT data not found at {data_dir}. "
                "Either provide existing data or set download=True"
            ],
            warnings=warnings,
            metadata=metadata,
        )

    def validate(self, data: Any) -> List[str]:
        """Validate TAMSAT data.

        Args:
            data: TAMSATData object to validate

        Returns:
            List of validation error/warning messages
        """
        warnings = []

        if not isinstance(data, TAMSATData):
            return [f"Expected TAMSATData, got {type(data)}"]

        # Check file count
        expected_days = (data.end_date - data.start_date).days + 1
        if data.file_count < expected_days * 0.95:  # Allow 5% missing
            warnings.append(
                f"Only {data.file_count}/{expected_days} daily files available "
                f"({100*data.file_count/expected_days:.1f}%)"
            )

        # Check data directory exists
        if not data.data_dir.exists():
            warnings.append(f"Data directory does not exist: {data.data_dir}")

        # Validate bounds
        if data.bounds[0] >= data.bounds[2]:  # minx >= maxx
            warnings.append("Invalid bounds: minx >= maxx")
        if data.bounds[1] >= data.bounds[3]:  # miny >= maxy
            warnings.append("Invalid bounds: miny >= maxy")

        return warnings

    def load_daily_rainfall(
        self,
        data_dir: Union[str, Path],
        target_date: date,
        region_name: str,
    ) -> Optional[np.ndarray]:
        """Load rainfall data for a single day.

        Args:
            data_dir: Directory containing TAMSAT files
            target_date: Date to load
            region_name: Region name for file pattern

        Returns:
            2D numpy array of rainfall values, or None if file not found
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required for loading TAMSAT files")

        data_dir = Path(data_dir)
        filename = self.FILE_PATTERN.format(
            version=self.config.version,
            region=region_name,
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
        )

        file_path = data_dir / filename

        if not file_path.exists():
            # Try alternative patterns
            patterns = [
                f"TAMSAT*{target_date.year}_{target_date.month:02d}_{target_date.day:02d}.tif",
                f"*rfe*{target_date.year}_{target_date.month:02d}_{target_date.day:02d}.tif",
            ]
            for pattern in patterns:
                matches = list(data_dir.glob(pattern))
                if matches:
                    file_path = matches[0]
                    break
            else:
                return None

        with rasterio.open(file_path) as src:
            data = src.read(1)  # First band
            return data

    def load_timeseries(
        self,
        data_dir: Union[str, Path],
        start_date: date,
        end_date: date,
        region_name: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Load rainfall time series from TAMSAT files.

        Args:
            data_dir: Directory containing TAMSAT files
            start_date: Start date
            end_date: End date
            region_name: Region name
            lat: Optional latitude for point extraction
            lon: Optional longitude for point extraction

        Returns:
            Dictionary with dates and rainfall values
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required for loading TAMSAT files")

        data_dir = Path(data_dir)
        dates = []
        rainfall = []

        current = start_date
        while current <= end_date:
            data = self.load_daily_rainfall(data_dir, current, region_name)

            if data is not None:
                if lat is not None and lon is not None:
                    # Extract point value (would need transform info)
                    # For now, use center value as placeholder
                    value = float(np.nanmean(data))
                else:
                    # Use spatial mean
                    value = float(np.nanmean(data))

                dates.append(current)
                rainfall.append(value)
            else:
                dates.append(current)
                rainfall.append(np.nan)

            current += timedelta(days=1)

        return {
            "dates": dates,
            "rainfall": rainfall,
            "unit": "mm/day",
            "source": self.NAME,
        }

    def _download_tamsat(
        self,
        bounds: List[float],
        start_date: date,
        end_date: date,
        output_dir: Path,
        region_name: str,
    ) -> None:
        """Download TAMSAT data using SARRA_data_download library.

        Args:
            bounds: Bounding box in SARRA-Py format [lat_NW, lon_NW, lat_SE, lon_SE]
            start_date: Start date
            end_date: End date
            output_dir: Output directory
            region_name: Region name for file naming
        """
        import shutil
        from SARRA_data_download.get_satellite_rainfall_estimates import (
            download_TAMSAT_year_parallel,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # SARRA_data_download expects area dict
        area = {region_name: bounds}

        years = range(start_date.year, end_date.year + 1)
        for year in years:
            self.logger.info(f"Downloading TAMSAT data for {year}...")
            download_TAMSAT_year_parallel(year, area, region_name, str(output_dir))

        # SARRA_data_download writes cropped .tif files to a hardcoded relative
        # path (../data/3_output/TAMSAT_v3.1_{region}_rfe_filled/) instead of
        # output_dir. Relocate them to our cache directory.
        hardcoded_dir = Path("../data/3_output") / f"TAMSAT_v3.1_{region_name}_rfe_filled"
        if hardcoded_dir.exists():
            relocated = 0
            for tif in hardcoded_dir.glob("*.tif"):
                shutil.move(str(tif), str(output_dir / tif.name))
                relocated += 1
            if relocated:
                self.logger.info(f"Relocated {relocated} TAMSAT .tif files to {output_dir}")
            # Clean up empty hardcoded directory
            try:
                hardcoded_dir.rmdir()
            except OSError:
                pass

    def _validate_local_files(
        self,
        data_dir: Path,
        region_name: str,
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        """Validate local TAMSAT files for completeness.

        Args:
            data_dir: Directory containing files
            region_name: Region name
            start_date: Expected start date
            end_date: Expected end date

        Returns:
            Dictionary with validation results
        """
        # Find all TAMSAT files
        patterns = [
            "TAMSAT*.tif",
            "*rfe*.tif",
        ]

        files = []
        for pattern in patterns:
            files.extend(data_dir.glob(pattern))

        # Parse dates from filenames
        found_dates = set()
        for f in files:
            basename = f.name
            # Try to extract date from filename
            # Pattern: ..._YYYY_MM_DD.tif
            parts = basename.replace(".tif", "").split("_")
            if len(parts) >= 3:
                try:
                    year = int(parts[-3])
                    month = int(parts[-2])
                    day = int(parts[-1])
                    found_dates.add(date(year, month, day))
                except (ValueError, IndexError):
                    pass

        # Calculate expected dates
        expected_dates = set()
        current = start_date
        while current <= end_date:
            expected_dates.add(current)
            current += timedelta(days=1)

        # Find missing dates
        missing_dates = sorted(expected_dates - found_dates)

        return {
            "file_count": len(found_dates),
            "expected_count": len(expected_dates),
            "missing_dates": missing_dates,
            "complete": len(missing_dates) == 0,
            "coverage_pct": 100 * len(found_dates) / max(len(expected_dates), 1),
        }

    def _parse_date(self, date_input: Union[str, date]) -> date:
        """Parse date from string or date object."""
        if isinstance(date_input, date):
            return date_input
        return datetime.strptime(date_input, "%Y-%m-%d").date()

    def get_expected_file_path(
        self,
        data_dir: Path,
        region_name: str,
        target_date: date,
    ) -> Path:
        """Get expected file path for a given date.

        Args:
            data_dir: Base data directory
            region_name: Region name
            target_date: Target date

        Returns:
            Expected file path
        """
        filename = self.FILE_PATTERN.format(
            version=self.config.version,
            region=region_name,
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
        )
        return data_dir / filename
