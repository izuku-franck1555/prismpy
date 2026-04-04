"""
AgERA5 climate data source retriever.

This module provides functionality to access AgERA5 (Agrometeorological indicators
from ERA5) climate data via the Copernicus Climate Data Store (CDS) API.

AgERA5 provides daily climate variables at ~10km resolution including:
- Temperature (mean, min, max)
- Solar radiation
- ET0 (computed using Hargreaves method)

Primarily used by SARRA-Py for temperature and radiation data.

Reference: SARRA-Py/02-WEATHER-PREPARATION/ implementation patterns.
"""

import glob
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from prismpy.models.region import Region
from prismpy.provenance.tracker import ProvenanceTracker
from prismpy.sources.base import DataSource, RetrievalResult


logger = logging.getLogger(__name__)


# AgERA5 variable names
AGERA5_VARIABLES = [
    "2m_temperature_24_hour_mean",
    "2m_temperature_24_hour_maximum",
    "2m_temperature_24_hour_minimum",
    "solar_radiation_flux_daily",
]

# Variable name mappings to internal names
AGERA5_TO_INTERNAL = {
    "2m_temperature_24_hour_mean": "tmean",
    "2m_temperature_24_hour_maximum": "tmax",
    "2m_temperature_24_hour_minimum": "tmin",
    "solar_radiation_flux_daily": "srad",
    "ET0Hargreaves": "et0",
}


@dataclass
class AgERA5Config:
    """Configuration for AgERA5 data access."""

    cds_url: str = "https://cds.climate.copernicus.eu/api/v2"
    dataset: str = "sis-agrometeorological-indicators"
    resolution: float = 0.1  # ~10km
    data_dir: Optional[Path] = None
    use_sarra_download: bool = True
    timeout: int = 300


@dataclass
class AgERA5Data:
    """Container for AgERA5 climate data.

    Attributes:
        region_name: Name of the region
        bounds: Bounding box (GIS format)
        bounds_sarra_py: Bounding box (SARRA-Py format)
        start_date: Start of data coverage
        end_date: End of data coverage
        resolution: Spatial resolution in degrees
        data_dir: Directory containing the data files
        variables: Dictionary mapping variable names to file counts
    """

    region_name: str
    bounds: List[float]
    bounds_sarra_py: List[float]
    start_date: date
    end_date: date
    resolution: float
    data_dir: Path
    variables: Dict[str, int]  # {variable_name: file_count}


class AgERA5Source(DataSource):
    """Data source for AgERA5 climate data.

    AgERA5 provides agrometeorological indicators derived from ERA5 reanalysis,
    available from the Copernicus Climate Data Store (CDS).

    The data can be accessed in two ways:
    1. Using the SARRA_data_download library (if available)
    2. Loading from pre-downloaded GeoTIFF files

    Attributes:
        NAME: Data source identifier
        EARLIEST_DATE: Earliest available date
        RESOLUTION: Native resolution in degrees (~10km)
        VARIABLES: Available variables
    """

    NAME = "agera5"
    EARLIEST_DATE = date(1979, 1, 1)
    RESOLUTION = 0.1  # ~10km
    CRS = "EPSG:4326"
    VARIABLES = AGERA5_VARIABLES

    def __init__(
        self,
        config: Optional[AgERA5Config] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the AgERA5 data source.

        Args:
            config: AgERA5 configuration
            cache_dir: Directory for caching data
            provenance: Provenance tracker
        """
        super().__init__(cache_dir=cache_dir, provenance=provenance)
        self.config = config or AgERA5Config()
        self._sarra_download_available = None
        self._cdsapi_available = None

    @property
    def sarra_download_available(self) -> bool:
        """Check if SARRA_data_download library is available."""
        if self._sarra_download_available is None:
            try:
                from SARRA_data_download.get_AgERA5_data import download_AgERA5_year
                self._sarra_download_available = True
            except ImportError:
                self._sarra_download_available = False
        return self._sarra_download_available

    @property
    def cdsapi_available(self) -> bool:
        """Check if cdsapi library is available."""
        if self._cdsapi_available is None:
            try:
                import cdsapi
                self._cdsapi_available = True
            except ImportError:
                self._cdsapi_available = False
        return self._cdsapi_available

    def retrieve(
        self,
        region: Region,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        variables: Optional[List[str]] = None,
        download: bool = False,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve AgERA5 climate data for a region.

        Args:
            region: Region with bounding box
            start_date: Start date
            end_date: End date
            data_dir: Directory containing/for AgERA5 data
            variables: Variables to retrieve (default: all)
            download: Whether to download if not available locally
            **kwargs: Additional parameters

        Returns:
            RetrievalResult containing AgERA5Data object
        """
        errors = []
        warnings = []
        metadata = {
            "source": self.NAME,
            "resolution": self.config.resolution,
        }

        # Parse dates
        start_date = self._parse_date(start_date) if start_date else date(2015, 1, 1)
        end_date = self._parse_date(end_date) if end_date else date.today() - timedelta(days=1)

        metadata["start_date"] = start_date.isoformat()
        metadata["end_date"] = end_date.isoformat()

        # Determine variables
        variables = variables or self.VARIABLES

        # Determine data directory
        if data_dir:
            data_dir = Path(data_dir)
        elif self.config.data_dir:
            data_dir = self.config.data_dir
        else:
            from prismpy.utils.sanitization import normalize_region_name
            safe_name = normalize_region_name(region.name)
            data_dir = self.cache_dir / "agera5" / f"AgERA5_{safe_name}"

        # Get bounds
        bounds_gis = region.bounds.to_gis_format()
        bounds_sarra_py = region.bounds.to_sarra_py_format()

        metadata["bounds_gis"] = bounds_gis
        metadata["bounds_sarra_py"] = bounds_sarra_py
        metadata["data_dir"] = str(data_dir)

        # Check if data exists locally
        if data_dir.exists():
            file_info = self._validate_local_files(
                data_dir=data_dir,
                start_date=start_date,
                end_date=end_date,
                variables=variables,
            )

            if file_info["complete"]:
                self.logger.info(
                    f"Found AgERA5 data for {region.name}: {file_info['total_files']} files"
                )

                agera5_data = AgERA5Data(
                    region_name=region.name,
                    bounds=bounds_gis,
                    bounds_sarra_py=bounds_sarra_py,
                    start_date=start_date,
                    end_date=end_date,
                    resolution=self.config.resolution,
                    data_dir=data_dir,
                    variables=file_info["variable_counts"],
                )

                metadata["from_local"] = True
                metadata["variable_counts"] = file_info["variable_counts"]

                if file_info["missing_by_variable"]:
                    for var, missing in file_info["missing_by_variable"].items():
                        if missing:
                            warnings.append(f"{var}: {len(missing)} dates missing")

                return self.create_result(
                    success=True,
                    data=agera5_data,
                    output_path=data_dir,
                    warnings=warnings,
                    metadata=metadata,
                )
            else:
                warnings.append(
                    f"Local data incomplete. Found: {file_info['total_files']} files"
                )

        # Download if requested
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
                self._download_agera5(
                    bounds=bounds_sarra_py,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=data_dir.parent,  # Library creates subdir
                    region_name=region.name,
                )

                # Re-validate
                file_info = self._validate_local_files(
                    data_dir=data_dir,
                    start_date=start_date,
                    end_date=end_date,
                    variables=variables,
                )

                agera5_data = AgERA5Data(
                    region_name=region.name,
                    bounds=bounds_gis,
                    bounds_sarra_py=bounds_sarra_py,
                    start_date=start_date,
                    end_date=end_date,
                    resolution=self.config.resolution,
                    data_dir=data_dir,
                    variables=file_info["variable_counts"],
                )

                metadata["downloaded"] = True
                metadata["variable_counts"] = file_info["variable_counts"]

                return self.create_result(
                    success=True,
                    data=agera5_data,
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

        return self.create_result(
            success=False,
            errors=[
                f"AgERA5 data not found at {data_dir}. "
                "Either provide existing data or set download=True"
            ],
            warnings=warnings,
            metadata=metadata,
        )

    def validate(self, data: Any) -> List[str]:
        """Validate AgERA5 data.

        Args:
            data: AgERA5Data object

        Returns:
            List of validation messages
        """
        warnings = []

        if not isinstance(data, AgERA5Data):
            return [f"Expected AgERA5Data, got {type(data)}"]

        # Check data directory
        if not data.data_dir.exists():
            warnings.append(f"Data directory does not exist: {data.data_dir}")

        # Check variable coverage
        expected_days = (data.end_date - data.start_date).days + 1
        for var, count in data.variables.items():
            if count < expected_days * 0.95:
                warnings.append(
                    f"{var}: only {count}/{expected_days} files ({100*count/expected_days:.1f}%)"
                )

        return warnings

    def load_variable(
        self,
        data_dir: Union[str, Path],
        variable: str,
        target_date: date,
    ) -> Optional[np.ndarray]:
        """Load a single variable for a specific date.

        Args:
            data_dir: Base data directory
            variable: Variable name
            target_date: Target date

        Returns:
            2D numpy array or None if not found
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required for loading AgERA5 files")

        data_dir = Path(data_dir)
        var_dir = data_dir / variable

        if not var_dir.exists():
            return None

        # File pattern: {variable}_{YYYY}_{MM}_{DD}.tif
        filename = f"{variable}_{target_date.year}_{target_date.month:02d}_{target_date.day:02d}.tif"
        file_path = var_dir / filename

        if not file_path.exists():
            # Try alternative patterns
            patterns = [
                f"{variable}*{target_date.year}_{target_date.month:02d}_{target_date.day:02d}.tif",
                f"*{target_date.year}_{target_date.month:02d}_{target_date.day:02d}.tif",
            ]
            for pattern in patterns:
                matches = list(var_dir.glob(pattern))
                if matches:
                    file_path = matches[0]
                    break
            else:
                return None

        with rasterio.open(file_path) as src:
            return src.read(1)

    def load_timeseries(
        self,
        data_dir: Union[str, Path],
        start_date: date,
        end_date: date,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Load time series for multiple variables.

        Args:
            data_dir: Base data directory
            start_date: Start date
            end_date: End date
            variables: Variables to load (default: all)

        Returns:
            Dictionary of variable data
        """
        variables = variables or self.VARIABLES
        data_dir = Path(data_dir)

        result = {}
        for variable in variables:
            dates = []
            values = []

            current = start_date
            while current <= end_date:
                data = self.load_variable(data_dir, variable, current)
                dates.append(current)

                if data is not None:
                    values.append(float(np.nanmean(data)))
                else:
                    values.append(np.nan)

                current += timedelta(days=1)

            internal_name = AGERA5_TO_INTERNAL.get(variable, variable)
            result[internal_name] = {
                "dates": dates,
                "values": values,
                "source": self.NAME,
                "variable": variable,
            }

        return result

    def _download_agera5(
        self,
        bounds: List[float],
        start_date: date,
        end_date: date,
        output_dir: Path,
        region_name: str,
    ) -> None:
        """Download AgERA5 data using SARRA_data_download library.

        Args:
            bounds: Bounding box in SARRA-Py format
            start_date: Start date
            end_date: End date
            output_dir: Output directory
            region_name: Region name
        """
        import shutil
        from SARRA_data_download.get_AgERA5_data import download_AgERA5_year

        output_dir.mkdir(parents=True, exist_ok=True)

        area = {region_name: bounds}

        years = range(start_date.year, end_date.year + 1)
        for year in years:
            self.logger.info(f"Downloading AgERA5 data for {year}...")
            download_AgERA5_year(
                query_year=year,
                area=area,
                selected_area=region_name,
                save_path=str(output_dir),
                version="SARRA-Py",
            )

        # SARRA_data_download writes final GeoTIFFs to a hardcoded path
        # (../data/3_output/AgERA5_{region}/ relative to CWD, or under save_path).
        # Check both locations and relocate to our cache directory.
        target_dir = output_dir / f"AgERA5_{region_name}"
        hardcoded_dir = None
        for candidate in [
            output_dir / "3_output" / f"AgERA5_{region_name}",
            Path("../data/3_output") / f"AgERA5_{region_name}",
        ]:
            if candidate.exists():
                hardcoded_dir = candidate
                break

        if hardcoded_dir:
            target_dir.mkdir(parents=True, exist_ok=True)
            total_relocated = 0
            for var_dir in hardcoded_dir.iterdir():
                if var_dir.is_dir():
                    dest_var_dir = target_dir / var_dir.name
                    dest_var_dir.mkdir(parents=True, exist_ok=True)
                    for tif in var_dir.glob("*.tif"):
                        shutil.move(str(tif), str(dest_var_dir / tif.name))
                        total_relocated += 1
            if total_relocated:
                self.logger.info(
                    f"Relocated {total_relocated} AgERA5 .tif files to {target_dir}"
                )
            # Clean up hardcoded directories
            try:
                shutil.rmtree(output_dir / "3_output", ignore_errors=True)
                shutil.rmtree(output_dir / "2_conversion", ignore_errors=True)
                shutil.rmtree(output_dir / "1_extraction", ignore_errors=True)
                shutil.rmtree(output_dir / "0_downloads", ignore_errors=True)
            except OSError:
                pass

    def _validate_local_files(
        self,
        data_dir: Path,
        start_date: date,
        end_date: date,
        variables: List[str],
    ) -> Dict[str, Any]:
        """Validate local AgERA5 files.

        Args:
            data_dir: Base data directory
            start_date: Expected start date
            end_date: Expected end date
            variables: Variables to check

        Returns:
            Validation results
        """
        variable_counts = {}
        missing_by_variable = {}
        total_files = 0

        expected_days = (end_date - start_date).days + 1

        for variable in variables:
            var_dir = data_dir / variable

            if not var_dir.exists():
                variable_counts[variable] = 0
                missing_by_variable[variable] = list(
                    self._date_range(start_date, end_date)
                )
                continue

            # Find files
            files = list(var_dir.glob("*.tif"))
            found_dates = set()

            for f in files:
                # Parse date from filename
                parts = f.stem.split("_")
                if len(parts) >= 3:
                    try:
                        year = int(parts[-3])
                        month = int(parts[-2])
                        day = int(parts[-1])
                        found_dates.add(date(year, month, day))
                    except (ValueError, IndexError):
                        pass

            variable_counts[variable] = len(found_dates)
            total_files += len(found_dates)

            # Find missing dates
            expected_dates = set(self._date_range(start_date, end_date))
            missing = sorted(expected_dates - found_dates)
            missing_by_variable[variable] = missing

        # Determine completeness (95% threshold)
        complete = all(
            count >= expected_days * 0.95
            for count in variable_counts.values()
        )

        return {
            "variable_counts": variable_counts,
            "missing_by_variable": missing_by_variable,
            "total_files": total_files,
            "expected_per_variable": expected_days,
            "complete": complete,
        }

    def _date_range(self, start: date, end: date):
        """Generate dates in range."""
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)

    def _parse_date(self, date_input: Union[str, date]) -> date:
        """Parse date from string or date object."""
        if isinstance(date_input, date):
            return date_input
        return datetime.strptime(date_input, "%Y-%m-%d").date()
