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
    1. Direct download from the JASMIN TAMSAT server (crop + GeoTIFF)
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

    @property
    def sarra_download_available(self) -> bool:
        """Check if TAMSAT download capability is available.

        Always True — uses direct HTTP download to JASMIN server,
        no external library dependency.
        """
        return True

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
                    progress_callback=kwargs.get('progress_callback'),
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
        progress_callback=None,
        max_workers: int = 4,
    ) -> None:
        """Download TAMSAT daily rainfall and crop to region bounds.

        Two-phase architecture to avoid SIGSEGV from concurrent
        rasterio/PROJ writes:

        Phase 1 — Parallel HTTP download (thread-safe):
            4 threads fetch raw .nc files from JASMIN. Pure HTTP,
            no GDAL/PROJ loaded. 4x network speedup.

        Phase 2 — Sequential crop + convert (rasterio-safe):
            Single-threaded xarray crop + rioxarray GeoTIFF write.
            No concurrency on PROJ/GDAL, no SIGSEGV.

        Args:
            bounds: Bounding box in SARRA-Py format [lat_NW, lon_NW, lat_SE, lon_SE]
            start_date: Start date
            end_date: End date
            output_dir: Output directory for cropped GeoTIFFs
            region_name: Region name for file naming
            progress_callback: Optional callback(current, total, detail)
            max_workers: Number of parallel download threads (default 4)
        """
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        TAMSAT_URL = (
            "https://gws-access.jasmin.ac.uk/public/tamsat/rfe/data/"
            "v3.1/daily/{year}/{month:02d}/"
            "rfe{year}_{month:02d}_{day:02d}.v3.1.nc"
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse SARRA-Py bounds: [lat_NW, lon_NW, lat_SE, lon_SE]
        lat_nw, lon_nw, lat_se, lon_se = bounds
        lat_min = min(lat_nw, lat_se)
        lat_max = max(lat_nw, lat_se)
        lon_min = min(lon_nw, lon_se)
        lon_max = max(lon_nw, lon_se)

        # Partition dates into cached vs. to-download
        dates_to_download = []
        already_have = 0
        current_date = start_date
        while current_date <= end_date:
            tif_name = self.FILE_PATTERN.format(
                version=self.config.version, region=region_name,
                year=current_date.year, month=current_date.month,
                day=current_date.day,
            )
            if (output_dir / tif_name).exists():
                already_have += 1
            else:
                dates_to_download.append(current_date)
            current_date += timedelta(days=1)

        total_days = (end_date - start_date).days + 1

        self.logger.info(
            f"TAMSAT download: {len(dates_to_download)} to fetch, "
            f"{already_have} cached, total={total_days}, "
            f"workers={max_workers}, region={region_name}"
        )

        if not dates_to_download:
            if progress_callback:
                progress_callback(total_days, total_days, '')
            return

        # Temp dir for raw .nc files (Phase 1 output → Phase 2 input)
        nc_dir = output_dir / "_raw_nc"
        nc_dir.mkdir(exist_ok=True)

        # ── Phase 1: Parallel HTTP download (thread-safe, no GDAL) ──

        def _download_nc(target_date):
            """Download a single raw .nc file. Pure HTTP, no rasterio."""
            nc_name = f"rfe{target_date.year}_{target_date.month:02d}_{target_date.day:02d}.nc"
            nc_path = nc_dir / nc_name

            if nc_path.exists():
                return "cached"

            url = TAMSAT_URL.format(
                year=target_date.year,
                month=target_date.month,
                day=target_date.day,
            )

            try:
                resp = requests.get(url, timeout=(30, 60))

                if resp.status_code == 404:
                    return "skipped"

                if resp.status_code >= 500:
                    self.logger.warning(
                        f"TAMSAT server {resp.status_code} for "
                        f"{target_date}, retrying..."
                    )
                    resp = requests.get(url, timeout=(30, 60))
                    if resp.status_code != 200:
                        return f"HTTP {resp.status_code}"

                resp.raise_for_status()
                nc_path.write_bytes(resp.content)
                return "ok"

            except requests.exceptions.Timeout:
                return "timeout"
            except requests.exceptions.RequestException as e:
                return f"download error: {e}"

        dl_ok = 0
        dl_skipped = 0
        errors = []

        self.logger.info(
            f"TAMSAT Phase 1: downloading {len(dates_to_download)} "
            f".nc files ({max_workers} threads)..."
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_download_nc, d): d
                for d in dates_to_download
            }
            for future in as_completed(futures):
                target_date = futures[future]
                try:
                    result = future.result(timeout=90)
                    if result in ("ok", "cached"):
                        dl_ok += 1
                    elif result == "skipped":
                        dl_skipped += 1
                    else:
                        self.logger.warning(
                            f"TAMSAT {target_date}: {result}"
                        )
                        errors.append(f"{target_date}: {result}")
                        dl_skipped += 1
                except Exception as e:
                    self.logger.warning(
                        f"TAMSAT {target_date}: {e}"
                    )
                    errors.append(f"{target_date}: {e}")
                    dl_skipped += 1

                processed = dl_ok + dl_skipped
                if progress_callback and processed % 10 == 0:
                    progress_callback(
                        already_have + processed // 2,
                        total_days,
                        f'TAMSAT rainfall: downloading {processed}/'
                        f'{len(dates_to_download)} files',
                    )

        self.logger.info(
            f"TAMSAT Phase 1 complete: {dl_ok} downloaded, "
            f"{dl_skipped} skipped"
        )

        # ── Phase 2: Sequential crop + GeoTIFF (single-threaded, rasterio-safe) ──

        import xarray as xr
        import rioxarray  # noqa: F401

        converted = 0
        nc_files = sorted(nc_dir.glob("*.nc"))

        self.logger.info(
            f"TAMSAT Phase 2: converting {len(nc_files)} "
            f".nc → .tif (sequential)..."
        )

        for nc_path in nc_files:
            # Parse date from filename: rfe{Y}_{M}_{D}.nc
            stem = nc_path.stem  # rfe2020_01_15
            parts = stem.replace("rfe", "").split("_")
            try:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            except (ValueError, IndexError):
                continue

            tif_name = self.FILE_PATTERN.format(
                version=self.config.version, region=region_name,
                year=y, month=m, day=d,
            )
            tif_path = output_dir / tif_name

            if tif_path.exists():
                converted += 1
                nc_path.unlink()
                continue

            try:
                ds = xr.open_dataset(str(nc_path))
                try:
                    ds_cropped = ds.where(
                        (ds.lat >= lat_min) & (ds.lat <= lat_max)
                        & (ds.lon >= lon_min) & (ds.lon <= lon_max),
                        drop=True,
                    )
                    rfe = ds_cropped["rfe"]
                    rfe = rfe.rio.set_spatial_dims(
                        x_dim="lon", y_dim="lat"
                    )
                    rfe = rfe.rio.write_crs("EPSG:4326")
                    rfe.rio.to_raster(str(tif_path))
                    converted += 1
                except Exception as e:
                    self.logger.warning(
                        f"Failed to convert {nc_path.name}: {e}"
                    )
                    if tif_path.exists():
                        tif_path.unlink()
                finally:
                    ds.close()
            except Exception as e:
                self.logger.warning(
                    f"Failed to open {nc_path.name}: {e}"
                )

            # Clean up raw .nc after conversion
            try:
                nc_path.unlink()
            except OSError:
                pass

            if progress_callback and converted % 10 == 0:
                progress_callback(
                    already_have + converted,
                    total_days,
                    f'TAMSAT rainfall: converting {converted}/'
                    f'{len(nc_files)} files',
                )

        # Clean up temp dir
        try:
            nc_dir.rmdir()
        except OSError:
            pass

        # Final progress
        if progress_callback:
            progress_callback(total_days, total_days, '')

        total_done = already_have + converted
        self.logger.info(
            f"TAMSAT download complete: {total_done}/{total_days} files, "
            f"{dl_skipped} skipped, {len(errors)} errors"
        )
        if errors:
            self.logger.warning(
                f"TAMSAT download errors (first 10): {errors[:10]}"
            )

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
