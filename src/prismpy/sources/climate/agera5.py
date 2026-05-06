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

V2-19 note on CDS API configuration:
    The ``AgERA5Config.dataset`` field below is the SINGLE SOURCE OF TRUTH
    for the CDS dataset name ("sis-agrometeorological-indicators"). The
    previous stale duplicate in ``config/defaults.py`` (CDS_API_CONFIG,
    which had dataset="reanalysis-era5-land") was deleted in V2-19
    CD-13 — it was never imported and drifted from this file's value.
"""

import glob
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from filelock import FileLock, Timeout

from prismpy.models.region import Region
from prismpy.provenance.tracker import ProvenanceTracker
from prismpy.sources.base import DataSource, RetrievalResult
from prismpy.sources.climate._cancel import PipelineCancelled, raise_if_cancelled
# V2-22a B2: cache-isolation helpers (canonical home is tamsat.py;
# imported here to keep the manifest + filelock contract identical
# across both climate sources).
from prismpy.sources.climate.tamsat import (
    DOWNLOAD_LOCK_TIMEOUT_SECONDS,
    MANIFEST_FILENAME,
    MARKER_FILENAME,
    bbox_field_for_log,
    bbox_to_dict,
    cache_lock_path,
    check_cache_manifest,
    count_tif_files,
    delete_marker,
    warn_legacy_cache_once,
    write_cache_manifest,
    write_marker,
)
from prismpy.utils.sanitization import normalize_region_name


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


def _count_agera5_stage_files(
    stages_base: Path,
    region_name: str,
    year: int,
) -> Dict[str, int]:
    """Return per-year file counts across the 4 SARRA_data_download stages.

    V2-22a 1.5 route-back fix — the SARRA_data_download library writes to
    a CWD-relative path (``../data/``) regardless of the ``save_path``
    argument forwarded through ``download_AgERA5_year``. The relocation
    logic in ``AgERA5Source.retrieve`` explicitly handles both candidate
    paths after the fact, which confirms the quirk empirically. Callers
    must pass the SAME base the library actually uses (``Path("../data")``
    in production) so these counts reflect ground truth. Reading from the
    per-run ``output_dir`` returns zeros and pins the var counter at 1/6
    for the entire download phase.

    The zip glob matches the flat layout ``0_downloads/AgERA5_{region}*_{year}.zip``.
    The conversion and output globs are year-scoped via ``f'*_{year}_*.tif'`` so
    year 1's accumulated TIFFs do not leak into year 2's counts (the
    ``2_conversion`` dir is only wiped at the end of the full year loop).

    Returns a dict with keys ``n_zips``, ``n_extracted``, ``n_converted``,
    ``n_output``.
    """
    year_glob = f"*_{year}_*.tif"

    dl_dir = stages_base / "0_downloads"
    n_zips = (
        len(list(dl_dir.glob(f"AgERA5_{region_name}*_{year}.zip")))
        if dl_dir.exists()
        else 0
    )

    ext_dir = stages_base / "1_extraction" / f"AgERA5_{region_name}" / str(year)
    n_extracted = (
        sum(
            len(list(vd.glob("*.nc")))
            for vd in ext_dir.iterdir()
            if vd.is_dir()
        )
        if ext_dir.exists()
        else 0
    )

    conv_dir = stages_base / "2_conversion" / f"AgERA5_{region_name}"
    n_converted = (
        sum(
            len(list(vd.glob(year_glob)))
            for vd in conv_dir.iterdir()
            if vd.is_dir()
        )
        if conv_dir.exists()
        else 0
    )

    out_dir = stages_base / "3_output" / f"AgERA5_{region_name}"
    n_output = (
        sum(1 for _ in out_dir.rglob(year_glob))
        if out_dir.exists()
        else 0
    )

    return {
        "n_zips": n_zips,
        "n_extracted": n_extracted,
        "n_converted": n_converted,
        "n_output": n_output,
    }


@dataclass
class AgERA5Config:
    """Configuration for AgERA5 data access."""

    cds_url: str = "https://cds.climate.copernicus.eu/api/v2"
    dataset: str = "sis-agrometeorological-indicators"
    resolution: float = 0.1  # ~10km
    data_dir: Optional[Path] = None
    use_sarra_download: bool = True
    timeout: int = 600


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
        """Confirm the vendored SARRA_data_download module is importable.

        Post-vendor (prismpy.vendor.sarra_data_download), the import
        always succeeds because the package ships with the prismpy
        wheel. The previous silent-skip pattern returned ``False`` on
        ImportError, which let a missing-library install fall through
        to a 1/4-climate-variables outcome on fresh venvs that had not
        re-applied the local editable install. The current pattern
        raises ``ModuleNotFoundError`` if the vendor goes missing
        (e.g., wheel-build dropped the subpackage) so the configuration
        error surfaces loudly at first call, rather than soft-failing
        into the partial-climate code path. Mirrors the broad-except
        carve-out discipline at the property surface.
        """
        if self._sarra_download_available is None:
            try:
                from prismpy.vendor.sarra_data_download.get_AgERA5_data import (  # noqa: F401
                    download_AgERA5_year,
                )
                self._sarra_download_available = True
            except ImportError as e:
                raise ModuleNotFoundError(
                    "prismpy.vendor.sarra_data_download is required for "
                    "AgERA5 retrieval but did not import. The package "
                    "ships with prismpy; if this raised, the wheel "
                    "build is missing the vendored subpackage. "
                    "Reinstall prismpy or check "
                    "[tool.setuptools.packages.find] in pyproject.toml."
                ) from e
        return self._sarra_download_available

    @property
    def cdsapi_available(self) -> bool:
        """Confirm cdsapi is importable.

        cdsapi is a required dependency declared in pyproject.toml
        (per the prior partial-climate substrate fix). The previous
        silent-skip pattern would return ``False`` on ImportError; the
        current pattern raises ``ModuleNotFoundError`` so a broken
        install surfaces loudly at first call rather than soft-failing
        into the partial-climate code path.
        """
        if self._cdsapi_available is None:
            try:
                import cdsapi  # noqa: F401
                self._cdsapi_available = True
            except ImportError as e:
                raise ModuleNotFoundError(
                    "cdsapi is required for AgERA5 retrieval but did "
                    "not import. The package is declared in "
                    "pyproject.toml [project] dependencies; reinstall "
                    "prismpy with `pip install -e .` to refresh the "
                    "venv."
                ) from e
        return self._cdsapi_available

    def retrieve(
        self,
        region: Region,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        variables: Optional[List[str]] = None,
        download: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
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
        run_id = kwargs.get('run_id')
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
            # V2-22b/P.1 — bbox-keyed cache dir for manual regions
            # so two unnamed-manual projects sharing "Unnamed study
            # area" don't collide on the same on-disk cache. GADM
            # regions fall through to the name-based key unchanged.
            from prismpy.utils.sanitization import region_cache_key_from_region
            safe_name = region_cache_key_from_region(region)
            data_dir = self.cache_dir / "agera5" / f"AgERA5_{safe_name}"

        # Get bounds
        bounds_gis = region.bounds.to_gis_format()
        bounds_sarra_py = region.bounds.to_sarra_py_format()
        bbox_dict = bbox_to_dict(region.bounds)

        metadata["bounds_gis"] = bounds_gis
        metadata["bounds_sarra_py"] = bounds_sarra_py
        metadata["data_dir"] = str(data_dir)

        # Cache-isolation paths (V2-22a B2)
        manifest_path = data_dir / MANIFEST_FILENAME
        marker_path = data_dir / MARKER_FILENAME
        force_redownload = False
        # V2-22b L F-5: initialize `state` so the download-branch
        # legacy-warning guard doesn't NameError when data_dir does
        # not exist (fresh cache).
        state = None

        # V2-22a B2 + Gate B BLOCKER fix: consult the manifest BEFORE the
        # file_info.complete branch. A partial AgERA5 cache (some var
        # subdirs populated, others empty) whose manifest carries a
        # stale bbox must still trigger the wipe + re-download — the
        # old ordering gated the whole manifest check behind completeness
        # and let the SARRA_data_download library's internal caching
        # preserve stale-bbox .tif files in the partial subdirs.
        if data_dir.exists():
            file_info = self._validate_local_files(
                data_dir=data_dir,
                start_date=start_date,
                end_date=end_date,
                variables=variables,
            )
            actual_count = count_tif_files(data_dir)
            state = check_cache_manifest(
                manifest_path,
                marker_path,
                expected_bbox=bbox_dict,
                actual_file_count=actual_count,
                data_files_present=actual_count > 0,
            )

            # Invalidation signals fire independent of completeness.
            force_redownload = state.force_redownload
            metadata["cache_state"] = state.reason

            if state.reason == "bbox_mismatch":
                self.logger.info(
                    "AgERA5 cache bbox mismatch for %s — prior=%s "
                    "requested=%s — re-downloading",
                    region.name,
                    bbox_field_for_log(state.prior_bbox),
                    bbox_field_for_log(bbox_dict),
                )
            elif state.reason == "manifest_corrupt":
                self.logger.warning(
                    "AgERA5 manifest at %s is corrupt/unreadable — "
                    "treating as cold",
                    manifest_path,
                )
            elif state.reason == "marker_present":
                self.logger.warning(
                    "AgERA5 marker present at %s (started_at=%s) — "
                    "prior download interrupted; re-downloading",
                    marker_path,
                    state.marker_started_at,
                )
            elif state.reason == "file_count_drift":
                # Gate B LOW 1: include expected + actual in the log
                self.logger.warning(
                    "AgERA5 manifest file_count drift at %s — expected=%s "
                    "(from manifest) actual=%d (disk count) — treating as "
                    "cold",
                    manifest_path,
                    state.expected_file_count,
                    actual_count,
                )

            # Cache hit only when BOTH the manifest says OK AND the
            # completeness check says we have every date we need.
            if state.cache_hit and file_info["complete"]:
                if state.reason == "legacy_assume_valid":
                    warn_legacy_cache_once(data_dir, self.logger)

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

            if not file_info["complete"]:
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

            # V2-22a B2: per-source-per-region filelock. Different sources
            # (TAMSAT vs. AgERA5) on the same region run concurrently —
            # separate lock files (.tamsat-<region>.lock vs.
            # .agera5-<region>.lock) keep a single SARRA-Py run from
            # self-blocking when it sequences TAMSAT then AgERA5.
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            lock_path = cache_lock_path(self.cache_dir, source=self.NAME, region_name=region)
            lock = FileLock(str(lock_path))

            try:
                with lock.acquire(timeout=DOWNLOAD_LOCK_TIMEOUT_SECONDS):
                    # Marker BEFORE any data write so a SIGKILL during the
                    # download leaves a "this cache may be partial" signal
                    # the next reader will respect (AC 1.7.3c).
                    write_marker(
                        marker_path,
                        source=self.NAME,
                        region_name=region.name,
                        run_id=run_id,
                    )

                    # V2-22b L F-5: emit legacy-cache warning when we
                    # reach the download branch on a pre-B2 cache —
                    # unconditional of force_redownload. See tamsat.py
                    # counterpart for full rationale.
                    if state is not None and state.reason == "legacy_assume_valid":
                        warn_legacy_cache_once(data_dir, self.logger)

                    # AC 1.7.3e (AgERA5 flavor): force_redownload wipes the
                    # per-region .tif files only — staging dirs at
                    # Path("../data") are CWD-relative and may be shared
                    # with concurrently-running AgERA5 calls on a different
                    # region; touching them is out of B2 scope (Drift 4
                    # backlog). The opaque SARRA_data_download library may
                    # have its own skip-if-exists shortcuts — wiping the
                    # per-region cache contents is the safe path.
                    if force_redownload and data_dir.exists():
                        for tif in data_dir.rglob("*.tif"):
                            try:
                                tif.unlink()
                            except OSError:
                                pass

                    # Codex Path A — pass the SAME cache key used to
                    # build `data_dir` so SARRA_data_download's
                    # library-side subdir creation (`AgERA5_{key}`)
                    # lands in the directory `retrieve()` later
                    # validates and manifests. Using `region.name`
                    # here would split brain: manual-unnamed runs
                    # wrote to `AgERA5_Unnamed study area/` while
                    # data_dir pointed at `AgERA5_manual_…/`.
                    from prismpy.utils.sanitization import region_cache_key_from_region
                    self._download_agera5(
                        bounds=bounds_sarra_py,
                        start_date=start_date,
                        end_date=end_date,
                        output_dir=data_dir.parent,  # Library creates subdir
                        region_name=region_cache_key_from_region(region),
                        progress_callback=kwargs.get('progress_callback'),
                        cancel_check=cancel_check,
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
                    if force_redownload:
                        metadata["force_redownload"] = True

                    # AC 1.7.3d: manifest replace BEFORE marker delete.
                    # Marker stays on disk if any of these fail so the next
                    # reader sees the cache as cold.
                    write_cache_manifest(
                        manifest_path,
                        source=self.NAME,
                        region_name=region.name,
                        bbox=bbox_dict,
                        start_date=start_date,
                        end_date=end_date,
                        run_id=run_id,
                        file_count=count_tif_files(data_dir),
                    )
                    delete_marker(marker_path)

                    return self.create_result(
                        success=True,
                        data=agera5_data,
                        output_path=data_dir,
                        warnings=warnings,
                        metadata=metadata,
                    )

            except Timeout:
                return self.create_result(
                    success=False,
                    errors=[
                        "Another run on this region is downloading data "
                        "(~90 min max). Please wait and retry."
                    ],
                    warnings=warnings,
                    metadata=metadata,
                )
            except PipelineCancelled:
                # V2-22b L F-2: carve-out so user cancel unwinds past
                # this broad except instead of being rewritten as
                # "Download failed: {e}".
                raise
            except (ImportError, ModuleNotFoundError):
                # Mirror the tamsat carve-out: an undeclared
                # transitive dep is a configuration error and must
                # propagate so pip / CI / startup surfaces it loudly
                # rather than soft-failing into the placeholder
                # climate dict.
                raise
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
        progress_callback=None,
        cancel_check=None,
    ) -> None:
        """Download AgERA5 data using SARRA_data_download library.

        Args:
            bounds: Bounding box in SARRA-Py format
            start_date: Start date
            end_date: End date
            output_dir: Output directory
            region_name: Region name
            progress_callback: Optional callback(current, total) for progress
        """
        import shutil
        from prismpy.vendor.sarra_data_download.get_AgERA5_data import (
            download_AgERA5_year,
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        area = {region_name: bounds}

        import threading as _threading
        import logging as _logging
        from pathlib import Path as _P

        total_days = (end_date - start_date).days + 1
        estimated_total = total_days * len(AGERA5_VARIABLES)

        # Variable names the library downloads (6 total)
        var_labels = {
            '2m_temperature_24_hour_minimum': 'tmin',
            '2m_temperature_24_hour_maximum': 'tmax',
            'solar_radiation_flux_daily': 'srad',
            'vapour_pressure_24_hour_mean': 'humidity',
            '10m_wind_speed_24_hour_mean': 'wind',
            '2m_temperature_24_hour_mean': 'tmean',
        }

        years = list(range(start_date.year, end_date.year + 1))
        for i, year in enumerate(years):
            # V2-22b L (AC L.4, BLOCKER 2): year-top cancel MUST raise,
            # not return. A bare `return` drops back into retrieve()
            # which then runs _validate_local_files + write_cache_manifest
            # + delete_marker + success return — treating the cancelled
            # download as successful (silent contamination). Raising
            # unwinds past all of that; the B2 marker stays on disk so
            # the next reader treats the cache as cold.
            raise_if_cancelled(cancel_check, f"agera5.year={year}")

            # V2-22a 1.5 — W4 (main-thread year-header callback) deleted.
            # _phase_monitor is now the sole writer of substage.detail for
            # AgERA5 (see ownership comment below). The ≤8s reporting lag
            # at year boundaries is acceptable.
            self.logger.info(
                f"Downloading AgERA5 data for {year} "
                f"(year {i + 1}/{len(years)})..."
            )

            # ── Sophisticated progress: CDS log interceptor + multi-stage scanner ──
            # Captures CDS request lifecycle (queued → running → successful) and
            # scans all 4 library stages to determine current phase.
            stop_monitor = _threading.Event()
            # V2-22a 1.5 — year-tagged state: 'year' key added so
            # _phase_monitor reads are explicit about per-year scoping,
            # and reset at year-loop entry prevents cross-year carry-over.
            _cds_state = {'year': year, 'status': 'initializing', 'var_index': 0}

            # CDS log interceptor: captures per-request status transitions
            class _CDSHandler(_logging.Handler):
                def emit(self, record):
                    msg = record.getMessage()
                    if 'status has been updated to' in msg:
                        for token in ['accepted', 'running', 'successful', 'failed']:
                            if token in msg:
                                _cds_state['status'] = token
                                break

            cds_handler = _CDSHandler()
            cds_handler.setLevel(_logging.INFO)
            for logger_name in ['ecmwf.datastores.legacy_client', 'cdsapi']:
                _logging.getLogger(logger_name).addHandler(cds_handler)

            # V2-22b L (Gate A round 1 MEDIUM 2): cancel observed after
            # log-handler attachment but before the monitor thread
            # starts must clean up the handler in its own hands — the
            # `finally` block below only runs after download_AgERA5_year
            # has been called. Without this cleanup, cancel leaks a
            # handler registration on the CDS logger for the rest of
            # the process.
            try:
                raise_if_cancelled(
                    cancel_check, f"agera5.before_monitor.year={year}",
                )
            except PipelineCancelled:
                for logger_name in ['ecmwf.datastores.legacy_client', 'cdsapi']:
                    _logging.getLogger(logger_name).removeHandler(cds_handler)
                raise

            # V2-22a 1.5 — SINGLE WRITER OWNERSHIP
            # _phase_monitor is the sole writer of progress_callback (and
            # therefore substage.detail) during AgERA5 downloads. The
            # main-thread year-header callback (W4) was deleted above so
            # labels no longer alternate between this monitor's rich CDS
            # state and a main-thread file-count fallback. This single
            # writer is what lets item 1.1's detail-diff guard correctly
            # preserve updated_at during CDS queue waits — without the
            # consolidation, labels from two sources are never byte-
            # identical and the guard never fires. AC 1.5.6 grep test
            # locks this invariant structurally (asserts zero
            # progress_callback() call sites outside _phase_monitor).
            def _phase_monitor():
                """Scan all 4 library stages + CDS state every 8s."""
                prev_zips = 0
                # Post-vendor save_path forwarding — the vendored
                # ``download_AgERA5_year`` now propagates ``save_path``
                # through every nested call, so each stage writes
                # under the per-region ``output_dir`` instead of a
                # CWD-relative ``../data/`` tree. The previous binding
                # was ``stages_base = Path("../data")`` to match the
                # upstream-verbatim quirk where ``save_path`` was
                # silently dropped; that quirk is fixed at the vendor
                # layer now, so the monitor reads from the same
                # ``output_dir`` the executor already passed in.
                stages_base = output_dir
                while not stop_monitor.wait(8):
                    try:
                        counts = _count_agera5_stage_files(
                            stages_base, region_name, year,
                        )
                        n_zips = counts['n_zips']
                        n_extracted = counts['n_extracted']
                        n_converted = counts['n_converted']
                        n_output = counts['n_output']

                        # Track variable progression
                        if n_zips > prev_zips:
                            _cds_state['var_index'] = n_zips
                            _cds_state['status'] = 'initializing'
                            prev_zips = n_zips

                        # Build rich detail string
                        yr_label = f'year {i + 1}/{len(years)}'
                        cds_status = _cds_state['status']
                        var_idx = _cds_state.get('var_index', 0)

                        if n_converted > 0 or n_extracted > 0:
                            # Post-download phase
                            if n_converted > n_extracted:
                                detail = (
                                    f'{yr_label} — converting to GeoTIFF '
                                    f'({n_converted} files)'
                                )
                            else:
                                detail = (
                                    f'{yr_label} — extracting NetCDF '
                                    f'({n_extracted} files)'
                                )
                        elif n_zips > 0 and cds_status in ('initializing', 'accepted', 'running'):
                            # Downloading next variable from CDS
                            cds_label = {
                                'initializing': 'preparing request',
                                'accepted': 'queued on CDS',
                                'running': 'downloading',
                            }.get(cds_status, cds_status)
                            detail = (
                                f'{yr_label} — var {n_zips + 1}/6 '
                                f'— {cds_label}'
                            )
                        elif n_zips > 0 and cds_status == 'successful':
                            detail = (
                                f'{yr_label} — {n_zips}/6 vars '
                                f'downloaded'
                            )
                        else:
                            # First variable, waiting for CDS
                            cds_label = {
                                'initializing': 'preparing request',
                                'accepted': 'queued on CDS',
                                'running': 'downloading',
                                'successful': 'downloaded',
                            }.get(cds_status, 'connecting...')
                            detail = (
                                f'{yr_label} — var 1/6 '
                                f'— {cds_label}'
                            )

                        if progress_callback:
                            progress_callback(
                                n_output, estimated_total,
                                f'AgERA5: {detail}',
                            )
                        self.logger.info(
                            f'  AgERA5 {year}: {detail} '
                            f'({n_output}/{estimated_total} output)'
                        )
                    except OSError:
                        pass

            monitor = _threading.Thread(target=_phase_monitor, daemon=True)
            monitor.start()

            # V2-22b L (AC L.4): second year-level check — catches
            # cancel fired between the pre-monitor check and the
            # opaque `download_AgERA5_year` submit. Library opacity
            # means 12-30 min per year is intrinsic; this check is
            # the last observation point before that window opens.
            # The monitor thread is already running, so a raise here
            # must also stop it (the outer `finally` at the try below
            # handles stop_monitor.set() + monitor.join; we need to
            # fire it manually here before raising).
            try:
                raise_if_cancelled(
                    cancel_check, f"agera5.before_cds.year={year}",
                )
            except PipelineCancelled:
                stop_monitor.set()
                monitor.join(timeout=2)
                for logger_name in ['ecmwf.datastores.legacy_client', 'cdsapi']:
                    _logging.getLogger(logger_name).removeHandler(cds_handler)
                raise

            # Call library directly — no timeout. CDS requests take
            # 2-5 min each × 6 variables = 12-30 min per year.
            # Heartbeat + watchdog provide the safety net. Cancel
            # granularity here is year-boundary — the library call is
            # opaque (cdsapi internals); 12-30 min per year is inherent
            # to CDS queue + 6-variable sequential fetch. Do not try to
            # interrupt — orphan threads consume CDS quota across
            # cancellations. User-facing expectation documented in AC L.4.
            try:
                download_AgERA5_year(
                    query_year=year,
                    area=area,
                    selected_area=region_name,
                    save_path=str(output_dir),
                    version="SARRA-Py",
                )
            finally:
                stop_monitor.set()
                monitor.join(timeout=2)
                for logger_name in ['ecmwf.datastores.legacy_client', 'cdsapi']:
                    _logging.getLogger(logger_name).removeHandler(cds_handler)

            self.logger.info(f"AgERA5 {year} download complete.")

        # Post-vendor save_path forwarding — the vendored
        # ``download_AgERA5_year`` writes stage outputs under the
        # per-region ``output_dir`` instead of the CWD-relative
        # ``../data/`` tree the upstream copy used. The relocated
        # final GeoTIFFs land at ``output_dir / "3_output" /
        # f"AgERA5_{region_name}"``; the legacy ``Path("../data/3_output")``
        # fallback that previously bridged the CWD-relative quirk is
        # retired.
        target_dir = output_dir / f"AgERA5_{region_name}"
        hardcoded_dir = output_dir / "3_output" / f"AgERA5_{region_name}"

        if hardcoded_dir.exists():
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
            # Clean up the per-region stage directories so the cache
            # only carries the relocated final outputs. ``ignore_errors``
            # covers the case where one of the four stage dirs was
            # never created (e.g., the vendor short-circuited before
            # the conversion or output stage ran).
            shutil.rmtree(output_dir / "3_output", ignore_errors=True)
            shutil.rmtree(output_dir / "2_conversion", ignore_errors=True)
            shutil.rmtree(output_dir / "1_extraction", ignore_errors=True)
            shutil.rmtree(output_dir / "0_downloads", ignore_errors=True)

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
