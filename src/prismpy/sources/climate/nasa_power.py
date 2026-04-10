"""
NASA POWER climate data source retriever.

This module provides functionality to download daily climate data from the
NASA POWER (Prediction of Worldwide Energy Resources) API.

Reference: PYTHIA/02-WEATHER-PREPARATION/ and ACEA/02-CLIMATE-PREPARATION/

V2-19 note on climate validation:
    After V2-19 deletions (config/defaults.py CLIMATE_VALIDATION_RANGES
    removed as dead code; harmonizers/quality.py CLIMATE_LIMITS module
    deleted wholesale), climate validation is distributed as inline
    ad-hoc checks below (search for "range" or "bounds" in the module).
    No centralized source-of-truth table exists. Add one via
    sources/climate/_quality.py if centralization is ever needed
    (tracked as V2-20 carryover).
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.region import Region
from prismpy.provenance.tracker import DecisionType, ProvenanceTracker
from prismpy.sources.base import DataSource, RetrievalResult


logger = logging.getLogger(__name__)


# NASA POWER API parameter mappings
NASA_POWER_PARAMETERS = {
    "tmax": "T2M_MAX",
    "tmin": "T2M_MIN",
    "tmean": "T2M",
    "tdew": "T2MDEW",
    "precip": "PRECTOTCORR",
    "wind": "WS2M",
    "srad": "ALLSKY_SFC_SW_DWN",
    "rh": "RH2M",
    "pressure": "PS",
}

# Reverse mapping for response parsing
NASA_TO_INTERNAL = {v: k for k, v in NASA_POWER_PARAMETERS.items()}

# Default parameters to retrieve
DEFAULT_PARAMETERS = [
    "T2M_MAX",
    "T2M_MIN",
    "T2M",
    "T2MDEW",
    "PRECTOTCORR",
    "WS2M",
    "ALLSKY_SFC_SW_DWN",
    "RH2M",
]


@dataclass
class NASAPowerConfig:
    """Configuration for NASA POWER API requests."""

    base_url: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    community: str = "AG"  # Agriculture community
    timeout: int = 120  # Request timeout in seconds
    retry_count: int = 3  # Number of retries on failure
    retry_delay: float = 5.0  # Delay between retries (seconds)
    request_delay: float = 1.0  # Delay between requests (rate limiting)
    parameters: List[str] = None  # Parameters to retrieve

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = DEFAULT_PARAMETERS.copy()


class NASAPowerSource(DataSource):
    """Data source for NASA POWER daily climate data.

    NASA POWER provides global daily climate data including:
    - Temperature (min, max, mean, dew point)
    - Precipitation
    - Solar radiation
    - Wind speed
    - Relative humidity

    Data is available from 1981-01-01 for most parameters, and from
    1984-01-01 for solar radiation (ALLSKY_SFC_SW_DWN).

    Attributes:
        NAME: Data source identifier
        EARLIEST_DATE: Earliest available date for full data
        SRAD_EARLIEST_DATE: Earliest date for solar radiation
    """

    NAME = "nasa_power"
    EARLIEST_DATE = date(1981, 1, 1)
    SRAD_EARLIEST_DATE = date(1984, 1, 1)

    def __init__(
        self,
        config: Optional[NASAPowerConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the NASA POWER data source.

        Args:
            config: API configuration
            cache_dir: Directory for caching downloaded data
            provenance: Provenance tracker for audit trail
        """
        super().__init__(cache_dir=cache_dir, provenance=provenance)
        self.config = config or NASAPowerConfig()

    def retrieve(
        self,
        region: Region = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        parameters: Optional[List[str]] = None,
        location_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve climate data from NASA POWER API.

        Can retrieve data for a single point (lat/lon) or for the center
        of a Region's bounding box. Uses per-year caching so overlapping
        date ranges reuse already-downloaded years.

        Args:
            region: Region object (uses center point if lat/lon not provided)
            lat: Latitude of point
            lon: Longitude of point
            start_date: Start date (YYYY-MM-DD string or date object)
            end_date: End date (YYYY-MM-DD string or date object)
            parameters: List of NASA POWER parameters to retrieve
            location_id: Optional ID for the location (for multi-point retrieval)
            use_cache: Use cached data if available
            **kwargs: Additional parameters

        Returns:
            RetrievalResult containing ClimateTimeSeries object
        """
        errors = []
        warnings = []
        metadata = {"source": self.NAME}

        # Determine coordinates
        if lat is None or lon is None:
            if region is None:
                return self.create_result(
                    success=False,
                    errors=["Either (lat, lon) or region must be provided"],
                )
            # Use center of region
            lon, lat = region.bounds.center
            metadata["coordinate_source"] = "region_center"
            # V2-19 C6 (RN-04): NASA POWER center-point sampling rationale
            if self.provenance:
                self.provenance.record_decision(
                    decision_type=DecisionType.SOURCE_SELECTION,
                    description=(
                        "NASA POWER: single time series from region centre"
                    ),
                    rationale=(
                        "The region bounding-box centre is used as the "
                        "single query point for NASA POWER's /point "
                        "endpoint. NASA POWER provides gridded reanalysis "
                        "at 0.5\u00b0 (~56 km) native resolution (Stackhouse "
                        "et al. 2018). For regions smaller than one NASA "
                        "POWER grid cell (~3000 km\u00b2), the center-point "
                        "returns the same data as any other point in the "
                        "region — spatial homogeneity is a property of "
                        "the source, not an assumption. Valid for "
                        "administrative units up to ~ADM2 in West Africa "
                        "(Koutiala \u2248 7000 km\u00b2, spans ~2 POWER cells). "
                        "NOT valid for large regions spanning multiple "
                        "climate zones (e.g., national-scale ADM0) where "
                        "interior climate variability exceeds the 0.5\u00b0 "
                        "grid — use AgERA5 gridded fields or multi-point "
                        "sampling instead. A reviewer would ask: 'Is one "
                        "point representative of the study area?' For "
                        "ADM2 regions in the Sahel, yes — NASA POWER's "
                        "56 km cell typically covers the entire district."
                    ),
                    alternatives=[
                        "Query NASA POWER at every grid cell (API-expensive, "
                        "same 0.5\u00b0 data for small regions)",
                        "AgERA5 gridded fields (0.1\u00b0 resolution, higher fidelity)",
                        "Multi-point sampling with spatial interpolation",
                    ],
                    reference=(
                        "prismpy.sources.climate.nasa_power.NASAPowerSource."
                        "retrieve (region.bounds.center)"
                    ),
                    artifact_id="climate",
                )
        else:
            metadata["coordinate_source"] = "explicit"

        metadata["latitude"] = lat
        metadata["longitude"] = lon

        # Parse dates
        start_date = self._parse_date(start_date) if start_date else self.SRAD_EARLIEST_DATE
        end_date = self._parse_date(end_date) if end_date else date.today() - timedelta(days=1)

        # Validate date range
        if start_date < self.SRAD_EARLIEST_DATE:
            warnings.append(
                f"Start date {start_date} is before SRAD availability ({self.SRAD_EARLIEST_DATE}). "
                "Solar radiation data may be missing."
            )

        metadata["start_date"] = start_date.isoformat()
        metadata["end_date"] = end_date.isoformat()

        # Determine parameters
        params_to_fetch = parameters or self.config.parameters

        # Determine which years span the requested range
        years_needed = list(range(start_date.year, end_date.year + 1))
        cached_records = []
        years_to_fetch = []

        if use_cache:
            for year in years_needed:
                year_cache_path = self._get_year_cache_path(lat, lon, year)
                if year_cache_path.exists():
                    try:
                        year_ts = self._load_cached_climate(year_cache_path)
                        cached_records.extend(year_ts.records)
                        self.logger.info(f"Loaded cached year {year} for ({lat}, {lon})")
                    except Exception as e:
                        warnings.append(f"Cache read failed for year {year}: {e}")
                        years_to_fetch.append(year)
                else:
                    # Fallback: check old per-date-range cache for the full span
                    years_to_fetch.append(year)

            # If all years cached, check old-style cache as additional fallback
            if years_to_fetch and not cached_records:
                old_cache_path = self._get_climate_cache_path(lat, lon, start_date, end_date)
                if old_cache_path.exists():
                    try:
                        cached_data = self._load_cached_climate(old_cache_path)
                        self.logger.info(
                            f"Loaded old-style cached climate data for ({lat}, {lon})"
                        )
                        # Migrate: save as per-year cache files
                        self._save_per_year_cache(cached_data, lat, lon)
                        metadata["from_cache"] = True
                        metadata["migrated_from_old_cache"] = True
                        # Filter to requested date range
                        filtered = self._filter_records_to_range(
                            cached_data.records, start_date, end_date
                        )
                        cached_data.records = filtered
                        return self.create_result(
                            success=True,
                            data=cached_data,
                            metadata=metadata,
                        )
                    except Exception as e:
                        warnings.append(f"Old cache read failed, fetching fresh: {e}")
        else:
            years_to_fetch = years_needed

        # Fetch missing years from API
        for year in years_to_fetch:
            year_start = date(year, 1, 1)
            year_end = date(year, 12, 31)
            # Clamp to actual request bounds for first/last year isn't needed —
            # we always cache full years for maximum reuse
            try:
                nasa_data = self._fetch_from_api(
                    lat=lat,
                    lon=lon,
                    start_date=year_start,
                    end_date=year_end,
                    parameters=params_to_fetch,
                )
                year_ts = self._convert_to_climate_timeseries(
                    nasa_data=nasa_data,
                    lat=lat,
                    lon=lon,
                    location_id=location_id,
                )
                cached_records.extend(year_ts.records)

                # Cache this year
                year_cache_path = self._get_year_cache_path(lat, lon, year)
                try:
                    self._save_climate_cache(year_ts, year_cache_path)
                except Exception as e:
                    warnings.append(f"Failed to cache year {year}: {e}")

                # Rate limiting between year requests
                if year != years_to_fetch[-1]:
                    time.sleep(self.config.request_delay)

            except Exception as e:
                return self.create_result(
                    success=False,
                    errors=[f"API request failed for year {year}: {e}"],
                    metadata=metadata,
                )

        # Filter combined records to the exact requested date range and sort
        filtered_records = self._filter_records_to_range(
            cached_records, start_date, end_date
        )

        climate_ts = ClimateTimeSeries(
            location_id=location_id or 0,
            lat=lat,
            lon=lon,
            source=self.NAME,
            records=filtered_records,
        )

        # Validate the data
        validation_errors = self.validate(climate_ts)
        if validation_errors:
            warnings.extend(validation_errors)

        # Record provenance
        if self.provenance:
            self.provenance.record_retrieval(
                source=self.NAME,
                parameters={
                    "lat": lat,
                    "lon": lon,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "parameters": params_to_fetch,
                },
                output_path=None,
                decisions=[],
            )

        metadata["from_cache"] = len(years_to_fetch) == 0
        metadata["years_cached"] = [y for y in years_needed if y not in years_to_fetch]
        metadata["years_fetched"] = years_to_fetch
        metadata["record_count"] = len(climate_ts.records)
        metadata["parameters_retrieved"] = params_to_fetch

        return self.create_result(
            success=True,
            data=climate_ts,
            warnings=warnings,
            metadata=metadata,
        )

    def retrieve_grid(
        self,
        points: List[Tuple[float, float]],
        start_date: Union[str, date],
        end_date: Union[str, date],
        parameters: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
        **kwargs,
    ) -> Dict[int, RetrievalResult]:
        """Retrieve climate data for multiple points.

        Args:
            points: List of (lat, lon) tuples
            start_date: Start date
            end_date: End date
            parameters: Parameters to retrieve
            progress_callback: Optional callback(current, total) for progress
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping point index to RetrievalResult
        """
        results = {}
        total = len(points)

        for i, (lat, lon) in enumerate(points):
            if progress_callback:
                progress_callback(i + 1, total)

            result = self.retrieve(
                lat=lat,
                lon=lon,
                start_date=start_date,
                end_date=end_date,
                parameters=parameters,
                location_id=i,
                **kwargs,
            )
            results[i] = result

            # Rate limiting between requests
            if i < total - 1:
                time.sleep(self.config.request_delay)

        return results

    def validate(self, data: Any) -> List[str]:
        """Validate retrieved climate data.

        Args:
            data: ClimateTimeSeries object to validate

        Returns:
            List of validation warning/error messages
        """
        warnings = []

        if not isinstance(data, ClimateTimeSeries):
            return [f"Expected ClimateTimeSeries, got {type(data)}"]

        if not data.records:
            return ["No climate records in time series"]

        # Check for missing values
        missing_counts = {}
        for record in data.records:
            for field in ["tmax", "tmin", "precip", "srad"]:
                value = getattr(record, field, None)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    missing_counts[field] = missing_counts.get(field, 0) + 1

        for field, count in missing_counts.items():
            pct = 100 * count / len(data.records)
            if pct > 5:
                warnings.append(f"{field} has {count} missing values ({pct:.1f}%)")

        # Validate value ranges
        for record in data.records:
            if record.tmax is not None and record.tmin is not None:
                if record.tmin > record.tmax:
                    warnings.append(
                        f"tmin > tmax on {record.date}: {record.tmin} > {record.tmax}"
                    )
            if record.precip is not None and record.precip < 0:
                warnings.append(f"Negative precipitation on {record.date}: {record.precip}")
            if record.srad is not None and record.srad < 0:
                warnings.append(f"Negative solar radiation on {record.date}: {record.srad}")

        return warnings

    def _fetch_from_api(
        self,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
        parameters: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Fetch data from NASA POWER API.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date
            end_date: End date
            parameters: Parameters to fetch

        Returns:
            Dictionary of parameter data {param: {date_str: value}}

        Raises:
            Exception: If API request fails after retries
        """
        # Format dates as YYYYMMDD
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        params = {
            "start": start_str,
            "end": end_str,
            "latitude": lat,
            "longitude": lon,
            "community": self.config.community,
            "parameters": ",".join(parameters),
            "format": "JSON",
            "header": "false",
        }

        self.logger.info(
            f"Fetching NASA POWER data for ({lat}, {lon}) "
            f"from {start_date} to {end_date}"
        )

        last_error = None
        for attempt in range(self.config.retry_count):
            try:
                response = requests.get(
                    self.config.base_url,
                    params=params,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                data = response.json()

                # Validate response structure
                if "properties" not in data or "parameter" not in data["properties"]:
                    raise ValueError("Invalid response format from NASA POWER")

                return data["properties"]["parameter"]

            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.config.timeout}s"
                self.logger.warning(f"Attempt {attempt + 1} timeout, retrying...")
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
            except ValueError as e:
                last_error = str(e)
                self.logger.warning(f"Attempt {attempt + 1} invalid response: {e}")

            if attempt < self.config.retry_count - 1:
                time.sleep(self.config.retry_delay)

        raise Exception(f"API request failed after {self.config.retry_count} attempts: {last_error}")

    def _convert_to_climate_timeseries(
        self,
        nasa_data: Dict[str, Dict[str, float]],
        lat: float,
        lon: float,
        location_id: Optional[int] = None,
    ) -> ClimateTimeSeries:
        """Convert NASA POWER response to ClimateTimeSeries.

        Args:
            nasa_data: NASA POWER API response data
            lat: Latitude
            lon: Longitude
            location_id: Optional location identifier

        Returns:
            ClimateTimeSeries object
        """
        # Get dates from first available parameter
        first_param = next(iter(nasa_data.keys()))
        date_strings = sorted(nasa_data[first_param].keys())

        records = []
        for date_str in date_strings:
            # Parse date
            record_date = datetime.strptime(date_str, "%Y%m%d").date()

            # Extract values with None handling for missing data
            def get_value(param_name: str) -> Optional[float]:
                if param_name not in nasa_data:
                    return None
                value = nasa_data[param_name].get(date_str)
                # NASA POWER uses -999 for missing values
                if value is None or value == -999 or value == -999.0:
                    return None
                return float(value)

            record = ClimateRecord(
                date=record_date,
                tmax=get_value("T2M_MAX"),
                tmin=get_value("T2M_MIN"),
                tmean=get_value("T2M"),
                precip=get_value("PRECTOTCORR"),
                srad=get_value("ALLSKY_SFC_SW_DWN"),
                wind=get_value("WS2M"),
                rh=get_value("RH2M"),
                tdew=get_value("T2MDEW"),
            )
            records.append(record)

        return ClimateTimeSeries(
            location_id=location_id or 0,
            lat=lat,
            lon=lon,
            source=self.NAME,
            records=records,
        )

    def _parse_date(self, date_input: Union[str, date]) -> date:
        """Parse date from string or date object.

        Args:
            date_input: Date string (YYYY-MM-DD) or date object

        Returns:
            date object
        """
        if isinstance(date_input, date):
            return date_input
        return datetime.strptime(date_input, "%Y-%m-%d").date()

    def _filter_records_to_range(
        self,
        records: List[ClimateRecord],
        start_date: date,
        end_date: date,
    ) -> List[ClimateRecord]:
        """Filter and sort records to the requested date range."""
        filtered = [r for r in records if start_date <= r.date <= end_date]
        filtered.sort(key=lambda r: r.date)
        return filtered

    def _get_year_cache_path(self, lat: float, lon: float, year: int) -> Path:
        """Get per-year cache file path for climate data."""
        lat_key = f"{lat:.4f}".replace(".", "_").replace("-", "m")
        lon_key = f"{lon:.4f}".replace(".", "_").replace("-", "m")
        filename = f"nasapower_{lat_key}_{lon_key}_{year}.json"
        return self.cache_dir / "climate" / filename

    def _save_per_year_cache(
        self, climate_ts: ClimateTimeSeries, lat: float, lon: float
    ) -> None:
        """Split a ClimateTimeSeries into per-year cache files."""
        by_year: Dict[int, List[ClimateRecord]] = {}
        for r in climate_ts.records:
            by_year.setdefault(r.date.year, []).append(r)
        for year, records in by_year.items():
            year_ts = ClimateTimeSeries(
                location_id=climate_ts.location_id,
                lat=climate_ts.lat,
                lon=climate_ts.lon,
                source=climate_ts.source,
                records=sorted(records, key=lambda r: r.date),
            )
            year_path = self._get_year_cache_path(lat, lon, year)
            try:
                self._save_climate_cache(year_ts, year_path)
            except Exception as e:
                self.logger.warning(f"Failed to migrate cache for year {year}: {e}")

    def _get_climate_cache_path(
        self,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
    ) -> Path:
        """Get cache file path for climate data.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date
            end_date: End date

        Returns:
            Path to cache file
        """
        # Round coordinates for cache key
        lat_key = f"{lat:.4f}".replace(".", "_").replace("-", "m")
        lon_key = f"{lon:.4f}".replace(".", "_").replace("-", "m")
        date_key = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        filename = f"nasapower_{lat_key}_{lon_key}_{date_key}.json"
        return self.cache_dir / "climate" / filename

    def _save_climate_cache(self, climate_ts: ClimateTimeSeries, cache_path: Path) -> None:
        """Save climate data to cache file.

        Args:
            climate_ts: ClimateTimeSeries object
            cache_path: Path to cache file
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        cache_data = {
            "location_id": climate_ts.location_id,
            "lat": climate_ts.lat,
            "lon": climate_ts.lon,
            "source": climate_ts.source,
            "records": [
                {
                    "date": r.date.isoformat(),
                    "tmax": r.tmax,
                    "tmin": r.tmin,
                    "tmean": r.tmean,
                    "precip": r.precip,
                    "srad": r.srad,
                    "wind": r.wind,
                    "rh": r.rh,
                    "tdew": r.tdew,
                    "et0": r.et0,
                }
                for r in climate_ts.records
            ],
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

        self.logger.debug(f"Cached climate data to {cache_path}")

    def _load_cached_climate(self, cache_path: Path) -> ClimateTimeSeries:
        """Load climate data from cache file.

        Args:
            cache_path: Path to cache file

        Returns:
            ClimateTimeSeries object
        """
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = [
            ClimateRecord(
                date=datetime.strptime(r["date"], "%Y-%m-%d").date(),
                tmax=r.get("tmax"),
                tmin=r.get("tmin"),
                tmean=r.get("tmean"),
                precip=r.get("precip"),
                srad=r.get("srad"),
                wind=r.get("wind"),
                rh=r.get("rh"),
                tdew=r.get("tdew"),
                et0=r.get("et0"),
            )
            for r in data["records"]
        ]

        return ClimateTimeSeries(
            location_id=data["location_id"],
            lat=data["lat"],
            lon=data["lon"],
            source=data["source"],
            records=records,
        )

    def to_dataframe(self, climate_ts: ClimateTimeSeries) -> pd.DataFrame:
        """Convert ClimateTimeSeries to pandas DataFrame.

        Args:
            climate_ts: ClimateTimeSeries object

        Returns:
            DataFrame with date index and climate columns
        """
        data = {
            "date": [r.date for r in climate_ts.records],
            "tmax": [r.tmax for r in climate_ts.records],
            "tmin": [r.tmin for r in climate_ts.records],
            "tmean": [r.tmean for r in climate_ts.records],
            "precip": [r.precip for r in climate_ts.records],
            "srad": [r.srad for r in climate_ts.records],
            "wind": [r.wind for r in climate_ts.records],
            "rh": [r.rh for r in climate_ts.records],
            "tdew": [r.tdew for r in climate_ts.records],
            "et0": [r.et0 for r in climate_ts.records],
        }

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        return df
