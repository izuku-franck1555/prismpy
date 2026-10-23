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
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

from prismpy.sources.climate._cancel import PipelineCancelled, raise_if_cancelled
from prismpy.sources.climate._availability import (
    DEFAULT_LAG_DAYS,
    nasa_power_latest_available_date,
)
from prismpy.sources.climate._gapfill import (
    fill_short_gaps,
    normalize_missing,
    recompute_means,
)
from prismpy.sources.climate.errors import NasaPowerScaledPrecipError

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

# NASA POWER intermittently serves PRECTOTCORR ×24 in windows reaching its near-real-time period.
PRECIP_LATTICE_MM = 0.24
X24_FACTOR = 24
X24_DESCALE_REASON = "nasa_power_x24_response"
# A fetch window ending this many days or more before latest_available is not guarded.
X24_EXPOSURE_MARGIN_DAYS = 120
_LATTICE_TOLERANCE = 1e-6
_GUARD_MAX_VALIDATIONS = 6
_GUARD_MAX_HTTP_ATTEMPTS = 12
_REF_MIN_OVERLAP_WET_DAYS = 3
_REF_MIN_MAGNITUDE_MM = 1.0


class _NasaPowerTransportError(Exception):
    """The transport retries of one NASA POWER request are exhausted."""


def _is_wet(value: Any) -> bool:
    """A wet day: a finite number above zero (never the -999 fill value,
    None, NaN, ±inf, 0, a negative or a bool)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _on_lattice(value: float) -> bool:
    ratio = value / PRECIP_LATTICE_MM
    return abs(ratio - round(ratio)) <= _LATTICE_TOLERANCE


def _lattice_counts(series: Mapping[str, Any]) -> Tuple[int, int]:
    """(k, m): a PRECTOTCORR series' wet days and those on the lattice."""
    wet = [value for value in series.values() if _is_wet(value)]
    return len(wet), sum(1 for value in wet if _on_lattice(value))


def _classify_precip(
    series: Optional[Mapping[str, Any]],
) -> Literal["clean", "scaled", "unconfirmed", "n/a"]:
    """"clean", "scaled" (k >= 10 with >= half on the lattice, or 3-9 all on
    it), "unconfirmed" (1-2 wet days, all on it) or "n/a" (no series)."""
    if series is None:
        return "n/a"
    k, m = _lattice_counts(series)
    if (k >= 10 and 2 * m >= k) or (3 <= k <= 9 and m == k):
        return "scaled"
    if 1 <= k <= 2 and m == k:
        return "unconfirmed"
    return "clean"


def _classification_evidence(series: Mapping[str, Any], classification: str) -> str:
    k, m = _lattice_counts(series)
    if classification == "unconfirmed":
        return f"only {k} wet day(s), all on the {PRECIP_LATTICE_MM} mm lattice"
    return f"{m} of {k} wet days on the {PRECIP_LATTICE_MM} mm lattice"


def _scaled_by(a: float, b: float, factor: int) -> bool:
    """a == factor · b at the API's resolution."""
    return abs(a - factor * b) <= _LATTICE_TOLERANCE * max(1.0, abs(a))


def _descale_precip(series: Mapping[str, Any]) -> Dict[str, Any]:
    """Divide each wet value by 24 at the API's 0.01 mm; every other value
    passes through unchanged."""
    return {
        day: round(value / X24_FACTOR, 2) if _is_wet(value) else value
        for day, value in series.items()
    }


class _Measurement(NamedTuple):
    factor: Optional[int]
    k_ov: int
    descaled: Optional[Dict[str, Any]]


def _within(series: Optional[Mapping[str, Any]], first: str, last: str) -> Optional[Dict[str, Any]]:
    """The days of a PRECTOTCORR series within [first, last] (``YYYYMMDD``)."""
    if series is None:
        return None
    return {day: value for day, value in series.items() if first <= day <= last}


def _day_keys(first: date, last: date) -> List[str]:
    """Every calendar day of [first, last] as ``YYYYMMDD``."""
    return [(first + timedelta(days=n)).strftime("%Y%m%d") for n in range((last - first).days + 1)]


def _measure_factor(
    response: Mapping[str, Any],
    ref: Mapping[str, Any],
    ref_end: date,
    year_end: date,
) -> _Measurement:
    """Measure a response's factor (24, 1 or None) against REF, a CLEAN
    retrieval of [1 January, ``ref_end``] with >= 3 wet days.

    The overlap is every day of [1 January, ref_end]: both series must carry
    each one and share its wet dates. ×24 needs every wet overlap value at
    24 × REF's, a REF value >= 1.0 mm, every wet value of the used span
    [1 January, year_end] on the lattice and a CLEAN de-scaled used span
    equal to REF on the overlap. ×1 needs every wet overlap value equal to
    REF's at 0.01 mm and a tail (ref_end, year_end] that is dry or has a
    value off the lattice.
    """
    no_measurement = _Measurement(None, 0, None)
    first = date(ref_end.year, 1, 1)
    overlap = _day_keys(first, ref_end)
    if any(day not in ref or day not in response for day in overlap):
        return no_measurement
    if _classify_precip({day: ref[day] for day in overlap}) != "clean":
        return no_measurement
    ref_wet = [day for day in overlap if _is_wet(ref[day])]
    k_ov = len(ref_wet)
    if k_ov < _REF_MIN_OVERLAP_WET_DAYS:
        return no_measurement
    if [day for day in overlap if _is_wet(response[day])] != ref_wet:
        return no_measurement
    used = {day: response[day] for day in _day_keys(first, year_end) if day in response}
    unproven = _Measurement(None, k_ov, None)
    if all(_scaled_by(response[day], ref[day], X24_FACTOR) for day in ref_wet):
        if not any(ref[day] >= _REF_MIN_MAGNITUDE_MM for day in ref_wet):
            return unproven
        if not all(_on_lattice(v) for v in used.values() if _is_wet(v)):
            return unproven
        descaled = _descale_precip(used)
        if any(descaled[day] != ref[day] for day in overlap):
            return unproven
        if _classify_precip(descaled) != "clean":
            return unproven
        return _Measurement(X24_FACTOR, k_ov, descaled)
    if all(round(response[day], 2) == round(ref[day], 2) for day in ref_wet):
        tail_wet = [v for day, v in used.items() if day > overlap[-1] and _is_wet(v)]
        if tail_wet and all(_on_lattice(v) for v in tail_wet):
            return unproven
        return _Measurement(1, k_ov, None)
    return unproven


def _x24_pair_smaller(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> Optional[str]:
    """"first" or "second": the smaller of two PRECTOTCORR series when one
    is exactly 24 × the other on the same wet dates; else None."""
    wet_first = {day for day, value in first.items() if _is_wet(value)}
    wet_second = {day for day, value in second.items() if _is_wet(value)}
    if not wet_first or wet_first != wet_second:
        return None
    if all(_scaled_by(first[day], second[day], X24_FACTOR) for day in wet_first):
        return "second"
    if all(_scaled_by(second[day], first[day], X24_FACTOR) for day in wet_first):
        return "first"
    return None


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
    # Days behind "today" (UTC) that NASA POWER data is assumed published;
    # raise it for an even more conservative published-date estimate.
    climate_lag_days: int = DEFAULT_LAG_DAYS

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
        cancel_check: Optional[Callable[[], bool]] = None,
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
        # V2-22b L F-8: cancel_check now an explicit kwarg on the
        # signature — typo safety.
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

        # Single source of NASA POWER's latest published date — the default
        # end date, the per-year fetch clamp, and the coverage check all read
        # this one value so they cannot disagree.
        latest_available = nasa_power_latest_available_date(
            self.config.climate_lag_days,
        )

        # Parse dates
        start_date = self._parse_date(start_date) if start_date else self.SRAD_EARLIEST_DATE
        end_date = self._parse_date(end_date) if end_date else latest_available

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
                        coverage_error, fill_provenance = self._recover_coverage(
                            filtered, start_date, end_date, latest_available,
                        )
                        if coverage_error is not None:
                            return self.create_result(
                                success=False,
                                errors=[coverage_error],
                                metadata=metadata,
                            )
                        cached_data.metadata["gap_fill"] = fill_provenance
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
        descaled: List[Dict[str, Any]] = []
        for year in years_to_fetch:
            # V2-22b L: year-top cancel — AC L.2. Operator clicks cancel
            # and the loop exits cleanly via PipelineCancelled rather
            # than firing a third year's API call.
            raise_if_cancelled(cancel_check, f"nasa_power.year={year}")

            year_start = date(year, 1, 1)
            # Clamp the year end to the requested window and the latest
            # published date so a fetch never reaches into the future.
            year_end = min(date(year, 12, 31), end_date, latest_available)
            # Skip a year with no in-window published day (wholly future, or a
            # window starting after the latest published date) — nothing to fetch.
            if year_end < max(year_start, start_date):
                continue
            try:
                nasa_data = self._fetch_from_api(
                    lat=lat,
                    lon=lon,
                    start_date=year_start,
                    end_date=year_end,
                    parameters=params_to_fetch,
                    cancel_check=cancel_check,
                    on_attempt=kwargs.get('on_attempt'),
                    latest_available=latest_available,
                    descaled=descaled,
                )
                year_ts = self._convert_to_climate_timeseries(
                    nasa_data=nasa_data,
                    lat=lat,
                    lon=lon,
                    location_id=location_id,
                )
                cached_records.extend(year_ts.records)

                # Cache a year only when fetched in full; a clamp-shortened
                # (partial) year must not be reused as complete on a later run.
                # A de-scaled year is never cached, so every package using it carries its record.
                if year_end == date(year, 12, 31) and not any(
                    entry["year"] == year for entry in descaled
                ):
                    year_cache_path = self._get_year_cache_path(lat, lon, year)
                    try:
                        self._save_climate_cache(year_ts, year_cache_path)
                    except Exception as e:
                        warnings.append(f"Failed to cache year {year}: {e}")

                # V2-22b L: cancel check BEFORE the inter-year sleep so
                # operator cancel is observed on the next iteration
                # rather than after a full request_delay wait.
                raise_if_cancelled(
                    cancel_check, f"nasa_power.before_request_delay={year}",
                )
                # Rate limiting between year requests
                if year != years_to_fetch[-1]:
                    time.sleep(self.config.request_delay)

            except PipelineCancelled:
                # V2-22b L F-1: carve-out so user cancel unwinds past
                # this broad except instead of being rewritten as
                # "API request failed for year {year}".
                raise
            except Exception as e:
                return self.create_result(
                    success=False,
                    errors=[f"failed to load climate for {year}: {e}"],
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

        # Producer-honesty: report success only when the loaded climate covers
        # the full requested window. An uncovered window is reported
        # unavailable with an honest reason, never a silent partial.
        coverage_error, fill_provenance = self._recover_coverage(
            filtered_records, start_date, end_date, latest_available,
        )
        if coverage_error is not None:
            return self.create_result(
                success=False,
                errors=[coverage_error],
                warnings=warnings,
                metadata=metadata,
            )
        climate_ts.metadata["gap_fill"] = fill_provenance
        descale_record = self._descale_record(descaled, filtered_records)
        if descale_record:
            climate_ts.metadata["precip_x24_descaled"] = descale_record

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
        cancel_check: Optional[Callable[[], bool]] = None,
        on_attempt: Optional[Callable[[int, int, float], None]] = None,
        latest_available: Optional[date] = None,
        descaled: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Fetch data from NASA POWER API.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date
            end_date: End date
            parameters: Parameters to fetch
            latest_available: ``retrieve``'s estimate of the latest
                published day. A window ending after it minus
                ``X24_EXPOSURE_MARGIN_DAYS`` is EXPOSED and passes the ×24
                precipitation guard; ``None`` fetches without the guard.
            descaled: receives one entry per response divided by 24.

        Returns:
            Dictionary of parameter data {param: {date_str: value}}

        Raises:
            Exception: If API request fails after retries
            NasaPowerScaledPrecipError: If the guard's budget ends without
                a clean, measured or paired response
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

        from prismpy.sources.common.retry import (
            retry_with_exponential_backoff,
        )

        # F-AG-NASA-RETRY minimum floors per deployment-engineer R1 —
        # transient external-API blips need a ≥120 s cumulative budget
        # to absorb. Config-supplied values raise the ceiling but
        # never lower below the contract floor.
        attempt_counter = {"i": 0, "parsed": 0}
        current = {"params": params}

        def _attempt() -> Dict[str, Dict[str, float]]:
            # V2-22b L: per-attempt cancel check — fires at the top of
            # every iteration, including after a backoff sleep completes
            # so a cancel during the sleep window aborts before the
            # next potentially 120-s NASA request.
            raise_if_cancelled(
                cancel_check,
                f"nasa_power.fetch.attempt={attempt_counter['i']}",
            )
            attempt_counter["i"] += 1
            response = requests.get(
                self.config.base_url,
                params=current["params"],
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            if "properties" not in data or "parameter" not in data["properties"]:
                raise ValueError("Invalid response format from NASA POWER")
            attempt_counter["parsed"] += 1
            return data["properties"]["parameter"]

        def _on_retry(attempt_index, exc, sleep_s):
            self.logger.warning(
                "Attempt %d failed: %s; sleeping %.2fs before retry",
                attempt_index + 1, exc, sleep_s,
            )
            # V2-22b L: pre-sleep cancel check — cancel fired before
            # the backoff wait short-circuits the sleep entirely.
            raise_if_cancelled(
                cancel_check,
                f"nasa_power.before_retry_delay={attempt_index}",
            )

        # Honor caller-supplied retry config but never below the
        # contract floor (max_attempts >= 6, base_delay >= 5.0 s).
        max_attempts = max(6, int(self.config.retry_count))
        base_delay_s = max(5.0, float(self.config.retry_delay))

        def _request(window: Dict[str, Any], attempts: int) -> Dict[str, Dict[str, float]]:
            current["params"] = window
            try:
                return retry_with_exponential_backoff(
                    _attempt,
                    max_attempts=attempts,
                    base_delay_s=base_delay_s,
                    jitter_ratio=0.2,
                    exception_classes=(
                        requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.HTTPError,
                        requests.exceptions.ChunkedEncodingError,
                        ValueError,
                    ),
                    on_retry=_on_retry,
                    on_attempt=on_attempt,
                )
            except (requests.exceptions.RequestException, ValueError) as exc:
                raise _NasaPowerTransportError(
                    f"API request failed after retries: {exc}"
                ) from exc

        if latest_available is None:
            return _request(params, max_attempts)
        if end_date <= latest_available - timedelta(days=X24_EXPOSURE_MARGIN_DAYS):
            parameter = _request(params, max_attempts)
            self._warn_unguarded_x24_signature(
                parameter, lat, lon, start_date.year, end_date,
            )
            return parameter
        return self._guard_exposed_window(
            _request, params, attempt_counter, max_attempts,
            lat, lon, start_date, end_date, latest_available,
            cancel_check, descaled,
        )

    def _warn_unguarded_x24_signature(
        self,
        parameter: Dict[str, Dict[str, float]],
        lat: float,
        lon: float,
        year: int,
        end_date: date,
    ) -> None:
        """Log a window outside the guard that shows the ×24 signature; its
        data is used as received (never refused, never re-fetched)."""
        precip = parameter.get("PRECTOTCORR")
        if precip is None:
            return
        k, m = _lattice_counts(precip)
        if k >= 10 and 2 * m >= k:
            self.logger.warning(
                f"NASA POWER rain guard ({lat}, {lon}) {year}: precipitation "
                f"shows the ×24 signature ({m} of {k} wet days on the "
                f"{PRECIP_LATTICE_MM} mm lattice) in a window ending "
                f"{end_date.isoformat()}, before the guarded period; "
                f"used as received"
            )

    def _guard_exposed_window(
        self,
        request: Callable[[Dict[str, Any], int], Dict[str, Dict[str, float]]],
        window: Dict[str, Any],
        counter: Dict[str, int],
        max_attempts: int,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
        latest_available: date,
        cancel_check: Optional[Callable[[], bool]],
        descaled: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, float]]:
        """Return a CLEAN, ×24-corrected, ×1-proven or ×24-paired response for
        an exposed window, or raise NasaPowerScaledPrecipError. REF (1 January
        to latest_available - margin) is requested at most once, never empty."""
        year = start_date.year
        ref_end = latest_available - timedelta(days=X24_EXPOSURE_MARGIN_DAYS)
        ref_end_key = ref_end.strftime("%Y%m%d")
        tag = f"NASA POWER rain guard ({lat}, {lon}) {year}"
        rejected: List[Tuple[Dict[str, Dict[str, float]], Optional[Dict[str, Any]], str]] = []
        reference: Dict[str, Any] = {"requested": False, "precip": None}

        def remaining() -> int:
            return _GUARD_MAX_HTTP_ATTEMPTS - counter["i"]

        def transport_retries() -> int:
            return counter["i"] - counter["parsed"]

        def pause(step: str) -> None:
            raise_if_cancelled(cancel_check, f"nasa_power.x24.before_{step}_delay={year}")
            time.sleep(self.config.request_delay)
            raise_if_cancelled(cancel_check, f"nasa_power.x24.before_{step}={year}")

        def finish(outcome: str) -> None:
            self.logger.info(
                f"{tag}: {outcome} after {len(rejected)} validation attempts "
                f"and {counter['i']} HTTP attempts ({transport_retries()} "
                f"transport retries)"
            )

        def ref_precip() -> Optional[Dict[str, Any]]:
            if reference["requested"]:
                return reference["precip"]
            reference["requested"] = True
            if ref_end < date(year, 1, 1) or remaining() <= 0:
                return None
            pause("ref")
            try:
                body = request(
                    dict(window, start=f"{year}0101", end=ref_end_key),
                    min(max_attempts, _GUARD_MAX_HTTP_ATTEMPTS - counter["i"]),
                )
            except _NasaPowerTransportError:
                return None
            reference["precip"] = body.get("PRECTOTCORR")
            return reference["precip"]

        first_request = True
        while True:
            if not first_request:
                if len(rejected) >= _GUARD_MAX_VALIDATIONS or remaining() <= 0:
                    break
                pause("refetch")
            first_request = False
            try:
                parameter = request(
                    window, min(max_attempts, _GUARD_MAX_HTTP_ATTEMPTS - counter["i"]),
                )
            except _NasaPowerTransportError:
                if not rejected:
                    raise
                continue
            precip = _within(parameter.get("PRECTOTCORR"), window["start"], window["end"])
            classification = _classify_precip(precip)
            if classification in ("clean", "n/a"):
                if rejected:
                    finish("resolved by a clean response")
                return parameter
            evidence = _classification_evidence(precip, classification)
            rejected.append((parameter, precip, evidence))
            self.logger.info(
                f"{tag}: validation attempt {len(rejected)} of "
                f"{_GUARD_MAX_VALIDATIONS} is {classification}: {evidence}; "
                f"HTTP attempts so far {counter['i']} of "
                f"{_GUARD_MAX_HTTP_ATTEMPTS} ({transport_retries()} transport retries)"
            )
            ref = ref_precip()
            if ref is not None:
                measured = _measure_factor(precip, ref, ref_end, end_date)
                if measured.factor == X24_FACTOR:
                    self.logger.info(
                        f"{tag}: ×24 measured against a clean retrieval to "
                        f"{ref_end.isoformat()} ({measured.k_ov} wet days); "
                        f"divided by 24"
                    )
                    finish("resolved by dividing a ×24 response by 24")
                    if descaled is not None:
                        descaled.append({
                            "year": year,
                            "ref_end": ref_end,
                            "k_ov": measured.k_ov,
                            "divided": {d for d, v in precip.items() if _is_wet(v)},
                        })
                    return dict(parameter, PRECTOTCORR=measured.descaled)
                if measured.factor == 1:
                    finish(
                        f"resolved by measuring ×1 against a clean retrieval "
                        f"to {ref_end.isoformat()}"
                    )
                    return parameter
            for earlier, earlier_precip, _ in rejected[:-1]:
                smaller = _x24_pair_smaller(precip, earlier_precip)
                if smaller is not None:
                    finish("resolved by the smaller of a ×24 pair")
                    return parameter if smaller == "first" else earlier
        finish("refused")
        raise NasaPowerScaledPrecipError(
            lat=lat,
            lon=lon,
            year=year,
            evidence=[evidence for _, _, evidence in rejected],
            year_end=end_date,
            latest_available=latest_available,
        )

    @staticmethod
    def _descale_record(
        descaled: List[Dict[str, Any]],
        records: List[ClimateRecord],
    ) -> List[Dict[str, Any]]:
        """One provenance record per de-scaled year, sorted by year, over
        the RETURNED records (the final filter to the caller's range)."""
        out = []
        for entry in sorted(descaled, key=lambda e: e["year"]):
            year_records = [r for r in records if r.date.year == entry["year"]]
            if not year_records:
                continue
            divided_dates = sorted(
                r.date.isoformat() for r in year_records
                if r.date.strftime("%Y%m%d") in entry["divided"]
            )
            out.append({
                "year": entry["year"],
                "first_date": year_records[0].date.isoformat(),
                "last_date": year_records[-1].date.isoformat(),
                "wet_days_divided": len(divided_dates),
                "divided_dates": divided_dates,
                "factor": X24_FACTOR,
                "reason": X24_DESCALE_REASON,
                "evidence": (
                    f"×24 measured on {entry['k_ov']} wet days against a clean "
                    f"retrieval of {entry['year']}-01-01 to "
                    f"{entry['ref_end'].isoformat()}"
                ),
            })
        return out

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

    def _recover_coverage(self, records, start_date, end_date, latest_available):
        """Recover short interior gaps, then assess window coverage.

        Normalizes impossible/sentinel values to missing and linearly fills
        short interior runs of solar radiation and temperature from their
        bracketing real days (rain is never filled), then returns an honest
        coverage error (or ``None``) plus the per-variable fill record for
        provenance. Mutates ``records`` in place.
        """
        normalize_missing(records)
        fill_provenance = fill_short_gaps(records)
        # An interpolated temperature can cross its real counterpart; drop any
        # tmin/tmax pair the fill left inverted rather than emit an impossible
        # record — the day then fails coverage honestly.
        normalize_missing(records)
        recompute_means(records)
        coverage_error = self._coverage_error(
            records, start_date, end_date, latest_available,
        )
        return coverage_error, fill_provenance

    def _coverage_error(self, records, start_date, end_date, latest_available):
        """Return an honest error when ``records`` do not cover every day of
        ``[start_date, end_date]`` with all required values, else ``None``.

        The bar is endpoint-bracket plus zero interior missing days: the
        series must begin on or before ``start_date``, end on or after
        ``end_date``, and carry tmin / tmax / srad / precip for every calendar
        day in the window (NASA POWER serves -999 for a gap, mapped to ``None``
        on convert, so a fill counts as missing). Coverage honesty lives here
        at the producer boundary; the climate is never gap-filled to pass.
        """
        required = ("tmin", "tmax", "srad", "precip")
        complete_days = {
            r.date for r in records
            if start_date <= r.date <= end_date
            and all(getattr(r, field) is not None for field in required)
        }
        needed_days = (end_date - start_date).days + 1
        if len(complete_days) == needed_days:
            return None

        # A shortfall whose last covered day stops before a window that runs
        # past the latest published date is unpublished; any other is corrupt.
        in_window = [r.date for r in records
                     if start_date <= r.date <= end_date]
        last_covered = max(in_window) if in_window else None
        if (last_covered is None or last_covered < end_date) \
                and latest_available < end_date:
            return (
                f"climate through {end_date.isoformat()} is not yet published "
                f"by NASA POWER (latest available: "
                f"{latest_available.isoformat()})"
            )
        return (
            f"incomplete or corrupt climate over {start_date.isoformat()} to "
            f"{end_date.isoformat()} — refetch"
        )

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
