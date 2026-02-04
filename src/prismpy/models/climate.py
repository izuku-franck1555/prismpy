"""
Climate data models for prismpy.

These models represent daily climate records and time series.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class ClimateRecord:
    """Daily climate record in canonical form.

    All units are standardized:
    - Temperatures in degrees Celsius
    - Precipitation in mm/day
    - Solar radiation in MJ/m²/day
    - Wind speed in m/s
    - Relative humidity in %
    - Reference ET in mm/day

    Attributes:
        date: Date of the record
        tmax: Maximum temperature (°C)
        tmin: Minimum temperature (°C)
        tmean: Mean temperature (°C), computed if not provided
        precip: Precipitation (mm/day)
        srad: Solar radiation (MJ/m²/day)
        wind: Wind speed at 2m (m/s), optional
        rh: Relative humidity (%), optional
        tdew: Dew point temperature (°C), optional
        et0: Reference evapotranspiration (mm/day), optional
    """
    date: date
    tmax: float
    tmin: float
    precip: float
    srad: float
    tmean: Optional[float] = None
    wind: Optional[float] = None
    rh: Optional[float] = None
    tdew: Optional[float] = None
    et0: Optional[float] = None

    def __post_init__(self):
        """Compute derived values if not provided."""
        if self.tmean is None:
            self.tmean = (self.tmax + self.tmin) / 2

    @property
    def doy(self) -> int:
        """Day of year (1-366)."""
        return self.date.timetuple().tm_yday

    @property
    def year(self) -> int:
        """Year of the record."""
        return self.date.year

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "year": self.year,
            "doy": self.doy,
            "tmax": self.tmax,
            "tmin": self.tmin,
            "tmean": self.tmean,
            "precip": self.precip,
            "srad": self.srad,
            "wind": self.wind,
            "rh": self.rh,
            "tdew": self.tdew,
            "et0": self.et0,
        }

    def validate(
        self,
        tmax_range: Tuple[float, float] = (-40, 60),
        tmin_range: Tuple[float, float] = (-50, 50),
        precip_range: Tuple[float, float] = (0, 500),
        srad_range: Tuple[float, float] = (0, 40),
    ) -> List[str]:
        """Validate the climate record against reasonable ranges.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not (tmax_range[0] <= self.tmax <= tmax_range[1]):
            errors.append(f"tmax {self.tmax} outside range {tmax_range}")
        if not (tmin_range[0] <= self.tmin <= tmin_range[1]):
            errors.append(f"tmin {self.tmin} outside range {tmin_range}")
        if self.tmin > self.tmax:
            errors.append(f"tmin {self.tmin} > tmax {self.tmax}")
        if not (precip_range[0] <= self.precip <= precip_range[1]):
            errors.append(f"precip {self.precip} outside range {precip_range}")
        if not (srad_range[0] <= self.srad <= srad_range[1]):
            errors.append(f"srad {self.srad} outside range {srad_range}")

        return errors


@dataclass
class ClimateTimeSeries:
    """Complete climate time series for a location.

    This is the canonical representation of climate data that all
    platform translators consume.

    Attributes:
        location_id: Unique location identifier (cell ID or site ID)
        lat: Latitude of the location
        lon: Longitude of the location
        source: Data source identifier (e.g., "NASA_POWER", "TAMSAT")
        records: List of daily climate records
        elevation: Elevation in meters (optional, for ET0 calculation)
        metadata: Additional metadata
    """
    location_id: int
    lat: float
    lon: float
    source: str
    records: List[ClimateRecord] = field(default_factory=list)
    elevation: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_records(self) -> int:
        """Number of daily records."""
        return len(self.records)

    @property
    def date_range(self) -> Optional[Tuple[date, date]]:
        """Start and end dates of the time series."""
        if not self.records:
            return None
        dates = [r.date for r in self.records]
        return (min(dates), max(dates))

    @property
    def years(self) -> List[int]:
        """List of years in the time series."""
        return sorted(set(r.year for r in self.records))

    def iter_records(self) -> Iterator[ClimateRecord]:
        """Iterate over all records."""
        return iter(self.records)

    def get_records_for_year(self, year: int) -> List[ClimateRecord]:
        """Get all records for a specific year."""
        return [r for r in self.records if r.year == year]

    def get_record_for_date(self, target_date: date) -> Optional[ClimateRecord]:
        """Get record for a specific date."""
        for record in self.records:
            if record.date == target_date:
                return record
        return None

    def to_numpy_arrays(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary of numpy arrays.

        Returns:
            Dictionary with arrays for each variable
        """
        n = len(self.records)
        arrays = {
            "tmax": np.zeros(n),
            "tmin": np.zeros(n),
            "tmean": np.zeros(n),
            "precip": np.zeros(n),
            "srad": np.zeros(n),
        }

        for i, record in enumerate(self.records):
            arrays["tmax"][i] = record.tmax
            arrays["tmin"][i] = record.tmin
            arrays["tmean"][i] = record.tmean if record.tmean else (record.tmax + record.tmin) / 2
            arrays["precip"][i] = record.precip
            arrays["srad"][i] = record.srad

        # Optional variables
        if any(r.et0 is not None for r in self.records):
            arrays["et0"] = np.array([r.et0 if r.et0 else np.nan for r in self.records])
        if any(r.wind is not None for r in self.records):
            arrays["wind"] = np.array([r.wind if r.wind else np.nan for r in self.records])

        return arrays

    def to_acea_pickle_format(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert to ACEA pickle format (tmax, tmin, prec, et0).

        Returns:
            Tuple of numpy arrays (tmax, tmin, prec, et0)
        """
        arrays = self.to_numpy_arrays()
        et0 = arrays.get("et0", np.zeros(len(self.records)))
        return (arrays["tmax"], arrays["tmin"], arrays["precip"], et0)

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in self.records])

    def validate_all(self) -> Dict[date, List[str]]:
        """Validate all records and return errors by date."""
        errors = {}
        for record in self.records:
            record_errors = record.validate()
            if record_errors:
                errors[record.date] = record_errors
        return errors

    def check_completeness(self) -> Dict[str, Any]:
        """Check for missing dates in the time series.

        Returns:
            Dictionary with completeness statistics
        """
        if not self.records:
            return {"complete": False, "missing_dates": [], "coverage_pct": 0}

        start_date, end_date = self.date_range
        expected_dates = set()
        current = start_date
        while current <= end_date:
            expected_dates.add(current)
            current = date(
                current.year,
                current.month,
                current.day + 1 if current.day < 28 else 1
            )
            # Handle month/year transitions properly
            from datetime import timedelta
            current = start_date
            while current <= end_date:
                expected_dates.add(current)
                current += timedelta(days=1)

        actual_dates = set(r.date for r in self.records)
        missing_dates = sorted(expected_dates - actual_dates)

        return {
            "complete": len(missing_dates) == 0,
            "missing_dates": missing_dates,
            "n_expected": len(expected_dates),
            "n_actual": len(actual_dates),
            "coverage_pct": 100 * len(actual_dates) / len(expected_dates) if expected_dates else 0,
        }
