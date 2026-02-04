"""
Temporal harmonization utilities for prismpy.

This module provides functionality for:
- Gap-filling missing daily values
- Temporal interpolation
- Date alignment across sources
- Climatological filling for extended gaps

Reference: Platform requirements for continuous daily climate data.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.provenance.tracker import DecisionType, ProvenanceTracker


logger = logging.getLogger(__name__)


class GapFillMethod(str, Enum):
    """Methods for filling gaps in time series data."""
    LINEAR = "linear"           # Linear interpolation
    SPLINE = "spline"           # Spline interpolation
    NEAREST = "nearest"         # Nearest valid value
    CLIMATOLOGY = "climatology" # Day-of-year climatology
    FORWARD = "forward"         # Forward fill (last valid)
    BACKWARD = "backward"       # Backward fill (next valid)
    MEAN = "mean"               # Mean of surrounding values


@dataclass
class GapInfo:
    """Information about a gap in time series data.

    Attributes:
        start_date: First missing date
        end_date: Last missing date
        length: Number of missing days
        variable: Variable name (if applicable)
        fill_method: Method used/recommended for filling
    """
    start_date: date
    end_date: date
    length: int
    variable: Optional[str] = None
    fill_method: Optional[GapFillMethod] = None


@dataclass
class TemporalHarmonizationResult:
    """Result of temporal harmonization.

    Attributes:
        success: Whether harmonization succeeded
        data: Harmonized data
        gaps_filled: Number of gaps filled
        gap_details: List of GapInfo objects
        method: Primary method used
        metadata: Additional metadata
        warnings: List of warnings
    """
    success: bool
    data: Any
    gaps_filled: int
    gap_details: List[GapInfo]
    method: str
    metadata: Dict[str, Any]
    warnings: List[str]


class TemporalHarmonizer:
    """Handles temporal harmonization of climate time series data.

    Key responsibilities:
    1. Detect and fill gaps in daily data
    2. Align time series to common date ranges
    3. Apply appropriate filling strategies based on gap length
    4. Generate climatologies for long-gap filling
    """

    # Default thresholds for gap-filling strategy selection
    SHORT_GAP_THRESHOLD = 5   # Days - use interpolation
    MEDIUM_GAP_THRESHOLD = 30 # Days - use climatology
    # Longer gaps may require external data or flagging

    def __init__(
        self,
        provenance: Optional[ProvenanceTracker] = None,
        short_gap_threshold: int = 5,
        medium_gap_threshold: int = 30,
    ):
        """Initialize the temporal harmonizer.

        Args:
            provenance: Provenance tracker
            short_gap_threshold: Max days for interpolation
            medium_gap_threshold: Max days for climatology filling
        """
        self.provenance = provenance
        self.short_gap_threshold = short_gap_threshold
        self.medium_gap_threshold = medium_gap_threshold

    def fill_gaps(
        self,
        climate_ts: ClimateTimeSeries,
        variables: Optional[List[str]] = None,
        method: Optional[GapFillMethod] = None,
        climatology: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> TemporalHarmonizationResult:
        """Fill gaps in a climate time series.

        Args:
            climate_ts: ClimateTimeSeries with potential gaps
            variables: Variables to fill (default: all)
            method: Override method (default: auto-select based on gap length)
            climatology: Pre-computed climatology for DOY filling

        Returns:
            TemporalHarmonizationResult with filled data
        """
        warnings = []
        gap_details = []
        total_filled = 0

        # Convert to DataFrame for easier manipulation
        df = self._timeseries_to_dataframe(climate_ts)

        # Identify variables to process
        if variables is None:
            variables = ["tmax", "tmin", "tmean", "precip", "srad", "wind", "rh", "et0"]
        variables = [v for v in variables if v in df.columns]

        # Detect and fill gaps for each variable
        for var in variables:
            if var not in df.columns:
                continue

            # Find gaps
            gaps = self._detect_gaps(df, var)

            for gap in gaps:
                gap.variable = var

                # Select method based on gap length
                if method:
                    fill_method = method
                elif gap.length <= self.short_gap_threshold:
                    fill_method = GapFillMethod.LINEAR
                elif gap.length <= self.medium_gap_threshold:
                    fill_method = GapFillMethod.CLIMATOLOGY
                else:
                    fill_method = GapFillMethod.CLIMATOLOGY
                    warnings.append(
                        f"Long gap ({gap.length} days) for {var} at {gap.start_date}: "
                        "using climatology"
                    )

                gap.fill_method = fill_method

                # Apply filling
                filled_count = self._fill_gap(
                    df=df,
                    variable=var,
                    gap=gap,
                    method=fill_method,
                    climatology=climatology,
                )
                total_filled += filled_count
                gap_details.append(gap)

        # Convert back to ClimateTimeSeries
        filled_ts = self._dataframe_to_timeseries(df, climate_ts)

        # Record provenance
        if self.provenance and total_filled > 0:
            self.provenance.record_decision(
                decision_type=DecisionType.GAP_FILL_METHOD,
                description=f"Filled {total_filled} missing values in climate data",
                rationale="Continuous daily data required for crop simulation",
                alternatives=["linear", "climatology", "nearest"],
                reference=f"Gaps: {len(gap_details)}",
            )

        return TemporalHarmonizationResult(
            success=True,
            data=filled_ts,
            gaps_filled=total_filled,
            gap_details=gap_details,
            method="auto" if method is None else method.value,
            metadata={
                "variables_processed": variables,
                "total_records": len(df),
            },
            warnings=warnings,
        )

    def compute_climatology(
        self,
        climate_ts: ClimateTimeSeries,
        variables: Optional[List[str]] = None,
        min_years: int = 3,
    ) -> Dict[int, Dict[str, float]]:
        """Compute day-of-year climatology from time series.

        Args:
            climate_ts: ClimateTimeSeries with historical data
            variables: Variables to compute climatology for
            min_years: Minimum years required for climatology

        Returns:
            Dictionary mapping DOY (1-366) to variable means
        """
        df = self._timeseries_to_dataframe(climate_ts)

        if variables is None:
            variables = ["tmax", "tmin", "tmean", "precip", "srad"]
        variables = [v for v in variables if v in df.columns]

        # Add DOY column
        df["doy"] = df.index.dayofyear

        # Check data coverage
        years = df.index.year.nunique()
        if years < min_years:
            logger.warning(
                f"Only {years} years of data for climatology (min: {min_years})"
            )

        # Compute climatology
        climatology = {}
        for doy in range(1, 367):
            doy_data = df[df["doy"] == doy]
            climatology[doy] = {}
            for var in variables:
                if var in doy_data.columns:
                    climatology[doy][var] = float(doy_data[var].mean())

        return climatology

    def align_date_range(
        self,
        climate_ts: ClimateTimeSeries,
        start_date: date,
        end_date: date,
        fill_missing: bool = True,
    ) -> TemporalHarmonizationResult:
        """Align time series to a specific date range.

        Ensures continuous daily coverage from start_date to end_date.

        Args:
            climate_ts: Input time series
            start_date: Required start date
            end_date: Required end date
            fill_missing: Fill missing dates with NaN

        Returns:
            TemporalHarmonizationResult with aligned data
        """
        warnings = []
        df = self._timeseries_to_dataframe(climate_ts)

        # Create complete date index
        full_index = pd.date_range(start=start_date, end=end_date, freq="D")

        # Check coverage
        existing_dates = set(df.index.date)
        required_dates = set(d.date() for d in full_index)
        missing_dates = required_dates - existing_dates
        extra_dates = existing_dates - required_dates

        if missing_dates:
            warnings.append(f"{len(missing_dates)} dates missing from input data")

        if extra_dates:
            warnings.append(f"{len(extra_dates)} dates outside requested range")

        # Reindex to full date range
        df = df.reindex(full_index)

        # Create gap details for missing dates
        gap_details = []
        if missing_dates:
            sorted_missing = sorted(missing_dates)
            gap_start = sorted_missing[0]
            gap_end = sorted_missing[0]

            for d in sorted_missing[1:]:
                if (d - gap_end).days == 1:
                    gap_end = d
                else:
                    gap_details.append(GapInfo(
                        start_date=gap_start,
                        end_date=gap_end,
                        length=(gap_end - gap_start).days + 1,
                    ))
                    gap_start = d
                    gap_end = d

            gap_details.append(GapInfo(
                start_date=gap_start,
                end_date=gap_end,
                length=(gap_end - gap_start).days + 1,
            ))

        # Convert back
        aligned_ts = self._dataframe_to_timeseries(df, climate_ts)

        return TemporalHarmonizationResult(
            success=True,
            data=aligned_ts,
            gaps_filled=0,  # Alignment doesn't fill
            gap_details=gap_details,
            method="date_alignment",
            metadata={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_days": len(full_index),
                "missing_days": len(missing_dates),
            },
            warnings=warnings,
        )

    def detect_gaps(
        self,
        climate_ts: ClimateTimeSeries,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, List[GapInfo]]:
        """Detect all gaps in a time series.

        Args:
            climate_ts: ClimateTimeSeries to analyze
            variables: Variables to check (default: all)

        Returns:
            Dictionary mapping variable names to lists of GapInfo
        """
        df = self._timeseries_to_dataframe(climate_ts)

        if variables is None:
            variables = [c for c in df.columns if c not in ["date"]]

        result = {}
        for var in variables:
            if var in df.columns:
                gaps = self._detect_gaps(df, var)
                if gaps:
                    result[var] = gaps

        return result

    def _detect_gaps(
        self,
        df: pd.DataFrame,
        variable: str,
    ) -> List[GapInfo]:
        """Detect gaps in a single variable.

        Args:
            df: DataFrame with DatetimeIndex
            variable: Column name

        Returns:
            List of GapInfo objects
        """
        gaps = []

        # Find missing values
        missing_mask = df[variable].isna()

        if not missing_mask.any():
            return gaps

        # Find contiguous missing regions
        missing_dates = df.index[missing_mask]

        if len(missing_dates) == 0:
            return gaps

        gap_start = missing_dates[0].date()
        gap_end = missing_dates[0].date()

        for dt in missing_dates[1:]:
            current_date = dt.date()
            if (current_date - gap_end).days == 1:
                gap_end = current_date
            else:
                gaps.append(GapInfo(
                    start_date=gap_start,
                    end_date=gap_end,
                    length=(gap_end - gap_start).days + 1,
                ))
                gap_start = current_date
                gap_end = current_date

        # Add final gap
        gaps.append(GapInfo(
            start_date=gap_start,
            end_date=gap_end,
            length=(gap_end - gap_start).days + 1,
        ))

        return gaps

    def _fill_gap(
        self,
        df: pd.DataFrame,
        variable: str,
        gap: GapInfo,
        method: GapFillMethod,
        climatology: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> int:
        """Fill a single gap in the data.

        Args:
            df: DataFrame to modify in place
            variable: Variable column name
            gap: GapInfo describing the gap
            method: Filling method
            climatology: DOY climatology for climatology filling

        Returns:
            Number of values filled
        """
        filled = 0
        gap_mask = (df.index.date >= gap.start_date) & (df.index.date <= gap.end_date)
        gap_indices = df.index[gap_mask & df[variable].isna()]

        if len(gap_indices) == 0:
            return 0

        if method == GapFillMethod.LINEAR:
            df[variable] = df[variable].interpolate(method="linear")
            filled = len(gap_indices)

        elif method == GapFillMethod.SPLINE:
            try:
                df[variable] = df[variable].interpolate(method="spline", order=3)
            except Exception:
                df[variable] = df[variable].interpolate(method="linear")
            filled = len(gap_indices)

        elif method == GapFillMethod.NEAREST:
            df[variable] = df[variable].interpolate(method="nearest")
            filled = len(gap_indices)

        elif method == GapFillMethod.FORWARD:
            df[variable] = df[variable].ffill()
            filled = len(gap_indices)

        elif method == GapFillMethod.BACKWARD:
            df[variable] = df[variable].bfill()
            filled = len(gap_indices)

        elif method == GapFillMethod.CLIMATOLOGY:
            if climatology:
                for idx in gap_indices:
                    doy = idx.dayofyear
                    if doy in climatology and variable in climatology[doy]:
                        df.loc[idx, variable] = climatology[doy][variable]
                        filled += 1
            else:
                # Fall back to interpolation
                df[variable] = df[variable].interpolate(method="linear")
                filled = len(gap_indices)

        elif method == GapFillMethod.MEAN:
            # Use mean of values before and after gap
            before = df.loc[df.index < gap_indices[0], variable].dropna()
            after = df.loc[df.index > gap_indices[-1], variable].dropna()

            fill_value = np.nan
            if len(before) > 0 and len(after) > 0:
                fill_value = (before.iloc[-1] + after.iloc[0]) / 2
            elif len(before) > 0:
                fill_value = before.iloc[-1]
            elif len(after) > 0:
                fill_value = after.iloc[0]

            if not np.isnan(fill_value):
                df.loc[gap_indices, variable] = fill_value
                filled = len(gap_indices)

        return filled

    def _timeseries_to_dataframe(
        self,
        climate_ts: ClimateTimeSeries,
    ) -> pd.DataFrame:
        """Convert ClimateTimeSeries to DataFrame.

        Args:
            climate_ts: ClimateTimeSeries object

        Returns:
            DataFrame with DatetimeIndex
        """
        records = []
        for rec in climate_ts.records:
            records.append({
                "date": rec.date,
                "tmax": rec.tmax,
                "tmin": rec.tmin,
                "tmean": rec.tmean,
                "precip": rec.precip,
                "srad": rec.srad,
                "wind": rec.wind,
                "rh": rec.rh,
                "tdew": rec.tdew,
                "et0": rec.et0,
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        return df

    def _dataframe_to_timeseries(
        self,
        df: pd.DataFrame,
        original_ts: ClimateTimeSeries,
    ) -> ClimateTimeSeries:
        """Convert DataFrame back to ClimateTimeSeries.

        Args:
            df: DataFrame with climate data
            original_ts: Original time series (for metadata)

        Returns:
            ClimateTimeSeries object
        """
        records = []
        for idx, row in df.iterrows():
            rec = ClimateRecord(
                date=idx.date(),
                tmax=row.get("tmax") if pd.notna(row.get("tmax")) else None,
                tmin=row.get("tmin") if pd.notna(row.get("tmin")) else None,
                tmean=row.get("tmean") if pd.notna(row.get("tmean")) else None,
                precip=row.get("precip") if pd.notna(row.get("precip")) else None,
                srad=row.get("srad") if pd.notna(row.get("srad")) else None,
                wind=row.get("wind") if pd.notna(row.get("wind")) else None,
                rh=row.get("rh") if pd.notna(row.get("rh")) else None,
                tdew=row.get("tdew") if pd.notna(row.get("tdew")) else None,
                et0=row.get("et0") if pd.notna(row.get("et0")) else None,
            )
            records.append(rec)

        return ClimateTimeSeries(
            location_id=original_ts.location_id,
            lat=original_ts.lat,
            lon=original_ts.lon,
            source=original_ts.source,
            records=records,
        )
