"""
Quality control utilities for prismpy.

This module provides functionality for:
- Validating climate data against physical limits
- Detecting outliers and anomalies
- Cross-variable consistency checks
- Data quality flagging and reporting

Reference: OWASP-style validation for crop modeling inputs.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.provenance.tracker import ProvenanceTracker


logger = logging.getLogger(__name__)


class QualityFlag(str, Enum):
    """Quality control flags."""
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"
    MISSING = "missing"
    CORRECTED = "corrected"


class IssueType(str, Enum):
    """Types of quality issues."""
    OUT_OF_RANGE = "out_of_range"
    CONSISTENCY = "consistency"
    OUTLIER = "outlier"
    MISSING = "missing"
    DUPLICATE = "duplicate"
    TEMPORAL = "temporal"


@dataclass
class QualityIssue:
    """Description of a quality control issue.

    Attributes:
        issue_type: Type of issue
        variable: Affected variable
        date: Date of issue (if applicable)
        value: Problematic value
        expected_range: Expected range (if applicable)
        message: Human-readable description
        severity: Issue severity (1-3, 3=most severe)
        auto_correctable: Whether issue can be auto-corrected
    """
    issue_type: IssueType
    variable: str
    date: Optional[date] = None
    value: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None
    message: str = ""
    severity: int = 1
    auto_correctable: bool = False


@dataclass
class QualityReport:
    """Quality control report for a dataset.

    Attributes:
        total_records: Total number of records checked
        valid_records: Number of fully valid records
        issues: List of quality issues found
        flags_by_variable: Count of flags per variable
        overall_quality: Overall quality score (0-100)
        passed: Whether data passes minimum quality threshold
    """
    total_records: int
    valid_records: int
    issues: List[QualityIssue]
    flags_by_variable: Dict[str, Dict[str, int]]
    overall_quality: float
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


# Physical limits for climate variables
CLIMATE_LIMITS = {
    "tmax": (-50, 60),      # °C
    "tmin": (-60, 50),      # °C
    "tmean": (-55, 55),     # °C
    "precip": (0, 500),     # mm/day
    "srad": (0, 40),        # MJ/m²/day
    "wind": (0, 50),        # m/s
    "rh": (0, 100),         # %
    "et0": (0, 20),         # mm/day
    "tdew": (-60, 40),      # °C
}

# Physical limits for soil variables
SOIL_LIMITS = {
    "sand": (0, 100),
    "clay": (0, 100),
    "silt": (0, 100),
    "organic_carbon": (0, 30),
    "bulk_density": (0.5, 2.0),
    "ph": (3, 10),
    "field_capacity": (0, 1),
    "wilting_point": (0, 1),
}

# Consistency rules
CONSISTENCY_RULES = {
    "tmin_le_tmax": ("tmin", "<=", "tmax"),
    "tmin_le_tmean": ("tmin", "<=", "tmean"),
    "tmean_le_tmax": ("tmean", "<=", "tmax"),
    "tdew_le_tmax": ("tdew", "<=", "tmax"),
    "et0_positive": ("et0", ">=", 0),
    "precip_positive": ("precip", ">=", 0),
}


class QualityController:
    """Performs quality control checks on crop modeling data.

    Key responsibilities:
    1. Validate values against physical limits
    2. Check cross-variable consistency
    3. Detect statistical outliers
    4. Generate quality reports
    5. Optionally correct minor issues
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceTracker] = None,
        climate_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        soil_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        quality_threshold: float = 90.0,
    ):
        """Initialize the quality controller.

        Args:
            provenance: Provenance tracker
            climate_limits: Override climate variable limits
            soil_limits: Override soil variable limits
            quality_threshold: Minimum quality score to pass (%)
        """
        self.provenance = provenance
        self.climate_limits = climate_limits or CLIMATE_LIMITS.copy()
        self.soil_limits = soil_limits or SOIL_LIMITS.copy()
        self.quality_threshold = quality_threshold

    def validate_climate(
        self,
        climate_ts: ClimateTimeSeries,
        check_ranges: bool = True,
        check_consistency: bool = True,
        check_outliers: bool = True,
        outlier_std: float = 4.0,
    ) -> QualityReport:
        """Validate a climate time series.

        Args:
            climate_ts: ClimateTimeSeries to validate
            check_ranges: Check value ranges
            check_consistency: Check cross-variable consistency
            check_outliers: Check for statistical outliers
            outlier_std: Standard deviations for outlier detection

        Returns:
            QualityReport with validation results
        """
        issues = []
        flags_by_var = {}
        valid_count = 0

        # Convert to DataFrame for analysis
        df = self._climate_to_dataframe(climate_ts)
        total_records = len(df)

        # Initialize flag counts
        variables = ["tmax", "tmin", "tmean", "precip", "srad", "wind", "rh", "et0"]
        for var in variables:
            flags_by_var[var] = {f.value: 0 for f in QualityFlag}

        # Check each record
        for idx, row in df.iterrows():
            record_valid = True
            record_date = idx.date()

            # Range checks
            if check_ranges:
                for var in variables:
                    if var in row and pd.notna(row[var]):
                        value = row[var]
                        limits = self.climate_limits.get(var)

                        if limits and (value < limits[0] or value > limits[1]):
                            issues.append(QualityIssue(
                                issue_type=IssueType.OUT_OF_RANGE,
                                variable=var,
                                date=record_date,
                                value=value,
                                expected_range=limits,
                                message=f"{var}={value} outside range {limits}",
                                severity=2 if abs(value - np.mean(limits)) > (limits[1] - limits[0]) else 1,
                                auto_correctable=False,
                            ))
                            flags_by_var[var][QualityFlag.INVALID.value] += 1
                            record_valid = False
                        else:
                            flags_by_var[var][QualityFlag.VALID.value] += 1
                    elif var in row:
                        flags_by_var[var][QualityFlag.MISSING.value] += 1

            # Consistency checks
            if check_consistency:
                # tmin <= tmean <= tmax
                tmin = row.get("tmin")
                tmax = row.get("tmax")
                tmean = row.get("tmean")

                if pd.notna(tmin) and pd.notna(tmax) and tmin > tmax:
                    issues.append(QualityIssue(
                        issue_type=IssueType.CONSISTENCY,
                        variable="tmin/tmax",
                        date=record_date,
                        message=f"tmin ({tmin}) > tmax ({tmax})",
                        severity=2,
                        auto_correctable=True,
                    ))
                    record_valid = False

                if pd.notna(tmean) and pd.notna(tmin) and tmean < tmin:
                    issues.append(QualityIssue(
                        issue_type=IssueType.CONSISTENCY,
                        variable="tmean/tmin",
                        date=record_date,
                        message=f"tmean ({tmean}) < tmin ({tmin})",
                        severity=1,
                        auto_correctable=True,
                    ))

                if pd.notna(tmean) and pd.notna(tmax) and tmean > tmax:
                    issues.append(QualityIssue(
                        issue_type=IssueType.CONSISTENCY,
                        variable="tmean/tmax",
                        date=record_date,
                        message=f"tmean ({tmean}) > tmax ({tmax})",
                        severity=1,
                        auto_correctable=True,
                    ))

            if record_valid:
                valid_count += 1

        # Outlier checks (on full series)
        if check_outliers:
            for var in variables:
                if var in df.columns:
                    outliers = self._detect_outliers(df[var], outlier_std)
                    for idx in outliers:
                        issues.append(QualityIssue(
                            issue_type=IssueType.OUTLIER,
                            variable=var,
                            date=idx.date(),
                            value=df.loc[idx, var],
                            message=f"Statistical outlier (>{outlier_std} std)",
                            severity=1,
                            auto_correctable=False,
                        ))
                        flags_by_var[var][QualityFlag.SUSPECT.value] += 1

        # Calculate quality score
        quality_score = 100 * valid_count / total_records if total_records > 0 else 0

        return QualityReport(
            total_records=total_records,
            valid_records=valid_count,
            issues=issues,
            flags_by_variable=flags_by_var,
            overall_quality=quality_score,
            passed=quality_score >= self.quality_threshold,
            metadata={
                "location_id": climate_ts.location_id,
                "lat": climate_ts.lat,
                "lon": climate_ts.lon,
                "source": climate_ts.source,
            },
        )

    def validate_soil(
        self,
        profile: SoilProfile,
        check_ranges: bool = True,
        check_texture: bool = True,
    ) -> QualityReport:
        """Validate a soil profile.

        Args:
            profile: SoilProfile to validate
            check_ranges: Check value ranges
            check_texture: Check texture fractions sum to ~100%

        Returns:
            QualityReport with validation results
        """
        issues = []
        flags_by_var = {}
        total_checks = 0
        valid_checks = 0

        variables = ["sand", "clay", "silt", "organic_carbon", "bulk_density", "ph"]
        for var in variables:
            flags_by_var[var] = {f.value: 0 for f in QualityFlag}

        for layer_idx, layer in enumerate(profile.layers):
            layer_depth = f"{layer.depth_top}-{layer.depth_bottom}m"

            # Range checks
            if check_ranges:
                for var in variables:
                    value = getattr(layer, var, None)
                    total_checks += 1

                    if value is None:
                        flags_by_var[var][QualityFlag.MISSING.value] += 1
                        continue

                    limits = self.soil_limits.get(var)
                    if limits and (value < limits[0] or value > limits[1]):
                        issues.append(QualityIssue(
                            issue_type=IssueType.OUT_OF_RANGE,
                            variable=var,
                            message=f"Layer {layer_depth}: {var}={value} outside {limits}",
                            value=value,
                            expected_range=limits,
                            severity=2,
                        ))
                        flags_by_var[var][QualityFlag.INVALID.value] += 1
                    else:
                        flags_by_var[var][QualityFlag.VALID.value] += 1
                        valid_checks += 1

            # Texture check
            if check_texture:
                sand = layer.sand or 0
                clay = layer.clay or 0
                silt = layer.silt or 0
                total_texture = sand + clay + silt

                if total_texture > 0 and abs(total_texture - 100) > 5:
                    issues.append(QualityIssue(
                        issue_type=IssueType.CONSISTENCY,
                        variable="texture",
                        message=f"Layer {layer_depth}: sand+clay+silt={total_texture}% (expected ~100%)",
                        severity=2,
                        auto_correctable=True,
                    ))

        quality_score = 100 * valid_checks / total_checks if total_checks > 0 else 0

        return QualityReport(
            total_records=len(profile.layers),
            valid_records=sum(1 for _ in profile.layers),  # Simplified
            issues=issues,
            flags_by_variable=flags_by_var,
            overall_quality=quality_score,
            passed=quality_score >= self.quality_threshold,
            metadata={
                "profile_id": profile.profile_id,
                "lat": profile.lat,
                "lon": profile.lon,
                "source": profile.source,
            },
        )

    def correct_issues(
        self,
        climate_ts: ClimateTimeSeries,
        issues: List[QualityIssue],
    ) -> Tuple[ClimateTimeSeries, int]:
        """Auto-correct fixable quality issues.

        Args:
            climate_ts: ClimateTimeSeries to correct
            issues: List of issues to correct

        Returns:
            Tuple of (corrected ClimateTimeSeries, number of corrections)
        """
        df = self._climate_to_dataframe(climate_ts)
        corrections = 0

        for issue in issues:
            if not issue.auto_correctable:
                continue

            if issue.issue_type == IssueType.CONSISTENCY:
                if issue.variable == "tmin/tmax" and issue.date:
                    # Swap tmin and tmax
                    idx = pd.Timestamp(issue.date)
                    if idx in df.index:
                        tmin = df.loc[idx, "tmin"]
                        tmax = df.loc[idx, "tmax"]
                        df.loc[idx, "tmin"] = min(tmin, tmax)
                        df.loc[idx, "tmax"] = max(tmin, tmax)
                        corrections += 1

                elif issue.variable in ["tmean/tmin", "tmean/tmax"] and issue.date:
                    # Recalculate tmean
                    idx = pd.Timestamp(issue.date)
                    if idx in df.index:
                        tmin = df.loc[idx, "tmin"]
                        tmax = df.loc[idx, "tmax"]
                        if pd.notna(tmin) and pd.notna(tmax):
                            df.loc[idx, "tmean"] = (tmin + tmax) / 2
                            corrections += 1

        # Convert back
        corrected_ts = self._dataframe_to_climate(df, climate_ts)

        return corrected_ts, corrections

    def generate_summary(
        self,
        reports: List[QualityReport],
    ) -> Dict[str, Any]:
        """Generate summary statistics from multiple QC reports.

        Args:
            reports: List of QualityReport objects

        Returns:
            Summary statistics dictionary
        """
        if not reports:
            return {"error": "No reports provided"}

        total_records = sum(r.total_records for r in reports)
        total_valid = sum(r.valid_records for r in reports)
        total_issues = sum(len(r.issues) for r in reports)
        passed_count = sum(1 for r in reports if r.passed)

        # Aggregate issue types
        issue_counts = {}
        for report in reports:
            for issue in report.issues:
                key = issue.issue_type.value
                issue_counts[key] = issue_counts.get(key, 0) + 1

        return {
            "total_reports": len(reports),
            "total_records": total_records,
            "total_valid": total_valid,
            "total_issues": total_issues,
            "overall_quality": 100 * total_valid / total_records if total_records > 0 else 0,
            "passed_count": passed_count,
            "pass_rate": 100 * passed_count / len(reports),
            "issue_counts": issue_counts,
        }

    def _climate_to_dataframe(
        self,
        climate_ts: ClimateTimeSeries,
    ) -> pd.DataFrame:
        """Convert ClimateTimeSeries to DataFrame."""
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

    def _dataframe_to_climate(
        self,
        df: pd.DataFrame,
        original_ts: ClimateTimeSeries,
    ) -> ClimateTimeSeries:
        """Convert DataFrame back to ClimateTimeSeries."""
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

    def _detect_outliers(
        self,
        series: pd.Series,
        n_std: float = 4.0,
    ) -> List:
        """Detect statistical outliers using z-score method.

        Args:
            series: Pandas Series to check
            n_std: Number of standard deviations for threshold

        Returns:
            List of outlier indices
        """
        if series.isna().all():
            return []

        mean = series.mean()
        std = series.std()

        if std == 0 or pd.isna(std):
            return []

        z_scores = np.abs((series - mean) / std)
        outlier_mask = z_scores > n_std

        return series.index[outlier_mask].tolist()
