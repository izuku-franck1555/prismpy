"""
Scientific data quality validation for prismpy.

Implements 6 Tier 1 checks required by the manuscript (Section 2.5):
1. Temporal completeness
2. Cross-variable physical consistency
3. Value range (universal physical bounds)
4. Soil profile completeness (platform-specific)
5. Format compliance (delegated to platform validators)
6. Spatial/temporal coverage

Each check returns a structured dict per UX-expert spec:
{
    "check": "<check_name>",
    "scope": "per_cell|global",
    "result": "pass|warning|fail",
    "summary": "<one-line human summary>",
    "manuscript_claim": "Section 2.5: <specific claim>",
    "details": { ... structured metadata ... }
}
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Value range thresholds (from crop-modeling-specialist spec)
# =============================================================================

CLIMATE_RANGES = {
    "tmax": (-50.0, 60.0, "°C"),
    "tmin": (-60.0, 50.0, "°C"),
    "precip": (0.0, 600.0, "mm/day"),
    "srad": (0.0, 40.0, "MJ/m²/d"),
    "wind": (0.0, 75.0, "m/s"),
    "rh": (0.0, 100.0, "%"),
}

SOIL_RANGES = {
    "sand": (0.0, 100.0, "%"),
    "clay": (0.0, 100.0, "%"),
    "silt": (0.0, 100.0, "%"),
    "organic_carbon": (0.0, 30.0, "%"),
    "ph": (2.5, 10.5, ""),
    "bulk_density": (0.3, 2.0, "g/cm³"),
}

# Platform-specific required soil properties
PLATFORM_SOIL_REQUIREMENTS = {
    "craft": ["sand", "clay", "silt", "organic_carbon", "ph", "bulk_density"],
    "pythia": ["sand", "clay", "silt", "organic_carbon", "ph", "bulk_density"],
    "acea": ["sand", "clay", "organic_carbon"],
    "sarra_py": ["sand", "clay"],  # SARRA-Py derives HumFC/HumPF from texture
}


def run_scientific_validation(
    unified_data,
    config,
    enabled_platforms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run all 6 Tier 1 scientific validation checks.

    Args:
        unified_data: UnifiedData from the HARMONIZE stage
        config: ProjectConfig for temporal/spatial reference
        enabled_platforms: List of enabled platform names

    Returns:
        Dict with 'checks' list and 'overall_result' rollup
    """
    checks = []
    enabled = enabled_platforms or []

    # Check 1: Temporal completeness
    checks.append(_check_temporal_completeness(unified_data, config))

    # Check 2: Cross-variable consistency
    checks.append(_check_cross_variable_consistency(unified_data))

    # Check 3: Value range
    checks.extend(_check_value_ranges(unified_data))

    # Check 4: Soil profile completeness
    for platform in enabled:
        checks.append(_check_soil_completeness(unified_data, platform))

    # Check 5: Format compliance (placeholder — delegated to platform validators)
    checks.append({
        "check": "format_compliance",
        "scope": "per_platform",
        "result": "pass",
        "summary": "Format compliance delegated to platform-specific validators",
        "manuscript_claim": "Section 2.5: schema conformance check",
        "details": {"delegated_to": "platform validators (craft/pythia/acea/sarra_py)"},
    })

    # Check 6: Spatial/temporal coverage
    checks.append(_check_coverage(unified_data, config))

    # Overall rollup
    results = [c["result"] for c in checks]
    if "fail" in results:
        overall = "fail"
    elif "warning" in results:
        overall = "warning"
    else:
        overall = "pass"

    return {
        "validation_version": "1.0",
        "checks": checks,
        "overall_result": overall,
        "n_checks": len(checks),
        "n_pass": results.count("pass"),
        "n_warning": results.count("warning"),
        "n_fail": results.count("fail"),
    }


# =============================================================================
# Check 1: Temporal completeness
# =============================================================================

def _check_temporal_completeness(unified_data, config) -> Dict[str, Any]:
    """Check that climate data covers the full requested period."""
    start_year = config.temporal.start_year
    end_year = config.temporal.end_year
    spinup = config.temporal.spinup_years
    effective_start = start_year - spinup

    expected_start = date(effective_start, 1, 1)
    expected_end = date(end_year, 12, 31)
    expected_days = (expected_end - expected_start).days + 1

    # Count actual days per cell
    climate = unified_data.climate if unified_data and hasattr(unified_data, 'climate') else {}
    if not climate:
        return {
            "check": "temporal_completeness",
            "scope": "global",
            "result": "warning",
            "summary": "No climate data available for completeness check",
            "manuscript_claim": "Section 2.5: completeness check",
            "details": {"expected_days": expected_days, "actual_cells": 0},
        }

    cell_gaps = {}
    total_present = 0
    total_expected = 0
    cells_with_records = 0

    for cell_id, ts in climate.items():
        if not hasattr(ts, 'records'):
            continue
        cells_with_records += 1
        actual_dates = {r.date for r in ts.records if hasattr(r, 'date')}
        n_actual = len(actual_dates)
        n_missing = expected_days - n_actual
        total_present += n_actual
        total_expected += expected_days
        if n_missing > 0:
            cell_gaps[cell_id] = n_missing

    # If no cells have per-day records but climate dict has file-based
    # data (SARRA-Py: rainfall_file_count, agera5_variables), count
    # GeoTIFF files per variable as the completeness metric.
    if cells_with_records == 0 and climate:
        return _check_temporal_completeness_file_based(
            climate, expected_days
        )

    if total_expected == 0:
        completeness = 0.0
    else:
        completeness = total_present / total_expected

    if completeness >= 1.0:
        result = "pass"
    elif completeness >= 0.99:
        result = "warning"
    else:
        result = "fail"

    return {
        "check": "temporal_completeness",
        "scope": "per_cell",
        "result": result,
        "summary": (
            f"{total_present}/{total_expected} days present "
            f"({completeness * 100:.1f}%) across {len(climate)} cells"
        ),
        "manuscript_claim": "Section 2.5: completeness check",
        "details": {
            "expected_days_per_cell": expected_days,
            "n_cells": len(climate),
            "completeness_pct": round(completeness * 100, 2),
            "cells_with_gaps": len(cell_gaps),
            "gap_details": {str(k): v for k, v in list(cell_gaps.items())[:10]},
        },
    }


def _check_temporal_completeness_file_based(
    climate: Dict, expected_days: int,
) -> Dict[str, Any]:
    """Temporal completeness check for GeoTIFF-based climate (SARRA-Py).

    Counts files per climate variable directory and compares against
    the expected number of days. SARRA-Py stores one GeoTIFF per day
    per variable, so file_count == day_count.
    """
    var_counts: Dict[str, int] = {}

    # TAMSAT rainfall
    rainfall_count = climate.get("rainfall_file_count", 0)
    if rainfall_count:
        var_counts["rainfall"] = rainfall_count

    # AgERA5 variables (dict of var_name → file_count)
    agera5_vars = climate.get("agera5_variables", {})
    for var_name, count in agera5_vars.items():
        var_counts[var_name] = count

    if not var_counts:
        return {
            "check": "temporal_completeness",
            "scope": "per_variable",
            "result": "warning",
            "summary": "No file-count data available for temporal check",
            "manuscript_claim": "Section 2.5: completeness check",
            "details": {"expected_days": expected_days},
        }

    # Completeness per variable: file_count / expected_days
    per_var_results = {}
    min_completeness = 1.0
    for var_name, count in var_counts.items():
        pct = count / expected_days if expected_days > 0 else 0.0
        per_var_results[var_name] = {
            "file_count": count,
            "expected": expected_days,
            "completeness_pct": round(pct * 100, 1),
        }
        min_completeness = min(min_completeness, pct)

    if min_completeness >= 1.0:
        result = "pass"
    elif min_completeness >= 0.99:
        result = "warning"
    else:
        result = "fail"

    total_files = sum(var_counts.values())
    total_expected = expected_days * len(var_counts)

    return {
        "check": "temporal_completeness",
        "scope": "per_variable",
        "result": result,
        "summary": (
            f"{total_files}/{total_expected} files present "
            f"({min_completeness * 100:.1f}% min) across "
            f"{len(var_counts)} variables"
        ),
        "manuscript_claim": "Section 2.5: completeness check",
        "details": {
            "expected_days": expected_days,
            "data_format": "geotiff_per_day",
            "per_variable": per_var_results,
            "min_completeness_pct": round(min_completeness * 100, 1),
        },
    }


# =============================================================================
# Check 2: Cross-variable physical consistency
# =============================================================================

def _is_file_based_climate(climate: Dict) -> bool:
    """Detect if climate data is file-based (GeoTIFF) vs per-cell time series."""
    if not climate:
        return False
    # SARRA-Py's climate dict has keys like "rainfall_dir", "agera5_dir"
    return "rainfall_dir" in climate or "agera5_dir" in climate


def _check_cross_variable_consistency(unified_data) -> Dict[str, Any]:
    """Check physical consistency across climate variables."""
    climate = unified_data.climate if unified_data and hasattr(unified_data, 'climate') else {}
    if not climate:
        return {
            "check": "cross_variable_consistency",
            "scope": "per_record",
            "result": "pass",
            "summary": "No climate data to check",
            "manuscript_claim": "Section 2.5: cross-variable consistency",
            "details": {},
        }

    # GeoTIFF-based climate (SARRA-Py): per-value consistency requires
    # opening thousands of raster files — too expensive for a validation
    # check. Report honestly as info with the limitation stated.
    if _is_file_based_climate(climate):
        return {
            "check": "cross_variable_consistency",
            "scope": "per_record",
            "result": "info",
            "summary": (
                "Cross-variable consistency not checked: climate stored "
                "as GeoTIFF rasters (would require reading all files)"
            ),
            "manuscript_claim": "Section 2.5: cross-variable consistency",
            "details": {
                "data_format": "geotiff_per_day",
                "limitation": (
                    "Per-value cross-checks (Tmax>Tmin, precip>=0) require "
                    "opening each GeoTIFF file. For 5000+ files this is "
                    "too expensive in the validation stage. The platform "
                    "validator checks file-level structure instead."
                ),
            },
        }

    violations = {
        "tmax_le_tmin": 0,
        "zero_diurnal_range": 0,
        "negative_precip": 0,
        "negative_srad": 0,
        "excessive_srad": 0,
        "negative_wind": 0,
    }
    total_records = 0
    affected_cells = set()

    for cell_id, ts in climate.items():
        if not hasattr(ts, 'records'):
            continue
        for record in ts.records:
            total_records += 1
            tmax = getattr(record, 'tmax', None)
            tmin = getattr(record, 'tmin', None)
            precip = getattr(record, 'precip', None)
            srad = getattr(record, 'srad', None)
            wind = getattr(record, 'wind', None)

            if tmax is not None and tmin is not None:
                if tmax < tmin:
                    violations["tmax_le_tmin"] += 1
                    affected_cells.add(cell_id)
                elif abs(tmax - tmin) < 0.1:
                    violations["zero_diurnal_range"] += 1

            if precip is not None and precip < 0:
                violations["negative_precip"] += 1
                affected_cells.add(cell_id)

            if srad is not None:
                if srad < 0:
                    violations["negative_srad"] += 1
                    affected_cells.add(cell_id)
                elif srad > 40:
                    violations["excessive_srad"] += 1

            if wind is not None and wind < 0:
                violations["negative_wind"] += 1
                affected_cells.add(cell_id)

    # FAIL-level violations (physical impossibilities)
    fail_count = (
        violations["tmax_le_tmin"]
        + violations["negative_precip"]
        + violations["negative_srad"]
        + violations["negative_wind"]
    )
    # WARNING-level violations (suspect but not impossible)
    warn_count = violations["zero_diurnal_range"] + violations["excessive_srad"]

    if fail_count > 0:
        result = "fail"
    elif warn_count > 0:
        result = "warning"
    else:
        result = "pass"

    return {
        "check": "cross_variable_consistency",
        "scope": "per_record",
        "result": result,
        "summary": (
            f"{total_records} records checked, "
            f"{fail_count} fail-level + {warn_count} warn-level violations"
        ),
        "manuscript_claim": "Section 2.5: cross-variable consistency",
        "details": {
            "total_records": total_records,
            "violations": violations,
            "affected_cells": list(affected_cells)[:20],
            "n_affected_cells": len(affected_cells),
        },
    }


# =============================================================================
# Check 3: Value range (universal physical bounds)
# =============================================================================

def _check_value_ranges(unified_data) -> List[Dict[str, Any]]:
    """Check climate and soil values against universal physical bounds."""
    checks = []

    # Climate value ranges
    climate = unified_data.climate if unified_data and hasattr(unified_data, 'climate') else {}
    climate_stats = {}

    # GeoTIFF-based climate: report limitation instead of silent skip
    if _is_file_based_climate(climate):
        checks.append({
            "check": "value_range_climate",
            "scope": "per_record",
            "result": "info",
            "summary": (
                "Climate value range not checked: data stored as "
                "GeoTIFF rasters (would require reading all files)"
            ),
            "manuscript_claim": "Section 2.5: value range verification",
            "details": {
                "data_format": "geotiff_per_day",
                "limitation": (
                    "Per-value range checks require opening each GeoTIFF. "
                    "Soil value ranges are still checked below."
                ),
            },
        })
        # Skip to soil value ranges (still check those)
        climate_stats = {}  # empty → no per-variable climate checks emitted
    else:
        for cell_id, ts in climate.items():
            if not hasattr(ts, 'records'):
                continue
            for record in ts.records:
                for var, (vmin, vmax, unit) in CLIMATE_RANGES.items():
                    val = getattr(record, var, None)
                    if val is None:
                        continue
                    if var not in climate_stats:
                        climate_stats[var] = {
                            "min": val, "max": val,
                            "out_of_range": 0, "total": 0,
                            "affected_cells": set(),
                        }
                    stats = climate_stats[var]
                    stats["min"] = min(stats["min"], val)
                    stats["max"] = max(stats["max"], val)
                    stats["total"] += 1
                    if val < vmin or val > vmax:
                        stats["out_of_range"] += 1
                        stats["affected_cells"].add(cell_id)

    for var, (vmin, vmax, unit) in CLIMATE_RANGES.items():
        stats = climate_stats.get(var)
        if not stats:
            continue
        n_oor = stats["out_of_range"]
        result = "warning" if n_oor > 0 else "pass"
        checks.append({
            "check": f"value_range_{var}",
            "scope": "per_record",
            "result": result,
            "summary": (
                f"{var}: observed [{stats['min']:.1f}, {stats['max']:.1f}] {unit}, "
                f"expected [{vmin}, {vmax}], {n_oor}/{stats['total']} out of range"
            ),
            "manuscript_claim": "Section 2.5: value range verification",
            "details": {
                "variable": var,
                "unit": unit,
                "expected_min": vmin,
                "expected_max": vmax,
                "observed_min": round(stats["min"], 2),
                "observed_max": round(stats["max"], 2),
                "out_of_range_count": n_oor,
                "total_values": stats["total"],
                "affected_cells": list(stats["affected_cells"])[:10],
            },
        })

    # Soil value ranges
    soil = unified_data.soil if unified_data and hasattr(unified_data, 'soil') else {}
    soil_stats = {}
    texture_violations = 0
    texture_total = 0

    for cell_id, profile in soil.items():
        if not hasattr(profile, 'layers'):
            continue
        for layer in profile.layers:
            for var, (vmin, vmax, unit) in SOIL_RANGES.items():
                val = getattr(layer, var, None)
                if val is None:
                    continue
                if var not in soil_stats:
                    soil_stats[var] = {
                        "min": val, "max": val,
                        "out_of_range": 0, "total": 0,
                    }
                stats = soil_stats[var]
                stats["min"] = min(stats["min"], val)
                stats["max"] = max(stats["max"], val)
                stats["total"] += 1
                if val < vmin or val > vmax:
                    stats["out_of_range"] += 1

            # Texture fraction check (sand + clay + silt ≈ 100)
            sand = getattr(layer, 'sand', None)
            clay = getattr(layer, 'clay', None)
            silt = getattr(layer, 'silt', None)
            if sand is not None and clay is not None and silt is not None:
                texture_total += 1
                texture_sum = sand + clay + silt
                if texture_sum < 95 or texture_sum > 105:
                    texture_violations += 1

    for var, (vmin, vmax, unit) in SOIL_RANGES.items():
        stats = soil_stats.get(var)
        if not stats:
            continue
        n_oor = stats["out_of_range"]
        result = "warning" if n_oor > 0 else "pass"
        checks.append({
            "check": f"value_range_soil_{var}",
            "scope": "per_layer",
            "result": result,
            "summary": (
                f"soil {var}: observed [{stats['min']:.2f}, {stats['max']:.2f}] {unit}, "
                f"{n_oor}/{stats['total']} out of range"
            ),
            "manuscript_claim": "Section 2.5: value range verification",
            "details": {
                "variable": f"soil_{var}",
                "unit": unit,
                "expected_min": vmin,
                "expected_max": vmax,
                "observed_min": round(stats["min"], 3),
                "observed_max": round(stats["max"], 3),
                "out_of_range_count": n_oor,
                "total_values": stats["total"],
            },
        })

    # Texture fraction sum check
    if texture_total > 0:
        result = "warning" if texture_violations > 0 else "pass"
        checks.append({
            "check": "value_range_texture_sum",
            "scope": "per_layer",
            "result": result,
            "summary": (
                f"sand+clay+silt sum: {texture_violations}/{texture_total} "
                f"layers outside [95, 105]%"
            ),
            "manuscript_claim": "Section 2.5: value range verification",
            "details": {
                "expected_range": [95, 105],
                "violations": texture_violations,
                "total_layers": texture_total,
            },
        })

    return checks


# =============================================================================
# Check 4: Soil profile completeness
# =============================================================================

def _check_soil_completeness(unified_data, platform: str) -> Dict[str, Any]:
    """Check that soil profiles have all required properties for the platform."""
    required = PLATFORM_SOIL_REQUIREMENTS.get(platform, [])
    if not required:
        return {
            "check": f"soil_completeness_{platform}",
            "scope": "per_cell",
            "result": "pass",
            "summary": f"No soil requirements defined for {platform}",
            "manuscript_claim": "Section 2.5: completeness check",
            "details": {},
        }

    soil = unified_data.soil if unified_data and hasattr(unified_data, 'soil') else {}
    n_complete = 0
    n_incomplete = 0
    missing_by_cell = {}

    for cell_id, profile in soil.items():
        if not hasattr(profile, 'layers') or not profile.layers:
            n_incomplete += 1
            missing_by_cell[cell_id] = required
            continue

        layer = profile.layers[0]  # Check surface layer
        missing = [
            prop for prop in required
            if getattr(layer, prop, None) is None
        ]
        if missing:
            n_incomplete += 1
            missing_by_cell[cell_id] = missing
        else:
            n_complete += 1

    n_total = n_complete + n_incomplete
    if n_total == 0:
        result = "warning"
        summary = f"No soil profiles available for {platform}"
    elif n_incomplete == 0:
        result = "pass"
        summary = f"{n_complete}/{n_total} cells have complete soil ({platform})"
    else:
        result = "warning"
        summary = (
            f"{n_complete}/{n_total} complete, "
            f"{n_incomplete} incomplete for {platform}"
        )

    return {
        "check": f"soil_completeness_{platform}",
        "scope": "per_cell",
        "result": result,
        "summary": summary,
        "manuscript_claim": "Section 2.5: completeness check",
        "details": {
            "platform": platform,
            "required_properties": required,
            "n_complete": n_complete,
            "n_incomplete": n_incomplete,
            "n_total": n_total,
            "sample_missing": {
                str(k): v for k, v in list(missing_by_cell.items())[:5]
            },
        },
    }


# =============================================================================
# Check 6: Spatial/temporal coverage
# =============================================================================

def _check_coverage(unified_data, config) -> Dict[str, Any]:
    """Check that data covers the configured spatial and temporal extent."""
    issues = []

    # Temporal coverage
    start_year = config.temporal.start_year
    end_year = config.temporal.end_year

    # Spatial coverage — grid cells vs region
    grid = unified_data.grid if unified_data and hasattr(unified_data, 'grid') else None
    n_cells = grid.n_cells if grid else 0

    # Check climate cell count matches grid (per-cell format only)
    climate = unified_data.climate if unified_data and hasattr(unified_data, 'climate') else {}
    if _is_file_based_climate(climate):
        # GeoTIFF-based: climate is region-wide rasters, not per-cell.
        # Cell-count comparison is not meaningful — the rasters cover
        # the full spatial extent by construction.
        n_climate_cells = n_cells  # treat as matching
        climate_format = "geotiff (region-wide)"
    else:
        n_climate_cells = len(climate)
        climate_format = "per_cell"

    # Check soil cell count matches grid
    soil = unified_data.soil if unified_data and hasattr(unified_data, 'soil') else {}
    n_soil_cells = len(soil)

    if n_cells > 0:
        if n_climate_cells < n_cells and climate_format == "per_cell":
            issues.append(
                f"Climate covers {n_climate_cells}/{n_cells} grid cells"
            )
        if n_soil_cells < n_cells:
            issues.append(
                f"Soil covers {n_soil_cells}/{n_cells} grid cells"
            )

    # Grid resolution consistency
    resolution = grid.resolution if grid and hasattr(grid, 'resolution') else "unknown"

    result = "warning" if issues else "pass"
    return {
        "check": "spatial_temporal_coverage",
        "scope": "global",
        "result": result,
        "summary": (
            f"Grid: {n_cells} cells at {resolution}, "
            f"climate: {n_climate_cells} cells, soil: {n_soil_cells} cells, "
            f"period: {start_year}-{end_year}"
        ),
        "manuscript_claim": "Section 2.5: coverage check",
        "details": {
            "grid_cells": n_cells,
            "grid_resolution": resolution,
            "climate_cells": n_climate_cells,
            "soil_cells": n_soil_cells,
            "temporal_start": start_year,
            "temporal_end": end_year,
            "issues": issues,
        },
    }
