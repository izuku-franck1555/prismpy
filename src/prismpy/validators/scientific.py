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

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _load_region_bounds() -> Dict[str, Any]:
    """Load region-specific validation bounds from JSON config."""
    bounds_path = Path(__file__).parent / "region_bounds.json"
    if not bounds_path.exists():
        return {"regions": [], "universal": {"thresholds": {}}}
    with open(bounds_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _detect_region(lat: float, lon: float) -> Dict[str, Any]:
    """Auto-detect agro-ecological region from coordinates.

    Checks the study area centroid against each region's bounding box.
    Returns the first matching region, or the universal fallback.
    """
    config = _load_region_bounds()
    for region in config.get("regions", []):
        bbox = region.get("bbox", [])
        if len(bbox) == 4:
            minx, miny, maxx, maxy = bbox
            if minx <= lon <= maxx and miny <= lat <= maxy:
                return region
    return config.get("universal", {"id": "universal", "thresholds": {}})


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
    "bulk_density": (0.5, 1.9, "g/cm³"),  # Tightened per specialist — catches unit errors
}

# Platform-specific required soil properties
PLATFORM_SOIL_REQUIREMENTS = {
    "craft": ["sand", "clay", "silt", "organic_carbon", "ph", "bulk_density"],
    "pythia": ["sand", "clay", "silt", "organic_carbon", "ph", "bulk_density"],
    "acea": ["sand", "clay", "organic_carbon"],
    "sarra_py": ["sand", "clay"],  # SARRA-Py derives HumFC/HumPF from texture
}

# F16 — canonical per-layer count each platform's translator emits.
# Per-layer validators (e.g., ``value_range_texture_sum``) and the
# soil-layer narrative consumer use this as the platform-side
# expectation; the actual per-cell layer count comes from iterating
# the profile's layers (which can vary by source cascade outcome,
# especially for PYTHIA where iSDA → HWSD fallback may surface
# different layer depths). The constant captures the conventional
# count; downstream consumers compare actual against expected to
# narrate "X of Y layers" honestly.
LAYERS_PER_PLATFORM: Dict[str, int] = {
    "craft": 6,     # 6-layer profile (surface to ~200 cm)
    "pythia": 2,    # surface + subsurface
    "acea": 4,      # 4-layer profile (canonical ACEA depth horizons)
    "sarra_py": 2,  # surface texture (root + subsoil)
}


def run_scientific_validation(
    unified_data,
    config,
    enabled_platforms: Optional[List[str]] = None,
    sarra_climate_per_cell: Optional[Dict[int, Dict[str, list]]] = None,
) -> Dict[str, Any]:
    """Run all 6 Tier 1 scientific validation checks.

    Args:
        unified_data: UnifiedData from the HARMONIZE stage
        config: ProjectConfig for temporal/spatial reference
        enabled_platforms: List of enabled platform names
        sarra_climate_per_cell: V2-22c-PRE.2.4 (D14) — per-cell
            sampled climate values for SARRA-Py file-based runs.
            Shape: ``{cell_id: {var: [pixel_value, ...]}}``. When
            provided, the value-range check synthesizes per-cell
            ``value_range_<var>`` checks with `affected_cells`
            lists; without it, file-based climate stays on the
            delegated-info path. Populated by
            :func:`prismpy.validators.post_translate.sample_sarra_py_per_cell`
            in the validate stage when SARRA-Py is enabled.

    Returns:
        Dict with 'checks' list and 'overall_result' rollup
    """
    checks = []
    enabled = enabled_platforms or []

    # Check 1: Temporal completeness
    checks.append(_check_temporal_completeness(
        unified_data, config, enabled_platforms=enabled,
    ))

    # Check 2: Cross-variable consistency
    checks.append(_check_cross_variable_consistency(unified_data))

    # Check 3: Value range — pass through the per-cell sampled
    # values so the SARRA-Py file-based branch synthesizes per-cell
    # checks (PRE.2.4).
    checks.extend(_check_value_ranges(
        unified_data, sarra_climate_per_cell=sarra_climate_per_cell,
    ))

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

    # V2-22c-PRE.1.9 (D36) — per-cell coverage checks. SEPARATE from
    # the existing global `spatial_temporal_coverage` (kept above);
    # these emit per-cell `affected_cells` lists that PRE.1.2's
    # failed_checks pivot reads. Cockpit's Layer 1 fill rule + Layer
    # 3 dimension toggle (Coverage) depend on these.
    checks.append(_check_coverage_climate_cells(unified_data))
    checks.append(_check_coverage_soil_cells(unified_data))

    # Check 7: Region-specific bounds (V2-20)
    checks.append(_check_region_bounds(unified_data, config))

    # Overall rollup. G7 §2 added the ``unavailable`` result; the
    # rollup keeps backward-compat by mapping it to a non-fail
    # non-warning state — the existing certificate consumer treats
    # the run as "pass" when its checks ran cleanly, and the per-
    # cell ``data_availability`` surface carries the absence
    # narrative. The one new top-level state, ``unavailable``,
    # fires only when EVERY ran-or-not check came back as
    # unavailable — i.e., neither axis had data — so the consumer
    # can distinguish "no checks ran" from "all checks passed".
    results = [c["result"] for c in checks]
    n_unavailable = results.count("unavailable")
    runnable = [r for r in results if r != "unavailable"]
    if "fail" in runnable:
        overall = "fail"
    elif "warning" in runnable:
        overall = "warning"
    elif n_unavailable > 0 and not any(r in ("pass", "warning", "fail") for r in runnable):
        # No check produced a runnable verdict; the entire validation
        # short-circuited on unavailable axes.
        overall = "unavailable"
    else:
        overall = "pass"

    # Summary statistics block (V2-20 specialist request #6)
    summary_stats = _compute_summary_stats(unified_data, config)

    # V2-20 UX-expert item 5: restructure into 5 manuscript-aligned
    # categories + items 6-8 (passed boolean, unit field, category field)
    categories = _restructure_to_categories(checks)

    # Overall rollup at category level
    categories_passed = sum(1 for c in categories.values() if c["passed"])

    return {
        # G7 §2 — version bump to "2.1" once the validator can emit
        # the unavailable result. v2.0 reports never carried
        # ``unavailable`` records and the absent ``n_unavailable``
        # key is the explicit shim for v2.0 readers (defaults to 0
        # via ``.get("n_unavailable", 0)``).
        "validation_version": "2.1",
        "passed": overall != "fail",
        "overall_result": overall,
        "categories_passed": categories_passed,
        "categories_total": len(categories),
        "categories": categories,
        "summary_statistics": summary_stats,
        # Flat list preserved for backward compat + flat-list operations
        "checks": checks,
        "n_checks": len(checks),
        "n_pass": results.count("pass"),
        "n_warning": results.count("warning"),
        "n_fail": results.count("fail"),
        "n_unavailable": n_unavailable,
    }


# Category mapping: check name prefix → manuscript category
_CATEGORY_MAP = {
    "temporal_completeness": "completeness",
    "cross_variable_consistency": "ranges",
    "value_range_": "ranges",
    "soil_completeness_": "completeness",
    "format_compliance": "schema",
    "spatial_temporal_coverage": "coverage",
    # V2-22c-PRE.1.9 codex P2 #3 — the per-cell coverage checks
    # added by D36 must classify under "coverage" so the validation
    # report's category rollup matches the cockpit's coverage chip
    # rendering. Without these mappings the fallback category in
    # _get_check_category puts them under "schema" → category drift
    # vs. the cockpit's per-cell `failed_checks` pivot which labels
    # them `coverage_per_cell`.
    "coverage_climate_cells": "coverage",
    "coverage_soil_cells": "coverage",
    "region_specific_bounds": "ranges",
    "post_translate_consistency_": "ranges",
    "post_translate_range_": "ranges",
    "post_translate_date_continuity_": "completeness",
    "post_translate_climate_": "ranges",
}

_CATEGORY_META = {
    "schema": {
        "label": "Schema Conformance",
        "subtitle": "File structure, headers, and format compliance",
    },
    "ranges": {
        "label": "Value Ranges",
        "subtitle": "Climate and soil values within physical and regional bounds",
    },
    "completeness": {
        "label": "Completeness",
        "subtitle": "Temporal coverage and soil profile availability",
    },
    "spatial": {
        "label": "Spatial Consistency",
        "subtitle": "CRS consistency and grid alignment",
    },
    "coverage": {
        "label": "Coverage",
        "subtitle": "Temporal and spatial extent match configuration",
    },
}


def _get_check_category(check_name: str) -> str:
    """Map a check name to its manuscript category."""
    for prefix, category in _CATEGORY_MAP.items():
        if check_name.startswith(prefix):
            return category
    return "schema"  # default for unknown checks


def _restructure_to_categories(
    checks: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Restructure flat check list into 5 manuscript-aligned categories.

    UX-expert item 5: groups checks under schema/ranges/completeness/
    spatial/coverage. Also applies items 6-8:
    - Item 6: `passed: bool` on each check
    - Item 7: `unit` field on range checks (from details)
    - Item 8: `category` field on each check
    """
    categories: Dict[str, Dict[str, Any]] = {}

    # Initialize all 5 categories
    for cat_id, meta in _CATEGORY_META.items():
        categories[cat_id] = {
            "passed": True,
            "label": meta["label"],
            "subtitle": meta["subtitle"],
            "checks": [],
        }

    for check in checks:
        check_name = check.get("check", "")
        cat = _get_check_category(check_name)

        # Item 6: add passed boolean. G7 §2 — ``unavailable`` is not
        # a pass; treat it like info-but-explicit so the cockpit's
        # category card renders the unavailable badge rather than a
        # green tick. The category-level ``passed`` flag below stays
        # true on unavailable (the category did not fail; it
        # short-circuited), matching the §1 schema invariant where
        # ``data_availability='unavailable'`` is a fourth state
        # alongside pass/warning/fail.
        result = check.get("result", "pass")
        check["passed"] = result in ("pass", "info")

        # Item 7: add unit field for range checks (extract from details)
        details = check.get("details", {})
        if "unit" in details and "unit" not in check:
            check["unit"] = details["unit"]

        # Item 8: add category field
        check["category"] = cat

        # Add to category
        if cat in categories:
            categories[cat]["checks"].append(check)
            if result == "fail":
                categories[cat]["passed"] = False

    return categories


def _compute_summary_stats(unified_data, config) -> Dict[str, Any]:
    """Compute descriptive summary statistics for the validation report.

    Saves researchers from computing basic stats themselves.
    """
    stats: Dict[str, Any] = {
        "temporal": {
            "start_year": config.temporal.start_year,
            "end_year": config.temporal.end_year,
            "spinup_years": config.temporal.spinup_years,
        },
    }

    # Grid stats
    grid = unified_data.grid if unified_data and hasattr(unified_data, 'grid') else None
    if grid:
        stats["grid"] = {
            "n_cells": grid.n_cells,
            "resolution": getattr(grid, 'resolution', 'unknown'),
        }

    # Soil source breakdown
    soil = unified_data.soil if unified_data and hasattr(unified_data, 'soil') else {}
    if soil:
        sources = {}
        for profile in soil.values():
            src = getattr(profile, 'source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        stats["soil"] = {
            "n_profiles": len(soil),
            "source_breakdown": sources,
        }

    # Climate stats (per-cell format only)
    climate = unified_data.climate if unified_data and hasattr(unified_data, 'climate') else {}
    if climate and not _is_file_based_climate(climate):
        tmax_vals, tmin_vals, precip_vals, srad_vals = [], [], [], []
        n_days = 0
        for ts in climate.values():
            if not hasattr(ts, 'records'):
                continue
            for r in ts.records:
                n_days += 1
                if r.tmax is not None:
                    tmax_vals.append(r.tmax)
                if r.tmin is not None:
                    tmin_vals.append(r.tmin)
                if r.precip is not None:
                    precip_vals.append(r.precip)
                if r.srad is not None:
                    srad_vals.append(r.srad)

        stats["climate"] = {
            "n_cells": len(climate),
            "n_days_total": n_days,
        }
        if tmax_vals:
            stats["climate"]["tmax_range"] = [
                round(min(tmax_vals), 1), round(max(tmax_vals), 1)
            ]
        if tmin_vals:
            stats["climate"]["tmin_range"] = [
                round(min(tmin_vals), 1), round(max(tmin_vals), 1)
            ]
        if precip_vals:
            annual_total = sum(precip_vals)
            n_years = max(1, config.temporal.end_year - config.temporal.start_year + 1)
            stats["climate"]["precip_annual_mean_mm"] = round(
                annual_total / max(1, len(climate)) / n_years, 0
            )
        if srad_vals:
            stats["climate"]["srad_mean"] = round(
                sum(srad_vals) / len(srad_vals), 1
            )
    elif climate and _is_file_based_climate(climate):
        stats["climate"] = {
            "data_format": "geotiff_per_day",
            "rainfall_files": climate.get("rainfall_file_count", 0),
            "agera5_variables": climate.get("agera5_variables", {}),
        }

    # Region detection
    region = unified_data.region if unified_data and hasattr(unified_data, 'region') else None
    if region and hasattr(region, 'bounds'):
        center_lat = (region.bounds.miny + region.bounds.maxy) / 2
        center_lon = (region.bounds.minx + region.bounds.maxx) / 2
        detected = _detect_region(center_lat, center_lon)
        stats["region_detected"] = detected.get("id", "universal")
        stats["region_name"] = detected.get("name", "Universal")

    return stats


# =============================================================================
# Check 1: Temporal completeness
# =============================================================================

def _check_temporal_completeness(
    unified_data, config,
    enabled_platforms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Check that climate data covers the full requested period.

    The expected per-cell range matches each translator's actual
    fetch contract — verified empirically by re-grepping the
    download sites:

    * **PYTHIA** ``pythia/translator.py:1987`` and **CRAFT**
      ``craft/translator.py:1395`` both fetch from
      ``f"{start_year}-01-01"`` — the warmup period is handled by
      the model internally, same precedent as the SARRA-Py
      file-based branch (see comment at the file-based fallback
      below). For these platforms the validator's per-cell
      expectation must be ``expected_days_no_spinup``; using the
      with-spinup denominator silently halves the reported
      completeness on any project with ``spinup_years > 0`` and
      breaks Aminata's "is my package complete?" trust check.
    * **ACEA** ``acea/translator.py:410-413`` fetches from
      ``date(start_year - spinup, 1, 1)`` — the warmup period is
      pre-fetched and lives in the per-cell records. For ACEA
      the expectation must be ``expected_days_with_spinup``;
      using the no-spinup denominator would over-claim
      completeness (n_actual > expected) and silently mask
      genuine gap detection.
    * **SARRA-Py** file-based (delegated below) keeps the
      no-spinup expectation — model handles warmup internally,
      same as PYTHIA + CRAFT.

    The whether-ACEA-should-fetch-pre-spinup-data domain question
    (DSSAT vs CSM vs AquaCropOS warmup-state-generation
    conventions) is deferred to a future sprint with crop-
    modeling-specialist time; this check matches the validator
    to what each translator currently does, not to a chosen-
    canonical-fetch-contract.

    Args:
        unified_data: UnifiedData container (climate dict + region).
        config: ProjectConfig for temporal bounds + crop calendar.
        enabled_platforms: List of enabled platform names. When
            ``"acea"`` is among them the per-cell expectation
            includes the spinup window; otherwise it does not.
            ``None`` (back-compat default) maps to the
            no-spinup expectation, matching the empirically-most-
            common PYTHIA + CRAFT path.
    """
    start_year = config.temporal.start_year
    end_year = config.temporal.end_year
    spinup = config.temporal.spinup_years

    expected_start_with_spinup = date(start_year - spinup, 1, 1)
    expected_start_no_spinup = date(start_year, 1, 1)
    crop_cal = config.crop.calendar if config.crop else None
    expected_end = config.temporal.get_climate_end_date(crop_cal)
    expected_days_with_spinup = (expected_end - expected_start_with_spinup).days + 1
    expected_days_no_spinup = (expected_end - expected_start_no_spinup).days + 1

    # Pick the per-cell expectation that matches the active
    # translator's actual fetch range. ACEA pre-fetches the
    # spinup window into per-cell records; PYTHIA + CRAFT do not.
    enabled = enabled_platforms or []
    if "acea" in enabled:
        expected_per_cell = expected_days_with_spinup
    else:
        expected_per_cell = expected_days_no_spinup

    # Count actual days per cell
    climate = unified_data.climate if unified_data and hasattr(unified_data, 'climate') else {}
    # G7 §2 — short-circuit when the climate axis is unavailable. The
    # prior "warning" result was misleading: the validator did not
    # run a check, so reporting a low-severity outcome implied data
    # was inspected and found wanting. The unavailable record is the
    # ICASA MISDAT marker the cockpit banner + cross-hatch palette
    # bind against.
    if not _has_climate_data(climate):
        return _unavailable(
            check_id="temporal_completeness",
            scope="global",
            cause="no_climate_fetch",
            details={
                "expected_days": expected_per_cell,
                "actual_cells": 0,
            },
        )

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
        n_missing = expected_per_cell - n_actual
        total_present += n_actual
        total_expected += expected_per_cell
        if n_missing > 0:
            cell_gaps[cell_id] = n_missing

    # If no cells have per-day records but climate dict has file-based
    # data (SARRA-Py: rainfall_file_count, agera5_variables), count
    # GeoTIFF files per variable as the completeness metric. SARRA-Py
    # downloads data for the actual period only; spinup is handled by
    # the model internally — same precedent as PYTHIA + CRAFT above.
    if cells_with_records == 0 and climate:
        return _check_temporal_completeness_file_based(
            climate, expected_days_no_spinup
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
            "expected_days_per_cell": expected_per_cell,
            "n_cells": len(climate),
            "completeness_pct": round(completeness * 100, 2),
            "cells_with_gaps": len(cell_gaps),
            # V2-22c-PRE.1.3 — un-truncated. Cockpit's per-cell drill-down
            # is the strictest downstream consumer (§6.4 schema-bounds
            # discipline); `[:10]` was a UI-driven pre-cockpit optimization.
            "gap_details": {str(k): v for k, v in cell_gaps.items()},
            # V2-22c-PRE codex P2 #1 — `_build_cell_summary` pivots
            # per-cell `failed_checks` from `details.affected_cells`,
            # not `gap_details`. Without this list a per-cell check
            # with `result='fail'` or `'warning'` and `scope='per_cell'`
            # would silently omit the `temporal_completeness` entry
            # from every cell's `failed_checks` array. Sorted ASC for
            # deterministic JSON diffs.
            "affected_cells": sorted(cell_gaps.keys()),
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

    # If AgERA5 was expected (SARRA-Py pipeline) but has no files,
    # report it as zero-count entries so completeness check fails
    # instead of silently passing on rainfall alone.
    agera5_expected = climate.get("agera5_expected", False)
    if agera5_expected and not agera5_vars:
        for missing_var in [
            "2m_temperature_24_hour_maximum",
            "2m_temperature_24_hour_minimum",
            "solar_radiation_flux_daily",
            "2m_temperature_24_hour_mean",
        ]:
            var_counts[missing_var] = 0

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


# =============================================================================
# G7 §2 — Unavailable-axis short-circuit helper
# =============================================================================
#
# The ICASA MISDAT convention says: when an input substrate (climate or soil)
# could not be obtained, downstream validation must record that absence
# explicitly rather than reporting a misleading pass/warning/fail. The
# ``unavailable`` result is a fourth value alongside pass/warning/fail/info;
# consumers (cockpit banner, region-health card, fetch-fallback metrics) read
# it to route the cell to the cross-hatch palette and the documented-MISDAT
# banner copy.
#
# The helper centralises the canonical dict shape so every short-circuit site
# emits the same {check, scope, result, summary, manuscript_claim, details}
# structure with the ``cause`` discriminator on details. A copy-edit at one
# call site can no longer drift from the rest — the prior diffuse pattern
# (each validator hand-rolling its "no climate data" branch with subtly
# different summaries and result strings) was the bug class §2 aims to retire.

_UNAVAILABLE_RESULT: str = "unavailable"

# Canonical cause vocabulary aligned with crop-modeling-specialist's §2
# pack. The strings appear verbatim in ``details.cause`` on every
# unavailable record; downstream consumers (banner copy, Methods-tab
# audit, Dr. Kofi grep) match against these literals. Renaming any of
# them is a contract change — bump ``validation_version`` if you do.
UNAVAILABLE_CAUSES: tuple[str, ...] = (
    "no_climate_fetch",
    "no_soil_match",
    "soil_cascade_exhausted",
    "no_climate_and_soil_fetch",
    "no_translated_output",
)
_UNAVAILABLE_CAUSE_LABELS: Dict[str, str] = {
    "no_climate_fetch": "climate data was not fetched",
    "no_soil_match": "soil cascade returned no profile",
    "soil_cascade_exhausted": "soil cascade exhausted (iSDA → HWSD all empty)",
    "no_climate_and_soil_fetch": "neither climate nor soil data was fetched",
    "no_translated_output": "platform translation produced no output for this cell",
}


# Canonical iteration constants for the Tier 1 climate / soil checks.
# crop-modeling-specialist's §2 pack §3 + §3.4 follow-up calls for
# these as the source of truth derived from the existing range +
# requirement dicts; production code uses them at the sibling-sweep
# callsites AND test code uses them for DRY assertions. Generator
# expressions over CLIMATE_RANGES / SOIL_RANGES /
# PLATFORM_SOIL_REQUIREMENTS keep the iteration sets in lockstep
# with the source-of-truth dicts — adding a new variable to any
# range dict automatically updates the corresponding iteration set
# without a parallel rename pass.
CLIMATE_TIER1_CHECKS: tuple[str, ...] = (
    "temporal_completeness",
    "cross_variable_consistency",
    # Axis-level aggregator the §2 short-circuit emits when the
    # whole climate axis is unavailable — distinct from the per-
    # variable records below.
    "value_range_climate",
    *(f"value_range_{var}" for var in CLIMATE_RANGES),
    "coverage_climate_cells",
    "region_specific_bounds",
)
SOIL_CHECKS_ALL: tuple[str, ...] = (
    # Axis-level aggregator emitted when the soil axis is fully
    # unavailable; per-variable records derive from SOIL_RANGES.
    "value_range_soil",
    *(f"value_range_soil_{var}" for var in SOIL_RANGES),
    "value_range_texture_sum",
    "coverage_soil_cells",
    # Platform-specific completeness check_ids follow the
    # ``soil_completeness_<platform>`` pattern. Generated over
    # PLATFORM_SOIL_REQUIREMENTS so a new platform key is picked
    # up automatically — adding ``oryza`` to the requirements
    # dict propagates here without a parallel iteration update.
    *(f"soil_completeness_{platform}" for platform in PLATFORM_SOIL_REQUIREMENTS),
)


def _unavailable(
    check_id: str,
    scope: str,
    cause: str,
    cell_id: Optional[Any] = None,
    *,
    summary: Optional[str] = None,
    manuscript_claim: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    affected_cells: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical "validator did not run" record for an axis
    that has no input data.

    Args:
        check_id: The validator's stable check id (e.g.,
            ``temporal_completeness``, ``value_range_soil_sand``).
            Same value the validator would emit on a successful run,
            so the cockpit's check-id pivot still binds.
        scope: ``per_cell`` / ``per_record`` / ``per_layer`` / ``global``
            — same vocabulary as a normal record. Per-cell scopes carry
            an empty ``affected_cells`` list when the axis is fully
            unavailable region-wide.
        cause: Discriminator string from ``UNAVAILABLE_CAUSES``. Selects
            the human-readable label woven into the default summary
            and the audit-grep marker on ``details.cause``.
        cell_id: Optional per-cell id. When set, ``details.cell_id``
            carries the value so the consumer can match the record
            to a single cell rather than scanning ``affected_cells``.
            The crop-modeling-specialist §2 pack signature pinned
            this position; it stays optional for axis-level callers.
        summary: Optional override of the default summary. Pass when
            the validator wants to add nuance (e.g., "and the
            file-based delegate also could not run") without losing
            the unavailable framing.
        manuscript_claim: Optional override of the default manuscript
            claim.
        details: Optional extra details to merge after ``cause``. The
            helper always sets ``details.cause`` and
            ``details.icasa_misdat=True``; caller-supplied keys
            override neither.
        affected_cells: Optional explicit per-cell list — only set
            when the validator scope is per-cell AND the absence is
            partial (e.g., one cell missing climate while others
            have it). Region-wide unavailability leaves the list at
            ``[]`` so the consumer can render the banner once
            without scanning a redundant per-cell echo.

    Returns:
        Dict with the same six top-level keys as a regular validator
        record. ``result`` is the literal ``"unavailable"`` constant;
        downstream rollups (run_scientific_validation overall,
        cockpit category passed flag) treat it as a non-fail signal
        that nonetheless does not promote to ``pass``.
    """
    label = _UNAVAILABLE_CAUSE_LABELS.get(cause, "input data was not available")
    default_summary = "Data not available; validation skipped."
    default_manuscript = (
        f"Cell data was not available at fetch time. Validation was "
        f"not performed for {check_id}. Per ICASA MISDAT convention, "
        f"missing-data cells are reported as unavailable, distinct "
        f"from validation failures. ({label})"
    )
    final_manuscript = manuscript_claim if manuscript_claim is not None else default_manuscript
    # G7 §2 — defensive guard per crop-modeling-specialist's pack §3
    # note 3: a caller who overrides ``manuscript_claim`` MUST
    # preserve the ICASA / MISDAT semantic anchor so Dr. Kofi's
    # audit grep continues to find the standards lineage on every
    # unavailable record. The check is a logger.warning rather than
    # a raise because (a) the contract is stricter than what would
    # block a production run, and (b) older fixtures or external
    # writers may legitimately predate the discipline. The warning
    # surfaces the violation in audit logs without breaking the
    # pipeline.
    if "ICASA" not in final_manuscript and "MISDAT" not in final_manuscript:
        logger.warning(
            "_unavailable() override of manuscript_claim missing the "
            "ICASA / MISDAT semantic anchor for check_id=%r; Dr. Kofi "
            "audit-grep continuity at risk. Got: %r",
            check_id, final_manuscript,
        )
    merged_details: Dict[str, Any] = {
        "cause": cause,
        "icasa_misdat": True,
    }
    if cell_id is not None:
        merged_details["cell_id"] = cell_id
    if details:
        for k, v in details.items():
            if k not in merged_details:
                merged_details[k] = v
    # The empty list keeps the executor's per-cell pivot key uniform
    # — the pivot reads ``details.affected_cells`` and would emit
    # category=None entries on an absent key.
    merged_details.setdefault("affected_cells", list(affected_cells or []))
    return {
        "check": check_id,
        "scope": scope,
        "result": _UNAVAILABLE_RESULT,
        "summary": summary or default_summary,
        "manuscript_claim": final_manuscript,
        "details": merged_details,
    }


def _has_climate_data(climate: Any) -> bool:
    """True when ``unified_data.climate`` carries enough shape for the
    climate-axis validators to do meaningful work.

    The contract is intentionally narrow: file-based climate
    (SARRA-Py's ``{rainfall_dir, agera5_dir, ...}`` shape) counts as
    available even before sampling, because the file-based validator
    branches at each call site emit their own info/warning records
    (the validator did run; it just delegated to a downstream
    sampler). A truly empty ``{}`` dict — neither per-cell records nor
    file-based directories — is the unavailable case.
    """
    if not climate:
        return False
    if _is_file_based_climate(climate):
        return True
    if not isinstance(climate, dict):
        return False
    # Per-cell shape: at least one cell whose ts.records is non-empty.
    for ts in climate.values():
        records = getattr(ts, "records", None)
        if records:
            return True
    return False


def _has_soil_data(soil: Any) -> bool:
    """True when ``unified_data.soil`` carries at least one profile
    with non-empty layers. The validator's per-cell loop already
    skips profiles whose ``layers`` list is empty; this helper makes
    the absence explicit at the entry guard so the validator doesn't
    iterate an empty dict and emit a hollow ``pass``."""
    if not soil:
        return False
    if not isinstance(soil, dict):
        return False
    for profile in soil.values():
        layers = getattr(profile, "layers", None)
        if layers:
            return True
    return False


def _check_cross_variable_consistency(unified_data) -> Dict[str, Any]:
    """Check physical consistency across climate variables."""
    climate = unified_data.climate if unified_data and hasattr(unified_data, 'climate') else {}
    # G7 §2 — short-circuit on unavailable climate. The prior `pass`
    # branch overclaimed (the check never ran on any record); the
    # unavailable record makes the absence the consumer's signal
    # rather than a phantom green-tick.
    if not _has_climate_data(climate):
        return _unavailable(
            check_id="cross_variable_consistency",
            scope="per_record",
            cause="no_climate_fetch",
        )

    # GeoTIFF-based climate (SARRA-Py): per-value consistency requires
    # opening thousands of raster files — too expensive for a validation
    # check. Report honestly as info with the limitation stated.
    #
    # NOTE(V2-22b/P.2): same emission pattern as value_range_climate
    # above, but no delegated post-translate target exists today for
    # cross-variable consistency. If a `post_translate_cross_variable_*`
    # check ever lands, mirror the executor post-merge gate so this
    # info escalates to a warning when the delegated records are
    # missing — see prismpy/pipeline/executor.py around the
    # `post_translate_range_sarra_py_` gate.
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
            f"Physical consistency checked across {total_records} daily records"
            + (f" — {fail_count} critical + {warn_count} advisory issues found"
               if (fail_count + warn_count) > 0
               else " — all consistent")
        ),
        "manuscript_claim": "Section 2.5: cross-variable consistency",
        "details": {
            "total_records": total_records,
            "violations": violations,
            # V2-22c-PRE.1.3 — un-truncated for cockpit drill-down.
            "affected_cells": list(affected_cells),
            "n_affected_cells": len(affected_cells),
        },
    }


# =============================================================================
# Check 3: Value range (universal physical bounds)
# =============================================================================

def _check_value_ranges(
    unified_data,
    sarra_climate_per_cell: Optional[Dict[int, Dict[str, list]]] = None,
) -> List[Dict[str, Any]]:
    """Check climate and soil values against universal physical bounds.

    V2-22c-PRE.2.4 (D14) — when ``sarra_climate_per_cell`` is provided,
    the file-based climate branch consumes those sampled values to
    emit per-cell value-range checks (with `affected_cells` lists)
    identical to the in-memory CRAFT/PYTHIA/ACEA shape. Without the
    parameter, the file-based branch keeps emitting only the
    delegated info record.
    """
    checks = []

    # Climate value ranges
    climate = unified_data.climate if unified_data and hasattr(unified_data, 'climate') else {}
    climate_stats = {}

    # G7 §2 — when the climate axis is unavailable the value-range
    # check cannot run on any climate variable. Emit a single axis-
    # level ``value_range_climate`` unavailable record rather than
    # six per-variable ones; downstream consumers (cockpit Value
    # Ranges category card, fetch banner) read the axis-level signal,
    # and a per-variable explosion would inflate the unavailable
    # count without adding information. The soil branch below
    # follows the same pattern.
    climate_axis_unavailable = not _has_climate_data(climate)
    if climate_axis_unavailable:
        checks.append(_unavailable(
            check_id="value_range_climate",
            scope="per_record",
            cause="no_climate_fetch",
        ))

    # GeoTIFF-based climate (SARRA-Py today; ACEA once it activates
    # srad): the scientific validator's in-memory path cannot check
    # value ranges — the data is on disk as per-day rasters. The
    # per-platform post-translate validator (for SARRA-Py,
    # `post_translate._validate_sarra_py_geotiffs`) opens a random
    # 10-file subset per variable and emits the authoritative
    # `post_translate_range_sarra_py_<var>` records at report time.
    #
    # The info record below DELEGATES to that downstream check rather
    # than CLAIMING it ran. Codex self-check HIGH: the prior
    # "spot-checked" phrasing asserted a sampled check had happened
    # even in cases where translation failed or post-translate
    # validation was skipped — the user would then see the info
    # record but no post_translate_range_* records and have no way
    # to distinguish "platform validation succeeded on a sample"
    # from "platform validation never ran". The delegating phrasing
    # tells them where to look and what absence means.
    if _is_file_based_climate(climate):
        checks.append({
            "check": "value_range_climate",
            "scope": "per_record",
            "result": "info",
            "summary": (
                "Climate value ranges for SARRA-Py are computed "
                "from 10 output files per variable: first, last, "
                "and 8 interior files chosen by filename-digest "
                "within evenly-spaced buckets. When available, "
                "the per-variable ranges appear below."
            ),
            "manuscript_claim": "Section 2.5: value range verification (delegated)",
            "details": {
                "data_format": "geotiff_per_day",
                "delegated_to": "post_translate._validate_sarra_py_geotiffs",
                "sample_policy": (
                    "stratified buckets, digest-steered interior"
                ),
                "coverage_kind": "delegated",
            },
        })
        # V2-22c-PRE.2.4 (D14) — when the caller supplied per-cell
        # sampled values, populate climate_stats so the per-cell
        # `value_range_<var>` checks emit with `affected_cells` lists
        # identical to the CRAFT/PYTHIA/ACEA shape. The cockpit's
        # Layer 1 fill rule reads these for SARRA-Py runs; without
        # this branch the cockpit shows every SARRA-Py cell red
        # because no per-cell value-range check produces a `pass`
        # record (the universal-fail bug per V2-22c contract §1.1).
        #
        # The canonical var mapping is `rain` → SARRA's "rain"
        # output, but CLIMATE_RANGES uses `rain` natively too.
        # Each per-cell list is a sample across N file reads;
        # min/max-aggregating per cell mirrors the per-cell-records
        # path's range computation.
        if sarra_climate_per_cell:
            # SARRA_PY_VAR_MAPPING (post_translate.py) emits canonical
            # vars `rain / tmax / tmin / srad`; scientific.py's
            # CLIMATE_RANGES uses `precip` for the precipitation
            # threshold. Translate the SARRA-Py-side `rain` → the
            # canonical scientific.py `precip` when populating stats
            # so the check_id ends up `value_range_precip` (matching
            # the CRAFT/PYTHIA/ACEA shape — the cockpit drill-down
            # treats SARRA-Py and the in-memory platforms identically
            # at the check_id surface).
            _sarra_to_canonical = {
                "rain": "precip",
                "tmax": "tmax",
                "tmin": "tmin",
                "srad": "srad",
            }
            for cell_id, var_bucket in sarra_climate_per_cell.items():
                for sarra_var, vals in var_bucket.items():
                    var = _sarra_to_canonical.get(sarra_var, sarra_var)
                    if var not in CLIMATE_RANGES:
                        continue
                    if not vals:
                        continue
                    vmin, vmax, unit = CLIMATE_RANGES[var]
                    if var not in climate_stats:
                        climate_stats[var] = {
                            "min": min(vals), "max": max(vals),
                            "out_of_range": 0, "total": 0,
                            "affected_cells": set(),
                        }
                    stats = climate_stats[var]
                    stats["min"] = min(stats["min"], min(vals))
                    stats["max"] = max(stats["max"], max(vals))
                    stats["total"] += len(vals)
                    for v in vals:
                        if v < vmin or v > vmax:
                            stats["out_of_range"] += 1
                            stats["affected_cells"].add(cell_id)
        # else: empty climate_stats → no per-variable climate checks
        # emitted; the delegated info record above is the only output.
    else:
        for cell_id, ts in climate.items():
            if not hasattr(ts, 'records'):
                continue
            if getattr(ts, 'source', '') == 'placeholder':
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
                f"{var}: {stats['min']:.1f} to {stats['max']:.1f} {unit} "
                f"(expected {vmin} to {vmax})"
                + (f" — {n_oor} of {stats['total']} values outside range" if n_oor > 0 else "")
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
                # V2-22c-PRE.1.3 — un-truncated for cockpit drill-down.
                "affected_cells": list(stats["affected_cells"]),
            },
        })

    # Soil value ranges
    soil = unified_data.soil if unified_data and hasattr(unified_data, 'soil') else {}
    # G7 §2 — short-circuit when the soil axis is unavailable. The
    # ``value_range_soil`` aggregator id mirrors the climate axis-
    # level record above; the per-variable subdivision is meaningful
    # only when at least one layer has data.
    if not _has_soil_data(soil):
        checks.append(_unavailable(
            check_id="value_range_soil",
            scope="per_layer",
            cause="no_soil_match",
        ))
        return checks
    soil_stats = {}
    texture_violations = 0
    texture_total = 0
    # V2-22c-PRE.1.4 / 1.5 — track per-(cell_id, layer_idx) violations
    # alongside the existing aggregate counters so the validator emits a
    # cockpit-consumable `affected_cells` list per check. Net-new
    # collection logic; the validator never recorded the failed cell IDs
    # before V2-22c. ``soil_violations_by_var[var]`` is the list per
    # soil variable; ``texture_violation_cells`` is the list for the
    # composite texture-sum check; ``soil_violation_details`` is the
    # downstream-feed payload for PRE.1.8 ``cell_failed_check_details``
    # flattening (each entry carries the cell + layer + value + bounds
    # context the cockpit drawer needs to render the violation row
    # directly without a second JSON load).
    soil_violations_by_var: Dict[str, List[Tuple[int, int]]] = {}
    texture_violation_cells: List[Tuple[int, int]] = []
    soil_violation_details: List[Dict[str, Any]] = []

    for cell_id, profile in soil.items():
        if not hasattr(profile, 'layers'):
            continue
        for layer_idx, layer in enumerate(profile.layers):
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
                    soil_violations_by_var.setdefault(var, []).append(
                        (cell_id, layer_idx),
                    )
                    soil_violation_details.append({
                        "cell_id": cell_id,
                        "layer_idx": layer_idx,
                        "variable": var,
                        "value": float(val),
                        "unit": unit,
                        "bounds": [vmin, vmax],
                    })

            # Texture fraction check (sand + clay + silt ≈ 100)
            sand = getattr(layer, 'sand', None)
            clay = getattr(layer, 'clay', None)
            silt = getattr(layer, 'silt', None)
            if sand is not None and clay is not None and silt is not None:
                texture_total += 1
                texture_sum = sand + clay + silt
                if texture_sum < 95 or texture_sum > 105:
                    texture_violations += 1
                    texture_violation_cells.append((cell_id, layer_idx))
                    soil_violation_details.append({
                        "cell_id": cell_id,
                        "layer_idx": layer_idx,
                        "variable": "texture_sum",
                        "value": float(texture_sum),
                        "unit": "%",
                        "bounds": [95, 105],
                    })

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
                f"Soil {var.replace('_', ' ')}: {stats['min']:.2f} to {stats['max']:.2f} {unit} "
                f"(expected {vmin} to {vmax})"
                + (f" — {n_oor} of {stats['total']} soil layers outside range" if n_oor > 0 else "")
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
                # V2-22c-PRE.1.5 — per-(cell_id, layer_idx) tuples for
                # each layer that violated this variable's range.
                # Sorted by (cell_id, layer_idx) ASC for deterministic
                # JSON diffs across runs (evaluator §2 binding).
                "affected_cells": sorted(soil_violations_by_var.get(var, [])),
                # V2-22c-PRE.1.8 — per-violation context for the cockpit
                # drawer; flattened into top-level cell_failed_check_details
                # by _build_cell_summary so consumers can render
                # "Cell #N layer L — <var>=<value> outside [<lo>, <hi>]"
                # without a second JSON join. Sorted to match the
                # affected_cells ordering for cockpit cursor stability.
                "violation_details": sorted(
                    [d for d in soil_violation_details if d["variable"] == var],
                    key=lambda d: (d["cell_id"], d["layer_idx"]),
                ),
            },
        })

    # Texture fraction sum check.
    #
    # F16 — soil-layer narrative emit. The summary wording was
    # "abnormal totals" pre-F16; renamed to the explicit physical
    # invariant "texture fractions that don't sum to 100%" so a
    # researcher (Aminata first-time, Moussa stakeholder) reads
    # the check name + summary as a single complete sentence
    # without needing to remember which "abnormal totals" the
    # validator was guarding against. The metadata block adds
    # per-cell vs per-layer counts so the consumer renders chip
    # text ("7 cells flagged") and drawer detail ("8 of 136
    # layers") from one canonical emit. The convention generalizes
    # to any future per-layer validator: emit per-layer counts
    # alongside affected_cells / violation_details so the cockpit
    # / drawer can decide which granularity to surface.
    if texture_total > 0:
        result = "warning" if texture_violations > 0 else "pass"
        # Per-cell metadata derived from the per-(cell_id, layer_idx)
        # violation list. Cell-level counts feed chip rendering;
        # layer-level counts feed drawer detail.
        flagged_cell_ids = {cid for cid, _ in texture_violation_cells}
        n_flagged_cells = len(flagged_cell_ids)
        # A cell is "multi-flagged" when more than one of its layers
        # carry texture violations — useful for the drawer's
        # "Cell #N: 3 of 6 layers flagged" rendering vs the more
        # common "1 of 6" single-layer case.
        n_per_cell: Dict[int, int] = {}
        for cid, _ in texture_violation_cells:
            n_per_cell[cid] = n_per_cell.get(cid, 0) + 1
        n_multi_flagged_cells = sum(1 for n in n_per_cell.values() if n > 1)
        # Per-cell layer-count typical (mode) — describes the cell-
        # level layer count Aminata/Moussa expect, derived from the
        # soil profiles actually iterated. Falls back to None when
        # no profiles carry layer counts (defensive; the texture
        # check only runs when at least one profile had layers).
        layer_counts = [
            len(profile.layers)
            for profile in soil.values()
            if hasattr(profile, "layers") and profile.layers
        ]
        n_layers_per_cell_typical = (
            max(set(layer_counts), key=layer_counts.count)
            if layer_counts else None
        )
        # Depth-range description across all profiles' layers.
        # Renders as e.g. "0–200 cm" for the canonical CRAFT 6-layer
        # profile. Falls back to None when depth_top/depth_bottom
        # are unset (defensive).
        depth_min = None
        depth_max = None
        for profile in soil.values():
            for layer in getattr(profile, "layers", []) or []:
                top = getattr(layer, "depth_top", None)
                bot = getattr(layer, "depth_bottom", None)
                if isinstance(top, (int, float)):
                    depth_min = top if depth_min is None else min(depth_min, top)
                if isinstance(bot, (int, float)):
                    depth_max = bot if depth_max is None else max(depth_max, bot)
        depth_range_description: Optional[str] = None
        if depth_min is not None and depth_max is not None:
            # Profiles sometimes carry depth in metres (CRAFT) or
            # centimetres (PYTHIA / ACEA); the convention here is
            # the canonical metric the loader writes — keep the
            # raw numbers and append the unit downstream loaders
            # populate. Until we resolve the m-vs-cm ambiguity,
            # render the bare range so the consumer sees the
            # actual numbers without an assumed unit.
            depth_range_description = f"{depth_min}–{depth_max}"
        checks.append({
            "check": "value_range_texture_sum",
            "scope": "per_layer",
            "result": result,
            "summary": (
                f"Soil texture fractions (sand + clay + silt) sum to ~100%"
                + (
                    f" — {texture_violations} of {texture_total} soil "
                    f"layers carry texture fractions that don't sum to 100%"
                    if texture_violations > 0
                    else f" for all {texture_total} soil layers"
                )
            ),
            "manuscript_claim": "Section 2.5: value range verification",
            "details": {
                "expected_range": [95, 105],
                "violations": texture_violations,
                "total_layers": texture_total,
                # F16 per-cell vs per-layer breakdown — chip text
                # binds to ``n_flagged_cells`` (cell-count surface);
                # drawer detail binds to ``n_flagged_layers`` /
                # ``n_total_layers`` (layer-count surface).
                "n_flagged_cells": n_flagged_cells,
                "n_flagged_layers": texture_violations,
                "n_total_layers": texture_total,
                "n_multi_flagged_cells": n_multi_flagged_cells,
                "n_layers_per_cell_typical": n_layers_per_cell_typical,
                "depth_range_description": depth_range_description,
                # V2-22c-PRE.1.4 — per-(cell_id, layer_idx) tuples for
                # each layer where sand+clay+silt fell outside [95, 105].
                # Sorted ASC for deterministic JSON diffs (evaluator §2).
                "affected_cells": sorted(texture_violation_cells),
                # V2-22c-PRE.1.8 — same flatten-into-cell_failed_check_details
                # rationale as the soil-quality family above. Sorted to
                # match affected_cells ordering for cockpit cursor stability.
                "violation_details": sorted(
                    [d for d in soil_violation_details
                     if d["variable"] == "texture_sum"],
                    key=lambda d: (d["cell_id"], d["layer_idx"]),
                ),
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
    # G7 §2 — when the soil axis carries no profiles or every
    # profile has empty layers, the completeness check cannot run.
    # Distinguish that absence from "no soil requirements" (the
    # branch above) — the latter is a platform configuration; this
    # is missing input data.
    if not _has_soil_data(soil):
        return _unavailable(
            check_id=f"soil_completeness_{platform}",
            scope="per_cell",
            cause="no_soil_match",
            details={
                "platform": platform,
                "required_properties": required,
                "n_complete": 0,
                "n_incomplete": 0,
                "n_total": 0,
            },
        )
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
            # V2-22c-PRE.1.3 — un-truncated for cockpit drill-down.
            "sample_missing": {
                str(k): v for k, v in missing_by_cell.items()
            },
            # V2-22c-PRE codex P2 #2 — `_build_cell_summary` pivots
            # per-cell `failed_checks` from `details.affected_cells`,
            # not `sample_missing`. Without this list, incomplete
            # soil cells never surface the `soil_completeness_<platform>`
            # entry on their per-cell chip strip. Sorted ASC for
            # deterministic JSON diffs.
            "affected_cells": sorted(missing_by_cell.keys()),
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
    soil = unified_data.soil if unified_data and hasattr(unified_data, 'soil') else {}

    # G7 §2 — when BOTH axes are unavailable the coverage check has
    # nothing meaningful to compare. Short-circuit with the
    # ``no_climate_and_soil_fetch`` cause so the banner copy can lift
    # the unified-unavailable narrative without parsing per-axis
    # records. When only one axis is missing, the existing "covers
    # X/N cells" issue rendering still works on the available axis,
    # so we let the regular path run.
    if not _has_climate_data(climate) and not _has_soil_data(soil):
        return _unavailable(
            check_id="spatial_temporal_coverage",
            scope="global",
            cause="no_climate_and_soil_fetch",
            details={
                "grid_cells": n_cells,
                "climate_cells": 0,
                "soil_cells": 0,
                "temporal_start": start_year,
                "temporal_end": end_year,
            },
        )
    if _is_file_based_climate(climate):
        # GeoTIFF-based: climate is region-wide rasters, not per-cell.
        # Cell-count comparison is not meaningful — the rasters cover
        # the full spatial extent by construction.
        n_climate_cells = n_cells  # treat as matching
        climate_format = "geotiff (region-wide)"
    else:
        n_climate_cells = len(climate)
        climate_format = "per_cell"

    # Check soil cell count matches grid (``soil`` already pulled
    # off ``unified_data`` above for the both-axes guard).
    n_soil_cells = len(soil) if isinstance(soil, dict) else 0

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


# =============================================================================
# Check 6b: Per-cell coverage (V2-22c-PRE.1.9 / D36)
# =============================================================================

def _check_coverage_climate_cells(unified_data) -> Dict[str, Any]:
    """V2-22c-PRE.1.9 (D36) — per-cell climate coverage check.

    Cells where the unified climate data has no entry, OR has an
    entry whose ``records`` is empty, count as missing. Emits
    ``scope='per_cell'`` so PRE.1.2's failed_checks pivot picks it
    up under the ``coverage_per_cell`` category.

    SARRA-Py file-based climate path: emits ``info`` with a
    delegation note rather than ``fail``. Per-cell coverage for
    SARRA-Py is V2-22d backlog (#11) — PRE.2.4 ships per-cell
    climate sampling but the coverage synthesis defers. The
    ``info`` result lets the cockpit Coverage chip render
    "delegated" instead of red.
    """
    grid = (
        unified_data.grid
        if unified_data and hasattr(unified_data, 'grid')
        else None
    )
    if grid is None or not getattr(grid, 'cells', None):
        return {
            "check": "coverage_climate_cells",
            "scope": "per_cell",
            "result": "info",
            "summary": "Per-cell climate coverage skipped: no grid available",
            "manuscript_claim": "Section 2.5: per-cell coverage check",
            "details": {"affected_cells": [], "n_missing": 0, "n_total": 0},
        }
    climate = (
        unified_data.climate
        if unified_data and hasattr(unified_data, 'climate')
        else {}
    )
    # G7 §2 — when the climate axis is fully unavailable (empty
    # dict, neither per-cell records nor file-based directories),
    # emit unavailable instead of fail. The prior behavior listed
    # every grid cell in ``affected_cells``, which fed the per-cell
    # pivot a climate-fail entry for every cell — a direct
    # violation of the §1 per-axis invariant 3 (climate fails on a
    # cell whose climate axis is unavailable). Partial coverage
    # (some cells missing) keeps the existing fail+affected_cells
    # path; the per-axis invariant only forbids the AXIS-fully-
    # unavailable case from emitting climate fails.
    if not _has_climate_data(climate):
        return _unavailable(
            check_id="coverage_climate_cells",
            scope="per_cell",
            cause="no_climate_fetch",
            details={
                "n_total": grid.n_cells,
                "n_missing": 0,
            },
        )
    if _is_file_based_climate(climate):
        return {
            "check": "coverage_climate_cells",
            "scope": "per_cell",
            "result": "info",
            "summary": (
                "File-based climate (SARRA-Py) — per-cell coverage "
                "delegated to PRE.2 sampling (V2-22d backlog)"
            ),
            "manuscript_claim": "Section 2.5: per-cell coverage check",
            "details": {
                "delegated_to": "PRE.2 sampling",
                "affected_cells": [],
                "n_missing": 0,
                "n_total": grid.n_cells,
            },
        }

    grid_ids = {c.cell_id for c in grid.cells}
    missing = set()
    if isinstance(climate, dict):
        for cid in grid_ids:
            ts = climate.get(cid)
            if ts is None:
                missing.add(cid)
                continue
            records = getattr(ts, 'records', None)
            if records is None or len(records) == 0:
                missing.add(cid)
    else:
        # Non-dict climate that isn't file-based: treat as missing.
        missing = set(grid_ids)

    affected_cells = sorted(missing)
    n_missing = len(affected_cells)
    n_total = len(grid_ids)
    result = "fail" if n_missing > 0 else "pass"
    return {
        "check": "coverage_climate_cells",
        "scope": "per_cell",
        "result": result,
        "summary": (
            f"{n_total - n_missing}/{n_total} cells covered by climate data"
            + (f" — {n_missing} cells missing" if n_missing else "")
        ),
        "manuscript_claim": "Section 2.5: per-cell coverage check",
        "details": {
            "n_total": n_total,
            "n_missing": n_missing,
            "affected_cells": affected_cells,
            # PRE.1.8 — per-violation context. Coverage failures
            # don't carry a value/unit/bounds tuple (the failure is
            # absence-of-data, not out-of-range), so those fields
            # are null. Cockpit drawer renders "no climate data"
            # for these rows.
            "violation_details": [
                {
                    "cell_id": cid, "layer_idx": None,
                    "variable": "climate", "value": None,
                    "unit": None, "bounds": None,
                }
                for cid in affected_cells
            ],
        },
    }


def _check_coverage_soil_cells(unified_data) -> Dict[str, Any]:
    """V2-22c-PRE.1.9 (D36) — per-cell soil coverage check. Same
    shape as ``_check_coverage_climate_cells`` but reads
    ``unified_data.soil``. SoilProfile cells where the profile is
    None OR ``layers`` is empty count as missing.

    No file-based asymmetry here — soil is always per-cell across
    all four platforms today.
    """
    grid = (
        unified_data.grid
        if unified_data and hasattr(unified_data, 'grid')
        else None
    )
    if grid is None or not getattr(grid, 'cells', None):
        return {
            "check": "coverage_soil_cells",
            "scope": "per_cell",
            "result": "info",
            "summary": "Per-cell soil coverage skipped: no grid available",
            "manuscript_claim": "Section 2.5: per-cell coverage check",
            "details": {"affected_cells": [], "n_missing": 0, "n_total": 0},
        }
    soil = (
        unified_data.soil
        if unified_data and hasattr(unified_data, 'soil')
        else {}
    )
    # G7 §2 — mirror of the climate-side guard: fully-unavailable
    # soil emits unavailable instead of a fail-every-cell record
    # that would violate the §1 per-axis invariant 3.
    if not _has_soil_data(soil):
        return _unavailable(
            check_id="coverage_soil_cells",
            scope="per_cell",
            cause="no_soil_match",
            details={
                "n_total": grid.n_cells,
                "n_missing": 0,
            },
        )

    grid_ids = {c.cell_id for c in grid.cells}
    missing = set()
    if isinstance(soil, dict):
        for cid in grid_ids:
            profile = soil.get(cid)
            if profile is None:
                missing.add(cid)
                continue
            layers = getattr(profile, 'layers', None)
            if layers is None or len(layers) == 0:
                missing.add(cid)
    else:
        missing = set(grid_ids)

    affected_cells = sorted(missing)
    n_missing = len(affected_cells)
    n_total = len(grid_ids)
    result = "fail" if n_missing > 0 else "pass"
    return {
        "check": "coverage_soil_cells",
        "scope": "per_cell",
        "result": result,
        "summary": (
            f"{n_total - n_missing}/{n_total} cells covered by soil data"
            + (f" — {n_missing} cells missing" if n_missing else "")
        ),
        "manuscript_claim": "Section 2.5: per-cell coverage check",
        "details": {
            "n_total": n_total,
            "n_missing": n_missing,
            "affected_cells": affected_cells,
            "violation_details": [
                {
                    "cell_id": cid, "layer_idx": None,
                    "variable": "soil", "value": None,
                    "unit": None, "bounds": None,
                }
                for cid in affected_cells
            ],
        },
    }


# =============================================================================
# Check 7: Region-specific bounds (V2-20)
# =============================================================================

def _check_region_bounds(unified_data, config) -> Dict[str, Any]:
    """Check climate values against region-specific thresholds.

    Auto-detects the agro-ecological region from the study area's
    centroid coordinates, then applies tighter bounds than the
    universal physical limits in Check 3.
    """
    # Determine centroid from region bounds
    region = unified_data.region if unified_data and hasattr(unified_data, 'region') else None
    if not region or not hasattr(region, 'bounds'):
        return {
            "check": "region_specific_bounds",
            "scope": "global",
            "result": "info",
            "summary": "Region bounds check skipped: no region available",
            "manuscript_claim": "Section 2.5: region-appropriate thresholds",
            "details": {},
        }

    bounds = region.bounds
    center_lat = (bounds.miny + bounds.maxy) / 2
    center_lon = (bounds.minx + bounds.maxx) / 2

    detected = _detect_region(center_lat, center_lon)
    region_id = detected.get("id", "universal")
    region_name = detected.get("name", "Universal")
    thresholds = detected.get("thresholds", {})

    if not thresholds or region_id == "universal":
        return {
            "check": "region_specific_bounds",
            "scope": "global",
            "result": "info",
            "summary": (
                f"No region-specific bounds for centroid "
                f"({center_lat:.2f}, {center_lon:.2f}) — using universal"
            ),
            "manuscript_claim": "Section 2.5: region-appropriate thresholds",
            "details": {
                "centroid_lat": round(center_lat, 4),
                "centroid_lon": round(center_lon, 4),
                "region_detected": "universal",
            },
        }

    # Check climate values against region-specific thresholds
    climate = unified_data.climate if unified_data and hasattr(unified_data, 'climate') else {}
    # G7 §2 — region bounds is purely a climate-axis check; emit
    # unavailable when climate is missing rather than a vacuous pass
    # built on zero records inspected.
    if not _has_climate_data(climate):
        return _unavailable(
            check_id="region_specific_bounds",
            scope="global",
            cause="no_climate_fetch",
            details={
                "centroid_lat": round(center_lat, 4),
                "centroid_lon": round(center_lon, 4),
                "region_detected": region_id,
            },
        )
    # V2-22c-PRE.1.7 — track per-cell violations alongside the existing
    # diagnostic text-string list. Two parallel structures:
    #
    # * ``violation_texts``: human-readable strings (existing
    #   diagnostic; capped at 10 in the emitted ``sample_violations``
    #   per PRE.1.7 spec — text-string list is preserved at [:10]).
    # * ``violation_records``: structured `(cell_id, variable, value,
    #   bounds, unit)` tuples that drive the un-truncated
    #   ``affected_cells`` list AND the ``violation_details`` flatten
    #   used by PRE.1.8 ``cell_failed_check_details``.
    #
    # Without the structured records the cockpit's Layer 1
    # `region_specific_bounds` failure-fill silently drops every cell
    # because PRE.1.2's pivot reads `details.affected_cells` which
    # would be empty.
    violation_texts: List[str] = []
    violation_records: List[Dict[str, Any]] = []

    if not _is_file_based_climate(climate):
        tmax_range = thresholds.get("tmax")
        tmin_range = thresholds.get("tmin")
        srad_range = thresholds.get("srad")
        precip_daily_max = thresholds.get("precip_daily_max")

        for cell_id, ts in climate.items():
            if not hasattr(ts, 'records'):
                continue
            for record in ts.records:
                if tmax_range and hasattr(record, 'tmax') and record.tmax is not None:
                    if record.tmax < tmax_range[0] or record.tmax > tmax_range[1]:
                        violation_texts.append(
                            f"tmax={record.tmax:.1f} outside {region_name} "
                            f"range {tmax_range}"
                        )
                        violation_records.append({
                            "cell_id": cell_id, "variable": "tmax",
                            "value": float(record.tmax), "unit": "°C",
                            "bounds": list(tmax_range),
                        })
                if tmin_range and hasattr(record, 'tmin') and record.tmin is not None:
                    if record.tmin < tmin_range[0] or record.tmin > tmin_range[1]:
                        violation_texts.append(
                            f"tmin={record.tmin:.1f} outside {region_name} "
                            f"range {tmin_range}"
                        )
                        violation_records.append({
                            "cell_id": cell_id, "variable": "tmin",
                            "value": float(record.tmin), "unit": "°C",
                            "bounds": list(tmin_range),
                        })
                # Daily precipitation uses precip_daily_max (NOT
                # precip_annual_mm which is an annual total)
                if precip_daily_max and hasattr(record, 'precip') and record.precip is not None:
                    if record.precip > precip_daily_max:
                        violation_texts.append(
                            f"precip={record.precip:.1f} mm/day exceeds "
                            f"{region_name} daily max {precip_daily_max}"
                        )
                        violation_records.append({
                            "cell_id": cell_id, "variable": "precip",
                            "value": float(record.precip),
                            "unit": "mm/day",
                            "bounds": [None, float(precip_daily_max)],
                        })
                if srad_range and hasattr(record, 'srad') and record.srad is not None:
                    if record.srad < srad_range[0] or record.srad > srad_range[1]:
                        violation_texts.append(
                            f"srad={record.srad:.1f} outside {region_name} "
                            f"range {srad_range}"
                        )
                        violation_records.append({
                            "cell_id": cell_id, "variable": "srad",
                            "value": float(record.srad),
                            "unit": "MJ/m²/d",
                            "bounds": list(srad_range),
                        })

    n_violations = len(violation_texts)
    if n_violations > 0:
        result = "warning"
    else:
        result = "pass"

    # V2-22c-PRE.1.7 — un-truncated, deduplicated, sorted ASC for
    # deterministic JSON diffs (evaluator §2 binding).
    affected_cells_sorted = sorted({r["cell_id"] for r in violation_records})

    return {
        "check": "region_specific_bounds",
        "scope": "per_record",
        "result": result,
        "summary": (
            f"Region: {region_name} (centroid {center_lat:.2f}°N, "
            f"{center_lon:.2f}°E) — {n_violations} values outside "
            f"region-specific bounds"
        ),
        "manuscript_claim": "Section 2.5: region-appropriate thresholds",
        "details": {
            "region_detected": region_id,
            "region_name": region_name,
            "centroid_lat": round(center_lat, 4),
            "centroid_lon": round(center_lon, 4),
            "thresholds": thresholds,
            "n_violations": n_violations,
            # V2-22c-PRE.1.7 — text-string sample stays capped at 10
            # for human-readable diagnostic; the un-truncated cell-id
            # list lives on `affected_cells` below. Structurally
            # captured by tests/unit/test_scientific_un_truncation.py
            # via the ALLOWED_TRUNCATED_KEYS allowlist.
            "sample_violations": violation_texts[:10],
            "affected_cells": affected_cells_sorted,
            # V2-22c-PRE.1.8 — sorted by (cell_id, variable) for
            # cockpit-cursor stability + reproducible JSON diffs.
            "violation_details": sorted(
                violation_records,
                key=lambda r: (r["cell_id"], r["variable"]),
            ),
        },
    }
