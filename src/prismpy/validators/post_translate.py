"""
Post-translate climate validation for prismpy.

V2-20: validates ACTUAL per-cell weather data from translator output
files, replacing the V2-19 placeholder-based checks for PYTHIA and ACEA.

Platform support:
  - PYTHIA: parses .WTH files (DSSAT weather format)
  - ACEA: unpickles .pckl files (numpy arrays)
  - CRAFT: not applicable (weather downloaded externally by CRAFT GUI)
  - SARRA-Py: not applicable (GeoTIFF file-count validation in scientific.py)
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Same thresholds as scientific.py — single source of truth
CLIMATE_RANGES = {
    "srad": (0.0, 40.0, "MJ/m²/d"),
    "tmax": (-50.0, 60.0, "°C"),
    "tmin": (-60.0, 50.0, "°C"),
    "rain": (0.0, 600.0, "mm/day"),
}

MISSING_VALUE = -99.0


def run_post_translate_validation(
    translation_results: Dict,
    base_dir: Path,
) -> Dict[str, Any]:
    """Run climate validation on actual translator output files.

    Args:
        translation_results: Dict of platform_name → TranslationResult
        base_dir: Pipeline output base directory

    Returns:
        Dict with 'checks' list and 'overall_result', same structure
        as scientific.run_scientific_validation.
    """
    checks = []

    for platform_name, result in translation_results.items():
        if not result.success:
            continue

        platform_dir = base_dir / platform_name

        if platform_name == "pythia":
            checks.extend(_validate_pythia_wth(platform_dir))
        elif platform_name == "acea":
            checks.extend(_validate_acea_pickles(platform_dir))
        elif platform_name == "craft":
            checks.extend(_validate_craft_weather(platform_dir))
        elif platform_name == "sarra_py":
            checks.extend(_validate_sarra_py_geotiffs(platform_dir))

    results = [c["result"] for c in checks]
    if "fail" in results:
        overall = "fail"
    elif "warning" in results:
        overall = "warning"
    else:
        overall = "pass"

    return {
        "post_translate_version": "1.0",
        "checks": checks,
        "overall_result": overall,
        "n_checks": len(checks),
        "n_pass": results.count("pass"),
        "n_warning": results.count("warning"),
        "n_fail": results.count("fail"),
    }


# =============================================================================
# PYTHIA .WTH parser + validator
# =============================================================================

def _validate_pythia_wth(platform_dir: Path) -> List[Dict[str, Any]]:
    """Parse PYTHIA .WTH files and run climate checks on real values."""
    weather_dir = platform_dir / "weather"
    if not weather_dir.exists():
        return [{
            "check": "post_translate_climate_pythia",
            "scope": "platform",
            "result": "info",
            "summary": "No weather directory found for PYTHIA",
            "manuscript_claim": "Section 2.5: value range verification",
            "details": {"platform_dir": str(platform_dir)},
        }]

    wth_files = sorted(weather_dir.glob("*.WTH"))
    if not wth_files:
        return [{
            "check": "post_translate_climate_pythia",
            "scope": "platform",
            "result": "warning",
            "summary": "No .WTH files found in PYTHIA weather directory",
            "manuscript_claim": "Section 2.5: value range verification",
            "details": {"weather_dir": str(weather_dir)},
        }]

    # Parse all WTH files and collect values
    all_values: Dict[str, List[float]] = {
        "srad": [], "tmax": [], "tmin": [], "rain": [],
    }
    consistency_violations = {
        "tmax_le_tmin": 0,
        "negative_rain": 0,
        "negative_srad": 0,
    }
    total_records = 0
    n_files = len(wth_files)

    date_issues = []

    for wth_file in wth_files:
        try:
            records, dates = _parse_wth_file(wth_file)
            # Date continuity check per file
            date_check = _check_date_continuity(
                dates, "pythia", wth_file.name
            )
            if date_check:
                date_issues.append(date_check)

            for srad, tmax, tmin, rain in records:
                total_records += 1
                if srad != MISSING_VALUE:
                    all_values["srad"].append(srad)
                    if srad < 0:
                        consistency_violations["negative_srad"] += 1
                if tmax != MISSING_VALUE:
                    all_values["tmax"].append(tmax)
                if tmin != MISSING_VALUE:
                    all_values["tmin"].append(tmin)
                if rain != MISSING_VALUE:
                    all_values["rain"].append(rain)
                    if rain < 0:
                        consistency_violations["negative_rain"] += 1
                if tmax != MISSING_VALUE and tmin != MISSING_VALUE:
                    if tmax < tmin:
                        consistency_violations["tmax_le_tmin"] += 1
        except Exception as e:
            logger.warning(f"Failed to parse {wth_file.name}: {e}")

    checks = _build_climate_checks(
        "pythia", all_values, consistency_violations,
        total_records, n_files,
    )
    checks.extend(date_issues)
    return checks


def _parse_wth_file(path: Path) -> Tuple[List[Tuple[float, float, float, float]], List[int]]:
    """Parse a DSSAT .WTH file and extract values + dates.

    Returns:
        Tuple of (records list of (srad, tmax, tmin, rain), dates list of YRDOY ints)
    """
    records = []
    dates = []
    in_data = False

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("@"):
                if "SRAD" in line and "TMAX" in line:
                    in_data = True
                continue
            if not in_data or not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                yrdoy = int(parts[0])
                srad = float(parts[1])
                tmax = float(parts[2])
                tmin = float(parts[3])
                rain = float(parts[4])
                records.append((srad, tmax, tmin, rain))
                dates.append(yrdoy)
            except (ValueError, IndexError):
                continue

    return records, dates


# =============================================================================
# CRAFT weather parser + validator
# =============================================================================

def _validate_craft_weather(platform_dir: Path) -> List[Dict[str, Any]]:
    """Parse CRAFT tab-separated weather files and run climate checks.

    CRAFT generates per-cell weather files (weather/cell_*.txt) when
    real climate data is available. Format: YRDOY<tab>SRAD<tab>TMAX<tab>TMIN<tab>RAIN.
    Falls back to "info" if only input.csv (coordinate file) exists.
    """
    weather_dir = platform_dir / "weather"
    if not weather_dir.exists():
        return [{
            "check": "post_translate_climate_craft",
            "scope": "platform",
            "result": "info",
            "summary": "No weather directory found for CRAFT",
            "manuscript_claim": "Section 2.5: value range verification",
            "details": {"platform_dir": str(platform_dir)},
        }]

    # CRAFT weather files: {CellID}.txt (e.g., 4054256.txt) or
    # legacy cell_*.txt format. Glob for .txt excluding input.csv.
    weather_files = sorted(
        f for f in weather_dir.glob("*.txt")
        if f.name != "input.csv" and not f.name.startswith(".")
    )
    if not weather_files:
        # Only input.csv exists (coordinate file for external download)
        return [{
            "check": "post_translate_climate_craft",
            "scope": "platform",
            "result": "info",
            "summary": (
                "CRAFT weather: only input.csv found (coordinates for "
                "external NASA POWER download via CRAFT GUI)"
            ),
            "manuscript_claim": "Section 2.5: value range verification",
            "details": {"weather_dir": str(weather_dir)},
        }]

    all_values: Dict[str, List[float]] = {
        "srad": [], "tmax": [], "tmin": [], "rain": [],
    }
    consistency_violations = {
        "tmax_le_tmin": 0,
        "negative_rain": 0,
        "negative_srad": 0,
    }
    total_records = 0
    n_files = len(weather_files)

    for wf in weather_files:
        try:
            with open(wf, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("YRDOY"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 5:
                        continue
                    try:
                        srad = float(parts[1])
                        tmax = float(parts[2])
                        tmin = float(parts[3])
                        rain = float(parts[4])
                        total_records += 1
                        if srad != MISSING_VALUE:
                            all_values["srad"].append(srad)
                            if srad < 0:
                                consistency_violations["negative_srad"] += 1
                        if tmax != MISSING_VALUE:
                            all_values["tmax"].append(tmax)
                        if tmin != MISSING_VALUE:
                            all_values["tmin"].append(tmin)
                        if rain != MISSING_VALUE:
                            all_values["rain"].append(rain)
                            if rain < 0:
                                consistency_violations["negative_rain"] += 1
                        if tmax != MISSING_VALUE and tmin != MISSING_VALUE:
                            if tmax < tmin:
                                consistency_violations["tmax_le_tmin"] += 1
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            logger.warning(f"Failed to parse {wf.name}: {e}")

    return _build_climate_checks(
        "craft", all_values, consistency_violations,
        total_records, n_files,
    )


# =============================================================================
# ACEA pickle parser + validator
# =============================================================================

def _validate_acea_pickles(platform_dir: Path) -> List[Dict[str, Any]]:
    """Parse ACEA .pckl files and run climate checks on real values."""
    climate_dir = platform_dir / "climate"
    if not climate_dir.exists():
        return [{
            "check": "post_translate_climate_acea",
            "scope": "platform",
            "result": "info",
            "summary": "No climate directory found for ACEA",
            "manuscript_claim": "Section 2.5: value range verification",
            "details": {"platform_dir": str(platform_dir)},
        }]

    pkl_files = sorted(climate_dir.glob("*.pckl"))
    if not pkl_files:
        return [{
            "check": "post_translate_climate_acea",
            "scope": "platform",
            "result": "warning",
            "summary": "No .pckl files found in ACEA climate directory",
            "manuscript_claim": "Section 2.5: value range verification",
            "details": {"climate_dir": str(climate_dir)},
        }]

    all_values: Dict[str, List[float]] = {
        "tmax": [], "tmin": [], "rain": [], "srad": [],
    }
    consistency_violations = {
        "tmax_le_tmin": 0,
        "negative_rain": 0,
        "negative_srad": 0,
    }
    total_records = 0
    n_files = len(pkl_files)

    for pkl_file in pkl_files:
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            # ACEA pickle format: tuple of (tmax, tmin, prec, et0[, srad])
            if isinstance(data, tuple) and len(data) >= 3:
                tmax_arr, tmin_arr, prec_arr = data[0], data[1], data[2]
                srad_arr = data[4] if len(data) >= 5 else None
                n = min(len(tmax_arr), len(tmin_arr), len(prec_arr))
                total_records += n
                for i in range(n):
                    tmax_val = float(tmax_arr[i])
                    tmin_val = float(tmin_arr[i])
                    rain_val = float(prec_arr[i])
                    all_values["tmax"].append(tmax_val)
                    all_values["tmin"].append(tmin_val)
                    all_values["rain"].append(rain_val)
                    if srad_arr is not None and i < len(srad_arr):
                        srad_val = float(srad_arr[i])
                        all_values["srad"].append(srad_val)
                        if srad_val < 0:
                            consistency_violations["negative_srad"] += 1
                    if tmax_val < tmin_val:
                        consistency_violations["tmax_le_tmin"] += 1
                    if rain_val < 0:
                        consistency_violations["negative_rain"] += 1
        except Exception as e:
            logger.warning(f"Failed to parse {pkl_file.name}: {e}")

    return _build_climate_checks(
        "acea", all_values, consistency_violations,
        total_records, n_files,
    )


# =============================================================================
# Shared check builder
# =============================================================================

def _check_date_continuity(
    dates: List[int], platform: str, filename: str,
) -> Optional[Dict[str, Any]]:
    """Check that YRDOY date sequence is monotonically increasing with no gaps.

    Args:
        dates: List of YRDOY values (e.g., 2020001, 2020002, ...)
        platform: Platform name for reporting
        filename: Source file for reporting

    Returns:
        Check result dict if issues found, None if clean
    """
    if len(dates) < 2:
        return None

    gaps = []
    duplicates = []
    out_of_order = 0

    for i in range(1, len(dates)):
        if dates[i] == dates[i - 1]:
            duplicates.append(dates[i])
        elif dates[i] < dates[i - 1]:
            out_of_order += 1

    # Check for gaps (expected: consecutive days)
    # YRDOY format: YYYYDDD — can't just subtract, need date math
    from datetime import datetime, timedelta
    try:
        parsed = []
        for yrdoy in dates:
            yr = yrdoy // 1000
            doy = yrdoy % 1000
            dt = datetime(yr, 1, 1) + timedelta(days=doy - 1)
            parsed.append(dt)

        for i in range(1, len(parsed)):
            delta = (parsed[i] - parsed[i - 1]).days
            if delta > 1:
                gaps.append((dates[i - 1], dates[i], delta - 1))
    except (ValueError, OverflowError):
        pass

    if not gaps and not duplicates and out_of_order == 0:
        return None

    return {
        "check": f"post_translate_date_continuity_{platform}",
        "scope": "per_file",
        "result": "warning",
        "summary": (
            f"{platform.upper()} {filename}: "
            f"{len(gaps)} gaps, {len(duplicates)} duplicates, "
            f"{out_of_order} out-of-order"
        ),
        "manuscript_claim": "Section 2.5: temporal completeness (date sequence)",
        "details": {
            "platform": platform,
            "file": filename,
            "n_gaps": len(gaps),
            "n_duplicates": len(duplicates),
            "n_out_of_order": out_of_order,
            "sample_gaps": gaps[:5],
            "sample_duplicates": duplicates[:5],
        },
    }


def _build_climate_checks(
    platform: str,
    all_values: Dict[str, List[float]],
    consistency_violations: Dict[str, int],
    total_records: int,
    n_files: int,
) -> List[Dict[str, Any]]:
    """Build hierarchical check results from collected values."""
    checks = []

    # Cross-variable consistency check
    fail_count = (
        consistency_violations["tmax_le_tmin"]
        + consistency_violations["negative_rain"]
        + consistency_violations["negative_srad"]
    )
    if fail_count > 0:
        result = "fail"
    else:
        result = "pass"

    checks.append({
        "check": f"post_translate_consistency_{platform}",
        "scope": "per_record",
        "result": result,
        "summary": (
            f"{platform.upper()}: {total_records} daily records from "
            f"{n_files} files, {fail_count} violations"
        ),
        "manuscript_claim": "Section 2.5: cross-variable consistency (real data)",
        "details": {
            "platform": platform,
            "total_records": total_records,
            "n_files": n_files,
            "violations": consistency_violations,
            "data_source": "actual translator output (not placeholder)",
        },
    })

    # Per-variable value range checks
    for var, (vmin, vmax, unit) in CLIMATE_RANGES.items():
        values = all_values.get(var, [])
        if not values:
            continue
        obs_min = min(values)
        obs_max = max(values)
        n_oor = sum(1 for v in values if v < vmin or v > vmax)
        result = "warning" if n_oor > 0 else "pass"
        checks.append({
            "check": f"post_translate_range_{platform}_{var}",
            "scope": "per_record",
            "result": result,
            "summary": (
                f"{platform.upper()} {var}: [{obs_min:.1f}, {obs_max:.1f}] "
                f"{unit}, {n_oor}/{len(values)} out of range"
            ),
            "manuscript_claim": "Section 2.5: value range verification (real data)",
            "details": {
                "platform": platform,
                "variable": var,
                "unit": unit,
                "expected_min": vmin,
                "expected_max": vmax,
                "observed_min": round(obs_min, 2),
                "observed_max": round(obs_max, 2),
                "out_of_range_count": n_oor,
                "total_values": len(values),
                "data_source": "actual translator output",
            },
        })

    return checks


# Explicit variable mapping for SARRA-Py GeoTIFF subdirectories.
# Each entry: subdir_name → (internal_var, operation, operand)
# Operations: "noop" = no conversion, "add" = additive, "mul" = multiplicative.
#
# Issue 5 (warning-auditor HIGH) fix — the SARRA_data_download library
# already converts AgERA5 to researcher-friendly units before writing
# its .tif files (tmax / tmin in °C, srad in kJ/m²/day, rain in mm/day).
# The prior mapping assumed raw AgERA5 units and applied a redundant
# second conversion, so the validator reported tmax around -251 °C and
# srad around 0.02 MJ/m²/d. Actual file-unit verification on run
# 121e66b5-... (Maradi 2020-01-01) confirmed:
#   tmax: 18.6 .. 26.4 (already °C)  → noop
#   tmin:  9.4 .. 14.3 (already °C)  → noop
#   srad: 20388 .. 21168 (kJ/m²/d)   → mul 1e-3 to get 20.4 .. 21.2 MJ/m²/d
SARRA_PY_VAR_MAPPING = {
    "rainfall": ("rain", "noop", 0.0),
    "2m_temperature_24_hour_maximum": ("tmax", "noop", 0.0),  # already °C
    "2m_temperature_24_hour_minimum": ("tmin", "noop", 0.0),  # already °C
    "solar_radiation_flux_daily": ("srad", "mul", 1e-3),  # kJ/m²/d → MJ/m²/d
}


def _validate_sarra_py_geotiffs(
    platform_dir: Path,
) -> List[Dict[str, Any]]:
    """Validate SARRA-Py climate by sampling GeoTIFF pixel values from
    the per-run output tree at platform_dir/data/climate/<var_subdir>/.

    Reads from the translator's authoritative output location (matching
    the pattern of _validate_pythia_wth and _validate_acea_pickles),
    not from the shared download cache.
    """
    import random

    checks = []
    climate_base = platform_dir / "data" / "climate"

    searched_paths = []
    sampled_vars = {}
    total_files = 0

    for subdir_name, (var, op, operand) in SARRA_PY_VAR_MAPPING.items():
        var_dir = climate_base / subdir_name
        searched_paths.append(str(var_dir))
        if not var_dir.is_dir():
            continue
        tifs = list(var_dir.glob("*.tif"))
        total_files += len(tifs)
        if not tifs:
            continue
        sample = random.sample(tifs, min(10, len(tifs)))
        sampled_vars[var] = (sample, op, operand)

    if not sampled_vars:
        checks.append({
            "check": "post_translate_climate_sarra_py",
            "scope": "platform",
            "result": "warning",
            "summary": "No climate files found in SARRA-Py output",
            "details": {
                "platform": "sarra_py",
                "searched_paths": searched_paths,
            },
        })
        return checks

    # Partial-data detection (AC 1.4.10)
    expected_vars = len(SARRA_PY_VAR_MAPPING)
    found_vars = len(sampled_vars)
    if found_vars < expected_vars:
        missing = [v for _, (v, _, _) in SARRA_PY_VAR_MAPPING.items()
                   if v not in sampled_vars]
        checks.append({
            "check": "post_translate_completeness_sarra_py",
            "scope": "platform",
            "result": "warning",
            "summary": (
                f"Partial climate data: {found_vars}/{expected_vars} "
                f"variables present ({total_files} files)"
            ),
            "details": {
                "platform": "sarra_py",
                "expected_variables": expected_vars,
                "found_variables": found_vars,
                "missing_variables": missing,
                "total_files": total_files,
            },
        })

    try:
        import rasterio
        import numpy as np
    except ImportError:
        checks.append({
            "check": "post_translate_climate_sarra_py",
            "scope": "platform",
            "result": "info",
            "summary": "rasterio not available for climate range check",
            "details": {"platform": "sarra_py"},
        })
        return checks

    for var, (sample_files, op, operand) in sampled_vars.items():
        all_vals = []
        unreadable = 0
        for tif_path in sample_files:
            try:
                with rasterio.open(tif_path) as src:
                    data = src.read(1).astype(float)
                    nodata = src.nodata
                    if nodata is not None:
                        data = data[data != nodata]
                    valid = data[np.isfinite(data)]
                    if valid.size > 0:
                        if op == "add":
                            valid = valid + operand
                        elif op == "mul":
                            valid = valid * operand
                        all_vals.extend([float(valid.min()), float(valid.max())])
            except Exception as e:
                logger.debug(f"Skipping {tif_path.name}: {e}")
                unreadable += 1
                continue

        if not all_vals:
            # Codex self-check R3 HIGH — the scientific validator's
            # `value_range_climate` info record promises that "if a
            # variable is missing, look for another SARRA-Py
            # post-translate message in this report explaining
            # why." Without this branch, a variable whose entire
            # 10-file sample fails to open / has only nodata would
            # produce NO record of any kind, contradicting that
            # promise. Emit an explicit warning so the user always
            # has a paper trail for the missing range record.
            checks.append({
                "check": f"post_translate_range_sarra_py_{var}",
                "scope": "per_record",
                "result": "warning",
                "summary": (
                    f"SARRA-Py {var} range not computed: all "
                    f"{len(sample_files)} sampled GeoTIFFs either "
                    f"failed to open or contained no finite values "
                    f"({unreadable} unreadable)."
                ),
                "details": {
                    "platform": "sarra_py",
                    "variable": var,
                    "sample_size": len(sample_files),
                    "unreadable_count": unreadable,
                    "reason": "all_sampled_files_yielded_no_values",
                    "data_source": "GeoTIFF pixel values (10-file sample)",
                },
            })
            continue

        obs_min = min(all_vals)
        obs_max = max(all_vals)

        vmin, vmax, unit = CLIMATE_RANGES.get(var, (None, None, ""))
        if vmin is None:
            continue

        n_oor = sum(1 for v in all_vals if v < vmin or v > vmax)
        # Codex self-check R4 HIGH — downgrade to warning when the
        # effective sample collapsed below the target size. Without
        # this, a 9-of-10-unreadable sample emits a clean 'pass'
        # record that claims "10-file sample" while actually drawing
        # from one file — broad climate-file corruption would hide
        # behind the happy-path.
        effective_sample = len(sample_files) - unreadable
        if n_oor > 0 or unreadable > 0:
            result = "warning"
        else:
            result = "pass"
        summary = (
            f"SARRA-Py {var}: [{obs_min:.1f}, {obs_max:.1f}] "
            f"{unit} (from output files)"
        )
        if unreadable > 0:
            summary += (
                f" — sample degraded: {unreadable} of "
                f"{len(sample_files)} files were unreadable, effective "
                f"sample = {effective_sample}"
            )

        checks.append({
            "check": f"post_translate_range_sarra_py_{var}",
            "scope": "per_record",
            "result": result,
            "summary": summary,
            "details": {
                "platform": "sarra_py",
                "variable": var,
                "unit": unit,
                "expected_min": vmin,
                "expected_max": vmax,
                "observed_min": round(obs_min, 2),
                "observed_max": round(obs_max, 2),
                "out_of_range_count": n_oor,
                "total_values": len(all_vals),
                # Codex R4 HIGH — persist sample coverage so auditors
                # reading the packaged `validation_report.json` can
                # tell when a pass-looking record was backed by a
                # degraded sample.
                "sample_size": len(sample_files),
                "unreadable_count": unreadable,
                "effective_sample_size": effective_sample,
                "data_source": "GeoTIFF pixel values (10-file sample)",
            },
        })

    return checks
