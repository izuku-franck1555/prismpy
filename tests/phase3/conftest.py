"""Shared fixtures for PR3 Phase 3 spec-driven behavior probes (eval-2).

Spec source: ``EXP/prism-runner/PHASE-B-PR3-CONTRACT.md`` v0.5 — §1.3 IN-scope
items 1-12 + §2.1.1 canonical helpers + §2.7.3 keysets + §2.7.6 schema +
§2.7.6.1 MUSTs + §2.7.7 4 display-guide flags.

These fixtures build SPEC-COMPLIANT ``project_config`` dicts that exercise
the contract's emit clauses. They MUST NOT mirror impl details; they encode
the public contract surface verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prismpy.packaging.manifest import UC_CONFIG_KEY_TABLE


# ──────────────────────────────────────────────────────────────────────────
# Per-UC use_case_config keysets (verbatim from PR3 contract §1.3 item 8e)
# ──────────────────────────────────────────────────────────────────────────


# SINGLE SSOT: the expected per-UC use_case_config keyset is derived from the
# generator's bake contract UC_CONFIG_KEY_TABLE, not a re-hardcoded copy (which
# silently drifted — UC5 metric/agg_level/organic_decomp + UC6
# output_metric/agg_level are no longer baked, and climate_scenarios' `years` is
# not bake-eligible). The T1/T9 probes still catch a REAL drift: the emit path
# (use_case_config_for, with its per-translator encodings) is distinct from this
# table, so an emit that diverges from the declared table still fails.
#
# n_response_skill is a no-raw post-processor: it carries a baked config in the
# table but build_project_config does not construct it, so phase3 stays scoped
# to the six platform-run UCs (its phase3 coverage is a fixture follow-up).
_PHASE3_UCS = (
    "yield_forecast", "climate_scenarios", "sowing_optimization",
    "drought_management", "soil_fertility", "livestock_feed",
)
UC_KEYSETS: dict[str, frozenset[str]] = {
    uc: frozenset(UC_CONFIG_KEY_TABLE[uc]) for uc in _PHASE3_UCS
}

KNOWN_UC_NAMES: frozenset[str] = frozenset(UC_KEYSETS)


# ──────────────────────────────────────────────────────────────────────────
# Per-UC readiness gates (verbatim from parent contract §2.7.6 + PR2
# manifest_gates.py SSOT: 22 slot references / 13 unique members per
# v1.1.3 SH-2 reconciliation)
# ──────────────────────────────────────────────────────────────────────────


# Hard gates mirror the generator SSOT manifest.py:PER_UC_GATES (hard subset).
# UC1: temporal floor raised n_years_gte_3 -> n_years_gte_4 (analog planner needs
# >=4 package years to guarantee >=3 analogs); forecast_or_analog_mode_resolved
# was REMOVED (dispatch-only — reads a runtime target_year absent at packaging).
UC1_HARD_GATES = frozenset({
    "n_years_gte_4",
    "manifest_cells_populated",
    "manifest_crops_populated",
})
# UC1 gained a >=30-year forecast-adequacy ADVISORY (WMO climate normal) in the
# same reclassification that removed forecast_or_analog_mode_resolved.
UC1_ADVISORY_GATES = frozenset({
    "n_years_gte_30_for_forecast_adequacy",
})
UC2_HARD_GATES = frozenset({
    "base_package_temporal_complete",
    "at_least_one_scenario_package_present",
    "scenario_packages_temporal_aligned",
})
# UC3 has 1 advisory gate (manifest_adapter_capability_sowing_rule_default_present)
UC3_HARD_GATES = frozenset({
    "n_years_gte_5",
    "crop_supported_per_platform",
})
UC3_ADVISORY_GATES = frozenset({
    "manifest_adapter_capability_sowing_rule_default_present",
})
# UC4 has 1 advisory gate (n_years_gte_9_for_drought_freq_anomaly)
UC4_HARD_GATES = frozenset({
    "n_years_gte_5",
    "crop_supported_per_platform",
    "manifest_cells_populated",
})
UC4_ADVISORY_GATES = frozenset({
    "n_years_gte_9_for_drought_freq_anomaly",
})
UC5_HARD_GATES = frozenset({
    "n_years_gte_3",
    "manifest_cells_populated",
    "crop_supported_per_platform",
    "fertilizer_scenarios_resolvable",
})
UC6_HARD_GATES = frozenset({
    "n_years_gte_3",
    "manifest_cells_populated",
    "manifest_cell_areas_populated",
    "manifest_crops_populated",
})


PER_UC_HARD_GATES: dict[str, frozenset[str]] = {
    "yield_forecast": UC1_HARD_GATES,
    "climate_scenarios": UC2_HARD_GATES,
    "sowing_optimization": UC3_HARD_GATES,
    "drought_management": UC4_HARD_GATES,
    "soil_fertility": UC5_HARD_GATES,
    "livestock_feed": UC6_HARD_GATES,
}

PER_UC_ADVISORY_GATES: dict[str, frozenset[str]] = {
    "yield_forecast": UC1_ADVISORY_GATES,
    "climate_scenarios": frozenset(),
    "sowing_optimization": UC3_ADVISORY_GATES,
    "drought_management": UC4_ADVISORY_GATES,
    "soil_fertility": frozenset(),
    "livestock_feed": frozenset(),
}

# All 22 slot references (per-UC × applicable-gates sum) → 14 unique strings
# (the UC1 reclassification added n_years_gte_4 + n_years_gte_30_for_forecast_
# adequacy and dropped forecast_or_analog_mode_resolved: net +1 vs the prior 13).
ALL_GATE_STRINGS: frozenset[str] = (
    UC1_HARD_GATES | UC1_ADVISORY_GATES | UC2_HARD_GATES | UC3_HARD_GATES
    | UC3_ADVISORY_GATES | UC4_HARD_GATES | UC4_ADVISORY_GATES | UC5_HARD_GATES
    | UC6_HARD_GATES
)


# ──────────────────────────────────────────────────────────────────────────
# Literal advisory_flag strings (verbatim from PR3 contract §1.3 items 6-8d)
# ──────────────────────────────────────────────────────────────────────────


# §1.3 item 6 — MUST-3 unconditional
ADVISORY_UC3_SOWING = "sowing_rule_default_absent:falls_back_to_manifest_default"

# §1.3 item 7 — MUST-6 conditional (platform=pythia AND ACEA path)
ADVISORY_UC5_PYTHIA_PK = "pythia_pk_silent_no_op:fertility_stress_unmodeled_v3.1"

# §1.3 item 8a-8d — 4 display-guide flags (per v0.3 F2 fold)
ADVISORY_UC1_SHORTFALL_PREFIX = "shortfall_threshold:viz_layer_default_"
ADVISORY_UC4_SEVERITY_TIER = "severity_tier:viz_layer_thresholds_v1"
ADVISORY_UC5_ROI_PRICES = "roi_prices:viz_layer_regional_defaults"
ADVISORY_UC6_HERD_DENSITY = "herd_density:GLW_2020_default_supply_side_only"


# ──────────────────────────────────────────────────────────────────────────
# Synthetic project_config builders
# ──────────────────────────────────────────────────────────────────────────


def _minimal_uc_config(uc_name: str) -> dict[str, Any]:
    """Build a minimal valid use_case_config sub-dict for a UC.

    Values match the cli.py argparse SSOT (per parent v1.1.6 §2.7.3
    + PR3 §1.3 item 8e closed keysets).
    """
    defaults = {
        "yield_forecast": {
            "cores": 1,
            "target_year": 2024,
            "forecast_date": None,
            "max_runs": 500,
            "cultivar_ids": None,
            "n_analogs": None,
        },
        "climate_scenarios": {
            "cores": 1,
            "years": [2018, 2020],
            "scenario_packages": [],
        },
        "sowing_optimization": {
            "cores": 1,
            "sowing_window_start": 120,
            "sowing_window_end": 200,
            "sowing_stride": 7,
            "sowing_rule": "manifest_default",
            "subsistence_yield": None,
        },
        "drought_management": {
            "cores": 1,
            "risk_metric": "prob_drought",
            "critical_window": "full_season",
            "critical_window_start": None,
            "critical_window_end": None,
            "drought_threshold": 0.4,
            "min_consecutive_days": 7,
            "baseline_start": None,
            "baseline_end": None,
            "no_cell_day_output": False,
            "drought_threshold_grid": None,
        },
        "soil_fertility": {
            "cores": 1,
            "scenarios": None,
            "metric": "agronomic_efficiency",
            "agg_level": "cell",
            "organic_decomp_rate": None,
            "enable_cost_benefit": False,
        },
        "livestock_feed": {
            "cores": 1,
            "harvestable_fraction": 0.5,
            "rg_ratio": None,
            "dpi_residue_weight": 0.4,
            "output_metric": "dpi_grain",
            "agg_level": "cell",
            "no_grid_output": False,
            "enable_livestock_demand": False,
            "feed_scenarios": None,
        },
    }
    return dict(defaults[uc_name])


def build_project_config(
    *,
    use_cases: list[str] | None = None,
    cultivar_ids: list[str] | None = None,
    drought_threshold_grid: list[float] | None = None,
    feed_scenarios: str | None = None,
    n_years: int = 6,
    platform_translator: str = "default",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a spec-compliant project_config dict for PR3 emit tests.

    Per parent contract v1.1.6 §2.7.6 + PR3 §1.3, ``project_config`` is the
    canonical input shape consumed by ``create_manifest``. PR3 introduces
    a ``use_case_config`` sub-dict that signals which UCs to emit (the
    CLOSED-WORLD source per v0.3 BL-2).
    """
    use_cases = use_cases or []
    start_year = 2018
    end_year = start_year + max(n_years - 1, 0)

    use_case_config: dict[str, dict[str, Any]] = {}
    for uc in use_cases:
        cfg = _minimal_uc_config(uc)
        if uc == "yield_forecast" and cultivar_ids is not None:
            cfg["cultivar_ids"] = cultivar_ids
        if uc == "drought_management" and drought_threshold_grid is not None:
            cfg["drought_threshold_grid"] = drought_threshold_grid
        if uc == "livestock_feed" and feed_scenarios is not None:
            cfg["feed_scenarios"] = feed_scenarios
        use_case_config[uc] = cfg

    base: dict[str, Any] = {
        "project_name": "phase3_pr3_eval2_pkg",
        "region_name": "Koutiala",
        "country": "Mali",
        "gadm_level": 2,
        "crop_name": "sorghum",
        "planting_doy": 175,
        "maturity_doy": 305,
        "start_year": start_year,
        "end_year": end_year,
        "spinup_years": 0,
        # data_sources is REQUIRED_AT_CREATION + must be NON-EMPTY (manifest.py
        # MANIFEST_L2_TIERS / create_manifest validation): a realistic climate +
        # soil source pair so the fixture builds a valid manifest.
        "data_sources": {"climate": "AgERA5", "soil": "iSDA"},
        "use_case_config": use_case_config,
        "platform_translator": platform_translator,
    }
    base.update(overrides)
    return base


# ──────────────────────────────────────────────────────────────────────────
# package_dir fixture — minimal valid directory that create_manifest can scan
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def package_dir(tmp_path: Path) -> Path:
    """Minimal package directory: one tiny file so ``create_manifest``'s
    ``collect_files_with_checksums`` produces ≥ 1 entry and a valid summary
    block. Specific contents don't affect uc_readiness / use_case_config /
    crops emit clauses (those derive from ``project_config`` only).
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "metadata.json").write_text(json.dumps({"placeholder": True}))
    return pkg
