"""Phase 3 spec-driven behavior tests — advisory_flag emit matrix (eval-2).

Spec source: ``EXP/prism-runner/PHASE-B-PR3-CONTRACT.md`` v0.5 §1.3 items 6-8 +
§2.1.2 ACEA translator append + §2.7.6.1 MUST-3 + MUST-6 (parent v1.1.6) +
§2.7.7 4 display-guide flag dispositions.

Probes cover:
    T1   uc3_sowing_advisory_unconditional_all_4_platforms (MUST-3)
    T2   uc5_pk_advisory_positive_dssat_engine_platforms (MUST-6 positive: pythia+craft)
    T3   uc5_pk_advisory_negative_non_dssat_engine_platforms (MUST-6 negative-by-platform)
    T4   uc5_pk_advisory_negative_when_trigger_unset (MUST-6 negative-by-trigger)
    T5   uc1_shortfall_threshold_literal_flag_emit (8a)
    T6   uc4_severity_tier_literal_flag_emit (8b)
    T7   uc5_roi_prices_literal_flag_emit (8c)
    T8   uc6_herd_density_literal_flag_emit (8d)
    T9   display_guide_flags_emit_only_for_emitted_uc (closed-world cross-flag)
    T10  per_platform_per_uc_matrix_no_cross_pollination
    T11  wire_format_compliance_split_on_first_colon
    T12  unknown_advisory_keys_fall_through_no_crash (§2.7.6 fail-open)
    T13  flag_uniqueness_per_uc_no_duplicate_emit
    T14  flag_list_stable_across_repeated_emits (determinism)
"""

from __future__ import annotations

import re

import pytest

from prismpy.packaging.manifest import create_manifest

from .conftest import (
    ADVISORY_UC1_SHORTFALL_PREFIX,
    ADVISORY_UC3_SOWING,
    ADVISORY_UC4_SEVERITY_TIER,
    ADVISORY_UC5_PYTHIA_PK,
    ADVISORY_UC5_ROI_PRICES,
    ADVISORY_UC6_HERD_DENSITY,
    KNOWN_UC_NAMES,
    build_project_config,
)


# ──────────────────────────────────────────────────────────────────────────
# T1 — UC3 MUST-3 unconditional emit across 4 platforms
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("platform", ["sarra_py", "pythia", "craft", "dssat"])
def test_t1_uc3_sowing_advisory_unconditional_all_4_platforms(platform, package_dir):
    """§2.7.6.1 MUST-3 + PR3 §1.3 item 6: ``sowing_rule_default_absent:
    falls_back_to_manifest_default`` emitted UNCONDITIONALLY for any
    platform when UC3 declared (until v3.2 adapter_capability emit lands
    per OQ-A1-15-v3.2). Lesson #31 honest-signal — silent-fallback
    closure on legacy packages.
    """
    cfg = build_project_config(use_cases=["sowing_optimization"], n_years=10)
    m = create_manifest(package_dir, cfg, platform=platform)
    flags = m["uc_readiness"]["sowing_optimization"]["advisory_flags"]
    assert ADVISORY_UC3_SOWING in flags, (
        f"MUST-3 violation: platform={platform!r} UC3 advisory missing: {flags}"
    )


# The P+K advisory emit gate (generator SSOT, manifest.py): fires iff
# uc_name == "soil_fertility" AND platform in {"pythia", "craft"} (the DSSAT-
# engine producers — craft's .SNX hard-codes PHOSP=N + POTAS=N, so P/K is
# unmodeled on the same engine as pythia; the flag name keeps its historical
# "pythia" prefix) AND the P/K-unmodeled trigger is set. The trigger is set here
# directly via additional_metadata (the emit-level surface), mirroring the live
# translator handoff (test_pythia_uc5_trigger_handoff.py). Verified empirically:
# pythia/craft + trigger EMIT; acea/sarra_py/dssat don't; trigger-unset doesn't.
_PK_TRIGGER = {"_acea_uc5_p_k_silent_no_op_triggered": True}


# ──────────────────────────────────────────────────────────────────────────
# T2 — UC5 MUST-6 positive: emits for the DSSAT-engine platforms {pythia, craft}
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("platform", ["pythia", "craft"])
def test_t2_uc5_pk_advisory_positive_dssat_engine_platforms(platform, package_dir):
    """§2.7.6.1 MUST-6: ``pythia_pk_silent_no_op:fertility_stress_unmodeled_v3.1``
    emits when the UC5 P/K-unmodeled trigger is set AND the platform is a
    DSSAT-engine producer (``{pythia, craft}``). Drives Dr. Kofi paper-
    replication + Moussa cross-country fertilizer ROI bulletin honest-signal.
    """
    cfg = build_project_config(use_cases=["soil_fertility"], n_years=10)
    m = create_manifest(
        package_dir, cfg, platform=platform, additional_metadata=_PK_TRIGGER,
    )
    flags = m["uc_readiness"]["soil_fertility"]["advisory_flags"]
    assert ADVISORY_UC5_PYTHIA_PK in flags, (
        f"MUST-6: {platform!r} (DSSAT-engine) + trigger must emit the P+K flag; "
        f"got: {flags}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T3 — UC5 MUST-6 negative-by-platform: non-DSSAT-engine platforms don't emit
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("platform", ["sarra_py", "acea", "dssat"])
def test_t3_uc5_pk_advisory_negative_non_dssat_engine_platforms(platform, package_dir):
    """MUST-6 negative: even WITH the trigger set, a platform NOT in
    ``{pythia, craft}`` must NOT emit the P+K flag — the emit gate is
    platform-scoped. acea produces via its own engine (not the DSSAT ``.SNX``
    P/K hard-code), so it is filtered here despite its translator setting the
    trigger upstream.
    """
    cfg = build_project_config(use_cases=["soil_fertility"], n_years=10)
    m = create_manifest(
        package_dir, cfg, platform=platform, additional_metadata=_PK_TRIGGER,
    )
    flags = m["uc_readiness"]["soil_fertility"]["advisory_flags"]
    assert ADVISORY_UC5_PYTHIA_PK not in flags, (
        f"MUST-6 false-positive: {platform!r} (non-DSSAT-engine) must NOT emit "
        f"the P+K flag even with the trigger set; got: {flags}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T4 — UC5 MUST-6 negative-by-trigger: DSSAT-engine platform, trigger UNSET
# ──────────────────────────────────────────────────────────────────────────


def test_t4_uc5_pk_advisory_negative_when_trigger_unset(package_dir):
    """MUST-6 negative: a DSSAT-engine platform (pythia) with the P/K-unmodeled
    trigger NOT set must NOT emit the flag — the emit gate requires the trigger,
    not just the platform. (The live translators set the trigger unconditionally
    at dispatch; this proves the emit-side gate still requires it.)
    """
    cfg = build_project_config(use_cases=["soil_fertility"], n_years=10)
    m = create_manifest(package_dir, cfg, platform="pythia")  # no trigger
    flags = m["uc_readiness"]["soil_fertility"]["advisory_flags"]
    assert ADVISORY_UC5_PYTHIA_PK not in flags, (
        f"MUST-6 false-positive: pythia WITHOUT the trigger must NOT emit the "
        f"P+K flag; got: {flags}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T5-T8 — Literal display-guide flag emit (§1.3 items 8a-8d)
# ──────────────────────────────────────────────────────────────────────────


def test_t5_uc1_shortfall_threshold_literal_flag_emit(package_dir):
    """§1.3 item 8a + §2.7.7 OQ-A1-7: literal flag string
    ``shortfall_threshold:viz_layer_default_<value>_kgha_<crop>_<region>``
    emitted always when UC1 in use_case_config. Per F2 fold, the prefix
    ``shortfall_threshold:viz_layer_default_`` is the binding literal;
    the suffix interpolates region/crop-specific defaults.
    """
    cfg = build_project_config(
        use_cases=["yield_forecast"], n_years=10,
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    flags = m["uc_readiness"]["yield_forecast"]["advisory_flags"]
    matches = [f for f in flags if f.startswith(ADVISORY_UC1_SHORTFALL_PREFIX)]
    assert matches, (
        f"§1.3 item 8a violation: UC1 shortfall_threshold flag with prefix "
        f"{ADVISORY_UC1_SHORTFALL_PREFIX!r} not in flags={flags}"
    )
    # Single emit (not duplicated)
    assert len(matches) == 1, f"UC1 shortfall flag duplicated: {matches}"


def test_t6_uc4_severity_tier_literal_flag_emit(package_dir):
    """§1.3 item 8b + §2.7.7 OQ-A1-11: literal constant string
    ``severity_tier:viz_layer_thresholds_v1`` emitted always when UC4
    in use_case_config.
    """
    cfg = build_project_config(
        use_cases=["drought_management"], n_years=10,
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    flags = m["uc_readiness"]["drought_management"]["advisory_flags"]
    assert ADVISORY_UC4_SEVERITY_TIER in flags, (
        f"§1.3 item 8b violation: UC4 severity_tier flag missing: {flags}"
    )


def test_t7_uc5_roi_prices_literal_flag_emit(package_dir):
    """§1.3 item 8c + §2.7.7 OQ-A1-12: literal constant string
    ``roi_prices:viz_layer_regional_defaults`` emitted always when UC5
    in use_case_config.
    """
    cfg = build_project_config(
        use_cases=["soil_fertility"], n_years=10,
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    flags = m["uc_readiness"]["soil_fertility"]["advisory_flags"]
    assert ADVISORY_UC5_ROI_PRICES in flags, (
        f"§1.3 item 8c violation: UC5 roi_prices flag missing: {flags}"
    )


def test_t8_uc6_herd_density_literal_flag_emit(package_dir):
    """§1.3 item 8d + §2.7.7 OQ-A1-13: literal constant string
    ``herd_density:GLW_2020_default_supply_side_only`` emitted always
    when UC6 in use_case_config.
    """
    cfg = build_project_config(
        use_cases=["livestock_feed"], n_years=10,
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    flags = m["uc_readiness"]["livestock_feed"]["advisory_flags"]
    assert ADVISORY_UC6_HERD_DENSITY in flags, (
        f"§1.3 item 8d violation: UC6 herd_density flag missing: {flags}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T9 — Closed-world: display-guide flags emit only for declared UCs
# ──────────────────────────────────────────────────────────────────────────


def test_t9_display_guide_flags_emit_only_for_emitted_uc(package_dir):
    """v0.3 BL-2 closed-world: when UC1 is NOT declared in
    use_case_config, the UC1 shortfall flag MUST NOT appear in ANY UC's
    advisory_flags (and the UC1 key must be ABSENT from uc_readiness).
    Cross-flag closed-world consistency.
    """
    # UC4 + UC6 only — UC1/UC5 NOT declared
    cfg = build_project_config(
        use_cases=["drought_management", "livestock_feed"], n_years=10,
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    ucr = m["uc_readiness"]
    # UC1 not emitted
    assert "yield_forecast" not in ucr, (
        f"UC1 not declared but emitted: {set(ucr)}"
    )
    assert "soil_fertility" not in ucr, (
        f"UC5 not declared but emitted: {set(ucr)}"
    )
    # No UC1 / UC5 flags anywhere
    for uc, entry in ucr.items():
        for f in entry["advisory_flags"]:
            assert not f.startswith(ADVISORY_UC1_SHORTFALL_PREFIX), (
                f"UC1 flag leaked into {uc}: {f}"
            )
            assert f != ADVISORY_UC5_ROI_PRICES, (
                f"UC5 flag leaked into {uc}: {f}"
            )
            assert f != ADVISORY_UC5_PYTHIA_PK, (
                f"UC5 flag leaked into {uc}: {f}"
            )


# ──────────────────────────────────────────────────────────────────────────
# T10 — Per-platform × per-UC matrix: no cross-pollination
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("platform", ["sarra_py", "pythia", "craft", "dssat"])
def test_t10_per_platform_per_uc_matrix_no_cross_pollination(platform, package_dir):
    """All 6 UCs declared + 4 platforms = 24-cell matrix. For each cell,
    each UC's advisory_flags MUST contain only that UC's flags
    (cross-UC contamination check).

    Per-UC expected flag substrings (anchor-prefix or full literal):
        UC1: 'shortfall_threshold:'
        UC3: 'sowing_rule_default_absent:'
        UC4: 'severity_tier:'
        UC5: 'roi_prices:'
        UC5 (pythia+acea only): 'pythia_pk_silent_no_op:'
        UC6: 'herd_density:'
    """
    cfg = build_project_config(
        use_cases=sorted(KNOWN_UC_NAMES), n_years=10,
        platform_translator="acea",
    )
    m = create_manifest(package_dir, cfg, platform=platform)
    expected_prefix_per_uc = {
        "yield_forecast": "shortfall_threshold:",
        "sowing_optimization": "sowing_rule_default_absent:",
        "drought_management": "severity_tier:",
        "soil_fertility": "roi_prices:",
        "livestock_feed": "herd_density:",
        "climate_scenarios": None,  # no display-guide flag for UC2
    }
    for uc, expected_prefix in expected_prefix_per_uc.items():
        entry_flags = m["uc_readiness"][uc]["advisory_flags"]
        if expected_prefix:
            has_own = any(f.startswith(expected_prefix) for f in entry_flags)
            assert has_own, (
                f"platform={platform} UC {uc!r} missing own flag prefix "
                f"{expected_prefix!r}: {entry_flags}"
            )
        # Cross-pollination: no OTHER UC's flag should appear here
        for other_uc, other_prefix in expected_prefix_per_uc.items():
            if other_uc == uc or other_prefix is None:
                continue
            contaminated = [f for f in entry_flags if f.startswith(other_prefix)]
            assert not contaminated, (
                f"platform={platform} UC {uc!r} contaminated with UC {other_uc!r} "
                f"flag(s) {contaminated} (prefix {other_prefix!r})"
            )


# ──────────────────────────────────────────────────────────────────────────
# T11 — Wire-format compliance per §2.7.6 FORMAT clause
# ──────────────────────────────────────────────────────────────────────────


_WIRE_FORMAT_RE = re.compile(r"^[a-z][a-z0-9_]*:[A-Za-z0-9_\-:<>]+$")


def test_t11_wire_format_compliance_split_on_first_colon(package_dir):
    """§2.7.6 FORMAT clause: every flag is ``<snake_case_key>:
    <kebab-or-snake-description>``; split-on-first-colon yields
    well-formed (key, description) tuple. Key matches ``[a-z][a-z0-9_]*``.

    Sweep all 6 UCs + PYTHIA-ACEA conditional + display-guide flags.
    """
    cfg = build_project_config(
        use_cases=sorted(KNOWN_UC_NAMES), n_years=10,
        platform_translator="acea",
    )
    m = create_manifest(package_dir, cfg, platform="pythia")
    violations = []
    for uc, entry in m["uc_readiness"].items():
        for f in entry["advisory_flags"]:
            if ":" not in f:
                violations.append(f"{uc}: no-colon flag {f!r}")
                continue
            key, _, desc = f.partition(":")
            if not re.match(r"^[a-z][a-z0-9_]*$", key):
                violations.append(f"{uc}: bad key {key!r} in flag {f!r}")
            if not desc:
                violations.append(f"{uc}: empty description in flag {f!r}")
    assert not violations, f"wire-format violations: {violations}"


# ──────────────────────────────────────────────────────────────────────────
# T12 — Unknown advisory keys do not crash (fail-open per Lesson #31)
# ──────────────────────────────────────────────────────────────────────────


def test_t12_unknown_advisory_keys_fall_through_no_crash(package_dir):
    """§2.7.6 FORMAT: 'parser is permissive — unknown keys surface as raw
    strings in prismweb honest-signal panel (fail-open per Lesson #31)'.
    The prismpy emitter never generates 'unknown' keys, but if a future
    cycle adds a new flag class, the cross-codebase consumer must
    fail-open. This probe ensures the emit path doesn't crash on the
    standard happy-path (i.e., flag generation is robust).
    """
    # This is mostly a smoke test: if PR3 emit is robust, the result is
    # well-formed; we don't try to inject a malformed flag (that's a
    # consumer-side test). Just verify no crash on the full matrix.
    for plat in ["sarra_py", "pythia"]:
        for translator in ["default", "acea"]:
            cfg = build_project_config(
                use_cases=sorted(KNOWN_UC_NAMES), n_years=10,
                platform_translator=translator,
            )
            try:
                m = create_manifest(package_dir, cfg, platform=plat)
            except Exception as exc:  # noqa: BLE001
                pytest.fail(
                    f"emit crashed on platform={plat}, translator={translator}: "
                    f"{type(exc).__name__}: {exc}"
                )
            # Sanity: at least UC3 sowing flag must be present
            uc3_flags = m["uc_readiness"]["sowing_optimization"]["advisory_flags"]
            assert ADVISORY_UC3_SOWING in uc3_flags


# ──────────────────────────────────────────────────────────────────────────
# T13 — Flag uniqueness per UC (no duplicate emit)
# ──────────────────────────────────────────────────────────────────────────


def test_t13_flag_uniqueness_per_uc_no_duplicate_emit(package_dir):
    """Adversarial: every flag in a UC's advisory_flags list should be
    UNIQUE. Duplicate emits would indicate two code paths both append
    the same string (a class of bugs to avoid).
    """
    cfg = build_project_config(
        use_cases=sorted(KNOWN_UC_NAMES), n_years=10,
        platform_translator="acea",
    )
    m = create_manifest(package_dir, cfg, platform="pythia")
    for uc, entry in m["uc_readiness"].items():
        flags = entry["advisory_flags"]
        unique = set(flags)
        if len(flags) != len(unique):
            duplicates = [f for f in unique if flags.count(f) > 1]
            pytest.fail(
                f"UC {uc!r} has duplicate flag(s): {duplicates}; full list: {flags}"
            )


# ──────────────────────────────────────────────────────────────────────────
# T14 — Flag list stability across repeated emits (determinism)
# ──────────────────────────────────────────────────────────────────────────


def test_t14_flag_list_stable_across_repeated_emits(package_dir):
    """Determinism: two emits with same input produce same advisory_flags
    list (same elements + same order). Non-deterministic ordering would
    cause spurious diffs in CI / cross-codebase comparison.
    """
    cfg = build_project_config(
        use_cases=sorted(KNOWN_UC_NAMES), n_years=10,
        platform_translator="acea",
    )
    m1 = create_manifest(package_dir, cfg, platform="pythia")
    m2 = create_manifest(package_dir, cfg, platform="pythia")
    for uc in m1["uc_readiness"]:
        f1 = m1["uc_readiness"][uc]["advisory_flags"]
        f2 = m2["uc_readiness"][uc]["advisory_flags"]
        assert f1 == f2, (
            f"UC {uc!r} advisory_flags order drift: {f1} vs {f2}"
        )
