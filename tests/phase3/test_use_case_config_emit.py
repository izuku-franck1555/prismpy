"""Phase 3 spec-driven behavior tests — `manifest.use_case_config` emit (eval-2).

Spec source: ``EXP/prism-runner/PHASE-B-PR3-CONTRACT.md`` v0.5 §1.3 item 8e +
§2.1.1 `canonical_use_case_config_serializer` (v0.4 codex BL-2 residual
CLOSED-WORLD closure) + parent v1.1.6 §2.7.3 keysets + §2.7.8 SSOT enforcement.

Probes cover:
    T1   closed_keyset_per_uc_matches_table (§2.7.3)
    T2   no_extra_keys_per_uc (no pollution)
    T3   multi_candidate_fields_round_trip_uc1_uc4_uc6
    T4   json_encode_decode_preserves_list_str_shape
    T5   closed_world_non_emitted_uc_absent (v0.4 BL-2)
    T6   cross_uc_contamination_uc1_field_not_in_uc4
    T7   enum_values_match_argparse_ssot
    T8   path_c_blind_merge_does_not_pollute_emit (OQ-PR3-6)
    T9   all_six_ucs_emit_simultaneously_no_drift
    T10  empty_use_case_config_produces_empty_emit
"""

from __future__ import annotations

import json

import pytest

from prismpy.packaging.manifest import create_manifest

from .conftest import (
    KNOWN_UC_NAMES,
    UC_KEYSETS,
    build_project_config,
)


# ──────────────────────────────────────────────────────────────────────────
# T1 — closed-keyset per UC matches §2.7.3 table
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("uc_kind", sorted(KNOWN_UC_NAMES))
def test_t1_closed_keyset_per_uc_matches_table(uc_kind, package_dir):
    """§1.3 item 8e + parent §2.7.3: each UC's ``manifest.use_case_config[uc]``
    contains EXACTLY the keys declared in the §2.7.3 table for that UC
    (verbatim from PR3 §1.3 item 8e). No missing required key; no extra
    key. AST guard T4 (§2.2.4) enforces.
    """
    cfg = build_project_config(use_cases=[uc_kind])
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    assert "use_case_config" in m, "use_case_config key absent from manifest"
    assert uc_kind in m["use_case_config"], (
        f"UC {uc_kind!r} not in use_case_config: {set(m['use_case_config'])}"
    )
    actual_keys = set(m["use_case_config"][uc_kind].keys())
    expected_keys = UC_KEYSETS[uc_kind]
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    assert not missing, (
        f"UC {uc_kind!r} missing required keys: {missing}; "
        f"actual: {actual_keys}; expected: {expected_keys}"
    )
    assert not extra, (
        f"UC {uc_kind!r} has extra keys (closed-keyset violation): {extra}; "
        f"actual: {actual_keys}; expected: {expected_keys}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T2 — no extra keys per UC (defense-in-depth complement to T1)
# ──────────────────────────────────────────────────────────────────────────


def test_t2_no_extra_keys_per_uc_aggregated(package_dir):
    """All 6 UCs emitted simultaneously; assert each UC's keyset stays
    within the §2.7.3 closed-keyset. Aggregated probe to catch
    cross-UC leakage at the use_case_config level.
    """
    cfg = build_project_config(use_cases=sorted(KNOWN_UC_NAMES))
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    violations = []
    for uc_kind in KNOWN_UC_NAMES:
        actual = set(m["use_case_config"][uc_kind].keys())
        expected = UC_KEYSETS[uc_kind]
        extra = actual - expected
        if extra:
            violations.append(f"{uc_kind!r}: extra keys {extra}")
    assert not violations, f"closed-keyset violations: {violations}"


# ──────────────────────────────────────────────────────────────────────────
# T3 — multi-candidate fields round-trip (cultivar_ids / drought_grid /
#       feed_scenarios) preserve list-str / list-float / str shape
# ──────────────────────────────────────────────────────────────────────────


def test_t3_multi_candidate_fields_round_trip_uc1_uc4_uc6(package_dir):
    """§1.3 items 1-3 + PR1 §2.1.1 multi-candidate design:
    - UC1 cultivar_ids: list[str]
    - UC4 drought_threshold_grid: list[float]
    - UC6 feed_scenarios: str (CSV/preset/comma-list)
    Each must survive emit with type preserved (not coerced to str).
    """
    cfg = build_project_config(
        use_cases=["yield_forecast", "drought_management", "livestock_feed"],
        cultivar_ids=["IT89KD-288", "IT94K-1"],
        drought_threshold_grid=[0.3, 0.4, 0.5],
        feed_scenarios="preset:dpi-factorial-3x2",
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    ucc = m["use_case_config"]
    # UC1 cultivar_ids
    cv = ucc["yield_forecast"]["cultivar_ids"]
    assert cv == ["IT89KD-288", "IT94K-1"], (
        f"UC1 cultivar_ids type/value drift: {cv!r} (type {type(cv).__name__})"
    )
    assert isinstance(cv, list) and all(isinstance(c, str) for c in cv)
    # UC4 drought_threshold_grid
    gr = ucc["drought_management"]["drought_threshold_grid"]
    assert gr == [0.3, 0.4, 0.5], (
        f"UC4 drought_threshold_grid drift: {gr!r}"
    )
    assert isinstance(gr, list) and all(isinstance(x, float) for x in gr)
    # UC6 feed_scenarios
    fs = ucc["livestock_feed"]["feed_scenarios"]
    assert fs == "preset:dpi-factorial-3x2", (
        f"UC6 feed_scenarios drift: {fs!r}"
    )
    assert isinstance(fs, str)


# ──────────────────────────────────────────────────────────────────────────
# T4 — JSON encode/decode preserves list/str shape (cross-codebase wire)
# ──────────────────────────────────────────────────────────────────────────


def test_t4_json_encode_decode_preserves_list_str_shape(package_dir):
    """v1.1.5 strict Celery boundary discipline: multi-candidate fields
    must survive JSON wire (prismpy → manifest.json → prismweb → prism-
    runner Celery boundary shape-check).
    """
    cfg = build_project_config(
        use_cases=["yield_forecast"],
        cultivar_ids=["IT89KD-288", "IT94K-1", "IT93K-503-1"],
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    # JSON round-trip
    json_str = json.dumps(m, sort_keys=True, ensure_ascii=False)
    parsed = json.loads(json_str)
    cv = parsed["use_case_config"]["yield_forecast"]["cultivar_ids"]
    assert cv == ["IT89KD-288", "IT94K-1", "IT93K-503-1"], (
        f"JSON round-trip lost cultivar_ids: {cv!r}"
    )
    # Critical: stayed list[str], NOT coerced to single string
    assert isinstance(cv, list), (
        f"JSON round-trip coerced cultivar_ids to {type(cv).__name__} (must "
        f"remain list)"
    )


# ──────────────────────────────────────────────────────────────────────────
# T5 — CLOSED-WORLD: non-emitted UCs ABSENT from use_case_config (v0.4 BL-2)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "emitted_ucs,expected_keys",
    [
        (["yield_forecast"], {"yield_forecast"}),
        (["climate_scenarios"], {"climate_scenarios"}),
        (["yield_forecast", "drought_management"],
         {"yield_forecast", "drought_management"}),
        (["soil_fertility"], {"soil_fertility"}),
    ],
)
def test_t5_closed_world_non_emitted_uc_absent(
    emitted_ucs, expected_keys, package_dir,
):
    """v0.4 codex BL-2 residual + PR3 §2.1.1 row 164: 'Non-emitted UCs
    MUST be ABSENT from manifest.use_case_config dict — NOT present with
    empty sub-dicts'. Iterating KNOWN_USE_CASE_NAMES directly would
    re-expand the emitted set back to 6 → same UI confirm-card
    regression class as the BL-2 closure on uc_readiness.
    """
    cfg = build_project_config(use_cases=emitted_ucs)
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    actual = set(m["use_case_config"].keys())
    assert actual == expected_keys, (
        f"closed-world violation: emitted {emitted_ucs}, expected "
        f"use_case_config keys {expected_keys}, got {actual}"
    )
    # No empty sub-dict either
    for uc, sub in m["use_case_config"].items():
        assert sub, f"UC {uc!r} present with empty sub-dict: {sub!r}"


# ──────────────────────────────────────────────────────────────────────────
# T6 — Cross-UC contamination at the use_case_config level
# ──────────────────────────────────────────────────────────────────────────


def test_t6_cross_uc_contamination_uc1_field_not_in_uc4(package_dir):
    """UC1's cultivar_ids field must NOT leak into UC4's use_case_config
    sub-dict. UC4's drought_threshold_grid must NOT leak into UC1's
    sub-dict. Strict per-UC keyset isolation.
    """
    cfg = build_project_config(
        use_cases=["yield_forecast", "drought_management"],
        cultivar_ids=["IT94K-1"],
        drought_threshold_grid=[0.4],
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    uc1 = m["use_case_config"]["yield_forecast"]
    uc4 = m["use_case_config"]["drought_management"]
    assert "drought_threshold_grid" not in uc1, (
        f"UC4 field leaked into UC1: {uc1!r}"
    )
    assert "drought_threshold" not in uc1
    assert "cultivar_ids" not in uc4, (
        f"UC1 field leaked into UC4: {uc4!r}"
    )
    assert "n_analogs" not in uc4


# ──────────────────────────────────────────────────────────────────────────
# T7 — Enum values match argparse SSOT (sample subset; full check is
#      cross-codebase AST T4 in §2.2.4 — runs as manual smoke per OQ-PR3-4)
# ──────────────────────────────────────────────────────────────────────────


def test_t7_enum_values_match_argparse_ssot(package_dir):
    """§2.7.8 SSOT enforcement + PR3 §1.3 item 8e: emitted enum-typed
    values match the cli.py argparse SSOT. Sample: UC4 risk_metric must
    be one of the 5 documented values per parent v1.1.6 §2.7.7.1.

    Full cross-codebase AST verification is §2.2.4 T4 (manual smoke per
    OQ-PR3-4 v0.3 disposition). This probe is a fast sanity check.
    """
    expected_uc4_risk_metric = {
        "prob_drought", "SPI", "SPEI",
        "drought_yield_loss", "drought_freq_anomaly",
    }
    cfg = build_project_config(use_cases=["drought_management"])
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    rm = m["use_case_config"]["drought_management"]["risk_metric"]
    assert rm in expected_uc4_risk_metric, (
        f"UC4 risk_metric value {rm!r} not in argparse SSOT set "
        f"{expected_uc4_risk_metric}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T8 — Path-(c) blind-merge regression check (OQ-PR3-6)
# ──────────────────────────────────────────────────────────────────────────


def test_t8_path_c_blind_merge_does_not_pollute_emit(package_dir):
    """OQ-PR3-6 v0.1 recommendation was path-(b) introspection
    (project_config['use_case_config']) OR path-(a) signature change —
    NOT path-(c) blind-merge of additional_metadata.

    Adversarial: if builder accidentally used path-(c) (passing
    additional_metadata directly through to create_manifest body),
    a user-set additional_metadata['use_case_config'] = {'frobnication':
    {...}} would silently land in manifest.use_case_config.

    Path-(c) blind-merge is now PREVENTED FAIL-LOUD: create_manifest RAISES on a
    reserved-key collision (additional_metadata may not carry a canonical
    manifest key like use_case_config) — a STRONGER guarantee than the original
    silent-non-pollution contract. This probe asserts the raise, then that a
    non-reserved additional_metadata key leaves a clean UC emit.
    """
    cfg = build_project_config(use_cases=["yield_forecast"])
    # A bogus use_case_config injected via additional_metadata is REJECTED
    # (reserved-key collision), never silently merged into use_case_config.
    with pytest.raises(ValueError, match="canonical manifest keys"):
        create_manifest(
            package_dir, cfg, platform="sarra_py",
            additional_metadata={"use_case_config": {"frobnication": {"cores": 99}}},
        )
    # A non-reserved additional_metadata key does NOT pollute the UC emit.
    m = create_manifest(
        package_dir, cfg, platform="sarra_py",
        additional_metadata={"scenario_info": "preserved"},
    )
    ucc = m.get("use_case_config", {})
    assert "frobnication" not in ucc, (
        f"'frobnication' polluted use_case_config: {set(ucc.keys())}"
    )
    assert "yield_forecast" in ucc, f"legitimate UC1 emit lost; ucc={ucc}"


# ──────────────────────────────────────────────────────────────────────────
# T9 — All-6-UCs simultaneously emit no drift
# ──────────────────────────────────────────────────────────────────────────


def test_t9_all_six_ucs_emit_simultaneously_no_drift(package_dir):
    cfg = build_project_config(use_cases=sorted(KNOWN_UC_NAMES))
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    actual = set(m["use_case_config"].keys())
    assert actual == KNOWN_UC_NAMES, (
        f"all-6 declared but emit set drift: {actual} != {KNOWN_UC_NAMES}"
    )
    # Per-UC keyset compliance preserved at high cardinality
    for uc_kind in KNOWN_UC_NAMES:
        actual_keys = set(m["use_case_config"][uc_kind].keys())
        expected_keys = UC_KEYSETS[uc_kind]
        assert actual_keys == expected_keys, (
            f"all-6 emit: UC {uc_kind!r} keyset drift: "
            f"{actual_keys} vs {expected_keys}"
        )


# ──────────────────────────────────────────────────────────────────────────
# T10 — Empty use_case_config produces empty (or absent) emit
# ──────────────────────────────────────────────────────────────────────────


def test_t10_empty_use_case_config_produces_empty_emit(package_dir):
    """Adversarial: project_config['use_case_config'] = {} (explicit empty)
    → manifest.use_case_config = {} OR ABSENT (NOT auto-populated with
    KNOWN_USE_CASE_NAMES — that would be the v0.1.1 regression class).
    """
    cfg = build_project_config(use_cases=[])
    cfg["use_case_config"] = {}  # explicit empty
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    ucc = m.get("use_case_config", {})
    assert ucc == {}, (
        f"empty use_case_config should yield empty/absent emit; got: {ucc!r}"
    )
