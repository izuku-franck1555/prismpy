"""Phase 3 spec-driven behavior tests — `manifest.crops` widening (eval-2).

Spec source: ``EXP/prism-runner/PHASE-B-PR3-CONTRACT.md`` v0.5 §1.3 item 5 +
§2.1.1 `canonical_crops_emitter` + §2.7.6.1 MUST-7 (parent v1.1.6).

CLOSURE: prismpy emits ``manifest.crops: list[dict]`` (ALWAYS); legacy
``manifest.crop: dict`` ALSO emitted for backward-compat per §2.1.1 row 4
("backward-compat: also emit legacy 'crop:' singleton key"). Singleton
shape: 1-element list when cultivar_ids absent / None / length ≤ 1.
Multi-cultivar shape: N-element list when cultivar_ids declared with N ≥ 2.

Probes:
    T1   singleton_cardinality_absent_cultivar_ids
    T2   singleton_cardinality_none_cultivar_ids
    T3   singleton_cardinality_single_cultivar
    T4   multi_cardinality_two_cultivars
    T5   multi_cardinality_three_cultivars
    T6   multi_cardinality_five_cultivars
    T7   multi_cardinality_ten_cultivars
    T8   each_crop_dict_carries_required_keys
    T9   crop_name_shared_across_multi_cultivar_emit
    T10  cultivar_id_unique_across_multi_cultivar_emit
    T11  backward_compat_legacy_crop_singleton_key_co_emitted
    T12  legacy_crop_singleton_matches_first_crops_entry
    T13  empty_list_cultivar_ids_falls_back_to_singleton
    T14  duplicate_cultivar_ids_each_preserved_no_dedup
    T15  whitespace_cultivar_id_handling
    T16  crops_is_list_dict_never_dict
"""

from __future__ import annotations

import pytest

from prismpy.packaging.manifest import create_manifest

from .conftest import build_project_config


# ──────────────────────────────────────────────────────────────────────────
# T1-T3 — Singleton path (cultivar_ids absent / None / single-element)
# ──────────────────────────────────────────────────────────────────────────


def test_t1_singleton_cardinality_absent_cultivar_ids(package_dir):
    """§2.1.1 canonical_crops_emitter: 'when cultivar_ids ... absent ...
    emits 1-element list with cultivar_id derived from project_config's
    primary crop entry'.
    """
    cfg = build_project_config(use_cases=["yield_forecast"])
    # cultivar_ids defaults to None inside _minimal_uc_config; assert
    cfg["use_case_config"]["yield_forecast"].pop("cultivar_ids", None)
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    crops = m.get("crops")
    assert isinstance(crops, list), (
        f"crops must be list, got {type(crops).__name__}: {crops!r}"
    )
    assert len(crops) == 1, (
        f"absent cultivar_ids should yield 1-element list, got {len(crops)}: "
        f"{crops!r}"
    )


def test_t2_singleton_cardinality_none_cultivar_ids(package_dir):
    cfg = build_project_config(
        use_cases=["yield_forecast"], cultivar_ids=None,
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    assert isinstance(m["crops"], list)
    assert len(m["crops"]) == 1


def test_t3_singleton_cardinality_single_cultivar(package_dir):
    cfg = build_project_config(
        use_cases=["yield_forecast"], cultivar_ids=["IT94K-1"],
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    assert isinstance(m["crops"], list)
    assert len(m["crops"]) == 1
    assert m["crops"][0].get("cultivar_id") == "IT94K-1"


# ──────────────────────────────────────────────────────────────────────────
# T4-T7 — Multi-cultivar matrix (2 / 3 / 5 / 10 cultivars)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cultivars",
    [
        ["IT89KD-288", "IT94K-1"],
        ["IT89KD-288", "IT94K-1", "IT93K-503-1"],
        ["A", "B", "C", "D", "E"],
        ["c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"],
    ],
)
def test_t4_to_t7_multi_cardinality_matrix(cultivars, package_dir):
    """§2.7.6.1 MUST-7 + PR3 §1.3 item 5: cultivar_ids length N ≥ 2
    produces N-element list."""
    cfg = build_project_config(
        use_cases=["yield_forecast"], cultivar_ids=cultivars,
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    crops = m["crops"]
    assert isinstance(crops, list)
    assert len(crops) == len(cultivars), (
        f"cultivar_ids cardinality {len(cultivars)} → crops {len(crops)}; "
        f"crops={crops!r}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T8 — Per-crop dict schema
# ──────────────────────────────────────────────────────────────────────────


def test_t8_each_crop_dict_carries_required_keys(package_dir):
    """§2.1.1 canonical_crops_emitter: 'each dict carries crop_name,
    planting_doy, maturity_doy, cultivar_id'.
    """
    cfg = build_project_config(
        use_cases=["yield_forecast"],
        cultivar_ids=["IT89KD-288", "IT94K-1"],
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    required = {"crop_name", "planting_doy", "maturity_doy", "cultivar_id"}
    for i, crop in enumerate(m["crops"]):
        actual = set(crop.keys())
        missing = required - actual
        assert not missing, (
            f"crops[{i}] missing required keys {missing}: {crop!r}"
        )


# ──────────────────────────────────────────────────────────────────────────
# T9 — Shared crop_name across multi-cultivar emit
# ──────────────────────────────────────────────────────────────────────────


def test_t9_crop_name_shared_across_multi_cultivar_emit(package_dir):
    """All cultivars of the same crop share the same crop_name; only
    cultivar_id differs. Adversarial check: builder should NOT emit
    different crop_names per cultivar (e.g., from a per-cultivar
    metadata dict mistakenly indexed by cultivar_id).
    """
    cfg = build_project_config(
        use_cases=["yield_forecast"],
        cultivar_ids=["IT89KD-288", "IT94K-1", "IT93K-503-1"],
    )
    cfg["crop_name"] = "cowpea"
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    names = {c["crop_name"] for c in m["crops"]}
    assert names == {"cowpea"}, (
        f"crop_name should be shared 'cowpea' across cultivars; got {names}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T10 — cultivar_id unique across emit
# ──────────────────────────────────────────────────────────────────────────


def test_t10_cultivar_id_unique_across_multi_cultivar_emit(package_dir):
    cultivars = ["IT89KD-288", "IT94K-1", "IT93K-503-1"]
    cfg = build_project_config(
        use_cases=["yield_forecast"], cultivar_ids=cultivars,
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    emitted_ids = [c["cultivar_id"] for c in m["crops"]]
    assert sorted(emitted_ids) == sorted(cultivars), (
        f"emitted cultivar_ids {emitted_ids} != input {cultivars}"
    )
    assert len(set(emitted_ids)) == len(cultivars), (
        f"duplicate cultivar_ids in emit: {emitted_ids}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T11 — backward-compat: legacy `manifest.crop` singleton co-emitted
# ──────────────────────────────────────────────────────────────────────────


def test_t11_backward_compat_legacy_crop_singleton_key_co_emitted(package_dir):
    """§2.1.1 row 4 + PR3 §2.7.6.1 MUST-7: 'also emit legacy "crop:"
    singleton key (for packages built before PR1's consumer-side
    widening lands in production)'. PR1 UC1 processor accepts BOTH
    shapes; PR3 emit MUST preserve the legacy key for forward-compat
    until v3.2 cutoff.
    """
    cfg = build_project_config(
        use_cases=["yield_forecast"], cultivar_ids=["IT89KD-288", "IT94K-1"],
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    assert "crop" in m, "legacy 'crop' singleton key MISSING (backward-compat broken)"
    legacy = m["crop"]
    assert isinstance(legacy, dict), (
        f"legacy 'crop' must be dict (not list), got {type(legacy).__name__}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T12 — Legacy crop singleton matches first crops entry
# ──────────────────────────────────────────────────────────────────────────


def test_t12_legacy_crop_singleton_matches_first_crops_entry(package_dir):
    """Sanity: legacy ``crop`` should be coherent with new ``crops[0]`` —
    they describe the same data (crop_name + planting/maturity DOY). If
    they diverged, a pre-PR3 prism-runner would read different values
    than a post-PR1 prism-runner.
    """
    cfg = build_project_config(
        use_cases=["yield_forecast"], cultivar_ids=["IT89KD-288"],
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    legacy = m["crop"]
    first = m["crops"][0]
    # crop_name field-name disambiguation: legacy uses 'name'; new uses 'crop_name'
    legacy_name = legacy.get("name") or legacy.get("crop_name")
    new_name = first.get("crop_name")
    assert legacy_name == new_name, (
        f"legacy crop.name={legacy_name!r} != crops[0].crop_name={new_name!r}"
    )
    assert legacy.get("planting_doy") == first.get("planting_doy"), (
        "planting_doy diverges between legacy 'crop' and new 'crops[0]'"
    )


# ──────────────────────────────────────────────────────────────────────────
# T13 — Empty list cultivar_ids → singleton fallback
# ──────────────────────────────────────────────────────────────────────────


def test_t13_empty_list_cultivar_ids_falls_back_to_singleton(package_dir):
    """Adversarial: cultivar_ids=[] (explicit empty list) should behave
    identically to None / absent — falls back to 1-element list with the
    project_config's primary crop. PR1 consumer-side single-cultivar
    legacy path expects this.
    """
    cfg = build_project_config(use_cases=["yield_forecast"], cultivar_ids=[])
    try:
        m = create_manifest(package_dir, cfg, platform="sarra_py")
    except (ValueError, KeyError) as exc:
        pytest.fail(
            f"empty cultivar_ids=[] should not raise; got {type(exc).__name__}: {exc}"
        )
    assert isinstance(m["crops"], list)
    assert len(m["crops"]) == 1, (
        f"empty cultivar_ids=[] should yield 1-element fallback list; "
        f"got {len(m['crops'])}: {m['crops']!r}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T14 — Duplicate cultivar_ids preserved (no dedup at emit)
# ──────────────────────────────────────────────────────────────────────────


def test_t14_duplicate_cultivar_ids_each_preserved_no_dedup(package_dir):
    """Adversarial: user-supplied duplicates should NOT be silently deduped
    at emit (user is responsible for input hygiene; silent dedup would
    miscount cardinality on the consumer side). Either preserve duplicates
    OR raise — never silently drop.
    """
    cultivars = ["IT94K-1", "IT94K-1", "IT89KD-288"]
    cfg = build_project_config(
        use_cases=["yield_forecast"], cultivar_ids=cultivars,
    )
    try:
        m = create_manifest(package_dir, cfg, platform="sarra_py")
    except (ValueError, KeyError):
        return  # raise path acceptable
    crops = m["crops"]
    if len(crops) == len(cultivars):
        # Preserved
        emitted_ids = [c["cultivar_id"] for c in crops]
        assert sorted(emitted_ids) == sorted(cultivars)
    elif len(crops) < len(cultivars):
        pytest.fail(
            f"silent dedup: input {cultivars} ({len(cultivars)}) → emit "
            f"{[c['cultivar_id'] for c in crops]} ({len(crops)})"
        )
    else:
        pytest.fail(
            f"unexpected cardinality: input {len(cultivars)} → emit {len(crops)}"
        )


# ──────────────────────────────────────────────────────────────────────────
# T15 — Whitespace handling on cultivar_id
# ──────────────────────────────────────────────────────────────────────────


def test_t15_whitespace_cultivar_id_handling(package_dir):
    """Adversarial: leading/trailing whitespace in cultivar_id strings.
    Acceptable behaviors: (a) strip + emit; (b) preserve verbatim; (c) raise.
    NOT acceptable: silently drop entries.
    """
    cultivars = ["IT94K-1", " IT89KD-288 "]
    cfg = build_project_config(
        use_cases=["yield_forecast"], cultivar_ids=cultivars,
    )
    try:
        m = create_manifest(package_dir, cfg, platform="sarra_py")
    except (ValueError, KeyError):
        return  # raise path acceptable
    crops = m["crops"]
    assert len(crops) == 2, (
        f"whitespace handling dropped element: input 2 → emit {len(crops)}: "
        f"{crops!r}"
    )
    emitted = [c["cultivar_id"] for c in crops]
    # Either preserved-verbatim OR stripped — both acceptable, but
    # stripped values must still resolve to the original cultivars
    stripped = [s.strip() for s in emitted]
    assert sorted(stripped) == sorted(s.strip() for s in cultivars), (
        f"whitespace handling lost info: emitted {emitted}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T16 — Type invariant: crops is ALWAYS list[dict], never dict (singleton)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "scenario",
    [
        {"use_cases": ["yield_forecast"]},  # no cultivar_ids
        {"use_cases": ["yield_forecast"], "cultivar_ids": ["IT94K-1"]},  # 1
        {"use_cases": ["yield_forecast"], "cultivar_ids": ["a", "b"]},  # 2
        {"use_cases": ["yield_forecast"], "cultivar_ids": ["a", "b", "c"]},  # 3
        {"use_cases": ["climate_scenarios"]},  # non-UC1 path
        {"use_cases": []},  # no UCs at all
    ],
)
def test_t16_crops_is_list_dict_never_dict(scenario, package_dir):
    """§2.1.1 row 4: ``crops`` MUST be ``list[dict]`` for ALL scenarios
    (never reverts to legacy singleton dict shape under any input).
    """
    cfg = build_project_config(**scenario)
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    assert "crops" in m, f"crops key missing for scenario {scenario!r}"
    assert isinstance(m["crops"], list), (
        f"scenario {scenario!r}: crops must be list, got "
        f"{type(m['crops']).__name__}: {m['crops']!r}"
    )
    for i, c in enumerate(m["crops"]):
        assert isinstance(c, dict), (
            f"scenario {scenario!r}: crops[{i}] must be dict, got "
            f"{type(c).__name__}"
        )
