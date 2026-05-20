"""PYTHIA translator UC5 trigger-handoff — behavioral tests.

The PYTHIA translator now sets ``_acea_uc5_p_k_silent_no_op_triggered:
True`` in ``additional_metadata`` before calling ``create_manifest(
platform="pythia")``, mirroring the ACEA pattern. Without this trigger,
the joint advisory_flag ``pythia_pk_silent_no_op:fertility_stress_
unmodeled_v3.1`` (manifest.py emit gate at ``packaging/manifest.py:644-
650``) never landed on real PYTHIA packages in production; the flag was
spec'd but unreachable.

The deck splits into two tiers:

- **Real-translator path** (the codex Gate B SH-1 closure): instantiates
  ``PythiaTranslator`` and drives the real ``_generate_manifest()``
  method with the heavy file-generation parts monkeypatched out, then
  captures the call to ``create_manifest`` and asserts the
  ``additional_metadata`` kwarg carries the trigger. This is the
  production-code-path coverage.
- **Emitter-level** behavioural tests: mirror the exact ``project_config``
  + ``additional_metadata`` shape that ``PythiaTranslator`` builds, then
  drive the ``create_manifest`` emitter directly. Verify the joint flag
  lands on the ``soil_fertility`` ``uc_readiness`` entry, that the
  counterfactual (no ``soil_fertility`` in ``use_case_config``) produces
  no flag, that the sibling UCs do NOT carry the flag, and that the
  ACEA-platform gate keeps the disclosure PYTHIA-only.

The deck is self-contained — no relative ``conftest`` imports, no
shared test packaging — so a clean ``git archive HEAD`` checkout (what
CI sees) can collect and run it. The previous fold relied on an
untracked ``tests/phase3/conftest.py`` for the flag constant + the
``package_dir`` fixture; both are now local.

AST coverage for the wiring lives in the sibling
``test_pythia_trigger_setter_invariants_ast.py`` file.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from prismpy.packaging.manifest import (
    ADVISORY_FLAG_UC5_PYTHIA_PK_SILENT_NO_OP as ADVISORY_UC5_PYTHIA_PK,
    create_manifest,
)


# ── Local fixtures (no conftest dependency) ─────────────────────────────────


@pytest.fixture
def package_dir(tmp_path: Path) -> Path:
    """Minimal package directory: one tiny file so ``create_manifest``'s
    ``collect_files_with_checksums`` produces ≥ 1 entry. Specific
    contents don't affect ``uc_readiness`` / ``use_case_config`` /
    ``crops`` emit clauses (those derive from ``project_config`` only).

    Inlined locally so the test file does not depend on the untracked
    ``tests/phase3/conftest.py`` — clean ``git archive HEAD`` checkout
    can collect and run this file as-is.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "metadata.json").write_text(json.dumps({"placeholder": True}))
    return pkg


def _pythia_project_config(
    *,
    include_soil_fertility: bool = True,
    include_other_ucs: bool = True,
    include_livestock_feed: bool = True,
) -> dict[str, Any]:
    """Build a project_config matching what ``PythiaTranslator`` emits
    at ``translators/pythia/translator.py:3055-3078``.

    ``soil_fertility`` is the UC the disclosure targets; the other UCs
    drive the sibling-sweep matrix. ``include_livestock_feed`` covers
    UC6 (Gate B SH-2 closure) so the matrix is now UC1 / UC3 / UC4 / UC6.
    The counterfactual test omits ``soil_fertility`` to prove the
    downstream emit gate filters per UC name.
    """
    use_case_config: dict[str, dict[str, Any]] = {}
    if include_other_ucs:
        use_case_config["yield_forecast"] = {}
        use_case_config["sowing_optimization"] = {}
        use_case_config["drought_management"] = {}
    if include_livestock_feed:
        use_case_config["livestock_feed"] = {}
    if include_soil_fertility:
        use_case_config["soil_fertility"] = {}
    return {
        "project_name": "phase3_pythia_uc5_trigger_handoff",
        "region_name": "Koutiala",
        "country": "Mali",
        "gadm_level": 2,
        "crop_name": "sorghum",
        "planting_doy": 175,
        "maturity_doy": 305,
        "start_year": 2018,
        "end_year": 2022,
        "spinup_years": 0,
        "data_sources": {
            "climate": "NASA POWER",
            "soil": "eGHR",
            "crop_mask": "SPAM 2020",
            "boundaries": "GADM 4.1",
        },
        "use_case_config": use_case_config,
    }


# ── Real-translator-path test (Gate B SH-1 closure) ─────────────────────────


def test_pythia_translator_generate_manifest_real_path(tmp_path, monkeypatch):
    """Drive the REAL ``PythiaTranslator._generate_manifest`` method
    end-to-end with the heavy file-generation parts monkeypatched out,
    and capture the actual ``create_manifest`` call to assert
    ``additional_metadata`` carries the trigger. This is the
    production-code coverage (the other tests below are emitter-level
    — they reconstruct the project_config rather than driving the real
    translator method).

    Closes the cycle-4 Gate B SH-1 (the prior fold's "production"
    test was actually emitter-level — it built a mirrored
    project_config and called create_manifest directly, never
    exercising the translator method).
    """
    from prismpy.translators.pythia.translator import PythiaTranslator

    captured: dict[str, Any] = {}

    def fake_create_manifest(*, package_dir, project_config, platform,
                             scenario=None, additional_metadata=None,
                             **extra):
        captured["package_dir"] = package_dir
        captured["project_config"] = project_config
        captured["platform"] = platform
        captured["scenario"] = scenario
        captured["additional_metadata"] = additional_metadata
        return {"_captured": True, "uc_readiness": {}}

    # Monkeypatch the emitter + the disk-writer + the boundary helper
    # at their canonical module paths so the local ``from ... import``
    # inside ``_generate_manifest`` picks up the test doubles.
    monkeypatch.setattr(
        "prismpy.packaging.manifest.create_manifest", fake_create_manifest,
    )
    monkeypatch.setattr(
        "prismpy.packaging.manifest.save_manifest",
        lambda manifest, path: path,
    )
    monkeypatch.setattr(
        "prismpy.packaging.manifest.derive_boundary_label",
        lambda source, gadm_level: ("gadm_l2", "ok"),
    )
    monkeypatch.setattr(
        "prismpy.packaging.scenario_helpers."
        "build_baseline_scenario_block_for_period",
        lambda **kw: None,
    )

    # Minimal config + data shape: ``_generate_manifest`` only reads a
    # handful of attributes off ``self.config`` and ``data.region``.
    config = SimpleNamespace(
        project=SimpleNamespace(name="real-path-test"),
        crop=SimpleNamespace(
            name="sorghum",
            calendar=SimpleNamespace(planting_doy=175, maturity_doy=305),
        ),
        temporal=SimpleNamespace(
            start_year=2018, end_year=2022, spinup_years=0,
        ),
        region=SimpleNamespace(
            boundary=SimpleNamespace(
                source=SimpleNamespace(value="gadm"),
                gadm_level=2,
            ),
        ),
    )
    data = SimpleNamespace(
        region=SimpleNamespace(
            name="Koutiala", country="Mali", boundary_source="gadm",
        ),
    )

    # Bypass __init__ — _generate_manifest only touches self.config +
    # self.output_dir, so a manually-constructed instance suffices.
    translator = PythiaTranslator.__new__(PythiaTranslator)
    translator.config = config
    translator.output_dir = tmp_path

    translator._generate_manifest(data)

    assert captured.get("platform") == "pythia", (
        f"PythiaTranslator must call create_manifest with platform='pythia'; "
        f"got {captured.get('platform')!r}"
    )
    additional = captured.get("additional_metadata") or {}
    assert additional.get("_acea_uc5_p_k_silent_no_op_triggered") is True, (
        f"PythiaTranslator._generate_manifest must pass "
        f"additional_metadata['_acea_uc5_p_k_silent_no_op_triggered']=True; "
        f"got {additional!r}"
    )
    # Sibling check: the project_config the translator built carries
    # the soil_fertility entry unconditionally (the downstream gate
    # filters per UC; only soil_fertility carries the disclosure).
    use_case_cfg = (
        captured.get("project_config", {}).get("use_case_config", {})
    )
    assert "soil_fertility" in use_case_cfg, (
        f"PythiaTranslator must include soil_fertility unconditionally "
        f"in use_case_config; got keys {sorted(use_case_cfg)}"
    )


# ── Emitter-level: mirror the production project_config + metadata ──────────


def test_pythia_translator_emits_uc5_joint_flag_emitter_level(
    package_dir,
) -> None:
    """Emitter-level coverage: mirror the exact ``project_config`` +
    ``additional_metadata`` shape ``PythiaTranslator`` builds, then
    drive ``create_manifest`` directly. Asserts the joint flag lands
    on ``uc_readiness.soil_fertility.advisory_flags``.

    This is the emit-side surface, complementary to the real-translator
    path test above (which proves the translator passes the right
    additional_metadata) — together they cover the full chain from
    translator → emitter → manifest.
    """
    cfg = _pythia_project_config()
    additional_metadata = {"_acea_uc5_p_k_silent_no_op_triggered": True}
    m = create_manifest(
        package_dir, cfg, platform="pythia",
        additional_metadata=additional_metadata,
    )
    flags = m["uc_readiness"]["soil_fertility"]["advisory_flags"]
    assert ADVISORY_UC5_PYTHIA_PK in flags, (
        f"PYTHIA UC5 packages must emit the joint P+K silent-no-op flag "
        f"via the new trigger handoff; got flags={flags}"
    )


# ── T-counterfactual: emit gate filters when soil_fertility absent ──────────


def test_pythia_trigger_does_not_emit_when_soil_fertility_omitted(
    package_dir,
) -> None:
    """Trigger set but ``use_case_config`` omits ``soil_fertility`` →
    the joint flag does NOT emit. Proves the downstream emit gate at
    ``manifest.py:644-650`` filters per UC name so the unconditional
    trigger setter cannot leak the flag onto packages that were not
    built for UC5."""
    cfg = _pythia_project_config(include_soil_fertility=False)
    m = create_manifest(
        package_dir, cfg, platform="pythia",
        additional_metadata={"_acea_uc5_p_k_silent_no_op_triggered": True},
    )
    uc_readiness = m.get("uc_readiness", {})
    assert "soil_fertility" not in uc_readiness, (
        f"closed-world violation: soil_fertility appeared in uc_readiness "
        f"despite being omitted from use_case_config; entries: "
        f"{sorted(uc_readiness)}"
    )
    for uc_name, uc_entry in uc_readiness.items():
        flags = (
            uc_entry.get("advisory_flags", [])
            if isinstance(uc_entry, dict) else []
        )
        assert ADVISORY_UC5_PYTHIA_PK not in flags, (
            f"sibling-sweep violation: joint flag leaked onto {uc_name!r} "
            f"advisory_flags={flags}"
        )


# ── T-sibling-sweep: flag ONLY on soil_fertility, not on UC1/UC3/UC4/UC6 ────


def test_pythia_joint_flag_emits_only_on_soil_fertility_uc(
    package_dir,
) -> None:
    """A PYTHIA package built for all 5 UCs (yield_forecast +
    sowing_optimization + drought_management + livestock_feed +
    soil_fertility) emits the joint flag on the ``soil_fertility``
    entry ONLY; UC1 / UC3 / UC4 / UC6 entries do NOT carry it (the emit
    gate keys on ``uc_name == 'soil_fertility'``). UC6 added per Gate B
    SH-2 closure (prior fold only covered UC1 / UC3 / UC4).
    """
    cfg = _pythia_project_config()
    m = create_manifest(
        package_dir, cfg, platform="pythia",
        additional_metadata={"_acea_uc5_p_k_silent_no_op_triggered": True},
    )
    uc_readiness = m["uc_readiness"]
    sf_flags = uc_readiness["soil_fertility"]["advisory_flags"]
    assert ADVISORY_UC5_PYTHIA_PK in sf_flags
    for uc_name in (
        "yield_forecast",
        "sowing_optimization",
        "drought_management",
        "livestock_feed",
    ):
        flags = uc_readiness[uc_name].get("advisory_flags", [])
        assert ADVISORY_UC5_PYTHIA_PK not in flags, (
            f"sibling-sweep violation: joint flag leaked from "
            f"soil_fertility to {uc_name!r} advisory_flags={flags}"
        )


# ── T-acea-non-leak: ACEA-platform manifests still gated out ────────────────


def test_acea_platform_does_not_emit_joint_flag_even_with_trigger(
    package_dir,
) -> None:
    """ACEA-target packages (``platform='acea'``) carry the trigger in
    ``additional_metadata`` (from the ACEA translator's own pattern at
    ``acea/translator.py:2408-2417``) but never emit the joint flag —
    the manifest.py emit gate at ``:644-650`` requires
    ``platform == 'pythia'`` so ACEA-platform packages are filtered
    out. Verifies the cycle-4 prismpy fix does NOT collaterally flip
    ACEA into emitting the PYTHIA-only disclosure.
    """
    cfg = _pythia_project_config()
    m = create_manifest(
        package_dir, cfg, platform="acea",
        additional_metadata={"_acea_uc5_p_k_silent_no_op_triggered": True},
    )
    sf_entry = m["uc_readiness"].get("soil_fertility", {})
    flags = sf_entry.get("advisory_flags", [])
    assert ADVISORY_UC5_PYTHIA_PK not in flags, (
        f"ACEA-platform manifests must NOT emit the PYTHIA-only P+K "
        f"flag even when the trigger is set; got flags={flags}"
    )
