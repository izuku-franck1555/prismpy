"""PYTHIA translator UC5 trigger-handoff — behavioral tests.

The PYTHIA translator now sets ``_acea_uc5_p_k_silent_no_op_triggered:
True`` in ``additional_metadata`` before calling ``create_manifest(
platform="pythia")``, mirroring the ACEA pattern. Without this trigger,
the joint advisory_flag ``pythia_pk_silent_no_op:fertility_stress_
unmodeled_v3.1`` (manifest.py emit gate at ``packaging/manifest.py:644-
650``) never landed on real PYTHIA packages in production; the flag was
spec'd but unreachable.

The tests here mirror the exact ``project_config`` + ``additional_
metadata`` shape that ``PythiaTranslator.translate()`` builds at
``translators/pythia/translator.py:3055-3132``, then drives the
manifest emitter directly. AST coverage for the wiring lives in the
sibling ``test_pythia_trigger_setter_invariants_ast.py`` file.
"""

from __future__ import annotations

from typing import Any

from prismpy.packaging.manifest import create_manifest

from .conftest import ADVISORY_UC5_PYTHIA_PK


def _pythia_project_config(
    *,
    include_soil_fertility: bool = True,
    include_other_ucs: bool = True,
) -> dict[str, Any]:
    """Build a project_config matching what PythiaTranslator emits.

    The structure mirrors translators/pythia/translator.py:3055-3078:
    PR3 closed-world ``use_case_config`` is the source of which UCs
    appear in ``uc_readiness``. ``soil_fertility`` is included by
    default (the translator emits it unconditionally for every PYTHIA
    package); the counterfactual test omits it to prove the downstream
    emit gate filters per-UC.
    """
    use_case_config: dict[str, dict[str, Any]] = {}
    if include_other_ucs:
        use_case_config["yield_forecast"] = {}
        use_case_config["sowing_optimization"] = {}
        use_case_config["drought_management"] = {}
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


# ── T-prod: production trigger handoff round-trip ───────────────────────────


def test_pythia_translator_emits_uc5_joint_flag_in_production(
    package_dir,
) -> None:
    """Mirror the exact ``additional_metadata`` shape that
    ``PythiaTranslator.translate()`` now passes to ``create_manifest``;
    assert the joint advisory_flag lands on the ``soil_fertility``
    entry of ``uc_readiness``. Closes the cycle-4 v0.1 BLOCKING-1
    production gap (the synthetic ``platform_translator='acea'``
    fixture did NOT exercise this path).
    """
    cfg = _pythia_project_config()
    additional_metadata = {"_acea_uc5_p_k_silent_no_op_triggered": True}
    m = create_manifest(
        package_dir, cfg, platform="pythia",
        additional_metadata=additional_metadata,
    )
    flags = m["uc_readiness"]["soil_fertility"]["advisory_flags"]
    assert ADVISORY_UC5_PYTHIA_PK in flags, (
        f"BLOCKING-1 regression: PYTHIA UC5 packages must emit the joint "
        f"P+K silent-no-op flag via the new trigger handoff; got "
        f"flags={flags}"
    )


# ── T-counterfactual: emit gate filters when soil_fertility absent ──────────


def test_pythia_trigger_does_not_emit_when_soil_fertility_omitted(
    package_dir,
) -> None:
    """Direct emitter-level counterfactual: trigger set but
    ``use_case_config`` omits ``soil_fertility`` → the joint flag does
    NOT emit. Proves the downstream emit gate at
    ``manifest.py:644-650`` filters per UC name (``uc_name ==
    'soil_fertility'``) so the unconditional trigger setter cannot leak
    the flag onto packages that were not built for UC5.
    """
    cfg = _pythia_project_config(include_soil_fertility=False)
    m = create_manifest(
        package_dir, cfg, platform="pythia",
        additional_metadata={"_acea_uc5_p_k_silent_no_op_triggered": True},
    )
    uc_readiness = m.get("uc_readiness", {})
    # soil_fertility is not in use_case_config, so it must not appear
    # in uc_readiness at all (closed-world per PR3).
    assert "soil_fertility" not in uc_readiness, (
        f"closed-world violation: soil_fertility appeared in uc_readiness "
        f"despite being omitted from use_case_config; entries: "
        f"{sorted(uc_readiness)}"
    )
    # Sibling-sweep: the joint flag must not leak into any OTHER UC's
    # advisory_flags either (the manifest.py gate keys on uc_name).
    for uc_name, uc_entry in uc_readiness.items():
        flags = uc_entry.get("advisory_flags", []) if isinstance(uc_entry, dict) else []
        assert ADVISORY_UC5_PYTHIA_PK not in flags, (
            f"sibling-sweep violation: joint flag leaked onto {uc_name!r} "
            f"advisory_flags={flags}"
        )


# ── T-sibling-sweep: flag ONLY on soil_fertility, not on UC1/UC3/UC4 ────────


def test_pythia_joint_flag_emits_only_on_soil_fertility_uc(
    package_dir,
) -> None:
    """A PYTHIA package built for ALL 4 UCs (yield_forecast +
    sowing_optimization + drought_management + soil_fertility) emits
    the joint flag on the ``soil_fertility`` entry ONLY; the other UCs'
    ``advisory_flags`` lists do NOT carry it (the emit gate keys on
    ``uc_name == 'soil_fertility'``).
    """
    cfg = _pythia_project_config()
    m = create_manifest(
        package_dir, cfg, platform="pythia",
        additional_metadata={"_acea_uc5_p_k_silent_no_op_triggered": True},
    )
    uc_readiness = m["uc_readiness"]
    sf_flags = uc_readiness["soil_fertility"]["advisory_flags"]
    assert ADVISORY_UC5_PYTHIA_PK in sf_flags
    for uc_name in ("yield_forecast", "sowing_optimization", "drought_management"):
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
    cfg = _pythia_project_config()  # use_case_config includes soil_fertility
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
