"""§7 generator — n_response_skill (UC7) registration + gates + trial-CSV copy.

Proves a modeler can generate a package that serves n_response_skill (the model-
skill post-processor on soil_fertility) for ANY region/crop/engine with zero code
change (AC-7 Part-1). Covers the 6 manifest surfaces, the 3 readiness gates, the
uc_readiness emit across platforms, and the shared fail-loud trial-CSV copy helper.

Self-contained: builds minimal project_configs with a NON-empty ``data_sources``
so create_manifest's required-at-creation check is satisfied (independent of the
shared conftest builder).
"""

from __future__ import annotations

import logging
import types
from pathlib import Path

import pytest

from prismpy.packaging.manifest import (
    ADVISORY_GATES,
    KNOWN_USE_CASE_NAMES,
    PER_UC_GATES,
    UC_CONFIG_KEY_TABLE,
    _PACKAGEABLE_UCS,
    _UC_SUPPORTED_PLATFORMS,
    canonical_uc_readiness_emitter,
    create_manifest,
    use_case_config_for,
)
from prismpy.translators.base import BaseTranslator

_UC = "n_response_skill"


# ── (a) the 6 registration surfaces ──────────────────────────────────────────


def test_uc7_registered_in_all_six_surfaces():
    assert _UC in KNOWN_USE_CASE_NAMES
    assert _UC in _PACKAGEABLE_UCS
    assert PER_UC_GATES[_UC] == frozenset({
        "n_trials_present",
        "soil_fertility_dependency_declared",
        "platform_dssat_family",
    })
    assert _UC_SUPPORTED_PLATFORMS[_UC] == frozenset({"pythia", "acea", "craft"})
    assert UC_CONFIG_KEY_TABLE[_UC] == (
        "cores", "observed_trials", "skill_calibrated",
        "trial_coord_tol_deg", "full_curve_levels",
    )
    assert "platform_dssat_family" in ADVISORY_GATES  # acea reduced-coverage advisory


@pytest.mark.parametrize("platform,present", [
    ("pythia", True), ("acea", True), ("craft", True), ("sarra_py", False),
])
def test_uc7_auto_declared_only_on_supported_platforms(platform, present):
    # A post-processor is packageable (mirrors drought_management) → auto-declared
    # on the platforms that support it; sarra_py (no soil_fertility DSSAT path) omits it.
    assert (_UC in use_case_config_for(platform)) is present


# ── (b) the uc_readiness emit across platforms (honest per-state) ─────────────


def _emit(platform, *, trials):
    pc = {"use_case_config": use_case_config_for(platform), "_n_trials_present": trials}
    return canonical_uc_readiness_emitter(pc, platform, {}).get(_UC)


def test_uc7_pythia_with_trials_all_gates_pass_ready():
    e = _emit("pythia", trials=True)
    assert set(e["gates_passed"]) == {
        "n_trials_present", "soil_fertility_dependency_declared", "platform_dssat_family",
    }
    assert not e.get("gates_failed")


def test_uc7_pythia_without_trials_n_trials_present_hard_fails_honestly():
    e = _emit("pythia", trials=False)
    failed = {g["gate_id"]: g for g in e.get("gates_failed", [])}
    assert "n_trials_present" in failed and failed["n_trials_present"]["severity"] == "hard"
    # the substrate + engine gates still pass — only the trials gate is not-ready.
    assert "soil_fertility_dependency_declared" in e["gates_passed"]


def test_uc7_acea_surfaces_platform_dssat_family_advisory_not_a_block():
    e = _emit("acea", trials=True)
    assert not e.get("gates_failed")                      # advisory is NOT a hard block
    assert any("platform_dssat_family" in f for f in e["advisory_flags"])
    assert "n_trials_present" in e["gates_passed"]        # trials still scored


def test_uc7_sarra_py_absent_from_readiness_closed_world():
    # sarra_py doesn't support n_response_skill → not declared → absent (not empty).
    pc = {"use_case_config": use_case_config_for("sarra_py"), "_n_trials_present": True}
    assert _UC not in canonical_uc_readiness_emitter(pc, "sarra_py", {})


# ── (c) create_manifest injects the DEST-based _n_trials_present ──────────────


def _cfg(platform):
    return {
        "project_name": "nrisk_uc7_test", "region_name": "Kano", "country": "Nigeria",
        "gadm_level": 2, "crop_name": "sorghum", "planting_doy": 150, "maturity_doy": 280,
        "start_year": 2010, "end_year": 2019, "spinup_years": 0,
        "data_sources": {"climate": "NASA POWER"},   # non-empty (required-at-creation)
        "use_case_config": use_case_config_for(platform),
    }


def _pkg(tmp_path, name, with_trials):
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "metadata.json").write_text("{}")
    if with_trials:
        (pkg / "data").mkdir()
        (pkg / "data" / "n_trials.csv").write_text(
            "cell_id,year,scenario_label,n_level_kg_ha\nc1,2016,N60,60\n"
        )
    return pkg


def test_create_manifest_gate_reads_the_actual_copied_trials_artifact(tmp_path):
    cfg = _cfg("pythia")
    with_ = create_manifest(_pkg(tmp_path, "with", True), cfg, platform="pythia")
    without = create_manifest(_pkg(tmp_path, "without", False), cfg, platform="pythia")
    nr_with = with_["uc_readiness"][_UC]
    nr_without = without["uc_readiness"][_UC]
    # dest-present → gate PASSES (READY); dest-absent → HARD-fails (honest not-ready).
    assert "n_trials_present" in nr_with["gates_passed"]
    assert any(g["gate_id"] == "n_trials_present" for g in nr_without.get("gates_failed", []))


# ── (d) the shared fail-loud trial-CSV copy helper ───────────────────────────


def _translator_stub(src, out):
    o = types.SimpleNamespace()
    o.config = types.SimpleNamespace(n_trials_source_path=src)
    o.output_dir = Path(out)
    o.logger = logging.getLogger("nrisk_test")
    return o


def test_copy_observed_trials_happy_writes_data_n_trials_csv(tmp_path):
    src = tmp_path / "my_trials.csv"
    src.write_text("cell_id,year,scenario_label\nc1,2016,N60\n")
    out = tmp_path / "pkg"
    out.mkdir()
    dest = BaseTranslator._copy_observed_trials(_translator_stub(str(src), out))
    assert dest == out / "data" / "n_trials.csv"
    assert dest.read_text() == src.read_text()          # copied to the convention path


def test_copy_observed_trials_none_when_not_supplied(tmp_path):
    out = tmp_path / "pkg"
    out.mkdir()
    assert BaseTranslator._copy_observed_trials(_translator_stub(None, out)) is None
    assert not (out / "data").exists()                   # no data/ dir when no trials


def test_copy_observed_trials_fails_loud_on_missing_source(tmp_path):
    out = tmp_path / "pkg"
    out.mkdir()
    missing = tmp_path / "does_not_exist.csv"
    # supplied-but-uncopyable → RAISES (never a silent trials-absent package).
    with pytest.raises(FileNotFoundError):
        BaseTranslator._copy_observed_trials(_translator_stub(str(missing), out))
