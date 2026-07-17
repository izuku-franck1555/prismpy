"""§7 generator — n_response_skill (UC7) registration + gates + trial-CSV copy.

Proves a modeler can generate a package that serves n_response_skill (the model-
skill post-processor on soil_fertility) for ANY region/crop/engine with zero code
change. Covers the 6 manifest surfaces, the 3 readiness gates, the uc_readiness
emit across platforms, and the shared fail-loud trial-CSV copy helper.

Self-contained: builds minimal project_configs with a NON-empty ``data_sources``
so create_manifest's required-at-creation check is satisfied (independent of the
shared conftest builder).
"""

from __future__ import annotations

import logging
import os
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
from prismpy.translators.base import BaseTranslator, ObservedTrialsCopyError
from prismpy.config.schema import Platform

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
    # bind the REAL fail-closed remover so the stub exercises the real helper
    o._remove_stale_trials_dest = BaseTranslator._remove_stale_trials_dest.__get__(o)
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
    # supplied-but-uncopyable → RAISES the typed error (never a silent trials-absent
    # package; the typed class lets the PACKAGE stage treat it as FATAL).
    with pytest.raises(ObservedTrialsCopyError):
        BaseTranslator._copy_observed_trials(_translator_stub(str(missing), out))


def test_copy_observed_trials_wraps_os_error_as_typed(tmp_path):
    # Root-cause: ALL failures placing the trials (mkdir / copy) surface as the
    # typed ObservedTrialsCopyError so the executor can route them fatally — not
    # only the missing-source case.
    src = tmp_path / "trials.csv"
    src.write_text("cell_id,year\nc1,2016\n")
    out = tmp_path / "pkg"
    out.mkdir()
    (out / "data").write_text("a file sits where the data/ dir must go")  # mkdir -> OSError
    with pytest.raises(ObservedTrialsCopyError):
        BaseTranslator._copy_observed_trials(_translator_stub(str(src), out))


# ── fail-loud + honest-signal edges (trials-copy fatal; no-source reconcile) ──


class _TrialsOnlyTranslator(BaseTranslator):
    """A real BaseTranslator subclass whose generate_package invokes the REAL
    _copy_observed_trials (the code under test) — so the fatal trials-copy path is
    exercised end-to-end through the executor, not faked by a stub that pre-raises."""

    PLATFORM = Platform.PYTHIA

    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger("nrisk_realish_translator")

    def translate(self, data):            # abstract stub — never called here
        raise NotImplementedError

    def validate_outputs(self):           # abstract stub — never called here
        return []

    def get_required_data(self):          # abstract stub — never called here
        return ["region"]

    def generate_package(self, unified_data, output_files):
        trials = self._copy_observed_trials()   # REAL helper — the real flow
        return [trials] if trials else []


def _run_package_stage(tmp_path, translator):
    """Drive the REAL executor PACKAGE stage over ``translator`` and return the
    PACKAGE StageResult, so a test asserts the package-level success flag (not just
    the helper raise)."""
    from prismpy.pipeline.executor import TranslationPipeline
    from prismpy.translators.base import TranslationResult

    class _Prov:
        session_id = "t"

        def save(self, output_path=None):
            p = Path(output_path)
            p.write_text("{}")
            return p

        def get_report(self):
            return ""

    pipe = TranslationPipeline.__new__(TranslationPipeline)
    pipe.logger = logging.getLogger("nrisk_pkg_stage")
    pipe._translators = {Platform.PYTHIA: translator}
    pipe.config = types.SimpleNamespace(
        output=types.SimpleNamespace(base_dir=str(tmp_path))
    )
    pipe.provenance = _Prov()
    tr = TranslationResult(
        success=True, platform=Platform.PYTHIA, output_dir=tmp_path,
        output_files=[], errors=[], warnings=[], metadata={},
    )
    return pipe._execute_package(
        unified_data=types.SimpleNamespace(grid=None, metadata=None),
        translation_results={"pythia": tr},
    )


def test_uncopyable_trials_make_package_fail_not_silent_success(tmp_path):
    # A supplied-but-uncopyable n_trials_source_path must land in PACKAGE errors
    # (success=False), NOT a swallowed warning. Drives a REAL BaseTranslator subclass
    # (its generate_package calls the real _copy_observed_trials) through the real
    # executor PACKAGE stage — asserts the PACKAGE flag, matching the real flow.
    out = tmp_path / "pkg"
    out.mkdir()
    cfg = types.SimpleNamespace(n_trials_source_path=str(tmp_path / "missing.csv"))
    result = _run_package_stage(tmp_path, _TrialsOnlyTranslator(cfg, out))
    assert any("trials" in e.lower() for e in result.errors), (
        f"trials-copy failure must be recorded in PACKAGE errors; got {result.errors}"
    )
    assert result.success is False


def test_unlink_failure_on_no_source_makes_package_fail(tmp_path, monkeypatch):
    # No-source rebuild in a reused dir: if removing the stale dest FAILS, the helper
    # must fail CLOSED (typed → PACKAGE success=False), not leave the stale artifact
    # and report success. Pre-fix: a bare unwrapped unlink lets a raw OSError escape
    # → generic executor catch → warning → success=True.
    out = tmp_path / "pkg"
    (out / "data").mkdir(parents=True)
    (out / "data" / "n_trials.csv").write_text("stale trials from a prior build\n")
    cfg = types.SimpleNamespace(n_trials_source_path=None)     # no source this build
    real_unlink = Path.unlink

    def _guarded_unlink(self, *a, **k):
        if self.name == "n_trials.csv":
            raise PermissionError("cannot unlink n_trials.csv")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", _guarded_unlink)
    result = _run_package_stage(tmp_path, _TrialsOnlyTranslator(cfg, out))
    assert result.success is False, (
        f"unlink-failure must fail PACKAGE (fail-closed); errors={result.errors}"
    )
    assert any("trials" in e.lower() for e in result.errors)


def test_directory_dest_makes_package_fail(tmp_path):
    # A directory sitting at the EXACT dest path: shutil.copy2 would copy the source
    # BENEATH it (dest never becomes a regular file), silently. The helper must
    # reject it (typed → PACKAGE success=False). Pre-fix: copy2-into-dir, no raise,
    # success=True.
    out = tmp_path / "pkg"
    (out / "data" / "n_trials.csv").mkdir(parents=True)        # dir where the CSV must go
    src = tmp_path / "trials.csv"
    src.write_text("cell_id,year\nc1,2016\n")
    cfg = types.SimpleNamespace(n_trials_source_path=str(src))
    result = _run_package_stage(tmp_path, _TrialsOnlyTranslator(cfg, out))
    assert result.success is False, (
        f"directory-dest must fail PACKAGE; errors={result.errors}"
    )
    assert any("trials" in e.lower() for e in result.errors)


def test_probe_permission_error_makes_package_fail(tmp_path, monkeypatch):
    # A metadata probe that raises EACCES/EIO (Python's is_file re-raises non-not-
    # found OSErrors) must be caught by the outer fail-closed boundary (typed →
    # PACKAGE success=False), not escape raw → silent success. Pre-fix: the probe
    # re-raise escaped the per-op wraps → generic executor catch → warning → True.
    out = tmp_path / "pkg"
    (out / "data").mkdir(parents=True)
    (out / "data" / "n_trials.csv").write_text("stale\n")   # pre-existing dest → is_file() probed
    src = tmp_path / "trials.csv"
    src.write_text("cell_id,year\nc1,2016\n")
    cfg = types.SimpleNamespace(n_trials_source_path=str(src))
    real_is_file = Path.is_file

    def _boom_is_file(self, *a, **k):
        if self.name == "n_trials.csv":
            raise PermissionError("stat denied probing the trials dest")
        return real_is_file(self, *a, **k)

    monkeypatch.setattr(Path, "is_file", _boom_is_file)
    result = _run_package_stage(tmp_path, _TrialsOnlyTranslator(cfg, out))
    assert result.success is False, (
        f"a probe OSError must fail PACKAGE (outer boundary); errors={result.errors}"
    )
    assert any("trials" in e.lower() or "placement" in e.lower() for e in result.errors)


def test_symlinked_data_parent_rejected_and_external_not_deleted(tmp_path):
    # A symlinked data/ parent (data/ -> /external) must be REJECTED (typed →
    # PACKAGE success=False) so trials never write off-package, AND the no-source
    # reconcile must NOT delete the external target through the symlink. Pre-fix:
    # the reconcile unlinked the external artifact AND reported success=True.
    external = tmp_path / "external"
    external.mkdir()
    external_artifact = external / "n_trials.csv"
    external_artifact.write_text("PRECIOUS external data - must NOT be deleted\n")
    out = tmp_path / "pkg"
    out.mkdir()
    (out / "data").symlink_to(external, target_is_directory=True)   # data/ -> /external
    cfg = types.SimpleNamespace(n_trials_source_path=None)          # no-source build
    result = _run_package_stage(tmp_path, _TrialsOnlyTranslator(cfg, out))
    assert result.success is False, (
        f"symlinked data/ parent must fail PACKAGE; errors={result.errors}"
    )
    assert any(
        "contain" in e.lower() or "off-package" in e.lower() or "trials" in e.lower()
        for e in result.errors
    )
    # the external target + its artifact survive — no destructive delete via symlink.
    assert external.is_dir()
    assert external_artifact.is_file(), "reconcile must NOT delete the external target"


def test_symlink_loop_at_dest_makes_package_fail(tmp_path):
    # A symlink loop on the dest path makes Path.resolve() raise RuntimeError (NOT
    # OSError) on py3.11 — only a catch-all boundary types it → PACKAGE
    # success=False, rather than escaping raw (past an OSError-only handler) to a
    # silent success.
    out = tmp_path / "pkg"
    (out / "data").mkdir(parents=True)
    dest = out / "data" / "n_trials.csv"
    dest.symlink_to(dest)                      # self-referential symlink loop
    cfg = types.SimpleNamespace(n_trials_source_path=None)
    result = _run_package_stage(tmp_path, _TrialsOnlyTranslator(cfg, out))
    assert result.success is False, (
        f"a resolve() symlink loop must fail PACKAGE (catch-all); errors={result.errors}"
    )
    assert any("trials" in e.lower() or "placement" in e.lower() for e in result.errors)


def test_hardlinked_stale_dest_not_mutated_by_atomic_write(tmp_path):
    # A stale dest hard-linked to an external file shares its inode; a naive
    # copy2(src, dest) would write THROUGH the shared inode → mutate the external
    # file. The atomic write (copy to a temp + os.replace) breaks the link → the
    # external content is UNCHANGED, while the package still receives fresh trials.
    external = tmp_path / "external.csv"
    external.write_text("EXTERNAL CONTENT - must stay unchanged\n")
    out = tmp_path / "pkg"
    (out / "data").mkdir(parents=True)
    dest = out / "data" / "n_trials.csv"
    os.link(external, dest)                     # dest hard-linked to external (shared inode)
    src = tmp_path / "trials.csv"
    src.write_text("cell_id,year\nc1,2016\n")
    cfg = types.SimpleNamespace(n_trials_source_path=str(src))
    result = _run_package_stage(tmp_path, _TrialsOnlyTranslator(cfg, out))
    assert result.success is True, (
        f"a valid copy over a hard-linked stale dest should succeed; errors={result.errors}"
    )
    assert dest.read_text() == src.read_text()          # the package got the fresh trials
    assert external.read_text() == "EXTERNAL CONTENT - must stay unchanged\n"  # external UNCHANGED


def test_stale_trials_dest_reconciled_on_no_source_rebuild(tmp_path):
    # A reused output dir that held trials from a prior build must NOT keep reading
    # trials-present after a no-source rebuild.
    src = tmp_path / "trials.csv"
    src.write_text("cell_id,year,scenario_label,n_level_kg_ha\nc1,2016,N60,60\n")
    out = _pkg(tmp_path, "pkg", with_trials=False)  # metadata.json, no data/
    cfg = _cfg("pythia")
    # Build 1: trials supplied → copied in → gate READY.
    dest = BaseTranslator._copy_observed_trials(_translator_stub(str(src), out))
    assert dest.is_file()
    m1 = create_manifest(out, cfg, platform="pythia")
    assert "n_trials_present" in m1["uc_readiness"][_UC]["gates_passed"]
    # Build 2: SAME dir, no source → reconcile must drop the stale artifact.
    assert BaseTranslator._copy_observed_trials(_translator_stub(None, out)) is None
    assert not dest.is_file()
    m2 = create_manifest(out, cfg, platform="pythia")
    assert any(
        g["gate_id"] == "n_trials_present"
        for g in m2["uc_readiness"][_UC].get("gates_failed", [])
    ), "stale trials dest must be reconciled → n_trials_present HARD-fails"


def test_soil_fertility_dependency_gate_hard_fails_when_substrate_undeclared():
    # A customized UC set that declares n_response_skill but DROPS its soil_fertility
    # substrate → the HARD dependency gate must fail (honest not-ready). Pins the
    # failing branch of the substrate gate.
    uc_cfg = dict(use_case_config_for("pythia"))
    uc_cfg.pop("soil_fertility", None)                 # drop the substrate
    assert _UC in uc_cfg                                # n_response_skill still declared
    pc = {"use_case_config": uc_cfg, "_n_trials_present": True}
    e = canonical_uc_readiness_emitter(pc, "pythia", {})[_UC]
    failed = {g["gate_id"] for g in e.get("gates_failed", [])}
    assert "soil_fertility_dependency_declared" in failed
    assert "n_trials_present" in e["gates_passed"]      # only the substrate gate fails
