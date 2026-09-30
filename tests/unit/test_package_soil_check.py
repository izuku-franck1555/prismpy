"""The package ``TranslationPipeline.execute`` returns as successful has been verified after its
last write: a soil file or manifest changed at any point before the return fails the PACKAGE
stage with a named error, once before the package callback and once at the return boundary."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from prismpy.config.schema import Platform
from prismpy.packaging import soil_declaration as sd
from prismpy.packaging.manifest import create_manifest
from prismpy.pipeline.executor import PipelineStage, StageResult, TranslationPipeline
from prismpy.translators.base import TranslationResult
from tests.package_soil import stamp_package_soil
from tests.unit.test_package_soil_values import _acea, _isda_s3
from tests.unit.test_pythia_canonical_substrate_flag import _build_project_config

NAMED = "soil declaration check failed on the final package"
EXECUTOR = Path(__file__).resolve().parents[2] / "src/prismpy/pipeline/executor.py"
CFG = {"project_name": "p", "region_name": "Koutiala", "country": "Mali", "crop_name": "Maize",
       "start_year": 2015, "end_year": 2016, "data_sources": {"climate": "NASA POWER"}}


def _soil(pkg, platform):
    """The soil files ``platform`` reads, written and bound by the production writers."""
    if platform == "acea":
        _acea(pkg)._generate_soil_netcdf_from_profiles({1: _isda_s3(1)}, [1000])
    else:
        stamp_package_soil(pkg, platform)
    return {"craft": pkg / "soil/ML.SOL", "pythia": pkg / "raster/soil.tif",
            "acea": pkg / sd.ACEA_SOIL_FILE}[platform]


class _Callback:
    def __init__(self, on_package=None):
        self.on_package, self.seen = on_package, []

    def on_stage_start(self, stage, description):
        pass

    def on_stage_complete(self, stage, result):
        self.seen.append((stage, result.success, list(result.errors)))
        if stage == "package" and self.on_package:
            self.on_package()


def _run(tmp_path, monkeypatch, platform, *, after_translate=None, inspect=None, pipe=None, callback=None):
    """Execute TRANSLATE + PACKAGE for one engine whose files the real writers produce."""
    pkg = tmp_path / platform
    pipe = pipe or TranslationPipeline(_build_project_config(tmp_path))

    def translate(unified):
        pkg.mkdir(parents=True, exist_ok=True)
        bound = _soil(pkg, platform)
        if after_translate:
            after_translate(bound)
        return {platform: TranslationResult(success=True, platform=Platform(platform), output_dir=pkg,
                                            output_files=[], errors=[], warnings=[], metadata={})}

    def package(unified, translation_results, validate_result):
        if inspect:          # an early, swallowed inspection, as the translators' metadata step does
            try:
                inspect(pkg)
            except sd.SoilDeclarationError:
                pass
        try:                 # the translators' metadata step logs a manifest failure and moves on
            (pkg / "manifest.json").write_text(json.dumps(create_manifest(pkg, CFG, platform=platform)))
        except Exception:
            pass
        return StageResult(stage=PipelineStage.PACKAGE, success=True, data={})

    monkeypatch.setattr(pipe, "_execute_translate", translate)
    monkeypatch.setattr(pipe, "_execute_package", package)
    result = pipe.execute(stages=[PipelineStage.TRANSLATE, PipelineStage.PACKAGE], progress_callback=callback)
    return result, pkg, pipe


def _corrupt(path):
    data = bytearray(Path(path).read_bytes())
    data[len(data) // 2] ^= 0xFF
    Path(path).write_bytes(bytes(data))


@pytest.mark.parametrize("platform", ["craft", "pythia", "acea"])
def test_a_soil_file_changed_before_packaging_fails_the_package_stage(tmp_path, monkeypatch, platform):
    callback = _Callback()
    result, _, _ = _run(tmp_path, monkeypatch, platform, after_translate=_corrupt, callback=callback)
    package = result.stages["package"]
    assert result.success is False and package.success is False
    assert any(NAMED in e and "changed after it was bound" in e for e in package.errors)
    # the package callback already sees the failed stage
    assert [s for s in callback.seen if s[0] == "package"] == [("package", False, package.errors)]


@pytest.mark.parametrize("platform", ["craft", "pythia", "acea"])
@pytest.mark.parametrize("where", ["callback", "finalize"])
def test_a_change_after_the_package_callback_or_finalize_fails_the_build(tmp_path, monkeypatch, platform, where):
    files = {}
    callback = _Callback(on_package=lambda: _corrupt(files["bound"])) if where == "callback" else None
    pipe = TranslationPipeline(_build_project_config(tmp_path))
    if where == "finalize":
        monkeypatch.setattr(pipe.provenance, "finalize", lambda *a, **k: _corrupt(files["bound"]))
    result, _, _ = _run(tmp_path, monkeypatch, platform, pipe=pipe, callback=callback,
                        after_translate=lambda bound: files.setdefault("bound", bound))
    assert result.success is False and result.stages["package"].success is False
    assert any(NAMED in e for e in result.stages["package"].errors)


@pytest.mark.parametrize("change", ["delete", "inputs_used", "data_sources"])
def test_the_final_check_compares_the_written_manifest(tmp_path, monkeypatch, change):
    pkg_holder = {}

    def late():
        path = pkg_holder["pkg"] / "manifest.json"
        if change == "delete":
            path.unlink()
            return
        manifest = json.loads(path.read_text())
        if change == "inputs_used":
            manifest["inputs_used"]["soil"]["label"] = "another soil"
        else:
            manifest["data_sources"]["soil"] = "another soil"
        path.write_text(json.dumps(manifest))

    result, _, _ = _run(tmp_path, monkeypatch, "craft", callback=_Callback(on_package=late),
                        after_translate=lambda bound: pkg_holder.setdefault("pkg", bound.parent.parent))
    assert result.success is False
    assert any(NAMED in e for e in result.stages["package"].errors)


def test_one_pipeline_carries_no_failure_into_its_next_build(tmp_path, monkeypatch):
    first, _, pipe = _run(tmp_path / "a", monkeypatch, "craft", after_translate=_corrupt)
    second, _, _ = _run(tmp_path / "b", monkeypatch, "craft", pipe=pipe)
    assert first.success is False and second.success is True


@pytest.mark.parametrize("platform", ["craft", "acea"])
def test_a_transient_change_restored_before_the_last_write_ships(tmp_path, monkeypatch, platform):
    """A byte changed during the early inspection and restored at once leaves a consistent package."""
    def inspect(pkg):
        target = {"craft": pkg / "soil/ML.SOL", "acea": pkg / sd.ACEA_SOIL_FILE}[platform]
        original = target.read_bytes()
        try:
            _corrupt(target)
            sd.declared_soil(pkg, platform)
        finally:
            target.write_bytes(original)
    result, _, _ = _run(tmp_path, monkeypatch, platform, inspect=inspect)
    assert result.success is True, result.stages["package"].errors


def test_a_bug_in_the_inspector_fails_the_build_loud(tmp_path, monkeypatch):
    """Only a soil declaration error is a finding about the package; any other exception is a bug."""
    def broken(package_dir, platform):
        raise TypeError("a bug in the inspector")
    monkeypatch.setattr(sd, "declared_soil", broken)
    result, _, _ = _run(tmp_path, monkeypatch, "craft")
    assert result.success is False and result.stages["execute"].errors == ["a bug in the inspector"]


def _execute_body():
    tree = ast.parse(EXECUTOR.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TranslationPipeline")
    return cls, next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "execute")


def _calls(node, name):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call) and getattr(c.func, "attr", None) == name]


def test_the_two_checks_sit_where_nothing_can_write_after_them():
    cls, execute = _execute_body()
    body = execute.body
    success = next(i for i, s in enumerate(body) if isinstance(s, ast.Assign)
                   and getattr(s.targets[0], "id", None) == "success")
    final = body[success - 1]
    # the final call: top level, the statement right before success = all(...), in no try/with
    assert _calls(final, "_package_soil_check") and isinstance(final, ast.Expr)
    assert isinstance(body[success - 2], ast.Try)       # after the stages' try/except/finally
    # the pre-notification call: the statement right before _notify_complete("package", result)
    for block in (n for n in ast.walk(execute) if hasattr(n, "body") and isinstance(n.body, list)):
        for i, stmt in enumerate(block.body):
            call = stmt.value if isinstance(stmt, ast.Expr) else None
            if (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "_notify_complete"
                    and isinstance(call.args[0], ast.Constant) and call.args[0].value == "package"):
                assert _calls(block.body[i - 1], "_package_soil_check")
                break
        else:
            continue
        break
    else:
        pytest.fail("no _notify_complete('package', ...) call")


def test_no_typed_catch_no_in_package_inspection_no_module_state():
    source = EXECUTOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    handlers = [h for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler) and h.type is not None]
    assert not any("SoilDeclarationError" in ast.unparse(h.type) for h in handlers)
    cls, _ = _execute_body()
    package = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_execute_package")
    assert not any(getattr(c.func, "id", getattr(c.func, "attr", None)) == "declared_soil"
                   for c in ast.walk(package) if isinstance(c, ast.Call))
    def mutable(value):
        return isinstance(value, (ast.List, ast.Dict, ast.Set)) or (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", getattr(value.func, "attr", "")) in
            ("list", "dict", "set", "ContextVar", "defaultdict"))
    state = [target.id for scope in (tree.body, cls.body) for n in scope
             if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None and mutable(n.value)
             for target in (n.targets if isinstance(n, ast.Assign) else [n.target])
             if isinstance(target, ast.Name) and "soil" in target.id.lower()]
    assert state == []
