"""Where the crop-presence rule and its reader sit in the code: the rule between the share threshold
and the user exclusions; the harmonize catch classifying the typed errors; CRAFT's mask extraction
and ACEA's clip routed through the one reader and the one resolver; no independent SPAM sampler."""
from __future__ import annotations

import ast
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "prismpy"


def _function(relpath, name):
    text = (_SRC / relpath).read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node, text
    raise AssertionError(f"{relpath}::{name} not found")


def _calls(node):
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            names.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
    return names


def _line_of(node, predicate):
    lines = [sub.lineno for sub in ast.walk(node) if predicate(sub)]
    assert lines, "expected statement not found"
    return min(lines)


def test_the_rule_runs_after_the_share_threshold_and_before_user_exclusions():
    harmonize, _ = _function("pipeline/executor.py", "_execute_harmonize")

    def assigns(target_text):
        return lambda n: isinstance(n, ast.Assign) and any(
            ast.unparse(t) == target_text for t in n.targets)

    threshold = _line_of(harmonize, lambda n: isinstance(n, ast.Assign) and any(
        ast.unparse(t) == "grid.cells" for t in n.targets) and "cells_post_threshold" in ast.unparse(n.value))
    rule = _line_of(harmonize, lambda n: isinstance(n, ast.Call)
                    and ast.unparse(n.func).endswith("apply_crop_presence_rule"))
    user_skip = _line_of(harmonize, assigns("user_excluded"))
    assert threshold < rule < user_skip


def test_the_harmonize_catch_classifies_the_typed_errors():
    harmonize, _ = _function("pipeline/executor.py", "_execute_harmonize")
    handlers = [n for n in ast.walk(harmonize) if isinstance(n, ast.ExceptHandler)
                and ast.unparse(n.type) == "Exception"]
    body = "\n".join(ast.unparse(h) for h in handlers)
    assert "classify_to_event_dict" in body and "CropPresenceEmptyError" in body
    stage_results = [n for n in ast.walk(harmonize) if isinstance(n, ast.Call)
                     and ast.unparse(n.func) == "StageResult"]
    assert stage_results and all(any(k.arg == "error_events" for k in c.keywords) for c in stage_results)


def test_craft_mask_extraction_reads_through_the_one_reader():
    extract, _ = _function("translators/craft/translator.py", "_extract_crop_mask_from_spam")
    calls = _calls(extract)
    assert "cell_presence" in calls and "sample" not in calls


def test_the_acea_clip_resolves_its_input_through_the_exposed_resolver():
    clip, _ = _function("translators/acea/translator.py", "_clip_spam_data")
    calls = _calls(clip)
    assert "resolve_acea_spam_input" in calls and "glob" not in calls


def test_translation_reads_back_each_engine_roster_when_the_rule_ran():
    translate, _ = _function("pipeline/executor.py", "_execute_translate")
    assert any("record_roster_readback" in name for name in _calls(translate))


# Functions that open a raster, read or sample it, and name SPAM / harvested / crop-area data. Each
# is a translator clip, the legacy SPAM source, the one reader, or a comment-only mention; a new
# independent SPAM sampler must read through ``cell_presence`` instead.
_ALLOWED_SPAM_READERS = {
    "sources/crop_areas/presence.py::cell_presence",
    "sources/crop_areas/spam.py::clip_to_file",
    "sources/crop_areas/spam.py::_sample_raster",
    "pipeline/executor.py::_ensure_isda_1km_cache",
}
_SPAM_TOKENS = re.compile(r"spam|harvest|crop_area|crop_mask", re.IGNORECASE)


def test_no_independent_spam_sampler():
    found = set()
    for path in sorted(_SRC.rglob("*.py")):
        if "vendor" in path.parts:
            continue
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            source = ast.get_source_segment(text, node) or ""
            if ("rasterio.open(" in source and (".sample(" in source or ".read(" in source)
                    and _SPAM_TOKENS.search(source)):
                found.add(f"{path.relative_to(_SRC).as_posix()}::{node.name}")
    assert found <= _ALLOWED_SPAM_READERS, sorted(found - _ALLOWED_SPAM_READERS)
    assert "sources/crop_areas/presence.py::cell_presence" in found
