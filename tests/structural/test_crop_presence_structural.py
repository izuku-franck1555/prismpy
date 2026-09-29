"""Where the crop-presence rule and its reader sit in the code: the rule between the share threshold
and the user exclusions, through the one > 0 predicate; the harmonize catch classifying the typed
errors; CRAFT's mask extraction and ACEA's clip routed through the one reader and the one resolver;
one roster-id digest encoding; and no raster reader beyond the classified inventory."""
from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "prismpy"


def _function(relpath, name):
    for node in ast.walk(ast.parse((_SRC / relpath).read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{relpath}::{name} not found")


def _calls(node):
    return {ast.unparse(sub.func).rsplit(".", 1)[-1] for sub in ast.walk(node) if isinstance(sub, ast.Call)}


def _first_line(node, predicate):
    lines = [sub.lineno for sub in ast.walk(node) if predicate(sub)]
    assert lines, "expected statement not found"
    return min(lines)


def test_the_rule_runs_after_the_share_threshold_and_before_user_exclusions():
    harmonize = _function("pipeline/executor.py", "_execute_harmonize")
    threshold = _first_line(harmonize, lambda n: isinstance(n, ast.Assign) and any(
        ast.unparse(t) == "grid.cells" for t in n.targets) and "cells_post_threshold" in ast.unparse(n.value))
    rule = _first_line(harmonize, lambda n: isinstance(n, ast.Call)
                       and ast.unparse(n.func).endswith("apply_crop_presence_rule"))
    user_skip = _first_line(harmonize, lambda n: isinstance(n, ast.Assign) and any(
        ast.unparse(t) == "user_excluded" for t in n.targets))
    assert threshold < rule < user_skip


def test_the_rule_judges_presence_through_the_one_predicate():
    rule = _function("pipeline/crop_presence.py", "apply_crop_presence_rule")
    assert "crop_area_present" in _calls(rule)
    inline = [ast.unparse(n) for n in ast.walk(rule) if isinstance(n, ast.Compare)
              and any(isinstance(op, (ast.Gt, ast.GtE)) for op in n.ops)
              and any(isinstance(c, ast.Constant) and c.value == 0 for c in n.comparators)]
    assert inline == []


def test_the_harmonize_catch_classifies_the_typed_errors():
    harmonize = _function("pipeline/executor.py", "_execute_harmonize")
    handlers = [n for n in ast.walk(harmonize) if isinstance(n, ast.ExceptHandler)
                and ast.unparse(n.type) == "Exception"]
    body = "\n".join(ast.unparse(h) for h in handlers)
    assert "classify_to_event_dict" in body and "CropPresenceEmptyError" in body
    assert "CropPresenceIdentityError" in body
    results = [n for n in ast.walk(harmonize) if isinstance(n, ast.Call) and ast.unparse(n.func) == "StageResult"]
    assert results and all(any(k.arg == "error_events" for k in c.keywords) for c in results)


def test_craft_mask_extraction_reads_through_the_one_reader_and_keeps_its_own_checks():
    extract = _function("translators/craft/translator.py", "_extract_crop_mask_from_spam")
    source = ast.unparse(extract)
    calls = _calls(extract)
    assert "cell_presence" in calls and not {"open", "sample"} & calls
    assert "cap_at_100" in source and "SpamVintageError" in source


def test_the_acea_clip_resolves_its_input_through_the_exposed_resolver():
    clip = _function("translators/acea/translator.py", "_clip_spam_data")
    calls = _calls(clip)
    assert "resolve_acea_spam_input" in calls and "glob" not in calls


def test_translation_reads_back_each_engine_roster():
    assert any("record_roster_readback" in name for name in _calls(
        _function("pipeline/executor.py", "_execute_translate")))


def test_one_roster_id_digest_encoding():
    for relpath in ("pipeline/crop_presence.py", "packaging/roster_readback.py"):
        tree = ast.parse((_SRC / relpath).read_text())
        assert "roster_id_digest" in _calls(tree), relpath
        hashing = {f.name for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and "sha256" in _calls(f)}
        assert hashing <= {"crop_presence_record_digest"}, (relpath, hashing)


# Every raster reader in src is classified here; a new SPAM area reader must use cell_presence.
_RASTER_READERS = {
    "sources/crop_areas/presence.py::cell_presence": "the one SPAM presence reader",
    "sources/crop_areas/presence.py::raster_identity": "the one reader's identity (header + checksum)",
    "sources/crop_areas/spam.py::clip_to_file": "ACEA's harvested-area clip",
    "sources/crop_areas/spam.py::_sample_raster": "legacy SPAMSource.retrieve (no production caller)",
    "sources/crop_areas/spam.py::_extract_from_bounds": "legacy SPAMSource.retrieve (no production caller)",
    "translators/acea/translator.py::_generate_dummy_spam_files": "ACEA's placeholder SPAM writer",
    "translators/pythia/translator.py::_cell_block_sums": "PYTHIA's per-cell crop area (kept separate)",
    "translators/pythia/translator.py::_check_mask_covers_cells": "PYTHIA's mask extent check",
    "translators/pythia/translator.py::_clip_global_raster": "PYTHIA's raster clips",
    "translators/pythia/translator.py::_snap_bounds_outward": "PYTHIA's clip extent",
    "translators/pythia/translator.py::_generate_management_rasters": "not SPAM: management writers",
    "translators/pythia/translator.py::_enumerate_countries_from_local_substrate": "not SPAM: soil",
    "translators/acea/translator.py::_generate_acea_soil_netcdf": "not SPAM: soil",
    "translators/sarra_py/translator.py::_generate_projection_climate_geotiffs": "not SPAM: climate",
    "translators/_shared/eghr_substrate.py::_write_soil_raster": "not SPAM: soil",
    "pipeline/executor.py::_ensure_isda_1km_cache": "not SPAM: soil",
    "pipeline/executor.py::_retrieve_isda_api_for_grid": "not SPAM: soil",
    "preprocess/zone_elevation_lookup.py::lookup_zone_and_elevation": "not SPAM: soil",
    "sources/climate/agera5.py::load_variable": "not SPAM: climate",
    "sources/climate/tamsat.py::load_daily_rainfall": "not SPAM: climate",
    "sources/soil/eghr.py::_clip_raster_to_bounds": "not SPAM: soil",
    "sources/soil/eghr.py::_sample_pixel_ids": "not SPAM: soil",
    "sources/soil/hwsd.py::_sample_bil_raster": "not SPAM: soil",
    "sources/soil/isda.py::load_variable": "not SPAM: soil",
    "sources/soil/isda.py::sample_at_points": "not SPAM: soil",
    "koppen/kg_classifier.py::__init__": "not SPAM: climate zones",
    "koppen/kg_classifier.py::_sample_one": "not SPAM: climate zones",
    "koppen/kg_classifier.py::classify_batch": "not SPAM: climate zones",
    "validators/post_translate.py::_validate_sarra_py_geotiffs": "not SPAM: climate",
    "validators/post_translate.py::sample_sarra_py_per_cell": "not SPAM: climate",
    "validators/post_translate.py::sarra_py_climate_rasters_readable": "not SPAM: climate",
}


def _raster_readers():
    found = set()
    for path in sorted(_SRC.rglob("*.py")):
        if "vendor" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    isinstance(sub, ast.Call) and (ast.unparse(sub.func) in ("rasterio.open", "rio.open")
                                                   or ast.unparse(sub.func).endswith((".sample", "open_rasterio")))
                    for sub in ast.walk(node)):
                found.add(f"{path.relative_to(_SRC).as_posix()}::{node.name}")
    return found


def test_no_raster_reader_beyond_the_classified_inventory():
    found = _raster_readers()
    assert found <= set(_RASTER_READERS), sorted(found - set(_RASTER_READERS))
    assert "translators/craft/translator.py::_extract_crop_mask_from_spam" not in found
    assert "sources/crop_areas/presence.py::cell_presence" in found


def test_spam_source_retrieve_keeps_no_production_caller():
    for path in sorted(_SRC.rglob("*.py")):
        if path.name == "spam.py" or "vendor" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        instances = {ast.unparse(t) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "SPAMSource"
                     for t in n.targets}
        used = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and ast.unparse(n.func.value) in instances}
        assert used <= {"clip_to_file"}, (path, used)
