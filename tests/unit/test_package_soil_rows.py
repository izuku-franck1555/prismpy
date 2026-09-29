"""The soil files' stamps, bindings and detail, and where they may be written: every consumed byte
is bound, the detail names each layer's provenance, and only the translators write soil files."""
from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest

from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.packaging import soil_declaration as sd
from prismpy.packaging.manifest import create_manifest
from prismpy.translators._shared.dssat_sol_writer import write_dssat_sol
from prismpy.translators._shared.eghr_substrate import build_eghr_substrate
from prismpy.translators.base import HwsdOutcome
from prismpy.validators.craft import CraftValidator
from tests.package_soil import REGION, isda_profile, stamp_package_soil
from tests.unit import test_craft_declared_default_soil as craft_t
from tests.unit.test_package_soil_values import _craft, _pythia, _sol_rows

SRC = Path(__file__).resolve().parents[2] / "src/prismpy"
CFG = {"project_name": "p", "region_name": "Koutiala", "country": "Mali", "crop_name": "Maize",
       "start_year": 2015, "end_year": 2016, "data_sources": {"climate": "NASA POWER"}}


def _tree(rel):
    return ast.parse((SRC / rel).read_text(encoding="utf-8"))


def _functions_calling(name):
    """(module, enclosing function) of every call to ``name`` under src/prismpy."""
    sites = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for call in ast.walk(fn):
                    if isinstance(call, ast.Call) and getattr(call.func, "id", getattr(call.func, "attr", None)) == name:
                        sites.add((path.relative_to(SRC).as_posix(), fn.name))
    return sites


# ── AST pins: required stamps, the writer census, bindings written last ─────────


def test_the_stamp_and_description_are_required_and_no_data_sources_literal_names_soil():
    fn = next(n for n in _tree("translators/_shared/dssat_sol_writer.py").body
              if isinstance(n, ast.FunctionDef) and n.name == "write_dssat_sol")
    required = [a.arg for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is None]
    assert {"stamp", "source_label_for_id"} <= set(required)
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                if "data_sources" in keys:
                    inner = node.values[keys.index("data_sources")]
                    if isinstance(inner, ast.Dict):
                        assert "soil" not in [k.value for k in inner.keys if isinstance(k, ast.Constant)], path
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "SoilStamp":
                assert not (node.args and isinstance(node.args[0], ast.Constant)), f"constant stamp in {path}"


def test_only_the_translators_write_and_bind_soil_files():
    census = {name: _functions_calling(name) for name in (
        "write_dssat_sol", "_generate_soil_mask", "_write_soil_raster", "_write_ghr_db",
        "_generate_acea_soil_netcdf", "_generate_soil_netcdf_from_profiles", "_include_eghr_data_legacy",
        "write_binding")}
    assert census["write_dssat_sol"] == {("translators/craft/translator.py", "_generate_soil_package"),
                                         ("translators/_shared/eghr_substrate.py", "build_eghr_substrate")}
    assert census["write_binding"] == {
        ("translators/craft/translator.py", "translate"),
        ("translators/_shared/eghr_substrate.py", "build_eghr_substrate"),
        ("translators/acea/translator.py", "_generate_acea_soil_netcdf"),
        ("translators/acea/translator.py", "_generate_soil_netcdf_from_profiles"),
        ("translators/pythia/translator.py", "_include_eghr_data_legacy")}
    for sites in census.values():
        assert all(module.startswith("translators/") for module, _ in sites), sites


@pytest.mark.parametrize("module, function, writers", [
    ("translators/craft/translator.py", "translate", ("_generate_soil_package", "_generate_soil_mask")),
    ("translators/_shared/eghr_substrate.py", "build_eghr_substrate",
     ("write_dssat_sol", "_write_soil_raster", "_write_ghr_db")),
    ("translators/pythia/translator.py", "_include_eghr_data_legacy", ("copy2",)),
    ("translators/acea/translator.py", "_generate_acea_soil_netcdf", ("Dataset",)),
    ("translators/acea/translator.py", "_generate_soil_netcdf_from_profiles", ("Dataset",)),
])
def test_each_binding_is_written_after_the_last_write_of_its_set(module, function, writers):
    fn = next(n for n in ast.walk(_tree(module)) if isinstance(n, ast.FunctionDef) and n.name == function)
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)]
    bind = min(c.lineno for c in calls if getattr(c.func, "id", getattr(c.func, "attr", None)) == "write_binding")
    last = max(c.lineno for c in calls if getattr(c.func, "id", getattr(c.func, "attr", None)) in writers)
    assert last < bind


# ── the files themselves ─────────────────────────────────────────────────────


def test_readers_see_short_fields_and_the_record_before_any_profile(tmp_path):
    profiles = {1: isda_profile(), 2: craft_t._hwsd(1469), 0: craft_t.craft._default_soil_profile(REGION, "ML")}
    write_dssat_sol(tmp_path / "ML.SOL", profiles, "ML", REGION, stamp=sd.SoilStamp("profiles"),
                    source_label_for_id=lambda key: craft_t.craft._profile_description(key, profiles[key]),
                    profile_cell_counts={1: 1, 2: 1, 0: 1})
    lines = (tmp_path / "ML.SOL").read_text().splitlines()
    first_profile = next(i for i, ln in enumerate(lines) if ln.startswith("*") and not ln.startswith("*SOILS"))
    assert lines[1].startswith(sd.RECORD_PREFIX) and 1 < first_profile
    for line in lines:
        if line.startswith("*") and not line.startswith("*SOILS"):
            assert len(line[13:24].strip()) <= 11 and len(line[37:]) <= 50


@pytest.mark.parametrize("platform", ["craft", "pythia", "acea", "sarra_py"])
def test_a_caller_cannot_state_the_soil(tmp_path, platform):
    stamp_package_soil(tmp_path, platform)
    with pytest.raises(ValueError, match="declared from the package's soil files"):
        create_manifest(tmp_path, {**CFG, "data_sources": {"climate": "x", "soil": "iSDA"}}, platform=platform)


def test_inputs_used_is_reserved_and_the_declaration_survives(tmp_path):
    stamp_package_soil(tmp_path, "craft")
    with pytest.raises(ValueError, match="reserved-key collisions"):
        create_manifest(tmp_path, CFG, platform="craft", additional_metadata={"inputs_used": {"soil": {}}})
    manifest = create_manifest(tmp_path, CFG, platform="craft")
    assert manifest["inputs_used"]["soil"] == sd.declared_soil(tmp_path, "craft").inputs_used()


def test_the_data_sources_keys_are_those_the_engines_always_wrote(tmp_path, fake_hwsd):
    grid = craft_t._grid([101])
    fake_hwsd.plan = {0: craft_t._hwsd(1469)}
    craft_pkg = _craft(tmp_path / "craft", grid, {101: isda_profile()},
                       craft_t._state(HwsdOutcome.NOT_QUERIED, isda=True), paths=True)
    pythia_pkg = _pythia(tmp_path / "pythia", {c: isda_profile() for c in range(6)})
    assert set(json.loads((craft_pkg / "manifest.json").read_text())["data_sources"]) == {
        "soil", "crop_mask", "boundaries", "climate"}
    assert set(json.loads((pythia_pkg / "manifest.json").read_text())["data_sources"]) == {
        "soil", "crop_mask", "boundaries", "climate"}


@pytest.fixture
def fake_hwsd(monkeypatch):
    craft_t._FakeHWSD.plan, craft_t._FakeHWSD.calls = {}, 0
    monkeypatch.setattr(craft_t.craft, "HWSDSource", craft_t._FakeHWSD)
    return craft_t._FakeHWSD


# ── tampering with a bound file ──────────────────────────────────────────────


def test_an_unreferenced_profile_is_not_counted(tmp_path):
    soil = tmp_path / "soil"
    soil.mkdir()
    profiles = {90_000_001: isda_profile(), 1469: craft_t._hwsd(1469)}
    names = write_dssat_sol(soil / "ML.SOL", profiles, "ML", REGION, stamp=sd.SoilStamp("profiles"),
                            source_label_for_id=str, profile_cell_counts={90_000_001: 2})
    (soil / "soil_mask.txt").write_text(f"CellID\tSoilProfile\tSharePCT\n1\t{names[90_000_001]}\t1\n"
                                        f"2\t{names[90_000_001]}\t1\n")
    sd.write_binding(tmp_path, "soil", [soil / "ML.SOL", soil / "soil_mask.txt", soil / sd.DETAIL_FILE])
    assert sd.declared_soil(tmp_path, "craft").record["profile_sources"] == {"isda_s3": 2}


@pytest.mark.parametrize("change", ["raster pixel", "profile_map row", "a .SOL value"])
def test_any_change_to_the_canonical_eghr_triple_is_refused(tmp_path, change):
    stamp_package_soil(tmp_path, "pythia")
    if change == "raster pixel":
        import rasterio
        with rasterio.open(tmp_path / "raster/soil.tif", "r+") as dst:
            dst.write(dst.read(1) + 1, 1)
    elif change == "profile_map row":
        with sqlite3.connect(tmp_path / "eGHR/GHR.db") as db:
            db.execute("UPDATE profile_map SET profile = 'ML99999999'")
    else:
        sol = tmp_path / "eGHR/ML.SOL"
        sol.write_text(sol.read_text().replace(" 1.40", " 1.41", 1))
    with pytest.raises(sd.SoilDeclarationError, match="changed after it was bound"):
        sd.declared_soil(tmp_path, "pythia")


@pytest.mark.parametrize("change", ["a copied .SOL", "GHR.db"])
def test_any_change_to_the_copied_database_is_refused(tmp_path, change):
    ghr = tmp_path / "eGHR"
    ghr.mkdir()
    (ghr / "GHR.db").write_bytes(b"db")
    (ghr / "CM.SOL").write_text("*SOILS: copied\n*CM00000001  eghr  SL  100 x\n    20 -9   0.100\n")
    sd.write_binding(tmp_path, "eGHR", [ghr / "CM.SOL", ghr / "GHR.db"], source=sd.EGHR_DATABASE)
    target = ghr / ("CM.SOL" if change == "a copied .SOL" else "GHR.db")
    target.write_bytes(target.read_bytes().replace(b"0.100", b"0.200") if change != "GHR.db" else b"db2")
    with pytest.raises(sd.SoilDeclarationError, match="changed after it was bound"):
        sd.declared_soil(tmp_path, "pythia")


# ── pedotransfer provenance and the per-layer detail ────────────────────────


def test_estimated_water_limits_are_counted_wherever_they_are_estimated(tmp_path):
    grid = SpatialGrid(resolution="5arcmin", cells=[GridCell(cell_id=1, lat=12.5, lon=-5.5, row=0, col=0)])
    bare = SoilProfile(profile_id="b", lat=12.5, lon=-5.5, source="isda",
                       layers=[SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=50.0, clay=20.0)])
    build_eghr_substrate(grid, {1: bare}, "ML", REGION, tmp_path / "pythia")
    pythia = sd.declared_soil(tmp_path / "pythia", "pythia")
    assert pythia.record["ptf_layers"] == 1 and "pedotransfer function" in pythia.label
    assert pythia.record["hydraulics"]["estimated_layers"] == 1
    supplied = SoilProfile(profile_id="s", lat=12.5, lon=-5.5, source="isda", layers=[
        SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=50.0, clay=20.0, wilting_point=0.1,
                  field_capacity=0.25, saturated_wc=0.45)])
    build_eghr_substrate(grid, {1: supplied}, "ML", REGION, tmp_path / "given")
    assert sd.declared_soil(tmp_path / "given", "pythia").record["ptf_layers"] == 0
    out = craft_t._run(craft_t._translator(tmp_path / "craft"), craft_t._grid([101]), {101: bare},
                       craft_t._state(HwsdOutcome.NO_ANSWER))
    assert out.record["ptf_layers"] == "1"


def test_the_detail_file_names_each_layers_provenance(tmp_path):
    profiles = {
        1469: SoilProfile(profile_id="h", lat=12.5, lon=-5.5, source="hwsd", total_depth=0.4,
                          layers=[SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=40.0, clay=30.0,
                                            organic_carbon=40.4, bulk_density=1.4, ph=6.5),
                                  SoilLayer(depth_top=0.2, depth_bottom=0.4, sand=40.0, clay=30.0,
                                            organic_carbon=1.0, bulk_density=None, ph=6.5)],
                          metadata={"hwsd_smu_id": 1469, "ptf_domain_flags": {0: "organic"}}),
        90_000_001: SoilProfile(profile_id="i", lat=12.5, lon=-5.5, source="iSDA S3 (30m)",
                                layers=[SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=50.0, clay=20.0,
                                                  wilting_point=0.1, field_capacity=0.25, saturated_wc=0.45)],
                                metadata={"source_cell_id": 7}),
        0: craft_t.craft._default_soil_profile(REGION, "ML"),
    }
    write_dssat_sol(tmp_path / "ML.SOL", profiles, "ML", REGION, stamp=sd.SoilStamp("profiles"),
                    source_label_for_id=str, profile_cell_counts={1469: 3, 90_000_001: 2, 0: 1})
    detail = json.loads((tmp_path / sd.DETAIL_FILE).read_text())
    assert detail["ML00001469"]["origin"] == {"hwsd_smu_id": 1469, "component_rule": "max_share_then_lowest_sequence"}
    assert detail["ML00001469"]["n_cells"] == 3
    assert detail["ML00001469"]["layers"] == {
        "0": {"depth_cm": [0, 20], "ptf_domain_flag": "organic", "chem_defaulted": [], "hydraulics_estimated": True},
        "1": {"depth_cm": [20, 40], "ptf_domain_flag": None, "chem_defaulted": ["bulk_density"],
              "hydraulics_estimated": True}}
    assert detail["ML90000001"]["origin"] == {"source_cell_id": 7}
    assert detail["ML90000001"]["layers"]["0"]["hydraulics_estimated"] is False
    assert detail["ML00000000"]["origin"] == {"generic_default": "branch-4 profile"}


# ── the validator, the header, the placeholder package ──────────────────────


def test_the_craft_validator_allows_exactly_the_detail_file(tmp_path):
    soil = tmp_path / "soil"
    soil.mkdir()
    (soil / sd.DETAIL_FILE).write_text("{}")
    validator = CraftValidator.__new__(CraftValidator)
    validator.output_dir = tmp_path
    assert validator.validate_file_types(soil, [".SOL", ".sol", ".txt"], "soil", allowed_names=(sd.DETAIL_FILE,)) == []
    (soil / "other.json").write_text("{}")
    assert validator.validate_file_types(soil, [".SOL", ".sol", ".txt"], "soil", allowed_names=(sd.DETAIL_FILE,))


def _header_rule(path):
    lines = Path(path).read_text().splitlines()
    record = sd.parse_record(lines[1])
    tokens = {token for token, _ in sd.read_sol(path).profiles.values()}
    written = [t for t in sd.PROFILE_SOURCE_TOKENS.values() if t in tokens]
    return lines[0].split(" - Generated by prismpy ")[1] == (
        f"({record['source']}: {', '.join(written)})" if written else f"({record['source']})")


def test_the_soils_line_names_the_stamp_and_the_profiles_written(tmp_path, fake_hwsd):
    grid = craft_t._grid([101, 102])
    fake_hwsd.plan = {0: craft_t._hwsd(1469)}
    mixed = _craft(tmp_path / "b2", grid, {101: isda_profile(), 102: isda_profile()},
                   craft_t._state(HwsdOutcome.NOT_QUERIED, isda=True), paths=True)
    assert (mixed / "soil/ML.SOL").read_text().splitlines()[0] == (
        "*SOILS: Koutiala - Generated by prismpy (profiles: hwsd, default)")
    fake_hwsd.plan = {0: craft_t._hwsd(1469), 1: craft_t._hwsd(1470, sand=50.0)}      # one token, two profiles
    two = _craft(tmp_path / "two", grid, {101: isda_profile(), 102: isda_profile()},
                 craft_t._state(HwsdOutcome.NOT_QUERIED, isda=True), paths=True)
    assert (two / "soil/ML.SOL").read_text().splitlines()[0] == (
        "*SOILS: Koutiala - Generated by prismpy (profiles: hwsd)")
    branch4 = craft_t._run(craft_t._translator(tmp_path / "b4"), grid, None, None)   # no soil source at all
    assert branch4.lines[0].endswith("(default_profile: default)")
    none = tmp_path / "b4/craft/Koutiala"
    stamp_package_soil(tmp_path / "p", "pythia")
    for path in (mixed / "soil/ML.SOL", two / "soil/ML.SOL", none / "soil/ML.SOL", tmp_path / "p/eGHR/ML.SOL"):
        assert _header_rule(path), path


def test_a_package_on_the_retrieval_placeholder_passes(tmp_path):
    grid = craft_t._grid([101, 102])
    pkg = _craft(tmp_path, grid, {0: craft_t._placeholder()}, craft_t._state(HwsdOutcome.NO_ANSWER))
    decl = sd.declared_soil(pkg, "craft")
    assert decl.record["default_cells"] == 2 == decl.record["profile_sources"]["placeholder"]
    assert decl.record["default_profile"] == "ML90000001"
    rows = (pkg / "soil/soil_mask.txt").read_text().splitlines()[1:]
    assert {row.split("\t")[1] for row in rows} == {"ML90000001"}
    assert _sol_rows(pkg / "soil/ML.SOL", "ML90000001")
