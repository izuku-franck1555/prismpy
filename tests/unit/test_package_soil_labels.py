"""The declaration's wording: each label is composed exactly as specified from the package's own
record, the generic-soil warning first, one join rule throughout, and no stray punctuation. Every
string a component emits is one of the inventory's literals (``fixtures/soil_label_inventory.json``)."""
from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

import pytest

from prismpy.models import soil as soil_model
from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.packaging import soil_declaration as sd
from prismpy.packaging.manifest import create_manifest
from prismpy.translators._shared.dssat_sol_writer import DefaultDeclaration, write_dssat_sol
from prismpy.translators._shared.eghr_substrate import build_eghr_substrate
from prismpy.translators.base import HwsdOutcome
from tests.package_soil import REGION
from tests.unit import test_craft_declared_default_soil as craft_t
from tests.unit import test_package_soil_check as check
from tests.unit import test_package_soil_rows as rows
from tests.unit import test_package_soil_values as values
from tests.unit import test_soil_declaration as unit
from tests.unit.test_hwsd_layered_profile import _extract, _unit
from tests.unit.test_package_soil_values import _agrees, _craft

SRC = Path(__file__).resolve().parents[2] / "src/prismpy"
INVENTORY = json.loads((Path(__file__).resolve().parents[1] / "fixtures/soil_label_inventory.json").read_text(
    encoding="utf-8"))
CFG = {"project_name": "p", "region_name": "Koutiala", "country": "Mali", "crop_name": "Maize",
       "start_year": 2015, "end_year": 2016, "data_sources": {"climate": "NASA POWER"}}
BAD_JOINS = (".;", ". ;", "..", ";;", "; .", "  ")


def _block(path, name):
    lines = Path(path).read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("*") and ln[1:11].strip() == name)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("*")), len(lines))
    return lines[start + 1:end]


# ── the three generic-soil packages, end to end ────────────────────────────────


def _mixed_fixture(monkeypatch):
    craft_t._FakeHWSD.plan = {0: craft_t._hwsd(1469)}          # cell 102's HWSD unit is not a soil
    monkeypatch.setattr(craft_t.craft, "HWSDSource", craft_t._FakeHWSD)
    return (craft_t._grid([101, 102]), {101: craft_t._isda(101), 102: craft_t._isda(102)},
            craft_t._state(HwsdOutcome.NOT_QUERIED, isda=True))


def test_a_mixed_package_warns_first_then_names_its_one_retrieved_source(tmp_path, monkeypatch):
    pkg = _craft(tmp_path / "pkg", *_mixed_fixture(monkeypatch), paths=True)
    decl = _agrees(pkg, "craft")
    mask = dict(line.split("\t")[:2] for line in (pkg / "soil/soil_mask.txt").read_text().splitlines()[1:])
    assert mask == {"101": "ML00001469", "102": "ML00000000"}
    (tmp_path / "ref").mkdir()
    assert _block(pkg / "soil/ML.SOL", "ML00000000") == craft_t._default_block(tmp_path / "ref")
    assert sd.read_sol(pkg / "soil/ML.SOL").profiles["ML00000000"][0] == "default"
    assert {key: decl.record[key] for key in ("profile_sources", "default_cells", "default_fraction",
                                               "default_warning", "default_depth_cm", "default_paw_mm")} == {
        "profile_sources": {"hwsd": 1, "default": 1}, "default_cells": 1, "default_fraction": "0.5000",
        "default_warning": 1, "default_depth_cm": "0-100", "default_paw_mm": 158}
    # the warning, then one space and the BASE, which names HWSD alone
    warning = INVENTORY["W-no_hwsd_soil_at_cell_centre-1of2"]
    assert decl.label.startswith(warning + " HWSD v2.0 dominant soil component")
    assert decl.label == warning + " " + INVENTORY["base-craft-profiles"] + (
        "; water limits estimated with a pedotransfer function adapted from Saxton & Rawls (2006) for 5 of 7 layers")


def test_a_package_without_a_soil_source_is_its_warning_alone(tmp_path):
    """CRAFT with no soil source at all, written, masked and bound in translate()'s order."""
    tr, grid = craft_t._translator(tmp_path), craft_t._grid(range(101, 121))
    sol, mapping = tr._generate_soil_package(grid, craft_t.REGION, None, None, [])
    mask = tr._generate_soil_mask(grid, craft_t.REGION, mapping)
    sd.write_binding(tr.output_dir, "soil", [sol, sol.parent / sd.DETAIL_FILE, mask])
    decl = sd.declared_soil(tr.output_dir, "craft")
    assert decl.label == INVENTORY["W-no_soil_source-20of20"] and "cell centre" not in decl.label
    assert create_manifest(tr.output_dir, CFG, platform="craft")["data_sources"]["soil"] == decl.label


def _placeholder_fixture():
    return craft_t._grid(range(101, 121)), {0: craft_t._placeholder()}, craft_t._state(HwsdOutcome.NO_ANSWER)


def test_a_placeholder_package_is_its_placeholder_warning_alone(tmp_path):
    decl = _agrees(_craft(tmp_path, *_placeholder_fixture()), "craft")
    assert (decl.record["default_cells"], decl.record["default_cause"]) == (20, {"retrieve_stage_placeholder": 20})
    assert decl.label == INVENTORY["W-retrieve_stage_placeholder-20of20"] and "cell centre" not in decl.label


@pytest.mark.parametrize("kind", ["default", "placeholder"])
def test_the_label_and_the_craft_warning_share_the_one_direction_clause(tmp_path, monkeypatch, kind):
    def both(where):
        grid, existing, state = _mixed_fixture(monkeypatch) if kind == "default" else _placeholder_fixture()
        label = sd.declared_soil(_craft(where / "pkg", grid, existing, state, paths=kind == "default"), "craft").label
        translator = craft_t._translator(where / "w", paths=kind == "default")
        return label, craft_t._run(translator, grid, existing, state).warnings[0]

    label, warning = both(tmp_path / "real")
    ending = "; " + soil_model.generic_soil_direction(kind) + "."
    assert label.count(ending) == 1 and warning.endswith(ending)
    monkeypatch.setattr(soil_model, "generic_soil_direction", lambda k: f"SENTINEL-{k}")
    for text in both(tmp_path / "sentinel"):
        assert f"; SENTINEL-{kind}." in text and "may hold more or less water" not in text


# ── the joins, on records ─────────────────────────────────────────────────────


def test_without_a_base_the_first_suffix_opens_a_sentence_after_the_warning(tmp_path):
    decl = sd.declared_soil(unit._craft(
        tmp_path / "b4", [("ML00000000", "default", 100)], ["ML00000000"] * 20, source="default_profile",
        profile_sources="default:20", cells=20, default_cells=20, default_cause="no_soil_source:20",
        default_fraction="1.0000", default_warning=1, default_profile="ML00000000", default_depth_cm="0-100",
        default_paw_mm=158, chem_defaulted="bulk_density:3,organic_carbon:3,ph:3"), "craft")
    assert decl.label == INVENTORY["W-no_soil_source-20of20"] + (
        " Chemistry defaults used (bulk density 1.40 g/cm³ ×3, organic carbon 0.50 % ×3, pH 6.5 ×3)")
    decl = sd.declared_soil(unit._craft(
        tmp_path / "b3", [("ML90000001", "placeholder", 100)], ["ML90000001"] * 20,
        profile_sources="placeholder:20", cells=20, default_cells=20, default_cause="retrieve_stage_placeholder:20",
        default_fraction="1.0000", default_warning=1, default_profile="ML90000001", default_depth_cm="0-100",
        default_paw_mm=158, ptf_layers=2, layers=2, chem_defaulted="bulk_density:0,organic_carbon:0,ph:1"), "craft")
    assert decl.label == INVENTORY["W-retrieve_stage_placeholder-20of20"] + (
        " Water limits estimated with a pedotransfer function adapted from Saxton & Rawls (2006) for 2 of 2 layers; "
        "chemistry defaults used (pH 6.5 ×1)")


@pytest.mark.parametrize("ptf_layers, before", [(0, "read at each cell centre, one profile per grid cell"),
                                                 (3, "for 3 of 5 layers")])
def test_a_clause_under_the_threshold_follows_the_chain_without_a_warning(tmp_path, ptf_layers, before):
    clause = ("1 of 20 candidate grid cells runs on a generic default soil profile (0-100 cm, about 158 mm "
              "plant-available water), because no soil value exists at the cell centre: the HWSD soil unit there "
              "is not a soil (e.g. water or urban) or has no data. Its result reflects this default soil, not the "
              "local soil.")
    label = unit._generic(tmp_path, 1, 20, ptf_layers=ptf_layers).label
    assert label.endswith(f"{before}. {clause}") and label.count(clause) == 1
    assert "WARNING" not in label and "may hold more or less water" not in label


def test_no_label_the_fixtures_produce_has_a_stray_join(tmp_path):
    """Every label composed while the declaration's value, row and wording tests run."""
    labels, real = [], sd._label
    modules = (unit, values, rows, check, sys.modules[__name__])
    for i, (fn, case) in enumerate(c for module in modules for c in _test_calls(module)):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sd, "_label", lambda *a: labels.append(real(*a)) or labels[-1])
            craft_t._FakeHWSD.plan, craft_t._FakeHWSD.calls = {}, 0
            mp.setattr(craft_t.craft, "HWSDSource", craft_t._FakeHWSD)
            (tmp_path / str(i)).mkdir()
            given = {"tmp_path": tmp_path / str(i), "monkeypatch": mp, "fake_hwsd": craft_t._FakeHWSD, **case}
            fn(**{name: given[name] for name in inspect.signature(fn).parameters})
    assert len(labels) > 80
    stray = [label for label in labels if label != label.strip() or any(bad in label for bad in BAD_JOINS)]
    assert stray == []


def _test_calls(module):
    """(test function, parameters) for each case pytest would collect from ``module``, bar the scan."""
    for name, fn in vars(module).items():
        scan = fn is test_no_label_the_fixtures_produce_has_a_stray_join
        if scan or not name.startswith("test_") or not inspect.isfunction(fn):
            continue
        cases = [{}]
        for mark in getattr(fn, "pytestmark", []):
            if mark.name == "parametrize":
                names = [n.strip() for n in mark.args[0].split(",")]
                cases = [{**case, **dict(zip(names, argvalues if len(names) > 1 else (argvalues,)))}
                         for case in cases for argvalues in mark.args[1]]
        for case in cases:
            yield fn, case


# ── the pedotransfer token ────────────────────────────────────────────────────


def test_the_record_names_the_adapted_method_and_the_label_says_the_same(tmp_path):
    grid = SpatialGrid(resolution="5arcmin", cells=[GridCell(cell_id=1, lat=12.5, lon=-5.5, row=0, col=0)])
    profile = SoilProfile(profile_id="p", lat=12.5, lon=-5.5, source="isda", layers=[
        SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=50.0, clay=20.0),        # estimated at normalisation
        SoilLayer(depth_top=0.2, depth_bottom=0.5, sand=48.0, clay=24.0, wilting_point=0.1,
                  field_capacity=0.25, saturated_wc=0.45)])
    build_eghr_substrate(grid, {1: profile}, "ML", REGION, tmp_path)
    manifest = create_manifest(tmp_path, CFG, platform="pythia")
    record, label = manifest["inputs_used"]["soil"]["record"], manifest["data_sources"]["soil"]
    assert (record["ptf_layers"], record["layers"]) == (1, 2)
    assert record["hydraulics"] == {"method": "adapted_from_saxton_rawls_2006", "estimated_layers": 1, "layers": 2,
                                    "reference": "Saxton KE, Rawls WJ (2006) Soil Sci. Soc. Am. J. 70:1569–1578"}
    assert ("adapted from Saxton & Rawls (2006)" in label) == (
        record["hydraulics"]["method"] == "adapted_from_saxton_rawls_2006")
    assert label == (
        "iSDA Africa topsoil properties, 0–50 cm, read at each cell centre, one profile per grid cell, stored in "
        "the eGHR file format and looked up at each simulated site; water limits estimated with a pedotransfer "
        "function adapted from Saxton & Rawls (2006) for 1 of 2 layers; 1 of 1 profiles is shallower than "
        "100 cm (depth 50 cm); chemistry defaults used (bulk density 1.40 g/cm³ ×2, organic carbon 0.50 % ×2, "
        "pH 6.5 ×2)")
    soil_source = (SRC / "models/soil.py").read_text(encoding="utf-8")
    assert "simplif" not in soil_source.lower()
    assert "Uses a pedotransfer function adapted from Saxton & Rawls (2006)." in (
        soil_model.SoilLayer.estimate_hydraulic_properties.__doc__)
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "saxton_rawls_2006_simplified" not in text, path
        assert not [ln for ln in text.splitlines() if "saxton" in ln.lower() and "simplif" in ln.lower()], path


# ── the inventory: every emitted string is one of its literals ─────────────────

MOPTI = Region(name="Mopti", country="Mali", country_iso3="MLI",
               bounds=BoundingBox(minx=-4.5, miny=14.0, maxx=-3.5, maxy=15.0))
TOKEN_SOURCE = {token: source for source, token in sd.PROFILE_SOURCE_TOKENS.items()}
SHALLOW = {"1of1-20": [20], "1of2-20": [20, 100], "3of40-40": [40] * 3 + [100] * 37, "2of2-20_50": [20, 50]}
ORGANIC_ANDIC = {"2_1": (2, 1, 12, 3, 5), "1_0": (1, 0, 4, 1, 1), "0_3": (0, 3, 12, 3, 2)}
ACEA_COUNT = {"default": "default", "skip": "masked", "field": "field_default"}
PYTHIA_ISDA = {"isda_s3": [50]}
BASES = {
    "craft-profiles": ("craft", {"source": "profiles", "profile_sources": {"hwsd": 1}}, {"hwsd": [100]}),
    "pythia-profiles-z0": ("pythia", {"source": "profiles", "profile_sources": {"isda_s3": 10}, "cells": 10,
                                      "no_profile_cells": 0}, PYTHIA_ISDA),
    "pythia-profiles-covered7of10": ("pythia", {"source": "profiles", "profile_sources": {"isda_s3": 7},
                                                "cells": 10, "no_profile_cells": 3}, PYTHIA_ISDA),
    "pythia-profiles-covered1of10": ("pythia", {"source": "profiles", "profile_sources": {"isda_s3": 1},
                                                "cells": 10, "no_profile_cells": 9}, PYTHIA_ISDA),
    "pythia-eghr_database": ("pythia", {"source": "eghr_database"}, {}),
    "pythia-none": ("pythia", {"source": "none"}, {}),
    **{f"acea-{source}": ("acea", {"source": source}, {})
       for source in ("hwsd_upper_layer_texture", "default_values", "engine_installed")},
    **{f"acea-list-{token}": ("acea", {"source": "profiles_by_list_position", "profile_sources": {token: 1}}, {})
       for token in ("isda_s3", "isda", "hwsd", "placeholder")},
    "sarra": ("sarra_py", {"source": "engine_bundled_africa"}, {}),
}


def _record(**fields):
    record = dict(source="profiles", cells=10, ptf_layers=0, layers=12, no_profile_cells=0, override_cells=0,
                  chem_default_values="bulk_density:1.40,organic_carbon:0.50,ph:6.5",
                  chem_defaulted={"bulk_density": 0, "organic_carbon": 0, "ph": 0}, organic_layers=0,
                  andic_layers=0, profiles=3, cells_with_flagged_layers=0)
    return {**record, **fields}


def _generic_record(cause, counts):
    k, n = map(int, counts.split("of"))
    return {"cells": n, "default_cause": {cause: k}, "default_depth_cm": "0-100", "default_paw_mm": 158}


def _depths(token, bottoms):
    """The written profiles' depths, through the one derivation from a detail file's layers."""
    detail = {f"P{i}": {"source": token, "layers": {"0": {"depth_cm": [0, bottom]}}}
              for i, bottom in enumerate(bottoms)}
    return sd.profile_depths_cm(detail)


def _suffix(platform, record, depths=None):
    items = sd._suffixes(platform, record, depths or {})
    assert len(items) == 1, items
    return "; " + items[0]


def _soils_line(where, source, tokens):
    profiles = {i: SoilProfile(profile_id=f"p{i}", lat=14.5, lon=-4.0, source=TOKEN_SOURCE[token],
                               layers=[SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=50.0, clay=20.0)])
                for i, token in enumerate(tokens, start=1)}
    path = where / f"{source}-{'+'.join(tokens)}.SOL"
    write_dssat_sol(path, profiles, "ML", MOPTI, stamp=sd.SoilStamp(source), source_label_for_id=str,
                    profile_cell_counts={i: 1 for i in profiles})
    return path.read_text().splitlines()[0]


def _built(row, where, engine="craft"):
    """What the built component emits for inventory ``row``'s axis values."""
    family, _, rest = row.partition("-")
    parts = rest.split("-")
    if family == "hdr":
        return _soils_line(where, parts[1], [] if parts[2] == "none" else parts[2].split("+"))
    if family == "S":
        return sd._describe(parts[0], engine, {} if parts[1] == "nodepth" else _depths(
            parts[0], [int(b) for b in parts[1].split("_")]))
    if family == "Sacea":
        return sd._describe(rest, "acea", {})
    if family == "several":
        counts = {"2x": {"isda_s3": 15, "hwsd": 5}, "count1": {"isda_s3": 9, "placeholder": 1}}[parts[0]]
        depths = {"isda_s3": [50], "hwsd": [100]}
        return sd._sources_text(counts, ("default",), lambda t: sd._describe(t, "pythia", depths))
    if family == "base":
        return sd._base(*BASES[rest])
    if family == "ptf":
        p, n = map(int, rest.split("of"))
        return _suffix("craft", _record(ptf_layers=p, layers=n))
    if family == "shallow":
        return _suffix("craft", _record(), {"hwsd": SHALLOW[rest]})
    if family == "noprof":
        z, n = map(int, rest.split("of"))
        return _suffix("pythia", _record(no_profile_cells=z, cells=n))
    if family == "over":
        return _suffix("pythia", _record(override_cells=int(rest)))
    if family == "chem":
        counts = dict(zip(("bulk_density", "organic_carbon", "ph"), map(int, rest.split("_"))))
        return _suffix("craft", _record(chem_defaulted=counts))
    if family == "organd":
        k, m, layers, profiles, cells = ORGANIC_ANDIC[rest]
        return _suffix("craft", _record(organic_layers=k, andic_layers=m, layers=layers, profiles=profiles,
                                        cells_with_flagged_layers=cells))
    if family == "acea":
        return _suffix("acea", {"source": "hwsd_upper_layer_texture", "cells": 20,
                                ACEA_COUNT[parts[0]]: int(parts[1].split("of")[0])})
    if family == "clause":
        return sd._generic_clause(_generic_record(parts[0], parts[1]))
    if family == "W":
        return sd._warning(_generic_record(parts[0], parts[1]))
    if family == "dir":
        return soil_model.generic_soil_direction(rest)
    if family == "craft":
        k, n = map(int, parts[1].split("of"))
        cause = "retrieve_stage_placeholder" if parts[0] == "placeholder" else "no_hwsd_soil_at_cell_centre"
        return craft_t.craft._default_soil_warning(DefaultDeclaration(
            cells=n, default_cells=k, causes=((cause, k),), fraction=k / n, warning=True, profile="ML00000000",
            depth_cm="0-100", paw_mm=158))
    raise AssertionError(f"no component for {row}")


@pytest.mark.parametrize("row", sorted(INVENTORY))
def test_each_inventory_string_is_what_its_component_emits(tmp_path, row):
    assert _built(row, tmp_path) == INVENTORY[row]


@pytest.mark.parametrize("row", sorted(r for r in INVENTORY if r.startswith("S-")))
def test_both_sol_engines_describe_a_source_through_the_one_function(row):
    assert _built(row, None, engine="pythia") == INVENTORY[row]


@pytest.mark.parametrize("consumer", ["label", "craft"])
@pytest.mark.parametrize("plant_available, organic_carbon", [(0.30, 2.0), (0.30, 0.2), (0.06, 2.0), (0.06, 0.2)])
@pytest.mark.parametrize("kind", ["default", "placeholder"])
def test_the_generic_soil_warning_compares_nothing_with_the_packages_own_soils(
        tmp_path, kind, plant_available, organic_carbon, consumer):
    """The package's own profiles hold 300 or 60 mm and are rich or poor in organic matter: one text."""
    def retrieved(cid):
        return SoilProfile(profile_id=f"r{cid}", lat=12.0, lon=-5.0, source="iSDA S3 (30m)", total_depth=1.0, layers=[
            SoilLayer(depth_top=i / 5, depth_bottom=(i + 1) / 5, sand=40.0, clay=25.0, silt=35.0,
                      organic_carbon=organic_carbon, bulk_density=1.4, ph=6.5, wilting_point=0.10,
                      field_capacity=0.10 + plant_available, saturated_wc=0.45) for i in range(5)])

    if kind == "default":          # a real package: 18 cells on retrieved profiles, 2 without one
        cells = list(range(1, 21))
        grid, existing = craft_t._grid(cells), {c: retrieved(c) for c in cells[2:]}
        state = craft_t._state(HwsdOutcome.NO_ANSWER)
        if consumer == "craft":
            assert craft_t._run(craft_t._translator(tmp_path), grid, existing, state).warnings == [
                INVENTORY["craft-default-2of20"]]
        else:
            decl = sd.declared_soil(_craft(tmp_path, grid, existing, state), "craft")
            assert decl.label.startswith(INVENTORY["W-no_retrieved_soil_at_cell-2of20"] + " ")
        return
    # every cell of a placeholder package runs on the placeholder, so its other profiles are written unused
    soil, declarations = tmp_path / "soil", []
    soil.mkdir()
    names = write_dssat_sol(soil / "ML.SOL", {90_000_001: craft_t._placeholder(), 90_000_002: retrieved(0)}, "ML",
                            craft_t.REGION, stamp=sd.SoilStamp("profiles"), source_label_for_id=str,
                            profile_cell_counts={90_000_001: 20}, default_causes={"retrieve_stage_placeholder": 20},
                            default_declaration_out=declarations)
    (soil / "soil_mask.txt").write_text("CellID\tSoilProfile\tSharePCT\n" +
                                        "".join(f"{c}\t{names[90_000_001]}\t1\n" for c in range(1, 21)))
    sd.write_binding(tmp_path, "soil", [soil / "ML.SOL", soil / sd.DETAIL_FILE, soil / "soil_mask.txt"])
    if consumer == "craft":
        assert craft_t.craft._default_soil_warning(declarations[0]) == INVENTORY["craft-placeholder-20of20"]
    else:
        assert sd.declared_soil(tmp_path, "craft").label == INVENTORY["W-retrieve_stage_placeholder-20of20"]


def test_the_source_depth_and_the_shallow_suffix_read_the_one_derivation(tmp_path, monkeypatch):
    # every written profile counts, including one no cell uses
    unused = unit._craft(tmp_path / "unused", [("ML00001469", "hwsd", 100), ("ML00001470", "hwsd", 60)],
                         ["ML00001469"], profile_sources="hwsd:1", cells=1)
    label = sd.declared_soil(unused, "craft").label
    assert label.startswith(INVENTORY["S-hwsd-60_100"])
    assert "; 1 of 2 profiles is shallower than 100 cm (depth 60 cm)" in label
    pkg = unit._craft(tmp_path, [("ML00001469", "hwsd", 100)], ["ML00001469"], profile_sources="hwsd:1", cells=1)
    monkeypatch.setattr(sd, "profile_depths_cm", lambda detail: {"hwsd": [37]})
    label = sd.declared_soil(pkg, "craft").label
    assert "HWSD v2.0 dominant soil component (max share), 0–37 cm from its own HWSD layers" in label
    assert "; 1 of 1 profiles is shallower than 100 cm (depth 37 cm)" in label


def test_acea_reads_only_the_top_layers_sand_and_clay():
    tree = ast.parse((SRC / "translators/acea/translator.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_generate_soil_netcdf_from_profiles")
    subscripts = [n for n in ast.walk(fn)
                  if isinstance(n, ast.Subscript) and getattr(n.value, "attr", None) == "layers"]
    assert [ast.unparse(n.slice) for n in subscripts] == ["0"]
    layer_names = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign) and n.value in subscripts for t in n.targets}
    read = {n.attr for n in ast.walk(fn)
            if isinstance(n, ast.Attribute) and getattr(n.value, "id", None) in layer_names}
    assert layer_names and read == {"sand", "clay"}


def test_an_hwsd_component_without_its_second_layer_is_declared_at_its_written_depth(tmp_path):
    profiles, _ = _extract([row for row in _unit(1469) if row["LAYER"] != "D2"], [1469])
    pkg = _craft(tmp_path, craft_t._grid([101]), {101: profiles[0]}, craft_t._state(HwsdOutcome.SERVED))
    label = _agrees(pkg, "craft").label
    assert INVENTORY["S-hwsd-20"] in label and INVENTORY["shallow-1of1-20"] in label


def test_an_isda_profile_without_its_lower_layer_is_declared_at_its_written_depth(tmp_path):
    grid = SpatialGrid(resolution="5arcmin", cells=[GridCell(cell_id=1, lat=12.5, lon=-5.5, row=0, col=0)])
    topsoil_only = SoilProfile(profile_id="isda_1", lat=12.5, lon=-5.5, source="iSDA S3 (30m)", layers=[
        SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=52.0, clay=20.0, organic_carbon=0.8, bulk_density=1.4, ph=6.2)])
    build_eghr_substrate(grid, {1: topsoil_only}, "ML", REGION, tmp_path)   # the executor keeps no nodata layer
    assert sd.declared_soil(tmp_path, "pythia").label.startswith(INVENTORY["S-isda_s3-20"] + ", ")


def test_a_pythia_package_with_unmapped_cells_names_the_cells_it_covers(tmp_path):
    grid = SpatialGrid(resolution="5arcmin", cells=[GridCell(cell_id=i, lat=12.5, lon=-5.5 + i / 12, row=0, col=i)
                                                     for i in range(1, 11)])
    build_eghr_substrate(grid, {i: values._isda_s3(i) for i in range(1, 8)}, "ML", REGION, tmp_path)
    label = sd.declared_soil(tmp_path, "pythia").label
    assert label.startswith(INVENTORY["base-pythia-profiles-covered7of10"]) and INVENTORY["noprof-3of10"] in label
