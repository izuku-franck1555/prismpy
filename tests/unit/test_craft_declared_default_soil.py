"""CRAFT soil: every grid cell runs on its own soil or on the DECLARED default
profile (written into the .SOL, counted in its record line, warned about above
5 %), never on another cell's soil; the branch follows the harmonize stage's
soil-cascade state; no soil at all after HWSD answered fails the build."""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators._shared import dssat_sol_writer
from prismpy.translators.base import HwsdOutcome, SoilCascadeState, UnifiedData
from prismpy.translators.craft import translator as craft

sys.path.insert(0, str(Path(__file__).parent))
from test_craft_spam_vintage_honesty import _cfg  # noqa: E402

REGION = Region(name="Koutiala", country="Mali", country_iso3="MLI",
                bounds=BoundingBox(minx=-6.0, miny=11.0, maxx=-4.0, maxy=13.0))
DIRECTION = ("may be overstated where water limits growth and understated where soil "
             "nitrogen does")


def _state(outcome, isda=False):
    return SoilCascadeState(isda_served=isda, hwsd=outcome)


def _grid(ids):
    return SpatialGrid(resolution="5arcmin", cells=[
        GridCell(cell_id=cid, lat=12.0 + k / 12, lon=-5.0 + k / 12, row=k, col=k)
        for k, cid in enumerate(ids)])


def _hwsd(smu, n_layers=5, sand=40.0):
    layers = [SoilLayer(depth_top=i / 5, depth_bottom=(i + 1) / 5, sand=sand + i, clay=30.0 - i,
                        silt=30.0, organic_carbon=1.0, bulk_density=1.4, ph=6.5)
              for i in range(n_layers)]
    return SoilProfile(profile_id=f"HWSD_MLI_{smu}", lat=12.0, lon=-5.0, source="hwsd",
                       layers=layers, total_depth=n_layers / 5,
                       metadata={"hwsd_smu_id": smu})


def _isda(cid, sand=50.0):
    return SoilProfile(profile_id=f"isda_{cid}", lat=12.0 + cid / 1000, lon=-5.0, source="isda",
                       layers=[SoilLayer(depth_top=0.0, depth_bottom=0.3, sand=sand, clay=20.0,
                                         silt=100.0 - sand - 20.0, organic_carbon=0.8,
                                         bulk_density=1.5, ph=6.1)],
                       total_depth=0.3)


def _placeholder():
    from prismpy.pipeline.executor import TranslationPipeline
    return next(iter(TranslationPipeline._create_placeholder_soil(None, REGION).values()))


class _FakeHWSD:
    """Stands in for the translator's own HWSD query: index -> profile."""
    plan: dict = {}
    calls = 0

    def __init__(self, config):
        pass

    def retrieve(self, region, cell_coords):
        type(self).calls += 1
        profiles = dict(type(self).plan)
        return SimpleNamespace(success=bool(profiles), errors=[],
                               data=SimpleNamespace(profiles=profiles) if profiles else None)


@pytest.fixture
def fake_hwsd(monkeypatch):
    _FakeHWSD.plan, _FakeHWSD.calls = {}, 0
    monkeypatch.setattr(craft, "HWSDSource", _FakeHWSD)
    return _FakeHWSD


def _translator(tmp_path, paths=False):
    cfg = _cfg(tmp_path)
    if paths:
        cfg.data_sources.soil.hwsd_bil_path = str(tmp_path / "HWSD2.bil")
        cfg.data_sources.soil.hwsd_mdb_path = str(tmp_path / "HWSD2.mdb")
    tr = craft.CraftTranslator(cfg)
    (tr.output_dir / "soil").mkdir(parents=True, exist_ok=True)
    return tr


def _run(tr, grid, existing=None, state=None):
    warnings: list = []
    path, mask = tr._generate_soil_package(grid, REGION, existing, state, warnings)
    lines = path.read_text().splitlines()
    record = dict(f.split("=", 1) for f in lines[1].split(": ", 1)[1].split())
    blocks, name = {}, None
    for ln in lines:
        if ln.startswith("*") and not ln.startswith("*SOILS"):
            name = ln[1:11].strip()
            blocks[name] = []
        elif name is not None:
            blocks[name].append(ln)
    return SimpleNamespace(mask=mask, record=record, warnings=warnings, lines=lines, blocks=blocks)


def _reference_block(tmp_path, key, profile):
    path = tmp_path / "reference.SOL"
    dssat_sol_writer.write_dssat_sol(path, {key: profile}, "ML", REGION)
    return [ln for ln in path.read_text().splitlines()[2:]][1:]


def _default_block(tmp_path):
    return _reference_block(tmp_path, 0, craft._default_soil_profile(REGION, "ML"))


def _integrity(tmp_path, out, key0_expected):
    """Every mask name is a .SOL profile; key 0 exists exactly when a cell uses
    it, with exactly the declared default's layers."""
    assert set(out.mask.values()) <= set(out.blocks)
    if key0_expected:
        assert out.blocks["ML00000000"] == _default_block(tmp_path)
    else:
        assert "ML00000000" not in out.blocks


# ── branch 2: the translator's own HWSD query ──────────────────────────────


def test_a_cell_without_hwsd_soil_runs_on_the_declared_default(tmp_path, fake_hwsd):
    fake_hwsd.plan = {0: _hwsd(1469)}
    out = _run(_translator(tmp_path, paths=True), _grid([101, 102]))
    assert out.mask == {101: "ML00001469", 102: "ML00000000"}
    _integrity(tmp_path, out, key0_expected=True)
    assert {k: out.record[k] for k in ("cells", "default_cells", "default_cause",
                                       "default_fraction", "default_warning")} == {
        "cells": "2", "default_cells": "1",
        "default_cause": "no_hwsd_soil_at_cell_centre:1",
        "default_fraction": "0.5000", "default_warning": "1"}
    assert len(out.warnings) == 1 and out.warnings[0].startswith("1 of 2 grid cells (50.0%)")


def test_layered_hwsd_profile_is_written_to_100_cm(tmp_path, fake_hwsd):
    fake_hwsd.plan = {0: _hwsd(1469)}
    out = _run(_translator(tmp_path, paths=True), _grid([101]))
    header = next(ln for ln in out.lines if ln.startswith("*ML00001469"))
    assert int(header[31:36]) == 100
    rows = [ln.split() for ln in out.blocks["ML00001469"] if ln[:6].strip().isdigit()]
    assert [int(r[0]) for r in rows] == [20, 40, 60, 80, 100]
    assert [float(r[5]) for r in rows] == pytest.approx(
        [round(1 - 0.8 * b / 100, 2) for b in (20, 40, 60, 80, 100)])


def test_no_hwsd_soil_in_any_cell_fails_the_build(tmp_path, fake_hwsd):
    tr = _translator(tmp_path, paths=True)
    with pytest.raises(craft.CraftSoilUnavailableError, match="refusing to invent a default soil"):
        tr._generate_soil_package(_grid([101, 102]), REGION, None, None, [])
    assert not (tr.output_dir / "soil" / "ML.SOL").exists()
    tr.validate_input_data = lambda data: []
    result = tr.translate(UnifiedData(region=REGION, grid=_grid([101]), soil={},
                                      soil_cascade=_state(HwsdOutcome.ANSWERED_NONE)))
    assert result.success is False
    assert result.errors == ["no grid cell in Koutiala has an HWSD soil (non-soil units or "
                             "HWSD gaps); refusing to invent a default soil"]


# ── branch selection follows the cascade state ─────────────────────────────


def test_branch_follows_the_state_not_the_shape(tmp_path, fake_hwsd):
    served = _run(_translator(tmp_path / "a", paths=True), _grid([101]),
                  {101: _hwsd(1469)}, _state(HwsdOutcome.SERVED))
    assert served.mask == {101: "ML00001469"} and fake_hwsd.calls == 0
    fake_hwsd.plan = {0: _hwsd(1470), 1: _hwsd(1471)}
    queried = _run(_translator(tmp_path / "b", paths=True), _grid([101, 102]),
                   {101: _hwsd(1469), 102: _hwsd(1469)}, None)
    assert queried.mask == {101: "ML00001470", 102: "ML00001471"} and fake_hwsd.calls == 1
    with pytest.raises(craft.CraftSoilUnavailableError):
        _run(_translator(tmp_path / "c", paths=True), _grid([101]), {0: _placeholder()},
             _state(HwsdOutcome.ANSWERED_NONE))
    assert fake_hwsd.calls == 1


# ── branch 1: HWSD served at harmonize ─────────────────────────────────────


def test_each_served_cell_keeps_its_own_unit_and_the_unserved_one_gets_the_default(tmp_path):
    existing = {101: _hwsd(1469), 102: _hwsd(1470, sand=60.0), 104: _hwsd(1469)}
    out = _run(_translator(tmp_path), _grid([101, 102, 103, 104]), existing,
               _state(HwsdOutcome.SERVED))
    assert out.mask == {101: "ML00001469", 102: "ML00001470", 103: "ML00000000",
                        104: "ML00001469"}
    assert out.record["default_cause"] == "no_hwsd_soil_at_cell_centre:1"
    _integrity(tmp_path, out, key0_expected=True)


def test_cell_id_zero_keeps_its_own_soil(tmp_path):
    branch1 = _run(_translator(tmp_path / "a"), _grid([0, 7]), {0: _hwsd(1469), 7: _hwsd(1469)},
                   _state(HwsdOutcome.SERVED))
    assert branch1.mask == {0: "ML00001469", 7: "ML00001469"}
    assert branch1.record["default_cells"] == "0"
    branch3 = _run(_translator(tmp_path / "b"), _grid([0, 7]), {0: _isda(0), 7: _isda(7, 55.0)},
                   _state(HwsdOutcome.NOT_QUERIED, isda=True))
    assert branch3.mask == {0: "ML90000001", 7: "ML90000002"}
    assert branch3.record["default_cells"] == "0"


def test_an_all_served_package_writes_no_default(tmp_path):
    out = _run(_translator(tmp_path), _grid([101, 102]), {101: _hwsd(1469), 102: _hwsd(1470)},
               _state(HwsdOutcome.SERVED))
    _integrity(tmp_path, out, key0_expected=False)


# ── branch 3: retrieved soil ───────────────────────────────────────────────


def test_a_lone_retrieved_profile_is_never_fanned_out(tmp_path):
    out = _run(_translator(tmp_path), _grid([101, 102]), {101: _isda(101)},
               _state(HwsdOutcome.NOT_QUERIED, isda=True))
    assert out.mask == {101: "ML90000001", 102: "ML00000000"}
    assert out.record["default_cause"] == "no_retrieved_soil_at_cell:1"
    _integrity(tmp_path, out, key0_expected=True)


def test_retrieved_values_are_unchanged_and_keyed_in_cell_order(tmp_path):
    existing = {303: _isda(303, 60.0), 101: _isda(101, 40.0), 202: _isda(202, 50.0)}
    out = _run(_translator(tmp_path), _grid([303, 101, 202]), existing,
               _state(HwsdOutcome.NOT_QUERIED, isda=True))
    assert out.mask == {101: "ML90000001", 202: "ML90000002", 303: "ML90000003"}
    for key, cid in ((90000001, 101), (90000002, 202), (90000003, 303)):
        assert out.blocks[f"ML{key:08d}"] == _reference_block(tmp_path, key, existing[cid])


def test_the_placeholder_covers_every_cell_and_is_declared(tmp_path):
    placeholder = _placeholder()
    out = _run(_translator(tmp_path), _grid([101, 102, 103]), {0: placeholder},
               _state(HwsdOutcome.NO_ANSWER))
    assert set(out.mask.values()) == {"ML90000001"}
    assert out.blocks["ML90000001"] == _reference_block(tmp_path, 90000001, _placeholder())
    assert {k: out.record[k] for k in ("default_cells", "default_cause", "default_fraction",
                                       "default_warning", "default_depth_cm",
                                       "default_paw_mm")} == {
        "default_cells": "3", "default_cause": "retrieve_stage_placeholder:3",
        "default_fraction": "1.0000", "default_warning": "1", "default_depth_cm": "0-100",
        "default_paw_mm": "158"}
    assert "generic placeholder soil profile (0-100 cm, about 158 mm" in out.warnings[0]
    _integrity(tmp_path, out, key0_expected=False)


def test_no_answer_with_configured_paths_queries_hwsd(tmp_path, fake_hwsd):
    fake_hwsd.plan = {0: _hwsd(1469)}
    out = _run(_translator(tmp_path, paths=True), _grid([101]), {0: _placeholder()},
               _state(HwsdOutcome.NO_ANSWER))
    assert out.mask == {101: "ML00001469"} and fake_hwsd.calls == 1


# ── branch 4: no soil source ───────────────────────────────────────────────


def test_no_soil_source_is_the_base_default_plus_its_record(tmp_path):
    out = _run(_translator(tmp_path), _grid([101, 102]), None, None)
    assert set(out.mask.values()) == {"ML00000000"}
    reference = tmp_path / "base.SOL"
    dssat_sol_writer.write_dssat_sol(reference, {0: craft._default_soil_profile(REGION, "ML")},
                                     "ML", REGION)
    assert [ln for ln in out.lines if not ln.startswith("! prismpy soil record:")] == \
        reference.read_text().splitlines()
    assert {k: out.record[k] for k in ("default_cells", "default_cause", "default_fraction",
                                       "default_warning", "default_depth_cm",
                                       "default_paw_mm")} == {
        "default_cells": "2", "default_cause": "no_soil_source:2", "default_fraction": "1.0000",
        "default_warning": "1", "default_depth_cm": "0-100", "default_paw_mm": "158"}
    assert DIRECTION in out.warnings[0]
    _integrity(tmp_path, out, key0_expected=True)


# ── the > 5 % warning: one computation for the record and the warning ──────


def _mixed(tmp_path, n, k0):
    cells = list(range(1, n + 1))
    existing = {cid: _hwsd(1469) for cid in cells[k0:]}
    return _run(_translator(tmp_path), _grid(cells), existing, _state(HwsdOutcome.SERVED))


def test_the_warning_states_the_direction_and_does_not_fail(tmp_path):
    out = _mixed(tmp_path / "a", 20, 2)
    assert out.record["default_warning"] == "1" and len(out.warnings) == 1
    warning = out.warnings[0]
    for part in ("2 of 20", "10.0%", "0-100 cm", "about 158 mm", DIRECTION):
        assert part in warning
    assert "likely overstated" not in warning and ".;" not in warning
    quiet = _mixed(tmp_path / "b", 20, 1)
    assert quiet.record["default_warning"] == "0" and quiet.warnings == []


def test_the_warning_uses_the_exact_ratio(tmp_path):
    over = _mixed(tmp_path / "a", 20001, 1001)
    assert (over.record["default_fraction"], over.record["default_warning"]) == ("0.0500", "1")
    assert len(over.warnings) == 1
    exact = _mixed(tmp_path / "b", 20000, 1000)
    assert exact.record["default_warning"] == "0" and exact.warnings == []


def test_record_and_warning_follow_the_one_declaration(tmp_path, monkeypatch):
    real = dssat_sol_writer._default_declaration
    monkeypatch.setattr(dssat_sol_writer, "_default_declaration",
                        lambda *a, **k: dataclasses.replace(real(*a, **k), warning=True))
    out = _mixed(tmp_path, 20, 1)
    assert out.record["default_warning"] == "1"
    assert len(out.warnings) == 1 and out.warnings[0].startswith("1 of 20 grid cells")


@pytest.mark.parametrize("lower_fc, bottom, depth, paw", [
    (0.28, 1.0, "0-100", "158"),
    (0.24, 1.0, "0-100", "126"),
    (0.28, 0.8, "0-80", "126"),
])
def test_depth_and_water_are_derived_in_the_record_and_the_warning(
        tmp_path, monkeypatch, lower_fc, bottom, depth, paw):
    real = craft._default_soil_profile

    def modified(region, country_code):
        profile = real(region, country_code)
        profile.layers[1].field_capacity = lower_fc
        profile.layers[1].depth_bottom = bottom
        return profile

    monkeypatch.setattr(craft, "_default_soil_profile", modified)
    out = _run(_translator(tmp_path), _grid([101, 102]), None, None)
    assert (out.record["default_depth_cm"], out.record["default_paw_mm"]) == (depth, paw)
    assert f"({depth} cm, about {paw} mm plant-available water)" in out.warnings[0]


# ── every branch keeps every cell ──────────────────────────────────────────


def test_no_branch_loses_a_cell(tmp_path, fake_hwsd):
    ids = [101, 102, 103]
    fake_hwsd.plan = {0: _hwsd(1469)}
    runs = [
        _run(_translator(tmp_path / "1"), _grid(ids), {101: _hwsd(1469)}, _state(HwsdOutcome.SERVED)),
        _run(_translator(tmp_path / "2", paths=True), _grid(ids), None, None),
        _run(_translator(tmp_path / "3"), _grid(ids), {101: _isda(101)},
             _state(HwsdOutcome.NOT_QUERIED, isda=True)),
        _run(_translator(tmp_path / "4"), _grid(ids), None, None),
        _run(_translator(tmp_path / "p"), _grid(ids), {0: _placeholder()}, _state(HwsdOutcome.NO_ANSWER)),
    ]
    for out in runs:
        assert set(out.mask) == set(ids)
        assert set(out.mask.values()) <= set(out.blocks)


def test_the_soil_mask_and_other_craft_files_keep_the_grid_cells(tmp_path):
    tr = _translator(tmp_path)
    grid = _grid([101, 102, 103])
    _, mapping = tr._generate_soil_package(grid, REGION, {101: _isda(101)},
                                           _state(HwsdOutcome.NOT_QUERIED, isda=True), [])
    mask_file = tr._generate_soil_mask(grid=grid, region=REGION, cell_to_profile=mapping)
    first = [ln.split()[0] for ln in Path(mask_file).read_text().splitlines() if ln.strip()]
    assert sorted(int(f) for f in first if f.isdigit()) == [101, 102, 103]


# ── stable keys ────────────────────────────────────────────────────────────

_DETERMINISM = r"""
import sys, pathlib
sys.path.insert(0, sys.argv[2])
from test_craft_declared_default_soil import _translator, _grid, _isda, _state, REGION
from prismpy.translators.base import HwsdOutcome
out = pathlib.Path(sys.argv[1])
tr = _translator(out)
ids = list(range(1000, 1646))
path, mask = tr._generate_soil_package(_grid(ids), REGION, {c: _isda(c) for c in ids},
                                       _state(HwsdOutcome.NOT_QUERIED, isda=True), [])
(out / "mask.txt").write_text("\n".join(f"{c},{mask[c]}" for c in sorted(mask)))
(out / "sol.txt").write_bytes(path.read_bytes())
"""


def test_keys_are_stable_across_hash_seeds_and_never_collide(tmp_path):
    outputs = []
    for seed in ("0", "1", "2", "3", "random"):
        out = tmp_path / seed
        env = dict(os.environ, PYTHONHASHSEED=seed,
                   PYTHONPATH=os.pathsep.join(filter(None, [
                       str(Path(craft.__file__).parents[3]), os.environ.get("PYTHONPATH")])))
        subprocess.run([sys.executable, "-c", _DETERMINISM, str(out), str(Path(__file__).parent)],
                       check=True, env=env, capture_output=True)
        outputs.append(((out / "sol.txt").read_bytes(), (out / "mask.txt").read_text()))
    assert all(o == outputs[0] for o in outputs)
    assert outputs[0][0].count(b"\n*ML9") == 646
