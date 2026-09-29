"""Each engine's package declares the soil its files hold: the values the engine reads come from
the source the declaration names, and the manifest (data_sources.soil, inputs_used.soil) and the
README state exactly that declaration."""
from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.packaging import soil_declaration as sd
from prismpy.packaging.manifest import create_manifest
from prismpy.packaging.readme_generator import generate_readme
from prismpy.translators.acea import translator as acea
from prismpy.translators.base import HwsdOutcome, UnifiedData
from prismpy.translators.pythia.translator import PythiaTranslator
from tests.unit import test_craft_declared_default_soil as craft_t
from tests.unit.test_apply_soil_overrides_to_assignment import _make_sidecar, _sand_override_entry
from tests.unit.test_pythia_canonical_substrate_flag import _build_grid_2x3
from tests.unit.test_pythia_canonical_substrate_flag import _build_project_config as _pythia_config


def _agrees(package_dir, platform):
    """The manifest and the README state exactly the declaration read from the soil files."""
    decl = sd.declared_soil(package_dir, platform)
    manifest = json.loads((package_dir / "manifest.json").read_text())
    assert manifest["data_sources"]["soil"] == decl.label
    assert manifest["inputs_used"] == {"schema_version": 1, "soil": decl.inputs_used()}
    assert decl.label in (package_dir / "README.md").read_text()
    return decl


def _sol_rows(path, name):
    """(SLB, clay, silt) per layer of profile ``name``, as written (SLCL and SLSI)."""
    lines, rows, inside = Path(path).read_text().splitlines(), [], False
    for line in lines:
        if line.startswith("*"):
            inside = line[1:11].strip() == name
        elif inside and line[:6].strip().isdigit():
            rows.append((int(line[:6]), float(line[54:60]), float(line[60:66])))
    return rows


# ── CRAFT, through translate() and generate_package() ─────────────────────────


def _craft(tmp_path, grid, existing, state, paths=False):
    records = [ClimateRecord(date=date(2015, 1, 1) + timedelta(days=d), tmax=32.0, tmin=21.0,
                             precip=2.0, srad=20.0) for d in range(6 * 365)]
    climate = {c.cell_id: ClimateTimeSeries(location_id=c.cell_id, lat=c.lat, lon=c.lon,
                                            source="nasa_power", records=records) for c in grid.cells}
    tr = craft_t._translator(tmp_path, paths=paths)
    data = UnifiedData(region=craft_t.REGION, grid=grid, climate=climate, soil=existing,
                       soil_cascade=state)
    result = tr.translate(data)
    assert result.success, result.errors
    tr.generate_package(data, result.output_files)
    return tr.output_dir


@pytest.fixture
def fake_hwsd(monkeypatch):
    craft_t._FakeHWSD.plan, craft_t._FakeHWSD.calls = {}, 0
    monkeypatch.setattr(craft_t.craft, "HWSDSource", craft_t._FakeHWSD)
    return craft_t._FakeHWSD


def _isda_s3(cid, sand=52.0):
    return SoilProfile(profile_id=f"isda_{cid}", lat=12.0, lon=-5.0, source="iSDA S3 (30m)", layers=[
        SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=sand, clay=20.0, organic_carbon=0.8,
                  bulk_density=1.4, ph=6.2),
        SoilLayer(depth_top=0.2, depth_bottom=0.5, sand=sand - 2, clay=24.0, organic_carbon=0.5,
                  bulk_density=1.45, ph=6.3)])


def test_craft_hwsd_queried_by_craft_is_declared_hwsd(tmp_path, fake_hwsd):
    """The prismweb path: iSDA served at harmonize, CRAFT's own HWSD query answers."""
    grid = craft_t._grid([101, 102])
    fake_hwsd.plan = {0: craft_t._hwsd(1469), 1: craft_t._hwsd(1470, sand=50.0)}
    pkg = _craft(tmp_path, grid, {101: _isda_s3(101), 102: _isda_s3(102)},
                 craft_t._state(HwsdOutcome.NOT_QUERIED, isda=True), paths=True)
    decl = _agrees(pkg, "craft")
    assert decl.label.startswith("HWSD v2.0 dominant soil component (max share), 0–100 cm from its own HWSD layers, "
                                 "read at each cell centre, one profile per grid cell")
    assert decl.record["profile_sources"] == {"hwsd": 2}
    assert _sol_rows(pkg / "soil/ML.SOL", "ML00001470") == [(20 * (i + 1), 30.0 - i, 30.0) for i in range(5)]


def test_craft_hwsd_served_at_harmonize_is_declared_hwsd(tmp_path):
    grid = craft_t._grid([101, 102])
    pkg = _craft(tmp_path, grid, {101: craft_t._hwsd(1469), 102: craft_t._hwsd(1469)},
                 craft_t._state(HwsdOutcome.SERVED))
    decl = _agrees(pkg, "craft")
    assert decl.record["profile_sources"] == {"hwsd": 2}
    assert _sol_rows(pkg / "soil/ML.SOL", "ML00001469") == [(20 * (i + 1), 30.0 - i, 30.0) for i in range(5)]


def test_craft_on_isda_alone_is_declared_isda(tmp_path):
    grid = craft_t._grid([101, 102])
    pkg = _craft(tmp_path, grid, {101: _isda_s3(101), 102: _isda_s3(102, sand=60.0)},
                 craft_t._state(HwsdOutcome.NO_ANSWER))
    decl = _agrees(pkg, "craft")
    assert decl.label.startswith("iSDA Africa soil properties, 0–50 cm, read at each cell centre, one profile per "
                                 "grid cell")
    assert {token for token, _ in sd.read_sol(pkg / "soil/ML.SOL").profiles.values()} == {"isda_s3"}
    assert _sol_rows(pkg / "soil/ML.SOL", "ML90000002") == [(20, 20.0, 20.0), (50, 24.0, 18.0)]


# ── PYTHIA, through the eGHR step, the manifest and the README ─────────────────


def _pythia(tmp_path, soil, grid=None, sidecar=None, canonical=True):
    translator = PythiaTranslator(config=_pythia_config(tmp_path), output_dir=str(tmp_path),
                                  prefer_canonical_substrate=canonical)
    translator.cockpit_override_sidecar = sidecar
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    data = UnifiedData(region=craft_t.REGION, grid=grid or _build_grid_2x3(), soil=soil)
    translator._include_eghr_data(data)
    translator._generate_manifest(data)
    translator._generate_readme(data)
    return tmp_path


def test_pythia_canonical_isda_is_declared_isda_not_egh_r(tmp_path):
    grid = _build_grid_2x3()
    pkg = _pythia(tmp_path, {c.cell_id: _isda_s3(c.cell_id, sand=40.0 + c.cell_id) for c in grid.cells})
    decl = _agrees(pkg, "pythia")
    assert decl.label.startswith("iSDA Africa soil properties, 0–50 cm, read at each cell centre, one profile per "
                                 "grid cell, stored in the eGHR file format")
    assert decl.record["profile_sources"] == {"isda_s3": len(grid.cells)}
    sols = list((pkg / "eGHR").glob("*.SOL"))
    top_silt = sorted(_sol_rows(sols[0], p)[0][2] for p in sd.read_sol(sols[0]).profiles)
    assert top_silt == sorted(40.0 - c.cell_id for c in grid.cells)      # 100 - sand - clay per cell


def test_pythia_overrides_are_counted_after_they_apply(tmp_path):
    grid = _build_grid_2x3()
    ids = [c.cell_id for c in grid.cells]
    sidecar = _make_sidecar([_sand_override_entry(str(ids[0]), 70.0), _sand_override_entry(str(ids[1]), 72.0)])
    decl = _agrees(_pythia(tmp_path, {i: _isda_s3(i) for i in ids}, sidecar=sidecar), "pythia")
    assert "user soil overrides applied to the top layer of 2 cells" in decl.label


def test_pythia_with_no_profile_for_any_cell_declares_none(tmp_path):
    grid = SpatialGrid(resolution="5arcmin", cells=[GridCell(cell_id=i, lat=12.0, lon=-5.0 + i / 12, row=0, col=i)
                                                   for i in (1, 2)])
    pkg = _pythia(tmp_path, {0: craft_t._placeholder()}, grid=grid)     # no grid cell has id 0
    decl = _agrees(pkg, "pythia")
    assert decl.source_id == "none" and decl.label == sd.NO_PROFILE
    assert sd.read_sol(next((pkg / "eGHR").glob("*.SOL"))).profiles == {}


def test_pythia_cell_zero_runs_on_the_executor_placeholder(tmp_path):
    grid = SpatialGrid(resolution="5arcmin", cells=[GridCell(cell_id=0, lat=12.0, lon=-5.0, row=0, col=0),
                                                   GridCell(cell_id=1, lat=12.0, lon=-4.9, row=0, col=1)])
    decl = _agrees(_pythia(tmp_path, {0: craft_t._placeholder(), 1: _isda_s3(1)}, grid=grid), "pythia")
    assert decl.record["profile_sources"] == {"isda_s3": 1, "placeholder": 1}
    assert f"{sd.SOURCE_DESCRIPTIONS['placeholder']} for 1 cell" in decl.label


def test_pythia_legacy_copies_are_bound_and_declared_as_the_database(tmp_path):
    source = tmp_path / "global"
    (source / "SOL").mkdir(parents=True)
    with sqlite3.connect(source / "GHR.db") as db:
        db.execute("CREATE TABLE profile_map(id INTEGER PRIMARY KEY, profile TEXT NOT NULL)")
    (source / "SOL" / "CM.SOL").write_text("*SOILS: global database\r\n")
    pkg = tmp_path / "pkg"
    translator = PythiaTranslator(config=_pythia_config(pkg), output_dir=str(pkg),
                                  prefer_canonical_substrate=False)
    translator.config.platform_config.pythia.eghr_database_path = str(source / "GHR.db")
    translator.config.platform_config.pythia.eghr_sol_dir = str(source / "SOL")
    translator._include_eghr_data_legacy()
    decl = sd.declared_soil(pkg, "pythia")
    assert decl.source_id == "eghr_database" and decl.label == sd.BASE[("pythia", "eghr_database")]
    assert (pkg / "eGHR/GHR.db").read_bytes() == (source / "GHR.db").read_bytes()


# ── ACEA, through its two netCDF writers ──────────────────────────────────────


def _acea(tmp_path):
    tr = acea.AceaTranslator.__new__(acea.AceaTranslator)
    tr.output_dir = tmp_path
    (tmp_path / "soil").mkdir(parents=True, exist_ok=True)
    return tr


def _acea_manifest(pkg):
    cfg = {"project_name": "acea", "region_name": "Koutiala", "country": "Mali", "crop_name": "Maize",
           "start_year": 2015, "end_year": 2016, "data_sources": {"climate": "NASA POWER"}}
    (pkg / "manifest.json").write_text(json.dumps(create_manifest(pkg, cfg, platform="acea")))
    generate_readme(pkg / "README.md", cfg, platform="acea")
    return _agrees(pkg, "acea")


def test_acea_hwsd_tiles_are_counted_by_outcome(tmp_path, monkeypatch):
    """One tile on a known unit, one on a unit missing from the table, one on nodata (unit 0)."""
    cells = [360 * 720 // 2 + 400, 360 * 720 // 2 + 401, 360 * 720 // 2 + 402]  # one row, three columns
    lat, lons = 89.75 - (cells[0] // 720) * 0.5, [-179.75 + (c % 720) * 0.5 for c in cells]
    bil = tmp_path / "HWSD2.bil"
    with rasterio.open(bil, "w", driver="GTiff", height=1, width=3, count=1, dtype="int32", crs="EPSG:4326",
                       transform=from_origin(lons[0] - 0.25, lat + 0.25, 0.5, 0.5)) as dst:
        dst.write(np.array([[1469, 9999, 0]], dtype="int32"), 1)
    (tmp_path / "HWSD2.mdb").write_bytes(b"mdb")
    csv = "HWSD2_SMU_ID,LAYER,SAND,CLAY\n1469,D1,61.0,17.0\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=csv, stderr=""))
    tr = _acea(tmp_path / "pkg")
    tr._generate_acea_soil_netcdf(cells, bil, tmp_path / "HWSD2.mdb")
    decl = _acea_manifest(tr.output_dir)
    assert decl.record == {"source": "hwsd_upper_layer_texture", "cells": 3, "hwsd": 1, "default": 1, "masked": 1}
    assert decl.label.endswith("; 1 of 3 cells uses a generic default soil (sand 40%, clay 25%); 1 of 3 cells "
                               "has no soil value and is skipped by the engine")


def test_acea_profiles_by_list_position_and_the_empty_cases(tmp_path, monkeypatch):
    tr = _acea(tmp_path / "list")
    tr._generate_soil_netcdf_from_profiles({7: _isda_s3(7), 8: craft_t._placeholder()}, [1000, 1001, 1002])
    decl = _acea_manifest(tr.output_dir)
    assert decl.record["profile_sources"] == {"isda_s3": 1, "placeholder": 1}
    assert decl.label.startswith("Retrieved soil: ") and "assigned by list order, not by location" in decl.label
    # an empty HWSD lookup, and no soil file at all
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="x"))
    empty = _acea(tmp_path / "empty")
    for name in ("empty.bil", "empty.mdb"):
        (tmp_path / name).write_bytes(b"")
    empty._generate_acea_soil_netcdf([1000], tmp_path / "empty.bil", tmp_path / "empty.mdb")
    assert _acea_manifest(empty.output_dir).source_id == "default_values"
    assert _acea_manifest(_acea(tmp_path / "bare").output_dir).source_id == "engine_installed"


def test_sarra_declares_its_bundled_soil_in_the_manifest(tmp_path):
    cfg = {"project_name": "sarra", "region_name": "Niamey", "country": "Niger", "crop_name": "Millet",
           "start_year": 2015, "end_year": 2016, "data_sources": {"rainfall": "TAMSAT v3.1"}}
    manifest = create_manifest(tmp_path, cfg, platform="sarra_py")
    assert manifest["data_sources"]["soil"] == sd.BASE[("sarra_py", "engine_bundled_africa")]
    shutil.rmtree(tmp_path)
