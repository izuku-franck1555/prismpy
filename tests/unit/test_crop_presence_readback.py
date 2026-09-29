"""Each engine's roster, as its real writer wrote it for a restricted run, is read back from the
package and digested with the one encoding: PYTHIA's sites by CellID, CRAFT's soil mask, ACEA's
gridcells (parsed, never executed). A read-back that differs from the final roster is visible."""
from __future__ import annotations

import ast
import csv

import geopandas as gpd
import pytest

from prismpy.config.schema import Platform
from prismpy.models.spatial import SpatialGrid
from prismpy.translators.base import UnifiedData
from tests.unit._crop_presence_fixtures import (
    COAST_MAIZ, cell, identity_of, make_config, region_of, run_grid_stages,
)


def _digest(ids):
    from prismpy.cells.roster_digest import roster_id_digest

    return roster_id_digest(ids)


def _restricted_run(tmp_path, monkeypatch, platform):
    from prismpy.config.schema import CropPresenceRule

    cfg = make_config(tmp_path, rule=CropPresenceRule(**identity_of(COAST_MAIZ)), rule_path=COAST_MAIZ,
                      targets=(platform.value,))
    _, boundary, ids, pipe = run_grid_stages(cfg, monkeypatch)
    grid = SpatialGrid(resolution="5arcmin", cells=[cell(i) for i in ids])
    return pipe, boundary["crop_presence"], grid, region_of(cfg)


def _write(pipe, platform, grid, region):
    translator = pipe._get_translator(platform)
    for sub in ("shapes", "soil", "config"):  # translate() creates the package folders first
        (translator.output_dir / sub).mkdir(parents=True, exist_ok=True)
    if platform is Platform.PYTHIA:
        translator._generate_sites_shapefile(grid, region)
        return translator.output_dir / "shapes" / "sites.shp"
    if platform is Platform.CRAFT:
        return translator._generate_soil_mask(grid, region)
    gridcells = translator._compute_30arcmin_cell_ids(grid)
    return translator._generate_acea_config(UnifiedData(region=region, grid=grid), gridcells, "nasa_power")


def _parse(platform, path):
    """My own parse of the written roster file."""
    if platform is Platform.PYTHIA:
        return [int(v) for v in gpd.read_file(path)["CellID"]]
    if platform is Platform.CRAFT:
        return [int(row["CellID"]) for row in csv.DictReader(path.read_text().splitlines(), delimiter="\t")]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "gridcells" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise AssertionError("no gridcells")


def _final_digest(platform, record):
    return record["n30_final_digest" if platform is Platform.ACEA else "n5_final_digest"]


@pytest.mark.parametrize("platform", [Platform.PYTHIA, Platform.CRAFT, Platform.ACEA])
def test_each_engine_reads_back_what_its_writer_wrote(tmp_path, monkeypatch, platform):
    pipe, record, grid, region = _restricted_run(tmp_path, monkeypatch, platform)
    written = _write(pipe, platform, grid, region)
    pipe._record_roster_readback(platform)

    readback = pipe.provenance.record.boundary["crop_presence"]["roster_readback"][platform.value]
    assert readback["id_digest"] == _digest(_parse(platform, written)) == _final_digest(platform, record)
    if platform is Platform.ACEA:
        assert readback["n"] == record["n30_final"] == len(_parse(platform, written))


def _tamper(platform, path, fault):
    if fault == "missing":
        for sibling in path.parent.glob(path.stem + ".*"):
            sibling.unlink()
        return
    if platform is Platform.PYTHIA:
        sites = gpd.read_file(path)
        if fault in ("extra", "duplicate"):
            extra = sites.iloc[[0]].copy()
            if fault == "extra":
                extra["CellID"] = int(sites["CellID"].max()) + 1
            sites = gpd.GeoDataFrame(pd_concat([sites, extra]), crs=sites.crs)
        elif fault == "different":
            sites.loc[0, "CellID"] = int(sites["CellID"].max()) + 1
        else:
            sites = sites.rename(columns={"CellID": "CellNo"})
        sites.to_file(path)
        return
    text = path.read_text()
    if platform is Platform.CRAFT:
        lines = text.splitlines(keepends=True)
        first_id = lines[1].split("\t", 1)[0]
        lines = {"extra": lines + [lines[1].replace(first_id, str(int(first_id) + 10**6), 1)],
                 "duplicate": lines + [lines[1]],
                 "different": [lines[0], lines[1].replace(first_id, str(int(first_id) + 10**6), 1)] + lines[2:],
                 "malformed": ["CellId\tSoilProfile\tSharePCT\r\n"] + lines[1:]}[fault]
        path.write_text("".join(lines))
        return
    ids = _parse(platform, path)
    new = {"extra": ids + [max(ids) + 1], "duplicate": ids + [ids[0]], "different": [max(ids) + 1] + ids[1:],
           "malformed": None}[fault]
    path.write_text(text.replace("gridcells = ", "gridcells = int(", 1) if new is None
                    else text.replace(f"gridcells = {_format(ids)}", f"gridcells = {new!r}", 1))


def _format(ids):
    from prismpy.translators.acea.translator import AceaTranslator

    return AceaTranslator.__new__(AceaTranslator)._format_gridcells(ids)


def pd_concat(frames):
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "different", "malformed"])
@pytest.mark.parametrize("platform", [Platform.PYTHIA, Platform.CRAFT, Platform.ACEA])
def test_a_writer_fault_is_visible_in_the_readback(tmp_path, monkeypatch, platform, fault):
    pipe, record, grid, region = _restricted_run(tmp_path, monkeypatch, platform)
    _tamper(platform, _write(pipe, platform, grid, region), fault)
    pipe._record_roster_readback(platform)

    readback = pipe.provenance.record.boundary["crop_presence"]["roster_readback"][platform.value]
    assert readback.get("id_digest") != _final_digest(platform, record)
    if fault in ("missing", "duplicate", "malformed"):
        assert readback.get("id_digest") is None and readback["error"]


def test_the_acea_config_is_parsed_never_executed(tmp_path):
    from prismpy.packaging.roster_readback import read_back_roster

    (tmp_path / "config").mkdir()
    sentinel = tmp_path / "executed"
    (tmp_path / "config" / "coast_config.py").write_text(
        f"open({str(sentinel)!r}, 'w').close()\n\nclass project_conf:\n    gridcells = [59081, 59801]\n")
    assert sorted(read_back_roster("acea", tmp_path).ids) == [59081, 59801] and not sentinel.exists()


def test_the_acea_readback_needs_exactly_one_config(tmp_path):
    from prismpy.packaging.roster_readback import read_back_roster

    (tmp_path / "config").mkdir()
    for name in ("a_config.py", "b_config.py"):
        (tmp_path / "config" / name).write_text("class project_conf:\n    gridcells = [59081]\n")
    with pytest.raises(ValueError):
        read_back_roster("acea", tmp_path)
