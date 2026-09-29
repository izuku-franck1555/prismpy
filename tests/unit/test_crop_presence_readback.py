"""Each engine's written roster is read back from its package and digested with the one encoding:
PYTHIA's sites by CellID, CRAFT's soil mask, ACEA's gridcells (parsed, never executed)."""
from __future__ import annotations

import numpy as np
import pytest

from prismpy.models.spatial import GridCell, SpatialGrid
from tests.unit._crop_presence_fixtures import identity_of, make_config, run_grid_stages, write_layer

_IDS = [4024014, 4024015, 4028334]


def _digest(ids):
    from prismpy.cells.roster_digest import roster_id_digest

    return roster_id_digest(ids)


def test_pythia_reads_back_cell_ids_not_site_numbers(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point

    from prismpy.packaging.roster_readback import read_back_roster

    (tmp_path / "shapes").mkdir()
    gpd.GeoDataFrame(
        [{"geometry": Point(0, 0), "ID": n + 1, "CellID": cid, "Latitude": 0.0, "Longitude": 0.0,
          "Region": "Koutiala"} for n, cid in enumerate(_IDS)],
        crs="EPSG:4326").to_file(tmp_path / "shapes" / "sites.shp")

    readback = read_back_roster("pythia", tmp_path)
    assert sorted(readback.ids) == _IDS and readback.id_digest == _digest(_IDS)
    assert (readback.file, readback.field) == ("shapes/sites.shp", "CellID")


def test_craft_reads_back_the_soil_mask(tmp_path):
    from prismpy.packaging.roster_readback import read_back_roster

    (tmp_path / "soil").mkdir()
    rows = "".join(f"{cid}\tMZ{n:08d}\t100\r\n" for n, cid in enumerate(sorted(_IDS, reverse=True)))
    (tmp_path / "soil" / "soil_mask.txt").write_bytes(f"CellID\tSoilProfile\tSharePCT\r\n{rows}".encode())

    readback = read_back_roster("craft", tmp_path)
    assert sorted(readback.ids) == _IDS and readback.id_digest == _digest(_IDS)
    assert (readback.file, readback.field) == ("soil/soil_mask.txt", "CellID")


@pytest.mark.parametrize("text", [
    "CellId\tSoilProfile\tSharePCT\r\n4024014\tMZ00000001\t100\r\n",
    "CellID\tSoilProfile\tSharePCT\r\n4024014\tMZ00000001\t100\r\n4024014\tMZ00000002\t100\r\n",
    "CellID\tSoilProfile\tSharePCT\r\nx\tMZ00000001\t100\r\n",
])
def test_craft_readback_refuses_a_malformed_mask(tmp_path, text):
    from prismpy.packaging.roster_readback import read_back_roster

    (tmp_path / "soil").mkdir()
    (tmp_path / "soil" / "soil_mask.txt").write_bytes(text.encode())
    with pytest.raises(ValueError):
        read_back_roster("craft", tmp_path)


def _acea_config(gridcells, prelude=""):
    return (f'{prelude}"""ACEA project configuration."""\n\n\nclass project_conf:\n'
            f"    project_name = 'koutiala'\n    resolution = 0\n    gridcells = [\n"
            + "".join(f"        {cid},\n" for cid in gridcells) + "    ]\n")


def test_acea_reads_back_gridcells_without_executing_the_package(tmp_path):
    from prismpy.packaging.roster_readback import read_back_roster

    (tmp_path / "config").mkdir()
    sentinel = tmp_path / "executed"
    prelude = f"open({str(sentinel)!r}, 'w').close()\n"
    (tmp_path / "config" / "koutiala_config.py").write_text(_acea_config([59801, 59081, 59082], prelude))

    readback = read_back_roster("acea", tmp_path)
    assert sorted(readback.ids) == [59081, 59082, 59801] and not sentinel.exists()
    assert readback.id_digest == _digest([59081, 59082, 59801])


def test_acea_readback_needs_exactly_one_config(tmp_path):
    from prismpy.packaging.roster_readback import read_back_roster

    (tmp_path / "config").mkdir()
    for name in ("a_config.py", "b_config.py"):
        (tmp_path / "config" / name).write_text(_acea_config([59081]))
    with pytest.raises(ValueError):
        read_back_roster("acea", tmp_path)


def test_n30_final_equals_the_acea_gridcells(tmp_path, monkeypatch):
    from prismpy.config.schema import CropPresenceRule
    from prismpy.translators.acea.translator import AceaTranslator

    arr = np.zeros((18, 18), dtype="float32")
    arr[4, 4], arr[5, 9], arr[10, 12], arr[15, 15] = 12.0, 0.5, 3.0, 1.0
    path = write_layer(tmp_path / "spam2020_V2r0_global_H_MAIZ_A.tif", arr)
    rule = CropPresenceRule(**identity_of(path, release="V2r0"))
    _, boundary, ids = run_grid_stages(
        make_config(tmp_path, rule=rule, rule_path=path, targets=("acea",)), monkeypatch)

    cells = []
    for cid in ids:
        row, col = divmod(cid, SpatialGrid.GLOBAL_COLS_5ARCMIN)
        lat, lon = SpatialGrid.latlon_from_rowcol_5arcmin(row, col)
        cells.append(GridCell(cell_id=cid, lat=lat, lon=lon, row=row, col=col))
    gridcells = AceaTranslator.__new__(AceaTranslator)._compute_30arcmin_cell_ids(
        SpatialGrid(resolution="5arcmin", cells=cells))
    record = boundary["crop_presence"]
    assert record["n30_final"] == len(gridcells)
    assert record["n30_final_digest"] == _digest(gridcells)
