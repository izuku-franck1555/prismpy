"""Sprint S AC-10 — eGHR triple-symmetry structural pin.

Elevates the pair-wise membership pins from AC-3 (raster ↔ DB) and
AC-4 (DB ↔ SOL) into a single triple-symmetry assertion that
checks all three artifact id sets at once. Pair-wise pins catch
drift between two artifacts; this pin catches the rare case where
A ⊆ B and B ⊆ C but A ⊄ C (set-membership transitivity holds in
mathematics but the failure mode here is artifact-level: a
producer that emits the same id under different normalization in
two writers can ship a triple where each pair-wise pin passes but
the three sets do not collectively form an equivalence class).

The failure message names all three artifacts plus their id sets
so a developer chasing a regression can localize the drift
directly to the artifact responsible — the find-AND-localize form
of structural pinning per durable lesson §24.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict

import rasterio

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators._shared import build_eghr_substrate


def _build_substrate(tmp_path: Path):
    cells = []
    cell_id = 0
    for row in range(2):
        for col in range(3):
            cells.append(
                GridCell(
                    cell_id=cell_id,
                    lat=12.0 - row * 0.5,
                    lon=2.0 + col * 0.5,
                    row=row,
                    col=col,
                    resolution="custom",
                )
            )
            cell_id += 1
    grid = SpatialGrid(
        resolution="custom",
        cells=cells,
        increment_deg=0.5,
        bounds=BoundingBox(minx=2.0, miny=11.0, maxx=3.5, maxy=12.5),
    )
    profiles: Dict[int, SoilProfile] = {
        0: SoilProfile(
            profile_id="P0",
            lat=12.0,
            lon=2.0,
            source="hwsd2",
            layers=[
                SoilLayer(
                    depth_top=0.0,
                    depth_bottom=0.2,
                    sand=30.0,
                    clay=45.0,
                    silt=25.0,
                    organic_carbon=0.5,
                    bulk_density=1.4,
                    ph=6.5,
                    field_capacity=0.30,
                    wilting_point=0.18,
                ),
            ],
        ),
        1: SoilProfile(
            profile_id="P1",
            lat=12.0,
            lon=2.5,
            source="hwsd2",
            layers=[
                SoilLayer(
                    depth_top=0.0,
                    depth_bottom=0.3,
                    sand=82.0,
                    clay=8.0,
                    silt=10.0,
                    organic_carbon=0.2,
                    bulk_density=1.55,
                    ph=7.0,
                    field_capacity=0.15,
                    wilting_point=0.05,
                ),
            ],
        ),
        2: SoilProfile(
            profile_id="P2",
            lat=12.0,
            lon=3.0,
            source="hwsd2",
            layers=[
                SoilLayer(
                    depth_top=0.0,
                    depth_bottom=0.25,
                    sand=40.0,
                    clay=20.0,
                    silt=40.0,
                    organic_carbon=0.4,
                    bulk_density=1.45,
                    ph=6.7,
                    field_capacity=0.22,
                    wilting_point=0.12,
                ),
            ],
        ),
    }
    region = Region(
        name="Test Region",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=2.0, miny=11.0, maxx=3.5, maxy=12.5),
    )
    result = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=profiles,
        country_code="CM",
        region=region,
        output_dir=tmp_path,
    )
    return result


def test_eghr_triple_symmetry_via_id_set_equivalence(tmp_path: Path) -> None:
    """The id sets across raster, profile_map, and SOL form an equivalence class.

    Reads the integer id space from each artifact and asserts:

    1. ``raster_ids == db_ids`` (raster pixel ids match profile_map.id rows)
    2. ``db_ids == sol_ids`` (profile_map.id rows match the integer id
       carried in each ``*<CC>{NNNNNNNN}`` SOL profile name)
    3. ``raster_ids == sol_ids`` (transitive: every raster id maps to a
       SOL profile name back through profile_map)

    The failure message names all three sets plus their differences
    so a future regression can be localized directly. This is the
    elevated triple-symmetry form of the AC-3/4 pair-wise pins.
    """
    result = _build_substrate(tmp_path)

    # 1) Raster pixel ids (excluding nodata).
    with rasterio.open(result.soil_raster_path) as src:
        band = src.read(1)
        nodata = src.nodata
    raster_ids = {int(v) for v in band.flatten() if v != nodata}

    # 2) profile_map ids.
    with sqlite3.connect(result.ghr_db_path) as conn:
        db_rows = conn.execute("SELECT id, profile FROM profile_map").fetchall()
    db_ids = {int(rid) for rid, _ in db_rows}
    db_id_to_profile = {int(rid): str(name) for rid, name in db_rows}

    # 3) SOL profile-id integers. The canonical writer emits names
    # of the form "{CC}{ID:08d}" (e.g., "CM00000001"); the integer
    # part is the same id stored in profile_map and as raster pixel
    # values. Parsing back from the SOL gives us the third id set.
    sol_text = result.sol_path.read_text()
    sol_ids: set[int] = set()
    for line in sol_text.splitlines():
        if line.startswith("*") and not line.startswith("*SOILS:"):
            name = line[1:11].rstrip()
            if len(name) >= 3 and name[2:].isdigit():
                sol_ids.add(int(name[2:]))

    # Triple-symmetry assertion. Build a single failure message that
    # localizes any drift to the responsible artifact pair.
    differences = []
    if raster_ids != db_ids:
        differences.append(
            f"raster vs db: only-in-raster={sorted(raster_ids - db_ids)} "
            f"only-in-db={sorted(db_ids - raster_ids)}"
        )
    if db_ids != sol_ids:
        differences.append(
            f"db vs sol: only-in-db={sorted(db_ids - sol_ids)} "
            f"only-in-sol={sorted(sol_ids - db_ids)}"
        )
    if raster_ids != sol_ids:
        differences.append(
            f"raster vs sol: only-in-raster={sorted(raster_ids - sol_ids)} "
            f"only-in-sol={sorted(sol_ids - raster_ids)}"
        )

    assert not differences, (
        "eGHR triple-symmetry violated. Id sets must form an equivalence class:\n"
        f"  raster: {sorted(raster_ids)}\n"
        f"  db:     {sorted(db_ids)}\n"
        f"  sol:    {sorted(sol_ids)}\n"
        f"Drift detail:\n  " + "\n  ".join(differences)
    )

    # Sanity check: every raster id resolves to a SOL profile name
    # via the database, asserting the round-trip the consumer
    # actually performs.
    for pid in raster_ids:
        name = db_id_to_profile[pid]
        assert name in {
            line[1:11].rstrip()
            for line in sol_text.splitlines()
            if line.startswith("*") and not line.startswith("*SOILS:")
        }, f"raster id {pid} routes through db to SOL name {name!r} not present in SOL."
