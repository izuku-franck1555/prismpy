"""Structural pins for the eGHR substrate's SQLite schema and raster <-> DB invariant.

These tests treat the GHR.db schema and the raster <-> profile_map
relationship as cross-boundary invariants and pin them via direct
inspection of the produced artifacts (durable lesson §24,
canonical-source-or-pin). PYTHIA's existing ``profile_map`` consumer
relies on the column shape; if a future refactor drops the NOT NULL
constraint, renames a column, or lets the writer emit raster pixels
that have no matching ``profile_map`` row (orphan pixels), every
package built afterwards would silently break for the consumer.

The schema is read with ``PRAGMA table_info(profile_map)`` rather than
asserted by Python-side imperative checks (per durable lesson §6.4
schema-layer discipline — the SQLite schema is the canonical source;
the Python writer should not be re-asserting what the schema already
encodes).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List

import rasterio

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators._shared import build_eghr_substrate


# Canonical schema: profile_map(id INTEGER PRIMARY KEY, profile TEXT NOT NULL).
# Recorded as the producer's expected output of ``PRAGMA table_info``. SQLite
# reports ``(cid, name, type, notnull, dflt_value, pk)`` per column; the
# values below are what every eGHR substrate ships, in column-id order.
_EXPECTED_TABLE_INFO: List[tuple] = [
    # cid, name,      type,      notnull, dflt_value, pk
    (0,    "id",      "INTEGER", 0,       None,       1),
    (1,    "profile", "TEXT",    1,       None,       0),
]


def _build_substrate(tmp_path: Path) -> "Path":
    """Build a tiny three-cell, two-profile substrate and return the GHR.db path."""
    cells = [
        GridCell(cell_id=0, lat=12.0, lon=2.0, row=0, col=0, resolution="custom"),
        GridCell(cell_id=1, lat=12.0, lon=2.5, row=0, col=1, resolution="custom"),
        GridCell(cell_id=2, lat=12.0, lon=3.0, row=0, col=2, resolution="custom"),
    ]
    grid = SpatialGrid(
        resolution="custom",
        cells=cells,
        increment_deg=0.5,
        bounds=BoundingBox(minx=2.0, miny=11.5, maxx=3.5, maxy=12.5),
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
                    depth_bottom=0.2,
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
        # Cell 2 deliberately absent so the raster carries a nodata pixel
        # that the no-orphan-pixel pin will exclude from its membership check.
    }
    region = Region(
        name="Test Region",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=2.0, miny=11.5, maxx=3.5, maxy=12.5),
    )
    result = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=profiles,
        country_code="CM",
        region=region,
        output_dir=tmp_path,
    )
    return result.ghr_db_path, result.soil_raster_path


def test_ghr_db_profile_map_schema_pinned(tmp_path: Path) -> None:
    """``profile_map`` column shape must match the canonical schema exactly.

    Every PYTHIA package's GHR.db must satisfy the canonical schema
    ``profile_map(id INTEGER PRIMARY KEY, profile TEXT NOT NULL)``.
    PRAGMA table_info returns ``(cid, name, type, notnull, dflt_value,
    pk)`` per column; this test pins the full row for both columns so
    a future refactor that drops NOT NULL on ``profile``, renames
    columns, or changes the type fails loud.
    """
    db_path, _ = _build_substrate(tmp_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("PRAGMA table_info(profile_map)").fetchall()

    assert rows == _EXPECTED_TABLE_INFO, (
        "GHR.db profile_map schema drifted from the canonical column layout.\n"
        f"Expected: {_EXPECTED_TABLE_INFO}\n"
        f"Actual:   {rows}\n"
        "If this is an intentional schema change, update _EXPECTED_TABLE_INFO "
        "and check every consumer of profile_map (PYTHIA translator, "
        "prismpy/sources/soil/eghr.py)."
    )


def test_ghr_db_only_has_profile_map_table(tmp_path: Path) -> None:
    """The substrate must not bundle stray tables (sqlite_sequence etc.).

    PYTHIA's consumer queries ``profile_map`` directly; an extra table
    from an inadvertent AUTOINCREMENT or a debug helper would not break
    PYTHIA but would expand the SHA-256 surface area and weaken
    byte-identical idempotency guarantees. This pin keeps the database
    schema minimal.
    """
    db_path, _ = _build_substrate(tmp_path)

    with sqlite3.connect(db_path) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]

    assert tables == ["profile_map"], (
        f"GHR.db carries unexpected tables: {tables}. "
        "The substrate writer should only create profile_map."
    )


def test_no_orphan_pixels_in_soil_raster(tmp_path: Path) -> None:
    """Every non-nodata raster pixel id has a matching ``profile_map`` row.

    PYTHIA samples soil.tif at each cell's coordinates, reads the
    integer pixel value, and looks it up in ``profile_map`` to find
    the soil profile name. An orphan pixel id (in the raster but
    missing from the database) returns NULL on lookup, which silently
    falls back to a default profile and corrupts the simulation. This
    pin asserts the producer-side invariant explicitly.
    """
    db_path, raster_path = _build_substrate(tmp_path)

    with rasterio.open(raster_path) as src:
        band = src.read(1)
        nodata = src.nodata
    raster_ids = {int(v) for v in band.flatten() if v != nodata}

    with sqlite3.connect(db_path) as conn:
        db_ids = {
            int(row[0])
            for row in conn.execute("SELECT id FROM profile_map").fetchall()
        }

    orphans = raster_ids - db_ids
    assert not orphans, (
        f"Raster carries pixel ids with no matching profile_map row: "
        f"{sorted(orphans)}. PYTHIA's lookup against these ids will return "
        "NULL and silently fall back to a default profile."
    )


def test_no_unused_profile_map_rows(tmp_path: Path) -> None:
    """Every ``profile_map`` row appears as at least one raster pixel.

    The reverse of the orphan-pixel invariant: a profile_map row that
    no raster pixel references is dead weight that bloats the
    substrate. This is not strictly fatal for PYTHIA (the consumer
    only queries ids it sees in the raster), but it indicates the
    producer's deduplication or assignment logic shipped extra
    profiles beyond what the raster needs.
    """
    db_path, raster_path = _build_substrate(tmp_path)

    with rasterio.open(raster_path) as src:
        band = src.read(1)
        nodata = src.nodata
    raster_ids = {int(v) for v in band.flatten() if v != nodata}

    with sqlite3.connect(db_path) as conn:
        db_ids = {
            int(row[0])
            for row in conn.execute("SELECT id FROM profile_map").fetchall()
        }

    unused = db_ids - raster_ids
    assert not unused, (
        f"profile_map carries rows with no raster pixel referencing them: "
        f"{sorted(unused)}. The producer is shipping more profiles than the "
        "raster uses."
    )


def _build_substrate_with_sol(tmp_path: Path):
    """Build the same tiny substrate as ``_build_substrate`` and return
    all three artifact paths (raster, GHR.db, .SOL).
    """
    db_path, raster_path = _build_substrate(tmp_path)
    sol_path = db_path.parent / "CM.SOL"
    return raster_path, db_path, sol_path


def _parse_sol_profile_names(sol_path: Path) -> set:
    """Read ``*<name>`` profile-block headers from a .SOL file.

    Returns the 10-character profile names (rstripped) the canonical
    SOL writer emits for each profile block. Skips the file-level
    ``*SOILS:`` banner.
    """
    text = sol_path.read_text()
    return {
        line[1:11].rstrip()
        for line in text.splitlines()
        if line.startswith("*") and not line.startswith("*SOILS:")
    }


def test_every_db_profile_name_appears_as_sol_block_header(tmp_path: Path) -> None:
    """Every ``profile_map.profile`` value appears as a ``*<name>`` header.

    The producer-side guarantee that PYTHIA's lookup chain works
    end-to-end: a raster pixel returns an id, the id returns a name
    via ``profile_map``, and the name resolves to a profile block in
    ``{CC}.SOL``. If any DB row has no matching SOL header, the
    PYTHIA simulation hits the lookup chain and fails to find the
    profile content; this pin asserts the producer never ships that
    state.
    """
    raster_path, db_path, sol_path = _build_substrate_with_sol(tmp_path)

    with sqlite3.connect(db_path) as conn:
        db_profile_names = {
            str(row[0])
            for row in conn.execute("SELECT profile FROM profile_map").fetchall()
        }

    sol_profile_names = _parse_sol_profile_names(sol_path)

    missing = db_profile_names - sol_profile_names
    assert not missing, (
        f"profile_map carries names that have no matching ``*<name>`` block "
        f"in the .SOL: {sorted(missing)}. PYTHIA's lookup chain (raster -> "
        f"profile_map -> SOL) will fail for any of these names."
    )


def test_every_sol_block_header_appears_in_profile_map(tmp_path: Path) -> None:
    """Every ``*<name>`` SOL block has a matching ``profile_map`` row.

    The reverse of the previous pin: a SOL block whose name is absent
    from ``profile_map`` is a profile that no raster pixel can ever
    reach (because the lookup goes raster -> profile_map -> SOL, and
    a missing profile_map row means no raster pixel can resolve to
    that name). Such a block is dead weight in the .SOL — it bloats
    the file, weakens byte-identical idempotency guarantees, and
    indicates the SOL writer received profiles the substrate builder
    never registered.
    """
    raster_path, db_path, sol_path = _build_substrate_with_sol(tmp_path)

    with sqlite3.connect(db_path) as conn:
        db_profile_names = {
            str(row[0])
            for row in conn.execute("SELECT profile FROM profile_map").fetchall()
        }

    sol_profile_names = _parse_sol_profile_names(sol_path)

    orphan_blocks = sol_profile_names - db_profile_names
    assert not orphan_blocks, (
        f".SOL contains ``*<name>`` blocks with no matching profile_map row: "
        f"{sorted(orphan_blocks)}. No raster pixel can reach these blocks."
    )


def test_sol_block_count_matches_profile_map_row_count(tmp_path: Path) -> None:
    """The .SOL has exactly one ``*<name>`` block per ``profile_map`` row.

    Ties the membership-symmetry pins together with a count assertion.
    Two membership pins (DB ⊆ SOL and SOL ⊆ DB) collectively imply
    name-set equality, but a count mismatch — for example, a SOL
    writer that emitted the same profile block twice — would slip
    through both membership checks (the duplicate name is in both
    sets). This pin catches duplicate emission at the .SOL level.
    """
    raster_path, db_path, sol_path = _build_substrate_with_sol(tmp_path)

    with sqlite3.connect(db_path) as conn:
        db_row_count = int(
            conn.execute("SELECT COUNT(*) FROM profile_map").fetchone()[0]
        )

    sol_block_count = sum(
        1
        for line in sol_path.read_text().splitlines()
        if line.startswith("*") and not line.startswith("*SOILS:")
    )

    assert sol_block_count == db_row_count, (
        f".SOL block count ({sol_block_count}) differs from profile_map row "
        f"count ({db_row_count}). The producer is emitting a different number "
        "of profile blocks than the database registered."
    )
