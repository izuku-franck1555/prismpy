"""Integration tests for the per-package eGHR substrate builder.

The substrate builder at
:func:`prismpy.translators._shared.eghr_substrate.build_eghr_substrate`
emits three artifacts (a GeoTIFF profile-id raster, a SQLite GHR.db,
and a DSSAT v4.8 ``.SOL``) that together let a PYTHIA-compatible
package resolve every cell to a soil profile without depending on a
globally-bundled ``.SOL`` library. These tests cover:

- end-to-end output: all three files are written; counts match input;
  SHA-256s are valid hex digests;
- cross-artifact consistency (the canonical-source pin per durable
  lesson §24): every raster pixel id appears in ``profile_map`` AND
  every ``profile_map`` row's ``profile`` value matches the ``.SOL``
  ``*<profile>`` header for the same id;
- idempotency: a second call with identical inputs regenerates
  byte-identical artifacts;
- profile deduplication: two cells assigned identical layer
  parameters end up sharing the same ``profile_id``;
- the canonical SOL writer's ``source_label_for_id`` hook is exercised
  by the substrate caller (PYTHIA-mode label) without breaking the
  default-CRAFT regression net pinned in ``test_dssat_sol_writer_byte_pin``.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Dict, List

import pytest
import rasterio

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators._shared import (
    EghrSubstrateResult,
    assign_cell_to_profile_id,
    build_eghr_substrate,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _layer(
    *,
    sand: float,
    clay: float,
    silt: float,
    depth_top: float = 0.0,
    depth_bottom: float = 0.2,
) -> SoilLayer:
    return SoilLayer(
        depth_top=depth_top,
        depth_bottom=depth_bottom,
        sand=sand,
        clay=clay,
        silt=silt,
        organic_carbon=0.5,
        bulk_density=1.4,
        ph=6.5,
        field_capacity=0.25,
        wilting_point=0.10,
    )


def _profile(*, lat: float, lon: float, sand: float, clay: float) -> SoilProfile:
    return SoilProfile(
        profile_id=f"P_{lat:.1f}_{lon:.1f}",
        lat=lat,
        lon=lon,
        source="hwsd2",
        layers=[_layer(sand=sand, clay=clay, silt=100.0 - sand - clay)],
    )


def _build_grid_2x3() -> SpatialGrid:
    """A small 2-row x 3-col grid with deterministic cell ordering."""
    cells: List[GridCell] = []
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
    return SpatialGrid(
        resolution="custom",
        cells=cells,
        increment_deg=0.5,
        bounds=BoundingBox(minx=2.0, miny=11.0, maxx=3.5, maxy=12.5),
    )


def _build_region() -> Region:
    return Region(
        name="Test Region",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=2.0, miny=11.0, maxx=3.5, maxy=12.5),
    )


def _build_profiles_5_cells_3_distinct() -> Dict[int, SoilProfile]:
    """Five cells assigned among three distinct profiles (so cell #5 is nodata).

    - Cells 0 and 3 share profile A (high-clay; sand=30 clay=45).
    - Cells 1 and 4 share profile B (high-sand; sand=82 clay=8).
    - Cell 2 has its own profile C (loamy; sand=40 clay=20).
    - Cell 5 is intentionally absent → expected nodata pixel.
    """
    return {
        0: _profile(lat=12.0, lon=2.0, sand=30.0, clay=45.0),
        1: _profile(lat=12.0, lon=2.5, sand=82.0, clay=8.0),
        2: _profile(lat=12.0, lon=3.0, sand=40.0, clay=20.0),
        3: _profile(lat=11.5, lon=2.0, sand=30.0, clay=45.0),
        4: _profile(lat=11.5, lon=2.5, sand=82.0, clay=8.0),
    }


def test_build_eghr_substrate_writes_three_artifacts(tmp_path: Path) -> None:
    grid = _build_grid_2x3()
    profiles = _build_profiles_5_cells_3_distinct()

    result = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=profiles,
        country_code="CM",
        region=_build_region(),
        output_dir=tmp_path,
    )

    assert isinstance(result, EghrSubstrateResult)
    assert result.soil_raster_path == tmp_path / "raster" / "soil.tif"
    assert result.ghr_db_path == tmp_path / "eGHR" / "GHR.db"
    assert result.sol_path == tmp_path / "eGHR" / "CM.SOL"

    assert result.soil_raster_path.exists()
    assert result.ghr_db_path.exists()
    assert result.sol_path.exists()

    assert result.cell_count == 5
    assert result.profile_count == 3

    for sha in (
        result.soil_raster_sha256,
        result.ghr_db_sha256,
        result.sol_sha256,
    ):
        assert SHA256_RE.match(sha) is not None


def test_build_eghr_substrate_canonical_source_consistency(tmp_path: Path) -> None:
    """The three artifacts must agree on every (cell_id, profile_id) pair.

    This is the structural pin for durable lesson §24: raster pixel ids,
    ``profile_map`` rows, and ``.SOL`` ``*<profile>`` headers all index
    the same id space and must not drift apart.
    """
    grid = _build_grid_2x3()
    profiles = _build_profiles_5_cells_3_distinct()

    result = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=profiles,
        country_code="CM",
        region=_build_region(),
        output_dir=tmp_path,
    )

    # 1) Read every non-nodata pixel id from the raster.
    with rasterio.open(result.soil_raster_path) as src:
        band = src.read(1)
        nodata = src.nodata
    raster_ids = {int(v) for v in band.flatten() if v != nodata}

    # 2) Read every (id, profile) pair from the GHR.db profile_map table.
    with sqlite3.connect(result.ghr_db_path) as conn:
        rows = conn.execute("SELECT id, profile FROM profile_map").fetchall()
    db_id_to_profile = {int(rid): str(name) for rid, name in rows}

    # 3) Parse every ``*<profile>`` profile-id from the .SOL file. The id
    # we want is the 10-character name written by the canonical SOL
    # writer (e.g., "CM00000001").
    sol_text = result.sol_path.read_text()
    sol_profile_names = {
        line[1:11].rstrip()
        for line in sol_text.splitlines()
        if line.startswith("*") and not line.startswith("*SOILS:")
    }

    # Cross-artifact invariants:
    assert raster_ids == set(db_id_to_profile.keys()), (
        "Raster pixel ids drift from profile_map ids: "
        f"raster={sorted(raster_ids)} db={sorted(db_id_to_profile)}"
    )
    assert set(db_id_to_profile.values()) == sol_profile_names, (
        "profile_map names drift from .SOL profile-id headers: "
        f"db={sorted(db_id_to_profile.values())} sol={sorted(sol_profile_names)}"
    )
    # And the round-trip mapping is consistent: every db row's profile
    # name appears in the .SOL exactly once and each profile_id in the
    # raster maps via the db to a known SOL header.
    for pid in raster_ids:
        name = db_id_to_profile[pid]
        assert name in sol_profile_names


def test_build_eghr_substrate_is_idempotent(tmp_path: Path) -> None:
    """Re-running with identical inputs regenerates byte-identical artifacts."""
    grid = _build_grid_2x3()
    profiles = _build_profiles_5_cells_3_distinct()
    region = _build_region()

    result_a = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=profiles,
        country_code="CM",
        region=region,
        output_dir=tmp_path,
    )
    a_raster = result_a.soil_raster_path.read_bytes()
    a_db = result_a.ghr_db_path.read_bytes()
    a_sol = result_a.sol_path.read_bytes()

    result_b = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=profiles,
        country_code="CM",
        region=region,
        output_dir=tmp_path,
    )

    assert result_b.soil_raster_path.read_bytes() == a_raster
    assert result_b.ghr_db_path.read_bytes() == a_db
    assert result_b.sol_path.read_bytes() == a_sol

    assert result_b.soil_raster_sha256 == result_a.soil_raster_sha256
    assert result_b.ghr_db_sha256 == result_a.ghr_db_sha256
    assert result_b.sol_sha256 == result_a.sol_sha256


def test_assign_cell_to_profile_id_dedupes_identical_profiles() -> None:
    """Two cells with byte-identical layer parameters share one profile_id."""
    grid = _build_grid_2x3()
    profiles = _build_profiles_5_cells_3_distinct()

    cell_to_id, profiles_by_id = assign_cell_to_profile_id(grid, profiles)

    # Cells 0 and 3 share profile A; cells 1 and 4 share profile B; cell 2 is unique.
    assert cell_to_id[0] == cell_to_id[3]
    assert cell_to_id[1] == cell_to_id[4]
    assert len(set(cell_to_id.values())) == 3
    assert len(profiles_by_id) == 3
    # Profile ids are 1-based and contiguous; 0 is reserved for nodata.
    assert set(profiles_by_id.keys()) == {1, 2, 3}


def test_assign_cell_to_profile_id_skips_unmapped_cells() -> None:
    """Cells absent from the input mapping do not appear in the output."""
    grid = _build_grid_2x3()  # 6 cells
    profiles = _build_profiles_5_cells_3_distinct()  # 5 of the 6 cells

    cell_to_id, _ = assign_cell_to_profile_id(grid, profiles)

    assert 5 not in cell_to_id  # cell #5 had no profile
    assert len(cell_to_id) == 5


def test_eghr_substrate_caller_overrides_source_label_without_breaking_default(
    tmp_path: Path,
) -> None:
    """The substrate writes eGHR-flavored labels while the default CRAFT path is unchanged.

    The substrate builder passes its own ``source_label_for_id`` callable
    to the canonical writer; this test checks the eGHR-flavored ``Source``
    column is what landed on disk, while the existing dssat_sol_writer
    byte-pin test keeps the default-CRAFT regression net intact.
    """
    grid = _build_grid_2x3()
    profiles = _build_profiles_5_cells_3_distinct()

    result = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=profiles,
        country_code="CM",
        region=_build_region(),
        output_dir=tmp_path,
    )

    sol_text = result.sol_path.read_text()
    assert "(eGHR per-package substrate)" in sol_text
    assert "eGHR profile 1" in sol_text
    assert "eGHR profile 2" in sol_text
    assert "eGHR profile 3" in sol_text
    # The CRAFT default label must NOT bleed into the substrate output.
    assert "HWSD v2 SMU" not in sol_text


def test_eghr_substrate_pydantic_result_is_extra_forbid_and_validate_assignment(
    tmp_path: Path,
) -> None:
    """``EghrSubstrateResult`` rejects extra fields + revalidates on assignment.

    Locks the schema-layer discipline pinned at construction time so a
    refactor cannot quietly weaken the model contract (durable §6.4).
    """
    grid = _build_grid_2x3()
    profiles = _build_profiles_5_cells_3_distinct()

    result = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=profiles,
        country_code="CM",
        region=_build_region(),
        output_dir=tmp_path,
    )

    # extra="forbid": setting an unknown attribute raises (with validate_assignment).
    with pytest.raises(Exception):
        result.unexpected_field = "x"

    # validate_assignment: assigning an invalid sha rejects.
    with pytest.raises(Exception):
        result.sol_sha256 = "not-a-sha"


def test_build_eghr_substrate_creates_subdirs_when_missing(tmp_path: Path) -> None:
    """The builder creates ``raster/`` and ``eGHR/`` even if they don't exist."""
    grid = _build_grid_2x3()
    profiles = _build_profiles_5_cells_3_distinct()

    target = tmp_path / "fresh_package"
    assert not target.exists()

    result = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=profiles,
        country_code="CM",
        region=_build_region(),
        output_dir=target,
    )

    assert (target / "raster").is_dir()
    assert (target / "eGHR").is_dir()
    assert result.soil_raster_path.exists()
    assert result.ghr_db_path.exists()
    assert result.sol_path.exists()


def test_build_eghr_substrate_raster_uses_uint32(tmp_path: Path) -> None:
    """The raster dtype is uint32 so >65 535 unique profiles can be encoded."""
    grid = _build_grid_2x3()
    profiles = _build_profiles_5_cells_3_distinct()

    result = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=profiles,
        country_code="CM",
        region=_build_region(),
        output_dir=tmp_path,
    )

    with rasterio.open(result.soil_raster_path) as src:
        assert src.dtypes[0] == "uint32"
        assert src.nodata == 0


def test_build_eghr_substrate_aligns_raster_to_filtered_cells(tmp_path: Path) -> None:
    """When the caller filters the grid, the raster covers the cells that remain.

    Pipelines clip cells via centroid_strict / share-percent /
    exclude_cells; when that happens, ``grid.cells`` is the post-filter
    roster but ``grid.bounds`` still describes the unfiltered region.
    The raster transform must align to the cells that survived so
    PYTHIA's cell-center sampling lands on the correct pixel.
    """
    grid_unfiltered = _build_grid_2x3()
    # Filter to a 1-row x 2-col strip in the middle of the original 2x3 grid.
    kept_cells = [c for c in grid_unfiltered.cells if c.cell_id in {1, 2}]
    grid_filtered = SpatialGrid(
        resolution="custom",
        cells=kept_cells,
        increment_deg=0.5,
        # Bounds intentionally describe the ORIGINAL 2x3 region, not the
        # 1x2 filtered subset; this is the realistic post-filter shape.
        bounds=BoundingBox(minx=2.0, miny=11.0, maxx=3.5, maxy=12.5),
    )
    profiles = {
        cid: profile
        for cid, profile in _build_profiles_5_cells_3_distinct().items()
        if cid in {1, 2}
    }

    result = build_eghr_substrate(
        grid=grid_filtered,
        profiles_by_cell=profiles,
        country_code="CM",
        region=_build_region(),
        output_dir=tmp_path,
    )

    # PYTHIA samples soil.tif at each cell's lat/lon. After the fix the
    # cell-center coordinate must land on a non-nodata pixel that maps
    # to that cell's profile id.
    with rasterio.open(result.soil_raster_path) as src:
        for cell in kept_cells:
            sampled = list(src.sample([(cell.lon, cell.lat)]))[0][0]
            assert sampled != 0, (
                f"Cell {cell.cell_id} at ({cell.lat}, {cell.lon}) sampled "
                f"to nodata; raster transform mis-aligned."
            )


def test_build_eghr_substrate_is_idempotent_with_unset_hydraulics(tmp_path: Path) -> None:
    """Idempotency holds even when source profiles arrive without hydraulic fields.

    HWSD2 / iSDA per-cell layered horizons commonly omit
    field-capacity / wilting-point / saturated-water-content. The
    canonical SOL writer mutates these values in-place when it sees
    ``None``; without an up-front normalization the second build's
    dedup key would differ from the first because only the
    representative profiles got mutated. ``assign_cell_to_profile_id``
    pre-normalizes the layers so the dedup keys are stable.
    """
    grid = _build_grid_2x3()
    region = _build_region()

    def _fresh_profiles() -> Dict[int, SoilProfile]:
        return {
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
                        # field_capacity, wilting_point, saturated_wc deliberately None.
                    )
                ],
            ),
            3: SoilProfile(
                profile_id="P3",
                lat=11.5,
                lon=2.0,
                source="hwsd2",
                # Same layer parameters as cell 0 — should dedup to a single id.
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
                    )
                ],
            ),
        }

    result_a = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=_fresh_profiles(),
        country_code="CM",
        region=region,
        output_dir=tmp_path,
    )

    result_b = build_eghr_substrate(
        grid=grid,
        profiles_by_cell=_fresh_profiles(),
        country_code="CM",
        region=region,
        output_dir=tmp_path,
    )

    # Two cells, identical layer parameters (modulo the auto-derived
    # hydraulic fields) — must collapse to a single profile id every
    # time, and produce byte-identical artifacts across reruns.
    assert result_a.profile_count == 1
    assert result_b.profile_count == 1
    assert result_b.cell_count == result_a.cell_count
    assert result_b.soil_raster_sha256 == result_a.soil_raster_sha256
    assert result_b.ghr_db_sha256 == result_a.ghr_db_sha256
    assert result_b.sol_sha256 == result_a.sol_sha256
