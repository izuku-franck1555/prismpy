"""PYTHIA's SPAM crop mask covers every grid cell's full footprint on each registered
vintage's own pixel grid, and the grid cells must sit on the canonical lattice."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from prismpy.config.schema import (
    BoundaryConfig, BoundarySource, CropCalendarConfig, CropConfig, ManualBoundsConfig,
    OutputConfig, Platform, PlatformConfigGroup, ProjectConfig, ProjectInfo, PythiaConfig,
    RegionConfig, TemporalConfig,
)
from prismpy.models.region import BoundingBox, Region
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators.base import UnifiedData
from prismpy.translators.pythia import translator as pythia

CENTRES = json.loads((Path(__file__).parents[1] / "fixtures" / "mopti_oromia_cell_centres.json").read_text())

# Each registered vintage's real global transform: its pixel is not exactly 1/12°.
VINTAGES = {
    ("2020", "V2r2"): dict(pixel=0.0833333332727273, north=90.00000000000004,
                           nodata=-3.4028234663852886e38,
                           name="spam2020_V2r2_global_H_{code}_{tech}.tif"),
    ("2010", "V2r0"): dict(pixel=0.083333, north=90.0, nodata=-1.0,
                           name="spam2010V2r0_global_H_{code}_{tech}.tif"),
}


def _write_layer(spam_dir, vintage, code, tech, bounds, values=None):
    """A window of the vintage's global raster (same pixel size and phase) over ``bounds``."""
    spec = VINTAGES[vintage]
    world = Affine(spec["pixel"], 0.0, -180.0, 0.0, -spec["pixel"], spec["north"])
    col0, row0 = ~world * (bounds[0], bounds[3])
    col1, row1 = ~world * (bounds[2], bounds[1])
    col0, row0, col1, row1 = math.floor(col0), math.floor(row0), math.ceil(col1), math.ceil(row1)
    transform = world * Affine.translation(col0, row0)
    shape = (row1 - row0, col1 - col0)
    array = np.ones(shape, dtype="float32") if values is None else values(transform, shape)
    path = spam_dir / spec["name"].format(code=code, tech=tech)
    with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=spec["nodata"]) as dst:
        dst.write(array.astype("float32"), 1)
    return path


def _grid(centres, resolution="30arcmin"):
    return SpatialGrid(resolution=resolution, cells=[
        GridCell(cell_id=i, lat=lat, lon=lon, row=i, col=i) for i, (lat, lon) in enumerate(centres)])


def _translator(tmp_path, spam_dir, vintage, crop="Millet"):
    version, release = vintage
    cfg = ProjectConfig(
        project=ProjectInfo(name="cell-crop-area", description="cell crop area"),
        region=RegionConfig(
            name="Region", country="Mali", country_iso3="MLI", grid_resolution="30arcmin",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(minx=-6.0, miny=13.0, maxx=-2.0, maxy=17.0),
                inclusion_rule="bbox_intersects",
            ),
        ),
        crop=CropConfig(name=crop, name_short="crp", variety="M",
                        calendar=CropCalendarConfig(planting_doy=180, maturity_doy=280)),
        temporal=TemporalConfig(start_year=2015, end_year=2016, spinup_years=0),
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir=str(tmp_path), structure="by_platform"),
        platform_config=PlatformConfigGroup(pythia=PythiaConfig(
            spam_raster_dir=spam_dir, spam_version=version, spam_release=release)),
    )
    return pythia.PythiaTranslator(config=cfg, output_dir=tmp_path / "out")


def _region_setup(tmp_path, vintage, region):
    centres = CENTRES[region]
    lats, lons = [c[0] for c in centres], [c[1] for c in centres]
    spam = tmp_path / "spam"
    spam.mkdir()
    code = "PMIL" if region == "mopti" else "MAIZ"
    _write_layer(spam, vintage, code, "A",
                 (min(lons) - 2, min(lats) - 2, max(lons) + 2, max(lats) + 2))
    tr = _translator(tmp_path, spam, vintage, "Millet" if region == "mopti" else "Maize")
    box = BoundingBox(minx=min(lons) - 0.25, miny=min(lats) - 0.25,
                      maxx=max(lons) + 0.25, maxy=max(lats) + 0.25)
    data = UnifiedData(region=Region(name=region, country="Mali", country_iso3="MLI", bounds=box,
                                     boundary_source="manual"),
                       grid=_grid(centres))
    return tr, data


# ── the extent, on both registered vintages' real transforms ───────────────


@pytest.mark.parametrize("vintage", list(VINTAGES))
@pytest.mark.parametrize("region", ["mopti", "oromia"])
def test_the_mask_covers_every_cell_footprint(tmp_path, vintage, region):
    tr, data = _region_setup(tmp_path, vintage, region)
    mask = tr._generate_crop_mask_raster(data)
    with rasterio.open(mask) as src:
        left, bottom, right, top = src.bounds
        for cell in data.grid.cells:
            assert left <= cell.lon - 0.25 + 1e-9 and cell.lon + 0.25 <= right + 1e-9
            assert bottom <= cell.lat - 0.25 + 1e-9 and cell.lat + 0.25 <= top + 1e-9
            row, col = src.index(cell.lon, cell.lat)
            assert 0 <= row < src.height and 0 <= col < src.width


@pytest.mark.parametrize("vintage", list(VINTAGES))
@pytest.mark.parametrize("region", ["mopti", "oromia"])
def test_unsnapped_footprint_bounds_fall_short_and_the_build_says_so(
        tmp_path, monkeypatch, vintage, region):
    """Geographic footprint bounds alone fall short of the footprints on the real pixel grids."""
    tr, data = _region_setup(tmp_path, vintage, region)
    monkeypatch.setattr(pythia.PythiaTranslator, "_snap_bounds_outward",
                        staticmethod(lambda raster_path, bounds: bounds))
    with pytest.raises(pythia.CropMaskExtentError, match="does not cover the footprint"):
        tr._generate_crop_mask_raster(data)


# ── the lattice guard, anchored to the canonical phase ─────────────────────


@pytest.mark.parametrize("centres", [
    [(3.80, 36.25)],                       # off the 0.5° lattice
    [(0.258333, 36.25), (0.758333, 36.25)],  # a uniformly shifted grid
    [(0.258333, 36.25)],                   # an off-phase single cell
])
def test_an_off_lattice_grid_is_refused(centres):
    with pytest.raises(pythia.CellGridMismatchError, match="off the 0.5° lattice"):
        pythia._check_cell_lattice(_grid(centres).cells, 0.5)


@pytest.mark.parametrize("centres", [
    [(3.75, 36.25), (4.75, 36.25)],        # a gapped grid
    [(3.75, 36.25), (3.75, 36.75), (3.75, 38.25)],  # a single row
    [(0.25, 36.25)],                       # an aligned single cell
    [tuple(c) for c in CENTRES["mopti"]],
    [tuple(c) for c in CENTRES["oromia"]],
])
def test_grid_cells_on_the_lattice_pass(centres):
    pythia._check_cell_lattice(_grid(centres).cells, 0.5)


def test_the_build_refuses_an_off_lattice_grid(tmp_path):
    tr, data = _region_setup(tmp_path, ("2020", "V2r2"), "mopti")
    data.grid = _grid([(14.80, -4.25), (15.25, -4.25)])
    with pytest.raises(pythia.CellGridMismatchError):
        tr._generate_crop_mask_raster(data)


# ── unchanged surfaces ─────────────────────────────────────────────────────

# The AST of _clip_global_raster at prismpy 74af733 (the legacy soil clip shares it).
CLIP_AST_SHA256 = "1e93f9c802edf2d6d62216198a4ccdaef3f753bb398c03a8814dcd40f9015d94"


def test_the_shared_clip_is_unchanged():
    source = textwrap.dedent(inspect.getsource(pythia.PythiaTranslator._clip_global_raster))
    digest = hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest()
    assert digest == CLIP_AST_SHA256


def test_the_applied_vintage_is_unchanged(tmp_path):
    tr, data = _region_setup(tmp_path, ("2020", "V2r2"), "mopti")
    tr._generate_crop_mask_raster(data)
    from prismpy.sources.crop_areas.spam_vintage import AppliedVintage
    assert tr._applied_vintage == AppliedVintage(
        year="2020", release="V2r2", source_filename="spam2020_V2r2_global_H_PMIL_A.tif",
        mask_filename="harvest_area_2020_V2r2.tif")


def test_only_the_pythia_translator_and_tests_change():
    repo = Path(__file__).parents[2]
    try:
        base = subprocess.run(["git", "merge-base", "HEAD", "origin/main"], cwd=repo, check=True,
                              capture_output=True, text=True).stdout.strip()
        changed = subprocess.run(["git", "diff", "--name-only", base, "HEAD"], cwd=repo,
                                 check=True, capture_output=True, text=True).stdout.split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("origin/main is not available in this checkout")
    allowed = {"src/prismpy/translators/pythia/translator.py"}
    assert [p for p in changed if p not in allowed and not p.startswith("tests/")] == []
