"""PYTHIA's SPAM crop mask covers every grid cell's full footprint on each registered
vintage's own pixel grid, and the grid cells must sit on the canonical lattice."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from prismpy.config.schema import (
    BoundaryConfig, BoundarySource, CropCalendarConfig, CropConfig, ManagementConfig,
    ManualBoundsConfig,
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

# The source of _clip_global_raster at prismpy 74af733 (the legacy soil clip shares it),
# line endings and trailing spaces normalised, so the digest is the same on every Python.
CLIP_SOURCE_SHA256 = "5861f82d0b8e271cba14057e74ca2d80966512ab9b81f4173b22ae1307ed8adc"


def test_the_shared_clip_is_unchanged():
    source = textwrap.dedent(inspect.getsource(pythia.PythiaTranslator._clip_global_raster))
    text = "\n".join(line.rstrip() for line in source.splitlines())
    assert hashlib.sha256(text.encode()).hexdigest() == CLIP_SOURCE_SHA256


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


# ── the per-cell crop-area file ─────────────────────────────────────────────

V2R2 = ("2020", "V2r2")
CELLS = [(10.25, 36.25), (10.25, 36.75)]


def _one_pixel(per_cell):
    """Layer values: each cell's whole value in one pixel (so every sum is exact)."""
    def values(transform, shape):
        array = np.zeros(shape, dtype="float64")
        for (lat, lon), value in per_cell.items():
            row, col = rasterio.transform.rowcol(transform, lon - 0.08, lat + 0.08)
            array[row, col] = value
        return array
    return values


def _build(tmp_path, cells, layers, *, wkt=None, source="gadm", crop="Maize",
           irrigation=False, vintage=V2R2, run=True):
    """A translator + data over ``cells``; ``layers`` maps a technology to a values function."""
    spam = tmp_path / "spam"
    spam.mkdir(parents=True, exist_ok=True)
    lats, lons = [c[0] for c in cells] or [10.25], [c[1] for c in cells] or [36.25]
    window = (min(lons) - 1, min(lats) - 1, max(lons) + 1, max(lats) + 1)
    code = {"Maize": "MAIZ"}.get(crop, "PMIL")
    for tech, values in layers.items():
        _write_layer(spam, vintage, code, tech, window, values)
    tr = _translator(tmp_path, spam, vintage, crop)
    if irrigation:
        tr.config.management = ManagementConfig(planting_density=62500, irrigation=True)
    box = BoundingBox(minx=min(lons) - 0.25, miny=min(lats) - 0.25,
                      maxx=max(lons) + 0.25, maxy=max(lats) + 0.25)
    region = Region(name="Region", country="Ethiopia", country_iso3="ETH", bounds=box,
                    geometry_wkt=wkt if wkt is not None else box_wkt(box), boundary_source=source)
    data = UnifiedData(region=region, grid=_grid(cells))
    if run:
        tr._generate_crop_mask_raster(data)
    return tr, data


def box_wkt(b):
    return f"POLYGON(({b.minx} {b.miny}, {b.maxx} {b.miny}, {b.maxx} {b.maxy}, {b.minx} {b.maxy}, {b.minx} {b.miny}))"


COLUMNS = ("lat", "lon", "crop_area_ha", "crop_area_full_cell_ha", "cell_area_ha", "h_j",
           "irrigated_share", "technology", "spam_crop_code", "spam_version", "spam_release",
           "crop_area_rule")


def _csv(tr, crop="Maize"):
    import csv as _csv_module
    path = tr.output_dir / "data" / "masks" / f"{crop.lower().replace(' ', '_')}_harvested_area.csv"
    with open(path) as f:
        reader = _csv_module.DictReader(f)
        return reader.fieldnames, list(reader)


@pytest.mark.parametrize("region", ["mopti", "oromia"])
def test_one_clean_row_per_grid_cell(tmp_path, region):
    tr, data = _region_setup(tmp_path, V2R2, region)
    tr._generate_crop_mask_raster(data)
    crop = "Millet" if region == "mopti" else "Maize"
    header, rows = _csv(tr, crop)
    assert tuple(header) == COLUMNS
    assert [(r["lat"], r["lon"]) for r in rows] == [
        (f"{lat:.6f}", f"{lon:.6f}") for lat, lon in CENTRES[region]]
    for row in rows:
        for column in ("crop_area_ha", "crop_area_full_cell_ha", "cell_area_ha", "h_j"):
            assert row[column] != "" and math.isfinite(float(row[column]))


def test_an_empty_grid_writes_the_header_with_the_mask(tmp_path):
    tr, data = _build(tmp_path, [], {"A": None}, run=False)
    data.grid = None
    assert tr._generate_crop_mask_raster(data) is not None
    header, rows = _csv(tr)
    assert tuple(header) == COLUMNS and rows == []


# ── in-region sums from the global raster ──────────────────────────────────


def _pixels(values_at):
    """Layer values from a function of the pixel centre (lon, lat)."""
    def values(transform, shape):
        rows, cols = np.indices(shape)
        lons, lats = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
        return np.array([values_at(x, y) for x, y in zip(lons, lats)]).reshape(shape)
    return values


def test_off_centre_crop_and_the_whole_footprint_count(tmp_path):
    """(i) crop only off the centre pixel; (ii) an edge cell's crop in its outer column."""
    layer = _pixels(lambda x, y: 7.0 if (36.0 < x < 36.09 and 10.41 < y < 10.5)
                    else 5.0 if (37.41 < x < 37.5) else 0.0)
    tr, _ = _build(tmp_path, [(10.25, 36.25), (10.25, 37.25)], {"A": layer})
    _, rows = _csv(tr)
    assert float(rows[0]["crop_area_ha"]) == 7.0
    assert float(rows[1]["crop_area_ha"]) == 30.0   # the east column: 6 pixels of 5


def test_nodata_nan_negative_and_empty_cells_count_zero(tmp_path):
    nodata = VINTAGES[V2R2]["nodata"]
    layer = _pixels(lambda x, y: (np.nan if x < 36.09 and y > 10.41 else nodata if x < 36.09
                                  else -4.0 if y > 10.41 else 1.0) if x < 36.5 else 0.0)
    tr, _ = _build(tmp_path, CELLS, {"A": layer})
    _, rows = _csv(tr)
    assert float(rows[0]["crop_area_ha"]) == 25.0   # 36 pixels less 11 nodata/NaN/negative
    assert (float(rows[1]["crop_area_ha"]), float(rows[1]["h_j"])) == (0.0, 0.0)


def _raster(path, pixel, north, west, shape, value, nodata=None):
    transform = Affine(pixel, 0.0, west, 0.0, -pixel, north)
    with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform, nodata=nodata) as dst:
        dst.write(np.asarray(value, dtype="float32") * np.ones(shape, dtype="float32"), 1)
    return path


def test_a_positive_nodata_value_counts_zero(tmp_path):
    from shapely.geometry import box
    array = np.ones((12, 12), dtype="float32")
    array[:6, :6] = 9999.0
    path = _raster(tmp_path / "layer.tif", 1 / 12, 10.5, 36.0, (12, 12), array, nodata=9999.0)
    sums = pythia._cell_block_sums(path, _grid(CELLS[:1]).cells, 0.25, box(35, 9, 38, 12))
    assert sums == [(0.0, 0.0)]


def test_footprint_windows_round_to_the_nearest_pixel(tmp_path):
    """A pixel a hair wider than 1/12°: the footprint spans 5.9999 pixels, rounded to 6."""
    from shapely.geometry import box
    pixel = 1 / 12 + 1e-9
    # a small window keeping the global lattice phase: 2580 columns east, 900 rows south
    path = _raster(tmp_path / "layer.tif", pixel, 90.0 - 900 * pixel, -180.0 + 2580 * pixel,
                   (120, 120), 1.0)
    sums = pythia._cell_block_sums(path, _grid(CELLS[:1]).cells, 0.25, box(35, 9, 38, 12))
    assert sums == [(36.0, 36.0)]


def test_only_pixels_whose_centre_is_inside_the_region_count(tmp_path):
    """(v) a cell whose crop is wholly outside; (vi) a pixel that overlaps but is centred outside."""
    region = "POLYGON((36.0 10.0, 36.51 10.0, 36.51 10.5, 36.0 10.5, 36.0 10.0))"
    tr, _ = _build(tmp_path, CELLS, {"A": _pixels(lambda x, y: 1.0)}, wkt=region)
    _, rows = _csv(tr)
    assert float(rows[0]["crop_area_ha"]) == 36.0
    assert float(rows[1]["crop_area_ha"]) == 0.0 and float(rows[1]["h_j"]) == 0.0
    assert float(rows[1]["crop_area_full_cell_ha"]) == 36.0
    assert all(r["crop_area_rule"] == "pixel_centre_in_region_polygon" for r in rows)


def test_a_manual_box_without_a_polygon_uses_its_bounds(tmp_path):
    tr, data = _build(tmp_path, CELLS, {"A": _pixels(lambda x, y: 1.0)}, run=False)
    data.region.geometry_wkt, data.region.boundary_source = None, "manual"
    tr._generate_crop_mask_raster(data)
    _, rows = _csv(tr)
    assert {r["crop_area_rule"] for r in rows} == {"pixel_centre_in_region_bounds"}
    assert [float(r["crop_area_ha"]) for r in rows] == [36.0, 36.0]


@pytest.mark.parametrize("wkt, source", [
    ("POLYGON((36 10, 37", "gadm"),
    ("POLYGON EMPTY", "gadm"),
    ("GEOMETRYCOLLECTION EMPTY", "gadm"),
    (None, "gadm"),
])
def test_a_region_without_a_usable_polygon_fails_before_writing(tmp_path, wkt, source):
    tr, data = _build(tmp_path, CELLS, {"A": _pixels(lambda x, y: 1.0)}, run=False)
    if wkt is None:
        data.region = Region.from_dict(data.region.to_dict())   # the round trip drops the WKT
        assert data.region.geometry_wkt is None and data.region.boundary_source == "gadm"
    else:
        data.region.geometry_wkt = wkt
    with pytest.raises(pythia.RegionGeometryError):
        tr._generate_crop_mask_raster(data)
    assert not list(tr.output_dir.rglob("*.csv")) and not list(tr.output_dir.rglob("*.tif"))


def test_the_sums_read_the_global_layers_only(tmp_path, monkeypatch):
    tr, data = _build(tmp_path, CELLS, {tech: _pixels(lambda x, y: 1.0) for tech in "AIR"}, run=False)
    opened = []
    real_open = rasterio.open
    monkeypatch.setattr(rasterio, "open", lambda path, *a, **k: opened.append(Path(path)) or real_open(path, *a, **k))
    pythia_config = tr.config.platform_config.pythia
    tr._write_cell_crop_area(data, Path(pythia_config.spam_raster_dir), "2020", "V2r2", "MAIZ")
    assert opened and {p.parent for p in opened} == {tmp_path / "spam"}
    assert {p.name for p in opened} == {f"spam2020_V2r2_global_H_MAIZ_{t}.tif" for t in "AIR"}


# ── the fraction and the weights ───────────────────────────────────────────


def test_cell_area_is_the_spherical_footprint_and_h_j_is_never_clamped(tmp_path):
    area = 6_371_008.8 ** 2 * math.radians(0.5) * (
        math.sin(math.radians(10.5)) - math.sin(math.radians(10.0))) / 1e4
    tr, _ = _build(tmp_path, CELLS[:1], {"A": _one_pixel({CELLS[0]: 1.6 * area})})
    _, rows = _csv(tr)
    assert float(rows[0]["cell_area_ha"]) == pytest.approx(area, rel=1e-12)
    assert float(rows[0]["h_j"]) == pytest.approx(1.6, rel=1e-6)
    assert float(rows[0]["h_j"]) == float(rows[0]["crop_area_ha"]) / float(rows[0]["cell_area_ha"])


def test_uc1_weights_follow_the_in_region_crop_area(tmp_path):
    centres = [tuple(c) for c in CENTRES["oromia"]]
    crop = {c: 100.0 + 7 * i for i, c in enumerate(centres)}
    tr, _ = _build(tmp_path, centres, {"A": _one_pixel(crop)})
    _, rows = _csv(tr)
    weights = [84.92 * math.cos(math.radians(float(r["lat"]))) * float(r["h_j"]) for r in rows]
    areas = [float(r["crop_area_ha"]) for r in rows]
    for w, a in zip(weights, areas):
        assert w / sum(weights) == pytest.approx(a / sum(areas), rel=1e-9)


# ── the technology, decided once per package on in-region sums ─────────────

A, B = CELLS


def _tech(tmp_path, a, i=None, r=None, **kw):
    layers = {"A": _one_pixel(a)}
    if i is not None:
        layers["I"] = _one_pixel(i)
    if r is not None:
        layers["R"] = _one_pixel(r)
    tr, _ = _build(tmp_path, CELLS, layers, **kw)
    return _csv(tr)[1]


@pytest.mark.parametrize("a, i, r, irrigation, expected", [
    ({A: 1000}, {A: 50}, {A: 950}, False, "A"),                           # (i) 5%, no I > R
    ({A: 1000, B: 500}, {A: 150}, {A: 850, B: 500}, False, "R"),          # (ii) 15%, rainfed
    ({A: 245245, B: 5005}, {B: 5000}, {A: 245245, B: 5}, False, "R"),     # (iii) the sliver
    ({A: 99100, B: 900}, {B: 600}, {A: 99100, B: 300}, False, "A"),       # (iv) 0.9% of A
    ({A: 1000}, {A: 150}, {A: 850}, True, "I"),                           # (v) irrigated run
    ({A: 1000}, {A: 100}, {A: 900}, False, "R"),                          # (ix) exactly 10%
    ({A: 990, B: 10}, {B: 8}, {A: 990, B: 2}, False, "R"),                # (x) exactly 1%
    ({A: 988, B: 12}, {B: 7}, {A: 988, B: 5}, False, "R"),                # (xi) the numerator is A
])
def test_the_technology_follows_material_irrigation(tmp_path, a, i, r, irrigation, expected):
    rows = _tech(tmp_path, a, i, r, irrigation=irrigation)
    assert {row["technology"] for row in rows} == {expected}
    layer = {"A": a, "I": i, "R": r}[expected]
    assert [float(row["crop_area_ha"]) for row in rows] == [layer.get(A, 0), layer.get(B, 0)]


def test_a_missing_irrigated_or_rainfed_layer_keeps_all_technologies(tmp_path):
    rows = _tech(tmp_path / "noI", {A: 1000}, None, {A: 100})
    assert {row["technology"] for row in rows} == {"A"} and rows[0]["irrigated_share"] == ""
    rows = _tech(tmp_path / "noR", {A: 1000}, {A: 900}, None)
    assert {row["technology"] for row in rows} == {"A"}
    assert float(rows[0]["irrigated_share"]) == 0.9


def test_irrigated_crop_outside_the_region_does_not_count(tmp_path):
    """(vii) the only I > R cell's crop lies outside the polygon."""
    region = "POLYGON((36.0 10.0, 36.5 10.0, 36.5 10.5, 36.0 10.5, 36.0 10.0))"
    rows = _tech(tmp_path, {A: 1000, B: 500}, {B: 500}, {A: 1000}, wkt=region)
    assert {row["technology"] for row in rows} == {"A"}
    assert rows[1]["irrigated_share"] == ""


def test_an_all_zero_region_keeps_all_technologies_and_says_so(tmp_path, caplog):
    rows = _tech(tmp_path, {}, {}, {})
    assert {row["technology"] for row in rows} == {"A"}
    assert [float(row["h_j"]) for row in rows] == [0.0, 0.0]
    assert "No grid cell holds MAIZ crop inside the region" in caplog.text


def test_an_unregistered_vintage_still_raises(tmp_path):
    from prismpy.sources.crop_areas.spam_vintage import VintageNotRegisteredError
    tr, data = _build(tmp_path, CELLS, {"A": _pixels(lambda x, y: 1.0)}, run=False)
    tr.config.platform_config.pythia.spam_release = "V2r0"   # 2020 V2r0 is not registered
    with pytest.raises(VintageNotRegisteredError):
        tr._generate_crop_mask_raster(data)


# ── the file name ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("crop, code", [("Maize", "MAIZ"), ("Millet", "PMIL"), ("Pearl Millet", "PMIL")])
def test_the_file_name_matches_uc1_and_the_crop_code_is_recorded(tmp_path, crop, code):
    tr, _ = _build(tmp_path, CELLS, {"A": _pixels(lambda x, y: 1.0)}, crop=crop)
    stem = crop.lower().strip().replace(" ", "_") + "_harvested_area"
    assert [p.stem for p in (tr.output_dir / "data" / "masks").iterdir()] == [stem]
    assert {row["spam_crop_code"] for row in _csv(tr, crop)[1]} == {code}


# ── both files or neither ──────────────────────────────────────────────────


def _outputs(tr):
    return sorted(str(p.relative_to(tr.output_dir)) for p in tr.output_dir.rglob("*") if p.is_file())


def _seeded(tmp_path):
    """A directory an earlier SPAM build filled, plus files this step does not own."""
    tr, data = _build(tmp_path, CELLS, {"A": _pixels(lambda x, y: 1.0)})
    (tr.output_dir / "raster" / "soil.tif").write_bytes(b"soil")
    (tr.output_dir / "data" / "masks" / "notes.csv").write_text("x\n")
    assert _outputs(tr) == ["data/masks/maize_harvested_area.csv", "data/masks/notes.csv",
                            "raster/harvest_area_2020_V2r2.tif", "raster/soil.tif"]
    return tr, data


OTHERS = ["data/masks/notes.csv", "raster/soil.tif"]


def test_both_files_come_from_the_applied_vintage(tmp_path):
    tr, _ = _build(tmp_path, CELLS, {"A": _pixels(lambda x, y: 1.0)})
    rows = _csv(tr)[1]
    assert {(r["spam_version"], r["spam_release"]) for r in rows} == {
        (tr._applied_vintage.year, tr._applied_vintage.release)}
    assert (tr.output_dir / "raster" / tr._applied_vintage.mask_filename).exists()


def test_without_spam_a_reused_directory_loses_the_stale_pair_only(tmp_path):
    tr, data = _seeded(tmp_path)
    tr.config.platform_config.pythia.spam_raster_dir = None
    assert tr._generate_crop_mask_raster(data) is None
    assert _outputs(tr) == OTHERS


@pytest.mark.parametrize("fault", ["empty_wkt", "bad_wkt", "off_lattice", "unregistered", "csv_write"])
def test_a_failed_spam_build_leaves_neither_file(tmp_path, monkeypatch, fault):
    tr, data = _seeded(tmp_path)
    if fault == "empty_wkt":
        data.region.geometry_wkt = "POLYGON EMPTY"
    elif fault == "bad_wkt":
        data.region.geometry_wkt = "POLYGON((36 10"
    elif fault == "off_lattice":
        data.grid = _grid([(10.30, 36.25)])
    elif fault == "unregistered":
        tr.config.platform_config.pythia.spam_release = "V2r0"
    else:
        class Failing(pythia.csv.DictWriter):
            def writerows(self, rows):
                raise OSError("disk full")
        monkeypatch.setattr(pythia.csv, "DictWriter", Failing)
    with pytest.raises(Exception):
        tr._generate_crop_mask_raster(data)
    assert _outputs(tr) == OTHERS


def _replace_spy(monkeypatch, tr, fail_at=None):
    calls, real_replace = [], os.replace

    def spy(src, dst):
        calls.append(Path(dst).name)
        if len(calls) == fail_at:
            if fail_at == 2:
                assert (tr.output_dir / "raster" / "harvest_area_2020_V2r2.tif").exists()
            raise OSError("publish failed")
        real_replace(src, dst)
    monkeypatch.setattr(pythia.os, "replace", spy)
    return calls


def test_the_mask_is_published_before_the_file(tmp_path, monkeypatch):
    tr, data = _build(tmp_path, CELLS, {"A": _pixels(lambda x, y: 1.0)}, run=False)
    calls = _replace_spy(monkeypatch, tr)
    tr._generate_crop_mask_raster(data)
    assert calls == ["harvest_area_2020_V2r2.tif", "maize_harvested_area.csv"]


@pytest.mark.parametrize("fail_at", [1, 2])
def test_a_failed_publish_leaves_neither_file(tmp_path, monkeypatch, fail_at):
    tr, data = _seeded(tmp_path)
    _replace_spy(monkeypatch, tr, fail_at)
    with pytest.raises(OSError, match="publish failed"):
        tr._generate_crop_mask_raster(data)
    assert _outputs(tr) == OTHERS
