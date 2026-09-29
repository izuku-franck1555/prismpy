"""Shared fixtures for the crop-presence pins.

Real data: window crops of the real SPAM 2020 layers (``tests/fixtures/spam_crops``, provenance in its
PROVENANCE.txt). Synthetic GeoTIFFs only for faults a real file cannot show (a shifted transform, a
wrong CRS, +/-inf, an ordinary negative value).

Only modules that exist before the crop-presence rule are imported at module level, so every pin
that needs the new API fails on its own rather than at collection.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from prismpy.config.schema import (
    BoundaryConfig,
    BoundarySource,
    CropCalendarConfig,
    CropConfig,
    ManualBoundsConfig,
    OutputConfig,
    Platform,
    ProjectConfig,
    ProjectInfo,
    RegionConfig,
    TemporalConfig,
)
from prismpy.models.region import BoundingBox, Region
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.provenance.tracker import ProvenanceTracker

CROPS = Path(__file__).resolve().parents[1] / "fixtures" / "spam_crops"
COAST_MAIZ = CROPS / "coast_maiz_v2r2" / "spam2020_V2r2_global_H_MAIZ_A.tif"       # nodata -3.4028e38
COAST_MAIZ_V2R0 = CROPS / "coast_maiz_v2r0" / "spam2020_V2r0_global_H_MAIZ_A.tif"  # nodata NaN
INLAND_POTA = CROPS / "inland_pota_v2r2" / "spam2020_V2r2_global_H_POTA_A.tif"     # no potato mapped
# Region boxes 3 px inside each crop, so the grid's edge cells still lie on the crop.
COAST_BOX = (3.75, 5.75, 5.25, 7.25)
INLAND_BOX = (3.25, 8.25, 3.75, 8.75)

INC = 5 / 60
V2R2_NODATA = -3.4028234663852886e38
# The synthetic layer: lon [-6.25, -4.75] x lat [11.25, 12.75], 18 x 18 pixels on the 5' lattice.
WEST, NORTH, NCOLS, NROWS = -6.25, 12.75, 18, 18


class StopAfterGrid(Exception):
    """Raised in place of the first soil fetch, so harmonize stops right after the grid stages."""


def write_layer(path, values=None, *, nodata=V2R2_NODATA, dtype="float32", west=WEST, north=NORTH,
                crs="EPSG:4326", count=1):
    import rasterio
    from rasterio.transform import from_origin

    arr = np.zeros((NROWS, NCOLS), dtype=dtype) if values is None else np.asarray(values, dtype=dtype)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=count, dtype=dtype, crs=crs, transform=from_origin(west, north, INC, INC),
                       nodata=nodata) as dst:
        for band in range(1, count + 1):
            dst.write(arr, band)
    return Path(path)


def identity_of(path, *, year=2020, release="V2r2", crop_code="MAIZ"):
    """The layer's identity computed independently of the reader under test."""
    import rasterio

    with rasterio.open(path) as src:
        nodata = src.nodata
        return {
            "year": year, "release": release, "crop_code": crop_code, "stratum": "A",
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "crs": src.crs.to_string(),
            "transform": [src.transform.a, src.transform.b, src.transform.c,
                          src.transform.d, src.transform.e, src.transform.f],
            "band": 1, "width": src.width, "height": src.height,
            "nodata": None if nodata is None else ("nan" if math.isnan(nodata) else float(nodata)),
        }


def pixel_value(path, lat, lon):
    """Independent read: the value of the pixel that contains (lat, lon)."""
    import rasterio

    with rasterio.open(path) as src:
        row, col = src.index(lon, lat)
        return float(src.read(1)[row, col]), src.nodata


def mapped(path, lat, lon):
    """Harvested area > 0 at (lat, lon): finite, not the declared nodata, above zero."""
    value, nodata = pixel_value(path, lat, lon)
    return math.isfinite(value) and (nodata is None or value != nodata) and value > 0


def cell(cell_id):
    row, col = divmod(cell_id, SpatialGrid.GLOBAL_COLS_5ARCMIN)
    lat, lon = SpatialGrid.latlon_from_rowcol_5arcmin(row, col)
    return GridCell(cell_id=cell_id, lat=lat, lon=lon, row=row, col=col)


def box_cells(box):
    """The executor's full-extent 5' grid over a box (SpatialGrid.from_bounds)."""
    minx, miny, maxx, maxy = box
    return SpatialGrid.from_bounds(BoundingBox(minx=minx, miny=miny, maxx=maxx, maxy=maxy),
                                   resolution="5arcmin").cells


def make_config(tmp_path, *, rule=None, rule_path=None, targets=("craft",), exclude_cells=(),
                grid_resolution="5arcmin", box=COAST_BOX, crop=("Maize", "mai")):
    boundary = {"source": BoundarySource.MANUAL,
                "manual_bounds": ManualBoundsConfig(minx=box[0], miny=box[1], maxx=box[2], maxy=box[3])}
    if rule is not None:
        boundary["crop_presence"] = rule
    if rule_path is not None:
        boundary["crop_presence_path"] = str(rule_path)
    return ProjectConfig(
        project=ProjectInfo(name="crop_presence", description="crop-presence pins"),
        region=RegionConfig(name="Coast", country="Nigeria", country_iso3="NGA",
                            grid_resolution=grid_resolution, exclude_cells=list(exclude_cells),
                            boundary=BoundaryConfig(**boundary)),
        crop=CropConfig(name=crop[0], name_short=crop[1], variety="Medium-duration",
                        calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285)),
        temporal=TemporalConfig(start_year=2015, end_year=2020, spinup_years=2),
        targets=[Platform(t) for t in targets],
        output=OutputConfig(base_dir=str(tmp_path / "out"), structure="by_platform"),
    )


def region_of(config):
    mb = config.region.boundary.manual_bounds
    region = Region(name=config.region.name, country=config.region.country,
                    country_iso3=config.region.country_iso3,
                    bounds=BoundingBox(minx=mb.minx, miny=mb.miny, maxx=mb.maxx, maxy=mb.maxy),
                    geometry_wkt=(f"POLYGON(({mb.minx} {mb.miny}, {mb.maxx} {mb.miny}, {mb.maxx} {mb.maxy}, "
                                  f"{mb.minx} {mb.maxy}, {mb.minx} {mb.miny}))"))
    region.boundary_source = "manual"
    return region


def run_grid_stages(config, monkeypatch):
    """Run harmonize on the config's manual region; returns (stage result, boundary record,
    final cell ids, pipeline). The first soil fetch is replaced by StopAfterGrid, so the stage ends
    right after the grid stages and the boundary record, with no network access."""
    final = {}

    def _stop(self, grid, region):
        final["ids"] = [c.cell_id for c in grid.cells]
        raise StopAfterGrid("stopped after the grid stages")

    monkeypatch.setattr(TranslationPipeline, "_retrieve_isda_api_for_grid", _stop)
    pipe = TranslationPipeline(config, provenance=ProvenanceTracker(enabled=True,
                                                                    project_name="crop_presence"))
    result = pipe._execute_harmonize({"region": region_of(config)})
    return result, dict(pipe.provenance.record.boundary or {}), final.get("ids"), pipe


# Differ run to run: the clock, the session, and the config hash over the tmp output path.
VOLATILE_FIELDS = frozenset({"created_at", "session_id", "timestamp", "config_hash"})


def _scrub(value):
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items() if key not in VOLATILE_FIELDS}
    return [_scrub(item) for item in value] if isinstance(value, list) else value


def harmonize_output_digest(config, monkeypatch):
    """sha256 over what harmonize hands on for ``config``: the kept cell ids, the boundary record,
    the stage's errors, warnings and events, and the provenance record less its VOLATILE_FIELDS."""
    result, boundary, ids, pipe = run_grid_stages(config, monkeypatch)
    payload = {"ids": sorted(ids), "boundary": boundary, "errors": result.errors,
               "warnings": result.warnings, "events": getattr(result, "error_events", None),
               "provenance": _scrub(pipe.provenance.record.to_dict())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
