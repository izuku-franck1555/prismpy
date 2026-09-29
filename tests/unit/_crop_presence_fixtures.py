"""Shared fixtures for the crop-presence pins: a canonical 5-arcmin SPAM-shaped GeoTIFF around
Koutiala, and a minimal pipeline whose harmonize stage runs the grid stages with no network source.

Only modules that exist before the crop-presence rule are imported at module level, so every pin
that needs the new API fails on its own rather than at collection.
"""
from __future__ import annotations

import hashlib
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
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.provenance.tracker import ProvenanceTracker

INC = 5 / 60
# The layer spans lon [-6.25, -4.75] x lat [11.25, 12.75]: 18 x 18 pixels on the global 5' lattice,
# one pixel wider than the region box on every side so the grid's edge cells lie inside it.
WEST, NORTH, NCOLS, NROWS = -6.25, 12.75, 18, 18
V2R2_NODATA = -3.4028234663852886e38
REGION_BOX = (-6.0, 11.5, -5.0, 12.5)
REGION_WKT = "POLYGON((-6.0 11.5, -5.0 11.5, -5.0 12.5, -6.0 12.5, -6.0 11.5))"


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


def pixel_of(lat, lon):
    """(row, col) of the fixture pixel that contains (lat, lon)."""
    return int((NORTH - lat) / INC), int((lon - WEST) / INC)


def identity_of(path, *, year=2020, release="V2r2", crop_code="MAIZ"):
    """The layer's identity computed independently of the reader under test."""
    import math

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


def make_config(tmp_path, *, rule=None, rule_path=None, targets=("craft",), exclude_cells=(),
                grid_resolution="5arcmin"):
    boundary = {"source": BoundarySource.MANUAL,
                "manual_bounds": ManualBoundsConfig(minx=REGION_BOX[0], miny=REGION_BOX[1],
                                                    maxx=REGION_BOX[2], maxy=REGION_BOX[3])}
    if rule is not None:
        boundary["crop_presence"] = rule
    if rule_path is not None:
        boundary["crop_presence_path"] = str(rule_path)
    return ProjectConfig(
        project=ProjectInfo(name="crop_presence", description="crop-presence pins"),
        region=RegionConfig(name="Koutiala", country="Mali", country_iso3="MLI",
                            grid_resolution=grid_resolution, exclude_cells=list(exclude_cells),
                            boundary=BoundaryConfig(**boundary)),
        crop=CropConfig(name="Maize", name_short="mai", variety="Medium-duration",
                        calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285)),
        temporal=TemporalConfig(start_year=2015, end_year=2020, spinup_years=2),
        targets=[Platform(t) for t in targets],
        output=OutputConfig(base_dir=str(tmp_path / "out"), structure="by_platform"),
    )


def run_grid_stages(config, monkeypatch):
    """Run harmonize on the manual region; returns (stage result, boundary record, final cell ids).

    The first soil fetch is replaced by StopAfterGrid, so the stage ends right after the grid
    stages and the boundary record, with no network access."""
    final = {}

    def _stop(self, grid, region):
        final["ids"] = [c.cell_id for c in grid.cells]
        raise StopAfterGrid("stopped after the grid stages")

    monkeypatch.setattr(TranslationPipeline, "_retrieve_isda_api_for_grid", _stop)
    pipe = TranslationPipeline(config, provenance=ProvenanceTracker(enabled=True,
                                                                    project_name="crop_presence"))
    region = Region(name="Koutiala", country="Mali", country_iso3="MLI",
                    bounds=BoundingBox(minx=REGION_BOX[0], miny=REGION_BOX[1],
                                       maxx=REGION_BOX[2], maxy=REGION_BOX[3]),
                    geometry_wkt=REGION_WKT)
    region.boundary_source = "manual"
    result = pipe._execute_harmonize({"region": region})
    return result, dict(pipe.provenance.record.boundary or {}), final.get("ids")
