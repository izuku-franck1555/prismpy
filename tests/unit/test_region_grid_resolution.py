"""Configurable site-grid resolution (``region.grid_resolution``).

The pipeline previously hardcoded the site grid to 5-arcmin. UC2's ISIMIP3b
climate-scenario ensemble needs a 0.5° (30-arcmin) grid aligned to the ISIMIP
cells. ``region.grid_resolution`` makes it config-driven, defaulting to
5-arcmin so every existing UC is unaffected.
"""

from __future__ import annotations

import inspect

import pytest

from prismpy.config.schema import (
    BoundaryConfig,
    BoundarySource,
    ManualBoundsConfig,
    RegionConfig,
)
from prismpy.models.region import BoundingBox
from prismpy.models.spatial import SpatialGrid

# The UC2 ISIMIP3b-aligned 40-cell grid: 4 lat × 10 lon at 0.5° centers
# (k·0.5 + 0.25), spanning the half-cell-tightened Kano-belt bbox.
_UC2_BBOX = BoundingBox(minx=5.75, miny=11.25, maxx=10.25, maxy=12.75)
_UC2_LAT = [11.25, 11.75, 12.25, 12.75]
_UC2_LON = [5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25]


def _region(**kwargs) -> RegionConfig:
    return RegionConfig(
        name="Kano",
        country="Nigeria",
        country_iso3="NGA",
        boundary=BoundaryConfig(
            source=BoundarySource.MANUAL,
            manual_bounds=ManualBoundsConfig(
                minx=5.75, miny=11.25, maxx=10.25, maxy=12.75
            ),
        ),
        **kwargs,
    )


def test_grid_resolution_defaults_to_5arcmin():
    # Backward-compat: a config without grid_resolution keeps the 5-arcmin grid.
    assert _region().grid_resolution == "5arcmin"


def test_grid_resolution_accepts_30arcmin_rejects_invalid():
    assert _region(grid_resolution="30arcmin").grid_resolution == "30arcmin"
    # "custom" is rejected: SpatialGrid.from_bounds only implements 5arcmin /
    # 30arcmin (non-5arcmin silently maps to 30arcmin), so the config Literal
    # must not admit the unimplemented "custom".
    for invalid in ("bogus", "custom"):
        with pytest.raises(Exception):
            _region(grid_resolution=invalid)


def test_30arcmin_cell_center_bbox_is_exactly_40_isimip_cells():
    resolution = _region(grid_resolution="30arcmin").grid_resolution
    grid = SpatialGrid.from_bounds(_UC2_BBOX, resolution=resolution)
    assert len(grid.cells) == 40
    assert sorted({round(c.lat, 3) for c in grid.cells}) == _UC2_LAT
    assert sorted({round(c.lon, 3) for c in grid.cells}) == _UC2_LON


def test_5arcmin_default_grid_unchanged():
    # The established 5-arcmin grid is unchanged: the full Kano-belt AOI → 1525.
    aoi = BoundingBox(minx=5.5, miny=11.0, maxx=10.5, maxy=13.0)
    grid = SpatialGrid.from_bounds(aoi, resolution="5arcmin")
    assert len(grid.cells) == 1525


def test_executor_wires_config_grid_resolution_no_hardcode():
    # The executor reads the config field and passes it to BOTH from_bounds
    # sites; no hardcoded 5-arcmin grid resolution remains (revert guard —
    # fails on the pre-change executor).
    import prismpy.pipeline.executor as executor

    src = inspect.getsource(executor)
    assert 'resolution="5arcmin"' not in src
    assert src.count("resolution=self.config.region.grid_resolution") == 2
