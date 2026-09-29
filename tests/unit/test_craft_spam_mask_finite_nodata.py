"""CRAFT's SPAM crop-mask shares under a finite nodata value (SPAM 2020 V2r2 declares -3.4028e38):
a nodata, non-finite or negative pixel is 0 harvested area, never a negative share."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prismpy.config.schema import CraftConfig, PlatformConfigGroup
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators.craft.translator import CraftTranslator
from tests.unit._crop_presence_fixtures import INC, NORTH, V2R2_NODATA, WEST, make_config, write_layer


def _cells(pixels):
    cells = []
    for row, col in pixels:
        lat, lon = NORTH - (row + 0.5) * INC, WEST + (col + 0.5) * INC
        grow, gcol = SpatialGrid.rowcol_from_latlon_5arcmin(lat, lon)
        cells.append(GridCell(cell_id=SpatialGrid.compute_cell_id_5arcmin(grow, gcol),
                              lat=lat, lon=lon, row=grow, col=gcol))
    return cells


def _shares(tmp_path, nodata, values):
    arr = np.zeros((18, 18), dtype="float32")
    arr[3, 3:3 + len(values)] = values
    path = write_layer(tmp_path / "spam2020_V2r2_global_H_MAIZ_A.tif", arr, nodata=nodata)
    cfg = make_config(tmp_path)
    cfg.platform_config = PlatformConfigGroup(craft=CraftConfig(spam_raster_path=Path(path)))
    cells = _cells([(3, 3 + k) for k in range(len(values))])
    shares = CraftTranslator(cfg)._extract_crop_mask_from_spam(cells, path)
    return [shares[c.cell_id] for c in cells]


def test_finite_nodata_never_yields_a_negative_share(tmp_path):
    shares = _shares(tmp_path, V2R2_NODATA, [40.0, V2R2_NODATA, -3.0, np.nan])
    assert shares[0] > 0
    assert shares[1:] == [0.0, 0.0, 0.0]


def test_the_nan_nodata_path_is_unchanged(tmp_path):
    shares = _shares(tmp_path, np.nan, [40.0, np.nan, 0.0])
    assert shares[0] > 0 and shares[1:] == [0.0, 0.0]
