"""CRAFT's SPAM crop-mask shares on real SPAM 2020 crops. V2r2 declares a FINITE nodata
(-3.4028e38): a nodata, non-finite or negative pixel is 0 harvested area, never a negative share.
The NaN-nodata path (V2r0) and every valid pixel keep their pre-change shares."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prismpy.config.schema import CraftConfig, PlatformConfigGroup
from prismpy.translators.craft.translator import CraftTranslator
from tests.unit._crop_presence_fixtures import (
    COAST_BOX, COAST_MAIZ, COAST_MAIZ_V2R0, V2R2_NODATA, box_cells, make_config, pixel_value,
)

# The shares at the pre-change base 4e763a1, digested by _digest.
_BASE_V2R0_SHARES = "25f958321c8ebec0e5e7ca8693a8de32f6f5dce669a78cf196edd9c66960f7de"
_BASE_V2R2_VALID_SHARES = "29e42ba683d7db7a9070f8c9d9d3f4a8872698e3399919dae9a81f460edb9819"


def _shares(tmp_path, path):
    cfg = make_config(tmp_path)
    cfg.platform_config = PlatformConfigGroup(craft=CraftConfig(spam_raster_path=Path(path)))
    cells = box_cells(COAST_BOX)
    return cells, CraftTranslator(cfg)._extract_crop_mask_from_spam(cells, path)


def _digest(shares):
    return hashlib.sha256(json.dumps(sorted(shares.items())).encode()).hexdigest()


def _nodata_ids(cells):
    return {c.cell_id for c in cells if pixel_value(COAST_MAIZ, c.lat, c.lon)[0] == V2R2_NODATA}


def test_finite_nodata_never_yields_a_negative_share(tmp_path):
    cells, shares = _shares(tmp_path, COAST_MAIZ)
    nodata = _nodata_ids(cells)
    assert len(nodata) >= 18 and len(shares) == len(cells)
    assert all(value >= 0 for value in shares.values())
    assert all(shares[cid] == 0.0 for cid in nodata)


def test_the_nan_nodata_path_is_unchanged(tmp_path):
    _, shares = _shares(tmp_path, COAST_MAIZ_V2R0)
    assert _digest(shares) == _BASE_V2R0_SHARES


def test_valid_pixels_keep_their_shares(tmp_path):
    cells, shares = _shares(tmp_path, COAST_MAIZ)
    nodata = _nodata_ids(cells)
    assert _digest({cid: v for cid, v in shares.items() if cid not in nodata}) == _BASE_V2R2_VALID_SHARES
