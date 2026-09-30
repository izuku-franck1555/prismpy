"""The one crop-presence reader on real SPAM 2020 crops: the exact input keyset with nodata-safe zeros,
one windowed read, the layer identity verified before any cut (no false reject on the real 5' pixel),
agreement with #116's footprint sums at 5', and the one roster-id digest encoding."""
from __future__ import annotations

import hashlib
import math
import shutil

import numpy as np
import pytest

from tests.unit._crop_presence_fixtures import (
    COAST_BOX, COAST_MAIZ, COAST_MAIZ_V2R0, INC, INLAND_BOX, INLAND_POTA, NORTH, V2R2_NODATA, WEST,
    box_cells, cell, identity_of, pixel_value, write_layer,
)


def _pixel_cells(row, cols):
    """Grid cells at the synthetic layer's pixel centres (row, col) for each col."""
    from prismpy.models.spatial import SpatialGrid

    cells = []
    for col in cols:
        r, c = SpatialGrid.rowcol_from_latlon_5arcmin(NORTH - (row + 0.5) * INC, WEST + (col + 0.5) * INC)
        cells.append(cell(SpatialGrid.compute_cell_id_5arcmin(r, c)))
    return cells


def _expected_area(path, c):
    value, nodata = pixel_value(path, c.lat, c.lon)
    valid = math.isfinite(value) and (nodata is None or math.isnan(nodata) or value != nodata)
    return value if valid and value > 0 else 0.0


@pytest.mark.parametrize("path,box", [(COAST_MAIZ, COAST_BOX), (COAST_MAIZ_V2R0, COAST_BOX),
                                      (INLAND_POTA, INLAND_BOX)])
def test_exact_keyset_with_nodata_safe_zeros_on_real_layers(path, box):
    from prismpy.sources.crop_areas.presence import cell_presence

    cells = box_cells(box)
    got = cell_presence(cells, path, expected=identity_of(path))
    assert set(got.areas) == {c.cell_id for c in cells} and len(got.areas) == len(cells)
    assert [got.areas[c.cell_id] for c in cells] == pytest.approx([_expected_area(path, c) for c in cells])
    assert got.observed.crs == "EPSG:4326" and got.covered is True
    assert got.window[2] > 0 and got.window[3] > 0


def test_every_no_area_value_reads_as_zero(tmp_path):
    from prismpy.sources.crop_areas.presence import cell_presence

    values = [5.0, 0.0, np.nan, V2R2_NODATA, -1.0, np.inf, -np.inf, 0.25]
    arr = np.zeros((18, 18), dtype="float32")
    arr[2, 2:2 + len(values)] = values
    path = write_layer(tmp_path / "layer.tif", arr)
    cells = _pixel_cells(2, range(2, 2 + len(values)))
    got = cell_presence(cells, path)
    assert [got.areas[c.cell_id] for c in cells] == pytest.approx([5.0, 0, 0, 0, 0, 0, 0, 0.25])


@pytest.mark.parametrize("area,present", [
    (0.25, True), (1e-9, True), (0.0, False), (-1.0, False),
    (float("nan"), False), (V2R2_NODATA, False), (float("inf"), False),
])
def test_the_presence_predicate(area, present):
    from prismpy.sources.crop_areas.presence import crop_area_present

    assert crop_area_present(area) is present


def test_one_open_and_one_windowed_read(monkeypatch):
    import rasterio

    from prismpy.sources.crop_areas.presence import cell_presence

    opens, reads = [], []
    real_open = rasterio.open

    class Counting:
        def __init__(self, ds):
            self._ds = ds

        def __enter__(self):
            self._ds.__enter__()
            return self

        def __exit__(self, *exc):
            return self._ds.__exit__(*exc)

        def read(self, *args, **kwargs):
            reads.append(kwargs.get("window"))
            return self._ds.read(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._ds, name)

    expected, cells = identity_of(COAST_MAIZ), box_cells(COAST_BOX)
    monkeypatch.setattr(rasterio, "open", lambda *a, **k: (opens.append(a), Counting(real_open(*a, **k)))[1])
    cell_presence(cells, COAST_MAIZ, expected=expected)
    assert len(opens) == 1 and len(reads) == 1 and reads[0] is not None


def test_the_real_layers_pass_their_own_identity():
    from prismpy.sources.crop_areas.presence import cell_presence

    for path, box in ((COAST_MAIZ, COAST_BOX), (INLAND_POTA, INLAND_BOX)):
        expected = identity_of(path)
        assert expected["transform"][0] != 5 / 60  # the real pixel is 0.0833333332727273 deg
        observed = cell_presence(box_cells(box), path, expected=expected).observed.as_dict()
        assert observed == {key: expected[key] for key in observed}
        assert set(observed) == {"sha256", "crs", "transform", "band", "width", "height", "nodata"}


def _drift(path, field):
    expected = identity_of(path)
    if field == "transform":
        expected["transform"] = [*expected["transform"][:2], expected["transform"][2] + INC,
                                 *expected["transform"][3:]]
    else:
        expected[field] = {"sha256": "0" * 64, "crs": "EPSG:3857", "width": 25, "height": 23,
                           "nodata": "nan", "band": 2}[field]
    return expected


@pytest.mark.parametrize("field", ["sha256", "crs", "width", "height", "nodata", "band", "transform"])
def test_identity_drift_is_refused_before_any_cut(field):
    from prismpy.sources.crop_areas.presence import CropPresenceIdentityError, cell_presence

    with pytest.raises(CropPresenceIdentityError):
        cell_presence(box_cells(COAST_BOX), COAST_MAIZ, expected=_drift(COAST_MAIZ, field))


def test_an_in_place_swap_is_refused(tmp_path):
    from prismpy.sources.crop_areas.presence import CropPresenceIdentityError, cell_presence

    path = tmp_path / COAST_MAIZ.name
    shutil.copyfile(COAST_MAIZ, path)
    expected = identity_of(path)
    shutil.copyfile(COAST_MAIZ_V2R0, path)
    with pytest.raises(CropPresenceIdentityError):
        cell_presence(box_cells(COAST_BOX), path, expected=expected)


@pytest.mark.parametrize("west,crs", [(WEST + INC / 2, "EPSG:4326"), (WEST, "EPSG:3857")])
def test_a_layer_off_the_5_arcmin_geographic_lattice_is_refused(tmp_path, west, crs):
    from prismpy.sources.crop_areas.presence import CropPresenceIdentityError, cell_presence

    path = write_layer(tmp_path / "layer.tif", west=west, crs=crs)
    cells = _pixel_cells(3, range(3, 6))
    with pytest.raises(CropPresenceIdentityError):
        cell_presence(cells, path, expected=identity_of(path))


def test_partial_coverage_is_refused_never_silent_zeros():
    from prismpy.sources.crop_areas.presence import CropPresenceIdentityError, cell_presence

    minx, miny, maxx, maxy = COAST_BOX
    with pytest.raises(CropPresenceIdentityError):
        cell_presence(box_cells((minx - 1.0, miny, maxx, maxy)), COAST_MAIZ, expected=identity_of(COAST_MAIZ))


def test_agrees_with_the_footprint_sums_at_5_arcmin():
    from shapely.geometry import box

    from prismpy.sources.crop_areas.presence import cell_presence
    from prismpy.translators.pythia.translator import _cell_block_sums

    cells = box_cells(COAST_BOX)
    got = cell_presence(cells, COAST_MAIZ)
    for c, (_in_region, full) in zip(cells, _cell_block_sums(COAST_MAIZ, cells, INC / 2, box(*COAST_BOX))):
        assert got.areas[c.cell_id] == pytest.approx(full)


def test_the_one_roster_id_digest_encoding():
    from prismpy.cells.roster_digest import roster_id_digest

    assert roster_id_digest([3, 1, 2]) == hashlib.sha256(b"roster-ids/v1\n1\n2\n3").hexdigest()
    assert roster_id_digest([]) == hashlib.sha256(b"roster-ids/v1\n").hexdigest()
    assert roster_id_digest((2, 1)) == roster_id_digest({1, 2})
    for bad in ([1, 1], [True], [1.0], ["1"]):
        with pytest.raises(ValueError):
            roster_id_digest(bad)
