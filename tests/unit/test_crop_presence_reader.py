"""The one crop-presence reader: the exact input keyset with nodata-safe zeros, the layer identity
verified before any cut, agreement with the footprint sums at 5', and the one roster-id digest."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from prismpy.models.spatial import GridCell, SpatialGrid
from tests.unit._crop_presence_fixtures import INC, NORTH, V2R2_NODATA, WEST, identity_of, write_layer


def _centre(row, col):
    """(lat, lon) of the fixture pixel centre (row, col)."""
    return NORTH - (row + 0.5) * INC, WEST + (col + 0.5) * INC


def _cells(latlons):
    out = []
    for lat, lon in latlons:
        row, col = SpatialGrid.rowcol_from_latlon_5arcmin(lat, lon)
        out.append(GridCell(cell_id=SpatialGrid.compute_cell_id_5arcmin(row, col),
                            lat=lat, lon=lon, row=row, col=col))
    return out


def test_exact_keyset_with_nodata_safe_zeros(tmp_path):
    from prismpy.sources.crop_areas.presence import cell_presence

    values = [5.0, 0.0, np.nan, V2R2_NODATA, -1.0, np.inf, -np.inf, 0.25]
    arr = np.zeros((18, 18), dtype="float32")
    arr[2, 2:2 + len(values)] = values
    path = write_layer(tmp_path / "spam2020_V2r2_global_H_MAIZ_A.tif", arr)
    cells = _cells([_centre(2, 2 + k) for k in range(len(values))])

    got = cell_presence(cells, path)

    assert set(got.areas) == {c.cell_id for c in cells} and len(got.areas) == len(cells)
    assert [got.areas[c.cell_id] for c in cells] == pytest.approx([5.0, 0, 0, 0, 0, 0, 0, 0.25])


@pytest.mark.parametrize("area,present", [
    (0.25, True), (1e-9, True), (0.0, False), (-1.0, False),
    (float("nan"), False), (V2R2_NODATA, False), (float("inf"), False),
])
def test_the_presence_predicate(area, present):
    from prismpy.sources.crop_areas.presence import crop_area_present

    assert crop_area_present(area) is present


def test_one_windowed_read_for_all_cells(tmp_path, monkeypatch):
    import rasterio

    from prismpy.sources.crop_areas.presence import cell_presence

    path = write_layer(tmp_path / "layer.tif")
    reads = []
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

    monkeypatch.setattr(rasterio, "open", lambda *a, **k: Counting(real_open(*a, **k)))
    cell_presence(_cells([_centre(r, c) for r in range(3, 9) for c in range(3, 9)]), path)
    assert len(reads) == 1 and reads[0] is not None


def test_identity_is_verified_and_reported(tmp_path):
    from prismpy.sources.crop_areas.presence import cell_presence

    path = write_layer(tmp_path / "layer.tif")
    expected = identity_of(path)
    observed = cell_presence(_cells([_centre(3, 3)]), path, expected=expected).observed.as_dict()
    assert observed == {key: expected[key] for key in observed}
    assert set(observed) == {"sha256", "crs", "transform", "band", "width", "height", "nodata"}


@pytest.mark.parametrize("field,value", [
    ("sha256", "0" * 64), ("crs", "EPSG:3857"), ("width", 19), ("height", 17),
    ("nodata", "nan"), ("band", 2), ("transform", "shifted"),
])
def test_identity_drift_is_refused_before_any_cut(tmp_path, field, value):
    from prismpy.sources.crop_areas.presence import CropPresenceIdentityError, cell_presence

    path = write_layer(tmp_path / "layer.tif")
    expected = identity_of(path)
    if value == "shifted":
        value = [*expected["transform"][:2], expected["transform"][2] + INC, *expected["transform"][3:]]
    expected[field] = value
    with pytest.raises(CropPresenceIdentityError):
        cell_presence(_cells([_centre(3, 3)]), path, expected=expected)


def test_an_in_place_swap_is_refused(tmp_path):
    from prismpy.sources.crop_areas.presence import CropPresenceIdentityError, cell_presence

    path = write_layer(tmp_path / "layer.tif")
    expected = identity_of(path)
    swapped = np.zeros((18, 18), dtype="float32")
    swapped[3, 3] = 7.0
    write_layer(path, swapped)
    with pytest.raises(CropPresenceIdentityError):
        cell_presence(_cells([_centre(3, 3)]), path, expected=expected)


def test_a_layer_off_the_canonical_lattice_is_refused(tmp_path):
    from prismpy.sources.crop_areas.presence import CropPresenceIdentityError, cell_presence

    path = write_layer(tmp_path / "layer.tif", west=WEST + INC / 2)
    with pytest.raises(CropPresenceIdentityError):
        cell_presence(_cells([_centre(3, 3)]), path, expected=identity_of(path))


def test_partial_coverage_is_refused(tmp_path):
    from prismpy.sources.crop_areas.presence import CropPresenceIdentityError, cell_presence

    path = write_layer(tmp_path / "layer.tif")
    with pytest.raises(CropPresenceIdentityError):
        cell_presence(_cells([_centre(3, 3), (NORTH + INC / 2, WEST + INC / 2)]), path)


def test_agrees_with_the_footprint_sums_at_5_arcmin(tmp_path):
    from shapely.geometry import box

    from prismpy.sources.crop_areas.presence import cell_presence
    from prismpy.translators.pythia.translator import _cell_block_sums

    rng = np.random.default_rng(7)
    arr = rng.choice([0.0, 0.0, 3.5, np.nan, V2R2_NODATA, -2.0], size=(18, 18)).astype("float32")
    path = write_layer(tmp_path / "layer.tif", arr)
    cells = _cells([_centre(r, c) for r in range(1, 17) for c in range(1, 17)])

    got = cell_presence(cells, path)
    sums = _cell_block_sums(path, cells, INC / 2, box(WEST, NORTH - 18 * INC, WEST + 18 * INC, NORTH))
    for cell, (_in_region, full) in zip(cells, sums):
        assert got.areas[cell.cell_id] == pytest.approx(full)


def test_the_one_roster_id_digest_encoding():
    from prismpy.cells.roster_digest import roster_id_digest

    assert roster_id_digest([3, 1, 2]) == hashlib.sha256(b"roster-ids/v1\n1\n2\n3").hexdigest()
    assert roster_id_digest([]) == hashlib.sha256(b"roster-ids/v1\n").hexdigest()
    assert roster_id_digest((2, 1)) == roster_id_digest({1, 2})
    for bad in ([1, 1], [True], [1.0], ["1"]):
        with pytest.raises(ValueError):
            roster_id_digest(bad)
