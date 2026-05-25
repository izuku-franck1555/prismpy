"""Structural pin: AgERA5 NetCDF->GeoTIFF conversion georeferences every
grid size, including single-pixel-axis subsets.

Regression (2026-05-25): small study areas (sub-~0.2 deg) subset to a SINGLE
AgERA5 grid pixel in a dimension. ``convert_AgERA5_netcdf_to_geotiff`` used
``bT.rio.to_raster`` which relies on rioxarray inferring the affine from the
lon/lat coordinate spacing. A 1-pixel axis has no spacing to infer, so
rioxarray silently fell back to an identity transform — origin (0,0), bounds
(0, N, M, 0) at null-island. The downstream SARRA-Py reprojection onto the
(correctly georeferenced) TAMSAT grid then had zero overlap and produced
ALL-NaN temperature/radiation/ET0, so the yield_forecast returned 0 kg/ha for
every cell/year. The fix sets the geotransform explicitly from the coordinate
values + the AgERA5 native 0.1 deg resolution when an axis is single-pixel.

These tests run the REAL vendored converter on synthetic NetCDFs and assert
the output GeoTIFF bounds match the input coordinates (never the null-island
identity transform).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List

import numpy as np
import pytest
import rasterio
import xarray as xr

from prismpy.vendor.sarra_data_download.get_AgERA5_data import (
    convert_AgERA5_netcdf_to_geotiff,
)

_VAR = ("2m_temperature", "24_hour_minimum")
_REGION = "TestRegion"
_YEAR = 2017


def _convert_and_read(
    tmp_path: Path, lons: List[float], lats: List[float]
):
    """Write a synthetic AgERA5 extraction NetCDF, run the real converter,
    and return (bounds, shape) of the produced GeoTIFF."""
    save_path = tmp_path
    extraction = (
        save_path
        / "1_extraction"
        / f"AgERA5_{_REGION}"
        / str(_YEAR)
        / f"{_VAR[0]}_{_VAR[1]}"
    )
    extraction.mkdir(parents=True, exist_ok=True)

    da = xr.DataArray(
        np.full((1, len(lats), len(lons)), 295.0, dtype="float32"),
        coords={
            "time": [np.datetime64("2017-06-15", "ns")],
            "lat": np.asarray(lats, dtype="float64"),
            "lon": np.asarray(lons, dtype="float64"),
        },
        dims=("time", "lat", "lon"),
        name="Temperature_Air_2m_Min_24h",
    )
    da.to_dataset().to_netcdf(extraction / "synthetic_20170615.nc")

    convert_AgERA5_netcdf_to_geotiff(
        {}, _REGION, [_VAR], query=dt.date(_YEAR, 1, 1), save_path=str(save_path)
    )

    conv = (
        save_path
        / "2_conversion"
        / f"AgERA5_{_REGION}"
        / f"{_VAR[0]}_{_VAR[1]}"
    )
    tifs = sorted(conv.glob("*.tif"))
    assert tifs, f"converter produced no GeoTIFF in {conv}"
    with rasterio.open(tifs[0]) as ds:
        return tuple(ds.bounds), ds.shape


def test_single_lon_pixel_georeferenced_correctly(tmp_path: Path) -> None:
    """The Mopti bug case: 1 lon x 2 lat. Must NOT degenerate to the
    null-island identity transform."""
    bounds, shape = _convert_and_read(
        tmp_path, lons=[-4.2], lats=[14.5, 14.4]
    )
    assert shape == (2, 1)
    assert bounds != (0.0, 2.0, 1.0, 0.0), (
        "single-lon subset fell back to the null-island identity transform"
    )
    # left, bottom, right, top — pixel-corner bounds at 0.1 deg
    assert bounds == pytest.approx((-4.25, 14.35, -4.15, 14.55), abs=1e-6)


def test_single_lat_pixel_georeferenced_correctly(tmp_path: Path) -> None:
    """Symmetric case: 3 lon x 1 lat (single-row)."""
    bounds, shape = _convert_and_read(
        tmp_path, lons=[-4.3, -4.2, -4.1], lats=[14.5]
    )
    assert shape == (1, 3)
    assert bounds == pytest.approx((-4.35, 14.45, -4.05, 14.55), abs=1e-6)


def test_single_pixel_both_axes_georeferenced_correctly(
    tmp_path: Path,
) -> None:
    """Extreme case: 1 lon x 1 lat (region smaller than one AgERA5 cell)."""
    bounds, shape = _convert_and_read(tmp_path, lons=[-4.2], lats=[14.5])
    assert shape == (1, 1)
    assert bounds != (0.0, 1.0, 1.0, 0.0)
    assert bounds == pytest.approx((-4.25, 14.45, -4.15, 14.55), abs=1e-6)


def test_multi_pixel_unchanged(tmp_path: Path) -> None:
    """Regression guard: normal multi-pixel grids keep rioxarray's native
    inference and georeference correctly (no behaviour change)."""
    bounds, shape = _convert_and_read(
        tmp_path,
        lons=[-4.3, -4.2, -4.1],
        lats=[14.6, 14.5, 14.4],
    )
    assert shape == (3, 3)
    assert bounds == pytest.approx((-4.35, 14.35, -4.05, 14.65), abs=1e-6)
