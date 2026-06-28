"""Unit tests for the ISIMIP3b → ClimateTimeSeries composition bridge.

Covers :func:`prismpy.harmonize.isimip_to_climate.isimip_cutouts_to_climate_timeseries`
with synthetic in-memory xarray arrays (no network, no HDF5/netCDF): the
bridge must wire the existing calendar / unit / dewpoint primitives and
sample per cell, producing canonical-unit records and propagating the
calendar-conversion limitation.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from prismpy.harmonize.calendar_conversion import LIMITATION_KEY_NOLEAP_DROPPED
from prismpy.harmonize.isimip_to_climate import (
    CALENDAR_LIMITATION_KEY_FIELD,
    isimip_cutouts_to_climate_timeseries,
)
from prismpy.harmonize.tetens import derive_tdew
from prismpy.models.spatial import GridCell

_LATS = np.array([12.0, 12.1])
_LONS = np.array([8.4, 8.5])
_CELLS = [GridCell(cell_id=1, lat=12.0, lon=8.4, row=0, col=0, resolution="custom")]


def _field(times, value: float) -> xr.DataArray:
    data = np.full((len(times), len(_LATS), len(_LONS)), value, dtype="float64")
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": _LATS, "lon": _LONS},
    )


def _gregorian_times(n: int = 3):
    return np.array(
        [f"2050-06-0{i + 1}" for i in range(n)], dtype="datetime64[ns]"
    )


def _cutouts(times, *, tasmax=300.0, tasmin=290.0, pr=1e-4, rsds=200.0):
    return {
        "tasmax": _field(times, tasmax),
        "tasmin": _field(times, tasmin),
        "pr": _field(times, pr),
        "rsds": _field(times, rsds),
    }


def test_gregorian_unit_conversions_land_in_canonical_units():
    times = _gregorian_times(3)
    series = isimip_cutouts_to_climate_timeseries(
        _cutouts(times), _CELLS, gcm_source="ipsl-cm6a-lr"
    )
    assert set(series) == {1}
    ts = series[1]
    assert ts.n_records == 3
    record = ts.records[0]
    # 300 K → 26.85 °C; 290 K → 16.85 °C.
    assert record.tmax == pytest.approx(26.85, abs=1e-6)
    assert record.tmin == pytest.approx(16.85, abs=1e-6)
    # 1e-4 kg m⁻² s⁻¹ × 86400 → 8.64 mm/day.
    assert record.precip == pytest.approx(8.64, abs=1e-6)
    # 200 W m⁻² × 86.4 → 17280 kJ → ×1e-3 → 17.28 MJ m⁻² day⁻¹.
    assert record.srad == pytest.approx(17.28, abs=1e-6)
    # Plausible canonical values pass the record's own range validator.
    assert record.validate() == []
    # Gregorian source → no calendar limitation.
    assert ts.metadata == {}


def test_noleap_source_propagates_calendar_limitation():
    times = xr.date_range(
        "2050-06-01", periods=3, freq="D", calendar="noleap", use_cftime=True
    )
    series = isimip_cutouts_to_climate_timeseries(
        _cutouts(times), _CELLS, gcm_source="gfdl-esm4"
    )
    metadata = series[1].metadata
    assert metadata.get(CALENDAR_LIMITATION_KEY_FIELD) == LIMITATION_KEY_NOLEAP_DROPPED


def test_humidity_sets_rh_and_derives_real_dewpoint():
    times = _gregorian_times(2)
    cutouts = _cutouts(times)
    cutouts["hurs"] = _field(times, 60.0)
    series = isimip_cutouts_to_climate_timeseries(
        cutouts, _CELLS, gcm_source="ipsl-cm6a-lr"
    )
    record = series[1].records[0]
    # The humidity is preserved on the record (the WTH writers read rh for
    # RHUM); a None here would silently emit RHUM=-99 despite real hurs.
    assert record.rh == pytest.approx(60.0)
    # tdew is the real FAO-56 Tetens value from the actual RH, not a fabricated
    # saturation (tmean) fallback — so it is strictly below tmean at 60 %% RH.
    expected_tdew = derive_tdew(record.tmean, 60.0)
    assert record.tdew == pytest.approx(expected_tdew, abs=1e-9)
    assert record.tdew < record.tmean
    # The dewpoint policy is disclosed on the series metadata.
    policy = series[1].metadata.get("dewpoint_policy")
    assert policy is not None and "propagate-missing" in policy


def test_out_of_range_humidity_propagates_missing_not_saturation():
    # A relative humidity above 100 %% (e.g. a 360-day calendar interpolation
    # overshoot) is invalid input. The honest policy is propagate-missing:
    # leave rh and tdew None rather than fabricate a saturated dewpoint.
    times = _gregorian_times(2)
    cutouts = _cutouts(times)
    cutouts["hurs"] = _field(times, 103.0)
    series = isimip_cutouts_to_climate_timeseries(
        cutouts, _CELLS, gcm_source="ipsl-cm6a-lr"
    )
    record = series[1].records[0]
    assert record.tdew is None
    assert record.rh is None
    # Specifically NOT the old saturation-fabrication (tdew == tmean).
    assert record.tdew != record.tmean


def test_no_humidity_leaves_dewpoint_none():
    times = _gregorian_times(2)
    series = isimip_cutouts_to_climate_timeseries(
        _cutouts(times), _CELLS, gcm_source="ipsl-cm6a-lr"
    )
    assert series[1].records[0].tdew is None


def test_missing_required_variable_raises():
    times = _gregorian_times(2)
    cutouts = _cutouts(times)
    del cutouts["rsds"]
    with pytest.raises(ValueError, match="rsds"):
        isimip_cutouts_to_climate_timeseries(
            cutouts, _CELLS, gcm_source="ipsl-cm6a-lr"
        )


def test_unknown_gcm_raises():
    times = _gregorian_times(2)
    with pytest.raises(ValueError):
        isimip_cutouts_to_climate_timeseries(
            _cutouts(times), _CELLS, gcm_source="not-a-real-gcm"
        )


def test_source_label_default_and_override():
    times = _gregorian_times(2)
    default = isimip_cutouts_to_climate_timeseries(
        _cutouts(times), _CELLS, gcm_source="ipsl-cm6a-lr"
    )
    assert default[1].source == "ISIMIP3b_ipsl-cm6a-lr"
    overridden = isimip_cutouts_to_climate_timeseries(
        _cutouts(times), _CELLS, gcm_source="ipsl-cm6a-lr", source_label="custom"
    )
    assert overridden[1].source == "custom"
