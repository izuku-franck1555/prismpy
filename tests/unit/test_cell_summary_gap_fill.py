"""Pins for per-cell climate gap-fill provenance carried into cell_summary.

The canonical writer ``TranslationPipeline._build_cell_summary`` emits each
cell's recovered-day provenance (a per-variable list, method read from the
record), omits the key for a measured cell, and stamps a package-level flag so
a consumer tells an old producer (flag absent) from a measured cell (key
omitted). The headline cell uses the FROZEN real Hukunsti payload (no gap-free
mock — the real-data lesson); the rest use realistic per-variable records.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.sources.climate import nasa_power as np_mod
from prismpy.sources.climate.nasa_power import (
    DEFAULT_PARAMETERS,
    NASAPowerConfig,
    NASAPowerSource,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "hukunsti_frozen.json"
_EXECUTOR = (Path(__file__).resolve().parents[2] / "src" / "prismpy"
             / "pipeline" / "executor.py")
_NASA_POWER = (Path(__file__).resolve().parents[2] / "src" / "prismpy"
               / "sources" / "climate" / "nasa_power.py")
_LAT, _LON = -23.291666666666657, 20.041666666666657


# ── helpers ────────────────────────────────────────────────────────────────
class _Cell:
    def __init__(self, cid):
        self.cell_id = cid
        self.lat = 0.0
        self.lon = 0.0
        self.layers = None


def _pipeline():
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    return p


def _summary(climate, sarra=None, cell_ids=None):
    ids = cell_ids if cell_ids is not None else list(climate.keys())
    grid = SimpleNamespace(cells=[_Cell(i) for i in ids], resolution="5arcmin")
    ud = SimpleNamespace(grid=grid, soil={}, climate=climate, metadata={})
    return _pipeline()._build_cell_summary(ud, None, sarra_climate_per_cell=sarra)


def _ts(gap_fill=None):
    recs = [ClimateRecord(date=date(2025, 1, d), tmax=20.0, tmin=10.0,
                          precip=0.0, srad=15.0) for d in range(1, 5)]
    ts = ClimateTimeSeries(location_id=0, lat=0, lon=0, source="nasa_power",
                           records=recs)
    if gap_fill:
        ts.metadata["gap_fill"] = gap_fill
    return ts


def _frozen_params():
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))["properties"]["parameter"]


def _fetch_from_frozen(frozen):
    def _fetch(self, lat, lon, start_date, end_date, parameters,
               cancel_check=None, on_attempt=None):
        return {
            var: {k: v for k, v in days.items()
                  if start_date <= datetime.strptime(k, "%Y%m%d").date() <= end_date}
            for var, days in frozen.items()
        }
    return _fetch


# ── REAL Hukunsti gap-fill carried with fidelity ───────────────────────────
def test_real_hukunsti_gap_fill_carried_per_variable(tmp_path):
    src = NASAPowerSource(config=NASAPowerConfig(request_delay=0.0),
                          cache_dir=tmp_path)
    with mock.patch.object(np_mod, "nasa_power_latest_available_date",
                           lambda *a, **k: date(2026, 5, 24)), \
            mock.patch.object(NASAPowerSource, "_fetch_from_api",
                              _fetch_from_frozen(_frozen_params())):
        result = src.retrieve(lat=_LAT, lon=_LON, start_date=date(2025, 1, 1),
                              end_date=date(2026, 3, 15), use_cache=False)
    assert result.success is True
    gap_meta = result.data.metadata.get("gap_fill")
    assert gap_meta == {"srad": {"n_filled_days": 1, "method": "linear-interp"}}

    cell = _summary({7: result.data})["cells"][0]
    assert cell["climate_gap_fill"] == {
        "per_variable": [{"variable": "srad", "n_filled_days": 1,
                          "method": "linear-interp"}]}
    # end-to-end fidelity: emitted counts == producer metadata counts
    emitted = {r["variable"]: r["n_filled_days"]
               for r in cell["climate_gap_fill"]["per_variable"]}
    assert emitted == {v: d["n_filled_days"] for v, d in gap_meta.items()}


# ── Measured cell OMITS the key (anti-false-positive) ──────────────────────
def test_measured_cell_omits_climate_gap_fill():
    assert "climate_gap_fill" not in _summary({1: _ts()})["cells"][0]


# ── Multi-variable rows ────────────────────────────────────────────────────
def test_multi_variable_fill_per_variable_rows():
    gf = {"srad": {"n_filled_days": 5, "method": "linear-interp"},
          "tmax": {"n_filled_days": 3, "method": "linear-interp"}}
    rows = _summary({1: _ts(gf)})["cells"][0]["climate_gap_fill"]["per_variable"]
    assert {r["variable"]: r["n_filled_days"] for r in rows} == {"srad": 5, "tmax": 3}


# ── Canonical-writer parity + package flag + SARRA omit ────────────────────
def test_emit_is_in_the_single_canonical_writer():
    # The carry lives once in the shared writer, not a per-platform branch, so
    # every platform's cell_summary gets it identically (one canonical writer).
    assert _EXECUTOR.read_text(encoding="utf-8").count(
        'cell_data["climate_gap_fill"]') == 1


def test_package_flag_records_provenance():
    summary = _summary({1: _ts({"srad": {"n_filled_days": 1,
                                         "method": "linear-interp"}})})
    assert summary["climate_gap_provenance_recorded"] is True


def test_sarra_py_cell_omits_key_but_package_flag_emits():
    # SARRA-Py climate never flows through unified_data.climate (ts is None) →
    # per-cell key absent (honest n=0), package flag still emitted.
    sarra = {1: {"tmax": [20.0, 21.0], "tmin": [10.0, 11.0],
                 "srad": [15.0, 16.0], "rain": [0.0, 1.0]}}
    summary = _summary({}, sarra=sarra, cell_ids=[1])
    assert summary["climate_gap_provenance_recorded"] is True
    assert "climate_gap_fill" not in summary["cells"][0]


# ── method read from the record, never hard-coded ───────────────────────────
def test_method_read_from_provenance_not_hardcoded():
    gf = {"srad": {"n_filled_days": 2, "method": "future-method"}}
    rows = _summary({1: _ts(gf)})["cells"][0]["climate_gap_fill"]["per_variable"]
    assert rows[0]["method"] == "future-method"


# ── Both producer set-sites carry it (cache + fresh) ───────────────────────
def test_gap_fill_set_at_both_nasa_power_return_paths():
    # The cache early-return AND the main return both stamp gap_fill, so a
    # cache-served cell carries provenance identically to a fresh fetch.
    text = _NASA_POWER.read_text(encoding="utf-8")
    assert text.count('.metadata["gap_fill"] = fill_provenance') == 2


def test_cache_served_gap_fill_carries_into_cell_summary(tmp_path):
    # A full past year with one interior srad gap is cached on run 1; run 2
    # serves it from cache, re-derives the gap-fill, and the cell_summary
    # carries it — the cache path is not a provenance hole.
    def fake(self, lat, lon, start_date, end_date, parameters,
             cancel_check=None, on_attempt=None):
        out = {p: {} for p in DEFAULT_PARAMETERS}
        day = start_date
        while day <= end_date:
            key = day.strftime("%Y%m%d")
            for p in DEFAULT_PARAMETERS:
                out[p][key] = (-999 if (p == "ALLSKY_SFC_SW_DWN"
                                        and day == date(2024, 6, 15)) else 5.0)
            day += timedelta(days=1)
        return out

    src = NASAPowerSource(config=NASAPowerConfig(request_delay=0.0),
                          cache_dir=tmp_path)

    def run():
        with mock.patch.object(np_mod, "nasa_power_latest_available_date",
                               lambda *a, **k: date(2026, 1, 1)), \
                mock.patch.object(NASAPowerSource, "_fetch_from_api", fake):
            return src.retrieve(lat=-23.29, lon=20.04, start_date=date(2024, 1, 1),
                                end_date=date(2024, 12, 31), use_cache=True)

    run()                 # populate the per-year cache
    result = run()        # second run serves 2024 from cache
    assert result.success is True
    assert result.data.metadata.get("gap_fill") == {
        "srad": {"n_filled_days": 1, "method": "linear-interp"}}
    cell = _summary({7: result.data})["cells"][0]
    assert cell["climate_gap_fill"]["per_variable"][0]["n_filled_days"] == 1
