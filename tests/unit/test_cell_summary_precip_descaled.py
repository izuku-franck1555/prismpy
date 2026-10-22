"""A cell's NASA POWER de-scaling record is copied by name, verbatim, into its cell summary
beside the gap fill; a cell whose series carries none (or an empty one) has no key."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.pipeline.executor import TranslationPipeline

_EXECUTOR = (Path(__file__).resolve().parents[2] / "src" / "prismpy" / "pipeline" / "executor.py")

RECORD = [{"year": 2026, "first_date": "2026-06-01", "last_date": "2026-10-10",
           "wet_days_divided": 2, "divided_dates": ["2026-07-14", "2026-08-02"], "factor": 24,
           "reason": "nasa_power_x24_response",
           "evidence": "the response's rain is 24 times a clean earlier window's"}]


def _summary(climate):
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    cells = [SimpleNamespace(cell_id=cell, lat=0.0, lon=0.0, layers=None) for cell in climate]
    unified = SimpleNamespace(grid=SimpleNamespace(cells=cells, resolution="5arcmin"), soil={},
                              climate=climate, metadata={})
    return pipeline._build_cell_summary(unified, None, sarra_climate_per_cell=None)


def _series(descaled=None):
    series = ClimateTimeSeries(
        location_id=7, lat=0.0, lon=0.0, source="nasa_power",
        records=[ClimateRecord(date=date(2026, 1, day), tmax=30.0, tmin=20.0, precip=1.0,
                               srad=18.0) for day in range(1, 5)])
    if descaled is not None:
        series.metadata["precip_x24_descaled"] = descaled
    return series


def test_a_cells_descaling_record_is_copied_verbatim_into_its_cell_summary():
    assert _summary({7: _series(RECORD)})["cells"][0].get("climate_precip_descaled") == RECORD


def test_a_cells_records_for_two_divided_years_are_kept_in_order_and_unchanged():
    two_years = [{"year": 2025, "first_date": "2025-07-01", "last_date": "2025-12-31",
                  "wet_days_divided": 1, "divided_dates": ["2025-08-09"], "factor": 24,
                  "reason": "nasa_power_x24_response",
                  "evidence": "the response's rain is 24 times a clean earlier window's"}, *RECORD]
    assert _summary({7: _series(two_years)})["cells"][0].get("climate_precip_descaled") == two_years


def test_a_cell_without_a_descaling_record_has_no_key():
    cells = _summary({1: _series(), 2: _series([])})["cells"]
    assert ["climate_precip_descaled" in cell for cell in cells] == [False, False]


def test_the_copy_lives_once_in_the_canonical_writer():
    assert _EXECUTOR.read_text(encoding="utf-8").count('cell_data["climate_precip_descaled"]') == 1
