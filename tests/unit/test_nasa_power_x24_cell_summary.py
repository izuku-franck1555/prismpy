"""The guard's real retrieval feeds the cell summary: a de-scaled exposed year's record reaches
its cell's ``climate_precip_descaled`` intact; a clean year's cell has no such key."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

from prismpy.pipeline.executor import TranslationPipeline
from tests.unit._nasa_x24_support import FakeNasa, body, load, run

LAT, LON = 10.75, 5.75
RATE = load("rate_probe_compact.json")
W, REF = ("20260101", "20261011"), ("20260101", "20260630")
JAN1, END = date(2026, 1, 1), date(2026, 10, 11)


def _summary(climate):
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    cells = [SimpleNamespace(cell_id=cell, lat=LAT, lon=LON, layers=None) for cell in climate]
    unified = SimpleNamespace(
        grid=SimpleNamespace(cells=cells, resolution="5arcmin"),
        soil={},
        climate=climate,
        metadata={},
    )
    return pipeline._build_cell_summary(unified, None, sarra_climate_per_cell=None)


def test_a_descaled_years_record_reaches_its_cell_summary_intact(tmp_path):
    scaled = run(
        FakeNasa(
            {
                W: [body(RATE["scaled_precip"], RATE["other_variables"])],
                REF: [RATE["ref_jan_jun_parameter"]],
            }
        ),
        cache_dir=tmp_path / "scaled",
        lat=LAT,
        lon=LON,
        start=JAN1,
        end=END,
    )
    clean = run(
        FakeNasa({W: [body(RATE["clean_precip"], RATE["other_variables"])]}),
        cache_dir=tmp_path / "clean",
        lat=LAT,
        lon=LON,
        start=JAN1,
        end=END,
    )
    cells = _summary({1: scaled.data, 2: clean.data})["cells"]
    record = cells[0].get("climate_precip_descaled")
    assert record and record == scaled.data.metadata.get("precip_x24_descaled")
    assert [entry["year"] for entry in record] == [2026]
    assert json.loads(json.dumps(record)) == record
    assert "climate_precip_descaled" not in cells[1]
