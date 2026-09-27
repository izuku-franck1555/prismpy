"""The harmonize stage records what the soil cascade did (iSDA served; HWSD
served, answered with nothing, or gave no answer), and CRAFT's soil branch
follows that state: HWSD-served cells keep their units, an HWSD answer with no
soil fails the build, and no answer keeps today's retrieved-soil path."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.sources.soil.hwsd import HWSDSource
from prismpy.translators.base import HwsdOutcome
from prismpy.translators.craft import translator as craft

sys.path.insert(0, str(Path(__file__).parent))
from test_craft_declared_default_soil import _translator  # noqa: E402
from test_hwsd_layered_profile import _row  # noqa: E402
from test_executor_hwsd_remap import (  # noqa: E402
    _FakeCell,
    _FakeGrid,
    _make_pipeline_with_paths,
    _make_region,
)


def _hwsd(smu):
    layers = [SoilLayer(depth_top=i / 5, depth_bottom=(i + 1) / 5, sand=40.0 + i, clay=30.0 - i,
                        silt=30.0, organic_carbon=1.0, bulk_density=1.4, ph=6.5)
              for i in range(5)]
    return SoilProfile(profile_id=f"HWSD_MLI_{smu}", lat=12.0, lon=-5.0, source="hwsd",
                       layers=layers, total_depth=1.0, metadata={"hwsd_smu_id": smu})


def _source(mode, served=None):
    """A stand-in HWSDSource: ``served`` (index -> profile), ``stray`` (a profile at
    an index outside the grid), ``miss`` (every cell uncovered) or ``raise``."""

    class _Source:
        def __init__(self, config=None, cache_dir=None, provenance=None):
            self.unavailable_cells = []

        def retrieve(self, region=None, cell_coords=None, **kwargs):
            if mode == "raise":
                raise RuntimeError("HWSD unreachable")
            profiles = {"served": dict(served or {}), "stray": {999: _hwsd(1469)}}.get(mode, {})
            for i in range(len(cell_coords)):
                if i not in profiles:
                    self.unavailable_cells.append({"cell_id": i, "cause": "soil_no_hwsd_coverage"})
            return SimpleNamespace(success=bool(profiles), errors=["no HWSD soil"], metadata={},
                                   data=SimpleNamespace(profiles=profiles) if profiles else None)

    return _Source


def _paths(tmp_path):
    bil, mdb = tmp_path / "HWSD2.bil", tmp_path / "HWSD2.mdb"
    bil.touch()
    mdb.touch()
    return bil, mdb


def _outcome(pipe, grid, source):
    with patch("prismpy.sources.soil.hwsd.HWSDSource", source):
        return pipe._retrieve_hwsd_for_grid(grid, _make_region())


def test_each_return_point_reports_its_outcome(tmp_path):
    grid = _FakeGrid([_FakeCell(101, 11.95, -5.05), _FakeCell(202, 11.95, -5.00)])
    pipe = _make_pipeline_with_paths(*_paths(tmp_path))

    profiles, _, outcome = _outcome(pipe, grid, _source("served", {1: _hwsd(1470)}))
    assert outcome == HwsdOutcome.SERVED and set(profiles) == {202}
    assert _outcome(pipe, grid, _source("stray"))[2] == HwsdOutcome.ANSWERED_NONE
    assert _outcome(pipe, grid, _source("miss"))[2] == HwsdOutcome.ANSWERED_NONE
    assert _outcome(pipe, grid, _source("raise"))[2] == HwsdOutcome.NO_ANSWER
    assert _outcome(pipe, _FakeGrid([]), _source("served", {0: _hwsd(1)}))[2] == HwsdOutcome.NO_ANSWER

    unfound = _make_pipeline_with_paths(tmp_path / "absent.bil", tmp_path / "absent.mdb")
    assert _outcome(unfound, grid, _source("served", {0: _hwsd(1)}))[2] == HwsdOutcome.NO_ANSWER
    with patch.object(type(pipe.config), "get_enabled_platforms", return_value=[]):
        assert _outcome(pipe, grid, _source("served", {0: _hwsd(1)}))[2] == HwsdOutcome.NO_ANSWER

    # The real source: files it cannot read are no answer; a read table with no soil is one.
    assert _outcome(pipe, grid, HWSDSource)[2] == HwsdOutcome.NO_ANSWER
    with _non_soil_hwsd():
        assert _outcome(pipe, grid, HWSDSource)[2] == HwsdOutcome.ANSWERED_NONE


def _non_soil_hwsd():
    """The real HWSDSource reading a table where every sampled unit is not a soil."""
    rows = pd.DataFrame([_row(7001, "D1", 0, 20, sand=-9.0, silt=-9.0, clay=-9.0)])
    return patch.multiple(HWSDSource, _sample_bil_raster=lambda self, coords: [7001] * len(coords),
                          _export_mdb_table=lambda self: rows)


def _harmonize(pipe, source, placeholder=True, isda=None):
    region = _make_region()
    retrieved = {"region": region}
    if placeholder:
        retrieved["soil"] = pipe._create_placeholder_soil(region)
    pipe._retrieve_isda_api_for_grid = lambda grid, region: isda
    with patch("prismpy.sources.soil.hwsd.HWSDSource", source):
        result = pipe._execute_harmonize(retrieved)
    assert result.data is not None, result.errors
    return result.data


def test_hwsd_served_at_harmonize_runs_craft_on_the_real_units(tmp_path):
    pipe = _make_pipeline_with_paths(*_paths(tmp_path))
    data = _harmonize(pipe, _source("served", {i: _hwsd(1469 + i % 2) for i in range(500)}))
    assert data.soil_cascade.hwsd == HwsdOutcome.SERVED and not data.soil_cascade.isda_served
    assert data.soil_cascade.n_hwsd_served == len(data.grid.cells)
    _, mask = _translator(tmp_path / "craft")._generate_soil_package(
        data.grid, data.region, data.soil, data.soil_cascade, [])
    assert set(mask.values()) <= {"ML00001469", "ML00001470"}
    assert set(mask) == {cell.cell_id for cell in data.grid.cells}


def test_a_one_cell_region_served_by_hwsd_takes_the_served_branch(tmp_path):
    grid = _FakeGrid([_FakeCell(101, 11.95, -5.05)])
    pipe = _make_pipeline_with_paths(*_paths(tmp_path))
    profiles, _, outcome = _outcome(pipe, grid, _source("served", {0: _hwsd(1469)}))
    assert outcome == HwsdOutcome.SERVED
    from prismpy.translators.base import SoilCascadeState
    state = SoilCascadeState(isda_served=False, hwsd=outcome, n_hwsd_served=len(profiles))
    from test_craft_declared_default_soil import REGION, _grid
    _, mask = _translator(tmp_path / "craft", paths=True)._generate_soil_package(
        _grid([101]), REGION, profiles, state, [])
    assert mask == {101: "ML00001469"}


def test_hwsd_auto_discovered_but_every_unit_missing_fails_the_craft_build(tmp_path, monkeypatch):
    data_dir = tmp_path / "prism_data"
    (data_dir / "hwsd").mkdir(parents=True)
    _paths(data_dir / "hwsd")
    monkeypatch.setenv("PRISM_DATA_DIR", str(data_dir))
    pipe = _make_pipeline_with_paths(None, None)
    with _non_soil_hwsd():
        data = _harmonize(pipe, HWSDSource)
    assert data.soil_cascade.hwsd == HwsdOutcome.ANSWERED_NONE
    assert next(iter(data.soil.values())).source == "placeholder"
    with pytest.raises(craft.CraftSoilUnavailableError, match="refusing to invent a default soil"):
        _translator(tmp_path / "craft")._generate_soil_package(
            data.grid, data.region, data.soil, data.soil_cascade, [])


def test_an_unreadable_hwsd_keeps_the_placeholder_path(tmp_path):
    pipe = _make_pipeline_with_paths(*_paths(tmp_path))
    data = _harmonize(pipe, HWSDSource)
    assert data.soil_cascade.hwsd == HwsdOutcome.NO_ANSWER
    warnings: list = []
    _, mask = _translator(tmp_path / "craft")._generate_soil_package(
        data.grid, data.region, data.soil, data.soil_cascade, warnings)
    assert set(mask.values()) == {"ML90000001"}
    assert "generic placeholder soil profile" in warnings[0]


def test_isda_served_leaves_hwsd_unqueried(tmp_path):
    pipe = _make_pipeline_with_paths(*_paths(tmp_path))
    region = _make_region()
    isda = {0: SoilProfile(profile_id="isda", lat=11.95, lon=-5.05, source="isda",
                           layers=[SoilLayer(depth_top=0.0, depth_bottom=0.3, sand=50.0,
                                             clay=20.0, silt=30.0)], total_depth=0.3)}
    data = _harmonize(pipe, _source("raise"), isda=isda)
    assert data.soil_cascade.isda_served and data.soil_cascade.hwsd == HwsdOutcome.NOT_QUERIED
    assert data.region.name == region.name
