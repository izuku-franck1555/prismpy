"""A GADM region configured by its GID prepares exactly that unit, in both GADM backends; a GID never
falls back to a name lookup; a NAME config keeps today's behaviour. Nigeria has two districts named
Bassa (Kogi and Plateau), the shape of the duplicate-name defect."""
from __future__ import annotations

import sys

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from prismpy.config.schema import (
    BoundaryConfig, BoundarySource, CropCalendarConfig, CropConfig, DataSourcesConfig,
    GadmSourceConfig, ManualBoundsConfig, OutputConfig, Platform, ProjectConfig, ProjectInfo,
    RegionConfig, TemporalConfig,
)
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.provenance.tracker import ProvenanceTracker
from prismpy.sources.boundaries.gadm import GADMSource

_UNITS = {  # GID_2 -> (NAME_1, NAME_2, polygon)
    "NGA.23.2_1": ("Kogi", "Bassa", box(6.5, 7.5, 7.0, 8.0)),
    "NGA.32.3_1": ("Plateau", "Bassa", box(8.5, 9.5, 9.0, 10.0)),
}


class _FakePygadm:
    def __init__(self):
        self.calls = []

    def Names(self, admin, content_level):  # noqa: N802 - pygadm's API
        self.calls.append(("Names", admin))
        return pd.DataFrame([{"NAME_1": n1, "NAME_2": n2, "GID_2": gid}
                             for gid, (n1, n2, _) in _UNITS.items()])

    def Items(self, admin):  # noqa: N802 - pygadm's API
        self.calls.append(("Items", admin))
        rows = [{"GID_2": gid, "NAME_2": n2, "geometry": poly}
                for gid, (_, n2, poly) in _UNITS.items() if gid == admin]
        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326") if rows else None


class _StopAfterRegion(Exception):
    pass


def _retrieve(tmp_path, monkeypatch, *, field, value, manual=False, version="4.1"):
    fake = _FakePygadm()
    monkeypatch.setitem(sys.modules, "pygadm", fake)
    seen = {}

    def _stop(self, region):
        seen["region"] = region
        raise _StopAfterRegion("stopped after the region")

    monkeypatch.setattr(TranslationPipeline, "_load_climate_data", _stop)
    boundary = {"source": BoundarySource.GADM, "gadm_level": 2,
                "gadm_filter_field": field, "gadm_filter_value": value}
    if manual:
        boundary["manual_bounds"] = ManualBoundsConfig(minx=3.0, miny=4.0, maxx=4.0, maxy=5.0)
    cfg = ProjectConfig(
        project=ProjectInfo(name="gid"),
        region=RegionConfig(name="Bassa", country="Nigeria", country_iso3="NGA",
                            boundary=BoundaryConfig(**boundary)),
        crop=CropConfig(name="Maize", name_short="mai", variety="M",
                        calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285)),
        temporal=TemporalConfig(start_year=2015, end_year=2020, spinup_years=2),
        targets=[Platform.PYTHIA],
        data_sources=DataSourcesConfig(gadm=GadmSourceConfig(base_path=tmp_path / "no_gadm",
                                                             version=version),
                                       cache_enabled=False, cache_dir=tmp_path / "cache"),
        output=OutputConfig(base_dir=str(tmp_path / "out"), structure="by_platform"),
    )
    pipe = TranslationPipeline(cfg, provenance=ProvenanceTracker(enabled=True, project_name="gid"))
    result = pipe._execute_retrieve()
    return result, seen.get("region"), fake.calls


def test_a_gid_config_prepares_exactly_that_unit(tmp_path, monkeypatch):
    _, region, calls = _retrieve(tmp_path, monkeypatch, field="GID_2", value="NGA.32.3_1")
    assert calls == [("Items", "NGA.32.3_1")]
    assert tuple(region.bounds.to_gis_format()) == (8.5, 9.5, 9.0, 10.0)
    assert region.name == "Bassa"


@pytest.mark.parametrize("manual", [False, True])
def test_a_gid_miss_fails_loud_with_no_fallback(tmp_path, monkeypatch, manual):
    result, region, calls = _retrieve(tmp_path, monkeypatch, field="GID_2", value="NGA.32.9_1",
                                      manual=manual)
    assert result.success is False and region is None
    assert ("Names", "NGA") not in calls
    assert any("NGA.32.9_1" in error for error in result.errors)


@pytest.mark.parametrize("value", ["MLI.32.3_1", "NGA.32_1", "NGA.32.3"])
def test_a_gid_that_does_not_fit_its_country_and_level_fails_loud(tmp_path, monkeypatch, value):
    result, region, calls = _retrieve(tmp_path, monkeypatch, field="GID_2", value=value)
    assert result.success is False and region is None and calls == []
    assert any("GID" in error and value in error for error in result.errors)


def test_a_gid_config_needs_gadm_4_1(tmp_path, monkeypatch):
    result, region, calls = _retrieve(tmp_path, monkeypatch, field="GID_2", value="NGA.32.3_1",
                                      version="3.6")
    assert result.success is False and region is None and calls == []
    assert any("4.1" in error for error in result.errors)


def test_a_name_config_keeps_the_name_lookup(tmp_path, monkeypatch):
    _, region, calls = _retrieve(tmp_path, monkeypatch, field="NAME_2", value="Bassa")
    assert calls == [("Names", "NGA"), ("Items", "NGA.23.2_1")]
    assert tuple(region.bounds.to_gis_format()) == (6.5, 7.5, 7.0, 8.0)


def _shapefile(tmp_path):
    rows = [{"GID_0": "NGA", "NAME_0": "Nigeria", "GID_1": gid.rsplit(".", 1)[0] + "_1", "NAME_1": n1,
             "GID_2": gid, "NAME_2": n2, "geometry": poly} for gid, (n1, n2, poly) in _UNITS.items()]
    path = tmp_path / "gadm41_NGA_2.shp"
    gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_file(path)
    return path


def test_the_shapefile_backend_selects_one_unit_by_gid(tmp_path):
    result = GADMSource(base_path=tmp_path).retrieve(
        shapefile_path=_shapefile(tmp_path), gadm_level=2, filter_field="GID_2",
        filter_value="NGA.32.3_1", country_iso3="NGA", include_geometry=True, use_cache=False)
    assert result.success and tuple(result.data.bounds.to_gis_format()) == (8.5, 9.5, 9.0, 10.0)


def test_the_shapefile_backend_name_filter_is_unchanged(tmp_path):
    result = GADMSource(base_path=tmp_path).retrieve(
        shapefile_path=_shapefile(tmp_path), gadm_level=2, filter_field="NAME_2",
        filter_value="Bassa", country_iso3="NGA", include_geometry=True, use_cache=False)
    assert result.success and tuple(result.data.bounds.to_gis_format()) == (6.5, 7.5, 9.0, 10.0)
