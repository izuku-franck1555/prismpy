"""A GADM region configured by its GID prepares exactly that unit, in both GADM backends; a GID never
falls back to a name lookup; a NAME config keeps today's behaviour. The six real duplicate district
names of Nigeria (12 GIDs) come from the repo's GADM 4.1 subset, served to the real pygadm by the
local adapter (no network)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import geopandas as gpd
import pygadm
import pytest
from shapely import wkt

import prismpy.gadm_local as gadm_local
from prismpy.config.schema import (
    BoundaryConfig, BoundarySource, CropCalendarConfig, CropConfig, DataSourcesConfig,
    GadmSourceConfig, ManualBoundsConfig, OutputConfig, Platform, ProjectConfig, ProjectInfo,
    RegionConfig, TemporalConfig,
)
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.provenance.tracker import ProvenanceTracker
from prismpy.sources.boundaries.gadm import GADMSource

_GPKG = Path(__file__).resolve().parents[1] / "fixtures" / "gadm_subset_NGA_MLI.gpkg"
_DUPLICATES = {
    "Bassa": ("NGA.23.4_1", "NGA.32.2_1"), "Ifelodun": ("NGA.24.5_1", "NGA.30.16_1"),
    "Irepodun": ("NGA.24.9_1", "NGA.30.20_1"), "Nasarawa": ("NGA.20.31_1", "NGA.26.9_1"),
    "Obi": ("NGA.26.11_1", "NGA.7.14_1"), "Surulere": ("NGA.25.20_1", "NGA.31.33_1"),
}
_GIDS = [(name, gid, other) for name, pair in _DUPLICATES.items()
         for gid, other in (pair, pair[::-1])]
# Each legacy NAME config's region at base 4e763a1 (pygadm: first match; shapefile: union).
_BASE_NAME_REGIONS = {
    ("Bassa", "pygadm"): "0adb24c9d01cf099592fa5a60a741d77373320cbd552928a61d9831caea73b5a",
    ("Bassa", "shapefile"): "28942d125e33878e1ca96ce35754fa162af6ce4cca075749ce79c545c434b51f",
    ("Jos North", "pygadm"): "809a4bcdc0efe58dbe441c56aa162f88619dd34d3c1707ced838efc441b179af",
    ("Jos North", "shapefile"): "809a4bcdc0efe58dbe441c56aa162f88619dd34d3c1707ced838efc441b179af",
}


@pytest.fixture(scope="module")
def nga():
    rows = gpd.read_file(_GPKG, layer="gadm_410")
    return rows[rows["GID_0"] == "NGA"]


@pytest.fixture
def local_pygadm(monkeypatch):
    """The real pygadm, served offline by the local GADM 4.1 adapter on the repo subset."""
    monkeypatch.setattr(gadm_local.LocalGADMAdapter, "_verify_artifact", lambda self: True)
    prior = pygadm.session.adapters.get(gadm_local.GADM_HOST)
    prior_disabled = pygadm.session.settings.disabled
    pygadm.session.mount(gadm_local.GADM_HOST, gadm_local.LocalGADMAdapter(str(_GPKG)))
    pygadm.session.settings.disabled = True
    yield
    pygadm.session.adapters.pop(gadm_local.GADM_HOST, None)
    if prior is not None:
        pygadm.session.mount(gadm_local.GADM_HOST, prior)
    pygadm.session.settings.disabled = prior_disabled


def _shapefile_dir(tmp_path, nga):
    directory = tmp_path / "gadm"
    directory.mkdir()
    nga[["GID_0", "NAME_0", "GID_1", "NAME_1", "GID_2", "NAME_2", "geometry"]].to_file(
        directory / "gadm41_NGA_2.shp")
    return directory


class _StopAfterRegion(Exception):
    pass


def _retrieve(tmp_path, monkeypatch, *, field, value, gadm_dir=None, manual=False, version="4.1"):
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
        region=RegionConfig(name="Unit", country="Nigeria", country_iso3="NGA",
                            boundary=BoundaryConfig(**boundary)),
        crop=CropConfig(name="Maize", name_short="mai", variety="M",
                        calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285)),
        temporal=TemporalConfig(start_year=2015, end_year=2020, spinup_years=2),
        targets=[Platform.PYTHIA],
        data_sources=DataSourcesConfig(
            gadm=GadmSourceConfig(base_path=gadm_dir or tmp_path / "no_gadm", version=version),
            cache_enabled=False, cache_dir=tmp_path / "cache"),
        output=OutputConfig(base_dir=str(tmp_path / "out"), structure="by_platform"),
    )
    result = TranslationPipeline(cfg, provenance=ProvenanceTracker(enabled=True, project_name="gid"))._execute_retrieve()
    return result, seen.get("region")


def _unit(nga, gid):
    return nga[nga["GID_2"] == gid].geometry.union_all()


def _region_digest(region):
    import shapely

    geometry = wkt.loads(region.geometry_wkt)
    key = (tuple(round(v, 9) for v in region.bounds.to_gis_format()), round(geometry.area, 12),
           len(shapely.get_coordinates(geometry)))
    return hashlib.sha256(repr(key).encode()).hexdigest()


@pytest.mark.parametrize("name,gid,other", _GIDS)
def test_the_pygadm_path_loads_exactly_the_selected_unit(tmp_path, monkeypatch, local_pygadm, nga,
                                                          name, gid, other):
    _, region = _retrieve(tmp_path, monkeypatch, field="GID_2", value=gid)
    unit, namesake = _unit(nga, gid), _unit(nga, other)
    assert region.bounds.to_gis_format() == pytest.approx(list(unit.bounds))
    assert wkt.loads(region.geometry_wkt).area == pytest.approx(unit.area)
    assert region.bounds.to_gis_format() != pytest.approx(list(namesake.bounds))


@pytest.mark.parametrize("name,gid,other", _GIDS)
def test_the_shapefile_backend_selects_one_unit(tmp_path, nga, name, gid, other):
    result = GADMSource(base_path=tmp_path).retrieve(
        shapefile_path=_shapefile_dir(tmp_path, nga) / "gadm41_NGA_2.shp", gadm_level=2,
        filter_field="GID_2", filter_value=gid, country_iso3="NGA", include_geometry=True,
        use_cache=False)
    assert result.success and result.data.metadata["feature_count"] == 1
    assert result.data.bounds.to_gis_format() == pytest.approx(list(_unit(nga, gid).bounds))


@pytest.mark.parametrize("with_shapefile", [False, True])
@pytest.mark.parametrize("manual", [False, True])
def test_a_gid_miss_fails_loud_in_both_backends(tmp_path, monkeypatch, local_pygadm, nga,
                                                with_shapefile, manual):
    gadm_dir = _shapefile_dir(tmp_path, nga) if with_shapefile else None
    result, region = _retrieve(tmp_path, monkeypatch, field="GID_2", value="NGA.99.99_1",
                               gadm_dir=gadm_dir, manual=manual)
    assert result.success is False and region is None
    assert any("NGA.99.99_1" in error for error in result.errors)


@pytest.mark.parametrize("value", ["MLI.32.2_1", "NGA.32_1", "NGA.32.2"])
def test_a_gid_that_does_not_fit_its_country_and_level_fails_loud(tmp_path, monkeypatch, local_pygadm,
                                                                  value):
    result, region = _retrieve(tmp_path, monkeypatch, field="GID_2", value=value)
    assert result.success is False and region is None
    assert any("GID" in error and value in error for error in result.errors)


def test_a_gid_config_needs_gadm_4_1(tmp_path, monkeypatch, local_pygadm):
    result, region = _retrieve(tmp_path, monkeypatch, field="GID_2", value="NGA.32.2_1", version="3.6")
    assert result.success is False and region is None
    assert any("4.1" in error for error in result.errors)


@pytest.mark.parametrize("value,backend", [
    ("Bassa", "pygadm"), ("Bassa", "shapefile"), ("Jos North", "pygadm"), ("Jos North", "shapefile"),
])
def test_a_legacy_name_config_is_unchanged(tmp_path, monkeypatch, local_pygadm, nga, value, backend):
    gadm_dir = _shapefile_dir(tmp_path, nga) if backend == "shapefile" else None
    _, region = _retrieve(tmp_path, monkeypatch, field="NAME_2", value=value, gadm_dir=gadm_dir)
    assert _region_digest(region) == _BASE_NAME_REGIONS[(value, backend)]


def test_gadm_4_1_is_the_one_version():
    assert gadm_local.EXPECTED_GADM_DATASET_VERSION == "410"
    assert GadmSourceConfig().version == "4.1"
    assert "gadm41_{" in Path(gadm_local.__file__).parent.joinpath("pipeline", "executor.py").read_text()
