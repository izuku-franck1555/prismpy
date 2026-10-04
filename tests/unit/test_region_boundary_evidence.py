"""The region the pipeline resolves carries the evidence its boundary declaration is built from, on
both real paths: the GADM files (``GADMSource``) and the pygadm fallback, run through the
executor's own retrieve step on the repo's GADM 4.1 subset (pygadm served offline)."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pygadm
import pytest

from prismpy.config.schema import (
    BoundaryConfig, BoundarySource, CropCalendarConfig, CropConfig, DataSourcesConfig,
    GadmSourceConfig, OutputConfig, Platform, ProjectConfig, ProjectInfo, RegionConfig, TemporalConfig,
)
from prismpy.packaging.manifest import declared_region_boundary
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.provenance.tracker import ProvenanceTracker
from prismpy.sources.boundaries.gadm import GADMSource
from tests.unit.test_gadm_gid_identity import _GPKG, local_pygadm  # noqa: F401  (a fixture)

COUNTRIES = {"NGA": "Nigeria", "MLI": "Mali"}


@pytest.fixture(scope="module")
def gadm_rows():
    return gpd.read_file(_GPKG, layer="gadm_410")


def _gadm_dir(tmp_path, units, iso3, level):
    """A GADM file of ``units`` (one row per level-``level`` unit) where the executor looks for it."""
    directory = tmp_path / "gadm"
    directory.mkdir(exist_ok=True)
    columns = [c for n in range(level + 1) for c in (f"GID_{n}", f"NAME_{n}")]
    units = units[units["GID_0"] == iso3]
    units = units.dissolve(by=f"GID_{level}", as_index=False)[columns + ["geometry"]]
    units.to_file(directory / f"gadm41_{iso3}_{level}.shp")
    return directory


class _StopAfterRegion(Exception):
    pass


def _resolve(tmp_path, monkeypatch, *, iso3="NGA", level=2, field, value, gadm_dir=None):
    """The executor's retrieve step for a GADM config; returns (region, its boundary config)."""
    seen = {}

    def _stop(self, region):
        seen["region"] = region
        raise _StopAfterRegion("stopped after the region")

    monkeypatch.setattr(TranslationPipeline, "_load_climate_data", _stop)
    boundary = BoundaryConfig(source=BoundarySource.GADM, gadm_level=level, gadm_filter_field=field,
                              gadm_filter_value=value)
    cfg = ProjectConfig(
        project=ProjectInfo(name="boundary"),
        region=RegionConfig(name="Unit", country=COUNTRIES[iso3], country_iso3=iso3, boundary=boundary),
        crop=CropConfig(name="Maize", name_short="mai", variety="M",
                        calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285)),
        temporal=TemporalConfig(start_year=2015, end_year=2020, spinup_years=2),
        targets=[Platform.PYTHIA],
        data_sources=DataSourcesConfig(
            gadm=GadmSourceConfig(base_path=gadm_dir or tmp_path / "no_gadm", version="4.1"),
            cache_enabled=False, cache_dir=tmp_path / "cache"),
        output=OutputConfig(base_dir=str(tmp_path / "out"), structure="by_platform"),
    )
    TranslationPipeline(cfg, provenance=ProvenanceTracker(enabled=True, project_name="boundary"))._execute_retrieve()
    assert "region" in seen, "the retrieve step resolved no region"
    return seen["region"], boundary


# ── The GADM files ──────────────────────────────────────────────────────────


def test_same_named_units_dissolved_by_name_are_declared_a_union_with_their_gids(tmp_path, monkeypatch,
                                                                                    gadm_rows):
    region, boundary = _resolve(tmp_path, monkeypatch, field="NAME_2", value="Bassa",
                                gadm_dir=_gadm_dir(tmp_path, gadm_rows, "NGA", 2))
    assert region.metadata["feature_count"] == 2
    assert sorted(region.metadata["gids"]) == ["NGA.23.4_1", "NGA.32.2_1"]
    assert declared_region_boundary(region, boundary) == {
        "boundary_source": "gadm_union", "gadm_level": 2, "units": 2, "gids": ["NGA.23.4_1", "NGA.32.2_1"]}


@pytest.mark.parametrize("field, value", [("NAME_2", "Jos North"), ("GID_2", "NGA.32.2_1")])
def test_one_unit_by_name_or_gid_is_declared(tmp_path, monkeypatch, gadm_rows, field, value) -> None:
    region, boundary = _resolve(tmp_path, monkeypatch, field=field, value=value,
                                gadm_dir=_gadm_dir(tmp_path, gadm_rows, "NGA", 2))
    assert region.metadata["feature_count"] == 1 and region.metadata["filter_field"] == field
    assert declared_region_boundary(region, boundary) == {"boundary_source": "gadm", "gadm_level": 2}


def test_one_level_3_unit_by_gid_is_declared_at_level_3(tmp_path, monkeypatch, gadm_rows) -> None:
    region, boundary = _resolve(tmp_path, monkeypatch, iso3="MLI", level=3, field="GID_3", value="MLI.1.1.1_1",
                                gadm_dir=_gadm_dir(tmp_path, gadm_rows, "MLI", 3))
    assert region.metadata["gids"] == ["MLI.1.1.1_1"]
    assert declared_region_boundary(region, boundary) == {"boundary_source": "gadm", "gadm_level": 3}


def test_a_parent_area_matched_by_its_own_name_is_not_declared_same_named_units(tmp_path, monkeypatch,
                                                                                 gadm_rows) -> None:
    """Kano: 44 level-2 units carry NAME_1 "Kano"; they are the Kano state, not 44 namesakes."""
    region, boundary = _resolve(tmp_path, monkeypatch, field="NAME_1", value="Kano",
                                gadm_dir=_gadm_dir(tmp_path, gadm_rows, "NGA", 2))
    assert region.metadata["feature_count"] == 44 and region.metadata["filter_field"] == "NAME_1"
    assert declared_region_boundary(region, boundary) == {"gadm_level": 2}


def test_one_unit_matched_through_its_parents_name_is_declared(tmp_path, monkeypatch, gadm_rows) -> None:
    """A level-2 file in which the Kano state holds one unit: the match is that one unit."""
    kano = gadm_rows[gadm_rows["NAME_1"] == "Kano"]
    one = pd.concat([gadm_rows[gadm_rows["NAME_1"] != "Kano"], kano[kano["GID_2"] == kano["GID_2"].iloc[0]]])
    region, boundary = _resolve(tmp_path, monkeypatch, field="NAME_1", value="Kano",
                                gadm_dir=_gadm_dir(tmp_path, gpd.GeoDataFrame(one, crs=gadm_rows.crs), "NGA", 2))
    assert region.metadata["feature_count"] == 1
    assert declared_region_boundary(region, boundary) == {"boundary_source": "gadm", "gadm_level": 2}


def test_the_bounds_cache_keeps_the_evidence(tmp_path, gadm_rows) -> None:
    """A region served from the bounds cache (bounds only) carries the evidence it was built with."""
    shapefile = _gadm_dir(tmp_path, gadm_rows, "NGA", 2) / "gadm41_NGA_2.shp"
    source = GADMSource(base_path=tmp_path / "gadm", cache_dir=tmp_path / "cache")
    kwargs = dict(shapefile_path=shapefile, gadm_level=2, filter_field="NAME_2", filter_value="Bassa",
                  country_iso3="NGA", include_geometry=False, use_cache=True)
    built, cached = source.retrieve(**kwargs), source.retrieve(**kwargs)
    assert cached.metadata["from_cache"] is True
    evidence = {key: built.data.metadata[key] for key in ("feature_count", "filter_field", "gids")}
    assert {key: cached.data.metadata[key] for key in evidence} == evidence
    assert evidence["feature_count"] == 2 and len(evidence["gids"]) == 2


# ── The pygadm fallback ─────────────────────────────────────────────────────


def test_a_name_shared_by_two_units_is_never_declared(tmp_path, monkeypatch, local_pygadm) -> None:
    """pygadm picks the first of the two Bassa units: one unit was built, but not one proven."""
    region, boundary = _resolve(tmp_path, monkeypatch, field="NAME_2", value="Bassa")
    assert region.metadata == {"feature_count": 1, "name_matches": 2, "filter_field": "NAME_2"}
    assert declared_region_boundary(region, boundary) == {"gadm_level": 2}


@pytest.mark.parametrize("field, value, evidence", [
    ("NAME_2", "Jos North", {"feature_count": 1, "name_matches": 1, "filter_field": "NAME_2"}),
    ("GID_2", "NGA.32.2_1", {"feature_count": 1, "filter_field": "GID_2"}),
])
def test_one_unit_found_by_pygadm_is_declared(tmp_path, monkeypatch, local_pygadm, field, value, evidence) -> None:
    region, boundary = _resolve(tmp_path, monkeypatch, field=field, value=value)
    assert region.metadata == evidence
    assert declared_region_boundary(region, boundary) == {"boundary_source": "gadm", "gadm_level": 2}


@pytest.mark.parametrize("field, value, evidence", [
    ("GID_2", "NGA.32.2_1", {"feature_count": 2, "filter_field": "GID_2"}),
    ("NAME_2", "Jos North", {"feature_count": 2, "name_matches": 1, "filter_field": "NAME_2"}),
])
def test_a_unit_pygadm_returns_in_several_rows_is_never_declared(tmp_path, monkeypatch, local_pygadm, field,
                                                                  value, evidence) -> None:
    """pygadm's actual row count is recorded before the rows are dissolved. (pygadm's own result
    class cannot be dissolved, so the two rows are served as a plain GeoDataFrame.)"""
    real_items = pygadm.Items

    def _two_rows(*args, **kwargs):
        rows = real_items(*args, **kwargs)
        return gpd.GeoDataFrame(pd.concat([pd.DataFrame(rows)] * 2, ignore_index=True), geometry="geometry",
                                crs=rows.crs)

    monkeypatch.setattr(pygadm, "Items", _two_rows)
    region, boundary = _resolve(tmp_path, monkeypatch, field=field, value=value)
    assert region.metadata == evidence
    assert declared_region_boundary(region, boundary) == {"gadm_level": 2}
