"""A package manifest declares its region's boundary only as far as the build proves it.

``declared_region_boundary`` reads the resolved boundary source, the resolved GADM level and the
region's evidence (the features dissolved, the column matched, how many units carried the name,
their GIDs), and writes ``boundary_source`` only when it is proven: one GADM unit, a union of
same-named units, a box or a shapefile. Anything else keeps the level and declares no source.
"""
from __future__ import annotations

import pytest

from prismpy.config.schema import BoundaryConfig
from prismpy.models.region import BoundingBox, Region
from prismpy.packaging.manifest import (
    declared_region_boundary,
    derive_boundary_label,
    describe_boundary_label,
)

UNION_DESCRIPTION = ("Official administrative boundaries; several units share the region's name "
                     "and were merged")
OFFICIAL = "Official administrative boundaries"


def _region(level=2, metadata=None, source="gadm") -> Region:
    return Region(name="Bassa", country="Nigeria", country_iso3="NGA",
                  bounds=BoundingBox(minx=0.0, miny=0.0, maxx=1.0, maxy=1.0), gadm_level=level,
                  metadata={} if metadata is None else metadata, boundary_source=source)


def _config(**kwargs) -> BoundaryConfig:
    """A boundary config; ``gadm_level`` None unless given, so no level counts as substituted."""
    return BoundaryConfig(**{"gadm_level": None, "gadm_filter_value": "Bassa", "shapefile_path": "region.shp",
                             "manual_bounds": {"minx": 0.0, "miny": 0.0, "maxx": 1.0, "maxy": 1.0}, **kwargs})


ONE_UNIT = {"boundary_source": "gadm", "gadm_level": 2}
LEVEL_ONLY = {"gadm_level": 2}

# (id, region level, region metadata, expected declaration)
GADM_ROWS = [
    ("one_unit", 2, {"feature_count": 1, "filter_field": "NAME_2"}, ONE_UNIT),
    ("one_unit_one_name_match", 2, {"feature_count": 1, "name_matches": 1, "filter_field": "NAME_2"}, ONE_UNIT),
    ("one_of_two_same_names_picked", 2, {"feature_count": 1, "name_matches": 2, "filter_field": "NAME_2"},
     LEVEL_ONLY),
    ("no_name_match_recorded", 2, {"feature_count": 1, "name_matches": 0, "filter_field": "NAME_2"}, LEVEL_ONLY),
    ("union_with_gids", 2, {"feature_count": 2, "filter_field": "NAME_2", "gids": ["NGA.32.2_1", "NGA.23.4_1"]},
     {"boundary_source": "gadm_union", "gadm_level": 2, "units": 2, "gids": ["NGA.23.4_1", "NGA.32.2_1"]}),
    ("union_gids_incomplete", 2, {"feature_count": 2, "filter_field": "NAME_2", "gids": ["NGA.23.4_1"]},
     {"boundary_source": "gadm_union", "gadm_level": 2, "units": 2}),
    ("union_gids_not_text", 2, {"feature_count": 2, "filter_field": "NAME_2", "gids": [1, 2]},
     {"boundary_source": "gadm_union", "gadm_level": 2, "units": 2}),
    ("parent_area_cross_level", 2, {"feature_count": 2, "filter_field": "NAME_1"}, LEVEL_ONLY),
    ("other_field", 2, {"feature_count": 2, "filter_field": "VARNAME_2"}, LEVEL_ONLY),
    ("several_by_gid", 2, {"feature_count": 2, "filter_field": "GID_2"}, LEVEL_ONLY),
    ("several_no_field", 2, {"feature_count": 2}, LEVEL_ONLY),
    ("one_unit_cross_level", 2, {"feature_count": 1, "filter_field": "NAME_1"}, ONE_UNIT),
    ("one_unit_other_field", 2, {"feature_count": 1, "filter_field": "HASC_2"}, ONE_UNIT),
    ("one_unit_level_3", 3, {"feature_count": 1, "filter_field": "NAME_3"}, {"boundary_source": "gadm", "gadm_level": 3}),
    ("union_level_3", 3, {"feature_count": 3, "filter_field": "NAME_3"},
     {"boundary_source": "gadm_union", "gadm_level": 3, "units": 3}),
    ("union_level_0", 0, {"feature_count": 2, "filter_field": "COUNTRY"},
     {"boundary_source": "gadm_union", "gadm_level": 0, "units": 2}),
    ("name_column_at_level_0", 0, {"feature_count": 2, "filter_field": "NAME_0"}, {"gadm_level": 0}),
    ("level_out_of_range", 7, {"feature_count": 1, "filter_field": "NAME_7"}, {"gadm_level": 7}),
    ("level_not_an_int", "2", {"feature_count": 1, "filter_field": "NAME_2"}, {"gadm_level": "2"}),
    ("count_is_a_bool", 2, {"feature_count": True, "filter_field": "NAME_2"}, LEVEL_ONLY),
    ("count_is_a_float", 2, {"feature_count": 1.0, "filter_field": "NAME_2"}, LEVEL_ONLY),
    ("count_zero", 2, {"feature_count": 0, "filter_field": "NAME_2"}, LEVEL_ONLY),
    ("name_matches_a_bool", 2, {"feature_count": 1, "name_matches": True, "filter_field": "NAME_2"}, LEVEL_ONLY),
    ("count_a_bool_with_one_match", 2, {"feature_count": True, "name_matches": 1, "filter_field": "NAME_2"},
     LEVEL_ONLY),
    ("metadata_not_a_dict", 2, ["feature_count", 1], LEVEL_ONLY),
    ("no_metadata", 2, {}, LEVEL_ONLY),
    ("fallback_parts_of_one_match", 2, {"feature_count": 2, "name_matches": 1, "filter_field": "NAME_2"},
     LEVEL_ONLY),
    ("fallback_parts_by_gid", 2, {"feature_count": 2, "filter_field": "GID_2"}, LEVEL_ONLY),
]


@pytest.mark.parametrize("level, metadata, expected", [row[1:] for row in GADM_ROWS],
                         ids=[row[0] for row in GADM_ROWS])
def test_a_gadm_region_declares_only_what_its_evidence_proves(level, metadata, expected) -> None:
    assert declared_region_boundary(_region(level, metadata), _config()) == expected


@pytest.mark.parametrize("source, configured, expected", [
    ("manual", "gadm", {"boundary_source": "manual", "gadm_level": None}),
    ("manual_bounds", "gadm", {"boundary_source": "manual", "gadm_level": None}),
    ("shapefile", "gadm", {"boundary_source": "shapefile", "gadm_level": None}),
    (None, "shapefile", {"boundary_source": "shapefile", "gadm_level": None}),
    (None, "manual", {"boundary_source": "manual", "gadm_level": None}),
])
def test_a_box_or_a_shapefile_is_declared_with_no_level(source, configured, expected) -> None:
    region = _region(2, {"feature_count": 1, "filter_field": "NAME_2"}, source=source)
    assert declared_region_boundary(region, _config(source=configured)) == expected


def test_an_unknown_boundary_source_is_refused() -> None:
    with pytest.raises(ValueError, match="geojson"):
        declared_region_boundary(_region(source="geojson"), _config())


def test_a_substituted_level_is_never_recorded() -> None:
    """A level-0 config resolved at level 2: neither level nor source is declared."""
    region = _region(2, {"feature_count": 5, "filter_field": "COUNTRY"})
    assert declared_region_boundary(region, _config(gadm_level=0, gadm_filter_field="COUNTRY")) == {
        "gadm_level": None}


@pytest.mark.parametrize("configured, field, level", [(None, "NAME_2", 2), (2, "NAME_2", 2), (None, "GID_1", 1),
                                                      (1, "GID_1", 1)])
def test_a_level_resolved_as_configured_or_from_a_gid_is_not_a_substitution(configured, field, level) -> None:
    region = _region(level, {"feature_count": 1, "filter_field": field})
    assert declared_region_boundary(region, _config(gadm_level=configured, gadm_filter_field=field)) == {
        "boundary_source": "gadm", "gadm_level": level}


def test_the_schema_default_level_is_declared_with_its_level() -> None:
    """With the level left at the schema's default the region is resolved at 2, declared, and
    labelled with that level."""
    config = BoundaryConfig(gadm_filter_value="Bassa")
    declared = declared_region_boundary(_region(config.gadm_level, {"feature_count": 1, "filter_field": "NAME_2"}),
                                        config)
    assert declared == {"boundary_source": "gadm", "gadm_level": 2}
    assert derive_boundary_label("gadm", declared["gadm_level"]) == ("GADM v4.1 admin level 2", OFFICIAL)


@pytest.mark.parametrize("source", ["gadm", "manual"])
def test_cells_from_a_unit_the_platform_selected_declare_nothing(source) -> None:
    region = _region(2, {"feature_count": 1, "filter_field": "NAME_2"}, source=source)
    assert declared_region_boundary(region, _config(), own_unit_override=True) == {"gadm_level": None}


# ── The labels and their descriptions ────────────────────────────────────────


@pytest.mark.parametrize("args, expected", [
    (("gadm", 2), ("GADM v4.1 admin level 2", OFFICIAL)),
    (("gadm", 3), ("GADM v4.1 admin level 3", OFFICIAL)),
    (("gadm", None), ("GADM v4.1", OFFICIAL)),
    (("gadm", 2, 2), ("GADM v4.1 admin level 2, 2 same-named units merged", UNION_DESCRIPTION)),
    (("gadm", 3, 3), ("GADM v4.1 admin level 3, 3 same-named units merged", UNION_DESCRIPTION)),
    (("gadm", 2, 1), ("GADM v4.1 admin level 2", OFFICIAL)),
    (("gadm", 2, True), ("GADM v4.1 admin level 2", OFFICIAL)),
    (("manual", None), ("Bounding box", "Manual coordinate bounds")),
    (("manual_bounds", None), ("Bounding box", "Manual coordinate bounds")),
    (("shapefile", None), ("Custom shapefile", "User-provided boundary")),
])
def test_each_label_and_its_description(args, expected) -> None:
    assert derive_boundary_label(*args) == expected
    assert describe_boundary_label(expected[0]) == expected[1]


@pytest.mark.parametrize("source", ["gadm_union", None, "geojson"])
def test_a_label_is_never_derived_from_a_declaration_token(source) -> None:
    with pytest.raises(ValueError):
        derive_boundary_label(source, 2)


@pytest.mark.parametrize("label", ["not recorded", None, "GADM", "GADM v4.1 admin level two", ""])
def test_a_label_the_table_does_not_know_has_no_description(label) -> None:
    assert describe_boundary_label(label) == "—"
