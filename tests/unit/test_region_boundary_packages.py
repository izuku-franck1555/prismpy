"""Every translator writes the region's boundary declaration into the manifest the package ships
(for CRAFT, ACEA and SARRA-Py the one the executor writes last, from the configuration the
translator stashed) and the one boundary label into ``data_sources.boundaries``; the READMEs
render that label."""
from __future__ import annotations

import ast
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from prismpy.config.schema import (
    BoundaryConfig, BoundarySource, CraftConfig, Platform, PlatformConfigGroup, ProjectConfig,
)
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.region import BoundingBox, Region
from prismpy.packaging.manifest import create_manifest, save_manifest
from prismpy.translators import craft
from prismpy.translators.acea.translator import AceaTranslator
from prismpy.translators.base import HwsdOutcome, UnifiedData
from prismpy.translators.pythia.translator import PythiaTranslator
from prismpy.translators.sarra_py.translator import SarraPyTranslator
from tests.package_spam import provision_spam
from tests.unit import test_craft_declared_default_soil as craft_t
from tests.unit.test_acea_harvested_area_layer import _config as _acea_config
from tests.unit.test_craft_spam_vintage_honesty import _cfg as _craft_config
from tests.unit.test_pythia_canonical_substrate_flag import _build_grid_2x3
from tests.unit.test_pythia_canonical_substrate_flag import _build_project_config as _pythia_config

SRC = Path(__file__).resolve().parents[2] / "src" / "prismpy"
BOX = {"minx": -6.0, "miny": 11.0, "maxx": -4.0, "maxy": 13.0}
OFFICIAL = "Official administrative boundaries"
GIDS = ["MLI.6.6_1", "MLI.6.2_1"]

# The states every translator is built in: (the executor's resolved region, the boundary config,
# the declaration, the label).
STATES = {
    "one_unit": (("gadm", 2, {"feature_count": 1, "filter_field": "NAME_2"}),
                 {"gadm_filter_value": "Mopti"},
                 {"boundary_source": "gadm", "gadm_level": 2}, "GADM v4.1 admin level 2"),
    "union": (("gadm", 2, {"feature_count": 2, "filter_field": "NAME_2", "gids": GIDS}),
              {"gadm_filter_value": "Mopti"},
              {"boundary_source": "gadm_union", "gadm_level": 2, "units": 2, "gids": sorted(GIDS)},
              "GADM v4.1 admin level 2, 2 same-named units merged"),
    "ambiguous_pick": (("gadm", 2, {"feature_count": 1, "name_matches": 2, "filter_field": "NAME_2"}),
                       {"gadm_filter_value": "Mopti"}, {"gadm_level": 2}, "GADM v4.1 admin level 2"),
    "manual": (("manual", 2, {}), {"source": BoundarySource.MANUAL, "manual_bounds": BOX},
               {"boundary_source": "manual", "gadm_level": None}, "Bounding box"),
    "shapefile": (("shapefile", 2, {}), {"source": BoundarySource.SHAPEFILE, "shapefile_path": "region.shp"},
                  {"boundary_source": "shapefile", "gadm_level": None}, "Custom shapefile"),
    "gadm_failed_manual": (("manual", 2, {}), {"gadm_filter_value": "Mopti", "manual_bounds": BOX},
                           {"boundary_source": "manual", "gadm_level": None}, "Bounding box"),
}


def _region(source, level, metadata) -> Region:
    return Region(name="Mopti", country="Mali", country_iso3="MLI", bounds=BoundingBox(**BOX), gadm_level=level,
                  metadata=metadata, boundary_source=source)


def _with_boundary(cfg: ProjectConfig, boundary: dict) -> ProjectConfig:
    cfg.region.boundary = BoundaryConfig(**{"source": BoundarySource.GADM, **boundary})
    return cfg


def _final_manifest(translator) -> dict:
    """The manifest the package ships: re-emitted as the executor does from a translator's stash."""
    deferred = getattr(translator, "_deferred_manifest", None)
    if deferred:
        out, package_config, platform, extra = deferred
        save_manifest(create_manifest(out, package_config, platform=platform, additional_metadata=extra),
                      out / "manifest.json")
    return json.loads((translator.output_dir / "manifest.json").read_text())


def _region_block(manifest: dict) -> dict:
    """The manifest's region block without the box's coordinates (``bounds_*``)."""
    return {key: value for key, value in manifest["region"].items() if not key.startswith("bounds_")}


def _pythia(tmp_path, monkeypatch, boundary, region):
    tr = PythiaTranslator(config=_with_boundary(_pythia_config(tmp_path), boundary), output_dir=str(tmp_path / "pkg"),
                          prefer_canonical_substrate=True)
    (tr.output_dir / "config").mkdir(parents=True, exist_ok=True)
    data = UnifiedData(region=region, grid=_build_grid_2x3(), soil={0: craft_t._placeholder()})
    tr._include_eghr_data(data)
    tr._generate_manifest(data)
    tr._generate_readme(data)
    return tr


def _craft(tmp_path, monkeypatch, boundary, region, tr=None, cells=(101, 102)):
    tr = tr or craft.CraftTranslator(_with_boundary(_craft_config(tmp_path), boundary))
    (tr.output_dir / "soil").mkdir(parents=True, exist_ok=True)
    grid = craft_t._grid(list(cells))
    records = [ClimateRecord(date=date(2015, 1, 1) + timedelta(days=d), tmax=32.0, tmin=21.0, precip=2.0,
                             srad=20.0) for d in range(6 * 365)]
    climate = {c.cell_id: ClimateTimeSeries(location_id=c.cell_id, lat=c.lat, lon=c.lon, source="nasa_power",
                                            records=records) for c in grid.cells}
    data = UnifiedData(region=region, grid=grid, climate=climate,
                       soil={cell: craft_t._isda(cell) for cell in cells},
                       soil_cascade=craft_t._state(HwsdOutcome.NO_ANSWER))
    result = tr.translate(data)
    assert result.success, result.errors
    tr.generate_package(data, result.output_files)
    return tr


def _acea(tmp_path, monkeypatch, boundary, region):
    provision_spam(tmp_path / "spam", monkeypatch)
    tr = AceaTranslator(_with_boundary(_acea_config(tmp_path), boundary))
    tr.output_dir = tmp_path / "pkg"
    tr.output_dir.mkdir(exist_ok=True)
    tr._write_harvested_area_layer()
    tr._generate_package_metadata(UnifiedData(region=region), [], "mopti_nasapower", [])
    return tr


def _sarra(tmp_path, monkeypatch, boundary, region):
    cfg = _with_boundary(_acea_config(tmp_path).model_copy(update={"targets": [Platform.SARRA_PY]}), boundary)
    tr = SarraPyTranslator(cfg, output_dir=tmp_path / "pkg")
    tr.output_dir.mkdir(parents=True, exist_ok=True)
    tr._generate_package_files(UnifiedData(region=region), [])
    return tr


BUILDERS = {"pythia": _pythia, "craft": _craft, "acea": _acea, "sarra_py": _sarra}

#: Each label's README description.
DESCRIPTIONS = {
    "GADM v4.1": OFFICIAL,
    "GADM v4.1 admin level 2": OFFICIAL,
    "GADM v4.1 admin level 1": OFFICIAL,
    "GADM v4.1 admin level 2, 2 same-named units merged":
        "Official administrative boundaries; several units share the region's name and were merged",
    "Bounding box": "Manual coordinate bounds",
    "Custom shapefile": "User-provided boundary",
    "not recorded": "—",
}


def _readme_row(package: Path, platform: str):
    """The README's boundary row: PYTHIA and CRAFT whole, SARRA-Py up to its format column, ACEA none."""
    starts = {"pythia": "| Boundary |", "craft": "| **Boundaries** |", "sarra_py": "| Boundaries |"}
    rows = [line for line in (package / "README.md").read_text(encoding="utf-8").splitlines()
            if line.startswith(("| Boundary |", "| **Boundaries** |", "| Boundaries |"))]
    if platform == "acea":
        return rows or None
    (row,) = rows
    assert row.startswith(starts[platform]), row
    return row if platform != "sarra_py" else row.rsplit("| JSON |", 1)[0] + "| JSON |"


def _expected_row(platform: str, label: str):
    return {"pythia": f"| Boundary | {label} | — | {DESCRIPTIONS[label]} |",
            "craft": f"| **Boundaries** | {label} | {DESCRIPTIONS[label]} |",
            "sarra_py": f"| Boundaries | {label} | JSON |", "acea": None}[platform]


@pytest.mark.parametrize("state", list(STATES))
@pytest.mark.parametrize("platform", list(BUILDERS))
def test_each_translator_declares_the_region_and_labels_it(tmp_path, monkeypatch, platform, state) -> None:
    resolved, boundary, declared, label = STATES[state]
    translator = BUILDERS[platform](tmp_path, monkeypatch, boundary, _region(*resolved))
    manifest = _final_manifest(translator)
    assert _region_block(manifest) == {"name": "Mopti", "country": "Mali", **declared}
    assert manifest["data_sources"]["boundaries"] == label
    assert _readme_row(translator.output_dir, platform) == _expected_row(platform, label)


@pytest.mark.parametrize("platform", ["craft", "acea", "sarra_py"])
def test_the_manifest_the_executor_writes_last_keeps_the_declaration(tmp_path, monkeypatch, platform) -> None:
    resolved, boundary, declared, _ = STATES["union"]
    translator = BUILDERS[platform](tmp_path, monkeypatch, boundary, _region(*resolved))
    first = _region_block(json.loads((translator.output_dir / "manifest.json").read_text()))
    assert _region_block(_final_manifest(translator)) == first == {"name": "Mopti", "country": "Mali", **declared}


def test_a_gid_package_declares_the_level_of_its_gid(tmp_path, monkeypatch) -> None:
    region = _region("gadm", 1, {"feature_count": 1, "filter_field": "GID_1"})
    tr = _pythia(tmp_path, monkeypatch, {"gadm_level": None, "gadm_filter_field": "GID_1",
                                         "gadm_filter_value": "MLI.6_1"}, region)
    manifest = _final_manifest(tr)
    assert _region_block(manifest) == {"name": "Mopti", "country": "Mali", "boundary_source": "gadm", "gadm_level": 1}
    assert manifest["data_sources"]["boundaries"] == "GADM v4.1 admin level 1"
    assert _readme_row(tr.output_dir, "pythia") == _expected_row("pythia", "GADM v4.1 admin level 1")


@pytest.mark.parametrize("platform", list(BUILDERS))
def test_a_substituted_level_is_neither_declared_nor_labelled(tmp_path, monkeypatch, platform) -> None:
    """A level-0 config the executor resolved at level 2: no level, no source, a label naming none."""
    region = _region("gadm", 2, {"feature_count": 5, "filter_field": "COUNTRY"})
    translator = BUILDERS[platform](tmp_path, monkeypatch, {"gadm_level": 0, "gadm_filter_field": "COUNTRY",
                                                            "gadm_filter_value": "Mali"}, region)
    manifest = _final_manifest(translator)
    assert _region_block(manifest) == {"name": "Mopti", "country": "Mali", "gadm_level": None}
    assert manifest["data_sources"]["boundaries"] == "GADM v4.1"
    assert _readme_row(translator.output_dir, platform) == _expected_row(platform, "GADM v4.1")


# ── CRAFT's own GADM schema ─────────────────────────────────────────────────


def _craft_on_its_own_unit(tmp_path, monkeypatch, rows: bool):
    """A CRAFT build whose platform config names a GADM unit of its own; the schema step finds
    one row inside the grid (``rows``) or none."""
    cfg = _with_boundary(_craft_config(tmp_path), STATES["one_unit"][1])
    cfg.platform_config = PlatformConfigGroup(craft=CraftConfig(
        gadm_data_path=tmp_path / "gadm", gadm_country_iso3="MLI", gadm_admin_name="Koutiala"))
    tr = craft.CraftTranslator(cfg)
    schema = [{"cellid": tr._to_craft_cellid(101), "share_percent": 100}] if rows else []
    monkeypatch.setattr(craft.CraftTranslator, "_generate_schema_from_gadm", lambda self, **kwargs: (schema, []))
    monkeypatch.setattr(craft.CraftTranslator, "_copy_gadm_shapefile_to_shape_dir", lambda self, *a, **k: None)
    region = _region(*STATES["one_unit"][0])
    return _craft(tmp_path, monkeypatch, None, region, tr=tr), region


def test_cells_from_crafts_own_gadm_unit_declare_nothing_about_the_region(tmp_path, monkeypatch) -> None:
    tr, _ = _craft_on_its_own_unit(tmp_path, monkeypatch, rows=True)
    manifest = _final_manifest(tr)
    assert _region_block(manifest) == {"name": "Mopti", "country": "Mali", "gadm_level": None}
    assert manifest["data_sources"]["boundaries"] == "GADM v4.1"
    assert _readme_row(tr.output_dir, "craft") == _expected_row("craft", "GADM v4.1")


def test_crafts_own_unit_without_cells_falls_back_to_the_region(tmp_path, monkeypatch) -> None:
    tr, _ = _craft_on_its_own_unit(tmp_path, monkeypatch, rows=False)
    assert _region_block(_final_manifest(tr)) == {"name": "Mopti", "country": "Mali", **STATES["one_unit"][2]}


def test_the_next_build_on_the_same_translator_declares_the_region_again(tmp_path, monkeypatch) -> None:
    tr, region = _craft_on_its_own_unit(tmp_path, monkeypatch, rows=True)
    tr.config.platform_config = PlatformConfigGroup(craft=CraftConfig())
    _craft(tmp_path, monkeypatch, None, region, tr=tr)
    assert _region_block(_final_manifest(tr)) == {"name": "Mopti", "country": "Mali", **STATES["one_unit"][2]}


def test_cells_from_the_regions_own_geometry_declare_the_region(tmp_path, monkeypatch) -> None:
    """The schema built from the executor's region geometry simulates the declared region."""
    import geopandas as gpd
    from shapely import wkt

    from prismpy.data_sources.gadm import GADMDataSource

    resolved, boundary, declared, label = STATES["one_unit"]
    region = _region(*resolved)
    region.geometry_wkt = "POLYGON ((-5.0 12.0, -4.9 12.0, -4.9 12.1, -5.0 12.1, -5.0 12.0))"
    rows, _ = GADMDataSource(gadm_path=None).generate_schema_data(
        gdf=gpd.GeoDataFrame(geometry=[wkt.loads(region.geometry_wkt)], crs="EPSG:4326"), resolution_deg=5 / 60,
        decimal_places=2, threshold=BoundaryConfig(gadm_filter_value="Mopti").min_share_percent)
    cells = sorted(row["cellid"] for row in rows)[:2]
    tr = _craft(tmp_path, monkeypatch, boundary, region, cells=cells)
    assert tr._valid_cellids == set(cells)  # the schema came from the region's geometry
    manifest = _final_manifest(tr)
    assert _region_block(manifest) == {"name": "Mopti", "country": "Mali", **declared}
    assert manifest["data_sources"]["boundaries"] == label


# ── create_manifest's region block ──────────────────────────────────────────


PROJECT = {"project_name": "p", "region_name": "Mopti", "country": "Mali", "crop_name": "Millet",
           "start_year": 2015, "end_year": 2016, "data_sources": {"rainfall": "TAMSAT v3.1"}}


def test_a_declaration_without_a_source_writes_no_source(tmp_path) -> None:
    manifest = create_manifest(tmp_path, {**PROJECT, "gadm_level": 2, "region_boundary": {"gadm_level": 1}})
    assert manifest["region"] == {"name": "Mopti", "country": "Mali", "gadm_level": 1}


@pytest.mark.parametrize("config", [{}, {"gadm_level": None}, {"gadm_level": 1}])
def test_a_caller_without_a_declaration_gets_the_level_alone_as_before(tmp_path, config) -> None:
    manifest = create_manifest(tmp_path, {**PROJECT, **config})
    assert manifest["region"] == {"name": "Mopti", "country": "Mali", "gadm_level": config.get("gadm_level", 2)}


# ── One producer of the declaration ─────────────────────────────────────────


def _calls(path: Path, name: str) -> list:
    return [node for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == name]


def test_every_manifest_is_written_from_a_configuration_holding_the_declaration() -> None:
    """Each translator's create_manifest call reads the configuration that carries
    ``region_boundary`` from declared_region_boundary; the executor's late call re-emits a
    translator's own stashed configuration."""
    sites = {"pythia": "project_config", "craft": "package_config", "acea": "package_config",
             "sarra_py": "project_config"}
    for platform, config_name in sites.items():
        source = SRC / "translators" / platform / "translator.py"
        (call,) = _calls(source, "create_manifest")
        passed = call.args[1] if len(call.args) > 1 else next(k.value for k in call.keywords
                                                              if k.arg == "project_config")
        assert ast.unparse(passed) == config_name, platform
        text = source.read_text(encoding="utf-8")
        assert text.count("region_boundary = declared_region_boundary(") == 1 + (platform == "pythia"), platform
        assert f'"region_boundary": region_boundary' in text or "'region_boundary': region_boundary" in text
    (late,) = _calls(SRC / "pipeline" / "executor.py", "_create_manifest")
    assert [ast.unparse(a) for a in late.args] == ["_out_dir", "_pkg_config"]
    owners = [p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
              if "own_unit_override=" in p.read_text(encoding="utf-8") and p.name != "manifest.py"]
    assert owners == ["translators/craft/translator.py"]


# ── The READMEs and the projections ─────────────────────────────────────────


@pytest.mark.parametrize("platform", ["pythia", "craft", "sarra_py"])
def test_a_readme_without_a_boundary_label_says_it_is_not_recorded(tmp_path, platform) -> None:
    from prismpy.packaging.readme_generator import generate_readme
    from tests.package_soil import stamp_package_soil

    stamp_package_soil(tmp_path, platform)
    generate_readme(tmp_path / "README.md", {**PROJECT, "data_sources": {}}, platform=platform)
    assert _readme_row(tmp_path, platform) == _expected_row(platform, "not recorded")
    assert "GADM" not in _readme_row(tmp_path, platform)


BASELINE = {"project_name": "baseline", "platform": "pythia",
            "region": {"name": "Mopti", "country": "Mali", "boundary_source": "gadm", "gadm_level": 1},
            "crop": {"name": "Millet"}, "temporal": {"start_year": 2001, "end_year": 2010},
            "data_sources": {"climate": "NASA POWER", "rainfall": "CHIRPS v2.0", "temperature": "ERA5",
                             "boundaries": "GADM v4.1 admin level 1", "crop_mask": "SPAM 2020 V2r2"}}


def _projection_readme(tmp_path, monkeypatch, baseline):
    """The projection README's config and its boundary row, rebuilt from ``baseline``."""
    from prismpy.packaging import scenario_set_generator as ssg

    seen, real = {}, ssg.generate_readme

    def _capture(path, config, **kwargs):
        seen["config"] = config
        return real(path, config, **kwargs)

    monkeypatch.setattr(ssg, "generate_readme", _capture)
    ssg._rewrite_projection_readme(tmp_path, baseline_manifest=baseline, start_year=2041, end_year=2050,
                                   gcm_source="GFDL-ESM4", rcp_or_ssp="ssp585")
    return seen["config"], _readme_row(tmp_path, "pythia")


def test_a_projection_readme_carries_its_own_climate_and_only_the_baselines_boundary(tmp_path, monkeypatch) -> None:
    config, row = _projection_readme(tmp_path, monkeypatch, BASELINE)
    assert config["data_sources"] == {"climate": "ISIMIP3b GFDL-ESM4 ssp585",
                                      "boundaries": "GADM v4.1 admin level 1"}
    assert row == _expected_row("pythia", "GADM v4.1 admin level 1")


def test_a_projection_of_a_baseline_without_a_boundary_label_says_not_recorded(tmp_path, monkeypatch) -> None:
    baseline = {**BASELINE, "data_sources": {"climate": "NASA POWER"}}
    config, row = _projection_readme(tmp_path, monkeypatch, baseline)
    assert config["data_sources"] == {"climate": "ISIMIP3b GFDL-ESM4 ssp585"}
    assert row == _expected_row("pythia", "not recorded")


def test_a_projection_keeps_its_baselines_region_block(tmp_path) -> None:
    from types import SimpleNamespace

    from prismpy.packaging import scenario_set_generator as ssg

    (tmp_path / "manifest.json").write_text(json.dumps(BASELINE))
    ssg._rewrite_projection_manifest(tmp_path, scenario_block=SimpleNamespace(model_dump=lambda: {}),
                                     gcm_source="GFDL-ESM4", rcp_or_ssp="ssp585", start_year=2041, end_year=2050,
                                     projection_climate={})
    assert json.loads((tmp_path / "manifest.json").read_text())["region"] == BASELINE["region"]


# ── One meaning per key ──────────────────────────────────────────────────────


def test_no_translator_writes_a_boundary_source_into_its_configuration() -> None:
    """The manifest's ``boundary_source`` comes only from the declaration; the label lives in
    ``data_sources.boundaries``."""
    for platform in ("pythia", "craft", "acea", "sarra_py"):
        tree = ast.parse((SRC / "translators" / platform / "translator.py").read_text(encoding="utf-8"))
        keys = [k.value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                for k in node.keys if isinstance(k, ast.Constant)]
        assert "boundary_source" not in keys and "boundary_description" not in keys, platform


def test_a_label_is_derived_only_from_a_runtime_source() -> None:
    for path in SRC.rglob("*.py"):
        for call in _calls(path, "derive_boundary_label"):
            first = call.args[0] if call.args else None
            assert not (isinstance(first, ast.Constant) and first.value in ("gadm_union", None)), path
