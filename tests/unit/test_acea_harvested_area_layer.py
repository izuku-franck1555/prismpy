"""An ACEA package carries the registered SPAM layer of its vintage (R, I and A) under the engine's
fixed read-keys, verified against the pinned content digests, and declares it: the manifest's
data_sources and both READMEs state it, and a README that cannot state it fails the package."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from prismpy.models.region import BoundingBox, Region
from prismpy.config.schema import (
    BoundaryConfig, BoundarySource, CropCalendarConfig, CropConfig, ManualBoundsConfig, OutputConfig,
    Platform, ProjectConfig, ProjectInfo, RegionConfig, TemporalConfig,
)
from prismpy.packaging.readme_generator import generate_readme
from prismpy.packaging.scenario_set_generator import _write_acea_forced_co2_readme
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.sources.crop_areas import spam as spam_module
from prismpy.sources.crop_areas import spam_vintage as sv
from prismpy.translators.acea import translator as acea
from prismpy.translators.acea.translator import ACEA_FAO_CODE_MAP, SPAM_CODE_MAP, AceaTranslator
from prismpy.translators.base import RequiredPackageArtifactError, TranslationResult
from tests.package_spam import provision_spam

LOCAL_V2R2 = Path(__file__).resolve().parents[3] / "data" / "spam_2020_V2r2"


def _config(tmp_path, *, crop=("Maize", "mai"), vintage=None, spam_dir=None):
    acea_cfg = {"spam_data_dir": str(spam_dir or tmp_path / "spam")}
    if vintage:
        acea_cfg.update(spam_version=vintage[0], spam_release=vintage[1])
    return ProjectConfig(
        project=ProjectInfo(name="acea_layer"),
        region=RegionConfig(name="Mopti", country="Mali", country_iso3="MLI", boundary=BoundaryConfig(
            source=BoundarySource.MANUAL, manual_bounds=ManualBoundsConfig(minx=-4.5, miny=14.0, maxx=-3.5, maxy=15.0))),
        crop=CropConfig(name=crop[0], name_short=crop[1],
                        calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285)),
        temporal=TemporalConfig(start_year=2015, end_year=2016),
        targets=[Platform.ACEA], platform_config={"acea": acea_cfg}, output=OutputConfig(base_dir=str(tmp_path / "out")))


def _unified():
    from prismpy.translators.base import UnifiedData

    return UnifiedData(region=Region(name="Mopti", country="Mali", country_iso3="MLI",
                                     bounds=BoundingBox(minx=-4.5, miny=14.0, maxx=-3.5, maxy=15.0)))


def _translator(tmp_path, **config):
    translator = AceaTranslator(_config(tmp_path, **config))
    translator.output_dir = tmp_path / "pkg"
    translator.output_dir.mkdir(exist_ok=True)
    return translator


def _layer_files(translator, fao):
    folder = translator.output_dir / "harvested_areas" / str(fao)
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def _digest(path):
    with rasterio.open(path) as ds:
        return sv.content_digest(ds)


# ── the carried layer and its declaration ────────────────────────────────────


@pytest.mark.parametrize("nodata", [np.nan, -1.0])
def test_a_default_package_carries_the_registered_2020_v2r2_layer_and_declares_it(tmp_path, monkeypatch, nodata):
    spam = provision_spam(tmp_path / "spam", monkeypatch, nodata=nodata)
    translator = _translator(tmp_path)
    written = translator._write_harvested_area_layer()
    names = [f"spam2020V2r0_global_H_56_{tech}.tif" for tech in "RIA"]
    assert [p.name for p in written] == names and _layer_files(translator, 56) == sorted(names)
    for tech, path in zip("RIA", written):
        source = spam / f"spam2020_V2r2_global_H_MAIZ_{tech}.tif"
        assert _digest(path) == sv.SPAM_CONTENT_DIGESTS[("2020", "V2r2", "MAIZ", tech)] == _digest(source)
        with rasterio.open(path) as copy, rasterio.open(source) as src:
            structure = copy.tags(ns="IMAGE_STRUCTURE")
            assert (structure["COMPRESSION"], structure["PREDICTOR"], copy.profile["tiled"]) == ("DEFLATE", "3", True)
            assert (copy.transform, copy.crs, copy.dtypes, copy.nodata is None or np.isnan(copy.nodata) == np.isnan(src.nodata)) == (
                src.transform, src.crs, src.dtypes, True)
            assert np.array_equal(copy.read(1).view(np.uint32), src.read(1).view(np.uint32))
    assert translator._harvested_area_declaration() == {
        "harvested_areas": "SPAM 2020 V2r2 (default; no vintage selected)",
        "harvested_areas_layer": {
            "contract": "acea-spam-identity/1", "year": "2020", "release": "V2r2", "selection": "default",
            "crop": "Maize", "crop_code": "MAIZ", "fao": 56,
            "files": {tech: {"name": f"spam2020V2r0_global_H_56_{tech}.tif",
                             "content_digest": sv.SPAM_CONTENT_DIGESTS[("2020", "V2r2", "MAIZ", tech)]}
                      for tech in "RIA"}},
    }


def test_a_selected_vintage_is_carried_under_the_same_read_keys_and_declares_its_crop_mask(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch, vintage=("2010", "V2r0"), codes=("SMIL",))
    translator = _translator(tmp_path, crop=("Finger Millet", "mil"), vintage=("2010", "V2r0"))
    translator._write_harvested_area_layer()
    declaration = translator._harvested_area_declaration()
    assert declaration["harvested_areas"] == "SPAM 2010 V2r0"
    layer = declaration["harvested_areas_layer"]
    assert (layer["year"], layer["release"], layer["selection"], layer["crop"], layer["crop_code"], layer["fao"]) == (
        "2010", "V2r0", "selected", "Finger Millet", "SMIL", 79)
    assert declaration["crop_mask_vintage"] == {"year": "2010", "release": "V2r0",
                                                "source_filename": "spam2010V2r0_global_H_SMIL_A.tif",
                                                "mask_filename": "spam2020V2r0_global_H_79_A.tif"}


@pytest.mark.skipif(not LOCAL_V2R2.is_dir(), reason="the provisioned SPAM 2020 V2r2 layers are not on this machine")
def test_the_provisioned_maize_layer_is_carried_against_its_real_pinned_digests(tmp_path):
    translator = _translator(tmp_path, spam_dir=LOCAL_V2R2)
    translator._write_harvested_area_layer()
    assert translator._harvested_area_declaration()["harvested_areas_layer"]["crop_code"] == "MAIZ"


# ── failing loud, before anything is written ─────────────────────────────────


def test_an_absent_source_fails_before_anything_is_written(tmp_path):
    translator = _translator(tmp_path)
    with pytest.raises(sv.VintageRasterAbsentError):
        translator._write_harvested_area_layer()
    assert _layer_files(translator, 56) == []


def test_an_unregistered_vintage_fails_before_anything_is_written(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch)
    translator = _translator(tmp_path, vintage=("2020", "V2r0"))
    with pytest.raises(sv.VintageNotRegisteredError):
        translator._write_harvested_area_layer()
    assert _layer_files(translator, 56) == []


def test_a_source_that_is_not_the_registered_content_fails_before_anything_is_written(tmp_path, monkeypatch):
    spam = provision_spam(tmp_path / "spam", monkeypatch)
    source = spam / "spam2020_V2r2_global_H_MAIZ_I.tif"
    with rasterio.open(source, "r+") as ds:
        band = ds.read(1)
        band[10, 10] += 1.0
        ds.write(band, 1)
    copies = []
    monkeypatch.setattr(acea, "_reencode_losslessly", lambda source, target: copies.append(source))
    translator = _translator(tmp_path)
    with pytest.raises(sv.SpamVintageError, match="MAIZ_I"):
        translator._write_harvested_area_layer()
    assert copies == [] and _layer_files(translator, 56) == []


def test_a_copy_that_does_not_reproduce_its_source_is_refused_and_removed(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch)
    real = acea._reencode_losslessly

    def faulty(source, target):
        real(source, target)
        if source.name.endswith("_A.tif"):
            with rasterio.open(target, "r+") as ds:
                band = ds.read(1)
                band[5, 5] += 1.0
                ds.write(band, 1)

    monkeypatch.setattr(acea, "_reencode_losslessly", faulty)
    translator = _translator(tmp_path)
    with pytest.raises(sv.SpamVintageError, match="does not reproduce"):
        translator._write_harvested_area_layer()
    assert _layer_files(translator, 56) == []


def test_a_crop_the_vintage_does_not_map_is_refused(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch)
    monkeypatch.setitem(SPAM_CODE_MAP, "Maize", "ZZZZ")
    with pytest.raises(sv.CropNotInVintageError):
        _translator(tmp_path)._write_harvested_area_layer()


def test_a_crop_without_a_spam_code_is_refused(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch)
    crop = sorted(set(ACEA_FAO_CODE_MAP) - set(SPAM_CODE_MAP))[0]
    with pytest.raises(sv.SpamVintageError, match="has no SPAM code"):
        _translator(tmp_path, crop=(crop, "xx"))._write_harvested_area_layer()


def test_a_failed_copy_leaves_no_layer_file(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch)
    real = acea._reencode_losslessly

    def fail_on_irrigated(source, target):
        if source.name.endswith("_I.tif"):
            raise OSError("disk full")
        real(source, target)

    monkeypatch.setattr(acea, "_reencode_losslessly", fail_on_irrigated)
    translator = _translator(tmp_path)
    with pytest.raises(OSError, match="disk full"):
        translator._write_harvested_area_layer()
    assert _layer_files(translator, 56) == []


def test_a_package_without_its_layer_cannot_declare_one(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch)
    with pytest.raises(RequiredPackageArtifactError, match="spam2020V2r0_global_H_56_R.tif"):
        _translator(tmp_path)._harvested_area_declaration()


# ── what is gone ─────────────────────────────────────────────────────────────


def test_no_placeholder_layer_no_wildcard_and_no_clip_remain():
    assert not hasattr(AceaTranslator, "_generate_dummy_spam_files")
    assert not hasattr(AceaTranslator, "_clip_spam_data")
    for name in ("resolve_acea_spam_input", "acea_spam_input_patterns", "AceaSpamInput"):
        assert not hasattr(acea, name), name
    assert not hasattr(spam_module.SPAMSource, "clip_to_file")


# ── the READMEs state the declaration ────────────────────────────────────────


def _declared(tmp_path, monkeypatch, vintage=None):
    if vintage:
        provision_spam(tmp_path / "spam", monkeypatch, vintage=vintage)
    else:
        provision_spam(tmp_path / "spam", monkeypatch)
    translator = _translator(tmp_path, vintage=vintage)
    translator._write_harvested_area_layer()
    return translator._harvested_area_declaration()


def _row(readme):
    (row,) = [line for line in Path(readme).read_text().splitlines() if line.startswith("| **Harvested Area** |")]
    return row


# The row as worded for the default and a selected vintage (the label as derived for each).
DEFAULT_ROW = ("| **Harvested Area** | SPAM 2020 V2r2 (default; no vintage selected), crop layer MAIZ. Its rainfed, "
               "irrigated and all-technology layers are carried under the engine's fixed file names "
               "`spam2020V2r0_global_H_56_R.tif`, `spam2020V2r0_global_H_56_I.tif` and "
               "`spam2020V2r0_global_H_56_A.tif`, which name the engine's read keys, not the SPAM version. "
               "| Crop area fractions |")
SELECTED_ROW = DEFAULT_ROW.replace("SPAM 2020 V2r2 (default; no vintage selected)", "SPAM 2010 V2r0")


@pytest.mark.parametrize("vintage, row", [(None, DEFAULT_ROW), (("2010", "V2r0"), SELECTED_ROW)])
def test_the_readme_states_the_declared_layer(tmp_path, monkeypatch, vintage, row):
    declaration = _declared(tmp_path, monkeypatch, vintage)
    readme = generate_readme(tmp_path / "README.md", {"data_sources": declaration}, platform="acea", soil_label="x")
    assert _row(readme) == row


def test_a_readme_without_the_declaration_is_refused(tmp_path):
    with pytest.raises(RequiredPackageArtifactError, match="harvested-area declaration"):
        generate_readme(tmp_path / "README.md", {"data_sources": {"harvested_areas": "SPAM 2020"}},
                        platform="acea", soil_label="x")


def test_the_forced_co2_readme_states_its_baseline_declaration(tmp_path, monkeypatch):
    declaration = _declared(tmp_path, monkeypatch)
    package = generate_readme(tmp_path / "README.md", {"data_sources": declaration}, platform="acea", soil_label="x")
    projection = tmp_path / "projection"
    projection.mkdir()
    baseline = {"data_sources": {**declaration, "crop_suitability": "FAO GAEZ v4"},
                "region": {"name": "Mopti"}, "crop": {"name": "Maize"}}
    _write_acea_forced_co2_readme(projection, baseline, co2=600.0, ssp="ssp245", ts_start=2041, ts_end=2060,
                                  b_start=2015, b_end=2016)
    assert _row(projection / "README.md") == _row(package)
    with pytest.raises(RequiredPackageArtifactError):
        _write_acea_forced_co2_readme(projection, {"data_sources": {}, "region": {}, "crop": {}}, co2=600.0,
                                      ssp="ssp245", ts_start=2041, ts_end=2060, b_start=2015, b_end=2016)


# ── a README failure fails the package ───────────────────────────────────────


class _Stub:
    def __init__(self, error):
        self.error = error

    def generate_package(self, unified_data, output_files):
        raise self.error


@pytest.mark.parametrize("error, fatal", [(RequiredPackageArtifactError("README could not be written"), True),
                                          (RuntimeError("another package-file failure"), False)])
def test_a_required_package_file_failure_fails_the_package_stage(tmp_path, error, fatal):
    pipeline = TranslationPipeline(_config(tmp_path))
    pipeline._translators[Platform.ACEA] = _Stub(error)
    result = TranslationResult(success=True, platform=Platform.ACEA, output_dir=tmp_path / "pkg", output_files=[],
                               errors=[], warnings=[], metadata={})
    stage = pipeline._execute_package(_unified(), {"acea": result})
    assert any(str(error) in e for e in stage.errors) == fatal
    assert any(str(error) in w for w in stage.warnings) == (not fatal)
    if fatal:
        assert stage.success is False


def test_an_acea_readme_failure_is_the_typed_error(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch)
    translator = _translator(tmp_path)
    translator._write_harvested_area_layer()

    def broken(*args, **kwargs):
        raise ValueError("template error")

    monkeypatch.setattr("prismpy.packaging.readme_generator.generate_readme", broken)
    monkeypatch.setattr(acea, "generate_readme", broken, raising=False)
    with pytest.raises(RequiredPackageArtifactError, match="template error"):
        translator._generate_package_metadata(_unified(), [], "mopti_nasapower", [])
    json.dumps(translator._harvested_area_declaration())


def test_the_manifest_and_readme_state_the_declaration(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch)
    translator = _translator(tmp_path)
    translator._write_harvested_area_layer()
    declaration = translator._harvested_area_declaration()
    translator._generate_package_metadata(_unified(), [], "mopti_nasapower", [])
    data_sources = json.loads((translator.output_dir / "manifest.json").read_text())["data_sources"]
    assert {key: data_sources.get(key) for key in declaration} == declaration
    assert "crop_mask_vintage" not in data_sources
    assert _row(translator.output_dir / "README.md") == DEFAULT_ROW


def test_a_crop_without_an_acea_fao_code_is_refused(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch)
    monkeypatch.delitem(ACEA_FAO_CODE_MAP, "Maize")
    with pytest.raises(sv.SpamVintageError, match="not an ACEA crop"):
        _translator(tmp_path)._write_harvested_area_layer()
