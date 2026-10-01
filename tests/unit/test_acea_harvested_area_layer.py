"""An ACEA package carries the registered SPAM layer of its vintage (R, I and A) under the engine's
fixed read-keys, verified against the pinned content digests, and declares it: the manifest's
data_sources and both READMEs state it, and a README that cannot state it fails the package."""
from __future__ import annotations

import json
import shutil
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
from prismpy.packaging.scenario_set_generator import (
    _verify_carried_harvested_area_layer, _write_acea_forced_co2_readme, finalize_acea_forced_co2_projection,
)
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



def test_a_second_build_in_the_same_output_carries_only_its_own_layer(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch, codes=("MAIZ", "RICE"))
    _translator(tmp_path)._write_harvested_area_layer()
    rice = _translator(tmp_path, crop=("Rice", "ric"))
    (rice.output_dir / "harvested_areas" / "27").mkdir()
    (rice.output_dir / "harvested_areas" / "27" / ".left.partial").write_bytes(b"x")
    rice._write_harvested_area_layer()
    root = rice.output_dir / "harvested_areas"
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == [
        "27", *(f"27/spam2020V2r0_global_H_27_{tech}.tif" for tech in "AIR")]


def _projection(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch)
    translator = _translator(tmp_path)
    translator._write_harvested_area_layer()
    baseline = {"data_sources": translator._harvested_area_declaration(),
                "temporal": {"start_year": 2015, "end_year": 2016}}
    (translator.output_dir / "manifest.json").write_text(json.dumps(baseline))
    projection = tmp_path / "projection"
    shutil.copytree(translator.output_dir, projection)
    return translator.output_dir, projection, baseline


def test_a_projection_carrying_its_baselines_layer_is_verified(tmp_path, monkeypatch):
    _, projection, baseline = _projection(tmp_path, monkeypatch)
    _verify_carried_harvested_area_layer(projection, baseline)


@pytest.mark.parametrize("fault", ["corrupted", "missing"])
def test_a_projection_whose_carried_layer_is_not_the_declared_one_is_refused(tmp_path, monkeypatch, fault):
    package, projection, _ = _projection(tmp_path, monkeypatch)
    alias = projection / "harvested_areas" / "56" / "spam2020V2r0_global_H_56_I.tif"
    if fault == "missing":
        alias.unlink()
    else:
        with rasterio.open(alias, "r+") as ds:
            band = ds.read(1)
            band[3, 3] += 1.0
            ds.write(band, 1)
    with pytest.raises(RequiredPackageArtifactError, match="MAIZ_I"):
        finalize_acea_forced_co2_projection(projection, package)



@pytest.mark.parametrize("linked", ["harvested_areas", "harvested_areas/56"])
def test_a_symlinked_layer_folder_is_refused_and_nothing_outside_is_touched(tmp_path, monkeypatch, linked):
    provision_spam(tmp_path / "spam", monkeypatch)
    outside = tmp_path / "outside"
    (outside / "56").mkdir(parents=True)
    (outside / "keep.txt").write_text("x")
    (outside / "56" / "keep.tif").write_text("y")
    translator = _translator(tmp_path)
    link = translator.output_dir / linked
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside if linked == "harvested_areas" else outside / "56", target_is_directory=True)
    with pytest.raises(RequiredPackageArtifactError, match="symbolic link"):
        translator._write_harvested_area_layer()
    assert sorted(p.relative_to(outside).as_posix() for p in outside.rglob("*")) == ["56", "56/keep.tif", "keep.txt"]


def test_a_symlinked_stray_is_unlinked_and_what_it_points_to_is_kept(tmp_path, monkeypatch):
    provision_spam(tmp_path / "spam", monkeypatch)
    outside = tmp_path / "outside"
    (outside / "dir").mkdir(parents=True)
    (outside / "dir" / "a.txt").write_text("a")
    (outside / "file.txt").write_text("f")
    translator = _translator(tmp_path)
    layer = translator.output_dir / "harvested_areas"
    (layer / "56").mkdir(parents=True)
    (layer / "99").symlink_to(outside / "dir", target_is_directory=True)
    (layer / "56" / "stray.tif").symlink_to(outside / "file.txt")
    translator._write_harvested_area_layer()
    assert [p.name for p in layer.iterdir()] == ["56"]
    assert _layer_files(translator, 56) == sorted(f"spam2020V2r0_global_H_56_{tech}.tif" for tech in "RIA")
    assert (outside / "dir" / "a.txt").read_text() == "a" and (outside / "file.txt").read_text() == "f"


def test_a_truncated_layer_file_is_the_typed_error(tmp_path, monkeypatch):
    package, projection, baseline = _projection(tmp_path, monkeypatch)
    for root in (package, projection):
        alias = root / "harvested_areas" / "56" / "spam2020V2r0_global_H_56_R.tif"
        alias.write_bytes(alias.read_bytes()[:200])
    with pytest.raises(RequiredPackageArtifactError, match="MAIZ_R"):
        _verify_carried_harvested_area_layer(projection, baseline)
    with pytest.raises(RequiredPackageArtifactError, match="MAIZ_R"):
        _translator(tmp_path)._harvested_area_declaration()


# ── a projection is complete or absent ───────────────────────────────────────


def _assemble_projection(tmp_path, baseline_root=None, **extra):
    from prismpy.packaging import scenario_set_generator as ssg
    from tests.structural.test_scenario_set_generator import (
        _BASELINE_LABEL, _SLICE, _baseline_fixture, _project_config, _projection_climate)

    return ssg.assemble_projection_package(
        baseline_package=_baseline_fixture(baseline_root or tmp_path), baseline_config=_project_config(tmp_path / "out"),
        projection_climate=_projection_climate(), grid=None, region_name="Kano", crop_name="Cowpea",
        gcm_source="gfdl-esm4", rcp_or_ssp="ssp245", time_slice=_SLICE, baseline_reference_label=_BASELINE_LABEL,
        output_dir=tmp_path / "out", **extra)


def test_a_projection_is_moved_to_its_final_name_only_when_complete(tmp_path):
    projection = _assemble_projection(tmp_path)
    assert projection.is_dir() and [p.name for p in projection.parent.iterdir()] == [projection.name]
    for path in projection.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".md", ".WTH", ".txt", ".yaml"):
            assert ".staging-" not in path.read_text(errors="ignore"), path


@pytest.mark.parametrize("fail_at", ["readme", "manifest", "finalize"])
def test_a_failed_projection_leaves_neither_a_projection_nor_its_staging(tmp_path, monkeypatch, fail_at):
    from prismpy.packaging import scenario_set_generator as ssg

    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    extra = {}
    if fail_at == "readme":
        monkeypatch.setattr(ssg, "_rewrite_projection_readme", disk_full)
    elif fail_at == "manifest":
        monkeypatch.setattr(ssg, "_rewrite_projection_manifest", disk_full)
    else:
        extra["finalize"] = disk_full
    with pytest.raises(OSError, match="disk full"):
        _assemble_projection(tmp_path, **extra)
    assert list((tmp_path / "out").iterdir()) == []


def test_a_projection_whose_finalization_refuses_its_layer_is_not_left_behind(tmp_path):
    from prismpy.packaging import scenario_set_generator as ssg
    from tests.structural.test_scenario_set_generator import _baseline_fixture

    baseline = _baseline_fixture(tmp_path / "b")
    with pytest.raises(RequiredPackageArtifactError, match="declares no harvested-area layer"):
        _assemble_projection(tmp_path, finalize=lambda staged: ssg.finalize_acea_forced_co2_projection(staged, baseline))
    assert list((tmp_path / "out").iterdir()) == []



@pytest.mark.parametrize("target", ["outside_file", "dangling", "outside_directory"])
def test_a_planted_partial_link_is_removed_never_written_through(tmp_path, monkeypatch, target):
    import hashlib

    provision_spam(tmp_path / "spam", monkeypatch)
    outside = tmp_path / "outside"
    (outside / "d").mkdir(parents=True)
    (outside / "users_file.bin").write_bytes(b"precious bytes")
    (outside / "d" / "kept.txt").write_text("kept")

    def state():
        return {p.relative_to(outside).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir"
                for p in sorted(outside.rglob("*"))}

    before = state()
    translator = _translator(tmp_path)
    folder = translator.output_dir / "harvested_areas" / "56"
    folder.mkdir(parents=True)
    link = folder / ".spam2020V2r0_global_H_56_R.tif.partial"
    link.symlink_to({"outside_file": outside / "users_file.bin", "dangling": outside / "new_outside.tif",
                     "outside_directory": outside / "d"}[target])
    translator._write_harvested_area_layer()
    assert state() == before
    assert _layer_files(translator, 56) == sorted(f"spam2020V2r0_global_H_56_{tech}.tif" for tech in "RIA")
    assert not any((folder / name).is_symlink() for name in _layer_files(translator, 56))


def test_a_linked_read_key_is_never_declared(tmp_path, monkeypatch):
    import shutil as sh

    provision_spam(tmp_path / "spam", monkeypatch)
    translator = _translator(tmp_path)
    translator._write_harvested_area_layer()
    read_key = translator.output_dir / "harvested_areas" / "56" / "spam2020V2r0_global_H_56_R.tif"
    same_bytes = tmp_path / "elsewhere.tif"
    sh.copy2(read_key, same_bytes)          # the registered content, held outside the package
    read_key.unlink()
    read_key.symlink_to(same_bytes)
    with pytest.raises(RequiredPackageArtifactError, match="symbolic link"):
        translator._harvested_area_declaration()


def test_a_projection_that_cannot_be_moved_into_place_keeps_the_previous_one(tmp_path, monkeypatch):
    final = _assemble_projection(tmp_path)
    (final / "previous.txt").write_text("the previous projection")
    real = Path.rename

    def failing(self, target):
        if self.name == final.name and self.parent.name.startswith(".staging-"):
            raise OSError("rename failed")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", failing)
    with pytest.raises(OSError, match="rename failed"):
        _assemble_projection(tmp_path, baseline_root=tmp_path / "second")
    assert (final / "previous.txt").read_text() == "the previous projection"
    assert [p.name for p in final.parent.iterdir()] == [final.name]


def test_a_staging_directory_that_cannot_be_removed_is_logged_and_the_new_projection_kept(tmp_path, monkeypatch, caplog):
    from prismpy.packaging import scenario_set_generator as ssg

    final = _assemble_projection(tmp_path)
    (final / "previous.txt").write_text("the previous projection")
    real = ssg.shutil.rmtree

    def failing(path, *args, **kwargs):
        if Path(path).name.startswith(".staging-"):
            raise OSError("permission denied")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(ssg.shutil, "rmtree", failing)
    with caplog.at_level("WARNING", logger="prismpy.packaging.scenario_set_generator"):
        assert _assemble_projection(tmp_path, baseline_root=tmp_path / "second") == final
    assert final.is_dir() and not (final / "previous.txt").exists()
    assert any("could not remove the projection staging directory" in r.getMessage() for r in caplog.records)
