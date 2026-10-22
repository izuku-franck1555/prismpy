"""SPAM 2020 V2r0 is registered with the inventory ACEA's own folder provisions: 6 crops, strata
A, I and R, and the 18 content digests of those files. Another ACEA crop on it is refused before
any file is read, a present file with other bytes is refused, and the earlier vintages are
unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prismpy.config.schema import (
    BoundaryConfig,
    BoundarySource,
    CropCalendarConfig,
    CropConfig,
    ManualBoundsConfig,
    OutputConfig,
    Platform,
    ProjectConfig,
    ProjectInfo,
    RegionConfig,
    TemporalConfig,
)
from prismpy.sources.crop_areas import spam_vintage as sv
from prismpy.translators.acea.translator import AceaTranslator
from tests.package_spam import provision_spam

V2R0 = ("2020", "V2r0")
CROPS = frozenset({"COWP", "GROU", "MAIZ", "PMIL", "RICE", "SORG"})
ACEA_FOLDER = Path.home() / "Downloads" / "DATA-TO-MODEL-TRANSLATION" / "data" / "spam"
EARLIER_VINTAGES = "b440a05719f5960cae05212cd7bacf5e7ce5ba48476a7b468f31042bc336af5f"


def _translator(tmp_path, crop, spam_dir):
    translator = AceaTranslator(ProjectConfig(
        project=ProjectInfo(name="acea_v2r0"),
        region=RegionConfig(name="Mopti", country="Mali", country_iso3="MLI", boundary=BoundaryConfig(
            source=BoundarySource.MANUAL, manual_bounds=ManualBoundsConfig(
                minx=-4.5, miny=14.0, maxx=-3.5, maxy=15.0))),
        crop=CropConfig(name=crop[0], name_short=crop[1],
                        calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285)),
        temporal=TemporalConfig(start_year=2015, end_year=2016), targets=[Platform.ACEA],
        platform_config={"acea": {"spam_data_dir": str(spam_dir), "spam_version": "2020",
                                  "spam_release": "V2r0"}},
        output=OutputConfig(base_dir=str(tmp_path / "out"))))
    translator.output_dir = tmp_path / "pkg"
    translator.output_dir.mkdir()
    return translator


def test_2020_v2r0_is_registered_with_its_provisioned_inventory_and_digests():
    spec = sv.SPAM_VINTAGES.get(V2R0)
    assert spec is not None and (spec.pattern, spec.crops, spec.strata) == (
        "spam2020_V2r0_global_H_{code}_{tech}.tif", CROPS, frozenset("AIR"))
    assert {key[2:] for key in sv.SPAM_CONTENT_DIGESTS if key[:2] == V2R0} == {
        (code, tech) for code in CROPS for tech in "AIR"}


@pytest.mark.skipif(not ACEA_FOLDER.is_dir(), reason="ACEA's SPAM folder is not on this machine")
@pytest.mark.parametrize("code,tech", sorted((code, tech) for code in CROPS for tech in "AIR"))
def test_each_provisioned_file_resolves_to_its_pinned_digest(code, tech):
    assert V2R0 in sv.SPAM_VINTAGES
    path = sv.resolve_spam_raster(ACEA_FOLDER, *V2R0, code, tech)
    assert sv.file_content_digest(path) == sv.SPAM_CONTENT_DIGESTS[(*V2R0, code, tech)]


def _refusal(call):
    try:
        call()
    except Exception as exc:   # any failure, so a KeyError reads as the wrong refusal, not a crash
        return exc
    return None


@pytest.mark.parametrize("crop", [("Wheat", "whe"), ("Beans", "bea"), ("Soybean", "soy"),
                                  ("Oats", "oat")])   # Oats has no SPAM code at all
def test_another_acea_crop_on_2020_v2r0_is_refused_before_any_file_is_read(tmp_path, crop):
    assert V2R0 in sv.SPAM_VINTAGES
    refusal = _refusal(_translator(tmp_path, crop, tmp_path / "no-folder")._write_harvested_area_layer)
    assert type(refusal) is sv.CropNotInVintageError, refusal


def test_a_present_file_with_other_bytes_is_refused(tmp_path, monkeypatch):
    assert V2R0 in sv.SPAM_VINTAGES
    provision_spam(tmp_path / "spam", monkeypatch, vintage=V2R0, codes=("MAIZ",))
    monkeypatch.undo()   # keep the synthetic files, drop their pins: the registry's own digests stand
    refusal = _refusal(_translator(tmp_path, ("Maize", "mai"), tmp_path / "spam")._write_harvested_area_layer)
    assert type(refusal) is sv.SpamVintageError, refusal
    assert "is not the registered SPAM 2020 V2r0 MAIZ_" in str(refusal)


def test_the_earlier_vintages_are_unchanged():
    earlier = sorted((list(key), sv.SPAM_CONTENT_DIGESTS[key], sv.SPAM_CONTENT_HEADERS[key])
                     for key in sv.SPAM_CONTENT_DIGESTS if key[:2] != V2R0)
    specs = sorted((list(key), spec.pattern, sorted(spec.crops), sorted(spec.strata))
                   for key, spec in sv.SPAM_VINTAGES.items() if key != V2R0)
    assert hashlib.sha256(json.dumps([earlier, specs], sort_keys=True).encode()).hexdigest() == (
        EARLIER_VINTAGES)
