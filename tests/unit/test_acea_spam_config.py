"""ACEA's harvested-area configuration: the SPAM vintage it selects (or the default), the SPAM
directory and the carried layer it needs only when ACEA is enabled, and a crop-presence rule that
must name the same SPAM layer ACEA uses."""
from __future__ import annotations

from pathlib import Path

import pytest

from prismpy.config.loader import load_config
from prismpy.config.schema import (
    AceaConfig, BoundaryConfig, BoundarySource, CropCalendarConfig, CropConfig, CropPresenceRule,
    ManualBoundsConfig, OutputConfig, Platform, ProjectConfig, ProjectInfo, RegionConfig, TemporalConfig,
)


def _rule(year, release, code):
    return CropPresenceRule(year=year, release=release, crop_code=code, sha256="0" * 64, crs="EPSG:4326",
                            transform=(1 / 12, 0.0, -180.0, 0.0, -1 / 12, 90.0), width=4320, height=2160,
                            nodata="nan")


def _config(*, targets=("acea",), acea=None, rule=None, crop=("Maize", "mai"), base_dir="out"):
    boundary = {"source": BoundarySource.MANUAL,
                "manual_bounds": ManualBoundsConfig(minx=-6.0, miny=11.0, maxx=-5.0, maxy=12.0)}
    if rule is not None:
        boundary["crop_presence"] = rule
    return ProjectConfig(
        project=ProjectInfo(name="acea_spam"),
        region=RegionConfig(name="Koutiala", country="Mali", country_iso3="MLI", grid_resolution="5arcmin",
                            boundary=BoundaryConfig(**boundary)),
        crop=CropConfig(name=crop[0], name_short=crop[1],
                        calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285)),
        temporal=TemporalConfig(start_year=2015, end_year=2016),
        targets=[Platform(t) for t in targets],
        platform_config={} if acea is None else {"acea": acea},
        output=OutputConfig(base_dir=base_dir),
    )


SPAM = {"spam_data_dir": "spam"}


# ── the SPAM directory and the carried layer, only when ACEA is enabled ──────


def test_a_project_without_acea_needs_no_spam_directory():
    assert _config(targets=("pythia",)).platform_config.acea.spam_data_dir is None


@pytest.mark.parametrize("targets, acea", [(("pythia",), {"enabled": True}), (("acea",), {"enabled": False})])
def test_an_acea_block_of_a_project_that_does_not_run_acea_needs_no_spam_directory(targets, acea):
    _config(targets=targets, acea=acea)


def test_acea_enabled_without_a_spam_directory_is_refused():
    with pytest.raises(ValueError, match="spam_data_dir is required when ACEA is enabled"):
        _config()


def test_acea_cannot_leave_its_harvested_area_layer_out_of_the_package():
    with pytest.raises(ValueError, match="include_spam_in_package cannot be false"):
        _config(acea={**SPAM, "include_spam_in_package": False})
    _config(targets=("pythia",), acea={"include_spam_in_package": False})


def test_spam_required_is_still_accepted():
    _config(acea={**SPAM, "spam_required": True})


# ── the selected vintage, or the default ─────────────────────────────────────


@pytest.mark.parametrize("fields", [{"spam_version": "2010"}, {"spam_release": "V2r0"}])
def test_a_vintage_is_selected_by_its_year_and_release_together(fields):
    with pytest.raises(ValueError, match="set both, or neither"):
        AceaConfig(**fields)


def test_no_selection_applies_the_default_vintage_and_a_selection_applies_itself():
    assert AceaConfig().applied_spam_vintage() == ("2020", "V2r2", "default")
    assert AceaConfig(spam_version="2010", spam_release="V2r0").applied_spam_vintage() == (
        "2010", "V2r0", "selected")
    assert AceaConfig(spam_version="2020", spam_release="V2r2").applied_spam_vintage() == (
        "2020", "V2r2", "selected")


# ── a crop-presence rule names the layer ACEA uses ───────────────────────────


def test_a_rule_of_another_vintage_is_refused_naming_both():
    with pytest.raises(ValueError, match="SPAM 2020 V2r2 MAIZ, but ACEA uses SPAM 2010 V2r0 MAIZ"):
        _config(acea={**SPAM, "spam_version": "2010", "spam_release": "V2r0"}, rule=_rule(2020, "V2r2", "MAIZ"))


def test_a_rule_of_the_same_vintage_and_crop_builds():
    _config(acea=SPAM, rule=_rule(2020, "V2r2", "MAIZ"))
    _config(acea={**SPAM, "spam_version": "2010", "spam_release": "V2r0"}, rule=_rule(2010, "V2r0", "MAIZ"))


def test_a_rule_for_another_crop_of_the_same_vintage_is_refused():
    with pytest.raises(ValueError, match="SPAM 2020 V2r2 WHEA, but ACEA uses SPAM 2020 V2r2 MAIZ"):
        _config(acea=SPAM, rule=_rule(2020, "V2r2", "WHEA"))


@pytest.mark.parametrize("vintage, code", [(("2020", "V2r2"), "MILL"), (("2010", "V2r0"), "SMIL")])
def test_a_renamed_crop_is_compared_in_its_vintage_spelling(vintage, code):
    acea = {**SPAM, "spam_version": vintage[0], "spam_release": vintage[1]}
    _config(acea=acea, rule=_rule(int(vintage[0]), vintage[1], code), crop=("Finger Millet", "mil"))
    other = "SMIL" if code == "MILL" else "MILL"
    with pytest.raises(ValueError, match=f"{other}, but ACEA uses SPAM {vintage[0]} {vintage[1]} {code}"):
        _config(acea=acea, rule=_rule(int(vintage[0]), vintage[1], other), crop=("Finger Millet", "mil"))


def test_a_rule_is_not_checked_against_acea_when_acea_does_not_run():
    _config(targets=("pythia",), rule=_rule(2010, "V2r0", "WHEA"))


# ── every configuration the repository ships still validates ────────────────

_ROOT = Path(__file__).resolve().parents[2]
# The full project configurations (configs/domes and configs/templates hold fragments of one).
SHIPPED = sorted((_ROOT / "configs" / "base").glob("*.yaml")) + sorted((_ROOT / "examples").glob("*.yaml"))


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_shipped_configuration_validates(path):
    load_config(path)


# ── targets switched to ACEA after the configuration was built are checked again ──

_AFTER = {"include_false": ({**SPAM, "include_spam_in_package": False}, None, "include_spam_in_package cannot be false"),
          "rule_mismatch": (SPAM, ("2010", "V2r0", "WHEA"), "but ACEA uses SPAM 2020 V2r2 MAIZ"),
          "no_spam_dir": ({}, None, "spam_data_dir is required when ACEA is enabled")}


@pytest.mark.parametrize("case", sorted(_AFTER))
def test_targets_switched_to_acea_are_refused_before_anything_is_written(tmp_path, case):
    from prismpy.pipeline.executor import TranslationPipeline

    acea, rule, message = _AFTER[case]
    rule = _rule(int(rule[0]), rule[1], rule[2]) if rule else None
    out = tmp_path / "out"
    for switch in (lambda c: setattr(c, "targets", [Platform.ACEA]), lambda c: c.targets.append(Platform.ACEA)):
        cfg = _config(targets=("pythia",), acea=acea, rule=rule, base_dir=str(out))
        switch(cfg)
        with pytest.raises(ValueError, match=message):
            TranslationPipeline(cfg)
    assert not out.exists()


@pytest.mark.parametrize("case", sorted(_AFTER))
def test_the_cli_refuses_targets_switched_to_acea_before_anything_is_written(tmp_path, case):
    import argparse

    from prismpy.cli import cmd_translate
    from prismpy.config.loader import save_config

    acea, rule, _ = _AFTER[case]
    rule = _rule(int(rule[0]), rule[1], rule[2]) if rule else None
    out = tmp_path / "out"
    path = tmp_path / "config.yaml"
    save_config(_config(targets=("pythia",), acea=acea, rule=rule, base_dir=str(out)), path)
    assert cmd_translate(argparse.Namespace(base=None, dome=None, config=str(path), targets=["acea"])) == 1
    assert not out.exists()


# ── an ACEA translator refuses, whatever the targets, before writing ─────────

@pytest.mark.parametrize("acea, rule", [({}, None), ({"enabled": False}, None),
                                        ({**SPAM, "include_spam_in_package": False}, None),
                                        (SPAM, ("2010", "V2r0", "WHEA"))])
def test_an_acea_translator_refuses_a_project_whose_acea_rules_fail_before_writing(tmp_path, acea, rule):
    from prismpy.translators.acea.translator import AceaTranslator

    rule = _rule(int(rule[0]), rule[1], rule[2]) if rule else None
    out = tmp_path / "acea"
    with pytest.raises(ValueError, match="spam_data_dir|include_spam_in_package|crop-presence rule"):
        AceaTranslator(_config(targets=("pythia",), acea=acea, rule=rule), output_dir=out)
    assert not out.exists()
    AceaTranslator(_config(targets=("pythia",), acea=SPAM), output_dir=out)
    assert out.is_dir()


@pytest.mark.parametrize("switch", ["assign", "append"])
def test_targets_switched_to_acea_after_the_pipeline_is_built_fail_before_any_acea_file(tmp_path, switch):
    from prismpy.models.region import BoundingBox, Region
    from prismpy.pipeline.executor import PipelineStage, StageResult, TranslationPipeline
    from prismpy.translators.base import UnifiedData

    out = tmp_path / "out"
    cfg = _config(targets=("pythia",), acea={}, base_dir=str(out))
    pipeline = TranslationPipeline(cfg)
    if switch == "assign":
        cfg.targets = [Platform.PYTHIA, Platform.ACEA]
    else:
        cfg.targets.append(Platform.ACEA)
    region = Region(name="Koutiala", country="Mali", country_iso3="MLI",
                    bounds=BoundingBox(minx=-6.0, miny=11.0, maxx=-5.0, maxy=12.0))
    pipeline._execute_harmonize = lambda *a, **k: StageResult(stage=PipelineStage.HARMONIZE, success=True,
                                                              data=UnifiedData(region=region))
    result = pipeline.execute(stages=[PipelineStage.HARMONIZE, PipelineStage.TRANSLATE])
    assert not result.success
    assert any("spam_data_dir is required" in e for e in result.stages["execute"].errors)
    assert not (out / "acea").exists()
