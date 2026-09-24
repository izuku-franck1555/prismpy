"""A PYTHIA package README reports the package's own run configuration.

The README summary block (cultivar, experiment template, weather-station prefix, planting window,
fertilizer, population, row spacing, water regime) is read from the package's
``config/pythia_config.json`` — the inputs the runner actually uses — for baseline packages and for
the climate-projection packages cloned from them. A value the package does not record reads
"not recorded"; the README never falls back to fabricated defaults.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from prismpy.config.schema import (
    BoundaryConfig,
    BoundarySource,
    CropCalendarConfig,
    CropConfig,
    ManagementConfig,
    ManualBoundsConfig,
    OutputConfig,
    Platform,
    ProjectConfig,
    ProjectInfo,
    RegionConfig,
    TemporalConfig,
)
from prismpy.models.region import BoundingBox, Region
from prismpy.packaging.scenario_set_generator import _rewrite_projection_readme
from prismpy.translators.base import UnifiedData
from prismpy.translators.pythia.translator import PythiaTranslator
from tests.unit.test_pythia_canonical_substrate_flag import _build_grid_2x3, _build_profiles

_COWPEA_PROJECTION = Path(__file__).resolve().parents[1] / "fixtures" / "uc2_projection_cowpea"

_SUMMARY_ROW = re.compile(
    r"^\| (?:\*\*)?(Weather Station Prefix|Cultivar Code|Cultivar Name|Planting Window|"
    r"Fertilizer N|Plant Population|Row Spacing|Irrigation)(?:\*\*)? \| (.*) \|$", re.M)
_TEMPLATE_ROW = re.compile(r"└── (.+?)\s+# DSSAT experiment template")

# The values the README used to invent when a caller passed none (a Koutiala maize package).
_FABRICATED = {
    "Weather Station Prefix": "MLKO", "Experiment template": "KOMZ8001.SNX",
    "Cultivar Code": "990002", "Cultivar Name": "MEDIUM_SEASON",
    "Fertilizer N": "60 kg/ha", "Plant Population": "5.0 plants/m²", "Row Spacing": "70 cm",
    "Irrigation": "Rainfed", "Planting Window": "N/A to N/A",
}


def _summary(readme: str) -> dict:
    fields = dict(_SUMMARY_ROW.findall(readme))
    template = _TEMPLATE_ROW.search(readme)
    fields["Experiment template"] = template.group(1) if template else None
    return fields


def _baseline(out: Path, crop: str, short: str) -> Path:
    """A baseline PYTHIA package emitted by the real translator, with run values that differ
    from every fabricated default (Oromia, Ethiopia · 50 cm rows · 30 kg N · irrigated)."""
    cfg = ProjectConfig(
        project=ProjectInfo(name=f"{crop}_oromia", description="projection README values"),
        region=RegionConfig(
            name="Oromia", country="Ethiopia", country_iso3="ETH",
            boundary=BoundaryConfig(source=BoundarySource.MANUAL, manual_bounds=(
                ManualBoundsConfig(minx=38.0, miny=7.0, maxx=39.0, maxy=8.0))),
        ),
        crop=CropConfig(name=crop, name_short=short,
                        calendar=CropCalendarConfig(planting_doy=74, maturity_doy=169)),
        temporal=TemporalConfig(start_year=2015, end_year=2015, spinup_years=0),
        management=ManagementConfig(planting_density=200000.0, row_spacing_cm=50.0,
                                    irrigation=True, fertilizer_n_total=30.0),
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir=str(out), structure="by_platform"),
    )
    translator = PythiaTranslator(config=cfg, output_dir=str(out))
    (out / "config").mkdir(parents=True, exist_ok=True)
    data = UnifiedData(region=Region(name="Oromia", country="Ethiopia", country_iso3="ETH",
                                     bounds=BoundingBox(minx=38.0, miny=7.0, maxx=39.0, maxy=8.0)),
                       grid=_build_grid_2x3(), soil=_build_profiles())
    translator._generate_pythia_json(data)
    translator._generate_snx_template(data)
    return out


def _rewrite(projection: Path, platform: str, region: dict, crop: dict) -> str:
    _rewrite_projection_readme(
        projection,
        baseline_manifest={"platform": platform, "project_name": "projection_readme",
                           "region": region, "crop": crop},
        start_year=2046, end_year=2065, gcm_source="gfdl-esm4", rcp_or_ssp="ssp245",
    )
    return (projection / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("crop,short", [("Beans", "bns"), ("Potato", "pot")])
def test_projection_readme_reports_the_packages_own_run_config(tmp_path, crop, short):
    baseline = _baseline(tmp_path / "baseline", crop, short)
    projection = tmp_path / "projection"
    shutil.copytree(baseline, projection)
    run_config = json.loads((projection / "config" / "pythia_config.json").read_text())
    setup = run_config["default_setup"]
    fertilized = next(r for r in run_config["runs"] if r["name"].endswith("_fertilized"))

    fields = _summary(_rewrite(projection, "pythia", {"name": "Oromia", "country": "Ethiopia"},
                               {"name": crop, "planting_doy": 74, "maturity_doy": 169}))

    assert fields == {
        "Weather Station Prefix": setup["wsta"].split("::")[1],
        "Experiment template": setup["template"],
        "Cultivar Code": setup["ingeno"],
        "Cultivar Name": setup["cname"],
        "Planting Window": f"{setup['pfrst']} to {setup['plast']}",
        "Fertilizer N": f"{fertilized['fen_tot']} kg/ha",
        "Plant Population": f"{setup['ppop']} plants/m²",
        "Row Spacing": f"{setup['plrs']} cm",
        "Irrigation": "Enabled",
    }
    assert not {k for k, v in fields.items() if _FABRICATED[k] == v}


def test_the_real_cowpea_projection_readme_invents_nothing(tmp_path):
    projection = tmp_path / "projection"
    shutil.copytree(_COWPEA_PROJECTION, projection)
    manifest = json.loads((projection / "manifest.json").read_text())

    fields = _summary(_rewrite(projection, manifest["platform"], manifest["region"],
                               manifest["crop"]))

    # this package's run config records none of the summary values
    assert fields == {name: "not recorded" for name in _FABRICATED}


def test_a_craft_baseline_projection_readme_invents_nothing(tmp_path):
    projection = tmp_path / "projection"
    (projection / "management").mkdir(parents=True)

    fields = _summary(_rewrite(projection, "craft", {"name": "Kano", "country": "Nigeria"},
                               {"name": "Maize", "planting_doy": 166, "maturity_doy": 285}))

    assert fields == {name: "not recorded" for name in _FABRICATED}
