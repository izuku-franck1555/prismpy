"""Case C — sowing_mode -> DSSAT PLANT method wired into pythia_config default_setup.

The SNX ``@N MANAGEMENT PLANT`` field is a Jinja placeholder PYTHIA fills at runtime
from ``pythia_config.json``'s ``default_setup`` (and per-run dicts). This asserts at
that WIRE (the generated JSON), not the intermediate ``_map_generic_to_pythia_config``
dict: opportunistic -> "A" (automatic; DSSAT uses the PFRST/PLAST window), fixed_date
-> "R" (on PDATE). "F" is non-standard and never emitted; unknown modes raise. Covers
the generic (management-driven) AND else/legacy branches, which both feed default_setup.
"""
from __future__ import annotations

import inspect
import json
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
from prismpy.translators.base import UnifiedData
from prismpy.translators.pythia.translator import PythiaTranslator
from tests.unit.test_pythia_canonical_substrate_flag import (
    _build_grid_2x3,
    _build_profiles,
)


def _cfg(output_dir: Path, management) -> ProjectConfig:
    return ProjectConfig(
        project=ProjectInfo(name="caseC_plantmode", description="plant_mode wire test"),
        region=RegionConfig(
            name="Wami",
            country="Tanzania",
            country_iso3="TZA",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(minx=37.0, miny=-7.0, maxx=38.0, maxy=-6.0),
            ),
        ),
        crop=CropConfig(
            name="Maize",
            name_short="mze",
            variety="medium",
            calendar=CropCalendarConfig(planting_doy=330, maturity_doy=120),
        ),
        temporal=TemporalConfig(start_year=2015, end_year=2015, spinup_years=0),
        management=management,
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir=str(output_dir), structure="by_platform"),
    )


def _generated_pythia_json(tmp_path: Path, management) -> dict:
    """Render pythia_config.json via the translator and return it parsed (the WIRE)."""
    translator = PythiaTranslator(config=_cfg(tmp_path, management), output_dir=str(tmp_path))
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    data = UnifiedData(
        region=Region(
            name="Wami", country="Tanzania", country_iso3="TZA",
            bounds=BoundingBox(minx=37.0, miny=-7.0, maxx=38.0, maxy=-6.0),
        ),
        grid=_build_grid_2x3(),
        soil=_build_profiles(),
    )
    return json.loads(Path(translator._generate_pythia_json(data)).read_text())


# ── generic (management-driven) path — asserts at the wire ──────────────────

def test_opportunistic_wires_default_setup_plant_A(tmp_path):
    j = _generated_pythia_json(tmp_path, ManagementConfig(planting_density=62500.0, sowing_mode="opportunistic"))
    assert j["default_setup"]["plant_mode"] == "A"
    assert j["default_setup"]["pdate"] == j["default_setup"]["pfrst"]  # PDATE=window start
    assert all(r["plant_mode"] == "A" for r in j["runs"])  # mirrored on every run dict


def test_fixed_date_wires_default_setup_plant_R(tmp_path):
    j = _generated_pythia_json(tmp_path, ManagementConfig(planting_density=62500.0, sowing_mode="fixed_date"))
    assert j["default_setup"]["plant_mode"] == "R"
    assert all(r["plant_mode"] == "R" for r in j["runs"])


def test_fixed_alias_normalizes_to_R(tmp_path):
    # schema normalizes "fixed" -> "fixed_date" -> "R"
    j = _generated_pythia_json(tmp_path, ManagementConfig(planting_density=62500.0, sowing_mode="fixed"))
    assert j["default_setup"]["plant_mode"] == "R"


# ── else/legacy path (no management/phenology/physiology) feeds default_setup too ──

def test_else_path_no_management_defaults_plant_A(tmp_path):
    # use_generic_mapping is False -> else branch -> mgmt None -> opportunistic default -> "A".
    # Proves the else branch also sets default_setup["plant_mode"] (not a NameError / missing key).
    j = _generated_pythia_json(tmp_path, None)
    assert j["default_setup"]["plant_mode"] == "A"
    assert all(r["plant_mode"] == "A" for r in j["runs"])


# ── helper unit + SNX placeholder presence ─────────────────────────────────

def test_helper_maps_and_raises_on_unknown():
    assert PythiaTranslator._plant_mode_from_sowing("opportunistic") == "A"
    assert PythiaTranslator._plant_mode_from_sowing("fixed_date") == "R"
    with pytest.raises(ValueError, match="Unknown sowing_mode"):
        PythiaTranslator._plant_mode_from_sowing("planting_window")


def test_snx_template_has_plant_mode_placeholder():
    # the @N MANAGEMENT PLANT field must be a {{ plant_mode }} placeholder PYTHIA fills.
    src = inspect.getsource(PythiaTranslator._build_snx_content)
    assert "plant_mode" in src
