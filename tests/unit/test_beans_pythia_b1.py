"""Common-bean PYTHIA-only B1 pins.

Two honesty invariants the runner makes load-bearing:
1. **No-mask harvest-area emit.** The prism-runner drops every site whose ``harvestArea`` raster
   value is missing, so a crop with no SPAM mask (beans) MUST OMIT the ``harvestArea`` directive
   entirely — not emit an unresolved ``raster::`` path — so every otherwise-eligible region site is
   attempted instead of all sites being dropped. A masked crop still emits ``raster::`` on both runs.
2. **Honest package provenance.** When no crop mask was produced the manifest + README must NOT claim
   a "SPAM 2020" crop mask / ``harvest_area.tif``; they must record that none was applied.
Plus the MANDATORY ``PLANTING_DEFAULTS['beans']`` density on BOTH management paths (explicit density
and absent-management), so an absent-management beans run renders bean density (~20 plants/m²), never
the generic 6.25 maize fallback.
"""
from __future__ import annotations

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
from prismpy.packaging.readme_generator import generate_readme
from prismpy.translators.base import UnifiedData
from prismpy.translators.pythia.translator import PythiaTranslator
from tests.unit.test_pythia_canonical_substrate_flag import _build_grid_2x3, _build_profiles


def _beans_cfg(output_dir: Path, management, crop_name: str = "Beans",
               crop_short: str = "bns") -> ProjectConfig:
    return ProjectConfig(
        project=ProjectInfo(name="beans_b1", description="beans pythia no-mask + density"),
        region=RegionConfig(
            name="Kacheliba",
            country="Kenya",
            country_iso3="KEN",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(minx=34.0, miny=1.0, maxx=35.0, maxy=2.0),
            ),
        ),
        crop=CropConfig(
            name=crop_name,
            name_short=crop_short,
            variety="medium",
            calendar=CropCalendarConfig(planting_doy=74, maturity_doy=169),
        ),
        temporal=TemporalConfig(start_year=2015, end_year=2015, spinup_years=0),
        management=management,
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir=str(output_dir), structure="by_platform"),
    )


def _translator_and_data(tmp_path: Path, management, *, mask_present: bool,
                         crop_name: str = "Beans", crop_short: str = "bns"):
    """A PYTHIA translator + minimal UnifiedData, with ``_mask_present`` injected to the state
    that ``translate()`` sets at its crop-mask step (the helper renders the artifacts directly)."""
    translator = PythiaTranslator(config=_beans_cfg(tmp_path, management, crop_name, crop_short),
                                  output_dir=str(tmp_path))
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    translator._mask_present = mask_present
    data = UnifiedData(
        region=Region(
            name="Kacheliba", country="Kenya", country_iso3="KEN",
            bounds=BoundingBox(minx=34.0, miny=1.0, maxx=35.0, maxy=2.0),
        ),
        grid=_build_grid_2x3(),
        soil=_build_profiles(),
    )
    return translator, data


# ── mandatory PLANTING_DEFAULTS density — both management paths ──────────────

def test_beans_planting_defaults_entry_is_bean_density():
    assert PythiaTranslator.PLANTING_DEFAULTS["beans"] == {"ppop": 20.0, "plrs": 50.0, "pldp": 5.0}


def test_generic_fallback_stays_generic_and_beans_is_mapped():
    # The catch-all fallback stays the generic 6.25 maize density; beans is a REAL entry, so it
    # never resolves to the fallback (the silent-wrong-density the mandatory entry closes).
    assert PythiaTranslator.PLANTING_DEFAULT_FALLBACK["ppop"] == 6.25
    assert "beans" in PythiaTranslator.PLANTING_DEFAULTS


def test_beans_absent_management_renders_bean_density_not_fallback(tmp_path):
    # management=None -> the per-crop default is used; for beans it MUST be 20 plants/m², not 6.25.
    translator, _ = _translator_and_data(tmp_path, None, mask_present=False)
    assert translator._resolve_planting_params()["ppop"] == 20.0


def test_beans_explicit_density_converts_to_bean_ppop(tmp_path):
    # 200,000 plants/ha -> 20 plants/m² (a single /10000); the explicit path also lands on 20.
    translator, _ = _translator_and_data(
        tmp_path, ManagementConfig(planting_density=200000.0), mask_present=False)
    assert translator._resolve_planting_params()["ppop"] == 20.0


# ── no-mask harvest-area emit (both sides; masked-crop regression) ───────────

def test_beans_no_mask_omits_harvest_area_on_every_run(tmp_path):
    # Beans (no SPAM mask) must OMIT harvestArea entirely on every run — not emit an
    # unresolved raster:: path the runner would drop every site on.
    translator, data = _translator_and_data(
        tmp_path, ManagementConfig(planting_density=200000.0), mask_present=False)
    runs = json.loads(Path(translator._generate_pythia_json(data)).read_text())["runs"]
    assert len(runs) >= 2
    for run in runs:
        assert "harvestArea" not in run


@pytest.mark.parametrize("crop_name", ["Maize", "Sorghum", "Millet", "Cowpea", "Rice", "Groundnut"])
def test_masked_crops_still_emit_raster_on_every_run(tmp_path, crop_name):
    # Masked-crop regression across the REAL masked crops: with a mask present the raster::
    # harvest-area filter stays on EVERY run (the no-mask change must not un-mask them).
    translator, data = _translator_and_data(
        tmp_path, ManagementConfig(planting_density=55000.0), mask_present=True,
        crop_name=crop_name, crop_short=crop_name[:3].lower())
    runs = json.loads(Path(translator._generate_pythia_json(data)).read_text())["runs"]
    assert len(runs) >= 2
    for run in runs:
        assert str(run.get("harvestArea", "")).startswith("raster::")


# ── honest package provenance — manifest + README (both sides) ───────────────

def test_crop_mask_provenance_label_is_honest(tmp_path):
    translator, _ = _translator_and_data(tmp_path, None, mask_present=True)
    assert translator._crop_mask_provenance_label() == "SPAM 2020"
    translator._mask_present = False
    assert "no crop mask applied" in translator._crop_mask_provenance_label().lower()


@pytest.mark.parametrize("mask_present", [True, False])
def test_manifest_and_readme_are_honest_about_the_mask(tmp_path, mask_present):
    translator, data = _translator_and_data(
        tmp_path, ManagementConfig(planting_density=200000.0), mask_present=mask_present)
    manifest_text = Path(translator._generate_manifest(data)).read_text()
    readme_text = Path(translator._generate_readme(data)).read_text()
    if mask_present:
        assert '"crop_mask": "SPAM 2020"' in manifest_text
        assert "| Crop Mask | SPAM | 2020 v2.0 |" in readme_text
    else:
        # No false SPAM claim; the honest "none" label instead, in BOTH surfaces.
        assert '"crop_mask": "SPAM 2020"' not in manifest_text
        assert '"crop_mask": "none' in manifest_text
        assert "No crop mask applied" in readme_text
        assert "| Crop Mask | SPAM |" not in readme_text


# ── README template unit (the centralized generator, direct) ─────────────────

@pytest.mark.parametrize("mask_present", [True, False])
def test_pythia_readme_template_crop_mask_row(tmp_path, mask_present):
    out = tmp_path / "README.md"
    generate_readme(out, {}, platform="pythia", mask_present=mask_present)
    text = out.read_text()
    if mask_present:
        assert "| Crop Mask | SPAM | 2020 v2.0 |" in text
        assert "harvest_area.tif         # SPAM crop harvest area" in text
    else:
        assert "No crop mask applied" in text
        assert "| Crop Mask | SPAM |" not in text


# ── translate-time assignment + scenario-set projection README honesty ───────

def test_mask_present_defaults_false_until_translate_sets_it(tmp_path):
    # The class default is the conservative no-mask state; translate() assigns it at the
    # crop-mask step (crop_mask is not None). An un-run translator never claims a mask.
    translator, _ = _translator_and_data(tmp_path, None, mask_present=False)
    assert PythiaTranslator._mask_present is False
    translator._mask_present = True  # what translate() sets when a mask IS produced
    assert translator._crop_mask_provenance_label() == "SPAM 2020"


def _run_projection_readme(tmp_path: Path, crop_mask_value: str) -> str:
    from prismpy.packaging.scenario_set_generator import _rewrite_projection_readme
    proj = tmp_path / "proj_ssp585"
    (proj / "weather").mkdir(parents=True, exist_ok=True)
    baseline_manifest = {
        "platform": "pythia",
        "project_name": "beans_projection",
        "region": {"name": "Kacheliba", "country": "Kenya"},
        "crop": {"name": "Beans", "planting_doy": 74, "maturity_doy": 169},
        "data_sources": {"crop_mask": crop_mask_value},
    }
    _rewrite_projection_readme(
        proj, baseline_manifest=baseline_manifest,
        start_year=2040, end_year=2049, gcm_source="GFDL-ESM4", rcp_or_ssp="ssp585",
    )
    return (proj / "README.md").read_text()


def test_projection_readme_inherits_no_mask_from_baseline(tmp_path):
    # Regression: a projection cloned from a NO-MASK beans baseline must NOT re-claim a SPAM
    # mask (the projection README previously defaulted the mask state to present). RED pre-fix.
    text = _run_projection_readme(
        tmp_path, "none (no crop mask applied; run not restricted to harvested crop area)")
    assert "No crop mask applied" in text
    assert "| Crop Mask | SPAM |" not in text
    assert "harvest_area.tif         # SPAM crop harvest area" not in text


def test_projection_readme_keeps_mask_from_masked_baseline(tmp_path):
    # A projection from a MASKED baseline (real SPAM) correctly keeps the SPAM claim.
    text = _run_projection_readme(tmp_path, "SPAM 2020")
    assert "| Crop Mask | SPAM | 2020 v2.0 |" in text
    assert "harvest_area.tif         # SPAM crop harvest area" in text
