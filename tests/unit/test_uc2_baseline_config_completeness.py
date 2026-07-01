"""UC2 baseline package-completeness guards (Bar-1 raster fix).

The Bar-1 smoke found the UC2 baseline configs omitted
``platform_config.pythia.spam_raster_dir`` → the pythia translator's crop-mask
step skipped generation → the package carried no ``raster/harvest_area.tif`` →
the pythia adapter skipped ALL sites ("Prepared 0 sites") and the run silently
produced no yield. The structural spot-check (cells / weather / manifest /
provenance) did not cover raster-completeness, so the gap only surfaced at the
empirical DSSAT run.

These tests assert the REAL behaviour, not inert YAML:
  1. ``load_config`` (the real loader) parses ``spam_raster_dir`` into the
     runtime ``PythiaConfig`` — the field the crop-mask step actually reads.
  2. with ``spam_raster_dir`` set (+ a SPAM raster present) the crop-mask step
     actually WRITES ``raster/harvest_area.tif``; without it, it is skipped.

Note ``include_spam_in_package`` is an ``AceaConfig`` field — under
``platform_config.pythia`` it is silently dropped, so a config/test that
asserts it there is a masquerade (asserts an inert key that never reaches
``PythiaConfig``). The UC2 configs therefore set only the real PythiaConfig
knobs ``spam_raster_dir`` + ``spam_version``.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from prismpy.config.loader import load_config
from prismpy.translators.pythia.translator import PythiaTranslator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UC2_BASELINE_CONFIGS = (
    "uc2_kano_cowpea_baseline.yaml",
    "uc2_kano_sorghum_baseline.yaml",
)


@pytest.mark.parametrize("cfg_name", _UC2_BASELINE_CONFIGS)
def test_uc2_baseline_pythia_config_carries_effective_spam_raster_dir(
    cfg_name: str,
) -> None:
    """REAL parse (not raw YAML): ``spam_raster_dir`` must survive ``load_config``
    into the runtime ``PythiaConfig`` — the field the crop-mask step reads. If it
    is dropped/misnested/unset, harvest_area.tif is never produced."""
    cfg = load_config(_REPO_ROOT / cfg_name)
    pythia = cfg.platform_config.pythia
    assert pythia is not None, f"{cfg_name}: no platform_config.pythia"
    assert pythia.spam_raster_dir is not None, (
        f"{cfg_name}: PythiaConfig.spam_raster_dir is unset after parse → the "
        "crop-mask step skips harvest_area.tif → pythia 'Prepared 0 sites'."
    )
    # Guard against re-introducing the inert masquerade key on pythia.
    assert not hasattr(pythia, "include_spam_in_package"), (
        "include_spam_in_package is an AceaConfig field; it must not be relied "
        "on under platform_config.pythia (it is silently dropped)."
    )


def _mock_data() -> types.SimpleNamespace:
    """grid=None + region.bounds without to_gis_format → the crop-mask method
    falls back to config.region.boundary.manual_bounds."""
    return types.SimpleNamespace(
        grid=None, region=types.SimpleNamespace(bounds=None)
    )


def _write_synthetic_spam(spam_dir: Path, crop_lower: str) -> Path:
    """Minimal SPAM raster covering the Kano belt (simplified naming the
    translator accepts: ``spam2020_<crop>.tif``)."""
    path = spam_dir / f"spam2020_{crop_lower}.tif"
    width, height = 60, 40
    transform = from_bounds(5.0, 11.0, 11.0, 13.0, width, height)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(np.full((height, width), 100.0, dtype="float32"), 1)
    return path


def test_spam_raster_dir_produces_harvest_area_tif(tmp_path: Path) -> None:
    """REAL behaviour: with spam_raster_dir set (+ a SPAM raster present), the
    crop-mask step writes ``raster/harvest_area.tif`` — what actually unblocks
    the pythia run. Uses a SYNTHETIC raster (no external data dependency)."""
    spam_dir = tmp_path / "spam"
    spam_dir.mkdir()
    _write_synthetic_spam(spam_dir, "cowpea")

    cfg = load_config(_REPO_ROOT / "uc2_kano_cowpea_baseline.yaml")
    cfg.platform_config.pythia.spam_raster_dir = spam_dir  # override to synthetic
    t = PythiaTranslator(config=cfg, output_dir=tmp_path / "out")
    (t.output_dir / "raster").mkdir(parents=True, exist_ok=True)

    result = t._generate_crop_mask_raster(_mock_data())
    assert result is not None, "crop-mask returned None despite spam_raster_dir set"
    assert (t.output_dir / "raster" / "harvest_area.tif").exists(), (
        "harvest_area.tif not written — the pythia run would 'Prepared 0 sites'"
    )


def test_crop_mask_skipped_without_spam_raster_dir(tmp_path: Path) -> None:
    """The exact gap: no spam_raster_dir → crop-mask skipped → no
    harvest_area.tif (the failure the smoke caught)."""
    cfg = load_config(_REPO_ROOT / "uc2_kano_cowpea_baseline.yaml")
    cfg.platform_config.pythia.spam_raster_dir = None
    t = PythiaTranslator(config=cfg, output_dir=tmp_path / "out")
    (t.output_dir / "raster").mkdir(parents=True, exist_ok=True)

    assert t._generate_crop_mask_raster(_mock_data()) is None
    assert not (t.output_dir / "raster" / "harvest_area.tif").exists()
