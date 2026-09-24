"""Every crop but potato prepares, byte for byte, the PYTHIA package it did before potato support.

For the seven other crops, in each of the three emit shapes, the RAW SNX template and the RAW
``config/pythia_config.json`` are hashed and compared with the digests prismpy ``714abec`` (the
release before potato support) emitted from this module's own ``_emit``. The one allowed change is
the cultivar a minimal-config legume now resolves (its own default, not the CERES-maize
placeholder); the fixture lists those tokens, and the emitted file must hash to the old digest once
they are put back.

Regenerate the digests against a ``714abec`` source tree with
``PYTHONPATH=<714abec tree>/src:. python -m tests.unit.test_pythia_crop_scope_goldens``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

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
from tests.unit.test_pythia_canonical_substrate_flag import _build_grid_2x3, _build_profiles

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pythia_crop_scope_714abec.json"
_CROPS = {"Beans": "bns", "Cowpea": "cpe", "Groundnut": "gnt", "Maize": "mze", "Millet": "mil",
          "Rice": "ric", "Sorghum": "sor"}
_SOWING = {"minimal-config": None, "opportunistic": "opportunistic", "fixed_date": "fixed_date"}


def _emit(out: Path, crop: str, shape: str) -> Dict[str, bytes]:
    sowing = _SOWING[shape]
    bounds = dict(minx=36.2, miny=-0.6, maxx=36.8, maxy=0.0)
    cfg = ProjectConfig(
        project=ProjectInfo(name="crop_scope", description="crop scope goldens"),
        region=RegionConfig(name="Nyandarua", country="Kenya", country_iso3="KEN",
                            boundary=BoundaryConfig(source=BoundarySource.MANUAL,
                                                    manual_bounds=ManualBoundsConfig(**bounds))),
        crop=CropConfig(name=crop, name_short=_CROPS[crop],
                        calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285)),
        temporal=TemporalConfig(start_year=2015, end_year=2016, spinup_years=0),
        management=(None if sowing is None
                    else ManagementConfig(planting_density=50000.0, sowing_mode=sowing)),
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir=str(out), structure="by_platform"),
    )
    translator = PythiaTranslator(config=cfg, output_dir=str(out))
    (out / "config").mkdir(parents=True, exist_ok=True)
    data = UnifiedData(region=Region(name="Nyandarua", country="Kenya", country_iso3="KEN",
                                     bounds=BoundingBox(**bounds)),
                       grid=_build_grid_2x3(), soil=_build_profiles())
    return {"template": Path(translator._generate_snx_template(data)).read_bytes(),
            "config": Path(translator._generate_pythia_json(data)).read_bytes()}


@pytest.mark.parametrize("shape", sorted(_SOWING))
@pytest.mark.parametrize("crop", sorted(_CROPS))
def test_other_crops_emit_the_template_and_run_config_they_did_before_potato(
        tmp_path, crop, shape):
    golden = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    case = f"{crop}/{shape}"
    emitted = _emit(tmp_path, crop, shape)
    config = emitted["config"].decode("utf-8")
    for field, (before, now) in golden["cultivar_resolved_since"].get(case, {}).items():
        config = config.replace(f'"{field}": "{now}', f'"{field}": "{before}')
    assert hashlib.sha256(emitted["template"]).hexdigest() == golden["sha256"][case]["template"]
    assert hashlib.sha256(config.encode("utf-8")).hexdigest() == golden["sha256"][case]["config"]


if __name__ == "__main__":
    import tempfile

    digests = {}
    with tempfile.TemporaryDirectory() as tmp:
        for crop in sorted(_CROPS):
            for shape in sorted(_SOWING):
                emitted = _emit(Path(tmp) / crop / shape, crop, shape)
                digests[f"{crop}/{shape}"] = {
                    kind: hashlib.sha256(blob).hexdigest() for kind, blob in emitted.items()}
    print(json.dumps(digests, indent=2, sort_keys=True))
