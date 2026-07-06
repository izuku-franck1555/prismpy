"""CRAFT is fixed to the 5-arcmin global grid (cell-id row*4320+col), so a
CRAFT + non-5arcmin config must fail loud at config validation.

Guard: ProjectConfig.validate_craft_requires_5arcmin. CRAFT-SPECIFIC — ACEA
and PYTHIA (which support 30-arcmin) must NOT be rejected.
"""
from __future__ import annotations

from unittest import TestCase

import pytest
from pydantic import ValidationError

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


def _cfg(*, targets, resolution: str, base_dir: str = 'outputs') -> ProjectConfig:
    return ProjectConfig(
        project=ProjectInfo(name='craft_guard', description='guard test'),
        region=RegionConfig(
            name='Koutiala', country='Mali', country_iso3='MLI',
            grid_resolution=resolution,
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=-6.0, miny=11.0, maxx=-4.0, maxy=13.0,
                ),
                inclusion_rule='bbox_intersects',
            ),
        ),
        crop=CropConfig(
            name='Maize', name_short='mai', variety='M',
            calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285),
        ),
        temporal=TemporalConfig(start_year=2015, end_year=2020, spinup_years=2),
        targets=targets,
        output=OutputConfig(base_dir=base_dir, structure="by_platform"),
    )


class TestCraftRequires5Arcmin(TestCase):
    def test_craft_30arcmin_rejected_with_actionable_error(self):
        with pytest.raises(ValidationError) as excinfo:
            _cfg(targets=[Platform.CRAFT], resolution='30arcmin')
        msg = str(excinfo.value)
        self.assertIn('CRAFT does not support', msg)   # WHAT went wrong
        self.assertIn('row*4320+col', msg)             # WHY (5-arcmin cell-id)
        self.assertIn("'5arcmin'", msg)                # WHAT TO DO (use 5arcmin)
        self.assertIn('PYTHIA', msg)                   # same-physics 30-arcmin pointer (DSSAT)
        self.assertIn('AquaCrop', msg)                 # ACEA mentioned only WITH the different-model caveat

    def test_craft_alongside_acea_30arcmin_still_rejected(self):
        # CRAFT anywhere in targets triggers the guard, even with a 30arcmin-valid platform.
        with pytest.raises(ValidationError):
            _cfg(targets=[Platform.CRAFT, Platform.ACEA], resolution='30arcmin')

    def test_craft_5arcmin_accepted(self):
        cfg = _cfg(targets=[Platform.CRAFT], resolution='5arcmin')
        self.assertIn(Platform.CRAFT, cfg.targets)

    def test_acea_30arcmin_not_rejected(self):
        cfg = _cfg(targets=[Platform.ACEA], resolution='30arcmin')
        self.assertEqual(cfg.region.grid_resolution, '30arcmin')

    def test_pythia_30arcmin_not_rejected(self):
        cfg = _cfg(targets=[Platform.PYTHIA], resolution='30arcmin')
        self.assertEqual(cfg.region.grid_resolution, '30arcmin')

    def test_targets_mutation_bypass_caught_at_pipeline_boundary(self):
        # A post-construction targets mutation (the CLI --targets path) skips the
        # model_validator; the pipeline boundary must re-check and fail loud.
        from prismpy.pipeline.executor import TranslationPipeline
        cfg = _cfg(targets=[Platform.ACEA], resolution='30arcmin')
        cfg.targets = [Platform.CRAFT]
        with pytest.raises(ValueError) as excinfo:
            TranslationPipeline(cfg)
        msg = str(excinfo.value)
        self.assertIn('CRAFT does not support', msg)
        self.assertIn('row*4320+col', msg)
        # Also the in-place append path (programmatic mutation).
        cfg2 = _cfg(targets=[Platform.ACEA], resolution='30arcmin')
        cfg2.targets.append(Platform.CRAFT)
        with pytest.raises(ValueError):
            TranslationPipeline(cfg2)

    def test_craft_translator_direct_instantiation_rejected(self):
        # Root choke point: instantiating CraftTranslator directly with a valid
        # non-CRAFT config still produces CRAFT output -> must fail loud.
        from prismpy.translators.craft.translator import CraftTranslator
        cfg = _cfg(targets=[Platform.ACEA], resolution='30arcmin')
        with pytest.raises(ValueError) as excinfo:
            CraftTranslator(cfg)
        msg = str(excinfo.value)
        self.assertIn('CRAFT does not support', msg)
        self.assertIn('PYTHIA', msg)

    def test_translate_time_data_grid_bypass_rejected(self):
        # The consumption point: a VALID 5arcmin CRAFT config + translator, then
        # translate() with a 30arcmin RUNTIME grid -> reject before any output.
        from prismpy.translators.craft.translator import CraftTranslator
        from prismpy.translators.base import UnifiedData
        from prismpy.models.spatial import SpatialGrid
        from prismpy.models.region import BoundingBox, Region
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(targets=[Platform.CRAFT], resolution='5arcmin', base_dir=tmp)
            tr = CraftTranslator(cfg)
            data = UnifiedData(
                region=Region(name='Koutiala', country='Mali', country_iso3='MLI',
                              bounds=BoundingBox(minx=-6.0, miny=11.0, maxx=-4.0, maxy=13.0)),
                grid=SpatialGrid(resolution='30arcmin'),
            )
            with pytest.raises(ValueError) as excinfo:
                tr.translate(data)
            msg = str(excinfo.value)
            self.assertIn('CRAFT does not support', msg)
            self.assertIn('PYTHIA', msg)
            self.assertEqual(list(tr.output_dir.glob("*")), [])  # isolated temp -> no flake
