"""Config-drift canonical-soil pins (Slice A) — genuine RED-on-revert.

Guards for the Option-C fix: soil HWSD paths resolve through the SOLE canonical
``data_sources.soil`` location, never a per-platform copy.

- **P-CONFIG-DRIFT-RESOLVES (launchability):** a PYTHIA / SarraPy config with HWSD
  paths ONLY in ``data_sources.soil`` must NOT trip the ``_execute_retrieve``
  ``has_hwsd`` launchability probe into warning "Soil data not available - using
  placeholder" — the drift that failed every non-Africa PYTHIA/SarraPy run.
  Pre-fix (``ba0dd70``) that probe scanned ``platform_config.<engine>.hwsd_bil_path``
  (absent for PYTHIA/SarraPy → Pydantic dropped it) so it warned; post-fix it reads
  the canonical ``data_sources.soil``. This drives the REAL ``_execute_retrieve``
  (never a re-simulation) with the iSDA-local check forced absent so ``has_hwsd`` is
  the sole decider — so it is RED on the pre-fix SHA.
- **P-CANONICAL-FORWARD-PROOF (structural):** no platform config class may declare
  HWSD fields; ``SoilSourceConfig`` is the one declarer. Re-adding a per-platform
  copy → RED.
"""
from __future__ import annotations

import pathlib
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from prismpy.config.schema import (
    AceaConfig,
    BoundaryConfig,
    BoundarySource,
    CraftConfig,
    CropCalendarConfig,
    CropConfig,
    DataSourcesConfig,
    ManualBoundsConfig,
    OutputConfig,
    Platform,
    ProjectConfig,
    ProjectInfo,
    PythiaConfig,
    RegionConfig,
    SarraPyConfig,
    SoilSourceConfig,
    TemporalConfig,
)
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.provenance.tracker import ProvenanceTracker


def _pipeline(platform, bil, mdb, out_dir) -> TranslationPipeline:
    cfg = ProjectConfig(
        project=ProjectInfo(name='cfg_drift', description='canonical soil pin'),
        region=RegionConfig(
            name='Calvados', country='France', country_iso3='FRA',
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(minx=-0.6, miny=48.9, maxx=-0.5, maxy=49.0),
                inclusion_rule='bbox_intersects', min_share_percent=0.0,
            ),
        ),
        crop=CropConfig(
            name='Maize', name_short='mai', variety='Medium',
            calendar=CropCalendarConfig(planting_doy=120, maturity_doy=270),
        ),
        temporal=TemporalConfig(start_year=2015, end_year=2018, spinup_years=2),
        targets=[platform],
        data_sources=DataSourcesConfig(
            soil=SoilSourceConfig(hwsd_bil_path=bil, hwsd_mdb_path=mdb),
        ),
        output=OutputConfig(base_dir=out_dir, structure='by_platform'),
    )
    return TranslationPipeline(
        cfg, provenance=ProvenanceTracker(enabled=False, project_name='cfg_drift'),
    )


_real_path_exists = Path.exists


def _isda_absent(self):
    """Force the iSDA-local existence check False so the has_hwsd probe is the sole
    decider of the placeholder-soil warning (isolates the drift surface)."""
    s = str(self)
    if 'sand_content_1km.tif' in s or '/isda' in s:
        return False
    return _real_path_exists(self)


class TestConfigDriftLaunchabilityViaCanonicalSoil(TestCase):
    def _assert_launchable(self, platform):
        with tempfile.TemporaryDirectory() as td:
            tp = pathlib.Path(td)
            bil = tp / 'HWSD2.bil'
            mdb = tp / 'HWSD2.mdb'
            bil.touch()
            mdb.touch()
            pipe = _pipeline(platform, bil, mdb, str(tp / 'out'))
            with patch.object(Path, 'exists', _isda_absent):
                result = pipe._execute_retrieve()
            self.assertNotIn(
                'Soil data not available - using placeholder', result.warnings or [],
                f"{platform.value}: with HWSD paths in the canonical data_sources.soil, the "
                f"_execute_retrieve has_hwsd probe must NOT warn placeholder-soil. Pre-fix the "
                f"probe scanned platform_config (no HWSD field for {platform.value}) → warned; "
                f"reverting the routing fix makes this RED.",
            )

    def test_pythia_launchable_via_data_sources_soil(self):
        self._assert_launchable(Platform.PYTHIA)

    def test_sarra_py_launchable_via_data_sources_soil(self):
        self._assert_launchable(Platform.SARRA_PY)


class TestNoPlatformConfigDeclaresHWSD(TestCase):
    def test_platform_configs_do_not_declare_hwsd(self):
        for cls in (CraftConfig, AceaConfig, PythiaConfig, SarraPyConfig):
            for field in ('hwsd_bil_path', 'hwsd_mdb_path'):
                self.assertNotIn(
                    field, cls.model_fields,
                    f"{cls.__name__} declares {field}: HWSD soil paths must live ONLY "
                    f"in SoilSourceConfig (data_sources.soil). A per-platform copy "
                    f"re-opens the config-drift the fix closed.",
                )

    def test_soil_source_config_is_the_canonical_declarer(self):
        for field in ('hwsd_bil_path', 'hwsd_mdb_path'):
            self.assertIn(
                field, SoilSourceConfig.model_fields,
                f"SoilSourceConfig must declare {field} — the sole canonical location.",
            )


if __name__ == '__main__':
    import unittest
    unittest.main()
