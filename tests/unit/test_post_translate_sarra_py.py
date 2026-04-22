"""Production tests for SARRA-Py GeoTIFF post-translate validation.

Covers three acceptance criteria:
  AC 1.4.6 — variable coverage: all 4 mapped subdirs produce checks
  AC 1.4.7 — silent-failure regression: empty dir → warning + searched_paths
  AC 1.4.8 — unit mapping matches actual SARRA-Py .tif contents. Issue 5
             (warning-auditor HIGH): .tifs are already converted by the
             SARRA_data_download library, so tmax/tmin are °C (noop) and
             srad is kJ/m²/d (mul 1e-3 → MJ/m²/d). The prior K→°C and
             J→MJ mapping applied a redundant second conversion and
             reported tmax around -251 °C.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from prismpy.validators.post_translate import (
    SARRA_PY_VAR_MAPPING,
    _validate_sarra_py_geotiffs,
)


def _write_fixture_tiff(path: Path, value: float):
    """Write a minimal 2x2 single-band GeoTIFF with all pixels set to `value`."""
    transform = from_bounds(9.5, 5.0, 10.0, 5.5, 2, 2)
    with rasterio.open(
        path, 'w', driver='GTiff', height=2, width=2,
        count=1, dtype='float32', crs='EPSG:4326', transform=transform,
    ) as dst:
        dst.write(np.full((1, 2, 2), value, dtype=np.float32))


class TestSarraPyVariableCoverage(unittest.TestCase):
    """AC 1.4.6 — all 4 mapped variable subdirs produce range checks."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix='sarra-py-test-'))
        climate = self.tmpdir / 'data' / 'climate'
        for subdir_name in SARRA_PY_VAR_MAPPING:
            var_dir = climate / subdir_name
            var_dir.mkdir(parents=True)
            for i in range(3):
                _write_fixture_tiff(var_dir / f'day_{i:03d}.tif', 300.0)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_all_four_variables_produce_range_checks(self):
        checks = _validate_sarra_py_geotiffs(self.tmpdir)
        range_names = {c['check'] for c in checks if c['check'].startswith('post_translate_range_')}
        expected = {
            'post_translate_range_sarra_py_rain',
            'post_translate_range_sarra_py_tmin',
            'post_translate_range_sarra_py_tmax',
            'post_translate_range_sarra_py_srad',
        }
        self.assertEqual(range_names, expected)


class TestSarraPySilentFailureRegression(unittest.TestCase):
    """AC 1.4.7 — empty climate dir → warning (not info) + searched_paths."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix='sarra-py-test-'))
        (self.tmpdir / 'data' / 'climate').mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_dir_returns_warning_with_searched_paths(self):
        checks = _validate_sarra_py_geotiffs(self.tmpdir)
        self.assertEqual(len(checks), 1)
        check = checks[0]
        self.assertEqual(check['result'], 'warning')
        self.assertIn('searched_paths', check['details'])
        self.assertTrue(len(check['details']['searched_paths']) > 0)
        self.assertNotEqual(check['check'], 'post_translate_range_sarra_py_rain')


class TestSarraPyUnitConversion(unittest.TestCase):
    """AC 1.4.8 — unit mapping reflects the actual SARRA_data_download
    .tif contents (already-converted units), not raw AgERA5. The prior
    K→°C + J→MJ mapping double-converted and reported impossible
    ranges on screen. Real .tif inspection on a Maradi-2020 run:
    tmax / tmin already in °C, srad in kJ/m²/d."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix='sarra-py-test-'))
        climate = self.tmpdir / 'data' / 'climate'
        # tmax fixture at 22.0 (already °C) — noop passes through.
        tmax_dir = climate / '2m_temperature_24_hour_maximum'
        tmax_dir.mkdir(parents=True)
        _write_fixture_tiff(tmax_dir / 'day_001.tif', 22.0)
        # tmin fixture at 12.0 (already °C) — noop passes through.
        tmin_dir = climate / '2m_temperature_24_hour_minimum'
        tmin_dir.mkdir(parents=True)
        _write_fixture_tiff(tmin_dir / 'day_001.tif', 12.0)
        # srad fixture at 20_000 (kJ/m²/d) — mul 1e-3 → 20.0 MJ/m²/d.
        srad_dir = climate / 'solar_radiation_flux_daily'
        srad_dir.mkdir(parents=True)
        _write_fixture_tiff(srad_dir / 'day_001.tif', 20_000.0)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sarra_py_tmax_validates_in_celsius_range(self):
        """tmax fixture at 22 °C must report ~22 (not ~-251 °C from
        the old redundant K→°C subtraction). Mutation-proof: reverting
        the mapping to ("add", -273.15) produces -251.15 which falls
        outside the ± 2 °C tolerance."""
        checks = _validate_sarra_py_geotiffs(self.tmpdir)
        by_var = {c['details']['variable']: c for c in checks
                  if c['check'].startswith('post_translate_range_')}
        self.assertIn('tmax', by_var)
        self.assertAlmostEqual(
            by_var['tmax']['details']['observed_max'], 22.0, places=1,
        )

    def test_sarra_py_tmin_noop_conversion(self):
        """Same noop path for tmin — the .tif is already °C."""
        checks = _validate_sarra_py_geotiffs(self.tmpdir)
        by_var = {c['details']['variable']: c for c in checks
                  if c['check'].startswith('post_translate_range_')}
        self.assertIn('tmin', by_var)
        self.assertAlmostEqual(
            by_var['tmin']['details']['observed_max'], 12.0, places=1,
        )

    def test_sarra_py_srad_kj_to_mj(self):
        """srad fixture at 20_000 kJ/m²/d must report ~20 MJ/m²/d
        (not ~0.02 from the old 1e-6 mapping). Mutation-proof:
        reverting to ("mul", 1e-6) produces 0.02, which does not
        round-trip through `assertAlmostEqual(..., 20.0, places=1)`."""
        checks = _validate_sarra_py_geotiffs(self.tmpdir)
        by_var = {c['details']['variable']: c for c in checks
                  if c['check'].startswith('post_translate_range_')}
        self.assertIn('srad', by_var)
        self.assertAlmostEqual(
            by_var['srad']['details']['observed_max'], 20.0, places=1,
        )
