"""Production tests for SARRA-Py GeoTIFF post-translate validation.

Covers three acceptance criteria:
  AC 1.4.6 — variable coverage: all 4 mapped subdirs produce checks
  AC 1.4.7 — silent-failure regression: empty dir → warning + searched_paths
  AC 1.4.8 — unit conversion: Kelvin→°C (add) and J/m²→MJ/m² (mul)
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
    """AC 1.4.8 — tmax Kelvin→°C and srad J/m²→MJ/m² conversions."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix='sarra-py-test-'))
        climate = self.tmpdir / 'data' / 'climate'
        # tmax at 295.15 K = 22.0 °C
        tmax_dir = climate / '2m_temperature_24_hour_maximum'
        tmax_dir.mkdir(parents=True)
        _write_fixture_tiff(tmax_dir / 'day_001.tif', 295.15)
        # srad at 20_000_000 J/m² = 20.0 MJ/m²
        srad_dir = climate / 'solar_radiation_flux_daily'
        srad_dir.mkdir(parents=True)
        _write_fixture_tiff(srad_dir / 'day_001.tif', 20_000_000.0)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_kelvin_to_celsius_and_joules_to_megajoules(self):
        checks = _validate_sarra_py_geotiffs(self.tmpdir)
        by_var = {c['details']['variable']: c for c in checks
                  if c['check'].startswith('post_translate_range_')}

        self.assertIn('tmax', by_var)
        self.assertAlmostEqual(by_var['tmax']['details']['observed_max'], 22.0, places=2)

        self.assertIn('srad', by_var)
        self.assertAlmostEqual(by_var['srad']['details']['observed_max'], 20.0, places=2)
