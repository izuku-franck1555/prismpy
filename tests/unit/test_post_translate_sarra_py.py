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


class TestSarraPyPartialUnreadableSample(unittest.TestCase):
    """Codex self-check R4 HIGH — a range check that draws from a
    degraded sample (some files unreadable) must be flagged as
    warning, not pass. Otherwise broad climate-file corruption
    hides behind a happy-path record that still claims "10-file
    sample" while actually drawing from one good file."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix='sarra-py-partial-'))
        climate = self.tmpdir / 'data' / 'climate'
        # tmax only — 2 valid °C TIFFs, 8 placeholder non-TIFF bytes.
        # The helper's random.sample picks 10 files; whatever mix it
        # picks, SOME will be valid and some won't, so the warning
        # path fires deterministically as long as at least one of
        # each is in the sample.
        #
        # Simpler fixture: 1 valid + 9 bad guarantees partial
        # coverage regardless of sample selection.
        tmax_dir = climate / '2m_temperature_24_hour_maximum'
        tmax_dir.mkdir(parents=True)
        _write_fixture_tiff(tmax_dir / 'valid_001.tif', 22.0)
        for i in range(9):
            (tmax_dir / f'bad_{i:03d}.tif').write_bytes(b'not-a-tif')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_partial_unreadable_sample_downgrades_to_warning(self):
        """A 1-good + 9-bad sample must emit `result: warning`, not
        `pass`, so operators see the effective sample collapsed."""
        checks = _validate_sarra_py_geotiffs(self.tmpdir)
        tmax_records = [
            c for c in checks
            if c.get('check') == 'post_translate_range_sarra_py_tmax'
        ]
        self.assertEqual(len(tmax_records), 1)
        self.assertEqual(
            tmax_records[0]['result'], 'warning',
            f'degraded sample should warn, got {tmax_records[0]}',
        )

    def test_partial_unreadable_sample_persists_effective_coverage(self):
        """The record must carry `unreadable_count` and
        `effective_sample_size` so the packaged
        validation_report.json has machine-readable coverage data."""
        checks = _validate_sarra_py_geotiffs(self.tmpdir)
        record = next(
            c for c in checks
            if c.get('check') == 'post_translate_range_sarra_py_tmax'
        )
        details = record['details']
        self.assertEqual(details.get('sample_size'), 10)
        self.assertGreater(details.get('unreadable_count'), 0)
        self.assertLess(details.get('effective_sample_size'), 10)
        # The summary must name the degradation in plain text too.
        self.assertIn('sample degraded', record['summary'])


class TestSarraPyEmptySampleFiles(unittest.TestCase):
    """Codex self-check — a SARRA-Py sample mix where files open
    cleanly but yield no finite values (all-nodata, all-NaN) must
    be flagged as degraded coverage. Earlier accounting only
    counted `unreadable` (open-failure) files, so a 1-good +
    9-empty sample would still pass silently as a "10-file
    sample". The empty count now feeds into effective_sample and
    the warning decision."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix='sarra-py-empty-'))
        climate = self.tmpdir / 'data' / 'climate'
        # 1 valid tmax TIFF + 9 nodata-only TIFFs (written with the
        # fixture helper, which writes a constant value; setting the
        # nodata tag to the written value makes every pixel fall out
        # of `data[data != nodata]`).
        tmax_dir = climate / '2m_temperature_24_hour_maximum'
        tmax_dir.mkdir(parents=True)
        _write_fixture_tiff(tmax_dir / 'valid_001.tif', 22.0)
        for i in range(9):
            path = tmax_dir / f'nodata_{i:03d}.tif'
            # Write an all-zeros TIFF with nodata=0 so every pixel
            # gets filtered out.
            from rasterio.transform import from_bounds as _from_bounds
            with rasterio.open(
                path, 'w', driver='GTiff', height=2, width=2,
                count=1, dtype='float32', crs='EPSG:4326',
                transform=_from_bounds(9.5, 5.0, 10.0, 5.5, 2, 2),
                nodata=0.0,
            ) as dst:
                dst.write(np.zeros((1, 2, 2), dtype=np.float32))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sample_is_deterministic_and_persists_file_identity(self):
        """Gate B MEDIUM — the sample must be reproducible and the
        result must persist the actual filenames sampled + which
        ones failed. Previously `random.sample` was unseeded and
        details only had counts, so an operator couldn't reproduce
        a warning or open the corrupt files. Two back-to-back
        invocations must produce the same sampled_files list."""
        checks_first = _validate_sarra_py_geotiffs(self.tmpdir)
        checks_second = _validate_sarra_py_geotiffs(self.tmpdir)
        first = next(
            c for c in checks_first
            if c.get('check') == 'post_translate_range_sarra_py_tmax'
        )
        second = next(
            c for c in checks_second
            if c.get('check') == 'post_translate_range_sarra_py_tmax'
        )
        # Deterministic sample: two invocations on the same input
        # produce identical sampled_files lists.
        self.assertEqual(
            first['details']['sampled_files'],
            second['details']['sampled_files'],
        )
        # Each sampled entry is a bare filename (not a full path)
        # so the payload stays compact in the packaged report.
        for name in first['details']['sampled_files']:
            self.assertNotIn('/', name)

    def test_sample_changes_when_file_set_changes(self):
        """Codex Path A follow-up MEDIUM — the sample must depend
        on the FILE SET, not just the file count. Earlier seed
        was `{var}:{len(tifs)}` so every run with the same count
        picked the same 10 positions — a corruption outside those
        positions was systematically missed. The content-hashed
        seed makes the sample reproducible AND responsive to any
        file-set change."""
        checks_before = _validate_sarra_py_geotiffs(self.tmpdir)
        sampled_before = next(
            c['details']['sampled_files'] for c in checks_before
            if c.get('check') == 'post_translate_range_sarra_py_tmax'
        )
        # Add a new valid tif for tmax — different file set, same count+1.
        new_path = (
            self.tmpdir / 'data' / 'climate'
            / '2m_temperature_24_hour_maximum' / 'extra_valid.tif'
        )
        _write_fixture_tiff(new_path, 23.0)
        checks_after = _validate_sarra_py_geotiffs(self.tmpdir)
        sampled_after = next(
            c['details']['sampled_files'] for c in checks_after
            if c.get('check') == 'post_translate_range_sarra_py_tmax'
        )
        # Different file set → the sample has changed.
        self.assertNotEqual(sampled_before, sampled_after)

    def test_partial_unreadable_sample_persists_file_identity(self):
        """Companion to the above — when files DO fail, their
        names land in the details payload so an operator can
        grep which files corrupted. Persisted names are
        filename-only, not absolute paths."""
        checks = _validate_sarra_py_geotiffs(self.tmpdir)
        record = next(
            c for c in checks
            if c.get('check') == 'post_translate_range_sarra_py_tmax'
        )
        details = record['details']
        self.assertIn('unreadable_files', details)
        self.assertIn('empty_files', details)
        # 9-of-10 bad bytes → unreadable list is non-empty and
        # the counts agree with the lists' lengths.
        self.assertEqual(
            len(details['unreadable_files']),
            details['unreadable_count'],
        )
        self.assertEqual(
            len(details['empty_files']),
            details['empty_count'],
        )

    def test_mixed_valid_and_empty_sample_warns_and_counts_empty(self):
        """The 1-good + 9-empty mix must produce `result=warning`
        AND report `empty_count` on the details payload."""
        checks = _validate_sarra_py_geotiffs(self.tmpdir)
        record = next(
            c for c in checks
            if c.get('check') == 'post_translate_range_sarra_py_tmax'
        )
        details = record['details']
        # Every file opened cleanly → unreadable is 0. But 9 of them
        # had no finite values → empty counts them.
        self.assertEqual(details.get('unreadable_count'), 0)
        self.assertGreater(details.get('empty_count'), 0)
        # Effective sample collapsed below target.
        self.assertLess(details.get('effective_sample_size'), 10)
        # And the record downgrades to warning so the user sees it.
        self.assertEqual(record['result'], 'warning')
        # Summary names BOTH failure modes so operator can diagnose.
        self.assertIn('empty', record['summary'])
