"""Issue 5 (warning-auditor HIGH) — the scientific validator's
per-cell value-range check used to emit a stale
'Climate value range not checked' info record for file-based
(GeoTIFF) climate. The post-translate sampled validator
(`_validate_sarra_py_geotiffs`) provides authoritative per-variable
coverage by opening ~10 random rasters per variable, so the info
line contradicted the real coverage users see in the ranges card.

This file guards the suppression: file-based climate → no info
record. Non-file-based (per-cell) climate → normal value-range
records emitted as before.
"""

import types
import unittest

from prismpy.validators.scientific import _check_value_ranges


def _make_unified_data(*, climate, soil=None):
    """Minimal duck-typed stand-in for the UnifiedData contract."""
    ud = types.SimpleNamespace()
    ud.climate = climate
    ud.soil = soil if soil is not None else {}
    return ud


class TestClimateInfoLineSuppressedForPlatformsWithSampledCheck(unittest.TestCase):
    """The info line was the only `value_range_climate` record
    `_check_value_ranges` emitted on file-based climate. With the
    suppression in place, no such record appears at all — the
    post-translate sampled check owns that coverage."""

    def test_file_based_climate_emits_no_value_range_climate_info(self):
        """SARRA-Py-shape climate dict (keyed by rainfall_dir /
        agera5_dir) must NOT produce a `value_range_climate` info
        record. Mutation-proof: restoring the append call would
        re-introduce one record with result='info' and a summary
        starting with 'Climate value range not checked'."""
        climate = {
            'rainfall_dir': '/tmp/fake-rainfall',
            'agera5_dir': '/tmp/fake-agera5',
        }
        checks = _check_value_ranges(_make_unified_data(climate=climate))
        summaries = [c.get('summary', '') for c in checks]
        checks_with_name = [c for c in checks if c.get('check') == 'value_range_climate']
        self.assertEqual(
            checks_with_name, [],
            f'expected no value_range_climate info on file-based climate, '
            f'got: {checks_with_name}',
        )
        # Belt-and-braces: the specific suppressed phrase must not
        # appear in any emitted summary.
        self.assertFalse(
            any('Climate value range not checked' in s for s in summaries),
            f'stale info phrase leaked into summaries: {summaries}',
        )

    def test_per_cell_climate_still_emits_value_range_records(self):
        """Two-way binding — suppression only fires on file-based
        climate. A per-cell (in-memory records) climate must still
        produce the per-variable value_range_<var> records. Without
        this guard, an accidental change that suppresses on ALL
        climate shapes would go undetected."""

        class _Record:
            def __init__(self, tmax, tmin, precip, srad):
                self.tmax = tmax
                self.tmin = tmin
                self.precip = precip
                self.srad = srad

        class _Series:
            def __init__(self):
                self.source = 'agera5'
                self.records = [_Record(32.0, 19.0, 1.2, 22.0)]

        climate = {'cell-0001': _Series()}
        checks = _check_value_ranges(_make_unified_data(climate=climate))
        value_range_checks = [
            c for c in checks
            if c.get('check', '').startswith('value_range_')
            and not c.get('check', '').startswith('value_range_soil_')
        ]
        self.assertGreater(
            len(value_range_checks), 0,
            'per-cell climate path produced no value_range_<var> records',
        )
