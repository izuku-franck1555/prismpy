"""Issue 5 (warning-auditor HIGH + codex self-check MEDIUM) — the
scientific validator's per-cell value-range check used to emit a
'Climate value range not checked' info record on file-based
(GeoTIFF) climate. That was wrong: the post-translate sampled
validator (`_validate_sarra_py_geotiffs`) DOES check the ranges
by opening 10 random rasters per variable. The first fix
suppressed the record entirely, but codex caught that this
removed the only user-visible caveat that the check is sampled,
not exhaustive — a bad GeoTIFF outside the sample would pass
silently.

The current contract: file-based climate → one info record with
the sampling policy disclosed; per-cell climate → normal
`value_range_<var>` records as before.
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


class TestClimateInfoLineDisclosesSampledCoverage(unittest.TestCase):
    """The info record must stay visible (one per file-based climate
    call) and its summary must describe the sampled-coverage policy
    — not the misleading 'not checked' phrasing the first Issue 5
    fix replaced. The details block must also tag the coverage as
    'sampled' so downstream surfaces can distinguish exhaustive from
    sampled checks."""

    def test_file_based_climate_emits_one_value_range_climate_record(self):
        """Exactly one `value_range_climate` info record for
        file-based climate. More than one would double-disclose;
        zero would re-introduce the codex regression where users
        lose the sampled-vs-exhaustive caveat."""
        climate = {
            'rainfall_dir': '/tmp/fake-rainfall',
            'agera5_dir': '/tmp/fake-agera5',
        }
        checks = _check_value_ranges(_make_unified_data(climate=climate))
        records = [c for c in checks if c.get('check') == 'value_range_climate']
        self.assertEqual(
            len(records), 1,
            f'expected exactly one value_range_climate record, got: {records}',
        )
        self.assertEqual(records[0]['result'], 'info')

    def test_summary_describes_sampled_coverage_not_skip(self):
        """The summary must advertise that the check IS running on
        a sample, not that it was skipped. Mutation-proof against a
        revert to the original 'not checked' phrasing."""
        climate = {'rainfall_dir': '/tmp/x', 'agera5_dir': '/tmp/y'}
        checks = _check_value_ranges(_make_unified_data(climate=climate))
        summary = next(
            c['summary'] for c in checks
            if c.get('check') == 'value_range_climate'
        )
        # The summary must name the sampled policy.
        self.assertIn('sample', summary.lower())
        # And must NOT resurrect the prior misleading phrasing.
        self.assertNotIn('not checked', summary.lower())
        # And must steer the reader to the authoritative per-variable
        # records emitted by the post-translate validator.
        self.assertIn('post_translate_range_sarra_py', summary)

    def test_details_tag_coverage_kind_as_sampled(self):
        """Downstream surfaces (report generation, UI) need a
        machine-readable flag to differentiate sampled from
        exhaustive climate checks. `coverage_kind='sampled'` on
        the details dict is that flag."""
        climate = {'rainfall_dir': '/tmp/x', 'agera5_dir': '/tmp/y'}
        checks = _check_value_ranges(_make_unified_data(climate=climate))
        details = next(
            c['details'] for c in checks
            if c.get('check') == 'value_range_climate'
        )
        self.assertEqual(details.get('coverage_kind'), 'sampled')
        self.assertIn('sample_policy', details)

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
