"""Issue 5 (warning-auditor HIGH + codex self-check HIGH + V2-22b/P.1
persona copy pass) — the scientific validator's per-cell
value-range check used to emit a 'Climate value range not checked'
info record on file-based (GeoTIFF) climate. That was wrong: the
post-translate sampled validator (`_validate_sarra_py_geotiffs`)
DOES check the ranges by opening 10 random rasters per variable.
The first fix suppressed the record entirely, losing the
sampled-coverage caveat. The second fix asserted the check had
happened ("spot-checked"), overclaiming when translation failed.
The third fix introduced a "delegated to the per-platform
post-translate validator" copy which was honest but read as
technical to non-developer personas — user flagged it.

The current contract (P.1 persona copy): file-based climate →
one info record whose summary explains the sampling in plain
language ("random sample of 10 output files per variable"),
points the reader at the per-variable records that follow, and
tells them where to look if those records are missing ("pipeline
steps above" — unambiguously the Retrieve/Harmonize/Translate/
Package cards, not some other part of the validation list).
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


class TestClimateInfoLineDelegatesToPostTranslate(unittest.TestCase):
    """The info record must stay visible (one per file-based climate
    call) and its summary must DELEGATE to the per-platform
    post-translate check rather than CLAIM the check already
    happened. The details block must tag `coverage_kind='delegated'`
    so downstream surfaces can distinguish "we ran this" from "we
    handed this off"."""

    def test_file_based_climate_emits_one_value_range_climate_record(self):
        """Exactly one `value_range_climate` info record for
        file-based climate. More than one would double-disclose;
        zero would re-introduce the "silent skip" regression."""
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

    def test_summary_uses_plain_language_with_honest_caveats(self):
        """V2-22b/P.1 persona copy — the summary must read in plain
        language and carry four load-bearing properties:

        1. Name the sampling policy honestly ('random sample of 10'
           — the scientific-honesty signal the user still needs).
        2. NOT include the internal code identifier
           `post_translate_range_sarra_py_*` (persona copy, not
           developer copy).
        3. Point the reader at 'pipeline steps above' — the
           disambiguating phrase evaluator added so the user can't
           misread 'earlier steps' as 'earlier in this report'.
        4. NOT resurrect any of the prior misleading phrasings
           ('not checked' underclaim, 'spot-checked' overclaim,
           'delegated' developer-copy)."""
        climate = {'rainfall_dir': '/tmp/x', 'agera5_dir': '/tmp/y'}
        checks = _check_value_ranges(_make_unified_data(climate=climate))
        summary = next(
            c['summary'] for c in checks
            if c.get('check') == 'value_range_climate'
        )
        # (1) scientific-honesty signal preserved.
        self.assertIn('random sample of 10', summary)
        # (2) code identifier stays out of persona copy.
        self.assertNotIn('post_translate_range_sarra_py', summary)
        # (3) disambiguating phrase for the "where to look on error"
        # signal.
        self.assertIn('pipeline steps above', summary)
        # (4) prior misleading phrasings must not regress.
        self.assertNotIn('not checked', summary.lower())
        self.assertNotIn('spot-checked', summary.lower())
        self.assertNotIn('delegated', summary.lower())

    def test_details_tag_coverage_kind_as_delegated(self):
        """Downstream surfaces need a machine-readable flag that
        differentiates a delegated check (runs elsewhere, may or
        may not exist in the report) from an in-line one (always
        ran at this code site). `coverage_kind='delegated'` is
        that flag."""
        climate = {'rainfall_dir': '/tmp/x', 'agera5_dir': '/tmp/y'}
        checks = _check_value_ranges(_make_unified_data(climate=climate))
        details = next(
            c['details'] for c in checks
            if c.get('check') == 'value_range_climate'
        )
        self.assertEqual(details.get('coverage_kind'), 'delegated')
        self.assertIn('delegated_to', details)
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
