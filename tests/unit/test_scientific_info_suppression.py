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

The current contract (P.1 persona copy, post codex rounds 2+3+R3):
file-based climate → one info record whose summary explains the
sampling in plain language ("random sample of 10 output files per
variable") and tells the reader to look for "another SARRA-Py
post-translate message in this report" if a variable is missing.

Copy is self-contained — it refers only to artifacts present IN
THIS REPORT (the validation record list), not to "pipeline steps
above" (which is a web-UI-only concept, missing from the packaged
`validation_report.json` that researchers audit offline). Codex
R3 HIGH caught that phrasing pointed users at dead-end context
in the JSON artifact.

To uphold the "look for another SARRA-Py message" promise, the
post-translate validator (`_validate_sarra_py_geotiffs`) now
emits an explicit warning for any variable whose 10-file sample
yields zero readable values — previously that path silently
skipped the variable, contradicting the copy's promise. The
regression tests below prove the promise is honored across all
three failure modes (empty climate dir, missing rasterio,
unreadable sampled tiffs).
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
        """V2-22b/P.1 persona copy (post codex HIGH round 2) — the
        summary must read in plain language and carry five
        load-bearing properties:

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
           'delegated' developer-copy).
        5. NOT attribute absent records to a SPECIFIC failure mode
           (e.g., 'translation didn't complete') — the post-translate
           validator also emits zero range records on empty-output
           paths and missing-rasterio paths, and the prior wording
           would misdirect operators to the wrong pipeline stage."""
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
        # (3) copy is self-contained — refers only to content
        # present in the packaged `validation_report.json` AND the
        # web UI. Codex R3 HIGH flagged that "pipeline steps above"
        # was dead text inside the JSON artifact researchers audit.
        self.assertNotIn('pipeline steps above', summary)
        # Must still point users at the in-report follow-up so the
        # silent-skip story codex caught is addressable.
        self.assertIn('in this report', summary)
        # (4) prior misleading phrasings must not regress.
        self.assertNotIn('not checked', summary.lower())
        self.assertNotIn('spot-checked', summary.lower())
        self.assertNotIn('delegated', summary.lower())
        # (5) must not attribute absence to a single failure mode.
        self.assertNotIn("translation didn't complete", summary)
        self.assertNotIn("translation did not complete", summary)
        self.assertNotIn(
            "one of the pipeline steps above didn't finish", summary,
            "narrower didn't-finish phrasing was the codex-R2 HIGH — "
            "must not resurface",
        )

    def test_unreadable_sampled_tifs_emit_explicit_warning(self):
        """Codex self-check R3 HIGH — the info-copy promise that
        "another SARRA-Py post-translate message explains why a
        variable is missing" is only honest if the validator
        actually emits such a message on every silent-skip path.
        Previously, a variable whose entire 10-file sample failed
        to open had NO record emitted — contradicting the
        promise. The fix adds an explicit warning record for that
        path so the user always has a paper trail."""
        import shutil
        import tempfile
        from pathlib import Path as _P
        from prismpy.validators.post_translate import (
            _validate_sarra_py_geotiffs,
            SARRA_PY_VAR_MAPPING,
        )

        tmpdir = _P(tempfile.mkdtemp(prefix='p1-unreadable-'))
        try:
            # Write placeholder bytes that rasterio cannot open as
            # GeoTIFF. Every sampled file fails → the silent-skip
            # branch is hit → must emit an explicit warning.
            climate = tmpdir / 'data' / 'climate'
            for subdir_name in SARRA_PY_VAR_MAPPING:
                var_dir = climate / subdir_name
                var_dir.mkdir(parents=True)
                # Need at least one tif per subdir so the validator
                # reaches the read loop (empty dirs short-circuit).
                (var_dir / 'day_001.tif').write_bytes(b'not-a-tif')

            checks = _validate_sarra_py_geotiffs(tmpdir)

            # Exactly one warning per variable (4 total).
            warnings = [
                c for c in checks
                if c.get('check', '').startswith('post_translate_range_sarra_py_')
                and c.get('result') == 'warning'
            ]
            self.assertEqual(
                len(warnings), len(SARRA_PY_VAR_MAPPING),
                f'expected one warning per variable on all-unreadable path, got: {warnings}',
            )
            # Each warning must name the variable and the
            # sample-size / unreadable count so an auditor reading
            # the JSON report has grep-able context.
            for w in warnings:
                self.assertIn('sampled GeoTIFFs', w['summary'])
                self.assertIn('unreadable', w['summary'])
                self.assertIn('variable', w['details'])
                self.assertIn('unreadable_count', w['details'])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_absent_range_records_when_rasterio_missing(self):
        """Codex self-check R2 HIGH regression — the validator-local
        failure mode the R1 test didn't cover. `_validate_sarra_py_geotiffs`
        has a try/except around its `import rasterio` that emits a
        `post_translate_climate_sarra_py` info record ("rasterio not
        available") and returns WITHOUT any `post_translate_range_sarra_py_*`
        records. All pipeline steps finished, the translation produced
        files, but the validator couldn't read them.

        If the info-copy guidance said only "check the pipeline steps
        above", a user hitting this environment issue would investigate
        the wrong subsystem. The "post-translate messages in this
        report" branch of the guidance is what covers this path.
        """
        import shutil
        import sys
        import tempfile
        from pathlib import Path as _P
        from unittest.mock import patch as _patch
        from prismpy.validators.post_translate import (
            _validate_sarra_py_geotiffs,
            SARRA_PY_VAR_MAPPING,
        )

        tmpdir = _P(tempfile.mkdtemp(prefix='p1-rasterio-miss-'))
        try:
            # Create the climate tree with at least one tif per
            # variable so the validator reaches the `import rasterio`
            # branch (empty-dir case short-circuits earlier).
            climate = tmpdir / 'data' / 'climate'
            for subdir_name in SARRA_PY_VAR_MAPPING:
                var_dir = climate / subdir_name
                var_dir.mkdir(parents=True)
                # Placeholder file — won't actually be read because
                # the import fails before the read loop.
                (var_dir / 'fake.tif').write_bytes(b'')

            # Simulate `import rasterio` failing with ImportError.
            import builtins
            real_import = builtins.__import__

            def _patched_import(name, *args, **kwargs):
                if name == 'rasterio':
                    raise ImportError('simulated missing rasterio')
                return real_import(name, *args, **kwargs)

            with _patch.object(builtins, '__import__', side_effect=_patched_import):
                checks = _validate_sarra_py_geotiffs(tmpdir)

            # No range records emitted.
            range_records = [
                c for c in checks
                if c.get('check', '').startswith('post_translate_range_sarra_py_')
            ]
            self.assertEqual(
                range_records, [],
                f'expected no range records on rasterio-missing path, got: {range_records}',
            )
            # But an info record IS emitted explaining rasterio
            # unavailability — this is the thing the info-copy
            # guidance points the user at.
            rasterio_info = [
                c for c in checks
                if c.get('check') == 'post_translate_climate_sarra_py'
                and 'rasterio' in c.get('summary', '').lower()
            ]
            self.assertGreater(
                len(rasterio_info), 0,
                'expected a post_translate_climate_sarra_py info record mentioning rasterio',
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_absent_range_records_can_come_from_post_translate_failure_paths(self):
        """Codex self-check HIGH regression — proves that absent
        `post_translate_range_sarra_py_*` records DO NOT imply a
        failed translation. `_validate_sarra_py_geotiffs` emits a
        warning record with NO range records when the climate
        directory structure is missing files (empty-output path) —
        the translation step may have completed and still produced
        no files. If the info copy said "translation didn't
        complete", it would misdirect operators to the wrong stage
        for this case. The current "one of the pipeline steps
        above" phrasing covers both the missing-files path AND
        the translation-failed path without over-committing to
        either diagnosis."""
        import shutil
        import tempfile
        from pathlib import Path as _P
        from prismpy.validators.post_translate import (
            _validate_sarra_py_geotiffs,
        )

        tmpdir = _P(tempfile.mkdtemp(prefix='p1-regression-'))
        try:
            # Present the climate directory but no per-variable
            # subdirs — mimics the post-translate-only failure mode
            # where translation finished with empty output.
            (tmpdir / 'data' / 'climate').mkdir(parents=True)

            checks = _validate_sarra_py_geotiffs(tmpdir)

            # A warning record IS emitted (the post-translate
            # validator ran and found nothing) — but NO
            # `post_translate_range_sarra_py_*` records. So the
            # Results page user would see this warning but no
            # per-variable ranges, and the scientific info copy's
            # "one of the pipeline steps" guidance correctly points
            # them at the report's pipeline cards to find which
            # step produced the empty output.
            range_records = [
                c for c in checks
                if c.get('check', '').startswith('post_translate_range_sarra_py_')
            ]
            self.assertEqual(
                range_records, [],
                f'expected no range records on empty-output path, got: {range_records}',
            )
            # Positive control — the warning DID fire (so the user
            # gets a signal, it just isn't a range record).
            warnings = [
                c for c in checks if c.get('result') == 'warning'
            ]
            self.assertGreater(
                len(warnings), 0,
                'expected a warning record on empty-output path',
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

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
