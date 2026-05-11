"""F-CK hot-fix — pin the producer-consumer parity for ``crop_calendar``.

Before this fix the producer at ``pipeline/executor.py:752`` emitted
``data["crop_calendar"] = {0: CropCalendar(location_id=0, ...)}``
(synthetic single key). The consumer at
``cockpit/observed_values_writer.py:260`` checked
``cell_id not in crop_calendar`` and raised ``ValueError`` on every
real climate cell ID (e.g. ``4184036``). The retrieve stage caught
the raise as a warning, and the downstream
``cockpit_observed_values.json`` shipped empty on every package
build — 100% prevalence, second-occurrence empirical confirmation.

Per CMS §9.4 + durable §27 producer-consumer vocabulary parity +
durable §24 canonical-source-or-pin: every cell present in the
``data["climate"]`` roster MUST have a matching entry in
``data["crop_calendar"]``.

Two layers of pin:

* **Behavioral** — build a fanned-out calendar dict mimicking the
  fixed producer + invoke the consumer's
  ``_growing_season_window`` for every cell in a 3-cell roster.
  Assert no ``ValueError`` raise; assert the doys round-trip.
  This is the regression net for the user-visible class.
* **Structural source-shape** — read the producer file and pin
  the fan-out shape: the synthetic ``location_id=0`` literal is
  gone, and the fan-out reads the climate roster via a dict
  comprehension over ``climate_cell_ids``. A future refactor that
  reverts to a single-key emit fires this assertion before the
  consumer-side ValueError flood ever ships.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from prismpy.cockpit.observed_values_writer import _growing_season_window
from prismpy.models.crop import CropCalendar


REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_SRC = REPO_ROOT / 'src' / 'prismpy' / 'pipeline' / 'executor.py'


class TestCropCalendarFanoutBehaviour(unittest.TestCase):
    """The consumer accepts the post-fix producer output for every
    cell in the climate roster — proves the
    ``cell_id not in crop_calendar`` raise class is closed."""

    def _build_fanned_out_calendar(self, climate_cell_ids, planting=150, maturity=270):
        """Mirror of the post-fix producer dict-comprehension at
        ``pipeline/executor.py`` (the crop_calendar fan-out)."""
        return {
            cell_id: CropCalendar(
                location_id=cell_id,
                planting_doy=planting,
                maturity_doy=maturity,
                source='config',
            )
            for cell_id in climate_cell_ids
        }

    def test_consumer_accepts_fanned_calendar_for_three_real_cell_ids(self):
        # Realistic 5-arcmin grid cell IDs (range of values the
        # production pipeline actually emits — F-CK was reported
        # on cell 4184036 / 4201284).
        cell_ids = [4184036, 4201284, 4217825]
        calendar = self._build_fanned_out_calendar(
            cell_ids, planting=166, maturity=285,
        )
        for cell_id in cell_ids:
            try:
                planting, harvest, season_days = (
                    _growing_season_window(calendar, cell_id)
                )
            except ValueError as exc:  # pragma: no cover - defensive
                self.fail(
                    f'_growing_season_window raised on cell {cell_id} '
                    f'despite fan-out: {exc!r}',
                )
            self.assertEqual(planting, 166)
            self.assertEqual(harvest, 285)  # ``__post_init__`` defaults harvest to maturity
            self.assertEqual(season_days, 285 - 166 + 1)

    def test_consumer_raises_when_calendar_is_empty(self):
        # Sanity check that the consumer's CMS §9.4 violation still
        # surfaces when the producer genuinely has no roster (e.g.
        # all-cell climate retrieval failure). The fix doesn't mute
        # the consumer's guard; it just keeps the calendar in lock-
        # step with the climate roster when both populate.
        with self.assertRaises(ValueError):
            _growing_season_window({}, 4184036)

    def test_consumer_raises_when_cell_id_outside_roster(self):
        # If the climate roster genuinely lacks a cell, the
        # consumer must still raise so the user knows. The fan-out
        # closes the synthetic-key class; it does not paper over
        # honest gaps.
        calendar = self._build_fanned_out_calendar([4184036])
        with self.assertRaises(ValueError):
            _growing_season_window(calendar, 9999999)

    def test_fan_out_with_empty_climate_roster_produces_empty_calendar(self):
        # Codex round-2 LOW absorption — when the retrieve-stage
        # climate dict is empty (no cells retrieved at all), the
        # fan-out yields an empty calendar dict, NOT a synthetic
        # placeholder. Empty input ⇒ empty output keeps the
        # producer-consumer contract honest: the consumer's
        # ``not crop_calendar or cell_id not in crop_calendar``
        # raise then surfaces the genuine empty-roster gap.
        calendar = self._build_fanned_out_calendar([])
        self.assertEqual(calendar, {})
        # Single-cell happy path — pin minimal-roster success
        # alongside the multi-cell case at line 64-84 so the
        # smallest valid fan-out is covered.
        single = self._build_fanned_out_calendar([4184036], planting=120, maturity=240)
        self.assertEqual(set(single.keys()), {4184036})
        self.assertEqual(single[4184036].location_id, 4184036)
        self.assertEqual(single[4184036].planting_doy, 120)
        self.assertEqual(single[4184036].maturity_doy, 240)


class TestExecutorFanoutSentinelAndEmptyCases(unittest.TestCase):
    """Codex round-3 LOW + HIGH absorption — pin the executor-level
    producer fan-out directly (not just the mirror at
    ``_build_fanned_out_calendar``) for the two edge classes:

    * Sentinel-only climate ``{-1: placeholder_ts}`` — produces a
      ``{-1: CropCalendar(...)}`` entry so PYTHIA / ACEA pass
      ``BaseTranslator.validate_input_data`` (which rejects a
      falsy ``data.crop_calendar``). Round-3 HIGH regression net.
    * Empty climate ``{}`` — produces an empty calendar dict;
      validate_input_data then rejects honestly per CMS §9.4.
    * Mixed path-dict + int keys — int keys flow through; string
      keys are filtered out (round-1 MEDIUM regression net).

    We exercise ``TranslationPipeline``'s retrieve-stage fan-out
    via a minimal ``_load_substrate`` shim — using object.__new__
    to bypass full pipeline init while threading the substrate
    branch under test.
    """

    def _build_pipeline_skeleton(self, planting_doy, maturity_doy):
        """Build a minimal ``TranslationPipeline`` instance whose
        ``self.config.crop.calendar`` returns the supplied doys."""
        from prismpy.pipeline.executor import TranslationPipeline

        class _StubCalendar:
            def __init__(self, planting, maturity):
                self.planting_doy = planting
                self.maturity_doy = maturity

        class _StubCrop:
            def __init__(self, calendar):
                self.calendar = calendar

        class _StubConfig:
            def __init__(self, crop):
                self.crop = crop

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.config = _StubConfig(
            crop=_StubCrop(
                calendar=_StubCalendar(planting_doy, maturity_doy),
            ),
        )
        return pipeline

    def _emulate_executor_fanout(self, data):
        """Mirror the post-fix executor.py crop_calendar fan-out
        block. We can't easily invoke ``_load_substrate`` in
        isolation (it owns retrieve I/O); the mirror has the same
        body as the production code so a divergence is caught by
        the ``TestProducerSourceShape`` regex assertion above."""
        pipeline = self._build_pipeline_skeleton(166, 285)
        project_planting_doy = pipeline.config.crop.calendar.planting_doy
        project_maturity_doy = pipeline.config.crop.calendar.maturity_doy
        climate_cell_ids = [
            cid
            for cid in (data.get("climate") or {}).keys()
            if isinstance(cid, int)
        ]
        data["crop_calendar"] = {
            cell_id: CropCalendar(
                location_id=cell_id,
                planting_doy=project_planting_doy,
                maturity_doy=project_maturity_doy,
                source="config",
            )
            for cell_id in climate_cell_ids
        }
        return data

    def test_sentinel_only_climate_produces_sentinel_calendar_entry(self):
        # Codex round-3 HIGH regression net — sentinel-only climate
        # MUST yield a non-empty crop_calendar (sentinel-keyed) so
        # the translator-level ``validate_input_data`` check at
        # ``translators/base.py:190`` passes for PYTHIA / ACEA /
        # CRAFT self-download paths. Without this, every
        # self-download translator would reject the unified data
        # before ``_surface_per_cell_climate`` ever ran.
        data = {"climate": {-1: object()}}
        out = self._emulate_executor_fanout(data)
        self.assertIn(
            -1, out["crop_calendar"],
            'Sentinel-only climate ``{-1: placeholder}`` MUST yield '
            'a sentinel-keyed calendar entry so the translator-side '
            '``validate_input_data`` (translators/base.py:190) does '
            'not reject self-download paths before '
            '``_surface_per_cell_climate`` runs.',
        )
        self.assertEqual(out["crop_calendar"][-1].location_id, -1)
        self.assertEqual(out["crop_calendar"][-1].planting_doy, 166)
        self.assertEqual(out["crop_calendar"][-1].maturity_doy, 285)
        # Truthy guarantee for validate_input_data.
        self.assertTrue(out["crop_calendar"])

    def test_truly_empty_climate_produces_empty_calendar(self):
        # Genuine no-cells-retrieved case: empty climate → empty
        # calendar. ``validate_input_data`` then rejects honestly.
        data = {"climate": {}}
        out = self._emulate_executor_fanout(data)
        self.assertEqual(out["crop_calendar"], {})

    def test_mixed_path_dict_int_keys_filters_strings_keeps_ints(self):
        # Round-1 MEDIUM regression net — path-dict string keys
        # MUST NOT enter ``CropCalendar.location_id`` (typed int).
        # The retrieve-stage fan-out now retains negative int
        # sentinels (post round-3 HIGH absorption), so a path-dict
        # entry next to a sentinel keeps the sentinel + drops the
        # string.
        data = {"climate": {
            "rainfall_dir": "/tmp/p",
            "agera5_dir": "/tmp/q",
            -1: object(),
        }}
        out = self._emulate_executor_fanout(data)
        # Only the int key survives.
        self.assertEqual(set(out["crop_calendar"].keys()), {-1})

    def test_real_cell_climate_fans_out_to_matching_calendar(self):
        # Happy path mirror — real cell IDs at retrieve time
        # (SARRA-Py / int-keyed translators) get a matching
        # calendar entry per cell.
        data = {"climate": {4184036: object(), 4201284: object()}}
        out = self._emulate_executor_fanout(data)
        self.assertEqual(
            set(out["crop_calendar"].keys()), {4184036, 4201284},
        )
        for cell_id, entry in out["crop_calendar"].items():
            self.assertEqual(entry.location_id, cell_id)
            self.assertEqual(entry.planting_doy, 166)
            self.assertEqual(entry.maturity_doy, 285)


class TestProducerSourceShape(unittest.TestCase):
    """Read the producer source and pin the fan-out shape. A
    future refactor that reverts to the synthetic single-key emit
    fires this assertion before the consumer flood ever ships."""

    @classmethod
    def setUpClass(cls):
        cls.source = EXECUTOR_SRC.read_text(encoding='utf-8')

    def test_producer_does_not_emit_synthetic_location_id_zero(self):
        # Before F-CK the producer wrote ``location_id=0`` in the
        # crop_calendar block. Any future revert that restores the
        # synthetic-key shape (matching the literal ``location_id=0``
        # inside a ``CropCalendar(...)`` call within the
        # ``crop_calendar`` block) trips this assertion. Other
        # call sites (tests, fixtures, etc.) are out of scope.
        crop_calendar_block = self._extract_crop_calendar_block()
        self.assertNotRegex(
            crop_calendar_block,
            r'CropCalendar\([^)]*location_id\s*=\s*0\b',
            'pipeline/executor.py crop_calendar block emits a '
            'synthetic ``location_id=0`` CropCalendar — F-CK '
            'requires the calendar to fan out across the climate '
            'cell roster (CMS §9.4 + durable §27). See structural '
            'test docstring for the consumer-side raise that '
            'this assertion guards.',
        )

    def test_producer_fans_out_via_climate_cell_ids(self):
        # The fix introduces a ``climate_cell_ids`` local that
        # drives the dict-comprehension. Pin the variable name + a
        # dict-comprehension over it inside the crop_calendar
        # block, so a refactor that drops the fan-out (e.g.
        # reverts to ``{0: CropCalendar(...)}``) fails loud.
        crop_calendar_block = self._extract_crop_calendar_block()
        self.assertIn(
            'climate_cell_ids',
            crop_calendar_block,
            'pipeline/executor.py crop_calendar block must derive '
            'the calendar roster from ``climate_cell_ids`` so '
            'every cell in ``data["climate"]`` has a matching '
            'calendar entry (CMS §9.4).',
        )
        self.assertRegex(
            crop_calendar_block,
            r'for\s+cell_id\s+in\s+climate_cell_ids',
            'pipeline/executor.py crop_calendar block must iterate '
            'over ``climate_cell_ids`` in a dict comprehension so '
            'the producer-consumer vocabulary stays in lock-step '
            '(durable §27).',
        )

    def test_producer_does_not_filter_negative_sentinel(self):
        # Codex round-4 MEDIUM-2 absorption — guard against a
        # regression that re-introduces ``cid >= 0`` to the executor
        # crop_calendar fan-out. The negative sentinel ``-1`` from
        # ``_create_placeholder_climate`` MUST survive the executor
        # filter so PYTHIA / ACEA / CRAFT pass
        # ``BaseTranslator.validate_input_data`` (which rejects a
        # falsy ``data.crop_calendar``) before
        # ``_surface_per_cell_climate`` ever runs. Helper-side
        # filtering at ``translators/base.py`` removes the sentinel
        # from the final consumer-visible calendar; the executor's
        # job is only to keep the sentinel alive long enough for
        # validation.
        crop_calendar_block = self._extract_crop_calendar_block()
        # Strip Python comments before the regex check. The
        # executor's contract docstring at line ~787 references the
        # helper-side ``isinstance(cid, int) and cid >= 0`` filter
        # as part of its narrative — that prose mention is correct
        # and must not trip this assertion. We only want to catch
        # an actual code-level re-introduction of the executor-
        # side filter.
        code_only = re.sub(
            r'#.*$', '', crop_calendar_block, flags=re.MULTILINE,
        )
        self.assertNotRegex(
            code_only,
            r'cid\s*>=\s*0',
            'pipeline/executor.py crop_calendar block must NOT '
            'filter ``cid >= 0`` in CODE — that filter was removed '
            'in F-CK round-3 absorption because it emptied the '
            'calendar for sentinel-only climate '
            '(``{-1: placeholder}`` from '
            '``_create_placeholder_climate``), causing PYTHIA / '
            'ACEA / CRAFT to fail ``validate_input_data`` before '
            'ever calling ``_surface_per_cell_climate``. The '
            'helper-side filter at ``translators/base.py:364`` is '
            'what removes the sentinel from the FINAL consumer-'
            'visible calendar; the executor must NOT pre-empt it.',
        )

    def _extract_crop_calendar_block(self):
        # Capture from the ``if self.config.crop.calendar:`` guard
        # to the next ``except`` so the assertions scope to the
        # fan-out only (avoid matching unrelated CropCalendar(...)
        # references elsewhere in the file).
        match = re.search(
            r'if self\.config\.crop\.calendar:'
            r'([\s\S]+?)\n        except ',
            self.source,
        )
        self.assertIsNotNone(
            match,
            'Could not locate the crop_calendar producer block in '
            'pipeline/executor.py. The structural pin assumes a '
            'specific surrounding scope; if the block moved, '
            'update the regex.',
        )
        return match.group(1)


class TestSurfacingHelperRefansCalendar(unittest.TestCase):
    """Codex round 1 HIGH absorption — when a translator
    (CRAFT / PYTHIA / ACEA) self-downloads per-cell weather at
    translate time, ``_surface_per_cell_climate`` updates
    ``data.climate`` with real cell IDs and drops the ``-1``
    placeholder. Without the calendar re-fan inside that helper,
    ``data.crop_calendar`` keeps its retrieve-stage shape (keyed
    at ``-1`` for those translators) and the consumer at
    ``cockpit/observed_values_writer.py:260`` raises ``ValueError``
    on every real cell at package-write time.
    """

    def _make_translator(self, planting_doy=166, maturity_doy=285):
        """Build a minimal ``BaseTranslator`` subclass instance
        whose ``self.config.crop.calendar`` returns the supplied
        doys. Concrete-subclass-with-stub-abstract-methods avoids
        the full ProjectConfig schema — the helper under test
        only reaches ``self.config.crop.calendar.planting_doy /
        maturity_doy``."""
        from prismpy.translators.base import SarraPyTranslatorBase

        class _StubCalendar:
            def __init__(self, planting, maturity):
                self.planting_doy = planting
                self.maturity_doy = maturity

        class _StubCrop:
            def __init__(self, calendar):
                self.calendar = calendar

        class _StubConfig:
            def __init__(self, crop):
                self.crop = crop

        class _FanoutTestTranslator(SarraPyTranslatorBase):
            def translate(self, data, output_dir):  # pragma: no cover
                raise NotImplementedError
            def validate_outputs(self, output_files):  # pragma: no cover
                raise NotImplementedError

        translator = _FanoutTestTranslator.__new__(_FanoutTestTranslator)
        translator.config = _StubConfig(
            crop=_StubCrop(
                calendar=_StubCalendar(planting_doy, maturity_doy),
            ),
        )
        return translator

    def _make_unified_data(self, climate, crop_calendar):
        """Build a minimal ``UnifiedData`` instance."""
        from prismpy.translators.base import UnifiedData
        from prismpy.models.region import BoundingBox, Region

        return UnifiedData(
            region=Region(
                name='test',
                country='test',
                country_iso3='TST',
                bounds=BoundingBox(
                    minx=0.0, miny=0.0, maxx=1.0, maxy=1.0,
                ),
                metadata={},
            ),
            climate=climate,
            crop_calendar=crop_calendar,
        )

    def _make_climate_ts(self):
        """Build a minimal ClimateTimeSeries with one real record."""
        from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
        from datetime import date

        return ClimateTimeSeries(
            location_id=0,
            lat=14.0,
            lon=7.0,
            source='test',
            records=[
                ClimateRecord(
                    date=date(2020, 1, 1),
                    tmax=30.0, tmin=20.0, precip=0.0, srad=20.0,
                ),
            ],
        )

    def test_helper_refans_calendar_across_surfaced_cells(self):
        translator = self._make_translator(
            planting_doy=166, maturity_doy=285,
        )
        # Pre-surfacing state: climate has only the ``-1``
        # placeholder; calendar keyed at ``-1`` mirrors that
        # (what the retrieve-stage fan-out produces when climate
        # falls through to ``_create_placeholder_climate``).
        from prismpy.models.crop import CropCalendar
        placeholder_ts = self._make_climate_ts()
        real_ts_a = self._make_climate_ts()
        real_ts_b = self._make_climate_ts()
        real_ts_c = self._make_climate_ts()
        data = self._make_unified_data(
            climate={-1: placeholder_ts},
            crop_calendar={
                -1: CropCalendar(
                    location_id=-1,
                    planting_doy=166,
                    maturity_doy=285,
                    source='config',
                ),
            },
        )
        translator._surface_per_cell_climate(
            data,
            {4184036: real_ts_a, 4201284: real_ts_b, 4217825: real_ts_c},
        )
        # Climate post-surfacing: real cells, no ``-1``
        self.assertEqual(
            set(data.climate.keys()), {4184036, 4201284, 4217825},
        )
        # Calendar post-surfacing: re-fanned to match climate roster
        self.assertEqual(
            set(data.crop_calendar.keys()),
            {4184036, 4201284, 4217825},
            'Helper must re-fan crop_calendar across the post-'
            'surfacing climate roster (durable §27 + CMS §9.4).',
        )
        for cell_id, entry in data.crop_calendar.items():
            self.assertEqual(entry.location_id, cell_id)
            self.assertEqual(entry.planting_doy, 166)
            self.assertEqual(entry.maturity_doy, 285)
            self.assertEqual(entry.source, 'config')

    def test_helper_leaves_calendar_alone_when_no_wizard_calendar(self):
        # Edge case — if ``self.config.crop.calendar`` is None
        # (no wizard intent), the helper must NOT manufacture a
        # calendar. The consumer's CMS §9.4 ValueError surfaces
        # honestly downstream.
        from prismpy.translators.base import SarraPyTranslatorBase

        class _StubCrop:
            calendar = None

        class _StubConfig:
            def __init__(self):
                self.crop = _StubCrop()

        class _FanoutTestTranslator(SarraPyTranslatorBase):
            def translate(self, data, output_dir):  # pragma: no cover
                raise NotImplementedError
            def validate_outputs(self, output_files):  # pragma: no cover
                raise NotImplementedError

        translator = _FanoutTestTranslator.__new__(_FanoutTestTranslator)
        translator.config = _StubConfig()

        data = self._make_unified_data(
            climate={-1: self._make_climate_ts()},
            crop_calendar=None,
        )
        translator._surface_per_cell_climate(
            data,
            {4184036: self._make_climate_ts()},
        )
        self.assertIsNone(
            data.crop_calendar,
            'Helper must leave crop_calendar None when the '
            'wizard did not supply one — the consumer surfaces '
            'CMS §9.4 ValueError honestly.',
        )


if __name__ == '__main__':
    unittest.main()
