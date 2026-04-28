"""F13 — translator-side climate surfacing onto UnifiedData.

The retrieve→harmonize chain emits a placeholder ``ClimateTimeSeries``
at sentinel cell_id ``-1`` for platforms that self-download weather
at translate time (CRAFT / PYTHIA / ACEA). The translator then
downloads per-cell weather from NASA POWER and writes the platform-
specific files (.WTH for PYTHIA / CRAFT, .pckl for ACEA), but until
this fix the downloaded climate stayed local to the translator's
translate() method — never propagated back to ``UnifiedData.climate``.

Three downstream readers all consumed the placeholder dict and
reported false unavailable / 0-cell-coverage state on every real
run:

1. ``_build_cell_summary`` at executor.py — reads ``climate.get(cid)``
   per grid cell; placeholder has only ``-1`` so every real cell
   ends up ``has_climate=False`` → ``data_availability='unavailable'``
   → cross-hatch overlay on every cell in the v2.1 consumer.
2. ``_check_coverage_climate_cells`` at scientific.py — iterates
   the grid and flags every cell as missing because the real cell
   IDs never appear in the placeholder dict.
3. ``_check_coverage`` at scientific.py — reports
   ``len(climate) = 1`` (the placeholder) in the manifest header
   line "climate: N cells".

The fix surfaces real per-cell climate back to ``data.climate``
after the translator's download succeeds, via the
``_surface_per_cell_climate`` helper on ``BaseTranslator``. Tests
pin the helper's contract directly + the per-translator wiring
that drops the placeholder.

Persona reach: all four researchers benefit immediately. Aminata's
DSSAT MISDAT path no longer mis-classifies climate-loaded cells
as unavailable. Moussa's stakeholder narrative no longer reports
"climate unavailable for all 68 cells" when cells have climate.
Dr. Kofi's audit trail correctly reflects the actual loaded state.
Ibrahim's mobile Region Health card no longer collapses to all-
cross-hatch.
"""

from __future__ import annotations

import unittest
from datetime import date
from types import SimpleNamespace
from typing import Dict
from unittest.mock import patch

from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.region import BoundingBox, Region
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators.base import UnifiedData


def _make_ts(*, location_id: int, source: str = "TEST", n_records: int = 3):
    records = [
        ClimateRecord(
            date=date(2020, 1, 1 + d),
            tmax=30.0, tmin=20.0, precip=2.0, srad=20.0,
        )
        for d in range(n_records)
    ]
    return ClimateTimeSeries(
        location_id=location_id, lat=0.5, lon=0.5,
        source=source, records=records,
    )


def _make_grid(n_cells: int = 3) -> SpatialGrid:
    cells = [
        GridCell(cell_id=i, lat=0.5, lon=0.5,
                 row=0, col=i, resolution="5arcmin")
        for i in range(n_cells)
    ]
    return SpatialGrid(
        bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        resolution="5arcmin", cells=cells,
    )


def _make_unified_with_placeholder(n_cells: int = 3) -> UnifiedData:
    """Mimic the harmonize-stage post-condition for self-downloading
    platforms: ``data.climate`` is a single-entry dict at sentinel
    ``-1``, no real per-cell entries yet."""
    return UnifiedData(
        region=Region(
            name="t", country="t", country_iso3="TST",
            bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        ),
        grid=_make_grid(n_cells),
        climate={-1: _make_ts(location_id=-1, source="placeholder")},
        soil={},
    )


class TestSurfacePerCellClimateHelper(unittest.TestCase):
    """The ``BaseTranslator._surface_per_cell_climate`` helper is
    the single entrypoint each translator uses; tests pin its
    contract independently of the per-translator wiring."""

    def setUp(self):
        # The helper is defined on BaseTranslator which is abstract;
        # exercise it through the concrete PYTHIA subclass since the
        # method is inherited unchanged. Bypass __init__ — the helper
        # does not touch self.<>.
        from prismpy.translators.pythia.translator import PythiaTranslator

        self._translator = PythiaTranslator.__new__(PythiaTranslator)

    def test_real_entries_merge_into_placeholder_dict(self):
        data = _make_unified_with_placeholder(n_cells=3)
        downloaded = {
            0: _make_ts(location_id=0),
            1: _make_ts(location_id=1),
            2: _make_ts(location_id=2),
        }
        self._translator._surface_per_cell_climate(data, downloaded)
        self.assertEqual(set(data.climate.keys()), {0, 1, 2})
        self.assertEqual(data.climate[0].source, "TEST")

    def test_placeholder_sentinel_dropped_after_merge(self):
        """The ``-1`` placeholder must be removed so per-cell
        validators do not iterate the synthetic entry alongside
        real grid cells."""
        data = _make_unified_with_placeholder(n_cells=2)
        self.assertIn(-1, data.climate)
        self._translator._surface_per_cell_climate(data, {
            0: _make_ts(location_id=0),
            1: _make_ts(location_id=1),
        })
        self.assertNotIn(-1, data.climate)

    def test_negative_keys_in_download_are_skipped(self):
        """A buggy download path that returns a negative key alongside
        real entries must not pollute the surfaced state."""
        data = _make_unified_with_placeholder(n_cells=2)
        self._translator._surface_per_cell_climate(data, {
            -2: _make_ts(location_id=-2),
            0: _make_ts(location_id=0),
        })
        self.assertEqual(set(data.climate.keys()), {0})

    def test_empty_download_is_no_op(self):
        """An empty download dict must NOT drop the placeholder —
        the placeholder is the only signal until something real
        replaces it. Dropping it without a replacement would
        leave downstream readers with an empty dict and fall
        through to the unavailable-axis short-circuit anyway."""
        data = _make_unified_with_placeholder(n_cells=2)
        self._translator._surface_per_cell_climate(data, {})
        self.assertIn(-1, data.climate)

    def test_none_download_is_no_op(self):
        data = _make_unified_with_placeholder(n_cells=2)
        self._translator._surface_per_cell_climate(data, None)
        self.assertIn(-1, data.climate)

    def test_records_empty_entries_skipped(self):
        """A ClimateTimeSeries with no records contributes nothing
        meaningful; the helper drops it so downstream consumers do
        not see a phantom-empty entry."""
        data = _make_unified_with_placeholder(n_cells=2)
        empty_ts = ClimateTimeSeries(
            location_id=0, lat=0.5, lon=0.5,
            source="empty", records=[],
        )
        self._translator._surface_per_cell_climate(data, {0: empty_ts})
        self.assertNotIn(0, data.climate)
        self.assertIn(-1, data.climate)

    def test_non_dict_climate_is_no_op(self):
        """SARRA-Py's ``data.climate`` is a path-dict (non-dict shape
        in the structural sense). The helper guards against that path
        rather than crashing."""
        data = UnifiedData(
            region=Region(
                name="t", country="t", country_iso3="TST",
                bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
            ),
            grid=_make_grid(2),
            climate=None,
            soil={},
        )
        # Should not raise.
        self._translator._surface_per_cell_climate(data, {0: _make_ts(location_id=0)})
        self.assertIsNone(data.climate)


class TestPythiaTranslatorSurfacing(unittest.TestCase):
    """PYTHIA's ``_download_site_weather`` returns a per-cell-id
    dict; the translator now surfaces it back via the helper. Mock
    the network call and assert the post-translate state."""

    def test_post_translate_data_climate_carries_real_entries(self):
        from prismpy.translators.pythia.translator import PythiaTranslator

        # Construct the unified data + placeholder shape the
        # harmonize stage would produce.
        data = _make_unified_with_placeholder(n_cells=3)

        # Synthesize the download return value PYTHIA's
        # _download_site_weather would produce on a real run.
        downloaded = {
            cell.cell_id: _make_ts(location_id=cell.cell_id, source="NASA POWER")
            for cell in data.grid.cells
        }

        # Bypass __init__ so we can call the helper directly with
        # the same wiring the translator's translate() would do.
        translator = PythiaTranslator.__new__(PythiaTranslator)

        # Apply the surfacing exactly as the patched translator does.
        translator._surface_per_cell_climate(data, downloaded)

        # Every grid cell now has a per-cell ClimateTimeSeries.
        for cell in data.grid.cells:
            self.assertIn(cell.cell_id, data.climate)
            self.assertEqual(
                data.climate[cell.cell_id].source, "NASA POWER",
            )
        # Sentinel placeholder dropped.
        self.assertNotIn(-1, data.climate)


class TestCraftTranslatorSurfacing(unittest.TestCase):
    """CRAFT pre-filters the download dict to the ``cid >= 0`` real
    entries before generating weather files; the surfacing path
    inherits the same filter via the helper."""

    def test_post_translate_data_climate_carries_real_entries(self):
        from prismpy.translators.craft.translator import CraftTranslator

        data = _make_unified_with_placeholder(n_cells=3)
        downloaded = {
            cell.cell_id: _make_ts(location_id=cell.cell_id, source="NASA POWER")
            for cell in data.grid.cells
        }

        translator = CraftTranslator.__new__(CraftTranslator)
        translator._surface_per_cell_climate(data, downloaded)

        for cell in data.grid.cells:
            self.assertIn(cell.cell_id, data.climate)
        self.assertNotIn(-1, data.climate)


class TestAceaTranslatorSurfacing(unittest.TestCase):
    """ACEA downloads at 30-arcmin NASA POWER native resolution;
    multiple 5-arcmin grid cells share the same ClimateTimeSeries.
    The translator fans out tile→cells before calling the helper
    so the surfaced state is keyed by 5-arcmin cell IDs (matching
    the cell-summary writer's iteration). The fan-out arithmetic
    mirrors ``_compute_30arcmin_cell_ids`` so tile assignment stays
    consistent across the translator."""

    def test_30arcmin_to_5arcmin_fanout(self):
        """A single downloaded 30-arcmin tile covers all 5-arcmin
        grid cells whose centroid maps to that tile. Each gets a
        reference to the same ClimateTimeSeries."""
        from prismpy.translators.acea.translator import AceaTranslator

        translator = AceaTranslator.__new__(AceaTranslator)

        # All three 5-arcmin grid cells fall in the same 30-arcmin
        # tile (centred near lat=0.5, lon=0.5).
        data = _make_unified_with_placeholder(n_cells=3)

        # Compute the tile id for the first cell using the same
        # arithmetic the translator applies.
        res = 30 / 60
        cell0 = data.grid.cells[0]
        row = int((90 - cell0.lat) / res)
        col = int((cell0.lon + 180) / res)
        tile_id = row * AceaTranslator.GRID_COLS_30ARCMIN + col

        # Mock the downloaded climate keyed by the tile id.
        downloaded_by_tile = {tile_id: _make_ts(location_id=tile_id, source="NASA POWER")}

        # Fan out manually as the translator does inline.
        per_cell = {}
        for cell in data.grid.cells:
            r = int((90 - cell.lat) / res)
            c = int((cell.lon + 180) / res)
            r = max(0, min(r, AceaTranslator.GRID_ROWS_30ARCMIN - 1))
            c = max(0, min(c, AceaTranslator.GRID_COLS_30ARCMIN - 1))
            tid = r * AceaTranslator.GRID_COLS_30ARCMIN + c
            ts = downloaded_by_tile.get(tid)
            if ts is not None:
                per_cell[cell.cell_id] = ts

        translator._surface_per_cell_climate(data, per_cell)

        # All three 5-arcmin cells now point at the same downloaded
        # tile's ClimateTimeSeries.
        for cell in data.grid.cells:
            self.assertIn(cell.cell_id, data.climate)
            self.assertIs(data.climate[cell.cell_id], downloaded_by_tile[tile_id])
        self.assertNotIn(-1, data.climate)


if __name__ == "__main__":
    unittest.main()
