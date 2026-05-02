"""Sprint D.1 AC-4 — unit pin for the HWSD-index → grid-cell-id remap.

Pins the remap block inside ``_retrieve_hwsd_for_grid`` that
translates the HWSD source's index-keyed unavailable_cells list
to the actual grid cell IDs. Without this pin, a future
contributor could break the translation (e.g., pass the index
straight through as cell_id) and downstream consumers would
attach the data-availability routing to the wrong cells.

The E2E test in ``tests/integration/test_hwsd_unavailable_e2e.py``
stubs ``_retrieve_hwsd_for_grid`` itself, so it locks the caller-
side wiring (the cascade orchestrator unpacks the tuple and
stashes the second element) but does NOT exercise the remap
block. This file fills that gap by patching ``HWSDSource`` with
a fake that records misses by index, calling
``_retrieve_hwsd_for_grid`` directly with non-sequential grid
cell IDs, and asserting the returned list is keyed by grid
cell IDs (not indices).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest import TestCase
from unittest.mock import patch

from prismpy.config.schema import (
    BoundaryConfig,
    BoundarySource,
    CropCalendarConfig,
    CropConfig,
    DataSourcesConfig,
    ManualBoundsConfig,
    OutputConfig,
    Platform,
    ProjectConfig,
    ProjectInfo,
    RegionConfig,
    SoilSourceConfig,
    TemporalConfig,
)
from prismpy.models.region import BoundingBox, Region
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.provenance.tracker import ProvenanceTracker


# ---------------------------------------------------------------------------
# Test doubles — minimal fakes used in place of the real HWSD source
# so the test does not depend on rasterio / mdbtools / actual data.
# ---------------------------------------------------------------------------


class _FakeCell:
    """Stand-in for ``SpatialCell``; provides only the attributes
    ``_retrieve_hwsd_for_grid`` reads (cell_id, lat, lon)."""

    def __init__(self, cell_id: int, lat: float, lon: float):
        self.cell_id = cell_id
        self.lat = lat
        self.lon = lon


class _FakeGrid:
    """Stand-in for ``SpatialGrid``; provides only ``.cells``."""

    def __init__(self, cells: List[_FakeCell]):
        self.cells = cells


class _FakeRetrievalResult:
    """Minimal duck-type for the ``RetrievalResult`` shape that
    ``_retrieve_hwsd_for_grid`` reads (success, data, errors)."""

    def __init__(self, success: bool, data=None, errors=None):
        self.success = success
        self.data = data
        self.errors = errors or []


def _make_fake_hwsd_source_class(
    record_indices: List[int],
):
    """Build a fake ``HWSDSource`` class whose ``retrieve()``
    populates ``self.unavailable_cells`` with the supplied
    index-keyed entries and returns a failure result. The fake
    intentionally records misses by index (matching the real
    source's enumerate-based contract) so the executor's remap
    block has work to do.
    """

    class _FakeHWSDSource:
        def __init__(self, config=None, cache_dir=None, provenance=None):
            self.unavailable_cells: List[Dict[str, Any]] = []

        def retrieve(self, region=None, cell_coords=None, **kwargs):
            for idx in record_indices:
                self.unavailable_cells.append({
                    "cell_id": idx,
                    "cause": "soil_no_hwsd_coverage",
                })
            # Failure result — exercises the all-miss propagation
            # path. The remap still runs because the executor
            # captures ``hwsd_source.unavailable_cells`` BEFORE
            # the success check.
            return _FakeRetrievalResult(
                success=False, data=None,
                errors=["fake retrieval failure"],
            )

    return _FakeHWSDSource


def _make_pipeline_with_paths(bil_path, mdb_path) -> TranslationPipeline:
    """Build a minimal CRAFT-targeted pipeline whose
    ``data_sources.soil`` carries the stub bil/mdb paths so the
    executor's path-existence gate at ``_retrieve_hwsd_for_grid``
    is satisfied and the method reaches the try block where the
    remap fires."""
    cfg = ProjectConfig(
        project=ProjectInfo(
            name='executor_hwsd_remap_unit',
            description='Sprint D.1 AC-4 remap unit pin',
        ),
        region=RegionConfig(
            name='Koutiala', country='Mali', country_iso3='MLI',
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=-5.10, miny=11.90, maxx=-5.00, maxy=12.00,
                ),
                inclusion_rule='bbox_intersects',
                min_share_percent=0.0,
            ),
        ),
        crop=CropConfig(
            name='Maize', name_short='mai', variety='Medium-duration',
            calendar=CropCalendarConfig(
                planting_doy=166, maturity_doy=285,
            ),
        ),
        temporal=TemporalConfig(
            start_year=2015, end_year=2020, spinup_years=2,
        ),
        targets=[Platform.CRAFT],
        data_sources=DataSourcesConfig(
            soil=SoilSourceConfig(
                hwsd_bil_path=bil_path,
                hwsd_mdb_path=mdb_path,
            ),
        ),
        output=OutputConfig(
            base_dir='outputs', structure='by_platform',
        ),
    )
    # Provenance disabled so the executor's RESAMPLING_METHOD
    # record_decision call (which depends on the cascade
    # orchestrator having previously called ``start_artifact('soil')``)
    # does not raise when ``_retrieve_hwsd_for_grid`` is invoked
    # directly. The remap block is independent of the provenance
    # path.
    return TranslationPipeline(
        cfg,
        provenance=ProvenanceTracker(
            enabled=False, project_name='executor_hwsd_remap_unit',
        ),
    )


def _make_region() -> Region:
    return Region(
        name='Koutiala', country='Mali', country_iso3='MLI',
        bounds=BoundingBox(
            minx=-5.10, miny=11.90, maxx=-5.00, maxy=12.00,
        ),
    )


# ---------------------------------------------------------------------------
# AC-4 remap pin — the source records by index; the executor must
# remap to actual grid cell IDs.
# ---------------------------------------------------------------------------


class TestRetrieveHWSDIndexToGridCellIdRemap(TestCase):
    """Drives ``_retrieve_hwsd_for_grid`` with a patched
    ``HWSDSource`` and asserts the returned unavailable_cells
    list carries actual grid cell IDs, not enumerate indices."""

    def _run_remap(self, tmp_path, recorded_indices, grid_cells):
        """Build the stub fixtures + run ``_retrieve_hwsd_for_grid``
        with a patched HWSDSource. Returns the (profiles,
        unavailable_cells) tuple the method produced."""
        # Stub bil/mdb files — only need ``Path.exists()`` to be
        # True; the fake HWSDSource never actually opens them.
        bil = tmp_path / "HWSD2.bil"
        mdb = tmp_path / "HWSD2.mdb"
        bil.touch()
        mdb.touch()

        pipe = _make_pipeline_with_paths(bil, mdb)
        grid = _FakeGrid(grid_cells)
        region = _make_region()

        FakeHWSDSource = _make_fake_hwsd_source_class(
            record_indices=recorded_indices,
        )
        # Patch at the import site inside the executor's local
        # ``from prismpy.sources.soil.hwsd import HWSDSource``.
        with patch(
            'prismpy.sources.soil.hwsd.HWSDSource', FakeHWSDSource,
        ):
            result = pipe._retrieve_hwsd_for_grid(grid, region)
        return result

    def test_remap_translates_indices_to_grid_cell_ids(self, tmp_path=None):
        """The remap must translate the source's enumerate-based
        index keys to the executor's grid cell IDs. Source records
        indices [0, 1, 2]; grid cells carry IDs [101, 202, 303];
        the returned list must carry [101, 202, 303]."""
        if tmp_path is None:
            import tempfile, pathlib
            with tempfile.TemporaryDirectory() as td:
                tmp_path = pathlib.Path(td)
                self._assert_remap_translates(tmp_path)
        else:
            self._assert_remap_translates(tmp_path)

    def _assert_remap_translates(self, tmp_path):
        cells = [
            _FakeCell(101, 11.95, -5.05),
            _FakeCell(202, 11.95, -5.00),
            _FakeCell(303, 12.00, -5.05),
        ]
        profiles, unavailable = self._run_remap(
            tmp_path,
            recorded_indices=[0, 1, 2],
            grid_cells=cells,
        )
        # Failure path — no profiles produced.
        self.assertIsNone(
            profiles,
            "Fake source returned success=False; profiles must be "
            "None on the all-miss path.",
        )
        recorded_ids = sorted(e["cell_id"] for e in unavailable)
        self.assertEqual(
            recorded_ids, [101, 202, 303],
            f"AC-4 remap pin: source recorded indices [0, 1, 2] "
            f"but grid cells carry IDs [101, 202, 303]. The remap "
            f"must produce [101, 202, 303]; produced "
            f"{recorded_ids}. A future regression that passes "
            f"the index straight through would surface "
            f"[0, 1, 2] here.",
        )

    def test_remap_preserves_cause_strings(self, tmp_path=None):
        """The remap must carry the cause string forward intact —
        the executor never rewrites the loader's cause."""
        if tmp_path is None:
            import tempfile, pathlib
            with tempfile.TemporaryDirectory() as td:
                tmp_path = pathlib.Path(td)
                self._assert_cause_preserved(tmp_path)
        else:
            self._assert_cause_preserved(tmp_path)

    def _assert_cause_preserved(self, tmp_path):
        cells = [_FakeCell(500, 11.95, -5.05)]
        _, unavailable = self._run_remap(
            tmp_path, recorded_indices=[0], grid_cells=cells,
        )
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(
            unavailable[0]["cause"], "soil_no_hwsd_coverage",
            "AC-4 remap pin: cause string must round-trip "
            "through the remap block unchanged.",
        )

    def test_remap_drops_out_of_range_indices(self, tmp_path=None):
        """An index outside ``[0, len(cell_ids))`` must be dropped
        rather than producing a bogus entry. Defensive guard
        against a malformed source-side recording."""
        if tmp_path is None:
            import tempfile, pathlib
            with tempfile.TemporaryDirectory() as td:
                tmp_path = pathlib.Path(td)
                self._assert_drops_out_of_range(tmp_path)
        else:
            self._assert_drops_out_of_range(tmp_path)

    def _assert_drops_out_of_range(self, tmp_path):
        cells = [
            _FakeCell(11, 11.95, -5.05),
            _FakeCell(22, 11.95, -5.00),
        ]
        # Source records indices [0, 99] — index 99 is out of
        # range (only 2 cells in the grid).
        _, unavailable = self._run_remap(
            tmp_path,
            recorded_indices=[0, 99],
            grid_cells=cells,
        )
        recorded_ids = [e["cell_id"] for e in unavailable]
        self.assertEqual(
            recorded_ids, [11],
            f"AC-4 remap pin: out-of-range index 99 (grid has 2 "
            f"cells, valid range [0, 2)) must be dropped. Index 0 "
            f"must remap to grid cell ID 11. Produced "
            f"{recorded_ids}.",
        )
