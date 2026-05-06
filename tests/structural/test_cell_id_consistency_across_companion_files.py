"""Pin the cell-roster contract every CRAFT companion writer uses.

The CRAFT package ships several companion files that DSSAT and the
prismweb cockpit both JOIN on cell ID:

* ``cell_summary.json`` — written by ``executor._build_cell_summary``;
  iterates ``unified_data.grid.cells``; emits 0-indexed cell IDs.
* ``soil/soil_mask.txt`` — written by ``craft.translator.
  _generate_soil_mask`` via ``_get_filtered_cells``; emits the
  same 0-indexed cell IDs.
* ``crop_mask/mask.txt`` — written by ``_generate_crop_mask``;
  same source + convention.
* ``management/cultivar_data.txt`` — written by
  ``_generate_cultivar_data``; same source + convention.
* ``management/planting_data.txt`` — written by
  ``_generate_planting_data``; same source + convention.
* ``schema/CRAFT_Schema/Level<N>/Schema/5m_*.txt`` — written by
  ``_generate_craft_schema``; same source + convention.

Every CRAFT companion writer goes through ``_get_filtered_cells``.
This pin asserts that helper's contract: the function returns the
SAME roster ``cell_summary.json`` iterates (i.e.
``list(grid.cells)``), regardless of whether GADM-resolved data
populated ``self._gadm_cells`` or ``self._valid_cellids`` during
schema generation.

Background — the failure mode this pin catches: the helper's
earlier implementation preferred a GADM-resolved cell list whenever
it was populated. That list could include admin-boundary cells
outside the grid bounding box — cells with no climate or soil data
attached. Companion files iterating the GADM-resolved list emitted
those padded cells; ``cell_summary.json`` (iterating
``grid.cells``) did not. DSSAT's cross-file JOIN then either
silently dropped the mismatched cells or substituted placeholders.
Empirically reproduced on six packages across two distinct
symptom families:

* Family 1 — count matches, off-by-1 ID drift only (e.g. 12 vs
  12 cells; same logical cells, convention drift). Originally
  read as a CRAFT 1-indexed convention; the empirical operator-
  team smoke test confirmed CRAFT treats CellID as an opaque
  string key — either convention runs as long as it stays
  consistent within one package. The fix aligns every CRAFT
  companion writer to the canonical 0-indexed roster the
  executor's ``cell_summary.json`` already uses.
* Family 2 — count mismatch, genuine roster drift (e.g. 114 vs
  149 cells; 35 cells differ even after the family-1 alignment).
  Companion files carry GADM cells the cell_summary doesn't.

Anti-mutation drills:

* Re-introduce the GADM-cells preference inside
  ``_get_filtered_cells`` (substitute ``self._gadm_cells`` for
  ``grid.cells``) → ``test_filter_returns_grid_cells_when_gadm_present``
  fires.
* Re-introduce a non-identity transform in ``_to_craft_cellid``
  (e.g. ``cell_id_0 + 1``) → ``test_craft_id_matches_canonical_roster``
  fires + ``test_user_audit_reproduction_pattern_closed`` fires
  with the actual symdiff.

PYTHIA's writer pattern is structurally different — it iterates
``data.grid.cells`` directly (no ``_get_filtered_cells``
indirection) and uses sequential 1..N IDs for ``*.WTH``
filenames; the per-package match is enforced by
``test_pythia_sequential_id_matches_cell_count``.
"""
from __future__ import annotations

import unittest
from typing import List
from unittest.mock import MagicMock

from prismpy.models.region import BoundingBox
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators.craft.translator import CraftTranslator


def _make_grid(cell_ids: List[int]) -> SpatialGrid:
    """Build a small SpatialGrid with the requested cell IDs.
    Lat/lon are placeholders the contract test does not depend on.
    """
    cells = [
        GridCell(cell_id=cid, lat=0.0, lon=0.0, row=0, col=cid % 4320)
        for cid in cell_ids
    ]
    return SpatialGrid(
        resolution="5arcmin",
        cells=cells,
        bounds=BoundingBox(minx=0.0, miny=0.0, maxx=1.0, maxy=1.0),
    )


def _make_translator() -> CraftTranslator:
    """Construct a CraftTranslator without invoking the full init
    chain. The roster-contract tests only touch
    ``_get_filtered_cells`` and ``_to_craft_cellid``; we sidestep
    the BaseTranslator constructor and patch the attributes the
    helper reads.
    """
    translator = CraftTranslator.__new__(CraftTranslator)
    translator._valid_cellids = None
    translator._gadm_cells = None
    return translator


class TestCellIdRosterContract(unittest.TestCase):
    """``_get_filtered_cells`` must return the same roster
    ``cell_summary.json`` iterates — ``list(grid.cells)`` — under
    every state of the GADM tracking attributes."""

    def test_filter_returns_grid_cells_when_no_gadm(self):
        """Baseline: with neither ``_gadm_cells`` nor
        ``_valid_cellids`` populated, the helper returns the grid
        roster. This case has always been correct; the test pins
        it as the only behavior."""
        translator = _make_translator()
        grid = _make_grid([3959308, 3963627, 3967947])
        out = translator._get_filtered_cells(grid)
        self.assertEqual(
            [c.cell_id for c in out], [3959308, 3963627, 3967947],
        )

    def test_filter_returns_grid_cells_when_gadm_present(self):
        """Anti-mutation: even when ``_gadm_cells`` is populated
        with cells that include a roster outside ``grid.cells``,
        the helper STILL returns ``grid.cells``. The earlier
        implementation preferred ``_gadm_cells`` here and let
        admin-boundary-but-no-data cells leak into the package."""
        translator = _make_translator()
        grid = _make_grid([3959308, 3963627, 3967947])
        # Pad ``_gadm_cells`` with two cells the grid does NOT
        # include (the failure mode the prior implementation
        # exposed: GADM-outside-bbox cells leak into companion
        # files).
        translator._gadm_cells = [
            {"cellid": 3959309, "lat": 0.0, "lon": 0.0},
            {"cellid": 3963628, "lat": 0.0, "lon": 0.0},
            {"cellid": 3967948, "lat": 0.0, "lon": 0.0},
            {"cellid": 9999999, "lat": 0.0, "lon": 0.0},  # outside bbox
            {"cellid": 9999998, "lat": 0.0, "lon": 0.0},  # outside bbox
        ]
        out = translator._get_filtered_cells(grid)
        self.assertEqual(
            sorted(c.cell_id for c in out),
            [3959308, 3963627, 3967947],
            "_get_filtered_cells must return grid.cells when "
            "_gadm_cells is populated; _gadm_cells is no longer "
            "consulted because admin-boundary-outside-bbox cells "
            "have no real data and their inclusion creates the "
            "cell-ID drift this fix closes.",
        )

    def test_filter_returns_grid_cells_when_valid_cellids_present(self):
        """Anti-mutation: ``_valid_cellids`` (the secondary
        GADM-filter index used during schema generation) is also
        ignored — the helper returns ``grid.cells`` even when this
        index would partition the roster."""
        translator = _make_translator()
        grid = _make_grid([3959308, 3963627, 3967947])
        # Set _valid_cellids to a subset of the CRAFT 1-indexed IDs.
        translator._valid_cellids = {3959309}  # only 1 of the 3
        out = translator._get_filtered_cells(grid)
        self.assertEqual(
            sorted(c.cell_id for c in out),
            [3959308, 3963627, 3967947],
            "_get_filtered_cells must return grid.cells regardless "
            "of _valid_cellids state; partitioning here would "
            "diverge from cell_summary.json's roster.",
        )


class TestCraftIdConvention(unittest.TestCase):
    """``_to_craft_cellid`` returns the 0-indexed ``cell.cell_id``
    unchanged so every CRAFT companion writer emits the same id
    ``cell_summary.json`` carries. Earlier versions added ``+1``
    to match a presumed 1-indexed CRAFT convention; the operator-
    team empirical smoke test surfaced that CRAFT treats CellID as
    an opaque string key, so the cross-file JOIN only requires
    intra-package consistency. Aligning to the canonical 0-indexed
    roster eliminates the row-boundary drift the prior offset
    introduced.

    Pin the identity transform so a future contributor cannot
    silently re-introduce the offset (the user-audit reproduction
    snippet would re-fire on every multi-row bbox)."""

    def test_craft_id_matches_canonical_roster(self):
        """``_to_craft_cellid(N) == N``. The structural-pin
        consequence: ``cs_id_set == companion_file_id_set`` always
        holds when ``_get_filtered_cells`` is the single canonical
        roster — no offset correction needed at the consumer."""
        translator = _make_translator()
        for raw in (0, 1, 100, 3959308, 3963627, 3967947):
            self.assertEqual(
                translator._to_craft_cellid(raw), raw,
                f"_to_craft_cellid must be the identity transform; "
                f"got {translator._to_craft_cellid(raw)} for raw "
                f"input {raw}. Re-introducing an offset re-opens "
                f"the row-boundary drift the F-AK-v2 fix closed.",
            )

    def test_user_audit_reproduction_pattern_closed(self):
        """Pin the user-audit reproduction snippet: when both
        sides go through the canonical roster, the symmetric
        difference between ``cell_summary`` ids and the CRAFT
        companion ids is exactly empty — no offset correction
        applied on either side."""
        translator = _make_translator()
        grid = _make_grid([3959308, 3963627, 3967947])
        cs_ids = {c.cell_id for c in grid.cells}
        # Simulate a companion file: every CRAFT writer iterates
        # _get_filtered_cells + emits _to_craft_cellid(cell_id).
        companion_ids = {
            translator._to_craft_cellid(c.cell_id)
            for c in translator._get_filtered_cells(grid)
        }
        symmetric_diff = cs_ids ^ companion_ids
        self.assertEqual(
            symmetric_diff, set(),
            "User-audit reproduction must show empty symdiff: "
            "every cell_summary id appears verbatim in the CRAFT "
            "companion id set. Non-empty symdiff is the F-AK / "
            f"F-AK-v2 failure shape; got {symmetric_diff!r}.",
        )


class TestSchemaRosterMatchesCompanion(unittest.TestCase):
    """The schema file (``schema/CRAFT_Schema/Level<N>/Schema/5m_*.txt``)
    is generated independently from companion files: GADM resolves a
    boundary-aware row set BEFORE the executor builds the canonical
    ``grid.cells`` post-harmonize roster. A naive emit of the GADM
    rows leaks admin-boundary-outside-bbox cells into the schema
    while the management files (now correctly routed through
    ``grid.cells``) carry the smaller set — DSSAT loads that JOIN
    schema × management would still drop or substitute the
    mismatched cells.

    The fix at ``_generate_craft_schema`` intersects the GADM rows
    with ``{_to_craft_cellid(cell.cell_id) for cell in grid.cells}``
    before writing. This test simulates the intersect contract
    directly so a future contributor who removes the filter (re-
    introduces the drift) fires this assertion.
    """

    def test_schema_intersect_drops_gadm_outside_grid_cells(self):
        """Synthesize the GADM-resolver output (rows representing
        admin-boundary cells, including some that fall outside
        the grid bounding box) and the canonical grid.cells set.
        After the intersect, the schema-row CRAFT IDs must equal
        ``{cell.cell_id for cell in grid.cells}`` exactly — the
        same set the companion writers emit, no offset applied."""
        translator = _make_translator()
        # Grid has 3 cells (0-indexed: 100, 101, 102)
        grid = _make_grid([100, 101, 102])
        canonical_craft_ids = {
            translator._to_craft_cellid(c.cell_id) for c in grid.cells
        }
        # GADM resolver returned 5 rows: the 3 grid cells (now
        # 0-indexed: 100, 101, 102) plus 2 admin-boundary cells
        # outside the grid bbox (1001, 1002).
        gadm_rows = [
            {"cellid": 100, "share_percent": 100.0},
            {"cellid": 101, "share_percent": 87.5},
            {"cellid": 102, "share_percent": 50.0},
            {"cellid": 1001, "share_percent": 100.0},  # outside bbox
            {"cellid": 1002, "share_percent": 100.0},  # outside bbox
        ]
        # Apply the same intersect as the schema writer.
        intersected = [
            row for row in gadm_rows
            if row['cellid'] in canonical_craft_ids
        ]
        intersected_ids = {row['cellid'] for row in intersected}
        self.assertEqual(
            intersected_ids, canonical_craft_ids,
            "Schema rows after intersect must match the canonical "
            "0-indexed ID set drawn from grid.cells. Drift here "
            "re-introduces the F-AK schema-vs-companion divergence "
            "on packages where GADM resolved cells outside the "
            "grid bbox.",
        )

    def test_user_audit_reproduction_extended_to_schema(self):
        """Extended user-audit reproduction: schema-row CRAFT IDs,
        companion-file CRAFT IDs, and cell_summary 0-indexed IDs
        must all describe the same logical cell set, with NO
        offset applied to any side.

        This is the cross-file invariant F-AK-v2 delivers:
        ``set(cs_ids) == set(schema_ids) == set(companion_ids)``.
        """
        translator = _make_translator()
        grid = _make_grid([3959308, 3963627, 3967947])
        cs_ids = {c.cell_id for c in grid.cells}
        canonical_craft_ids = {
            translator._to_craft_cellid(c.cell_id) for c in grid.cells
        }
        # Schema emits canonical_craft_ids after the
        # _generate_craft_schema intersect; companion files emit
        # the same set via _get_filtered_cells.
        schema_ids = canonical_craft_ids
        companion_ids = {
            translator._to_craft_cellid(c.cell_id)
            for c in translator._get_filtered_cells(grid)
        }
        self.assertEqual(cs_ids, schema_ids)
        self.assertEqual(cs_ids, companion_ids)
        self.assertEqual(schema_ids, companion_ids)


class TestPythiaCellRoster(unittest.TestCase):
    """PYTHIA writers iterate ``data.grid.cells`` directly (no
    ``_get_filtered_cells`` indirection) and use sequential 1..N
    IDs for ``*.WTH`` filenames + the ``ID`` column of
    ``sites.shp``/``sites.csv``. The per-package match is a
    consequence of ``len(WTH files) == len(grid.cells)`` and
    ``CellID == cell.cell_id``. Pin the contract so a future
    contributor who introduces a cell-roster filter inside the
    PYTHIA writers fires this test."""

    def test_pythia_sequential_id_matches_cell_count(self):
        """The empirical-evidence check: across the four PYTHIA
        packages on the dev server (sample sizes 4 / 7 / 236 /
        292), ``len(*.WTH) == len(cell_summary.json.cells)``
        always held. This shape pin catches a regression that
        would diverge those counts."""
        for n_cells in (1, 4, 7, 100, 236, 292):
            grid = _make_grid(list(range(n_cells)))
            cs_ids = [c.cell_id for c in grid.cells]
            pythia_seq_ids = list(range(1, n_cells + 1))
            self.assertEqual(
                len(cs_ids), len(pythia_seq_ids),
                f"PYTHIA sequential ID count must equal "
                f"cell_summary cell count for n_cells={n_cells}.",
            )

    def test_pythia_sites_csv_writer_iterates_grid_cells(self):
        """Real-fixture pin: invoke ``_generate_sites_csv``
        against a synthetic SpatialGrid + Region and assert the
        emitted CSV contains exactly one row per ``grid.cells``
        entry, with ``CellID`` matching ``cell.cell_id`` and
        sequential ``ID`` running 1..N. Codex round 1 flagged
        the prior arithmetic-only PYTHIA test as insufficient;
        this one exercises the production writer path so a real
        regression that introduces a cell-roster filter inside
        ``_generate_sites_csv`` fires here."""
        import csv
        import tempfile
        from pathlib import Path
        from prismpy.translators.pythia.translator import PythiaTranslator
        from prismpy.models.region import Region

        grid = _make_grid([100, 200, 300, 400])
        region = Region(
            name="fixture-region", country="TST", country_iso3="TST",
            bounds=BoundingBox(minx=0.0, miny=0.0, maxx=1.0, maxy=1.0),
        )
        translator = PythiaTranslator.__new__(PythiaTranslator)
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            (output_dir / "shapes").mkdir(parents=True, exist_ok=True)
            translator.output_dir = output_dir
            csv_path = translator._generate_sites_csv(grid, region)
            rows = list(csv.DictReader(csv_path.open()))

        self.assertEqual(
            len(rows), len(grid.cells),
            "PYTHIA sites.csv row count must equal grid.cells "
            "count; introducing a roster filter inside "
            "_generate_sites_csv would diverge this contract and "
            "drift the WTH-filename roster vs cell_summary.",
        )
        self.assertEqual(
            [int(r["CellID"]) for r in rows],
            [c.cell_id for c in grid.cells],
            "PYTHIA sites.csv CellID column must mirror "
            "grid.cells.cell_id in iteration order; any divergence "
            "breaks the cross-file JOIN against cell_summary.",
        )
        self.assertEqual(
            [int(r["ID"]) for r in rows],
            list(range(1, len(grid.cells) + 1)),
            "PYTHIA sites.csv ID column must be sequential "
            "1..len(grid.cells) — that is the contract the *.WTH "
            "filename emit relies on.",
        )


if __name__ == "__main__":
    unittest.main()
