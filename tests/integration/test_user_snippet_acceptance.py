"""Per the user-reproduction-snippet acceptance discipline:
the verbatim snippet the user runs against a generated CRAFT
package on the dev server is the canonical Gate B test for
the cell-ID-drift class. Any change to the canonical-source
contract or the per-writer convention must keep the snippet's
output at ``len(cs_ids ^ companion_ids) == 0``.

The user's snippet shape:

::

    cs_ids = {c['id'] for c in json.load(
        (package_dir / 'cell_summary.json').open()
    )['cells']}
    soil_ids = {int(line.split()[0])
                for line in (package_dir / 'soil/soil_mask.txt').read_text().splitlines()[1:]
                if line.strip()}
    crop_ids = {int(line.split()[0])
                for line in (package_dir / 'crop_mask/mask.txt').read_text().splitlines()[1:]
                if line.strip()}
    weather_ids = {int(p.stem) for p in (package_dir / 'weather').glob('*.txt')
                   if p.stem.isdigit()}
    companion_ids = soil_ids | crop_ids | weather_ids
    drift = len(cs_ids ^ companion_ids)
    assert drift == 0

This test simulates the package generation by:

1. Synthesizing a 12-cell SpatialGrid shaped like a multi-row
   bbox (the row-boundary case where the prior ``+1`` offset
   produced visible drift).
2. Writing a ``cell_summary.json`` that mirrors what the
   executor's ``_build_cell_summary`` emits (0-indexed
   ``cell.cell_id`` per row).
3. Writing CRAFT-companion files (soil_mask, crop_mask, weather
   per-cell .txt) by iterating the same canonical roster
   through ``_to_craft_cellid`` — the same code path the real
   writers use.
4. Running the user's snippet against the disk layout.

Anti-mutation drill: revert ``_to_craft_cellid`` to
``cell_id_0 + 1`` → the test fires with the actual xor count
plus the row-boundary diagnostic naming the offending cell ids.

Why a 12-cell 3-row Niamey-shaped fixture: the prior offset
hid in single-cell or single-row probes (a +1 shift on a
single row produces no row-boundary collisions). The drift
empirically surfaces only at multi-row bboxes; the fixture
chooses 4 cols × 3 rows = 12 cells specifically to exercise
the row-boundary code path the user audit hit.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import List

from prismpy.models.region import BoundingBox
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators.craft.translator import CraftTranslator


def _build_niamey_shaped_grid() -> SpatialGrid:
    """12 cells arranged as 4 cols × 3 rows so the cell ids
    span three row boundaries — the empirical reproduction
    shape from the user's Niamey audit. Row stride 4320 mirrors
    the SpatialGrid's 5-arcmin global grid; cell ids are picked
    from the Niamey region so the test reads as familiar to
    a contributor scanning the diff.
    """
    row_stride = 4320  # 5-arcmin global grid columns
    base = 3959304    # bottom-left cell of the Niamey-shaped fixture
    cells: List[GridCell] = []
    for r in range(3):
        for c in range(4):
            cid = base + r * row_stride + c
            cells.append(
                GridCell(cell_id=cid, lat=13.5, lon=2.1, row=r, col=c)
            )
    return SpatialGrid(
        resolution="5arcmin",
        cells=cells,
        bounds=BoundingBox(minx=2.0, miny=13.4, maxx=2.4, maxy=13.7),
    )


def _write_cell_summary(package_dir: Path, grid: SpatialGrid) -> None:
    """Mirror what ``executor._build_cell_summary`` writes:
    a ``cell_summary.json`` whose ``cells`` array carries
    ``id == cell.cell_id`` (0-indexed) for every grid cell."""
    payload = {
        "cells": [
            {"id": c.cell_id, "lat": c.lat, "lon": c.lon}
            for c in grid.cells
        ],
    }
    (package_dir / "cell_summary.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def _write_soil_mask(
    package_dir: Path, translator: CraftTranslator, grid: SpatialGrid,
) -> None:
    """Mirror the CRAFT soil-mask companion writer's per-row
    layout: a header line followed by one row per cell with
    the CRAFT cell id in column 1. Real writer pulls the
    same roster via ``_get_filtered_cells`` + emits
    ``_to_craft_cellid(cell.cell_id)``; the simulation does the
    same so the user's snippet reconciliation is faithful."""
    soil_dir = package_dir / "soil"
    soil_dir.mkdir(parents=True, exist_ok=True)
    lines = ["CellID Soil"]
    for cell in translator._get_filtered_cells(grid):
        cid = translator._to_craft_cellid(cell.cell_id)
        lines.append(f"{cid} HWSD_DEFAULT")
    (soil_dir / "soil_mask.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


def _write_crop_mask(
    package_dir: Path, translator: CraftTranslator, grid: SpatialGrid,
) -> None:
    """Mirror the CRAFT crop-mask companion writer."""
    crop_dir = package_dir / "crop_mask"
    crop_dir.mkdir(parents=True, exist_ok=True)
    lines = ["CellID CropMask"]
    for cell in translator._get_filtered_cells(grid):
        cid = translator._to_craft_cellid(cell.cell_id)
        lines.append(f"{cid} 1")
    (crop_dir / "mask.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


def _write_weather_files(
    package_dir: Path, translator: CraftTranslator, grid: SpatialGrid,
) -> None:
    """Mirror the per-cell weather file writer: one ``.txt``
    per cell, filename = CRAFT cell id."""
    weather_dir = package_dir / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)
    for cell in translator._get_filtered_cells(grid):
        cid = translator._to_craft_cellid(cell.cell_id)
        (weather_dir / f"{cid}.txt").write_text(
            "DATE TMAX TMIN RAIN\n", encoding="utf-8",
        )


def _make_translator() -> CraftTranslator:
    """Sidestep the BaseTranslator constructor — the helpers
    we exercise (``_get_filtered_cells`` / ``_to_craft_cellid``)
    only read the GADM tracking attributes, not the full
    translator state."""
    translator = CraftTranslator.__new__(CraftTranslator)
    translator._valid_cellids = None
    translator._gadm_cells = None
    return translator


class TestUserSnippetAcceptance(unittest.TestCase):
    """The verbatim user-reproduction snippet's output is the
    canonical Gate B test. ``len(cs_ids ^ companion_ids) == 0``
    must hold across the simulated package layout that mirrors
    the on-disk shape the user audited."""

    def test_user_snippet_xor_zero_on_simulated_package(self):
        grid = _build_niamey_shaped_grid()
        translator = _make_translator()
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td)
            _write_cell_summary(pkg, grid)
            _write_soil_mask(pkg, translator, grid)
            _write_crop_mask(pkg, translator, grid)
            _write_weather_files(pkg, translator, grid)

            # The user's verbatim snippet — kept as close to
            # the original as possible so any drift the
            # snippet would surface on the dev server fires
            # the same way here.
            cs_ids = {
                c["id"] for c in json.loads(
                    (pkg / "cell_summary.json").read_text(encoding="utf-8")
                )["cells"]
            }
            soil_ids = {
                int(line.split()[0])
                for line in (pkg / "soil/soil_mask.txt")
                .read_text(encoding="utf-8")
                .splitlines()[1:]
                if line.strip()
            }
            crop_ids = {
                int(line.split()[0])
                for line in (pkg / "crop_mask/mask.txt")
                .read_text(encoding="utf-8")
                .splitlines()[1:]
                if line.strip()
            }
            weather_ids = {
                int(p.stem)
                for p in (pkg / "weather").glob("*.txt")
                if p.stem.isdigit()
            }
            companion_ids = soil_ids | crop_ids | weather_ids
            drift_set = cs_ids ^ companion_ids
            drift = len(drift_set)

        cs_only = sorted(cs_ids - companion_ids)
        companion_only = sorted(companion_ids - cs_ids)
        self.assertEqual(
            drift, 0,
            f"User-snippet drift = {drift}; expected 0. The "
            f"row-boundary diagnostic: cells appearing only in "
            f"cell_summary = {cs_only!r}; cells appearing only "
            f"in CRAFT companions = {companion_only!r}. A "
            f"non-empty diff with paired (start, end+1) ids per "
            f"row is the +1-offset reproduction shape. The fix "
            f"is to keep ``_to_craft_cellid`` as the identity "
            f"transform.",
        )


if __name__ == "__main__":
    unittest.main()
