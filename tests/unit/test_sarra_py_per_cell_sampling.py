"""V2-22c-PRE.2.1 (D17 amortized-open binding) — structural test.

This is **the load-bearing test for PRE.2** per the contract R4 risk
note. The amortized-open pattern is the difference between ~2.2s and
~3.5h for a 5k-cell SARRA-Py run; a refactor that swaps the loop
nesting (cells outer, files inner) opens each file N times and
balloons the cost by ~5000×.

Two layers of verification:

1. **Runtime call-count check (evaluator §12.3)**: mock
   ``rasterio.open`` and assert
   ``mock_rasterio_open.call_count == len(sample_files)``
   PARAMETRIC — covers all D4 scales (1k MUST, 5k STRETCH).
   Literal-10 binding would only catch the default fixture; a
   refactor that wraps the file-open in a per-cell helper would
   silently slip through. The parametric form is the contract's
   §12.3 binding.

2. **AST loop-nesting check**: walk the source AST of the
   ``sample_sarra_py_per_cell`` function and assert the OUTER
   ``for`` iterates ``sample_files`` and the INNER ``for``
   iterates ``grid_cells`` (or the symbol the caller supplies).
   AST verification catches the case where a refactor
   superficially preserves call_count but inverts the nesting via
   ``itertools.product`` or other obfuscation.

Negative coverage by symmetry: if either invariant regresses, the
respective test fails — operators don't have to wait for a 5k-run
GateB smoke to discover the slowdown.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from prismpy.validators.post_translate import (
    SARRA_PY_VAR_MAPPING,
    sample_sarra_py_per_cell,
)


def _make_cells(n):
    """Build N synthetic GridCell-like objects spread across a
    small bbox in the Sahel (lon ~ 0, lat ~ 14)."""
    return [
        SimpleNamespace(
            cell_id=i,
            lat=14.0 + (i * 0.01),
            lon=0.0 + (i * 0.01),
        )
        for i in range(n)
    ]


def _seed_var_subdirs(platform_dir: Path, *, n_files: int):
    """Create the SARRA-Py output tree shape the helper expects:
    ``platform_dir/data/climate/<var_subdir>/*.tif``. Files are
    empty placeholders — the test mocks ``rasterio.open`` so the
    actual content never matters."""
    climate_dir = platform_dir / "data" / "climate"
    for subdir_name in SARRA_PY_VAR_MAPPING:
        var_dir = climate_dir / subdir_name
        var_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_files):
            (var_dir / f"day_{i:03d}.tif").touch()


def _fake_rasterio_open_factory():
    """Build a context-manager mock for ``rasterio.open`` that
    yields a synthetic dataset whose ``read`` always returns a
    1×1 array of value 25.0. The transform inverts to
    pixel-coords (1, 1) for any input, so every cell read lands
    inside (width, height) = (10, 10).
    """
    import numpy as np

    fake_dataset = MagicMock()
    fake_dataset.width = 10
    fake_dataset.height = 10
    fake_dataset.nodata = -9999.0
    fake_dataset.read.return_value = np.array([[25.0]], dtype=float)

    # Affine inverse — `~transform * (lon, lat)` returns (col, row).
    # The MagicMock's __invert__ returns another MagicMock; calling
    # that with (lon, lat) returns a 2-tuple. The real `Affine`
    # behaviour is irrelevant for this test — we only need a 2-tuple.
    fake_inverse = MagicMock(return_value=(1.0, 1.0))
    fake_dataset.transform = MagicMock()
    fake_dataset.transform.__invert__ = MagicMock(return_value=fake_inverse)

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=fake_dataset)
    cm.__exit__ = MagicMock(return_value=False)

    fake_open = MagicMock(return_value=cm)
    return fake_open, fake_dataset


class TestRasterioOpenCallCount:
    """Evaluator §12.3 — call_count is parametric, NOT literal 10."""

    def test_open_called_once_per_sample_file(self, tmp_path):
        """For 4 mapped variables × 5 files each = 20 sample files
        total, ``rasterio.open`` must be called exactly 20 times,
        regardless of the cell count. This is the amortized-open
        binding."""
        _seed_var_subdirs(tmp_path, n_files=5)
        cells = _make_cells(100)

        fake_open, fake_dataset = _fake_rasterio_open_factory()
        with patch("rasterio.open", fake_open), \
                patch("rasterio.windows.Window", MagicMock):
            sample_sarra_py_per_cell(tmp_path, cells)

        # 4 variables × 5 files (≤ 10 → full list) = 20 opens.
        # Parametric — derived from the seeded fixture, NOT the
        # literal `10`. If a refactor caps the sample count or
        # adds a fifth mapped variable, this assertion follows.
        n_vars = len(SARRA_PY_VAR_MAPPING)
        expected = n_vars * 5
        assert fake_open.call_count == expected, (
            f"D17 amortized-open binding regressed: opened "
            f"{fake_open.call_count} times for {n_vars} vars × "
            f"5 files = {expected} expected. Reverse nesting would "
            f"give {expected * len(cells)} opens at this cell count."
        )

    def test_open_count_independent_of_cell_count(self, tmp_path):
        """Same call_count for 10 cells vs 1000 cells — the
        amortized invariant is that file-opens scale with sample
        files, not with cells."""
        _seed_var_subdirs(tmp_path, n_files=3)

        # Run with 10 cells.
        fake_open_a, _ = _fake_rasterio_open_factory()
        with patch("rasterio.open", fake_open_a), \
                patch("rasterio.windows.Window", MagicMock):
            sample_sarra_py_per_cell(tmp_path, _make_cells(10))

        # Run with 1000 cells.
        fake_open_b, _ = _fake_rasterio_open_factory()
        with patch("rasterio.open", fake_open_b), \
                patch("rasterio.windows.Window", MagicMock):
            sample_sarra_py_per_cell(tmp_path, _make_cells(1000))

        assert fake_open_a.call_count == fake_open_b.call_count, (
            f"open count varied with cell count "
            f"({fake_open_a.call_count} vs {fake_open_b.call_count}); "
            "D17 amortized-open binding regressed — refactor likely "
            "swapped the loop nesting."
        )


class TestLoopNestingASTBinding:
    """Evaluator §12.3 binding — AST walk asserts OUTER loop
    iterates ``sample_files`` and INNER loop iterates the cell
    iterable. Catches the obfuscated case where call_count holds
    but the structural pattern regressed (e.g., a refactor that
    uses ``itertools.product`` with files-first-cells-second
    semantics, breaking the amortized-open intent)."""

    def test_outer_loop_iterates_sample_files(self):
        source = textwrap.dedent(inspect.getsource(sample_sarra_py_per_cell))
        tree = ast.parse(source)
        # Find the outermost `for X in ...:` loops inside the
        # function body that walk SARRA_PY_VAR_MAPPING items.
        # The structure we expect (per the helper's design):
        #
        #   for subdir_name, (var, op, operand) in
        #       SARRA_PY_VAR_MAPPING.items():
        #     ...
        #     for tif_path in sample_files:    ← OUTER (load-bearing)
        #       with rasterio.open(...) as src:
        #         for cell in grid_cells:      ← INNER
        #           ...
        sample_loop = None
        cell_loop = None
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                target = ast.unparse(node.target)
                iter_ = ast.unparse(node.iter)
                if target == "tif_path" and "sample_files" in iter_:
                    sample_loop = node
                elif target == "cell" and "grid_cells" in iter_:
                    cell_loop = node

        assert sample_loop is not None, (
            "AST walk could not find `for tif_path in sample_files:` "
            "outer loop in sample_sarra_py_per_cell — has the "
            "function been renamed or restructured?"
        )
        assert cell_loop is not None, (
            "AST walk could not find `for cell in grid_cells:` inner "
            "loop in sample_sarra_py_per_cell."
        )

        # The cell loop MUST be a descendant of the sample-files
        # loop (i.e., nested inside) — that's the amortized-open
        # invariant.
        sample_descendants = set()
        for node in ast.walk(sample_loop):
            sample_descendants.add(id(node))

        assert id(cell_loop) in sample_descendants, (
            "D17 amortized-open binding REGRESSED: the `for cell` "
            "loop is NOT nested inside the `for tif_path` loop. "
            "Refactor likely swapped the nesting order (cells outer, "
            "files inner) — that opens each file N times and "
            "balloons the cost from ~2.2s to ~3.5h on a 5k-cell run."
        )


class TestPerCellResultStructure:
    """Sanity checks on the helper's return shape — the cockpit
    consumes ``per_cell[cell_id][var]`` as a list of floats. PRE.2.3
    will derive ``has_climate`` from this, and PRE.2.4 will compute
    per-cell value-range checks from it."""

    def test_returns_dict_keyed_by_cell_id(self, tmp_path):
        _seed_var_subdirs(tmp_path, n_files=3)
        cells = _make_cells(5)
        fake_open, _ = _fake_rasterio_open_factory()
        with patch("rasterio.open", fake_open), \
                patch("rasterio.windows.Window", MagicMock):
            result = sample_sarra_py_per_cell(tmp_path, cells)

        assert isinstance(result, dict)
        # Every cell that the mock-read landed inside (1, 1) has
        # values for every variable; the fixture's transform makes
        # all cells valid.
        for cid in result:
            assert isinstance(cid, int)
            assert isinstance(result[cid], dict)

    def test_returns_canonical_var_keys(self, tmp_path):
        """PRE.2.2 binding — the keys are the canonical
        SARRA_PY_VAR_MAPPING values (`rain` / `tmax` / `tmin` /
        `srad`), NOT the raw subdir names."""
        _seed_var_subdirs(tmp_path, n_files=2)
        cells = _make_cells(3)
        fake_open, _ = _fake_rasterio_open_factory()
        with patch("rasterio.open", fake_open), \
                patch("rasterio.windows.Window", MagicMock):
            result = sample_sarra_py_per_cell(tmp_path, cells)

        expected_vars = {
            v[0] for v in SARRA_PY_VAR_MAPPING.values()
        }  # {rain, tmax, tmin, srad}
        for cid, var_bucket in result.items():
            assert set(var_bucket.keys()).issubset(expected_vars), (
                f"cell {cid} bucket has unexpected keys: "
                f"{set(var_bucket.keys()) - expected_vars}"
            )

    def test_empty_when_no_climate_dir(self, tmp_path):
        """No ``data/climate/`` tree → empty dict (no crash)."""
        cells = _make_cells(3)
        fake_open, _ = _fake_rasterio_open_factory()
        with patch("rasterio.open", fake_open), \
                patch("rasterio.windows.Window", MagicMock):
            result = sample_sarra_py_per_cell(tmp_path, cells)
        assert result == {}


class TestCellSummaryHasClimateFromSampledValues:
    """V2-22c-PRE.2.3 (D14) — cell_summary's ``has_climate`` derives
    from sampled-values presence for SARRA-Py runs, NOT from
    ``unified_data.climate`` dict structure (which is path-dict
    shaped for SARRA-Py and would yield universal-fail without
    this fix per V2-22c contract §1.1)."""

    def test_has_climate_true_when_sampled_values_present(self):
        from prismpy.models.region import Region, BoundingBox
        from prismpy.models.spatial import GridCell, SpatialGrid
        from prismpy.pipeline.executor import TranslationPipeline
        from prismpy.translators.base import UnifiedData

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        cells = [GridCell(cell_id=0, lat=14.0, lon=0.0,
                          row=0, col=0, resolution="5arcmin")]
        grid = SpatialGrid(
            bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15),
            resolution="5arcmin", cells=cells,
        )
        unified = UnifiedData(
            region=Region(name="t", country="t", country_iso3="TST",
                          bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15)),
            grid=grid,
            climate={"rainfall_dir": "/tmp/x"},  # path-dict shape
            soil={},
        )
        sarra_per_cell = {
            0: {"tmax": [25.0, 26.0, 27.0], "tmin": [10.0, 11.0]},
        }
        out = pipeline._build_cell_summary(
            unified, sarra_climate_per_cell=sarra_per_cell,
        )
        cell = out["cells"][0]
        assert cell["has_climate"] is True
        assert cell["tmax_range"] == [25.0, 27.0]
        assert cell["tmin_range"] == [10.0, 11.0]
        # V2-22c-PRE codex Gate B P2 #2 — `n_days` is the per-cell
        # day count derived from the longest variable's sample list,
        # NOT a sum across variables. tmax has 3 samples + tmin has
        # 2 → expected n_days = max(3, 2) = 3 (pre-fix this was
        # 5 = 3 + 2).
        assert cell["n_days"] == 3, (
            f"n_days inflation regression: got {cell['n_days']}, "
            "expected max-across-variables = 3."
        )

    def test_has_climate_false_when_no_sampled_values_for_cell(self):
        from prismpy.models.region import Region, BoundingBox
        from prismpy.models.spatial import GridCell, SpatialGrid
        from prismpy.pipeline.executor import TranslationPipeline
        from prismpy.translators.base import UnifiedData

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        cells = [GridCell(cell_id=0, lat=14.0, lon=0.0,
                          row=0, col=0, resolution="5arcmin")]
        grid = SpatialGrid(
            bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15),
            resolution="5arcmin", cells=cells,
        )
        unified = UnifiedData(
            region=Region(name="t", country="t", country_iso3="TST",
                          bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15)),
            grid=grid,
            climate={"rainfall_dir": "/tmp/x"},
            soil={},
        )
        # Empty sampler result for this cell.
        out = pipeline._build_cell_summary(
            unified, sarra_climate_per_cell={},
        )
        assert out["cells"][0]["has_climate"] is False


class TestSarraPyValueRangeSynthesis:
    """V2-22c-PRE.2.4 (D14) — per-cell value-range checks for SARRA-Py
    runs. The validator's file-based-climate branch consumes the
    per-cell sampled values and emits ``value_range_<var>`` checks
    with `affected_cells` lists, identical to the
    CRAFT/PYTHIA/ACEA shape."""

    def _make_unified(self, *, n_cells=3):
        from prismpy.models.region import Region, BoundingBox
        from prismpy.models.spatial import GridCell, SpatialGrid
        from prismpy.translators.base import UnifiedData
        cells = [
            GridCell(cell_id=i, lat=14.0, lon=0.0 + i * 0.01,
                     row=0, col=i, resolution="5arcmin")
            for i in range(n_cells)
        ]
        return UnifiedData(
            region=Region(name="t", country="t", country_iso3="TST",
                          bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15)),
            grid=SpatialGrid(
                bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15),
                resolution="5arcmin", cells=cells,
            ),
            # path-dict shape triggers _is_file_based_climate branch
            climate={"rainfall_dir": "/tmp/x"},
            soil={},
        )

    def test_in_range_sampled_values_yield_pass(self):
        from prismpy.validators.scientific import _check_value_ranges
        unified = self._make_unified()
        sarra_per_cell = {
            0: {"tmax": [25.0, 26.0]},
            1: {"tmax": [27.0, 28.0]},
            2: {"tmax": [29.0, 30.0]},
        }
        out = _check_value_ranges(
            unified, sarra_climate_per_cell=sarra_per_cell,
        )
        tmax_check = next(c for c in out if c["check"] == "value_range_tmax")
        # All in [-50, 60] → no out-of-range; result is `pass` per
        # the standard climate-range check rule.
        assert tmax_check["result"] == "pass"
        assert tmax_check["details"]["out_of_range_count"] == 0
        assert tmax_check["details"]["affected_cells"] == []

    def test_out_of_range_sampled_values_populate_affected_cells(self):
        from prismpy.validators.scientific import _check_value_ranges
        unified = self._make_unified()
        # CLIMATE_RANGES["tmax"] = (-50, 60). 70.0 violates upper bound.
        sarra_per_cell = {
            0: {"tmax": [25.0]},   # in range
            1: {"tmax": [70.0]},   # OUT (> 60)
            2: {"tmax": [-60.0]},  # OUT (< -50)
        }
        out = _check_value_ranges(
            unified, sarra_climate_per_cell=sarra_per_cell,
        )
        tmax_check = next(c for c in out if c["check"] == "value_range_tmax")
        assert tmax_check["details"]["out_of_range_count"] == 2
        assert set(tmax_check["details"]["affected_cells"]) == {1, 2}

    def test_no_sampled_values_keeps_delegated_info_only(self):
        """Backward-compat — without `sarra_climate_per_cell`, the
        file-based branch keeps emitting only the delegated info
        record (no per-cell value-range synthesis)."""
        from prismpy.validators.scientific import _check_value_ranges
        unified = self._make_unified()
        out = _check_value_ranges(unified)  # no per-cell arg
        # The delegated info record is present...
        info_records = [c for c in out if c["check"] == "value_range_climate"]
        assert len(info_records) == 1
        assert info_records[0]["result"] == "info"
        # ...and there are NO per-cell value_range_<var> checks.
        per_var = [c for c in out if c["check"].startswith("value_range_t")]
        assert per_var == []

    def test_multiple_vars_per_cell_each_emit_separate_check(self):
        """4 mapped vars (rain, tmax, tmin, srad) — each gets its
        own check in the validator output. Note SARRA-Py-side
        `rain` translates to the canonical `precip` check_id so
        the cockpit drill-down treats SARRA-Py and the in-memory
        platforms identically."""
        from prismpy.validators.scientific import _check_value_ranges
        unified = self._make_unified(n_cells=1)
        sarra_per_cell = {
            0: {"tmax": [25.0], "tmin": [10.0],
                "rain": [5.0], "srad": [20.0]},
        }
        out = _check_value_ranges(
            unified, sarra_climate_per_cell=sarra_per_cell,
        )
        check_ids = {c["check"] for c in out}
        for canonical_var in ("tmax", "tmin", "precip", "srad"):
            assert f"value_range_{canonical_var}" in check_ids, (
                f"value_range_{canonical_var} missing from SARRA-Py "
                "per-cell synthesis"
            )


class TestSarraPyClimateSourceProvenance:
    """V2-22c-PRE codex Gate B P2 #1 — every SARRA-Py cell with
    `has_climate=True` must carry a populated `sources.climate`
    block. The pre-fix path-dict shape of `unified_data.climate`
    (`{rainfall_dir, agera5_dir}`) caused `climate.get(cid)` to
    always return None, leaving SARRA-Py cells with no climate
    provenance even when the sampled-values branch marked them
    `has_climate=True`. Dr. Kofi's audit trail in the cockpit
    drawer would have rendered "Climate: unknown" for every
    SARRA-Py cell."""

    def test_sarra_py_cell_has_climate_source_block(self):
        from prismpy.models.region import Region, BoundingBox
        from prismpy.models.spatial import GridCell, SpatialGrid
        from prismpy.pipeline.executor import TranslationPipeline
        from prismpy.translators.base import UnifiedData

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        cells = [GridCell(cell_id=0, lat=14.0, lon=0.0,
                          row=0, col=0, resolution="5arcmin")]
        unified = UnifiedData(
            region=Region(
                name="t", country="t", country_iso3="TST",
                bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15),
            ),
            grid=SpatialGrid(
                bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15),
                resolution="5arcmin", cells=cells,
            ),
            climate={"rainfall_dir": "/tmp/x"},  # path-dict shape
            soil={},
        )
        out = pipeline._build_cell_summary(
            unified,
            sarra_climate_per_cell={0: {"tmax": [25.0], "rain": [5.0]}},
        )
        cell = out["cells"][0]
        # has_climate must be True so we know the SARRA-Py branch fired.
        assert cell["has_climate"] is True
        # sources.climate present with the four canonical fields.
        assert "sources" in cell
        assert "climate" in cell["sources"]
        sb = cell["sources"]["climate"]
        # Composite name reflects the SARRA-Py two-source pattern
        # (TAMSAT for rainfall + AgERA5 for temperature/solar).
        assert sb["name"] == "TAMSAT + AgERA5"
        assert sb["cascade_rank"] == 1
        assert sb["fallback_attempts"] == []

    def test_sarra_py_cell_without_climate_has_no_climate_source(self):
        """Backstop — when `has_climate=False` (no sampled values),
        sources.climate is NOT spuriously populated. Keeps the
        D37 elision rule intact."""
        from prismpy.models.region import Region, BoundingBox
        from prismpy.models.spatial import GridCell, SpatialGrid
        from prismpy.pipeline.executor import TranslationPipeline
        from prismpy.translators.base import UnifiedData

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        cells = [GridCell(cell_id=0, lat=14.0, lon=0.0,
                          row=0, col=0, resolution="5arcmin")]
        unified = UnifiedData(
            region=Region(
                name="t", country="t", country_iso3="TST",
                bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15),
            ),
            grid=SpatialGrid(
                bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15),
                resolution="5arcmin", cells=cells,
            ),
            climate={"rainfall_dir": "/tmp/x"},
            soil={},
        )
        # Empty sampler dict → no sampled values → has_climate=False.
        out = pipeline._build_cell_summary(
            unified, sarra_climate_per_cell={},
        )
        cell = out["cells"][0]
        assert cell["has_climate"] is False
        # sources block either absent or has no climate sub-block —
        # both are acceptable per D37 elision (no soil + no climate
        # → entire sources block elided).
        if "sources" in cell:
            assert "climate" not in cell["sources"]


class TestSarraPyNDaysNotInflated:
    """V2-22c-PRE codex Gate B P2 #2 — `n_days` is per-cell day
    count, not a sum across variables. The pre-fix
    ``sum(len(v) for v in bucket.values())`` form would report
    `n_days=40` for a 10-file × 4-var sample, inflating the
    cockpit's temporal-coverage display by 4×."""

    def test_n_days_uses_max_across_variables(self):
        from prismpy.models.region import Region, BoundingBox
        from prismpy.models.spatial import GridCell, SpatialGrid
        from prismpy.pipeline.executor import TranslationPipeline
        from prismpy.translators.base import UnifiedData

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        cells = [GridCell(cell_id=0, lat=14.0, lon=0.0,
                          row=0, col=0, resolution="5arcmin")]
        unified = UnifiedData(
            region=Region(
                name="t", country="t", country_iso3="TST",
                bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15),
            ),
            grid=SpatialGrid(
                bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15),
                resolution="5arcmin", cells=cells,
            ),
            climate={"rainfall_dir": "/tmp/x"},
            soil={},
        )
        # 10 sampled days per variable across 4 variables.
        # Pre-fix: n_days = 40 (sum). Post-fix: n_days = 10 (max).
        sarra_per_cell = {
            0: {
                "rain": [5.0] * 10,
                "tmax": [25.0] * 10,
                "tmin": [10.0] * 10,
                "srad": [20.0] * 10,
            },
        }
        out = pipeline._build_cell_summary(
            unified, sarra_climate_per_cell=sarra_per_cell,
        )
        cell = out["cells"][0]
        assert cell["n_days"] == 10, (
            f"n_days inflation regression — got {cell['n_days']}, "
            "expected 10 (max across variables). The pre-fix form "
            "would have summed 4 vars × 10 samples = 40."
        )

    def test_n_days_handles_uneven_variable_lengths(self):
        """When variables have different sample lengths (one
        partially missing), the canonical day count is the
        longest list's length, NOT the sum."""
        from prismpy.models.region import Region, BoundingBox
        from prismpy.models.spatial import GridCell, SpatialGrid
        from prismpy.pipeline.executor import TranslationPipeline
        from prismpy.translators.base import UnifiedData

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        cells = [GridCell(cell_id=0, lat=14.0, lon=0.0,
                          row=0, col=0, resolution="5arcmin")]
        unified = UnifiedData(
            region=Region(
                name="t", country="t", country_iso3="TST",
                bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15),
            ),
            grid=SpatialGrid(
                bounds=BoundingBox(minx=0, miny=14, maxx=1, maxy=15),
                resolution="5arcmin", cells=cells,
            ),
            climate={"rainfall_dir": "/tmp/x"},
            soil={},
        )
        # tmax has 8 days; tmin has 10; rain has 10; srad has 7.
        # Max = 10 (the day count for the most-complete variable).
        sarra_per_cell = {
            0: {
                "rain": [5.0] * 10,
                "tmax": [25.0] * 8,
                "tmin": [10.0] * 10,
                "srad": [20.0] * 7,
            },
        }
        out = pipeline._build_cell_summary(
            unified, sarra_climate_per_cell=sarra_per_cell,
        )
        cell = out["cells"][0]
        assert cell["n_days"] == 10


class TestFloorRejectsWestNorthEdgeCells:
    """Climate-coverage honesty (fix B) — the per-cell sampler converts
    world→pixel coords with ``math.floor()``, NOT ``int()``.

    ``int()`` truncates toward zero, so a cell whose center sits just WEST
    or NORTH of the raster origin (negative ``col_f``/``row_f``, e.g. -0.42)
    maps to pixel 0 and silently mis-samples an edge pixel ~0.09° away,
    over-reporting ``has_climate`` — the real Mopti 6/9-reported vs honest-3/9
    defect. ``floor(-0.42) = -1`` is correctly rejected by the existing
    ``0 <= idx < dim`` bounds check. ``floor`` and ``int`` agree for all
    non-negative coords, so no NEW in-bounds cell is dropped.

    These pin the fix with a REAL affine (not the MagicMock used elsewhere)
    so the floor-vs-int boundary is exercised by actual world→pixel math.
    """

    def _fake_open_real_affine(self):
        """1×1 raster, 1° pixel, origin (lon=0, lat=15). The inverse affine
        gives ``~transform * (lon, lat) == (lon, 15 - lat)``, so only
        ``lon ∈ [0, 1)`` and ``lat ∈ (14, 15]`` land inside pixel (0, 0)."""
        from affine import Affine
        import numpy as np

        fake_dataset = MagicMock()
        fake_dataset.width = 1
        fake_dataset.height = 1
        fake_dataset.nodata = -9999.0
        fake_dataset.read.return_value = np.array([[25.0]], dtype=float)
        fake_dataset.transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 15.0)

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=fake_dataset)
        cm.__exit__ = MagicMock(return_value=False)
        return MagicMock(return_value=cm)

    def test_west_of_origin_cell_rejected_not_missampled(self, tmp_path):
        _seed_var_subdirs(tmp_path, n_files=1)
        # in_bounds: col_f=0.5, row_f=0.5 → floor (0,0) → IN.
        # west_edge: col_f=-0.42 → floor=-1 → OUT (int(-0.42)=0 would be IN).
        in_bounds = SimpleNamespace(cell_id=1, lon=0.5, lat=14.5)
        west_edge = SimpleNamespace(cell_id=2, lon=-0.42, lat=14.5)
        fake_open = self._fake_open_real_affine()
        with patch("rasterio.open", fake_open):
            result = sample_sarra_py_per_cell(tmp_path, [in_bounds, west_edge])
        assert 1 in result, "in-bounds cell must sample climate"
        assert 2 not in result, (
            "west-of-origin cell (col_f=-0.42) must be REJECTED by floor(); "
            "int() truncates to pixel 0 and over-reports has_climate"
        )

    def test_north_of_origin_cell_rejected(self, tmp_path):
        _seed_var_subdirs(tmp_path, n_files=1)
        # north_edge: lat=15.3 → row_f = 15 - 15.3 = -0.3 → floor=-1 → OUT.
        in_bounds = SimpleNamespace(cell_id=1, lon=0.5, lat=14.5)
        north_edge = SimpleNamespace(cell_id=3, lon=0.5, lat=15.3)
        fake_open = self._fake_open_real_affine()
        with patch("rasterio.open", fake_open):
            result = sample_sarra_py_per_cell(tmp_path, [in_bounds, north_edge])
        assert 1 in result
        assert 3 not in result, (
            "north-of-origin cell (row_f=-0.3) must be REJECTED by floor()"
        )

    def test_non_negative_coords_unaffected_by_floor(self, tmp_path):
        """floor and int agree for non-negative coords — an in-bounds cell
        still samples (no regression to the honest in-extent cells)."""
        _seed_var_subdirs(tmp_path, n_files=1)
        in_bounds = SimpleNamespace(cell_id=7, lon=0.5, lat=14.5)
        fake_open = self._fake_open_real_affine()
        with patch("rasterio.open", fake_open):
            result = sample_sarra_py_per_cell(tmp_path, [in_bounds])
        assert 7 in result and result[7], "non-negative-coord cell must sample"
