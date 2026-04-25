"""V2-22c-PRE.3 — `exclude_cells` field + `SpatialGrid` factory
filter + executor thread-through.

The cockpit's bulk-fix Exclude class (D8) submits a new
PipelineRun derived from the prior config with
`region.exclude_cells = [<user-selected cell ids>]`. The grid
construction drops those cells before any translator iterates,
so excluded cells never participate in the run — they don't
appear in `unified_data.grid.cells`, they don't get climate
fetched, they don't get soil sampled.

Coverage:

- PRE.3.1: schema validation (Pydantic) — positive int list
  validates, default empty, negative / string / bool rejected.
- PRE.3.2: factory filter — without `exclude_cells` baseline
  unchanged; with `exclude_cells=[X]`, output has no cell with
  id == X. Applied INSIDE the inner loop (no full-grid
  construction then filter).
- PRE.3.3: executor thread-through — `RegionConfig.exclude_cells`
  propagates to `SpatialGrid.from_bounds` via the factory call
  site at `executor.py:2078`. Verified via AST walk on the call
  site (full-pipeline e2e is Gate B integration smoke).
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from prismpy.config.schema import RegionConfig
from prismpy.models.region import BoundingBox
from prismpy.models.spatial import SpatialGrid


# Sahel-ish bbox: ~12.5 cells wide × 12.5 cells tall at 5arcmin.
SMALL_BBOX = BoundingBox(minx=0.0, miny=14.0, maxx=1.0, maxy=15.0)


class TestRegionConfigExcludeCells:
    """V2-22c-PRE.3.1 — Pydantic schema validation."""

    def _make_config_kwargs(self, **overrides):
        # BoundaryConfig requires `gadm_filter_value` when
        # source='gadm' (the default). Pass a valid boundary so
        # the RegionConfig validator can focus on exclude_cells.
        base = {
            "name": "Koutiala", "country": "Mali", "country_iso3": "MLI",
            "boundary": {
                "source": "gadm",
                "gadm_level": 2,
                "gadm_filter_field": "NAME_2",
                "gadm_filter_value": "Koutiala",
            },
        }
        base.update(overrides)
        return base

    def test_default_is_empty_list(self):
        cfg = RegionConfig(**self._make_config_kwargs())
        assert cfg.exclude_cells == []

    def test_positive_int_list_validates(self):
        cfg = RegionConfig(
            **self._make_config_kwargs(exclude_cells=[3, 7, 12]),
        )
        assert cfg.exclude_cells == [3, 7, 12]

    def test_zero_is_accepted(self):
        """cell_id=0 is the first cell, valid per
        SpatialGrid.compute_id contract."""
        cfg = RegionConfig(
            **self._make_config_kwargs(exclude_cells=[0]),
        )
        assert cfg.exclude_cells == [0]

    def test_negative_int_rejected(self):
        with pytest.raises(ValidationError):
            RegionConfig(
                **self._make_config_kwargs(exclude_cells=[-1]),
            )

    def test_string_entry_rejected(self):
        with pytest.raises(ValidationError):
            RegionConfig(
                **self._make_config_kwargs(exclude_cells=["3"]),
            )

    def test_bool_entry_rejected(self):
        """Python bools are subclasses of int — explicit reject so
        True/False can't accidentally slip through."""
        with pytest.raises(ValidationError):
            RegionConfig(
                **self._make_config_kwargs(exclude_cells=[True]),
            )

    def test_omitting_field_succeeds_backward_compat(self):
        """RegionConfig payload without `exclude_cells` validates
        cleanly (existing fixtures don't carry the new field)."""
        cfg = RegionConfig.model_validate({
            "name": "Koutiala", "country": "Mali", "country_iso3": "MLI",
            "boundary": {
                "source": "gadm", "gadm_level": 2,
                "gadm_filter_field": "NAME_2",
                "gadm_filter_value": "Koutiala",
            },
        })
        assert cfg.exclude_cells == []


class TestSpatialGridFromBoundsFilter:
    """V2-22c-PRE.3.2 — `SpatialGrid.from_bounds` accepts an
    `exclude_cells` parameter and filters at construction time."""

    def test_no_filter_yields_baseline_cell_count(self):
        baseline = SpatialGrid.from_bounds(SMALL_BBOX, resolution="5arcmin")
        assert baseline.n_cells > 0
        baseline_ids = {c.cell_id for c in baseline.cells}
        # Sanity: the bbox covered enough cells to give us a
        # meaningful exclusion test below.
        assert len(baseline_ids) >= 5

    def test_excluded_cells_removed_from_output(self):
        """Pick a known cell from the baseline output and exclude
        it; grid must not contain that cell_id."""
        baseline = SpatialGrid.from_bounds(SMALL_BBOX, resolution="5arcmin")
        baseline_ids = sorted({c.cell_id for c in baseline.cells})
        target = baseline_ids[len(baseline_ids) // 2]   # middle cell
        filtered = SpatialGrid.from_bounds(
            SMALL_BBOX, resolution="5arcmin", exclude_cells=[target],
        )
        filtered_ids = {c.cell_id for c in filtered.cells}
        assert target not in filtered_ids
        assert len(filtered_ids) == len(baseline_ids) - 1

    def test_multiple_excludes_all_removed(self):
        baseline = SpatialGrid.from_bounds(SMALL_BBOX, resolution="5arcmin")
        baseline_ids = sorted({c.cell_id for c in baseline.cells})
        targets = [baseline_ids[0], baseline_ids[-1], baseline_ids[2]]
        filtered = SpatialGrid.from_bounds(
            SMALL_BBOX, resolution="5arcmin", exclude_cells=targets,
        )
        filtered_ids = {c.cell_id for c in filtered.cells}
        for t in targets:
            assert t not in filtered_ids
        assert len(filtered_ids) == len(baseline_ids) - 3

    def test_exclude_outside_bbox_is_noop(self):
        """An exclude cell_id that isn't inside the bbox is silently
        a no-op (the cell never gets constructed in the first place
        — set membership check just doesn't fire)."""
        baseline = SpatialGrid.from_bounds(SMALL_BBOX, resolution="5arcmin")
        # Pick a cell ID much higher than anything in the bbox.
        out_of_bbox = max(c.cell_id for c in baseline.cells) + 100000
        filtered = SpatialGrid.from_bounds(
            SMALL_BBOX, resolution="5arcmin",
            exclude_cells=[out_of_bbox],
        )
        assert filtered.n_cells == baseline.n_cells

    def test_none_and_empty_list_equivalent(self):
        """`exclude_cells=None` (default) and `exclude_cells=[]`
        produce identical output — backward-compat for the call
        sites that didn't pass anything."""
        a = SpatialGrid.from_bounds(SMALL_BBOX, resolution="5arcmin")
        b = SpatialGrid.from_bounds(
            SMALL_BBOX, resolution="5arcmin", exclude_cells=[],
        )
        c = SpatialGrid.from_bounds(
            SMALL_BBOX, resolution="5arcmin", exclude_cells=None,
        )
        assert a.n_cells == b.n_cells == c.n_cells

    def test_filter_runs_inside_inner_loop_not_post_construction(self):
        """Evaluator §2 — filter happens INSIDE the inner row/col
        loop, not as a post-construction list comprehension. The
        AST walk asserts the filter expression is nested inside
        the row+col loop body. Prevents a regression that builds
        the full grid and then filters at the cls(...) constructor
        — wasteful at 5k cells with hundreds of exclusions."""
        import textwrap as _textwrap
        # `inspect.getsource(method)` returns the source with
        # class-level indentation; dedent so ast.parse accepts it.
        source = _textwrap.dedent(inspect.getsource(SpatialGrid.from_bounds))
        tree = ast.parse(source)
        # Find the outer `for row in range(...):` loop.
        outer_loops = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.For) and ast.unparse(n.target) == "row"
        ]
        assert outer_loops, (
            "from_bounds AST: outer `for row in range(...):` loop "
            "not found — has the function been refactored?"
        )
        # Inside the outer loop, find the `if cell_id in exclude_set: continue`
        # pattern.
        outer = outer_loops[0]
        cell_id_filter = None
        for node in ast.walk(outer):
            if isinstance(node, ast.If):
                cond_src = ast.unparse(node.test)
                if "cell_id" in cond_src and "exclude_set" in cond_src:
                    cell_id_filter = node
                    break
        assert cell_id_filter is not None, (
            "Filter `if cell_id in exclude_set: continue` NOT found "
            "inside the row/col construction loop. PRE.3.2 §2 binding "
            "requires the filter at the inner-loop locus, NOT post-"
            "construction (post-construction filtering is wasteful at "
            "5k cells with hundreds of exclusions)."
        )


class TestExecutorThreadThrough:
    """V2-22c-PRE.3.3 — `RegionConfig.exclude_cells` is threaded
    through to `SpatialGrid.from_bounds` at the executor's grid-
    construction call site. Verified via AST walk; full-pipeline
    e2e is reserved for Gate B integration smoke."""

    def test_executor_passes_exclude_cells_to_factory(self):
        executor_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "prismpy" / "pipeline" / "executor.py"
        )
        tree = ast.parse(executor_path.read_text())
        # Find every `SpatialGrid.from_bounds(...)` call in the file.
        from_bounds_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "from_bounds"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "SpatialGrid"
                ):
                    from_bounds_calls.append(node)

        assert from_bounds_calls, (
            "AST walk found no SpatialGrid.from_bounds calls in "
            "executor.py — has the call site been moved or renamed?"
        )

        for call in from_bounds_calls:
            kwarg_names = {kw.arg for kw in call.keywords}
            assert "exclude_cells" in kwarg_names, (
                "executor.py SpatialGrid.from_bounds call site is "
                "missing the `exclude_cells=` keyword arg. PRE.3.3 "
                "thread-through regressed — translators inherit grid "
                "pruning automatically only if the factory call passes "
                "the exclude_cells through."
            )
            # The corresponding value should reference
            # `self.config.region.exclude_cells` (or a getattr
            # equivalent). Soft check via source unparse.
            for kw in call.keywords:
                if kw.arg == "exclude_cells":
                    src = ast.unparse(kw.value)
                    assert "exclude_cells" in src, (
                        "exclude_cells= keyword's value doesn't "
                        f"reference exclude_cells; got {src!r}. The "
                        "thread-through must read from "
                        "self.config.region.exclude_cells."
                    )
