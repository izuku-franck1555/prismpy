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

    def test_executor_filters_grid_cells_with_exclude_cells(self):
        """Translators inherit grid pruning automatically because the
        executor filters ``grid.cells`` post-construction. Per F-R
        AC-2 (Stage 4), user-skip is applied AFTER the inclusion_rule
        + share_percent filters so the canonical-grid arithmetic
        identity holds (`n_cells_full_extent = excluded_by_rule +
        excluded_by_threshold + admitted + user_excluded`). The
        thread-through guarantee is preserved — translators still
        see only non-excluded cells in ``grid.cells``; the factory
        call is now ``exclude_cells=None`` and the filtering happens
        in a post-Stage-3 list comprehension that reads from
        ``self.config.region.exclude_cells``.

        Asserts both: (a) the post-Stage-3 ``grid.cells`` filter
        comprehension references ``exclude_cells``; (b) NO
        ``SpatialGrid.from_bounds`` callsite passes
        ``exclude_cells=`` other than ``None`` (the canonical-grid
        arithmetic invariant — user-skip lives at Stage 4, not
        inside the factory).
        """
        executor_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "prismpy" / "pipeline" / "executor.py"
        )
        source = executor_path.read_text()
        tree = ast.parse(source)

        # AC-2 invariant a: a list comprehension over grid.cells
        # references exclude_cells (the Stage 4 user-skip filter).
        assert "user_excluded" in source and "exclude_cells" in source, (
            "F-R AC-2 Stage 4: executor.py must apply a "
            "user-skip filter referencing "
            "``self.config.region.exclude_cells`` (via a "
            "``user_excluded = set(...)`` derivation) and a list "
            "comprehension that drops excluded cells from "
            "``grid.cells``. Translators inherit grid pruning "
            "automatically because Stage 4 mutates grid.cells "
            "before any translator reads it."
        )
        assert "c.cell_id not in user_excluded" in source, (
            "F-R AC-2 Stage 4: the user-skip filter must "
            "comprehension-skip cells whose cell_id is in the "
            "user_excluded set. Pin enforces the canonical "
            "arithmetic identity (`n_cells_full_extent = "
            "excluded_by_rule + excluded_by_threshold + "
            "admitted + user_excluded`)."
        )

        # AC-2 invariant b: SpatialGrid.from_bounds calls must NOT
        # pass exclude_cells= other than None. The harmonize
        # 5-stage flow uses the factory's clip_geometry param for
        # the inclusion_rule, but exclude_cells stays None at
        # construction so the cell-count arithmetic holds.
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
            for kw in call.keywords:
                if kw.arg == "exclude_cells":
                    src = ast.unparse(kw.value)
                    assert src == "None", (
                        f"F-R AC-2: SpatialGrid.from_bounds at "
                        f"line {call.lineno} passes "
                        f"``exclude_cells={src}``; the post-F-R "
                        f"flow filters at Stage 4 (post-construction "
                        f"comprehension), not at the factory. "
                        f"Pre-F-R thread-through has been replaced "
                        f"by the canonical-grid arithmetic identity."
                    )
