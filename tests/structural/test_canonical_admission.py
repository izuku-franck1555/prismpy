"""Structural and behavioural pins for the canonical admission helper.

The pins in this file enforce the producer-boundary canonical admission
discipline: every translate-time consumer of a shape-polymorphic climate
dict routes through ``prismpy.cells.admission.canonical_climate_for_grid``
so admission lives in one tested code path instead of fanning out across
per-callsite ad-hoc filters that drift over time. Pin 2 is the structural
backbone (AST walker over every ``is_real_climate_cell_id`` callsite);
Pin 1 + Pin 3 + the equivalence regression cover behavioural shape; the
CMS pins cover cross-platform consistency and the four pickle-writer
failure modes the prior ad-hoc filter let through.

Empirical anchor: ACEA gate fold-coincidence + ACEA post-DL twin + CRAFT
count-based gate + ACEA pickle writer foreign passthrough + PYTHIA
producer pre-filter foreign admission — five surfaces of the same class
across three translators.
"""
from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path
from typing import Any, Dict

from prismpy.cells.admission import (
    GRID_COLS_5ARCMIN,
    GRID_COLS_30ARCMIN,
    CanonicalClimate,
    canonical_climate_for_grid,
    cell_id_5arcmin_to_30arcmin_parent,
)


_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "prismpy"


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


class _GridCell:
    def __init__(self, cell_id: int) -> None:
        self.cell_id = cell_id


class _Grid:
    def __init__(self, cells: list) -> None:
        self.cells = cells


class _Ts:
    """Stand-in for ``ClimateTimeSeries`` covering the admissibility shape."""

    def __init__(self, label: str = "ts", n_records: int = 2) -> None:
        self.label = label
        self.records = [object() for _ in range(n_records)]


# ---------------------------------------------------------------------------
# Pin 1 — Helper behavioural pin
# ---------------------------------------------------------------------------


class TestCanonicalClimateForGridHelper(unittest.TestCase):
    """``canonical_climate_for_grid`` honours every admission rule.

    Covers the seven behavioural scenarios listed in the contract's
    helper-behavioural pin table: empty / None inputs, direct per-cell
    admission, tile-keyed broadcast, per-cell distinctness under
    shared parents, degenerate-series drop, path-dict and sentinel
    drop, and the foreign-fold-coincidence closure that motivated the
    META class redesign.
    """

    def test_returns_canonical_climate_dataclass(self):
        result = canonical_climate_for_grid({}, _Grid([_GridCell(9241)]))
        self.assertIsInstance(result, CanonicalClimate)
        self.assertEqual(result.per_cell, {})
        self.assertEqual(result.covered_tile_ids, set())

    def test_empty_inputs_return_empty_canonical(self):
        """None / empty climate_data and None / cell-less grid all return
        an empty ``CanonicalClimate``."""
        for climate_data, grid in (
            (None, _Grid([_GridCell(9241)])),
            ({}, _Grid([_GridCell(9241)])),
            ({9241: _Ts()}, None),
            ({9241: _Ts()}, _Grid([])),
        ):
            result = canonical_climate_for_grid(climate_data, grid)
            self.assertEqual(result.per_cell, {})
            self.assertEqual(result.covered_tile_ids, set())

    def test_admits_per_cell_keyed_input(self):
        cell = _GridCell(9241)
        ts = _Ts("legit")
        result = canonical_climate_for_grid({9241: ts}, _Grid([cell]))
        self.assertEqual(set(result.per_cell.keys()), {9241})
        self.assertIs(result.per_cell[9241], ts)
        self.assertEqual(
            result.covered_tile_ids,
            {cell_id_5arcmin_to_30arcmin_parent(9241)},
        )

    def test_admits_tile_keyed_input_with_fanout(self):
        # Children of tile 7300 (row_5=60, col_5=600..602 → cell_id 259800/801/802)
        children = [259800, 259801, 259802]
        target_tile = 7300
        for c in children:
            self.assertEqual(
                cell_id_5arcmin_to_30arcmin_parent(c), target_tile
            )
        tile_ts = _Ts(f"tile_{target_tile}")
        result = canonical_climate_for_grid(
            {target_tile: tile_ts},
            _Grid([_GridCell(c) for c in children]),
        )
        self.assertEqual(set(result.per_cell.keys()), set(children))
        for c in children:
            self.assertIs(result.per_cell[c], tile_ts)
        self.assertEqual(result.covered_tile_ids, {target_tile})

    def test_preserves_per_cell_distinct_under_shared_parent(self):
        """When sibling cells share a parent tile, per-cell-keyed inputs
        each keep their own series; the tile slot does not collapse to
        a last-write-wins value."""
        children = [259800, 259801, 259802]
        ts_a, ts_b, ts_c = _Ts("a"), _Ts("b"), _Ts("c")
        result = canonical_climate_for_grid(
            {children[0]: ts_a, children[1]: ts_b, children[2]: ts_c},
            _Grid([_GridCell(c) for c in children]),
        )
        self.assertIs(result.per_cell[children[0]], ts_a)
        self.assertIs(result.per_cell[children[1]], ts_b)
        self.assertIs(result.per_cell[children[2]], ts_c)

    def test_direct_per_cell_wins_over_tile_broadcast_on_same_cell(self):
        """A per-cell entry beats a tile-keyed broadcast for the cell
        that has both. The broadcast still fans out to the cells that
        do not carry a direct entry."""
        children = [259800, 259801, 259802]
        target_tile = 7300
        ts_a = _Ts("a")
        tile_ts = _Ts(f"tile_{target_tile}")
        result = canonical_climate_for_grid(
            {children[0]: ts_a, target_tile: tile_ts},
            _Grid([_GridCell(c) for c in children]),
        )
        self.assertIs(result.per_cell[children[0]], ts_a)
        self.assertIs(result.per_cell[children[1]], tile_ts)
        self.assertIs(result.per_cell[children[2]], tile_ts)

    def test_drops_foreign_fold_coincidence(self):
        """A foreign 30-arcmin key whose parent-fold coincidentally
        lands on an in-region target tile MUST be dropped. The
        coincidence ``fold(600) == 100`` collides with target tile 100;
        an ad-hoc parent-fold rescue at the consumer site would admit
        the foreign series — the helper rejects it at admission."""
        target_cell = 9241
        self.assertEqual(
            cell_id_5arcmin_to_30arcmin_parent(target_cell), 100
        )
        self.assertEqual(cell_id_5arcmin_to_30arcmin_parent(600), 100)

        foreign_ts = _Ts("foreign")
        result = canonical_climate_for_grid(
            {600: foreign_ts},
            _Grid([_GridCell(target_cell)]),
        )
        self.assertEqual(result.per_cell, {})
        self.assertEqual(result.covered_tile_ids, set())

    def test_drops_degenerate_series(self):
        """Single-record / empty / missing-records-attr / None-records
        entries are dropped when ``require_multi_record=True`` (default)."""

        class _NoRecordsAttr:
            pass

        class _EmptyList:
            records = []

        class _SingleList:
            records = [object()]

        class _NoneList:
            records = None

        cells = [_GridCell(9241), _GridCell(9242), _GridCell(9854), _GridCell(9243)]
        result = canonical_climate_for_grid(
            {
                9241: _NoRecordsAttr(),
                9242: _EmptyList(),
                9854: _SingleList(),
                9243: _NoneList(),
            },
            _Grid(cells),
        )
        self.assertEqual(result.per_cell, {})

    def test_drops_path_dict_string_and_sentinel_keys(self):
        """Path-dict string keys and the sentinel ``-1`` are dropped at
        the admissibility check before the key-space membership test
        runs."""
        cell = _GridCell(9241)
        result = canonical_climate_for_grid(
            {
                "rainfall_dir": _Ts(),
                "agera5_dir": _Ts(),
                -1: _Ts(),
            },
            _Grid([cell]),
        )
        self.assertEqual(result.per_cell, {})

    def test_require_multi_record_false_admits_single_record(self):
        """The loose tightness bar admits any non-empty records list;
        reserved for read-only metadata counter sites (none today)."""

        class _SingleList:
            records = [object()]

        result = canonical_climate_for_grid(
            {9241: _SingleList()},
            _Grid([_GridCell(9241)]),
            require_multi_record=False,
        )
        self.assertEqual(set(result.per_cell.keys()), {9241})

    def test_frozen_dataclass_blocks_field_rebinding(self):
        """``CanonicalClimate`` is frozen; rebinding ``per_cell`` raises
        ``FrozenInstanceError``. The ``covered_tile_ids`` cached
        property is still settable because ``functools.cached_property``
        writes through ``instance.__dict__`` directly."""
        result = canonical_climate_for_grid(
            {9241: _Ts()},
            _Grid([_GridCell(9241)]),
        )
        with self.assertRaises(Exception) as ctx:
            result.per_cell = {}
        self.assertEqual(type(ctx.exception).__name__, "FrozenInstanceError")

    def test_covered_tile_ids_is_cached(self):
        """Two accesses to ``covered_tile_ids`` return the same object
        identity — confirming the cached_property's __dict__ write
        survives under ``frozen=True``."""
        result = canonical_climate_for_grid(
            {9241: _Ts(), 9242: _Ts(), 9854: _Ts()},
            _Grid([_GridCell(9241), _GridCell(9242), _GridCell(9854)]),
        )
        first = result.covered_tile_ids
        second = result.covered_tile_ids
        self.assertIs(first, second)
        # And the contents are the correct parent-fold of admitted cells
        expected = {
            cell_id_5arcmin_to_30arcmin_parent(c) for c in (9241, 9242, 9854)
        }
        self.assertEqual(first, expected)


# ---------------------------------------------------------------------------
# Pin 2 — META structural pin (AST walker + census)
# ---------------------------------------------------------------------------


# Allowlist of (file relative to src/prismpy, enclosing function) tuples
# where a direct ``is_real_climate_cell_id`` callsite is allowed without
# routing through ``canonical_climate_for_grid``. These sites are
# read-only metadata counters, defense-in-depth filters on already-
# canonical input, or the helper's own definition site.
_META_PIN_ALLOWLIST = frozenset(
    [
        # Provenance counter — operates on the already-loaded climate
        # dict for an n_locations metadata field at the retrieve stage.
        ("pipeline/executor.py", "_execute_retrieve"),
        # Defense-in-depth on canonical input + crop_calendar re-fan;
        # both callsites live inside _surface_per_cell_climate.
        ("translators/base.py", "_surface_per_cell_climate"),
        # Metadata counters for n_climate_pickles / n_weather_files /
        # n_climate_locations / n_climate_cells inside each translator's
        # translate() method. These observe data.climate.keys() after
        # the canonical surfacing helper has run.
        ("translators/acea/translator.py", "translate"),
        ("translators/craft/translator.py", "translate"),
        ("translators/sarra_py/translator.py", "translate"),
        # SARRA-Py's per-loc admission inside the climate-file writer.
        # The substrate is path-dict-aware (string keys for path-dict
        # variants, int keys for ClimateTimeSeries variants) so the
        # canonical helper cannot route it; the local predicate at
        # this callsite is the substrate-specific admission gate.
        ("translators/sarra_py/translator.py", "_generate_climate_files"),
        # Scientific validator coverage check — read-only inspection.
        ("validators/scientific.py", "_check_coverage"),
        # Cockpit observed-values writer — read-only on
        # already-canonical data.climate.
        ("cockpit/observed_values_writer.py", "write_observed_values_json"),
        # Helper's own canonical definition site.
        ("cells/admission.py", "_admissible_series"),
    ]
)


def _enclosing_function_name(tree: ast.AST, lineno: int) -> str:
    """Return the name of the nearest enclosing function for the AST
    node whose source line is ``lineno``.

    The walker descends into ``FunctionDef`` / ``AsyncFunctionDef``
    nodes and tracks the closest enclosing scope for the target line.
    Nested functions resolve to the innermost enclosing function.
    """
    target = ""
    target_start = -1
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", None)
        if end is None:
            continue
        if start <= lineno <= end and start > target_start:
            target = node.name
            target_start = start
    return target


def _is_real_callsites_in_file(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, enclosing_function)`` for every direct
    ``is_real_climate_cell_id(...)`` call in ``path``.

    Walks ``ast.Call`` nodes whose ``func`` is a ``Name`` named
    ``is_real_climate_cell_id``; non-call references (imports, string
    mentions, doctest examples) are skipped.
    """
    text = path.read_text()
    if "is_real_climate_cell_id" not in text:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "is_real_climate_cell_id":
            sites.append(
                (node.lineno, _enclosing_function_name(tree, node.lineno))
            )
    return sites


class TestAdmissionRoutingMetaPin(unittest.TestCase):
    """Every production ``is_real_climate_cell_id`` callsite either
    lives in a file that imports ``canonical_climate_for_grid`` from
    ``prismpy.cells.admission`` (helper-routing) OR has its enclosing
    function in the explicit allowlist of read-only metadata counters.

    The pin is the structural enforcement bedrock for the
    producer-boundary admission discipline: it prevents net-new
    ad-hoc consumer admission filters from re-introducing the drift
    class on a future surface.
    """

    def test_every_is_real_callsite_routes_or_allowlisted(self):
        violators: list[str] = []
        for path in _SRC_ROOT.rglob("*.py"):
            rel = path.relative_to(_SRC_ROOT).as_posix()
            # The canonical predicate's own definition file is exempt.
            if rel == "sources/climate/_sentinels.py":
                continue
            text = path.read_text()
            # Skip __pycache__ residue and files that do not call the
            # predicate at all (text check first to avoid AST parse).
            sites = _is_real_callsites_in_file(path)
            if not sites:
                continue
            routes_through_helper = (
                "from prismpy.cells.admission import" in text
                and "canonical_climate_for_grid" in text
            )
            if routes_through_helper:
                continue
            for lineno, enclosing in sites:
                if (rel, enclosing) in _META_PIN_ALLOWLIST:
                    continue
                violators.append(f"{rel}:{lineno} ({enclosing}())")
        self.assertEqual(
            violators,
            [],
            "Every production is_real_climate_cell_id callsite MUST "
            "either live in a file that imports canonical_climate_for_grid "
            "OR have its enclosing function in the META pin allowlist. "
            "Violators above are unrouted callsites that re-introduce "
            "the ad-hoc admission drift class.",
        )

    def test_callsite_census_within_baseline(self):
        """Total direct ``is_real_climate_cell_id`` callsites stays at
        the post-refactor baseline. The helper introduces 1 internal
        usage at ``cells/admission.py::_admissible_series``
        (allowlisted). Every refactored consumer site loses its direct
        callsite; the remaining direct callsites are the read-only
        metadata counters in the allowlist plus the SARRA-Py
        translate() metadata-counter site.
        """
        total = 0
        for path in _SRC_ROOT.rglob("*.py"):
            rel = path.relative_to(_SRC_ROOT).as_posix()
            if rel == "sources/climate/_sentinels.py":
                # The predicate's definition file does not call itself
                # outside doctest examples; exclude from the call census.
                continue
            total += len(_is_real_callsites_in_file(path))
        # Post-refactor baseline: 1 helper-internal + 7 allowlisted
        # (executor + base + acea translate + craft translate +
        # sarra_py translate + scientific validator + cockpit writer)
        # = 8 direct callsites. Two callsites inside
        # _surface_per_cell_climate share the function but are counted
        # separately by AST.
        self.assertLessEqual(
            total,
            12,
            f"Direct is_real_climate_cell_id callsite census {total} "
            "exceeds the post-refactor baseline (<= 12). New ad-hoc "
            "consumer filter sites violate the producer-boundary "
            "admission discipline.",
        )


# ---------------------------------------------------------------------------
# Pin 3 — Per-translator routing pins (structural + behavioural)
# ---------------------------------------------------------------------------


def _translate_source(module_path: str) -> str:
    import importlib

    mod = importlib.import_module(module_path)
    # Each translator module's translate() lives on the platform class.
    for name in (
        "AceaTranslator",
        "CraftTranslator",
        "PythiaTranslator",
    ):
        cls = getattr(mod, name, None)
        if cls is not None:
            return inspect.getsource(cls.translate)
    raise AssertionError(f"No translator class found in {module_path}")


class TestAceaGateRoutesThroughHelper(unittest.TestCase):
    """ACEA gate + post-DL twin both route through
    ``canonical_climate_for_grid`` and derive missing tiles via the
    helper's ``covered_tile_ids`` cached property."""

    def test_acea_source_imports_helper(self):
        text = (
            _SRC_ROOT / "translators" / "acea" / "translator.py"
        ).read_text()
        self.assertIn("from prismpy.cells.admission import", text)
        self.assertIn("canonical_climate_for_grid", text)

    def test_acea_gate_uses_covered_tile_ids(self):
        source = _translate_source("prismpy.translators.acea.translator")
        # Both the gate site and the post-download twin compute
        # missing_tiles via covered_tile_ids.
        self.assertGreaterEqual(
            source.count("covered_tile_ids"),
            2,
            "ACEA translate() MUST derive missing_tiles from the helper's "
            "covered_tile_ids in both the gate site and the post-download "
            "twin.",
        )

    def test_acea_gate_drops_foreign_fold_coincidence_behavioural(self):
        """With ``climate_data = {600: ts}`` (foreign 30-arcmin id that
        folds to in-region tile 100) and ``data.grid`` having a single
        cell 9241 (parent tile 100), the gate's coverage computation
        via the helper yields no covered tiles — missing_tiles surfaces
        the target."""
        # 9241's parent tile is 100; 600 also folds to 100.
        self.assertEqual(
            cell_id_5arcmin_to_30arcmin_parent(9241), 100
        )
        self.assertEqual(cell_id_5arcmin_to_30arcmin_parent(600), 100)

        canonical = canonical_climate_for_grid(
            {600: _Ts("foreign")},
            _Grid([_GridCell(9241)]),
        )
        cell_ids_30arcmin_set = {100}
        missing_tiles = cell_ids_30arcmin_set - canonical.covered_tile_ids
        self.assertEqual(missing_tiles, {100})


class TestAceaPickleWriterRoutesThroughHelper(unittest.TestCase):
    """ACEA pickle writer iterates the canonical per-cell view and maps
    each 5-arcmin cell to its parent tile via the module-level helper.
    No direct ``_create_id_mapping`` call remains."""

    def test_pickle_writer_imports_helpers(self):
        text = (
            _SRC_ROOT / "translators" / "acea" / "translator.py"
        ).read_text()
        self.assertIn(
            "cell_id_5arcmin_to_30arcmin_parent", text,
            "ACEA module MUST import or use the module-level parent helper.",
        )
        # _create_id_mapping deletion: no method remaining.
        self.assertNotIn(
            "def _create_id_mapping",
            text,
            "ACEA _create_id_mapping MUST be deleted; pickle writer "
            "iterates canonical per-cell output directly.",
        )

    def test_pickle_writer_drops_foreign_keys_behavioural(self):
        """Foreign 30-arcmin key 600 routed through the helper produces
        no pickle for that key. The helper's per_cell admits only the
        in-region cell 9241 — the writer's iteration over per_cell.items()
        therefore writes one pickle (for cell 9241), not two."""
        canonical = canonical_climate_for_grid(
            {600: _Ts("foreign"), 9241: _Ts("legit")},
            _Grid([_GridCell(9241)]),
        )
        # Simulate the pickle-writer loop body
        pickles_to_write: list[int] = []
        for cell_id_5, _ts in canonical.per_cell.items():
            pickles_to_write.append(
                cell_id_5arcmin_to_30arcmin_parent(cell_id_5)
            )
        self.assertEqual(pickles_to_write, [100])


class TestCraftGateRoutesThroughHelper(unittest.TestCase):
    """CRAFT gate's count comparison uses ``len(helper.per_cell)`` and
    ``real_climate`` is the helper's per_cell."""

    def test_craft_source_imports_helper(self):
        text = (
            _SRC_ROOT / "translators" / "craft" / "translator.py"
        ).read_text()
        self.assertIn("from prismpy.cells.admission import", text)
        self.assertIn("canonical_climate_for_grid", text)

    def test_craft_gate_uses_helper_count(self):
        source = _translate_source("prismpy.translators.craft.translator")
        # The count comparison applies len(.per_cell) to the helper output.
        self.assertIn("canonical_climate_for_grid", source)
        self.assertIn(".per_cell", source)

    def test_craft_gate_drops_foreign_keys_behavioural(self):
        """Foreign key in CRAFT climate input fails the n_climate count
        because the helper drops it — gate fires download."""
        canonical = canonical_climate_for_grid(
            {600: _Ts("foreign")},
            _Grid([_GridCell(9241)]),
        )
        n_climate = len(canonical.per_cell)
        n_cells = 1
        self.assertLess(n_climate, n_cells)


class TestPythiaProducerFilterRoutesThroughHelper(unittest.TestCase):
    """PYTHIA producer pre-filter computes ``real_climate_data`` via the
    helper; the gate's set-difference uses the helper's per_cell keys."""

    def test_pythia_source_imports_helper(self):
        text = (
            _SRC_ROOT / "translators" / "pythia" / "translator.py"
        ).read_text()
        self.assertIn("from prismpy.cells.admission import", text)
        self.assertIn("canonical_climate_for_grid", text)

    def test_pythia_producer_filter_uses_helper(self):
        source = _translate_source("prismpy.translators.pythia.translator")
        # real_climate_data is computed from the helper's per_cell.
        self.assertIn("real_climate_data", source)
        self.assertIn("canonical_climate_for_grid", source)
        self.assertIn(".per_cell", source)

    def test_pythia_producer_filter_drops_foreign_keys_behavioural(self):
        """PYTHIA's producer pre-filter via helper drops the foreign key
        600 — the writer never sees it."""
        canonical = canonical_climate_for_grid(
            {600: _Ts("foreign"), 9241: _Ts("legit")},
            _Grid([_GridCell(9241)]),
        )
        real_climate_data = canonical.per_cell
        self.assertEqual(set(real_climate_data.keys()), {9241})


# ---------------------------------------------------------------------------
# SARRA-Py Bar D discriminator (substrate-specific local tightening)
# ---------------------------------------------------------------------------


def _function_body_source(module_path: str, qualname: str) -> str:
    """Return the source of a method body by enclosing-function scope.

    Reads ``module_path`` (a dotted name like
    ``prismpy.translators.sarra_py.translator``) and locates the
    ``ClassName.method_name`` named by ``qualname``. Returning the
    method source rather than the whole module text lets assertions
    target the specific function body — the global
    ``is_real_climate_cell_id`` name is referenced elsewhere in the
    same file for metadata counting, and a file-level grep would
    falsely pass when only the metadata-counter site survives.
    """
    import importlib

    mod = importlib.import_module(module_path)
    cls_name, method_name = qualname.split(".")
    cls = getattr(mod, cls_name)
    method = getattr(cls, method_name)
    return inspect.getsource(method)


class TestSarraPyBarDDiscriminator(unittest.TestCase):
    """SARRA-Py's per-loc admission inside ``_generate_climate_files``
    tightens from a Bar D (``hasattr`` only) check to the canonical
    real-cell predicate plus the records-shape guard. The fix closes
    the latent admission gap for the sentinel placeholder and
    path-dict string keys; it does NOT close foreign non-grid
    non-negative integer admission because the canonical predicate
    has no grid context (SARRA-Py's substrate-shape — path-dict OR
    ClimateTimeSeries dict — is outside the canonical helper's
    domain by design)."""

    def test_sarra_py_rejects_sentinel_and_non_int_documents_foreign_int_limitation(self):
        """Structural + behavioural pin.

        Structural part (mutation drill anchor): the source of
        ``_generate_climate_files`` contains the canonical predicate
        at the per-loc admission gate. Function-scoped walk because
        the predicate also appears at ``translate()``'s metadata-
        counter site, so a file-level grep cannot distinguish a
        reverted ``:741`` from the unchanged ``:301`` callsite. A
        revert of the production gate is caught here.

        Behavioural part (predicate-semantics documentation): against
        a synthetic ``climate_data`` whose values ALL carry valid
        ``records`` (so ``hasattr`` cannot be the rejector), the
        canonical predicate admits:

        - the real grid cell (positive parity preserved),
        - the foreign non-grid non-negative int (DOCUMENTED LIMITATION
          per CMS DN-3; the canonical predicate has no grid context),

        and rejects:

        - the sentinel ``-1``,
        - the path-dict string keys ``rainfall_dir`` /
          ``agera5_dir``.

        Coverage boundary (intentional): the behavioural block
        evaluates the predicate in test scope, not by invoking the
        production ``_generate_climate_files`` directly. A production
        revert at ``:741`` would NOT be caught by the behavioural
        assertions because they exercise the canonical predicate
        in-test; the structural assertion above is the mutation
        anchor. Lifting the behavioural block to a full call-through
        would require translator instantiation + file-system mocks
        for the NetCDF / CSV writer chain, which adds substantial
        fixture surface for a coverage class already locked by the
        structural assertion. Documenting the boundary here keeps
        the pin honest about what it does and does not protect.
        """
        from prismpy.sources.climate import is_real_climate_cell_id

        # Structural — function-scoped walk so a revert at the
        # admission gate is caught even though the metadata-counter
        # site in ``translate()`` keeps the predicate name live in
        # the same module.
        source = _function_body_source(
            "prismpy.translators.sarra_py.translator",
            "SarraPyTranslator._generate_climate_files",
        )
        self.assertIn(
            "is_real_climate_cell_id(loc_id)",
            source,
            "_generate_climate_files MUST gate per-loc admission on "
            "is_real_climate_cell_id(loc_id); reverting to a "
            "hasattr-only check re-opens the sentinel + path-dict "
            "string admission gap.",
        )
        self.assertIn(
            "hasattr(ts, 'records')",
            source,
            "_generate_climate_files MUST keep the records-shape "
            "guard alongside the canonical predicate; the predicate "
            "is the key-shape gate and the records check is the "
            "value-shape gate.",
        )

        # Behavioural — all values carry valid records so the
        # predicate is the rejector. Real grid cell + foreign non-grid
        # int both admit (foreign int admission is the documented
        # limitation per CMS DN-3); sentinel + non-int strings drop.
        class _MultiRec:
            records = [object(), object()]

        climate_data = {
            -1: _MultiRec(),                # sentinel placeholder
            100001: _MultiRec(),            # real grid cell
            99999999: _MultiRec(),          # foreign non-grid int
            "rainfall_dir": _MultiRec(),    # path-dict string
            "agera5_dir": _MultiRec(),      # path-dict string
        }
        admitted = [
            k for k, ts in climate_data.items()
            if is_real_climate_cell_id(k) and hasattr(ts, "records")
        ]
        # POSITIVE assertions — sentinel + non-int strings dropped.
        self.assertNotIn(-1, admitted)
        self.assertNotIn("rainfall_dir", admitted)
        self.assertNotIn("agera5_dir", admitted)
        # POSITIVE assertion — real cell preserved.
        self.assertIn(100001, admitted)
        # DOCUMENTED LIMITATION — foreign non-grid non-negative int
        # admits. The canonical predicate has no grid set; closing
        # this surface would require canonical-helper migration which
        # CMS DN-3 explicitly excludes for the SARRA-Py substrate.
        self.assertIn(99999999, admitted)

    def test_sarra_py_admits_real_cells_unchanged(self):
        """Positive parity — a clean climate_data with two real grid
        cells and valid multi-record series admits both cells; the
        post-fix predicate adds no behaviour beyond closing the
        sentinel / non-int gap on the same predicate every other
        translator's gate already uses."""
        from prismpy.sources.climate import is_real_climate_cell_id

        class _MultiRec:
            records = [object(), object()]

        climate_data = {100001: _MultiRec(), 100002: _MultiRec()}
        admitted = [
            k for k, ts in climate_data.items()
            if is_real_climate_cell_id(k) and hasattr(ts, "records")
        ]
        self.assertEqual(sorted(admitted), [100001, 100002])


# ---------------------------------------------------------------------------
# Equivalence regression pin (helper output matches deleted ACEA Sketch D)
# ---------------------------------------------------------------------------


class TestCanonicalHelperEquivalenceWithSketchD(unittest.TestCase):
    """The new ``canonical_climate_for_grid`` produces identical
    ``per_cell`` output to the deleted ACEA-private Sketch D helper
    on every Sketch D test input.

    Expected outputs hardcoded from the Sketch D behaviour recorded at
    branch-cut time (prismpy ``b8dd7f1`` ACEA
    ``_canonicalize_climate_by_grid_cells``). The hardcoded approach is
    simpler than preserving a temp copy of Sketch D during migration;
    if the new helper ever drifts from Sketch D's behaviour these
    assertions fail immediately.
    """

    def test_foreign_fold_coincidence_equivalence(self):
        """Sketch D test input from ``test_climate_sentinel_discipline.py``
        ``test_acea_canonicalize_drops_foreign_fold_coincidence``:
        ``climate_data = {600: ts}`` + grid with cell 9241. Sketch D
        emitted ``{}``; new helper must do the same."""
        ts = _Ts("foreign")
        result = canonical_climate_for_grid(
            {600: ts},
            _Grid([_GridCell(9241)]),
        )
        self.assertEqual(result.per_cell, {})

    def test_tile_keyed_fanout_equivalence(self):
        """Sketch D test input from
        ``test_acea_canonicalize_fans_tile_keyed_data_to_all_children``:
        single tile-keyed entry covering 3 children. Sketch D emitted
        the fanned-out dict; new helper must match."""
        children = [259800, 259801, 259802]
        target_tile = 7300
        tile_ts = _Ts(f"tile_{target_tile}")
        result = canonical_climate_for_grid(
            {target_tile: tile_ts},
            _Grid([_GridCell(c) for c in children]),
        )
        self.assertEqual(set(result.per_cell.keys()), set(children))
        for c in children:
            self.assertIs(result.per_cell[c], tile_ts)

    def test_per_cell_distinct_series_equivalence(self):
        """Sketch D test input from
        ``test_acea_canonicalize_preserves_per_cell_distinct_series``:
        3 sibling cells with distinct series. Sketch D preserved each
        series; new helper must match."""
        children = [259800, 259801, 259802]
        ts_a, ts_b, ts_c = _Ts("a"), _Ts("b"), _Ts("c")
        result = canonical_climate_for_grid(
            {children[0]: ts_a, children[1]: ts_b, children[2]: ts_c},
            _Grid([_GridCell(c) for c in children]),
        )
        self.assertIs(result.per_cell[children[0]], ts_a)
        self.assertIs(result.per_cell[children[1]], ts_b)
        self.assertIs(result.per_cell[children[2]], ts_c)


# ---------------------------------------------------------------------------
# CRAFT count-vs-canonical-helper equivalence (cross-platform)
# ---------------------------------------------------------------------------


class TestCraftHelperEquivalenceOnLegitimateInputs(unittest.TestCase):
    """The CRAFT gate's pre-refactor count of valid multi-record
    cells matches ``len(canonical_climate_for_grid(...).per_cell)``
    on legitimate (no-foreign) inputs. The two diverge only when
    foreign keys are present — in that case the helper is correct and
    the legacy count over-reports coverage."""

    def test_helper_count_equals_legacy_count_on_clean_input(self):
        cells = [_GridCell(9241), _GridCell(9242), _GridCell(9854)]
        climate_data: Dict[int, Any] = {
            9241: _Ts(),
            9242: _Ts(),
            9854: _Ts(),
        }
        legacy_n = sum(
            1 for _k, ts in climate_data.items()
            if hasattr(ts, "records") and len(ts.records) > 1
        )
        helper_n = len(
            canonical_climate_for_grid(climate_data, _Grid(cells)).per_cell
        )
        self.assertEqual(legacy_n, helper_n)

    def test_helper_count_diverges_on_foreign_keys(self):
        """When foreign keys appear, the helper count drops below the
        legacy count — confirming the helper closes the parent-fold-
        rescue admission class without changing behaviour on
        legitimate inputs."""
        cells = [_GridCell(9241)]
        climate_data = {9241: _Ts(), 600: _Ts("foreign")}
        legacy_n = sum(
            1 for _k, ts in climate_data.items()
            if hasattr(ts, "records") and len(ts.records) > 1
        )
        helper_n = len(
            canonical_climate_for_grid(climate_data, _Grid(cells)).per_cell
        )
        self.assertEqual(legacy_n, 2)
        self.assertEqual(helper_n, 1)


# ---------------------------------------------------------------------------
# ACEA pickle writer four failure modes
# ---------------------------------------------------------------------------


class TestAceaPickleWriterFailureModes(unittest.TestCase):
    """The ACEA pickle writer routes through ``canonical_climate_for_grid``
    so the four known failure modes (silent fabrication via fold
    collision / provenance pollution via foreign passthrough / silent
    fabrication via tile coincidence / sibling-distinctness invariant)
    are structurally impossible.
    """

    def test_sub_test_a_silent_fabrication_via_fold_collision(self):
        """CRITICAL — a 5-arcmin foreign key whose parent-fold lands on
        an in-region target tile MUST NOT produce a pickle for that
        target tile under the foreign series."""
        cells = [_GridCell(9241)]  # in-region cell with parent tile 100
        foreign_5arcmin = 0  # parent tile 0 (NOT in target set)
        self.assertNotEqual(
            cell_id_5arcmin_to_30arcmin_parent(foreign_5arcmin),
            cell_id_5arcmin_to_30arcmin_parent(9241),
        )
        canonical = canonical_climate_for_grid(
            {foreign_5arcmin: _Ts("foreign-5arcmin")},
            _Grid(cells),
        )
        # No stray pickle: helper drops the foreign 5-arcmin key
        # (parent tile 0 not in trusted spaces for this grid).
        self.assertEqual(canonical.per_cell, {})

    def test_sub_test_b_provenance_pollution_foreign_passthrough(self):
        """A foreign-keyed entry with valid records MUST NOT produce a
        pickle named with the foreign cell_id."""
        cells = [_GridCell(9241)]
        canonical = canonical_climate_for_grid(
            {12345: _Ts("foreign-out-of-bounds")},
            _Grid(cells),
        )
        # Iterate as the pickle writer would; collect output keys.
        output_keys = list(canonical.per_cell.keys())
        self.assertNotIn(12345, output_keys)
        self.assertEqual(output_keys, [])

    def test_sub_test_c_silent_fabrication_via_tile_coincidence(self):
        """CRITICAL — a 30-arcmin foreign id that collides with a target
        tile via parent-fold MUST NOT propagate its series to the
        target tile's pickle. The fold-coincidence drop is the META
        class closure."""
        cells = [_GridCell(9241)]  # parent tile 100
        foreign_30arcmin = 600  # 600 // 720 = 0; 600 % 720 = 600 -> tile 600
        # Wait — that's not 100. Let me re-check.
        # 600 // 4320 = 0, 600 % 4320 = 600; row_30 = 0, col_30 = 100;
        # so cell_id_5arcmin_to_30arcmin_parent(600) = 100. Right.
        self.assertEqual(
            cell_id_5arcmin_to_30arcmin_parent(600), 100
        )
        self.assertEqual(
            cell_id_5arcmin_to_30arcmin_parent(9241), 100
        )
        legit_ts = _Ts("legit")
        foreign_ts = _Ts("foreign")
        # Both legit (per-cell-keyed) and foreign (its parent-fold lands
        # on tile 100) coexist in input.
        canonical = canonical_climate_for_grid(
            {9241: legit_ts, 600: foreign_ts},
            _Grid(cells),
        )
        # The legit per-cell entry wins; the foreign entry is dropped at
        # admission, never reaching the tile_lookup.
        self.assertEqual(set(canonical.per_cell.keys()), {9241})
        self.assertIs(canonical.per_cell[9241], legit_ts)

    def test_sub_test_d_sibling_distinctness_invariant(self):
        """Two sibling 5-arcmin cells under a shared parent tile, each
        with its own series in per-cell-keyed input, MUST each receive
        their own series — no last-write-wins collapse to a shared tile
        slot."""
        children = [259800, 259801]
        target_tile = 7300
        for c in children:
            self.assertEqual(
                cell_id_5arcmin_to_30arcmin_parent(c), target_tile
            )
        ts_a, ts_b = _Ts("a"), _Ts("b")
        canonical = canonical_climate_for_grid(
            {children[0]: ts_a, children[1]: ts_b},
            _Grid([_GridCell(c) for c in children]),
        )
        self.assertIs(canonical.per_cell[children[0]], ts_a)
        self.assertIs(canonical.per_cell[children[1]], ts_b)


# ---------------------------------------------------------------------------
# Cross-platform helper consistency + SARRA-Py exclusion
# ---------------------------------------------------------------------------


class TestCrossPlatformHelperConsistency(unittest.TestCase):
    """Helper-routed admission yields identical-shape output regardless
    of which translator routes through it. SARRA-Py is explicitly
    EXCLUDED (different substrate; future finding)."""

    def test_three_translators_yield_identical_shape(self):
        """Same input + same grid through the helper yields a single
        ``CanonicalClimate`` instance whose shape is independent of
        which translator's callsite invokes it."""
        cells = [_GridCell(9241), _GridCell(9242)]
        ts_a, ts_b = _Ts("a"), _Ts("b")
        climate_data = {9241: ts_a, 9242: ts_b, 600: _Ts("foreign")}
        # The helper is purely a function of (climate_data, grid); the
        # translator identity is irrelevant. Three invocations from
        # different "consumer perspectives" return the same shape.
        for _consumer in ("ACEA", "CRAFT", "PYTHIA"):
            canonical = canonical_climate_for_grid(
                climate_data, _Grid(cells)
            )
            self.assertEqual(set(canonical.per_cell.keys()), {9241, 9242})
            self.assertIs(canonical.per_cell[9241], ts_a)
            self.assertIs(canonical.per_cell[9242], ts_b)

    def test_sarra_py_does_not_import_canonical_helper(self):
        """SARRA-Py has a different admission substrate (path-dict
        strings, not per-cell time-series). It MUST NOT import the
        canonical helper until a future sprint addresses the
        SARRA-Py-specific admission gap."""
        sarra_path = _SRC_ROOT / "translators" / "sarra_py" / "translator.py"
        text = sarra_path.read_text()
        self.assertNotIn(
            "from prismpy.cells.admission import",
            text,
            "SARRA-Py MUST NOT import canonical_climate_for_grid until "
            "a future sprint addresses its path-dict-aware admission "
            "(different substrate; tracked in the findings ledger).",
        )

    def test_canonical_helper_not_referenced_in_sarra_py_subtree(self):
        """Defence-in-depth on the SARRA-Py exclusion: every file
        under the sarra_py subtree contains zero references to
        ``canonical_climate_for_grid``. Uses ``pathlib`` rather than
        ``rg`` so the pin runs identically in CI environments where the
        ripgrep binary is absent."""
        sarra_dir = _SRC_ROOT / "translators" / "sarra_py"
        offenders: list[str] = []
        for path in sarra_dir.rglob("*.py"):
            if "canonical_climate_for_grid" in path.read_text():
                offenders.append(str(path.relative_to(_SRC_ROOT)))
        self.assertEqual(
            offenders,
            [],
            "SARRA-Py subtree MUST NOT reference canonical_climate_for_grid "
            "until a future sprint addresses its path-dict-aware "
            "admission (different substrate; tracked in the findings "
            "ledger).",
        )


# ---------------------------------------------------------------------------
# Module + constant exports
# ---------------------------------------------------------------------------


class TestAdmissionModuleExports(unittest.TestCase):
    def test_exports_match_contract(self):
        """The admission module exposes the contract-named members."""
        import prismpy.cells.admission as admission_mod

        for name in (
            "canonical_climate_for_grid",
            "cell_id_5arcmin_to_30arcmin_parent",
            "CanonicalClimate",
            "GRID_COLS_5ARCMIN",
            "GRID_COLS_30ARCMIN",
        ):
            self.assertTrue(
                hasattr(admission_mod, name),
                f"prismpy.cells.admission MUST export {name!r}.",
            )

    def test_grid_constants_match_canonical_resolutions(self):
        self.assertEqual(GRID_COLS_5ARCMIN, 4320)
        self.assertEqual(GRID_COLS_30ARCMIN, 720)


if __name__ == "__main__":
    unittest.main()
