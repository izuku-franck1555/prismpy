"""Sprint F-CP fixup structural pins for AC-F-CP-12/13/13.5/14.

These pins close the small-region climate-download regression where the
gate compared raw ``len(data.climate)`` against ``n_cells`` and counted
the executor's sentinel-keyed placeholder as a "real" cell — so a 1-cell
project saw ``1 < 1 == False`` and silently skipped the NASA POWER
download.

Coverage:

- AC-F-CP-12 (3): ACEA climate gate uses canonical helpers / typed error
  on incomplete download / partial-tile coverage downloads missing only
- AC-F-CP-13 (1): PYTHIA climate gate uses canonical helper
- AC-F-CP-13.5 (1): metadata writers exclude sentinel climate
- AC-F-CP-14 (4): placeholder sentinel canonical-source / no raw climate
  key filters outside helper / circular-import safety / non-int key
  handling

Per durable §24 + §27 + §28: the placeholder sentinel has one canonical
constant + helper; producer (executor.create_placeholder_climate) and
consumers (translators + validators + metadata writers) share that
vocabulary.
"""
from __future__ import annotations

import ast
import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock


_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "prismpy"


# ---------------------------------------------------------------------------
# AC-F-CP-14 — Canonical helpers for sentinel discipline
# ---------------------------------------------------------------------------


class TestPlaceholderSentinelCanonical(unittest.TestCase):
    """``PLACEHOLDER_CLIMATE_SENTINEL_ID`` + ``is_real_climate_cell_id``
    live in exactly one module under ``src/prismpy/``."""

    def test_placeholder_sentinel_id_canonical_source(self):
        """Only ``_sentinels.py`` declares the constant + helper."""
        offenders_const: list[str] = []
        offenders_helper: list[str] = []
        for path in _SRC_ROOT.rglob("*.py"):
            if path.name == "_sentinels.py":
                continue
            text = path.read_text()
            # Look for declaration (Final-typed annotation OR plain assign)
            if "PLACEHOLDER_CLIMATE_SENTINEL_ID: Final" in text:
                offenders_const.append(str(path))
            elif "PLACEHOLDER_CLIMATE_SENTINEL_ID =" in text:
                # Only flag declarations (LHS), not import-from references.
                # The simplest discriminator: declaration has no preceding
                # `import` or `from ... import` on the same line.
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("PLACEHOLDER_CLIMATE_SENTINEL_ID ="):
                        offenders_const.append(str(path))
                        break
            if "def is_real_climate_cell_id" in text:
                offenders_helper.append(str(path))
        self.assertEqual(
            offenders_const, [],
            "PLACEHOLDER_CLIMATE_SENTINEL_ID MUST be declared only in "
            "prismpy/sources/climate/_sentinels.py (durable §24). "
            f"Other declarations: {offenders_const!r}",
        )
        self.assertEqual(
            offenders_helper, [],
            "is_real_climate_cell_id MUST be defined only in "
            "prismpy/sources/climate/_sentinels.py (durable §24). "
            f"Other definitions: {offenders_helper!r}",
        )

    def test_is_real_climate_cell_id_handles_non_int_keys(self):
        """The helper is non-int safe (codex Draft 4 V5 catch).

        Climate dicts have mixed-shape variants in the codebase — SARRA-Py
        path-dicts carry string keys; a raw ``key >= 0`` would TypeError.
        Returning False for non-int keys keeps the helper safe at every
        metadata-writer + calendar-fanout site.
        """
        from prismpy.sources.climate import is_real_climate_cell_id

        # Real cells
        self.assertTrue(is_real_climate_cell_id(0))
        self.assertTrue(is_real_climate_cell_id(100))
        self.assertTrue(is_real_climate_cell_id(9_331_199))
        # Sentinel
        self.assertFalse(is_real_climate_cell_id(-1))
        self.assertFalse(is_real_climate_cell_id(-100))
        # SARRA-Py path-dict variants
        self.assertFalse(is_real_climate_cell_id("rainfall_dir"))
        self.assertFalse(is_real_climate_cell_id("agera5_dir"))
        # Defensive non-int cases
        self.assertFalse(is_real_climate_cell_id(None))
        self.assertFalse(is_real_climate_cell_id(1.0))
        self.assertFalse(is_real_climate_cell_id([]))
        self.assertFalse(is_real_climate_cell_id({}))

    def test_circular_import_safety(self):
        """Importing ``_sentinels`` from a fresh interpreter state does
        not cascade into heavy translator / executor modules."""
        # Drop any cached modules that could short-circuit the import.
        for mod_name in list(sys.modules.keys()):
            if (
                mod_name.startswith("prismpy.sources.climate._sentinels")
                or mod_name == "prismpy.sources.climate._sentinels"
            ):
                del sys.modules[mod_name]

        # Track what got loaded as a side effect of the import.
        before = set(sys.modules.keys())
        # Use importlib to load the module directly so we don't pull
        # the climate package's __init__.py (which imports NASA POWER + etc.).
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_sentinels_isolated",
            _SRC_ROOT / "sources" / "climate" / "_sentinels.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        after = set(sys.modules.keys())
        new_modules = after - before

        # Assert no heavy prismpy module came along for the ride.
        forbidden_prefixes = (
            "prismpy.pipeline.executor",
            "prismpy.translators.acea",
            "prismpy.translators.pythia",
            "prismpy.translators.craft",
            "prismpy.translators.sarra_py",
            "prismpy.validators.scientific",
        )
        side_effect_imports = [
            m for m in new_modules
            if any(m == p or m.startswith(p + ".") for p in forbidden_prefixes)
        ]
        self.assertEqual(
            side_effect_imports, [],
            f"_sentinels.py MUST be importable without cascading into "
            f"executor / translators / validators (per AC-F-CP-14 + "
            f"durable §24). Side-effect imports observed: "
            f"{side_effect_imports!r}",
        )
        # And the helper + constant are present.
        self.assertEqual(mod.PLACEHOLDER_CLIMATE_SENTINEL_ID, -1)
        self.assertTrue(callable(mod.is_real_climate_cell_id))

    def test_no_raw_climate_key_filters_outside_helper(self):
        """No raw ``>= 0`` / ``< 0`` / ``== -1`` comparisons on climate-
        keyed identifiers outside ``_sentinels.py``.

        Scope: ``src/prismpy/translators/`` + ``src/prismpy/validators/`` +
        ``src/prismpy/packaging/``. The walker scans for ``Compare`` AST
        nodes whose left operand is a ``Name`` matching the climate-key
        roster (``cid``, ``loc_id``, ``k``, ``key``, ``cell_id`` IN a
        climate-related call/loop context) and asserts the source file
        imports ``is_real_climate_cell_id`` somewhere — i.e., the raw
        comparison either lives ALONGSIDE the helper as transient code
        OR doesn't exist.
        """
        # Tightened scope: only complain when the file (a) contains a raw
        # ``cid >= 0`` / ``loc_id < 0`` / ``-1`` comparison on an
        # identifier from the climate-key roster AND (b) does NOT import
        # ``is_real_climate_cell_id``. Files that already adopted the
        # helper but still carry transient unrelated comparisons are
        # exempt — the pin enforces the migration, not a stylistic ban.
        climate_key_names = {"cid", "loc_id", "k", "key"}
        scopes = [
            _SRC_ROOT / "translators",
            _SRC_ROOT / "validators",
            _SRC_ROOT / "packaging",
        ]
        offenders: list[tuple[str, int]] = []
        for scope in scopes:
            for path in scope.rglob("*.py"):
                text = path.read_text()
                # Skip files that explicitly migrated (helper import
                # present somewhere in the module).
                if "is_real_climate_cell_id" in text:
                    continue
                # AST-walk for raw climate-key comparisons.
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Compare):
                        continue
                    if not isinstance(node.left, ast.Name):
                        continue
                    if node.left.id not in climate_key_names:
                        continue
                    # Look for >= 0 / < 0 / == -1 / != -1 comparators.
                    for op, comparator in zip(node.ops, node.comparators):
                        if not isinstance(comparator, ast.Constant):
                            continue
                        cv = comparator.value
                        if cv == 0 and isinstance(op, (ast.GtE, ast.Gt, ast.Lt, ast.LtE)):
                            offenders.append((str(path), node.lineno))
                            break
                        if cv == -1 and isinstance(op, (ast.Eq, ast.NotEq, ast.Lt)):
                            offenders.append((str(path), node.lineno))
                            break
        self.assertEqual(
            offenders, [],
            "Raw climate-key sentinel comparisons MUST migrate to the "
            "canonical ``is_real_climate_cell_id`` helper (AC-F-CP-14 + "
            f"durable §24). Sites still using raw filters: {offenders!r}",
        )


# ---------------------------------------------------------------------------
# AC-F-CP-12 — ACEA climate gate
# ---------------------------------------------------------------------------


class TestAceaClimateGate(unittest.TestCase):
    """ACEA gate uses canonical helpers + raises typed error on
    incomplete download + only downloads missing tiles."""

    def test_acea_climate_gate_uses_canonical_helpers(self):
        """AST walker — translate() body routes the climate gate
        through the canonical admission helper and surfaces a typed
        error on incomplete coverage."""
        import prismpy.translators.acea.translator as acea_mod

        source = inspect.getsource(acea_mod.AceaTranslator.translate)
        for name in (
            "canonical_climate_for_grid",
            "covered_tile_ids",
            "ClimateDownloadError",
            "missing_tiles",
        ):
            self.assertIn(
                name, source,
                f"ACEA translate() MUST reference {name!r} so the gate "
                "lives on the canonical admission helper.",
            )
        # NO stale ``elif n_climate < n_cells`` literal in the gate.
        self.assertNotIn(
            "n_climate < n_cells", source,
            "ACEA translate() MUST drop any stale `n_climate < n_cells` "
            "check; coverage runs through `covered_tile_ids`.",
        )

    def test_acea_climate_failure_raises_typed_error(self):
        """When the post-download retry leaves tiles uncovered, the gate
        raises ``ClimateDownloadError`` (NOT silent ``return``)."""
        import prismpy.translators.acea.translator as acea_mod

        source = inspect.getsource(acea_mod.AceaTranslator.translate)
        # Look for ``raise ClimateDownloadError`` in the gate body.
        self.assertIn(
            "raise ClimateDownloadError(", source,
            "ACEA gate MUST raise ClimateDownloadError on incomplete "
            "post-download coverage (AC-F-CP-12 + F-AG class).",
        )
        # And the error carries ``missing_tiles`` + ``source`` kwargs.
        self.assertIn("missing_tiles=", source)
        self.assertIn("source='nasa_power'", source)

    def test_acea_partial_tile_coverage_downloads_missing_only(self):
        """AST walker — the download call passes ``sorted(missing_tiles)``,
        not the full ``cell_ids_30arcmin``."""
        import prismpy.translators.acea.translator as acea_mod

        source = inspect.getsource(acea_mod.AceaTranslator.translate)
        # The post-AC-F-CP-12 gate calls _download_climate_30arcmin with
        # sorted(missing_tiles) — verify that pattern exists.
        self.assertIn(
            "self._download_climate_30arcmin(", source,
            "ACEA gate must invoke _download_climate_30arcmin.",
        )
        self.assertIn(
            "sorted(missing_tiles)", source,
            "ACEA gate MUST pass sorted(missing_tiles) (not the full "
            "cell_ids_30arcmin) to the downloader per AC-F-CP-12 partial-"
            "tile-coverage Option B contract.",
        )


# ---------------------------------------------------------------------------
# AC-F-CP-13 — PYTHIA climate gate
# ---------------------------------------------------------------------------


class TestPythiaClimateGate(unittest.TestCase):

    def test_pythia_climate_gate_uses_canonical_helper(self):
        """PYTHIA translate() body routes the climate gate through the
        canonical admission helper and downloads only the missing
        subset rather than the full grid."""
        import prismpy.translators.pythia.translator as pythia_mod

        source = inspect.getsource(pythia_mod.PythiaTranslator.translate)
        for name in (
            "canonical_climate_for_grid",
            "gate_canonical",
            "missing_sites",
        ):
            self.assertIn(
                name, source,
                f"PYTHIA translate() MUST reference {name!r} so the "
                "missing-sites computation lives on the canonical helper.",
            )
        # Subset-fetch pattern: _download_site_weather is called with
        # subset_site_ids kwarg.
        self.assertIn(
            "subset_site_ids=", source,
            "PYTHIA gate MUST request the missing-sites subset only via "
            "_download_site_weather(..., subset_site_ids=...) so a partial "
            "pre-retrieve state is preserved instead of double-fetched.",
        )
        # No more raw ``n_climate < n_sites`` check.
        self.assertNotIn(
            "n_climate < n_sites", source,
            "PYTHIA gate MUST stay off any stale `n_climate < n_sites` "
            "check; coverage runs through `canonical.per_cell.keys()`.",
        )


# ---------------------------------------------------------------------------
# AC-F-CP-13.5 — metadata writers exclude sentinel climate
# ---------------------------------------------------------------------------


class TestMetadataWritersExcludeSentinel(unittest.TestCase):
    """Each translator's metadata writer + scientific validator counts
    REAL climate cells, not the sentinel placeholder."""

    def test_metadata_writers_exclude_sentinel_climate(self):
        """AST walker — ACEA / CRAFT / SARRA-Py / scientific validator
        metadata-emit sites import ``is_real_climate_cell_id``."""
        targets = [
            (_SRC_ROOT / "translators" / "acea" / "translator.py",
             "n_climate_pickles"),
            (_SRC_ROOT / "translators" / "craft" / "translator.py",
             "n_weather_files"),
            (_SRC_ROOT / "translators" / "sarra_py" / "translator.py",
             "n_climate_locations"),
            (_SRC_ROOT / "validators" / "scientific.py",
             "n_climate_cells"),
            (_SRC_ROOT / "pipeline" / "executor.py",
             "n_locations"),
        ]
        offenders: list[str] = []
        for path, metadata_key in targets:
            text = path.read_text()
            self.assertIn(
                metadata_key, text,
                f"Expected key {metadata_key!r} in {path.name}",
            )
            if "is_real_climate_cell_id" not in text:
                offenders.append(f"{path}: missing is_real_climate_cell_id")
        self.assertEqual(
            offenders, [],
            "Metadata writers + scientific validator MUST import "
            "is_real_climate_cell_id per AC-F-CP-13.5. Sites still on "
            f"raw len() counts: {offenders!r}",
        )


class TestPythiaWriterFiltersPlaceholder(unittest.TestCase):
    """Codex post-rebase finding: the PYTHIA writer caller must filter
    the sentinel before invoking ``_generate_weather_files``. Without
    the filter, the placeholder ends up as ``1.WTH`` after sorting and
    every real-site filename shifts away from the shapefile ``ID``
    values PYTHIA uses at runtime."""

    def test_pythia_filters_placeholder_before_weather_file_write(self):
        """Structural pin — the PYTHIA translate() body builds a
        ``real_climate_data`` view from the canonical admission helper
        BEFORE calling ``_generate_weather_files`` so the writer never
        sees the sentinel + foreign + degenerate entries."""
        import prismpy.translators.pythia.translator as pythia_mod

        source = inspect.getsource(pythia_mod.PythiaTranslator.translate)
        # The filtered view comes from the canonical admission helper.
        self.assertIn(
            "real_climate_data = canonical_climate_for_grid", source,
            "PYTHIA translate() MUST build `real_climate_data` from "
            "`canonical_climate_for_grid` so admission discipline lives "
            "on the canonical helper at the producer boundary.",
        )
        # The writer call uses the filtered name, NOT the raw
        # climate_data. The call may pass additional keyword arguments
        # (e.g., ``grid=data.grid`` for the canonical sites-shapefile
        # parity contract); accept either ``(real_climate_data)`` or
        # ``(real_climate_data,`` as evidence the filtered name is the
        # first positional argument.
        self.assertTrue(
            "_generate_weather_files(real_climate_data)" in source
            or "_generate_weather_files(\n                    real_climate_data," in source
            or "_generate_weather_files(real_climate_data," in source,
            "PYTHIA translate() MUST pass the filtered `real_climate_data` "
            "(not raw `climate_data`) as the first positional arg of "
            "`_generate_weather_files`. Without the filter the writer's "
            "sorted index puts the sentinel at `1.WTH` and every real "
            "site's filename shifts away from the shapefile IDs.",
        )

    def test_pythia_writer_filter_drops_sentinel(self):
        """Behavioural pin — the filter expression evaluated against a
        synthetic mixed-key dict yields a dict containing only the
        real-cell entries (no sentinel)."""
        from prismpy.sources.climate import is_real_climate_cell_id

        sentinel = -1
        # Synthetic stand-in for ClimateTimeSeries; the filter checks
        # only the key, not the value shape.
        fake_ts = object()
        mixed = {sentinel: fake_ts, 0: fake_ts, 1001: fake_ts, 2050: fake_ts}
        real = {k: ts for k, ts in mixed.items() if is_real_climate_cell_id(k)}
        self.assertNotIn(
            sentinel, real,
            "real_climate_data MUST exclude the placeholder sentinel.",
        )
        self.assertEqual(
            set(real.keys()), {0, 1001, 2050},
            "real_climate_data MUST retain every real-cell entry.",
        )


# Sketch D canonical-emit tests previously lived here; they were
# replaced by the helper-level Pin 1 + equivalence regression +
# per-translator Pin 3 in ``tests/structural/test_canonical_admission.py``
# when the producer-boundary helper subsumed the ACEA-private Sketch D
# implementation. Keep this comment block as a navigation hint for
# anyone searching git history.


class TestPythiaWeatherFilesParityWithSitesShapefile(unittest.TestCase):
    """Sketch D Part 2 — PYTHIA's ``_generate_weather_files`` must
    iterate ``grid.cells`` with the same ``enumerate(start=1)`` ordering
    that ``_generate_sites_shapefile`` uses, so the ``.WTH`` filename
    sequential IDs match the shapefile's ``ID`` column. Missing-climate
    cells emit a sentinel WTH preserving the numbering — the Coverage
    validator surfaces those gaps honestly instead of the writer
    silently renumbering surviving sites."""

    def test_pythia_writer_iterates_grid_cells_in_enumerate_order(self):
        """Structural pin — the writer's emission loop is built from
        ``enumerate(grid.cells, start=1)`` (matching the shapefile
        producer at ``_generate_sites_shapefile``) and accepts a
        ``grid`` keyword argument."""
        import prismpy.translators.pythia.translator as pythia_mod

        source = inspect.getsource(pythia_mod.PythiaTranslator._generate_weather_files)
        # The writer signature now carries a ``grid`` keyword.
        sig = inspect.signature(
            pythia_mod.PythiaTranslator._generate_weather_files
        )
        self.assertIn(
            "grid", sig.parameters,
            "_generate_weather_files MUST accept a `grid` keyword "
            "argument so callers can request sites-shapefile parity.",
        )
        # The writer body iterates ``enumerate(grid.cells, start=1)``
        # under the grid-provided branch.
        self.assertIn(
            "enumerate(grid.cells, start=1)", source,
            "_generate_weather_files MUST iterate "
            "`enumerate(grid.cells, start=1)` to align WTH sequential "
            "IDs with the shapefile's ID column (codex R15 P2 #2 "
            "absorption).",
        )

    def test_pythia_writer_emits_sentinel_wth_for_missing_climate(self):
        """Behavioural pin (R15 P2 #2 closure) — when the climate dict
        is missing the entry for a grid cell that DOES appear in the
        shapefile roster, the writer emits a sentinel WTH at the
        corresponding seq-id so the filename ↔ ID mapping is preserved.

        Set-up: 3-cell grid; ``climate_data`` carries valid series for
        cells 1 and 3 (seq 1 and 3) but NOT cell 2 (seq 2). Expected:
        ``1.WTH``, ``2.WTH``, ``3.WTH`` all exist; ``1.WTH`` and
        ``3.WTH`` carry data rows; ``2.WTH`` is the header-only
        sentinel."""
        import tempfile
        from datetime import date
        from pathlib import Path
        from prismpy.translators.pythia.translator import PythiaTranslator
        from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
        from prismpy.models.spatial import SpatialGrid, GridCell, BoundingBox

        records = [
            ClimateRecord(
                date=date(2020, 1, 1) + __import__("datetime").timedelta(days=i),
                tmax=30.0, tmin=20.0, precip=0.0, srad=20.0,
            )
            for i in range(3)
        ]
        ts_with_data = ClimateTimeSeries(
            location_id=1, lat=10.0, lon=10.0, source="test",
            records=records, elevation=300.0,
        )

        cells = [
            GridCell(cell_id=1001, lat=10.0, lon=10.0, row=0, col=0),
            GridCell(cell_id=1002, lat=10.1, lon=10.1, row=0, col=1),
            GridCell(cell_id=1003, lat=10.2, lon=10.2, row=0, col=2),
        ]
        grid = SpatialGrid(
            bounds=BoundingBox(minx=10.0, miny=10.0, maxx=10.2, maxy=10.2),
            resolution=0.1, cells=cells,
        )

        with tempfile.TemporaryDirectory() as td:
            inst = PythiaTranslator.__new__(PythiaTranslator)
            inst.output_dir = Path(td)
            (inst.output_dir / "weather").mkdir(parents=True, exist_ok=True)
            # Minimal attributes the writer body reads via ``self.``.
            inst.provenance = None
            inst.cockpit_override_sidecar = None

            # Climate carries series for cell 1001 and 1003 ONLY; cell
            # 1002 (seq 2) is the gap that must surface as a sentinel.
            climate_data = {1001: ts_with_data, 1003: ts_with_data}

            files = inst._generate_weather_files(climate_data, grid=grid)

            wth_dir = inst.output_dir / "weather"
            for seq in (1, 2, 3):
                self.assertTrue(
                    (wth_dir / f"{seq}.WTH").exists(),
                    f"{seq}.WTH MUST exist so the sites.shp ID → WTH "
                    "lookup never points at a missing file.",
                )

            # Seq 2 is the sentinel — header-only, no data rows.
            seq2_content = (wth_dir / "2.WTH").read_text()
            data_rows = [
                line for line in seq2_content.splitlines()
                if line and not line.startswith("@")
                and not line.startswith("$")
            ]
            self.assertEqual(
                data_rows, [],
                "2.WTH MUST be a header-only sentinel (no data rows) "
                "when cell 1002 has no climate series. Coverage "
                "validator surfaces the gap honestly downstream "
                "(codex R15 P2 #2 absorption).",
            )

            # Seq 1 and seq 3 carry real data rows.
            for seq, label in [(1, "1001"), (3, "1003")]:
                content = (wth_dir / f"{seq}.WTH").read_text()
                rows = [
                    line for line in content.splitlines()
                    if line and not line.startswith("@")
                    and not line.startswith("$")
                ]
                self.assertGreater(
                    len(rows), 0,
                    f"{seq}.WTH for cell {label} MUST carry data rows.",
                )


class TestPythiaSurfaceCallUsesFilteredDict(unittest.TestCase):
    """PYTHIA's call to ``_surface_per_cell_climate`` MUST pass the
    records-validity-filtered dict (``real_climate_data``), not the raw
    ``climate_data``. The producer filter at the per-translator gate
    (``len(ts.records) > 1``) is strictly tighter than the consumer
    helper filter (``ts.records`` truthy), so passing the raw dict
    marks degenerate single-record cells as covered in ``data.climate``
    without a corresponding ``.WTH`` file on disk."""

    def test_pythia_surfaces_filtered_dict_not_raw(self):
        """Structural pin — the surface call site inside
        ``PythiaTranslator.translate`` MUST reference
        ``real_climate_data``, never raw ``climate_data``."""
        import inspect
        import prismpy.translators.pythia.translator as pythia_mod

        source = inspect.getsource(pythia_mod.PythiaTranslator.translate)
        idx = source.index("_surface_per_cell_climate(data,")
        surface_line = source[idx:idx + source[idx:].index(")") + 1]
        self.assertIn(
            "real_climate_data", surface_line,
            "PYTHIA MUST pass `real_climate_data` (the filtered dict) "
            "to _surface_per_cell_climate; passing raw `climate_data` "
            "marks degenerate single-record cells as covered in "
            "data.climate without writing .WTH files for them.",
        )

    def test_pythia_surface_drops_single_record_cells(self):
        """Behavioural pin — when ``climate_data`` carries a mix of
        multi-record, single-record, empty, and sentinel entries, the
        post-filter producer roster and the consumer helper roster MUST
        agree on the same cell set (only multi-record cells)."""
        from prismpy.sources.climate import is_real_climate_cell_id

        class _Empty:
            records = []

        class _Single:
            records = [object()]

        class _Multi:
            records = [object(), object()]

        climate_data = {
            1001: _Multi(),
            1002: _Single(),
            1003: _Empty(),
            -1: _Multi(),
        }
        real_climate_data = {
            k: ts for k, ts in climate_data.items()
            if is_real_climate_cell_id(k)
            and hasattr(ts, "records")
            and len(ts.records) > 1
        }
        surfaced = {
            cid: ts for cid, ts in real_climate_data.items()
            if is_real_climate_cell_id(cid)
            and hasattr(ts, "records")
            and ts.records
        }
        self.assertEqual(
            set(surfaced.keys()), {1001},
            "Post-filter producer roster and consumer helper roster "
            "MUST agree: only multi-record cells survive. Single-record, "
            "empty, and sentinel entries MUST be excluded.",
        )


class TestPythiaSurfaceCallEndToEnd(unittest.TestCase):
    """End-to-end regression for the PYTHIA producer-consumer parity:
    invokes the actual ``BaseTranslator._surface_per_cell_climate``
    helper with the producer-filtered roster and asserts the resulting
    ``data.climate`` state matches what ``.WTH`` files were emitted.
    Exact-tolerance counterpart to ``TestPythiaSurfaceCallUsesFilteredDict``
    behavioural pin."""

    def _build_real_climate_data(self, climate_data):
        from prismpy.sources.climate import is_real_climate_cell_id

        return {
            k: ts for k, ts in climate_data.items()
            if is_real_climate_cell_id(k)
            and hasattr(ts, "records")
            and len(ts.records) > 1
        }

    def test_single_record_cell_absent_from_data_climate_after_surface(self):
        """Single-record cell MUST NOT appear in ``data.climate`` after
        the post-fix PYTHIA surface call. The producer filter rejects
        it upstream so the consumer helper receives an empty roster and
        the harmonize-stage placeholder remains in place unchanged."""
        import types
        from prismpy.translators.pythia.translator import PythiaTranslator

        target_cell = 1001
        placeholder_ts = object()

        class _Single:
            records = [object()]

        climate_data = {target_cell: _Single()}
        real_climate_data = self._build_real_climate_data(climate_data)

        data = types.SimpleNamespace(climate={-1: placeholder_ts})
        inst = PythiaTranslator.__new__(PythiaTranslator)

        inst._surface_per_cell_climate(data, real_climate_data)

        self.assertIsNone(
            data.climate.get(target_cell),
            "Single-record cell MUST NOT appear in data.climate after "
            "the PYTHIA surface call — the producer filter excludes it "
            "so no .WTH gets written for that cell.",
        )
        self.assertEqual(
            data.climate, {-1: placeholder_ts},
            "data.climate MUST remain at the harmonize-stage "
            "placeholder when the producer-filtered roster is empty.",
        )

    def test_multi_record_cell_present_in_data_climate_after_surface(self):
        """Multi-record cell MUST appear in ``data.climate`` after the
        post-fix PYTHIA surface call with its records preserved exactly,
        and the harmonize-stage ``-1`` placeholder MUST be dropped."""
        import types
        from prismpy.translators.pythia.translator import PythiaTranslator

        target_cell = 1001
        multi_records = [object(), object()]

        class _Multi:
            records = multi_records

        climate_data = {target_cell: _Multi()}
        real_climate_data = self._build_real_climate_data(climate_data)

        data = types.SimpleNamespace(climate={-1: object()})
        inst = PythiaTranslator.__new__(PythiaTranslator)

        inst._surface_per_cell_climate(data, real_climate_data)

        self.assertIn(
            target_cell, data.climate,
            "Multi-record cell MUST appear in data.climate after the "
            "PYTHIA surface call — the producer filter admits it.",
        )
        self.assertEqual(
            data.climate[target_cell].records, multi_records,
            "Records list MUST be preserved exactly (no transformation "
            "by the surface helper).",
        )
        self.assertNotIn(
            -1, data.climate,
            "Harmonize-stage placeholder at -1 MUST be dropped after "
            "the surface call so downstream consumers do not iterate "
            "the synthetic sentinel alongside real cells.",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
