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
        """AST walker — translate() body refers to both
        ``is_real_climate_cell_id`` AND
        ``_cell_id_5arcmin_to_30arcmin_parent`` AND
        ``ClimateDownloadError``."""
        import prismpy.translators.acea.translator as acea_mod

        source = inspect.getsource(acea_mod.AceaTranslator.translate)
        for name in (
            "is_real_climate_cell_id",
            "_cell_id_5arcmin_to_30arcmin_parent",
            "ClimateDownloadError",
            "real_30arcmin_tiles",
            "missing_tiles",
        ):
            self.assertIn(
                name, source,
                f"ACEA translate() MUST reference {name!r} per AC-F-CP-12.",
            )
        # NO stale ``elif n_climate < n_cells`` literal in the gate.
        self.assertNotIn(
            "n_climate < n_cells", source,
            "ACEA translate() MUST drop the stale `n_climate < n_cells` "
            "check per AC-F-CP-12 + codex Draft 3 H2.",
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
        """PYTHIA translate() body imports + uses
        ``is_real_climate_cell_id`` and set-difference pattern."""
        import prismpy.translators.pythia.translator as pythia_mod

        source = inspect.getsource(pythia_mod.PythiaTranslator.translate)
        for name in (
            "is_real_climate_cell_id",
            "real_climate_keys",
            "missing_sites",
        ):
            self.assertIn(
                name, source,
                f"PYTHIA translate() MUST reference {name!r} per AC-F-CP-13.",
            )
        # Subset-fetch pattern: _download_site_weather is called with
        # subset_site_ids kwarg.
        self.assertIn(
            "subset_site_ids=", source,
            "PYTHIA gate MUST request the missing-sites subset only via "
            "_download_site_weather(..., subset_site_ids=...) per "
            "AC-F-CP-13 merge-semantics.",
        )
        # No more raw ``n_climate < n_sites`` check.
        self.assertNotIn(
            "n_climate < n_sites", source,
            "PYTHIA gate MUST drop the stale `n_climate < n_sites` check "
            "per AC-F-CP-13.",
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
        ``real_climate_data`` dict via ``is_real_climate_cell_id``
        BEFORE calling ``_generate_weather_files`` so the writer never
        sees the sentinel."""
        import prismpy.translators.pythia.translator as pythia_mod

        source = inspect.getsource(pythia_mod.PythiaTranslator.translate)
        # The filtered dict comprehension uses the canonical helper
        self.assertIn(
            "real_climate_data = {", source,
            "PYTHIA translate() MUST construct a `real_climate_data` "
            "dict filtered via is_real_climate_cell_id before invoking "
            "_generate_weather_files. Without the filter, the sentinel "
            "shifts every real site's filename and PYTHIA fails to "
            "locate weather at runtime (codex post-rebase BLOCKING).",
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
            "_generate_weather_files. Codex post-rebase BLOCKING: the "
            "unfiltered call shifts site filenames.",
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


class TestAceaToleratesAlready30ArcminKeys(unittest.TestCase):
    """Codex post-rebase finding: when a caller hands ACEA a climate
    dict already keyed by 30-arcmin tile IDs (a shape
    ``_create_id_mapping`` supports), the gate's 5→30 parent fold MUST
    pass those keys through unchanged. Otherwise, valid coverage gets
    re-mapped to a different tile and ``missing_tiles`` triggers a
    spurious NASA POWER download that can fail offline runs."""

    def test_acea_gate_uses_set_membership_to_detect_30arcmin_keys(self):
        """Structural pin — ACEA translate() builds
        ``cell_ids_30arcmin_set`` and uses it as the discriminator
        between already-30-arcmin and 5-arcmin keys."""
        import prismpy.translators.acea.translator as acea_mod

        source = inspect.getsource(acea_mod.AceaTranslator.translate)
        self.assertIn(
            "cell_ids_30arcmin_set = set(cell_ids_30arcmin)", source,
            "ACEA translate() MUST build a `cell_ids_30arcmin_set` so "
            "the 5→30 mapping can short-circuit on already-30-arcmin "
            "keys (codex post-rebase SHOULD-FIX).",
        )
        # The mapping expression keeps keys that are already 30-arcmin
        self.assertIn(
            "k if k in cell_ids_30arcmin_set", source,
            "ACEA 30-arcmin coverage set MUST pass already-30-arcmin "
            "keys through unchanged instead of re-mapping via "
            "_cell_id_5arcmin_to_30arcmin_parent (codex post-rebase "
            "SHOULD-FIX).",
        )

    def test_acea_30arcmin_discriminator_yields_empty_missing_tiles(self):
        """Behavioural pin — when the climate dict's keys are all in
        the target 30-arcmin set, the discriminator computes
        ``missing_tiles == set()`` so no spurious download fires."""
        from prismpy.sources.climate import is_real_climate_cell_id

        # Simulate a region needing 3 already-30-arcmin tiles.
        cell_ids_30arcmin = [100, 200, 300]
        cell_ids_30arcmin_set = set(cell_ids_30arcmin)

        # Synthetic time-series that satisfies the gate's record-validity
        # check (>1 records). Only the ``records`` attribute length is
        # inspected; the items themselves are not unpacked here.
        class _Ts:
            records = [object(), object()]

        fake_ts = _Ts()
        climate_data = {100: fake_ts, 200: fake_ts, 300: fake_ts}

        real_30arcmin_tiles = {
            (k if k in cell_ids_30arcmin_set
             else None)  # `None` would surface as a spurious tile id
            for k, ts in climate_data.items()
            if is_real_climate_cell_id(k)
            and hasattr(ts, 'records')
            and len(ts.records) > 1
        }
        missing_tiles = cell_ids_30arcmin_set - real_30arcmin_tiles
        self.assertEqual(
            missing_tiles, set(),
            "When every climate key is already a 30-arcmin tile id, the "
            "discriminator MUST yield zero missing tiles. A non-empty "
            "missing_tiles here means the 5→30 mapping is re-running on "
            "already-30-arcmin keys and shifting the coverage set "
            "(codex post-rebase SHOULD-FIX).",
        )


class TestAceaCoverageBoundedToRegion(unittest.TestCase):
    """Codex R14 cycle-3: the ACEA tile-coverage computation must
    intersect with the region's expected 30-arcmin tile set so a
    stray foreign-region tile in ``climate_data`` cannot collapse
    onto a target tile via the 5→30 parent helper and falsely
    declare coverage complete."""

    def test_acea_coverage_intersects_with_region_tile_set(self):
        """Structural pin — both the gate computation and the
        post-download verification end the tile-coverage
        comprehension with ``& cell_ids_30arcmin_set`` so out-of-
        region folds are dropped from the coverage set."""
        import prismpy.translators.acea.translator as acea_mod

        source = inspect.getsource(acea_mod.AceaTranslator.translate)
        # Both comprehensions terminate with the intersection.
        self.assertGreaterEqual(
            source.count("} & cell_ids_30arcmin_set"), 2,
            "ACEA tile-coverage comprehensions (gate + post-download) "
            "MUST end with `& cell_ids_30arcmin_set` so foreign-region "
            "folds are dropped from the coverage set (codex R14 cycle-3 "
            "SHOULD-FIX).",
        )

    def test_acea_intersection_drops_out_of_region_fold(self):
        """Behavioural pin — when a foreign 5-arcmin key folds to a
        tile NOT in the region's expected set, the intersection drops
        the spurious coverage entry and ``missing_tiles`` correctly
        surfaces the still-uncovered targets."""
        from prismpy.sources.climate import is_real_climate_cell_id
        from prismpy.translators.acea.translator import AceaTranslator

        inst = AceaTranslator.__new__(AceaTranslator)

        class _Ts:
            records = [object(), object()]

        ts = _Ts()
        # Foreign 5-arcmin id 0 folds to 30-arcmin tile 0 (outside
        # the region's expected set {500}); intersection must drop it.
        cell_ids_30arcmin_set = {500}
        climate_data = {0: ts}

        real_30arcmin_tiles = {
            (k if k in cell_ids_30arcmin_set
             else inst._cell_id_5arcmin_to_30arcmin_parent(k))
            for k, ts_val in climate_data.items()
            if is_real_climate_cell_id(k)
            and hasattr(ts_val, 'records')
            and len(ts_val.records) > 1
        } & cell_ids_30arcmin_set
        missing_tiles = cell_ids_30arcmin_set - real_30arcmin_tiles

        self.assertEqual(
            real_30arcmin_tiles, set(),
            "Foreign-region tile that folds to a non-target tile MUST "
            "be dropped by the intersection; pre-intersection set may "
            "contain it but post-intersection set MUST NOT.",
        )
        self.assertEqual(
            missing_tiles, {500},
            "missing_tiles MUST surface the region's actually-uncovered "
            "tile when the only `climate_data` entry is foreign + folds "
            "outside the target set.",
        )


class TestPythiaWriterRequiresValidSeries(unittest.TestCase):
    """Codex R14 cycle-3: the PYTHIA writer-input filter must also
    require records validity. Without that check, a real cell key
    with an empty or one-record series slips into
    ``_generate_weather_files`` and either writes an empty ``.WTH``
    or crashes the writer."""

    def test_pythia_real_climate_data_filter_includes_records_check(self):
        """Structural pin — the ``real_climate_data`` comprehension
        ANDs in the same records-validity predicate the missing-
        sites gate uses (``hasattr(ts, 'records') and len(ts.records)
        > 1``)."""
        import prismpy.translators.pythia.translator as pythia_mod

        source = inspect.getsource(pythia_mod.PythiaTranslator.translate)
        # Look for the real_climate_data comprehension body with both
        # the canonical helper AND the records predicate.
        self.assertIn("real_climate_data = {", source)
        # Pull the slice containing the comprehension and confirm the
        # records-validity predicate is present in the same block.
        idx = source.index("real_climate_data = {")
        # The next "}" closes the comprehension; everything up to that
        # token is the predicate region.
        comp_slice = source[idx:idx + source[idx:].index("}\n") + 2]
        for token in ("is_real_climate_cell_id(k)", "hasattr(ts, 'records')", "len(ts.records) > 1"):
            self.assertIn(
                token, comp_slice,
                f"PYTHIA `real_climate_data` comprehension MUST include "
                f"`{token}` so the writer never sees an empty or "
                "degenerate series (codex R14 cycle-3 SHOULD-FIX).",
            )

    def test_pythia_writer_filter_drops_empty_and_single_record_series(self):
        """Behavioural pin — the filter expression evaluated against
        a synthetic mix of (multi-record, single-record, empty,
        no-records-attr) cases admits only the multi-record case."""
        from prismpy.sources.climate import is_real_climate_cell_id

        class _NoRecordsAttr:
            pass

        class _EmptyList:
            records = []

        class _SingleList:
            records = [object()]

        class _MultiList:
            records = [object(), object()]

        cases = {
            1001: _MultiList(),       # GOOD: real cell + valid series
            1002: _SingleList(),      # BAD: single record (degenerate)
            1003: _EmptyList(),       # BAD: empty series
            1004: _NoRecordsAttr(),   # BAD: no records attr at all
        }
        admitted = {
            k: ts for k, ts in cases.items()
            if is_real_climate_cell_id(k)
            and hasattr(ts, 'records')
            and len(ts.records) > 1
        }
        self.assertEqual(
            set(admitted.keys()), {1001},
            "PYTHIA writer filter MUST admit only the multi-record "
            "case. Degenerate single-record / empty / missing-attr "
            "series MUST be dropped before reaching "
            "_generate_weather_files (codex R14 cycle-3 SHOULD-FIX).",
        )


class TestAceaCanonicalEmit(unittest.TestCase):
    """Sketch D — ACEA's translator emits the per-cell climate dict by
    iterating ``grid.cells`` rather than ``climate_data.keys``. This
    moves the foreign-key filter from a per-site downstream guard to
    the canonical-emit boundary (no fold called on climate_data keys
    during the build, so a foreign tile whose parent-fold coincidentally
    lands on an in-region target cannot reach the canonical dict)."""

    def test_acea_canonicalize_helper_iterates_grid_cells(self):
        """Structural pin — ``_canonicalize_climate_by_grid_cells``
        exists on AceaTranslator AND iterates ``grid.cells`` (not
        ``climate_data.keys()``) per AST inspection."""
        import prismpy.translators.acea.translator as acea_mod

        self.assertTrue(
            hasattr(acea_mod.AceaTranslator, "_canonicalize_climate_by_grid_cells"),
            "AceaTranslator MUST expose _canonicalize_climate_by_grid_cells "
            "per Sketch D (codex R15 §6.3-redesign-trigger absorption).",
        )
        source = inspect.getsource(
            acea_mod.AceaTranslator._canonicalize_climate_by_grid_cells
        )
        # The helper iterates grid.cells (canonical source) AND admits
        # climate_data entries only via the known cell_ids / tile_ids
        # spaces — never via fold-during-build.
        self.assertIn(
            "for cell in grid.cells", source,
            "_canonicalize_climate_by_grid_cells MUST iterate "
            "grid.cells to emit the canonical per-cell dict.",
        )
        self.assertIn(
            "grid_cell_ids", source,
            "_canonicalize_climate_by_grid_cells MUST track the known "
            "grid cell-id space (admit climate_data keys that match a "
            "real cell.cell_id).",
        )
        self.assertIn(
            "grid_tile_ids", source,
            "_canonicalize_climate_by_grid_cells MUST track the known "
            "target tile-id space (admit climate_data keys that match "
            "a real cell's parent tile).",
        )

    def test_acea_canonicalize_drops_foreign_fold_coincidence(self):
        """Behavioural pin (Scenario B closure) — when ``climate_data``
        contains a foreign 30-arcmin id whose parent-fold lands by
        coincidence on an in-region target tile, the canonical helper
        MUST NOT admit the foreign series into the canonical dict.
        ``fold(600) == 100`` empirically demonstrates the coincidence
        the cycle-3 intersection couldn't close on its own."""
        from prismpy.translators.acea.translator import AceaTranslator

        inst = AceaTranslator.__new__(AceaTranslator)

        class _GridCell:
            def __init__(self, cell_id):
                self.cell_id = cell_id

        class _Grid:
            def __init__(self, cells):
                self.cells = cells

        class _Ts:
            def __init__(self, label):
                self.label = label
                self.records = [object(), object()]

        # Real grid cell 9241 has parent tile 100 (5-arcmin →
        # 30-arcmin via the helper). Foreign 30-arcmin id 600 also
        # folds to 100. Without the structural filter, a naive
        # tile_lookup would key 100 to the foreign series.
        target_cell_5arcmin = 9241
        assert inst._cell_id_5arcmin_to_30arcmin_parent(
            target_cell_5arcmin
        ) == 100
        assert inst._cell_id_5arcmin_to_30arcmin_parent(600) == 100

        foreign_ts = _Ts("FOREIGN")
        # climate_data has ONLY the foreign entry (no in-region tile
        # 100 entry and no in-region cell 9241 entry). The canonical
        # helper must produce an empty dict — the grid cell surfaces
        # as missing-coverage downstream.
        climate_data = {600: foreign_ts}
        grid = _Grid([_GridCell(target_cell_5arcmin)])

        canonical = inst._canonicalize_climate_by_grid_cells(climate_data, grid)

        self.assertEqual(
            canonical, {},
            "_canonicalize_climate_by_grid_cells MUST drop foreign keys "
            "whose parent-fold coincidentally lands on an in-region "
            "target tile. fold(600)=100 collides with target tile 100; "
            "without the structural filter the foreign series would "
            "have been admitted (Scenario B residual). Empty canonical "
            "dict here is the correct behaviour — the grid cell "
            "surfaces as missing-coverage downstream.",
        )

    def test_acea_canonicalize_fans_tile_keyed_data_to_all_children(self):
        """Behavioural pin (R15 P2 #1 fan-out) — when ``climate_data``
        is 30-arcmin tile-keyed (a single tile entry covering multiple
        5-arcmin children in the region), every child cell receives the
        tile's series in the canonical dict."""
        from prismpy.translators.acea.translator import AceaTranslator

        inst = AceaTranslator.__new__(AceaTranslator)

        class _GridCell:
            def __init__(self, cell_id):
                self.cell_id = cell_id

        class _Grid:
            def __init__(self, cells):
                self.cells = cells

        class _Ts:
            def __init__(self, label):
                self.label = label
                self.records = [object(), object()]

        # Pick a target tile + 3 of its 5-arcmin children, all in the
        # unambiguously-5-arcmin range (cell_id > 259199 = 30-arcmin
        # max) so the children cannot be misread as foreign 30-arcmin
        # tiles. 5-arcmin grid: 2160 rows × 4320 cols; 30-arcmin grid:
        # 360 rows × 720 cols; 6× fanout per axis. Target tile 7300 =
        # row_30 10 + col_30 100. Children at row_5=60, col_5 in
        # 600..602: 60*4320 + 600 = 259800.
        children = [259800, 259801, 259802]  # row_5=60, cols 600/601/602
        target_tile = 7300
        for c in children:
            assert inst._cell_id_5arcmin_to_30arcmin_parent(c) == target_tile, (
                f"fixture math wrong: fold({c}) = "
                f"{inst._cell_id_5arcmin_to_30arcmin_parent(c)}, "
                f"expected {target_tile}"
            )

        tile_ts = _Ts(f"TILE_{target_tile}")
        climate_data = {target_tile: tile_ts}
        grid = _Grid([_GridCell(c) for c in children])

        canonical = inst._canonicalize_climate_by_grid_cells(climate_data, grid)

        self.assertEqual(
            set(canonical.keys()), set(children),
            "_canonicalize_climate_by_grid_cells MUST fan a single "
            "tile-keyed climate entry out to every child grid cell "
            "of that tile. Missing fan-out is codex R15 P2 #1 — "
            "tile-keyed input with no per-child entries would surface "
            "as zero-coverage even when the tile data is fully real.",
        )
        for cell_id in children:
            self.assertIs(
                canonical[cell_id], tile_ts,
                f"Child cell {cell_id} MUST receive the tile-"
                f"{target_tile} series.",
            )

    def test_acea_canonicalize_preserves_per_cell_distinct_series(self):
        """Behavioural anti-corruption pin — when ``climate_data`` is
        already keyed by 5-arcmin grid cell and multiple cells share a
        parent 30-arcmin tile, each cell MUST receive its own distinct
        series.

        Earlier shape stored every direct-keyed 5-arcmin entry under
        ``tile_lookup[parent_tile]`` and then fanned the parent tile
        back out to every child grid cell. Last-write-wins on the
        tile slot collapsed all sibling cells to the LAST iterated
        child's series — silent per-cell data corruption that is
        invisible in the dict's coverage shape and impossible to
        recover from downstream.

        Currently latent in production callers (today's producers
        emit one series per tile, fanned), but the trap waits for the
        first producer that emits per-cell-distinct 5-arcmin series.
        """
        from prismpy.translators.acea.translator import AceaTranslator

        inst = AceaTranslator.__new__(AceaTranslator)

        class _GridCell:
            def __init__(self, cell_id):
                self.cell_id = cell_id

        class _Grid:
            def __init__(self, cells):
                self.cells = cells

        class _Ts:
            def __init__(self, label):
                self.label = label
                self.records = [object(), object()]

        # Three sibling 5-arcmin children of tile 7300 (same fixture
        # math as the fan-out test above). Each child gets a distinct
        # series so we can detect the corruption directly.
        children = [259800, 259801, 259802]
        target_tile = 7300
        for c in children:
            assert inst._cell_id_5arcmin_to_30arcmin_parent(c) == target_tile

        ts_a = _Ts("CELL_259800")
        ts_b = _Ts("CELL_259801")
        ts_c = _Ts("CELL_259802")
        climate_data = {
            children[0]: ts_a,
            children[1]: ts_b,
            children[2]: ts_c,
        }
        grid = _Grid([_GridCell(c) for c in children])

        canonical = inst._canonicalize_climate_by_grid_cells(climate_data, grid)

        self.assertEqual(
            set(canonical.keys()), set(children),
            "_canonicalize_climate_by_grid_cells MUST emit one entry "
            "per grid cell when climate_data is 5-arcmin direct-keyed.",
        )
        self.assertIs(
            canonical[children[0]], ts_a,
            f"Cell {children[0]} MUST keep its own ts_a; collapsed-to-"
            "last-write-wins corruption would assign ts_c here.",
        )
        self.assertIs(
            canonical[children[1]], ts_b,
            f"Cell {children[1]} MUST keep its own ts_b; collapsed-to-"
            "last-write-wins corruption would assign ts_c here.",
        )
        self.assertIs(
            canonical[children[2]], ts_c,
            f"Cell {children[2]} MUST keep its own ts_c.",
        )


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
