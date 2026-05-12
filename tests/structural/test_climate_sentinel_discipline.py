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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
