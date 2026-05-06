"""Pin every external (non-stdlib / non-prismpy) import that a climate
source module performs to a matching ``pyproject.toml`` dependency.

The undeclared-rioxarray gap surfaced through a SARRA-Py retrieval
that fell into the broad-except path in ``executor._load_climate_data``,
got logged as a generic "TAMSAT download failed" warning, returned
``None``, and let the caller fall through to the placeholder climate
shape. Validators that read the placeholder shape then reported zero
per-cell climate coverage on a project that should have produced a
file-based climate dict instead.

Declaring every climate-source import in ``pyproject.toml`` keeps the
fail mode honest: a missing dep on a fresh venv installs surfaces as
a pip-resolution error at install time, not as a silent climate-data
absence at validate time.

Anti-mutation drill: drop ``rioxarray`` from ``pyproject.toml`` →
this test fails with the specific module name + the climate source
file that imports it.
"""
from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLIMATE_SOURCE_DIR = _REPO_ROOT / "src" / "prismpy" / "sources" / "climate"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


# Top-level package names that ship with the CPython standard library.
# Restricted to the modules climate sources actually import; expand the
# set when a new stdlib import lands. The check stays narrow on purpose
# — the real signal is the external-import set, not exhaustive stdlib
# coverage.
_STDLIB_PACKAGES = frozenset({
    "__future__", "abc", "argparse", "ast", "calendar", "collections",
    "concurrent", "contextlib", "copy", "csv", "dataclasses", "datetime",
    "decimal", "enum", "functools", "glob", "gzip", "hashlib", "io",
    "itertools", "json", "logging", "math", "multiprocessing", "os",
    "pathlib", "pickle", "platform", "queue", "re", "shutil", "signal",
    "socket", "ssl", "stat", "string", "struct", "subprocess", "sys",
    "tempfile", "textwrap", "threading", "time", "traceback", "tomllib",
    "types", "typing", "unittest", "urllib", "uuid", "warnings",
    "weakref", "xml", "zipfile",
})


# Imports wrapped in ``try: ... except ImportError: ...`` are treated
# as optional — the module gracefully degrades when the dep is absent.
# The walker below skips imports it finds inside such a guard so the
# test only fires on hard-required imports.
def _import_under_optional_guard(node: ast.AST, ancestors: list[ast.AST]) -> bool:
    """True iff ``node`` (an ``ast.Import`` / ``ast.ImportFrom``) sits
    in the ``body`` of a ``try: ... except (ImportError |
    ModuleNotFoundError): ...`` block AND is not nested inside a
    function / lambda / comprehension scope between the import and the
    try statement.

    Tightening the check to ``try.body`` only (not ``orelse``,
    ``finalbody``, or any handler bodies) matches the graceful-degrade
    semantics: an import in the try body fires ImportError → the
    handler runs → the package degrades. An import in ``orelse`` runs
    only AFTER the try body succeeds (so it is not actually guarded by
    that try). An import inside a nested function runs at call time,
    which the enclosing try did not bracket.

    The function-scope guard prevents the false-exemption case where a
    helper defined inside a try body imports a third-party module — the
    function body executes when the helper is called, not when the
    enclosing try runs, so the import is effectively unguarded.
    """
    # Walk ancestors from the import outward. The first try we hit
    # whose ``body`` contains the import (with no intervening function
    # / lambda / comprehension scope) is the guard candidate. Anything
    # farther out is irrelevant — Python's ``except`` only catches
    # exceptions raised in the immediately-enclosing try body.
    last = node
    for parent in reversed(ancestors):
        # If a function / lambda / comprehension intervenes between
        # the import and an outer try, the import runs at the inner
        # call/eval site and is NOT bracketed by the outer try.
        if isinstance(parent, (
            ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
            ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        )):
            return False
        if isinstance(parent, ast.Try):
            # Only ``try.body`` carries the graceful-degrade semantics.
            # Imports in ``orelse`` / ``finalbody`` / handler bodies
            # are NOT exempt.
            if last not in parent.body:
                return False
            for handler in parent.handlers:
                exc = handler.type
                if exc is None:
                    # bare ``except:`` is broad enough to cover ImportError.
                    return True
                names = []
                if isinstance(exc, ast.Name):
                    names = [exc.id]
                elif isinstance(exc, ast.Tuple):
                    names = [
                        e.id for e in exc.elts if isinstance(e, ast.Name)
                    ]
                if "ImportError" in names or "ModuleNotFoundError" in names:
                    return True
            return False
        last = parent
    return False


def _is_external(top_level: str) -> bool:
    """True iff ``top_level`` is a third-party module that pyproject
    must declare. Filters out stdlib modules and the in-tree prismpy
    package itself."""
    if not top_level:
        return False
    if top_level == "prismpy":
        return False
    if top_level in _STDLIB_PACKAGES:
        return False
    if top_level in sys.builtin_module_names:
        return False
    return True


def _collect_external_imports(py_file: Path) -> set[str]:
    """Walk the AST of ``py_file`` and return every top-level package
    name from ``import X`` and ``from X import Y`` statements that is
    classified as external AND not inside a ``try/except ImportError``
    optional-guard block.

    The walk is full-tree (function-local imports included) so the
    rioxarray-style "imported inside the function for its side effect"
    pattern is captured. Relative imports (``from . import X``) are
    skipped because ``ImportFrom.module is None`` in that case.
    Optional imports gated by ``try: ... except ImportError`` are also
    skipped per the ``_import_under_optional_guard`` predicate.
    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    externals: set[str] = set()
    for node, ancestors in _walk_with_ancestors(tree):
        if isinstance(node, ast.Import):
            if _import_under_optional_guard(node, ancestors):
                continue
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if _is_external(top):
                    externals.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module is None:
                continue
            if _import_under_optional_guard(node, ancestors):
                continue
            top = node.module.split(".", 1)[0]
            if _is_external(top):
                externals.add(top)
    return externals


def _walk_with_ancestors(root: ast.AST):
    """Yield ``(node, ancestors)`` pairs across the tree so the
    optional-guard predicate can inspect a node's enclosing scopes
    without re-walking. ``ancestors`` is ordered root-first."""
    stack: list[tuple[ast.AST, list[ast.AST]]] = [(root, [])]
    while stack:
        node, ancestors = stack.pop()
        yield node, ancestors
        new_ancestors = ancestors + [node]
        for child in ast.iter_child_nodes(node):
            stack.append((child, new_ancestors))


def _declared_dep_names() -> set[str]:
    """Parse the ``[project] dependencies`` array from ``pyproject.toml``
    and return the set of declared top-level package names. The parser
    stays line-based so the test runs on Python 3.10 (no ``tomllib``
    required) without adding a dev dep."""
    src = _PYPROJECT.read_text(encoding="utf-8")
    match = re.search(
        r"^dependencies\s*=\s*\[(?P<body>.*?)^\]",
        src, re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        "pyproject.toml must contain a top-level "
        "[project] dependencies array."
    )
    body = match.group("body")
    declared: set[str] = set()
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Each dep entry: "name>=spec",  or  "name[extra]>=spec",
        m = re.match(r'"\s*([A-Za-z0-9_.\-]+)', line)
        if m:
            # Normalize to the canonical lowercase / hyphen form a
            # CPython import would resolve to. PyPI normalizes the
            # underscore / hyphen split, but the ``import name`` is
            # case-sensitive; rioxarray vs rio-xarray etc. all
            # normalize to lowercase here.
            declared.add(m.group(1).lower().replace("-", "_"))
    return declared


# Some pyproject deps map onto an import-name that differs from the
# package name. The mapping below records every case the climate source
# tree relies on so the test can match a declaration against the import
# without false-positive failures.
_PYPI_TO_IMPORT_ALIASES: dict[str, str] = {
    # python-dateutil exposes ``dateutil``.
    "python_dateutil": "dateutil",
    # PyYAML exposes ``yaml``.
    "pyyaml": "yaml",
    # netCDF4 exposes ``netCDF4`` — case differs from PyPI norm.
    "netcdf4": "netcdf4",
}


# Imports that are intentionally not declared in pyproject.toml because
# the package is not published on PyPI and ships only via a local
# editable install. Each entry pairs the import name with a short
# rationale so future readers know why the test exempts it.
_KNOWN_LOCAL_ONLY_OPTIONAL_IMPORTS: dict[str, str] = {
    # ``SARRA_data_download`` is an internal SARRA-Py utility package
    # that lives at ``../SARRA-Py-documents/SARRA_data-download`` as
    # an editable install; it is not on PyPI. The agera5 source has
    # an ``ImportError``-guarded probe at the constructor level
    # (``self._sarra_download_available = False`` on absent) and a
    # second hard import inside ``_download_with_sarra`` that only
    # executes when the probe succeeded. Wrapping the second import
    # in a redundant try/except is the cleaner long-term refactor;
    # task #155 ("Move year-top raise_if_cancelled before SARRA
    # import in agera5.py") tracks the related cleanup.
    "sarra_data_download": (
        "Local-only utility, not on PyPI. The first import at "
        "agera5.py is ImportError-guarded; the second import "
        "inside _download_with_sarra runs only when the probe "
        "passed. See task #155 for the related refactor."
    ),
}


def _normalize_for_match(name: str) -> str:
    """Return the case- and separator-normalized name a Python import
    would resolve to so declared deps and discovered imports compare
    cleanly."""
    norm = name.lower().replace("-", "_")
    return _PYPI_TO_IMPORT_ALIASES.get(norm, norm)


class TestClimateSourceImportsDeclared(unittest.TestCase):
    """Every external import a climate source module performs must
    appear in ``pyproject.toml`` ``[project] dependencies``.

    Climate sources fan into many third-party packages (rasterio,
    rioxarray, xarray, netCDF4, cdsapi, ...). A missing declaration
    surfaces as a runtime ``ModuleNotFoundError`` deep inside the
    retrieve stage and, because the executor's broad-except path
    catches and downgrades the error to a warning, the climate dict
    silently falls through to the placeholder shape.
    """

    @classmethod
    def setUpClass(cls):
        cls.declared = {
            _normalize_for_match(d) for d in _declared_dep_names()
        }

    def test_rioxarray_is_declared(self):
        """rioxarray registers the ``.rio`` accessor on xarray
        DataArrays via import side-effect; tamsat.py imports it
        for the Phase 2 GeoTIFF write. Without the declaration,
        a fresh venv install fails the TAMSAT retrieve with a
        ``ModuleNotFoundError`` that the broad-except path
        downgrades to a TAMSAT warning and the climate dict
        falls through to the placeholder shape."""
        self.assertIn(
            "rioxarray", self.declared,
            "pyproject.toml must declare 'rioxarray' in "
            "[project] dependencies. tamsat.py imports it for "
            "the .nc → .tif Phase 2 conversion; a missing "
            "declaration surfaces as a silent climate-data gap "
            "on fresh venv installs.",
        )

    def test_every_climate_source_import_is_declared(self):
        """Walk every ``.py`` module under ``sources/climate/`` and
        assert the top-level package of each external import is
        declared in pyproject.toml. The walker covers function-local
        imports so the rioxarray pattern (imported inside a method
        for its side-effect registration) is included."""
        # Optional-extras (deps under ``[project.optional-dependencies]``)
        # are also acceptable since installations that opt into them
        # carry the dep. The agera5 extra exposes cdsapi this way.
        src = _PYPROJECT.read_text(encoding="utf-8")
        optional_deps: set[str] = set()
        for m in re.finditer(
            r'"\s*([A-Za-z0-9_.\-]+)\s*(?:>=|==|<=|>|<)?[^"]*"',
            src,
        ):
            optional_deps.add(_normalize_for_match(m.group(1)))
        acceptable = self.declared | optional_deps

        problems: list[str] = []
        py_files = sorted(_CLIMATE_SOURCE_DIR.rglob("*.py"))
        self.assertGreater(
            len(py_files), 0,
            f"expected at least one .py file under {_CLIMATE_SOURCE_DIR}",
        )
        for py_file in py_files:
            externals = _collect_external_imports(py_file)
            for ext in sorted(externals):
                normalized = _normalize_for_match(ext)
                if normalized in acceptable:
                    continue
                if normalized in _KNOWN_LOCAL_ONLY_OPTIONAL_IMPORTS:
                    continue
                problems.append(
                    f"{py_file.relative_to(_REPO_ROOT)}: "
                    f"imports {ext!r} but pyproject.toml does "
                    "not declare it"
                )
        self.assertEqual(
            problems, [],
            "Climate source modules must only import packages that "
            "pyproject.toml declares (either in [project] "
            "dependencies or [project.optional-dependencies]). "
            "Undeclared imports surface as silent climate-data gaps "
            "on fresh venv installs:\n  " + "\n  ".join(problems),
        )


if __name__ == "__main__":
    unittest.main()
