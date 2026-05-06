"""Pin every external (non-stdlib / non-prismpy) import that a source
module performs to a matching ``pyproject.toml`` dependency.

The undeclared-rioxarray gap surfaced through a SARRA-Py retrieval
that fell into the broad-except path in ``executor._load_climate_data``,
got logged as a generic "TAMSAT download failed" warning, returned
``None``, and let the caller fall through to the placeholder climate
shape. Validators that read the placeholder shape then reported zero
per-cell climate coverage on a project that should have produced a
file-based climate dict instead.

The same regression class repeated three times across consecutive
substrate-fix sprints — rioxarray, then cdsapi, then the
``SARRA_data_download`` library that AgERA5 retrieve actually calls.
The first two had public PyPI entries and were declared upward.
``SARRA_data_download`` had no PyPI entry and no upstream
``pyproject.toml``, so it was vendored under
``prismpy/src/prismpy/vendor/sarra_data_download/``. The structural
pin generalizes: an external import that cannot resolve to either
``[project] dependencies`` / ``[project.optional-dependencies]`` /
the in-tree ``prismpy.vendor.*`` namespace fails the test.

Declaring every source import in ``pyproject.toml`` keeps the fail
mode honest: a missing dep on a fresh venv install surfaces as a
pip-resolution error at install time, not as a silent climate-data
absence at validate time.

Anti-mutation drill 1: drop ``rioxarray`` from ``pyproject.toml`` →
the per-source-tree test fails with the specific module name + the
source file that imports it.
Anti-mutation drill 2: delete the vendored
``prismpy/src/prismpy/vendor/sarra_data_download/get_AgERA5_data.py``
→ ``test_sarra_data_download_vendor_files_present`` fails with a
diagnostic that names the missing path.
"""
from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLIMATE_SOURCE_DIR = _REPO_ROOT / "src" / "prismpy" / "sources" / "climate"
_SOURCES_DIR = _REPO_ROOT / "src" / "prismpy" / "sources"
_VENDOR_DIR = _REPO_ROOT / "src" / "prismpy" / "vendor"
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


# Imports that are intentionally not declared in pyproject.toml.
# Each entry pairs the import name with a short rationale so future
# readers know why the test exempts it. ``SARRA_data_download`` was
# previously the sole resident and was removed when the substrate
# switched to the ``prismpy.vendor.sarra_data_download`` namespace
# (which is internal and therefore not flagged as external by the
# AST scan). Adding a new entry should be paired with surfacing the
# rationale to a sprint-review checkpoint per the canonical-source
# discipline.
_KNOWN_LOCAL_ONLY_OPTIONAL_IMPORTS: dict[str, str] = {
    # ``django`` is imported only inside the ``cache_paths.py`` shim
    # (``from django.conf import settings``) so the cache-eviction
    # helpers can read ``settings.PRISMWEB_CACHE_DIR`` when running
    # under prismweb's ``evict_cache.handle()`` lifecycle. Bare
    # prismpy unit tests run without Django booted, so the
    # try/except returns ``None`` and the call site falls back to
    # an empty in-use set. Declaring django in prismpy's pyproject
    # would invert the dependency direction (prismpy is the upstream
    # library, prismweb is the consumer); the shim keeps the
    # correct direction while remaining import-safe under bare
    # prismpy. Not in any retrieve / download path.
    "django": (
        "Imported only inside cache_paths.py as a prismweb-side "
        "shim; declaring django in prismpy would invert the "
        "library dependency direction. Not in any retrieve path."
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
        # carry the dep.
        acceptable = self.declared | _optional_dep_names()

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


def _optional_dep_names() -> set[str]:
    """Parse the ``[project.optional-dependencies]`` block from
    ``pyproject.toml`` and return the set of declared top-level
    package names (normalized). Each ``<group> = [ ... ]`` array
    body is scanned individually so quoted strings outside that
    block (e.g., the migration-rationale comments inside
    ``[project] dependencies``) cannot leak in as false positives."""
    src = _PYPROJECT.read_text(encoding="utf-8")
    section_match = re.search(
        r"^\[project\.optional-dependencies\](?P<body>.*?)(?=^\[)",
        src, re.DOTALL | re.MULTILINE,
    )
    if section_match is None:
        return set()
    section_body = section_match.group("body")
    declared: set[str] = set()
    for array_match in re.finditer(
        r"^[A-Za-z0-9_-]+\s*=\s*\[(?P<arr>.*?)^\]",
        section_body, re.DOTALL | re.MULTILINE,
    ):
        for line in array_match.group("arr").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name_match = re.match(r'"\s*([A-Za-z0-9_.\-]+)', line)
            if name_match:
                declared.add(_normalize_for_match(name_match.group(1)))
    return declared


def _collect_guarded_imports(py_file: Path) -> set[str]:
    """Return the set of external top-level package names that
    appear inside ``try: ... except ImportError: ...`` blocks in
    ``py_file``. These are the graceful-degrade / silent-skip
    imports the F-AL audit targets — declarative-substrate
    discipline still requires each to be declared in pyproject (or
    vendored under ``prismpy.vendor.*``) so the substrate cannot
    quietly fall back to a placeholder code path on a fresh venv
    install that lacks the optional library."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    externals: set[str] = set()
    for node, ancestors in _walk_with_ancestors(tree):
        if isinstance(node, ast.Import):
            if not _import_under_optional_guard(node, ancestors):
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
            if not _import_under_optional_guard(node, ancestors):
                continue
            top = node.module.split(".", 1)[0]
            if _is_external(top):
                externals.add(top)
    return externals


class TestSourceImportsDeclared(unittest.TestCase):
    """Every import that appears inside a ``try: ... except
    ImportError: ...`` block across the entire ``prismpy/sources/``
    tree must resolve to either:

    - ``pyproject.toml`` ``[project] dependencies``,
    - ``pyproject.toml`` ``[project.optional-dependencies]``,
    - The in-tree vendored namespace ``prismpy.vendor.*`` (top-level
      ``prismpy`` is filtered out by ``_is_external`` and therefore
      never appears in the externals set), or
    - An entry in ``_KNOWN_LOCAL_ONLY_OPTIONAL_IMPORTS`` (currently
      empty).

    The F-AL audit broadened the scan from ``sources/climate/`` to
    every subdirectory because the same silent-skip class repeated
    three times across consecutive substrate-fix sprints; declaring
    every guarded import keeps the redesign-trigger discipline
    enforced at every retrieve path, not just the climate ones.
    """

    @classmethod
    def setUpClass(cls):
        cls.declared = {
            _normalize_for_match(d) for d in _declared_dep_names()
        }
        cls.optional = _optional_dep_names()

    def test_every_try_except_import_in_sources_is_declared(self):
        """Walk every ``.py`` module under ``sources/`` and assert the
        top-level package of each guarded external import is declared
        in pyproject.toml. Vendored prismpy.vendor.* imports are
        internal and pass the ``_is_external`` filter automatically."""
        acceptable = self.declared | self.optional

        problems: list[str] = []
        py_files = sorted(_SOURCES_DIR.rglob("*.py"))
        self.assertGreater(
            len(py_files), 0,
            f"expected at least one .py file under {_SOURCES_DIR}",
        )
        for py_file in py_files:
            for ext in sorted(_collect_guarded_imports(py_file)):
                normalized = _normalize_for_match(ext)
                if normalized in acceptable:
                    continue
                if normalized in _KNOWN_LOCAL_ONLY_OPTIONAL_IMPORTS:
                    continue
                problems.append(
                    f"{py_file.relative_to(_REPO_ROOT)}: "
                    f"imports {ext!r} inside try/except ImportError "
                    "but pyproject.toml does not declare it (and it "
                    "is not vendored under prismpy.vendor.*)"
                )
        self.assertEqual(
            problems, [],
            "Sources may not silent-skip on undeclared optional "
            "libraries. Each guarded import must appear in "
            "[project] dependencies, [project.optional-dependencies], "
            "or as a vendored package under prismpy/vendor/. "
            "F-AL closed three same-class regressions (rioxarray, "
            "cdsapi, SARRA_data_download); this pin keeps a fourth "
            "from landing silently:\n  " + "\n  ".join(problems),
        )


class TestVendoredPackagesPresent(unittest.TestCase):
    """Each vendored package under ``prismpy/vendor/`` must have the
    expected source files (and LICENSE) on disk. Without this pin a
    refactor that deletes the vendored source would be caught only at
    runtime — at first AgERA5 call — instead of at structural-test
    time. Per durable lesson #22, the wheel-contents pin lives in
    pyproject.toml ``[tool.setuptools.package-data]``; this test is
    its source-tree counterpart."""

    def test_sarra_data_download_vendor_files_present(self):
        """The vendored SARRA_data_download package must include
        ``__init__.py``, ``get_AgERA5_data.py``, and the original
        upstream LICENSE alongside the source. The first two satisfy
        the agera5.py runtime imports; LICENSE preserves the
        attribution record that the open-source contract requires."""
        vendor_root = _VENDOR_DIR / "sarra_data_download"
        expected_files = [
            "__init__.py",
            "get_AgERA5_data.py",
            "LICENSE",
        ]
        missing: list[str] = []
        for name in expected_files:
            path = vendor_root / name
            if not path.exists():
                missing.append(str(path.relative_to(_REPO_ROOT)))
        self.assertEqual(
            missing, [],
            "Vendored SARRA_data_download package is incomplete. "
            "agera5.py imports `from "
            "prismpy.vendor.sarra_data_download.get_AgERA5_data` "
            "and the broad-except carve-outs at "
            "executor.py + agera5.py + tamsat.py would surface a "
            "missing module as a fail-loud ModuleNotFoundError at "
            "first AgERA5 call. Restore the missing files:\n  "
            + "\n  ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()
