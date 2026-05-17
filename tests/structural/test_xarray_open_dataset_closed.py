"""F-DP Pin DP-1 — every resource-handle call in prismpy is safely closed.

The AgERA5 SIGSEGV class root-caused at
`prismpy/src/prismpy/vendor/sarra_data_download/get_AgERA5_data.py:273`
was an `xr.open_dataset(...)` reassigned each iteration without
explicit close. The handle leaked through xarray's
``CachingFileManager`` LRU; at iteration ~700-800 the LRU's eviction
path triggered libhdf5 1.14.6 close-time state corruption and the
worker SIGSEGVed. AC-DP-1a closes the smoking gun + the
`prismpy/src/prismpy/sources/soil/hwsd.py:438` sibling by wrapping
the open in a `with` block.

Pin DP-1 is the structural net that catches the **class** of bug:
every direct construction of an HDF5/netCDF resource handle in
prismpy source MUST be either

* (a) the iterable of a `with` statement (handle released by
      ``__exit__`` on every exit path), OR
* (b) assigned in the body of a `try` statement with a matching
      ``finally:`` block calling ``<name>.close()`` (the tamsat
      pattern at `sources/climate/tamsat.py:1010` — functionally
      equivalent to `with` for handle release), OR
* (c) listed in ``WHITELIST`` with a committed rationale.

TARGET_CALLS covers the four primitive resource-acquisition surfaces
that share the libhdf5 close-path:

* ``xr.open_dataset`` / ``xarray.open_dataset``
* ``xr.open_mfdataset`` / ``xarray.open_mfdataset``
* ``netCDF4.Dataset`` (direct constructor)
* ``h5py.File`` (direct HDF5 usage)

Empirical count at PR1 (verified by builder grounding):
* `xr.open_dataset`: 3 sites (smoking gun + hwsd + tamsat)
* All other TARGET_CALLS patterns: 0 sites in production code
The walker defends against future introduction of any pattern.

Anti-mutation probes at the bottom of this file flex the walker
against synthetic source strings so regressions in the walker itself
are caught alongside regressions in the prismpy source it scans.

Per F-DP contract LOCKED cycle-4 §Z.1 + §X.3 + §C Pin DP-1 base
logic + infrastructure_rules.md durable §24.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import pytest


# ── Source tree root ───────────────────────────────────────────────


PRISMPY_SRC = Path(__file__).resolve().parents[2] / "src" / "prismpy"


# ── Target call patterns ───────────────────────────────────────────


# Each entry = (module_name, attribute_name). Walker matches
# `Attribute(value=Name(id=module_name), attr=attribute_name)` ASTs
# inside Call nodes. Per §X.3 cycle-2 expansion: 4 primitive surfaces
# that share the libhdf5 close-path.
TARGET_CALLS: Tuple[Tuple[str, str], ...] = (
    ("xr", "open_dataset"),
    ("xarray", "open_dataset"),
    ("xr", "open_mfdataset"),
    ("xarray", "open_mfdataset"),
    ("netCDF4", "Dataset"),
    ("h5py", "File"),
)


# Format: "src/prismpy/<relative path>.py:<lineno>" strings. Empty at
# PR1 per §X.3; any future entry MUST cite the why (e.g., function
# returns the dataset for the caller to manage) in the commit message
# adding the entry.
WHITELIST: Set[str] = set()


# ── AST helpers ────────────────────────────────────────────────────


def _is_target_call(node: ast.AST) -> Optional[Tuple[str, str]]:
    """If ``node`` is a Call whose ``func`` is ``Attribute(Name(X), Y)``
    and ``(X, Y)`` is in ``TARGET_CALLS``, return that tuple. Else None.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name):
        return None
    pair = (func.value.id, func.attr)
    return pair if pair in TARGET_CALLS else None


def _walk_stmts(stmts: List[ast.stmt]) -> Iterator[ast.AST]:
    """Walk every sub-node of every stmt in ``stmts``. Used to scan a
    Try's body or finalbody independently of the rest of the tree."""
    for stmt in stmts:
        for node in ast.walk(stmt):
            yield node


def _collect_with_protected_calls(tree: ast.AST) -> Set[int]:
    """Return ``id()`` of every Call node that is the ``context_expr``
    of a ``WithItem`` inside a ``With`` (i.e., wrapped by `with`)."""
    protected: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    protected.add(id(item.context_expr))
    return protected


def _enclosing_scope_by_lineno(node: ast.AST, tree: ast.AST) -> ast.AST:
    """Return the smallest ``FunctionDef``/``AsyncFunctionDef`` whose
    ``[lineno, end_lineno]`` range contains ``node.lineno``. Falls back
    to the module if no enclosing function exists.

    Uses lineno containment instead of a parent-pointer chain so the
    walker stays a pure read of the parsed ``ast.AST`` (no in-place
    mutation of nodes). The ``end_lineno`` attribute is Python 3.8+;
    prismpy's interpreter floor is 3.10 so always available.
    """
    best: ast.AST = tree
    best_span = float("inf")
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(fn, "end_lineno", fn.lineno)
        if fn.lineno <= node.lineno <= end:
            span = end - fn.lineno
            if span < best_span:
                best = fn
                best_span = span
    return best


def _collect_try_finally_protected_calls(tree: ast.AST) -> Set[int]:
    """Function-scope-local matching (cycle-1.5 SF-2 codex absorption):
    for each ``Assign(target=Name(X), value=TARGET_CALL)``, find its
    enclosing ``FunctionDef`` (or module). Collect ``X.close()``
    targets from any ``Try.finalbody`` whose enclosing scope is the
    SAME function/module. Mark the Assign's Call as protected only
    when ``X`` appears in that scope-local close set.

    Pattern matched (the textbook tamsat case at
    ``sources/climate/tamsat.py:1010``)::

        def phase2_convert_nc_to_tif(...):
            try:                                # outer try
                ds = xr.open_dataset(str(nc))   # this Call protected
                try:                            # inner try (same fn)
                    ds_cropped = ds.where(...)
                    ...
                finally:
                    ds.close()                  # close in same fn
            except ...:
                ...

    Pre-SF-2 the walker used module-wide name matching, which would
    false-accept this cross-function pattern::

        def f(p):
            ds = xr.open_dataset(p)   # bare; NOT protected
            return ds.values
        def g():
            try: pass
            finally:
                ds.close()            # unrelated close on same name

    Scoping the close lookup to the Assign's enclosing function fixes
    both the false-negative (tamsat nested-try) and the false-positive
    (cross-function close) classes.
    """
    protected: Set[int] = set()

    # Memoise (scope_id → close_names) so the scope-walk cost is O(N)
    # not O(N²) on modules with many Assigns.
    scope_to_closed: dict = {}

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue

        scope = _enclosing_scope_by_lineno(node, tree)
        scope_key = id(scope)
        if scope_key not in scope_to_closed:
            closed: Set[str] = set()
            for tnode in ast.walk(tree):
                if not isinstance(tnode, ast.Try) or not tnode.finalbody:
                    continue
                if _enclosing_scope_by_lineno(tnode, tree) is not scope:
                    continue
                for fnode in _walk_stmts(tnode.finalbody):
                    if (
                        isinstance(fnode, ast.Expr)
                        and isinstance(fnode.value, ast.Call)
                        and isinstance(fnode.value.func, ast.Attribute)
                        and fnode.value.func.attr == "close"
                        and isinstance(fnode.value.func.value, ast.Name)
                    ):
                        closed.add(fnode.value.func.value.id)
            scope_to_closed[scope_key] = closed

        if node.targets[0].id in scope_to_closed[scope_key]:
            protected.add(id(node.value))

    return protected


def _violations_in(source: str, source_label: str) -> List[str]:
    """Return human-readable violation strings for ``source``. Empty
    list when every TARGET_CALLS site is safely managed."""
    try:
        tree = ast.parse(source, filename=source_label)
    except SyntaxError as exc:  # noqa: BLE001 — surface parse errors
        pytest.fail(f"AST parse failed for {source_label}: {exc}")

    # Collect every TARGET_CALLS-matching Call and its safety status
    matches: List[Tuple[ast.Call, int]] = []
    for node in ast.walk(tree):
        if _is_target_call(node) is not None and isinstance(node, ast.Call):
            matches.append((node, node.lineno))
    if not matches:
        return []

    with_protected = _collect_with_protected_calls(tree)
    try_protected = _collect_try_finally_protected_calls(tree)

    violations: List[str] = []
    for call_node, lineno in matches:
        wl_key = f"{source_label}:{lineno}"
        if wl_key in WHITELIST:
            continue
        if id(call_node) in with_protected:
            continue
        if id(call_node) in try_protected:
            continue
        try:
            expr = ast.unparse(call_node)
        except Exception:  # noqa: BLE001 — formatting only
            expr = "Call(...)"
        violations.append(
            f"  {source_label}:{lineno} — {expr} — must be wrapped by "
            "`with` OR assigned-and-closed in `try/finally`"
        )
    return violations


# ── Anti-vacuous guards ────────────────────────────────────────────


def test_target_calls_tuple_non_empty() -> None:
    """``TARGET_CALLS`` MUST cover at least one resource-acquisition
    primitive. An empty tuple would make the main scan trivially
    return no violations."""
    assert TARGET_CALLS, (
        "TARGET_CALLS is empty — Pin DP-1 cannot find anything to "
        "assert against. Add at least one (module, attr) pair."
    )


def test_prismpy_source_root_exists() -> None:
    """The walker's source root MUST resolve to an existing directory
    containing Python files. If the repo layout moves and the path
    derivation here goes stale, the main scan would silently scan
    nothing and pass."""
    assert PRISMPY_SRC.is_dir(), (
        f"Expected prismpy source root at {PRISMPY_SRC!r}; not a directory. "
        "Pin DP-1's source-root derivation needs updating."
    )
    py_files = list(PRISMPY_SRC.rglob("*.py"))
    assert py_files, (
        f"No .py files under {PRISMPY_SRC!r}; walker would scan nothing."
    )


# ── Main invariant ─────────────────────────────────────────────────


def test_all_target_calls_in_prismpy_are_safely_managed() -> None:
    """Pin DP-1 main assertion: every TARGET_CALLS site under
    ``prismpy/src/prismpy/`` MUST be safely managed (case a / b /
    or whitelist). Currently expected to PASS — the 3 known sites
    are all wrapped post-AC-DP-1a:

    * `vendor/sarra_data_download/get_AgERA5_data.py:273` — with
    * `sources/soil/hwsd.py:438` — with
    * `sources/climate/tamsat.py:1010` — try/finally + ds.close()

    Anti-mutation: temporarily removing the `with` from any of these
    sites makes this test fail with a clear file:line + call-text
    citation. The whitelist starts empty + the bar for adding to it
    is a commit-message justification.
    """
    violations: List[str] = []
    for path in sorted(PRISMPY_SRC.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(PRISMPY_SRC.parent.parent)
        violations.extend(_violations_in(source, str(rel)))

    assert not violations, (
        "F-DP Pin DP-1: prismpy source contains TARGET_CALLS "
        "resource-handle acquisitions that are NOT safely managed:\n"
        + "\n".join(violations)
        + f"\n\nTARGET_CALLS: {TARGET_CALLS}"
        + f"\nWHITELIST: {sorted(WHITELIST)}"
    )


# ── Anti-mutation probes (synthetic source strings) ────────────────


_WITH_FORM = """
import xarray as xr
def f(p):
    with xr.open_dataset(p) as ds:
        return ds
"""

_TRY_FINALLY_FORM = """
import xarray as xr
def f(p):
    try:
        ds = xr.open_dataset(p)
        x = ds.values
    finally:
        ds.close()
    return x
"""

_BARE_OPEN_FORM = """
import xarray as xr
def f(p):
    ds = xr.open_dataset(p)
    return ds.values
"""

_BARE_NETCDF4_FORM = """
import netCDF4
def f(p):
    ds = netCDF4.Dataset(p, 'r')
    return ds.variables
"""

_BARE_MFDATASET_FORM = """
import xarray as xr
def f(pat):
    ds = xr.open_mfdataset(pat)
    return ds
"""

_BARE_H5PY_FORM = """
import h5py
def f(p):
    fh = h5py.File(p, 'r')
    return fh['x']
"""

_CROSS_FUNCTION_BARE_FORM = """
import xarray as xr

def f(p):
    ds = xr.open_dataset(p)
    return ds.values


def g():
    try:
        pass
    finally:
        ds.close()
"""


def test_anti_mutation_with_form_passes() -> None:
    """The ``with`` form satisfies case (a). Walker reports zero
    violations."""
    assert _violations_in(_WITH_FORM, "synthetic/with.py") == []


def test_anti_mutation_try_finally_form_passes() -> None:
    """The ``try`` + ``finally: ds.close()`` form satisfies case (b).
    Walker reports zero violations."""
    assert _violations_in(_TRY_FINALLY_FORM, "synthetic/try.py") == []


def test_anti_mutation_bare_open_fails() -> None:
    """A bare ``ds = xr.open_dataset(...)`` with no `with` and no
    `try/finally: ds.close()` MUST violate. This is the exact
    AC-DP-1a smoking gun shape."""
    violations = _violations_in(_BARE_OPEN_FORM, "synthetic/bare.py")
    assert len(violations) == 1, violations
    assert "xr.open_dataset" in violations[0]
    assert "synthetic/bare.py" in violations[0]


def test_anti_mutation_bare_netcdf4_fails() -> None:
    """A bare ``netCDF4.Dataset(...)`` constructor call MUST violate
    — same close-path class as xarray-via-netCDF4."""
    violations = _violations_in(_BARE_NETCDF4_FORM, "synthetic/nc4.py")
    assert len(violations) == 1, violations
    assert "netCDF4.Dataset" in violations[0]


def test_anti_mutation_bare_mfdataset_fails() -> None:
    """``xr.open_mfdataset`` is a TARGET_CALLS entry per §X.3 cycle-2
    scope expansion. A bare assignment without `with`/`try/finally`
    MUST violate."""
    violations = _violations_in(_BARE_MFDATASET_FORM, "synthetic/mf.py")
    assert len(violations) == 1, violations
    assert "xr.open_mfdataset" in violations[0]


def test_anti_mutation_bare_h5py_fails() -> None:
    """``h5py.File`` is a TARGET_CALLS entry per §X.3 cycle-2 scope
    expansion. A bare assignment without `with`/`try/finally` MUST
    violate."""
    violations = _violations_in(_BARE_H5PY_FORM, "synthetic/h5.py")
    assert len(violations) == 1, violations
    assert "h5py.File" in violations[0]


def test_anti_mutation_cross_function_close_does_not_protect() -> None:
    """Cycle-1.5 SF-2 codex absorption: a ``<name>.close()`` in
    function G MUST NOT protect a bare Assign of the same name in
    function F. Pre-SF-2 the walker matched close-names module-wide
    and would have falsely accepted this case; scope-local matching
    fires correctly on the bare Assign in F."""
    violations = _violations_in(
        _CROSS_FUNCTION_BARE_FORM, "synthetic/cross.py"
    )
    assert len(violations) == 1, violations
    assert "xr.open_dataset" in violations[0]
    assert "synthetic/cross.py" in violations[0]


def test_anti_mutation_whitelist_suppresses() -> None:
    """When the violator's file:line is in ``WHITELIST``, the walker
    suppresses it. Confirms the whitelist mechanism actually works
    (so a future legitimate carve-out can be added)."""
    label = "synthetic/whitelisted.py"
    # Find the lineno of the bare call to compute the whitelist key.
    tree = ast.parse(_BARE_OPEN_FORM, filename=label)
    lineno = next(
        node.lineno for node in ast.walk(tree)
        if _is_target_call(node) is not None
    )
    WHITELIST.add(f"{label}:{lineno}")
    try:
        violations = _violations_in(_BARE_OPEN_FORM, label)
        assert violations == [], (
            f"Whitelist did not suppress violation: {violations}"
        )
    finally:
        WHITELIST.discard(f"{label}:{lineno}")
