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


def _collect_try_finally_protected_calls(tree: ast.AST) -> Set[int]:
    """Return ``id()`` of every Call node whose Assign target is later
    closed by a matching ``<name>.close()`` in any ``finalbody`` of any
    enclosing Try in the same module.

    The walker matches the textbook tamsat pattern (smoking-gun-adjacent
    `sources/climate/tamsat.py:1010`) where the assign sits in an
    OUTER try's body and the close lives in an INNER try's finalbody::

        try:                                # outer try
            ds = xr.open_dataset(str(nc))   # this Call is protected
            try:                            # inner try
                ds_cropped = ds.where(...)
                ...
            finally:
                ds.close()                  # close lives here
        except ...:
            ...

    Restricting the match to "same Try's body and finalbody" would
    false-negative this case (and tamsat is contract §H sibling-sweep
    NO CHANGE — must pass the pin without refactor). Module-wide
    matching is acceptably permissive: a close in function F covering
    an Assign in function G is exceptionally unlikely in prismpy
    Python and would surface in code review.
    """
    # First pass: every name that has `.close()` called on it inside
    # any Try.finalbody anywhere in the module.
    closed_names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not node.finalbody:
            continue
        for fnode in _walk_stmts(node.finalbody):
            if (
                isinstance(fnode, ast.Expr)
                and isinstance(fnode.value, ast.Call)
                and isinstance(fnode.value.func, ast.Attribute)
                and fnode.value.func.attr == "close"
                and isinstance(fnode.value.func.value, ast.Name)
            ):
                closed_names.add(fnode.value.func.value.id)

    if not closed_names:
        return set()

    # Second pass: every Assign(target=Name(X), value=Call(...)) where
    # X is in the closed set. The Call (the rhs of the Assign) is the
    # node we want to protect — its position in the AST is what
    # ``_collect_target_calls`` finds.
    protected: Set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in closed_names
        ):
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
