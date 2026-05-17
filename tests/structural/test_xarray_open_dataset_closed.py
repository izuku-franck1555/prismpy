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
that share the libhdf5 close-path; ``TARGET_MODULES`` is the set of
canonical module names (whatever Python sees in ``sys.modules`` after
``import <name>``):

* ``xarray.open_dataset`` / ``xarray.open_mfdataset``
* ``netCDF4.Dataset`` (direct constructor)
* ``h5py.File`` (direct HDF5 usage)

The walker matches both the canonical receiver name (``xarray.``,
``netCDF4.``, ``h5py.``) AND any local alias bound via
``import <canonical> as <alias>`` (e.g., ``import netCDF4 as nc``
→ ``nc.Dataset(...)``). Alias tracking is per-scope — see the
"Alias tracking" section below — and walks both module-level
imports and function-local imports (the common lazy-import idiom
``try: import netCDF4 as nc except ImportError: ...``).

Alias tracking (alias-extension amendment):
production code at ``translators/acea/translator.py:1264, 1454, 1997``
binds ``import netCDF4 as nc`` inside a function-local
``try/except`` and then writes ``nc.Dataset(...)`` later in the same
method (5 sites: L1319, L1625, L2006, L2015, L2043). The receiver
of these constructor calls is ``ast.Name(id="nc")``. Pre-extension
the walker only matched receivers whose ``id`` was one of the
canonical names declared in ``TARGET_CALLS`` (e.g., ``netCDF4``);
``"nc"`` was not in that set, so the walker returned zero hits for
every production ``nc.Dataset`` call and the pin assertion passed
vacuously. A future regression to the smoking-gun module that
uses ``import xarray as xx`` (or any non-canonical alias name)
would also slip through. The alias-extension walks every
``ast.Import`` and ``ast.ImportFrom`` reachable without crossing a
function boundary in the relevant scope, maps each
``import <canonical> as <alias>`` to ``alias → canonical``, and
resolves the receiver name through that map before checking
``TARGET_CALLS`` membership. Function-local imports apply only
within their enclosing function; module-level imports inherit
into every function in the module. Per F-DL Pin DL-1 cycle-6
alias-tracking pattern (caught BL-1 codex R-independent at
``observed_values_writer.py:499``).

Empirical count at the alias-extension commit (verified by
post-extension run against the source tree):
* ``xr.open_dataset`` — 3 sites (smoking gun + hwsd + tamsat),
  all safely managed (with / try-finally).
* ``nc.Dataset`` — 5 sites in ``translators/acea/translator.py``
  (newly visible after alias tracking). Sibling-swept per durable
  §20: each wrapped in a ``with`` block in the same commit as the
  walker extension (or whitelisted with rationale per scope-amend
  decision).
* ``xr.open_mfdataset`` / ``h5py.File`` / canonical
  ``xarray.open_dataset`` / ``netCDF4.Dataset`` — 0 sites in
  production. The walker defends against future introduction of
  any pattern through any alias.

Anti-mutation probes at the bottom of this file flex the walker
against synthetic source strings so regressions in the walker itself
are caught alongside regressions in the prismpy source it scans.
Cycle-alias adds Name-receiver alias probes for both module-level
``import xarray as xa`` AND function-local
``import netCDF4 as nc`` patterns.

Per F-DP contract LOCKED cycle-4 §Z.1 + §X.3 + §C Pin DP-1 base
logic + alias-extension CORRECTION + infrastructure_rules.md
durable §20 sibling-sweep + §24 canonical source.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pytest


# ── Source tree root ───────────────────────────────────────────────


PRISMPY_SRC = Path(__file__).resolve().parents[2] / "src" / "prismpy"


# ── Target call patterns ───────────────────────────────────────────


# Each entry = (canonical_module_name, attribute_name). Walker
# matches ``Attribute(value=Name(id=R), attr=attribute_name)`` ASTs
# inside Call nodes, AFTER resolving ``R`` through the per-scope
# import-alias map (so ``nc.Dataset(...)`` resolves to
# ``(netCDF4, Dataset)`` when ``import netCDF4 as nc`` is in scope).
# Per §X.3 cycle-2 expansion: 4 primitive surfaces that share the
# libhdf5 close-path. Aliases are not duplicated as separate entries
# — the alias map handles that — so adding a new TARGET_CALLS entry
# is a one-line change.
TARGET_CALLS: Tuple[Tuple[str, str], ...] = (
    ("xarray", "open_dataset"),
    ("xarray", "open_mfdataset"),
    ("netCDF4", "Dataset"),
    ("h5py", "File"),
)


# Canonical module names referenced by TARGET_CALLS. Derived once so
# the alias collector can short-circuit on imports whose target
# module is irrelevant to Pin DP-1's scope.
TARGET_MODULES: Set[str] = {module for module, _ in TARGET_CALLS}


# Format: "src/prismpy/<relative path>.py:<lineno>" strings. Empty at
# PR1 per §X.3; any future entry MUST cite the why (e.g., function
# returns the dataset for the caller to manage) in the commit message
# adding the entry.
WHITELIST: Set[str] = set()


# ── AST helpers ────────────────────────────────────────────────────


def _walk_skipping_nested_scopes(scope_node: ast.AST) -> Iterator[ast.AST]:
    """Yield every descendant of ``scope_node`` EXCEPT nodes inside a
    nested ``FunctionDef`` / ``AsyncFunctionDef``. The ``scope_node``
    itself is yielded so callers can match against the scope's own
    header attributes when needed.

    Used by ``_collect_import_aliases_in_scope`` so a module-level
    walk does not absorb function-local imports (which belong to the
    function scope, not the module scope), and so a function-local
    walk does not absorb imports from nested helper functions.

    The walk descends into all other compound nodes (``If``, ``Try``,
    ``With``, ``For``, ``While``, ``ClassDef`` body for class-level
    statements). The intent is "everything reachable from this scope
    without crossing a function boundary".
    """
    yield scope_node
    for child in ast.iter_child_nodes(scope_node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield from _walk_skipping_nested_scopes(child)


def _collect_import_aliases_in_scope(
    scope_node: ast.AST,
) -> Dict[str, str]:
    """Walk ``scope_node`` (without crossing nested function
    boundaries) and return ``{local_name → canonical_module_name}``
    for every ``import <canonical> [as <alias>]`` and
    ``from <pkg> import <canonical> [as <alias>]`` that binds a name
    referring to one of ``TARGET_MODULES``.

    Two import idioms recognized:

    1. **Plain import**: ``import xarray`` → ``{"xarray": "xarray"}``;
       ``import netCDF4 as nc`` → ``{"nc": "netCDF4"}``.
       Sub-modules (``import xarray.something as xr``) are walked
       through the root: the canonical key tracked is the root
       module so the alias resolves to the same name Pin DP-1
       enforces. Only roots in ``TARGET_MODULES`` produce an entry.

    2. **From-import of submodule**:
       ``from xarray import open_dataset`` is rare in production but
       would bind the function ``open_dataset`` directly to a Name
       in the local scope, sidestepping the receiver pattern Pin
       DP-1 matches against. This case is NOT tracked here — it is
       documented in the limitations section of the module
       docstring; if it appears in production, extend the walker
       with a "function-callable alias" map.

    The identity mapping (``"xarray" → "xarray"``) is intentional:
    it lets the matcher resolve a bare ``import xarray; xarray.
    open_dataset(...)`` site the same way it resolves an aliased
    one, without a separate code path. ``import netCDF4`` (no alias)
    therefore still works even though there are no such sites in
    production today.

    Per alias-extension CORRECTION (F-DL Pin DL-1 cycle-6 precedent).
    """
    aliases: Dict[str, str] = {}
    for node in _walk_skipping_nested_scopes(scope_node):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            # ``alias.name`` is the dotted module path; the root is
            # what ``sys.modules`` keys against and what Pin DP-1
            # asserts against.
            root_canonical = alias.name.split(".", 1)[0]
            if root_canonical not in TARGET_MODULES:
                continue
            local_name = alias.asname or alias.name
            # Strip dotted-form local names (``import xarray.foo``
            # binds ``xarray`` in the local scope, not ``xarray.foo``).
            local_name = local_name.split(".", 1)[0]
            aliases[local_name] = root_canonical
    return aliases


def _is_target_call(
    node: ast.AST,
    module_aliases: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[str, str]]:
    """If ``node`` is a Call whose ``func`` is ``Attribute(Name(R), Y)``
    and ``(canonical_of(R), Y)`` is in ``TARGET_CALLS``, return that
    tuple. Else None.

    ``canonical_of(R)`` resolves ``R`` through ``module_aliases``
    (e.g., ``nc → netCDF4`` when ``import netCDF4 as nc`` is in
    scope). When ``module_aliases`` is None or missing the receiver,
    the receiver name is used verbatim — preserving direct matches
    on canonical names (``netCDF4.Dataset(...)`` with no aliasing).
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name):
        return None
    receiver = func.value.id
    canonical_receiver = (
        module_aliases.get(receiver, receiver) if module_aliases else receiver
    )
    pair = (canonical_receiver, func.attr)
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
    list when every TARGET_CALLS site is safely managed.

    The matcher runs per-scope so each Call is checked against the
    alias map visible at its position: module-level imports are
    inherited by every function in the module, function-local
    imports apply only inside that function (the common lazy-import
    idiom). Per alias-extension CORRECTION.
    """
    try:
        tree = ast.parse(source, filename=source_label)
    except SyntaxError as exc:  # noqa: BLE001 — surface parse errors
        pytest.fail(f"AST parse failed for {source_label}: {exc}")

    # Module-level alias map — shared by every function in the file.
    module_aliases = _collect_import_aliases_in_scope(tree)

    # Memoise the merged alias map per function scope. The scope
    # node's ``id()`` is stable for the lifetime of the tree.
    scope_aliases_cache: Dict[int, Dict[str, str]] = {
        id(tree): module_aliases,
    }

    def _aliases_for_scope(scope_node: ast.AST) -> Dict[str, str]:
        key = id(scope_node)
        cached = scope_aliases_cache.get(key)
        if cached is not None:
            return cached
        # Function-local imports inherit + may shadow module-level
        # entries (the same alias name rebound inside the function).
        local = _collect_import_aliases_in_scope(scope_node)
        merged = {**module_aliases, **local}
        scope_aliases_cache[key] = merged
        return merged

    # Collect every TARGET_CALLS-matching Call (after alias
    # resolution against the Call's enclosing scope) and its safety
    # status.
    matches: List[Tuple[ast.Call, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        scope = _enclosing_scope_by_lineno(node, tree)
        aliases = _aliases_for_scope(scope)
        if _is_target_call(node, aliases) is not None:
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
    or whitelist). Currently expected to PASS — the 5 known sites
    (3 xarray + 2 netCDF4-via-alias) are all wrapped:

    * `vendor/sarra_data_download/get_AgERA5_data.py:273` — with
      (AC-DP-1a smoking gun)
    * `sources/soil/hwsd.py:438` — with (AC-DP-1a sibling)
    * `sources/climate/tamsat.py:1010` — try/finally + ds.close()
    * `translators/acea/translator.py:1319` — with (alias-extension
      sibling-sweep; ``import netCDF4 as nc`` newly tracked)
    * `translators/acea/translator.py:1625` — with (alias-extension
      sibling-sweep; same alias)

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
    # Resolve the synthetic's ``import xarray as xr`` alias so the
    # matcher recognises ``xr.open_dataset`` as a canonical target.
    tree = ast.parse(_BARE_OPEN_FORM, filename=label)
    aliases = _collect_import_aliases_in_scope(tree)
    lineno = next(
        node.lineno for node in ast.walk(tree)
        if _is_target_call(node, aliases) is not None
    )
    WHITELIST.add(f"{label}:{lineno}")
    try:
        violations = _violations_in(_BARE_OPEN_FORM, label)
        assert violations == [], (
            f"Whitelist did not suppress violation: {violations}"
        )
    finally:
        WHITELIST.discard(f"{label}:{lineno}")


# ── Alias-extension anti-mutation probes ───────────────────────────


_BARE_NETCDF4_ALIAS_FORM = """
import netCDF4 as nc
def f(p):
    ds = nc.Dataset(p, 'r')
    return ds.variables
"""

_WITH_NETCDF4_ALIAS_FORM = """
import netCDF4 as nc
def f(p):
    with nc.Dataset(p, 'r') as ds:
        return list(ds.variables)
"""

_BARE_XARRAY_NONSTANDARD_ALIAS_FORM = """
import xarray as xa
def f(p):
    ds = xa.open_dataset(p)
    return ds.values
"""

_FUNCTION_LOCAL_BARE_NETCDF4_ALIAS_FORM = """
def f(p):
    import netCDF4 as nc
    ds = nc.Dataset(p, 'r')
    return ds.variables
"""

_TRY_EXCEPT_BARE_NETCDF4_ALIAS_FORM = """
def f(p):
    try:
        import netCDF4 as nc
    except ImportError:
        return None
    ds = nc.Dataset(p, 'r')
    return ds.variables
"""

_FUNCTION_LOCAL_WITH_NETCDF4_ALIAS_FORM = """
def f(p):
    import netCDF4 as nc
    with nc.Dataset(p, 'r') as ds:
        return list(ds.variables)
"""

_NESTED_FUNCTION_ALIAS_DOES_NOT_LEAK = """
def outer(p):
    import netCDF4 as nc
    with nc.Dataset(p, 'r') as ds:
        return list(ds.variables)

def sibling(p):
    # ``nc`` is NOT in scope here; this `nc.Dataset(...)` would be
    # a NameError at runtime, but for the walker it must still be
    # treated as a Name receiver with no resolvable canonical
    # mapping (i.e., not a TARGET_CALLS match, NOT a false-positive
    # leak from outer()).
    ds = nc.Dataset(p, 'r')
    return ds.variables
"""


def test_anti_mutation_bare_netcdf4_alias_fails() -> None:
    """Alias-extension primary regression: ``import netCDF4 as nc``
    followed by a bare ``ds = nc.Dataset(...)`` MUST violate. The
    pre-extension walker matched only canonical receivers
    (``netCDF4.Dataset``), so this synthetic — and the production
    sites at ``translators/acea/translator.py`` it mirrors —
    silently passed. A regression that drops alias resolution
    would re-introduce that silent-pass and this test would fail
    loud."""
    violations = _violations_in(
        _BARE_NETCDF4_ALIAS_FORM, "synthetic/bare_nc_alias.py"
    )
    assert len(violations) == 1, violations
    assert "nc.Dataset" in violations[0]
    assert "synthetic/bare_nc_alias.py" in violations[0]


def test_anti_mutation_with_netcdf4_alias_passes() -> None:
    """The ``with`` form of an aliased ``nc.Dataset(...)`` site is
    safely managed (case (a)). Walker reports zero violations.
    Pins that the alias resolution does not break the ``with``
    detection path."""
    assert _violations_in(
        _WITH_NETCDF4_ALIAS_FORM, "synthetic/with_nc_alias.py"
    ) == []


def test_anti_mutation_bare_xarray_nonstandard_alias_fails() -> None:
    """A non-canonical xarray alias (``import xarray as xa``) MUST be
    resolved the same as the canonical ``xr``. Pre-extension the
    walker had ``("xr", "open_dataset")`` hard-coded in TARGET_CALLS
    and would silently pass ``xa.open_dataset(...)``. The
    alias-aware matcher resolves ``xa → xarray`` and fires the
    violation."""
    violations = _violations_in(
        _BARE_XARRAY_NONSTANDARD_ALIAS_FORM, "synthetic/bare_xa.py"
    )
    assert len(violations) == 1, violations
    assert "xa.open_dataset" in violations[0]


def test_anti_mutation_function_local_bare_netcdf4_alias_fails() -> None:
    """Function-local ``import netCDF4 as nc`` (the production lazy
    -import idiom at ``translators/acea/translator.py:1264``) MUST
    still resolve ``nc → netCDF4`` within the same function body.
    Confirms ``_collect_import_aliases_in_scope`` walks function-
    local imports, not just module-level ones."""
    violations = _violations_in(
        _FUNCTION_LOCAL_BARE_NETCDF4_ALIAS_FORM,
        "synthetic/fn_local_nc.py",
    )
    assert len(violations) == 1, violations
    assert "nc.Dataset" in violations[0]


def test_anti_mutation_try_except_bare_netcdf4_alias_fails() -> None:
    """The production idiom at ``translators/acea/translator.py:1997``
    wraps the import in ``try/except ImportError`` (defensive against
    missing netCDF4). The walker's scope walk MUST descend into the
    try-body so the alias is collected even when conditionally
    imported. Empirically: the 3 ``import netCDF4 as nc`` sites in
    production all use this pattern; if the walker skips try-bodies
    they all bypass."""
    violations = _violations_in(
        _TRY_EXCEPT_BARE_NETCDF4_ALIAS_FORM,
        "synthetic/try_except_nc.py",
    )
    assert len(violations) == 1, violations
    assert "nc.Dataset" in violations[0]


def test_anti_mutation_function_local_with_netcdf4_alias_passes() -> None:
    """Function-local alias + ``with``-wrapped call is safely managed.
    Pins that alias resolution does not turn a legitimate
    ``with nc.Dataset(...)`` into a false positive."""
    assert _violations_in(
        _FUNCTION_LOCAL_WITH_NETCDF4_ALIAS_FORM,
        "synthetic/fn_local_with_nc.py",
    ) == []


def test_nested_function_alias_does_not_leak() -> None:
    """Per-function scoping: an alias defined inside ``outer()`` MUST
    NOT be visible inside ``sibling()``. The walker collects aliases
    per enclosing scope; a leak from ``outer`` into ``sibling`` would
    cause ``nc.Dataset`` in ``sibling`` to resolve to
    ``netCDF4.Dataset`` and report a violation. Without the leak,
    the bare ``nc.Dataset`` in ``sibling`` is treated as an unknown
    receiver and silently passes (the runtime NameError is a
    separate Python concern, not a Pin DP-1 invariant).

    This pin guards against a regression where the alias collector
    walks across function boundaries and creates ghost mappings."""
    violations = _violations_in(
        _NESTED_FUNCTION_ALIAS_DOES_NOT_LEAK,
        "synthetic/nested_no_leak.py",
    )
    # ``outer`` is `with`-protected → 0 violations from there.
    # ``sibling`` has no in-scope alias for ``nc`` → walker does
    # not recognise it as a TARGET_CALLS site → 0 violations.
    assert violations == [], (
        f"Alias leaked across function boundaries: {violations}"
    )
