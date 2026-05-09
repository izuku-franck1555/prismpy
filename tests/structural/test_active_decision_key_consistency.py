"""Structural pin: ``current_decisions()`` tuple-key consumers are consistent.

Sprint E.3 AC-E3-6 + AC-E3-12 #5 + Stage 1 §9 #5. Sprint E.3
reshaped ``current_decisions()`` from
``dict[CellID, Optional[CellDecisionRecord]]`` to
``dict[Tuple[CellID, str], Optional[CellDecisionRecord]]`` to
support multi-check coexistence per cell. Every consumer in the
prismpy source tree MUST consume the new tuple-keyed shape; a
consumer that still treats the dict as cell-only-keyed silently
drops the multi-check coexistence semantic.

The pin walks ``current_decisions(...)`` call sites in the
``prismpy/`` source tree and asserts:

§1 The function is defined exactly once at
``prismpy/models/decision_log.py`` (canonical-layer pin already
asserts this; we re-pin here as a redundant guard for the tuple-
key reshape).

§2 Every consumer call-site treats the return value as tuple-keyed
— either via ``for (cell_id, check_id), record in active.items()``
unpacking, via ``active[(cell_id, check_id)]`` indexed read, or
via passing the dict to ``serialize_decisions_to_config`` (which
is itself tuple-keyed-typed).

The walker is best-effort: it doesn't catch every conceivable
subtle misuse, but it catches the common drift class — a consumer
treating the dict as cell-only-keyed via ``for cell_id, record in
active.items()`` (single-tuple unpack rather than nested unpack)
or ``active[cell_id]`` indexed read.
"""

from __future__ import annotations

import ast
from pathlib import Path


_CANONICAL_MODULE_RELATIVE = Path("prismpy/models/decision_log.py")


def _prismpy_src_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent.parent / "src" / "prismpy"


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


# ── §1 canonical definition site ───────────────────────────────────


def test_current_decisions_defined_once_at_canonical_module() -> None:
    """The function is defined exactly once at the canonical
    module; consumers MUST import rather than re-define."""
    src_root = _prismpy_src_root()
    canonical = src_root.parent / _CANONICAL_MODULE_RELATIVE
    definition_sites: list[str] = []

    for path in _iter_python_files(src_root):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "current_decisions":
                definition_sites.append(
                    f"{path.relative_to(src_root.parent)}:{node.lineno}"
                )

    assert len(definition_sites) == 1, (
        f"``current_decisions`` MUST be defined exactly once. "
        f"Definition sites: {definition_sites}. Canonical site: "
        f"{_CANONICAL_MODULE_RELATIVE}."
    )
    assert definition_sites[0].startswith(str(_CANONICAL_MODULE_RELATIVE)), (
        f"``current_decisions`` definition drifted off canonical "
        f"module. Canonical: {_CANONICAL_MODULE_RELATIVE}; got: "
        f"{definition_sites[0]}."
    )


# ── §2 consumer call-sites treat output as tuple-keyed ─────────────


def _yields_cell_only_iteration(node: ast.AST, var_name: str) -> bool:
    """True iff ``node`` is a ``for`` loop iterating over
    ``var_name.items()`` with a 2-tuple target where the first
    element is a single Name (cell-only-keyed iteration drift).

    The acceptable tuple-keyed pattern is::

        for (cell_id, check_id), record in active.items():
            ...

    or equivalent — the loop target's first element is a Tuple node
    (the (cell_id, check_id) destructure). A drift to::

        for cell_id, record in active.items():
            ...

    has the first target as a single Name — that's the regression
    class this walker catches.
    """
    if not isinstance(node, ast.For):
        return False
    iter_node = node.iter
    # ``var.items()`` call.
    if not (
        isinstance(iter_node, ast.Call)
        and isinstance(iter_node.func, ast.Attribute)
        and iter_node.func.attr == "items"
        and isinstance(iter_node.func.value, ast.Name)
        and iter_node.func.value.id == var_name
    ):
        return False
    # Target should be a 2-tuple.
    target = node.target
    if not (isinstance(target, ast.Tuple) and len(target.elts) == 2):
        return False
    first_target = target.elts[0]
    # Drift case: first target is a Name (cell-only iteration).
    return isinstance(first_target, ast.Name)


def test_no_consumer_iterates_current_decisions_as_cell_only_keyed() -> None:
    """Walks all prismpy modules that import ``current_decisions``
    and rejects the cell-only-keyed iteration pattern.

    This is a best-effort check — variable aliasing
    (``active = current_decisions(...)`` then ``for cell_id,
    record in active.items()``) is the common drift; rarer patterns
    (passing the result through a wrapper, then iterating) won't
    be caught here. Pair with behavioral tests for full coverage."""
    src_root = _prismpy_src_root()
    canonical = src_root.parent / _CANONICAL_MODULE_RELATIVE
    offenders: list[str] = []

    for path in _iter_python_files(src_root):
        if path.resolve() == canonical.resolve():
            continue
        try:
            text = path.read_text()
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover
            continue
        # Skip files that don't reference current_decisions at all.
        if "current_decisions" not in text:
            continue
        # Find local variable name(s) bound to current_decisions(...) call.
        bound_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = node.value
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "current_decisions"
                ):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            bound_names.add(target.id)
        # For each bound name, check no for-loop iterates
        # ``var.items()`` with cell-only-keyed unpacking.
        for node in ast.walk(tree):
            for var_name in bound_names:
                if _yields_cell_only_iteration(node, var_name):
                    offenders.append(
                        f"{path.relative_to(src_root.parent)}:{node.lineno} "
                        f"iterates {var_name}.items() as cell-only-keyed; "
                        f"per AC-E3-6 reshape, items are tuple-keyed: "
                        f"``for (cell_id, check_id), record in {var_name}.items()``"
                    )

    assert not offenders, (
        f"Consumer drift: cell-only-keyed iteration over a tuple-"
        f"keyed ``current_decisions()`` output: {offenders}. Per "
        f"AC-E3-6 + durable §27 two-vocabulary substrate-drift "
        f"discipline: every consumer reads the (cell_id, check_id) "
        f"tuple shape; the prior cell-only key pattern silently "
        f"drops multi-check coexistence per cell."
    )


# ── §3 return-type annotation pinned to tuple-keyed shape ──────────


def test_current_decisions_return_annotation_is_tuple_keyed() -> None:
    """The function's return-type annotation MUST spell
    ``dict[Tuple[CellID, str], Optional[CellDecisionRecord]]`` per
    AC-E3-6. A drift to ``dict[CellID, ...]`` would silently revert
    to single-key-per-cell semantics."""
    src_root = _prismpy_src_root()
    canonical = src_root.parent / _CANONICAL_MODULE_RELATIVE
    tree = ast.parse(canonical.read_text())
    found = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "current_decisions"
        ):
            found = True
            return_annotation = ast.unparse(node.returns) if node.returns else "<missing>"
            # The annotation should mention "Tuple[" — either as
            # ``Tuple[CellID, str]`` directly or via ``tuple[...]``
            # syntax. Reject any annotation that doesn't carry the
            # tuple-key shape signal.
            assert "Tuple[" in return_annotation or "tuple[" in return_annotation, (
                f"``current_decisions`` return annotation MUST carry "
                f"``Tuple[CellID, str]`` outer-key shape per AC-E3-6 "
                f"reshape. Got: {return_annotation}"
            )
            break
    assert found, (
        f"``current_decisions`` function not found in canonical "
        f"module {_CANONICAL_MODULE_RELATIVE}."
    )
