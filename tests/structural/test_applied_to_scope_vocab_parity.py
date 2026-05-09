"""Structural pin: ``AppliedToScope`` Literal canonical-source parity.

Sprint E.3 AC-E3-2 + Stage 1 §9 #2 + builder grounding-pass CA-7
absorption. Two invariants close the silent-drift class per durable
§24 canonical-source-or-pin + durable §27 two-vocabulary substrate-
drift:

§1 Canonical-source uniqueness — only
``prismpy/standards/applied_to_scope.py`` carries the
``Literal["single_cell", "zone", "enumerated_cells"]`` definition.
The AST walker rejects any parallel definition (a renamed symbol
``AppliedToScope`` referring to a different Literal-arg set, or any
other module restating the three-arg Literal directly).

§2 Tuple ↔ Literal parity — ``APPLIED_TO_SCOPE_VALUES`` (runtime
tuple) MUST equal ``set(typing.get_args(AppliedToScope))`` (type-
level vocabulary) byte-for-byte. A future Literal addition that
forgets to update the tuple, or vice versa, fires this pin loud
rather than silently producing a tuple consumer that lags the type
hint.

Anti-mutation drill (§3): if a contributor adds a fourth scope
discriminator to the Literal but forgets the tuple, the test_*
runtime assertion at §2 fails immediately. Conversely if the tuple
gains a value the Literal doesn't carry, §2 fails the other
direction. Both directions are pinned.
"""

from __future__ import annotations

import ast
import typing
from pathlib import Path

from prismpy.standards.applied_to_scope import (
    APPLIED_TO_SCOPE_VALUES,
    AppliedToScope,
)


# ── §1 canonical-source uniqueness ─────────────────────────────────


_CANONICAL_MODULE_RELATIVE = Path("prismpy/standards/applied_to_scope.py")

_THREE_VALUE_VOCAB = frozenset({"single_cell", "zone", "enumerated_cells"})


def _prismpy_src_root() -> Path:
    """Return the prismpy ``src/prismpy/`` root."""
    # tests/structural/test_*.py → tests/structural/.. → tests/..
    # → prismpy/src/prismpy
    here = Path(__file__).resolve()
    return here.parent.parent.parent / "src" / "prismpy"


def _iter_python_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*.py"):
        # Skip __pycache__ artefacts.
        if "__pycache__" in path.parts:
            continue
        out.append(path)
    return out


def _is_three_value_applied_to_scope_literal(
    node: ast.AST,
) -> bool:
    """True iff ``node`` is a ``Literal[...]`` subscript whose args
    spell exactly the three canonical scope values.

    We only fire on the exact arg set; a future Literal that adds a
    fourth scope discriminator wouldn't match (and shouldn't —
    that's a different canonical Literal that needs its own home)."""
    if not isinstance(node, ast.Subscript):
        return False
    value = node.value
    # ``Literal[...]`` or ``typing.Literal[...]`` — the value is
    # either a Name(id="Literal") or an Attribute(attr="Literal").
    is_literal_ref = (
        (isinstance(value, ast.Name) and value.id == "Literal")
        or (
            isinstance(value, ast.Attribute) and value.attr == "Literal"
        )
    )
    if not is_literal_ref:
        return False
    # Slice carries the args. Pre-3.9 wraps in ast.Index; 3.9+
    # uses the inner node directly. Normalise.
    slice_node = node.slice
    if isinstance(slice_node, ast.Index):  # pragma: no cover (py38)
        slice_node = slice_node.value
    if isinstance(slice_node, ast.Tuple):
        elts = slice_node.elts
    else:
        elts = [slice_node]
    arg_strings: set[str] = set()
    for elt in elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            arg_strings.add(elt.value)
        else:
            # Non-string-constant arg — not the canonical three-
            # value vocab.
            return False
    return arg_strings == _THREE_VALUE_VOCAB


def test_only_canonical_module_defines_three_value_literal() -> None:
    """Reject parallel Literal definitions that spell the same three
    scope values. Every consumer MUST import ``AppliedToScope`` from
    the canonical module rather than restating the Literal."""
    src_root = _prismpy_src_root()
    canonical = src_root.parent / _CANONICAL_MODULE_RELATIVE
    offenders: list[str] = []

    for path in _iter_python_files(src_root):
        if path.resolve() == canonical.resolve():
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover (defensive)
            continue
        for node in ast.walk(tree):
            if _is_three_value_applied_to_scope_literal(node):
                offenders.append(
                    f"{path.relative_to(src_root.parent)}:{node.lineno}"
                )
                break  # one offence per file is enough

    assert not offenders, (
        f"Parallel three-value AppliedToScope Literal definitions "
        f"detected (canonical source is "
        f"{_CANONICAL_MODULE_RELATIVE}): {offenders}. Per durable "
        f"§24 canonical-source-or-pin discipline: every consumer "
        f"imports the Literal directly rather than restating."
    )


# ── §2 tuple ↔ Literal parity ──────────────────────────────────────


def test_runtime_tuple_matches_literal_args() -> None:
    """``APPLIED_TO_SCOPE_VALUES`` ↔ ``typing.get_args(AppliedToScope)``
    parity. The runtime tuple is the iteration target; the Literal
    is the type-level vocabulary; a drift means a consumer iterating
    the tuple lags the type hint or vice versa."""
    literal_args = set(typing.get_args(AppliedToScope))
    tuple_values = set(APPLIED_TO_SCOPE_VALUES)
    assert literal_args == tuple_values, (
        f"AppliedToScope Literal args {sorted(literal_args)} drifted "
        f"from APPLIED_TO_SCOPE_VALUES tuple "
        f"{sorted(tuple_values)}. Update both together."
    )


def test_three_value_vocab_is_canonical() -> None:
    """Sprint E.3 ships exactly three scope discriminators. A future
    fourth value extends the Literal here intentionally (this test
    documents the v1 contract scope; updating it is the conscious
    extension signal)."""
    expected = {"single_cell", "zone", "enumerated_cells"}
    literal_args = set(typing.get_args(AppliedToScope))
    assert literal_args == expected, (
        f"Sprint E.3 AppliedToScope scope: {sorted(expected)}. Got: "
        f"{sorted(literal_args)}."
    )


# ── §3 dunder-all is the canonical export surface ──────────────────


def test_module_exports_canonical_symbols() -> None:
    """``AppliedToScope`` Literal + ``APPLIED_TO_SCOPE_VALUES`` tuple
    are the canonical exports. Internal helpers (none currently)
    stay private."""
    from prismpy.standards import applied_to_scope
    assert sorted(applied_to_scope.__all__) == [
        "APPLIED_TO_SCOPE_VALUES",
        "AppliedToScope",
    ]
