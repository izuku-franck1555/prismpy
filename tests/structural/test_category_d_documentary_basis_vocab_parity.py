"""Structural pin: ``CategoryDDocumentaryBasis`` Literal canonical-source.

Sprint E.3 AC-E3-12 #12 + AC-E3-4 + WA CA-17 absorbed. The
canonical 4-value Literal at ``prismpy/models/override.py`` is the
single source of truth for the Cat D documentary-basis
discriminator. Three invariants close the silent-drift class per
durable §24 canonical-source-or-pin + durable §27 two-vocabulary
substrate-drift:

§1 Canonical-source uniqueness — only ``prismpy/models/override.py``
carries the 4-value Literal definition. The AST walker rejects any
parallel definition (typo'd or otherwise restated 4-arg
documentary-basis Literal) elsewhere in the prismpy source tree.

§2 OverrideRecord field annotation resolves to the canonical
Literal — the schema's ``category_d_documentary_basis`` field
MUST be typed with the exact ``CategoryDDocumentaryBasis`` symbol.

§3 4-value vocab is canonical — Sprint E.3 v1 ships exactly the
4 values per WA CA-17 + AC-E3-4 contract text; updating this
assertion is a conscious vocabulary-extension signal.
"""

from __future__ import annotations

import ast
import typing
from pathlib import Path

from prismpy.models.override import (
    CategoryDDocumentaryBasis,
    OverrideRecord,
)


_CANONICAL_MODULE_RELATIVE = Path("prismpy/models/override.py")

_FOUR_VALUE_VOCAB = frozenset({
    "irrigation_infrastructure",
    "documented_microclimate",
    "shallow_rooted_crop_variety",
    "other",
})


def _prismpy_src_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent.parent / "src" / "prismpy"


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _is_four_value_documentary_basis_literal(node: ast.AST) -> bool:
    """True iff ``node`` is a ``Literal[...]`` subscript whose
    args spell exactly the four canonical documentary-basis values."""
    if not isinstance(node, ast.Subscript):
        return False
    value = node.value
    is_literal_ref = (
        (isinstance(value, ast.Name) and value.id == "Literal")
        or (isinstance(value, ast.Attribute) and value.attr == "Literal")
    )
    if not is_literal_ref:
        return False
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
            return False
    return arg_strings == _FOUR_VALUE_VOCAB


# ── §1 canonical-source uniqueness ─────────────────────────────────


def test_only_canonical_module_defines_four_value_literal() -> None:
    """Reject parallel Literal definitions that spell the same
    four documentary-basis values. Per durable §24 + AC-E3-4."""
    src_root = _prismpy_src_root()
    canonical = src_root.parent / _CANONICAL_MODULE_RELATIVE
    offenders: list[str] = []

    for path in _iter_python_files(src_root):
        if path.resolve() == canonical.resolve():
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if _is_four_value_documentary_basis_literal(node):
                offenders.append(
                    f"{path.relative_to(src_root.parent)}:{node.lineno}"
                )
                break

    assert not offenders, (
        f"Parallel four-value CategoryDDocumentaryBasis Literal "
        f"definitions detected (canonical source is "
        f"{_CANONICAL_MODULE_RELATIVE}): {offenders}. Per durable "
        f"§24 + AC-E3-4: every consumer imports the Literal "
        f"directly rather than restating."
    )


# ── §2 OverrideRecord uses canonical Literal ──────────────────────


def test_override_record_field_resolves_to_canonical_literal() -> None:
    """``OverrideRecord.category_d_documentary_basis`` field MUST
    type to the canonical Literal. The Optional wrapper is allowed
    (Cat A/B/C records carry None); the inner Literal args must
    match the canonical 4-value set."""
    field_info = OverrideRecord.model_fields["category_d_documentary_basis"]
    annotation = field_info.annotation
    # The annotation is Optional[CategoryDDocumentaryBasis] which
    # is Union[CategoryDDocumentaryBasis, None]. Find the non-None
    # arm and inspect its Literal args.
    union_args = typing.get_args(annotation)
    non_none_arms = [a for a in union_args if a is not type(None)]
    assert len(non_none_arms) == 1, (
        f"category_d_documentary_basis annotation expected to be "
        f"Optional[CategoryDDocumentaryBasis] (one non-None arm); "
        f"got {len(non_none_arms)} arms: {non_none_arms}"
    )
    field_args = set(typing.get_args(non_none_arms[0]))
    canonical_args = set(typing.get_args(CategoryDDocumentaryBasis))
    assert field_args == canonical_args, (
        f"OverrideRecord.category_d_documentary_basis Literal args "
        f"{sorted(field_args)} drifted from canonical "
        f"CategoryDDocumentaryBasis {sorted(canonical_args)} at "
        f"{_CANONICAL_MODULE_RELATIVE}."
    )


# ── §3 4-value vocab is canonical ──────────────────────────────────


def test_four_value_vocab_is_canonical() -> None:
    """Sprint E.3 v1 ships exactly the 4 values per WA CA-17 +
    AC-E3-4. Updating this assertion is a conscious vocabulary-
    extension signal."""
    canonical_args = set(typing.get_args(CategoryDDocumentaryBasis))
    assert canonical_args == _FOUR_VALUE_VOCAB, (
        f"Sprint E.3 CategoryDDocumentaryBasis scope: "
        f"{sorted(_FOUR_VALUE_VOCAB)}. Got: {sorted(canonical_args)}."
    )
