"""Structural pin: ``EvidenceType`` Literal canonical-source parity.

Sprint E.3 AC-E3-1 + AC-E3-12 #1 + Stage 1 §9 #1. The canonical
``EvidenceType`` Literal lives at
``prismpy/provenance/wizard_decisions.py:119``; Sprint E.3 cockpit
``OverrideRecord`` re-imports it directly per AC-E3-1 sub-1 +
sub-3 (NO parallel enum). Two invariants close the silent-drift
class per durable §24 canonical-source-or-pin + durable §27 two-
vocabulary substrate-drift:

§1 Canonical-source uniqueness — only
``prismpy/provenance/wizard_decisions.py`` carries the 6-value
EvidenceType Literal. The AST walker rejects any parallel
definition (typo'd or otherwise restated 6-arg evidence-type
Literal) elsewhere in the prismpy source tree.

§2 OverrideRecord re-uses canonical Literal directly — the
``evidence_type`` field on OverrideRecord MUST resolve to the
exact ``EvidenceType`` symbol from ``wizard_decisions``. A future
refactor that copies the Literal definition into ``override.py``
fires the §1 walker; a refactor that types ``evidence_type`` with
a different name fires §2.

Same discipline as ``test_caveat_code_completeness.py`` (the
caveats canonical-source-with-phrase-dict pattern).
"""

from __future__ import annotations

import ast
import typing
from pathlib import Path

from prismpy.models.override import OverrideRecord
from prismpy.provenance.wizard_decisions import EvidenceType


# ── helpers ─────────────────────────────────────────────────────────


_CANONICAL_MODULE_RELATIVE = Path("prismpy/provenance/wizard_decisions.py")

_SIX_VALUE_VOCAB = frozenset({
    "local_trial",
    "irrigation",
    "cultivar_specific",
    "citation",
    "field_observation",
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


def _is_six_value_evidence_type_literal(node: ast.AST) -> bool:
    """True iff ``node`` is a ``Literal[...]`` subscript whose args
    spell exactly the six canonical evidence_type values."""
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
    return arg_strings == _SIX_VALUE_VOCAB


# ── §1 canonical-source uniqueness ─────────────────────────────────


def test_only_canonical_module_defines_six_value_literal() -> None:
    """Reject parallel Literal definitions that spell the same six
    evidence-type values. Every consumer MUST import
    ``EvidenceType`` from ``wizard_decisions.py`` rather than
    restating the Literal."""
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
            if _is_six_value_evidence_type_literal(node):
                offenders.append(
                    f"{path.relative_to(src_root.parent)}:{node.lineno}"
                )
                break

    assert not offenders, (
        f"Parallel six-value EvidenceType Literal definitions "
        f"detected (canonical source is "
        f"{_CANONICAL_MODULE_RELATIVE}): {offenders}. Per durable "
        f"§24 canonical-source-or-pin discipline + AC-E3-1 sub-3: "
        f"every consumer imports the Literal directly rather than "
        f"restating."
    )


# ── §2 OverrideRecord re-uses canonical Literal ────────────────────


def test_override_record_evidence_type_field_resolves_to_canonical_literal() -> None:
    """``OverrideRecord.evidence_type`` field MUST resolve to the
    exact ``EvidenceType`` symbol from ``wizard_decisions``. A
    future refactor that copies the Literal into ``override.py``
    or types the field with a different Literal-arg-set drifts
    silently from the canonical vocabulary; this pin enforces the
    direct symbol re-use."""
    field_info = OverrideRecord.model_fields["evidence_type"]
    annotation = field_info.annotation
    canonical_args = set(typing.get_args(EvidenceType))
    field_args = set(typing.get_args(annotation))
    assert canonical_args == field_args, (
        f"OverrideRecord.evidence_type Literal args "
        f"{sorted(field_args)} drifted from canonical EvidenceType "
        f"{sorted(canonical_args)} at "
        f"{_CANONICAL_MODULE_RELATIVE}:119. Per AC-E3-1 sub-1 + "
        f"sub-3: re-import the canonical Literal directly; do NOT "
        f"restate."
    )


def test_six_value_vocab_is_canonical() -> None:
    """Sprint E.3 v1 ships the six values per AC-E3-1. Updating
    this assertion is a conscious vocabulary extension signal."""
    canonical_args = set(typing.get_args(EvidenceType))
    assert canonical_args == _SIX_VALUE_VOCAB, (
        f"Sprint E.3 EvidenceType scope: "
        f"{sorted(_SIX_VALUE_VOCAB)}. Got: "
        f"{sorted(canonical_args)}."
    )
