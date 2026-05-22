"""Cancel-during-retry + single-emitter structural pins.

Two invariants over the prismpy source tree:

1. cancel-during-retry (``test_every_retry_call_site_is_cancel_wired``):
   every call to ``retry_with_exponential_backoff`` in ``src/prismpy/**``
   MUST pass an ``on_retry=`` callback whose body reaches
   ``raise_if_cancelled`` — so a cancel during a backoff sleep raises
   promptly instead of waiting out the full retry budget.

2. single-emitter (``test_no_inline_retry_substage_dict``): the
   ``{'kind': 'retry', ...}`` payload MUST be built in exactly one place —
   ``_bridge_helper_on_attempt`` — and never inline-constructed by an adapter.

Resolver scope: the cancel-wire walk resolves the ``on_retry`` argument one
level by NAME — it must be a ``Name`` referring to a ``def`` in the same
module whose (recursive) body calls ``raise_if_cancelled``. Lambdas, callback
factories, attribute callbacks, aliased imports, and variable indirection are
NOT resolved and will FAIL the pin — forcing the local-``_on_retry``-def
pattern. Arity is validated against the 3-arg ``(attempt, exc, sleep_s)``
contract.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

_PRISMPY_ROOT = Path(__file__).resolve().parents[2]
_SOURCES = _PRISMPY_ROOT / "src" / "prismpy"
_RETRY = _SOURCES / "sources" / "common" / "retry.py"

_HELPER_NAME = "retry_with_exponential_backoff"
_CANCEL_NAME = "raise_if_cancelled"


def _iter_py_files() -> List[Path]:
    return sorted(_SOURCES.rglob("*.py"))


def _collect_funcdefs_by_name(tree: ast.AST) -> Dict[str, List[ast.AST]]:
    """Map every (possibly nested) function name → its def nodes."""
    out: Dict[str, List[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, []).append(node)
    return out


def _body_calls_cancel(func: ast.AST) -> bool:
    for sub in ast.walk(func):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name) and f.id == _CANCEL_NAME:
                return True
            if isinstance(f, ast.Attribute) and f.attr == _CANCEL_NAME:
                return True
    return False


def _func_positional_arity(func: ast.AST) -> int:
    a = func.args
    return len(a.posonlyargs) + len(a.args)


def _retry_call_sites(tree: ast.AST) -> List[ast.Call]:
    sites: List[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == _HELPER_NAME:
                sites.append(node)
            elif isinstance(f, ast.Attribute) and f.attr == _HELPER_NAME:
                sites.append(node)
    return sites


def test_every_retry_call_site_is_cancel_wired() -> None:
    """Each helper call site passes ``on_retry=`` whose def reaches
    ``raise_if_cancelled`` with the canonical 3-arg arity."""
    checked = 0
    failures: List[str] = []
    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8")
        if _HELPER_NAME not in text:
            continue
        tree = ast.parse(text)
        sites = _retry_call_sites(tree)
        if not sites:
            continue
        funcs = _collect_funcdefs_by_name(tree)
        rel = path.relative_to(_PRISMPY_ROOT)
        for call in sites:
            checked += 1
            on_retry = next(
                (kw.value for kw in call.keywords if kw.arg == "on_retry"),
                None,
            )
            if on_retry is None:
                failures.append(f"{rel}:{call.lineno} missing on_retry=")
                continue
            if not isinstance(on_retry, ast.Name):
                failures.append(
                    f"{rel}:{call.lineno} on_retry is not a resolvable "
                    f"Name (lambdas / factories / attrs are rejected by "
                    f"design — use a local _on_retry def)"
                )
                continue
            candidates = funcs.get(on_retry.id, [])
            if not candidates:
                failures.append(
                    f"{rel}:{call.lineno} on_retry={on_retry.id!r} has no "
                    f"def in this module"
                )
                continue
            if not any(_body_calls_cancel(fn) for fn in candidates):
                failures.append(
                    f"{rel}:{call.lineno} on_retry={on_retry.id!r} body "
                    f"does not call {_CANCEL_NAME}"
                )
                continue
            if not any(_func_positional_arity(fn) == 3 for fn in candidates):
                failures.append(
                    f"{rel}:{call.lineno} on_retry={on_retry.id!r} arity "
                    f"!= 3 (canonical (attempt, exc, sleep_s))"
                )
    assert checked >= 3, (
        f"expected ≥3 canonical retry call sites (NASA + GAEZ + TAMSAT); "
        f"found {checked} — has an adapter regressed off the helper?"
    )
    assert not failures, "cancel-during-retry pin failures:\n" + "\n".join(
        failures
    )


def _dicts_with_retry_kind(tree: ast.AST) -> List[int]:
    """Line numbers of Dict literals carrying a ``'kind': 'retry'`` entry."""
    hits: List[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if (
                isinstance(k, ast.Constant)
                and k.value == "kind"
                and isinstance(v, ast.Constant)
                and v.value == "retry"
            ):
                hits.append(node.lineno)
    return hits


def test_no_inline_retry_substage_dict() -> None:
    """No adapter inline-builds a ``{'kind': 'retry', ...}`` substage dict;
    the only producer is ``_bridge_helper_on_attempt`` in
    ``sources/common/retry.py``."""
    offenders: List[Tuple[str, int]] = []
    for path in _iter_py_files():
        if path.resolve() == _RETRY.resolve():
            continue  # the single allowed emitter
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno in _dicts_with_retry_kind(tree):
            offenders.append((str(path.relative_to(_PRISMPY_ROOT)), lineno))
    assert not offenders, (
        "inline retry-substage dict(s) outside _bridge_helper_on_attempt:\n"
        + "\n".join(f"  {f}:{ln}" for f, ln in offenders)
    )


def test_bridge_factory_is_the_emitter() -> None:
    """Positive companion to the negative pin above: retry.py's
    ``_bridge_helper_on_attempt`` MUST build the canonical retry dict."""
    tree = ast.parse(_RETRY.read_text(encoding="utf-8"))
    in_bridge = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_bridge_helper_on_attempt"
    ]
    assert in_bridge, "_bridge_helper_on_attempt def missing from retry.py"
    assert _dicts_with_retry_kind(in_bridge[0]), (
        "_bridge_helper_on_attempt MUST construct the {'kind': 'retry', ...} "
        "payload (single canonical emitter)"
    )
