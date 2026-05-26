"""Producer-boundary error-classification structural pin.

Two invariants this pin protects:

1. The result dataclasses (``StageResult`` + ``TranslationResult``) carry an
   ``error_events`` field, and the executor's per-platform broad-except
   catch populates it via ``classify_to_event_dict`` (the producer-
   boundary classification site that closes the "exception class flattened
   to ``str(e)``" surface where a typed exception would otherwise be
   masked to the consumer).

2. ``F_AG_GATE_SITES`` enumerates the explicit raise sites in every
   F-AG-class translator: the ACEA site is typed (raises
   ``ClimateDownloadError`` with ``source='nasa_power'``); the deferred
   sites still raise (``ValueError`` / ``BuildEghrSubstrateError``) so the
   pipeline fails loudly, but their typed migration is later-sprint scope
   — the allowlist keeps the gap visible.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from prismpy.errors import ErrorEventDict, classify_to_event_dict
from prismpy.pipeline.executor import StageResult
from prismpy.translators.base import BaseTranslator, TranslationResult

_PRISMPY_ROOT = Path(__file__).resolve().parents[2]
_EXECUTOR = _PRISMPY_ROOT / "src" / "prismpy" / "pipeline" / "executor.py"


# Per-site raise expectations: (relative path, line number, expected class).
# Lines verified by spot-grep at the branch cut. ACEA is typed; the rest
# are deferred-untyped — the pin keeps the gap visible while letting them
# stay in production (they still fail loudly).
F_AG_GATE_SITES: Tuple[Tuple[str, int, str], ...] = (
    ("src/prismpy/translators/acea/translator.py", 514, "ClimateDownloadError"),
    ("src/prismpy/translators/pythia/translator.py", 970, "ValueError"),
    ("src/prismpy/translators/pythia/translator.py", 1763, "ValueError"),
    ("src/prismpy/translators/pythia/translator.py", 2511, "BuildEghrSubstrateError"),
    ("src/prismpy/translators/sarra_py/translator.py", 711, "ValueError"),
    ("src/prismpy/translators/sarra_py/translator.py", 1675, "ValueError"),
    ("src/prismpy/translators/craft/translator.py", 1565, "ValueError"),
    ("src/prismpy/translators/_shared/eghr_substrate.py", 461, "ValueError"),
    ("src/prismpy/translators/_shared/eghr_substrate.py", 466, "ValueError"),
)


def test_stage_result_has_error_events_field() -> None:
    names = {f.name for f in dataclasses.fields(StageResult)}
    assert "error_events" in names, (
        "StageResult MUST carry an error_events list for the consumer "
        "to dispatch on error class instead of pattern-matching strings"
    )


def test_translation_result_has_error_events_field() -> None:
    names = {f.name for f in dataclasses.fields(TranslationResult)}
    assert "error_events" in names, (
        "TranslationResult MUST carry an error_events list (the per-"
        "platform catch populates it; the TRANSLATE stage aggregates)"
    )


def test_create_result_accepts_error_events_kwarg() -> None:
    sig = inspect.signature(BaseTranslator.create_result)
    assert "error_events" in sig.parameters, (
        "BaseTranslator.create_result MUST accept an error_events kwarg "
        "so translator implementations can attach the structured payload"
    )


def test_classify_returned_keys_match_typed_dict() -> None:
    ev = classify_to_event_dict(ValueError("x"))
    declared = set(ErrorEventDict.__annotations__.keys())
    assert set(ev.keys()) == declared, (
        "classify output keys MUST match ErrorEventDict declaration "
        "(prevents producer-consumer drift on the wire shape)"
    )


def _find_call(tree: ast.AST, name: str) -> List[ast.Call]:
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == name)
            or (isinstance(n.func, ast.Attribute) and n.func.attr == name)
        )
    ]


def test_translate_catch_populates_error_events() -> None:
    """The per-platform broad-except in ``_execute_translate`` MUST call
    ``classify_to_event_dict`` AND pass ``error_events=`` to the
    ``TranslationResult`` it constructs in that catch."""
    text = _EXECUTOR.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert _find_call(tree, "classify_to_event_dict"), (
        "executor.py MUST invoke classify_to_event_dict at the catch site"
    )
    # At least one TranslationResult(...) construction carries error_events=
    tr_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "TranslationResult"
    ]
    assert any(
        any(kw.arg == "error_events" for kw in c.keywords) for c in tr_calls
    ), (
        "the catch-site TranslationResult MUST pass error_events=[...]"
    )


def test_translate_stage_result_aggregates_error_events() -> None:
    """The TRANSLATE stage's ``StageResult`` MUST aggregate ``error_events``
    across the per-platform ``translation_results`` so the consumer sees a
    single structured list for the stage."""
    text = _EXECUTOR.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "StageResult"):
            continue
        # find the StageResult(...) call that has stage=PipelineStage.TRANSLATE
        is_translate = any(
            kw.arg == "stage"
            and isinstance(kw.value, ast.Attribute)
            and kw.value.attr == "TRANSLATE"
            for kw in node.keywords
        )
        if not is_translate:
            continue
        kws = {kw.arg: kw for kw in node.keywords}
        assert "error_events" in kws, (
            "TRANSLATE StageResult MUST aggregate error_events from "
            "translation_results.values()"
        )
        # the aggregator references translation_results so we know it's
        # not just an empty placeholder.
        seg = ast.unparse(kws["error_events"].value)
        assert "translation_results" in seg, (
            f"error_events aggregator must read from translation_results; "
            f"got: {seg!r}"
        )
        return
    pytest.fail("could not find a TRANSLATE StageResult construction site")


def test_f_ag_gate_sites_raise_as_expected() -> None:
    """Every F-AG-class translator raise site stays a real raise of the
    expected exception class. The ACEA site keeps the typed shape
    (source=); the deferred sites stay loud (their typed migration is
    later-sprint scope, but the allowlist documents the gap)."""
    violations: List[str] = []
    for rel, line, cls in F_AG_GATE_SITES:
        path = _PRISMPY_ROOT / rel
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not 1 <= line <= len(lines):
            violations.append(f"{rel}:{line} out of range (file is {len(lines)} lines)")
            continue
        src = lines[line - 1]
        if f"raise {cls}" not in src:
            violations.append(
                f"{rel}:{line} expected `raise {cls}(...)`; got: {src.strip()!r}"
            )
    assert not violations, (
        "F_AG_GATE_SITES drift — update the allowlist after auditing the "
        "raise sites:\n  " + "\n  ".join(violations)
    )


_TRANSLATOR_PATHS = (
    "src/prismpy/translators/acea/translator.py",
    "src/prismpy/translators/pythia/translator.py",
    "src/prismpy/translators/sarra_py/translator.py",
    "src/prismpy/translators/craft/translator.py",
)


def test_each_translator_classifies_in_its_outer_catch() -> None:
    """Each translator's outer ``except Exception`` in ``translate`` MUST
    call ``classify_to_event_dict`` AND pass ``error_events=`` to its
    ``create_result`` return — otherwise the typed exception is flattened
    to ``str(e)`` before the executor-level catch ever sees it, recreating
    the Bester mask the producer-boundary classification is meant to close.
    """
    violations: List[str] = []
    for rel in _TRANSLATOR_PATHS:
        path = _PRISMPY_ROOT / rel
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        translate = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "translate"),
            None,
        )
        if translate is None:
            violations.append(f"{rel}: no `translate` function found")
            continue
        # The classify call MUST appear inside translate; AND at least one
        # create_result(...) call in translate MUST carry error_events=.
        classifies = False
        passes_events = False
        for sub in ast.walk(translate):
            if isinstance(sub, ast.Call):
                f = sub.func
                if (isinstance(f, ast.Name) and f.id == "classify_to_event_dict") \
                        or (isinstance(f, ast.Attribute) and f.attr == "classify_to_event_dict"):
                    classifies = True
                if isinstance(f, ast.Attribute) and f.attr == "create_result":
                    if any(kw.arg == "error_events" for kw in sub.keywords):
                        passes_events = True
        if not classifies:
            violations.append(f"{rel}: translate() does not call classify_to_event_dict")
        if not passes_events:
            violations.append(
                f"{rel}: translate() create_result(...) does not carry error_events="
            )
    assert not violations, (
        "translator-catch classification gap (would reintroduce the "
        "Bester mask):\n  " + "\n  ".join(violations)
    )


def test_acea_raise_carries_total_in_cell_unit() -> None:
    """ACEA's typed raise MUST pass ``total=len(cell_ids_30arcmin)`` so
    the downstream ``partial_progress`` is in the SAME unit as
    ``missing_tiles`` (30-arcmin cell count, NOT pixel-grid size). Without
    this, the consumer would report "47,996 of 48,000 cells" instead of
    the honest "96 of 100 cells" — quiet honest-signal violation."""
    acea = _PRISMPY_ROOT / "src/prismpy/translators/acea/translator.py"
    text = acea.read_text(encoding="utf-8")
    m = re.search(
        r"raise ClimateDownloadError\(.*?total\s*=\s*len\(\s*cell_ids_30arcmin",
        text, re.DOTALL,
    )
    assert m, (
        "ACEA raise MUST carry total=len(cell_ids_30arcmin) so "
        "partial_progress reports counts in the cell unit, not pixels"
    )


def test_acea_site_carries_typed_source_kwarg() -> None:
    """The ACEA raise site MUST keep ``source='nasa_power'`` on the
    typed ``ClimateDownloadError`` so the consumer (prismweb) can
    dispatch on provider."""
    acea = _PRISMPY_ROOT / "src/prismpy/translators/acea/translator.py"
    text = acea.read_text(encoding="utf-8")
    # Anchor on the ClimateDownloadError raise + its source= kwarg in
    # the immediate vicinity (a multi-line raise is normal in this file).
    m = re.search(
        r"raise ClimateDownloadError\(.*?source\s*=\s*['\"]nasa_power['\"]",
        text, re.DOTALL,
    )
    assert m, (
        "ACEA ClimateDownloadError raise MUST carry source='nasa_power' "
        "so the consumer can dispatch on provider"
    )
