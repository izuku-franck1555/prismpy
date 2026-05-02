"""Sprint D.1 AC-2 — translator-side silent-zero rain pattern absent.

The structural net asserts that the silent-zero rain default and
the silent zero-clamp on the missing-value sentinel are not
present in any of the four translators. The grep-style check is
a regression net for the exact shape the audit caught in PYTHIA;
the per-translator audit table at the top of the file documents
each translator's status at the time Sprint D.1 shipped so a
future reader of the test understands what each grep is pinning.

Per-translator audit table (Sprint D.1 dispatch + verification):

| Translator | File:line | Pattern | Status |
|---|---|---|---|
| CRAFT | craft/translator.py:~1559 | ``else -99.0`` (no clamp) | already correct |
| PYTHIA | pythia/translator.py:~459 | ``else 0.0`` rain default | AC-2 target — fixed in this sprint |
| PYTHIA | pythia/translator.py:~466 | ``max(0, rain) if rain != -99 else 0.0`` clamp | AC-2 target — fixed in this sprint |
| ACEA | acea/translator.py:~813 | ``np.array([..], dtype=np.float32)`` (None -> NaN) | None preserved as NaN by numpy default |
| SARRA-Py | sarra_py/translator.py:~753 | list comp + xarray Dataset | None preserved by construction |

Both ACEA's float32 array and SARRA-Py's list-comp-into-xarray
paths are None-preserving by construction; neither translator
exhibits the silent-zero pattern the audit flagged in PYTHIA.
The structural greps below pin these shapes so a future caller
that adds ``.fillna(0)`` or ``np.nan_to_num(...)`` to either
translator's rain path surfaces here.
"""
from __future__ import annotations

import inspect
import re

import pytest


def _read_source(module_name: str) -> str:
    """Helper: import a module by name and return its source."""
    import importlib

    mod = importlib.import_module(module_name)
    return inspect.getsource(mod)


# ---------------------------------------------------------------------------
# AC-2 — PYTHIA silent-zero patterns absent
# ---------------------------------------------------------------------------


def test_pythia_no_silent_zero_rain_default():
    """PYTHIA's rain default uses -99.0 (the DSSAT MISDAT
    sentinel), not 0.0. The grep is anchored to ``rain =
    record.precip`` to avoid matching unrelated 0.0 literals
    elsewhere in the translator."""
    src = _read_source("prismpy.translators.pythia.translator")
    bad_pattern = re.compile(
        r"rain\s*=\s*record\.precip\s+if\s+record\.precip\s+is\s+not\s+None\s+else\s+0\.0"
    )
    assert not bad_pattern.search(src), (
        "PYTHIA rain default still uses else 0.0 — AC-2 contract "
        "requires else MISDAT (-99.0). Without -99.0, papers "
        "using PYTHIA-derived rain analyses get phantom zero-rain "
        "days for missing inputs."
    )


def test_pythia_no_negative_99_to_zero_clamp():
    """The previous clamp ``rain = max(0, rain) if rain != -99
    else 0.0`` re-converted any -99 that did make it through to
    0.0 (the silent-zero #2). It must not appear after Sprint D.1."""
    src = _read_source("prismpy.translators.pythia.translator")
    bad_pattern = re.compile(
        r"max\(\s*0\s*,\s*rain\s*\)\s*if\s+rain\s*!=\s*-99\s+else\s+0\.0"
    )
    assert not bad_pattern.search(src), (
        "PYTHIA still re-clamps the -99 sentinel back to 0.0 — "
        "AC-2 contract preserves the missing-value sentinel "
        "through to the .WTH output."
    )


def test_pythia_uses_misdat_constant():
    """The replacement pattern names the sentinel via a local
    MISDAT constant so the meaning is explicit at the call
    site."""
    src = _read_source("prismpy.translators.pythia.translator")
    assert "MISDAT" in src, (
        "PYTHIA AC-2 fix is expected to introduce a local MISDAT "
        "constant naming the -99.0 sentinel."
    )


# ---------------------------------------------------------------------------
# AC-2 — CRAFT regression pin (already correct)
# ---------------------------------------------------------------------------


def test_craft_rain_uses_minus_99_default():
    """CRAFT's rain default was already correct prior to Sprint
    D.1 — the regression pin asserts the contract did not slip
    while AC-2 was being applied to PYTHIA."""
    src = _read_source("prismpy.translators.craft.translator")
    good_pattern = re.compile(
        r"rain\s*=\s*record\.precip\s+if\s+record\.precip\s+is\s+not\s+None\s+else\s+-99\.0"
    )
    assert good_pattern.search(src), (
        "CRAFT rain default no longer uses else -99.0 — "
        "regression of the already-correct path is a Sprint D.1 "
        "watch item."
    )


def test_craft_no_silent_zero_rain():
    """CRAFT must not introduce the PYTHIA-style ``else 0.0``
    rain default (a regression watch)."""
    src = _read_source("prismpy.translators.craft.translator")
    bad_pattern = re.compile(
        r"rain\s*=\s*record\.precip\s+if\s+record\.precip\s+is\s+not\s+None\s+else\s+0\.0"
    )
    assert not bad_pattern.search(src), (
        "CRAFT rain default uses else 0.0 — regression of the "
        "already-correct -99.0 default."
    )


# ---------------------------------------------------------------------------
# AC-2 — ACEA / SARRA-Py None-preserving regression pins
# ---------------------------------------------------------------------------


def test_acea_does_not_introduce_fillna_zero_or_nan_to_num():
    """ACEA preserves None as NaN through numpy's float32
    default. A future writer that adds ``.fillna(0)`` or
    ``np.nan_to_num(...)`` to the rain path would silently
    re-introduce the silent-zero shape — the structural pin
    catches the regression."""
    src = _read_source("prismpy.translators.acea.translator")
    fillna_zero = re.compile(r"\.fillna\(\s*0[\.\s,)]", re.MULTILINE)
    nan_to_num = re.compile(r"np\.nan_to_num\(", re.MULTILINE)
    assert not fillna_zero.search(src), (
        "ACEA introduced .fillna(0) — Sprint D.1 AC-2 sibling "
        "sweep contract requires None / NaN preservation across "
        "all translators."
    )
    assert not nan_to_num.search(src), (
        "ACEA introduced np.nan_to_num(...) — Sprint D.1 AC-2 "
        "sibling sweep contract requires explicit None / NaN "
        "preservation."
    )


def test_sarra_py_preserves_none_in_precip_list_comp():
    """SARRA-Py's precip extraction is a list comp into an
    xarray Dataset; the dataset preserves None via NaN. The
    pin asserts the list-comp pattern stays + that no
    ``.fillna(0)`` regression slips in."""
    src = _read_source("prismpy.translators.sarra_py.translator")
    list_comp_pattern = re.compile(
        r"precip\s*=\s*\[\s*r\.precip\s+for\s+r\s+in\s+ts\.records\s*\]"
    )
    assert list_comp_pattern.search(src), (
        "SARRA-Py precip extraction no longer uses the list-"
        "comprehension None-preserving path. Verify whether the "
        "replacement preserves None semantics through to xarray."
    )
    fillna_zero = re.compile(r"\.fillna\(\s*0[\.\s,)]", re.MULTILINE)
    assert not fillna_zero.search(src), (
        "SARRA-Py introduced .fillna(0) — Sprint D.1 AC-2 sibling "
        "sweep contract requires explicit None preservation."
    )
