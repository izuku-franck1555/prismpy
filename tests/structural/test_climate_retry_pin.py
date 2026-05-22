"""F-AG-NASA-RETRY Pin — canonical retry helper + NASA POWER call-site.

Three structural invariants + one anti-mutation drill:

* The helper at ``prismpy/src/prismpy/sources/common/retry.py``
  exposes ``retry_with_exponential_backoff`` with the canonical
  keyword-only signature (max_attempts / base_delay_s /
  jitter_ratio / exception_classes / on_retry / sleep_fn).
* The NASA POWER adapter (``nasa_power.py``) MUST invoke the
  helper with ``max_attempts >= 5`` AND ``base_delay_s >= 5``
  AND NOT route through the legacy inline
  ``for attempt in range(self.config.retry_count)`` pattern.
* Exponential schedule must amortise to ≥ 120 s cumulative
  retry budget (5 + 10 + 20 + 40 + 80 = 155 s for defaults).

Negative-case mutation drill: reverting the call-site to the
inline 3 × 5 s linear pattern → all three pins FAIL.
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest
import requests

from prismpy.sources.common.retry import (
    _bridge_helper_on_attempt,
    retry_with_exponential_backoff,
)

_PRISMPY_ROOT = Path(__file__).resolve().parents[2]
_NASA_POWER = _PRISMPY_ROOT / "src" / "prismpy" / "sources" / "climate" / "nasa_power.py"
_RETRY = _PRISMPY_ROOT / "src" / "prismpy" / "sources" / "common" / "retry.py"
# Ship 1' EXPANDED: GAEZ esri_client + TAMSAT also route through the helper.
_ESRI = _PRISMPY_ROOT / "src" / "prismpy" / "sources" / "gaez" / "esri_client.py"
_TAMSAT = _PRISMPY_ROOT / "src" / "prismpy" / "sources" / "climate" / "tamsat.py"


def _file_invokes_helper(path: Path) -> bool:
    """AST-walk ``path`` for a real Call node invoking the canonical helper
    (by name or attribute). Defeats substring / comment / docstring
    false-positives."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "retry_with_exponential_backoff":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "retry_with_exponential_backoff":
                return True
    return False


def _function_body_source(text: str, name: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"function {name!r} not found")


_EXPECTED_EXCEPTION_CLASSES = {
    requests.exceptions.HTTPError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
}


def test_helper_exposes_canonical_signature() -> None:
    sig = inspect.signature(retry_with_exponential_backoff)
    params = sig.parameters
    expected_kwargs = {
        "max_attempts": 6,
        "base_delay_s": 5.0,
        "jitter_ratio": 0.2,
    }
    for name, default in expected_kwargs.items():
        assert name in params, (
            f"retry_with_exponential_backoff missing keyword {name!r}"
        )
        assert params[name].default == default, (
            f"retry_with_exponential_backoff[{name}] default = "
            f"{params[name].default!r}; want {default!r}"
        )
    for required in ("exception_classes", "on_retry", "sleep_fn"):
        assert required in params, (
            f"retry_with_exponential_backoff missing keyword "
            f"{required!r}"
        )


def test_helper_exception_classes_default_is_canonical_set() -> None:
    """The default `exception_classes` tuple MUST contain exactly the
    4 transient-class types the contract names. Mutation that swaps
    one (e.g. ChunkedEncodingError → SomeOtherError) → pin FAILS."""
    sig = inspect.signature(retry_with_exponential_backoff)
    exc_default = sig.parameters["exception_classes"].default
    assert set(exc_default) == _EXPECTED_EXCEPTION_CLASSES, (
        f"retry_with_exponential_backoff default `exception_classes` "
        f"= {set(exc_default)}; MUST be {_EXPECTED_EXCEPTION_CLASSES} "
        f"(cycle-2 contract: HTTPError + ConnectionError + Timeout "
        f"+ ChunkedEncodingError; RequestException base over-catches "
        f"programmer bugs)."
    )


def test_nasa_power_routes_through_canonical_helper() -> None:
    """AST-walk asserts a real Call node invokes the canonical helper
    by name. Defeats substring / comment / docstring false-positives
    that a substring check would accept."""
    text = _NASA_POWER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    has_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) \
                    and func.id == "retry_with_exponential_backoff":
                has_call = True
                break
            if isinstance(func, ast.Attribute) \
                    and func.attr == "retry_with_exponential_backoff":
                has_call = True
                break
    assert has_call, (
        "nasa_power.py MUST contain a real Call node invoking "
        "`retry_with_exponential_backoff` (AST-walked). Substring "
        "matches in comments / docstrings don't satisfy the pin."
    )
    # The legacy inline iteration pattern MUST be gone.
    legacy_pattern = re.compile(
        r"for\s+attempt\s+in\s+range\(\s*self\.config\.retry_count",
    )
    assert not legacy_pattern.search(text), (
        "nasa_power.py still carries the legacy "
        "`for attempt in range(self.config.retry_count)` loop; "
        "F-AG-NASA-RETRY moves the retry policy into "
        "retry_with_exponential_backoff."
    )


def test_nasa_power_call_site_uses_required_budget_floor() -> None:
    """The call site MUST guarantee a minimum exponential budget
    regardless of caller-supplied config. The floor lives in a
    ``max(N, self.config.retry_count)`` (and matching base-delay)
    expression that promotes config values above the contract
    minimum but never below it."""
    text = _NASA_POWER.read_text(encoding="utf-8")
    # Require the second operand to be a config-derived expression
    # (self.config.retry_count / retry_delay), NOT a constant. A
    # mutation `max(6, 6)` would silently neutralise the floor by
    # making config irrelevant; this assertion catches that.
    floor_attempts = re.search(
        r"max_attempts\s*=\s*max\(\s*(\d+)\s*,\s*"
        r"int\(\s*self\.config\.retry_count\s*\)\s*\)",
        text,
    )
    floor_delay = re.search(
        r"base_delay_s\s*=\s*max\(\s*([\d.]+)\s*,\s*"
        r"float\(\s*self\.config\.retry_delay\s*\)\s*\)",
        text,
    )
    assert floor_attempts, (
        "nasa_power.py MUST pin "
        "`max_attempts = max(N, int(self.config.retry_count))` so "
        "the contract floor protects against config that lowers "
        "below the minimum while still honouring config overrides "
        "above it (deployment-engineer R1 + codex Gate B fix-up)."
    )
    assert floor_delay, (
        "nasa_power.py MUST pin "
        "`base_delay_s = max(N.0, float(self.config.retry_delay))` "
        "with the config-derived second operand (not a constant)."
    )
    floor_n = int(floor_attempts.group(1))
    floor_d = float(floor_delay.group(1))
    assert floor_n >= 5, (
        f"NASA POWER max_attempts floor MUST be ≥5; got {floor_n}"
    )
    assert floor_d >= 5.0, (
        f"NASA POWER base_delay floor MUST be ≥5.0 s; got {floor_d}"
    )
    n_sleeps = floor_n - 1
    budget = sum(floor_d * (2 ** i) for i in range(n_sleeps))
    assert budget >= 120, (
        f"Cumulative retry budget MUST be ≥120 s to absorb a "
        f"transient external-API blip; got {budget:.1f}s "
        f"({n_sleeps} sleeps from base={floor_d}s)"
    )


def test_anti_mutation_revert_to_linear_breaks_pin() -> None:
    """Drill: rewrite the call-site as the legacy linear loop in
    memory; assert the legacy-pattern check fires."""
    text = _NASA_POWER.read_text(encoding="utf-8")
    mutated = (
        text
        .replace("retry_with_exponential_backoff", "DELETED")
        + "\n# legacy fragment for drill\n"
        + "for attempt in range(self.config.retry_count):\n"
    )
    legacy_pattern = re.compile(
        r"for\s+attempt\s+in\s+range\(\s*self\.config\.retry_count",
    )
    assert legacy_pattern.search(mutated), (
        "anti-mutation drill: synthetic re-add of the legacy linear "
        "loop MUST surface in the same regex Pin 2 inspects"
    )
    assert "retry_with_exponential_backoff" not in mutated, (
        "anti-mutation drill: removing the helper call MUST be "
        "detectable by the same substring Pin 2 checks"
    )


def test_anti_mutation_swap_chunked_encoding_error_breaks_pin() -> None:
    """Drill: synthetic exception_classes tuple that swaps one of
    the 4 transient types (ChunkedEncodingError → str) MUST fail
    the canonical-set check."""
    mutated = {
        requests.exceptions.HTTPError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        str,  # swapped from ChunkedEncodingError
    }
    assert mutated != _EXPECTED_EXCEPTION_CLASSES, (
        "anti-mutation drill: swapping a transient class out of "
        "the default tuple MUST diverge from the canonical set "
        "Pin 2b inspects"
    )


# ---------------------------------------------------------------------------
# Ship 1' EXPANDED — GAEZ esri_client + TAMSAT migration + PRI-6 on_attempt.
# ---------------------------------------------------------------------------


def test_helper_exposes_on_attempt_keyword() -> None:
    """PRI-6: the canonical helper MUST expose an ``on_attempt`` keyword so
    producer-side adapters can emit a retry-attempt substage. Removing it
    breaks the retry-attempt threading substrate."""
    sig = inspect.signature(retry_with_exponential_backoff)
    assert "on_attempt" in sig.parameters, (
        "retry_with_exponential_backoff MUST expose `on_attempt` "
        "(PRI-6 producer-side retry-attempt threading)"
    )


def test_gaez_esri_routes_through_canonical_helper() -> None:
    """AC-S1E-2: GAEZ ``fetch_image`` (the PRODUCTION retry surface) MUST
    route through the canonical helper, and the legacy bespoke
    ``while attempt <= self.retries`` loop MUST be gone."""
    assert _file_invokes_helper(_ESRI), (
        "esri_client.py MUST invoke retry_with_exponential_backoff "
        "(AST-walked) — the bespoke while-loop is migrated to the "
        "canonical helper."
    )
    text = _ESRI.read_text(encoding="utf-8")
    legacy = re.compile(r"while\s+attempt\s*<=\s*self\.retries")
    assert not legacy.search(text), (
        "esri_client.py still carries the bespoke "
        "`while attempt <= self.retries` retry loop; AC-S1E-2 migrates "
        "fetch_image to retry_with_exponential_backoff."
    )


def test_gaez_fetch_image_has_cancel_and_progress_params() -> None:
    """AC-S1E-2 + AC-S1E-1: the innermost GAEZ retry surface MUST accept
    ``cancel_check`` (5-level cancel-wire terminus) AND ``progress_callback``
    (producer-side retry-attempt emit)."""
    from prismpy.sources.gaez.esri_client import EsriImageServiceClient

    params = inspect.signature(EsriImageServiceClient.fetch_image).parameters
    assert "cancel_check" in params, "fetch_image MUST accept cancel_check"
    assert "progress_callback" in params, (
        "fetch_image MUST accept progress_callback (PRI-6 retry-attempt emit)"
    )


def test_tamsat_routes_through_canonical_helper() -> None:
    """AC-S1E-3: TAMSAT ``_download_nc`` MUST route the 5xx fallback through
    the canonical helper instead of the bespoke double ``requests.get``."""
    assert _file_invokes_helper(_TAMSAT), (
        "tamsat.py MUST invoke retry_with_exponential_backoff (AC-S1E-3)"
    )


# ---------------------------------------------------------------------------
# Codex round-1 absorption: backoff-multiplier parity + legacy-callback
# additive fallback for the retry-substage bridge.
# ---------------------------------------------------------------------------


def test_helper_honors_backoff_multiplier() -> None:
    """The ``backoff_multiplier`` knob MUST drive the schedule so adapters
    (e.g. GAEZ ``self.backoff``) reproduce their own backoff rather than a
    forced doubling. With base=1.0 and multiplier=3.0 the sleeps are
    1, 3, 9 (jitter disabled)."""
    sleeps = []
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise requests.exceptions.Timeout("t")

    with pytest.raises(requests.exceptions.Timeout):
        retry_with_exponential_backoff(
            boom,
            max_attempts=4,
            base_delay_s=1.0,
            jitter_ratio=0.0,
            backoff_multiplier=3.0,
            exception_classes=(requests.exceptions.Timeout,),
            sleep_fn=sleeps.append,
        )
    assert sleeps == [1.0, 3.0, 9.0], f"expected 1,3,9 schedule; got {sleeps}"


class _LegacyFiveArgCallback:
    """Implements ONLY the documented 5-arg on_substage_progress protocol —
    no ``retry_info`` keyword (the pre-β consumer shape)."""

    def __init__(self):
        self.calls = []

    def on_substage_progress(self, stage, task, current, total, detail=""):
        self.calls.append((stage, task, current, total, detail))


class _ModernRetryInfoCallback:
    def __init__(self):
        self.calls = []

    def on_substage_progress(self, stage, task, current, total, detail="",
                             retry_info=None):
        self.calls.append((stage, task, current, total, detail, retry_info))


def test_bridge_falls_back_to_legacy_5arg_callback() -> None:
    """Codex P2: a retry storm against a legacy 5-arg callback MUST NOT
    raise TypeError — the bridge degrades to the 5-arg call shape."""
    cb = _LegacyFiveArgCallback()
    on_attempt = _bridge_helper_on_attempt(cb, "translate", "GAEZ")
    assert on_attempt is not None
    on_attempt(2, 6, 10.0)  # must not raise
    assert cb.calls == [("translate", "Retrying GAEZ", 2, 6,
                         "next attempt in 10s")]


def test_bridge_passes_retry_info_to_modern_callback() -> None:
    """A retry_info-aware consumer receives the structured payload."""
    cb = _ModernRetryInfoCallback()
    on_attempt = _bridge_helper_on_attempt(cb, "retrieve", "TAMSAT")
    on_attempt(1, 3, 5.0)
    assert len(cb.calls) == 1
    ri = cb.calls[0][5]
    assert ri == {
        "kind": "retry", "attempt": 1, "max_attempts": 3,
        "next_retry_delay_s": 5.0, "provider": "TAMSAT",
    }
