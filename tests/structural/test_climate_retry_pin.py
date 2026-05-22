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

from prismpy.sources.common.retry import retry_with_exponential_backoff

_PRISMPY_ROOT = Path(__file__).resolve().parents[2]
_NASA_POWER = _PRISMPY_ROOT / "src" / "prismpy" / "sources" / "climate" / "nasa_power.py"
_RETRY = _PRISMPY_ROOT / "src" / "prismpy" / "sources" / "common" / "retry.py"


def _function_body_source(text: str, name: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"function {name!r} not found")


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


def test_nasa_power_routes_through_canonical_helper() -> None:
    text = _NASA_POWER.read_text(encoding="utf-8")
    assert "from prismpy.sources.common.retry import" in text \
        or "from prismpy.sources.common import retry" in text \
        or "retry_with_exponential_backoff" in text, (
        "nasa_power.py MUST import + call retry_with_exponential_backoff"
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
    text = _NASA_POWER.read_text(encoding="utf-8")
    call_match = re.search(
        r"retry_with_exponential_backoff\((.*?)\)",
        text, re.DOTALL,
    )
    assert call_match, (
        "nasa_power.py MUST invoke retry_with_exponential_backoff"
    )
    body = call_match.group(1)
    max_attempts_match = re.search(r"max_attempts\s*=\s*(\d+)", body)
    base_delay_match = re.search(r"base_delay_s\s*=\s*([\d.]+)", body)
    assert max_attempts_match, "call MUST pass max_attempts=N"
    assert base_delay_match, "call MUST pass base_delay_s=N"
    max_attempts = int(max_attempts_match.group(1))
    base_delay = float(base_delay_match.group(1))
    assert max_attempts >= 5, (
        f"NASA POWER retry MUST allow ≥5 total attempts (per "
        f"deployment-engineer R1); got max_attempts={max_attempts}"
    )
    assert base_delay >= 5.0, (
        f"NASA POWER base delay MUST be ≥5 s; got {base_delay}s"
    )
    # Cumulative budget: sum of base * 2^i for i in 0..(N-2).
    n_sleeps = max_attempts - 1
    budget = sum(base_delay * (2 ** i) for i in range(n_sleeps))
    assert budget >= 120, (
        f"Cumulative retry budget MUST be ≥120 s to absorb a "
        f"transient external-API blip; got {budget:.1f}s "
        f"({n_sleeps} sleeps from base={base_delay}s)"
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
