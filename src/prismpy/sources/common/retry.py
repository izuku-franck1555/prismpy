"""Canonical retry helper for requests-based adapter calls.

Centralises exponential-backoff + jitter retry semantics for any
``requests``-backed adapter call so the policy lives in one place
per durable §24 canonical-source-or-pin. The current consumer is
the NASA POWER adapter; non-``requests`` adapters (cdsapi, custom
TCP, etc.) need their own retry wrapper and are not in scope.

Default schedule (max_attempts=6, base_delay_s=5.0):
    initial call + sleeps of 5, 10, 20, 40, 80 s ≈ 155 s budget
    (each sleep modulated by ±jitter_ratio to spread thundering-herd
    bursts when many tiles retry simultaneously).
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Optional, Tuple, Type

import requests

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    callable_: Callable[[], Any],
    *,
    max_attempts: int = 6,
    base_delay_s: float = 5.0,
    jitter_ratio: float = 0.2,
    exception_classes: Tuple[Type[BaseException], ...] = (
        requests.exceptions.HTTPError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    ),
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    """Invoke ``callable_`` with exponential-backoff retry on transient
    provider errors.

    Args:
        callable_: zero-argument callable wrapping the provider call.
        max_attempts: total attempts (1 initial + max_attempts - 1
            retries). Default 6 → 5 retry sleeps.
        base_delay_s: first retry sleep before jitter. Subsequent
            sleeps double (5 → 10 → 20 → 40 → 80 s).
        jitter_ratio: ±fraction of the sleep magnitude applied
            uniformly to spread herd retries.
        exception_classes: tuple of exception types treated as
            transient. ``requests.RequestException`` is deliberately
            NOT a default because it would catch programmer errors
            like InvalidURL.
        on_retry: optional callback ``(attempt_index, exc, sleep_s)``
            invoked between attempts (e.g., for cancel-aware
            consumers that want to raise during the sleep window).
        sleep_fn: injection seam for tests.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return callable_()
        except exception_classes as exc:
            last_exc = exc
            if attempt >= max_attempts - 1:
                break
            sleep_s = base_delay_s * (2 ** attempt)
            if jitter_ratio:
                spread = sleep_s * jitter_ratio
                sleep_s += random.uniform(-spread, spread)
            sleep_s = max(0.0, sleep_s)
            if on_retry is not None:
                on_retry(attempt, exc, sleep_s)
            logger.warning(
                "retry_with_exponential_backoff: attempt %d/%d "
                "raised %s; sleeping %.2fs",
                attempt + 1, max_attempts, type(exc).__name__, sleep_s,
            )
            sleep_fn(sleep_s)
    assert last_exc is not None  # exhausted loop implies last_exc set
    raise last_exc
