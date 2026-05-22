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
    on_attempt: Optional[Callable[[int, int, float], None]] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
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
            ``attempt_index`` is 0-based; fires BEFORE ``on_attempt``
            so a cancel raised here preempts the retry-substage emit.
        on_attempt: optional PRIMITIVE-only callback
            ``(attempt, max_attempts, sleep_s)`` for producer-side
            retry-attempt progress emission (PRI-6). ``attempt`` is the
            1-based count of the attempt that just failed (== the retry
            number about to be scheduled). The helper passes ONLY
            primitives; the structured retry-substage payload is built
            exclusively by ``_bridge_helper_on_attempt`` (single-emitter
            per durable #30 — the helper stays provider-agnostic and
            never constructs a substage dict).
        sleep_fn: injection seam for tests. ``None`` (default) resolves
            ``time.sleep`` DYNAMICALLY at call time so ``mock.patch(
            "time.sleep")`` intercepts the backoff (a fixed default bound at
            import time would not be patchable).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    _sleep = sleep_fn if sleep_fn is not None else time.sleep

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
            # on_attempt fires AFTER on_retry so a cancel raised in
            # on_retry preempts the "retrying" substage emit. Passes
            # the 1-based count of the just-failed attempt + primitives
            # only (the substage dict is built in the bridge factory).
            if on_attempt is not None:
                on_attempt(attempt + 1, max_attempts, sleep_s)
            logger.warning(
                "retry_with_exponential_backoff: attempt %d/%d "
                "raised %s; sleeping %.2fs",
                attempt + 1, max_attempts, type(exc).__name__, sleep_s,
            )
            _sleep(sleep_s)
    assert last_exc is not None  # exhausted loop implies last_exc set
    raise last_exc


def _bridge_helper_on_attempt(
    progress_callback: Any,
    stage: str,
    provider: str,
) -> Optional[Callable[[int, int, float], None]]:
    """Build the SINGLE canonical ``on_attempt`` emitter for a retry storm.

    This is the ONLY place a retry-substage payload (``kind='retry'``) is
    constructed — per durable #30 (canonical-emit at producer-boundary):
    no adapter inline-builds a retry dict; every adapter that wants the
    "retrying N/M" signal threads the closure this factory returns into
    ``retry_with_exponential_backoff(on_attempt=...)``.

    The returned closure converts the helper's PRIMITIVE
    ``(attempt, max_attempts, sleep_s)`` callback into a structured
    ``retry_info`` payload and emits it through the consumer's existing
    ``on_substage_progress`` seam (the ``retry_info`` channel is added
    consumer-side in prismweb β; the two repos ship atomically).

    Args:
        progress_callback: object exposing ``on_substage_progress``
            (the prismweb ``WebProgressCallback``). ``None`` or any
            object lacking the method → returns ``None`` (no-op; the
            adapter passes ``on_attempt=None`` and behaviour is
            unchanged for CLI / library-direct / unit-test callers).
        stage: pipeline stage key (``'translate'`` / ``'retrieve'``)
            the retrying adapter belongs to (static routing per S1v2-C3).
        provider: human display label bound here so the helper itself
            stays provider-agnostic (builder flag-2 resolution).

    Returns:
        A ``(attempt, max_attempts, sleep_s) -> None`` closure, or
        ``None`` when no usable callback was supplied.
    """
    if progress_callback is None or not hasattr(
        progress_callback, "on_substage_progress"
    ):
        return None

    def _on_attempt(attempt: int, max_attempts: int, sleep_s: float) -> None:
        retry_info = {
            "kind": "retry",
            "attempt": attempt,
            "max_attempts": max_attempts,
            "next_retry_delay_s": round(float(sleep_s), 1),
            "provider": provider,
        }
        progress_callback.on_substage_progress(
            stage,
            f"Retrying {provider}",
            attempt,
            max_attempts,
            f"next attempt in {sleep_s:.0f}s",
            retry_info=retry_info,
        )

    return _on_attempt
