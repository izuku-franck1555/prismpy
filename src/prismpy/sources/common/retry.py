"""Canonical retry helper for requests-based adapter calls.

Centralises exponential-backoff + jitter retry semantics for any
``requests``-backed adapter call so the policy lives in one place.
Non-``requests`` adapters (cdsapi, custom TCP, etc.) need their own
retry wrapper and are not in scope.

Default schedule (max_attempts=6, base_delay_s=5.0):
    initial call + sleeps of 5, 10, 20, 40, 80 s ≈ 155 s budget
    (each sleep modulated by ±jitter_ratio to spread thundering-herd
    bursts when many tiles retry simultaneously).
"""
from __future__ import annotations

import inspect
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
    backoff_multiplier: float = 2.0,
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
        backoff_multiplier: per-attempt growth factor of the sleep
            schedule (default 2.0 → doubling). Adapters with a different
            configured backoff pass their own factor so this helper
            reproduces their schedule rather than forcing a fixed doubling.
        exception_classes: tuple of exception types treated as
            transient. ``requests.RequestException`` is deliberately
            NOT a default because it would catch programmer errors
            like InvalidURL.
        on_retry: optional callback ``(attempt_index, exc, sleep_s)``,
            0-based, invoked between attempts (e.g. for cancel-aware
            consumers that raise during the sleep window). Fires BEFORE
            ``on_attempt`` so such a raise preempts the progress emit.
        on_attempt: optional primitive-only callback
            ``(attempt, max_attempts, sleep_s)`` for retry-attempt
            progress, where ``attempt`` is the 1-based count of the
            just-failed attempt. The helper passes only primitives; the
            structured progress payload is built solely by
            ``_bridge_helper_on_attempt`` so the helper stays
            provider-agnostic.
        sleep_fn: injection seam for tests. ``None`` (default) resolves
            ``time.sleep`` dynamically at call time so ``mock.patch(
            "time.sleep")`` intercepts the backoff (a default bound at
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
            sleep_s = base_delay_s * (backoff_multiplier ** attempt)
            if jitter_ratio:
                spread = sleep_s * jitter_ratio
                sleep_s += random.uniform(-spread, spread)
            sleep_s = max(0.0, sleep_s)
            if on_retry is not None:
                on_retry(attempt, exc, sleep_s)
            # After on_retry, so a cancel raised there preempts this emit.
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
    """Build the single ``on_attempt`` emitter for a retry storm.

    This is the only place a ``{'kind': 'retry', ...}`` progress payload is
    constructed, so no adapter inline-builds one; adapters thread the
    returned closure into ``retry_with_exponential_backoff(on_attempt=...)``.

    The closure converts the helper's primitive
    ``(attempt, max_attempts, sleep_s)`` callback into a structured
    ``retry_info`` payload emitted via ``on_substage_progress``. Callbacks
    that accept the ``retry_info`` keyword get the payload; callbacks that
    implement only the 5-arg ``on_substage_progress(stage, task, current,
    total, detail)`` get the same call WITHOUT it — so a retry storm never
    raises ``TypeError`` on an older callback.

    Args:
        progress_callback: object exposing ``on_substage_progress``.
            ``None`` or any object lacking the method → returns ``None``
            (no-op; behaviour unchanged for CLI / library / test callers).
        stage: pipeline stage key (``'translate'`` / ``'retrieve'``).
        provider: human display label, bound here so the helper stays
            provider-agnostic.

    Returns:
        A ``(attempt, max_attempts, sleep_s) -> None`` closure, or
        ``None`` when no usable callback was supplied.
    """
    if progress_callback is None or not hasattr(
        progress_callback, "on_substage_progress"
    ):
        return None

    emit = progress_callback.on_substage_progress

    # Detect retry_info support once (an older callback accepts only the
    # 5-arg shape; **kwargs counts as accepting it).
    try:
        params = inspect.signature(emit).parameters
        accepts_retry_info = "retry_info" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        accepts_retry_info = False

    def _on_attempt(attempt: int, max_attempts: int, sleep_s: float) -> None:
        task = f"Retrying {provider}"
        detail = f"next attempt in {sleep_s:.0f}s"
        if accepts_retry_info:
            emit(
                stage,
                task,
                attempt,
                max_attempts,
                detail,
                retry_info={
                    "kind": "retry",
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "next_retry_delay_s": round(float(sleep_s), 1),
                    "provider": provider,
                },
            )
        else:
            # 5-arg callback: omit retry_info so it never raises TypeError.
            emit(stage, task, attempt, max_attempts, detail)

    return _on_attempt
