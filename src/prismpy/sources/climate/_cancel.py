"""Shared cancellation primitives for climate + translator download loops.

V2-22b Group L — cooperative cancellation across prismpy's long-running
hot paths (TAMSAT Phase 1/Phase 2, AgERA5 year loop, NASA POWER year +
retry loops, per-translator per-cell loops).

Design notes
------------
* **Callable-based, not ``threading.Event``**: the DB is the authoritative
  source of cancellation state. A separate ``Event`` would add a second
  source of truth and a push-from-DB race surface. The ``cancel_check``
  callable closes over ``PipelineRun.objects.only('status').get(...)`` on
  the prismweb side; prismpy stays DB-agnostic.

* **Raise, not return-sentinel**: download loops are 2-4 levels deep
  (``retrieve`` → ``_download_tamsat`` → ``_download_nc`` → future body,
  or ``translate`` → ``_download_climate_for_cells`` → ``source.retrieve``).
  An exception unwinds cleanly without each level needing to propagate a
  sentinel return. Caught once at the ``pipeline.execute`` boundary and
  converted to the handler-local cleanup in ``_execute_pipeline``.

* **`cancel_check=None` is the unit-test fast path**: non-web callers
  (CLI, direct prismpy usage, unit tests without a pipeline context)
  pass ``None`` and every ``raise_if_cancelled`` call becomes a no-op.
  Zero behavioral change for non-pipeline usage.
"""
from __future__ import annotations

from typing import Callable, Optional


class PipelineCancelled(Exception):
    """Raised by download loops when a user cancel is observed mid-execution.

    Catchers at the pipeline layer convert this into a handler-local
    cleanup that mirrors the existing EARLY_EXIT path (``tasks.py``
    ``_execute_pipeline_cancelled_cleanup``). Not caught by the broad
    ``except Exception`` handlers that wrap third-party download work
    — those sites install a ``except PipelineCancelled: raise`` carve-out
    so the cancel propagates without being rewritten as an error-state.

    Attributes
    ----------
    where : str
        Short context identifier recorded at the raise site (e.g.,
        ``'tamsat.phase1.as_completed'``, ``'nasa_power.year=2022'``).
        Accessible as ``exc.where`` for test assertions and the
        ``[PIPELINE <id>] CANCELLED where=<site>`` log line.
    """

    def __init__(self, where: str):
        super().__init__(where)
        self.where = where


def raise_if_cancelled(
    cancel_check: Optional[Callable[[], bool]],
    where: str,
) -> None:
    """Raise ``PipelineCancelled(where)`` if ``cancel_check`` returns True.

    Parameters
    ----------
    cancel_check : Optional[Callable[[], bool]]
        Callable returning ``True`` when the user has requested
        cancellation. ``None`` disables cancellation (unit-test fast
        path, library-direct usage).
    where : str
        Short context string for the raise message. Recorded on the
        exception's ``.where`` attribute for log emission and tests.

    Raises
    ------
    PipelineCancelled
        When ``cancel_check`` is not ``None`` and ``cancel_check()``
        returned ``True``.
    """
    if cancel_check is not None and cancel_check():
        raise PipelineCancelled(where)
