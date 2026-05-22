"""Structural classification of a caught exception into an
``ErrorEventDict``.

This is producer-side (prismpy): no user-facing copy and no Django /
prismweb references — every consumer (prismweb, CLI, library-direct)
reads the same structured payload and owns its own presentation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from prismpy.errors.event import ErrorEventDict


def classify_to_event_dict(
    exc: BaseException,
    context: Optional[Dict[str, Any]] = None,
) -> ErrorEventDict:
    """Extract the structured fields off ``exc`` into an ``ErrorEventDict``.

    ``context`` is an optional dict the caller threads in for fields the
    exception alone cannot supply — currently ``grid_total`` (int) so a
    ``partial_progress`` count can be derived from a ``missing_tiles``
    list. Absent or partial context leaves derived fields ``None``.
    """
    missing = getattr(exc, "missing_tiles", None)
    if missing is None:
        # Some translators expose the missing-asset list under
        # ``missing_assets``; mirror that without forcing every typed
        # error to settle on one attribute name.
        missing = getattr(exc, "missing_assets", None)

    partial_progress: Optional[Dict[str, int]] = None
    if missing is not None and context:
        total = context.get("grid_total")
        if isinstance(total, int) and total > 0:
            failed = len(missing)
            partial_progress = {
                "succeeded": max(0, total - failed),
                "failed": failed,
                "total": total,
            }

    event: ErrorEventDict = {
        "error_class": type(exc).__name__,
        "message": str(exc),
        "source": getattr(exc, "source", None),
        "missing_tiles": list(missing) if missing is not None else None,
        "partial_progress": partial_progress,
        "recoverable": getattr(exc, "recoverable", None),
    }
    return event
