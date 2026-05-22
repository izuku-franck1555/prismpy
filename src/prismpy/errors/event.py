"""Canonical structured error-event payload that survives the
prismpy→prismweb wire boundary.

The producer (prismpy) populates this dict at the broad-except catch
sites where a typed exception would otherwise be flattened to ``str(e)``;
the consumer (prismweb) dispatches user-facing copy off
``error_class``.

The key is ``error_class`` (not ``class``) so the payload is JSON-/
TypedDict-safe and never collides with Python's soft-keyword tokenisation.
``message`` is always populated; the other keys are optional and absent
when the exception did not carry the corresponding attribute.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class ErrorEventDict(TypedDict, total=False):
    """Structural error-event payload.

    Required (populated by ``classify_to_event_dict``):
    - ``error_class``: ``type(exc).__name__`` — the consumer dispatch key.
    - ``message``: ``str(exc)`` — full formatted message.

    Optional (present iff the exception carries the corresponding attr):
    - ``source``: provider key (e.g. ``'nasa_power'``, ``'tamsat'``).
    - ``missing_tiles``: list of missing tile / cell / asset ids.
    - ``partial_progress``: derived ``{succeeded, failed, total}`` when a
      ``missing_tiles`` count + grid-total context are available.
    - ``recoverable``: whether the failure is treated as transient.
    """

    error_class: str
    message: str
    source: Optional[str]
    missing_tiles: Optional[List[Any]]
    partial_progress: Optional[Dict[str, int]]
    recoverable: Optional[bool]
