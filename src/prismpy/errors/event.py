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

import sys
from typing import Any, Dict, List, Optional, TypedDict

# Required / NotRequired moved into typing at 3.11; fall back to
# typing_extensions on 3.10 (prismpy supports >=3.10,<3.13).
if sys.version_info >= (3, 11):
    from typing import NotRequired
else:  # pragma: no cover - 3.10 path
    from typing_extensions import NotRequired


class ErrorEventDict(TypedDict):
    """Structural error-event payload.

    ``error_class`` and ``message`` are REQUIRED so the consumer can
    dispatch on class without ``KeyError``. The remaining keys are
    optional (``NotRequired``) — present iff the exception carried the
    corresponding attribute (or context supplied enough to derive it).
    """

    error_class: str
    message: str
    source: NotRequired[Optional[str]]
    missing_tiles: NotRequired[Optional[List[Any]]]
    partial_progress: NotRequired[Optional[Dict[str, int]]]
    recoverable: NotRequired[Optional[bool]]
