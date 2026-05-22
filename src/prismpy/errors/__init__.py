"""Producer-boundary error classification package.

Emits a structured ``ErrorEventDict`` payload at the broad-except catch
sites where a typed exception would otherwise be flattened to its
string form, so the consumer can dispatch on ``error_class`` instead
of pattern-matching the formatted message.
"""

from prismpy.errors.classify import classify_to_event_dict
from prismpy.errors.event import ErrorEventDict

__all__ = ["ErrorEventDict", "classify_to_event_dict"]
