"""The one digest of a roster's cell-id set, shared by the crop-presence record, each engine's
roster read-back and any consumer that compares them (prismweb's seal)."""
from __future__ import annotations

import hashlib
from typing import Iterable

ROSTER_DIGEST_ENCODING = "roster-ids/v1"


def roster_id_digest(ids: Iterable[int]) -> str:
    """sha256 of ``"roster-ids/v1\\n"`` followed by the sorted ids, one per line.

    A roster is a set of integer cell ids: a duplicate, a bool or a non-integer is refused rather
    than coerced, so two different rosters can never share a digest."""
    values = list(ids)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError("roster ids must be integers")
    if len(set(values)) != len(values):
        raise ValueError("a roster holds each cell id once")
    body = "\n".join(str(v) for v in sorted(values))
    return hashlib.sha256(f"{ROSTER_DIGEST_ENCODING}\n{body}".encode("ascii")).hexdigest()
