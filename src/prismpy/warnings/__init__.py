"""Cockpit warning-category taxonomy.

UNRELATED to Python's stdlib ``warnings`` module. This package
ships the canonical ``WarningCategory`` enum and the
``WARNING_BUCKET_MAP`` dictionary that drives prismweb's
5-bucket cockpit response taxonomy.

* :class:`WarningCategory` — Pydantic-compatible StrEnum whose
  values are the canonical strings every consumer compares
  against. Sprint D.1's existing ``UnavailableCause`` Literal
  values (``soil_no_hwsd_coverage``, ``soil_texture_invalid``,
  ``climate_rh_invalid``) are subsets of this enum and the
  StrEnum string-equality semantics keep the pre-existing
  call sites working without code change.

* :class:`WarningBucket` — the 5 cockpit response buckets
  (``AUTO_FIXABLE`` / ``INFORMATIONAL`` / ``TRUE_EXCLUDE`` /
  ``INTERPOLATABLE`` / ``MANUAL_OVERRIDE_WITH_EVIDENCE``).

* :data:`WARNING_BUCKET_MAP` — every category resolves to
  exactly one bucket; no orphans. The map is the chokepoint
  for "given this warning, how should the cockpit respond?"

* :func:`bucket_for` and :func:`categories_in_bucket` — thin
  helpers around the map, with deterministic ordering so a
  byte-identity serialization gate downstream cannot be
  defeated by hash-randomized iteration.

The package shape (single ``categories`` submodule plus this
``__init__`` re-export) keeps the import surface compact:
``from prismpy.warnings import WarningCategory`` works without
forcing every consumer to know the file layout.
"""
from __future__ import annotations

from prismpy.warnings.categories import (
    WARNING_BUCKET_MAP,
    WarningBucket,
    WarningCategory,
    bucket_for,
    categories_in_bucket,
)


__all__ = [
    "WARNING_BUCKET_MAP",
    "WarningBucket",
    "WarningCategory",
    "bucket_for",
    "categories_in_bucket",
]
