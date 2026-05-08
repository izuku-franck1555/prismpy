"""Canonical thresholds for cockpit per-cell bucket routing.

Sprint E.2 AC-E2-3 ext + Codex Gate A MEDIUM A1.

The cockpit's per-cell bucket assignment refines the legacy
per-category :data:`prismpy.cockpit.manifest._DIMENSION_BUCKET_MAP`
(every dimension-toggle category routed to bucket 3 TRUE_EXCLUDE
unconditionally) into per-cell-aware logic — short-gap variants
of ``temporal`` + ``coverage_per_cell`` route to bucket 4
INTERPOLATABLE, longer-gap variants stay at bucket 3. The split
threshold lives here so the producer (manifest builder) + tests
+ cockpit display all read the same number.

Per durable §24 canonical-source-or-pin:

* :data:`TEMPORAL_GAP_BUCKET_4_MAX_DAYS` — the gap-day count at
  or below which a temporal_completeness failure routes to
  bucket 4 (interpolatable). Above this, bucket 3 (true exclude).
* :data:`COVERAGE_PER_CELL_BUCKET_4_MIN_PCT` — the per-cell
  coverage percentage at or above which a coverage_per_cell
  failure routes to bucket 4. Below this, bucket 3.
* :data:`PROFILE_DEPTH_BUCKET_3_MIN_M` — the minimum DSSAT-
  required soil profile depth; cells below this stay bucket 3
  (profile depth varies physically, not interpolable).

A structural pin at
``tests/structural/test_bucket_routing_canonical_thresholds.py``
asserts these constants live ONLY in this module — every
:func:`bucket_for` callsite + every cockpit display surface
imports from here, NOT from a sibling literal.

Threshold rationale + precedents:

* 14-day gap cutoff matches the Sprint S precedent for
  short-gap interpolation (sub-fortnight gaps are the IDW
  sweet spot per crop-modeling-specialist guidance; longer
  gaps stretch the spatial-correlation assumption past the
  validator's confidence threshold).
* 80% coverage cutoff matches the Sprint F precedent for
  per-cell completeness — below 80% the cell has substantial
  data missing AND the IDW imputation would itself rely on
  too few observed days to be informative.
* 0.20m DSSAT minimum profile depth is the canonical ECOCROP
  +DSSAT consensus; profile depth varies physically across
  the landscape (bedrock-controlled), so IDW imputation is
  not appropriate even for short-gap cells.
"""
from __future__ import annotations


# Temporal completeness — gap-day count split between
# bucket 4 INTERPOLATABLE (short gap; IDW-able) and bucket 3
# TRUE_EXCLUDE (long gap; not interpolable).
#
# A cell with ``gap_count <= 14`` routes to bucket 4; above
# routes to bucket 3. Per Sprint S precedent + crop-modeling-
# specialist sub-fortnight guidance.
TEMPORAL_GAP_BUCKET_4_MAX_DAYS: int = 14


# Coverage-per-cell — coverage percentage split between
# bucket 4 INTERPOLATABLE (mostly-present; gaps fillable) and
# bucket 3 TRUE_EXCLUDE (substantially missing).
#
# A cell with ``coverage_pct >= 80`` routes to bucket 4; below
# routes to bucket 3. Per Sprint F precedent (80% per-cell
# completeness floor for IDW-with-confidence).
COVERAGE_PER_CELL_BUCKET_4_MIN_PCT: float = 80.0


# Soil profile depth — minimum DSSAT-required depth in meters.
# Cells with ``total_depth < 0.20`` route to bucket 3
# unconditionally; profile depth varies physically (bedrock-
# controlled) so IDW imputation isn't appropriate. Mirrors
# the validator at ``scientific.py::_check_soil_profile_depth``.
PROFILE_DEPTH_BUCKET_3_MIN_M: float = 0.20


__all__ = [
    "TEMPORAL_GAP_BUCKET_4_MAX_DAYS",
    "COVERAGE_PER_CELL_BUCKET_4_MIN_PCT",
    "PROFILE_DEPTH_BUCKET_3_MIN_M",
]
