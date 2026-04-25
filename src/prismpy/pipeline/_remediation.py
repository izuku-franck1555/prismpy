"""V2-22c-PRE.4 — REMEDIATION stage helpers.

Provides:

* :class:`RemediationBlocked` — exception raised by the remediation
  applier when a server-side invariant is violated. Carries a
  structured ``reason`` field per D33 (Veto #4 conservative
  thresholds) so the cockpit renders the correct AC-9.3 BLOCK copy
  variant + provenance.json captures the audit trail. The
  ``REASON_VALID`` frozenset enforces the enum at construction so
  schema drift surfaces as a ``ValueError`` at the boundary.

* :func:`_veto_4_tier` — re-validates the imputation against the
  target cell's neighborhood, returning a 3-state tier:

  - ``'silent'`` — all neighbors share the target's soil class;
    imputation is agronomically defensible. No exception.
  - ``'warn'`` — 1+ neighbors in different class but NOT all;
    acknowledgeable. Caller raises only when
    ``veto_4_acknowledged`` is False.
  - ``'block'`` — target soil_class null OR any neighbor null OR
    all neighbors in different class. NON-acknowledgeable per D33;
    caller raises unconditionally.

* :func:`_block_message` — server-side AC-9.3 copy mirror.

* :func:`_idw_neighbor_cells` — minimal IDW-style neighbor helper
  for PRE scope. Walks the grid for cells within ``max_radius_cells``
  of the target cell's row/col. The full IDW spec lives in V2-22c
  R5; PRE only needs enough to drive the Veto #4 server enforcement
  for the unit-test matrix.

Critical bindings (evaluator §12.4 + §12.5):

* ``RemediationBlocked.REASON_VALID == frozenset({'target_null',
  'neighbor_null', 'all_cross_class', 'veto_4_warn_unacknowledged'})``
  — exact strings, no synonym drift; cockpit AC-9.3 copy variants
  depend on these.

* HIGH 5 / Δ.6 closure pattern: build the FULL neighbor profile
  list FIRST (preserve nulls), THEN check. A refactor that filters
  ``None`` entries before the null-check would silently demote a
  ``neighbor_null`` BLOCK to ``silent`` / ``warn`` tier — Test #10
  in the 10-test matrix catches that regression.

* The ``if tier == 'block': raise`` clause runs UNCONDITIONALLY
  before any ``veto_4_acknowledged`` flag check. Adversarial-bypass
  tests #5/#6/#7 catch the refactor that accidentally inverts the
  order.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


class RemediationBlocked(Exception):
    """Raised by :func:`_execute_remediation` when a remediation
    action violates a server-side invariant.

    Carries a structured ``reason`` field per D33 (Veto #4
    conservative thresholds). The cockpit's AC-9.3 BLOCK copy
    renders one of three variants based on the reason, and
    ``provenance.json`` persists the structured payload for audit
    trail.

    Constructor enforces the reason enum via ``REASON_VALID`` —
    an unrecognized reason raises ``ValueError`` at the boundary,
    surfacing schema drift early instead of letting an invalid
    string propagate into the cockpit copy mapping (where it
    would render as the catch-all "remediation blocked" string and
    erase the user-actionable specifics).
    """

    # Reason enum (matches AC-9.3 cockpit copy variants exactly per
    # evaluator §12.4 binding — no synonym drift).
    REASON_TARGET_NULL = "target_null"
    REASON_NEIGHBOR_NULL = "neighbor_null"
    REASON_ALL_CROSS_CLASS = "all_cross_class"
    REASON_WARN_UNACKED = "veto_4_warn_unacknowledged"

    REASON_VALID = frozenset({
        REASON_TARGET_NULL,
        REASON_NEIGHBOR_NULL,
        REASON_ALL_CROSS_CLASS,
        REASON_WARN_UNACKED,
    })

    def __init__(
        self,
        cell_id: Optional[int] = None,
        reason: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        if reason is not None and reason not in self.REASON_VALID:
            raise ValueError(
                f"Invalid RemediationBlocked.reason={reason!r}; "
                f"must be one of {sorted(self.REASON_VALID)}"
            )
        self.cell_id = cell_id
        self.reason = reason
        super().__init__(
            message
            or f"RemediationBlocked: {reason} cell={cell_id}"
        )


def _idw_neighbor_cells(
    target_cell_id: int,
    radius: int,
    unified_data,
) -> List[int]:
    """Return cell IDs within ``radius`` rows/cols of the target.

    PRE scope: minimal grid-distance neighbor finder; the full IDW
    weighting + clipping logic is V2-22c R5 builder work. The
    return value drives :func:`_veto_4_tier` neighbor validation —
    we need a list of candidate neighbor cell IDs, not weighted
    contributions.

    Returns an empty list when the grid is missing or the target
    cell isn't on the grid; callers (``_veto_4_tier``) interpret
    that as a BLOCK ``neighbor_null`` condition.
    """
    grid = (
        unified_data.grid
        if unified_data and hasattr(unified_data, 'grid')
        else None
    )
    if grid is None or not getattr(grid, 'cells', None):
        return []
    target_cell = next(
        (c for c in grid.cells if c.cell_id == target_cell_id),
        None,
    )
    if target_cell is None:
        return []
    target_row = getattr(target_cell, 'row', None)
    target_col = getattr(target_cell, 'col', None)
    if target_row is None or target_col is None:
        return []

    neighbors: List[int] = []
    for cell in grid.cells:
        if cell.cell_id == target_cell_id:
            continue
        row = getattr(cell, 'row', None)
        col = getattr(cell, 'col', None)
        if row is None or col is None:
            continue
        # Chebyshev distance — cheap proxy for "within N grid cells".
        # Sufficient for Veto #4 same-class agreement testing.
        if max(abs(row - target_row), abs(col - target_col)) <= radius:
            neighbors.append(cell.cell_id)
    return neighbors


def _veto_4_tier(
    imputation_action: Dict[str, Any],
    unified_data,
) -> Tuple[str, Optional[str]]:
    """Server-side re-validation of Veto #4 per D33 conservative
    thresholds.

    Returns ``(tier, block_reason)`` where:

    * ``tier`` ∈ ``{'silent', 'warn', 'block'}``
    * ``block_reason`` ∈ ``{'target_null', 'neighbor_null',
      'all_cross_class'}`` or ``None``. Only populated when
      ``tier == 'block'``.

    Tier semantics (D33 binding):

    - ``silent`` — ALL neighbors in same soil class as target.
      Imputation is agronomically defensible.
    - ``warn`` — Some neighbors in same class, some not.
      Acknowledgeable: caller raises only if user did NOT
      acknowledge Veto #4 client-side.
    - ``block`` — Target ``soil_class`` null OR ANY neighbor null
      OR ALL neighbors in different class. NON-acknowledgeable per
      D33; caller raises unconditionally.

    D31 carve-out: Veto #4 only applies to donor-derived imputation
    methods (method name contains ``idw`` or ``neighbor``). Direct
    user-supplied Override values bypass — return ``('silent', None)``.
    """
    target_cell_id = imputation_action.get("cell_id")
    method = imputation_action.get("method", "") or ""

    # D31 carve-out — donor-provenance-only enforcement.
    if (
        "idw" not in method.lower()
        and "neighbor" not in method.lower()
    ):
        return ("silent", None)

    soil = (
        unified_data.soil
        if unified_data and hasattr(unified_data, 'soil')
        else {}
    )
    target_profile = soil.get(target_cell_id) if isinstance(soil, dict) else None

    # BLOCK case 1: target soil_class null
    if (
        target_profile is None
        or not getattr(target_profile, 'surface_texture', None)
    ):
        return ("block", RemediationBlocked.REASON_TARGET_NULL)
    target_class = target_profile.surface_texture

    radius = (
        imputation_action.get("parameters", {})
        .get("max_radius_cells", 3)
    )

    # HIGH 5 / Δ.6 closure pattern — build the FULL neighbor profile
    # list FIRST (preserve nulls), THEN check. Filtering None
    # entries before the null check would silently demote a
    # neighbor_null BLOCK to silent/warn — adversarial-bypass test
    # #10 catches a refactor that introduces that subtle reorder.
    neighbor_ids = _idw_neighbor_cells(target_cell_id, radius, unified_data)
    if not neighbor_ids:
        return ("block", RemediationBlocked.REASON_NEIGHBOR_NULL)

    neighbor_profiles = [
        soil.get(n) if isinstance(soil, dict) else None
        for n in neighbor_ids
    ]

    # BLOCK case 2: ANY neighbor profile is None OR has no
    # surface_texture. D33 conservative — distinguishes "we don't
    # know" from "we know all-different" (case 3 below).
    if any(
        p is None or not getattr(p, 'surface_texture', None)
        for p in neighbor_profiles
    ):
        return ("block", RemediationBlocked.REASON_NEIGHBOR_NULL)

    # All neighbors guaranteed valid past this point. Safe to
    # filter for class match.
    same_class = [
        p for p in neighbor_profiles
        if p.surface_texture == target_class
    ]

    if len(same_class) == len(neighbor_profiles):
        return ("silent", None)
    if len(same_class) == 0:
        # BLOCK case 3: ALL neighbors in different class — D33
        # NON-acknowledgeable. Imputation is not agronomically
        # defensible.
        return ("block", RemediationBlocked.REASON_ALL_CROSS_CLASS)
    return ("warn", None)


def _block_message(imp: Dict[str, Any], reason: str) -> str:
    """AC-9.3-aligned BLOCK reason copy — server-side mirror of
    cockpit text. Server emits the same string the cockpit will
    render so provenance.json carries the user-visible reason
    verbatim and the test matrix can cross-check identity."""
    cell_id = imp.get("cell_id")
    if reason == RemediationBlocked.REASON_TARGET_NULL:
        return (
            f"Imputation blocked: target cell #{cell_id} has unknown "
            f"soil class. Use Substitute (cascade) or Exclude instead."
        )
    if reason == RemediationBlocked.REASON_NEIGHBOR_NULL:
        return (
            f"Imputation blocked: cell #{cell_id} has neighbors with "
            f"unknown soil class. Use Substitute (cascade) instead."
        )
    if reason == RemediationBlocked.REASON_ALL_CROSS_CLASS:
        return (
            f"Imputation blocked: all neighbor cells of #{cell_id} are "
            f"in different soil classes. Imputation is not agronomically "
            f"defensible here. Use Substitute (cascade) instead."
        )
    return f"Imputation blocked for cell #{cell_id}: {reason}"
