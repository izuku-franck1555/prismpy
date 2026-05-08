"""Cockpit affordance routing — per-cell `(check_id, zone, elevation,
n_candidates_in_radius)` → ``AffordanceType``.

Sprint E.2 AC-E2-3 + §0.2 canonical-layer #1 (action vocabulary).
The cockpit's bucket-section UI calls ``route_affordance`` once per
flagged cell to decide whether to surface an Acknowledge / Skip /
Apply Interpolation / Document Override action — or to fall through
to ``"rerun_full_sources"`` (the EXIT BOUNDARY that spawns a new
pipeline run via the existing rerun mechanism rather than writing
a per-cell decision-log entry).

Per durable §24 canonical-source-or-pin: routing decisions live ONLY
in this helper. Two structural pins enforce:

* ``test_action_vocabulary_parity.py`` (per §0.2 #1) — asserts
  ``set(typing.get_args(AffordanceType))`` covers the
  ``AFFORDANCE_TO_ACTION_MAP`` keys ∪ the ``"rerun_full_sources"``
  exit-boundary set, AND ``set(AFFORDANCE_TO_ACTION_MAP.values()) -
  {None}`` ⊆ ``set(typing.get_args(DecisionAction))``. Drift in
  either direction fires loud at CI time.

* The "F-E2-5 affordance routing fragmentation" walker (per §4)
  asserts no other module returns ``AffordanceType`` literals
  directly.

Per §0.2 #1: the ``"rerun_full_sources"`` AffordanceType is an EXIT
BOUNDARY that does NOT map to a CellDecisionRecord action — it
spawns a new pipeline run via the existing rerun mechanism (no
decision-log entry; the rerun itself is the audit trail). Per Drill
Q (d): consumers seeing ``"rerun_full_sources"`` MUST NOT attempt to
create a CellDecisionRecord.

Routing rules (Sprint E.2 Decision 5 Bucket 4 + Stage-0 §11 +
Draft 5.2 Cwa Highland-precip exclusion):

* ``value_range_{tmax,tmin,srad,rh,wind}`` × any platform × any
  zone × ``n_candidates >= 1`` → ``"interpolate"``.
* Same check_ids × ``n_candidates == 0`` → ``"skip"`` (per WA
  CA-1; closes the 0-neighbour failure path BEFORE the engine
  raises).
* ``value_range_precip`` × any platform × non-highland zones ×
  ``n_candidates >= 1`` → ``"interpolate"``.
* ``value_range_precip`` × any platform × Köppen ∈ {Cwa} ×
  elevation_m > 1500.0 → ``"skip"`` (Decision 2 caveat 2;
  Highland-precip orographic exclusion per Daly 2006). Note: the
  registry's Cwa zone carries the "Subtropical highland" empirical
  label per Sprint F substrate; East African Highland zones (Cwb /
  Cwc proper) are V2-19.5 Data Bootstrapper future-explore.
* ``region_specific_bounds`` × any platform × any zone ×
  ``n_candidates >= 1`` → ``"interpolate"``.
* ``value_range_soil_*`` × any platform × any zone → ``"skip"``.
* ``temporal_completeness`` × ``sarra_py`` → ``"rerun_full_sources"``.
* ``cross_variable`` × any platform × any zone → ``"skip"``.
* ``coverage_per_cell`` × any platform × any zone → ``"rerun_full_sources"``.
* ``crop_region_mismatch`` × any platform × any zone → ``"override"``.
* All other check_ids → ``"acknowledge"``.
"""

from __future__ import annotations

from typing import Any, Dict, Final, Literal, Optional

from prismpy.config.schema import Platform
from prismpy.koppen.zones import KoppenZone
from prismpy.models.decision_log import DecisionAction
from prismpy.warnings.categories import WarningCategory


# ── Action vocabulary ───────────────────────────────────────────────


AffordanceType = Literal[
    "interpolate",
    "skip",
    "override",
    "acknowledge",
    "rerun_full_sources",
]


# ``AFFORDANCE_TO_ACTION_MAP`` per §0.2 canonical-source #1. Maps
# each AffordanceType to the corresponding ``CellDecisionRecord.action``
# Literal. ``"rerun_full_sources"`` maps to ``None`` (exit boundary;
# no decision-log entry is created — the rerun itself is the audit
# trail).
AFFORDANCE_TO_ACTION_MAP: Final[dict[AffordanceType, Optional[DecisionAction]]] = {
    "interpolate": "apply_interpolation",
    "skip": "skip_from_analysis",
    "override": "document_override",
    "acknowledge": "acknowledge",
    "rerun_full_sources": None,  # exit boundary
}


# ── Internal: check_id classification ───────────────────────────────


# value_range_X check_ids that get routed to "interpolate" when
# neighbours exist. AC-E2-3 + Stage-0 Decision 5 Bucket 4.
_VALUE_RANGE_INTERPOLATABLE: Final[frozenset[str]] = frozenset(
    {
        "value_range_tmax",
        "value_range_tmin",
        "value_range_srad",
        "value_range_rh",
        "value_range_wind",
    }
)

# Highland zones whose precipitation is orographically excluded
# from interpolation when elevation > 1500 m. Sprint E.2 ships
# with Cwa per Draft 5.2 (registry's "Subtropical highland" label);
# Cwb/Cwc proper are V2-19.5 future-explore.
_HIGHLAND_PRECIP_ZONES: Final[frozenset[str]] = frozenset({"Cwa"})

# Highland-precip elevation threshold (m). Per Decision 2 caveat 2
# anchored in Daly 2006; the orographic-effects regime kicks in
# above this elevation.
_HIGHLAND_ELEVATION_THRESHOLD_M: Final[float] = 1500.0


# ── Routing engine ──────────────────────────────────────────────────


def route_affordance(
    check_id: str,
    platform: Platform,
    zone: KoppenZone,
    elevation_m: float,
    n_candidates_in_radius: int,
    cell_failure_context: Dict[str, Any],
) -> AffordanceType:
    """Route a flagged cell to its cockpit affordance.

    Args:
        check_id: Per-cell warning identifier (Sprint F vocabulary).
        platform: Platform whose validation flagged the cell.
        zone: Köppen-Geiger zone code (canonical Literal per
            AC-E2-20).
        elevation_m: Cell elevation (m). Used for Highland-precip
            exclusion (Decision 2 caveat 2).
        n_candidates_in_radius: Count of candidate neighbour cells
            within the IDW search radius. Per §0.2 #1 + WA CA-1
            architectural concern: when 0, the cell is routed to
            ``"skip"`` BEFORE any IDW engine call would raise.
        cell_failure_context: Per-cell metric dict carrying
            additional signals the cockpit's per-cell routing
            engine reads (gap_count / coverage_pct / layer_idx /
            daily_failure_count / has_existing_override /
            is_highland_precip / profile_depth_m). Sprint E.2
            AC-E2-3 ext (Codex Gate A MEDIUM A1 + Builder
            Sub-CA #6): the context is consumed alongside the
            ``RoutingDecision`` triple in
            :func:`prismpy.cockpit.routing_decision.bucket_for`
            for per-cell-aware bucket assignment. Empty dict
            (``{}``) is the canonical "no per-cell context"
            sentinel — required as a positional argument so a
            structural pin enforces every callsite passes it
            explicitly + can't accidentally drop the context.

    Returns:
        ``AffordanceType`` — the cockpit affordance to surface.
    """
    # ``cell_failure_context`` is required by the signature so a
    # structural pin can enforce explicit pass-through at every
    # callsite (per durable §24 canonical-source-or-pin: the
    # context is the single shared input ``bucket_for`` reads
    # alongside the affordance, so callers MUST surface what they
    # know rather than silently defaulting to None). The argument
    # is read defensively to tolerate future expansions of the
    # context shape.
    _ = cell_failure_context  # explicit-discard placeholder; future
    # affordance refinements consume this. The kwarg's presence at
    # every callsite is what matters for the structural pin — its
    # actual contents wire through to ``bucket_for`` at the call
    # boundary, not inside this function (route_affordance owns
    # the affordance dimension; bucket_for owns the bucket +
    # variant dimensions).
    # value_range_precip with Highland zones + high elevation →
    # routed to skip per Decision 2 caveat 2 (orographic exclusion).
    # The Highland exclusion takes precedence over the n_candidates
    # check — even with neighbours available, IDW isn't reliable
    # for orographic precipitation (Daly 2006).
    if check_id == "value_range_precip":
        if zone in _HIGHLAND_PRECIP_ZONES and elevation_m > _HIGHLAND_ELEVATION_THRESHOLD_M:
            return "skip"
        if n_candidates_in_radius >= 1:
            return "interpolate"
        return "skip"  # 0-neighbour fallback

    # Other interpolatable value-range check_ids.
    if check_id in _VALUE_RANGE_INTERPOLATABLE:
        if n_candidates_in_radius >= 1:
            return "interpolate"
        return "skip"  # 0-neighbour fallback

    # region_specific_bounds: zonal envelope tail; interpolatable
    # like the value_range checks.
    if check_id == "region_specific_bounds":
        if n_candidates_in_radius >= 1:
            return "interpolate"
        return "skip"

    # Soil profile gaps don't follow climate gradients; never
    # interpolate.
    if check_id.startswith("value_range_soil_"):
        return "skip"

    # SARRA-Py temporal completeness: rerun upstream sources to
    # close the gap.
    if check_id == "temporal_completeness" and platform == Platform.SARRA_PY:
        return "rerun_full_sources"

    # Cross-variable consistency violations are too multi-dimensional
    # to interpolate.
    if check_id == "cross_variable":
        return "skip"

    # Coverage-per-cell: rerun with full sources to fill.
    if check_id == "coverage_per_cell":
        return "rerun_full_sources"

    # Crop-region-mismatch is wizard-time documented-override
    # territory. Reference the canonical WarningCategory enum value
    # rather than a bare string per durable §24 +
    # ``test_no_bare_warning_category_strings.py`` pin.
    if check_id == WarningCategory.CROP_REGION_MISMATCH.value:
        return "override"

    # Default: bucket-2 acknowledgement.
    return "acknowledge"


__all__ = [
    "AFFORDANCE_TO_ACTION_MAP",
    "AffordanceType",
    "route_affordance",
]
