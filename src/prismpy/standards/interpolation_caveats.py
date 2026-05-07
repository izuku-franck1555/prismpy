"""Per-zone × per-check_id caveat metadata for IDW interpolation.

Sprint E.2 AC-E2-6. The ``caveats_for()`` function below is the
domain-knowledge rule that maps (zone, check_id) → list of
``CaveatCode`` values — exactly the codes that should appear on an
imputed cell's ``InterpolatedCellRecord.caveat_codes`` field, and
exactly the codes whose phrases should appear in the methods-text
paragraph for that cell.

Three rules ship at MVP, each anchored in domain evidence:

* (Sahel BSh, value_range_precip) → ``["sahel-precip-convective"]``
  — Mathon et al. 2002 MCS decorrelation.
* (Sahel BSh, value_range_wind) → ``["sahel-wind-convective"]`` —
  same MCS regime.
* (Highland Cwa, value_range_precip) → ``["highland-orographic-excluded"]``
  — Daly 2006 orographic-effects principle.

Other (zone, check_id) pairs return ``[]`` — the absence of a caveat
is its own honest signal: the cell's imputation is straightforwardly
defensible without further qualification.

The function is pure (no I/O, no global state) so it's safe to call
from any context — schema validation, methods-text generation,
cockpit UI rendering. Determinism: returned lists are stable in
order across calls so test snapshots remain byte-identical.

Per durable §24 canonical-source-or-pin: this is the canonical
mapping. Consumers MUST route through this helper rather than
re-implementing the (zone, check_id) → caveats logic per-call-site.
``tests/structural/test_decision_workflow_canonical_layer.py`` (per
§0.2 #1-#7) walks the consumer surface to enforce.
"""

from __future__ import annotations

from prismpy.koppen.zones import KoppenZone
from prismpy.standards.caveat_codes import CaveatCode


# Sprint F shipped these check_id strings as the canonical
# per-cell warning identifiers (per
# ``prismpy/validators/scientific.py``). Sprint E.2's caveat rules
# match against them by string equality; a typed Literal for
# check_id is out-of-scope for E.2 (the validators emit raw strings;
# tightening to a Literal would touch ~12 validator surfaces and
# belongs in a dedicated check_id-canonicalisation sprint).
_CHECK_ID_PRECIP = "value_range_precip"
_CHECK_ID_WIND = "value_range_wind"


def caveats_for(zone: KoppenZone, check_id: str) -> list[CaveatCode]:
    """Return the caveat codes that apply to a (zone, check_id) pair.

    Args:
        zone: Köppen-Geiger zone code (one of the five
            ``KoppenZone`` Literal values).
        check_id: Per-cell warning identifier emitted by Sprint F's
            scientific validators. Unknown check_ids return ``[]``
            (absence of a caveat = no domain claim attached).

    Returns:
        List of ``CaveatCode`` Literal values. Order is stable across
        calls. Empty list when no caveat applies to this pair.

    The function is pure; the same arguments always produce the same
    output. Caveats:

    * ``(BSh, value_range_precip)`` — Sahel MCS decorrelation
      (Mathon 2002).
    * ``(BSh, value_range_wind)`` — same MCS regime applied to wind.
    * ``(Cwa, value_range_precip)`` — Highland orographic effects
      (Daly 2006). The affordance-routing rule at AC-E2-3 typically
      routes these cells to ``"skip"`` when elevation > 1500 m, but
      the caveat itself is independent of the routing decision —
      the code reflects the domain claim about the zone, not the
      operational decision about the cell.

    The (Sahel, wind) entry is ready for the day a cockpit caller
    surfaces ``value_range_wind`` cells; today the validators emit
    only ``value_range_precip`` for the Sahel-MCS class, but the
    rule is in place so a future expansion of the wind-validation
    surface lands the caveat without contract amendment.
    """
    if zone == "BSh":
        if check_id == _CHECK_ID_PRECIP:
            return ["sahel-precip-convective"]
        if check_id == _CHECK_ID_WIND:
            return ["sahel-wind-convective"]
    if zone == "Cwa":
        if check_id == _CHECK_ID_PRECIP:
            return ["highland-orographic-excluded"]
    return []


__all__ = [
    "caveats_for",
]
