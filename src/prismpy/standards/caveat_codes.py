"""Canonical caveat-code vocabulary for IDW interpolation.

Sprint E.2 §0.2 canonical-source #7 + AC-E2-6 + AC-E2-7. The
``CaveatCode`` Literal here is the single source of truth for the
domain-knowledge caveats the cockpit attaches to imputed cells.
Three downstream consumers reference it:

* ``InterpolatedCellRecord.caveat_codes: list[CaveatCode]`` (AC-E2-1)
  — schema-level validation; an unknown code raises at construction.
* ``caveats_for(zone, check_id) -> list[CaveatCode]`` at
  ``prismpy/standards/interpolation_caveats.py`` (AC-E2-6) — domain
  rule that determines which caveats apply for a zone × check_id
  pair.
* The methods-text generator at
  ``prismpy/packaging/methods_text.py`` (AC-E2-7) — emits the
  per-caveat phrase verbatim when an InterpolatedCellRecord carries
  the matching code.

Each ``CaveatCode`` member has exactly one entry in
``METHODS_TEXT_CAVEAT_PHRASES`` below; a structural pin at
``tests/structural/test_caveat_code_completeness.py`` asserts
``set(typing.get_args(CaveatCode)) == set(METHODS_TEXT_CAVEAT_PHRASES.keys())``
so a future code added to one without the other fails CI loud per
durable §24 canonical-source-or-pin.

The three caveats Sprint E.2 ships:

* ``sahel-precip-convective`` — Sahel-zone (Köppen BSh) precipitation
  has 5-15 km decorrelation lengths during the wet-season
  mesoscale-convective-system regime (Mathon, Laurent, Lebel 2002,
  *J. Appl. Meteor. Clim.* 41:1081). IDW(R=15 km) imputed precip in
  BSh may underestimate localised convective gradients; imputed
  cells should be cross-checked against TAMSAT gauge data when
  available.

* ``sahel-wind-convective`` — Same MCS decorrelation regime applies
  to wind during convective storm passage; same cross-check
  recommendation.

* ``highland-orographic-excluded`` — Highland zones (Köppen Cwa with
  elevation > 1500 m) have orographically-controlled precipitation
  that IDW cannot reliably interpolate (Daly 2006, *Int. J.
  Climatology* 26:707). The affordance-routing rule at AC-E2-3
  routes these cells to ``"skip"`` rather than ``"interpolate"`` —
  but the code remains in the Literal for honest-signal disclosure
  in methods text describing why the routing decision happened, and
  for future scope where lower-elevation Cwa cells might still be
  imputed with this caveat noted.

Future caveats (V2-19.5 Data Bootstrapper East African Highland
expansion, etc.) extend the Literal here + the phrase dict together,
gated by the completeness pin.
"""

from __future__ import annotations

from typing import Final, Literal


# ── Canonical Literal — three caveats Sprint E.2 ships ───────────────


CaveatCode = Literal[
    "sahel-precip-convective",
    "sahel-wind-convective",
    "highland-orographic-excluded",
]


# ── Methods-text phrases — one per CaveatCode, exact-string pinned ───


# Each phrase appears verbatim in ``manifest.methods_text`` when an
# imputed cell carries the matching code. The text is persona-tone-
# disciplined per VISION.md (concrete nouns + verbs; no marketing
# adjectives) and references the underlying peer-reviewed citation
# inline so a paper reviewer reading the manifest can trace the
# domain-evidence claim directly.
METHODS_TEXT_CAVEAT_PHRASES: Final[dict[CaveatCode, str]] = {
    "sahel-precip-convective": (
        "Sahel-zone precipitation interpolation may underestimate "
        "localised convective storm gradients during the wet-season "
        "mesoscale-convective-system regime (Mathon et al. 2002, "
        "Journal of Applied Meteorology and Climatology 41:1081); "
        "imputed Sahel cells should be cross-checked against TAMSAT "
        "gauge data when available."
    ),
    "sahel-wind-convective": (
        "Sahel-zone wind interpolation may underestimate localised "
        "convective storm gradients during the wet-season mesoscale-"
        "convective-system regime (Mathon et al. 2002, Journal of "
        "Applied Meteorology and Climatology 41:1081); imputed Sahel "
        "cells should be cross-checked against in-situ wind "
        "observations when available."
    ),
    "highland-orographic-excluded": (
        "Highland-zone precipitation cannot be reliably interpolated "
        "due to orographic effects (Daly 2006, International Journal "
        "of Climatology 26:707); cells in highland zones above "
        "1500 m elevation are routed to skip rather than imputation "
        "by affordance-routing policy."
    ),
}


__all__ = [
    "METHODS_TEXT_CAVEAT_PHRASES",
    "CaveatCode",
]
