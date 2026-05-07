"""Closed enum for climate-data provenance kind (Sprint G AC-G-7a).

Per durable §24 canonical-source-or-pin: the discriminator that selects
between observed-climate (TAMSAT/AgERA5 source) and projection-climate
(ISIMIP3b GCM bias-adjusted source) writer paths is a closed enum,
NOT a ``bool is_projection``. A bool admits only two states; if a
future climate-kind needs different writer behavior (reanalysis-blend,
ensemble-mean-with-uncertainty, observed-augmented-with-reanalysis,
etc.), the bool is a refactor magnet across every translator.

The closed enum lives at the ``harmonize`` layer because:
* Multiple translators (CRAFT/PYTHIA/ACEA/SARRA-Py) consume the same
  discriminator vocabulary.
* No platform-specific behavior; it's a pure semantic label.
* Pairs with the existing ``harmonize.calendar_conversion`` /
  ``harmonize.tetens`` modules for projection-path support.

Per protocol §1 builder-counter-contract: contract Draft 5 didn't
specify the discriminator vocabulary; team-lead's standing-pin #1 at
AC-G-7a green-light specified the closed enum form. This module is
the canonical source.

The MISDAT semantics distinction documented in
``prismpy/.local/SPRINT-G-IMPLEMENTATION-DISCIPLINE-NOTES.md``:

* OBSERVED: when the source carries None for tdew/hurs/wind (TAMSAT
  doesn't supply RH; some AgERA5 paths don't supply wind), the WTH
  writer emits the DSSAT MISDAT sentinel (``-99.0``) per the Jones
  2003 v4.7 spec. Honest-signal "data not available" semantic.

* PROJECTION: when the source carries None for tdew but DOES carry
  hurs + tmean (ISIMIP3b standard), the WTH writer derives tdew via
  Tetens (AC-G-8 ``derive_tdew``). Honest-signal "data computable
  from what we have" semantic.

The enum IS the semantic boundary: future audit asking "why doesn't
observed-climate also derive tdew?" reads the docstring + this module
+ the writer's branching to find the answer. Each kind has its own
documented data-availability contract.
"""

from __future__ import annotations

from enum import Enum


class ClimateKind(str, Enum):
    """Closed enum for climate-data provenance.

    ``OBSERVED`` is the default; passes through legacy WTH-writer
    behavior (CRAFT 5-col, PYTHIA 8-col with MISDAT fallback for
    fields the source genuinely lacks).

    ``PROJECTION`` activates the projection-shape WTH-writer path:
    8-column output for both DSSAT-format translators (CRAFT + PYTHIA)
    with TDEW derived via Tetens when source tdew is None but hurs +
    tmean are present. Per AC-G-7a + AC-G-8 contract.
    """

    OBSERVED = "observed"
    PROJECTION = "projection"


__all__ = ["ClimateKind"]
