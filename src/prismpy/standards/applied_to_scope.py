"""Canonical applied-to-scope vocabulary for cockpit Override decisions.

Sprint E.3 AC-E3-2 + builder grounding-pass CA-7. The ``AppliedToScope``
Literal here is the single source of truth for the three scope
discriminators a cockpit Override decision can carry:

* ``single_cell`` — the override applies only to the cell the persona
  was looking at when the decision was recorded.
* ``zone`` — the override applies to every cell sharing a Köppen-zone
  membership at decision time. The actual zone identifier (e.g.,
  ``Cwa``, ``BWh``) lives on the companion ``OverrideRecord
  .applied_to_zone_id`` field — Sprint E.3 AC-E3-2 sub-criterion 2 +
  AC-E3-4 cross-validator make the pairing explicit so a typo on the
  scope discriminator can't smuggle a free-form zone code into a
  ``single_cell`` row.
* ``enumerated_cells`` — the override applies to a hand-picked list of
  cells the persona named explicitly at decision time. The cell list
  is captured in the snapshot field on the enclosing ``OverrideRecord``
  (``applied_to_snapshot: Tuple[CellID, ...]``) per WA CA-19 immutable
  Tuple precedent at ``prismpy/cockpit/manifest.py:153``.

Per durable lesson §24 canonical-source-or-pin: a constant living in
two places without enforcement is a silent-drift class. Two structural
pins close the class:

1. ``tests/structural/test_applied_to_scope_vocab_parity.py`` AST-walks
   the prismpy + prismweb sources and rejects any parallel
   ``Literal["single_cell", "zone", "enumerated_cells"]`` definition.
   Every consumer (``OverrideRecord.applied_to_scope``, snapshot
   writer, new-cells-in-zone detector, cross-run comparator) MUST
   import this Literal directly per durable §27 two-vocabulary
   substrate-drift discipline.

2. ``tests/structural/test_applied_to_zone_id_conditional_required.py``
   (companion validator pin per AC-E3-2 sub-5) asserts the
   ``OverrideRecord`` cross-validator fires for the two error
   shapes — ``scope=zone`` with ``zone_id=None`` (must reject) and
   ``scope=single_cell`` with ``zone_id="Cwa"`` (must reject).

The enumeration is intentionally small (3 values). A future fourth
discriminator (``cross_zone_corridor``, ``elevation_band``, etc.)
extends the Literal here once the sprint that needs it specifies the
companion-field invariant.
"""

from __future__ import annotations

from typing import Final, Literal


# ── Canonical Literal — three scope discriminators Sprint E.3 ships ──


# Order matches the persona's mental model from least to most cells:
# ``single_cell`` (1 cell) → ``zone`` (every cell in a Köppen zone) →
# ``enumerated_cells`` (an explicit hand-picked list). Ordering doesn't
# affect Pydantic validation but reads cleanly in the type hint.
AppliedToScope = Literal[
    "single_cell",
    "zone",
    "enumerated_cells",
]


# Public tuple of the three values for runtime iteration. Kept as a
# ``Final[tuple[str, ...]]`` so ``set(typing.get_args(AppliedToScope))``
# is the type-level vocabulary and ``APPLIED_TO_SCOPE_VALUES`` is the
# runtime-iterable companion. The structural pin at
# ``test_applied_to_scope_vocab_parity.py`` asserts the two are
# byte-equivalent so a future Literal addition cannot slip past
# runtime consumers that iterate the tuple.
APPLIED_TO_SCOPE_VALUES: Final[tuple[str, ...]] = (
    "single_cell",
    "zone",
    "enumerated_cells",
)


__all__ = [
    "APPLIED_TO_SCOPE_VALUES",
    "AppliedToScope",
]
