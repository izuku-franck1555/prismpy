"""Cockpit per-cell diagnostic-variant Literal canonical source.

Sprint E.2 AC-E2-25 + WA CA-8 + Codex Gate A MEDIUM A4/B1.

Per-cell click on the cockpit map dispatches the right-side
panel into one of six State-C variants — the variant string
the JS template's ``cVariant`` getter reads off. The variant
is server-determined (per Codex Gate A HIGH 3 +
``stage0-5-ext-design.md`` §6.1): ``build_cell_detail`` (the
prismweb consumer) emits one of these literal values on
``selectedCell.diagnostic_variant``; the JS getter dispatches
to the matching template block.

This module is the cross-language CANONICAL source of the
variant vocabulary. Per durable §24 canonical-source-or-pin:

* The Python ``DiagnosticVariant`` :class:`typing.Literal`
  alias is what the producer's type checker enforces — a
  typo'd literal at any future ``build_cell_detail`` callsite
  fails at static analysis time, BEFORE shipping to the
  consumer.
* The :data:`DIAGNOSTIC_VARIANT_VALUES` :class:`frozenset` is
  the runtime-importable vocabulary set future consumers (in
  prismpy + cross-repo in prismweb) read instead of
  re-enumerating the literal arguments.

Phase 1.5 ships the canonical producer vocabulary only. The
consumer-side wiring + parity enforcement lands at Phase 2
prismweb UI work:

* The JS ``cVariant`` getter at
  ``prismweb/static/js/cockpit-redesign/cockpit-state.js``
  reads ``selectedCell.diagnostic_variant`` first + dispatches
  to the matching State C″ template branch (Variant A
  cell-level-scalar / Variant B climate-dual-scale / Variant
  C soil-layered) + the State C passing-cell + State C′
  highland-excluded + State C‴ documented-override branches.
* The structural pin
  ``prismweb/core/tests/structural/test_diagnostic_variant_vocab_parity.py``
  (WA CA-8) lands at Phase 2; it imports
  :data:`DIAGNOSTIC_VARIANT_VALUES` from this module + reads
  the JS getter's handled branches via regex parse over
  ``cockpit-state.js`` source + asserts ``set(producer Literal
  members) == set(consumer-handled branches)``. Until Phase 2
  ships, the JS getter only handles the legacy four variants
  (``documented-override`` / ``highland-excluded`` /
  ``bucket-3-exclude`` / ``interpolatable``); cells whose
  ``diagnostic_variant`` is one of the new three values
  (``climate-dual-scale`` / ``soil-layered`` / ``passing``)
  fall through the getter's ``else`` to the default
  ``interpolatable`` path. The Phase 2 pin closes that
  fallthrough.

Pair pattern with Sprint G ISIMIP→SARRA + Sprint E.0
``WarningCategory`` two-vocabulary precedents — both wire
producer + consumer + parity pin in lockstep at the consumer-
sprint close.
"""
from __future__ import annotations

from typing import Literal


# Canonical variant Literal — the producer (``build_cell_detail``
# in prismweb's cockpit_decisions service) annotates its return
# field with this Literal so a 7th value can't be shipped without
# ALSO landing in the consumer's JS branch + the structural pin
# refusing the build.
#
# Vocabulary anchors (per ``stage0-5-ext-design.md`` §5 + §6):
#
# * ``cell-level-scalar`` — non-interpolable cell-level
#   constraint (e.g., soil profile depth below DSSAT 20cm
#   minimum). Renders State C″ Mockup A.
# * ``climate-dual-scale`` — daily-failure on climate value-
#   range with seasonal-aggregate context (e.g., 3 of 122
#   days had tmax > 60°C; growing-season-mean 32°C is
#   regional-typical). Renders State C″ Mockup B.
# * ``soil-layered`` — per-layer soil violation with rootzone-
#   aggregate context (e.g., bulk_density 2.18 in layer 3 only;
#   rootzone-weighted-mean 1.62 is regional-typical). Renders
#   State C″ Mockup C.
# * ``highland-excluded`` — Decision 2 caveat 2 orographic
#   exclusion (zone signal). Renders State C′.
# * ``documented-override`` — cell carries a Bucket-5 researcher
#   override. Renders State C‴.
# * ``passing`` — cell has no flag (user clicked anyway).
#   Renders State C passing variant per Stage-0.5 EXT Fix 4.
DiagnosticVariant = Literal[
    "cell-level-scalar",
    "climate-dual-scale",
    "soil-layered",
    "highland-excluded",
    "documented-override",
    "passing",
]


# Frozenset of the literal values for runtime use. The structural
# pin imports this directly so the vocabulary parity test reads
# the same canonical source as the producer's type annotation
# (per durable §24: every consumer routes through ONE source).
#
# Mirror-pinned to the :class:`DiagnosticVariant` literal at
# ``tests/structural/test_diagnostic_variant_literal_completeness.py``
# — the pin asserts the frozenset equals the set of literal
# arguments parsed out of the Literal's annotation source.
DIAGNOSTIC_VARIANT_VALUES: frozenset[str] = frozenset({
    "cell-level-scalar",
    "climate-dual-scale",
    "soil-layered",
    "highland-excluded",
    "documented-override",
    "passing",
})


__all__ = [
    "DiagnosticVariant",
    "DIAGNOSTIC_VARIANT_VALUES",
]
