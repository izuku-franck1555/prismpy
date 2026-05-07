"""Canonical IDW interpolation constants — single source of truth.

Sprint E.2 AC-E2-19. The (k, R, weight power) tuple that defines the
prismweb cockpit's INTERPOLATABLE-bucket interpolation method lives
here once. Every consumer — the ``InterpolatedCellRecord`` schema's
``interpolation_method`` Literal (AC-E2-1), the ``interpolate_idw``
engine signature defaults (AC-E2-2), the methods-text generator's
per-cell paragraph (AC-E2-7), the persona-readable methods preview in
the cockpit's State A summary — references the constants below
rather than restating them.

Per durable lesson §24 canonical-source-or-pin: a constant living in
two places without enforcement is a silent-drift class. Three
structural pins close the class:

1. ``tests/structural/test_idw_method_literal_mirrors_constant.py``
   asserts ``InterpolatedCellRecord.model_fields['interpolation_method']``
   Literal arg equals ``IDW_DEFAULT_METHOD_LITERAL``. Drift between
   the schema-side spelling and the canonical constant fires loud at
   CI time.

2. ``tests/structural/test_idw_constants_no_external_hardcode.py``
   AST-walks the prismpy + prismweb sources, asserting no module
   outside ``standards/idw_methods.py`` hardcodes ``4`` for the
   neighbour count, ``15.0`` for the radius, or ``2.0`` for the
   weight exponent IN AN IDW CONTEXT (per Builder Sub-CA-A
   module-allow-list pattern: only this module may carry the
   literals). The walker uses module-name allow-listing — far simpler
   + far less false-positive-prone than surrounding-context regex on
   arbitrary literal-4 / literal-15 / literal-2 occurrences across
   the codebase.

3. ``tests/structural/test_decision_workflow_canonical_layer.py``
   (per §0.2 #1-#7 canonical-layer pin) asserts every IDW-method
   consumer imports from this module rather than re-defining.

The current MVP is one method ``"idw_k4_r15km_w_inverse_dist_sq"``
parameterised by:

* ``k = 4`` — number of nearest neighbours within the radius.
* ``R = 15 km`` — search radius for neighbour enumeration.
* ``w = 2`` — inverse-distance weighting exponent (``1 / d^w``).

A future method (e.g., source-bilinear from upstream gridded data, or
a per-zone tuned ``k``/``R``) extends ``IDW_DEFAULT_METHOD_LITERAL``
to a ``Literal[<old>, <new>]`` union; consumers that switch the
default migrate by changing one import.

The "Shepard 1968" foundational reference for IDW lives in the schema
docstring (AC-E2-1 ``method_doi`` field), not here — this module
holds machine-readable constants only.
"""

from __future__ import annotations

from typing import Final


# ── Canonical IDW method identifier ──────────────────────────────────


# String identifier persisted in ``InterpolatedCellRecord.interpolation_method``
# (AC-E2-1) and quoted by the methods-text generator (AC-E2-7) as the
# audit-grade "what method was used" handle. Format: ``idw_k{K}_r{R}km_w_inverse_dist_sq``
# — the ``inverse_dist_sq`` suffix names the weighting kernel explicitly so
# a future ``inverse_dist_lin`` (w=1) variant cannot collide on the
# (k, R) pair alone.
IDW_DEFAULT_METHOD_LITERAL: Final[str] = "idw_k4_r15km_w_inverse_dist_sq"


# ── Canonical numerical parameters ───────────────────────────────────


# Number of nearest neighbours to combine for the imputed value.
# At k=4 the cockpit reads as "imputed from up to 4 nearest cells
# within R" in the persona-readable methods text; below k the engine
# flags ``degraded_due_to_insufficient_neighbours=True``.
IDW_DEFAULT_K: Final[int] = 4

# Search radius (km) for neighbour enumeration. 15 km matches the 9-km
# grid spacing of the climate substrate with adequate decorrelation
# headroom for tmax/tmin/srad/rh; precip in highland zones (Köppen
# Cwb/Cwc, elevation > 1500 m) routes to ``"skip"`` per AC-E2-3 rather
# than relying on this radius (Daly 2006 orographic exclusion).
IDW_DEFAULT_R: Final[float] = 15.0

# Inverse-distance weighting exponent. ``2.0`` is Shepard's original
# formulation; the squared-distance kernel decays sharply enough that
# the four nearest neighbours dominate even when the search radius
# admits a fifth marginal cell. Lower exponents (1.0, 0.5) flatten
# the weighting toward an unweighted mean and lose the locality
# signal IDW exists to preserve.
IDW_DEFAULT_W: Final[float] = 2.0


__all__ = [
    "IDW_DEFAULT_K",
    "IDW_DEFAULT_METHOD_LITERAL",
    "IDW_DEFAULT_R",
    "IDW_DEFAULT_W",
]
