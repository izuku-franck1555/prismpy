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

from typing import Dict, Final


# ── Canonical IDW method identifier ──────────────────────────────────


# Sprint E.3 AC-E3-11 + post-Draft 4 codex HIGH-2 + MEDIUM-3 absorbed —
# the canonical post-E.3 method handle is plain ``"idw"`` (kernel-family
# identifier); the parameter-encoded form ``"idw_k4_r15km_w_inverse_dist_sq"``
# becomes deprecated, because per-record parameters now live in
# dedicated ``radius_km`` / ``k`` / ``weight_power`` fields on
# :class:`InterpolatedCellRecord`. During the migration window
# (this Sprint E.3 phase) the schema accepts BOTH literals via
# ``Literal["idw", "idw_k4_r15km_w_inverse_dist_sq"]``; the
# Django migration ``0024_interpolated_cell_record_schema_extension.py``
# (ships at AC-E3-16 prismweb-side per builder DELTA-CA-2) rewrites
# legacy rows to the canonical ``"idw"`` literal post-deployment.
IDW_CANONICAL_METHOD_LITERAL: Final[str] = "idw"
"""Post-E.3 canonical kernel-family identifier. The methods-text
generator + per-record provenance pin both read this constant."""

IDW_LEGACY_METHOD_LITERAL: Final[str] = "idw_k4_r15km_w_inverse_dist_sq"
"""Pre-E.3 parameter-encoded literal. Accepted during the migration
window; rewritten to :data:`IDW_CANONICAL_METHOD_LITERAL` by the
prismweb-side Django migration ``0024`` at AC-E3-16."""

# Backward-compat alias — Sprint E.2 consumers reference
# ``IDW_DEFAULT_METHOD_LITERAL`` directly. Bound to the legacy
# literal for the migration window so existing imports continue to
# resolve to a string the schema accepts.
IDW_DEFAULT_METHOD_LITERAL: Final[str] = IDW_LEGACY_METHOD_LITERAL


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


# ── Per-platform IDW radius registry (CMS CA-1 BLOCKING) ────────────


# Sprint E.3 AC-E3-11 + Draft 2 CMS CA-1 BLOCKING absorbed — the IDW
# orchestrator dispatches a platform-specific radius rather than the
# universal ``IDW_DEFAULT_R = 15.0`` because the four target
# platforms have radically different grid spacings:
#
# * SARRA-Py ≈ 4 km grid → 15 km radius (≈4× cell size; tight enough
#   to keep IDW within decorrelation envelope on Sahel precip).
# * CRAFT ≈ 1 km grid (30-arcsecond) → 15 km radius (same; the
#   1-km grid yields ample candidates within 15 km).
# * PYTHIA ≈ 9 km grid (5-arcmin) → 25 km radius (≈3× cell size;
#   modest expansion preserves locality without over-smoothing).
# * ACEA ≈ 50 km grid (30-arcmin) → 100 km radius (≈2× cell size;
#   tight by ratio, but absolute distance crosses Köppen-zone
#   boundaries so the orchestrator emits a cross-zone-warn flag
#   per AC-E3-11 sub-9 absorbed).
#
# Domain rationale (CMS Draft 2): Sahel decorrelation lengths
# (Boulanger et al. 2018 + AGRHYMET 30-year climatology + Lebel
# & Ali 2009 monsoon precipitation). ACEA at 50 km grid needs
# ~75-150 km IDW radius (1.5-3× cell size) to capture decorrelation
# envelope without crossing too many Köppen-zone boundaries. With
# the prior 15 km radius universal default, ACEA cells silently
# fail IDW (zero candidate neighbours at 50 km grid spacing) — a
# production-blocking class that the CMS BLOCKING absorption
# closed.
#
# The registry's keys are the canonical :class:`Platform` enum
# members at ``prismpy/config/schema.py:242``. The structural pin
# at ``tests/structural/test_idw_radius_by_platform.py`` AST-walks
# the registry against the enum to assert coverage of all
# ``Platform.*`` members; an enum addition without a registry
# update fires the pin loud per durable §24 canonical-source-or-pin.
def _build_idw_radius_by_platform() -> Dict[str, float]:
    """Lazy build to avoid circular import at module-load time.

    The :class:`Platform` enum lives at
    ``prismpy/config/schema.py``; importing it at module top
    triggers a Pydantic schema parse which transitively imports
    the validators → cockpit subpackage → this module. Lazy build
    breaks the cycle while keeping the registry keys typed via
    the enum values."""
    from prismpy.config.schema import Platform

    return {
        Platform.SARRA_PY.value: 15.0,
        Platform.CRAFT.value:    15.0,
        Platform.PYTHIA.value:   25.0,
        Platform.ACEA.value:     100.0,
    }


def get_idw_radius_for_platform(platform_value: str) -> float:
    """Return the canonical IDW radius (km) for ``platform_value``.

    Per CMS CA-1 BLOCKING: every IDW dispatch site routes through
    this helper rather than reading :data:`IDW_DEFAULT_R` directly.
    A platform value not in the registry raises ``KeyError`` —
    the orchestrator should never reach this case in production
    (the structural pin asserts coverage of all
    ``Platform.*`` members), but failing loud is the right
    discipline per ``feedback_no_data_cooking.md``.

    Args:
        platform_value: The canonical ``Platform.*.value`` string
            (e.g., ``"sarra_py"``, ``"acea"``).

    Returns:
        Search-radius in kilometres for the platform's grid.

    Raises:
        KeyError: when ``platform_value`` is not a known
            ``Platform.*.value``.
    """
    registry = _build_idw_radius_by_platform()
    if platform_value not in registry:
        raise KeyError(
            f"IDW radius dispatch unknown platform: {platform_value!r}. "
            f"Known: {sorted(registry.keys())}. Add a registry entry "
            f"in ``prismpy/standards/idw_methods.py`` if you've "
            f"introduced a new Platform.* member."
        )
    return registry[platform_value]


__all__ = [
    "IDW_CANONICAL_METHOD_LITERAL",
    "IDW_DEFAULT_K",
    "IDW_DEFAULT_METHOD_LITERAL",
    "IDW_DEFAULT_R",
    "IDW_DEFAULT_W",
    "IDW_LEGACY_METHOD_LITERAL",
    "get_idw_radius_for_platform",
]
