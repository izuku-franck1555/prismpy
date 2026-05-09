"""Structural pin: IDW method Literal mirrors the canonical constant.

Sprint E.2 AC-E2-1 + AC-E2-19 + WA Draft 2 CA-15 absorption (Sub-CA-A
parallel pattern). The ``InterpolatedCellRecord.interpolation_method``
field's Literal arg MUST equal ``IDW_DEFAULT_METHOD_LITERAL`` from the
canonical constants module. The Literal is hardcoded VERBATIM in the
schema (Python ``typing.Literal`` requires compile-time string
constants); this pin enforces the manual mirror via runtime
introspection.

Drift between the schema literal and the canonical constant fires
loud at CI time per durable §24 canonical-source-or-pin discipline:
a refactor that bumps the constant without updating the schema
silently introduces a never-validated method identifier.

Same discipline as ``test_koppen_zone_literal_mirrors_registry.py``
(KoppenZone Literal vs JSON registry).
"""

from __future__ import annotations

import typing

from prismpy.models.interpolated_cell import InterpolatedCellRecord
from prismpy.standards.idw_methods import IDW_DEFAULT_METHOD_LITERAL


def test_interpolation_method_literal_arg_equals_canonical_constant() -> None:
    """The Literal annotation on the schema field MUST carry the
    Sprint E.3 migration-window union per AC-E3-11 sub-2 absorbed —
    exactly two args: the post-E.3 canonical
    :data:`IDW_CANONICAL_METHOD_LITERAL` (``"idw"``) AND the legacy
    pre-E.3 :data:`IDW_LEGACY_METHOD_LITERAL`
    (``"idw_k4_r15km_w_inverse_dist_sq"``).

    Post-migration tightening (V3+ task) drops the legacy literal
    from the union; this pin asserts the migration-window
    coexistence so a refactor that drops the canonical literal
    early fires loud + a refactor that drops the legacy literal
    early fires loud (legacy rows still need the literal until
    the prismweb-side migration ``0024`` ships)."""
    from prismpy.standards.idw_methods import (
        IDW_CANONICAL_METHOD_LITERAL,
        IDW_LEGACY_METHOD_LITERAL,
    )

    field_info = InterpolatedCellRecord.model_fields["interpolation_method"]
    annotation = field_info.annotation
    literal_args = set(typing.get_args(annotation))
    expected = {
        IDW_CANONICAL_METHOD_LITERAL,
        IDW_LEGACY_METHOD_LITERAL,
    }
    assert literal_args == expected, (
        f"InterpolatedCellRecord.interpolation_method Literal "
        f"args {sorted(literal_args)} drifted from canonical "
        f"migration-window union {sorted(expected)} at "
        f"prismpy/standards/idw_methods.py. Per durable §24 + "
        f"AC-E3-11 sub-2: the schema mirrors the canonical "
        f"constants; update both atomically when extending the "
        f"method vocabulary."
    )
