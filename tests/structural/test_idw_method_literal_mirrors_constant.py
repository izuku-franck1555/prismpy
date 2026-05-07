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
    """The Literal annotation on the schema field carries exactly one
    arg — the canonical method identifier. Use ``model_fields`` +
    ``typing.get_args`` to introspect at runtime."""
    field_info = InterpolatedCellRecord.model_fields["interpolation_method"]
    annotation = field_info.annotation
    literal_args = typing.get_args(annotation)
    assert len(literal_args) == 1, (
        f"interpolation_method Literal MUST have exactly one arg "
        f"(MVP-fixed canonical method per AC-E2-19); got "
        f"{len(literal_args)}: {literal_args}"
    )
    assert literal_args[0] == IDW_DEFAULT_METHOD_LITERAL, (
        f"InterpolatedCellRecord.interpolation_method Literal "
        f"({literal_args[0]!r}) drifted from canonical constant "
        f"IDW_DEFAULT_METHOD_LITERAL ({IDW_DEFAULT_METHOD_LITERAL!r}) "
        f"at prismpy/standards/idw_methods.py. Per durable §24 "
        f"canonical-source-or-pin: the schema mirrors the canonical "
        f"constant; update both atomically when extending the method "
        f"vocabulary."
    )
