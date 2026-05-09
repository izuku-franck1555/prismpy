"""Canonical ``apply_override`` helper — 3-source precedence chain.

Sprint E.3 AC-E3-8 + Stage 1 §4 + CMS CA-3 + WA CA-18 + builder
CA-4 absorbed. Every per-cell translator-write site (across
SARRA-Py / CRAFT / PYTHIA / ACEA per AC-E3-9) routes through this
helper before writing per-cell platform files.

**Three-source precedence chain**: when a cell has BOTH an
Override AND an Interpolation for the same
``(cell_id, variable_key)`` (realistic per Drill-E3-K multi-check
coexistence: Override on ``value_range_tmax`` from documented
field measurement + Interpolate on ``coverage_climate_cells``
IDW-imputing ALL missing climate variables INCLUDING tmax),
translator output uses Override value. The user's documented
decision wins over mechanical imputation; mechanical imputation
wins over raw (which triggered the warning).

The InterpolatedCellRecord schema persists method + neighbours
+ uncertainty CI but NOT the imputed value itself — by the time
the translator-write site invokes ``apply_override``, the
``raw_value`` parameter ALREADY carries the IDW-imputed value
(if interpolation fired upstream) or the original raw value
(if no interpolation). The ``interpolation_record`` parameter
is informational: it signals the cell IS an IDW-imputed cell,
which lets the methods-text generator at AC-E3-21 emit
appropriate per-cell copy without re-walking the decision log.

**Helper is PURE**: does NOT mutate ``unified_data``; does NOT
mutate ``sidecar``; does NOT mutate ``interpolation_record``.
Behavioral drill at ``tests/unit/test_apply_override.py``
(``test_helper_does_not_mutate_sidecar`` +
``test_helper_does_not_mutate_interpolation_record``) asserts the
purity invariant on the Pydantic frozen contract; the integration-
level dual-equivalent purity drill (``unified_data`` byte-
equivalent before/after AND ``cockpit_observed_values.json``
byte-equivalent before/after — evaluator CA-E3-6 absorbed) lands
at AC-E3-9 4-translator wiring as a per-translator post-write
diff against a raw-substrate baseline, closing the data-cooking
honest-signal floor per ``feedback_no_data_cooking.md``.

**Translator-wiring precedence pin** (AC-E3-12 #3 rewritten per
builder CA-4 + WA CA-18; structural pin
``test_override_interpolation_precedence.py``): when sidecar has
``(cell_id, variable_key) → override_value`` AND
``interpolation_record`` has same ``(cell_id, variable_key)``,
per-translator output reflects override_value (drill: 4 platforms
× 3 cells × 3 override flavors = 36 byte-equivalent
comparisons).
"""

from __future__ import annotations

from typing import Optional, TypeVar, Union

from prismpy.cockpit.cockpit_overrides_writer import CockpitOverrideSidecar
from prismpy.models.interpolated_cell import CellID, InterpolatedCellRecord


T = TypeVar("T", float, int, str)


def apply_override(
    cell_id: CellID,
    variable_key: str,
    raw_value: T,
    sidecar: Optional[CockpitOverrideSidecar],
    interpolation_record: Optional[InterpolatedCellRecord] = None,
) -> Union[T, float, int, str]:
    """Apply the override-precedence chain at a per-cell write site.

    Args:
        cell_id: Canonical cell-id reference.
        variable_key: Canonical variable_key (e.g.,
            ``"tmax_growing_season_mean"``) — matches the registry
            entry's ``variable_key`` field at
            :data:`prismpy.standards.override_value_shapes.OVERRIDE_VALUE_SHAPES`.
        raw_value: The translator's would-be write value at this
            site, BEFORE override consideration. May already be
            an IDW-imputed value if interpolation fired upstream
            (the imputed value is in the upstream substrate
            before reaching the translator).
        sidecar: The :class:`CockpitOverrideSidecar` payload
            loaded from ``cockpit_overrides.json``. ``None`` is
            valid for runs with no overrides recorded — the
            helper short-circuits to ``raw_value``.
        interpolation_record: Informational signal that the cell
            IS an IDW-imputed cell. The imputed value is already
            in ``raw_value``; this argument exists for downstream
            methods-text use (lets the generator emit per-cell
            "imputed via IDW" copy without re-walking the
            decision log). The helper itself does NOT use this
            for the precedence chain — Override always wins,
            regardless of whether the cell was imputed.

    Returns:
        - ``override_value`` from ``sidecar`` if a matching
          ``(cell_id, variable_key)`` entry exists.
        - ``raw_value`` otherwise (which may be IDW-imputed
          upstream or the original raw value).

    The helper is PURE: ``sidecar`` is read-only via the Pydantic
    frozen+extra=forbid contract; ``interpolation_record`` is
    read-only via the same contract; ``raw_value`` is returned
    unmodified when no override matches.
    """
    if sidecar is None:
        return raw_value

    target_cell_id = str(cell_id)
    target_variable_key = str(variable_key)

    for entry in sidecar.overrides:
        if entry.cell_id != target_cell_id:
            continue
        if entry.variable_key != target_variable_key:
            continue
        # Match found — Override wins per the precedence chain.
        return entry.value

    return raw_value


__all__ = [
    "apply_override",
]
