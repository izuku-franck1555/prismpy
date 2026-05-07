"""Cross-document validator: manifest flags ↔ cell_summary entries.

Sprint E.2 AC-E2-8 + AC-E2-22 + §0.2 (validator runs at
``validate_scenario_set`` level since the Pydantic ``Manifest``
model alone can't read ``cell_summary.json`` — separate file).

Asserts the cross-document invariant:

* If ANY cell in ``cell_summary.json`` carries an
  ``interpolation_decision_id`` (i.e., was imputed via IDW),
  ``manifest.flags.interpolation_present`` MUST be ``True``.

* Symmetrically, if ``manifest.flags.interpolation_present`` is
  ``True``, at least one cell in ``cell_summary.json`` MUST carry
  an ``interpolation_decision_id``.

Drift in either direction surfaces as ``ManifestConsistencyError``
— the typed exception hierarchy that the
``validate_scenario_set`` integration (AC-E2-22) wraps when
running at scenario-set validation time.
"""

from __future__ import annotations

from typing import Any


class ManifestConsistencyError(ValueError):
    """Raised when manifest.flags.interpolation_present drifts from
    the per-cell interpolation_decision_id substrate. Sprint E.2
    AC-E2-8 + Drill H invariant."""


def validate_manifest_cell_summary_consistency(
    manifest: dict[str, Any],
    cell_summary: dict[str, Any],
) -> None:
    """Cross-document validator. Raises
    ``ManifestConsistencyError`` on drift between the manifest's
    interpolation flag and the cell_summary's per-cell
    interpolation-decision IDs.

    Args:
        manifest: Manifest dict. Expected to carry a ``flags``
            sub-dict with optional ``interpolation_present`` boolean.
        cell_summary: Cell-summary dict. Expected to carry a ``cells``
            list whose entries are dicts with optional
            ``interpolation_decision_id``.

    Raises:
        ManifestConsistencyError: When the manifest flag and the
            per-cell substrate disagree.
    """
    flags = manifest.get("flags") or {}
    flag_present = bool(flags.get("interpolation_present", False))

    cells = cell_summary.get("cells") or []
    cells_with_interpolation = [
        cell
        for cell in cells
        if isinstance(cell, dict) and cell.get("interpolation_decision_id")
    ]
    has_interpolated_cells = len(cells_with_interpolation) > 0

    if has_interpolated_cells and not flag_present:
        raise ManifestConsistencyError(
            f"manifest.flags.interpolation_present is False but "
            f"{len(cells_with_interpolation)} cells in cell_summary "
            f"carry an interpolation_decision_id. The manifest flag "
            f"MUST be True when ANY cell was imputed via IDW (per "
            f"AC-E2-8 invariant + feedback_no_data_cooking.md "
            f"flag-propagation contract)."
        )
    if flag_present and not has_interpolated_cells:
        raise ManifestConsistencyError(
            "manifest.flags.interpolation_present is True but no cell "
            "in cell_summary carries an interpolation_decision_id. "
            "The manifest flag is over-claimed; either set the flag "
            "to False or add the corresponding per-cell substrate "
            "(per current_decisions semantics — a fully-reverted "
            "package should flip the flag to False)."
        )


__all__ = [
    "ManifestConsistencyError",
    "validate_manifest_cell_summary_consistency",
]
