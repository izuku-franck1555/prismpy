"""Pipeline-stage application of the harmonize-stage transformations.

The pipeline executor invokes :func:`apply_harmonize_transformations`
between cascade resolution and the final ``UnifiedData`` assembly
in ``_execute_harmonize``. The helper iterates the post-cascade
soil profiles and climate time series, calls the per-event
helpers (:func:`texture_action_for`, :func:`rh_action_for`),
applies the resulting actions in place, and writes provenance
entries via the tracker's Sprint D.1 setter methods. Cells routed
to ``unavailable`` (texture-invalid layers, rh > 102, missing
HWSD coverage) are recorded for the executor to surface in the
final cell-summary block.

The helper is deliberately a thin orchestrator over the per-event
helpers in :mod:`prismpy.harmonize.texture_renormalize` and
:mod:`prismpy.harmonize.rh_clip`; the per-helper unit tests cover
the per-event behavior, this helper's tests cover the wiring +
the cell-unavailable propagation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prismpy.harmonize.rh_clip import rh_action_for, RHClipProvenance
from prismpy.warnings import WarningCategory
from prismpy.harmonize.texture_renormalize import (
    renormalize_layer,
    texture_action_for,
)


@dataclass
class HarmonizeStats:
    """Summary of every action the apply helper took during a
    harmonize-stage run. Returned to the executor so it can
    populate the cell-summary block + final provenance summary.
    """

    n_texture_renormalized: int = 0
    n_texture_warned: int = 0
    n_rh_clipped: int = 0
    cells_unavailable: List[Dict[str, Any]] = field(default_factory=list)


def apply_harmonize_transformations(
    soil_data: Optional[Dict[int, Any]],
    climate_data: Optional[Dict[int, Any]],
    provenance_tracker: Optional[Any] = None,
    hwsd_unavailable_cells: Optional[List[Dict[str, Any]]] = None,
) -> HarmonizeStats:
    """Apply texture renormalization, rh clipping, and missing-soil
    routing to a unified-data payload. Mutates the input
    structures in place; the return value carries summary stats
    + the list of cells routed to unavailable.

    Args:
        soil_data: Mapping of cell_id -> SoilProfile from the
            cascade. May be None if no soil source ran.
        climate_data: Mapping of cell_id -> ClimateTimeSeries.
            May be None.
        provenance_tracker: Optional ProvenanceTracker. When
            provided, every action writes a provenance entry via
            the tracker's setter methods (provenance is silently
            skipped when None).
        hwsd_unavailable_cells: Optional list of {cell_id, cause}
            entries the HWSD source surfaced in its retrieval
            metadata. Each entry is converted to a
            cell-unavailable record on the returned stats.

    Returns:
        :class:`HarmonizeStats` aggregating every action taken.
    """
    stats = HarmonizeStats()

    # -------------------------------------------------------------
    # AC-4: HWSD coverage misses surface as cell-unavailable
    # -------------------------------------------------------------
    if hwsd_unavailable_cells:
        for entry in hwsd_unavailable_cells:
            cell_id = entry.get("cell_id")
            cause = (
                entry.get("cause")
                or WarningCategory.SOIL_NO_HWSD_COVERAGE.value
            )
            stats.cells_unavailable.append(
                {
                    "cell_id": cell_id,
                    "unavailable_reason": "soil",
                    "unavailable_cause": cause,
                }
            )
            if provenance_tracker is not None:
                provenance_tracker.record_cell_unavailable(
                    cell_id=cell_id,
                    unavailable_reason="soil",
                    unavailable_cause=cause,
                )

    # -------------------------------------------------------------
    # AC-1: texture-fraction renormalization per soil layer
    # -------------------------------------------------------------
    if soil_data:
        excluded_for_texture: List[int] = []
        for cell_id, profile in soil_data.items():
            if profile is None or not getattr(profile, "layers", None):
                continue
            new_layers = []
            for layer_idx, layer in enumerate(profile.layers):
                # Skip layers without sand/clay populated — they
                # were not produced by a source the audit covers.
                if layer.sand is None or layer.clay is None:
                    new_layers.append(layer)
                    continue
                action = texture_action_for(layer)
                if action == "renormalize":
                    new_layer, prov = renormalize_layer(
                        layer, cell_id=int(cell_id), layer_idx=layer_idx,
                    )
                    new_layers.append(new_layer)
                    stats.n_texture_renormalized += 1
                    if provenance_tracker is not None:
                        provenance_tracker.record_texture_renormalization(prov)
                elif action == "warn_retain":
                    new_layers.append(layer)
                    stats.n_texture_warned += 1
                else:  # exclude
                    new_layers.append(layer)
                    if cell_id not in excluded_for_texture:
                        excluded_for_texture.append(cell_id)
            profile.layers = new_layers

        for cell_id in excluded_for_texture:
            stats.cells_unavailable.append(
                {
                    "cell_id": cell_id,
                    "unavailable_reason": "soil",
                    "unavailable_cause": WarningCategory.SOIL_TEXTURE_INVALID.value,
                }
            )
            if provenance_tracker is not None:
                provenance_tracker.record_cell_unavailable(
                    cell_id=cell_id,
                    unavailable_reason="soil",
                    unavailable_cause=WarningCategory.SOIL_TEXTURE_INVALID.value,
                )

    # -------------------------------------------------------------
    # AC-3: rh clip-to-100 per daily climate record
    # -------------------------------------------------------------
    if climate_data:
        excluded_for_rh: List[int] = []
        for cell_id, ts in climate_data.items():
            if ts is None or not getattr(ts, "records", None):
                continue
            for record in ts.records:
                rh = getattr(record, "rh", None)
                if rh is None:
                    continue
                action = rh_action_for(rh)
                if action.action == "clip":
                    record.rh = action.clipped_rh
                    stats.n_rh_clipped += 1
                    if provenance_tracker is not None:
                        provenance_tracker.record_rh_clip(
                            RHClipProvenance(
                                cell_id=int(cell_id),
                                date=str(getattr(record, "date", "")),
                                original_rh=rh,
                            )
                        )
                elif action.action == "exclude":
                    if cell_id not in excluded_for_rh:
                        excluded_for_rh.append(cell_id)

        for cell_id in excluded_for_rh:
            stats.cells_unavailable.append(
                {
                    "cell_id": cell_id,
                    "unavailable_reason": "climate",
                    "unavailable_cause": WarningCategory.CLIMATE_RH_INVALID.value,
                }
            )
            if provenance_tracker is not None:
                provenance_tracker.record_cell_unavailable(
                    cell_id=cell_id,
                    unavailable_reason="climate",
                    unavailable_cause=WarningCategory.CLIMATE_RH_INVALID.value,
                )

    return stats
