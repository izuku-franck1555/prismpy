"""Texture-fraction renormalization at the harmonize stage.

The helper takes a single soil layer and returns a decision string
in ``{'renormalize', 'warn_retain', 'exclude'}`` based on the
absolute deviation between ``sand + silt + clay`` and 100. The
caller (the pipeline executor's HARMONIZE stage) routes the layer
according to the decision:

* ``renormalize`` — call :func:`renormalize_layer` to scale each
  fraction by ``100 / sum`` and write a provenance entry. The
  resulting layer always sums to exactly 100 within float
  precision.
* ``warn_retain`` — keep the layer untouched and let the existing
  validator emit its standard warning. The harmonize stage does
  not transform layers in this band.
* ``exclude`` — route the cell containing this layer to
  ``data_availability='unavailable'`` with axis ``soil`` and cause
  ``soil_texture_invalid``; the layer is unsalvageable.

Threshold rationale + alignment with the existing validator's
``[95, 105]`` accept band lives in
:mod:`prismpy.harmonize.constants`.
"""
from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel

from prismpy.harmonize.constants import (
    TEXTURE_RENORMALIZE_THRESHOLD_PCT,
    TEXTURE_WARN_THRESHOLD_PCT,
)
from prismpy.models.soil import SoilLayer


class TextureRenormalizationProvenance(BaseModel):
    """Per-layer provenance entry written each time a layer's
    texture fractions are renormalized to sum to 100. The fields
    capture both the original triple and the post-renormalize
    triple so a downstream auditor can replay the transformation
    without re-reading the source-side raw data.
    """

    category: str = "texture_renormalize"
    cell_id: int
    layer_idx: int
    original_sand: float
    original_silt: float
    original_clay: float
    original_sum: float
    delta_from_100: float
    renormalization_factor: float
    new_sand: float
    new_silt: float
    new_clay: float


def texture_action_for(layer: SoilLayer) -> str:
    """Decide what to do with a layer's texture triple.

    Args:
        layer: A :class:`prismpy.models.soil.SoilLayer` with
            ``sand``, ``silt``, ``clay`` populated. The dataclass's
            ``__post_init__`` already auto-derives ``silt`` from
            ``100 - sand - clay`` when ``silt is None``, so a layer
            constructed without an explicit silt always sums to 100
            by construction; the function is most informative for
            layers built from a source that supplies all three
            independently (HWSD's T_SAND / T_SILT / T_CLAY columns).

    Returns:
        ``'renormalize'`` when ``|sum - 100| <= 3.0%``,
        ``'warn_retain'`` when the deviation is in ``(3.0, 5.0]``,
        ``'exclude'`` otherwise. The boundary at 3.0 is inclusive
        on the renormalize side; 5.0 is inclusive on the warn side.
    """
    sand = layer.sand
    clay = layer.clay
    silt = layer.silt if layer.silt is not None else (100.0 - sand - clay)
    delta = abs(sand + silt + clay - 100.0)
    if delta <= TEXTURE_RENORMALIZE_THRESHOLD_PCT:
        return "renormalize"
    if delta <= TEXTURE_WARN_THRESHOLD_PCT:
        return "warn_retain"
    return "exclude"


def renormalize_layer(
    layer: SoilLayer, cell_id: int, layer_idx: int,
) -> Tuple[SoilLayer, TextureRenormalizationProvenance]:
    """Scale a layer's sand / silt / clay so they sum to exactly
    100, and return both the renormalized layer and a provenance
    entry capturing the transformation. The caller is responsible
    for replacing the original layer with the returned one and
    persisting the provenance entry.

    Pre-condition: :func:`texture_action_for` returned ``'renormalize'``
    for this layer.
    """
    sand = layer.sand
    clay = layer.clay
    silt = layer.silt if layer.silt is not None else (100.0 - sand - clay)
    original_sum = sand + silt + clay
    if original_sum <= 0.0:
        # Degenerate input — should never reach here because the
        # action discriminator routes a sum of 0 to ``exclude`` (it
        # falls outside the warn band). Guard explicitly so a
        # mis-sequenced caller raises rather than divides by zero.
        raise ValueError(
            f"renormalize_layer called with non-positive sum "
            f"({original_sum}); this layer should have been routed "
            f"to 'exclude' upstream."
        )
    factor = 100.0 / original_sum
    new_sand = sand * factor
    new_silt = silt * factor
    new_clay = clay * factor
    new_layer = SoilLayer(
        depth_top=layer.depth_top,
        depth_bottom=layer.depth_bottom,
        sand=new_sand,
        clay=new_clay,
        silt=new_silt,
        organic_carbon=layer.organic_carbon,
        bulk_density=layer.bulk_density,
        ph=layer.ph,
        field_capacity=layer.field_capacity,
        wilting_point=layer.wilting_point,
        saturated_wc=layer.saturated_wc,
    )
    provenance = TextureRenormalizationProvenance(
        cell_id=cell_id,
        layer_idx=layer_idx,
        original_sand=sand,
        original_silt=silt,
        original_clay=clay,
        original_sum=original_sum,
        delta_from_100=original_sum - 100.0,
        renormalization_factor=factor,
        new_sand=new_sand,
        new_silt=new_silt,
        new_clay=new_clay,
    )
    return new_layer, provenance
