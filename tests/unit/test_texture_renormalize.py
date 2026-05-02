"""Sprint D.1 AC-1 — texture-fraction renormalization helper.

Pins the harmonize-stage decision function + the renormalization
transformation. The unit tests exercise the action discriminator
across every threshold band (renormalize / warn_retain / exclude)
including the inclusive boundaries (3.0% lower, 5.0% upper), and
the renormalize transformation against a representative HWSD-style
non-summing layer so the post-state always sums to exactly 100
within float tolerance.
"""
from __future__ import annotations

import pytest

from prismpy.harmonize.texture_renormalize import (
    TextureRenormalizationProvenance,
    renormalize_layer,
    texture_action_for,
)
from prismpy.models.soil import SoilLayer


def _layer(sand: float, clay: float, silt: float) -> SoilLayer:
    """Construct a layer with explicit silt so the dataclass's
    auto-derive does not paper over a non-summing source. Other
    fields are populated with sentinel values that pass through
    the renormalize transformation unchanged."""
    return SoilLayer(
        depth_top=0.0,
        depth_bottom=0.2,
        sand=sand,
        clay=clay,
        silt=silt,
        organic_carbon=0.5,
        bulk_density=1.4,
        ph=6.5,
    )


# ---------------------------------------------------------------------------
# AC-1 — texture_action_for boundaries + bands
# ---------------------------------------------------------------------------


def test_action_renormalize_for_small_deviation():
    """A layer whose texture sums to 100.5% (delta=0.5%) is
    routed to renormalize."""
    layer = _layer(sand=33.0, silt=33.0, clay=34.5)
    assert texture_action_for(layer) == "renormalize"


def test_action_renormalize_at_lower_boundary_inclusive():
    """The 3.0% boundary is inclusive on the renormalize side —
    a sum of 103.0 (delta=3.0%) is renormalized, not warned."""
    layer = _layer(sand=33.0, silt=33.0, clay=37.0)
    assert texture_action_for(layer) == "renormalize"


def test_action_warn_retain_for_band_above_3():
    """A layer with delta=4.0% sits in the warn band and retains
    its values without transformation."""
    layer = _layer(sand=33.0, silt=33.0, clay=38.0)
    assert texture_action_for(layer) == "warn_retain"


def test_action_warn_retain_at_upper_boundary_inclusive():
    """The 5.0% boundary is inclusive on the warn side — a sum of
    105.0 is in the warn band, not exclude."""
    layer = _layer(sand=33.0, silt=33.0, clay=39.0)
    assert texture_action_for(layer) == "warn_retain"


def test_action_exclude_for_large_deviation():
    """A layer with sum=110% (delta=10%) falls outside the
    accept band and is routed to exclude."""
    layer = _layer(sand=40.0, silt=30.0, clay=40.0)
    assert texture_action_for(layer) == "exclude"


def test_action_exclude_just_above_5_percent():
    """The threshold is strictly inclusive at 5.0; 5.001% deviation
    routes to exclude."""
    layer = _layer(sand=33.0, silt=33.001, clay=39.0)
    assert texture_action_for(layer) == "exclude"


def test_action_handles_layer_with_silt_none():
    """When ``silt`` is None the dataclass auto-derives it as
    ``100 - sand - clay`` so the layer always sums to 100 by
    construction. The action discriminator returns 'renormalize'
    (delta=0) under this path."""
    # Build via dataclass __post_init__ which sets silt = 100 - 30 - 30 = 40
    layer = SoilLayer(
        depth_top=0.0, depth_bottom=0.2,
        sand=30.0, clay=30.0,  # silt left None -> auto = 40.0
    )
    assert texture_action_for(layer) == "renormalize"


# ---------------------------------------------------------------------------
# AC-1 — renormalize_layer transformation
# ---------------------------------------------------------------------------


def test_renormalize_post_sum_is_exactly_100():
    """Post-renormalize sand + silt + clay sums to 100 within
    float tolerance."""
    layer = _layer(sand=33.0, silt=33.0, clay=34.5)
    new_layer, provenance = renormalize_layer(layer, cell_id=42, layer_idx=0)
    new_sum = new_layer.sand + new_layer.silt + new_layer.clay
    assert abs(new_sum - 100.0) < 1e-9


def test_renormalize_provenance_captures_original_triple():
    """The provenance entry preserves the pre-state so an auditor
    can replay the transformation exactly."""
    layer = _layer(sand=33.0, silt=33.0, clay=34.5)
    new_layer, prov = renormalize_layer(layer, cell_id=42, layer_idx=0)
    assert prov.original_sand == 33.0
    assert prov.original_silt == 33.0
    assert prov.original_clay == 34.5
    assert prov.original_sum == pytest.approx(100.5, abs=1e-9)
    assert prov.delta_from_100 == pytest.approx(0.5, abs=1e-9)
    assert prov.renormalization_factor == pytest.approx(100.0 / 100.5, abs=1e-12)
    assert prov.cell_id == 42
    assert prov.layer_idx == 0


def test_renormalize_provenance_is_pydantic():
    """Provenance is a Pydantic model so it can serialize cleanly."""
    layer = _layer(sand=33.0, silt=33.0, clay=34.5)
    _, prov = renormalize_layer(layer, cell_id=1, layer_idx=2)
    assert isinstance(prov, TextureRenormalizationProvenance)
    dumped = prov.model_dump()
    assert dumped["category"] == "texture_renormalize"
    assert dumped["cell_id"] == 1


def test_renormalize_preserves_non_texture_fields():
    """Renormalization only touches sand / silt / clay — other
    layer fields (organic carbon, pH, bulk density, depths) ride
    through unchanged."""
    layer = _layer(sand=33.0, silt=33.0, clay=34.5)
    new_layer, _ = renormalize_layer(layer, cell_id=0, layer_idx=0)
    assert new_layer.depth_top == layer.depth_top
    assert new_layer.depth_bottom == layer.depth_bottom
    assert new_layer.organic_carbon == layer.organic_carbon
    assert new_layer.bulk_density == layer.bulk_density
    assert new_layer.ph == layer.ph


def test_renormalize_zero_sum_raises():
    """A degenerate layer (sum=0) must not divide by zero. The
    function raises so the caller's mis-sequenced call surfaces
    instead of producing NaN."""
    layer = _layer(sand=0.0, silt=0.0, clay=0.0)
    with pytest.raises(ValueError, match="non-positive sum"):
        renormalize_layer(layer, cell_id=0, layer_idx=0)


# ---------------------------------------------------------------------------
# AC-1 — anti-mutation pin
# ---------------------------------------------------------------------------


def test_threshold_change_flips_boundary_behavior():
    """A delta of 4.0% sits in the warn band today. If the
    renormalize threshold is mutated upward to 5.0 (a common
    typo / regression direction), the same layer would flip to
    'renormalize' — this test pins the band assignment so that
    mutation surfaces as a failure here."""
    layer = _layer(sand=33.0, silt=33.0, clay=38.0)
    assert texture_action_for(layer) == "warn_retain"
    # Sanity: the same layer at 102.0% would be 'renormalize'
    layer_2pct = _layer(sand=33.0, silt=33.0, clay=36.0)
    assert texture_action_for(layer_2pct) == "renormalize"
