"""The seasonal thermal screen on the real Stage-1 path: the bundled envelope, its loader, the model and both validators."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from prismpy.koppen.envelopes import load_ecocrop_envelopes
from prismpy.koppen.zone_aggregates import build_zone_aggregate
from prismpy.validators.climate_envelope import ClimateEnvelopeValidator
from prismpy.validators.crop_physiological import CropPhysiologicalValidator
from prismpy.validators.input_base import CropEnvelope, InputValidationContext

_C, _I, _H = "compatible", "incompatible", "marginal_heterogeneous"
_SEASONAL = "marginal_thermal_seasonal"

# (thermal, precip) per bundled zone, as before the screen existed; none of these crops is seasonal.
_OTHER_CROPS = {
    "beans": {"Af": (_I, _C), "Aw": (_I, _C), "BSh": (_I, _H), "Cfa": (_I, _C), "Cwa": (_I, _C)},
    "cowpea": {"Af": (_C, _C), "Aw": (_C, _C), "BSh": (_C, _H), "Cfa": (_I, _C), "Cwa": (_I, _C)},
    "groundnut": {"Af": (_C, _C), "Aw": (_C, _C), "BSh": (_C, _H), "Cfa": (_C, _C), "Cwa": (_I, _C)},
    "maize": {"Af": (_C, _I), "Aw": (_C, _C), "BSh": (_C, _H), "Cfa": (_C, _C), "Cwa": (_I, _C)},
    "millet": {"Af": (_C, _I), "Aw": (_C, _C), "BSh": (_C, _H), "Cfa": (_I, _H), "Cwa": (_I, _C)},
    "rice": {"Af": (_C, _C), "Aw": (_C, _H), "BSh": (_C, _I), "Cfa": (_C, _C), "Cwa": (_I, _I)},
    "sorghum": {"Af": (_C, _I), "Aw": (_C, _I), "BSh": (_C, _H), "Cfa": (_C, _I), "Cwa": (_I, _I)},
}


def _stage1(crop, envelope, zones):
    """Run both Stage-1 validators on a context built the way the wizard builds it."""
    context = InputValidationContext(
        crop_name=crop, crop_envelope=envelope,
        zone_aggregates={zone: build_zone_aggregate(zone) for zone in zones},
    )
    return CropPhysiologicalValidator().validate(context), ClimateEnvelopeValidator().validate(context)


def test_from_ecocrop_keeps_every_model_field_of_each_bundled_crop():
    for crop, loaded in load_ecocrop_envelopes().items():
        expected = {key: loaded[key] for key in ("TMIN", "TMAX", "RMIN", "RMAX", "thermal_screen")}
        assert CropEnvelope.from_ecocrop(loaded).model_dump() == expected, crop


def test_from_ecocrop_refuses_a_key_it_does_not_know():
    with pytest.raises(ValidationError, match="TOPMN"):
        CropEnvelope.from_ecocrop({**load_ecocrop_envelopes()["potato"], "TOPMN": 15.0})


def test_potato_in_aw_is_a_seasonal_marginal_not_a_region_mismatch():
    # Aw alone: across all five zones, Af's rainfall keeps potato a region mismatch by design.
    envelope = CropEnvelope.from_ecocrop(load_ecocrop_envelopes()["potato"])
    physiological, climate = _stage1("potato", envelope, ["Aw"])
    for result in (physiological, climate):
        assert result.metadata["per_zone_verdicts"]["Aw"] == {"precip": _C, "thermal": _SEASONAL}
    assert physiological.issues == []
    assert [(i.category, i.details["zone"], i.details["variable"]) for i in climate.issues] == [
        ("climate_envelope_tail", "Aw", "thermal")]


def test_potato_in_aw_without_its_screen_is_a_region_mismatch():
    loaded = load_ecocrop_envelopes()["potato"]
    envelope = CropEnvelope.from_ecocrop({key: value for key, value in loaded.items() if key != "thermal_screen"})
    physiological, climate = _stage1("potato", envelope, ["Aw"])
    assert physiological.metadata["per_zone_verdicts"]["Aw"] == {"precip": _C, "thermal": _I}
    assert [i.category for i in physiological.issues] == ["crop_region_mismatch"]
    assert climate.issues == []


@pytest.mark.parametrize("crop", sorted(_OTHER_CROPS))
def test_the_other_crops_keep_their_stage1_verdicts(crop):
    envelope = CropEnvelope.from_ecocrop(load_ecocrop_envelopes()[crop])
    physiological, climate = _stage1(crop, envelope, sorted(_OTHER_CROPS[crop]))
    expected = {zone: {"thermal": thermal, "precip": precip} for zone, (thermal, precip) in _OTHER_CROPS[crop].items()}
    assert physiological.metadata["per_zone_verdicts"] == expected
    assert climate.metadata["per_zone_verdicts"] == expected
    assert {i.category for i in physiological.issues} == {"crop_region_mismatch"}
    assert {i.category for i in climate.issues} == {"climate_envelope_tail"}
