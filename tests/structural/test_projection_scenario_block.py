"""Unit tests for the projection scenario-block builder.

Covers :func:`prismpy.packaging.scenario_helpers.build_projection_scenario_block_for_period`
in isolation: the builder must produce a schema-valid PROJECTION
``ScenarioBlock`` whose CO₂ value + provenance agree with the canonical
lookup (so the model's CO₂ post-validator passes), and whose serialized
form carries every field the prism-runner UC2 consumer requires.
"""

from __future__ import annotations

import pytest

from prismpy.models.scenario import ScenarioBlock, ScenarioRole
from prismpy.packaging.scenario_helpers import (
    build_projection_scenario_block_for_period,
)
from prismpy.standards.co2_ppm import get_co2_ppm_with_provenance

# Mirrors prism_runner UC2
# ``climate_scenarios.schemas.REQUIRED_SCENARIO_MANIFEST_FIELDS`` — the fields
# UC2's pre-flight hard-fails on if absent. Pinned here so a drift in the
# producer surfaces against the consumer's contract.
_UC2_REQUIRED_FIELDS = (
    "scenario_label",
    "scenario_role",
    "gcm_source",
    "rcp_or_ssp",
    "time_slice_start",
    "time_slice_end",
    "baseline_reference_label",
    "bias_correction_method",
    "co2_ppm",
)

_REGISTERED_PERIODS = (
    ("ssp245", (2046, 2065)),
    ("ssp245", (2086, 2100)),
    ("ssp585", (2046, 2065)),
    ("ssp585", (2086, 2100)),
)


def _build(ssp: str = "ssp245", time_slice=(2046, 2065)) -> ScenarioBlock:
    return build_projection_scenario_block_for_period(
        region_name="Kano",
        crop_name="Cowpea",
        gcm_source="gfdl-esm4",
        rcp_or_ssp=ssp,
        time_slice_start=time_slice[0],
        time_slice_end=time_slice[1],
        baseline_reference_label="OBSERVED_KANO_COWPEA_2000-2020",
    )


def test_builds_valid_projection_block():
    block = _build()
    assert isinstance(block, ScenarioBlock)
    # use_enum_values=True → the stored value is the plain string.
    assert block.scenario_role == ScenarioRole.PROJECTION.value == "projection"
    assert block.gcm_source == "gfdl-esm4"
    assert block.baseline_reference_label == "OBSERVED_KANO_COWPEA_2000-2020"
    assert block.bias_correction_method == "quantile_mapping"


def test_co2_pair_matches_canonical_lookup_exactly():
    block = _build("ssp585", (2086, 2100))
    expected_ppm, expected_provenance = get_co2_ppm_with_provenance(
        "ssp585", (2086, 2100)
    )
    assert block.co2_ppm == expected_ppm == 1054.0
    assert block.co2_ppm_provenance == expected_provenance


@pytest.mark.parametrize("ssp,time_slice", _REGISTERED_PERIODS)
def test_all_registered_periods_build(ssp, time_slice):
    block = build_projection_scenario_block_for_period(
        region_name="Kano",
        crop_name="Sorghum",
        gcm_source="ukesm1-0-ll",
        rcp_or_ssp=ssp,
        time_slice_start=time_slice[0],
        time_slice_end=time_slice[1],
        baseline_reference_label="BASE",
    )
    expected_ppm, _ = get_co2_ppm_with_provenance(ssp, time_slice)
    assert block.co2_ppm == expected_ppm


def test_unregistered_period_raises():
    with pytest.raises(ValueError):
        build_projection_scenario_block_for_period(
            region_name="Kano",
            crop_name="Cowpea",
            gcm_source="gfdl-esm4",
            rcp_or_ssp="ssp245",
            time_slice_start=2030,
            time_slice_end=2049,
            baseline_reference_label="BASE",
        )


def test_serialized_block_carries_uc2_required_fields():
    dumped = _build().model_dump()
    for field in _UC2_REQUIRED_FIELDS:
        assert field in dumped, f"missing UC2-required field {field!r}"
        assert dumped[field] is not None, f"UC2-required field {field!r} is None"
    assert dumped["scenario_role"] == "projection"


def test_label_is_unique_per_dimension():
    a = _build("ssp245", (2046, 2065))
    b = _build("ssp585", (2046, 2065))
    assert a.scenario_label != b.scenario_label
    assert "SSP245".lower() in a.scenario_label.lower()
