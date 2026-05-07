"""Structural pin: ScenarioBlock + closed enums + manifest integration.

Sprint G AC-G-3 + AC-G-4 + AC-G-10 enforcement at the schema layer.
``prismpy/.local/SPRINT-G-VERIFICATION-STRATEGY.md`` §2 AC-G-3 + AC-G-4
+ §2 AC-G-10 verification probes mapped to per-test fixtures here.

Schema invariants:

* ``ScenarioRole.BASE`` is the literal string ``"baseline"`` per
  codex LOW-1 absorption (NOT ``"base"`` — the longer form is more
  durable for future UI/API consumers).
* ``BiasCorrectionMethod`` exhaustively covers
  ``{NONE, DELTA_METHOD, QUANTILE_MAPPING, TREND_PRESERVING, UNKNOWN}``.
* ``ScenarioBlock`` enforces 10 required fields with bounds:
  - ``scenario_label``, ``gcm_source``, ``rcp_or_ssp``,
    ``baseline_reference_label`` non-empty
  - ``time_slice_start`` / ``time_slice_end`` ∈ [1900, 2200]
  - ``time_slice_end >= time_slice_start`` (cross-field)
  - ``co2_ppm`` ∈ [200.0, 2000.0]
  - ``co2_ppm_provenance`` non-empty + non-whitespace (AC-G-10)
* ``create_manifest()`` accepts an optional ``scenario`` parameter;
  when None, no ``scenario`` key in output; when provided, embeds
  ``model_dump()`` payload at the ``scenario`` key.
* ``validate_manifest()`` validates the ``scenario`` key against
  ``ScenarioBlock`` if present; manifests without the key validate
  cleanly (codex H3 absorption — schema is OPTIONAL outside scenario
  contexts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from prismpy.models.scenario import (
    BiasCorrectionMethod,
    MissingProvenanceError,
    ScenarioBlock,
    ScenarioRole,
)
from prismpy.packaging.manifest import (
    create_manifest,
    save_manifest,
    validate_manifest,
)


# ── §AC-G-4 closed enums ─────────────────────────────────────────────


def test_scenario_role_members_are_baseline_scenario_projection() -> None:
    """BASE = 'baseline' per codex LOW-1 absorption."""
    members = {role.name: role.value for role in ScenarioRole}
    assert members == {
        "BASE": "baseline",
        "SCENARIO": "scenario",
        "PROJECTION": "projection",
    }


def test_scenario_role_base_is_baseline_not_base() -> None:
    """Codex LOW-1 absorption pin: BASE serializes as 'baseline'."""
    assert ScenarioRole.BASE.value == "baseline"
    assert ScenarioRole.BASE.value != "base"


def test_bias_correction_method_members() -> None:
    members = {m.name: m.value for m in BiasCorrectionMethod}
    assert members == {
        "NONE": "none",
        "DELTA_METHOD": "delta_method",
        "QUANTILE_MAPPING": "quantile_mapping",
        "TREND_PRESERVING": "trend_preserving",
        "UNKNOWN": "unknown",
    }


# ── §AC-G-3 ScenarioBlock required fields ────────────────────────────


def _valid_block_kwargs() -> Dict[str, Any]:
    return {
        "scenario_label": "niamey-millet-projection-1",
        "scenario_role": ScenarioRole.PROJECTION,
        "gcm_source": "gfdl-esm4",
        "rcp_or_ssp": "ssp585",
        "time_slice_start": 2046,
        "time_slice_end": 2065,
        "baseline_reference_label": "niamey-millet-baseline",
        "bias_correction_method": BiasCorrectionMethod.QUANTILE_MAPPING,
        "co2_ppm": 571.0,
        "co2_ppm_provenance": (
            "AR6 WG1 Annex III + RCMIP, mid-year-of-period convention"
        ),
    }


def test_scenario_block_round_trips_with_valid_inputs() -> None:
    block = ScenarioBlock(**_valid_block_kwargs())
    assert block.scenario_label == "niamey-millet-projection-1"
    assert block.scenario_role == "projection"
    assert block.bias_correction_method == "quantile_mapping"
    assert block.co2_ppm == 571.0


def test_scenario_block_required_fields_all_enforced() -> None:
    """Drop one field at a time and confirm each absence raises
    ValidationError. Catches a future commit that loosens a required
    field to optional."""
    base = _valid_block_kwargs()
    for missing in (
        "scenario_label",
        "scenario_role",
        "gcm_source",
        "rcp_or_ssp",
        "time_slice_start",
        "time_slice_end",
        "baseline_reference_label",
        "bias_correction_method",
        "co2_ppm",
    ):
        partial = {k: v for k, v in base.items() if k != missing}
        with pytest.raises(ValidationError):
            ScenarioBlock(**partial)


def test_scenario_label_must_be_non_empty() -> None:
    kwargs = _valid_block_kwargs()
    kwargs["scenario_label"] = ""
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


def test_gcm_source_must_be_non_empty() -> None:
    kwargs = _valid_block_kwargs()
    kwargs["gcm_source"] = ""
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


def test_rcp_or_ssp_must_be_non_empty() -> None:
    kwargs = _valid_block_kwargs()
    kwargs["rcp_or_ssp"] = ""
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


def test_baseline_reference_label_must_be_non_empty() -> None:
    kwargs = _valid_block_kwargs()
    kwargs["baseline_reference_label"] = ""
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


# ── §AC-G-3 schema bounds ────────────────────────────────────────────


@pytest.mark.parametrize("year", [1899, 2201])
def test_time_slice_start_rejects_out_of_bounds(year: int) -> None:
    kwargs = _valid_block_kwargs()
    kwargs["time_slice_start"] = year
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


@pytest.mark.parametrize("year", [1899, 2201])
def test_time_slice_end_rejects_out_of_bounds(year: int) -> None:
    kwargs = _valid_block_kwargs()
    kwargs["time_slice_end"] = year
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


@pytest.mark.parametrize("year", [1900, 2200])
def test_time_slice_boundary_years_accepted(year: int) -> None:
    """Inclusive boundaries accepted (1900 + 2200 are valid)."""
    kwargs = _valid_block_kwargs()
    kwargs["time_slice_start"] = year
    kwargs["time_slice_end"] = year
    block = ScenarioBlock(**kwargs)
    assert block.time_slice_start == year
    assert block.time_slice_end == year


def test_time_slice_end_must_not_be_before_start() -> None:
    kwargs = _valid_block_kwargs()
    kwargs["time_slice_start"] = 2065
    kwargs["time_slice_end"] = 2046  # reversed
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


def test_time_slice_end_equal_start_accepted() -> None:
    """A single-year slice (start == end) is valid."""
    kwargs = _valid_block_kwargs()
    kwargs["time_slice_start"] = 2050
    kwargs["time_slice_end"] = 2050
    block = ScenarioBlock(**kwargs)
    assert block.time_slice_end == 2050


@pytest.mark.parametrize("ppm", [199.99, 2000.01])
def test_co2_ppm_rejects_out_of_bounds(ppm: float) -> None:
    kwargs = _valid_block_kwargs()
    kwargs["co2_ppm"] = ppm
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


@pytest.mark.parametrize("ppm", [200.0, 2000.0])
def test_co2_ppm_boundary_accepted(ppm: float) -> None:
    kwargs = _valid_block_kwargs()
    kwargs["co2_ppm"] = ppm
    block = ScenarioBlock(**kwargs)
    assert block.co2_ppm == ppm


# ── §AC-G-4 closed enum out-of-domain rejection ──────────────────────


def test_scenario_role_rejects_out_of_domain_string() -> None:
    kwargs = _valid_block_kwargs()
    kwargs["scenario_role"] = "alternate_history"  # not a closed enum member
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


def test_bias_correction_method_rejects_out_of_domain_string() -> None:
    kwargs = _valid_block_kwargs()
    kwargs["bias_correction_method"] = "cubic_spline_invented"
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


# ── §AC-G-10 mandatory co2_ppm_provenance ────────────────────────────


def test_co2_ppm_provenance_missing_raises() -> None:
    kwargs = _valid_block_kwargs()
    del kwargs["co2_ppm_provenance"]
    with pytest.raises(ValidationError) as exc_info:
        ScenarioBlock(**kwargs)
    # Confirm the typed cause is preserved (callers can discriminate).
    causes = [e.get("type") for e in exc_info.value.errors()]
    assert any("missing_provenance" in str(c) or "value_error" in str(c) for c in causes)


def test_co2_ppm_provenance_none_raises() -> None:
    kwargs = _valid_block_kwargs()
    kwargs["co2_ppm_provenance"] = None
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


def test_co2_ppm_provenance_empty_string_raises() -> None:
    kwargs = _valid_block_kwargs()
    kwargs["co2_ppm_provenance"] = ""
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


def test_co2_ppm_provenance_whitespace_only_raises() -> None:
    """Per AC-G-10 §10.3 acceptance: ' ' counts as empty."""
    kwargs = _valid_block_kwargs()
    kwargs["co2_ppm_provenance"] = "   "
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


def test_missing_provenance_error_is_preserved_as_cause() -> None:
    """The typed ``MissingProvenanceError`` is the validator's
    underlying raise; ValidationError wraps it."""
    kwargs = _valid_block_kwargs()
    kwargs["co2_ppm_provenance"] = ""
    try:
        ScenarioBlock(**kwargs)
    except ValidationError as ve:
        # The pydantic v2 error struct includes the original message
        msg = str(ve)
        assert "co2_ppm_provenance" in msg or "provenance" in msg.lower()


# ── §AC-G-3 extra fields rejected ────────────────────────────────────


def test_extra_fields_rejected_with_validation_error() -> None:
    """Forbid extra fields so a producer typo surfaces as
    ValidationError instead of silently dropping at consumer read."""
    kwargs = _valid_block_kwargs()
    kwargs["typo_field"] = "should not be accepted"
    with pytest.raises(ValidationError):
        ScenarioBlock(**kwargs)


# ── §AC-G-3 model_dump round-trip ────────────────────────────────────


def test_model_dump_serializes_enums_as_string_values() -> None:
    """Config ``use_enum_values=True`` keeps the serialized form
    readable: 'baseline' not 'ScenarioRole.BASE'."""
    block = ScenarioBlock(**_valid_block_kwargs())
    payload = block.model_dump()
    assert payload["scenario_role"] == "projection"
    assert payload["bias_correction_method"] == "quantile_mapping"


def test_model_validate_round_trips_through_dump() -> None:
    """Serialize → re-validate is a no-op (idempotent contract)."""
    block_a = ScenarioBlock(**_valid_block_kwargs())
    block_b = ScenarioBlock.model_validate(block_a.model_dump())
    assert block_a.model_dump() == block_b.model_dump()


# ── §Manifest integration: optional scenario parameter ───────────────


@pytest.fixture
def synthetic_package(tmp_path: Path) -> Path:
    """Tiny on-disk package fixture for create_manifest tests."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    return pkg


@pytest.fixture
def synthetic_config() -> Dict[str, Any]:
    return {
        "project_name": "synthetic-test",
        "region_name": "Niamey",
        "country": "Niger",
        "gadm_level": 1,
        "crop_name": "millet",
        "planting_doy": 152,
        "maturity_doy": 273,
        "start_year": 2020,
        "end_year": 2022,
        "spinup_years": 0,
    }


def test_create_manifest_omits_scenario_key_when_none(
    synthetic_package: Path, synthetic_config: Dict[str, Any]
) -> None:
    """Existing observed-climate manifests do not carry scenario; the
    schema is OPTIONAL outside scenario contexts (codex H3)."""
    manifest = create_manifest(synthetic_package, synthetic_config)
    assert "scenario" not in manifest


def test_create_manifest_embeds_scenario_block_payload(
    synthetic_package: Path, synthetic_config: Dict[str, Any]
) -> None:
    block = ScenarioBlock(**_valid_block_kwargs())
    manifest = create_manifest(
        synthetic_package, synthetic_config, scenario=block
    )
    assert manifest["scenario"]["scenario_label"] == block.scenario_label
    assert manifest["scenario"]["scenario_role"] == "projection"
    assert manifest["scenario"]["co2_ppm"] == 571.0


def test_create_manifest_accepts_serialized_dict_scenario(
    synthetic_package: Path, synthetic_config: Dict[str, Any]
) -> None:
    """Callers that already serialized to a dict can pass it through;
    validate_manifest catches shape errors at the validation layer."""
    block = ScenarioBlock(**_valid_block_kwargs())
    payload = block.model_dump()
    manifest = create_manifest(
        synthetic_package, synthetic_config, scenario=payload
    )
    assert manifest["scenario"] == payload


def test_create_manifest_rejects_non_block_non_dict_scenario(
    synthetic_package: Path, synthetic_config: Dict[str, Any]
) -> None:
    with pytest.raises(TypeError):
        create_manifest(
            synthetic_package,
            synthetic_config,
            scenario="not-a-block",
        )


def test_validate_manifest_passes_when_no_scenario_key(
    synthetic_package: Path, synthetic_config: Dict[str, Any], tmp_path: Path
) -> None:
    """Observed-climate path: no scenario key, validate cleanly."""
    manifest = create_manifest(synthetic_package, synthetic_config)
    out = tmp_path / "manifest.json"
    save_manifest(manifest, out)
    results = validate_manifest(out, synthetic_package)
    assert results["valid"] is True


def test_validate_manifest_validates_scenario_against_schema(
    synthetic_package: Path, synthetic_config: Dict[str, Any], tmp_path: Path
) -> None:
    """When scenario key is present, the schema validator MUST run."""
    block = ScenarioBlock(**_valid_block_kwargs())
    manifest = create_manifest(
        synthetic_package, synthetic_config, scenario=block
    )
    out = tmp_path / "manifest.json"
    save_manifest(manifest, out)
    results = validate_manifest(out, synthetic_package)
    assert results["valid"] is True


def test_validate_manifest_rejects_invalid_scenario_payload(
    synthetic_package: Path, synthetic_config: Dict[str, Any], tmp_path: Path
) -> None:
    """An invalid scenario dict in a manifest.json on disk MUST fail
    validation at validate_manifest read time."""
    block = ScenarioBlock(**_valid_block_kwargs())
    manifest = create_manifest(
        synthetic_package, synthetic_config, scenario=block
    )
    # Tamper with the on-disk manifest: drop a required field.
    out = tmp_path / "manifest.json"
    save_manifest(manifest, out)
    import json

    raw = json.loads(out.read_text(encoding="utf-8"))
    del raw["scenario"]["scenario_label"]
    out.write_text(json.dumps(raw, sort_keys=True, indent=2), encoding="utf-8")

    with pytest.raises(ValidationError):
        validate_manifest(out, synthetic_package)


def test_validate_manifest_rejects_scenario_with_empty_provenance(
    synthetic_package: Path, synthetic_config: Dict[str, Any], tmp_path: Path
) -> None:
    """AC-G-10: empty co2_ppm_provenance in a scenario manifest must
    fail validation at read time."""
    block = ScenarioBlock(**_valid_block_kwargs())
    manifest = create_manifest(
        synthetic_package, synthetic_config, scenario=block
    )
    out = tmp_path / "manifest.json"
    save_manifest(manifest, out)
    import json

    raw = json.loads(out.read_text(encoding="utf-8"))
    raw["scenario"]["co2_ppm_provenance"] = ""
    out.write_text(json.dumps(raw, sort_keys=True, indent=2), encoding="utf-8")

    with pytest.raises(ValidationError):
        validate_manifest(out, synthetic_package)
