"""Structural pin: validate_scenario_set + mode disambiguation + AC-G-6 conflict rule.

Sprint G AC-G-5 + AC-G-6 verification probes mapped to per-test fixtures:

* §5.1-§5.4 ValidationMode enum + structured-trace fields + __str__ format
* §5.5 cell_id set-equality
* §5.6 lat/lon per-cell equality
* §5.7 SHA byte-identity on the 4+ identity files (crop_mask + soil_mask
  + soil/*.SOL + every management/*.txt)
* §5.8 differ-only-in-allowed paths (sanity — implicit; we don't strictly
  pin "weather differs" because the test fixtures don't generate weather)
* §5.9 pairing rule (baseline_reference_label == baseline.scenario_label)
* §5.10-§5.11 CLI wrapper exit codes
* §5.12 mode=ship default (F-G-3 active on 'unknown')
* §6.1-§6.5 bias-correction conflict rule + AC-G-6 mode-disambiguation

Plus 4 mode-disambiguation drills #6/#6a + #7/#7a per AC-G-12 partial
coverage at boundary 2/7 (full drill matrix lands at AC-G-12 commit).

Test fixture builder: ``_build_pair_fixture`` lays out a synthetic
baseline + projection on disk with all the identity-coupled files and
matching scenario-block payloads. Each drill mutates ONE invariant on
the projection side and asserts the right typed error fires.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from prismpy.models.scenario import (
    BiasCorrectionMethod,
    ScenarioBlock,
    ScenarioRole,
)
from prismpy.validators.scenario_set import (
    BiasCorrectionConflictError,
    IdentityDriftError,
    PairingRuleError,
    ScenarioSetValidationError,
    UnknownBiasCorrectionInShipModeError,
    UnregisteredScenarioInShipModeError,
    ValidationMode,
    validate_scenario_set,
)


# ── Fixture helpers ──────────────────────────────────────────────────


def _baseline_block_kwargs() -> Dict[str, Any]:
    return {
        "scenario_label": "niamey-millet-baseline",
        "scenario_role": ScenarioRole.BASE,
        "gcm_source": "observed",
        "rcp_or_ssp": "observed",
        "time_slice_start": 2017,
        "time_slice_end": 2021,
        "baseline_reference_label": "niamey-millet-baseline",
        "bias_correction_method": BiasCorrectionMethod.NONE,
        "co2_ppm": 410.0,
        "co2_ppm_provenance": (
            "AR6 WG1 Annex III observed atmospheric record, "
            "midpoint of 2017-2021"
        ),
    }


def _projection_block_kwargs(
    *,
    label: str = "niamey-millet-projection-1",
    method: BiasCorrectionMethod = BiasCorrectionMethod.QUANTILE_MAPPING,
    baseline_ref: str = "niamey-millet-baseline",
) -> Dict[str, Any]:
    return {
        "scenario_label": label,
        "scenario_role": ScenarioRole.PROJECTION,
        "gcm_source": "gfdl-esm4",
        "rcp_or_ssp": "ssp585",
        "time_slice_start": 2046,
        "time_slice_end": 2065,
        "baseline_reference_label": baseline_ref,
        "bias_correction_method": method,
        "co2_ppm": 571.0,
        "co2_ppm_provenance": (
            "AR6 WG1 Annex III + RCMIP, mid-year-of-period convention"
        ),
        # AC-G-11: bias-correction provenance mandatory for non-NONE method
        "scenario_bias_correction_provenance": (
            "ISIMIP3BASD v2.5.0 quantile-mapping against W5E5 v2.0"
        ),
    }


def _write_manifest(
    package_dir: Path,
    scenario_kwargs: Dict[str, Any],
) -> None:
    block = ScenarioBlock(**scenario_kwargs)
    payload = {
        "package_version": "1.0",
        "platform": "synthetic",
        "scenario": block.model_dump(),
    }
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_cell_summary(
    package_dir: Path,
    cells: Optional[List[Dict[str, Any]]] = None,
) -> None:
    cells = cells or [
        {"id": 1, "lat": 13.5, "lon": 2.1},
        {"id": 2, "lat": 13.6, "lon": 2.2},
    ]
    (package_dir / "cell_summary.json").write_text(
        json.dumps({"cells": cells}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_identity_files(package_dir: Path) -> None:
    """Create the identity-coupled files with deterministic content.

    The test fixture writes the same byte content to baseline + every
    projection by default; drills mutate ONE file's bytes to fire the
    SHA-mismatch path.
    """
    (package_dir / "crop_mask").mkdir(parents=True, exist_ok=True)
    (package_dir / "crop_mask" / "mask.txt").write_text(
        "0,0,1\n0,1,1\n1,1,1\n", encoding="utf-8"
    )
    (package_dir / "soil").mkdir(parents=True, exist_ok=True)
    (package_dir / "soil" / "soil_mask.txt").write_text(
        "1,1,1\n1,1,1\n1,1,1\n", encoding="utf-8"
    )
    (package_dir / "soil" / "NER.SOL").write_text(
        "*SOIL profile NER\n@profile-line-1\n", encoding="utf-8"
    )
    (package_dir / "management").mkdir(parents=True, exist_ok=True)
    (package_dir / "management" / "millet_calendar.txt").write_text(
        "planting=152\nmaturity=273\n", encoding="utf-8"
    )


def _write_weather_differ(package_dir: Path, marker: str) -> None:
    """Write a weather/ directory that differs between baseline and
    projection (so the sanity check of differ-only-in-allowed-paths
    passes in tests that exercise the full set)."""
    (package_dir / "weather").mkdir(parents=True, exist_ok=True)
    (package_dir / "weather" / "data.wth").write_text(
        f"weather-data-marker={marker}\n", encoding="utf-8"
    )


def _build_pair_fixture(
    tmp_path: Path,
    *,
    baseline_label: str = "niamey-millet-baseline",
    projection_label: str = "niamey-millet-projection-1",
    projection_baseline_ref: Optional[str] = None,
    projection_method: BiasCorrectionMethod = (
        BiasCorrectionMethod.QUANTILE_MAPPING
    ),
) -> Tuple[Path, Path]:
    """Build a synthetic baseline + projection pair on disk.

    Returns ``(baseline_path, projection_path)``. Both have:
    * manifest.json with the ScenarioBlock payload
    * cell_summary.json (default 2 cells; identity-coupled)
    * crop_mask/mask.txt + soil/soil_mask.txt + soil/NER.SOL +
      management/millet_calendar.txt — all SHA-equal between baseline
      + projection by default
    * weather/data.wth with different marker bytes (sanity:
      projections differ from baseline in the allowed weather path)
    """
    baseline_path = tmp_path / "baseline"
    projection_path = tmp_path / "projection_1"

    base_kwargs = _baseline_block_kwargs()
    base_kwargs["scenario_label"] = baseline_label
    base_kwargs["baseline_reference_label"] = baseline_label
    _write_manifest(baseline_path, base_kwargs)
    _write_cell_summary(baseline_path)
    _write_identity_files(baseline_path)
    _write_weather_differ(baseline_path, marker="baseline")

    proj_kwargs = _projection_block_kwargs(
        label=projection_label,
        method=projection_method,
        baseline_ref=projection_baseline_ref or baseline_label,
    )
    _write_manifest(projection_path, proj_kwargs)
    _write_cell_summary(projection_path)
    _write_identity_files(projection_path)
    _write_weather_differ(projection_path, marker="projection_1")

    return baseline_path, projection_path


# ── §5.1 ValidationMode enum ─────────────────────────────────────────


def test_validation_mode_members() -> None:
    members = {m.name: m.value for m in ValidationMode}
    assert members == {"SHIP": "ship", "LEGACY": "legacy"}


def test_validation_mode_default_is_ship() -> None:
    """``mode=ValidationMode.SHIP`` is the contract default — F-G-3
    active, which is the prismpy-generated deliverable path."""
    import inspect

    sig = inspect.signature(validate_scenario_set)
    assert sig.parameters["mode"].default is ValidationMode.SHIP


# ── §5.3-§5.4 structured-trace fields + __str__ format ───────────────


def test_scenario_set_validation_error_carries_structured_fields() -> None:
    err = ScenarioSetValidationError(
        package_label="projection_1",
        failing_field_path="manifest.scenario.gcm_source",
        expected="gfdl-esm4",
        actual="ipsl-cm6a-lr",
    )
    assert err.package_label == "projection_1"
    assert err.failing_field_path == "manifest.scenario.gcm_source"
    assert err.expected == "gfdl-esm4"
    assert err.actual == "ipsl-cm6a-lr"


def test_scenario_set_validation_error_str_format() -> None:
    err = ScenarioSetValidationError(
        package_label="projection_2",
        failing_field_path="cell_summary.cells[*].id",
        expected=[1, 2, 3],
        actual=[1, 2, 4],
    )
    msg = str(err)
    assert "projection_2" in msg
    assert "cell_summary.cells[*].id" in msg
    assert "mismatch" in msg.lower()


# ── §5.5 cell_id set-equality ────────────────────────────────────────


def test_drill_cell_id_mutated_fires_identity_drift(tmp_path: Path) -> None:
    base, proj = _build_pair_fixture(tmp_path)
    # Mutate one cell's id on the projection side.
    cs = json.loads((proj / "cell_summary.json").read_text(encoding="utf-8"))
    cs["cells"][0]["id"] = 99
    (proj / "cell_summary.json").write_text(
        json.dumps(cs, indent=2, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(IdentityDriftError) as exc_info:
        validate_scenario_set(base, [proj])
    assert "cell_summary.cells[*].id" in exc_info.value.failing_field_path


# ── §5.6 lat/lon per-cell equality ───────────────────────────────────


@pytest.mark.parametrize("axis", ["lat", "lon"])
def test_drill_lat_lon_drift_fires_identity_drift(
    tmp_path: Path, axis: str
) -> None:
    base, proj = _build_pair_fixture(tmp_path)
    cs = json.loads((proj / "cell_summary.json").read_text(encoding="utf-8"))
    cs["cells"][0][axis] = cs["cells"][0][axis] + 0.5
    (proj / "cell_summary.json").write_text(
        json.dumps(cs, indent=2, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(IdentityDriftError) as exc_info:
        validate_scenario_set(base, [proj])
    assert axis in exc_info.value.failing_field_path
    assert "id=" in exc_info.value.failing_field_path


# ── §5.7 SHA byte-identity on identity files ─────────────────────────


@pytest.mark.parametrize(
    "rel_path",
    [
        "crop_mask/mask.txt",
        "soil/soil_mask.txt",
        "soil/NER.SOL",
        "management/millet_calendar.txt",
    ],
)
def test_drill_identity_file_sha_mutation_fires(
    tmp_path: Path, rel_path: str
) -> None:
    """Mutate one byte of each identity-coupled file in turn; assert
    each fires IdentityDriftError on SHA mismatch."""
    base, proj = _build_pair_fixture(tmp_path)
    target = proj / rel_path
    original = target.read_bytes()
    target.write_bytes(original + b"\nmutation\n")

    with pytest.raises(IdentityDriftError) as exc_info:
        validate_scenario_set(base, [proj])
    assert rel_path in exc_info.value.failing_field_path
    assert "sha256" in exc_info.value.failing_field_path


def test_drill_identity_file_added_fires_identity_drift(
    tmp_path: Path,
) -> None:
    """Adding a file under management/ to the projection (but not
    baseline) breaks the key-set parity check before SHA comparison."""
    base, proj = _build_pair_fixture(tmp_path)
    (proj / "management" / "extra_management.txt").write_text(
        "extra=1\n", encoding="utf-8"
    )

    with pytest.raises(IdentityDriftError) as exc_info:
        validate_scenario_set(base, [proj])
    assert "identity_files[*].path" in exc_info.value.failing_field_path


# ── §5.9 pairing rule ────────────────────────────────────────────────


def test_drill_pairing_rule_wrong_baseline_reference(tmp_path: Path) -> None:
    base, proj = _build_pair_fixture(
        tmp_path, projection_baseline_ref="WRONG_REFERENCE_LABEL"
    )
    with pytest.raises(PairingRuleError) as exc_info:
        validate_scenario_set(base, [proj])
    assert (
        "manifest.scenario.baseline_reference_label"
        == exc_info.value.failing_field_path
    )
    assert exc_info.value.expected == "niamey-millet-baseline"
    assert exc_info.value.actual == "WRONG_REFERENCE_LABEL"


# ── Positive happy-path ──────────────────────────────────────────────


def test_happy_path_passes_in_ship_mode(tmp_path: Path) -> None:
    base, proj = _build_pair_fixture(tmp_path)
    # No raise expected. The function returns None on success.
    result = validate_scenario_set(base, [proj])
    assert result is None


def test_happy_path_passes_in_legacy_mode(tmp_path: Path) -> None:
    base, proj = _build_pair_fixture(tmp_path)
    result = validate_scenario_set(base, [proj], mode=ValidationMode.LEGACY)
    assert result is None


def test_happy_path_2_projections_same_method(tmp_path: Path) -> None:
    base, proj1 = _build_pair_fixture(tmp_path)
    # Build a second projection that passes identity coupling.
    proj2 = tmp_path / "projection_2"
    proj_kwargs = _projection_block_kwargs(
        label="niamey-millet-projection-2",
        method=BiasCorrectionMethod.QUANTILE_MAPPING,
    )
    _write_manifest(proj2, proj_kwargs)
    _write_cell_summary(proj2)
    _write_identity_files(proj2)
    _write_weather_differ(proj2, marker="projection_2")

    result = validate_scenario_set(base, [proj1, proj2])
    assert result is None


# ── §6.1 bias-correction conflict rule ───────────────────────────────


def test_drill_bias_correction_conflict_fires(tmp_path: Path) -> None:
    """Two projections in the same set with mutually-distinct
    methods (QUANTILE_MAPPING + DELTA_METHOD) → BiasCorrectionConflictError."""
    base, proj1 = _build_pair_fixture(
        tmp_path, projection_method=BiasCorrectionMethod.QUANTILE_MAPPING
    )
    proj2 = tmp_path / "projection_2"
    proj_kwargs = _projection_block_kwargs(
        label="niamey-millet-projection-2",
        method=BiasCorrectionMethod.DELTA_METHOD,
    )
    _write_manifest(proj2, proj_kwargs)
    _write_cell_summary(proj2)
    _write_identity_files(proj2)
    _write_weather_differ(proj2, marker="projection_2")

    with pytest.raises(BiasCorrectionConflictError) as exc_info:
        validate_scenario_set(base, [proj1, proj2])
    assert exc_info.value.method_a == "quantile_mapping"
    assert exc_info.value.method_b == "delta_method"


# ── F-G-3: ship-mode rejects bias_correction_method='unknown' ────────


def test_drill_7_unknown_method_rejected_in_ship_mode(tmp_path: Path) -> None:
    """Drill #7 per AC-G-12 §12 + Draft 5 — F-G-3 boundary in
    ship mode: any projection with method='unknown' is rejected."""
    base, proj = _build_pair_fixture(
        tmp_path, projection_method=BiasCorrectionMethod.UNKNOWN
    )
    with pytest.raises(UnknownBiasCorrectionInShipModeError):
        validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)


def test_drill_7a_unknown_method_allowed_in_legacy_mode(
    tmp_path: Path,
) -> None:
    """Drill #7a per AC-G-12 + Draft 5 MEDIUM-Pass3-1 — legacy mode
    + 2 projections (qm + unknown) → validator passes (AC-G-6
    unknown-exclusion live)."""
    base, proj1 = _build_pair_fixture(
        tmp_path, projection_method=BiasCorrectionMethod.QUANTILE_MAPPING
    )
    proj2 = tmp_path / "projection_2"
    proj_kwargs = _projection_block_kwargs(
        label="niamey-millet-projection-2",
        method=BiasCorrectionMethod.UNKNOWN,
    )
    _write_manifest(proj2, proj_kwargs)
    _write_cell_summary(proj2)
    _write_identity_files(proj2)
    _write_weather_differ(proj2, marker="projection_2")

    # In legacy mode: F-G-3 not applied; AC-G-6 unknown excluded from
    # conflict check; passes cleanly.
    result = validate_scenario_set(base, [proj1, proj2], mode=ValidationMode.LEGACY)
    assert result is None


# ── §AC-G-6 'none' exclusion from conflict check ─────────────────────


def test_method_none_does_not_trigger_conflict(tmp_path: Path) -> None:
    """A baseline with method=NONE coexisting with a projection using
    QUANTILE_MAPPING does NOT fire the conflict rule (the conflict
    check operates on PROJECTIONS only; baseline's NONE is never
    compared against a projection's method)."""
    base, proj = _build_pair_fixture(tmp_path)
    # Baseline already uses NONE per _baseline_block_kwargs.
    result = validate_scenario_set(base, [proj])
    assert result is None


# ── §AC-G-6 mode boundary (F-G-3 fires before AC-G-6 path) ───────────


def test_baseline_with_unknown_method_rejected_in_ship_mode(
    tmp_path: Path,
) -> None:
    """The baseline itself with method='unknown' is rejected outright
    in ship mode (F-G-3 covers all packages, not just projections)."""
    base, proj = _build_pair_fixture(tmp_path)
    # Mutate the baseline to use unknown. AC-G-11 requires a non-NONE
    # method to also carry a bias-correction provenance string;
    # populate it so this drill exercises F-G-3 (the unknown-method
    # ship-mode rejection) rather than tripping AC-G-11 first.
    base_manifest = json.loads(
        (base / "manifest.json").read_text(encoding="utf-8")
    )
    base_manifest["scenario"]["bias_correction_method"] = "unknown"
    base_manifest["scenario"]["scenario_bias_correction_provenance"] = (
        "synthetic legacy provenance for AC-G-11 compatibility"
    )
    (base / "manifest.json").write_text(
        json.dumps(base_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(UnknownBiasCorrectionInShipModeError):
        validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)


# ── Codex round 1 boundary 4/7 P2-2 absorption — unregistered scenario ─


def test_unregistered_scenario_rejected_in_ship_mode(tmp_path: Path) -> None:
    """Codex round 1 boundary 4/7 P2-2 absorption: a projection
    package with a scenario × period not in the canonical
    ``CO2_PPM_BY_SCENARIO_PERIOD`` table MUST be rejected in ship
    mode. Without this check, ``ScenarioBlock``'s Layer 2 silently
    skips and a shipped package could carry an arbitrary CO₂ value."""
    base, proj = _build_pair_fixture(tmp_path)
    # Mutate the projection's scenario to an unregistered tuple
    proj_manifest = json.loads(
        (proj / "manifest.json").read_text(encoding="utf-8")
    )
    proj_manifest["scenario"]["rcp_or_ssp"] = "ssp370"
    proj_manifest["scenario"]["time_slice_start"] = 2030
    proj_manifest["scenario"]["time_slice_end"] = 2049
    proj_manifest["scenario"]["co2_ppm"] = 999.0  # arbitrary
    (proj / "manifest.json").write_text(
        json.dumps(proj_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(UnregisteredScenarioInShipModeError) as exc_info:
        validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)
    err = exc_info.value
    assert "ssp370" in str(err)
    assert err.package_label == "projection_1"


def test_unregistered_scenario_allowed_in_legacy_mode(tmp_path: Path) -> None:
    """``mode=LEGACY`` honors the external-source path: an
    unregistered scenario is accepted at the validator level."""
    base, proj = _build_pair_fixture(tmp_path)
    proj_manifest = json.loads(
        (proj / "manifest.json").read_text(encoding="utf-8")
    )
    proj_manifest["scenario"]["rcp_or_ssp"] = "ssp370"
    proj_manifest["scenario"]["time_slice_start"] = 2030
    proj_manifest["scenario"]["time_slice_end"] = 2049
    proj_manifest["scenario"]["co2_ppm"] = 999.0
    (proj / "manifest.json").write_text(
        json.dumps(proj_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Legacy mode passes — ScenarioBlock Layer 2 skips silently for
    # unregistered scenarios; validator does not enforce ship-mode
    # registration check.
    result = validate_scenario_set(base, [proj], mode=ValidationMode.LEGACY)
    assert result is None


def test_unregistered_scenario_error_carries_structured_fields(
    tmp_path: Path,
) -> None:
    """``UnregisteredScenarioInShipModeError`` exposes the offending
    scenario + time_slice + package_label so callers (cockpit error
    rendering) get specific info, not a freeform message."""
    base, proj = _build_pair_fixture(tmp_path)
    proj_manifest = json.loads(
        (proj / "manifest.json").read_text(encoding="utf-8")
    )
    proj_manifest["scenario"]["rcp_or_ssp"] = "ssp126"
    proj_manifest["scenario"]["time_slice_start"] = 2050
    proj_manifest["scenario"]["time_slice_end"] = 2069
    proj_manifest["scenario"]["co2_ppm"] = 450.0
    (proj / "manifest.json").write_text(
        json.dumps(proj_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)
        assert False, "Expected UnregisteredScenarioInShipModeError"
    except UnregisteredScenarioInShipModeError as err:
        assert err.package_label == "projection_1"
        assert err.failing_field_path is not None
        # Message references the scenario + period
        msg = str(err)
        assert "ssp126" in msg
        assert "2050" in msg or "(2050, 2069)" in msg


# ── §5.10-§5.11 CLI exit codes ───────────────────────────────────────


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Subprocess-invoke the CLI module per behavior-bound test
    discipline (durable §5)."""
    return subprocess.run(
        [sys.executable, "-m", "prismpy.validators.scenario_set", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_exits_0_on_pass(tmp_path: Path) -> None:
    base, proj = _build_pair_fixture(tmp_path)
    result = _run_cli(str(base), str(proj))
    assert result.returncode == 0
    assert "PASS" in result.stdout


def test_cli_exits_1_on_pairing_failure(tmp_path: Path) -> None:
    base, proj = _build_pair_fixture(
        tmp_path, projection_baseline_ref="WRONG"
    )
    result = _run_cli(str(base), str(proj))
    assert result.returncode == 1
    assert "FAIL" in result.stderr
    assert "baseline_reference_label" in result.stderr


def test_cli_exits_1_on_bias_correction_conflict(tmp_path: Path) -> None:
    base, proj1 = _build_pair_fixture(
        tmp_path, projection_method=BiasCorrectionMethod.QUANTILE_MAPPING
    )
    proj2 = tmp_path / "projection_2"
    proj_kwargs = _projection_block_kwargs(
        label="niamey-millet-projection-2",
        method=BiasCorrectionMethod.DELTA_METHOD,
    )
    _write_manifest(proj2, proj_kwargs)
    _write_cell_summary(proj2)
    _write_identity_files(proj2)
    _write_weather_differ(proj2, marker="projection_2")

    result = _run_cli(str(base), str(proj1), str(proj2))
    assert result.returncode == 1
    assert "FAIL" in result.stderr
    assert "Bias-correction" in result.stderr


def test_cli_default_mode_is_ship(tmp_path: Path) -> None:
    """Omit ``--mode`` and assert ship-mode invariants apply (F-G-3
    fires on 'unknown'). Per §5.12."""
    base, proj = _build_pair_fixture(
        tmp_path, projection_method=BiasCorrectionMethod.UNKNOWN
    )
    result = _run_cli(str(base), str(proj))
    assert result.returncode == 1
    assert "F-G-3" in result.stderr or "ship mode" in result.stderr


def test_cli_legacy_mode_allows_unknown(tmp_path: Path) -> None:
    """``--mode=legacy`` permits projection method='unknown'."""
    base, proj = _build_pair_fixture(
        tmp_path, projection_method=BiasCorrectionMethod.UNKNOWN
    )
    result = _run_cli("--mode=legacy", str(base), str(proj))
    assert result.returncode == 0
    assert "PASS" in result.stdout


def test_cli_help_runs() -> None:
    """``--help`` returns 0 and prints usage."""
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "scenario" in result.stdout.lower()


# ── Manifest-shape errors surface as ScenarioSetValidationError ──────


def test_missing_manifest_fires_typed_error(tmp_path: Path) -> None:
    base = tmp_path / "no_manifest"
    base.mkdir()
    proj = tmp_path / "no_manifest_proj"
    proj.mkdir()
    with pytest.raises(ScenarioSetValidationError) as exc_info:
        validate_scenario_set(base, [proj])
    assert exc_info.value.failing_field_path == "manifest.json"
    assert exc_info.value.actual == "missing"


def test_missing_scenario_block_fires_typed_error(tmp_path: Path) -> None:
    """A manifest without a scenario block is rejected by the
    validator (the validator scope IS scenario sets)."""
    base = tmp_path / "no_scenario"
    base.mkdir()
    (base / "manifest.json").write_text(
        json.dumps({"package_version": "1.0"}), encoding="utf-8"
    )
    proj = tmp_path / "no_scenario_proj"
    proj.mkdir()
    (proj / "manifest.json").write_text(
        json.dumps({"package_version": "1.0"}), encoding="utf-8"
    )
    with pytest.raises(ScenarioSetValidationError) as exc_info:
        validate_scenario_set(base, [proj])
    assert exc_info.value.failing_field_path == "manifest.scenario"


def test_corrupt_scenario_block_fires_typed_error(tmp_path: Path) -> None:
    """A scenario block with an invalid field surfaces ValidationError
    wrapped in ScenarioSetValidationError."""
    base, proj = _build_pair_fixture(tmp_path)
    # Tamper: bias_correction_method to an out-of-domain string.
    raw = json.loads((proj / "manifest.json").read_text(encoding="utf-8"))
    raw["scenario"]["bias_correction_method"] = "cubic_spline_invented"
    (proj / "manifest.json").write_text(
        json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ScenarioSetValidationError) as exc_info:
        validate_scenario_set(base, [proj])
    assert exc_info.value.failing_field_path == "manifest.scenario"
    assert "ValidationError" in str(exc_info.value.actual)
