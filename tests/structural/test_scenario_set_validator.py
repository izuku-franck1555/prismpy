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
from tempfile import TemporaryDirectory
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
    WeatherSchemaAsymmetricWithoutLimitationError,
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


def _write_wth_with_columns(
    package_dir: Path,
    *,
    columns: int,
    marker: str = "synthetic",
    extension: str = "wth",
) -> None:
    """Write a synthetic DSSAT-style weather file with a specified
    daily-data column count for F-G-8 weather-schema-asymmetry drills.

    The fixture mirrors a real DSSAT/PYTHIA WTH layout:

    * ``$WEATHER DATA`` metadata header
    * Station-info ``@`` header + station-info data row (8 tokens —
      different column count from the climate data so the validator
      must skip past it)
    * Blank separator
    * Daily-data ``@`` header containing ``DATE`` + ``TMAX`` (the
      validator's data-header marker)
    * Daily data row with ``columns``-many whitespace-separated tokens

    The ``extension`` parameter exercises both the DSSAT path
    (``.wth`` / ``.WTH``) and the CRAFT path (``cell_<n>.txt`` style;
    CRAFT files have no ``@`` header — daily data lines start with a
    YRDOY 7-digit token).

    Codex round 1 boundary 6/7 P1-2 absorption: this fixture was
    previously a single ``@ <header>\\n<data>`` row, which the naive
    validator counted correctly; the realistic DSSAT layout with
    metadata + station-info exposed the gap that codex flagged.
    """
    (package_dir / "weather").mkdir(parents=True, exist_ok=True)
    # Remove any stale 1-col marker file from _write_weather_differ
    marker_file = package_dir / "weather" / "data.wth"
    if marker_file.exists():
        marker_file.unlink()
    header_tokens = [
        "DATE",
        "SRAD",
        "TMAX",
        "TMIN",
        "RAIN",
        "TDEW",
        "WIND",
        "RHUM",
        "ETO",
        "EXTRA",
    ][:columns]
    data_tokens = ["2046001"] + [f"{1.0 + i:.1f}" for i in range(columns - 1)]

    if extension.lower() in ("wth", "WTH"):
        filename = f"data.{extension}"
        body = (
            f"$WEATHER DATA: Generated by prismpy synthetic fixture {marker}\n"
            + "\n"
            + "@ INSI       LAT      LONG    ELEV   TAV   AMP REFHT WNDHT\n"
            + "  S001  13.5  2.1  0.0  25.0  10.0  2.0  2.0\n"
            + "\n"
            + "@ {}\n".format(" ".join(header_tokens))
            + " ".join(data_tokens)
            + f"\n# marker={marker}\n"
        )
    else:
        # CRAFT-style cell_*.txt — no @ header, tab-delimited daily
        # rows starting with the YRDOY 7-digit token.
        filename = f"cell_001.{extension}"
        body = (
            f"# CRAFT weather data — synthetic fixture {marker}\n"
            + "\t".join(data_tokens)
            + "\n"
        )
    (package_dir / "weather" / filename).write_text(body, encoding="utf-8")


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
        "ISIMIP3BASD v2.5.0 quantile-mapping against W5E5 v2.0"
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


# ── AC-G-12 drill #5 — CO₂ canonical mismatch via validator pipeline ─


def test_drill_5_co2_ppm_mismatch_via_validator_pipeline(
    tmp_path: Path,
) -> None:
    """AC-G-12 drill #5: when a projection's manifest carries a
    co2_ppm value that disagrees with the canonical lookup for
    its (rcp_or_ssp, time_slice) tuple, ``validate_scenario_set``
    surfaces the AC-G-9 Layer 2 ``CO2ProvenanceMismatchError``
    wrapped in ``ScenarioSetValidationError`` (because
    ``_validate_scenario_block`` re-validates the manifest payload
    via ``ScenarioBlock.model_validate`` after the disk read)."""
    base, proj = _build_pair_fixture(tmp_path)
    proj_manifest = json.loads(
        (proj / "manifest.json").read_text(encoding="utf-8")
    )
    proj_manifest["scenario"]["co2_ppm"] = 999.0  # canonical SSP585 (2046, 2065) is 571.0
    (proj / "manifest.json").write_text(
        json.dumps(proj_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ScenarioSetValidationError) as exc_info:
        validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)
    msg = str(exc_info.value)
    # Either the AC-G-9 Layer 2 wrapped message or the canonical key
    assert "canonical" in msg.lower() or "Layer 2" in msg or "999" in msg


def test_drill_5_co2_ppm_provenance_paraphrase_via_validator_pipeline(
    tmp_path: Path,
) -> None:
    """Companion drill: paraphrased provenance string (non-empty but
    not the canonical citation) also surfaces via the validator
    pipeline."""
    base, proj = _build_pair_fixture(tmp_path)
    proj_manifest = json.loads(
        (proj / "manifest.json").read_text(encoding="utf-8")
    )
    proj_manifest["scenario"]["co2_ppm_provenance"] = (
        "AR6 mid-period (paraphrased)"
    )
    (proj / "manifest.json").write_text(
        json.dumps(proj_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ScenarioSetValidationError):
        validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)


# ── AC-G-12 drill #6 — F-G-8 weather schema asymmetric without limitation ─


def test_drill_6_wth_asymmetric_without_limitation_fires_f_g_8(
    tmp_path: Path,
) -> None:
    """AC-G-12 drill #6 + F-G-8: 5-col WTH baseline + 8-col WTH
    projection MUST declare ``manifest.limitations.weather_schema_
    asymmetric_within_set``. Without the declaration, the validator
    fires ``WeatherSchemaAsymmetricWithoutLimitationError``."""
    base, proj = _build_pair_fixture(tmp_path)
    # Replace the trivial 1-col weather marker with structured WTH files
    _write_wth_with_columns(base, columns=5, marker="baseline-5-col")
    _write_wth_with_columns(proj, columns=8, marker="projection-8-col")

    with pytest.raises(
        WeatherSchemaAsymmetricWithoutLimitationError
    ) as exc_info:
        validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)
    err = exc_info.value
    assert err.package_label == "projection_1"
    msg = str(err)
    assert "5" in msg and "8" in msg


def test_drill_6_wth_asymmetric_error_carries_column_counts(
    tmp_path: Path,
) -> None:
    """``WeatherSchemaAsymmetricWithoutLimitationError`` exposes the
    baseline + projection column counts in the structured fields so
    the cockpit error rendering can show the mismatch quantitatively."""
    base, proj = _build_pair_fixture(tmp_path)
    _write_wth_with_columns(base, columns=5, marker="baseline")
    _write_wth_with_columns(proj, columns=8, marker="projection")

    try:
        validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)
        assert False, "Expected F-G-8 fire"
    except WeatherSchemaAsymmetricWithoutLimitationError as err:
        assert err.actual is not None
        assert "5-col" in err.actual
        assert "8-col" in err.actual
        assert (
            err.failing_field_path
            == "manifest.limitations.weather_schema_asymmetric_within_set"
        )


# ── AC-G-12 drill #6a — F-G-8 positive companion (limitation declared) ─


def test_drill_6a_wth_asymmetric_with_limitation_passes(
    tmp_path: Path,
) -> None:
    """AC-G-12 drill #6a positive companion: 5-col WTH baseline +
    8-col WTH projection WITH ``manifest.limitations.weather_schema_
    asymmetric_within_set`` populated → validator passes. Drill #6
    tests the fire path; #6a tests the pass path. Together they
    fully calibrate F-G-8 detection."""
    base, proj = _build_pair_fixture(tmp_path)
    _write_wth_with_columns(base, columns=5, marker="baseline-5-col")
    _write_wth_with_columns(proj, columns=8, marker="projection-8-col")

    # Populate the manifest.limitations declaration on the projection
    proj_manifest = json.loads(
        (proj / "manifest.json").read_text(encoding="utf-8")
    )
    proj_manifest["limitations"] = {
        "weather_schema_asymmetric_within_set": (
            "baseline ships 5-col WTH per existing observed-climate "
            "translators; projection ships 8-col WTH per AC-G-7"
        )
    }
    (proj / "manifest.json").write_text(
        json.dumps(proj_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Validator passes — asymmetry declared, no F-G-8 fire
    result = validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)
    assert result is None


def test_drill_6_walker_skips_when_no_wth_files(tmp_path: Path) -> None:
    """When neither baseline nor projection ships any WTH file, the
    F-G-8 check is out of scope and does not fire (the trivial
    ``data.wth`` marker file written by ``_build_pair_fixture`` is
    1-column on both sides → symmetric → no F-G-8)."""
    base, proj = _build_pair_fixture(tmp_path)
    # No mutation — both have 1-col data.wth from fixture default.
    # Validator passes cleanly.
    result = validate_scenario_set(base, [proj])
    assert result is None


def test_drill_6_walker_skips_when_symmetric_columns(tmp_path: Path) -> None:
    """8-col WTH on BOTH sides → symmetric → no F-G-8 fire even
    without the limitation declaration."""
    base, proj = _build_pair_fixture(tmp_path)
    _write_wth_with_columns(base, columns=8, marker="baseline-8-col")
    _write_wth_with_columns(proj, columns=8, marker="projection-8-col")

    # No limitations declared, but asymmetry doesn't exist.
    result = validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)
    assert result is None


# ── Codex round 1 boundary 6/7 P1 absorptions ────────────────────────


def test_drill_6_p1_2_skips_metadata_and_station_info_rows(
    tmp_path: Path,
) -> None:
    """Codex round 1 boundary 6/7 P1-2: real DSSAT WTH files have
    metadata ($WEATHER DATA) and station-info rows (8 tokens — INSI
    LAT LONG ELEV TAV AMP REFHT WNDHT) BEFORE the daily-data ``@ DATE``
    header. The validator MUST skip past those and count the daily
    data row, not the station-info row.

    Drill: 5-col baseline vs 8-col projection, both wrapped in the
    full DSSAT layout with metadata + 8-token station-info row.
    Without the P1-2 fix, the validator would return 8 for both
    sides (the station-info width) and miss the asymmetry."""
    base, proj = _build_pair_fixture(tmp_path)
    _write_wth_with_columns(base, columns=5, marker="baseline-real-dssat")
    _write_wth_with_columns(proj, columns=8, marker="projection-real-dssat")

    with pytest.raises(WeatherSchemaAsymmetricWithoutLimitationError):
        validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)


def test_drill_6_p1_1_detects_craft_cell_txt_files(tmp_path: Path) -> None:
    """Codex round 1 boundary 6/7 P1-1: CRAFT weather files are
    ``cell_*.txt`` (tab-delimited, no @ header). The validator must
    include them in the F-G-8 scan or a CRAFT 5-col baseline +
    8-col projection ships without the limitation declaration.

    Drill: 5-col baseline + 8-col projection, both shipped as
    ``cell_001.txt`` (CRAFT path), no manifest.limitations
    declaration → F-G-8 fires."""
    base, proj = _build_pair_fixture(tmp_path)
    _write_wth_with_columns(
        base, columns=5, marker="baseline-craft", extension="txt"
    )
    _write_wth_with_columns(
        proj, columns=8, marker="projection-craft", extension="txt"
    )

    with pytest.raises(WeatherSchemaAsymmetricWithoutLimitationError):
        validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)


def test_drill_6_p1_1_craft_with_declaration_passes(tmp_path: Path) -> None:
    """Companion to P1-1: CRAFT 5-col baseline + 8-col projection
    WITH the manifest.limitations declaration validates cleanly."""
    base, proj = _build_pair_fixture(tmp_path)
    _write_wth_with_columns(
        base, columns=5, marker="baseline-craft", extension="txt"
    )
    _write_wth_with_columns(
        proj, columns=8, marker="projection-craft", extension="txt"
    )
    proj_manifest = json.loads(
        (proj / "manifest.json").read_text(encoding="utf-8")
    )
    proj_manifest["limitations"] = {
        "weather_schema_asymmetric_within_set": (
            "CRAFT baseline 5-col observed weather; projection 8-col"
        )
    }
    (proj / "manifest.json").write_text(
        json.dumps(proj_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = validate_scenario_set(base, [proj], mode=ValidationMode.SHIP)
    assert result is None


def test_p1_1_unit_count_reads_dssat_data_row_past_metadata() -> None:
    """Direct unit test on ``_wth_column_count``: a DSSAT WTH text
    blob with metadata + station-info + daily-data sections returns
    the daily-data column count (5 in this case), not the station-info
    width (8)."""
    from prismpy.validators.scenario_set import _wth_column_count

    with TemporaryDirectory() as td:
        weather_dir = Path(td) / "weather"
        weather_dir.mkdir()
        (weather_dir / "data.wth").write_text(
            "$WEATHER DATA: Generated by prismpy\n"
            "\n"
            "@ INSI       LAT      LONG    ELEV   TAV   AMP REFHT WNDHT\n"
            "  S001  13.5  2.1  0.0  25.0  10.0  2.0  2.0\n"
            "\n"
            "@ DATE  SRAD  TMAX  TMIN  RAIN\n"
            "2046001 20.0 30.0 18.0 0.0\n",
            encoding="utf-8",
        )
        assert _wth_column_count(weather_dir) == 5


def test_p1_1_unit_count_reads_craft_cell_txt() -> None:
    """Direct unit test on ``_wth_column_count``: CRAFT
    ``cell_001.txt`` tab-delimited file (no @ header) returns the
    column count of the YRDOY-prefixed data row."""
    from prismpy.validators.scenario_set import _wth_column_count

    with TemporaryDirectory() as td:
        weather_dir = Path(td) / "weather"
        weather_dir.mkdir()
        (weather_dir / "cell_001.txt").write_text(
            "# CRAFT weather data\n"
            "2046001\t20.0\t30.0\t18.0\t0.0\n",
            encoding="utf-8",
        )
        assert _wth_column_count(weather_dir) == 5


def test_p1_1_unit_count_returns_none_when_only_metadata() -> None:
    """If the file only has metadata + station-info rows (no daily-
    data ``@ DATE TMAX`` header AND no YRDOY data row), the count
    returns None — the file is out of scope."""
    from prismpy.validators.scenario_set import _wth_column_count

    with TemporaryDirectory() as td:
        weather_dir = Path(td) / "weather"
        weather_dir.mkdir()
        (weather_dir / "data.wth").write_text(
            "$WEATHER DATA: only metadata, no daily data\n"
            "@ INSI LAT LONG\n"
            "S001 13.5 2.1\n",
            encoding="utf-8",
        )
        assert _wth_column_count(weather_dir) is None


# ── AC-G-12 fixture inventory pin ────────────────────────────────────


def test_ac_g_12_drill_inventory_complete() -> None:
    """Pin inventory of named drills per AC-G-12 contract §12. A
    refactor that drops a drill must update this list explicitly,
    making the omission impossible to miss in code review."""
    expected_drills = {
        # Drill #1 — happy path positive
        "test_happy_path_passes_in_ship_mode",
        "test_happy_path_passes_in_legacy_mode",
        "test_happy_path_2_projections_same_method",
        # Drills #2-#4 — identity coupling
        "test_drill_cell_id_mutated_fires_identity_drift",
        "test_drill_lat_lon_drift_fires_identity_drift",
        "test_drill_identity_file_sha_mutation_fires",
        "test_drill_identity_file_added_fires_identity_drift",
        # Pairing rule
        "test_drill_pairing_rule_wrong_baseline_reference",
        # Bias-correction conflict
        "test_drill_bias_correction_conflict_fires",
        # Drill #5 — CO₂ canonical mismatch
        "test_drill_5_co2_ppm_mismatch_via_validator_pipeline",
        "test_drill_5_co2_ppm_provenance_paraphrase_via_validator_pipeline",
        # Drill #6 + #6a — F-G-8 weather schema asymmetric
        "test_drill_6_wth_asymmetric_without_limitation_fires_f_g_8",
        "test_drill_6_wth_asymmetric_error_carries_column_counts",
        "test_drill_6a_wth_asymmetric_with_limitation_passes",
        # Drill #7 + #7a — mode disambiguation
        "test_drill_7_unknown_method_rejected_in_ship_mode",
        "test_drill_7a_unknown_method_allowed_in_legacy_mode",
    }
    # Spot-check by name presence in this module
    import sys

    this_module = sys.modules[__name__]
    actual_tests = {
        name
        for name in dir(this_module)
        if name.startswith("test_") and callable(getattr(this_module, name))
    }
    missing = expected_drills - actual_tests
    assert not missing, (
        f"AC-G-12 drill inventory incomplete. Missing tests: {missing}. "
        "Per Sprint G Draft 5 §AC-G-12 + warning-auditor pass-2/3 "
        "MEDIUM-Rebase-4/5 + MEDIUM-Pass3-1: every named drill MUST "
        "have a calibrating test."
    )


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
