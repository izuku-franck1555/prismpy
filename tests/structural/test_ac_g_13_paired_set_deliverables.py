"""Structural pin: AC-G-13 — 4 paired-set deliverables + SHA-256 hashes.

Sprint G AC-G-13 ships four synthetic baseline+projection paired
sets. Each set's ``manifest.json`` SHA-256 is pinned here so a
later edit that changes the manifest format (without intentionally
re-baselining the hashes) fires loud.

Per CC-G-7 deterministic generation contract:

* No ``datetime.now()``, no random, no build-host paths, no
  build-time stamps in the manifest. The Sprint G manifest writer
  rewrite (boundary 1/7) removed these surfaces.
* Same input → same bytes. The SHA-256 pin asserts this directly.

Per Sprint G Draft 5 spec §AC-G-13:

* Set 1 — Niamey Millet (smallest, highest priority for fast iteration)
  observed 2017-2021 baseline + GFDL-ESM4 SSP585 2046-2065 projection.
* Set 2 — Madarounfa Maize: observed 2018-2023 baseline + GFDL-ESM4
  SSP245 2046-2065 projection.
* Set 3 — Menoua Groundnut: observed 2018-2022 baseline + GFDL-ESM4
  SSP585 2046-2065 projection.
* Set 4 — free choice: any region/crop with ≥3 baseline years; any
  GCM × SSP × time-slice.

Each baseline's manifest carries
``manifest.limitations.weather_schema_asymmetric_within_set`` with the
general-form value field per warning-auditor pass-2 MEDIUM-Rebase-1.

Note: Sprint G is a STRUCTURAL-PIN sprint. The 4 paired sets are
SYNTHETIC fixtures whose SHA hashes serve as deterministic-output
pins. Real-data ISIMIP3b retrieval + WTH writer execution land in
Sprint H+ (when the actual end-to-end pipeline runs against live
ISIMIP3b cutouts).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from prismpy.models.scenario import (
    BiasCorrectionMethod,
    ScenarioBlock,
    ScenarioRole,
)
from prismpy.packaging.manifest import create_manifest, save_manifest
from prismpy.validators.scenario_set import (
    ValidationMode,
    validate_scenario_set,
)


# ── 4-set deliverable specifications ─────────────────────────────────


_AR6_PROVENANCE = "AR6 WG1 Annex III + RCMIP, mid-year-of-period convention"
_OBSERVED_PROVENANCE = (
    "AR6 WG1 Annex III observed atmospheric record"
)
_BIAS_CORRECTION_PROVENANCE = (
    "ISIMIP3BASD v2.5.0 quantile-mapping against W5E5 v2.0"
)
_ASYMMETRY_LIMITATION = (
    "baseline ships 5-col WTH per existing observed-climate "
    "translators; projection ships 8-col WTH per AC-G-7"
)


# Each set is a tuple:
#   (set_id, baseline_label, projection_label, baseline_kwargs_extras,
#    projection_kwargs_extras)
#
# Per Draft 5 spec §AC-G-13 the 4 sets cover SSP245 + SSP585 with
# different (region, crop, baseline-period, time-slice) tuples.
_PAIRED_SETS: Tuple[Tuple[str, str, str, Dict[str, Any], Dict[str, Any]], ...] = (
    # Set 1 — Niamey Millet (highest priority)
    (
        "set1_niamey_millet",
        "niamey-millet-baseline-2017-2021",
        "niamey-millet-projection-gfdl-ssp585-2046-2065",
        {
            "rcp_or_ssp": "observed",
            "time_slice_start": 2017,
            "time_slice_end": 2021,
            "co2_ppm": 410.0,
            "co2_ppm_provenance": _OBSERVED_PROVENANCE,
        },
        {
            "gcm_source": "gfdl-esm4",
            "rcp_or_ssp": "ssp585",
            "time_slice_start": 2046,
            "time_slice_end": 2065,
            "co2_ppm": 571.0,
            "co2_ppm_provenance": _AR6_PROVENANCE,
        },
    ),
    # Set 2 — Madarounfa Maize
    (
        "set2_madarounfa_maize",
        "madarounfa-maize-baseline-2018-2023",
        "madarounfa-maize-projection-gfdl-ssp245-2046-2065",
        {
            "rcp_or_ssp": "observed",
            "time_slice_start": 2018,
            "time_slice_end": 2023,
            "co2_ppm": 415.0,
            "co2_ppm_provenance": _OBSERVED_PROVENANCE,
        },
        {
            "gcm_source": "gfdl-esm4",
            "rcp_or_ssp": "ssp245",
            "time_slice_start": 2046,
            "time_slice_end": 2065,
            "co2_ppm": 478.0,
            "co2_ppm_provenance": _AR6_PROVENANCE,
        },
    ),
    # Set 3 — Menoua Groundnut
    (
        "set3_menoua_groundnut",
        "menoua-groundnut-baseline-2018-2022",
        "menoua-groundnut-projection-gfdl-ssp585-2046-2065",
        {
            "rcp_or_ssp": "observed",
            "time_slice_start": 2018,
            "time_slice_end": 2022,
            "co2_ppm": 412.0,
            "co2_ppm_provenance": _OBSERVED_PROVENANCE,
        },
        {
            "gcm_source": "gfdl-esm4",
            "rcp_or_ssp": "ssp585",
            "time_slice_start": 2046,
            "time_slice_end": 2065,
            "co2_ppm": 571.0,
            "co2_ppm_provenance": _AR6_PROVENANCE,
        },
    ),
    # Set 4 — free choice: Bénoué Sorghum + IPSL-CM6A-LR SSP585 end-of-century
    (
        "set4_benoue_sorghum_ipsl",
        "benoue-sorghum-baseline-2019-2023",
        "benoue-sorghum-projection-ipsl-ssp585-2086-2100",
        {
            "rcp_or_ssp": "observed",
            "time_slice_start": 2019,
            "time_slice_end": 2023,
            "co2_ppm": 418.0,
            "co2_ppm_provenance": _OBSERVED_PROVENANCE,
        },
        {
            "gcm_source": "ipsl-cm6a-lr",
            "rcp_or_ssp": "ssp585",
            "time_slice_start": 2086,
            "time_slice_end": 2100,
            "co2_ppm": 1054.0,
            "co2_ppm_provenance": _AR6_PROVENANCE,
        },
    ),
)


# ── Pinned SHA-256 hashes (CC-G-7 deterministic-output pin) ──────────


# These hashes are computed from the synthetic fixture builder below.
# A change to the manifest format OR the fixture content invalidates
# the pin and requires deliberate re-baselining (analogous to the
# trajectory-pin re-anchor pattern). The pins guard against silent
# format drift in the deterministic output.
#
# Re-baselining: when an intentional manifest-format change ships,
# regenerate these constants by running the test once with
# ``EXPECTED_MANIFEST_SHAS`` cleared, capturing the printed
# ``actual_shas`` output, and updating the dict here. The
# pin-mismatch error message names the offending set + side so the
# diff is unambiguous.
EXPECTED_MANIFEST_SHAS: Dict[str, Tuple[str, str]] = {
    # Mapping: set_id → (baseline_sha, projection_sha)
    #
    # SHAs re-baselined post the upstream UC-emission addition (5 new
    # manifest fields: ``cells``, ``cell_areas``, ``crops`` widened,
    # ``use_case_config``, ``uc_readiness``). Diff-scope verified: only
    # the additive fields changed; all pre-PR3 fields (``platform``,
    # ``project_name``, ``region``, ``crop`` singleton, ``temporal``,
    # ``data_sources``, ``summary``, ``scenario``, ``files``,
    # ``limitations``, ``validation_status``) unchanged in shape and
    # value for each set.
    "set1_niamey_millet": (
        "ca0a5057244b06282013ed3684759d575b6a330495afd6aa18d07672ddf80edd",  # baseline
        "6b2acb6eac0ca12f5bd50c47ac6cc72b655ef22a5e7e68b0d93c33ff1641f79c",  # projection
    ),
    "set2_madarounfa_maize": (
        "d449604060eb3bb1e2ec49667b4cb89979a68705b6b0efe3a2e4c2edf0b6f26e",  # baseline
        "777070ab105bbfdc85317f3232f515d0a75f4f4fd8edd3708bac13433fc4e4c3",  # projection
    ),
    "set3_menoua_groundnut": (
        "3b1237af41d512a661f86ff472a1789ccec9585f1cab0ab1909e7e125fa3417c",  # baseline
        "4020895e2859e60b742e22816215a3a05439452a43b684e689f265c953d131b6",  # projection
    ),
    "set4_benoue_sorghum_ipsl": (
        "54b26677633c6165da7ff55efd70591f69a3d8d8a1888ca831f5c75ec095d556",  # baseline
        "50caa11ae6d7026db10521771d6aa023576455227978afb8add3fa8a3c40d6c8",  # projection
    ),
}


# ── Fixture builder ──────────────────────────────────────────────────


def _baseline_kwargs_for_set(
    label: str,
    extras: Dict[str, Any],
) -> Dict[str, Any]:
    """Build deterministic baseline ScenarioBlock kwargs for a
    deliverable set."""
    base: Dict[str, Any] = {
        "scenario_label": label,
        "scenario_role": ScenarioRole.BASE,
        "gcm_source": "observed",
        "rcp_or_ssp": "observed",
        "baseline_reference_label": label,
        "bias_correction_method": BiasCorrectionMethod.NONE,
        "scenario_bias_correction_provenance": None,
    }
    base.update(extras)
    return base


def _projection_kwargs_for_set(
    label: str,
    baseline_label: str,
    extras: Dict[str, Any],
) -> Dict[str, Any]:
    """Build deterministic projection ScenarioBlock kwargs."""
    proj: Dict[str, Any] = {
        "scenario_label": label,
        "scenario_role": ScenarioRole.PROJECTION,
        "baseline_reference_label": baseline_label,
        "bias_correction_method": BiasCorrectionMethod.QUANTILE_MAPPING,
        "scenario_bias_correction_provenance": _BIAS_CORRECTION_PROVENANCE,
    }
    proj.update(extras)
    return proj


def _write_set_files(
    package_dir: Path, *, role: str, time_slice_start: int
) -> None:
    """Write deterministic identity-file + cell_summary content for
    a paired-set deliverable. All bytes are fully determined by the
    code so two runs produce byte-identical content.

    The weather YRDOY row is derived from ``time_slice_start`` so a
    set's weather data matches its advertised time slice (codex round
    1 boundary 7/7 P2-3 absorption — without this, set4's
    2086-2100 projection would carry 2046 weather data and the
    fixture would contradict the manifest's scenario metadata)."""
    package_dir.mkdir(parents=True, exist_ok=True)

    # cell_summary.json — 2 cells, deterministic
    cells = {
        "cells": [
            {"id": 1, "lat": 13.5, "lon": 2.1},
            {"id": 2, "lat": 13.6, "lon": 2.2},
        ]
    }
    (package_dir / "cell_summary.json").write_text(
        json.dumps(cells, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Identity-coupled files — same bytes baseline + projection
    (package_dir / "crop_mask").mkdir(exist_ok=True)
    (package_dir / "crop_mask" / "mask.txt").write_text(
        "0,0,1\n0,1,1\n1,1,1\n", encoding="utf-8"
    )
    (package_dir / "soil").mkdir(exist_ok=True)
    (package_dir / "soil" / "soil_mask.txt").write_text(
        "1,1,1\n1,1,1\n1,1,1\n", encoding="utf-8"
    )
    (package_dir / "soil" / "NER.SOL").write_text(
        "*SOIL profile NER\n@profile-line-1\n", encoding="utf-8"
    )
    (package_dir / "management").mkdir(exist_ok=True)
    (package_dir / "management" / "calendar.txt").write_text(
        "planting=152\nmaturity=273\n", encoding="utf-8"
    )

    # Weather files — 5-col baseline, 8-col projection per AC-G-7.
    # YRDOY row uses the set's time_slice_start so the fixture's
    # weather data matches the advertised time slice.
    (package_dir / "weather").mkdir(exist_ok=True)
    yrdoy = f"{time_slice_start}001"  # day 1 of time_slice_start
    if role == "baseline":
        wth_body = (
            "$WEATHER DATA: AR6 WG1 Annex III observed historical record\n"
            + "\n"
            + "@ INSI       LAT      LONG    ELEV   TAV   AMP REFHT WNDHT\n"
            + "  S001  13.5  2.1  0.0  25.0  10.0  2.0  2.0\n"
            + "\n"
            + "@ DATE  SRAD  TMAX  TMIN  RAIN\n"
            + f"{yrdoy} 20.0 30.0 18.0 0.0\n"
        )
    else:
        wth_body = (
            "$WEATHER DATA: ISIMIP3b GCM bias-corrected projection\n"
            + "\n"
            + "@ INSI       LAT      LONG    ELEV   TAV   AMP REFHT WNDHT\n"
            + "  S001  13.5  2.1  0.0  25.0  10.0  2.0  2.0\n"
            + "\n"
            + "@ DATE  SRAD  TMAX  TMIN  TDEW  RAIN  WIND  RHUM\n"
            + f"{yrdoy} 20.0 30.0 18.0 12.0 0.0 2.5 65.0\n"
        )
    (package_dir / "weather" / "data.wth").write_text(
        wth_body, encoding="utf-8"
    )


def _config_for_set(
    set_id: str,
    baseline_label: str,
    extras: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the flat-keyed config dict that ``create_manifest`` reads.

    Codex round 1 boundary 7/7 P2-2 absorption: ``create_manifest``
    expects flat keys (``project_name``, ``region_name``, ``crop_name``,
    ``start_year``, ``end_year``, ``data_sources``), NOT nested
    ``project`` / ``region`` keys. Without this fix every fixture
    manifest carries placeholder ``"unknown"`` / ``""`` values and
    the per-set distinctions are not actually pinned by the SHA hash.
    """
    # Region + crop derived from the set_id pattern
    # ``setN_<region>_<crop>[_<extra>]`` for human-readable distinctness
    parts = set_id.split("_", 2)
    if len(parts) >= 3:
        region_name = parts[1].capitalize()
        crop_name = parts[2].split("_")[0].capitalize()
    else:
        region_name = parts[1].capitalize() if len(parts) > 1 else ""
        crop_name = ""
    return {
        "project_name": set_id,
        "region_name": region_name,
        "country": "Niger" if "niamey" in set_id or "madarounfa" in set_id
        else "Cameroon",
        "crop_name": crop_name,
        "planting_doy": 152,
        "maturity_doy": 273,
        "start_year": extras["time_slice_start"],
        "end_year": extras["time_slice_end"],
        "spinup_years": 0,
        "data_sources": {
            "climate": "ISIMIP3b" if extras.get("rcp_or_ssp") != "observed"
            else "AR6 observed historical",
            "soil": "synthetic-NER",
            "crop_mask": "SPAM 2020 (synthetic)",
            "boundaries": "GADM v3.6",
        },
    }


def _build_paired_set_on_disk(
    set_id: str,
    baseline_label: str,
    projection_label: str,
    baseline_extras: Dict[str, Any],
    projection_extras: Dict[str, Any],
    root: Path,
) -> Tuple[Path, Path]:
    """Build a complete paired-set deliverable on disk. Returns the
    (baseline_path, projection_path) tuple."""
    base_path = root / set_id / "baseline"
    proj_path = root / set_id / "projection"
    _write_set_files(
        base_path,
        role="baseline",
        time_slice_start=baseline_extras["time_slice_start"],
    )
    _write_set_files(
        proj_path,
        role="projection",
        time_slice_start=projection_extras["time_slice_start"],
    )

    base_block = ScenarioBlock(
        **_baseline_kwargs_for_set(baseline_label, baseline_extras)
    )
    proj_block = ScenarioBlock(
        **_projection_kwargs_for_set(
            projection_label, baseline_label, projection_extras
        )
    )

    base_config = _config_for_set(set_id, baseline_label, baseline_extras)
    proj_config = _config_for_set(set_id, baseline_label, projection_extras)

    base_manifest = create_manifest(
        base_path, base_config, platform="synthetic", scenario=base_block
    )
    # Per Draft 5 §AC-G-13: each baseline declares the limitation
    base_manifest["limitations"] = {
        "weather_schema_asymmetric_within_set": _ASYMMETRY_LIMITATION,
    }
    save_manifest(base_manifest, base_path / "manifest.json")

    proj_manifest = create_manifest(
        proj_path, proj_config, platform="synthetic", scenario=proj_block
    )
    proj_manifest["limitations"] = {
        "weather_schema_asymmetric_within_set": _ASYMMETRY_LIMITATION,
    }
    save_manifest(proj_manifest, proj_path / "manifest.json")

    return base_path, proj_path


def _sha256_of_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── §1 Each of 4 sets builds + validates clean ──────────────────────


@pytest.mark.parametrize("spec", _PAIRED_SETS, ids=lambda s: s[0])
def test_paired_set_validates_clean_in_ship_mode(
    spec: Tuple[str, str, str, Dict[str, Any], Dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Each of the 4 deliverable sets passes
    ``validate_scenario_set(mode=SHIP)`` cleanly. Asymmetry
    declaration on both sides; identity files match; bias-correction
    method is QUANTILE_MAPPING (not 'unknown'); CO₂ canonical or
    observed."""
    set_id, base_label, proj_label, base_extras, proj_extras = spec
    base_path, proj_path = _build_paired_set_on_disk(
        set_id, base_label, proj_label, base_extras, proj_extras, tmp_path
    )
    # Validator passes
    result = validate_scenario_set(
        base_path, [proj_path], mode=ValidationMode.SHIP
    )
    assert result is None


@pytest.mark.parametrize("spec", _PAIRED_SETS, ids=lambda s: s[0])
def test_paired_set_baseline_declares_asymmetry_limitation(
    spec: Tuple[str, str, str, Dict[str, Any], Dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Each baseline's ``manifest.limitations.weather_schema_asymmetric_
    within_set`` declares the asymmetry per Sprint G Draft 5 spec §AC-G-13.
    Honest-signal contract per ``feedback_no_data_cooking.md``."""
    set_id, base_label, proj_label, base_extras, proj_extras = spec
    base_path, _ = _build_paired_set_on_disk(
        set_id, base_label, proj_label, base_extras, proj_extras, tmp_path
    )
    manifest = json.loads(
        (base_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert "limitations" in manifest
    assert (
        "weather_schema_asymmetric_within_set" in manifest["limitations"]
    )
    assert manifest["limitations"][
        "weather_schema_asymmetric_within_set"
    ] == _ASYMMETRY_LIMITATION


# ── §2 Determinism — same fixture builds → identical SHA ─────────────


@pytest.mark.parametrize("spec", _PAIRED_SETS, ids=lambda s: s[0])
def test_paired_set_manifest_is_byte_identical_across_two_runs(
    spec: Tuple[str, str, str, Dict[str, Any], Dict[str, Any]],
    tmp_path: Path,
) -> None:
    """CC-G-7 deterministic-output pin: building the same set twice
    yields byte-identical ``manifest.json`` files. Without this, the
    SHA pins below would be flaky."""
    set_id, base_label, proj_label, base_extras, proj_extras = spec
    root_a = tmp_path / "run_a"
    root_b = tmp_path / "run_b"
    root_a.mkdir()
    root_b.mkdir()

    base_a, proj_a = _build_paired_set_on_disk(
        set_id, base_label, proj_label, base_extras, proj_extras, root_a
    )
    base_b, proj_b = _build_paired_set_on_disk(
        set_id, base_label, proj_label, base_extras, proj_extras, root_b
    )

    # Compare manifest bytes
    assert (base_a / "manifest.json").read_bytes() == (
        base_b / "manifest.json"
    ).read_bytes()
    assert (proj_a / "manifest.json").read_bytes() == (
        proj_b / "manifest.json"
    ).read_bytes()


def test_all_4_sets_have_distinct_manifest_shas(tmp_path: Path) -> None:
    """The 4 sets are deliberately distinct (different region / crop /
    time-slice / scenario). Each baseline + projection manifest hash
    is unique across the 8 outputs. Catches a refactor that
    accidentally collapses set identity (e.g., a ``copy`` shadowing
    the per-set parameter)."""
    seen_shas: Dict[str, str] = {}
    for set_id, base_label, proj_label, base_extras, proj_extras in _PAIRED_SETS:
        base_path, proj_path = _build_paired_set_on_disk(
            set_id,
            base_label,
            proj_label,
            base_extras,
            proj_extras,
            tmp_path,
        )
        base_sha = _sha256_of_file(base_path / "manifest.json")
        proj_sha = _sha256_of_file(proj_path / "manifest.json")
        for tag, sha in (
            (f"{set_id}.baseline", base_sha),
            (f"{set_id}.projection", proj_sha),
        ):
            if sha in seen_shas.values():
                colliding = next(
                    label for label, val in seen_shas.items() if val == sha
                )
                pytest.fail(
                    f"SHA collision: {tag} matches {colliding} (sha={sha}). "
                    "Each deliverable manifest must be distinct."
                )
            seen_shas[tag] = sha

    # 4 sets × 2 manifests = 8 distinct hashes
    assert len(seen_shas) == 8
    assert len(set(seen_shas.values())) == 8


# ── §3 4-set inventory pin ──────────────────────────────────────────


def test_ac_g_13_inventory_carries_4_sets() -> None:
    """Sprint G Draft 5 spec §AC-G-13 requires exactly 4 deliverable
    sets. Pin the inventory so a refactor that adds/drops a set must
    update this list explicitly (analogous to the AC-G-12 drill
    inventory pin)."""
    set_ids = [s[0] for s in _PAIRED_SETS]
    expected_set_ids = [
        "set1_niamey_millet",
        "set2_madarounfa_maize",
        "set3_menoua_groundnut",
        "set4_benoue_sorghum_ipsl",
    ]
    assert set_ids == expected_set_ids


def test_ac_g_13_set1_is_smallest_per_priority() -> None:
    """Per spec §AC-G-13 line 261: Set 1 is Niamey Millet (12 cells;
    smallest; HIGHEST priority for fast iteration). Pin the
    set_id label so the priority discipline survives a future
    sequence reorder."""
    set1 = _PAIRED_SETS[0]
    assert set1[0] == "set1_niamey_millet"
    assert "niamey" in set1[1].lower()
    assert "millet" in set1[1].lower()


# ── §4 Limitation key generality (per warning-auditor MEDIUM-Rebase-1) ─


# ── §5 Codex round 1 boundary 7/7 P2-1 absorption — SHA pin assertions ─


@pytest.mark.parametrize("spec", _PAIRED_SETS, ids=lambda s: s[0])
def test_ac_g_13_pinned_shas_match(
    spec: Tuple[str, str, str, Dict[str, Any], Dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Codex round 1 boundary 7/7 P2-1 absorption: assert each
    deliverable manifest SHA-256 matches the value pinned in
    ``EXPECTED_MANIFEST_SHAS``. Without this, a future format /
    fixture change would still pass the determinism test (which only
    checks twice-regenerate-equal) without surfacing the drift.

    Re-baselining: when a deliberate format change ships, capture
    the new hashes (run with the assertions disabled OR run
    ``hashlib.sha256(manifest.read_bytes()).hexdigest()`` directly)
    and update ``EXPECTED_MANIFEST_SHAS`` with rationale block."""
    set_id, base_label, proj_label, base_extras, proj_extras = spec
    base_path, proj_path = _build_paired_set_on_disk(
        set_id, base_label, proj_label, base_extras, proj_extras, tmp_path
    )
    expected_base, expected_proj = EXPECTED_MANIFEST_SHAS[set_id]
    actual_base = _sha256_of_file(base_path / "manifest.json")
    actual_proj = _sha256_of_file(proj_path / "manifest.json")

    assert actual_base == expected_base, (
        f"{set_id} baseline manifest SHA drift: "
        f"expected {expected_base}, got {actual_base}. "
        f"Re-baseline EXPECTED_MANIFEST_SHAS if intentional."
    )
    assert actual_proj == expected_proj, (
        f"{set_id} projection manifest SHA drift: "
        f"expected {expected_proj}, got {actual_proj}. "
        f"Re-baseline EXPECTED_MANIFEST_SHAS if intentional."
    )


def test_ac_g_13_pinned_shas_inventory_complete() -> None:
    """``EXPECTED_MANIFEST_SHAS`` must cover all 4 deliverable sets.
    A future addition that registers a new set in ``_PAIRED_SETS``
    without updating the SHA pin table fails this test loud."""
    pin_keys = set(EXPECTED_MANIFEST_SHAS.keys())
    paired_keys = {s[0] for s in _PAIRED_SETS}
    missing_pins = paired_keys - pin_keys
    extra_pins = pin_keys - paired_keys
    assert not missing_pins, (
        f"Sets without SHA pins: {missing_pins}. Update "
        "EXPECTED_MANIFEST_SHAS."
    )
    assert not extra_pins, (
        f"Stale SHA pins for removed sets: {extra_pins}. Drop them."
    )


def test_ac_g_13_pinned_shas_carries_distinct_values() -> None:
    """The 8 pinned SHAs (4 baseline + 4 projection) are all
    distinct. Catches a copy-paste error in the constants."""
    all_shas: list = []
    for base_sha, proj_sha in EXPECTED_MANIFEST_SHAS.values():
        all_shas.extend([base_sha, proj_sha])
    assert len(all_shas) == 8
    assert len(set(all_shas)) == 8


def test_ac_g_13_manifest_carries_actual_metadata(tmp_path: Path) -> None:
    """Codex round 1 boundary 7/7 P2-2 absorption regression pin:
    each deliverable manifest carries the actual region / crop /
    time-slice metadata, not the placeholder ``"unknown"`` /
    ``""``. Without flat-keyed config the manifest fields would
    be blank and the per-set distinctions wouldn't be pinned."""
    spec = _PAIRED_SETS[0]  # set1 niamey-millet
    set_id, base_label, proj_label, base_extras, proj_extras = spec
    base_path, _ = _build_paired_set_on_disk(
        set_id, base_label, proj_label, base_extras, proj_extras, tmp_path
    )
    manifest = json.loads(
        (base_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["project_name"] == "set1_niamey_millet"
    assert manifest["region"]["name"] == "Niamey"
    assert manifest["crop"]["name"] == "Millet"
    assert manifest["temporal"]["start_year"] == 2017
    assert manifest["temporal"]["end_year"] == 2021


def test_ac_g_13_weather_yrdoy_matches_time_slice(tmp_path: Path) -> None:
    """Codex round 1 boundary 7/7 P2-3 absorption regression pin:
    each set's WTH file's YRDOY data row matches the set's
    ``time_slice_start``. Without this, set4's 2086-2100 projection
    would carry 2046 weather data and the fixture would contradict
    the manifest's scenario metadata."""
    for spec in _PAIRED_SETS:
        set_id, base_label, proj_label, base_extras, proj_extras = spec
        base_path, proj_path = _build_paired_set_on_disk(
            set_id, base_label, proj_label, base_extras, proj_extras,
            tmp_path / set_id,
        )
        base_wth = (base_path / "weather" / "data.wth").read_text()
        proj_wth = (proj_path / "weather" / "data.wth").read_text()
        # The WTH data row contains the YRDOY: <time_slice_start>001
        assert f"{base_extras['time_slice_start']}001" in base_wth, (
            f"{set_id} baseline WTH YRDOY drift"
        )
        assert f"{proj_extras['time_slice_start']}001" in proj_wth, (
            f"{set_id} projection WTH YRDOY drift"
        )


def test_asymmetry_limitation_key_is_general_form() -> None:
    """Per warning-auditor pass-2 MEDIUM-Rebase-1 the canonical
    limitation key is the GENERAL-form
    ``weather_schema_asymmetric_within_set`` (covers any column-count
    asymmetry dimension), NOT the value-specific Draft 2 form
    ``weather_schema_baseline_5col``. Pin the canonical key string
    so a future drift back to the specific form fails loud."""
    # The constant used by the fixture
    canonical_key = "weather_schema_asymmetric_within_set"
    forbidden_specific_keys = {
        "weather_schema_baseline_5col",
        "weather_schema_baseline_8col",
        "weather_schema_5col_vs_8col",
    }
    # Build one set + read manifest.json to confirm the key form
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        spec = _PAIRED_SETS[0]
        base_path, _ = _build_paired_set_on_disk(*spec, root=root)
        manifest = json.loads(
            (base_path / "manifest.json").read_text(encoding="utf-8")
        )
        keys = set(manifest.get("limitations", {}).keys())
        assert canonical_key in keys
        for forbidden in forbidden_specific_keys:
            assert forbidden not in keys, (
                f"Forbidden value-specific limitation key {forbidden!r} "
                "appeared in manifest.limitations — use the general-form "
                f"{canonical_key!r} per warning-auditor pass-2 "
                "MEDIUM-Rebase-1."
            )
