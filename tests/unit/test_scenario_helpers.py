"""Pin the scenario_helpers builder + rewriter contract.

Two helpers ship in ``prismpy.packaging.scenario_helpers``:

* :func:`build_baseline_scenario_block` constructs a ``BASE``-role
  scenario block for observed-climate baseline packages so the UC2
  pre-flight validator has a non-null ``manifest.scenario`` to read.

* :func:`rewrite_pythia_config_for_scenario` overwrites
  ``sdate`` / ``pfrst`` / ``plast`` / ``runs[*].startYear`` in a
  delivered ``pythia_config.json`` so projection packages emit DSSAT
  config aligned with the scenario's time-slice years instead of
  inheriting the baseline's.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismpy.models.scenario import (
    BiasCorrectionMethod,
    MissingProvenanceError,
    ScenarioBlock,
    ScenarioRole,
)
from prismpy.packaging.scenario_helpers import (
    build_baseline_scenario_block,
    rewrite_pythia_config_for_scenario,
)


# ── build_baseline_scenario_block ───────────────────────────────────


def test_baseline_scenario_block_has_canonical_role_and_method() -> None:
    """Baseline block carries scenario_role='baseline' + bias_correction_method='none'.

    The schema's enum value for ``BASE`` is the string ``"baseline"``
    (NOT ``"base"``); the helper returns a serialized form per
    ``use_enum_values=True`` config so downstream JSON consumers see
    the literal string.
    """
    block = build_baseline_scenario_block(
        scenario_label="OBSERVED_BENOUE_SORGHUM_2013-2015",
        time_slice_start=2013,
        time_slice_end=2015,
        co2_ppm=410.0,
        co2_ppm_provenance=(
            "AR6 WG1 Annex III + RCMIP, mid-year-of-period convention"
        ),
    )
    serialized = block.model_dump()
    assert serialized["scenario_role"] == "baseline"
    assert serialized["bias_correction_method"] == "none"
    # rcp_or_ssp default for observed.
    assert serialized["rcp_or_ssp"] == "historical"
    # gcm_source default for observed.
    assert serialized["gcm_source"] == "observed_NASA-POWER"
    # Self-reference convention: baseline_reference_label == scenario_label.
    assert serialized["baseline_reference_label"] == serialized["scenario_label"]


def test_baseline_block_self_reference_when_label_omitted() -> None:
    """``baseline_reference_label`` defaults to ``scenario_label``."""
    block = build_baseline_scenario_block(
        scenario_label="OBSERVED_TEST_2020-2022",
        time_slice_start=2020,
        time_slice_end=2022,
        co2_ppm=415.0,
        co2_ppm_provenance="NOAA Mauna Loa monthly mean (2020-2022)",
    )
    assert block.baseline_reference_label == "OBSERVED_TEST_2020-2022"
    assert block.scenario_label == block.baseline_reference_label


def test_baseline_block_explicit_reference_label_override() -> None:
    """An explicit ``baseline_reference_label`` overrides the self-reference default."""
    block = build_baseline_scenario_block(
        scenario_label="OBSERVED_NEW_LABEL",
        time_slice_start=2010,
        time_slice_end=2012,
        co2_ppm=389.0,
        co2_ppm_provenance="historical observed CO2",
        baseline_reference_label="EXPLICIT_PARENT_LABEL",
    )
    assert block.baseline_reference_label == "EXPLICIT_PARENT_LABEL"
    assert block.scenario_label == "OBSERVED_NEW_LABEL"


def test_baseline_block_rejects_empty_co2_provenance() -> None:
    """Empty ``co2_ppm_provenance`` raises MissingProvenanceError per AC-G-10."""
    with pytest.raises(Exception):  # ValidationError wraps MissingProvenanceError
        build_baseline_scenario_block(
            scenario_label="X",
            time_slice_start=2015,
            time_slice_end=2015,
            co2_ppm=400.0,
            co2_ppm_provenance="",
        )


def test_baseline_block_rejects_co2_outside_schema_bounds() -> None:
    """``co2_ppm`` < 200 or > 2000 fails Pydantic validation."""
    with pytest.raises(Exception):
        build_baseline_scenario_block(
            scenario_label="X",
            time_slice_start=2015,
            time_slice_end=2015,
            co2_ppm=100.0,  # below schema floor 200.0
            co2_ppm_provenance="test",
        )


def test_baseline_block_rejects_inverted_time_slice() -> None:
    """``time_slice_end < time_slice_start`` fails the cross-field validator."""
    with pytest.raises(Exception):
        build_baseline_scenario_block(
            scenario_label="X",
            time_slice_start=2020,
            time_slice_end=2018,
            co2_ppm=410.0,
            co2_ppm_provenance="test",
        )


def test_baseline_block_accepts_alternative_observed_source() -> None:
    """``gcm_source`` override flows through unchanged."""
    block = build_baseline_scenario_block(
        scenario_label="X",
        time_slice_start=2015,
        time_slice_end=2017,
        co2_ppm=403.0,
        co2_ppm_provenance="historical CO2 observation",
        gcm_source="observed_AgERA5",
    )
    assert block.gcm_source == "observed_AgERA5"


# ── rewrite_pythia_config_for_scenario ──────────────────────────────


@pytest.fixture
def baseline_pythia_config(tmp_path: Path) -> Path:
    """Drop a synthetic baseline pythia_config.json carrying baseline years."""
    config = {
        "default_setup": {
            "sdate": "2013-01-01",
            "pfrst": "2013-06-15",
            "plast": "2013-07-15",
            "wsta": "lookup_wth::CMBE::vector::./shapes/sites.shp::ID",
        },
        "runs": [
            {"name": "run-1", "startYear": 2013, "nyers": 3},
            {"name": "run-2", "startYear": 2013, "nyers": 3},
        ],
    }
    path = tmp_path / "pythia_config.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    return path


def test_rewrite_updates_sdate_to_scenario_start_year(
    baseline_pythia_config: Path,
) -> None:
    """``sdate`` rewrites to ``YYYY-01-01`` for the new start year."""
    rewrite_pythia_config_for_scenario(
        baseline_pythia_config,
        time_slice_start=2046,
        time_slice_end=2048,
    )
    with baseline_pythia_config.open() as fh:
        config = json.load(fh)
    assert config["default_setup"]["sdate"] == "2046-01-01"


def test_rewrite_preserves_pfrst_plast_month_day_when_no_doy(
    baseline_pythia_config: Path,
) -> None:
    """Without ``planting_doy``, pfrst/plast keep MM-DD; only year shifts."""
    rewrite_pythia_config_for_scenario(
        baseline_pythia_config,
        time_slice_start=2046,
        time_slice_end=2048,
    )
    with baseline_pythia_config.open() as fh:
        config = json.load(fh)
    # pfrst / plast keep month-day, only year shifts.
    assert config["default_setup"]["pfrst"] == "2046-06-15"
    assert config["default_setup"]["plast"] == "2046-07-15"


def test_rewrite_replaces_pfrst_plast_with_doy_when_provided(
    baseline_pythia_config: Path,
) -> None:
    """With ``planting_doy``, pfrst/plast are derived from DOY in new year."""
    rewrite_pythia_config_for_scenario(
        baseline_pythia_config,
        time_slice_start=2046,
        time_slice_end=2048,
        planting_doy=166,  # June 15 in non-leap year
        planting_window_days=30,
    )
    with baseline_pythia_config.open() as fh:
        config = json.load(fh)
    assert config["default_setup"]["pfrst"] == "2046-06-15"
    assert config["default_setup"]["plast"] == "2046-07-15"


def test_rewrite_updates_runs_start_year_and_nyers(
    baseline_pythia_config: Path,
) -> None:
    """Every ``runs[i].startYear`` and ``nyers`` aligns with new period."""
    rewrite_pythia_config_for_scenario(
        baseline_pythia_config,
        time_slice_start=2046,
        time_slice_end=2048,  # 3-year span
    )
    with baseline_pythia_config.open() as fh:
        config = json.load(fh)
    for run in config["runs"]:
        assert run["startYear"] == 2046
        assert run["nyers"] == 3


def test_rewrite_preserves_unrelated_fields(
    baseline_pythia_config: Path,
) -> None:
    """Fields not touched by the rewrite (wsta, run names) round-trip."""
    rewrite_pythia_config_for_scenario(
        baseline_pythia_config,
        time_slice_start=2046,
        time_slice_end=2048,
    )
    with baseline_pythia_config.open() as fh:
        config = json.load(fh)
    assert (
        config["default_setup"]["wsta"]
        == "lookup_wth::CMBE::vector::./shapes/sites.shp::ID"
    )
    assert config["runs"][0]["name"] == "run-1"
    assert config["runs"][1]["name"] == "run-2"


def test_rewrite_rejects_inverted_time_slice(
    baseline_pythia_config: Path,
) -> None:
    """``time_slice_end < time_slice_start`` raises ValueError."""
    with pytest.raises(ValueError, match="time_slice_end"):
        rewrite_pythia_config_for_scenario(
            baseline_pythia_config,
            time_slice_start=2050,
            time_slice_end=2046,
        )


def test_rewrite_raises_on_missing_file(tmp_path: Path) -> None:
    """Missing pythia_config.json raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        rewrite_pythia_config_for_scenario(
            missing,
            time_slice_start=2046,
            time_slice_end=2048,
        )


def test_rewrite_handles_missing_default_setup_block(tmp_path: Path) -> None:
    """A config without default_setup still has runs[] rewritten cleanly."""
    config = {"runs": [{"name": "run-1", "startYear": 2013, "nyers": 3}]}
    path = tmp_path / "pythia_config.json"
    with path.open("w") as fh:
        json.dump(config, fh)

    rewrite_pythia_config_for_scenario(
        path,
        time_slice_start=2046,
        time_slice_end=2048,
    )
    with path.open() as fh:
        rewritten = json.load(fh)
    assert "default_setup" not in rewritten
    assert rewritten["runs"][0]["startYear"] == 2046


# ── Codex round 1 LOW absorption — edge cases on rewriter ───────────


def test_rewrite_handles_missing_runs_block(tmp_path: Path) -> None:
    """A config without ``runs`` still has default_setup rewritten cleanly.

    Mirror of `test_rewrite_handles_missing_default_setup_block` for
    the inverse omission. The rewriter no-ops on the missing key
    without raising, preserving the rest of the config untouched.
    """
    config = {
        "default_setup": {
            "sdate": "2013-01-01",
            "pfrst": "2013-06-15",
            "plast": "2013-07-15",
        }
    }
    path = tmp_path / "pythia_config.json"
    with path.open("w") as fh:
        json.dump(config, fh)

    rewrite_pythia_config_for_scenario(
        path,
        time_slice_start=2046,
        time_slice_end=2048,
    )
    with path.open() as fh:
        rewritten = json.load(fh)
    # default_setup got rewritten as expected.
    assert rewritten["default_setup"]["sdate"] == "2046-01-01"
    # runs[] absence preserved (rewriter no-ops without raising).
    assert "runs" not in rewritten


def test_rewrite_raises_on_empty_file(tmp_path: Path) -> None:
    """An empty pythia_config.json raises a JSON decode error.

    Defensive: the rewriter relies on `json.load` to parse the
    input; an empty file produces a JSONDecodeError before the
    rewriter touches the on-disk content. This pin asserts the
    rewriter does NOT silently no-op or write malformed output.
    """
    path = tmp_path / "pythia_config.json"
    path.write_text("")  # empty file

    with pytest.raises(json.JSONDecodeError):
        rewrite_pythia_config_for_scenario(
            path,
            time_slice_start=2046,
            time_slice_end=2048,
        )


def test_rewrite_raises_on_malformed_json(tmp_path: Path) -> None:
    """A malformed pythia_config.json raises a JSON decode error.

    Mirror of the empty-file pin for the inverse failure mode: the
    rewriter must not silently overwrite the malformed input with
    "fixed" content. Failing loud at the parse step lets the caller
    inspect + decide (re-fetch upstream artifact, re-emit, etc.)
    rather than corrupting the on-disk state.
    """
    path = tmp_path / "pythia_config.json"
    path.write_text("{ this is not valid json,, }")

    with pytest.raises(json.JSONDecodeError):
        rewrite_pythia_config_for_scenario(
            path,
            time_slice_start=2046,
            time_slice_end=2048,
        )


# ── CA-1 wiring helpers: estimate_observed_co2_ppm + period builder ──


def test_estimate_observed_co2_ppm_at_anchor_year() -> None:
    """At an anchor year, the function returns the anchor value exactly."""
    from prismpy.packaging.scenario_helpers import estimate_observed_co2_ppm

    # 2014 is the canonical mid-2010s anchor neighborhood; the function
    # should return a value between the 2010 and 2015 anchors.
    assert 390.0 <= estimate_observed_co2_ppm(2014) <= 401.0
    # 2020 is exactly an anchor (414.0).
    assert estimate_observed_co2_ppm(2020) == 414.0
    # 2010 is exactly an anchor (390.0).
    assert estimate_observed_co2_ppm(2010) == 390.0


def test_estimate_observed_co2_ppm_saturates_at_extremes() -> None:
    """Years outside the anchor range return the closest anchor value."""
    from prismpy.packaging.scenario_helpers import estimate_observed_co2_ppm

    # Pre-1958: returns the first anchor.
    assert estimate_observed_co2_ppm(1900) == 315.0
    # Post-2023: returns the last anchor.
    assert estimate_observed_co2_ppm(2050) == 422.0


def test_estimate_observed_co2_ppm_increases_monotonically() -> None:
    """CO2 is strictly non-decreasing across the historical period.

    Sanity pin: any anchor table entry that violates monotonicity
    (e.g., a typo flipping two values) fails this check. Real
    Mauna Loa observations are monotonically non-decreasing year-
    over-year through 2023.
    """
    from prismpy.packaging.scenario_helpers import estimate_observed_co2_ppm

    years = [1960, 1970, 1980, 1990, 2000, 2010, 2014, 2020, 2023]
    values = [estimate_observed_co2_ppm(y) for y in years]
    for prev, curr in zip(values, values[1:]):
        assert curr >= prev, (
            f"CO2 estimator non-monotonic across anchors: {values}"
        )


def test_build_baseline_scenario_block_for_period_for_benoue() -> None:
    """The period-aware builder constructs a valid block with auto-derived defaults."""
    from prismpy.packaging.scenario_helpers import (
        build_baseline_scenario_block_for_period,
    )

    block = build_baseline_scenario_block_for_period(
        region_name="Bénoué",
        crop_name="Sorghum",
        time_slice_start=2013,
        time_slice_end=2015,
    )
    serialized = block.model_dump()
    assert serialized["scenario_role"] == "baseline"
    assert serialized["bias_correction_method"] == "none"
    assert serialized["rcp_or_ssp"] == "historical"
    assert serialized["gcm_source"] == "observed_NASA-POWER"
    assert serialized["time_slice_start"] == 2013
    assert serialized["time_slice_end"] == 2015
    # Label is ASCII-folded + uppercased.
    assert "BENOUE" in serialized["scenario_label"]
    assert "SORGHUM" in serialized["scenario_label"]
    assert "É" not in serialized["scenario_label"]
    # CO2 ppm for 2014 midpoint is between 2010 and 2015 anchors.
    assert 390.0 <= serialized["co2_ppm"] <= 401.0
    # Provenance string is non-empty (AC-G-10).
    assert serialized["co2_ppm_provenance"]
    assert "Mauna Loa" in serialized["co2_ppm_provenance"]


def test_build_baseline_scenario_block_for_period_handles_pure_nonascii_region() -> None:
    """A pure-non-ASCII region defaults to UNKNOWN in the label."""
    from prismpy.packaging.scenario_helpers import (
        build_baseline_scenario_block_for_period,
    )

    # Pure Chinese ideographs fold to empty → UNKNOWN sentinel.
    block = build_baseline_scenario_block_for_period(
        region_name="中文",
        crop_name="Sorghum",
        time_slice_start=2015,
        time_slice_end=2017,
    )
    assert "UNKNOWN" in block.scenario_label
    assert block.scenario_label.isascii()


def test_pythia_translator_baseline_manifest_has_scenario_block(tmp_path: Path) -> None:
    """CA-1 round-trip pin: PYTHIA translator's manifest carries a non-null scenario block.

    Synthetic Bénoué-Cameroon-Sorghum-2013-2015 project exercises
    the full ``_generate_manifest`` path; the on-disk manifest.json
    MUST have ``scenario.scenario_role == "baseline"`` populated by
    the period-aware builder. Without CA-1 wiring (i.e., on commit
    ``2ecc0a9`` before the fixup), the manifest's scenario field
    was null and UC2 hard-failed.
    """
    from prismpy.config.schema import (
        BoundaryConfig,
        BoundarySource,
        CropCalendarConfig,
        CropConfig,
        ManualBoundsConfig,
        OutputConfig,
        Platform,
        ProjectConfig,
        ProjectInfo,
        RegionConfig,
        TemporalConfig,
    )
    from prismpy.models.region import BoundingBox, Region
    from prismpy.translators.base import UnifiedData
    from prismpy.translators.pythia.translator import PythiaTranslator

    cfg = ProjectConfig(
        project=ProjectInfo(name="ca1_baseline_scenario_pin"),
        region=RegionConfig(
            name="Bénoué",
            country="Cameroon",
            country_iso3="CMR",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=13.5, miny=8.0, maxx=14.5, maxy=9.0,
                ),
            ),
        ),
        crop=CropConfig(
            name="Sorghum",
            name_short="sgh",
            calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285),
        ),
        temporal=TemporalConfig(
            start_year=2013, end_year=2015, spinup_years=0,
        ),
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir=str(tmp_path), structure="by_platform"),
    )

    region = Region(
        name="Bénoué",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=13.5, miny=8.0, maxx=14.5, maxy=9.0),
    )
    data = UnifiedData(region=region)

    translator = PythiaTranslator(config=cfg, output_dir=str(tmp_path))
    manifest_path = translator._generate_manifest(data)

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    assert "scenario" in manifest, (
        "manifest.json missing scenario block — CA-1 wiring not in effect"
    )
    assert manifest["scenario"] is not None
    assert manifest["scenario"]["scenario_role"] == "baseline"
    assert manifest["scenario"]["bias_correction_method"] == "none"
    assert manifest["scenario"]["time_slice_start"] == 2013
    assert manifest["scenario"]["time_slice_end"] == 2015


# ── End-to-end pin: helpers compose for a UC2 baseline ─────────────


def test_baseline_block_round_trips_through_manifest_creation(
    tmp_path: Path,
) -> None:
    """build_baseline_scenario_block + create_manifest produce the
    expected on-disk shape: ``manifest.scenario.scenario_role`` == ``"baseline"``.
    """
    from prismpy.packaging.manifest import create_manifest

    block = build_baseline_scenario_block(
        scenario_label="OBSERVED_BENOUE_SORGHUM_2013-2015",
        time_slice_start=2013,
        time_slice_end=2015,
        co2_ppm=399.0,
        co2_ppm_provenance="NOAA Mauna Loa annual mean for 2014",
    )

    # Minimal package directory + a placeholder file so file-walker has
    # something to enumerate.
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "README.md").write_text("test\n", encoding="utf-8")

    manifest = create_manifest(
        package_dir=pkg_dir,
        project_config={
            "project_name": "test",
            "region_name": "Bénoué",
            "country": "Cameroon",
            "country_iso3": "CMR",
            "crop_name": "Sorghum",
            "start_year": 2013,
            "end_year": 2015,
            "data_sources": {"climate": "AgERA5"},
        },
        platform="pythia",
        scenario=block,
    )
    assert "scenario" in manifest
    assert manifest["scenario"]["scenario_role"] == "baseline"
    assert manifest["scenario"]["bias_correction_method"] == "none"
    assert manifest["scenario"]["co2_ppm"] == 399.0
