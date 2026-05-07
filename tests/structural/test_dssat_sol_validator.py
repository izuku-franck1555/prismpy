"""Sprint S AC-7 — DSSAT v4.8 .SOL format validator.

Round-trip pin: every ``.SOL`` produced by
:func:`prismpy.translators._shared.dssat_sol_writer.write_dssat_sol`
must validate cleanly through
:func:`prismpy.translators._shared.dssat_sol_validator.validate_dssat_sol`.
The two helpers are a canonical-source pair (durable lesson §24):
the writer's output is the validator's contract; format drift in
either side is caught by this test file.

Negative-case coverage exercises the validator's error-detection
arms: a missing ``*SOILS:`` banner, a profile id line shorter than
11 characters, a layer row with the wrong field count, an empty
file, and a path that does not exist. Each case asserts the
validator surfaces an error rather than silently passing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.translators._shared import (
    validate_dssat_sol,
    write_dssat_sol,
)


# Reference fixture — the canonical writer's two-profile golden
# output that the byte-pin in test_dssat_sol_writer_byte_pin already
# pins. If the writer changes its output, both pins fire and the
# format change has to be deliberate on both sides.
_REFERENCE_FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "dssat_sol_writer"
    / "expected_two_profiles.SOL"
)


def _build_profiles() -> Dict[int, SoilProfile]:
    return {
        1: SoilProfile(
            profile_id="P1",
            lat=12.4,
            lon=-5.5,
            source="hwsd",
            layers=[
                SoilLayer(
                    depth_top=0.0,
                    depth_bottom=0.2,
                    sand=30.0,
                    clay=45.0,
                    silt=25.0,
                    organic_carbon=0.6,
                    bulk_density=1.35,
                    ph=6.4,
                    field_capacity=0.30,
                    wilting_point=0.18,
                ),
                SoilLayer(
                    depth_top=0.2,
                    depth_bottom=1.0,
                    sand=28.0,
                    clay=48.0,
                    silt=24.0,
                    organic_carbon=0.4,
                    bulk_density=1.45,
                    ph=6.2,
                    field_capacity=0.32,
                    wilting_point=0.20,
                ),
            ],
        ),
        2: SoilProfile(
            profile_id="P2",
            lat=13.5,
            lon=2.1,
            source="hwsd",
            layers=[
                SoilLayer(
                    depth_top=0.0,
                    depth_bottom=0.3,
                    sand=82.0,
                    clay=8.0,
                    silt=10.0,
                    organic_carbon=0.2,
                    bulk_density=1.55,
                    ph=7.0,
                    field_capacity=0.15,
                    wilting_point=0.05,
                ),
            ],
        ),
    }


def _build_region() -> Region:
    return Region(
        name="Test Region",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=2.0, miny=12.0, maxx=4.0, maxy=14.0),
    )


def test_canonical_writer_output_validates_cleanly(tmp_path: Path) -> None:
    """The canonical writer's output must round-trip through the validator with zero errors."""
    sol_path = tmp_path / "TEST.SOL"
    write_dssat_sol(
        soil_path=sol_path,
        profiles_by_id=_build_profiles(),
        country_code="CM",
        region=_build_region(),
    )

    result = validate_dssat_sol(sol_path)

    assert result.is_valid, (
        f"Canonical writer output failed validation:\n"
        + "\n".join(f"  [{i.severity}] line {i.line_number}: {i.message}" for i in result.errors)
    )
    assert result.profile_count == 2
    assert result.layer_count == 3  # profile 1 has 2 layers + profile 2 has 1


def test_reference_fixture_validates_cleanly() -> None:
    """The byte-pin reference fixture is the canonical-source acceptance for the validator.

    The fixture lives in ``tests/fixtures/dssat_sol_writer/`` and is
    the byte-pin reference for the writer; reusing it as the
    validator's reference closes the canonical-source-or-pin loop
    (durable §24): the fixture is the bridge between the writer's
    byte-pin test and the validator's format-conformance test.
    """
    if not _REFERENCE_FIXTURE.exists():
        pytest.skip("Reference fixture missing; AC-1 byte-pin test is the canonical generator.")

    result = validate_dssat_sol(_REFERENCE_FIXTURE)

    assert result.is_valid, (
        f"Reference fixture failed validation:\n"
        + "\n".join(f"  [{i.severity}] line {i.line_number}: {i.message}" for i in result.errors)
    )
    assert result.profile_count == 2
    assert result.layer_count == 3


def test_validator_flags_missing_soils_banner(tmp_path: Path) -> None:
    """A file without the ``*SOILS:`` banner fails validation with an error."""
    bad = tmp_path / "no_banner.SOL"
    bad.write_text("*CM00000001    CMR    Clay       100    HWSD v2 SMU 1\n")

    result = validate_dssat_sol(bad)

    assert not result.is_valid
    assert any(
        "*SOILS:" in issue.message for issue in result.errors
    ), f"Expected *SOILS: banner error; got {[i.message for i in result.errors]}"


def test_validator_flags_too_short_profile_id(tmp_path: Path) -> None:
    """A profile-id line shorter than 11 chars fails validation with an error."""
    bad = tmp_path / "short_id.SOL"
    bad.write_text(
        "*SOILS: test - Generated by prismpy\n"
        "\n"
        "*X\n"  # only 2 chars after the asterisk
    )

    result = validate_dssat_sol(bad)

    assert not result.is_valid
    assert any(
        "Profile id line must be at least 11 chars" in issue.message
        for issue in result.errors
    )


def test_validator_flags_layer_row_field_count_mismatch(tmp_path: Path) -> None:
    """A layer row with fewer than 17 fields fails validation with an error."""
    sol_path = tmp_path / "TEST.SOL"
    write_dssat_sol(
        soil_path=sol_path,
        profiles_by_id=_build_profiles(),
        country_code="CM",
        region=_build_region(),
    )

    # Truncate the last layer row by removing tokens.
    text = sol_path.read_text()
    lines = text.splitlines()
    # Find a layer row (any row whose first token parses as a digit).
    for idx, line in enumerate(lines):
        tokens = line.split()
        if tokens and tokens[0].isdigit() and len(tokens) >= 17:
            # Truncate to 10 tokens.
            lines[idx] = " ".join(tokens[:10])
            break
    sol_path.write_text("\n".join(lines))

    result = validate_dssat_sol(sol_path)

    assert not result.is_valid
    assert any(
        "must have exactly 17 fields" in issue.message
        for issue in result.errors
    )


def test_validator_handles_missing_file(tmp_path: Path) -> None:
    """A non-existent path produces a validation error, not an unhandled exception."""
    missing = tmp_path / "does_not_exist.SOL"
    result = validate_dssat_sol(missing)

    assert not result.is_valid
    assert any(
        "does not exist" in issue.message for issue in result.errors
    )


def test_validator_handles_empty_file(tmp_path: Path) -> None:
    """An empty file produces a validation error."""
    empty = tmp_path / "empty.SOL"
    empty.write_text("")
    result = validate_dssat_sol(empty)

    assert not result.is_valid
    assert any("empty" in issue.message for issue in result.errors)


def test_validator_handles_no_profile_blocks(tmp_path: Path) -> None:
    """A banner-only file with no profile blocks fails validation."""
    no_profiles = tmp_path / "banner_only.SOL"
    no_profiles.write_text("*SOILS: test - Generated by prismpy\n\n")
    result = validate_dssat_sol(no_profiles)

    assert not result.is_valid
    assert any(
        "No profile blocks found" in issue.message for issue in result.errors
    )


def test_validator_flags_profile_with_no_layer_rows(tmp_path: Path) -> None:
    """A profile whose @SLB header is followed by no data rows fails validation.

    Per Tsuji et al. 1994 §5.2.4 + the DSSAT v4.8 user guide, every
    profile must carry at least one layer row. The earlier validator
    accepted profiles with zero layer rows because the loop
    terminated cleanly on EOF / blank line / next-profile without
    checking the running ``layer_count``. This pin asserts the
    missing-rows error fires for each terminator path: blank line,
    EOF, and the start of the next profile block.
    """
    bad = tmp_path / "no_layer_rows.SOL"
    bad.write_text(
        "*SOILS: test - Generated by prismpy\n"
        "\n"
        "*CM00000001    CMR    Clay       100    HWSD v2 SMU 1\n"
        "@SITE        COUNTRY          LAT     LONG SCS FAMILY\n"
        " Test Region CMR            12.400  -5.500     Clay\n"
        "@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE\n"
        "    -9  0.09  6.00  0.20 85.00  1.00  1.00 IB001 IB001 IB001\n"
        "@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC\n"
        "\n"  # blank line straight after the layer header — no data rows
    )
    result = validate_dssat_sol(bad)

    assert not result.is_valid
    assert any(
        "no layer data rows" in issue.message for issue in result.errors
    )


def test_validator_flags_profile_with_no_layer_rows_at_eof(tmp_path: Path) -> None:
    """A profile whose @SLB header is followed by EOF fails validation."""
    bad = tmp_path / "no_layer_rows_eof.SOL"
    bad.write_text(
        "*SOILS: test - Generated by prismpy\n"
        "\n"
        "*CM00000001    CMR    Clay       100    HWSD v2 SMU 1\n"
        "@SITE        COUNTRY          LAT     LONG SCS FAMILY\n"
        " Test Region CMR            12.400  -5.500     Clay\n"
        "@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE\n"
        "    -9  0.09  6.00  0.20 85.00  1.00  1.00 IB001 IB001 IB001\n"
        "@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC\n"
        # EOF straight after the layer header, no blank line, no next profile
    )
    result = validate_dssat_sol(bad)

    assert not result.is_valid
    assert any(
        "no layer data rows" in issue.message for issue in result.errors
    )


def test_validator_flags_profile_with_no_layer_rows_before_next_profile(
    tmp_path: Path,
) -> None:
    """A profile whose @SLB header is followed immediately by another profile fails validation."""
    bad = tmp_path / "no_layer_rows_then_next.SOL"
    bad.write_text(
        "*SOILS: test - Generated by prismpy\n"
        "\n"
        "*CM00000001    CMR    Clay       100    HWSD v2 SMU 1\n"
        "@SITE        COUNTRY          LAT     LONG SCS FAMILY\n"
        " Test Region CMR            12.400  -5.500     Clay\n"
        "@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE\n"
        "    -9  0.09  6.00  0.20 85.00  1.00  1.00 IB001 IB001 IB001\n"
        "@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC\n"
        # Next profile begins immediately, no layer rows for the first one
        "*CM00000002    CMR    LoamySand   30    HWSD v2 SMU 2\n"
        "@SITE        COUNTRY          LAT     LONG SCS FAMILY\n"
        " Test Region CMR            13.500   2.100     Loamy Sand\n"
        "@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE\n"
        "    -9  0.13  6.00  0.60 60.00  1.00  1.00 IB001 IB001 IB001\n"
        "@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC\n"
        "    30 -9    0.050 0.150 0.450  0.20 10.00  1.55  0.20   8.0  10.0   0.0  0.00   7.0   7.0 -99.0 -99.0\n"
    )
    result = validate_dssat_sol(bad)

    assert not result.is_valid
    assert any(
        "no layer data rows" in issue.message for issue in result.errors
    )
