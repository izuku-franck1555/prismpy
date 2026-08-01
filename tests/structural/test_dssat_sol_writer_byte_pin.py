"""Byte-level regression pin for the canonical DSSAT v4.8 .SOL writer.

The writer at :func:`prismpy.translators._shared.dssat_sol_writer.write_dssat_sol`
is the single producer of DSSAT-format soil files for every translator
that emits one (CRAFT today; the eGHR substrate builder for PYTHIA next).
A byte-level regression here would silently shift the DSSAT-CSM consumer
experience for every package, so the format is pinned with both a
SHA-256 hash and a fixture file checked into the test tree.

The test fixture at ``tests/fixtures/dssat_sol_writer/expected_two_profiles.SOL``
covers two deterministic profiles:
- profile 1 hits the high-clay branch (clay > 40, SCOM 0.09 / 0.20 / 85)
- profile 2 hits the high-sand branch (sand > 70, SCOM 0.13 / 0.60 / 60)

so a refactor that breaks either SCOM-classification arm or any of the
fixed-width layer-row formatters fails the test deterministically.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.translators._shared.dssat_sol_writer import write_dssat_sol


FIXTURE_PATH = (
    Path(__file__).parent.parent
    / "fixtures"
    / "dssat_sol_writer"
    / "expected_two_profiles.SOL"
)

# Pin recorded 2026-05-07; regenerated 2026-08-01 for the SLTX-code fix +
# FORMAT 5030 column alignment. The *-header now emits the DSSAT SLTX code
# in the A5 field at cols 26-30 and depth (F5.0) at cols 32-36, per
# IPSOIL_Inp.for:627 (1X,A10,2X,A11,1X,A5,1X,F5.0,1X,A50) — so DSSAT reads
# BOTH texture and depth at their fixed columns. The prior spelled-out class
# overflowed the field and shifted the depth -> IPSOIL Error 5010. Update
# only when the fixture is intentionally regenerated; paired with the fixture
# file so a SHA update is meaningless without a corresponding fixture rewrite.
EXPECTED_SHA256 = "6e57f5fc5bbbc615e52648112249b831cb2ffe83e59fcd64fcd3746e783e6585"


def _build_profiles() -> dict[int, SoilProfile]:
    """Return two deterministic profiles exercising both SCOM branches."""
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
    """Return a minimal region the writer will populate every header line with."""
    return Region(
        name="Test Region",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=2.0, miny=12.0, maxx=4.0, maxy=14.0),
    )


def test_write_dssat_sol_returns_expected_profile_name_mapping(tmp_path: Path) -> None:
    """The helper must return ``{profile_id: 10-char profile name}``."""
    out_path = tmp_path / "TEST.SOL"
    mapping = write_dssat_sol(
        soil_path=out_path,
        profiles_by_id=_build_profiles(),
        country_code="CM",
        region=_build_region(),
    )
    assert mapping == {1: "CM00000001", 2: "CM00000002"}


def test_write_dssat_sol_byte_identical_to_fixture(tmp_path: Path) -> None:
    """The helper must produce byte-identical output across refactors."""
    out_path = tmp_path / "TEST.SOL"
    write_dssat_sol(
        soil_path=out_path,
        profiles_by_id=_build_profiles(),
        country_code="CM",
        region=_build_region(),
    )
    actual_bytes = out_path.read_bytes()
    expected_bytes = FIXTURE_PATH.read_bytes()
    assert actual_bytes == expected_bytes, (
        "DSSAT .SOL writer drifted from the canonical fixture.\n"
        f"Fixture:  {FIXTURE_PATH}\n"
        f"Actual SHA-256:   {hashlib.sha256(actual_bytes).hexdigest()}\n"
        f"Expected SHA-256: {hashlib.sha256(expected_bytes).hexdigest()}\n"
        "If this is an intentional format change, regenerate the fixture and"
        " update EXPECTED_SHA256 in this file."
    )


def test_write_dssat_sol_byte_identical_to_pinned_sha(tmp_path: Path) -> None:
    """The helper output's SHA-256 must match the pin recorded at first generation."""
    out_path = tmp_path / "TEST.SOL"
    write_dssat_sol(
        soil_path=out_path,
        profiles_by_id=_build_profiles(),
        country_code="CM",
        region=_build_region(),
    )
    actual_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    assert actual_sha == EXPECTED_SHA256, (
        f"DSSAT .SOL writer SHA-256 drifted from pinned constant.\n"
        f"Pinned:   {EXPECTED_SHA256}\n"
        f"Actual:   {actual_sha}\n"
        "If this is an intentional format change, regenerate the fixture and"
        " update EXPECTED_SHA256."
    )
