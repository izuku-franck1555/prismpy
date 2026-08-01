"""Regression: the .SOL ``*``-header SLTX field emits DSSAT texture CODES.

The profile-header SLTX field is fixed-width. Before the fix the writer
emitted the spelled-out class name; ``SandyClayLoam`` (13 chars) overflowed
the column and shifted the depth field, so DSSAT's IPSOIL parser rejected
~76% of the Case C profiles (Error 5010). The writer now emits the
canonical DSSAT SLTX code (<=4 chars) which always fits. These tests pin
the code mapping, the fail-loud on unmapped/absent classes, and — the crux
— that a SandyClayLoam profile renders ``SCL`` with the depth column still
aligned (an overflow would shift it right).
"""
from __future__ import annotations

import pytest

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.translators._shared.dssat_sol_writer import (
    _DSSAT_SLTX_CODES,
    _dssat_sltx_code,
    write_dssat_sol,
)


# ── the SLTX code mapping (helper) ───────────────────────────────────


@pytest.mark.parametrize(
    "texture, code",
    [
        ("Sand", "S"),
        ("Loamy Sand", "LS"),
        ("Sandy Loam", "SL"),
        ("Loam", "L"),
        ("Silt Loam", "SIL"),
        ("Silt", "SI"),
        ("Sandy Clay Loam", "SCL"),
        ("Clay Loam", "CL"),
        ("Silty Clay Loam", "SICL"),
        ("Sandy Clay", "SC"),
        ("Silty Clay", "SIC"),
        ("Clay", "C"),
    ],
)
def test_sltx_code_covers_every_usda_class(texture: str, code: str) -> None:
    """Every USDA class the ``surface_texture`` classifier emits maps to a code.

    Spelled-out (with spaces), space-stripped, and lower-cased forms all
    resolve — the writer strips spaces before the .SOL, and the classifier
    emits the spaced form.
    """
    assert _dssat_sltx_code(texture) == code
    assert _dssat_sltx_code(texture.replace(" ", "")) == code
    assert _dssat_sltx_code(texture.lower()) == code
    # every code is short enough to never overflow the fixed-width column.
    assert len(code) <= 4


def test_mapping_matches_the_twelve_usda_classes() -> None:
    """The map is exactly the USDA 12-class triangle — no more, no less."""
    assert len(_DSSAT_SLTX_CODES) == 12
    assert set(_DSSAT_SLTX_CODES.values()) == {
        "S", "LS", "SL", "L", "SIL", "SI", "SCL", "CL", "SICL", "SC", "SIC", "C",
    }


def test_sltx_code_raises_on_unmapped_class() -> None:
    with pytest.raises(ValueError, match="Unmapped soil texture class"):
        _dssat_sltx_code("Regolith")


def test_sltx_code_raises_on_none_and_empty() -> None:
    # absent texture (e.g. a layerless profile) fails loud, not a silent name.
    with pytest.raises(ValueError, match="Unmapped soil texture class"):
        _dssat_sltx_code(None)
    with pytest.raises(ValueError, match="Unmapped soil texture class"):
        _dssat_sltx_code("")


# ── the crux: SLTX + depth at DSSAT FORMAT 5030 columns (26-30 / 32-36) ─


def _region() -> Region:
    return Region(
        name="Wami", country="Tanzania", country_iso3="TZA",
        bounds=BoundingBox(minx=37.0, miny=-7.0, maxx=38.0, maxy=-6.0),
    )


def _profile(sand: float, clay: float) -> SoilProfile:
    return SoilProfile(
        profile_id="P", lat=-6.5, lon=37.5, source="isda",
        layers=[SoilLayer(
            depth_top=0.0, depth_bottom=0.5, sand=sand, clay=clay,
            silt=100.0 - sand - clay, organic_carbon=0.6, bulk_density=1.4,
            ph=6.5, field_capacity=0.28, wilting_point=0.15,
        )],
    )


def _star_line(sol_text: str, profile_name: str) -> str:
    for line in sol_text.splitlines():
        if line.startswith(f"*{profile_name}"):
            return line
    raise AssertionError(f"no *{profile_name} header line in .SOL")


def test_sandyclayloam_renders_sltx_code_at_dssat_column(tmp_path) -> None:
    """The census-dominant class emits ``SCL`` in the DSSAT A5 SLTX field (cols 26-30)."""
    scl = _profile(sand=55.0, clay=30.0)  # -> Sandy Clay Loam
    assert scl.surface_texture == "Sandy Clay Loam"  # premise guard
    out = tmp_path / "TZ.SOL"
    write_dssat_sol(out, {4: scl}, country_code="TZ", region=_region())
    line = _star_line(out.read_text(), "TZ00000004")
    assert "SandyClayLoam" not in line  # the overflowing spelled-out name is gone
    assert line[25:30] == "SCL  "  # SLTX A5 field at cols 26-30 (0-indexed 25:30)


@pytest.mark.parametrize(
    "sand, clay, texture, code",
    [
        (40.0, 22.0, "Loam", "L"),               # 1-char code
        (60.0, 10.0, "Sandy Loam", "SL"),        # 2-char code
        (55.0, 30.0, "Sandy Clay Loam", "SCL"),  # 3-char code (census-dominant)
        (20.0, 30.0, "Silty Clay Loam", "SICL"),  # 4-char code (misread risk)
    ],
)
def test_sltx_and_depth_at_dssat_format_5030_columns(
    tmp_path, sand: float, clay: float, texture: str, code: str
) -> None:
    """SLTX at cols 26-30 (A5) and depth at cols 32-36 (F5.0), per FORMAT 5030.

    DSSAT ``IPSOIL_Inp.for:627`` reads ``1X,A10,2X,A11,1X,A5,1X,F5.0,1X,A50``
    → SLTX at cols 26-30, depth at cols 32-36. Spans code widths 1-4 chars
    incl. the 4-char SICL a too-early field would misread (as ``L``). The depth
    column must not move regardless of the code length — that was the bug.
    """
    prof = _profile(sand=sand, clay=clay)
    assert prof.surface_texture == texture  # premise guard
    out = tmp_path / "TZ.SOL"
    write_dssat_sol(out, {1: prof}, country_code="TZ", region=_region())
    line = _star_line(out.read_text(), "TZ00000001")
    assert line[25:30] == f"{code:<5}"  # SLTX A5, cols 26-30
    assert line[31:36] == "   50"       # depth F5.0, cols 32-36 (50 cm)
    # the 1X separators DSSAT skips at cols 25 / 31 / 37 (0-indexed 24/30/36).
    assert line[24] == " " and line[30] == " " and line[36] == " "
    assert texture.replace(" ", "") not in line  # no spelled-out class anywhere
