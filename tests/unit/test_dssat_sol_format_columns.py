"""DSSAT .SOL full-row column conformance (FORMAT 5030 / 5035 + header-driven).

Every ``.SOL`` row must land at the columns DSSAT's IPSOIL reader expects:

- the ``*``-profile header (``IPSOIL_Inp.for`` FORMAT 5030) and the site value
  row (FORMAT 5035) are read at FIXED columns — an off-by-one truncates a field;
- the ``@ SCOM`` surface row and ``@ SLB`` layer rows are read *header-driven*:
  DSSAT ``PARSE_HEADERS`` records each ``@``-header token's column span and reads
  each value from that same span, so every value must sit under its header token.

These tests assert ``line[a:b]`` for every field on every row type (not a
whitespace split), and read lat/lon back to prove the FORMAT 5035 fix.
"""
from __future__ import annotations

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.translators._shared.dssat_sol_writer import write_dssat_sol


def _parse_headers(line: str):
    """Port of DSSAT ``READS.for`` PARSE_HEADERS → ``[(name, c1, c2)]`` 1-indexed.

    Splits the ``@``-header on whitespace (multiple spaces as one); a token's
    column span runs from the char after the previous token's separating space
    to the token's last char — exactly the span DSSAT reads each data value from.
    """
    s = line.rstrip("\n")
    length = len(s.rstrip())
    if length <= 1:
        return []
    cols: list[list[int]] = []
    c1 = 1
    spaces = True
    i = 2
    while i <= length:
        ch = s[i - 1]
        if ch == "!":
            length = i - 1
            break
        elif ch == " ":
            if not spaces:
                cols.append([c1, i - 1])
                c1 = i + 1
                spaces = True
        else:
            spaces = False
        i += 1
    cols.append([c1, length])
    out = []
    for idx, (a, b) in enumerate(cols):
        start = 2 if idx == 0 else a  # first token skips the leading '@'
        out.append((s[start - 1:b].strip(), a, b))
    return out


def _render(tmp_path, lat: float = -6.042, lon: float = 37.708):
    region = Region(
        name="Wami", country="Tanzania", country_iso3="TZA",
        bounds=BoundingBox(minx=37.0, miny=-7.0, maxx=38.0, maxy=-6.0),
    )
    prof = SoilProfile(
        profile_id="P", lat=lat, lon=lon, source="isda",
        layers=[SoilLayer(
            depth_top=0.0, depth_bottom=0.5, sand=55.0, clay=30.0, silt=15.0,
            organic_carbon=0.6, bulk_density=1.4, ph=6.5,
            field_capacity=0.28, wilting_point=0.15,
        )],
    )
    out = tmp_path / "TZ.SOL"
    write_dssat_sol(out, {4: prof}, country_code="TZ", region=region)
    return out.read_text().splitlines()


def _block(lines):
    for i, ln in enumerate(lines):
        if ln.startswith("*TZ00000004"):
            return {
                "star": ln, "site_hdr": lines[i + 1], "site_val": lines[i + 2],
                "scom_hdr": lines[i + 3], "scom_val": lines[i + 4],
                "layer_hdr": lines[i + 5], "layer_row": lines[i + 6],
            }
    raise AssertionError("no *TZ00000004 profile block")


def test_soils_title_line(tmp_path):
    assert _render(tmp_path)[0].startswith("*SOILS:")


def test_star_header_format_5030_columns(tmp_path):
    s = _block(_render(tmp_path))["star"]
    assert s[1:11] == "TZ00000004"  # PEDON A10, cols 2-11
    assert s[24] == " "             # 1X, col 25
    assert s[25:30] == "SCL  "      # SLTX A5, cols 26-30
    assert s[30] == " "             # 1X, col 31
    assert s[31:36] == "   50"      # SLDP F5.0, cols 32-36
    assert s[36] == " "             # 1X, col 37


def test_site_row_format_5035_lat_lon_exact(tmp_path):
    """Site row lat/lon land at FORMAT 5035 cols 26-33 / 35-42 and read EXACTLY.

    Regression for the truncation bug: a country field one char too wide put
    lat at col 27, so DSSAT (reading cols 26-33) dropped the last digit
    (-6.042 -> -6.04). Reverting the field widths reds this.
    """
    v = _block(_render(tmp_path, lat=-6.042, lon=37.708))["site_val"]
    assert v[13:24].strip() == "TZA"     # SCOUNT A11, cols 14-24
    assert v[24] == " "                  # 1X, col 25
    assert float(v[25:33]) == -6.042     # SLAT F8.3, cols 26-33 — no truncation
    assert v[33] == " "                  # 1X, col 34
    assert float(v[34:42]) == 37.708     # SLONG F8.3, cols 35-42
    assert v[42] == " "                  # 1X, col 43


def test_scom_values_sit_under_headers(tmp_path):
    b = _block(_render(tmp_path))
    hdrs = _parse_headers(b["scom_hdr"])
    val = b["scom_val"]
    assert [h[0] for h in hdrs] == [
        "SCOM", "SALB", "SLU1", "SLDR", "SLRO", "SLNF", "SLPF", "SMHB", "SMPX", "SMKE",
    ]
    numeric = {"SALB", "SLU1", "SLDR", "SLRO", "SLNF", "SLPF"}
    for name, c1, c2 in hdrs:
        field = val[c1 - 1:c2]
        assert field.strip(), f"{name} empty under header cols {c1}:{c2}"
        if name in numeric:
            float(field)  # value is a self-contained number within its own span


def test_layer_values_sit_under_headers(tmp_path):
    b = _block(_render(tmp_path))
    hdrs = _parse_headers(b["layer_hdr"])
    row = b["layer_row"]
    assert [h[0] for h in hdrs] == [
        "SLB", "SLMH", "SLLL", "SDUL", "SSAT", "SRGF", "SSKS", "SBDM", "SLOC",
        "SLCL", "SLSI", "SLCF", "SLNI", "SLHW", "SLHB", "SCEC", "SADC",
    ]
    for name, c1, c2 in hdrs:
        field = row[c1 - 1:c2]
        assert field.strip(), f"{name} empty under header cols {c1}:{c2}"
        if name != "SLMH":  # SLMH is a char master-horizon code ("-9")
            float(field)  # numeric value reads within its own column span
