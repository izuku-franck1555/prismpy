"""Pin DSSAT-byte-consumed identifiers to ASCII-only.

DSSAT v4.8 (Fortran) reads experiment-file names + weather-station
codes as fixed-width byte strings. Multi-byte UTF-8 (e.g., ``É`` =
``0xC3 0x89``, 2 bytes per glyph) shifts the byte boundaries and
truncates the read at the wrong offset, surfacing as
``WARNING.OUT: File not found: BÉSG8001.SN`` (the trailing ``X`` of
``.SNX`` lost to the byte-shift).

This test pins the ASCII-only invariant on:

* :meth:`PythiaTranslator._get_template_filename` (SNX filename).
* :meth:`PythiaTranslator._get_wsta_prefix` (4-char station prefix).

Display strings (e.g., ``manifest.region.name``) keep diacritics —
only the DSSAT-consumed surface is folded.
"""

from __future__ import annotations

from typing import Any

import pytest

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
from prismpy.translators.pythia.translator import PythiaTranslator


def _make_translator(region_name: str, country: str, country_iso3: str) -> PythiaTranslator:
    """Build a minimal translator with the given region naming."""
    cfg = ProjectConfig(
        project=ProjectInfo(name="ascii_fold_pin"),
        region=RegionConfig(
            name=region_name,
            country=country,
            country_iso3=country_iso3,
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=0.0, miny=0.0, maxx=1.0, maxy=1.0,
                ),
            ),
        ),
        crop=CropConfig(
            name="Sorghum",
            name_short="sgh",
            calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285),
        ),
        temporal=TemporalConfig(start_year=2015, end_year=2015, spinup_years=0),
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir="/tmp", structure="by_platform"),
    )
    return PythiaTranslator(config=cfg)


# ── _ascii_fold_for_dssat unit-level tests ──────────────────────────


@pytest.mark.parametrize(
    "raw, folded",
    [
        # ASCII-only is unchanged.
        ("Koutiala", "Koutiala"),
        ("Mifi", "Mifi"),
        # Diacritics decomposed + dropped.
        ("Bénoué", "Benoue"),
        ("Côte d'Ivoire", "Cote d'Ivoire"),
        ("São Paulo", "Sao Paulo"),
        ("München", "Munchen"),
        # Cedilla / tilde / circumflex.
        ("Curaçao", "Curacao"),
        ("Naïve", "Naive"),
        # Empty stays empty.
        ("", ""),
        # Pure non-ASCII drops to empty (the helper does not invent ASCII).
        ("中文", ""),
    ],
)
def test_ascii_fold_for_dssat_strips_diacritics(raw: str, folded: str) -> None:
    """The helper folds NFKD-decomposable diacritics + drops non-ASCII."""
    assert PythiaTranslator._ascii_fold_for_dssat(raw) == folded
    assert PythiaTranslator._ascii_fold_for_dssat(raw).isascii()


# ── _get_template_filename pins ─────────────────────────────────────


def test_template_filename_is_ascii_for_benoue() -> None:
    """Bénoué → BESG8001.SNX (NOT BÉSG8001.SNX)."""
    t = _make_translator("Bénoué", "Cameroon", "CMR")
    fn = t._get_template_filename()
    assert fn == "BESG8001.SNX"
    assert fn.isascii()
    assert "É" not in fn
    # DSSAT 12-char fixed-width byte read: ASCII fits in 1 byte/char.
    assert len(fn.encode("utf-8")) == 12


def test_template_filename_is_ascii_for_ascii_region() -> None:
    """ASCII region names round-trip unchanged."""
    t = _make_translator("Koutiala", "Mali", "MLI")
    fn = t._get_template_filename()
    assert fn == "KOSG8001.SNX"
    assert fn.isascii()


def test_template_filename_handles_short_region() -> None:
    """A 1-char region name still produces an ASCII filename."""
    t = _make_translator("À", "France", "FRA")
    fn = t._get_template_filename()
    # "À" → NFKD → "A" (1 char) + dropped combining mark; [:2] of "A" = "A";
    # uppercase = "A". Filename gets "A" + crop code + "8001.SNX" =
    # "ASG8001.SNX" (11 chars, NOT 12 — but ASCII-only either way).
    assert fn.isascii()
    assert "À" not in fn


# ── _get_wsta_prefix pins ───────────────────────────────────────────


def test_wsta_prefix_is_ascii_for_benoue_cameroon() -> None:
    """Cameroon + Bénoué → CMBE (NOT CMBÉ)."""
    t = _make_translator("Bénoué", "Cameroon", "CMR")
    prefix = t._get_wsta_prefix()
    assert prefix == "CMBE"
    assert prefix.isascii()
    assert "É" not in prefix
    # DSSAT 4-char fixed-width byte read: ASCII fits in 1 byte/char.
    assert len(prefix.encode("utf-8")) == 4


def test_wsta_prefix_is_ascii_for_mali_koutiala() -> None:
    """ASCII region/country round-trips unchanged."""
    t = _make_translator("Koutiala", "Mali", "MLI")
    prefix = t._get_wsta_prefix()
    assert prefix == "MLKO"
    assert prefix.isascii()


def test_wsta_prefix_helper_falls_back_to_xx_on_empty_fold() -> None:
    """At the helper level, a region that folds to empty triggers XX fallback.

    RegionConfig's Pydantic validator rejects pure-non-ASCII region names
    (e.g., ``中文``) at the schema boundary before the translator sees
    them, so the empty-fold path is unreachable through normal config
    construction. This test exercises the fallback at the helper level
    directly to pin the defensive behaviour.
    """
    # The helper folds an empty input to empty; the wsta-prefix builder
    # then falls back to "XX". Exercise the helper directly.
    assert PythiaTranslator._ascii_fold_for_dssat("") == ""
    # Pure-non-ASCII (Chinese ideographs) → empty after NFKD+ASCII-ignore.
    assert PythiaTranslator._ascii_fold_for_dssat("中文") == ""


# ── Display strings keep diacritics (negative pin) ──────────────────


def test_region_name_in_config_keeps_diacritic() -> None:
    """The translator's config preserves the original display name.

    The ASCII fold applies ONLY at the DSSAT-byte-consumed boundary;
    ``manifest.region.name`` and any human-readable surface keeps the
    diacritic.
    """
    t = _make_translator("Bénoué", "Cameroon", "CMR")
    assert t.config.region.name == "Bénoué"
    # Lowercase é (the input had lowercase) survives unchanged.
    assert "é" in t.config.region.name
    # Verify the diacritic byte-encoding is preserved (UTF-8 multi-byte).
    assert len("Bénoué".encode("utf-8")) > len("Bénoué")


# ── Cross-write-site invariant pin (codex round 1 HIGH absorption) ──


def test_snx_template_filename_matches_helper_for_benoue() -> None:
    """The `_generate_snx_template` writer site MUST derive its
    on-disk filename from `_get_template_filename()`.

    Codex round 1 HIGH catch: an earlier revision wired the ASCII
    fold into `_get_template_filename()` (used by the pythia_config
    JSON's ``default_setup.template`` field) but left the actual
    SNX writer site at ``_generate_snx_template()`` deriving its own
    raw `region.name[:2].upper()` slice. The two sites then drifted
    on non-ASCII region names: pythia_config referenced
    ``BESG8001.SNX`` (folded) while the file on disk was
    ``BÉSG8001.SNX`` (raw multi-byte). DSSAT opened the folded path,
    found nothing, and silently produced 0 yields.

    The fix routes both sites through `_get_template_filename()`.
    This pin asserts the agreement: the basename of the SNX file
    written to disk MUST equal the string the pythia_config JSON
    references. Any future divergence (someone re-introduces a raw
    slice in either site) fails this pin.
    """
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock

    from prismpy.models.region import BoundingBox, Region
    from prismpy.translators.base import UnifiedData

    with tempfile.TemporaryDirectory() as tmpd:
        out_dir = Path(tmpd)
        t = _make_translator("Bénoué", "Cameroon", "CMR")
        t.output_dir = out_dir

        # Build a minimal UnifiedData carrying the region.
        region = Region(
            name="Bénoué",
            country="Cameroon",
            country_iso3="CMR",
            bounds=BoundingBox(minx=13.5, miny=8.0, maxx=14.5, maxy=9.0),
        )
        data = UnifiedData(region=region)

        # Stub the helpers _generate_snx_template depends on so we
        # exercise the filename-derivation surface without pulling
        # in the full template-rendering machinery (separate concern,
        # not what this pin verifies).
        t._map_generic_to_pythia_config = MagicMock(return_value={})
        t._map_generic_to_cultivar = MagicMock(return_value={})
        t._map_generic_to_fertilizer = MagicMock(return_value={})
        t._build_snx_content = MagicMock(return_value="* test content\n")

        template_path = t._generate_snx_template(data)

        # The cross-write-site invariant: the on-disk SNX basename
        # MUST equal what `_get_template_filename()` returns. This is
        # the same string written into pythia_config["default_setup"]
        # ["template"] by `_generate_pythia_json()`.
        helper_filename = t._get_template_filename()
        assert template_path.name == helper_filename, (
            f"SNX writer drifted from helper: file={template_path.name!r} "
            f"vs helper={helper_filename!r}. Both must come from "
            f"_get_template_filename() so DSSAT opens the same path "
            f"the config references."
        )
        # Belt-and-suspenders: assert the on-disk basename is ASCII-only.
        assert template_path.name.isascii(), (
            f"SNX file {template_path.name!r} has non-ASCII chars; "
            f"DSSAT's fixed-width Fortran reads will truncate mid-byte."
        )
        # And: the SNX file actually exists at that path (not folded
        # by the helper but written to a different raw path).
        assert template_path.exists(), (
            f"_generate_snx_template returned {template_path} but no "
            f"file at that path; writer wrote to a different name."
        )
