"""The resolver ACEA's harvested-area clip uses, exposed unchanged: the same patterns in the same
order, the same pick, plus every match so a caller can see an ambiguous wildcard."""
from __future__ import annotations

import pytest

# The clip's pattern list at the base 4e763a1 (acea/translator.py:1774-1781), for MAIZ / FAO 56 / A.
_BASE_PATTERNS = [
    "spam2020V2r0_global_H_MAIZ_A.tif",
    "spam2020_V2r0_global_H_MAIZ_A.tif",
    "spam2020V2r0_global_H_56_A.tif",
    "spam2020v2r0_global_H_MAIZ_A.tif",
    "spam2010V1r0_global_H_MAIZ_A.tif",
    "*MAIZ*_A.tif",
]


def _touch(directory, *names):
    for name in names:
        (directory / name).write_bytes(b"layer")


def test_the_pattern_list_is_unchanged():
    from prismpy.translators.acea.translator import acea_spam_input_patterns

    assert acea_spam_input_patterns("MAIZ", 56, "A") == _BASE_PATTERNS


def test_the_pick_and_its_matches(tmp_path):
    from prismpy.translators.acea.translator import resolve_acea_spam_input

    _touch(tmp_path, "spam2020_V2r0_global_H_MAIZ_A.tif", "spam2020_maize.tif")
    got = resolve_acea_spam_input(tmp_path, "Maize", "A")
    assert got.pick == tmp_path / "spam2020_V2r0_global_H_MAIZ_A.tif" and got.matches == [got.pick]


def test_the_first_matching_pattern_wins(tmp_path):
    from prismpy.translators.acea.translator import resolve_acea_spam_input

    _touch(tmp_path, "spam2020_V2r0_global_H_MAIZ_A.tif", "spam2020V2r0_global_H_MAIZ_A.tif")
    got = resolve_acea_spam_input(tmp_path, "Maize", "A")
    assert got.pick.name == "spam2020V2r0_global_H_MAIZ_A.tif" and len(got.matches) == 1


def test_an_ambiguous_wildcard_is_visible(tmp_path):
    from prismpy.translators.acea.translator import resolve_acea_spam_input

    _touch(tmp_path, "a_MAIZ_1_A.tif", "b_MAIZ_2_A.tif")
    got = resolve_acea_spam_input(tmp_path, "Maize", "A")
    assert len(got.matches) == 2 and got.pick in got.matches


@pytest.mark.parametrize("crop", ["Teff", ""])
def test_a_crop_without_a_spam_code_resolves_nothing(tmp_path, crop):
    from prismpy.translators.acea.translator import resolve_acea_spam_input

    _touch(tmp_path, "spam2020_V2r0_global_H_MAIZ_A.tif")
    got = resolve_acea_spam_input(tmp_path, crop, "A")
    assert got.pick is None and got.matches == []
