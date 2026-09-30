"""The resolver ACEA's harvested-area clip uses, exposed unchanged: the same patterns in the same
order, the same pick (the first match of the first matching pattern, in glob order, never sorted),
plus every match of that pattern so a caller can see an ambiguous layer."""
from __future__ import annotations

from fnmatch import fnmatch

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


def _resolve(directory, tech="A", crop="Maize"):
    from prismpy.translators.acea.translator import resolve_acea_spam_input

    return resolve_acea_spam_input(directory, crop, tech)


def test_the_pattern_list_is_unchanged():
    from prismpy.translators.acea.translator import acea_spam_input_patterns

    assert acea_spam_input_patterns("MAIZ", 56, "A") == _BASE_PATTERNS


def test_the_staged_v2r0_set_resolves_one_file_per_technology(tmp_path):
    _touch(tmp_path, *(f"spam2020_V2r0_global_H_MAIZ_{t}.tif" for t in "RIA"), "spam2020_maize.tif")
    for tech in "RIA":
        got = _resolve(tmp_path, tech)
        assert got.pick == tmp_path / f"spam2020_V2r0_global_H_MAIZ_{tech}.tif" and got.matches == [got.pick]


def test_the_first_matching_pattern_wins(tmp_path):
    _touch(tmp_path, "spam2020_V2r0_global_H_MAIZ_A.tif", "spam2020V2r0_global_H_MAIZ_A.tif")
    got = _resolve(tmp_path)
    assert got.pick.name == "spam2020V2r0_global_H_MAIZ_A.tif" and len(got.matches) == 1


def test_a_2010_v1r0_file_alone_is_picked(tmp_path):
    _touch(tmp_path, "spam2010V1r0_global_H_MAIZ_A.tif")
    assert _resolve(tmp_path).pick.name == "spam2010V1r0_global_H_MAIZ_A.tif"


def test_two_files_for_one_pattern_are_both_listed_in_glob_order(tmp_path):
    _touch(tmp_path, "b_MAIZ_2_A.tif", "a_MAIZ_1_A.tif")
    got = _resolve(tmp_path)
    assert got.matches == list(tmp_path.glob("*MAIZ*_A.tif")) and got.pick == got.matches[0]
    assert len(got.matches) == 2


def test_the_pick_follows_the_directory_order_never_a_sort(tmp_path, monkeypatch):
    names = ["b_MAIZ_2_A.tif", "c_MAIZ_3_A.tif", "a_MAIZ_1_A.tif"]  # neither ascending nor descending
    _touch(tmp_path, *names)

    def listing(directory, pattern):
        return (directory / name for name in names if fnmatch(name, pattern))

    monkeypatch.setattr(type(tmp_path), "glob", listing)
    got = _resolve(tmp_path)
    assert [p.name for p in got.matches] == names and got.pick.name == names[0]


@pytest.mark.parametrize("crop,names", [("Maize", ()), ("Teff", ("spam2020_V2r0_global_H_MAIZ_A.tif",))])
def test_nothing_resolves_without_a_matching_file_or_a_spam_code(tmp_path, crop, names):
    _touch(tmp_path, *names)
    got = _resolve(tmp_path, crop=crop)
    assert got.pick is None and got.matches == []
