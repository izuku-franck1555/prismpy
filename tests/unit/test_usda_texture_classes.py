"""The soil texture class is the USDA (NRCS) texture triangle's, edge for edge.

Each edge is crossed along a path ``t -> (sand, clay)`` at -0.5, -0.01, on the edge, +0.01 and +0.5,
so both whole and fractional inputs land in the class the NRCS definitions give them.
"""
from __future__ import annotations

import pytest

from prismpy.models.soil import SoilProfile
from prismpy.translators._shared.dssat_sol_writer import _DSSAT_SLTX_CODES, _dssat_sltx_code

texture = SoilProfile._get_texture_class

THE_TWELVE = {
    "Sand", "Loamy Sand", "Sandy Loam", "Loam", "Silt Loam", "Silt", "Sandy Clay Loam",
    "Clay Loam", "Silty Clay Loam", "Sandy Clay", "Silty Clay", "Clay",
}
TRIANGLE = [(float(sand), float(clay)) for sand in range(101) for clay in range(101 - sand)]

# (edge, the path across it, t on the edge, the class below it, on it, above it)
EDGES = [
    ("clay 7 at silt 45", lambda t: (55 - t, t), 7, "Sandy Loam", "Loam", "Loam"),
    ("clay 12 at silt 82", lambda t: (18 - t, t), 12, "Silt", "Silt Loam", "Silt Loam"),
    ("clay 20 at silt 20", lambda t: (80 - t, t), 20, "Sandy Loam", "Sandy Clay Loam", "Sandy Clay Loam"),
    ("clay 27 at silt 30", lambda t: (70 - t, t), 27, "Loam", "Clay Loam", "Clay Loam"),
    ("clay 27 at silt 55", lambda t: (45 - t, t), 27, "Silt Loam", "Silty Clay Loam", "Silty Clay Loam"),
    ("clay 35 at silt 10", lambda t: (90 - t, t), 35, "Sandy Clay Loam", "Sandy Clay", "Sandy Clay"),
    ("clay 40 at sand 30", lambda t: (30, t), 40, "Clay Loam", "Clay", "Clay"),
    ("clay 40 at sand 10", lambda t: (10, t), 40, "Silty Clay Loam", "Silty Clay", "Silty Clay"),
    ("sand 20 at clay 33", lambda t: (t, 33), 20, "Silty Clay Loam", "Silty Clay Loam", "Clay Loam"),
    ("sand 45 at clay 30", lambda t: (t, 30), 45, "Clay Loam", "Clay Loam", "Sandy Clay Loam"),
    ("sand 45 at clay 45", lambda t: (t, 45), 45, "Clay", "Clay", "Sandy Clay"),
    ("sand 52 at clay 15", lambda t: (t, 15), 52, "Loam", "Loam", "Sandy Loam"),
    ("silt 28 at clay 22", lambda t: (78 - t, 22), 28, "Sandy Clay Loam", "Loam", "Loam"),
    ("silt 40 at clay 50", lambda t: (50 - t, 50), 40, "Clay", "Silty Clay", "Silty Clay"),
    ("silt 50 at clay 15", lambda t: (85 - t, 15), 50, "Loam", "Silt Loam", "Silt Loam"),
    ("silt 50 at clay 5", lambda t: (95 - t, 5), 50, "Sandy Loam", "Silt Loam", "Silt Loam"),
    ("silt 80 at clay 5", lambda t: (95 - t, 5), 80, "Silt Loam", "Silt", "Silt"),
    ("silt + 1.5 clay = 15 at clay 4", lambda t: (96 - t, 4), 9, "Sand", "Loamy Sand", "Loamy Sand"),
    ("silt + 2 clay = 30 at clay 10", lambda t: (90 - t, 10), 10, "Loamy Sand", "Sandy Loam", "Sandy Loam"),
    ("silt + 2 clay = 30 at clay 3", lambda t: (97 - t, 3), 24, "Loamy Sand", "Sandy Loam", "Sandy Loam"),
]


@pytest.mark.parametrize("name, path, edge, below, on, above", EDGES, ids=[row[0] for row in EDGES])
def test_each_edge_puts_both_sides_in_their_nrcs_class(name, path, edge, below, on, above) -> None:
    for offset, expected in ((-0.5, below), (-0.01, below), (0.0, on), (0.01, above), (0.5, above)):
        assert texture(*path(edge + offset)) == expected, (name, offset, path(edge + offset))


@pytest.mark.parametrize("sand, clay, usda", [
    (20, 10, "Silt Loam"),         # was "Silt"
    (25, 22, "Silt Loam"),         # was "Loam"
    (50, 25, "Sandy Clay Loam"),   # was "Loam"
    (75, 12, "Sandy Loam"),        # was "Loamy Sand"
    (50, 37, "Sandy Clay"),        # was "Sandy Clay Loam"
    (18, 38, "Silty Clay Loam"),   # was "Clay Loam"
    (86, 9, "Loamy Sand"),         # was "Sand"
    (50, 5, "Sandy Loam"),         # was "Loam"
])
def test_the_ranges_the_old_triangle_mislabelled(sand, clay, usda) -> None:
    assert texture(sand, clay) == usda


def test_exactly_one_rule_holds_at_every_whole_percent_of_the_triangle() -> None:
    from prismpy.models.soil import _USDA_TEXTURE_RULES
    for sand, clay in TRIANGLE:
        silt = 100 - sand - clay
        holding = [name for name, rule in _USDA_TEXTURE_RULES if rule(sand, silt, clay)]
        assert holding == [texture(sand, clay)], (sand, clay, holding)


def test_the_classes_are_the_same_twelve_strings_and_each_has_its_sltx_code() -> None:
    produced = {texture(sand, clay) for sand, clay in TRIANGLE}
    assert produced == THE_TWELVE
    assert {name.replace(" ", "").lower() for name in produced} == set(_DSSAT_SLTX_CODES)
    assert all(_dssat_sltx_code(name) for name in produced)
