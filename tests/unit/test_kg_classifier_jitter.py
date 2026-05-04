"""Sprint E.0.5 AC-Q1-A — sub-pixel jitter stability.

The jitter convention is **0.49 × native cell size** in
each axis. At 1/120° cell size this is ~0.00408° (~ 460 m
at the equator). Strictly less than half the cell size, so
the jittered point stays inside the original pixel under
rasterio's nearest-neighbor IEEE-754 floor convention. The
8 neighbor pixels at offsets ±cell_size from the jittered
center also stay in the same neighbor pixels for the same
reason.

Acceptance per AC-Q1-A: classification (and the transitional
flag) must be stable under this jitter for any cell.

Anti-mutation drills:

- Replace 0.49 → 0.51 → strict-inside-cell guarantee breaks;
  jittered points cross pixel boundaries; classification flips.
- Replace nearest-neighbor with bilinear interpolation in the
  loader → classification changes around boundaries even
  without any jitter.
"""
from __future__ import annotations

import unittest

from prismpy.koppen.kg_classifier import KGClassifier, KGZone
from prismpy.koppen.raster_loader import NATIVE_CELL_DEG
from prismpy.koppen.transitional import is_transitional_cell


# Jitter step strictly smaller than half-cell so the jittered
# point stays in the original pixel.
_JITTER = 0.49 * NATIVE_CELL_DEG


# 9-point jitter pattern: base + 8 corners (±jitter on each axis).
def _jitter_pattern(lat: float, lon: float) -> list[tuple[float, float]]:
    return [
        (lat,           lon),
        (lat + _JITTER, lon),
        (lat - _JITTER, lon),
        (lat,           lon + _JITTER),
        (lat,           lon - _JITTER),
        (lat + _JITTER, lon + _JITTER),
        (lat + _JITTER, lon - _JITTER),
        (lat - _JITTER, lon + _JITTER),
        (lat - _JITTER, lon - _JITTER),
    ]


# Empirically verified interior fixtures — same zone all
# 8 jitter corners.
_INTERIOR_FIXTURES: tuple[tuple[str, float, float, KGZone], ...] = (
    ("Niamey BSh interior", 13.5,    2.1,    KGZone.BSh),
    ("Bamako Aw interior",  12.65,  -8.0,    KGZone.Aw),
    ("Mid-Sahara BWh",      23.0,    8.0,    KGZone.BWh),
    ("Antarctica EF",      -85.0,    0.0,    KGZone.EF),
    ("Eastern US Cfa",      35.0,  -85.0,    KGZone.Cfa),
)


# Empirically verified seam fixture — center cell at the
# BSh/BWh boundary near Niamey. Found by 8-neighbor scan of
# the Sahel transition belt at sprint-implementation time.
# Center is BSh; at least one neighbor is BWh, so transitional.
_SEAM_FIXTURE: tuple[str, float, float, KGZone] = (
    "Sahel BSh/BWh seam",  13.4875,  2.65417,  KGZone.BSh,
)


_OCEAN_FIXTURE: tuple[float, float] = (0.0, -150.0)


class TestSubpixelJitterStability(unittest.TestCase):
    """Per AC-Q1-A: jitter ≤ 0.49 × native_cell_size keeps
    the lookup point strictly inside the original pixel; both
    classify() output and the transitional flag must be stable."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = KGClassifier()

    @classmethod
    def tearDownClass(cls):
        cls.classifier.close()

    def test_jitter_strictly_less_than_half_cell(self):
        # Sanity: 0.49 < 0.5 keeps the jittered point inside
        # the original pixel under rasterio's nearest-neighbor
        # IEEE-754 floor convention.
        self.assertLess(_JITTER, 0.5 * NATIVE_CELL_DEG)
        self.assertGreater(_JITTER, 0.0)

    def test_interior_fixtures_stable_under_jitter(self):
        for name, lat, lon, expected in _INTERIOR_FIXTURES:
            with self.subTest(fixture=name):
                for jitter_lat, jitter_lon in _jitter_pattern(lat, lon):
                    with self.subTest(jitter=(jitter_lat, jitter_lon)):
                        self.assertEqual(
                            self.classifier.classify(jitter_lat, jitter_lon),
                            expected,
                        )

    def test_seam_classification_stable_under_jitter(self):
        # The center cell value (the BSh "side" of the seam)
        # is what classify() returns. Even at a seam, the
        # center sample stays the same under jitter < 0.5*CELL.
        name, lat, lon, expected = _SEAM_FIXTURE
        for jitter_lat, jitter_lon in _jitter_pattern(lat, lon):
            with self.subTest(jitter=(jitter_lat, jitter_lon)):
                self.assertEqual(
                    self.classifier.classify(jitter_lat, jitter_lon),
                    expected,
                )

    def test_seam_transitional_flag_stable_under_jitter(self):
        # Per AC-Q1-A the TRANSITIONAL_ZONE classification is
        # stable under the jitter convention because the 8-
        # neighbor offsets (±CELL) from the jittered center
        # land in the same 8 neighbor pixels.
        _name, lat, lon, _expected = _SEAM_FIXTURE
        for jitter_lat, jitter_lon in _jitter_pattern(lat, lon):
            with self.subTest(jitter=(jitter_lat, jitter_lon)):
                self.assertTrue(
                    is_transitional_cell(
                        jitter_lat, jitter_lon, self.classifier,
                    ),
                )

    def test_interior_transitional_flag_stable_under_jitter(self):
        # Interior cells (Niamey BSh) are NOT transitional;
        # the flag stays False under jitter.
        for name, lat, lon, _zone in _INTERIOR_FIXTURES:
            with self.subTest(fixture=name):
                for jitter_lat, jitter_lon in _jitter_pattern(lat, lon):
                    with self.subTest(jitter=(jitter_lat, jitter_lon)):
                        self.assertFalse(
                            is_transitional_cell(
                                jitter_lat, jitter_lon, self.classifier,
                            ),
                        )

    def test_ocean_stable_under_jitter(self):
        lat, lon = _OCEAN_FIXTURE
        for jitter_lat, jitter_lon in _jitter_pattern(lat, lon):
            with self.subTest(jitter=(jitter_lat, jitter_lon)):
                self.assertIsNone(
                    self.classifier.classify(jitter_lat, jitter_lon),
                )


if __name__ == "__main__":
    unittest.main()
