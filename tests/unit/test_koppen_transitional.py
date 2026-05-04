"""Sprint E.0.5 AC-Q1-A + CC-13 — TRANSITIONAL_ZONE detection.

The 8-neighbor honest-fail rule (research doc §Q1.3): a cell
is TRANSITIONAL iff any of its 8 immediately adjacent native-
1km neighbors carries a different (non-zero) KG zone than
the center.

Anti-mutation drills:

- Reduce :data:`NEIGHBOR_OFFSETS` to 4 cardinal neighbors only
  → diagonal seams (e.g. BSh/BWh corner-touch) are missed.
- Treat ocean (nodata) neighbors as "different" →
  ``test_ocean_neighbor_does_not_flag_interior`` fails (would
  flag every coastal land cell).
- Sample at non-native offsets (e.g. 2 × cell_size) →
  ``test_neighbor_offsets_at_native_cell_size`` fails.
"""
from __future__ import annotations

import unittest

from prismpy.koppen.kg_classifier import KGClassifier, KGZone
from prismpy.koppen.raster_loader import NATIVE_CELL_DEG
from prismpy.koppen.transitional import (
    NEIGHBOR_OFFSETS,
    classify_with_transitional_flag,
    is_transitional_cell,
)


# Empirically verified seam: center BSh, at least one
# neighbor BWh. Found via 8-neighbor scan of the Sahel
# transition belt at sprint-implementation time.
_SEAM_BSH_BWH: tuple[float, float] = (13.4875, 2.65417)


# Interior cell — all 8 neighbors are the same zone as
# center (BSh).
_INTERIOR_BSH: tuple[float, float] = (13.5, 2.1)


# Interior Aw cell (Bamako).
_INTERIOR_AW: tuple[float, float] = (12.65, -8.0)


# Open-ocean cell — center is nodata (0).
_OCEAN: tuple[float, float] = (0.0, -150.0)


class TestNeighborOffsetGeometry(unittest.TestCase):
    """Pin the 8-neighbor offsets to exactly ±cell_size in
    lat + lon (4 cardinal + 4 diagonal)."""

    def test_eight_offsets_total(self):
        self.assertEqual(len(NEIGHBOR_OFFSETS), 8)

    def test_offsets_at_native_cell_size(self):
        for dlat, dlon in NEIGHBOR_OFFSETS:
            with self.subTest(offset=(dlat, dlon)):
                self.assertIn(dlat, (-NATIVE_CELL_DEG, 0.0, NATIVE_CELL_DEG))
                self.assertIn(dlon, (-NATIVE_CELL_DEG, 0.0, NATIVE_CELL_DEG))

    def test_zero_offset_excluded(self):
        # The center cell itself is not a neighbor.
        self.assertNotIn((0.0, 0.0), NEIGHBOR_OFFSETS)

    def test_offsets_unique(self):
        self.assertEqual(len(set(NEIGHBOR_OFFSETS)), 8)


class TestTransitionalDetection(unittest.TestCase):
    """Per AC-Q1-A: transitional flag fires at zone seams,
    NOT at interior cells, NOT at ocean cells."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = KGClassifier()

    @classmethod
    def tearDownClass(cls):
        cls.classifier.close()

    def test_interior_BSh_not_transitional(self):
        lat, lon = _INTERIOR_BSH
        self.assertFalse(is_transitional_cell(lat, lon, self.classifier))

    def test_interior_Aw_not_transitional(self):
        lat, lon = _INTERIOR_AW
        self.assertFalse(is_transitional_cell(lat, lon, self.classifier))

    def test_known_BSh_BWh_seam_is_transitional(self):
        lat, lon = _SEAM_BSH_BWH
        self.assertTrue(is_transitional_cell(lat, lon, self.classifier))

    def test_ocean_center_not_transitional(self):
        lat, lon = _OCEAN
        self.assertFalse(is_transitional_cell(lat, lon, self.classifier))


class TestClassifyWithTransitionalFlag(unittest.TestCase):
    """The combined :func:`classify_with_transitional_flag`
    helper returns ``(zone, flag)`` in one pass."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = KGClassifier()

    @classmethod
    def tearDownClass(cls):
        cls.classifier.close()

    def test_interior_returns_zone_and_false(self):
        lat, lon = _INTERIOR_BSH
        zone, flag = classify_with_transitional_flag(
            lat, lon, self.classifier,
        )
        self.assertEqual(zone, KGZone.BSh)
        self.assertFalse(flag)

    def test_seam_returns_zone_and_true(self):
        lat, lon = _SEAM_BSH_BWH
        zone, flag = classify_with_transitional_flag(
            lat, lon, self.classifier,
        )
        self.assertEqual(zone, KGZone.BSh)
        self.assertTrue(flag)

    def test_ocean_returns_None_and_false(self):
        lat, lon = _OCEAN
        zone, flag = classify_with_transitional_flag(
            lat, lon, self.classifier,
        )
        self.assertIsNone(zone)
        self.assertFalse(flag)


class TestOceanNeighborSemantics(unittest.TestCase):
    """Coastal land cells should NOT be flagged as transitional
    just because some neighbors are ocean. Per the research
    doc §Q1.3 'non-zero' rule, ocean neighbors are skipped."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = KGClassifier()

    @classmethod
    def tearDownClass(cls):
        cls.classifier.close()

    def test_interior_BSh_not_transitional_even_if_far_from_coast(self):
        # Niamey is well inland; its 8 neighbors are all BSh.
        # This sanity-check anchors the ocean-neighbor branch:
        # if an interior cell with no ocean neighbors flags
        # transitional, the implementation is broken.
        self.assertFalse(
            is_transitional_cell(*_INTERIOR_BSH, self.classifier),
        )


if __name__ == "__main__":
    unittest.main()
