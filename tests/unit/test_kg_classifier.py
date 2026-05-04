"""Sprint E.0.5 AC-Q1-A — KG zone classifier on the bundled Beck 2023 raster.

Pins the 30-zone enum + the code-to-name mapping + a small
set of known-zone fixtures sampled against the bundled
1991-2020 1km Beck raster. Together with
``test_kg_classifier_jitter.py`` (sub-pixel stability) and
``test_koppen_transitional.py`` (8-neighbor honest-fail flag),
this covers the AC-Q1-A acceptance contract.

Anti-mutation drills:

- Drop a zone from :data:`KG_CODE_TO_NAME` →
  ``test_code_to_name_covers_1_to_30`` fails.
- Mis-map a code (e.g. ``6: "BSk"`` swap) →
  ``test_niamey_returns_BSh`` fails empirically.
- Replace the bundled raster with a different file →
  fixture verdict tests fail.
- Remove the bundled raster file →
  ``test_bundled_raster_exists`` fails.
"""
from __future__ import annotations

import unittest

from prismpy.koppen.kg_classifier import (
    KG_CODE_TO_ZONE,
    KGClassifier,
    KGZone,
)
from prismpy.koppen.raster_loader import (
    BECK_2023_LEGEND_PATH,
    BECK_2023_RASTER_PATH,
    KG_CODE_TO_NAME,
    KG_NAME_TO_CODE,
    NATIVE_CELL_DEG,
    NODATA_CODE,
)


# Empirically verified fixtures — coordinates whose KG zone
# at the bundled Beck 2023 1991-2020 raster matches the
# expected. Verified at sprint-implementation time via direct
# rasterio sampling; used as anti-mutation pins.
_INTERIOR_FIXTURES: tuple[tuple[str, float, float, KGZone], ...] = (
    ("Niamey, Niger",       13.5,    2.1,    KGZone.BSh),
    ("Bamako, Mali",        12.65,  -8.0,    KGZone.Aw),
    ("Mid-Sahara",          23.0,    8.0,    KGZone.BWh),
    ("Antarctica",         -85.0,    0.0,    KGZone.EF),
    ("Eastern US (Cfa)",    35.0,  -85.0,    KGZone.Cfa),
)


# Open-ocean coordinate; Beck raster nodata code 0.
_OCEAN_FIXTURE: tuple[float, float] = (0.0, -150.0)


class TestKGZoneEnum(unittest.TestCase):
    """The 30-member :class:`KGZone` enum is the canonical
    Köppen-Geiger vocabulary for prismpy."""

    def test_30_zones_total(self):
        self.assertEqual(len(list(KGZone)), 30)

    def test_zone_value_equals_name(self):
        # Subclassing (str, Enum) means ``KGZone.BSh.value == "BSh"``.
        for zone in KGZone:
            self.assertEqual(zone.value, zone.name)

    def test_zone_is_str(self):
        # Subclassing str lets KGZone members serialize as
        # JSON strings without explicit conversion.
        self.assertIsInstance(KGZone.BSh, str)
        self.assertEqual(str(KGZone.BSh), "KGZone.BSh")
        self.assertEqual(KGZone.BSh.value, "BSh")


class TestKGCodeMapping(unittest.TestCase):
    """The integer code <-> zone-name mappings round-trip
    cleanly and cover all 30 zones."""

    def test_code_to_name_covers_1_to_30(self):
        self.assertEqual(set(KG_CODE_TO_NAME.keys()), set(range(1, 31)))

    def test_name_to_code_round_trips(self):
        for code, name in KG_CODE_TO_NAME.items():
            self.assertEqual(KG_NAME_TO_CODE[name], code)

    def test_code_to_zone_covers_1_to_30(self):
        self.assertEqual(set(KG_CODE_TO_ZONE.keys()), set(range(1, 31)))

    def test_code_to_zone_returns_KGZone_members(self):
        for code, zone in KG_CODE_TO_ZONE.items():
            with self.subTest(code=code):
                self.assertIsInstance(zone, KGZone)
                self.assertEqual(zone.value, KG_CODE_TO_NAME[code])

    def test_nodata_code_is_zero(self):
        self.assertEqual(NODATA_CODE, 0)

    def test_native_cell_size(self):
        # 1/120° ≈ 0.008333° per Beck 2023 raster spec.
        self.assertAlmostEqual(NATIVE_CELL_DEG, 1.0 / 120.0, places=10)


class TestBundledSubstrate(unittest.TestCase):
    """The Beck 2023 raster + legend ship next to the module."""

    def test_bundled_raster_exists(self):
        self.assertTrue(
            BECK_2023_RASTER_PATH.is_file(),
            f"Bundled Beck 2023 raster missing at "
            f"{BECK_2023_RASTER_PATH}; the wheel was likely built "
            f"without the [tool.setuptools.package-data] block.",
        )

    def test_bundled_legend_exists(self):
        self.assertTrue(
            BECK_2023_LEGEND_PATH.is_file(),
            f"Bundled Beck 2023 legend missing at "
            f"{BECK_2023_LEGEND_PATH}.",
        )

    def test_legend_lists_all_30_zones(self):
        text = BECK_2023_LEGEND_PATH.read_text(encoding="utf-8")
        for code, name in KG_CODE_TO_NAME.items():
            with self.subTest(code=code, name=name):
                # The legend formats lines like ``    1:  Af``;
                # the (code, name) pair is unique enough.
                self.assertIn(f"{code}:", text)
                self.assertIn(name, text)


class TestKGClassifier(unittest.TestCase):
    """Empirically verified zone classifications on the bundled
    Beck 2023 raster. Anti-mutation: replacing the raster or
    re-mapping codes flips these verdicts."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = KGClassifier()

    @classmethod
    def tearDownClass(cls):
        cls.classifier.close()

    def test_interior_fixtures_classify_correctly(self):
        for name, lat, lon, expected in _INTERIOR_FIXTURES:
            with self.subTest(fixture=name):
                self.assertEqual(
                    self.classifier.classify(lat, lon), expected,
                    f"Fixture {name!r} at ({lat}, {lon}) expected "
                    f"{expected.value!r}.",
                )

    def test_ocean_returns_None(self):
        lat, lon = _OCEAN_FIXTURE
        self.assertIsNone(self.classifier.classify(lat, lon))

    def test_classify_batch_matches_per_point(self):
        points = [(lat, lon) for _, lat, lon, _ in _INTERIOR_FIXTURES]
        batch = self.classifier.classify_batch(points)
        per_point = [self.classifier.classify(lat, lon) for lat, lon in points]
        self.assertEqual(batch, per_point)

    def test_classify_batch_includes_ocean(self):
        points = [
            (13.5, 2.1),     # Niamey BSh
            _OCEAN_FIXTURE,   # ocean
        ]
        batch = self.classifier.classify_batch(points)
        self.assertEqual(batch, [KGZone.BSh, None])

    def test_classify_returns_KGZone_or_None(self):
        for _, lat, lon, _ in _INTERIOR_FIXTURES:
            with self.subTest(coord=(lat, lon)):
                result = self.classifier.classify(lat, lon)
                self.assertIsInstance(result, KGZone)


class TestKGClassifierLifecycle(unittest.TestCase):
    """Context-manager + close discipline."""

    def test_context_manager_closes_dataset(self):
        with KGClassifier() as cls:
            self.assertIsNotNone(cls.classify(13.5, 2.1))
        # After exit, classify should fail because dataset closed.
        with self.assertRaises(RuntimeError):
            cls.classify(13.5, 2.1)

    def test_explicit_close_idempotent(self):
        cls = KGClassifier()
        cls.close()
        cls.close()  # second close is a no-op


if __name__ == "__main__":
    unittest.main()
