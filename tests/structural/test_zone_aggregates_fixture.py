"""Sprint F AC-F-3 — synthetic zone-aggregates fixture pin.

Pins the shape + the canonical 5-zone coverage of the bundled
``zone_aggregates_v1.json`` fixture so the wizard can rely on
the JSON keys + per-zone schema without an additional runtime
contract check.

Anti-mutation drills:

* Drop a percentile field for any zone → ``build_zone_aggregate``
  raises ``pydantic.ValidationError`` from
  :class:`ZoneAggregate`'s required-field validators →
  ``test_each_zone_constructs_zone_aggregate`` fails.
* Drop a canonical zone (e.g. ``BSh``) → coverage assertion
  fires.
* Add a forbidden ECOCROP field on any zone → out-of-scope at
  this layer (the ECOCROP scope walker polices the envelope
  layer, not the zone-aggregate layer); however the typed
  :class:`ZoneAggregate` Pydantic model still rejects extra
  fields when constructed.
* Substrate version drift without a JSON re-issue → version
  pin fires + cache invalidation per AC-F-5 picks up the new
  string.
"""
from __future__ import annotations

import json
import re
import unittest

from prismpy.koppen.zone_aggregates import (
    ZONE_AGGREGATES_PATH,
    build_zone_aggregate,
    load_zone_aggregates,
)
from prismpy.validators.input_base import ZoneAggregate


# Sprint F AC-F-3 contract — the 5 canonical zones. Any future
# expansion (e.g., adding ET Tundra for highland-Cwa-adjacent
# regions) lands as additive fields with a version bump.
CANONICAL_ZONES = frozenset({"BSh", "Aw", "Cfa", "Cwa", "Af"})

# ISO 8601 date pin — same shape used in the ECOCROP envelope
# substrate's verbatim_retrieval_date (F28 walker pattern).
ISO8601_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class TestZoneAggregatesFixtureShape(unittest.TestCase):
    """Pin the JSON shape, the version, the license footer,
    and the 5-zone canonical coverage."""

    @classmethod
    def setUpClass(cls):
        cls.payload = load_zone_aggregates()

    def test_default_path_loads(self):
        self.assertTrue(ZONE_AGGREGATES_PATH.exists())
        self.assertGreater(len(self.payload.get("zones", {})), 0)

    def test_metadata_block_present(self):
        for key in (
            "version", "license", "stage_1_scope_note", "fields", "zones",
        ):
            self.assertIn(key, self.payload)

    def test_version_pinned_v1(self):
        # AC-F-3 substrate version. AC-F-5 cache key composes
        # against this string; a future ratchet to V2-19.5
        # real data MUST bump this version so cached verdicts
        # invalidate cleanly.
        self.assertEqual(
            self.payload["version"],
            "zone_aggregates_v1_2026-05-04",
        )

    def test_license_footer_marks_synthetic(self):
        # Synthetic-fixture honest signal — the license string
        # explicitly tells consumers not to derive scientific
        # claims from these values until V2-19.5 lands.
        license_text = self.payload["license"]
        self.assertIn("Synthetic", license_text)
        self.assertIn("V2-19.5", license_text)
        self.assertIn("climate-realistic", license_text.lower())
        # AC-F-3 codex absorption: license also calls out the
        # specific BSh thermal extreme that's held below
        # realistic Sahel peaks so the four crops route to
        # MARGINAL_HETEROGENEOUS instead of INCOMPATIBLE.
        self.assertIn("calibrate", license_text.lower())
        self.assertIn("BSh", license_text)

    def test_canonical_zones_present(self):
        zones = set(self.payload["zones"].keys())
        # Coverage: every canonical zone in the contract has
        # an entry in the fixture. A missing zone breaks the
        # wizard's per-zone walk for any region the classifier
        # routes to that zone.
        missing = CANONICAL_ZONES - zones
        self.assertEqual(
            missing, set(),
            f"Missing canonical zones in fixture: {missing}",
        )

    def test_each_zone_has_required_subkeys(self):
        for zone, entry in self.payload["zones"].items():
            with self.subTest(zone=zone):
                for key in (
                    "label", "n_cells", "n_cell_days",
                    "precip", "thermal",
                ):
                    self.assertIn(key, entry)
                # Precip subkeys
                for key in ("p25", "p50", "p75"):
                    self.assertIn(key, entry["precip"])
                # Thermal subkeys
                for key in (
                    "p10_extreme_tmin", "p90_extreme_tmax",
                ):
                    self.assertIn(key, entry["thermal"])

    def test_each_zone_constructs_zone_aggregate(self):
        # Every zone in the fixture must construct a typed
        # ZoneAggregate without Pydantic raising. This is the
        # canonical compatibility pin — drift on any field
        # name or type fails here loudly.
        for zone in self.payload["zones"]:
            with self.subTest(zone=zone):
                agg = build_zone_aggregate(zone, payload=self.payload)
                self.assertIsInstance(agg, ZoneAggregate)
                self.assertGreaterEqual(agg.n_cell_days, 0)
                self.assertLessEqual(agg.p25, agg.p50)
                self.assertLessEqual(agg.p50, agg.p75)
                self.assertLess(
                    agg.p10_extreme_tmin, agg.p90_extreme_tmax,
                )

    def test_unknown_zone_raises_keyerror(self):
        with self.assertRaises(KeyError):
            build_zone_aggregate("ZZZ", payload=self.payload)

    def test_zones_pass_min_cell_days_threshold(self):
        # AC-F-3 fixtures should all clear the AC-Q2-E
        # MIN_CELL_DAYS_PER_ZONE = 1_000_000 threshold so the
        # wizard exercises the COMPATIBLE / MARGINAL_* /
        # INCOMPATIBLE branches rather than the sample-quality
        # skip path. Otherwise the synthetic fixture would
        # always route to "skipped_insufficient_sample" and
        # never test the verdict logic.
        from prismpy.bounds import MIN_CELL_DAYS_PER_ZONE
        for zone, entry in self.payload["zones"].items():
            with self.subTest(zone=zone):
                self.assertGreaterEqual(
                    entry["n_cell_days"], MIN_CELL_DAYS_PER_ZONE,
                )


class TestZoneAggregatesVerdictIntegration(unittest.TestCase):
    """Pin the Sahel-canonical test-case behavior per the
    contract's BLOCKER 1 anti-mutation rationale: rice on BSh
    routes to INCOMPATIBLE precip; sorghum / millet / cowpea /
    groundnut on BSh route to MARGINAL_HETEROGENEOUS precip
    (P25 below RMIN, P50 in envelope)."""

    @classmethod
    def setUpClass(cls):
        from prismpy.koppen.envelopes import load_ecocrop_envelopes
        from prismpy.validators.climate_envelope import (
            CompatibilityVerdict,
            compare_precip_iqr,
        )
        cls.envelopes = load_ecocrop_envelopes()
        cls._Verdict = CompatibilityVerdict
        cls._compare = staticmethod(compare_precip_iqr)
        cls.bsh = build_zone_aggregate("BSh")

    def test_rice_on_bsh_incompatible(self):
        env = self.envelopes["rice"]
        verdict = self._compare(
            self.bsh.p25, self.bsh.p50, self.bsh.p75,
            env["RMIN"], env["RMAX"],
        )
        # BSh P50 = 400 mm/yr; rice RMIN = 1000 → INCOMPATIBLE.
        self.assertIs(verdict, self._Verdict.INCOMPATIBLE)

    def test_sahel_4_crops_on_bsh_marginal_heterogeneous(self):
        # sorghum / millet / cowpea / groundnut all route to
        # MARGINAL_HETEROGENEOUS on BSh per the BLOCKER 1
        # rationale: P50 in envelope but P25 below RMIN. This
        # test pins the synthetic-fixture / envelope alignment
        # that BLOCKER 1 hinges on.
        for crop in ("sorghum", "millet", "cowpea", "groundnut"):
            with self.subTest(crop=crop):
                env = self.envelopes[crop]
                verdict = self._compare(
                    self.bsh.p25, self.bsh.p50, self.bsh.p75,
                    env["RMIN"], env["RMAX"],
                )
                self.assertIs(
                    verdict,
                    self._Verdict.MARGINAL_HETEROGENEOUS,
                    f"{crop} BSh verdict {verdict.value!r} "
                    f"breaks BLOCKER 1 invariant — fixture or "
                    f"envelope drift.",
                )


class TestZoneAggregatesPackageData(unittest.TestCase):
    """Pin the pyproject package-data declaration so the
    bundled JSON ships in installed wheels. Mirrors the
    ECOCROP envelope's release-safety pin (Sprint E.0.5)."""

    def test_pyproject_includes_json_pattern(self):
        from pathlib import Path as _P
        pyproject = (
            _P(__file__).resolve().parents[2] / "pyproject.toml"
        ).read_text(encoding="utf-8")
        # The koppen package-data block already includes
        # ``"*.json"`` which covers both ECOCROP envelopes and
        # the new zone-aggregates fixture. Verify the existing
        # block is still in force (a future refactor that
        # narrows the include list to e.g. ``"ecocrop_*.json"``
        # would silently drop the new fixture from wheels).
        self.assertRegex(
            pyproject,
            r'\[tool\.setuptools\.package-data\][\s\S]*?'
            r'"prismpy\.koppen"[\s\S]*?"\*\.json"',
            "pyproject.toml [tool.setuptools.package-data] "
            "must include '*.json' under 'prismpy.koppen' so "
            "zone_aggregates_v1.json ships in wheels.",
        )


if __name__ == "__main__":
    unittest.main()
