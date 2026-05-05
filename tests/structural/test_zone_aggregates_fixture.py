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
    _cached_zone_aggregates,
    build_zone_aggregate,
    label_for,
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
    bundled JSON ships in installed wheels.

    Two-stage pin discipline per
    ``feedback_intermediate_stage_pin_gap.md`` + the wheel-
    contents lesson learned post-Sprint-F: the SOURCE-side
    pyproject.toml literal is necessary but NOT sufficient.
    A pattern at the package root (``*.json``) does not
    recurse into subdirectories, so a fixture under
    ``data/`` requires its own ``data/*.json`` glob. The
    wheel-contents test below catches the gap empirically.
    """

    @classmethod
    def setUpClass(cls):
        from pathlib import Path as _P
        cls._pyproject_text = (
            _P(__file__).resolve().parents[2] / "pyproject.toml"
        ).read_text(encoding="utf-8")
        cls._koppen_list = cls._extract_koppen_package_data_list(
            cls._pyproject_text,
        )

    @staticmethod
    def _extract_koppen_package_data_list(pyproject_text: str) -> str:
        """Return the body of the ``"prismpy.koppen" = [...]``
        list from ``[tool.setuptools.package-data]``.

        Codex L-2 absorption: each per-pattern test bounds its
        regex to the koppen list specifically rather than
        scanning the whole pyproject. A future entry under a
        different package (e.g.,
        ``"prismpy.bounds" = ["data/*.json"]``) would
        otherwise satisfy a koppen-targeted assertion that
        wasn't bounded.
        """
        # Match: "prismpy.koppen" = [<body up to closing ]>]
        match = re.search(
            r'"prismpy\.koppen"\s*=\s*\[(?P<body>[\s\S]*?)\]',
            pyproject_text,
        )
        if match is None:
            raise AssertionError(
                "Could not locate 'prismpy.koppen' = [...] "
                "list in pyproject.toml. Verify the "
                "[tool.setuptools.package-data] section is "
                "still present.",
            )
        return match.group("body")

    def test_pyproject_includes_data_json_pattern(self):
        # Sprint F Stage 1 fix: the new fixture lives under
        # ``koppen/data/`` so it needs the ``data/*.json``
        # glob pattern, not just the package-root ``*.json``.
        # The original AC-F-3 test mistakenly assumed
        # ``*.json`` would cover the subdir; the wheel build
        # proved otherwise.
        self.assertIn(
            '"data/*.json"', self._koppen_list,
            "pyproject.toml [tool.setuptools.package-data] "
            "must include 'data/*.json' under 'prismpy.koppen' "
            "so zone_aggregates_v1.json (under data/) ships "
            "in installed wheels. The package-root '*.json' "
            "glob does NOT recurse into subdirs.",
        )

    def test_pyproject_includes_root_json_pattern(self):
        # Backstop for the existing ECOCROP envelopes file at
        # ``koppen/ecocrop_envelopes.json`` (package root).
        self.assertIn(
            '"*.json"', self._koppen_list,
            "pyproject.toml must keep '*.json' under "
            "'prismpy.koppen' for the ECOCROP envelopes file "
            "at the package root.",
        )

    def test_pyproject_includes_data_tif_pattern(self):
        # Backstop for the Beck 2023 raster shipped under
        # ``data/`` per Sprint E.0.5. The ``data/*.tif``
        # pattern was the original fix for that file. Pin
        # here so a future refactor that consolidates the
        # data globs cannot drop the raster pattern silently.
        self.assertIn(
            '"data/*.tif"', self._koppen_list,
            "pyproject.toml must keep 'data/*.tif' under "
            "'prismpy.koppen' for the Beck 2023 raster.",
        )

    # Allowed file extensions under ``koppen/data/`` per the
    # explicit globs declared in pyproject.toml. Any future
    # glob change that adds a new extension lands here in the
    # same commit so the wheel-contents pin stays in sync.
    _ALLOWED_DATA_EXTENSIONS = frozenset({".json", ".tif", ".txt"})

    def test_built_wheel_carries_zone_aggregates_fixture(self):
        # Two-stage pin (per team-lead Path-A directive +
        # ``feedback_intermediate_stage_pin_gap.md`` durable
        # lesson #22): the source-side ``assertRegex`` checks
        # above are necessary but NOT sufficient — they pin
        # the LITERAL text of the package-data list. This
        # test pins the DOWNSTREAM CONSUMER side: build a
        # wheel from the local source tree via ``pip wheel``
        # and assert the synthetic fixture is a member of
        # the resulting archive.
        #
        # Caveat documented for future builders: setuptools
        # ≥66.x can auto-include data files when invoked from
        # a local source tree (the ``setuptools.build_meta``
        # local-build shortcut bypasses the strict
        # ``package-data`` glob check). The same setuptools
        # applies the globs STRICTLY when the build is driven
        # through ``sdist → wheel`` (the path
        # ``pip install <git-url>`` uses). So in some test
        # environments this assertion may PASS against a
        # broken pyproject.toml. The source-side asserts
        # remain the load-bearing guard; this wheel-contents
        # test is defense-in-depth for environments where
        # the strict path applies.
        #
        # Skip discipline (codex M-1 absorption): only skip
        # on environment-unavailability (interpreter outside
        # the prismpy ``requires-python`` window OR ``pip
        # wheel`` missing). A genuine ``CalledProcessError``
        # on a supported interpreter falls through and fails
        # loudly — that's the regression catch.
        import subprocess
        import sys
        import tempfile
        import zipfile
        from pathlib import Path as _P

        # Pre-check interpreter version against the prismpy
        # ``requires-python`` cap (>=3.10,<3.13). Skip if
        # outside; otherwise the wheel build will hard-fail
        # with a "Package 'prismpy' requires a different
        # Python" diagnostic which is environmental, not a
        # source defect.
        py = sys.version_info
        if py.major != 3 or py.minor < 10 or py.minor >= 13:
            self.skipTest(
                f"Wheel-build test requires Python "
                f">=3.10,<3.13; running on "
                f"{py.major}.{py.minor}.{py.micro}. Source-"
                f"side asserts above still pin the package-"
                f"data declaration.",
            )

        prismpy_root = _P(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as td:
            wheel_dir = _P(td)
            try:
                subprocess.run(
                    [
                        sys.executable, "-m", "pip", "wheel",
                        "--no-deps", "--no-cache-dir",
                        "--wheel-dir", str(wheel_dir),
                        str(prismpy_root),
                    ],
                    check=True, capture_output=True, text=True,
                    timeout=240,
                )
            except FileNotFoundError as exc:
                # pip not installed in the test interpreter —
                # environmental skip.
                self.skipTest(
                    f"pip not available in test env: {exc}.",
                )
            except subprocess.TimeoutExpired as exc:
                # Wheel build hung past 240s — likely a CI
                # environmental issue (slow network,
                # contention). Skip with a loud message.
                self.skipTest(
                    f"pip wheel timed out (>240s): {exc}.",
                )
            # subprocess.CalledProcessError deliberately NOT
            # caught: a genuine wheel-build failure on a
            # supported interpreter is the exact regression
            # this test exists to catch. The exception bubbles
            # up and fails the test with pip's stdout/stderr
            # captured, which is what we want.
            wheels = list(wheel_dir.glob("prismpy-*.whl"))
            self.assertEqual(
                len(wheels), 1,
                f"expected exactly one prismpy wheel in "
                f"{wheel_dir}; got {wheels}",
            )
            with zipfile.ZipFile(wheels[0]) as zf:
                names = set(zf.namelist())
            target = "prismpy/koppen/data/zone_aggregates_v1.json"
            data_only = sorted(n for n in names if "data/" in n)
            self.assertIn(
                target, names,
                f"Two-stage pin failure: wheel does NOT carry "
                f"{target!r}. The pyproject.toml package-data "
                f"glob pattern likely doesn't match the file's "
                f"relative path under data/. Wheel contents "
                f"under data/: {data_only}",
            )
            # Codex L-1 absorption — tightened negative pin.
            # Every file shipped under ``koppen/data/`` MUST
            # have an extension in the explicit allow-list.
            # Catches not just bytecode caches (.pyc /
            # __pycache__) but also editor swap files
            # (.DS_Store / .bak / .swp), accidental
            # large-asset additions, etc. A future glob
            # broadening like ``data/**`` would over-match
            # and trip this assertion.
            data_files = [
                n for n in data_only
                if "/data/" in n and not n.endswith("/data/")
            ]
            unexpected = sorted(
                n for n in data_files
                if _P(n).suffix not in self._ALLOWED_DATA_EXTENSIONS
                or "__pycache__" in n
            )
            self.assertEqual(
                unexpected, [],
                f"Wheel under koppen/data/ unexpectedly "
                f"carries files outside the declared "
                f"extension allow-list "
                f"{sorted(self._ALLOWED_DATA_EXTENSIONS)}: "
                f"{unexpected}. The package-data globs should "
                f"only include the explicit declared "
                f"extensions, not a broad recursive match.",
            )

    def test_pyproject_includes_data_txt_pattern(self):
        # Backstop for the Beck 2023 legend shipped at
        # ``koppen/data/beck_2023_v1_legend.txt``.
        self.assertIn(
            '"data/*.txt"', self._koppen_list,
            "pyproject.toml must keep 'data/*.txt' under "
            "'prismpy.koppen' for the Beck 2023 raster legend.",
        )


class TestLabelFor(unittest.TestCase):
    """F-Path-β-1 — pin the human-readable zone-label resolver.

    The wizard banner needs ``"Hot semi-arid"`` (not ``"BSh"``)
    in plain-language explanation copy. ``label_for`` reads the
    label from the substrate, strips the trailing
    parenthetical disambiguator (a fixture-quality qualifier
    like ``" (Sahel-canonical)"``), and falls back to the zone
    code itself when the substrate doesn't carry a label.
    """

    def setUp(self):
        # Clear the LRU cache between tests so an in-memory
        # payload override doesn't leak across cases.
        _cached_zone_aggregates.cache_clear()

    def tearDown(self):
        _cached_zone_aggregates.cache_clear()

    def test_bsh_strips_sahel_canonical_parenthetical(self):
        # The fixture stores "Hot semi-arid (Sahel-canonical)";
        # the strip is the user-facing copy contract.
        self.assertEqual(label_for("BSh"), "Hot semi-arid")

    def test_aw_strips_sudan_savanna_parenthetical(self):
        self.assertEqual(label_for("Aw"), "Tropical savanna")

    def test_zones_without_parenthetical_round_trip(self):
        # "Humid subtropical" / "Subtropical highland" /
        # "Tropical rainforest" carry no parenthetical and
        # should pass through unchanged.
        self.assertEqual(label_for("Cfa"), "Humid subtropical")
        self.assertEqual(label_for("Cwa"), "Subtropical highland")
        self.assertEqual(label_for("Af"), "Tropical rainforest")

    def test_unknown_zone_falls_back_to_code(self):
        # Future-proofing — a zone code not in the fixture
        # falls back to the code itself so the banner stays
        # intelligible (no empty paragraph).
        self.assertEqual(label_for("ET"), "ET")

    def test_payload_override_skips_disk_load(self):
        # Tests can pass an in-memory payload to exercise
        # alternative fixtures without touching the cached
        # default substrate.
        payload = {
            "zones": {
                "Xx": {"label": "Imaginary climate (synthetic)"},
            },
        }
        self.assertEqual(
            label_for("Xx", payload=payload), "Imaginary climate",
        )

    def test_missing_label_falls_back_to_code(self):
        # Defensive — a substrate entry without a label key
        # falls back to the zone code rather than producing an
        # empty string.
        payload = {"zones": {"Yy": {"n_cell_days": 100}}}
        self.assertEqual(label_for("Yy", payload=payload), "Yy")

    def test_empty_label_falls_back_to_code(self):
        # An empty string after stripping the parenthetical
        # would otherwise leak through; fall back to the code.
        payload = {"zones": {"Zz": {"label": "(only-parens)"}}}
        self.assertEqual(label_for("Zz", payload=payload), "Zz")

    def test_non_string_label_falls_back_to_code(self):
        # Defensive type-check — a future drift to a non-string
        # label (e.g., a localized dict) does not break the
        # validator.
        payload = {"zones": {"Aa": {"label": 42}}}
        self.assertEqual(label_for("Aa", payload=payload), "Aa")

    def test_malformed_payload_falls_back_to_code(self):
        # Top-level shape drift (no ``zones`` key, or a list
        # instead of a dict) returns the code rather than
        # raising.
        self.assertEqual(label_for("BSh", payload={}), "BSh")
        self.assertEqual(label_for("BSh", payload={"zones": []}), "BSh")


if __name__ == "__main__":
    unittest.main()
