"""Sprint E.0.5 AC-Q3-A-d + F28 — bundled ECOCROP envelope substrate.

Pins the v2 ECOCROP envelope JSON shape (six crops covering the
prismweb wizard catalog: maize + rice from Sprint E.0.5;
sorghum / pearl millet / cowpea / groundnut from Sprint F
crop-specialist Chrome MCP retrieval 2026-05-04 per CC-28
envelope-coverage-parity invariant) and the F28 per-crop
provenance contract.

Anti-mutation drills:
- Drop a required field from any crop → loader raises
  EnvelopeValidationError →
  ``test_each_crop_has_4_required_envelope_fields`` fails.
- Introduce CLIZ (or any forbidden ECOCROP field) on any crop →
  ``test_no_out_of_scope_fields_in_any_crop`` fails.
- Change any crop's envelope value (incl. the 4 new Sprint F
  crops) → verbatim-pin test fails.
- Drop verbatim_source_url or verbatim_retrieval_date from any
  crop → loader raises (F28) →
  ``test_each_crop_has_provenance_block`` fails.
- Use http:// instead of https:// in URL → loader raises (F28).
- Use non-ISO 8601 date → loader raises (F28).
"""
from __future__ import annotations

import json
import re
import unittest
from datetime import date
from pathlib import Path

from prismpy.koppen.envelopes import (
    ECOCROP_ENVELOPE_PATH,
    REQUIRED_FIELDS,
    REQUIRED_PROVENANCE_FIELDS,
    load_ecocrop_envelopes,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


# Forbidden ECOCROP fields per AC-Q3-A-d + probe-1-A scope
# clarity. Mirrors the F27 AST walker's forbidden set; here
# enforced at the data layer.
FORBIDDEN_FIELDS = frozenset({
    "CLIZ", "ALTMX", "ALTMN",
    "PHMIN", "PHMAX", "PHOPMN", "PHOPMX",
    "PHOTOPERIOD",
    "GMIN", "GMAX",
    "LATMIN", "LATMAX",
    "TOPMN", "TOPMX",  # optimum-range fields excluded per probe-1-A
    "ROPMN", "ROPMX",
})


# Pinned FAO domain — the structural test pins the domain at
# PR-review-time even though the loader only enforces HTTPS at
# runtime (so a future ECOCROP migration does not require a
# loader code change).
PINNED_FAO_DOMAIN = "https://ecocrop.apps.fao.org/"


# ISO 8601 calendar-date pattern — strict YYYY-MM-DD form.
ISO8601_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class TestECOCROPEnvelopesShape(unittest.TestCase):
    """Pin the bundled v2 JSON shape + verbatim envelope values
    for the six wizard-catalog crops (maize / rice / sorghum /
    pearl millet / cowpea / groundnut) + scope discipline at
    the data layer."""

    @classmethod
    def setUpClass(cls):
        cls.envelopes = load_ecocrop_envelopes()
        with open(ECOCROP_ENVELOPE_PATH, encoding="utf-8") as fp:
            cls.payload = json.load(fp)

    def test_default_path_loads(self):
        self.assertGreaterEqual(len(self.envelopes), 6)

    def test_each_crop_has_4_required_envelope_fields(self):
        for crop, env in self.envelopes.items():
            with self.subTest(crop=crop):
                for field in REQUIRED_FIELDS:
                    self.assertIn(field, env)
                    self.assertIsInstance(env[field], float)

    def test_metadata_block_present(self):
        for key in (
            "version", "license", "stage_1_scope_note", "fields", "crops",
        ):
            self.assertIn(key, self.payload)

    def test_no_out_of_scope_fields_in_any_crop(self):
        for crop, env in self.payload["crops"].items():
            with self.subTest(crop=crop):
                forbidden_present = FORBIDDEN_FIELDS & set(env.keys())
                self.assertEqual(forbidden_present, set())

    def test_maize_verbatim_envelope(self):
        env = self.envelopes["maize"]
        self.assertEqual(env["TMIN"], 10.0)
        self.assertEqual(env["TMAX"], 47.0)
        self.assertEqual(env["RMIN"], 400.0)
        self.assertEqual(env["RMAX"], 1800.0)

    def test_rice_verbatim_envelope(self):
        env = self.envelopes["rice"]
        self.assertEqual(env["TMIN"], 10.0)
        self.assertEqual(env["TMAX"], 36.0)
        self.assertEqual(env["RMIN"], 1000.0)
        self.assertEqual(env["RMAX"], 4000.0)

    # Sprint F AC-F-0 — verbatim envelope pins for the four new
    # crops added per CC-28 envelope-coverage-parity. Values
    # retrieved 2026-05-04 by the crop-specialist via Chrome MCP
    # against the FAO ECOCROP web data sheets and cross-validated
    # against the OpenCLIM CSV mirror. Persona alignment: pearl
    # millet (Pennisetum glaucum, code 8418) chosen over finger
    # millet for Sahel-canonical-staple + drought-tolerance
    # rationale.
    def test_sorghum_verbatim_envelope(self):
        env = self.envelopes["sorghum"]
        self.assertEqual(env["TMIN"], 8.0)
        self.assertEqual(env["TMAX"], 40.0)
        self.assertEqual(env["RMIN"], 300.0)
        self.assertEqual(env["RMAX"], 700.0)

    def test_millet_verbatim_envelope(self):
        env = self.envelopes["millet"]
        self.assertEqual(env["TMIN"], 12.0)
        self.assertEqual(env["TMAX"], 40.0)
        self.assertEqual(env["RMIN"], 200.0)
        self.assertEqual(env["RMAX"], 1700.0)

    def test_cowpea_verbatim_envelope(self):
        env = self.envelopes["cowpea"]
        self.assertEqual(env["TMIN"], 15.0)
        self.assertEqual(env["TMAX"], 40.0)
        self.assertEqual(env["RMIN"], 300.0)
        self.assertEqual(env["RMAX"], 4100.0)

    def test_groundnut_verbatim_envelope(self):
        env = self.envelopes["groundnut"]
        self.assertEqual(env["TMIN"], 10.0)
        self.assertEqual(env["TMAX"], 45.0)
        self.assertEqual(env["RMIN"], 400.0)
        self.assertEqual(env["RMAX"], 4000.0)

    def test_version_pinned(self):
        # Sprint F AC-F-0 bumps the substrate version to v2 to
        # signal that the cache key composed in
        # ``Project.stage_1_verdicts`` (per AC-F-5) recomputes for
        # any project whose cached verdicts reference v1 — a
        # 4-crop expansion is a substrate change even if the two
        # original entries are unchanged.
        self.assertEqual(self.payload["version"], "ecocrop_v2_2026-05-04")

    def test_license_footer_reflects_six_crops(self):
        # Pin the license footer reframe per AC-F-0. A future
        # crop-coverage expansion that bumps to 32+ facts must
        # update both the JSON and this pin in the same commit
        # so the test catches an unsynchronized footer.
        self.assertIn(
            "24 numeric facts (6 crops × 4 fields)",
            self.payload["license"],
        )


class TestECOCROPProvenanceF28(unittest.TestCase):
    """F28: per-crop provenance block must be present + valid
    on every crop. Protects against future commits drifting to
    ship unverified crop values."""

    @classmethod
    def setUpClass(cls):
        cls.envelopes = load_ecocrop_envelopes()

    def test_each_crop_has_provenance_block(self):
        for crop, env in self.envelopes.items():
            with self.subTest(crop=crop):
                for field in REQUIRED_PROVENANCE_FIELDS:
                    self.assertIn(field, env)
                    self.assertIsInstance(env[field], str)

    def test_provenance_url_is_https(self):
        for crop, env in self.envelopes.items():
            with self.subTest(crop=crop):
                self.assertTrue(
                    env["verbatim_source_url"].startswith("https://"),
                    f"Provenance URL for {crop!r} must use https://; "
                    f"got {env['verbatim_source_url']!r}",
                )

    def test_provenance_url_matches_ecocrop_domain(self):
        for crop, env in self.envelopes.items():
            with self.subTest(crop=crop):
                self.assertTrue(
                    env["verbatim_source_url"].startswith(PINNED_FAO_DOMAIN),
                    f"Provenance URL for {crop!r} must start with "
                    f"{PINNED_FAO_DOMAIN!r}; "
                    f"got {env['verbatim_source_url']!r}",
                )

    def test_provenance_retrieval_date_is_iso8601(self):
        for crop, env in self.envelopes.items():
            with self.subTest(crop=crop):
                self.assertRegex(
                    env["verbatim_retrieval_date"], ISO8601_DATE_RE,
                )
                # Also asserts parseable.
                date.fromisoformat(env["verbatim_retrieval_date"])

    def test_maize_provenance_url(self):
        self.assertEqual(
            self.envelopes["maize"]["verbatim_source_url"],
            "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2175",
        )

    def test_rice_provenance_url(self):
        self.assertEqual(
            self.envelopes["rice"]["verbatim_source_url"],
            "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1574",
        )

    # Sprint F AC-F-0 — provenance URL pins for the four new
    # crops. Each ECOCROP data-sheet ID is the verbatim retrieval
    # target; a typo in the JSON would otherwise route the audit
    # link to a different (or missing) crop record on the FAO
    # site without surfacing.
    def test_sorghum_provenance_url(self):
        self.assertEqual(
            self.envelopes["sorghum"]["verbatim_source_url"],
            "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=48747",
        )

    def test_millet_provenance_url(self):
        self.assertEqual(
            self.envelopes["millet"]["verbatim_source_url"],
            "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=8418",
        )

    def test_cowpea_provenance_url(self):
        self.assertEqual(
            self.envelopes["cowpea"]["verbatim_source_url"],
            "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2153",
        )

    def test_groundnut_provenance_url(self):
        self.assertEqual(
            self.envelopes["groundnut"]["verbatim_source_url"],
            "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2199",
        )


class TestECOCROPPackageData(unittest.TestCase):
    """Pin the pyproject package-data declaration so the
    bundled JSON, the Beck 2023 raster, and its legend ship
    in installed wheels. Without these patterns the loader /
    classifier default-path resolution raises FileNotFoundError
    on a pip-installed prismpy even though source-tree tests
    pass — a release-safety hole codex Gate A flagged HIGH."""

    @classmethod
    def setUpClass(cls):
        cls.pyproject = (_REPO_ROOT / "pyproject.toml").read_text(
            encoding="utf-8",
        )

    def test_pyproject_has_koppen_package_data_block(self):
        self.assertRegex(
            self.pyproject,
            r'\[tool\.setuptools\.package-data\][\s\S]*?'
            r'"prismpy\.koppen"\s*=',
            "pyproject.toml must declare a "
            "[tool.setuptools.package-data] block keyed on "
            "'prismpy.koppen' so non-Python data files ship "
            "in installed wheels.",
        )

    def test_pyproject_includes_json_pattern(self):
        # Captures the koppen package-data list (the chunk
        # between '"prismpy.koppen" = [' and the closing ']')
        # and asserts the JSON pattern is in it.
        self._assert_pattern_in_koppen_list("*.json")

    def test_pyproject_includes_tif_pattern(self):
        # Beck 2023 raster (data/beck_2023_v1.tif) ships under
        # data/. Without the data/*.tif pattern the wheel omits
        # it and the classifier default-path raises.
        self._assert_pattern_in_koppen_list("data/*.tif")

    def test_pyproject_includes_txt_pattern(self):
        # Legend (data/beck_2023_v1_legend.txt). Same rationale
        # as the tif: the legend test depends on the file
        # shipping next to the module.
        self._assert_pattern_in_koppen_list("data/*.txt")

    def _assert_pattern_in_koppen_list(self, pattern: str):
        # Match the entire list value after "prismpy.koppen" =
        # whether the list is single-line or multi-line.
        match = re.search(
            r'"prismpy\.koppen"\s*=\s*\[(?P<items>[\s\S]*?)\]',
            self.pyproject,
        )
        self.assertIsNotNone(
            match,
            "Could not find 'prismpy.koppen = [...]' list in "
            "pyproject.toml [tool.setuptools.package-data] block.",
        )
        items = match.group("items")
        self.assertIn(
            f'"{pattern}"', items,
            f"pyproject.toml [tool.setuptools.package-data] "
            f"'prismpy.koppen' list must include {pattern!r} so "
            f"the corresponding file(s) ship in installed wheels.",
        )


if __name__ == "__main__":
    unittest.main()
