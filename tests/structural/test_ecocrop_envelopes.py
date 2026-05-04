"""Sprint E.0.5 AC-Q3-A-d + F28 — bundled ECOCROP envelope substrate.

Pins the v1 ECOCROP envelope JSON shape, the verbatim values
for maize + rice (the two crops verified via Chrome MCP at the
ECOCROP source on 2026-05-03), and the F28 per-crop provenance
contract.

Anti-mutation drills:
- Drop a required field from any crop → loader raises
  EnvelopeValidationError →
  ``test_each_crop_has_4_required_envelope_fields`` fails.
- Introduce CLIZ (or any forbidden ECOCROP field) on any crop →
  ``test_no_out_of_scope_fields_in_any_crop`` fails.
- Change maize/rice envelope value → verbatim-pin test fails.
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
    """Pin the bundled v1 JSON shape + maize/rice verbatim
    values + scope discipline at the data layer."""

    @classmethod
    def setUpClass(cls):
        cls.envelopes = load_ecocrop_envelopes()
        with open(ECOCROP_ENVELOPE_PATH, encoding="utf-8") as fp:
            cls.payload = json.load(fp)

    def test_default_path_loads(self):
        self.assertGreaterEqual(len(self.envelopes), 2)

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

    def test_version_pinned(self):
        self.assertEqual(self.payload["version"], "ecocrop_v1_2026-05-03")


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


class TestECOCROPPackageData(unittest.TestCase):
    """Pin the pyproject package-data declaration so the
    bundled JSON ships in installed wheels. Without this the
    loader's default-path resolution raises FileNotFoundError
    on a pip-installed prismpy even though source-tree tests
    pass — a release-safety hole codex Gate A flagged HIGH."""

    def test_pyproject_declares_koppen_package_data(self):
        pyproject = (_REPO_ROOT / "pyproject.toml").read_text(
            encoding="utf-8",
        )
        self.assertRegex(
            pyproject,
            r'\[tool\.setuptools\.package-data\][\s\S]*'
            r'"prismpy\.koppen"\s*=\s*\[\s*"\*\.json"\s*\]',
            "pyproject.toml must declare a "
            "[tool.setuptools.package-data] block with "
            '\'"prismpy.koppen" = ["*.json"]\' so the bundled '
            "ecocrop_envelopes.json ships in installed wheels.",
        )


if __name__ == "__main__":
    unittest.main()
