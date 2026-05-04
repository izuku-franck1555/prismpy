"""Sprint E.0.5 AC-Q3-A-NaN + F28 — envelope validation paths.

Unit-tests for the per-envelope validator. Synthetic JSON
fixtures exercise each rejection path; valid payload exercises
the happy path.

AC-Q3-A-NaN drills:

- NaN value rejected → EnvelopeValidationError
- inf value rejected → EnvelopeValidationError
- Missing required field → EnvelopeValidationError
- Non-numeric type → EnvelopeValidationError
- RMIN ≥ RMAX (boundary or inverted) → EnvelopeValidationError (strict <)
- TMIN ≥ TMAX (boundary or inverted) → EnvelopeValidationError (strict <)
- Non-dict envelope → EnvelopeValidationError
- Missing top-level "crops" → EnvelopeValidationError
- Empty "crops" dict → EnvelopeValidationError
- Error message names crop + field

F28 drills:

- Missing verbatim_source_url → EnvelopeValidationError
- Missing verbatim_retrieval_date → EnvelopeValidationError
- Non-string provenance field → EnvelopeValidationError
- Non-HTTPS URL → EnvelopeValidationError
- Unparseable ISO 8601 date → EnvelopeValidationError
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from prismpy.koppen.envelopes import (
    EnvelopeValidationError,
    load_ecocrop_envelopes,
)


def _valid_maize() -> dict:
    """Reusable verbatim-style maize envelope. Mutated per
    test to exercise specific rejection paths."""
    return {
        "TMIN": 10,
        "TMAX": 47,
        "RMIN": 400,
        "RMAX": 1800,
        "verbatim_source_url": (
            "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2175"
        ),
        "verbatim_retrieval_date": "2026-05-03",
    }


def _valid_payload(maize: dict | None = None) -> dict:
    return {
        "crops": {
            "maize": maize if maize is not None else _valid_maize(),
        },
    }


class _LoaderTestBase(unittest.TestCase):
    """Helper: write a payload to a temp JSON file + load it.

    The temp file is cleaned up regardless of whether the
    loader succeeded.
    """

    def _load(self, payload: dict):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        ) as fp:
            json.dump(payload, fp)
            tmp_path = Path(fp.name)
        try:
            return load_ecocrop_envelopes(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)


class TestEnvelopeValidationNaN(_LoaderTestBase):
    """AC-Q3-A-NaN — numeric envelope validation paths."""

    def test_nan_value_rejected(self):
        m = _valid_maize()
        m["TMIN"] = float("nan")
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("maize", str(ctx.exception))
        self.assertIn("TMIN", str(ctx.exception))

    def test_inf_value_rejected(self):
        m = _valid_maize()
        m["TMAX"] = math.inf
        with self.assertRaises(EnvelopeValidationError):
            self._load(_valid_payload(m))

    def test_missing_required_field_rejected(self):
        m = _valid_maize()
        del m["RMAX"]
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("RMAX", str(ctx.exception))

    def test_non_numeric_value_rejected(self):
        m = _valid_maize()
        m["TMIN"] = "ten"
        with self.assertRaises(EnvelopeValidationError):
            self._load(_valid_payload(m))

    def test_bool_value_rejected(self):
        # Defensive: bool is a subclass of int in Python; the
        # validator explicitly excludes it so True/False can't
        # silently coerce to 1.0/0.0 envelope values.
        m = _valid_maize()
        m["TMIN"] = True
        with self.assertRaises(EnvelopeValidationError):
            self._load(_valid_payload(m))

    def test_rmin_equal_to_rmax_rejected(self):
        # Strict ordering: RMIN < RMAX (not <=)
        m = _valid_maize()
        m["RMIN"] = 1800
        m["RMAX"] = 1800
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("RMIN", str(ctx.exception))
        self.assertIn("RMAX", str(ctx.exception))

    def test_rmin_greater_than_rmax_rejected(self):
        m = _valid_maize()
        m["RMIN"] = 1800
        m["RMAX"] = 400
        with self.assertRaises(EnvelopeValidationError):
            self._load(_valid_payload(m))

    def test_tmin_equal_to_tmax_rejected(self):
        m = _valid_maize()
        m["TMIN"] = 47
        m["TMAX"] = 47
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("TMIN", str(ctx.exception))
        self.assertIn("TMAX", str(ctx.exception))

    def test_tmin_greater_than_tmax_rejected(self):
        m = _valid_maize()
        m["TMIN"] = 47
        m["TMAX"] = 10
        with self.assertRaises(EnvelopeValidationError):
            self._load(_valid_payload(m))

    def test_non_dict_envelope_rejected(self):
        with self.assertRaises(EnvelopeValidationError):
            self._load({"crops": {"maize": [10, 47, 400, 1800]}})

    def test_missing_top_level_crops_rejected(self):
        with self.assertRaises(EnvelopeValidationError):
            self._load({"version": "v1"})

    def test_empty_crops_rejected(self):
        with self.assertRaises(EnvelopeValidationError):
            self._load({"crops": {}})

    def test_error_message_names_crop_and_field(self):
        m = _valid_maize()
        m["TMIN"] = float("nan")
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("'maize'", str(ctx.exception))
        self.assertIn("'TMIN'", str(ctx.exception))

    def test_valid_payload_passes(self):
        result = self._load(_valid_payload())
        self.assertIn("maize", result)
        self.assertEqual(result["maize"]["TMIN"], 10.0)
        self.assertEqual(result["maize"]["TMAX"], 47.0)
        self.assertEqual(result["maize"]["RMIN"], 400.0)
        self.assertEqual(result["maize"]["RMAX"], 1800.0)

    def test_int_values_coerced_to_float(self):
        result = self._load(_valid_payload())
        for field in ("TMIN", "TMAX", "RMIN", "RMAX"):
            self.assertIsInstance(result["maize"][field], float)


class TestEnvelopeValidationF28(_LoaderTestBase):
    """F28 — per-crop provenance block validation paths."""

    def test_missing_verbatim_source_url_rejected(self):
        m = _valid_maize()
        del m["verbatim_source_url"]
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("verbatim_source_url", str(ctx.exception))
        self.assertIn("F28", str(ctx.exception))

    def test_missing_verbatim_retrieval_date_rejected(self):
        m = _valid_maize()
        del m["verbatim_retrieval_date"]
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("verbatim_retrieval_date", str(ctx.exception))

    def test_non_string_url_rejected(self):
        m = _valid_maize()
        m["verbatim_source_url"] = 12345
        with self.assertRaises(EnvelopeValidationError):
            self._load(_valid_payload(m))

    def test_non_string_retrieval_date_rejected(self):
        m = _valid_maize()
        m["verbatim_retrieval_date"] = 20260503
        with self.assertRaises(EnvelopeValidationError):
            self._load(_valid_payload(m))

    def test_http_url_rejected(self):
        m = _valid_maize()
        m["verbatim_source_url"] = "http://ecocrop.apps.fao.org/example"
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("HTTPS", str(ctx.exception))

    def test_relative_url_rejected(self):
        m = _valid_maize()
        m["verbatim_source_url"] = "/ecocrop/srv/en/dataSheet?id=2175"
        with self.assertRaises(EnvelopeValidationError):
            self._load(_valid_payload(m))

    def test_invalid_date_format_rejected(self):
        m = _valid_maize()
        m["verbatim_retrieval_date"] = "2026/05/03"
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("ISO 8601", str(ctx.exception))

    def test_unparseable_date_rejected(self):
        m = _valid_maize()
        m["verbatim_retrieval_date"] = "not-a-date"
        with self.assertRaises(EnvelopeValidationError):
            self._load(_valid_payload(m))

    def test_compact_iso_date_rejected(self):
        # ``date.fromisoformat`` accepts the compact YYYYMMDD form
        # on Python 3.11+; F28 rejects it because downstream
        # provenance diffing expects canonical YYYY-MM-DD only.
        m = _valid_maize()
        m["verbatim_retrieval_date"] = "20260503"
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("YYYY-MM-DD", str(ctx.exception))

    def test_iso_week_date_rejected(self):
        # ``date.fromisoformat`` accepts the ISO week-date form
        # on Python 3.11+; F28 rejects it for the same reason.
        m = _valid_maize()
        m["verbatim_retrieval_date"] = "2026-W18-7"
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("YYYY-MM-DD", str(ctx.exception))

    def test_calendar_overflow_date_rejected(self):
        # YYYY-MM-DD shape passes the canonical regex but
        # ``date.fromisoformat`` rejects month overflow (13).
        m = _valid_maize()
        m["verbatim_retrieval_date"] = "2026-13-01"
        with self.assertRaises(EnvelopeValidationError) as ctx:
            self._load(_valid_payload(m))
        self.assertIn("real calendar date", str(ctx.exception))

    def test_empty_url_rejected(self):
        m = _valid_maize()
        m["verbatim_source_url"] = ""
        with self.assertRaises(EnvelopeValidationError):
            self._load(_valid_payload(m))

    def test_valid_provenance_passes(self):
        result = self._load(_valid_payload())
        self.assertEqual(
            result["maize"]["verbatim_source_url"],
            "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=2175",
        )
        self.assertEqual(
            result["maize"]["verbatim_retrieval_date"],
            "2026-05-03",
        )


if __name__ == "__main__":
    unittest.main()
