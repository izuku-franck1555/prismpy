"""Sprint F-CP structural pins for the GAEZ Esri Image Service migration.

Covers AC-F-CP-1 through AC-F-CP-11 except AC-F-CP-7 (canonical end-to-end
Gate B; evaluator-owned) and AC-F-CP-8 live-data probes. The structural pins
here enforce the producer-consumer contract per durable §27 + canonical-
source-or-pin per durable §24:

  - Public surface preservation (AC-F-CP-1)
  - 4-axis surjectivity + vocab parity (AC-F-CP-2 + WA CA-1)
  - URL canonical-source + old S3 ban (AC-F-CP-3 + codex H4)
  - Cache layout + atomic write + per-path lock (AC-F-CP-4 + codex H3)
  - Esri client integration shape (AC-F-CP-5)
  - Honest-signal HTTP status surfacing (AC-F-CP-6 + codex M3)
  - ThreadPoolExecutor max_workers backward compat (AC-F-CP-9 + codex M4)
  - GAEZDownloadError propagation (AC-F-CP-10 + codex H1)
  - LUT pixel-range / discrete-scheme parity (AC-F-CP-11)

The live-network probes (canonical Gate B) live in evaluator's framework,
not here — these structural pins do not hit the network.
"""
from __future__ import annotations

import ast
import inspect
import re
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

# ---------------------------------------------------------------------------
# AC-F-CP-1 — Module split + Pydantic schemas (6 tests)
# ---------------------------------------------------------------------------


class TestModuleSplitAndPydanticSchemas(unittest.TestCase):
    """Pin the 5-module layout + Pydantic frozen/extra= configuration."""

    def test_public_surface_preserved(self):
        """All pre-Esri exports continue to import + signatures preserved.

        AC-F-CP-1 drill: introduce ``download_v2`` and the assertion
        ``GAEZDownloader.download_v2`` would pass — that's intentional;
        the pin is on the EXISTENCE of the original methods, not the
        absence of new ones. ``max_workers`` is added as ADDITIVE kwarg.
        """
        from prismpy.sources.gaez import (
            GAEZDownloader,
            GAEZ_CROP_CODES,
            GAEZ_CULTIVAR_MAP,
        )

        for name in (
            "get_cultivars_for_crop",
            "_download_with_retry",
            "download_cultivar",
            "download_for_crop",
            "copy_to_output",
            "check_cached",
        ):
            self.assertTrue(
                hasattr(GAEZDownloader, name),
                f"GAEZDownloader.{name} must remain on the public surface "
                "per AC-F-CP-1.",
            )

        init_params = inspect.signature(GAEZDownloader.__init__).parameters
        for kw in ("cache_dir", "retries", "backoff", "max_workers"):
            self.assertIn(
                kw, init_params,
                f"GAEZDownloader.__init__ must accept {kw!r} kwarg.",
            )
        self.assertIsNot(
            init_params["max_workers"].default, inspect.Parameter.empty,
            "max_workers MUST have a default (per codex M4 back-compat).",
        )

        copy_params = inspect.signature(GAEZDownloader.copy_to_output).parameters
        for kw in ("crop_name", "output_dir", "water_supplies", "input_levels"):
            self.assertIn(kw, copy_params)

        # Constants present
        self.assertIn("Highland_sorghum", GAEZ_CROP_CODES)
        self.assertIn("Sorghum", GAEZ_CULTIVAR_MAP)

    def test_gaez_downloader_init_back_compat_kwargs(self):
        """``GAEZDownloader()`` + ``GAEZDownloader(retries=5)`` still work."""
        from prismpy.sources.gaez import GAEZDownloader

        d_default = GAEZDownloader()
        d_retries = GAEZDownloader(retries=5)
        self.assertEqual(d_default.retries, 3)
        self.assertEqual(d_default.max_workers, 10)
        self.assertEqual(d_retries.retries, 5)

    def test_export_image_request_frozen_and_forbid(self):
        """``EsriExportImageRequest`` is frozen + extra='forbid'."""
        from prismpy.sources.gaez import EsriExportImageRequest

        req = EsriExportImageRequest(
            bbox="-180,-90,180,90",
            size="4320,2160",
            mosaic_rule_json="{}",
        )
        with self.assertRaises(ValidationError):
            req.bbox = "0,0,1,1"  # type: ignore[misc]
        with self.assertRaises(ValidationError):
            EsriExportImageRequest(
                bbox="-180,-90,180,90",
                size="4320,2160",
                mosaic_rule_json="{}",
                unknown_kwarg="boom",
            )

    def test_esri_error_response_ignore_extra(self):
        """``EsriErrorResponse`` is the documented ``extra='ignore'`` exception
        per codex M2."""
        from prismpy.sources.gaez import EsriErrorResponse

        # Stale-schema body with extra fields must parse cleanly.
        body = {"code": 400, "message": "bad query", "messageCode": "ESRI_FOO",
                "details": ["detail-1"]}
        parsed = EsriErrorResponse.model_validate(body)
        self.assertEqual(parsed.code, 400)
        self.assertEqual(parsed.message, "bad query")

    def test_esri_query_spec_frozen(self):
        """``EsriQuerySpec`` immutable post-construction."""
        from prismpy.sources.gaez import EsriQuerySpec

        spec = EsriQuerySpec(
            crop="Highland sorghum",
            water_supply="ws",
            input_level="Low",
            variable_pattern="%v%",
        )
        with self.assertRaises(ValidationError):
            spec.crop = "mutated"  # type: ignore[misc]

    def test_export_image_request_bad_input_raises(self):
        """Malformed bbox / size / pixel_type / format raise ``ValidationError``."""
        from prismpy.sources.gaez import EsriExportImageRequest

        with self.assertRaises(ValidationError):
            EsriExportImageRequest(bbox="bad", size="1,1", mosaic_rule_json="{}")
        with self.assertRaises(ValidationError):
            EsriExportImageRequest(
                bbox="0,0,1,1", size="bad", mosaic_rule_json="{}",
            )
        with self.assertRaises(ValidationError):
            EsriExportImageRequest(
                bbox="0,0,1,1", size="1,1", mosaic_rule_json="{}",
                image_format="png",
            )
        with self.assertRaises(ValidationError):
            EsriExportImageRequest(
                bbox="0,0,1,1", size="1,1", mosaic_rule_json="{}",
                pixel_type="U8",
            )


# ---------------------------------------------------------------------------
# AC-F-CP-2 — Two-vocab translator surjectivity + AST walker (8 tests)
# ---------------------------------------------------------------------------


class TestTwoVocabTranslatorSurjectivity(unittest.TestCase):
    """4-axis surjectivity + UnknownCombination + per-axis verifications."""

    def test_full_esri_query_surjectivity(self):
        """Every (cultivar, water, level, var) in ACEA's domain resolves.

        Per WA CA-1: producer values ⊆ consumer keys. The Cartesian
        product is bounded:
          - cultivar: GAEZ_CROP_CODES keys (underscore-canonical form only)
          - water: ('irr','rf')
          - level: ('High','Low') — Intermediate excluded per res02 HIST
          - var: GAEZ_VARIABLES keys
        """
        from prismpy.sources.gaez import (
            GAEZ_CROP_CODES, GAEZ_VARIABLES, resolve_esri_query,
        )

        # Canonical cultivar set: dedupe via underscore normalization.
        cultivars = sorted({k.replace(" ", "_") for k in GAEZ_CROP_CODES.keys()})
        unresolved = []
        for cult in cultivars:
            for ws in ("irr", "rf"):
                for level in ("High", "Low"):
                    for var in GAEZ_VARIABLES.keys():
                        try:
                            resolve_esri_query(cult, ws, level, var)
                        except Exception as e:  # noqa
                            unresolved.append((cult, ws, level, var, str(e)))
        self.assertEqual(
            unresolved, [],
            f"Every ACEA 4-tuple MUST resolve via resolve_esri_query "
            f"(WA CA-1 + AC-F-CP-2). Unresolved: {unresolved!r}",
        )

    def test_unknown_combination_raises(self):
        """Unknown axis values raise ``UnknownCombination``."""
        from prismpy.sources.gaez import UnknownCombination, resolve_esri_query

        with self.assertRaises(UnknownCombination):
            resolve_esri_query("NotACultivar", "rf", "Low", "LUT")
        with self.assertRaises(UnknownCombination):
            resolve_esri_query("Highland_sorghum", "fizz", "Low", "LUT")
        with self.assertRaises(UnknownCombination):
            resolve_esri_query("Highland_sorghum", "rf", "Intermediate", "LUT")
        with self.assertRaises(UnknownCombination):
            resolve_esri_query("Highland_sorghum", "rf", "Low", "WrongVar")

    def test_cultivar_dual_form_accepted(self):
        """Space-form and underscore-form cultivars both resolve."""
        from prismpy.sources.gaez import resolve_esri_query

        a = resolve_esri_query("Highland sorghum", "rf", "Low", "LUT")
        b = resolve_esri_query("Highland_sorghum", "rf", "Low", "LUT")
        self.assertEqual(a.crop, b.crop)

    def test_water_vocab_map_complete(self):
        """``_WATER_TO_ESRI`` covers ACEA's default water set ('irr','rf')."""
        from prismpy.sources.gaez.raster_mapping import water_keys

        keys = set(water_keys().keys())
        self.assertEqual(keys, {"irr", "rf"})

    def test_level_vocab_map_excludes_intermediate(self):
        """res02 HIST baseline does NOT carry the Intermediate level."""
        from prismpy.sources.gaez.raster_mapping import level_keys

        keys = set(level_keys().keys())
        self.assertEqual(
            keys, {"High", "Low"},
            "res02 HIST baseline carries only High + Low per crop-specialist "
            "Q5 (Intermediate=8110I returns 403 on old S3 + is absent on Esri).",
        )

    def test_variable_pattern_uses_like_for_f3(self):
        """F3 pattern uses ``%constraint factor%3%agro-climatic%`` per
        crop-specialist HIGH-3 (Unicode quote brittleness)."""
        from prismpy.sources.gaez.raster_mapping import variable_keys

        f3_pattern = variable_keys()["F3"]
        self.assertIn("constraint factor", f3_pattern)
        self.assertIn("agro-climatic", f3_pattern)

    def test_no_inline_cultivar_map_redefinition(self):
        """AST walker — ACEA translator MUST NOT redefine cultivar_map locally.

        Per WA CA-2 + durable §24: the cultivar roster has a single source
        of truth (GAEZ_CULTIVAR_MAP). The walker greps for any local
        ``cultivar_map = {...}`` or ``gaez_cultivar_map = {...}`` assignment
        with a dict literal carrying GAEZ-known keys and fails the test.
        """
        import prismpy.translators.acea.translator as acea_translator_mod

        source = inspect.getsource(acea_translator_mod)
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # We only flag dict-literal assignments to a known cultivar-
                # map name. Imports + helper-call results are not in scope.
                if not isinstance(node.value, ast.Dict):
                    continue
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    name = target.id
                    if "cultivar_map" not in name.lower():
                        continue
                    # Detect the GAEZ signature: dict literal with at least
                    # 2 of {Maize, Sorghum, Wheat, Rice, Millet} as keys.
                    keys = []
                    for k in node.value.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            keys.append(k.value)
                    crop_keys = {"Maize", "Sorghum", "Wheat", "Rice", "Millet"}
                    if len(set(keys) & crop_keys) >= 2:
                        offenders.append((name, getattr(node, "lineno", "?")))
        self.assertEqual(
            offenders, [],
            f"Per WA CA-2 + durable §24, ACEA translator must import "
            f"GAEZ_CULTIVAR_MAP rather than redefining cultivar_map locally. "
            f"Inline redefinitions detected: {offenders!r}",
        )

    def test_to_where_renders_quoted_attribute_filter(self):
        """``EsriQuerySpec.to_where`` produces an SQL-style ``where`` clause."""
        from prismpy.sources.gaez import resolve_esri_query

        spec = resolve_esri_query("Highland_sorghum", "rf", "Low", "LUT")
        where = spec.to_where()
        self.assertIn("crop='Highland sorghum'", where)
        self.assertIn(
            "water_supply='Available water content of 200 mm/m "
            "(under rain-fed conditions)'",
            where,
        )
        self.assertIn("input_level='Low'", where)
        self.assertIn("variable LIKE '%Selected LUT sequence number%'", where)


# ---------------------------------------------------------------------------
# AC-F-CP-3 — URL canonical-source + old S3 ban-pin (2 tests)
# ---------------------------------------------------------------------------


def _list_prismpy_python_files() -> list[Path]:
    """Return all *.py files under ``src/prismpy/``."""
    src_root = Path(__file__).resolve().parents[2] / "src" / "prismpy"
    return sorted(src_root.rglob("*.py"))


class TestUrlCanonicalSourceAndOldS3Ban(unittest.TestCase):

    def test_no_hardcoded_image_service_urls_outside_client(self):
        """Only ``esri_client.py`` may contain ``gaez-services.fao.org``."""
        host = "gaez-services.fao.org"
        offenders = []
        for path in _list_prismpy_python_files():
            if path.name == "esri_client.py":
                continue
            text = path.read_text()
            if host in text:
                offenders.append(str(path))
        self.assertEqual(
            offenders, [],
            f"Per durable §24, the Esri host {host!r} MUST live ONLY in "
            f"esri_client.py. Found in: {offenders!r}",
        )

    def test_no_old_s3_strings_anywhere(self):
        """``data.gaezdev.aws.fao.org`` MUST be absent from production src."""
        host = "data.gaezdev.aws.fao.org"
        offenders = []
        for path in _list_prismpy_python_files():
            text = path.read_text()
            if host in text:
                offenders.append(str(path))
        self.assertEqual(
            offenders, [],
            f"Per codex H4 ban-pin + durable §24, the deprecated S3 host "
            f"{host!r} MUST NOT appear in src/prismpy/. Found in: {offenders!r}",
        )


# ---------------------------------------------------------------------------
# AC-F-CP-4 — Cache layout + atomic write + per-path lock (3 tests)
# ---------------------------------------------------------------------------


class TestCacheLayoutAndAtomicWrite(unittest.TestCase):

    def test_cache_path_matches_acea_filesystem_contract(self):
        """Filename format preserved: ``{var}/{var}_{cultivar_fname}_{ws}_{level}_1981_2010.tif``."""
        import tempfile
        from prismpy.sources.gaez import GAEZDownloader

        with tempfile.TemporaryDirectory() as tmp:
            d = GAEZDownloader(cache_dir=Path(tmp))
            p = d._compute_cache_path("Highland_sorghum", "rf", "Low", "LUT")
            self.assertEqual(
                p.name, "LUT_Highland_sorghum_rf_Low_1981_2010.tif",
            )
            self.assertEqual(p.parent.name, "LUT")

            # Space-form normalized
            p2 = d._compute_cache_path("Highland sorghum", "rf", "Low", "LUT")
            self.assertEqual(p2.name, p.name)

    def test_atomic_write_no_partial_on_failure(self):
        """If write of payload errors mid-stream, no partial file at output_path."""
        import tempfile
        from prismpy.sources.gaez import GAEZDownloader

        with tempfile.TemporaryDirectory() as tmp:
            d = GAEZDownloader(cache_dir=Path(tmp))
            target = Path(tmp) / "out.tif"

            # Simulate disk-write failure mid-stream by patching open() to
            # raise on close.
            class BadFile:
                def __init__(self):
                    self.written = bytearray()
                def write(self, b):
                    self.written.extend(b)
                def flush(self):
                    pass
                def fileno(self):
                    return 0
                def close(self):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    raise IOError("simulated disk failure")

            with mock.patch("builtins.open", return_value=BadFile()):
                with mock.patch("os.fsync"):
                    with self.assertRaises(IOError):
                        d._atomic_write(target, b"x" * 1024)

            # No final file at target
            self.assertFalse(target.exists(),
                "atomic_write MUST NOT leave partial output at target on failure.")
            # No stray tmp file
            self.assertFalse((target.with_suffix(".tif.tmp")).exists())

    def test_concurrent_write_same_path_does_not_corrupt(self):
        """Two threads writing same path: final = one of the two complete payloads.

        Per codex H3: the per-path lock + ``os.replace`` ensure the loser
        sees the winner's complete payload, not byte-interleaved garbage.
        """
        import tempfile
        from prismpy.sources.gaez import GAEZDownloader

        with tempfile.TemporaryDirectory() as tmp:
            d = GAEZDownloader(cache_dir=Path(tmp))
            target = Path(tmp) / "race.tif"
            payload_a = b"AAAA" * 4096
            payload_b = b"BBBB" * 4096

            def writer_a():
                d._atomic_write(target, payload_a)

            def writer_b():
                d._atomic_write(target, payload_b)

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(writer_a if i % 2 == 0 else writer_b)
                    for i in range(8)
                ]
                for f in futures:
                    f.result()

            data = target.read_bytes()
            self.assertIn(data, (payload_a, payload_b),
                "Concurrent same-path writes MUST produce one of the two "
                "complete payloads, never byte-interleaved corruption (per "
                "AC-F-CP-4 + codex H3).")


# ---------------------------------------------------------------------------
# AC-F-CP-5 — EsriImageServiceClient integration (6 tests)
# ---------------------------------------------------------------------------


class TestEsriImageServiceClient(unittest.TestCase):

    def _make_resp(self, status: int, content_type: str, body=b"", text="",
                   json_value=None):
        resp = mock.Mock()
        resp.status_code = status
        resp.headers = {"Content-Type": content_type}
        resp.content = body
        resp.text = text or (body.decode() if body else "")
        if json_value is not None:
            resp.json.return_value = json_value
        return resp

    def _spec(self):
        from prismpy.sources.gaez import resolve_esri_query
        return resolve_esri_query("Highland_sorghum", "rf", "Low", "LUT")

    def test_fetch_image_200_tiff_returns_bytes(self):
        from prismpy.sources.gaez import EsriImageServiceClient

        client = EsriImageServiceClient(retries=0, backoff=0.0)
        resp = self._make_resp(200, "image/tiff", body=b"II*\x00fake tiff bytes")
        with mock.patch("requests.get", return_value=resp):
            data = client.fetch_image(self._spec())
        self.assertEqual(data[:4], b"II*\x00")

    def test_fetch_image_200_json_in_band_error(self):
        from prismpy.sources.gaez import EsriFetchError, EsriImageServiceClient

        client = EsriImageServiceClient(retries=0, backoff=0.0)
        env = {"error": {"code": 400, "message": "Bad mosaicRule"}}
        resp = self._make_resp(200, "application/json", json_value=env)
        with mock.patch("requests.get", return_value=resp):
            with self.assertRaises(EsriFetchError) as ctx:
                client.fetch_image(self._spec())
        self.assertEqual(ctx.exception.status_code, 0)
        self.assertIn("ESRI_ERROR", ctx.exception.message)
        self.assertIn("400", ctx.exception.message)

    def test_fetch_image_403_typed_status(self):
        from prismpy.sources.gaez import EsriFetchError, EsriImageServiceClient

        client = EsriImageServiceClient(retries=0, backoff=0.0)
        resp = self._make_resp(403, "text/plain", body=b"Forbidden", text="Forbidden")
        with mock.patch("requests.get", return_value=resp):
            with self.assertRaises(EsriFetchError) as ctx:
                client.fetch_image(self._spec())
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("HTTP 403", ctx.exception.message)

    def test_fetch_image_timeout(self):
        import requests
        from prismpy.sources.gaez import EsriFetchError, EsriImageServiceClient

        client = EsriImageServiceClient(retries=0, backoff=0.0)
        with mock.patch("requests.get", side_effect=requests.exceptions.Timeout("t")):
            with self.assertRaises(EsriFetchError) as ctx:
                client.fetch_image(self._spec())
        self.assertEqual(ctx.exception.status_code, 0)
        self.assertIn("TIMEOUT", ctx.exception.message)

    def test_fetch_image_connection_error(self):
        import requests
        from prismpy.sources.gaez import EsriFetchError, EsriImageServiceClient

        client = EsriImageServiceClient(retries=0, backoff=0.0)
        with mock.patch(
            "requests.get",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            with self.assertRaises(EsriFetchError) as ctx:
                client.fetch_image(self._spec())
        self.assertIn("CONNECTION_ERROR", ctx.exception.message)

    def test_fetch_image_retry_then_success(self):
        import requests
        from prismpy.sources.gaez import EsriImageServiceClient

        client = EsriImageServiceClient(retries=2, backoff=1.0,
                                        jitter_range=(1.0, 1.0))
        ok = self._make_resp(200, "image/tiff", body=b"II*\x00ok")
        # First call raises Timeout; second returns success.
        with mock.patch(
            "requests.get",
            side_effect=[requests.exceptions.Timeout("t"), ok],
        ):
            with mock.patch("time.sleep") as sleep_mock:
                data = client.fetch_image(self._spec())
        self.assertEqual(data[:4], b"II*\x00")
        sleep_mock.assert_called()


# ---------------------------------------------------------------------------
# AC-F-CP-6 — Honest-signal HTTP status surfacing on legacy retry helper (3 tests)
# ---------------------------------------------------------------------------


class TestHonestSignalHttpStatusOnLegacyRetry(unittest.TestCase):
    """The legacy ``_download_with_retry`` path stays in place (back-compat)
    but the existing 403 / timeout / connection bug class needs explicit
    test coverage per builder DELTA-CA-1 + codex M3."""

    def test_log_surfaces_http_403(self):
        import requests
        from prismpy.sources.gaez import GAEZDownloader

        with self.assertLogs("prismpy.sources.gaez.downloader", level="ERROR") as cm:
            resp = mock.Mock()
            resp.status_code = 403
            err = requests.exceptions.HTTPError(response=resp)
            with mock.patch("requests.get") as get:
                bad = mock.Mock()
                bad.raise_for_status.side_effect = err
                bad.iter_content.return_value = iter([])
                get.return_value = bad
                with mock.patch("time.sleep"):
                    d = GAEZDownloader(retries=0)
                    ok, msg = d._download_with_retry(
                        "https://example.invalid/x.tif",
                        Path("/tmp/_pin_403_target.tif"),
                        overwrite=True,
                    )
        self.assertFalse(ok)
        self.assertEqual(msg, "HTTP 403")
        self.assertTrue(any("HTTP 403" in line for line in cm.output))

    def test_log_surfaces_timeout(self):
        import requests
        from prismpy.sources.gaez import GAEZDownloader

        with self.assertLogs("prismpy.sources.gaez.downloader", level="ERROR") as cm:
            with mock.patch(
                "requests.get",
                side_effect=requests.exceptions.Timeout("t"),
            ):
                with mock.patch("time.sleep"):
                    d = GAEZDownloader(retries=0)
                    ok, msg = d._download_with_retry(
                        "https://example.invalid/x.tif",
                        Path("/tmp/_pin_to_target.tif"),
                        overwrite=True,
                    )
        self.assertFalse(ok)
        self.assertEqual(msg, "TIMEOUT")
        self.assertTrue(any("TIMEOUT" in line for line in cm.output))

    def test_log_surfaces_connection_error(self):
        import requests
        from prismpy.sources.gaez import GAEZDownloader

        with self.assertLogs("prismpy.sources.gaez.downloader", level="ERROR") as cm:
            with mock.patch(
                "requests.get",
                side_effect=requests.exceptions.ConnectionError("refused"),
            ):
                with mock.patch("time.sleep"):
                    d = GAEZDownloader(retries=0)
                    ok, msg = d._download_with_retry(
                        "https://example.invalid/x.tif",
                        Path("/tmp/_pin_ce_target.tif"),
                        overwrite=True,
                    )
        self.assertFalse(ok)
        self.assertEqual(msg, "CONNECTION_ERROR")
        self.assertTrue(any("CONNECTION_ERROR" in line for line in cm.output))


# ---------------------------------------------------------------------------
# AC-F-CP-9 — ThreadPoolExecutor / max_workers back-compat + serial (3 tests)
# ---------------------------------------------------------------------------


class TestMaxWorkersBackCompatAndSerial(unittest.TestCase):

    def test_max_workers_default_is_ten(self):
        from prismpy.sources.gaez import GAEZDownloader
        d = GAEZDownloader()
        self.assertEqual(d.max_workers, 10)

    def test_max_workers_serial_path(self):
        """``max_workers=1`` runs serial without ThreadPoolExecutor errors."""
        import tempfile
        from prismpy.sources.gaez import GAEZDownloader

        client = mock.Mock()
        client.fetch_image.return_value = b"II*\x00fake"
        with tempfile.TemporaryDirectory() as tmp:
            d = GAEZDownloader(cache_dir=Path(tmp), max_workers=1, client=client)
            result = d.download_cultivar(
                "Highland_sorghum", water_supply="rf",
                input_levels=["Low"], overwrite=True,
            )
        # 1 level × 3 vars = 3 rasters
        self.assertEqual(len(result), 3)
        self.assertEqual(client.fetch_image.call_count, 3)

    def test_thread_pool_executor_cleanup(self):
        """ThreadPoolExecutor exits cleanly after fan-out (no leaked threads)."""
        import tempfile
        from prismpy.sources.gaez import GAEZDownloader

        client = mock.Mock()
        client.fetch_image.return_value = b"II*\x00fake"
        threads_before = threading.active_count()
        with tempfile.TemporaryDirectory() as tmp:
            d = GAEZDownloader(cache_dir=Path(tmp), max_workers=4, client=client)
            d.download_cultivar(
                "Highland_sorghum", water_supply="rf",
                input_levels=["High", "Low"], overwrite=True,
            )
        # Pool's worker threads are not guaranteed to be fully reaped
        # the instant ``with`` exits, so allow a small tolerance.
        threads_after = threading.active_count()
        self.assertLessEqual(
            threads_after - threads_before, 1,
            "ThreadPoolExecutor must not leak persistent threads "
            "(per AC-F-CP-9).",
        )


# ---------------------------------------------------------------------------
# AC-F-CP-10 — GAEZDownloadError propagation (3 tests)
# ---------------------------------------------------------------------------


class TestGaezDownloadErrorPropagation(unittest.TestCase):

    def test_partial_failure_raises_typed_error(self):
        """Any per-raster failure -> ``GAEZDownloadError`` (no silent {})."""
        import tempfile
        from prismpy.sources.gaez import (
            EsriFetchError, GAEZDownloadError, GAEZDownloader,
        )

        client = mock.Mock()
        # First call succeeds; second raises; third succeeds.
        call_count = {"n": 0}

        def fake_fetch(query, bbox="-180,-90,180,90", size="4320,2160",
                       **kwargs):
            # Ship 1' added cancel_check / progress_callback kwargs to the
            # real fetch_image; the mock must accept them.
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise EsriFetchError(403, "HTTP 403")
            return b"II*\x00fake"

        client.fetch_image.side_effect = fake_fetch
        with tempfile.TemporaryDirectory() as tmp:
            d = GAEZDownloader(cache_dir=Path(tmp), max_workers=1, client=client)
            with self.assertRaises(GAEZDownloadError) as ctx:
                d.download_cultivar(
                    "Highland_sorghum", water_supply="rf",
                    input_levels=["Low"], overwrite=True,
                )
        self.assertGreaterEqual(ctx.exception.summary.failed, 1)
        self.assertEqual(len(ctx.exception.summary.errors), 1)

    def test_all_failure_raises_with_aggregate_count(self):
        """All-503 simulation surfaces aggregate failure count."""
        import tempfile
        from prismpy.sources.gaez import (
            EsriFetchError, GAEZDownloadError, GAEZDownloader,
        )

        client = mock.Mock()
        client.fetch_image.side_effect = EsriFetchError(503, "HTTP 503")
        with tempfile.TemporaryDirectory() as tmp:
            d = GAEZDownloader(cache_dir=Path(tmp), max_workers=1, client=client)
            with self.assertRaises(GAEZDownloadError) as ctx:
                d.download_cultivar(
                    "Highland_sorghum", water_supply="rf",
                    input_levels=["High", "Low"], overwrite=True,
                )
        # 2 levels × 3 vars = 6 rasters, all fail.
        self.assertEqual(ctx.exception.summary.succeeded, 0)
        self.assertEqual(ctx.exception.summary.failed, 6)

    def test_acea_translator_propagates_gaez_download_error(self):
        """ACEA `_handle_gaez_data` re-raises ``GAEZDownloadError``.

        Replaces the previous ``logger.warning(...) + return []`` silent-
        skip pattern that caused the F-AG-class silent ``complete`` state.
        """
        import inspect as _inspect
        import prismpy.translators.acea.translator as acea_mod

        source = _inspect.getsource(acea_mod.AceaTranslator._handle_gaez_data)
        # The fixup MUST contain an explicit ``except GAEZDownloadError:``
        # branch that re-raises, NOT swallows.
        self.assertIn(
            "except GAEZDownloadError", source,
            "ACEA translator must re-raise GAEZDownloadError per AC-F-CP-10.",
        )
        # No remaining ``logger.warning(f\"GAEZ download failed:`` swallow.
        self.assertNotIn(
            'logger.warning(f"GAEZ download failed:', source,
            "ACEA translator must not swallow GAEZ download failures "
            "via logger.warning per AC-F-CP-10 + F-AG-class precedent.",
        )


# ---------------------------------------------------------------------------
# AC-F-CP-11 — LUT pixel-range parity (1 test)
# ---------------------------------------------------------------------------


class TestLutPixelRangeParity(unittest.TestCase):
    """Documents the empirically-verified pixel-range parity between Esri
    res02 HIST LUT rasters and the historical S3 ``_idx.tif`` cache shape.

    The full live-fetch comparison runs in evaluator's canonical Gate B
    (AC-F-CP-7). This pin is the documentation anchor: future contributors
    reading the module docstring must find the empirical evidence here.
    """

    def test_lut_pixel_range_documented_in_raster_mapping_module(self):
        """Raster-mapping module docstring records the empirical pixel range."""
        import prismpy.sources.gaez.raster_mapping as rm

        doc = rm.__doc__ or ""
        # The docstring must mention the empirical comparison anchors so a
        # future contributor reading the module finds the audit trail.
        for token in ("-9", "39", "82", "Esri", "S3"):
            self.assertIn(
                token, doc,
                f"raster_mapping.py docstring must document the LUT "
                f"pixel-range parity evidence (missing token: {token!r}).",
            )


# ---------------------------------------------------------------------------
# AC-F-CP-1 — One last sub-pin: schema-location consistency (codex M1 ask)
# ---------------------------------------------------------------------------


class TestSchemaLocationConsistency(unittest.TestCase):
    """Per codex M1: each Pydantic schema lives in exactly one module.

    ``EsriExportImageRequest`` + ``EsriErrorResponse`` live in
    ``esri_schemas.py``; ``EsriQuerySpec`` lives in ``raster_mapping.py``
    (closer to the vocab translator that constructs it). Cross-module
    re-imports are fine; class definitions are pinned to one location.
    """

    def test_each_schema_defined_in_exactly_one_module(self):
        from prismpy.sources.gaez import esri_schemas, raster_mapping

        # Class defined in its own module (i.e., not imported from elsewhere)
        self.assertEqual(
            esri_schemas.EsriExportImageRequest.__module__,
            "prismpy.sources.gaez.esri_schemas",
        )
        self.assertEqual(
            esri_schemas.EsriErrorResponse.__module__,
            "prismpy.sources.gaez.esri_schemas",
        )
        self.assertEqual(
            raster_mapping.EsriQuerySpec.__module__,
            "prismpy.sources.gaez.raster_mapping",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
