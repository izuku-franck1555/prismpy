"""Raw REST wrapper around FAO's GAEZ Esri ImageServer.

This module is the SOLE owner of the ``gaez-services.fao.org`` hostname in
``src/prismpy/`` per durable §24 canonical-source-or-pin discipline. The
ban-pin ``test_no_hardcoded_image_service_urls_outside_client`` walks the
source tree and asserts no other file embeds this hostname; the paired
ban-pin ``test_no_old_s3_strings_anywhere`` asserts no file embeds the
deprecated FAO dev S3 hostname anywhere in the package.

Endpoint contract (empirically validated against the live service on
2026-05-11):
  - ``maxImageHeight = 4100``, ``maxImageWidth = 15000`` per service
    metadata; global 5-arc-min export (4320×2160) fits cleanly.
  - Anonymous access; no API key required.
  - Service replies ``200 OK + Content-Type: image/tiff`` on success;
    ``200 OK + Content-Type: application/json`` with an in-band error
    envelope on attribute-filter / mosaicRule errors (per ``EsriErrorResponse``).
  - HTTP 4xx/5xx returned on transport-level failures (rate limit, server
    error, malformed URL).

Retry policy: exponential backoff with jitter; ``retries`` and ``backoff``
mirror the previous S3-era ``GAEZDownloader._download_with_retry`` contract
so the behavior under transient failure is unchanged from the consumer's
perspective. Jitter (per WA CA-3 R13 mitigation) avoids thundering-herd
retries when many concurrent workers hit a transient Esri outage at once.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Optional

from prismpy.sources.gaez.errors import EsriFetchError
from prismpy.sources.gaez.esri_schemas import (
    EsriErrorResponse,
    EsriExportImageRequest,
)
from prismpy.sources.gaez.raster_mapping import EsriQuerySpec

logger = logging.getLogger(__name__)


# Single canonical hostname per §24. Hardcoded inside the client only.
SERVICE_BASE_URL = (
    "https://gaez-services.fao.org/server/rest/services/res02/ImageServer"
)


class EsriImageServiceClient:
    """Thin REST wrapper around the GAEZ res02 HIST ImageServer.

    Anonymous; preserves the existing retry/backoff/jitter contract; surfaces
    typed ``EsriFetchError`` on every failure mode rather than the previous
    ``(False, msg)`` tuple shape so failures propagate honestly through the
    F-AG-class status chain.
    """

    def __init__(
        self,
        retries: int = 3,
        backoff: float = 1.5,
        timeout: float = 60.0,
        jitter_range: tuple = (0.8, 1.2),
        base_url: Optional[str] = None,
    ) -> None:
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.jitter_range = jitter_range
        self.base_url = base_url or SERVICE_BASE_URL

    def fetch_image(
        self,
        query: EsriQuerySpec,
        bbox: str = "-180,-90,180,90",
        size: str = "4320,2160",
    ) -> bytes:
        """Fetch a single raster as TIFF bytes.

        Raises ``EsriFetchError`` on every failure mode. The single fail-loud
        surface keeps the caller free of dual-shape success/error handling
        (the previous ``(success, msg)`` tuple was the source of the F-AG
        silent-skip class).
        """
        try:
            import requests
        except ImportError as e:  # pragma: no cover - declared dep
            raise ModuleNotFoundError(
                "requests is required for GAEZ Esri fetches but did not "
                "import. The package is declared in pyproject.toml; "
                "reinstall prismpy with `pip install -e .` to refresh."
            ) from e

        # Build + validate the wire-level parameters via Pydantic so the
        # request shape is guaranteed before the first attempt.
        request = EsriExportImageRequest(
            bbox=bbox,
            size=size,
            mosaic_rule_json=query.to_mosaic_rule_json(),
        )
        params = request.as_query()
        url = f"{self.base_url}/exportImage"

        # Sensible default UA so the FAO service doesn't bounce us on the
        # urllib default; mirrors the curl probe path that succeeded.
        headers = {"User-Agent": "prismpy/gaez-esri (+https://prismpy.local)"}

        attempt = 0
        wait = 1.0
        last_err: Optional[EsriFetchError] = None

        while attempt <= self.retries:
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.exceptions.Timeout as e:
                last_err = EsriFetchError(0, f"TIMEOUT: {e}")
            except requests.exceptions.ConnectionError as e:
                last_err = EsriFetchError(0, f"CONNECTION_ERROR: {e}")
            except requests.exceptions.RequestException as e:
                last_err = EsriFetchError(0, f"REQUEST_ERROR: {e}")
            else:
                # No transport-level exception. Inspect the response.
                status = resp.status_code
                content_type = (resp.headers.get("Content-Type") or "").lower()

                if status == 200 and content_type.startswith("image/"):
                    return resp.content

                if status == 200 and "application/json" in content_type:
                    # In-band Esri error envelope (the documented edge per
                    # codex M2). Parse via the ``extra='ignore'`` schema.
                    try:
                        body = resp.json()
                    except ValueError as e:
                        last_err = EsriFetchError(
                            0, f"ESRI_ERROR: non-JSON body: {e}"
                        )
                    else:
                        # Esri wraps the envelope in ``{"error": {...}}``.
                        err = body.get("error") if isinstance(body, dict) else None
                        if isinstance(err, dict):
                            try:
                                parsed = EsriErrorResponse.model_validate(err)
                                last_err = EsriFetchError(
                                    0,
                                    f"ESRI_ERROR: {parsed.code} {parsed.message}",
                                )
                            except Exception as e:  # pragma: no cover
                                last_err = EsriFetchError(
                                    0, f"ESRI_ERROR: parse failure: {e}"
                                )
                        else:
                            last_err = EsriFetchError(
                                0, f"ESRI_ERROR: unrecognized JSON shape: {body!r}"
                            )
                else:
                    # HTTP 4xx / 5xx — preserve the actual status code per
                    # builder DELTA-CA-1 + codex M3 honest-signal floor.
                    try:
                        body_snippet = resp.text[:200] if resp.text else ""
                    except Exception:
                        body_snippet = ""
                    last_err = EsriFetchError(
                        status, f"HTTP {status}: {body_snippet}"
                    )

            attempt += 1
            if attempt > self.retries:
                if last_err is None:  # pragma: no cover - defensive
                    last_err = EsriFetchError(0, "max retries exceeded; no error captured")
                logger.error(
                    f"Esri fetch failed for query={query.to_where()!r}: "
                    f"{last_err.message} (after {self.retries} retries)"
                )
                raise last_err

            # Exponential backoff with jitter per WA CA-3 R13 mitigation.
            jitter = random.uniform(*self.jitter_range)
            sleep_for = wait * jitter
            logger.warning(
                f"Retry {attempt}/{self.retries} for Esri fetch "
                f"({last_err.message if last_err else 'unknown'}); "
                f"sleeping {sleep_for:.2f}s"
            )
            time.sleep(sleep_for)
            wait *= self.backoff

        # Defensive: while-loop guarantees we either return or raise above.
        raise EsriFetchError(0, "exhausted retry loop without resolution")
