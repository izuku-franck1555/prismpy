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
from typing import Any, Callable, Optional

from prismpy.sources.climate._cancel import raise_if_cancelled
from prismpy.sources.common.retry import (
    _bridge_helper_on_attempt,
    retry_with_exponential_backoff,
)
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
        cancel_check: Optional[Callable[[], bool]] = None,
        progress_callback: Any = None,
    ) -> bytes:
        """Fetch a single raster as TIFF bytes.

        Raises ``EsriFetchError`` on every failure mode (the single
        fail-loud surface keeps callers free of dual-shape success/error
        handling). Retries via the canonical helper. Two non-obvious notes:

        * Cancellation is cooperative: ``cancel_check`` is polled at the top
          of every attempt and before each backoff sleep. ``requests.get``
          blocks up to ``self.timeout`` and cannot be interrupted mid-flight,
          so worst-case cancel latency is one ``timeout`` window.
        * In-band Esri errors (HTTP 200 + a JSON ``{"error": {...}}``
          envelope) are RAISED, not returned — returning the error body as
          raster bytes would be silent corruption.
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

        attempt_counter = {"i": 0}

        def _attempt() -> bytes:
            # Top-of-attempt cancel check (also covers the post-sleep path).
            raise_if_cancelled(
                cancel_check, f"esri.fetch.attempt={attempt_counter['i']}"
            )
            attempt_counter["i"] += 1

            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.exceptions.Timeout as e:
                raise EsriFetchError(0, f"TIMEOUT: {e}") from e
            except requests.exceptions.ConnectionError as e:
                raise EsriFetchError(0, f"CONNECTION_ERROR: {e}") from e
            except requests.exceptions.RequestException as e:
                raise EsriFetchError(0, f"REQUEST_ERROR: {e}") from e

            # No transport-level exception. Inspect the response.
            status = resp.status_code
            content_type = (resp.headers.get("Content-Type") or "").lower()

            if status == 200 and content_type.startswith("image/"):
                return resp.content

            if status == 200 and "application/json" in content_type:
                # In-band error envelope: raise so the helper retries it;
                # returning the error body as image bytes is silent corruption.
                try:
                    body = resp.json()
                except ValueError as e:
                    raise EsriFetchError(
                        0, f"ESRI_ERROR: non-JSON body: {e}"
                    ) from e
                err = body.get("error") if isinstance(body, dict) else None
                if not isinstance(err, dict):
                    raise EsriFetchError(
                        0, f"ESRI_ERROR: unrecognized JSON shape: {body!r}"
                    )
                try:
                    parsed = EsriErrorResponse.model_validate(err)
                except Exception as e:  # pragma: no cover - schema drift
                    raise EsriFetchError(
                        0, f"ESRI_ERROR: parse failure: {e}"
                    ) from e
                raise EsriFetchError(
                    0, f"ESRI_ERROR: {parsed.code} {parsed.message}"
                )

            # HTTP 4xx / 5xx — preserve the actual status code.
            try:
                body_snippet = resp.text[:200] if resp.text else ""
            except Exception:
                body_snippet = ""
            raise EsriFetchError(status, f"HTTP {status}: {body_snippet}")

        def _on_retry(attempt_index, exc, sleep_s):
            # Pre-sleep cancel check so a cancel aborts during the backoff.
            raise_if_cancelled(
                cancel_check, f"esri.before_retry={attempt_index}"
            )

        on_attempt = _bridge_helper_on_attempt(
            progress_callback, "translate", "GAEZ"
        )

        # initial attempt + configured retries (retries=0 → 1 attempt).
        max_attempts = max(1, int(self.retries) + 1)

        # base 1.0 s grown by self.backoff each attempt so a caller's custom
        # backoff is honoured; jitter is the helper's additive form
        # (jitter_range is no longer used).
        try:
            return retry_with_exponential_backoff(
                _attempt,
                max_attempts=max_attempts,
                base_delay_s=1.0,
                jitter_ratio=0.2,
                backoff_multiplier=float(self.backoff),
                exception_classes=(EsriFetchError,),
                on_retry=_on_retry,
                on_attempt=on_attempt,
            )
        except EsriFetchError as exc:
            logger.error(
                f"Esri fetch failed for query={query.to_where()!r}: "
                f"{exc.message} (after {max_attempts - 1} retries)"
            )
            raise
