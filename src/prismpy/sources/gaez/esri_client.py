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

        Raises ``EsriFetchError`` on every failure mode. The single fail-loud
        surface keeps the caller free of dual-shape success/error handling
        (the previous ``(success, msg)`` tuple was the source of the F-AG
        silent-skip class).

        Retry policy is the canonical ``retry_with_exponential_backoff``
        helper (durable #24 canonical-source-or-pin) instead of the former
        bespoke loop. Two semantic notes:

        * **Cancellation** is cooperative: ``cancel_check`` is polled at the
          top of every attempt and again before each backoff sleep, so a
          user cancel aborts at the next attempt boundary. ``requests.get``
          blocks for up to ``self.timeout`` and cannot be interrupted
          mid-flight, so worst-case cancel latency is one ``timeout`` window
          (same model as NASA POWER's per-attempt cancel — honest note I7).
        * **In-band Esri errors** (HTTP 200 + a JSON ``{"error": {...}}``
          envelope) are RAISED as ``EsriFetchError`` rather than recorded in
          a ``last_err`` accumulator, so the canonical helper retries them
          and an exhausted in-band error fails loudly — never returning the
          error body as if it were valid raster bytes (the silent-corruption
          class guarded by S1v2-C4).

        Jitter shifts from the bespoke MULTIPLICATIVE ``random.uniform(0.8,
        1.2)`` to the helper's additive ±``jitter_ratio``; the effective
        per-raster backoff budget is preserved at the same small magnitude.
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
            # Per-attempt cancel check — fires at the top of every attempt,
            # including after a backoff sleep, so a cancel during the sleep
            # window aborts before the next blocking request.
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
                # In-band Esri error envelope (the documented edge per codex
                # M2): RAISE so the canonical helper retries it. Never return
                # the error body as image bytes (S1v2-C4 silent-corruption
                # guard).
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

            # HTTP 4xx / 5xx — preserve the actual status code per builder
            # DELTA-CA-1 + codex M3 honest-signal floor.
            try:
                body_snippet = resp.text[:200] if resp.text else ""
            except Exception:
                body_snippet = ""
            raise EsriFetchError(status, f"HTTP {status}: {body_snippet}")

        def _on_retry(attempt_index, exc, sleep_s):
            # Pre-sleep cancel check — a cancel observed before the backoff
            # wait short-circuits the sleep entirely (mirrors NASA POWER).
            raise_if_cancelled(
                cancel_check, f"esri.before_retry={attempt_index}"
            )

        on_attempt = _bridge_helper_on_attempt(
            progress_callback, "translate", "GAEZ"
        )

        # Honour caller-supplied retry config; max_attempts = initial + the
        # configured retry count — EXACT parity with the former bespoke
        # attempt-counted loop (retries=3 → 4 attempts; retries=0 → 1
        # attempt, no retry). Floor of 1 guards the helper's
        # ``max_attempts >= 1`` contract.
        max_attempts = max(1, int(self.retries) + 1)

        # Reproduce the bespoke schedule exactly: base wait 1.0 s grown by
        # the configured ``self.backoff`` factor each attempt (so callers
        # passing a custom backoff — incl. 0.0 for near-immediate retries —
        # keep their behaviour). Jitter shifts from the bespoke MULTIPLICATIVE
        # ``random.uniform(*self.jitter_range)`` to the helper's additive
        # ±20 % (contract-sanctioned semantic shift; ``jitter_range`` is now a
        # legacy constructor arg that no longer alters the schedule).
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
