"""Typed errors + per-fetch summary for the GAEZ Esri Image Service path.

Replaces the previous silent-skip class where ``download_cultivar`` returned
an empty dict on every failure and the caller logged a warning that never
propagated to the top-level pipeline status. The fail-loud chain is:

    EsriImageServiceClient.fetch_image
        -> raises EsriFetchError(status, message)
    GAEZDownloader.download_cultivar (per-raster loop)
        -> accumulates EsriFetchError into GAEZFetchSummary
        -> raises GAEZDownloadError(summary) if summary.failed > 0
    translators/acea/translator.py::_handle_gaez_data
        -> catches GAEZDownloadError -> TranslationResult(success=False)
    prismpy.pipeline.executor + prismweb pipeline runner
        -> propagates success=False -> PipelineRun.status='error'

This closes the F-AG-class status-propagation gap pinned in the F-CP contract
AC-F-CP-10: any GAEZ raster unfetchable triggers an honest ``error`` status,
never a silent ``complete`` while N/N rasters failed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


class EsriFetchError(Exception):
    """Raised by ``EsriImageServiceClient.fetch_image`` on every failure mode.

    ``status_code`` mirrors the HTTP status when the server replied with one;
    is ``0`` for transport-level failures (timeout, connection refused, DNS),
    and is ``0`` when the server replied 200 + JSON-encoded ``error`` (the
    Esri ImageServer's documented in-band error shape) — the wrapped message
    starts with ``ESRI_ERROR:`` so callers can distinguish in-band errors from
    transport-level failures even though both share ``status_code == 0``.
    """

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"EsriFetchError[{status_code}]: {message}")


class UnknownCombination(KeyError):
    """Raised by ``raster_mapping.resolve_esri_query`` when a 4-tuple
    ``(cultivar, water_supply, input_level, gaez_var)`` has no Esri mapping.

    Inherits from ``KeyError`` so legacy ``dict.get`` style callers can still
    catch it as a missing-key signal; the explicit type lets the downloader
    surface the unknown 4-tuple in the fail-loud chain via ``EsriFetchError``
    framing so the cause is preserved through the pipeline status emit.
    """


@dataclass(frozen=True)
class GAEZFetchSummary:
    """Per-cultivar fan-out summary.

    Built by ``download_cultivar``; consumed by ``GAEZDownloadError``. Carries
    the per-raster ``EsriFetchError`` list so Dr. Kofi's audit-grep workflow
    can reconstruct the failure mode for every raster in the failed run.
    """

    succeeded: int
    failed: int
    errors: List[EsriFetchError] = field(default_factory=list)

    def is_complete(self) -> bool:
        return self.failed == 0


class GAEZDownloadError(Exception):
    """Raised by ``GAEZDownloader.download_cultivar`` when ``failed > 0``.

    Replaces the previous ``return {}`` silent-skip on per-cultivar failure.
    The fail-loud propagation chain closes the F-AG-class precedent: a GAEZ
    retrieval failure now surfaces honestly at ``PipelineRun.status='error'``
    rather than masquerading as a successful ``complete`` run with missing
    raster files in the package artifacts.
    """

    def __init__(self, summary: GAEZFetchSummary, cultivar: Optional[str] = None) -> None:
        self.summary = summary
        self.cultivar = cultivar
        cult_prefix = f"cultivar={cultivar!r}: " if cultivar else ""
        super().__init__(
            f"{cult_prefix}GAEZ retrieval: "
            f"{summary.succeeded}/{summary.succeeded + summary.failed} succeeded; "
            f"{summary.failed} errors"
        )
