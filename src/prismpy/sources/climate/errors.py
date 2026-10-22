"""Typed errors for the climate-source fail-loud propagation chain.

Sibling to ``prismpy/sources/gaez/errors.py``. Closes the F-AG-class
status-propagation gap on the climate axis: when a translator's
translate-time NASA POWER (or other source) download fails to fully
cover the project's grid, the translator raises ``ClimateDownloadError``
so ``BaseTranslator.translate`` returns
``TranslationResult(success=False)`` -> ``executor`` sets
``PipelineRun.status='error'`` (NOT a silent ``complete`` with missing
climate pickle files in the package).
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional, Sequence


class ClimateDownloadError(Exception):
    """Raised by a translator's climate-download gate when, after the
    retry path completes, the grid still has uncovered cells.

    ``missing_tiles`` carries the cell / tile IDs that the translator
    could not fetch so the audit trail in ``pipeline.log`` and
    ``validation_report.json`` records the exact gap. ``source`` is the
    upstream provider name (``"nasa_power"`` / ``"agera5"`` / etc.) so
    the failure mode is greppable for operator forensics (Dr. Kofi's
    audit workflow).
    """

    def __init__(
        self,
        message: str,
        *,
        missing_tiles: Optional[List[int]] = None,
        source: Optional[str] = None,
        total: Optional[int] = None,
        recoverable: Optional[bool] = None,
    ) -> None:
        self.missing_tiles: List[int] = list(missing_tiles or [])
        self.source = source
        # ``total`` is in the SAME unit as ``len(missing_tiles)`` (e.g.
        # 30-arcmin cell count, not pixel-grid size) so a partial-progress
        # derivation downstream reports honest counts in matching units.
        self.total = total
        # ``recoverable`` rides the Ship-3 ``error_event`` (classify reads
        # ``getattr(exc, 'recoverable', None)``) so the consumer can degrade
        # transient-vs-terminal. True = transient (timeout / 429 / 5xx /
        # network — retry-eligible); False = terminal (4xx-non-429 /
        # coordinate-rejected / out-of-domain — not worth retrying); None =
        # unknown. A land-covering provider still uncovered AFTER the retry
        # path is transient by default (the failure exhausted retries, it
        # was not a permanent rejection).
        self.recoverable = recoverable
        prefix = f"[{source}] " if source else ""
        suffix = (
            f" ({len(self.missing_tiles)} unfetched IDs: "
            f"{self.missing_tiles[:5]}{'...' if len(self.missing_tiles) > 5 else ''})"
            if self.missing_tiles
            else ""
        )
        super().__init__(f"{prefix}{message}{suffix}")


class NasaPowerScaledPrecipError(Exception):
    """Raised when NASA POWER's daily precipitation for one cell-year does
    not pass the ×24 scaling guard within its retry budget.

    ``evidence`` holds one text per classified response, in arrival order;
    ``year_end`` is the fetch's own end and ``latest_available`` prismpy's
    offline estimate of the latest published day (never a NASA statement).
    """

    def __init__(
        self,
        *,
        lat: float,
        lon: float,
        year: int,
        evidence: Sequence[str],
        year_end: date,
        latest_available: date,
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.year = year
        self.evidence: List[str] = list(evidence)
        self.year_end = year_end
        self.latest_available = latest_available
        super().__init__(
            f"NASA POWER's daily precipitation for ({lat}, {lon}) did not pass "
            f"the ×24 scaling guard in {len(self.evidence)} consecutive "
            f"responses ({'; '.join(self.evidence)}); this retrieval checked "
            f"{year} through {year_end.isoformat()}; prismpy's conservative "
            f"latest-available estimate was {latest_available.isoformat()}; "
            f"a later retrieval may resolve it"
        )
