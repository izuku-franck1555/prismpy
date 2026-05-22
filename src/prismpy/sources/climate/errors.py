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

from typing import List, Optional


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
    ) -> None:
        self.missing_tiles: List[int] = list(missing_tiles or [])
        self.source = source
        # ``total`` is in the SAME unit as ``len(missing_tiles)`` (e.g.
        # 30-arcmin cell count, not pixel-grid size) so a partial-progress
        # derivation downstream reports honest counts in matching units.
        self.total = total
        prefix = f"[{source}] " if source else ""
        suffix = (
            f" ({len(self.missing_tiles)} unfetched IDs: "
            f"{self.missing_tiles[:5]}{'...' if len(self.missing_tiles) > 5 else ''})"
            if self.missing_tiles
            else ""
        )
        super().__init__(f"{prefix}{message}{suffix}")
