"""GAEZ data source module for prismpy."""

from prismpy.sources.gaez.downloader import (
    GAEZDownloader,
    GAEZ_CROP_CODES,
    GAEZ_CULTIVAR_MAP,
)

__all__ = [
    "GAEZDownloader",
    "GAEZ_CROP_CODES",
    "GAEZ_CULTIVAR_MAP",
]
