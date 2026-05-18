"""GAEZ data source module for prismpy.

Public surface preserved across the Esri Image Service migration (Sprint
F-CP). Existing callers ``from prismpy.sources.gaez import ...`` continue
to work unchanged.

The new modules (``errors``, ``esri_schemas``, ``raster_mapping``,
``esri_client``) are also exposed at the package level so test pins and
internal callers can import them by their canonical name without reaching
into ``prismpy.sources.gaez.<submodule>``.
"""

from prismpy.sources.gaez.downloader import (
    GAEZDownloader,
    GAEZ_CROP_CODES,
    GAEZ_CULTIVAR_MAP,
    GAEZ_VARIABLES,
    INPUT_LEVELS,
)
from prismpy.sources.gaez.errors import (
    EsriFetchError,
    GAEZDownloadError,
    GAEZFetchSummary,
    UnknownCombination,
)
from prismpy.sources.gaez.esri_client import (
    EsriImageServiceClient,
    SERVICE_BASE_URL,
)
from prismpy.sources.gaez.esri_schemas import (
    EsriErrorResponse,
    EsriExportImageRequest,
)
from prismpy.sources.gaez.raster_mapping import (
    EsriQuerySpec,
    resolve_esri_query,
)

__all__ = [
    # Pre-existing public surface.
    "GAEZDownloader",
    "GAEZ_CROP_CODES",
    "GAEZ_CULTIVAR_MAP",
    "GAEZ_VARIABLES",
    "INPUT_LEVELS",
    # New surface for the Esri migration.
    "EsriFetchError",
    "GAEZDownloadError",
    "GAEZFetchSummary",
    "UnknownCombination",
    "EsriImageServiceClient",
    "SERVICE_BASE_URL",
    "EsriErrorResponse",
    "EsriExportImageRequest",
    "EsriQuerySpec",
    "resolve_esri_query",
]
