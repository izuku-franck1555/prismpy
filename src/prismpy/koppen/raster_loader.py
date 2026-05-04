"""Beck 2023 KG raster substrate paths + zone-code mapping.

Wraps the bundled Köppen-Geiger raster file paths and the
30-zone code-to-name mapping derived from the legend. The
actual sampling logic lives in
:mod:`prismpy.koppen.kg_classifier`; the transitional-cell
detection lives in :mod:`prismpy.koppen.transitional`.

Per AC-Q1-A and Beck et al. 2023 (Sci Data 10:724,
DOI 10.1038/s41597-023-02549-6, CC-BY 4.0):

* Raster format: GeoTIFF, single-band, ``uint8``.
* Resolution: 1/120° (≈ 1 km at the equator). Bounds: global.
* CRS: EPSG:4326 (WGS84 lat/lon).
* nodata: 0 (ocean / no data).
* 30 zone codes (1-30) per the legend.

The 1991-2020 historical period file is bundled as
``data/beck_2023_v1.tif`` next to this module. The
accompanying legend ships as ``data/beck_2023_v1_legend.txt``.
Both files are declared in the ``[tool.setuptools.package-data]``
block of ``pyproject.toml`` so a pip-installed prismpy
includes them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final


# Path to the bundled Beck 2023 1km KG raster. The file
# ships with the prismpy wheel; bundled at install time via
# the pyproject [tool.setuptools.package-data] declaration.
BECK_2023_RASTER_PATH: Final[Path] = (
    Path(__file__).parent / "data" / "beck_2023_v1.tif"
)


# Path to the bundled Beck 2023 legend describing the 30 KG
# zone codes + their human-readable names.
BECK_2023_LEGEND_PATH: Final[Path] = (
    Path(__file__).parent / "data" / "beck_2023_v1_legend.txt"
)


# Native cell size of the Beck 2023 1km raster in degrees.
# 1/120° ≈ 0.008333°. Used for sub-pixel jitter convention
# (per AC-Q1-A) and for 8-neighbor offsets in transitional-
# cell detection (per CC-13 + research doc §Q1.3).
NATIVE_CELL_DEG: Final[float] = 1.0 / 120.0


# Reserved nodata code in the raster (ocean / no data).
NODATA_CODE: Final[int] = 0


# Mapping from raster integer code (1-30) to the canonical
# Köppen-Geiger 2-3 letter zone name per Beck 2023
# ``legend.txt``. Code 0 is reserved for nodata (ocean) and
# is intentionally absent from this mapping.
KG_CODE_TO_NAME: Final[dict[int, str]] = {
    1: "Af",   2: "Am",   3: "Aw",
    4: "BWh",  5: "BWk",  6: "BSh",  7: "BSk",
    8: "Csa",  9: "Csb",  10: "Csc",
    11: "Cwa", 12: "Cwb", 13: "Cwc",
    14: "Cfa", 15: "Cfb", 16: "Cfc",
    17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd",
    21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd",
    25: "Dfa", 26: "Dfb", 27: "Dfc", 28: "Dfd",
    29: "ET",  30: "EF",
}


# Reverse mapping from zone name to integer code.
KG_NAME_TO_CODE: Final[dict[str, int]] = {
    name: code for code, name in KG_CODE_TO_NAME.items()
}
