"""KG zone classifier built on the Beck 2023 1km raster.

Classifies geographic points (lat, lon in EPSG:4326) into
one of the 30 Köppen-Geiger zones from Beck et al. 2023.
Returns :class:`KGZone` for non-ocean cells, ``None`` for
ocean / nodata.

Sampling is deterministic: rasterio's nearest-neighbor
sampling at the exact lookup point in pixel coordinates
returns the same code for the same input every call. This
is the determinism foundation the bound-generation
substrate (Sprint E.0.5 AC-Q2-B1) and the wizard-time
crop-region compatibility check (AC-Q3-A-a/b/c) build on.

The classifier holds the rasterio dataset open across calls
for performance; use as a context manager or call
``close()`` when done. Tests typically create one instance
in ``setUpClass`` and reuse it.

Future bound-gen work (AC-Q2-A1-c) extends the provenance
trail with the raster DOI + the bounds_version pin.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

import rasterio

from prismpy.koppen.raster_loader import (
    BECK_2023_RASTER_PATH,
    KG_CODE_TO_NAME,
    NODATA_CODE,
)


class KGZone(str, Enum):
    """The 30 Köppen-Geiger climate zones per Beck 2023.

    Subclasses :class:`str` so members are usable as JSON
    keys + string-formatted directly. The enum value matches
    the canonical 2-3 letter Köppen-Geiger code (the same
    string the Beck legend uses).

    Subclassing ``(str, Enum)`` rather than the 3.11+
    :class:`enum.StrEnum` preserves Python 3.10 compatibility
    per ``pyproject.toml`` ``requires-python`` floor.
    """
    Af = "Af"
    Am = "Am"
    Aw = "Aw"
    BWh = "BWh"
    BWk = "BWk"
    BSh = "BSh"
    BSk = "BSk"
    Csa = "Csa"
    Csb = "Csb"
    Csc = "Csc"
    Cwa = "Cwa"
    Cwb = "Cwb"
    Cwc = "Cwc"
    Cfa = "Cfa"
    Cfb = "Cfb"
    Cfc = "Cfc"
    Dsa = "Dsa"
    Dsb = "Dsb"
    Dsc = "Dsc"
    Dsd = "Dsd"
    Dwa = "Dwa"
    Dwb = "Dwb"
    Dwc = "Dwc"
    Dwd = "Dwd"
    Dfa = "Dfa"
    Dfb = "Dfb"
    Dfc = "Dfc"
    Dfd = "Dfd"
    ET = "ET"
    EF = "EF"


# Mapping from raster integer code (1-30) to :class:`KGZone`
# enum member. Built from :data:`KG_CODE_TO_NAME`; failing
# lookups indicate raster corruption (out-of-range code).
KG_CODE_TO_ZONE: dict[int, KGZone] = {
    code: KGZone(name) for code, name in KG_CODE_TO_NAME.items()
}


class KGClassifier:
    """Classify (lat, lon) points to KG zones.

    Holds the rasterio dataset open across calls. Initialize
    once + reuse; call :meth:`close` (or use as a context
    manager) to release the file handle.

    Tests typically create one instance in ``setUpClass``,
    reuse across test methods, and let teardown close it.
    """

    def __init__(self, raster_path: Path | None = None) -> None:
        target = (
            raster_path if raster_path is not None
            else BECK_2023_RASTER_PATH
        )
        self._dataset: Optional[rasterio.io.DatasetReader] = (
            rasterio.open(target)
        )

    def __enter__(self) -> "KGClassifier":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def classify(self, lat: float, lon: float) -> Optional[KGZone]:
        """Sample the raster at (lon, lat) and return the KG zone.

        Returns ``None`` for ocean / nodata cells. Coordinates
        are in EPSG:4326 (WGS84 lat/lon).
        """
        return self._code_to_zone(self._sample_one(lat, lon))

    def classify_batch(
        self, points: Iterable[tuple[float, float]],
    ) -> list[Optional[KGZone]]:
        """Batch classify ``(lat, lon)`` points.

        More efficient than per-point :meth:`classify` calls
        since rasterio amortizes the dataset traversal.
        """
        if self._dataset is None:
            raise RuntimeError(
                "KGClassifier dataset is closed; cannot classify."
            )
        # rasterio.sample expects (x, y) = (lon, lat) tuples
        coords = [(lon, lat) for lat, lon in points]
        codes = [int(v[0]) for v in self._dataset.sample(coords)]
        return [self._code_to_zone(c) for c in codes]

    def _sample_one(self, lat: float, lon: float) -> int:
        if self._dataset is None:
            raise RuntimeError(
                "KGClassifier dataset is closed; cannot classify."
            )
        sample_iter = self._dataset.sample([(lon, lat)])
        return int(next(iter(sample_iter))[0])

    @staticmethod
    def _code_to_zone(code: int) -> Optional[KGZone]:
        if code == NODATA_CODE:
            return None
        zone = KG_CODE_TO_ZONE.get(code)
        if zone is None:
            raise ValueError(
                f"Out-of-range KG raster code: {code}. Beck 2023 "
                f"defines codes 1-30 (plus 0 = nodata)."
            )
        return zone
