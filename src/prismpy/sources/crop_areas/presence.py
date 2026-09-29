"""The one crop-presence reader: SPAM harvested area per grid cell, read once over the cells'
window. Every no-area value (the declared nodata, a non-finite or a negative value) reads as 0 ha,
and when the caller froze the layer's identity, the file must match it before any value is used.

Callers: the crop-presence roster rule, prismweb's review estimate and CRAFT's crop-mask shares.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

RES_5_ARCMIN = 5 / 60
# SPAM's stored pixel is 0.0833333332727273 deg, not exactly 5/60: the lattice is matched within this.
_LATTICE_TOLERANCE_DEG = 1e-6
_ORIGIN_TOLERANCE_PX = 1e-3
IDENTITY_FIELDS = ("sha256", "crs", "transform", "band", "width", "height", "nodata")


class CropPresenceIdentityError(ValueError):
    """The layer is not the one the rule froze, is not on the 5' geographic lattice, does not
    cover every cell, or was not resolved for this run."""


class CropPresenceEmptyError(ValueError):
    """No cell of the region has harvested area > 0 in the rule's layer."""


def crop_area_present(area_ha: Any) -> bool:
    """The presence predicate: harvested area strictly above 0 ha, finite."""
    return (isinstance(area_ha, (int, float)) and not isinstance(area_ha, bool)
            and math.isfinite(area_ha) and area_ha > 0)


def nodata_class(nodata: Optional[float]) -> Union[None, str, float]:
    """A layer's nodata as a JSON-safe class: None, the "nan" sentinel, or the finite value."""
    if nodata is None:
        return None
    return "nan" if math.isnan(nodata) else float(nodata)


@dataclass(frozen=True)
class RasterIdentity:
    sha256: str
    crs: str
    transform: Tuple[float, float, float, float, float, float]
    band: int
    width: int
    height: int
    nodata: Union[None, str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {"sha256": self.sha256, "crs": self.crs, "transform": list(self.transform),
                "band": self.band, "width": self.width, "height": self.height, "nodata": self.nodata}


@dataclass(frozen=True)
class CellPresence:
    """The harvested area of every requested cell (the exact input keyset), the identity of the
    layer read, the one window read (col_off, row_off, width, height) and whether every cell lies
    on the layer."""
    areas: Dict[int, float]
    observed: RasterIdentity
    window: Tuple[int, int, int, int]
    covered: bool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_identity(expected: Mapping[str, Any], observed: RasterIdentity) -> Optional[str]:
    """The first identity field that differs, or None."""
    obs = observed.as_dict()
    for field in IDENTITY_FIELDS:
        want, got = expected.get(field), obs[field]
        if field == "transform":
            if (not isinstance(want, (list, tuple)) or len(want) != 6
                    or any(not math.isclose(float(w), g, rel_tol=0.0, abs_tol=1e-12) for w, g in zip(want, got))):
                return field
        elif want != got:
            return field
    return None


def _on_geographic_5_arcmin_lattice(crs, transform) -> bool:
    if crs is None or crs.to_epsg() != 4326:
        return False
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    if abs(b) > 1e-12 or abs(d) > 1e-12:
        return False
    if abs(a - RES_5_ARCMIN) > _LATTICE_TOLERANCE_DEG or abs(e + RES_5_ARCMIN) > _LATTICE_TOLERANCE_DEG:
        return False
    col0, row0 = (c + 180.0) / a, (90.0 - f) / -e
    return abs(col0 - round(col0)) < _ORIGIN_TOLERANCE_PX and abs(row0 - round(row0)) < _ORIGIN_TOLERANCE_PX


def cell_presence(
    cells: Iterable[Any],
    raster_path: Union[str, Path],
    *,
    expected: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
) -> CellPresence:
    """Harvested area (ha) for every cell, from ONE open and ONE windowed read of the layer.

    Each cell reads the pixel that contains its centre (at 5' the pixel IS the cell footprint).
    With ``expected`` (a frozen identity), the file's sha256, CRS, transform, band, dims and nodata
    class must match it and the layer must sit on the geographic 5' lattice, before any value is
    used. ``strict`` refuses a cell that lies off the layer; otherwise that cell reads 0 ha.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.windows import Window

    cells = list(cells)
    path = Path(raster_path)
    band = int(expected.get("band", 1)) if expected is not None else 1
    with rasterio.open(path) as src:
        observed = _identity(path, src, band)
        if expected is not None:
            if band < 1 or band > src.count:
                raise CropPresenceIdentityError(f"{path.name}: band {band} is not in the layer")
            field = _same_identity(expected, observed)
            if field is not None:
                raise CropPresenceIdentityError(
                    f"{path.name}: the layer's {field} differs from the rule's frozen identity")
            if not _on_geographic_5_arcmin_lattice(src.crs, src.transform):
                raise CropPresenceIdentityError(f"{path.name}: the layer is not on the geographic 5' lattice")

        if not cells:
            return CellPresence(areas={}, observed=observed, window=(0, 0, 0, 0), covered=True)
        rows, cols = rowcol(src.transform, [c.lon for c in cells], [c.lat for c in cells])
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        covered = bool(inside.all())
        if strict and not covered:
            raise CropPresenceIdentityError(
                f"{path.name}: the layer does not cover {int((~inside).sum())} of {len(cells)} cells")
        areas: Dict[int, float] = {c.cell_id: 0.0 for c in cells}
        if not inside.any():
            return CellPresence(areas=areas, observed=observed, window=(0, 0, 0, 0), covered=covered)
        r0, r1 = int(rows[inside].min()), int(rows[inside].max())
        c0, c1 = int(cols[inside].min()), int(cols[inside].max())
        window = (c0, r0, c1 - c0 + 1, r1 - r0 + 1)
        values = src.read(band, window=Window(*window)).astype("float64")
        nodata = src.nodata

    for c, row, col, ok in zip(cells, rows.tolist(), cols.tolist(), inside.tolist()):
        if not ok:
            continue
        value = float(values[row - r0, col - c0])
        if nodata is not None and not math.isnan(nodata) and value == nodata:
            continue
        if crop_area_present(value):
            areas[c.cell_id] = value
    return CellPresence(areas=areas, observed=observed, window=window, covered=covered)


def raster_identity(raster_path: Union[str, Path], *, band: int = 1) -> RasterIdentity:
    """The identity a caller freezes for a layer: the fields ``cell_presence`` verifies."""
    import rasterio

    path = Path(raster_path)
    with rasterio.open(path) as src:
        if band < 1 or band > src.count:
            raise CropPresenceIdentityError(f"{path.name}: band {band} is not in the layer")
        return _identity(path, src, band)


def _identity(path: Path, src, band: int) -> RasterIdentity:
    return RasterIdentity(
        sha256=_file_sha256(path), crs=src.crs.to_string() if src.crs else "",
        transform=tuple(float(v) for v in tuple(src.transform)[:6]), band=band,
        width=int(src.width), height=int(src.height), nodata=nodata_class(src.nodata))
