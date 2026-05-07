"""Per-cell zone + elevation lookup against HWSD2 + Köppen rasters.

Sprint E.2 AC-E2-5. The cockpit's affordance routing
(``route_affordance`` per AC-E2-3) needs both the Köppen-Geiger
zone code AND the elevation (m) for the Highland-precip exclusion
rule. Both come from raster sampling at the cell centroid:

* **Elevation** — HWSD2 (Harmonized World Soil Database v2; 30
  arcsec ≈ 1 km resolution; Sprint S substrate). The DEM band of
  the HWSD2 raster gives elevation in metres.

* **Zone** — Köppen-Geiger classification raster. The cell centroid
  is sampled and mapped to one of the canonical ``KoppenZone``
  Literal values via ``prismpy.koppen.kg_classifier``.

The lookup is intended to run ONCE per pipeline run during the
preparation phase; the per-cell `(zone, elevation)` tuple is then
persisted on each ``cell_summary.json`` entry as
``cells[].koppen_code`` + ``cells[].elevation_m`` so downstream
affordance-routing reads are O(1).

The module fails gracefully when rasterio is unavailable OR when
the raster paths don't exist — the caller receives a
``LookupSkipped`` exception they can catch to fall back to legacy
placeholder behaviour (e.g., ``elevation_m = -99.0`` per the
existing CRAFT translator pattern). Per ``feedback_no_data_cooking.md``:
the fallback is honestly signalled via the typed exception, NOT a
silent return.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from prismpy.harmonize.idw_interpolation import Cell
from prismpy.koppen.zones import KoppenZone


class LookupSkipped(Exception):
    """Raised when a raster lookup cannot complete (file missing,
    rasterio unavailable, sample out-of-bounds). Caller catches +
    falls back to legacy placeholder behaviour."""


@dataclass(frozen=True)
class ZoneElevationLookup:
    """Result of a per-cell zone + elevation lookup."""

    koppen_code: KoppenZone
    elevation_m: float


def lookup_zone_and_elevation(
    cell: Cell,
    *,
    hwsd2_path: Optional[Path] = None,
    koppen_path: Optional[Path] = None,
) -> ZoneElevationLookup:
    """Sample (zone, elevation) at ``cell``'s centroid via raster
    lookup.

    Args:
        cell: Cell whose lat/lon defines the centroid.
        hwsd2_path: Path to HWSD2 raster carrying the DEM band.
            Required.
        koppen_path: Path to the Köppen-Geiger classification raster.
            Required.

    Returns:
        ``ZoneElevationLookup`` carrying the canonical zone code +
        elevation (m).

    Raises:
        LookupSkipped: When rasterio is unavailable, paths don't
            exist, or the cell falls outside both rasters' bounds.
            Caller catches + falls back to legacy placeholder
            behaviour per ``feedback_no_data_cooking.md`` honest-
            signal contract.
    """
    if hwsd2_path is None or koppen_path is None:
        raise LookupSkipped(
            "lookup_zone_and_elevation requires both hwsd2_path and "
            "koppen_path; got at least one None."
        )
    if not hwsd2_path.exists():
        raise LookupSkipped(
            f"HWSD2 raster missing at {hwsd2_path}. Sprint S vendors "
            f"this substrate; verify build_eghr_substrate() ran for "
            f"this package."
        )
    if not koppen_path.exists():
        raise LookupSkipped(
            f"Köppen raster missing at {koppen_path}."
        )

    try:
        import rasterio  # type: ignore[import-untyped]
    except ImportError as exc:
        raise LookupSkipped(
            "rasterio not available in this environment; cannot "
            "sample HWSD2 / Köppen rasters."
        ) from exc

    centroid = (float(cell.lon), float(cell.lat))

    try:
        with rasterio.open(str(hwsd2_path)) as src:
            samples = list(src.sample([centroid]))
            if not samples or samples[0] is None or len(samples[0]) == 0:
                raise LookupSkipped(
                    f"HWSD2 sample at {centroid} returned no data; "
                    f"cell may be outside raster bounds."
                )
            elevation_m = float(samples[0][0])
    except (OSError, ValueError) as exc:
        raise LookupSkipped(
            f"HWSD2 raster read at {hwsd2_path} failed: {exc!r}"
        ) from exc

    # Codex HIGH #4 absorption: route through the existing
    # ``KGClassifier`` (canonical Köppen-zone classifier) rather
    # than a non-existent ``classify_zone_from_code`` helper. The
    # classifier opens the raster, samples (lat, lon), and maps
    # the integer code to a ``KGZone`` enum value via the bundled
    # ``KG_CODE_TO_ZONE`` table.
    from prismpy.koppen.kg_classifier import KGClassifier
    from prismpy.koppen.zones import KoppenZone as _KoppenZoneType  # noqa: F401

    # Sprint E.2's KoppenZone Literal is a 5-zone subset of the
    # full KG enum (Beck 2023 publishes 30 zones). When the
    # classifier returns a zone outside that subset (e.g., BSk,
    # BWh, Cwb) the lookup is "out-of-Sprint-E.2-scope" and we
    # raise LookupSkipped so the caller falls back honestly.
    try:
        with KGClassifier(koppen_path) as classifier:
            kg_zone = classifier.classify(float(cell.lat), float(cell.lon))
    except (OSError, ValueError) as exc:
        raise LookupSkipped(
            f"KGClassifier read at {koppen_path} failed: {exc!r}"
        ) from exc

    if kg_zone is None:
        raise LookupSkipped(
            f"Köppen sample at {centroid} returned no data (ocean / "
            f"nodata cell)."
        )
    # KGZone is a str-Enum whose value matches our KoppenZone
    # Literal members 1-1 for the 5 Sprint E.2 zones.
    code_str = kg_zone.value
    sprint_e2_zones = {"Af", "Aw", "BSh", "Cfa", "Cwa"}
    if code_str not in sprint_e2_zones:
        raise LookupSkipped(
            f"Köppen zone {code_str!r} at {centroid} is out of "
            f"Sprint E.2 scope (registry covers {sorted(sprint_e2_zones)}); "
            f"V2-19.5 Data Bootstrapper expansion required."
        )

    return ZoneElevationLookup(
        koppen_code=code_str,  # type: ignore[arg-type]
        elevation_m=elevation_m,
    )


__all__ = [
    "LookupSkipped",
    "ZoneElevationLookup",
    "lookup_zone_and_elevation",
]
