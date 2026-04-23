"""
Region and bounding box models for prismpy.

These models represent the canonical geographic extent of a study region.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BoundingBox:
    """Bounding box in standard GIS format [minx, miny, maxx, maxy].

    All coordinates are in WGS84 (EPSG:4326) decimal degrees.

    Attributes:
        minx: Minimum longitude (western edge)
        miny: Minimum latitude (southern edge)
        maxx: Maximum longitude (eastern edge)
        maxy: Maximum latitude (northern edge)
    """
    minx: float
    miny: float
    maxx: float
    maxy: float
    crs: str = "EPSG:4326"

    def __post_init__(self):
        """Validate bounding box coordinates."""
        if self.minx >= self.maxx:
            raise ValueError(f"minx ({self.minx}) must be less than maxx ({self.maxx})")
        if self.miny >= self.maxy:
            raise ValueError(f"miny ({self.miny}) must be less than maxy ({self.maxy})")

    @property
    def width(self) -> float:
        """Width of bounding box in degrees."""
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        """Height of bounding box in degrees."""
        return self.maxy - self.miny

    @property
    def center(self) -> Tuple[float, float]:
        """Center point (lon, lat) of bounding box."""
        return (
            (self.minx + self.maxx) / 2,
            (self.miny + self.maxy) / 2
        )

    def to_gis_format(self) -> List[float]:
        """Return bounds in standard GIS format [minx, miny, maxx, maxy]."""
        return [self.minx, self.miny, self.maxx, self.maxy]

    def to_sarra_py_format(self) -> List[float]:
        """Return bounds in SARRA-Py format [lat_NW, lon_NW, lat_SE, lon_SE].

        SARRA-Py uses a non-standard bounding box format where coordinates
        are ordered as [lat_NW, lon_NW, lat_SE, lon_SE] instead of the
        standard GIS [minx, miny, maxx, maxy].
        """
        return [self.maxy, self.minx, self.miny, self.maxx]

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return bounds as a tuple (minx, miny, maxx, maxy)."""
        return (self.minx, self.miny, self.maxx, self.maxy)

    def contains_point(self, lon: float, lat: float) -> bool:
        """Check if a point is within the bounding box."""
        return (self.minx <= lon <= self.maxx and
                self.miny <= lat <= self.maxy)

    def buffer(self, degrees: float) -> "BoundingBox":
        """Return a new bounding box expanded by the given amount."""
        return BoundingBox(
            minx=self.minx - degrees,
            miny=self.miny - degrees,
            maxx=self.maxx + degrees,
            maxy=self.maxy + degrees,
            crs=self.crs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "minx": self.minx,
            "miny": self.miny,
            "maxx": self.maxx,
            "maxy": self.maxy,
            "crs": self.crs,
        }

    @classmethod
    def from_gis_format(cls, bounds: List[float], crs: str = "EPSG:4326") -> "BoundingBox":
        """Create from GIS format [minx, miny, maxx, maxy]."""
        return cls(
            minx=bounds[0],
            miny=bounds[1],
            maxx=bounds[2],
            maxy=bounds[3],
            crs=crs
        )

    @classmethod
    def from_sarra_py_format(cls, bounds: List[float], crs: str = "EPSG:4326") -> "BoundingBox":
        """Create from SARRA-Py format [lat_NW, lon_NW, lat_SE, lon_SE]."""
        lat_nw, lon_nw, lat_se, lon_se = bounds
        return cls(
            minx=lon_nw,  # Western edge
            miny=lat_se,  # Southern edge
            maxx=lon_se,  # Eastern edge
            maxy=lat_nw,  # Northern edge
            crs=crs
        )


@dataclass
class Region:
    """Canonical representation of a study region.

    This is the unified region model used throughout the translation pipeline.

    Attributes:
        name: Region name (e.g., "Koutiala")
        country: Country name (e.g., "Mali")
        country_iso3: ISO 3166-1 alpha-3 country code (e.g., "MLI")
        bounds: Bounding box defining the spatial extent
        gadm_level: GADM administrative level (0=country, 1=region, 2=district)
        geometry_wkt: Optional WKT string of the full polygon geometry
        metadata: Optional additional metadata
    """
    name: str
    country: str
    country_iso3: str
    bounds: BoundingBox
    gadm_level: int = 2
    geometry_wkt: Optional[str] = None
    crs: str = "EPSG:4326"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # V2-22b/P.1 — populated from `RegionConfig.boundary.source.value`
    # at resolve-region time so cache/path helpers can route on it.
    # Optional (None) for legacy construction paths that predate the
    # field; helpers treat None as "use name-based key".
    boundary_source: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize the region."""
        self.country_iso3 = self.country_iso3.upper()
        if len(self.country_iso3) != 3:
            raise ValueError(f"country_iso3 must be 3 characters: {self.country_iso3}")

    @property
    def full_name(self) -> str:
        """Full region name including country."""
        return f"{self.name}, {self.country}"

    @property
    def area_deg2(self) -> float:
        """Approximate area in square degrees."""
        return self.bounds.width * self.bounds.height

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "country": self.country,
            "country_iso3": self.country_iso3,
            "bounds": self.bounds.to_dict(),
            "bounds_gis": self.bounds.to_gis_format(),
            "bounds_sarra_py": self.bounds.to_sarra_py_format(),
            "gadm_level": self.gadm_level,
            "crs": self.crs,
            "metadata": self.metadata,
            # V2-22b/P.1 — persist so `region_cache_key()` still
            # routes manual regions to bbox identity after a
            # to_dict/from_dict round-trip. Without this, reloaded
            # Region objects had `boundary_source=None` and fell
            # back to name-keyed cache paths.
            "boundary_source": self.boundary_source,
        }

    def to_json_file(self, path: str) -> None:
        """Save region to JSON file."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Region":
        """Create from dictionary."""
        bounds_data = data.get("bounds", {})
        bounds = BoundingBox(
            minx=bounds_data.get("minx", data.get("bounds_gis", [0, 0, 0, 0])[0]),
            miny=bounds_data.get("miny", data.get("bounds_gis", [0, 0, 0, 0])[1]),
            maxx=bounds_data.get("maxx", data.get("bounds_gis", [0, 0, 0, 0])[2]),
            maxy=bounds_data.get("maxy", data.get("bounds_gis", [0, 0, 0, 0])[3]),
            crs=bounds_data.get("crs", data.get("crs", "EPSG:4326"))
        )
        return cls(
            name=data["name"],
            country=data["country"],
            country_iso3=data["country_iso3"],
            bounds=bounds,
            gadm_level=data.get("gadm_level", 2),
            crs=data.get("crs", "EPSG:4326"),
            metadata=data.get("metadata", {}),
            boundary_source=data.get("boundary_source"),
        )
