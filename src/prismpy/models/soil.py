"""
Soil data models for prismpy.

These models represent soil profiles and layers with physical properties.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SoilLayer:
    """A single layer in a soil profile.

    Attributes:
        depth_top: Depth to top of layer (m)
        depth_bottom: Depth to bottom of layer (m)
        sand: Sand content (%)
        clay: Clay content (%)
        silt: Silt content (%), computed if not provided
        organic_carbon: Organic carbon content (%)
        bulk_density: Bulk density (g/cm³)
        ph: Soil pH
        field_capacity: Volumetric water content at field capacity (cm³/cm³)
        wilting_point: Volumetric water content at wilting point (cm³/cm³)
        saturated_wc: Saturated water content (cm³/cm³)
        hydraulics_estimated: The water limits came from the pedotransfer function, not the source
    """
    depth_top: float
    depth_bottom: float
    sand: float
    clay: float
    silt: Optional[float] = None
    organic_carbon: Optional[float] = None
    bulk_density: Optional[float] = None
    ph: Optional[float] = None
    field_capacity: Optional[float] = None
    wilting_point: Optional[float] = None
    saturated_wc: Optional[float] = None
    # Provenance only: outside equality, to_dict and the eGHR dedup key.
    hydraulics_estimated: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self):
        """Compute derived values."""
        # Compute silt if not provided (sand + clay + silt = 100)
        if self.silt is None:
            self.silt = 100 - self.sand - self.clay

    @property
    def thickness(self) -> float:
        """Layer thickness in meters."""
        return self.depth_bottom - self.depth_top

    @property
    def available_water_capacity(self) -> Optional[float]:
        """Available water capacity (field capacity - wilting point)."""
        if self.field_capacity is not None and self.wilting_point is not None:
            return self.field_capacity - self.wilting_point
        return None

    def estimate_hydraulic_properties(self) -> None:
        """Estimate hydraulic properties from texture using pedotransfer functions.

        Uses a pedotransfer function adapted from Saxton & Rawls (2006).
        """
        # Pedotransfer functions adapted from Saxton & Rawls (2006)
        S = self.sand / 100  # Sand fraction
        C = self.clay / 100  # Clay fraction
        OM = (self.organic_carbon or 1.0) / 100  # Organic matter fraction

        # Wilting point (1500 kPa)
        wp = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
        self.wilting_point = max(0.01, min(0.4, wp))

        # Field capacity (33 kPa)
        fc = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
        self.field_capacity = max(self.wilting_point + 0.05, min(0.5, fc))

        # Saturated water content
        sat = 0.332 - 0.0007251 * self.sand + 0.1276 * C
        self.saturated_wc = max(self.field_capacity + 0.05, min(0.6, sat))
        self.hydraulics_estimated = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "depth_top": self.depth_top,
            "depth_bottom": self.depth_bottom,
            "thickness": self.thickness,
            "sand": self.sand,
            "clay": self.clay,
            "silt": self.silt,
            "organic_carbon": self.organic_carbon,
            "bulk_density": self.bulk_density,
            "ph": self.ph,
            "field_capacity": self.field_capacity,
            "wilting_point": self.wilting_point,
            "saturated_wc": self.saturated_wc,
            "available_water_capacity": self.available_water_capacity,
        }

    def validate(self) -> List[str]:
        """Validate the soil layer."""
        errors = []

        if self.depth_top < 0:
            errors.append(f"depth_top must be >= 0: {self.depth_top}")
        if self.depth_bottom <= self.depth_top:
            errors.append(f"depth_bottom must be > depth_top: {self.depth_bottom} <= {self.depth_top}")
        if not (0 <= self.sand <= 100):
            errors.append(f"sand must be 0-100: {self.sand}")
        if not (0 <= self.clay <= 100):
            errors.append(f"clay must be 0-100: {self.clay}")
        if self.sand + self.clay > 100:
            errors.append(f"sand + clay must be <= 100: {self.sand + self.clay}")

        return errors


# The USDA (NRCS) soil texture classes by their published definitions: (class, rule on sand, silt, clay %).
# NRCS's sand bounds (sand > 85 %; loamy sand 70-91 %) follow from these lines inside the texture triangle.
_USDA_TEXTURE_RULES = (
    ("Sand", lambda sand, silt, clay: silt + 1.5 * clay < 15),
    ("Loamy Sand", lambda sand, silt, clay: silt + 1.5 * clay >= 15 and silt + 2 * clay < 30),
    ("Sandy Loam", lambda sand, silt, clay: silt + 2 * clay >= 30
        and ((7 <= clay < 20 and sand > 52) or (clay < 7 and silt < 50))),
    ("Loam", lambda sand, silt, clay: 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52),
    ("Silt Loam", lambda sand, silt, clay: (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12)),
    ("Silt", lambda sand, silt, clay: silt >= 80 and clay < 12),
    ("Sandy Clay Loam", lambda sand, silt, clay: 20 <= clay < 35 and silt < 28 and sand > 45),
    ("Clay Loam", lambda sand, silt, clay: 27 <= clay < 40 and 20 < sand <= 45),
    ("Silty Clay Loam", lambda sand, silt, clay: 27 <= clay < 40 and sand <= 20),
    ("Sandy Clay", lambda sand, silt, clay: clay >= 35 and sand > 45),
    ("Silty Clay", lambda sand, silt, clay: clay >= 40 and silt >= 40),
    ("Clay", lambda sand, silt, clay: clay >= 40 and sand <= 45 and silt < 40),
)


@dataclass
class SoilProfile:
    """Unified soil profile representation.

    This is the canonical soil profile model that all platform translators consume.

    Attributes:
        profile_id: Unique profile identifier
        lat: Latitude of the profile location
        lon: Longitude of the profile location
        source: Data source identifier (e.g., "iSDA", "HWSD", "eGHR")
        layers: List of soil layers from top to bottom
        total_depth: Total profile depth (m)
        metadata: Additional metadata
    """
    profile_id: str
    lat: float
    lon: float
    source: str
    layers: List[SoilLayer] = field(default_factory=list)
    total_depth: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived values."""
        if self.total_depth is None and self.layers:
            self.total_depth = max(layer.depth_bottom for layer in self.layers)

    @property
    def n_layers(self) -> int:
        """Number of layers."""
        return len(self.layers)

    @property
    def surface_texture(self) -> Optional[str]:
        """Get texture class of surface layer."""
        if not self.layers:
            return None
        return self._get_texture_class(self.layers[0].sand, self.layers[0].clay)

    @staticmethod
    def _get_texture_class(sand: float, clay: float) -> str:
        """The USDA (NRCS) texture class of sand and clay percentages, silt being the rest."""
        silt = 100 - sand - clay
        for name, rule in _USDA_TEXTURE_RULES:
            if rule(sand, silt, clay):
                return name
        raise ValueError(f"No USDA texture class for sand {sand} % and clay {clay} %")

    def get_layer_at_depth(self, depth: float) -> Optional[SoilLayer]:
        """Get the layer containing the specified depth."""
        for layer in self.layers:
            if layer.depth_top <= depth < layer.depth_bottom:
                return layer
        return None

    def get_weighted_average(self, property_name: str, max_depth: Optional[float] = None) -> Optional[float]:
        """Calculate depth-weighted average of a property.

        Args:
            property_name: Name of the property (e.g., "sand", "clay")
            max_depth: Maximum depth to consider (m)

        Returns:
            Weighted average value or None if property not available
        """
        if not self.layers:
            return None

        total_weight = 0
        weighted_sum = 0

        for layer in self.layers:
            if max_depth and layer.depth_top >= max_depth:
                break

            value = getattr(layer, property_name, None)
            if value is None:
                continue

            # Adjust layer thickness if it extends beyond max_depth
            top = layer.depth_top
            bottom = layer.depth_bottom if max_depth is None else min(layer.depth_bottom, max_depth)
            thickness = bottom - top

            if thickness > 0:
                weighted_sum += value * thickness
                total_weight += thickness

        return weighted_sum / total_weight if total_weight > 0 else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "profile_id": self.profile_id,
            "lat": self.lat,
            "lon": self.lon,
            "source": self.source,
            "total_depth": self.total_depth,
            "n_layers": self.n_layers,
            "surface_texture": self.surface_texture,
            "layers": [layer.to_dict() for layer in self.layers],
            "metadata": self.metadata,
        }

    def validate(self) -> List[str]:
        """Validate the soil profile."""
        errors = []

        if not self.layers:
            errors.append("Profile has no layers")
            return errors

        # Validate each layer
        for i, layer in enumerate(self.layers):
            layer_errors = layer.validate()
            for error in layer_errors:
                errors.append(f"Layer {i}: {error}")

        # Check layer continuity
        for i in range(1, len(self.layers)):
            if self.layers[i].depth_top != self.layers[i - 1].depth_bottom:
                errors.append(
                    f"Gap between layer {i - 1} bottom ({self.layers[i - 1].depth_bottom}) "
                    f"and layer {i} top ({self.layers[i].depth_top})"
                )

        return errors

    @classmethod
    def from_single_layer(
        cls,
        profile_id: str,
        lat: float,
        lon: float,
        source: str,
        sand: float,
        clay: float,
        depth: float = 1.5,
        **kwargs
    ) -> "SoilProfile":
        """Create a simple single-layer profile."""
        layer = SoilLayer(
            depth_top=0,
            depth_bottom=depth,
            sand=sand,
            clay=clay,
            **kwargs
        )
        return cls(
            profile_id=profile_id,
            lat=lat,
            lon=lon,
            source=source,
            layers=[layer],
            total_depth=depth
        )


def generic_soil_direction(kind: str) -> str:
    """The one wording of what a generic soil implies for its yields; callers add "; " and the final "."."""
    if kind not in ("default", "placeholder"):
        raise ValueError(f"no generic soil profile of kind {kind!r}")
    return (f"the {kind} soil may hold more or less water, and more or less organic matter, than the "
            "local soil, so the yields simulated on it may be over- or understated")
