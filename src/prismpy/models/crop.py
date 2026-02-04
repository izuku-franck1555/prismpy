"""
Crop parameter and calendar models for prismpy.

These models represent crop phenology, growth parameters, and planting calendars.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass
class CropCalendar:
    """Crop planting and harvest timing for a location.

    Attributes:
        location_id: Location identifier (cell ID or site ID)
        planting_doy: Planting day of year (1-366)
        maturity_doy: Maturity day of year (1-366)
        harvest_doy: Harvest day of year (optional, defaults to maturity)
        source: Source of calendar data
    """
    location_id: int
    planting_doy: int
    maturity_doy: int
    harvest_doy: Optional[int] = None
    source: str = "default"

    def __post_init__(self):
        """Set defaults."""
        if self.harvest_doy is None:
            self.harvest_doy = self.maturity_doy

    @property
    def growing_season_days(self) -> int:
        """Growing season length in days."""
        if self.maturity_doy >= self.planting_doy:
            return self.maturity_doy - self.planting_doy
        else:
            # Season spans year boundary
            return (365 - self.planting_doy) + self.maturity_doy

    def get_planting_date(self, year: int) -> date:
        """Get planting date for a specific year."""
        return date(year, 1, 1) + __import__('datetime').timedelta(days=self.planting_doy - 1)

    def get_maturity_date(self, year: int) -> date:
        """Get maturity date for a specific year."""
        planting = self.get_planting_date(year)
        return planting + __import__('datetime').timedelta(days=self.growing_season_days)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "location_id": self.location_id,
            "planting_doy": self.planting_doy,
            "maturity_doy": self.maturity_doy,
            "harvest_doy": self.harvest_doy,
            "growing_season_days": self.growing_season_days,
            "source": self.source,
        }


@dataclass
class CropParameters:
    """Crop parameters in unified format.

    This model supports parameters from different platforms by storing
    them in a flexible dictionary while providing accessors for common
    parameters.

    Attributes:
        crop_name: Crop name (e.g., "Maize", "Millet", "Sorghum")
        variety_name: Variety or cultivar name
        source: Source template or calibration reference
        parameters: Dictionary of all crop parameters
    """
    crop_name: str
    variety_name: Optional[str] = None
    source: str = "default"
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Common parameter keys across platforms
    PARAM_KEYS = {
        # Phenology (thermal time / GDD)
        "gdd_emergence": ["SDJLevee", "gdd_emergence", "P1"],
        "gdd_flowering": ["SDJBVP", "gdd_yield_form", "P2"],
        "gdd_maturity": ["SDJMatu1", "gdd_maturity", "P5"],
        # Temperature
        "base_temp": ["TBase", "base_temp", "tb"],
        "opt_temp": ["TOpt1", "temp_opt", "to"],
        # Yield
        "harvest_index": ["txConversion", "HI", "hi"],
        "potential_yield": ["KRdtPotB", "pot_yield"],
    }

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value, checking alternate keys."""
        if key in self.parameters:
            return self.parameters[key]

        # Check alternate keys
        if key in self.PARAM_KEYS:
            for alt_key in self.PARAM_KEYS[key]:
                if alt_key in self.parameters:
                    return self.parameters[alt_key]

        return default

    def set_param(self, key: str, value: Any) -> None:
        """Set a parameter value."""
        self.parameters[key] = value

    @property
    def gdd_emergence(self) -> Optional[float]:
        """GDD to emergence."""
        return self.get_param("gdd_emergence")

    @property
    def gdd_maturity(self) -> Optional[float]:
        """GDD to maturity."""
        return self.get_param("gdd_maturity")

    @property
    def base_temp(self) -> Optional[float]:
        """Base temperature for GDD accumulation."""
        return self.get_param("base_temp")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "crop_name": self.crop_name,
            "variety_name": self.variety_name,
            "source": self.source,
            "parameters": self.parameters,
        }

    def to_sarra_py_format(self) -> Dict[str, Any]:
        """Convert to SARRA-Py variety.yaml format.

        Returns parameters in the format expected by SARRA-Py.
        """
        return {
            # Phenology (thermal time)
            "SDJLevee": self.get_param("gdd_emergence", 90),
            "SDJBVP": self.get_param("gdd_flowering", 500),
            "SDJRPR": self.get_param("gdd_reproductive", 800),
            "SDJMatu1": self.get_param("gdd_maturity", 1200),
            # Temperature thresholds
            "TBase": self.get_param("base_temp", 8),
            "TOpt1": self.get_param("temp_opt1", 30),
            "TOpt2": self.get_param("temp_opt2", 35),
            "TLim": self.get_param("temp_limit", 45),
            # Yield parameters
            "KRdtPotA": self.get_param("krdtpota", 0),
            "KRdtPotB": self.get_param("potential_yield", 200),
            "txConversion": self.get_param("harvest_index", 0.4),
            "txResGrain": self.get_param("grain_fraction", 0.8),
            # Other parameters (include all extras)
            **{k: v for k, v in self.parameters.items() if k not in [
                "gdd_emergence", "gdd_flowering", "gdd_reproductive",
                "gdd_maturity", "base_temp", "temp_opt1", "temp_opt2",
                "temp_limit", "krdtpota", "potential_yield",
                "harvest_index", "grain_fraction"
            ]}
        }

    def to_acea_format(self) -> Dict[str, float]:
        """Convert to ACEA GDD-based crop parameters format.

        Returns the 11 parameters required by ACEA crop parameter NetCDF.
        """
        return {
            "gdd_emergence": float(self.get_param("gdd_emergence", 60)),
            "gdd_max_root": float(self.get_param("gdd_max_root", 800)),
            "gdd_senescence": float(self.get_param("gdd_senescence", 1400)),
            "gdd_maturity": float(self.get_param("gdd_maturity", 1700)),
            "gdd_yield_form": float(self.get_param("gdd_yield_form", 750)),
            "gdd_duration_flowering": float(self.get_param("gdd_duration_flowering", 200)),
            "gdd_duration_yield_form": float(self.get_param("gdd_duration_yield_form", 950)),
            "cgc": float(self.get_param("cgc", 0.013)),
            "cdc": float(self.get_param("cdc", 0.004)),
            "temp_max_avg": float(self.get_param("temp_max_avg", 32)),
            "temp_min_avg": float(self.get_param("temp_min_avg", 22)),
        }

    @classmethod
    def from_sarra_py_yaml(cls, yaml_data: Dict[str, Any], crop_name: str) -> "CropParameters":
        """Create from SARRA-Py variety.yaml format."""
        params = CropParameters(
            crop_name=crop_name,
            variety_name=yaml_data.get("variety_name"),
            source="sarra_py",
            parameters=yaml_data.copy()
        )
        # Map to canonical keys
        if "SDJLevee" in yaml_data:
            params.parameters["gdd_emergence"] = yaml_data["SDJLevee"]
        if "SDJMatu1" in yaml_data:
            params.parameters["gdd_maturity"] = yaml_data["SDJMatu1"]
        if "TBase" in yaml_data:
            params.parameters["base_temp"] = yaml_data["TBase"]
        return params


@dataclass
class ManagementParameters:
    """Crop management parameters (fertilizer, irrigation, etc.).

    Attributes:
        location_id: Location identifier
        total_n_kg_ha: Total nitrogen application (kg/ha)
        n_applications: Number of N applications
        irrigation_type: Irrigation type ("R" = rainfed, "I" = irrigated)
        plant_density: Plant density (plants/ha)
        row_spacing: Row spacing (cm)
        planting_depth: Planting depth (cm)
        cultivar_code: DSSAT cultivar code (e.g., "GH0010")
    """
    location_id: int
    total_n_kg_ha: float = 0
    n_applications: int = 1
    irrigation_type: str = "R"
    plant_density: float = 55000
    row_spacing: float = 75
    planting_depth: float = 5
    cultivar_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_fertilizer_schedule(self) -> List[Dict[str, Any]]:
        """Generate fertilizer application schedule.

        Returns list of applications with days after planting and amounts.
        """
        if self.n_applications <= 0 or self.total_n_kg_ha <= 0:
            return []

        schedules = {
            1: [(0, 1.0)],  # 100% at planting
            2: [(0, 0.4), (30, 0.6)],  # 40% at planting, 60% at 30 DAP
            3: [(0, 0.33), (30, 0.33), (60, 0.34)],  # Split evenly
        }

        schedule = schedules.get(self.n_applications, schedules[1])

        return [
            {
                "dap": dap,
                "n_kg_ha": self.total_n_kg_ha * fraction,
                "fraction": fraction
            }
            for dap, fraction in schedule
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "location_id": self.location_id,
            "total_n_kg_ha": self.total_n_kg_ha,
            "n_applications": self.n_applications,
            "irrigation_type": self.irrigation_type,
            "plant_density": self.plant_density,
            "row_spacing": self.row_spacing,
            "planting_depth": self.planting_depth,
            "cultivar_code": self.cultivar_code,
            "fertilizer_schedule": self.get_fertilizer_schedule(),
            "metadata": self.metadata,
        }
