"""
Default values and constants for prismpy.

These defaults are used when configuration values are not specified.
They are based on typical values for Sub-Saharan Africa crop modeling.
"""

from typing import Dict, Any

# =============================================================================
# Geographic Defaults
# =============================================================================

DEFAULT_CRS = "EPSG:4326"  # WGS84

# Grid resolutions in degrees
RESOLUTION_5ARCMIN = 5 / 60  # 0.0833...
RESOLUTION_30ARCMIN = 30 / 60  # 0.5
RESOLUTION_SARRA_PY = 0.0375  # ~4km (native TAMSAT resolution)

# Global grid dimensions
GLOBAL_GRID_5ARCMIN = {
    "rows": 2160,
    "cols": 4320,
    "total_cells": 9331200,
}

GLOBAL_GRID_30ARCMIN = {
    "rows": 360,
    "cols": 720,
    "total_cells": 259200,
}

# =============================================================================
# Temporal Defaults
# =============================================================================

# Earliest date with reliable data across all sources
DEFAULT_CLIMATE_START_YEAR = 1984  # NASA POWER SRAD available from 1984

# Typical spinup period for crop models
DEFAULT_SPINUP_YEARS = 2

# =============================================================================
# Climate Defaults
# =============================================================================

# V2-19 CD-05: CLIMATE_VALIDATION_RANGES was deleted as dead code.
# It was never imported anywhere. Centralized climate validation is now
# distributed as inline checks in sources/climate/nasa_power.py. If
# centralization is ever needed, add sources/climate/_quality.py as the
# canonical source-of-truth (tracked as V2-20 carryover).

# Gap-filling parameters
GAP_FILL_CONFIG = {
    "max_gap_days": 5,      # Maximum gap to fill with interpolation
    "method": "linear",     # Interpolation method
    "use_climatology_for_long_gaps": True,
}

# =============================================================================
# Soil Defaults
# =============================================================================

# Validation ranges for soil properties
SOIL_VALIDATION_RANGES = {
    "sand": (0, 100),
    "clay": (0, 100),
    "silt": (0, 100),
    "organic_carbon": (0, 30),
    "bulk_density": (0.5, 2.0),
    "ph": (3, 10),
}

# Default soil properties for gap-filling (typical Sahel values)
DEFAULT_SOIL_SAHEL = {
    "sand": 60,
    "clay": 18,
    "silt": 22,
    "organic_carbon": 0.5,
    "bulk_density": 1.4,
    "ph": 6.5,
}

# ACEA layer depths (meters)
ACEA_LAYER_DEPTHS = [0.1, 0.1, 0.1, 0.3, 0.4, 0.6, 0.7, 0.7]  # Total: 3.0m

# =============================================================================
# Crop Defaults
# =============================================================================

# Typical West African maize parameters
DEFAULT_MAIZE_PARAMS = {
    # Phenology (GDD)
    "gdd_emergence": 60,
    "gdd_flowering": 750,
    "gdd_maturity": 1700,
    "base_temp": 8,
    # Canopy
    "cgc": 0.013,
    "cdc": 0.004,
    # Yield
    "harvest_index": 0.4,
}

# Typical millet parameters
DEFAULT_MILLET_PARAMS = {
    "gdd_emergence": 90,
    "gdd_flowering": 500,
    "gdd_maturity": 1200,
    "base_temp": 10,
    "harvest_index": 0.25,
}

# Typical sorghum parameters
DEFAULT_SORGHUM_PARAMS = {
    "gdd_emergence": 80,
    "gdd_flowering": 600,
    "gdd_maturity": 1400,
    "base_temp": 8,
    "harvest_index": 0.35,
}

# Planting windows (DOY) for West Africa
PLANTING_WINDOWS_WEST_AFRICA = {
    "sahel": (152, 182),       # June 1 - July 1
    "sudan": (135, 166),       # May 15 - June 15
    "guinea": (121, 152),      # May 1 - June 1
    "forest": (74, 105),       # March 15 - April 15
}

# =============================================================================
# API Configuration Defaults
# =============================================================================

NASA_POWER_CONFIG = {
    "base_url": "https://power.larc.nasa.gov/api/temporal/daily/point",
    "timeout": 120,
    "retry_count": 3,
    "request_delay": 1.0,
    "parameters": [
        "T2M", "T2M_MIN", "T2M_MAX", "T2MDEW",
        "PRECTOTCORR", "WS2M", "ALLSKY_SFC_SW_DWN", "RH2M"
    ],
}

# V2-19 CD-13: CDS_API_CONFIG was deleted as dead code.
# It was never imported anywhere, and its "dataset" value
# ("reanalysis-era5-land") drifted from the actual dataset used by
# sources/climate/agera5.py ("sis-agrometeorological-indicators").
# The live CDS API config is now the single source of truth in agera5.py.

# =============================================================================
# Output Format Defaults
# =============================================================================

# DSSAT date format (YRDOY)
DSSAT_DATE_FORMAT = "%Y%j"  # e.g., 2015001

# CRAFT column separator
CRAFT_SEPARATOR = "\t"

# SARRA-Py YAML date format
SARRA_PY_DATE_FORMAT = "%Y-%m-%d"

# =============================================================================
# Platform-Specific Defaults
# =============================================================================

PLATFORM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "sarra_py": {
        "resolution": RESOLUTION_SARRA_PY,
        "climate_sources": {
            "rainfall": "tamsat",
            "temperature": "agera5",
            "radiation": "agera5",
        },
        "soil_source": "isda",
        "itk_template": "rainfed_opportunistic",
    },
    "craft": {
        "resolution_arcmin": 5,
        "climate_source": "nasa_power",
        "soil_source": "hwsd",
    },
    "pythia": {
        "resolution": RESOLUTION_5ARCMIN,
        "climate_source": "nasa_power",
        "soil_source": "eghr",
        "dssat_version": "4.8",
    },
    "acea": {
        "resolution": "5arcmin",
        "climate_source": "nasa_power",
        "soil_source": "hwsd",
        "compute_et0": True,
    },
}
