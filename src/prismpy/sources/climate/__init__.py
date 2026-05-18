"""Climate data sources (NASA POWER, TAMSAT, AgERA5).

Also exports the canonical sentinel discipline helpers used by every
translator that filters the executor's placeholder-climate entry out of
real-coverage counts (Sprint F-CP fixup, AC-F-CP-14).
"""

from prismpy.sources.climate._sentinels import (
    PLACEHOLDER_CLIMATE_SENTINEL_ID,
    is_real_climate_cell_id,
)
from prismpy.sources.climate.agera5 import AgERA5Config, AgERA5Data, AgERA5Source
from prismpy.sources.climate.errors import ClimateDownloadError
from prismpy.sources.climate.nasa_power import NASAPowerConfig, NASAPowerSource
from prismpy.sources.climate.tamsat import TAMSATConfig, TAMSATData, TAMSATSource

__all__ = [
    "NASAPowerSource",
    "NASAPowerConfig",
    "TAMSATSource",
    "TAMSATConfig",
    "TAMSATData",
    "AgERA5Source",
    "AgERA5Config",
    "AgERA5Data",
    # Canonical sentinel discipline (Sprint F-CP fixup AC-F-CP-14).
    "PLACEHOLDER_CLIMATE_SENTINEL_ID",
    "is_real_climate_cell_id",
    "ClimateDownloadError",
]
