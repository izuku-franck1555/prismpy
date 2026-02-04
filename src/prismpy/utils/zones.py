"""Zone matching utilities - framework agnostic.

This module provides reusable zone matching functionality that any translator
(CRAFT, Pythia, ACEA, etc.) can use to apply spatial management variability.

Key functions:
- cell_matches_filter: Check if a cell matches a zone filter
- get_management_for_cell: Get effective management params with zone overrides

Example usage in a translator:
    from prismpy.utils.zones import get_management_for_cell

    for cell in grid.cells:
        params = get_management_for_cell(
            cell=cell,
            management=config.management,
            zones=config.management_zones,
        )
        total_n = params['fertilizer_n_total']
        # ... use zone-specific params
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from prismpy.config.schema import (
        ManagementConfig,
        ManagementZone,
        BoundingBoxFilter,
        AdminLevelFilter,
    )

logger = logging.getLogger(__name__)


def cell_matches_bounding_box(
    lat: float,
    lon: float,
    filter_config: "BoundingBoxFilter",
) -> bool:
    """Check if cell centroid is within a bounding box.

    Args:
        lat: Cell centroid latitude (decimal degrees)
        lon: Cell centroid longitude (decimal degrees)
        filter_config: BoundingBoxFilter with lat/lon bounds

    Returns:
        True if cell is within bounds (inclusive)
    """
    return (
        filter_config.lat_min <= lat <= filter_config.lat_max and
        filter_config.lon_min <= lon <= filter_config.lon_max
    )


def cell_matches_admin_level(
    cell_id: int,
    filter_config: "AdminLevelFilter",
    schema_df: Optional[pd.DataFrame] = None,
) -> bool:
    """Check if cell matches an admin level filter.

    Requires a schema DataFrame with Level{N}Name columns (e.g., Level2Name).

    Args:
        cell_id: CRAFT CellID or internal cell_id
        filter_config: AdminLevelFilter with admin_level and names
        schema_df: DataFrame with CellID and Level{N}Name columns

    Returns:
        True if cell's admin name is in the filter's names list
    """
    if schema_df is None:
        logger.warning("Admin filter requires schema with admin columns - skipping")
        return False

    # Build column name (e.g., "Level2Name" for admin_level=2)
    level_col = f"Level{filter_config.admin_level}Name"

    if level_col not in schema_df.columns:
        logger.warning(f"{level_col} not in schema columns, cannot filter by admin")
        return False

    # Find cell's admin name
    # Try CellID first (CRAFT style), then cell_id
    cell_row = None
    if 'CellID' in schema_df.columns:
        cell_row = schema_df[schema_df['CellID'] == cell_id]
    if (cell_row is None or cell_row.empty) and 'cell_id' in schema_df.columns:
        cell_row = schema_df[schema_df['cell_id'] == cell_id]

    if cell_row is None or cell_row.empty:
        return False

    cell_admin = cell_row[level_col].iloc[0]
    return cell_admin in filter_config.names


def cell_matches_filter(
    lat: float,
    lon: float,
    cell_id: int,
    zone_filter,
    schema_df: Optional[pd.DataFrame] = None,
) -> bool:
    """Check if a cell matches a zone filter.

    Args:
        lat: Cell centroid latitude
        lon: Cell centroid longitude
        cell_id: Cell identifier (for admin filtering)
        zone_filter: BoundingBoxFilter or AdminLevelFilter
        schema_df: Optional DataFrame for admin filtering

    Returns:
        True if cell matches the filter
    """
    from prismpy.config.schema import BoundingBoxFilter, AdminLevelFilter

    if isinstance(zone_filter, BoundingBoxFilter):
        return cell_matches_bounding_box(lat, lon, zone_filter)
    elif isinstance(zone_filter, AdminLevelFilter):
        return cell_matches_admin_level(cell_id, zone_filter, schema_df)
    else:
        logger.warning(f"Unknown filter type: {type(zone_filter)}")
        return False


def get_management_for_cell(
    lat: float,
    lon: float,
    cell_id: int,
    management: "ManagementConfig",
    zones: List["ManagementZone"],
    schema_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Get effective management parameters for a cell.

    Starts with ManagementConfig defaults, then applies first matching zone's
    overrides. Only non-None override values replace defaults.

    Args:
        lat: Cell centroid latitude
        lon: Cell centroid longitude
        cell_id: Cell identifier
        management: ManagementConfig with default values
        zones: List of ManagementZone with filters and overrides
        schema_df: Optional DataFrame for admin filtering

    Returns:
        Dict with all management params ready for use:
        - fertilizer_n_total: float
        - fertilizer_n_splits: List[int]
        - fertilizer_n_fractions: List[float]
        - planting_density: float
        - row_spacing_cm: float
        - cultivar: Optional[str] (DSSAT cultivar code)
        - planting_date_mmdd: Optional[str] (MMDD format, e.g., '0604')
        - matched_zone: Optional[str] (name of matched zone, or None)
    """
    # Start with defaults from ManagementConfig
    params = {
        'fertilizer_n_total': management.fertilizer_n_total,
        'fertilizer_n_splits': list(management.fertilizer_n_splits),
        'fertilizer_n_fractions': list(management.fertilizer_n_fractions),
        'planting_density': management.planting_density,
        'row_spacing_cm': management.row_spacing_cm,
        'cultivar': getattr(management, 'default_cultivar', None),
        'planting_date_mmdd': None,  # No default - use config's planting_doy
        'matched_zone': None,
    }

    # Check zones in order (first match wins)
    for zone in zones:
        if cell_matches_filter(lat, lon, cell_id, zone.filter, schema_df):
            # Apply non-None overrides
            overrides = zone.overrides
            if overrides.fertilizer_n_total is not None:
                params['fertilizer_n_total'] = overrides.fertilizer_n_total
            if overrides.fertilizer_n_splits is not None:
                params['fertilizer_n_splits'] = list(overrides.fertilizer_n_splits)
            if overrides.fertilizer_n_fractions is not None:
                params['fertilizer_n_fractions'] = list(overrides.fertilizer_n_fractions)
            if overrides.planting_density is not None:
                params['planting_density'] = overrides.planting_density
            if overrides.row_spacing_cm is not None:
                params['row_spacing_cm'] = overrides.row_spacing_cm
            if overrides.cultivar is not None:
                params['cultivar'] = overrides.cultivar
            if overrides.planting_date_mmdd is not None:
                params['planting_date_mmdd'] = overrides.planting_date_mmdd

            params['matched_zone'] = zone.zone_name
            logger.debug(
                f"Cell ({lat:.4f}, {lon:.4f}) matched zone '{zone.zone_name}'"
            )
            break  # First matching zone wins

    return params


def get_zone_summary(
    zones: List["ManagementZone"],
) -> str:
    """Get a summary of configured management zones.

    Useful for logging/debugging.

    Args:
        zones: List of ManagementZone

    Returns:
        Multi-line string summarizing zones
    """
    if not zones:
        return "No management zones configured (uniform mode)"

    lines = [f"Management zones ({len(zones)} configured):"]
    for i, zone in enumerate(zones, 1):
        filter_type = zone.filter.type
        if filter_type == "bounding_box":
            filter_desc = (
                f"lat [{zone.filter.lat_min:.2f}, {zone.filter.lat_max:.2f}], "
                f"lon [{zone.filter.lon_min:.2f}, {zone.filter.lon_max:.2f}]"
            )
        elif filter_type == "admin_level":
            filter_desc = f"Level{zone.filter.admin_level} in {zone.filter.names}"
        else:
            filter_desc = str(zone.filter)

        # Summarize overrides
        overrides = []
        if zone.overrides.fertilizer_n_total is not None:
            overrides.append(f"N={zone.overrides.fertilizer_n_total}")
        if zone.overrides.planting_density is not None:
            overrides.append(f"density={zone.overrides.planting_density}")
        if zone.overrides.cultivar is not None:
            overrides.append(f"cultivar={zone.overrides.cultivar}")
        if zone.overrides.planting_date_mmdd is not None:
            overrides.append(f"pdate={zone.overrides.planting_date_mmdd}")
        override_str = ", ".join(overrides) if overrides else "(no overrides)"

        lines.append(f"  {i}. {zone.zone_name}: {filter_type} ({filter_desc}) -> {override_str}")

    return "\n".join(lines)
