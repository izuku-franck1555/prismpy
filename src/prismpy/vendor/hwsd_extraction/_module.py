#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HWSD v2.0 BIL+MDB Extraction Module

This module extracts soil properties (sand, clay, silt) from HWSD v2.0 data
which consists of:
  - HWSD2.bil: Raster containing Soil Mapping Unit (SMU) IDs
  - HWSD2.mdb: MS Access database containing soil properties keyed by SMU ID

The BIL raster does NOT contain soil values directly - it contains SMU IDs
that must be looked up in the MDB database.

Requirements:
  - mdbtools (system package): brew install mdbtools (macOS) or apt-get install mdbtools (Linux)
  - rasterio (Python package)
  - pandas (Python package)

Usage:
    from hwsd_extraction import extract_hwsd_soil_data

    soil_df = extract_hwsd_soil_data(
        bil_path="path/to/HWSD2.bil",
        mdb_path="path/to/HWSD2.mdb",
        cells_df=grid_cells_dataframe,  # Must have 'lat', 'lon' columns
        layer="D1",  # D1=topsoil (0-20cm), D2=subsoil (20-40cm)
    )

Reference:
    FAO & IIASA (2023). Harmonized World Soil Database version 2.0.
    Rome and Laxenburg. https://doi.org/10.4060/cc3823en

Author: ACEA Data-to-Model Translation Framework
Date: January 2026
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def export_mdb_table(mdb_path: str, table_name: str, output_csv: str = None) -> pd.DataFrame:
    """
    Export a table from MS Access MDB file using mdbtools.

    Args:
        mdb_path: Path to .mdb file
        table_name: Name of table to export (e.g., "HWSD2_LAYERS")
        output_csv: Optional path to save CSV (if None, uses temp file)

    Returns:
        DataFrame with table contents

    Raises:
        FileNotFoundError: If mdb file doesn't exist
        RuntimeError: If mdbtools is not installed or export fails
    """
    if not os.path.exists(mdb_path):
        raise FileNotFoundError(f"MDB file not found: {mdb_path}")

    # Check if mdbtools is available
    try:
        subprocess.run(["mdb-tables", "--help"], capture_output=True, check=False)
    except FileNotFoundError:
        raise RuntimeError(
            "mdbtools not installed. Install with:\n"
            "  macOS: brew install mdbtools\n"
            "  Linux: sudo apt-get install mdbtools"
        )

    # Export table to CSV
    if output_csv is None:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        output_csv = temp_file.name
        temp_file.close()

    try:
        result = subprocess.run(
            ["mdb-export", mdb_path, table_name],
            capture_output=True,
            text=True,
            check=True
        )

        # Write output to file
        with open(output_csv, 'w') as f:
            f.write(result.stdout)

        # Read CSV with pandas
        df = pd.read_csv(output_csv, low_memory=False)

        return df

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"mdb-export failed: {e.stderr}")
    finally:
        # Clean up temp file
        if output_csv and os.path.exists(output_csv):
            try:
                os.unlink(output_csv)
            except:
                pass


def extract_smu_ids_from_raster(bil_path: str, cells_df: pd.DataFrame) -> np.ndarray:
    """
    Extract Soil Mapping Unit (SMU) IDs from HWSD BIL raster for given coordinates.

    Args:
        bil_path: Path to HWSD2.bil raster file
        cells_df: DataFrame with 'lat' and 'lon' columns

    Returns:
        Array of SMU IDs for each cell
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio not installed. Install with: pip install rasterio")

    if not os.path.exists(bil_path):
        raise FileNotFoundError(f"BIL raster not found: {bil_path}")

    smu_ids = []

    with rasterio.open(bil_path) as src:
        for _, row in cells_df.iterrows():
            try:
                # Get pixel coordinates from geographic coordinates
                py, px = src.index(row['lon'], row['lat'])
                # Read SMU ID value at this location
                smu_id = src.read(1)[py, px]
                smu_ids.append(int(smu_id))
            except (IndexError, ValueError):
                # Cell outside raster bounds
                smu_ids.append(0)

    return np.array(smu_ids)


def create_smu_lookup(hwsd_layers_df: pd.DataFrame, layer: str = "D1") -> dict:
    """
    Create a lookup dictionary mapping SMU IDs to soil properties.

    Args:
        hwsd_layers_df: DataFrame from HWSD2_LAYERS table
        layer: Soil layer to use:
            - "D1": Topsoil (0-20cm) - recommended for crop modeling
            - "D2": Subsoil (20-40cm)

    Returns:
        Dictionary: {smu_id: {'sand': float, 'clay': float, 'silt': float}}
    """
    # Filter to specified layer and primary sequence
    # LAYER column: "D1" = 0-20cm topsoil, "D2" = 20-40cm subsoil
    # SEQUENCE column: 1 = dominant soil, 2+ = subdominant
    filtered = hwsd_layers_df[
        (hwsd_layers_df['LAYER'] == layer) &
        (hwsd_layers_df['SEQUENCE'] == 1)
    ]

    # Create lookup dictionary
    # IMPORTANT: Use 'HWSD2_SMU_ID' column, NOT 'ID' column!
    # - 'ID' is just a row number in the database
    # - 'HWSD2_SMU_ID' is the actual SMU ID that matches the raster values
    lookup = {}
    for _, row in filtered.iterrows():
        smu_id = int(row['HWSD2_SMU_ID'])
        lookup[smu_id] = {
            'sand': float(row['SAND']) if pd.notna(row['SAND']) else np.nan,
            'clay': float(row['CLAY']) if pd.notna(row['CLAY']) else np.nan,
            'silt': float(row['SILT']) if pd.notna(row['SILT']) else np.nan,
        }

    return lookup


def extract_hwsd_soil_data(
    bil_path: str,
    mdb_path: str,
    cells_df: pd.DataFrame,
    layer: str = "D1",
    default_sand: float = 60.0,
    default_clay: float = 18.0,
    default_silt: float = 22.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract HWSD soil properties for grid cells.

    This is the main function that combines all steps:
    1. Export HWSD2_LAYERS table from MDB
    2. Extract SMU IDs from BIL raster for each cell
    3. Look up soil properties for each SMU ID

    Args:
        bil_path: Path to HWSD2.bil raster file
        mdb_path: Path to HWSD2.mdb database file
        cells_df: DataFrame with 'lat', 'lon', 'cell_id' columns
        layer: Soil layer ("D1" for topsoil, "D2" for subsoil)
        default_sand: Default sand % for missing data
        default_clay: Default clay % for missing data
        default_silt: Default silt % for missing data
        verbose: Print progress messages

    Returns:
        DataFrame with columns: cell_id, lat, lon, sand, clay, silt, smu_id
    """
    if verbose:
        print(f"Extracting HWSD soil data...")
        print(f"  BIL raster: {bil_path}")
        print(f"  MDB database: {mdb_path}")
        print(f"  Layer: {layer} ({'topsoil 0-20cm' if layer == 'D1' else 'subsoil 20-40cm'})")

    # Step 1: Export HWSD2_LAYERS table from MDB
    if verbose:
        print(f"\n[1/3] Exporting HWSD2_LAYERS table from MDB...")

    hwsd_df = export_mdb_table(mdb_path, "HWSD2_LAYERS")

    if verbose:
        print(f"  Loaded {len(hwsd_df)} records from HWSD2_LAYERS")

    # Step 2: Create SMU -> soil properties lookup
    if verbose:
        print(f"\n[2/3] Creating SMU lookup table...")

    smu_lookup = create_smu_lookup(hwsd_df, layer=layer)

    if verbose:
        print(f"  Created lookup for {len(smu_lookup)} unique SMU IDs")

    # Step 3: Extract SMU IDs from raster for each cell
    if verbose:
        print(f"\n[3/3] Extracting SMU IDs from raster for {len(cells_df)} cells...")

    smu_ids = extract_smu_ids_from_raster(bil_path, cells_df)

    # Get unique SMU IDs in our region
    unique_smu = np.unique(smu_ids[smu_ids > 0])
    if verbose:
        print(f"  Found {len(unique_smu)} unique SMU IDs in region: {unique_smu}")

    # Step 4: Look up soil properties for each cell
    sand_values = []
    clay_values = []
    silt_values = []
    missing_count = 0

    for smu_id in smu_ids:
        if smu_id in smu_lookup:
            props = smu_lookup[smu_id]
            sand_values.append(props['sand'])
            clay_values.append(props['clay'])
            silt_values.append(props['silt'])
        else:
            # SMU ID not in lookup - use defaults
            sand_values.append(default_sand)
            clay_values.append(default_clay)
            silt_values.append(default_silt)
            missing_count += 1

    # Handle any NaN values
    sand_arr = np.array(sand_values)
    clay_arr = np.array(clay_values)
    silt_arr = np.array(silt_values)

    nan_mask = np.isnan(sand_arr) | np.isnan(clay_arr) | np.isnan(silt_arr)
    if np.any(nan_mask):
        sand_arr[nan_mask] = default_sand
        clay_arr[nan_mask] = default_clay
        silt_arr[nan_mask] = default_silt
        missing_count += np.sum(nan_mask)

    if verbose:
        print(f"\n  Results:")
        print(f"    Cells with HWSD data: {len(cells_df) - missing_count}")
        print(f"    Cells using defaults: {missing_count}")
        print(f"    Sand range: {np.min(sand_arr):.1f}% - {np.max(sand_arr):.1f}%")
        print(f"    Clay range: {np.min(clay_arr):.1f}% - {np.max(clay_arr):.1f}%")
        print(f"    Silt range: {np.min(silt_arr):.1f}% - {np.max(silt_arr):.1f}%")

    # Create output DataFrame
    result_df = cells_df.copy()
    result_df['sand'] = sand_arr
    result_df['clay'] = clay_arr
    result_df['silt'] = silt_arr
    result_df['smu_id'] = smu_ids

    return result_df


# =============================================================================
# Command-line interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract HWSD soil data from BIL+MDB format"
    )
    parser.add_argument(
        "--bil", required=True,
        help="Path to HWSD2.bil raster file"
    )
    parser.add_argument(
        "--mdb", required=True,
        help="Path to HWSD2.mdb database file"
    )
    parser.add_argument(
        "--cells", required=True,
        help="Path to grid cells CSV (must have lat, lon, cell_id columns)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path"
    )
    parser.add_argument(
        "--layer", default="D1", choices=["D1", "D2"],
        help="Soil layer: D1=topsoil (0-20cm), D2=subsoil (20-40cm)"
    )

    args = parser.parse_args()

    # Load grid cells
    cells_df = pd.read_csv(args.cells)

    # Extract soil data
    result_df = extract_hwsd_soil_data(
        bil_path=args.bil,
        mdb_path=args.mdb,
        cells_df=cells_df,
        layer=args.layer,
        verbose=True
    )

    # Save output
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
