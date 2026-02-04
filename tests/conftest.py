"""
Pytest configuration and shared fixtures for prismpy tests.
"""

import pytest
from datetime import date
from pathlib import Path
import tempfile
import shutil

from prismpy.config.schema import (
    ProjectConfig,
    ProjectInfo,
    RegionConfig,
    BoundaryConfig,
    BoundarySource,
    ManualBoundsConfig,
    CropConfig,
    CropCalendarConfig,
    TemporalConfig,
    OutputConfig,
    Platform,
)
from prismpy.models.region import Region, BoundingBox
from prismpy.models.spatial import SpatialGrid, GridCell
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.crop import CropParameters, CropCalendar


# =============================================================================
# Test Data Constants
# =============================================================================

TEST_REGION_NAME = "Koutiala"
TEST_COUNTRY = "Mali"
TEST_COUNTRY_ISO3 = "MLI"

# Koutiala, Mali bounding box (approximate)
TEST_MINX = -6.0
TEST_MINY = 11.5
TEST_MAXX = -5.0
TEST_MAXY = 12.5


# =============================================================================
# Fixtures: Configuration
# =============================================================================

@pytest.fixture
def sample_bounding_box():
    """Create a sample bounding box for Koutiala, Mali."""
    return BoundingBox(
        minx=TEST_MINX,
        miny=TEST_MINY,
        maxx=TEST_MAXX,
        maxy=TEST_MAXY,
    )


@pytest.fixture
def sample_region(sample_bounding_box):
    """Create a sample region."""
    return Region(
        name=TEST_REGION_NAME,
        country=TEST_COUNTRY,
        country_iso3=TEST_COUNTRY_ISO3,
        bounds=sample_bounding_box,
    )


@pytest.fixture
def sample_project_config():
    """Create a sample project configuration."""
    return ProjectConfig(
        project=ProjectInfo(
            name="test_mali_maize",
            description="Test project for Mali maize",
        ),
        region=RegionConfig(
            name=TEST_REGION_NAME,
            country=TEST_COUNTRY,
            country_iso3=TEST_COUNTRY_ISO3,
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=TEST_MINX,
                    miny=TEST_MINY,
                    maxx=TEST_MAXX,
                    maxy=TEST_MAXY,
                ),
            ),
        ),
        crop=CropConfig(
            name="Maize",
            name_short="mai",
            variety="Medium-duration",
            calendar=CropCalendarConfig(
                planting_doy=166,
                maturity_doy=285,
            ),
        ),
        temporal=TemporalConfig(
            start_year=2015,
            end_year=2020,
            spinup_years=2,
        ),
        targets=[Platform.SARRA_PY, Platform.CRAFT],
        output=OutputConfig(
            base_dir="outputs",
            structure="by_platform",
        ),
    )


# =============================================================================
# Fixtures: Climate Data
# =============================================================================

@pytest.fixture
def sample_climate_record():
    """Create a sample daily climate record."""
    return ClimateRecord(
        date=date(2015, 6, 15),
        tmax=35.0,
        tmin=22.0,
        precip=5.2,
        srad=22.5,
        wind=2.0,
        rh=65.0,
    )


@pytest.fixture
def sample_climate_timeseries():
    """Create a sample climate time series with 30 days of data."""
    records = []
    start_date = date(2015, 6, 1)

    for i in range(30):
        d = date(2015, 6, 1 + i) if i < 30 else date(2015, 7, i - 29)
        try:
            d = date(2015, 6, 1)
            from datetime import timedelta
            d = start_date + timedelta(days=i)
        except ValueError:
            continue

        records.append(ClimateRecord(
            date=d,
            tmax=32.0 + (i % 5),  # Vary between 32-36
            tmin=20.0 + (i % 3),  # Vary between 20-22
            precip=max(0, (i % 7) * 2 - 3),  # Some days with rain
            srad=20.0 + (i % 4),  # Vary between 20-23
        ))

    return ClimateTimeSeries(
        location_id=1001,
        lat=12.0,
        lon=-5.5,
        source="TEST_DATA",
        records=records,
        elevation=300.0,
    )


# =============================================================================
# Fixtures: Soil Data
# =============================================================================

@pytest.fixture
def sample_soil_layer():
    """Create a sample soil layer."""
    return SoilLayer(
        depth_top=0.0,
        depth_bottom=0.3,
        sand=45.0,
        clay=25.0,
        organic_carbon=1.5,
        bulk_density=1.35,
        ph=6.5,
    )


@pytest.fixture
def sample_soil_profile(sample_soil_layer):
    """Create a sample soil profile with multiple layers."""
    layers = [
        sample_soil_layer,  # 0-30cm
        SoilLayer(
            depth_top=0.3,
            depth_bottom=0.6,
            sand=40.0,
            clay=30.0,
            organic_carbon=0.8,
            bulk_density=1.40,
            ph=6.3,
        ),
        SoilLayer(
            depth_top=0.6,
            depth_bottom=1.0,
            sand=35.0,
            clay=35.0,
            organic_carbon=0.5,
            bulk_density=1.45,
            ph=6.0,
        ),
    ]

    return SoilProfile(
        profile_id="SOIL_001",
        lat=12.0,
        lon=-5.5,
        source="TEST_DATA",
        layers=layers,
    )


# =============================================================================
# Fixtures: Spatial Grid
# =============================================================================

@pytest.fixture
def sample_grid_cells():
    """Create sample grid cells for a small test area."""
    cells = []
    cell_id = 0

    # Create a 3x3 grid of cells
    for row in range(3):
        for col in range(3):
            lat = 12.0 - row * 0.0833  # ~5 arcmin spacing
            lon = -5.5 + col * 0.0833
            cells.append(GridCell(
                cell_id=cell_id,
                lat=lat,
                lon=lon,
                row=row,
                col=col,
            ))
            cell_id += 1

    return cells


@pytest.fixture
def sample_spatial_grid(sample_bounding_box, sample_grid_cells):
    """Create a sample spatial grid."""
    return SpatialGrid(
        bounds=sample_bounding_box,
        resolution=5 / 60,  # 5 arcmin
        cells=sample_grid_cells,
    )


# =============================================================================
# Fixtures: Crop Data
# =============================================================================

@pytest.fixture
def sample_crop_calendar():
    """Create a sample crop calendar."""
    return CropCalendar(
        location_id=1001,
        planting_doy=166,
        maturity_doy=285,
    )


@pytest.fixture
def sample_crop_params():
    """Create sample crop parameters for maize."""
    return CropParameters(
        crop_name="Maize",
        variety_name="Medium-duration",
        source="test",
        parameters={
            "base_temp": 8.0,
            "temp_opt1": 30.0,
            "temp_limit": 40.0,
            "gdd_emergence": 50,
            "gdd_flowering": 800,
            "gdd_maturity": 1500,
        },
    )


# =============================================================================
# Fixtures: Temporary Directories
# =============================================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="prismpy_test_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with sample data files."""
    temp_dir = tempfile.mkdtemp(prefix="prismpy_data_")
    temp_path = Path(temp_dir)

    # Create subdirectories
    (temp_path / "climate").mkdir()
    (temp_path / "soil").mkdir()
    (temp_path / "config").mkdir()

    yield temp_path
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)
