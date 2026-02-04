"""
Unit tests for prismpy data models.
"""

import pytest
from datetime import date
import numpy as np

from prismpy.models.region import BoundingBox, Region
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.soil import SoilLayer, SoilProfile


# =============================================================================
# BoundingBox Tests
# =============================================================================

class TestBoundingBox:
    """Tests for the BoundingBox class."""

    def test_creation(self, sample_bounding_box):
        """Test basic BoundingBox creation."""
        assert sample_bounding_box.minx == -6.0
        assert sample_bounding_box.miny == 11.5
        assert sample_bounding_box.maxx == -5.0
        assert sample_bounding_box.maxy == 12.5

    def test_width_height(self, sample_bounding_box):
        """Test width and height calculations."""
        assert sample_bounding_box.width == 1.0
        assert sample_bounding_box.height == 1.0

    def test_center(self, sample_bounding_box):
        """Test center point calculation."""
        center_lon, center_lat = sample_bounding_box.center
        assert center_lon == -5.5
        assert center_lat == 12.0

    def test_to_gis_format(self, sample_bounding_box):
        """Test GIS format output [minx, miny, maxx, maxy]."""
        gis_format = sample_bounding_box.to_gis_format()
        assert gis_format == [-6.0, 11.5, -5.0, 12.5]

    def test_to_sarra_py_format(self, sample_bounding_box):
        """Test SARRA-Py format output [lat_NW, lon_NW, lat_SE, lon_SE]."""
        sarra_format = sample_bounding_box.to_sarra_py_format()
        # SARRA-Py: [lat_NW, lon_NW, lat_SE, lon_SE]
        # NW corner: (maxy, minx), SE corner: (miny, maxx)
        assert sarra_format == [12.5, -6.0, 11.5, -5.0]

    def test_contains_point(self, sample_bounding_box):
        """Test point containment check."""
        # Point inside
        assert sample_bounding_box.contains_point(-5.5, 12.0)
        # Point outside
        assert not sample_bounding_box.contains_point(-7.0, 12.0)
        # Point on edge
        assert sample_bounding_box.contains_point(-6.0, 12.0)

    def test_from_gis_format(self):
        """Test creation from GIS format."""
        bounds = [-6.0, 11.5, -5.0, 12.5]
        bb = BoundingBox.from_gis_format(bounds)
        assert bb.minx == -6.0
        assert bb.miny == 11.5
        assert bb.maxx == -5.0
        assert bb.maxy == 12.5

    def test_validation_invalid_bounds(self):
        """Test validation catches invalid bounds."""
        # minx > maxx
        with pytest.raises(ValueError):
            BoundingBox(minx=0, miny=0, maxx=-1, maxy=1)

        # miny > maxy
        with pytest.raises(ValueError):
            BoundingBox(minx=0, miny=2, maxx=1, maxy=1)


# =============================================================================
# Region Tests
# =============================================================================

class TestRegion:
    """Tests for the Region class."""

    def test_creation(self, sample_region):
        """Test basic Region creation."""
        assert sample_region.name == "Koutiala"
        assert sample_region.country == "Mali"
        assert sample_region.country_iso3 == "MLI"

    def test_bounds_access(self, sample_region):
        """Test access to bounds."""
        assert sample_region.bounds is not None
        assert sample_region.bounds.minx == -6.0

    def test_to_dict(self, sample_region):
        """Test conversion to dictionary."""
        d = sample_region.to_dict()
        assert d["name"] == "Koutiala"
        assert d["country"] == "Mali"
        assert "bounds" in d


# =============================================================================
# ClimateRecord Tests
# =============================================================================

class TestClimateRecord:
    """Tests for the ClimateRecord class."""

    def test_creation(self, sample_climate_record):
        """Test basic ClimateRecord creation."""
        assert sample_climate_record.tmax == 35.0
        assert sample_climate_record.tmin == 22.0
        assert sample_climate_record.precip == 5.2
        assert sample_climate_record.srad == 22.5

    def test_tmean_computed(self, sample_climate_record):
        """Test that tmean is computed from tmax and tmin."""
        expected_tmean = (35.0 + 22.0) / 2
        assert sample_climate_record.tmean == expected_tmean

    def test_doy_property(self, sample_climate_record):
        """Test day of year property."""
        # June 15 = DOY 166
        assert sample_climate_record.doy == 166

    def test_year_property(self, sample_climate_record):
        """Test year property."""
        assert sample_climate_record.year == 2015

    def test_to_dict(self, sample_climate_record):
        """Test conversion to dictionary."""
        d = sample_climate_record.to_dict()
        assert d["tmax"] == 35.0
        assert d["doy"] == 166
        assert d["date"] == "2015-06-15"

    def test_validate_valid_record(self, sample_climate_record):
        """Test validation of a valid record."""
        errors = sample_climate_record.validate()
        assert len(errors) == 0

    def test_validate_invalid_tmax(self):
        """Test validation catches invalid tmax."""
        record = ClimateRecord(
            date=date(2015, 6, 15),
            tmax=70.0,  # Too high
            tmin=22.0,
            precip=5.0,
            srad=20.0,
        )
        errors = record.validate()
        assert len(errors) > 0
        assert any("tmax" in e for e in errors)

    def test_validate_tmin_greater_than_tmax(self):
        """Test validation catches tmin > tmax."""
        record = ClimateRecord(
            date=date(2015, 6, 15),
            tmax=20.0,
            tmin=25.0,  # Higher than tmax
            precip=5.0,
            srad=20.0,
        )
        errors = record.validate()
        assert len(errors) > 0
        assert any("tmin" in e and "tmax" in e for e in errors)


# =============================================================================
# ClimateTimeSeries Tests
# =============================================================================

class TestClimateTimeSeries:
    """Tests for the ClimateTimeSeries class."""

    def test_creation(self, sample_climate_timeseries):
        """Test basic ClimateTimeSeries creation."""
        assert sample_climate_timeseries.location_id == 1001
        assert sample_climate_timeseries.lat == 12.0
        assert sample_climate_timeseries.lon == -5.5
        assert sample_climate_timeseries.source == "TEST_DATA"

    def test_n_records(self, sample_climate_timeseries):
        """Test record count."""
        assert sample_climate_timeseries.n_records == 30

    def test_date_range(self, sample_climate_timeseries):
        """Test date range property."""
        start, end = sample_climate_timeseries.date_range
        assert start == date(2015, 6, 1)
        assert end == date(2015, 6, 30)

    def test_years_property(self, sample_climate_timeseries):
        """Test years list."""
        years = sample_climate_timeseries.years
        assert 2015 in years

    def test_get_records_for_year(self, sample_climate_timeseries):
        """Test getting records for a specific year."""
        records_2015 = sample_climate_timeseries.get_records_for_year(2015)
        assert len(records_2015) == 30

        records_2016 = sample_climate_timeseries.get_records_for_year(2016)
        assert len(records_2016) == 0

    def test_get_record_for_date(self, sample_climate_timeseries):
        """Test getting record for a specific date."""
        record = sample_climate_timeseries.get_record_for_date(date(2015, 6, 15))
        assert record is not None
        assert record.date == date(2015, 6, 15)

        # Non-existent date
        record = sample_climate_timeseries.get_record_for_date(date(2015, 7, 15))
        assert record is None

    def test_to_numpy_arrays(self, sample_climate_timeseries):
        """Test conversion to numpy arrays."""
        arrays = sample_climate_timeseries.to_numpy_arrays()
        assert "tmax" in arrays
        assert "tmin" in arrays
        assert "precip" in arrays
        assert "srad" in arrays
        assert len(arrays["tmax"]) == 30

    def test_to_acea_pickle_format(self, sample_climate_timeseries):
        """Test conversion to ACEA pickle format."""
        tmax, tmin, prec, et0 = sample_climate_timeseries.to_acea_pickle_format()
        assert isinstance(tmax, np.ndarray)
        assert isinstance(tmin, np.ndarray)
        assert isinstance(prec, np.ndarray)
        assert isinstance(et0, np.ndarray)
        assert len(tmax) == 30


# =============================================================================
# SoilLayer Tests
# =============================================================================

class TestSoilLayer:
    """Tests for the SoilLayer class."""

    def test_creation(self, sample_soil_layer):
        """Test basic SoilLayer creation."""
        assert sample_soil_layer.depth_top == 0.0
        assert sample_soil_layer.depth_bottom == 0.3
        assert sample_soil_layer.sand == 45.0
        assert sample_soil_layer.clay == 25.0

    def test_silt_computed(self, sample_soil_layer):
        """Test that silt is computed from sand and clay."""
        expected_silt = 100 - 45 - 25  # 30
        assert sample_soil_layer.silt == expected_silt

    def test_thickness(self, sample_soil_layer):
        """Test thickness property."""
        assert sample_soil_layer.thickness == 0.3

    def test_to_dict(self, sample_soil_layer):
        """Test conversion to dictionary."""
        d = sample_soil_layer.to_dict()
        assert d["sand"] == 45.0
        assert d["clay"] == 25.0
        assert d["thickness"] == 0.3

    def test_validate_valid_layer(self, sample_soil_layer):
        """Test validation of a valid layer."""
        errors = sample_soil_layer.validate()
        assert len(errors) == 0

    def test_validate_invalid_sand(self):
        """Test validation catches invalid sand percentage."""
        layer = SoilLayer(
            depth_top=0,
            depth_bottom=0.3,
            sand=110,  # Invalid
            clay=25,
        )
        errors = layer.validate()
        assert len(errors) > 0

    def test_validate_sand_clay_sum(self):
        """Test validation catches sand + clay > 100."""
        layer = SoilLayer(
            depth_top=0,
            depth_bottom=0.3,
            sand=60,
            clay=50,  # Sum = 110
        )
        errors = layer.validate()
        assert len(errors) > 0

    def test_estimate_hydraulic_properties(self):
        """Test hydraulic property estimation."""
        layer = SoilLayer(
            depth_top=0,
            depth_bottom=0.3,
            sand=40,
            clay=30,
            organic_carbon=1.5,
        )
        layer.estimate_hydraulic_properties()
        assert layer.wilting_point is not None
        assert layer.field_capacity is not None
        assert layer.saturated_wc is not None
        assert layer.wilting_point < layer.field_capacity < layer.saturated_wc


# =============================================================================
# SoilProfile Tests
# =============================================================================

class TestSoilProfile:
    """Tests for the SoilProfile class."""

    def test_creation(self, sample_soil_profile):
        """Test basic SoilProfile creation."""
        assert sample_soil_profile.profile_id == "SOIL_001"
        assert sample_soil_profile.lat == 12.0
        assert sample_soil_profile.lon == -5.5
        assert sample_soil_profile.source == "TEST_DATA"

    def test_n_layers(self, sample_soil_profile):
        """Test layer count."""
        assert sample_soil_profile.n_layers == 3

    def test_total_depth(self, sample_soil_profile):
        """Test total depth calculation."""
        assert sample_soil_profile.total_depth == 1.0

    def test_surface_texture(self, sample_soil_profile):
        """Test surface texture classification."""
        texture = sample_soil_profile.surface_texture
        assert texture is not None
        # Sand=45, Clay=25 -> Loam or Clay Loam
        assert texture in ["Loam", "Clay Loam", "Sandy Clay Loam"]

    def test_get_layer_at_depth(self, sample_soil_profile):
        """Test getting layer at a specific depth."""
        # Surface layer
        layer = sample_soil_profile.get_layer_at_depth(0.1)
        assert layer is not None
        assert layer.depth_top == 0.0

        # Middle layer
        layer = sample_soil_profile.get_layer_at_depth(0.4)
        assert layer is not None
        assert layer.depth_top == 0.3

        # Below profile
        layer = sample_soil_profile.get_layer_at_depth(1.5)
        assert layer is None

    def test_get_weighted_average(self, sample_soil_profile):
        """Test weighted average calculation."""
        avg_sand = sample_soil_profile.get_weighted_average("sand")
        assert avg_sand is not None
        # Should be between 35 and 45 (the range of sand values)
        assert 35 <= avg_sand <= 45

    def test_get_weighted_average_max_depth(self, sample_soil_profile):
        """Test weighted average with max depth."""
        avg_sand_top30 = sample_soil_profile.get_weighted_average("sand", max_depth=0.3)
        assert avg_sand_top30 == 45.0  # Only surface layer

    def test_validate_valid_profile(self, sample_soil_profile):
        """Test validation of a valid profile."""
        errors = sample_soil_profile.validate()
        assert len(errors) == 0

    def test_validate_empty_profile(self):
        """Test validation catches empty profile."""
        profile = SoilProfile(
            profile_id="EMPTY",
            lat=12.0,
            lon=-5.5,
            source="TEST",
            layers=[],
        )
        errors = profile.validate()
        assert len(errors) > 0
        assert any("no layers" in e.lower() for e in errors)

    def test_from_single_layer(self):
        """Test creation from single layer."""
        profile = SoilProfile.from_single_layer(
            profile_id="SIMPLE",
            lat=12.0,
            lon=-5.5,
            source="TEST",
            sand=40,
            clay=30,
            depth=1.0,
        )
        assert profile.n_layers == 1
        assert profile.total_depth == 1.0
        assert profile.layers[0].sand == 40


# =============================================================================
# GridCell Tests
# =============================================================================

class TestGridCell:
    """Tests for the GridCell class."""

    def test_creation(self):
        """Test basic GridCell creation."""
        cell = GridCell(
            cell_id=12345,
            lat=12.0,
            lon=-5.5,
            row=100,
            col=200,
        )
        assert cell.cell_id == 12345
        assert cell.lat == 12.0
        assert cell.lon == -5.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        cell = GridCell(
            cell_id=12345,
            lat=12.0,
            lon=-5.5,
            row=100,
            col=200,
        )
        d = cell.to_dict()
        assert d["cell_id"] == 12345
        assert d["lat"] == 12.0


# =============================================================================
# SpatialGrid Tests
# =============================================================================

class TestSpatialGrid:
    """Tests for the SpatialGrid class."""

    def test_creation(self, sample_spatial_grid):
        """Test basic SpatialGrid creation."""
        assert sample_spatial_grid.resolution == pytest.approx(5 / 60)
        assert len(sample_spatial_grid.cells) == 9

    def test_n_cells(self, sample_spatial_grid):
        """Test cell count."""
        assert sample_spatial_grid.n_cells == 9

    def test_get_cell_by_id(self, sample_spatial_grid):
        """Test getting cell by ID."""
        cell = sample_spatial_grid.get_cell_by_id(0)
        assert cell is not None
        assert cell.cell_id == 0

        # Non-existent ID
        cell = sample_spatial_grid.get_cell_by_id(999)
        assert cell is None
