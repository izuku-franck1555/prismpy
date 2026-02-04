"""
Integration tests for prismpy pipeline.

These tests verify the complete workflow from configuration to output generation.
"""

import pytest
from pathlib import Path
from datetime import date
import yaml
import json

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
from prismpy.models.provenance import DecisionType
from prismpy.config.loader import load_config, save_config
from prismpy.models.region import Region, BoundingBox
from prismpy.models.spatial import SpatialGrid, GridCell
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.crop import CropParameters, CropCalendar
from prismpy.translators.base import UnifiedData, TranslationResult
from prismpy.provenance.tracker import ProvenanceTracker


# =============================================================================
# Configuration Loading Tests
# =============================================================================

class TestConfigurationLoading:
    """Tests for configuration loading and saving."""

    def test_save_and_load_config(self, temp_output_dir, sample_project_config):
        """Test saving and loading a configuration file."""
        config_path = temp_output_dir / "test_config.yaml"

        # Save configuration
        save_config(sample_project_config, config_path)
        assert config_path.exists()

        # Load configuration
        loaded_config = load_config(config_path)

        # Verify key fields
        assert loaded_config.project.name == sample_project_config.project.name
        assert loaded_config.region.name == sample_project_config.region.name
        assert loaded_config.crop.name == sample_project_config.crop.name
        assert loaded_config.temporal.start_year == sample_project_config.temporal.start_year

    def test_config_yaml_format(self, temp_output_dir, sample_project_config):
        """Test that saved config is valid YAML."""
        config_path = temp_output_dir / "test_config.yaml"
        save_config(sample_project_config, config_path)

        # Load as raw YAML
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        assert 'project' in raw_config
        assert 'region' in raw_config
        assert 'crop' in raw_config
        assert 'temporal' in raw_config

    def test_config_validation(self, sample_project_config):
        """Test configuration validation."""
        # Valid configuration should pass
        assert sample_project_config.project.name == "test_mali_maize"

        # Invalid configuration should raise error
        with pytest.raises(ValueError):
            ProjectConfig(
                project=ProjectInfo(name="test"),
                region=RegionConfig(
                    name="Test",
                    country="XX",  # Invalid ISO code length
                    boundary=BoundaryConfig(source="manual", bounds=[0, 0, 1, 1]),
                ),
                crop=CropConfig(name="Maize"),
                temporal=TemporalConfig(start_year=2020, end_year=2015),  # Invalid range
                targets=[Platform.SARRA_PY],
                output=OutputConfig(base_dir="outputs"),
            )


# =============================================================================
# Data Model Integration Tests
# =============================================================================

class TestDataModelIntegration:
    """Tests for data model integration."""

    def test_unified_data_creation(
        self,
        sample_region,
        sample_spatial_grid,
        sample_climate_timeseries,
        sample_soil_profile,
        sample_crop_params,
        sample_crop_calendar,
    ):
        """Test creating UnifiedData container."""
        unified_data = UnifiedData(
            region=sample_region,
            grid=sample_spatial_grid,
            climate={1001: sample_climate_timeseries},
            soil={1001: sample_soil_profile},
            crop_params=sample_crop_params,
            crop_calendar={1001: sample_crop_calendar},
            metadata={"source": "test"},
        )

        assert unified_data.region.name == "Koutiala"
        assert unified_data.grid.n_cells == 9
        assert 1001 in unified_data.climate
        assert 1001 in unified_data.soil
        assert unified_data.crop_params.crop_name == "Maize"

    def test_climate_soil_consistency(
        self,
        sample_climate_timeseries,
        sample_soil_profile,
    ):
        """Test climate and soil data have consistent locations."""
        # Both should have the same location
        assert sample_climate_timeseries.lat == sample_soil_profile.lat
        assert sample_climate_timeseries.lon == sample_soil_profile.lon

    def test_bounding_box_format_conversion(self, sample_bounding_box):
        """Test bounding box format conversions are consistent."""
        gis_format = sample_bounding_box.to_gis_format()
        sarra_format = sample_bounding_box.to_sarra_py_format()

        # GIS: [minx, miny, maxx, maxy]
        # SARRA: [lat_NW, lon_NW, lat_SE, lon_SE]

        # NW corner: (maxy, minx)
        assert sarra_format[0] == gis_format[3]  # lat_NW = maxy
        assert sarra_format[1] == gis_format[0]  # lon_NW = minx

        # SE corner: (miny, maxx)
        assert sarra_format[2] == gis_format[1]  # lat_SE = miny
        assert sarra_format[3] == gis_format[2]  # lon_SE = maxx


# =============================================================================
# Provenance Tracking Tests
# =============================================================================

class TestProvenanceTracking:
    """Tests for provenance tracking system."""

    def test_provenance_tracker_creation(self):
        """Test creating a provenance tracker."""
        tracker = ProvenanceTracker(project_name="test_project")
        assert tracker.session_id is not None
        assert "tr_" in tracker.session_id

    def test_record_decision(self):
        """Test recording a decision."""
        tracker = ProvenanceTracker(project_name="test")

        # Start an artifact first (decisions are attached to artifacts)
        artifact_id = tracker.start_artifact("climate")

        # Create a decision
        decision = tracker.record_decision(
            decision_type=DecisionType.SOURCE_SELECTION,
            description="Selected NASA POWER for climate data",
            rationale="Best coverage for the region",
        )

        # Verify decision was created
        assert decision.description == "Selected NASA POWER for climate data"

        # Check summary includes artifact
        summary = tracker.get_summary()
        assert summary.get('n_artifacts', 0) >= 1

    def test_record_artifact(self, temp_output_dir):
        """Test recording an artifact."""
        tracker = ProvenanceTracker(project_name="test")

        # Create a test file
        test_file = temp_output_dir / "test_artifact.txt"
        test_file.write_text("test content")

        # Start tracking the artifact
        artifact_id = tracker.start_artifact(
            artifact_type="output",
            artifact_id="test_001",
        )

        # Verify artifact was started
        assert artifact_id == "test_001"

        # Record a retrieval for this artifact
        tracker.record_retrieval(
            source="test_source",
            parameters={"file": str(test_file)},
            output_path=test_file,
        )

        # Check that artifact exists
        summary = tracker.get_summary()
        assert summary.get('n_artifacts', 0) >= 1

    def test_export_provenance(self, temp_output_dir):
        """Test exporting provenance to JSON."""
        tracker = ProvenanceTracker(project_name="test")

        # Start an artifact and add a transformation
        tracker.start_artifact("test_artifact")
        tracker.record_retrieval(
            source="TEST",
            parameters={"test": "value"},
        )

        output_path = temp_output_dir / "provenance.json"
        tracker.save(output_path)

        assert output_path.exists()

        with open(output_path, 'r') as f:
            exported = json.load(f)

        assert 'session_id' in exported
        assert 'artifacts' in exported


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================

class TestEndToEndWorkflow:
    """Tests for complete workflow scenarios."""

    def test_mali_maize_workflow(
        self,
        temp_output_dir,
        sample_project_config,
        sample_region,
        sample_climate_timeseries,
        sample_soil_profile,
        sample_crop_params,
        sample_crop_calendar,
    ):
        """Test a complete workflow for Mali maize."""
        # 1. Create unified data
        unified_data = UnifiedData(
            region=sample_region,
            grid=None,  # SARRA-Py doesn't need grid
            climate={0: sample_climate_timeseries},
            soil={0: sample_soil_profile},
            crop_params=sample_crop_params,
            crop_calendar={0: sample_crop_calendar},
        )

        # 2. Verify data is complete
        assert unified_data.region is not None
        assert len(unified_data.climate) > 0
        assert len(unified_data.soil) > 0

        # 3. Check data consistency
        for loc_id, climate in unified_data.climate.items():
            assert climate.n_records > 0
            errors = climate.validate_all()
            # Should have no validation errors
            assert len(errors) == 0, f"Climate validation errors: {errors}"

        for loc_id, soil in unified_data.soil.items():
            errors = soil.validate()
            assert len(errors) == 0, f"Soil validation errors: {errors}"

    def test_multi_platform_config(self):
        """Test configuration for multiple platforms."""
        config = ProjectConfig(
            project=ProjectInfo(name="multi_platform_test"),
            region=RegionConfig(
                name="TestRegion",
                country="Mali",
                country_iso3="MLI",
                boundary=BoundaryConfig(
                    source=BoundarySource.MANUAL,
                    manual_bounds=ManualBoundsConfig(
                        minx=-6.0,
                        miny=11.0,
                        maxx=-5.0,
                        maxy=12.0,
                    ),
                ),
            ),
            crop=CropConfig(
                name="Maize",
                name_short="mai",
                calendar=CropCalendarConfig(
                    planting_doy=166,
                    maturity_doy=285,
                ),
            ),
            temporal=TemporalConfig(start_year=2015, end_year=2020),
            targets=[Platform.SARRA_PY, Platform.CRAFT, Platform.PYTHIA, Platform.ACEA],
            output=OutputConfig(base_dir="outputs"),
        )

        # Should have 4 targets
        assert len(config.targets) == 4
        assert Platform.SARRA_PY in config.targets
        assert Platform.CRAFT in config.targets
        assert Platform.PYTHIA in config.targets
        assert Platform.ACEA in config.targets


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_invalid_config_file(self, temp_output_dir):
        """Test loading an invalid config file."""
        invalid_path = temp_output_dir / "nonexistent.yaml"

        with pytest.raises(Exception):
            load_config(invalid_path)

    def test_empty_climate_timeseries(self):
        """Test handling empty climate time series."""
        ts = ClimateTimeSeries(
            location_id=1,
            lat=12.0,
            lon=-5.5,
            source="TEST",
            records=[],
        )

        assert ts.n_records == 0
        assert ts.date_range is None
        completeness = ts.check_completeness()
        assert completeness['complete'] is False

    def test_invalid_soil_profile(self):
        """Test handling invalid soil profile."""
        profile = SoilProfile(
            profile_id="INVALID",
            lat=12.0,
            lon=-5.5,
            source="TEST",
            layers=[],  # Empty layers
        )

        errors = profile.validate()
        assert len(errors) > 0

    def test_boundary_validation(self):
        """Test boundary coordinate validation."""
        # Valid bounds
        valid_bb = BoundingBox(minx=-6, miny=11, maxx=-5, maxy=12)
        assert valid_bb.width == 1.0

        # Invalid bounds (minx > maxx)
        with pytest.raises(ValueError):
            BoundingBox(minx=-5, miny=11, maxx=-6, maxy=12)


# =============================================================================
# Platform-Specific Integration Tests
# =============================================================================

class TestPlatformSpecificIntegration:
    """Tests for platform-specific integration."""

    def test_sarra_py_data_requirements(self, sample_region, sample_climate_timeseries, sample_soil_profile):
        """Test SARRA-Py data requirements."""
        # SARRA-Py needs: region, climate, soil, crop_params
        unified_data = UnifiedData(
            region=sample_region,
            climate={0: sample_climate_timeseries},
            soil={0: sample_soil_profile},
        )

        # Verify required data is present
        assert unified_data.region is not None
        assert unified_data.climate is not None
        assert unified_data.soil is not None

    def test_craft_cell_id_calculation(self, sample_bounding_box):
        """Test CRAFT cell ID calculation."""
        from prismpy.utils.gis_utils import latlon_to_rowcol, compute_cell_id_global

        # Calculate cell ID for center of bounding box
        center_lon, center_lat = sample_bounding_box.center
        resolution = 5 / 60  # 5 arcmin in degrees

        # First convert lat/lon to row/col
        row, col = latlon_to_rowcol(center_lat, center_lon, resolution)

        # Then compute cell ID
        cell_id = compute_cell_id_global(row, col)

        # Should be a valid cell ID
        assert 0 <= cell_id < 4320 * 2160

    def test_acea_pickle_format(self, sample_climate_timeseries):
        """Test ACEA pickle format generation."""
        tmax, tmin, prec, et0 = sample_climate_timeseries.to_acea_pickle_format()

        # Verify format
        assert len(tmax) == len(tmin) == len(prec) == len(et0)
        assert len(tmax) == sample_climate_timeseries.n_records

        # Verify values are reasonable
        assert all(-50 <= t <= 60 for t in tmax)
        assert all(-50 <= t <= 50 for t in tmin)
        assert all(p >= 0 for p in prec)

    def test_pythia_yrdoy_format(self, sample_climate_timeseries):
        """Test PYTHIA YRDOY date format."""
        from prismpy.utils.date_utils import date_to_yrdoy

        for record in sample_climate_timeseries.records[:5]:
            yrdoy = date_to_yrdoy(record.date)

            # Should be 7-digit number
            assert 1000000 <= yrdoy <= 9999366

            # Year should match
            assert yrdoy // 1000 == record.date.year

            # DOY should match
            assert yrdoy % 1000 == record.doy
