"""
Unit tests for prismpy validators.
"""

import pytest
import json
import pickle
import numpy as np
from pathlib import Path
import yaml

from prismpy.validators.base import (
    BaseValidator,
    ValidationIssue,
    ValidationResult,
)
from prismpy.validators.sarra_py import SarraPyValidator
from prismpy.validators.craft import CraftValidator
from prismpy.validators.pythia import PythiaValidator
from prismpy.validators.acea import AceaValidator
from prismpy.config.schema import Platform


# =============================================================================
# ValidationIssue Tests
# =============================================================================

class TestValidationIssue:
    """Tests for ValidationIssue class."""

    def test_creation(self):
        """Test basic ValidationIssue creation."""
        issue = ValidationIssue(
            severity='error',
            category='schema',
            message='Missing required field',
        )
        assert issue.severity == 'error'
        assert issue.category == 'schema'
        assert issue.message == 'Missing required field'

    def test_with_file_path(self):
        """Test ValidationIssue with file path."""
        issue = ValidationIssue(
            severity='warning',
            category='format',
            message='Unexpected value',
            file_path=Path('/test/file.yaml'),
        )
        assert issue.file_path == Path('/test/file.yaml')

    def test_with_details(self):
        """Test ValidationIssue with details."""
        issue = ValidationIssue(
            severity='error',
            category='range',
            message='Value out of range',
            details={'value': 500, 'max': 366},
        )
        assert issue.details['value'] == 500


# =============================================================================
# ValidationResult Tests
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_valid_result(self):
        """Test ValidationResult with no errors."""
        result = ValidationResult(
            valid=True,
            platform=Platform.SARRA_PY,
            output_dir=Path('/test'),
            issues=[],
            files_checked=5,
        )
        assert result.valid is True
        assert result.n_errors == 0

    def test_invalid_result(self):
        """Test ValidationResult with errors."""
        issues = [
            ValidationIssue(severity='error', category='schema', message='Error 1'),
            ValidationIssue(severity='warning', category='format', message='Warning 1'),
        ]
        result = ValidationResult(
            valid=False,
            platform=Platform.CRAFT,
            output_dir=Path('/test'),
            issues=issues,
            files_checked=3,
        )
        assert result.valid is False
        assert result.n_errors == 1
        assert result.n_warnings == 1

    def test_summary(self):
        """Test ValidationResult summary."""
        result = ValidationResult(
            valid=True,
            platform=Platform.SARRA_PY,
            output_dir=Path('/test'),
            issues=[],
            files_checked=10,
        )
        summary = result.summary()
        assert 'sarra_py' in summary.lower() or 'SARRA' in summary


# =============================================================================
# SARRA-Py Validator Tests
# =============================================================================

class TestSarraPyValidator:
    """Tests for SARRA-Py validator."""

    @pytest.fixture
    def valid_sarra_py_config(self, temp_data_dir):
        """Create a valid SARRA-Py config structure."""
        # Create required directories
        (temp_data_dir / "config").mkdir(exist_ok=True)
        (temp_data_dir / "data" / "boundaries").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "data" / "climate" / "rainfall").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "data" / "climate" / "2m_temperature_24_hour_maximum").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "data" / "climate" / "2m_temperature_24_hour_mean").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "data" / "climate" / "2m_temperature_24_hour_minimum").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "data" / "climate" / "ET0Hargeaves").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "data" / "climate" / "solar_radiation_flux_daily").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "data" / "soil").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "parameters").mkdir(exist_ok=True)
        (temp_data_dir / "validation").mkdir(exist_ok=True)

        # Create project_config.yaml (expected by validator)
        config = {
            'project': {
                'name': 'test_project',
                'description': 'Test description',
            },
            'region': {
                'name': 'Koutiala',
                'bounds': [12.5, -6.0, 11.5, -5.0],  # SARRA-Py format
            },
            'temporal': {
                'start_year': 2015,
                'end_year': 2020,
            },
            'crop': {
                'name': 'Maize',
                'planting_doy': 166,
            },
        }
        with open(temp_data_dir / "config" / "project_config.yaml", 'w') as f:
            yaml.dump(config, f)

        # Create bounds.json (expected by validator)
        bounds_data = {
            'bounds_sarra_py': [12.5, -6.0, 11.5, -5.0],
            'bounds_gis': [-6.0, 11.5, -5.0, 12.5],
        }
        with open(temp_data_dir / "data" / "boundaries" / "bounds.json", 'w') as f:
            json.dump(bounds_data, f)

        # Create required package files
        (temp_data_dir / "README.md").write_text("# Test Package\n")
        (temp_data_dir / "manifest.json").write_text(json.dumps({
            "package_version": "1.0",
            "generator": "test",
            "files": []
        }))
        (temp_data_dir / "provenance.json").write_text(json.dumps({"stages": []}))

        # Create parameter files
        (temp_data_dir / "parameters" / "variety.yaml").write_text(yaml.dump({
            "TBase": 8, "TOpt1": 30, "TOpt2": 35, "TLim": 45,
            "SDJLevee": 50, "SDJBVP": 500, "kcMax": 1.2
        }))
        (temp_data_dir / "parameters" / "itk.yaml").write_text(yaml.dump({
            "DateSemis": "2020-6-15", "densite": 5.5,
            "seuilEauSemis": 10, "irrigAuto": 0
        }))
        (temp_data_dir / "parameters" / "soil.yaml").write_text(yaml.dump({
            "epaisseurSurf": 200, "epaisseurProf": 800,
            "ru": 100, "seuilRuiss": 50
        }))

        # Create validation report
        (temp_data_dir / "validation" / "validation_report.json").write_text(json.dumps({"valid": True}))

        return temp_data_dir

    def test_validate_structure(self, valid_sarra_py_config):
        """Test SARRA-Py structure validation."""
        validator = SarraPyValidator(valid_sarra_py_config)
        issues = validator.validate_structure()
        # Should have some issues since we don't have all required dirs
        assert isinstance(issues, list)

    def test_validate_missing_dir(self, temp_data_dir):
        """Test validation catches missing directory."""
        validator = SarraPyValidator(temp_data_dir / "nonexistent")
        result = validator.validate()
        assert result.valid is False

    def test_bounding_box_format(self, valid_sarra_py_config):
        """Test SARRA-Py bounding box format validation."""
        validator = SarraPyValidator(valid_sarra_py_config)
        # The config has valid SARRA-Py format bounds
        result = validator.validate()
        # Check for bounding box errors
        bbox_errors = [i for i in result.issues if 'bound' in i.message.lower()]
        # Should not have bounding box format errors
        assert all(i.severity != 'error' for i in bbox_errors)


# =============================================================================
# CRAFT Validator Tests
# =============================================================================

class TestCraftValidator:
    """Tests for CRAFT validator."""

    @pytest.fixture
    def valid_craft_structure(self, temp_data_dir):
        """Create a valid CRAFT directory structure."""
        # Create required directories
        (temp_data_dir / "schema" / "CRAFT_Schema" / "Level2" / "Schema").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "schema" / "Python_Schemas" / "Level2").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "weather").mkdir(exist_ok=True)
        (temp_data_dir / "soil").mkdir(exist_ok=True)
        (temp_data_dir / "crop_mask").mkdir(exist_ok=True)
        (temp_data_dir / "management").mkdir(exist_ok=True)

        # Create CRAFT_Schema file (5m_*.txt format): CELLID\tSHAREPERCENT
        craft_schema = "CELLID\tSHAREPERCENT\n"
        craft_schema += "100000\t100\n"
        (temp_data_dir / "schema" / "CRAFT_Schema" / "Level2" / "Schema" / "5m_Mali_Koutiala.txt").write_text(craft_schema)

        # Create Python_Schema file (Schema_*.txt format): CellID\tLatitude\tLongitude\tElevation\tArea\tLevel1Name
        python_schema = "CellID\tLatitude\tLongitude\tElevation\tArea\tLevel1Name\tLevel2Name\n"
        python_schema += "100000\t12.00000000\t-5.50000000\t-99.00\t83.123456789012\tMali\tKoutiala\n"
        (temp_data_dir / "schema" / "Python_Schemas" / "Level2" / "Schema_Mali_Koutiala.txt").write_text(python_schema)

        # Create ML.SOL
        soil_content = "*SOIL001  MALI LOAM\n@SITE\n TEST\n"
        (temp_data_dir / "soil" / "ML.SOL").write_text(soil_content)

        return temp_data_dir

    def test_validate_structure(self, valid_craft_structure):
        """Test CRAFT structure validation."""
        validator = CraftValidator(valid_craft_structure)
        issues = validator.validate_structure()
        # Should be valid structure
        structure_errors = [i for i in issues if i.severity == 'error']
        assert len(structure_errors) == 0

    def test_validate_schema_file(self, valid_craft_structure):
        """Test CRAFT schema.txt validation."""
        validator = CraftValidator(valid_craft_structure)
        result = validator.validate()
        # Schema file should be valid
        schema_errors = [i for i in result.issues if i.category == 'schema' and i.severity == 'error']
        assert len(schema_errors) == 0

    def test_cell_id_range(self, temp_data_dir):
        """Test CRAFT cell ID range validation."""
        # Create FULL structure to avoid early return from structure validation
        (temp_data_dir / "schema" / "CRAFT_Schema" / "Level2" / "Schema").mkdir(parents=True, exist_ok=True)
        (temp_data_dir / "weather").mkdir(exist_ok=True)
        (temp_data_dir / "soil").mkdir(exist_ok=True)
        (temp_data_dir / "crop_mask").mkdir(exist_ok=True)
        (temp_data_dir / "management").mkdir(exist_ok=True)

        # Create CRAFT_Schema with invalid cell ID (99999999 > MAX_CELL_ID = 9331200)
        schema_content = "CELLID\tSHAREPERCENT\n"
        schema_content += "99999999\t100\n"
        (temp_data_dir / "schema" / "CRAFT_Schema" / "Level2" / "Schema" / "5m_Mali_Test.txt").write_text(schema_content)

        # Create ML.SOL so structure validation passes
        (temp_data_dir / "soil" / "ML.SOL").write_text("*SOIL001 TEST\n@SITE\n TEST\n")

        validator = CraftValidator(temp_data_dir)
        result = validator.validate()

        # Should have cell ID range error (category is 'schema' in CRAFT validator)
        range_errors = [i for i in result.issues if 'CellID' in i.message and 'range' in i.message]
        assert len(range_errors) > 0

    def test_missing_schema_file(self, temp_data_dir):
        """Test validation catches missing schema.txt."""
        (temp_data_dir / "schema").mkdir(exist_ok=True)
        # Don't create schema.txt

        validator = CraftValidator(temp_data_dir)
        result = validator.validate()

        # Should have structure error for missing schema.txt
        assert result.valid is False


# =============================================================================
# PYTHIA Validator Tests
# =============================================================================

class TestPythiaValidator:
    """Tests for PYTHIA validator."""

    @pytest.fixture
    def valid_pythia_structure(self, temp_data_dir):
        """Create a valid PYTHIA directory structure."""
        # Create required directories
        (temp_data_dir / "shapes").mkdir(exist_ok=True)
        (temp_data_dir / "weather").mkdir(exist_ok=True)
        (temp_data_dir / "raster").mkdir(exist_ok=True)
        (temp_data_dir / "config").mkdir(exist_ok=True)

        # Create JSON config
        config = {
            "name": "test_pythia_config",
            "default_setup": {
                "template": "MAIZE.SNX",
                "sites": "xy_from_vector::shapes/sites.shp",
                "nyers": 5,
            },
            "runs": [
                {"run_name": "run1"}
            ]
        }
        (temp_data_dir / "config" / "pythia_config.json").write_text(json.dumps(config))

        return temp_data_dir

    def test_validate_structure(self, valid_pythia_structure):
        """Test PYTHIA structure validation."""
        validator = PythiaValidator(valid_pythia_structure)
        issues = validator.validate_structure()
        # Should have minimal errors
        structure_errors = [i for i in issues if i.severity == 'error']
        assert len(structure_errors) == 0

    def test_validate_json_config(self, valid_pythia_structure):
        """Test PYTHIA JSON config validation."""
        validator = PythiaValidator(valid_pythia_structure)
        result = validator.validate()

        # JSON config should be valid
        json_errors = [i for i in result.issues if i.category == 'json' and i.severity == 'error']
        assert len(json_errors) == 0

    def test_missing_json_key(self, temp_data_dir):
        """Test validation catches missing JSON key."""
        # Create FULL structure to avoid early return from structure validation
        for subdir in ["shapes", "weather", "raster", "config"]:
            (temp_data_dir / subdir).mkdir(exist_ok=True)

        # Create config missing required keys
        config = {
            "name": "test",
            # Missing default_setup and runs
        }
        (temp_data_dir / "config" / "pythia_config.json").write_text(json.dumps(config))

        validator = PythiaValidator(temp_data_dir)
        result = validator.validate()

        # Should have errors for missing keys
        json_errors = [i for i in result.issues if i.category == 'json' and i.severity == 'error']
        assert len(json_errors) > 0

    def test_pythia_function_syntax(self, valid_pythia_structure):
        """Test PYTHIA function syntax detection."""
        validator = PythiaValidator(valid_pythia_structure)
        result = validator.validate()

        # Should detect xy_from_vector:: syntax
        # Check that no error for missing function syntax
        func_errors = [i for i in result.issues if 'function' in i.message.lower() and i.severity == 'error']
        assert len(func_errors) == 0


# =============================================================================
# ACEA Validator Tests
# =============================================================================

class TestAceaValidator:
    """Tests for ACEA validator."""

    @pytest.fixture
    def valid_acea_structure(self, temp_data_dir):
        """Create a valid ACEA directory structure."""
        # Create required directories
        (temp_data_dir / "climate").mkdir(exist_ok=True)
        (temp_data_dir / "soil").mkdir(exist_ok=True)
        (temp_data_dir / "crop_calendar").mkdir(exist_ok=True)
        (temp_data_dir / "crop_params").mkdir(exist_ok=True)
        (temp_data_dir / "config").mkdir(exist_ok=True)

        # Create Python config
        config_content = """class project_conf:
    project_name = 'test_project'
    crop_model = 'AquaCrop'
    gridcells = [1000, 1001, 1002]
    resolution = 0
    clock_start = '2015-01-01'
    clock_end = '2020-12-31'
    crop_name = 'Maize'
"""
        (temp_data_dir / "config" / "test_config.py").write_text(config_content)

        # Create climate pickle
        tmax = np.array([30.0, 31.0, 32.0], dtype=np.float32)
        tmin = np.array([20.0, 21.0, 22.0], dtype=np.float32)
        prec = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        et0 = np.array([5.0, 5.5, 6.0], dtype=np.float32)

        with open(temp_data_dir / "climate" / "climate_1000.pckl", 'wb') as f:
            pickle.dump((tmax, tmin, prec, et0), f)

        return temp_data_dir

    def test_validate_structure(self, valid_acea_structure):
        """Test ACEA structure validation."""
        validator = AceaValidator(valid_acea_structure)
        issues = validator.validate_structure()
        # Should have minimal errors
        structure_errors = [i for i in issues if i.severity == 'error']
        assert len(structure_errors) == 0

    def test_validate_python_config(self, valid_acea_structure):
        """Test ACEA Python config validation."""
        validator = AceaValidator(valid_acea_structure)
        result = validator.validate()

        # Should be valid
        config_errors = [i for i in result.issues if i.category == 'config' and i.severity == 'error']
        assert len(config_errors) == 0

    def test_crop_model_exact_string(self, temp_data_dir):
        """Test ACEA crop_model exact string check."""
        # Create FULL structure to avoid early return from structure validation
        for subdir in ["climate", "soil", "crop_calendar", "crop_params", "config"]:
            (temp_data_dir / subdir).mkdir(exist_ok=True)

        # Create config with wrong crop_model (must be exactly 'AquaCrop')
        config_content = """class project_conf:
    project_name = 'test'
    crop_model = 'AQUACROP'
    gridcells = [1000]
    resolution = 0
    clock_start = '2015-01-01'
    clock_end = '2020-12-31'
    crop_name = 'Maize'
"""
        (temp_data_dir / "config" / "test_config.py").write_text(config_content)

        validator = AceaValidator(temp_data_dir)
        result = validator.validate()

        # Should have error for incorrect crop_model (validator checks for exact 'AquaCrop')
        crop_model_errors = [i for i in result.issues if 'crop_model' in i.message.lower()]
        assert len(crop_model_errors) > 0

    def test_validate_climate_pickle(self, valid_acea_structure):
        """Test ACEA climate pickle validation."""
        validator = AceaValidator(valid_acea_structure)
        result = validator.validate()

        # Climate pickle should be valid
        climate_errors = [i for i in result.issues if i.category == 'climate' and i.severity == 'error']
        assert len(climate_errors) == 0

    def test_invalid_pickle_format(self, temp_data_dir):
        """Test validation catches invalid pickle format."""
        # Create FULL structure to avoid early return from structure validation
        for subdir in ["climate", "soil", "crop_calendar", "crop_params", "config"]:
            (temp_data_dir / subdir).mkdir(exist_ok=True)

        # Create pickle with wrong format (not a tuple of 4)
        with open(temp_data_dir / "climate" / "climate_1000.pckl", 'wb') as f:
            pickle.dump([1, 2, 3], f)  # Wrong type - should be tuple

        # Need a config file too
        config_content = """class project_conf:
    project_name = 'test'
    crop_model = 'AquaCrop'
    gridcells = [1000]
    resolution = 0
    clock_start = '2015-01-01'
    clock_end = '2020-12-31'
    crop_name = 'Maize'
"""
        (temp_data_dir / "config" / "test_config.py").write_text(config_content)

        validator = AceaValidator(temp_data_dir)
        result = validator.validate()

        # Should have climate error (expected tuple, got list)
        climate_errors = [i for i in result.issues if i.category == 'climate' and i.severity == 'error']
        assert len(climate_errors) > 0

    def test_cell_id_30arcmin(self, temp_data_dir):
        """Test ACEA 30-arcmin cell ID validation."""
        # Create FULL structure to avoid early return from structure validation
        for subdir in ["climate", "soil", "crop_calendar", "crop_params", "config"]:
            (temp_data_dir / subdir).mkdir(exist_ok=True)

        # Create config with cell ID exceeding 30-arcmin max (259,199)
        config_content = """class project_conf:
    project_name = 'test'
    crop_model = 'AquaCrop'
    gridcells = [300000]
    resolution = 0
    clock_start = '2015-01-01'
    clock_end = '2020-12-31'
    crop_name = 'Maize'
"""
        (temp_data_dir / "config" / "test_config.py").write_text(config_content)

        validator = AceaValidator(temp_data_dir)
        result = validator.validate()

        # Should have range error for cell ID (300000 > MAX_30ARCMIN_ID = 259199)
        range_errors = [i for i in result.issues if i.category == 'range']
        assert len(range_errors) > 0
