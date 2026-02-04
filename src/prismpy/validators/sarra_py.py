"""
SARRA-Py output validator for prismpy.

Validates SARRA-Py standardized package outputs:
- Package structure (README.md, manifest.json, provenance.json)
- config/project_config.yaml structure and required fields
- data/boundaries/bounds.json
- data/climate/ subdirectories (rainfall, temperature variables)
- parameters/ YAML files (variety.yaml, itk.yaml, soil.yaml)
- validation/validation_report.json
- Bounding box format [lat_NW, lon_NW, lat_SE, lon_SE]
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from prismpy.config.schema import Platform
from prismpy.validators.base import (
    BaseValidator,
    ValidationIssue,
    ValidationResult,
)


logger = logging.getLogger(__name__)


class SarraPyValidator(BaseValidator):
    """Validator for SARRA-Py model outputs.

    Validates standardized SARRA-Py package:
    1. Package files (README.md, manifest.json, provenance.json)
    2. config/project_config.yaml schema and required fields
    3. data/boundaries/bounds.json with SARRA-Py and GIS formats
    4. data/climate/ subdirectories (rainfall, temperature, ET0, radiation)
    5. parameters/ files (variety.yaml, itk.yaml, soil.yaml)
    6. validation/validation_report.json
    7. Bounding box format compliance [lat_NW, lon_NW, lat_SE, lon_SE]
    """

    PLATFORM = Platform.SARRA_PY

    # Standardized directory structure
    REQUIRED_DIRS = [
        "config",
        "data/boundaries",
        "data/climate/rainfall",
        "data/climate/2m_temperature_24_hour_maximum",
        "data/climate/2m_temperature_24_hour_mean",
        "data/climate/2m_temperature_24_hour_minimum",
        "data/climate/ET0Hargeaves",
        "data/climate/solar_radiation_flux_daily",
        "data/soil",
        "parameters",
        "validation",
    ]

    # Required package files
    REQUIRED_FILES = [
        "README.md",
        "manifest.json",
        "provenance.json",
        "config/project_config.yaml",
        "data/boundaries/bounds.json",
        "parameters/variety.yaml",
        "parameters/itk.yaml",
        "parameters/soil.yaml",
        "validation/validation_report.json",
    ]

    # Required config sections
    REQUIRED_CONFIG_SECTIONS = ["project", "region", "temporal", "crop"]

    # Bounding box expected length
    BBOX_LENGTH = 4

    # Expected variety.yaml parameters (SARRA-Py 52 parameters)
    VARIETY_PARAMS = [
        "TBase", "TOpt1", "TOpt2", "TLim", "SDJLevee", "SDJBVP", "kcMax",
    ]

    # Expected itk.yaml parameters (24 parameters)
    ITK_PARAMS = [
        "DateSemis", "densite", "seuilEauSemis", "irrigAuto",
    ]

    # Expected soil.yaml parameters (13 parameters)
    SOIL_PARAMS = [
        "epaisseurSurf", "epaisseurProf", "ru", "seuilRuiss",
    ]

    def validate(self) -> ValidationResult:
        """Validate all SARRA-Py outputs.

        Returns:
            ValidationResult with all issues found
        """
        self.logger.info(f"Validating SARRA-Py package in {self.output_dir}")
        issues = []
        files_checked = 0

        # 1. Validate package structure
        structure_issues = self.validate_structure()
        issues.extend(structure_issues)

        # If structure is invalid, can't validate files
        if any(i.severity == 'error' for i in structure_issues):
            return self.create_result(issues, files_checked)

        # 2. Validate package files
        package_issues = self._validate_package_files()
        issues.extend(package_issues)

        # 3. Validate file contents
        file_issues = self.validate_files()
        issues.extend(file_issues)
        files_checked = self._count_files()

        return self.create_result(
            issues,
            files_checked,
            metadata={
                "config_valid": not any(
                    i.category == 'config' and i.severity == 'error'
                    for i in issues
                ),
                "package_complete": not any(
                    i.category == 'package' and i.severity == 'error'
                    for i in issues
                ),
            },
        )

    def validate_structure(self) -> List[ValidationIssue]:
        """Validate SARRA-Py standardized package directory structure.

        Returns:
            List of validation issues
        """
        issues = self.validate_required_structure()

        # Check climate subdirectories for GeoTIFF files
        climate_vars = [
            "rainfall",
            "2m_temperature_24_hour_maximum",
            "2m_temperature_24_hour_mean",
            "2m_temperature_24_hour_minimum",
            "ET0Hargeaves",
            "solar_radiation_flux_daily",
        ]

        for var in climate_vars:
            var_dir = self.output_dir / "data" / "climate" / var
            if var_dir.exists():
                # Check for GeoTIFF files
                tif_files = list(var_dir.glob("*.tif")) + list(var_dir.glob("*.tiff"))
                if not tif_files:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='structure',
                        message=f"No GeoTIFF files found in {var}/ (expected .tif)",
                        file_path=var_dir,
                    ))

        return issues

    def _validate_package_files(self) -> List[ValidationIssue]:
        """Validate package metadata files (manifest, provenance, README).

        Returns:
            List of validation issues
        """
        issues = []

        # Check manifest.json
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)

                required_keys = ["package_version", "generator", "files"]
                for key in required_keys:
                    if key not in manifest:
                        issues.append(ValidationIssue(
                            severity='warning',
                            category='package',
                            message=f"manifest.json missing '{key}' field",
                            file_path=manifest_path,
                        ))

            except json.JSONDecodeError as e:
                issues.append(ValidationIssue(
                    severity='error',
                    category='package',
                    message=f"Invalid JSON in manifest.json: {e}",
                    file_path=manifest_path,
                ))

        # Check provenance.json
        provenance_path = self.output_dir / "provenance.json"
        if provenance_path.exists():
            try:
                with open(provenance_path, 'r') as f:
                    provenance = json.load(f)

                if "stages" not in provenance:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='package',
                        message="provenance.json missing 'stages' field",
                        file_path=provenance_path,
                    ))

            except json.JSONDecodeError as e:
                issues.append(ValidationIssue(
                    severity='error',
                    category='package',
                    message=f"Invalid JSON in provenance.json: {e}",
                    file_path=provenance_path,
                ))

        # Check bounds.json
        bounds_path = self.output_dir / "data" / "boundaries" / "bounds.json"
        if bounds_path.exists():
            issues.extend(self._validate_bounds_json(bounds_path))

        return issues

    def _validate_bounds_json(self, file_path: Path) -> List[ValidationIssue]:
        """Validate bounds.json file.

        Args:
            file_path: Path to bounds.json

        Returns:
            List of validation issues
        """
        issues = []

        try:
            with open(file_path, 'r') as f:
                bounds = json.load(f)

            # Check for SARRA-Py format bounds
            if "bounds_sarra_py" not in bounds:
                issues.append(ValidationIssue(
                    severity='error',
                    category='bounds',
                    message="bounds.json missing 'bounds_sarra_py' field",
                    file_path=file_path,
                ))
            else:
                sarra_bounds = bounds["bounds_sarra_py"]
                if len(sarra_bounds) != self.BBOX_LENGTH:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='bounds',
                        message=f"bounds_sarra_py must have {self.BBOX_LENGTH} values",
                        file_path=file_path,
                    ))

            # Check for GIS format bounds
            if "bounds_gis" not in bounds:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='bounds',
                    message="bounds.json missing 'bounds_gis' field",
                    file_path=file_path,
                ))

        except json.JSONDecodeError as e:
            issues.append(ValidationIssue(
                severity='error',
                category='bounds',
                message=f"Invalid JSON in bounds.json: {e}",
                file_path=file_path,
            ))

        return issues

    def validate_files(self) -> List[ValidationIssue]:
        """Validate SARRA-Py file contents.

        Returns:
            List of validation issues
        """
        issues = []

        # Validate config/project_config.yaml
        config_issues = self._validate_config()
        issues.extend(config_issues)

        # Validate standardized parameter files
        params_dir = self.output_dir / "parameters"
        if params_dir.exists():
            # Validate variety.yaml
            variety_path = params_dir / "variety.yaml"
            if variety_path.exists():
                issues.extend(self._validate_variety_yaml(variety_path))

            # Validate itk.yaml
            itk_path = params_dir / "itk.yaml"
            if itk_path.exists():
                issues.extend(self._validate_itk_yaml(itk_path))

            # Validate soil.yaml
            soil_path = params_dir / "soil.yaml"
            if soil_path.exists():
                issues.extend(self._validate_soil_yaml(soil_path))

        return issues

    def _validate_config(self) -> List[ValidationIssue]:
        """Validate config/project_config.yaml file.

        Returns:
            List of validation issues
        """
        issues = []
        config_path = self.output_dir / "config" / "project_config.yaml"

        if not config_path.exists():
            return issues  # Already caught in structure validation

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            if config is None:
                issues.append(ValidationIssue(
                    severity='error',
                    category='config',
                    message="project_config.yaml is empty or invalid",
                    file_path=config_path,
                ))
                return issues

            # Check required sections
            for section in self.REQUIRED_CONFIG_SECTIONS:
                if section not in config:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='config',
                        message=f"Missing required section: {section}",
                        file_path=config_path,
                    ))

            # Validate region section
            if "region" in config:
                region_issues = self._validate_region_config(config["region"], config_path)
                issues.extend(region_issues)

            # Validate temporal section
            if "temporal" in config:
                temporal_issues = self._validate_temporal_config(config["temporal"], config_path)
                issues.extend(temporal_issues)

        except yaml.YAMLError as e:
            issues.append(ValidationIssue(
                severity='error',
                category='config',
                message=f"Invalid YAML syntax: {e}",
                file_path=config_path,
            ))

        return issues

    def _validate_region_config(
        self,
        region: Dict[str, Any],
        config_path: Path,
    ) -> List[ValidationIssue]:
        """Validate region configuration.

        Args:
            region: Region config dictionary
            config_path: Path to config file

        Returns:
            List of validation issues
        """
        issues = []

        # Check required fields
        if "name" not in region:
            issues.append(ValidationIssue(
                severity='error',
                category='config',
                message="region.name is required",
                file_path=config_path,
            ))

        # Validate bounding box format
        if "bounds" in region:
            bounds = region["bounds"]
            if not isinstance(bounds, list):
                issues.append(ValidationIssue(
                    severity='error',
                    category='config',
                    message="region.bounds must be a list",
                    file_path=config_path,
                ))
            elif len(bounds) != self.BBOX_LENGTH:
                issues.append(ValidationIssue(
                    severity='error',
                    category='config',
                    message=f"region.bounds must have {self.BBOX_LENGTH} values [lat_NW, lon_NW, lat_SE, lon_SE]",
                    file_path=config_path,
                    details={'actual_length': len(bounds)},
                ))
            else:
                # Validate coordinate ranges
                lat_nw, lon_nw, lat_se, lon_se = bounds
                if not (-90 <= lat_nw <= 90 and -90 <= lat_se <= 90):
                    issues.append(ValidationIssue(
                        severity='error',
                        category='range',
                        message="Latitude values must be between -90 and 90",
                        file_path=config_path,
                    ))
                if not (-180 <= lon_nw <= 180 and -180 <= lon_se <= 180):
                    issues.append(ValidationIssue(
                        severity='error',
                        category='range',
                        message="Longitude values must be between -180 and 180",
                        file_path=config_path,
                    ))

        return issues

    def _validate_temporal_config(
        self,
        temporal: Dict[str, Any],
        config_path: Path,
    ) -> List[ValidationIssue]:
        """Validate temporal configuration.

        Args:
            temporal: Temporal config dictionary
            config_path: Path to config file

        Returns:
            List of validation issues
        """
        issues = []

        # Check date fields
        if "start_date" not in temporal and "start_year" not in temporal:
            issues.append(ValidationIssue(
                severity='warning',
                category='config',
                message="temporal section should have start_date or start_year",
                file_path=config_path,
            ))

        return issues

    def _validate_variety_yaml(self, file_path: Path) -> List[ValidationIssue]:
        """Validate variety.yaml (SARRA-Py 52 crop parameters).

        Args:
            file_path: Path to variety.yaml

        Returns:
            List of validation issues
        """
        issues = []

        try:
            with open(file_path, 'r') as f:
                params = yaml.safe_load(f)

            if params is None:
                issues.append(ValidationIssue(
                    severity='error',
                    category='params',
                    message="variety.yaml is empty",
                    file_path=file_path,
                ))
                return issues

            # Check for required SARRA-Py parameters
            for param in self.VARIETY_PARAMS:
                if param not in params:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='params',
                        message=f"variety.yaml missing parameter: {param}",
                        file_path=file_path,
                    ))

            # Validate temperature parameters
            if all(p in params for p in ["TBase", "TOpt1", "TOpt2", "TLim"]):
                if not (params["TBase"] < params["TOpt1"] <= params["TOpt2"] < params["TLim"]):
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='params',
                        message="Temperature parameters should follow: TBase < TOpt1 <= TOpt2 < TLim",
                        file_path=file_path,
                    ))

        except yaml.YAMLError as e:
            issues.append(ValidationIssue(
                severity='error',
                category='params',
                message=f"Invalid YAML in variety.yaml: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_itk_yaml(self, file_path: Path) -> List[ValidationIssue]:
        """Validate itk.yaml (SARRA-Py 24 ITK parameters).

        Args:
            file_path: Path to itk.yaml

        Returns:
            List of validation issues
        """
        issues = []

        try:
            with open(file_path, 'r') as f:
                params = yaml.safe_load(f)

            if params is None:
                issues.append(ValidationIssue(
                    severity='error',
                    category='params',
                    message="itk.yaml is empty",
                    file_path=file_path,
                ))
                return issues

            # Check for required ITK parameters
            for param in self.ITK_PARAMS:
                if param not in params:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='params',
                        message=f"itk.yaml missing parameter: {param}",
                        file_path=file_path,
                    ))

            # Validate DateSemis format
            if "DateSemis" in params:
                date_semis = params["DateSemis"]
                if isinstance(date_semis, str):
                    # Check date format (YYYY-M-D or YYYY-MM-DD)
                    parts = date_semis.split("-")
                    if len(parts) != 3:
                        issues.append(ValidationIssue(
                            severity='warning',
                            category='params',
                            message=f"DateSemis format should be YYYY-M-D, got: {date_semis}",
                            file_path=file_path,
                        ))

        except yaml.YAMLError as e:
            issues.append(ValidationIssue(
                severity='error',
                category='params',
                message=f"Invalid YAML in itk.yaml: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_soil_yaml(self, file_path: Path) -> List[ValidationIssue]:
        """Validate soil.yaml (SARRA-Py 13 soil parameters).

        Args:
            file_path: Path to soil.yaml

        Returns:
            List of validation issues
        """
        issues = []

        try:
            with open(file_path, 'r') as f:
                params = yaml.safe_load(f)

            if params is None:
                issues.append(ValidationIssue(
                    severity='error',
                    category='params',
                    message="soil.yaml is empty",
                    file_path=file_path,
                ))
                return issues

            # Check for required soil parameters
            for param in self.SOIL_PARAMS:
                if param not in params:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='params',
                        message=f"soil.yaml missing parameter: {param}",
                        file_path=file_path,
                    ))

            # Validate soil layer depths
            if "epaisseurSurf" in params and "epaisseurProf" in params:
                total_depth = params["epaisseurSurf"] + params["epaisseurProf"]
                if total_depth < 100 or total_depth > 5000:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='params',
                        message=f"Total soil depth ({total_depth}mm) outside typical range (100-5000mm)",
                        file_path=file_path,
                    ))

        except yaml.YAMLError as e:
            issues.append(ValidationIssue(
                severity='error',
                category='params',
                message=f"Invalid YAML in soil.yaml: {e}",
                file_path=file_path,
            ))

        return issues

    def _count_files(self) -> int:
        """Count total files in output directory."""
        if not self.output_dir.exists():
            return 0
        return sum(1 for _ in self.output_dir.rglob("*") if _.is_file())
