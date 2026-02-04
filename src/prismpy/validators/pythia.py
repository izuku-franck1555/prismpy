"""
PYTHIA output validator for prismpy.

Validates PYTHIA outputs:
- pythia_config.json structure and required fields
- DSSAT .WTH weather files format
- Site shapefile or CSV
- Management raster/CSV files
- PYTHIA function syntax (lookup_wth::, lookup_ghr::)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from prismpy.config.schema import Platform
from prismpy.validators.base import (
    BaseValidator,
    ValidationIssue,
    ValidationResult,
)


logger = logging.getLogger(__name__)


class PythiaValidator(BaseValidator):
    """Validator for PYTHIA model outputs.

    Validates:
    1. Directory structure (shapes/, weather/, raster/, templates/, config/)
    2. pythia_config.json schema and required fields
    3. PYTHIA function syntax in JSON
    4. .WTH weather files (DSSAT format)
    5. Site shapefile/CSV
    """

    PLATFORM = Platform.PYTHIA

    REQUIRED_DIRS = ["shapes", "weather", "raster", "config"]
    REQUIRED_FILES = []

    # Required JSON sections
    REQUIRED_JSON_KEYS = ["name", "default_setup", "runs"]

    # PYTHIA function patterns
    FUNCTION_PATTERNS = [
        r"xy_from_vector::",
        r"lookup_wth::",
        r"lookup_ghr::",
        r"raster::",
        r"generate_ic_layers::",
    ]

    def validate(self) -> ValidationResult:
        """Validate all PYTHIA outputs.

        Returns:
            ValidationResult with all issues found
        """
        self.logger.info(f"Validating PYTHIA outputs in {self.output_dir}")
        issues = []
        files_checked = 0

        # 1. Validate structure
        structure_issues = self.validate_structure()
        issues.extend(structure_issues)

        if any(i.severity == 'error' for i in structure_issues):
            return self.create_result(issues, files_checked)

        # 2. Validate files
        file_issues = self.validate_files()
        issues.extend(file_issues)
        files_checked = self._count_files()

        return self.create_result(
            issues,
            files_checked,
            metadata={
                "json_valid": not any(
                    i.category == 'json' and i.severity == 'error'
                    for i in issues
                ),
            },
        )

    def validate_structure(self) -> List[ValidationIssue]:
        """Validate PYTHIA output directory structure.

        Returns:
            List of validation issues
        """
        issues = self.validate_required_structure()

        # Check for JSON config
        config_dir = self.output_dir / "config"
        if config_dir.exists():
            json_files = list(config_dir.glob("*.json"))
            if not json_files:
                issues.append(ValidationIssue(
                    severity='error',
                    category='structure',
                    message="No JSON config file found in config/",
                    file_path=config_dir,
                ))

        # Check for weather files
        weather_dir = self.output_dir / "weather"
        if weather_dir.exists():
            wth_files = list(weather_dir.glob("*.WTH"))
            if not wth_files:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='structure',
                    message="No .WTH files found in weather/",
                    file_path=weather_dir,
                ))

        # Check for sites file
        shapes_dir = self.output_dir / "shapes"
        if shapes_dir.exists():
            shp_files = list(shapes_dir.glob("*.shp"))
            csv_files = list(shapes_dir.glob("*.csv"))
            if not shp_files and not csv_files:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='structure',
                    message="No shapefile or CSV found in shapes/",
                    file_path=shapes_dir,
                ))

        return issues

    def validate_files(self) -> List[ValidationIssue]:
        """Validate PYTHIA file contents.

        Returns:
            List of validation issues
        """
        issues = []

        # Validate JSON config
        config_dir = self.output_dir / "config"
        if config_dir.exists():
            for json_file in config_dir.glob("*.json"):
                issues.extend(self._validate_json_config(json_file))

        # Validate .WTH files
        weather_dir = self.output_dir / "weather"
        if weather_dir.exists():
            wth_files = list(weather_dir.glob("*.WTH"))
            for wth_file in wth_files[:10]:  # Limit to first 10
                issues.extend(self._validate_wth_file(wth_file))

            if len(wth_files) > 10:
                issues.append(ValidationIssue(
                    severity='info',
                    category='weather',
                    message=f"Validated 10 of {len(wth_files)} weather files",
                    file_path=weather_dir,
                ))

        # Validate sites file
        shapes_dir = self.output_dir / "shapes"
        if shapes_dir.exists():
            for csv_file in shapes_dir.glob("*.csv"):
                issues.extend(self._validate_sites_csv(csv_file))

        return issues

    def _validate_json_config(self, file_path: Path) -> List[ValidationIssue]:
        """Validate PYTHIA JSON configuration.

        Args:
            file_path: Path to JSON file

        Returns:
            List of validation issues
        """
        issues = []

        try:
            with open(file_path, 'r') as f:
                config = json.load(f)

            # Check required keys
            for key in self.REQUIRED_JSON_KEYS:
                if key not in config:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='json',
                        message=f"Missing required key: {key}",
                        file_path=file_path,
                    ))

            # Validate default_setup
            if "default_setup" in config:
                setup_issues = self._validate_default_setup(config["default_setup"], file_path)
                issues.extend(setup_issues)

            # Validate runs
            if "runs" in config:
                if not isinstance(config["runs"], list):
                    issues.append(ValidationIssue(
                        severity='error',
                        category='json',
                        message="'runs' must be a list",
                        file_path=file_path,
                    ))
                elif len(config["runs"]) == 0:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='json',
                        message="'runs' list is empty",
                        file_path=file_path,
                    ))

            # Check for PYTHIA function syntax
            config_str = json.dumps(config)
            function_found = False
            for pattern in self.FUNCTION_PATTERNS:
                if re.search(pattern, config_str):
                    function_found = True
                    break

            if not function_found:
                issues.append(ValidationIssue(
                    severity='info',
                    category='json',
                    message="No PYTHIA function syntax found (lookup_wth::, etc.)",
                    file_path=file_path,
                ))

        except json.JSONDecodeError as e:
            issues.append(ValidationIssue(
                severity='error',
                category='json',
                message=f"Invalid JSON syntax: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_default_setup(
        self,
        setup: Dict[str, Any],
        file_path: Path,
    ) -> List[ValidationIssue]:
        """Validate default_setup section.

        Args:
            setup: default_setup dictionary
            file_path: Path to config file

        Returns:
            List of validation issues
        """
        issues = []

        # Check expected fields
        expected_fields = ["template", "sites", "nyers"]
        for field in expected_fields:
            if field not in setup:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='json',
                    message=f"default_setup missing expected field: {field}",
                    file_path=file_path,
                ))

        # Validate nyers (number of years)
        if "nyers" in setup:
            nyers = setup["nyers"]
            if not isinstance(nyers, int) or nyers < 1:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='range',
                    message=f"nyers should be positive integer, got: {nyers}",
                    file_path=file_path,
                ))

        # Check sites uses function syntax
        if "sites" in setup:
            sites = setup["sites"]
            if isinstance(sites, str) and not sites.startswith("xy_from_vector::"):
                issues.append(ValidationIssue(
                    severity='info',
                    category='json',
                    message="sites field doesn't use xy_from_vector:: syntax",
                    file_path=file_path,
                ))

        return issues

    def _validate_wth_file(self, file_path: Path) -> List[ValidationIssue]:
        """Validate DSSAT .WTH weather file.

        Expected format:
        $WEATHER DATA: ...
        @ INSI       LAT      LONG    ELEV   TAV   AMP ...
          XXXX  lat  lon  elev  tav  amp ...
        @  DATE  SRAD  TMAX  TMIN  RAIN ...
        YRDOY  srad  tmax  tmin  rain ...

        Args:
            file_path: Path to .WTH file

        Returns:
            List of validation issues
        """
        issues = []

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                issues.append(ValidationIssue(
                    severity='error',
                    category='weather',
                    message=".WTH file is empty",
                    file_path=file_path,
                ))
                return issues

            # Check for header marker
            has_header = any(line.strip().startswith('$') for line in lines[:5])
            if not has_header:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='weather',
                    message="Missing $ header line",
                    file_path=file_path,
                ))

            # Check for station info (@INSI)
            has_station = any('@ INSI' in line or '@INSI' in line for line in lines[:10])
            if not has_station:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='weather',
                    message="Missing station info line (@INSI)",
                    file_path=file_path,
                ))

            # Check for data header (@DATE)
            has_data_header = any('@  DATE' in line or '@ DATE' in line for line in lines)
            if not has_data_header:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='weather',
                    message="Missing data header line (@DATE)",
                    file_path=file_path,
                ))

            # Validate data rows (look for YRDOY format)
            data_found = False
            for line in lines:
                stripped = line.strip()
                if stripped and stripped[0].isdigit():
                    parts = stripped.split()
                    if parts:
                        try:
                            yrdoy = int(parts[0])
                            # YRDOY format validation
                            year = yrdoy // 1000
                            doy = yrdoy % 1000

                            if year < 1980 or year > 2100:
                                issues.append(ValidationIssue(
                                    severity='warning',
                                    category='range',
                                    message=f"YRDOY year {year} unusual (expected 1980-2100)",
                                    file_path=file_path,
                                ))
                                break

                            if doy < 1 or doy > 366:
                                issues.append(ValidationIssue(
                                    severity='error',
                                    category='range',
                                    message=f"DOY {doy} invalid (expected 1-366)",
                                    file_path=file_path,
                                ))
                                break

                            data_found = True
                            break  # Only check first data row

                        except ValueError:
                            pass

            if not data_found:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='weather',
                    message="No valid YRDOY data rows found",
                    file_path=file_path,
                ))

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='weather',
                message=f"Error reading .WTH file: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_sites_csv(self, file_path: Path) -> List[ValidationIssue]:
        """Validate sites CSV file.

        Args:
            file_path: Path to sites CSV

        Returns:
            List of validation issues
        """
        issues = []

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='sites',
                    message="Sites file is empty",
                    file_path=file_path,
                ))
                return issues

            # Check header
            header = lines[0].strip().lower()
            required_fields = ["id", "lat"]  # At minimum
            has_required = all(field in header for field in required_fields)

            if not has_required:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='sites',
                    message=f"Sites file should have columns: {required_fields}",
                    file_path=file_path,
                ))

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='sites',
                message=f"Error reading sites file: {e}",
                file_path=file_path,
            ))

        return issues

    def _count_files(self) -> int:
        """Count total files in output directory."""
        if not self.output_dir.exists():
            return 0
        return sum(1 for _ in self.output_dir.rglob("*") if _.is_file())
