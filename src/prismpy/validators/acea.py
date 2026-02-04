"""
ACEA output validator for prismpy.

Validates ACEA outputs:
- Python project_conf configuration class
- Climate pickle files (tmax, tmin, prec, et0)
- 30-arcmin cell ID validity
- Soil, crop calendar, and harvested areas data
"""

import ast
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from prismpy.config.schema import Platform
from prismpy.validators.base import (
    BaseValidator,
    ValidationIssue,
    ValidationResult,
)


logger = logging.getLogger(__name__)


class AceaValidator(BaseValidator):
    """Validator for ACEA (AquaCrop) model outputs.

    Validates:
    1. Directory structure (climate/, soil/, crop_calendar/, crop_params/, config/)
    2. project_conf Python configuration class
    3. Climate pickle format: (tmax, tmin, prec, et0)
    4. 30-arcmin cell ID validity (max: 259,199)
    5. Required class attributes
    """

    PLATFORM = Platform.ACEA

    REQUIRED_DIRS = ["climate", "soil", "crop_calendar", "crop_params", "config"]
    REQUIRED_FILES = []

    # ACEA 30-arcmin grid dimensions
    GRID_ROWS = 360
    GRID_COLS = 720
    MAX_30ARCMIN_ID = GRID_ROWS * GRID_COLS - 1  # 259,199

    # Required config class attributes
    REQUIRED_CLASS_ATTRS = [
        "project_name",
        "crop_model",
        "gridcells",
        "resolution",
        "clock_start",
        "clock_end",
        "crop_name",
    ]

    def validate(self) -> ValidationResult:
        """Validate all ACEA outputs.

        Returns:
            ValidationResult with all issues found
        """
        self.logger.info(f"Validating ACEA outputs in {self.output_dir}")
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

        # Collect cell IDs from config
        cell_ids = self._extract_cell_ids_from_config()

        return self.create_result(
            issues,
            files_checked,
            metadata={
                "n_cells": len(cell_ids),
                "cell_id_range": [min(cell_ids), max(cell_ids)] if cell_ids else [0, 0],
                "max_valid_id": self.MAX_30ARCMIN_ID,
            },
        )

    def validate_structure(self) -> List[ValidationIssue]:
        """Validate ACEA output directory structure.

        Returns:
            List of validation issues
        """
        issues = self.validate_required_structure()

        # Check for Python config file
        config_dir = self.output_dir / "config"
        if config_dir.exists():
            py_files = list(config_dir.glob("*_config.py"))
            if not py_files:
                issues.append(ValidationIssue(
                    severity='error',
                    category='structure',
                    message="No *_config.py file found in config/",
                    file_path=config_dir,
                ))

        # Check for climate pickle files
        climate_dir = self.output_dir / "climate"
        if climate_dir.exists():
            pickle_files = list(climate_dir.glob("*.pckl"))
            if not pickle_files:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='structure',
                    message="No climate pickle files found",
                    file_path=climate_dir,
                ))

        return issues

    def validate_files(self) -> List[ValidationIssue]:
        """Validate ACEA file contents.

        Returns:
            List of validation issues
        """
        issues = []

        # Validate Python config
        config_dir = self.output_dir / "config"
        if config_dir.exists():
            for py_file in config_dir.glob("*_config.py"):
                issues.extend(self._validate_python_config(py_file))

        # Validate climate pickles
        climate_dir = self.output_dir / "climate"
        if climate_dir.exists():
            pickle_files = list(climate_dir.glob("*.pckl"))
            for pickle_file in pickle_files[:10]:  # Limit to first 10
                issues.extend(self._validate_climate_pickle(pickle_file))

            if len(pickle_files) > 10:
                issues.append(ValidationIssue(
                    severity='info',
                    category='climate',
                    message=f"Validated 10 of {len(pickle_files)} climate pickles",
                    file_path=climate_dir,
                ))

        # Validate soil CSV
        soil_file = self.output_dir / "soil" / "soil_data.csv"
        if soil_file.exists():
            issues.extend(self._validate_soil_csv(soil_file))

        # Validate crop calendar
        calendar_file = self.output_dir / "crop_calendar" / "calendar.csv"
        if calendar_file.exists():
            issues.extend(self._validate_crop_calendar(calendar_file))

        return issues

    def _validate_python_config(self, file_path: Path) -> List[ValidationIssue]:
        """Validate ACEA Python configuration file.

        Checks:
        - Syntax is valid Python
        - Contains 'class project_conf'
        - crop_model = 'AquaCrop'
        - Required attributes present
        - Cell IDs are 30-arcmin valid

        Args:
            file_path: Path to Python config file

        Returns:
            List of validation issues
        """
        issues = []

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Check for class project_conf
            if "class project_conf" not in content:
                issues.append(ValidationIssue(
                    severity='error',
                    category='config',
                    message="Missing 'class project_conf:' - ACEA requires this exact class name",
                    file_path=file_path,
                ))

            # Check for correct crop_model
            if "crop_model = 'AquaCrop'" not in content:
                issues.append(ValidationIssue(
                    severity='error',
                    category='config',
                    message="crop_model must be exactly 'AquaCrop'",
                    file_path=file_path,
                ))

            # Check for required attributes
            for attr in self.REQUIRED_CLASS_ATTRS:
                pattern = rf"{attr}\s*="
                if not re.search(pattern, content):
                    issues.append(ValidationIssue(
                        severity='error',
                        category='config',
                        message=f"Missing required attribute: {attr}",
                        file_path=file_path,
                    ))

            # Try to parse and validate Python syntax
            try:
                ast.parse(content)
            except SyntaxError as e:
                issues.append(ValidationIssue(
                    severity='error',
                    category='config',
                    message=f"Invalid Python syntax: {e}",
                    file_path=file_path,
                ))
                return issues

            # Extract and validate gridcells
            gridcells_issues = self._validate_gridcells(content, file_path)
            issues.extend(gridcells_issues)

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='config',
                message=f"Error reading config file: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_gridcells(self, content: str, file_path: Path) -> List[ValidationIssue]:
        """Validate gridcells attribute in config.

        Args:
            content: Python file content
            file_path: Path for error context

        Returns:
            List of validation issues
        """
        issues = []

        # Find gridcells assignment
        match = re.search(r"gridcells\s*=\s*(\[[\s\S]*?\])", content)
        if not match:
            issues.append(ValidationIssue(
                severity='warning',
                category='config',
                message="Could not parse gridcells list",
                file_path=file_path,
            ))
            return issues

        try:
            # Parse the list
            gridcells_str = match.group(1)
            # Clean up multiline format
            gridcells_str = re.sub(r'\s+', ' ', gridcells_str)
            cell_ids = ast.literal_eval(gridcells_str)

            if not isinstance(cell_ids, list):
                issues.append(ValidationIssue(
                    severity='error',
                    category='config',
                    message="gridcells must be a list",
                    file_path=file_path,
                ))
                return issues

            if not cell_ids:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='config',
                    message="gridcells list is empty",
                    file_path=file_path,
                ))
                return issues

            # Validate cell IDs are 30-arcmin
            max_id = max(cell_ids)
            if max_id > self.MAX_30ARCMIN_ID:
                issues.append(ValidationIssue(
                    severity='error',
                    category='range',
                    message=f"Cell ID {max_id} exceeds 30-arcmin max ({self.MAX_30ARCMIN_ID}). "
                           "These appear to be 5-arcmin IDs - ACEA requires 30-arcmin!",
                    file_path=file_path,
                    details={'max_id': max_id, 'max_valid': self.MAX_30ARCMIN_ID},
                ))

            # Check for negative IDs
            min_id = min(cell_ids)
            if min_id < 0:
                issues.append(ValidationIssue(
                    severity='error',
                    category='range',
                    message=f"Negative cell ID found: {min_id}",
                    file_path=file_path,
                ))

        except (SyntaxError, ValueError) as e:
            issues.append(ValidationIssue(
                severity='warning',
                category='config',
                message=f"Could not parse gridcells: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_climate_pickle(self, file_path: Path) -> List[ValidationIssue]:
        """Validate ACEA climate pickle file.

        Expected format: tuple of (tmax, tmin, prec, et0)
        Each element should be a numpy float32 array.

        Args:
            file_path: Path to pickle file

        Returns:
            List of validation issues
        """
        issues = []

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Check type
            if not isinstance(data, tuple):
                issues.append(ValidationIssue(
                    severity='error',
                    category='climate',
                    message=f"Expected tuple, got {type(data).__name__}",
                    file_path=file_path,
                ))
                return issues

            # Check length
            if len(data) != 4:
                issues.append(ValidationIssue(
                    severity='error',
                    category='climate',
                    message=f"Expected 4 elements (tmax, tmin, prec, et0), got {len(data)}",
                    file_path=file_path,
                ))
                return issues

            tmax, tmin, prec, et0 = data

            # Check all are arrays
            for arr, name in [(tmax, 'tmax'), (tmin, 'tmin'), (prec, 'prec'), (et0, 'et0')]:
                if not isinstance(arr, np.ndarray):
                    issues.append(ValidationIssue(
                        severity='error',
                        category='climate',
                        message=f"{name} is not numpy array (got {type(arr).__name__})",
                        file_path=file_path,
                    ))

            # Check same length
            lengths = [len(tmax), len(tmin), len(prec), len(et0)]
            if len(set(lengths)) != 1:
                issues.append(ValidationIssue(
                    severity='error',
                    category='climate',
                    message=f"Arrays have different lengths: {lengths}",
                    file_path=file_path,
                ))

            # Check for reasonable values
            if isinstance(tmax, np.ndarray) and len(tmax) > 0:
                tmax_max = float(np.nanmax(tmax))
                tmax_min = float(np.nanmin(tmax))
                if tmax_max > 60 or tmax_min < -50:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='range',
                        message=f"tmax range [{tmax_min:.1f}, {tmax_max:.1f}] seems unusual",
                        file_path=file_path,
                    ))

            # Validate cell ID in filename
            cell_id_match = re.search(r'_(\d+)\.pckl$', file_path.name)
            if cell_id_match:
                cell_id = int(cell_id_match.group(1))
                if cell_id > self.MAX_30ARCMIN_ID:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='range',
                        message=f"Cell ID {cell_id} in filename exceeds 30-arcmin max",
                        file_path=file_path,
                    ))

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='climate',
                message=f"Error reading pickle: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_soil_csv(self, file_path: Path) -> List[ValidationIssue]:
        """Validate soil CSV file.

        Args:
            file_path: Path to soil CSV

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
                    category='soil',
                    message="Soil file is empty",
                    file_path=file_path,
                ))
                return issues

            # Check header
            header = lines[0].strip().lower()
            required = ["cell_id", "sand", "clay"]
            for field in required:
                if field not in header:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='soil',
                        message=f"Missing expected column: {field}",
                        file_path=file_path,
                    ))

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='soil',
                message=f"Error reading soil file: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_crop_calendar(self, file_path: Path) -> List[ValidationIssue]:
        """Validate crop calendar CSV file.

        Args:
            file_path: Path to calendar CSV

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
                    category='calendar',
                    message="Crop calendar file is empty",
                    file_path=file_path,
                ))
                return issues

            # Check header
            header = lines[0].strip().lower()
            if "planting_doy" not in header:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='calendar',
                    message="Missing planting_doy column",
                    file_path=file_path,
                ))

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='calendar',
                message=f"Error reading calendar file: {e}",
                file_path=file_path,
            ))

        return issues

    def _extract_cell_ids_from_config(self) -> Set[int]:
        """Extract cell IDs from Python config file.

        Returns:
            Set of cell IDs
        """
        cell_ids = set()
        config_dir = self.output_dir / "config"

        if not config_dir.exists():
            return cell_ids

        for py_file in config_dir.glob("*_config.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()

                match = re.search(r"gridcells\s*=\s*(\[[\s\S]*?\])", content)
                if match:
                    gridcells_str = re.sub(r'\s+', ' ', match.group(1))
                    ids = ast.literal_eval(gridcells_str)
                    if isinstance(ids, list):
                        cell_ids.update(ids)

            except Exception:
                pass

        return cell_ids

    def _count_files(self) -> int:
        """Count total files in output directory."""
        if not self.output_dir.exists():
            return 0
        return sum(1 for _ in self.output_dir.rglob("*") if _.is_file())
