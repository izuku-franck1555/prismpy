"""
CRAFT output validator for prismpy.

Validates CRAFT outputs:
- schema/CRAFT_Schema/Level{N}/Schema/5m_{Admin}.txt - CRAFT database schema
- schema/Python_Schemas/Level{N}/Schema_{Admin}.txt - Python scripts schema
- Weather files with YRDOY dates
- {CC}.SOL soil profile database
- Crop mask and management files
- Cell ID formula: row * 4320 + col
"""

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


class CraftValidator(BaseValidator):
    """Validator for CRAFT model outputs.

    Validates:
    1. Directory structure (schema/, weather/, soil/, crop_mask/, management/)
    2. CRAFT_Schema and Python_Schemas format and cell ID validity
    3. Weather files format (tab-separated, YRDOY dates)
    4. {CC}.SOL DSSAT soil format
    5. Cell ID consistency (5-arcmin global grid)
    """

    PLATFORM = Platform.CRAFT

    REQUIRED_DIRS = ["schema", "weather", "soil", "crop_mask", "management"]
    REQUIRED_FILES = []  # Files are in subdirectories

    # CRAFT global grid dimensions
    GLOBAL_COLS = 4320
    GLOBAL_ROWS = 2160
    MAX_CELL_ID = GLOBAL_ROWS * GLOBAL_COLS - 1  # 9,331,199

    # Weather file expected columns
    WEATHER_COLUMNS = ["YRDOY", "SRAD", "TMAX", "TMIN", "RAIN"]

    def validate(self) -> ValidationResult:
        """Validate all CRAFT outputs.

        Returns:
            ValidationResult with all issues found
        """
        self.logger.info(f"Validating CRAFT outputs in {self.output_dir}")
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

        # Collect cell IDs for consistency check
        cell_ids = self._collect_cell_ids()

        return self.create_result(
            issues,
            files_checked,
            metadata={
                "n_cells": len(cell_ids),
                "cell_id_range": [min(cell_ids), max(cell_ids)] if cell_ids else [0, 0],
            },
        )

    def validate_structure(self) -> List[ValidationIssue]:
        """Validate CRAFT output directory structure.

        Returns:
            List of validation issues
        """
        issues = self.validate_required_structure()

        # Check for CRAFT_Schema (new structure)
        craft_schema_dir = self.output_dir / "schema" / "CRAFT_Schema"
        craft_schema_files = list(craft_schema_dir.glob("**/5m_*.txt")) if craft_schema_dir.exists() else []
        if not craft_schema_files:
            issues.append(ValidationIssue(
                severity='error',
                category='structure',
                message="Missing schema/CRAFT_Schema/Level*/Schema/5m_*.txt",
                file_path=craft_schema_dir,
            ))

        # Check for Python_Schemas
        python_schema_dir = self.output_dir / "schema" / "Python_Schemas"
        python_schema_files = list(python_schema_dir.glob("**/Schema_*.txt")) if python_schema_dir.exists() else []
        if not python_schema_files:
            issues.append(ValidationIssue(
                severity='warning',
                category='structure',
                message="Missing schema/Python_Schemas/Level*/Schema_*.txt",
                file_path=python_schema_dir,
            ))

        # Check for soil file (any .SOL file)
        soil_dir = self.output_dir / "soil"
        sol_files = list(soil_dir.glob("*.SOL")) if soil_dir.exists() else []
        if not sol_files:
            issues.append(ValidationIssue(
                severity='error',
                category='structure',
                message="Missing soil/*.SOL file",
                file_path=soil_dir,
            ))

        # Check weather directory has input.csv (CRAFT downloads weather at runtime)
        weather_dir = self.output_dir / "weather"
        if weather_dir.exists():
            input_csv = weather_dir / "input.csv"
            if not input_csv.exists():
                issues.append(ValidationIssue(
                    severity='warning',
                    category='structure',
                    message="Missing weather/input.csv for CRAFT weather download",
                    file_path=weather_dir,
                ))

        # Validate file types in key directories
        issues.extend(self.validate_file_types(
            self.output_dir / "weather", ['.csv'], dir_label="weather"
        ))
        issues.extend(self.validate_file_types(
            self.output_dir / "soil", ['.SOL', '.sol', '.txt'], dir_label="soil"
        ))
        issues.extend(self.validate_file_types(
            self.output_dir / "crop_mask", ['.txt'], dir_label="crop_mask"
        ))
        issues.extend(self.validate_file_types(
            self.output_dir / "management", ['.txt'], dir_label="management"
        ))

        return issues

    def validate_files(self) -> List[ValidationIssue]:
        """Validate CRAFT file contents.

        Returns:
            List of validation issues
        """
        issues = []

        # Validate CRAFT_Schema files
        craft_schema_dir = self.output_dir / "schema" / "CRAFT_Schema"
        if craft_schema_dir.exists():
            for schema_file in craft_schema_dir.glob("**/5m_*.txt"):
                issues.extend(self._validate_craft_schema(schema_file))

        # Validate Python_Schemas files
        python_schema_dir = self.output_dir / "schema" / "Python_Schemas"
        if python_schema_dir.exists():
            for schema_file in python_schema_dir.glob("**/Schema_*.txt"):
                issues.extend(self._validate_python_schema(schema_file))

        # Validate weather files
        weather_dir = self.output_dir / "weather"
        if weather_dir.exists():
            for weather_file in weather_dir.glob("*.txt"):
                issues.extend(self._validate_weather_file(weather_file))

        # Validate soil files (any .SOL file)
        soil_dir = self.output_dir / "soil"
        if soil_dir.exists():
            for sol_file in soil_dir.glob("*.SOL"):
                issues.extend(self._validate_sol_file(sol_file))

        # Validate crop mask
        mask_file = self.output_dir / "crop_mask" / "mask.txt"
        if mask_file.exists():
            issues.extend(self._validate_crop_mask(mask_file))

        return issues

    def _validate_schema(self, file_path: Path) -> List[ValidationIssue]:
        """Validate schema.txt file.

        Expected format:
        CellID\tLat\tLon\tRow\tCol\tAdminName

        Args:
            file_path: Path to schema.txt

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
                    category='schema',
                    message="schema.txt is empty",
                    file_path=file_path,
                ))
                return issues

            # Check header
            header = lines[0].strip().split('\t')
            required_cols = ["CellID", "Lat", "Lon"]
            for col in required_cols:
                if col not in header:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f"Missing required column: {col}",
                        file_path=file_path,
                    ))

            if any(i.severity == 'error' for i in issues):
                return issues

            # Validate data rows
            cell_id_idx = header.index("CellID")
            lat_idx = header.index("Lat")
            lon_idx = header.index("Lon")

            cell_ids = set()
            for i, line in enumerate(lines[1:], start=2):
                parts = line.strip().split('\t')
                if len(parts) < len(required_cols):
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='schema',
                        message=f"Line {i} has insufficient columns",
                        file_path=file_path,
                    ))
                    continue

                try:
                    cell_id = int(parts[cell_id_idx])
                    lat = float(parts[lat_idx])
                    lon = float(parts[lon_idx])

                    # Validate cell ID range
                    if cell_id < 0 or cell_id > self.MAX_CELL_ID:
                        issues.append(ValidationIssue(
                            severity='error',
                            category='range',
                            message=f"Cell ID {cell_id} out of range [0, {self.MAX_CELL_ID}]",
                            file_path=file_path,
                            details={'line': i, 'cell_id': cell_id},
                        ))

                    # Check for duplicate cell IDs
                    if cell_id in cell_ids:
                        issues.append(ValidationIssue(
                            severity='warning',
                            category='schema',
                            message=f"Duplicate cell ID: {cell_id}",
                            file_path=file_path,
                            details={'line': i},
                        ))
                    cell_ids.add(cell_id)

                    # Validate coordinates
                    if not (-90 <= lat <= 90):
                        issues.append(ValidationIssue(
                            severity='error',
                            category='range',
                            message=f"Latitude {lat} out of range [-90, 90]",
                            file_path=file_path,
                            details={'line': i},
                        ))
                    if not (-180 <= lon <= 180):
                        issues.append(ValidationIssue(
                            severity='error',
                            category='range',
                            message=f"Longitude {lon} out of range [-180, 180]",
                            file_path=file_path,
                            details={'line': i},
                        ))

                except ValueError as e:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f"Invalid value on line {i}: {e}",
                        file_path=file_path,
                    ))

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='schema',
                message=f"Error reading schema.txt: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_weather_file(self, file_path: Path) -> List[ValidationIssue]:
        """Validate a weather file.

        Expected format:
        YRDOY\tSRAD\tTMAX\tTMIN\tRAIN

        Args:
            file_path: Path to weather file

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
                    message="Weather file is empty",
                    file_path=file_path,
                ))
                return issues

            # Check header
            header = lines[0].strip().split('\t')
            for col in self.WEATHER_COLUMNS:
                if col not in header:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='weather',
                        message=f"Missing expected column: {col}",
                        file_path=file_path,
                    ))

            # Validate a sample of data rows
            yrdoy_idx = header.index("YRDOY") if "YRDOY" in header else 0
            for i, line in enumerate(lines[1:min(11, len(lines))], start=2):
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                try:
                    yrdoy = int(parts[yrdoy_idx])
                    # YRDOY format: YYYYDDD (e.g., 2015001)
                    year = yrdoy // 1000
                    doy = yrdoy % 1000

                    if year < 1980 or year > 2100:
                        issues.append(ValidationIssue(
                            severity='warning',
                            category='range',
                            message=f"YRDOY year {year} unusual",
                            file_path=file_path,
                            details={'line': i, 'yrdoy': yrdoy},
                        ))
                        break  # Only report once per file

                    if doy < 1 or doy > 366:
                        issues.append(ValidationIssue(
                            severity='error',
                            category='range',
                            message=f"DOY {doy} out of range [1, 366]",
                            file_path=file_path,
                            details={'line': i, 'yrdoy': yrdoy},
                        ))
                        break

                except ValueError:
                    pass  # Skip non-numeric

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='weather',
                message=f"Error reading weather file: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_ml_sol(self, file_path: Path) -> List[ValidationIssue]:
        """Validate ML.SOL DSSAT soil file.

        Args:
            file_path: Path to ML.SOL

        Returns:
            List of validation issues
        """
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                issues.append(ValidationIssue(
                    severity='error',
                    category='soil',
                    message="ML.SOL is empty",
                    file_path=file_path,
                ))
                return issues

            # Check for profile markers
            profile_count = content.count('*')
            if profile_count == 0:
                issues.append(ValidationIssue(
                    severity='error',
                    category='soil',
                    message="No soil profiles found (missing * markers)",
                    file_path=file_path,
                ))
            else:
                # Check for layer data markers
                if '@' not in content:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='soil',
                        message="No layer headers found (missing @ markers)",
                        file_path=file_path,
                    ))

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='soil',
                message=f"Error reading ML.SOL: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_crop_mask(self, file_path: Path) -> List[ValidationIssue]:
        """Validate crop mask file.

        Args:
            file_path: Path to mask.txt

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
                    category='mask',
                    message="Crop mask file is empty",
                    file_path=file_path,
                ))
                return issues

            # Check header
            header = lines[0].strip().split('\t')
            if "CellId" not in header:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='mask',
                    message="Missing CellId column in crop mask",
                    file_path=file_path,
                ))

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='mask',
                message=f"Error reading crop mask: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_craft_schema(self, file_path: Path) -> List[ValidationIssue]:
        """Validate CRAFT_Schema file (5m_*.txt format).

        Expected format: CELLID\\tSHAREPERCENT (tab-separated, 2 columns)
        - CELLID: integer, 1-indexed global grid cell ID
        - SHAREPERCENT: 0-100, percentage of cell covered by admin boundary

        Args:
            file_path: Path to CRAFT schema file

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
                    category='schema',
                    message="CRAFT schema file is empty",
                    file_path=file_path,
                ))
                return issues

            header = lines[0].strip().split('\t')

            # CRAFT schema uses CELLID and SHAREPERCENT (uppercase)
            required_cols = ["CELLID", "SHAREPERCENT"]
            for col in required_cols:
                if col not in header:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f"Missing required column: {col}",
                        file_path=file_path,
                    ))

            if any(i.severity == 'error' for i in issues):
                return issues

            cellid_idx = header.index("CELLID")
            share_idx = header.index("SHAREPERCENT")

            prev_cellid = -1
            for i, line in enumerate(lines[1:], start=2):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f"Line {i}: expected 2 columns, got {len(parts)}",
                        file_path=file_path,
                    ))
                    continue

                try:
                    cellid = int(parts[cellid_idx])
                    if cellid < 1 or cellid > 9331200:
                        issues.append(ValidationIssue(
                            severity='error',
                            category='schema',
                            message=f"Line {i}: CellID {cellid} out of valid range [1, 9331200]",
                            file_path=file_path,
                        ))
                    if cellid <= prev_cellid:
                        issues.append(ValidationIssue(
                            severity='warning',
                            category='schema',
                            message=f"Line {i}: CellID {cellid} not in ascending order",
                            file_path=file_path,
                        ))
                    prev_cellid = cellid
                except ValueError:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f"Line {i}: invalid CellID '{parts[cellid_idx]}'",
                        file_path=file_path,
                    ))

                try:
                    share = float(parts[share_idx])
                    if share < 0 or share > 100:
                        issues.append(ValidationIssue(
                            severity='error',
                            category='schema',
                            message=f"Line {i}: SharePercent {share} out of range [0, 100]",
                            file_path=file_path,
                        ))
                except ValueError:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f"Line {i}: invalid SharePercent '{parts[share_idx]}'",
                        file_path=file_path,
                    ))

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='schema',
                message=f"Failed to read CRAFT schema: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_python_schema(self, file_path: Path) -> List[ValidationIssue]:
        """Validate Python_Schema file (Schema_*.txt format).

        Expected format: CellID\\tLatitude\\tLongitude\\tElevation\\tArea\\tLevel{N}Name
        (tab-separated, 5+ columns)

        Args:
            file_path: Path to Python schema file

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
                    category='schema',
                    message="Python schema file is empty",
                    file_path=file_path,
                ))
                return issues

            header = lines[0].strip().split('\t')

            # Python schema uses CellID, Latitude, Longitude (not Lat/Lon)
            required_cols = ["CellID", "Latitude", "Longitude", "Area"]
            for col in required_cols:
                if col not in header:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f"Missing required column: {col}",
                        file_path=file_path,
                    ))

            if any(i.severity == 'error' for i in issues):
                return issues

            cellid_idx = header.index("CellID")
            lat_idx = header.index("Latitude")
            lon_idx = header.index("Longitude")

            for i, line in enumerate(lines[1:], start=2):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < len(required_cols):
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f"Line {i}: expected at least {len(required_cols)} columns, got {len(parts)}",
                        file_path=file_path,
                    ))
                    continue

                try:
                    cellid = int(parts[cellid_idx])
                    if cellid < 1 or cellid > 9331200:
                        issues.append(ValidationIssue(
                            severity='error',
                            category='schema',
                            message=f"Line {i}: CellID {cellid} out of valid range",
                            file_path=file_path,
                        ))
                except ValueError:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f"Line {i}: invalid CellID '{parts[cellid_idx]}'",
                        file_path=file_path,
                    ))

                try:
                    lat = float(parts[lat_idx])
                    if lat < -90 or lat > 90:
                        issues.append(ValidationIssue(
                            severity='error',
                            category='schema',
                            message=f"Line {i}: Latitude {lat} out of range [-90, 90]",
                            file_path=file_path,
                        ))
                except ValueError:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f"Line {i}: invalid Latitude",
                        file_path=file_path,
                    ))

                try:
                    lon = float(parts[lon_idx])
                    if lon < -180 or lon > 180:
                        issues.append(ValidationIssue(
                            severity='error',
                            category='schema',
                            message=f"Line {i}: Longitude {lon} out of range [-180, 180]",
                            file_path=file_path,
                        ))
                except ValueError:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='schema',
                        message=f"Line {i}: invalid Longitude",
                        file_path=file_path,
                    ))

        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                category='schema',
                message=f"Failed to read Python schema: {e}",
                file_path=file_path,
            ))

        return issues

    def _validate_sol_file(self, file_path: Path) -> List[ValidationIssue]:
        """Validate DSSAT soil file (*.SOL format).

        Args:
            file_path: Path to soil file

        Returns:
            List of validation issues
        """
        return self._validate_ml_sol(file_path)

    def _collect_cell_ids(self) -> Set[int]:
        """Collect all cell IDs from schema file.

        Returns:
            Set of cell IDs
        """
        cell_ids = set()
        schema_file = self.output_dir / "schema" / "schema.txt"

        if not schema_file.exists():
            return cell_ids

        try:
            with open(schema_file, 'r') as f:
                lines = f.readlines()

            if len(lines) < 2:
                return cell_ids

            header = lines[0].strip().split('\t')
            if "CellID" not in header:
                return cell_ids

            cell_id_idx = header.index("CellID")
            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) > cell_id_idx:
                    try:
                        cell_ids.add(int(parts[cell_id_idx]))
                    except ValueError:
                        pass

        except Exception:
            pass

        return cell_ids

    def _count_files(self) -> int:
        """Count total files in output directory."""
        if not self.output_dir.exists():
            return 0
        return sum(1 for _ in self.output_dir.rglob("*") if _.is_file())
