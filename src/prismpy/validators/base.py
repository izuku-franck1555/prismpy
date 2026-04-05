"""
Base validator class for prismpy output validation.

This module provides the abstract base class that all platform validators
must inherit from, ensuring consistent validation interface across platforms.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prismpy.config.schema import Platform


logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A single validation issue found during output validation.

    Attributes:
        severity: Issue severity ('error', 'warning', 'info')
        category: Issue category (e.g., 'schema', 'range', 'completeness')
        message: Human-readable description of the issue
        file_path: Path to the file with the issue (if applicable)
        details: Additional details about the issue
    """
    severity: str  # 'error', 'warning', 'info'
    category: str
    message: str
    file_path: Optional[Path] = None
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        prefix = f"[{self.severity.upper()}]"
        if self.file_path:
            return f"{prefix} {self.category}: {self.message} ({self.file_path})"
        return f"{prefix} {self.category}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating platform outputs.

    Attributes:
        valid: Whether all outputs are valid (no errors)
        platform: Platform that was validated
        output_dir: Directory that was validated
        issues: List of all validation issues found
        files_checked: Number of files checked
        metadata: Additional validation metadata
    """
    valid: bool
    platform: Platform
    output_dir: Path
    issues: List[ValidationIssue] = field(default_factory=list)
    files_checked: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get all error-severity issues."""
        return [i for i in self.issues if i.severity == 'error']

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get all warning-severity issues."""
        return [i for i in self.issues if i.severity == 'warning']

    @property
    def n_errors(self) -> int:
        """Count of errors."""
        return len(self.errors)

    @property
    def n_warnings(self) -> int:
        """Count of warnings."""
        return len(self.warnings)

    def summary(self) -> str:
        """Generate a summary string."""
        status = "VALID" if self.valid else "INVALID"
        return (
            f"{self.platform.value} validation: {status}\n"
            f"  Files checked: {self.files_checked}\n"
            f"  Errors: {self.n_errors}\n"
            f"  Warnings: {self.n_warnings}"
        )


class BaseValidator(ABC):
    """Abstract base class for platform output validators.

    Validators check that generated outputs conform to platform requirements:
    - File structure and naming conventions
    - Data schema and format compliance
    - Value ranges and consistency
    - Completeness of required data

    Subclasses must implement:
    - validate(): Main validation entry point
    - validate_structure(): Check directory/file structure
    - validate_files(): Check individual file contents
    """

    # Platform identifier (must be set by subclasses)
    PLATFORM: Platform = None

    # Required output files/directories for this platform
    REQUIRED_FILES: List[str] = []
    REQUIRED_DIRS: List[str] = []

    def __init__(self, output_dir: Union[str, Path]):
        """Initialize the validator.

        Args:
            output_dir: Path to the output directory to validate
        """
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def validate(self) -> ValidationResult:
        """Validate all outputs for this platform.

        Returns:
            ValidationResult with all issues found
        """
        pass

    @abstractmethod
    def validate_structure(self) -> List[ValidationIssue]:
        """Validate output directory structure.

        Checks:
        - Required directories exist
        - Required files exist
        - No unexpected files (optional)

        Returns:
            List of validation issues
        """
        pass

    @abstractmethod
    def validate_files(self) -> List[ValidationIssue]:
        """Validate individual file contents.

        Checks:
        - File format compliance
        - Data schema
        - Value ranges

        Returns:
            List of validation issues
        """
        pass

    def validate_file_types(
        self,
        directory: Path,
        allowed_extensions: List[str],
        dir_label: str = "",
    ) -> List[ValidationIssue]:
        """Check that a directory contains only expected file types.

        Packages should only contain final files the target framework consumes.
        Intermediate files (raw .nc downloads, temp files) must not leak in.

        Args:
            directory: Path to check
            allowed_extensions: List of allowed extensions (e.g., ['.tif', '.tiff'])
            dir_label: Human-readable directory name for messages

        Returns:
            List of validation issues for unexpected files
        """
        issues = []
        if not directory.exists() or not directory.is_dir():
            return issues

        label = dir_label or directory.name
        non_matching = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() not in [e.lower() for e in allowed_extensions]
        ]
        if non_matching:
            extensions = set(f.suffix for f in non_matching)
            issues.append(ValidationIssue(
                severity='warning',
                category='file_type',
                message=(
                    f"{label}/ contains {len(non_matching)} unexpected file(s) "
                    f"(found {', '.join(sorted(extensions))}; "
                    f"expected {', '.join(allowed_extensions)})"
                ),
                file_path=directory,
            ))
        return issues

    def validate_required_structure(self) -> List[ValidationIssue]:
        """Check that required files and directories exist.

        Returns:
            List of validation issues for missing items
        """
        issues = []

        # Check output directory exists
        if not self.output_dir.exists():
            issues.append(ValidationIssue(
                severity='error',
                category='structure',
                message=f"Output directory does not exist: {self.output_dir}",
            ))
            return issues

        # Check required directories
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.output_dir / dir_name
            if not dir_path.exists():
                issues.append(ValidationIssue(
                    severity='error',
                    category='structure',
                    message=f"Required directory missing: {dir_name}",
                    file_path=dir_path,
                ))
            elif not dir_path.is_dir():
                issues.append(ValidationIssue(
                    severity='error',
                    category='structure',
                    message=f"Expected directory but found file: {dir_name}",
                    file_path=dir_path,
                ))

        # Check required files
        for file_name in self.REQUIRED_FILES:
            file_path = self.output_dir / file_name
            if not file_path.exists():
                issues.append(ValidationIssue(
                    severity='error',
                    category='structure',
                    message=f"Required file missing: {file_name}",
                    file_path=file_path,
                ))

        return issues

    def check_file_not_empty(self, file_path: Path) -> Optional[ValidationIssue]:
        """Check that a file is not empty.

        Args:
            file_path: Path to check

        Returns:
            ValidationIssue if empty, None otherwise
        """
        if file_path.exists() and file_path.stat().st_size == 0:
            return ValidationIssue(
                severity='warning',
                category='completeness',
                message=f"File is empty",
                file_path=file_path,
            )
        return None

    def check_value_range(
        self,
        value: float,
        min_val: float,
        max_val: float,
        field_name: str,
        file_path: Optional[Path] = None,
    ) -> Optional[ValidationIssue]:
        """Check that a value is within expected range.

        Args:
            value: Value to check
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            field_name: Name of the field being checked
            file_path: Optional file path for context

        Returns:
            ValidationIssue if out of range, None otherwise
        """
        if value < min_val or value > max_val:
            return ValidationIssue(
                severity='warning',
                category='range',
                message=f"{field_name}={value} outside expected range [{min_val}, {max_val}]",
                file_path=file_path,
                details={'value': value, 'min': min_val, 'max': max_val},
            )
        return None

    def create_result(
        self,
        issues: List[ValidationIssue],
        files_checked: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Create a validation result from issues list.

        Args:
            issues: List of validation issues
            files_checked: Number of files checked
            metadata: Additional metadata

        Returns:
            ValidationResult
        """
        # Valid if no errors (warnings are OK)
        valid = all(i.severity != 'error' for i in issues)

        return ValidationResult(
            valid=valid,
            platform=self.PLATFORM,
            output_dir=self.output_dir,
            issues=issues,
            files_checked=files_checked,
            metadata=metadata or {},
        )
