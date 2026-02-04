"""
ICASA Validator: Validate configuration values against ICASA standards.

This module provides validation of prismpy configuration files against
ICASA (International Consortium for Agricultural Systems Applications) standards,
including range checking, code validation, and required field checking.

Example:
    >>> validator = IcasaValidator()
    >>> result = validator.validate_config({'emergence_gdd': 90.0, 'base_temperature': 8.0})
    >>> print(result.is_valid)
    True
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .icasa_mapper import IcasaMapper, get_mapper

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation messages."""
    ERROR = "error"      # Invalid value, must be fixed
    WARNING = "warning"  # Unusual value, should be reviewed
    INFO = "info"        # Informational message


@dataclass
class ValidationMessage:
    """A single validation message."""
    severity: ValidationSeverity
    parameter: str
    message: str
    value: Any = None
    expected: Any = None

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
        return f"{prefix} {self.parameter}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating a configuration."""
    messages: List[ValidationMessage] = field(default_factory=list)
    parameters_checked: int = 0
    parameters_with_issues: int = 0

    @property
    def is_valid(self) -> bool:
        """Config is valid if there are no errors."""
        return not any(m.severity == ValidationSeverity.ERROR for m in self.messages)

    @property
    def errors(self) -> List[ValidationMessage]:
        """Get all error messages."""
        return [m for m in self.messages if m.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationMessage]:
        """Get all warning messages."""
        return [m for m in self.messages if m.severity == ValidationSeverity.WARNING]

    @property
    def info(self) -> List[ValidationMessage]:
        """Get all info messages."""
        return [m for m in self.messages if m.severity == ValidationSeverity.INFO]

    def add_error(self, param: str, message: str, value: Any = None, expected: Any = None):
        """Add an error message."""
        self.messages.append(ValidationMessage(
            ValidationSeverity.ERROR, param, message, value, expected
        ))
        self.parameters_with_issues += 1

    def add_warning(self, param: str, message: str, value: Any = None, expected: Any = None):
        """Add a warning message."""
        self.messages.append(ValidationMessage(
            ValidationSeverity.WARNING, param, message, value, expected
        ))

    def add_info(self, param: str, message: str, value: Any = None):
        """Add an info message."""
        self.messages.append(ValidationMessage(
            ValidationSeverity.INFO, param, message, value
        ))

    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        self.messages.extend(other.messages)
        self.parameters_checked += other.parameters_checked
        self.parameters_with_issues += other.parameters_with_issues

    def summary(self) -> str:
        """Get a summary string of the validation result."""
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"Validation {status}: "
            f"{len(self.errors)} errors, "
            f"{len(self.warnings)} warnings, "
            f"{self.parameters_checked} parameters checked"
        )

    def __str__(self) -> str:
        lines = [self.summary()]
        for msg in self.messages:
            lines.append(f"  {msg}")
        return "\n".join(lines)


class IcasaValidator:
    """
    Validator for prismpy configurations against ICASA standards.

    Validates:
    - Numeric values against defined ranges (min/max)
    - Coded values against valid code lists
    - Required fields presence
    - Type correctness
    """

    # Parameters that should always be present in a valid base config
    REQUIRED_BASE_PARAMS = {
        'crop': {'name'},
        'phenology': set(),  # All optional but at least one recommended
        'physiology': {'base_temperature'},
        'management': {'planting_density'},
        'temporal': {'start_year', 'end_year'},
    }

    # Parameters where unusual values should trigger warnings
    UNUSUAL_VALUE_THRESHOLDS = {
        'emergence_gdd': (50, 150),       # Typical 60-120
        'grain_filling_gdd': (400, 1000), # Typical 500-800
        'base_temperature': (5, 12),      # Typical 8-10
        'harvest_index': (0.3, 0.55),     # Typical 0.4-0.5
        'planting_density': (30000, 100000),  # plants/ha
    }

    def __init__(self, mapper: Optional[IcasaMapper] = None):
        """
        Initialize the validator.

        Args:
            mapper: ICASA mapper instance. If None, uses default.
        """
        self.mapper = mapper or get_mapper()

    def validate_config(self, config: Dict[str, Any],
                        strict: bool = False,
                        check_required: bool = True) -> ValidationResult:
        """
        Validate a configuration dictionary.

        Args:
            config: Configuration dictionary to validate
            strict: If True, warnings become errors
            check_required: If True, check for required fields

        Returns:
            ValidationResult with all messages
        """
        result = ValidationResult()

        # Normalize config to handle mixed ICASA/human-readable names
        normalized = self.mapper.normalize_config(config)

        # Validate each section
        for section_name, section_data in normalized.items():
            if not isinstance(section_data, dict):
                continue

            section_result = self._validate_section(section_name, section_data, strict)
            result.merge(section_result)

        # Check required fields
        if check_required:
            required_result = self._check_required_fields(normalized)
            result.merge(required_result)

        return result

    def validate_value(self, param_name: str, value: Any,
                       strict: bool = False) -> ValidationResult:
        """
        Validate a single parameter value.

        Args:
            param_name: Parameter name
            value: Value to validate
            strict: If True, warnings become errors

        Returns:
            ValidationResult for this parameter
        """
        result = ValidationResult()
        result.parameters_checked = 1

        # Check range
        range_info = self.mapper.get_range(param_name)
        if range_info and isinstance(value, (int, float)):
            is_valid, message = self.mapper.is_in_range(param_name, value)
            if not is_valid:
                if strict:
                    result.add_error(param_name, message, value)
                else:
                    result.add_warning(param_name, message, value)

        # Check codes
        valid_codes = self.mapper.get_valid_codes(param_name)
        if valid_codes and isinstance(value, str):
            if value not in valid_codes:
                result.add_error(
                    param_name,
                    f"Invalid code '{value}'. Valid codes: {list(valid_codes.keys())}",
                    value,
                    list(valid_codes.keys())
                )

        # Check unusual values (warnings only)
        if param_name in self.UNUSUAL_VALUE_THRESHOLDS:
            low, high = self.UNUSUAL_VALUE_THRESHOLDS[param_name]
            if isinstance(value, (int, float)):
                if value < low or value > high:
                    result.add_warning(
                        param_name,
                        f"Unusual value {value}. Typical range: {low}-{high}",
                        value,
                        (low, high)
                    )

        return result

    def _validate_section(self, section_name: str, section_data: Dict[str, Any],
                          strict: bool) -> ValidationResult:
        """Validate a config section."""
        result = ValidationResult()

        for param_name, value in section_data.items():
            if isinstance(value, dict):
                # Recursively validate nested sections
                nested_result = self._validate_section(
                    f"{section_name}.{param_name}", value, strict
                )
                result.merge(nested_result)
            elif isinstance(value, list):
                # Validate list items
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        nested_result = self._validate_section(
                            f"{section_name}.{param_name}[{i}]", item, strict
                        )
                        result.merge(nested_result)
            else:
                # Validate individual value
                param_result = self.validate_value(param_name, value, strict)
                result.merge(param_result)

        return result

    def _check_required_fields(self, config: Dict[str, Any]) -> ValidationResult:
        """Check for required fields in the config."""
        result = ValidationResult()

        for section_name, required_params in self.REQUIRED_BASE_PARAMS.items():
            section_data = config.get(section_name, {})

            for param in required_params:
                if param not in section_data:
                    result.add_error(
                        f"{section_name}.{param}",
                        f"Required parameter missing"
                    )

        return result

    def validate_base_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate an ICASA-compliant base configuration.

        This is more strict than validate_config() and checks for:
        - All required ICASA fields
        - No platform-specific fields (those belong in DOME)
        - Proper ICASA formatting
        """
        result = self.validate_config(config, strict=True, check_required=True)

        # Check for platform-specific fields that shouldn't be in base config
        platform_specific = self._find_platform_specific_fields(config)
        for field_path in platform_specific:
            result.add_warning(
                field_path,
                "Platform-specific field found in base config. "
                "Consider moving to platform DOME."
            )

        return result

    def validate_dome(self, dome: Dict[str, Any],
                      platform: str) -> ValidationResult:
        """
        Validate a platform DOME configuration.

        Args:
            dome: DOME configuration dictionary
            platform: Target platform (craft, pythia, acea, sarra_py)

        Returns:
            ValidationResult for the DOME
        """
        result = ValidationResult()

        # Check dome_type
        if 'dome_type' not in dome:
            result.add_warning('dome_type', "Missing dome_type field")

        # Check platform
        dome_platform = dome.get('platform')
        if dome_platform and dome_platform != platform:
            result.add_error(
                'platform',
                f"DOME platform '{dome_platform}' doesn't match expected '{platform}'"
            )

        # Platform-specific validation
        if platform == 'craft':
            result.merge(self._validate_craft_dome(dome))
        elif platform == 'pythia':
            result.merge(self._validate_pythia_dome(dome))
        elif platform == 'acea':
            result.merge(self._validate_acea_dome(dome))
        elif platform == 'sarra_py':
            result.merge(self._validate_sarra_py_dome(dome))

        return result

    def _validate_craft_dome(self, dome: Dict[str, Any]) -> ValidationResult:
        """Validate CRAFT-specific DOME fields."""
        result = ValidationResult()

        # Check for recommended CRAFT fields
        cultivar = dome.get('cultivar', {})
        if not cultivar.get('dssat_code'):
            result.add_warning('cultivar.dssat_code', "DSSAT cultivar code recommended")

        data_sources = dome.get('data_sources', {})
        if not data_sources.get('hwsd'):
            result.add_info('data_sources.hwsd', "No HWSD paths specified")

        return result

    def _validate_pythia_dome(self, dome: Dict[str, Any]) -> ValidationResult:
        """Validate Pythia-specific DOME fields."""
        result = ValidationResult()

        data_sources = dome.get('data_sources', {})
        if not data_sources.get('eghr'):
            result.add_warning('data_sources.eghr', "eGHR database paths recommended")

        return result

    def _validate_acea_dome(self, dome: Dict[str, Any]) -> ValidationResult:
        """Validate ACEA-specific DOME fields."""
        result = ValidationResult()
        # ACEA-specific checks can be added here
        return result

    def _validate_sarra_py_dome(self, dome: Dict[str, Any]) -> ValidationResult:
        """Validate SARRA-Py-specific DOME fields."""
        result = ValidationResult()

        climate_sources = dome.get('climate_sources', {})
        if not climate_sources:
            result.add_warning(
                'climate_sources',
                "Climate sources (rainfall, temperature) should be specified"
            )

        return result

    def _find_platform_specific_fields(self, config: Dict[str, Any],
                                       path: str = "") -> List[str]:
        """Find fields that are platform-specific and should be in DOME."""
        platform_fields = []

        # Known platform-specific field patterns
        platform_patterns = {
            'dssat_code', 'cultivar_file', 'hwsd_', 'eghr_', 'spam_',
            'gadm_', 'schema', 'material_code', 'application_code'
        }

        for key, value in config.items():
            full_path = f"{path}.{key}" if path else key

            # Check if key matches platform-specific patterns
            for pattern in platform_patterns:
                if pattern in key.lower():
                    platform_fields.append(full_path)
                    break

            # Recurse into nested dicts
            if isinstance(value, dict):
                platform_fields.extend(
                    self._find_platform_specific_fields(value, full_path)
                )

        return platform_fields


# Convenience function
def validate_config(config: Dict[str, Any], strict: bool = False) -> ValidationResult:
    """Validate a configuration dictionary using default validator."""
    validator = IcasaValidator()
    return validator.validate_config(config, strict=strict)
