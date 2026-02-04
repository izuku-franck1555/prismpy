"""
ICASA Mapper: Bidirectional mapping between prismpy and ICASA naming conventions.

This module provides conversion between human-readable parameter names used in prismpy
configuration files and ICASA (International Consortium for Agricultural Systems Applications)
standard codes.

Example:
    >>> mapper = IcasaMapper()
    >>> mapper.to_icasa_code('emergence_gdd')
    'P1'
    >>> mapper.from_icasa_code('P1')
    'emergence_gdd'
    >>> mapper.to_icasa_config({'emergence_gdd': 90.0})
    {'P1': 90.0}
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


class IcasaMapper:
    """
    Bidirectional mapper between prismpy names and ICASA codes.

    Supports:
    - Name ↔ code conversion
    - Unit conversions (e.g., plants/ha ↔ plants/m²)
    - Platform-specific mappings (DSSAT, SARRA-Py, AquaCrop, APSIM)
    - Code lookups with descriptions
    """

    def __init__(self, mapping_file: Optional[Path] = None):
        """
        Initialize the ICASA mapper.

        Args:
            mapping_file: Path to icasa_mapping.yaml. If None, uses default location.
        """
        if mapping_file is None:
            mapping_file = Path(__file__).parent / "icasa_mapping.yaml"

        self.mapping_file = mapping_file
        self._mapping: Dict[str, Any] = {}
        self._name_to_icasa: Dict[str, str] = {}
        self._icasa_to_name: Dict[str, str] = {}
        self._conversions: Dict[str, Dict[str, str]] = {}
        self._ranges: Dict[str, Dict[str, float]] = {}
        self._codes: Dict[str, Dict[str, str]] = {}
        self._platform_mappings: Dict[str, Dict[str, str]] = {}

        self._load_mapping()

    def _load_mapping(self) -> None:
        """Load and index the ICASA mapping file."""
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"ICASA mapping file not found: {self.mapping_file}")

        with open(self.mapping_file, 'r') as f:
            self._mapping = yaml.safe_load(f)

        # Build indices for fast lookup
        self._build_indices()

        logger.info(f"Loaded ICASA mapping with {len(self._name_to_icasa)} parameters")

    def _build_indices(self) -> None:
        """Build lookup indices from the mapping data."""
        for section_name, section in self._mapping.items():
            if section_name.startswith('_'):
                continue  # Skip metadata

            if not isinstance(section, dict):
                continue

            for param_name, param_def in section.items():
                if not isinstance(param_def, dict):
                    continue

                icasa_code = param_def.get('icasa_code')
                if icasa_code:
                    # Name → ICASA code
                    full_name = param_name
                    self._name_to_icasa[full_name] = icasa_code

                    # ICASA code → name (first match wins)
                    if icasa_code not in self._icasa_to_name:
                        self._icasa_to_name[icasa_code] = full_name

                    # Store conversion info
                    if 'conversion' in param_def:
                        self._conversions[full_name] = param_def['conversion']

                    # Store range info
                    if 'min' in param_def or 'max' in param_def:
                        self._ranges[full_name] = {
                            'min': param_def.get('min'),
                            'max': param_def.get('max'),
                            'typical': param_def.get('typical_maize') or param_def.get('typical')
                        }

                    # Store valid codes
                    if 'codes' in param_def:
                        self._codes[full_name] = param_def['codes']

                    # Store platform mappings
                    if 'platform_mappings' in param_def:
                        self._platform_mappings[full_name] = param_def['platform_mappings']

    # =========================================================================
    # Core Conversion Methods
    # =========================================================================

    def to_icasa_code(self, name: str) -> Optional[str]:
        """
        Convert a human-readable parameter name to ICASA code.

        Args:
            name: Human-readable name (e.g., 'emergence_gdd')

        Returns:
            ICASA code (e.g., 'P1') or None if not found
        """
        return self._name_to_icasa.get(name)

    def from_icasa_code(self, code: str) -> Optional[str]:
        """
        Convert an ICASA code to human-readable parameter name.

        Args:
            code: ICASA code (e.g., 'P1')

        Returns:
            Human-readable name (e.g., 'emergence_gdd') or None if not found
        """
        return self._icasa_to_name.get(code)

    def to_icasa_config(self, config: Dict[str, Any],
                        apply_conversions: bool = True) -> Dict[str, Any]:
        """
        Convert a config dict from human-readable names to ICASA codes.

        Args:
            config: Config dict with human-readable names
            apply_conversions: Whether to apply unit conversions

        Returns:
            Config dict with ICASA codes as keys
        """
        result = {}

        for key, value in config.items():
            if isinstance(value, dict):
                # Recursively convert nested dicts
                result[key] = self.to_icasa_config(value, apply_conversions)
            else:
                icasa_code = self._name_to_icasa.get(key)
                if icasa_code:
                    # Apply unit conversion if needed
                    if apply_conversions and key in self._conversions:
                        value = self._apply_conversion(value, key, to_icasa=True)
                    result[icasa_code] = value
                else:
                    # Keep original key if no mapping
                    result[key] = value

        return result

    def from_icasa_config(self, config: Dict[str, Any],
                          apply_conversions: bool = True) -> Dict[str, Any]:
        """
        Convert a config dict from ICASA codes to human-readable names.

        Args:
            config: Config dict with ICASA codes
            apply_conversions: Whether to apply unit conversions

        Returns:
            Config dict with human-readable names as keys
        """
        result = {}

        for key, value in config.items():
            if isinstance(value, dict):
                # Recursively convert nested dicts
                result[key] = self.from_icasa_config(value, apply_conversions)
            else:
                human_name = self._icasa_to_name.get(key)
                if human_name:
                    # Apply unit conversion if needed
                    if apply_conversions and human_name in self._conversions:
                        value = self._apply_conversion(value, human_name, to_icasa=False)
                    result[human_name] = value
                else:
                    # Keep original key if no mapping
                    result[key] = value

        return result

    def normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a config to use human-readable names consistently.

        Handles configs that mix ICASA codes and human-readable names.

        Args:
            config: Config dict (may have mixed naming)

        Returns:
            Config dict with all human-readable names
        """
        result = {}

        for key, value in config.items():
            if isinstance(value, dict):
                result[key] = self.normalize_config(value)
            else:
                # Check if key is an ICASA code
                if key in self._icasa_to_name:
                    human_name = self._icasa_to_name[key]
                    result[human_name] = value
                else:
                    result[key] = value

        return result

    # =========================================================================
    # Unit Conversions
    # =========================================================================

    def _apply_conversion(self, value: Any, param_name: str,
                          to_icasa: bool) -> Any:
        """Apply unit conversion for a parameter value."""
        if value is None:
            return None

        conversion = self._conversions.get(param_name, {})

        if to_icasa:
            conv_type = conversion.get('to_icasa')
        else:
            conv_type = conversion.get('from_icasa')

        if not conv_type:
            return value

        try:
            if conv_type == 'divide_by_10000':
                return value / 10000.0
            elif conv_type == 'multiply_by_10000':
                return value * 10000.0
            elif conv_type == 'multiply_by_100':
                return value * 100.0
            elif conv_type == 'divide_by_100':
                return value / 100.0
            else:
                logger.warning(f"Unknown conversion type: {conv_type}")
                return value
        except (TypeError, ValueError):
            return value

    # =========================================================================
    # Platform-Specific Mappings
    # =========================================================================

    def get_platform_code(self, name: str, platform: str) -> Optional[str]:
        """
        Get the platform-specific code for a parameter.

        Args:
            name: Human-readable parameter name
            platform: Platform name (dssat, sarra_py, aquacrop, apsim, wofost)

        Returns:
            Platform-specific code or None
        """
        mappings = self._platform_mappings.get(name, {})
        return mappings.get(platform)

    def to_platform_config(self, config: Dict[str, Any],
                           platform: str) -> Dict[str, Any]:
        """
        Convert config to use platform-specific parameter names.

        Args:
            config: Config dict with human-readable names
            platform: Target platform

        Returns:
            Config dict with platform-specific names
        """
        result = {}

        for key, value in config.items():
            if isinstance(value, dict):
                result[key] = self.to_platform_config(value, platform)
            else:
                platform_code = self.get_platform_code(key, platform)
                if platform_code:
                    result[platform_code] = value
                else:
                    result[key] = value

        return result

    # =========================================================================
    # Lookup Methods
    # =========================================================================

    def get_valid_codes(self, param_name: str) -> Optional[Dict[str, str]]:
        """
        Get valid codes for a coded parameter.

        Args:
            param_name: Parameter name (e.g., 'planting_method')

        Returns:
            Dict of {code: description} or None
        """
        return self._codes.get(param_name)

    def get_range(self, param_name: str) -> Optional[Dict[str, float]]:
        """
        Get valid range for a numeric parameter.

        Args:
            param_name: Parameter name

        Returns:
            Dict with 'min', 'max', 'typical' or None
        """
        return self._ranges.get(param_name)

    def get_description(self, name: str) -> Optional[str]:
        """Get the description for a parameter."""
        for section in self._mapping.values():
            if isinstance(section, dict) and name in section:
                return section[name].get('description')
        return None

    def get_unit(self, name: str) -> Optional[str]:
        """Get the unit for a parameter."""
        for section in self._mapping.values():
            if isinstance(section, dict) and name in section:
                return section[name].get('unit')
        return None

    def list_parameters(self, section: Optional[str] = None) -> List[str]:
        """
        List all mapped parameter names.

        Args:
            section: Optional section filter (e.g., 'phenology', 'management')

        Returns:
            List of parameter names
        """
        if section:
            section_data = self._mapping.get(section, {})
            return [k for k in section_data.keys() if isinstance(section_data.get(k), dict)]
        return list(self._name_to_icasa.keys())

    def list_sections(self) -> List[str]:
        """List all sections in the mapping."""
        return [k for k in self._mapping.keys() if not k.startswith('_')]

    # =========================================================================
    # Validation Support
    # =========================================================================

    def is_valid_code(self, param_name: str, code: str) -> bool:
        """Check if a code is valid for a coded parameter."""
        valid_codes = self._codes.get(param_name)
        if valid_codes is None:
            return True  # No code restriction
        return code in valid_codes

    def is_in_range(self, param_name: str, value: float) -> Tuple[bool, Optional[str]]:
        """
        Check if a value is within the valid range.

        Returns:
            Tuple of (is_valid, message)
        """
        range_info = self._ranges.get(param_name)
        if range_info is None:
            return True, None

        min_val = range_info.get('min')
        max_val = range_info.get('max')

        if min_val is not None and value < min_val:
            return False, f"{param_name}={value} is below minimum {min_val}"

        if max_val is not None and value > max_val:
            return False, f"{param_name}={value} is above maximum {max_val}"

        return True, None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def resolve_alias(self, key: str) -> str:
        """
        Resolve a key to its canonical human-readable name.

        Handles both ICASA codes and human-readable names.
        """
        # If it's an ICASA code, convert to human-readable
        if key in self._icasa_to_name:
            return self._icasa_to_name[key]
        # Otherwise, return as-is (assume it's already human-readable)
        return key

    def get_metadata(self) -> Dict[str, Any]:
        """Get mapping file metadata."""
        return self._mapping.get('_meta', {})


# Module-level convenience instance
_default_mapper: Optional[IcasaMapper] = None


def get_mapper() -> IcasaMapper:
    """Get the default ICASA mapper instance (singleton)."""
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = IcasaMapper()
    return _default_mapper


# Convenience functions
def to_icasa(name: str) -> Optional[str]:
    """Convert human-readable name to ICASA code."""
    return get_mapper().to_icasa_code(name)


def from_icasa(code: str) -> Optional[str]:
    """Convert ICASA code to human-readable name."""
    return get_mapper().from_icasa_code(code)
