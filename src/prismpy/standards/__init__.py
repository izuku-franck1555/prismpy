"""
ICASA Standards Module for prismpy.

This module provides ICASA (International Consortium for Agricultural Systems Applications)
standards support for configuration files, enabling interoperability with the AgMIP ecosystem.

Components:
- icasa_mapping.yaml: Bidirectional mapping between ICASA codes and human-readable names
- IcasaMapper: Class for converting between naming conventions
- IcasaValidator: Class for validating values against ICASA ranges
- AceConverter: Class for AgMIP ACE JSON import/export

References:
- Porter et al. (2014): Harmonization and translation of crop modeling data
- White et al. (2013): ICASA Version 2.0 data standards
- ICASA Dictionary: https://github.com/agmip/ICASA-Dictionary
"""

from pathlib import Path

STANDARDS_DIR = Path(__file__).parent
ICASA_MAPPING_FILE = STANDARDS_DIR / "icasa_mapping.yaml"

# Import main classes
from .icasa_mapper import IcasaMapper, get_mapper, to_icasa, from_icasa
from .icasa_validator import IcasaValidator, ValidationResult, validate_config
from .ace_converter import AceConverter, export_ace, import_ace

__all__ = [
    "STANDARDS_DIR",
    "ICASA_MAPPING_FILE",
    "IcasaMapper",
    "get_mapper",
    "to_icasa",
    "from_icasa",
    "IcasaValidator",
    "ValidationResult",
    "validate_config",
    "AceConverter",
    "export_ace",
    "import_ace",
]
