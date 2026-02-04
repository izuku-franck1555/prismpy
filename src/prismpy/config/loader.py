"""
Configuration loader for prismpy.

This module provides functions to load and save project configurations
from/to YAML files with validation.

Supports two configuration formats:
1. Legacy (single-file): All settings in one YAML file
2. DOME (two-file): Base config (ICASA-compliant) + Platform DOME

Example (Legacy):
    config = load_config('project_config.yaml')

Example (DOME):
    config = load_dome_config('base.yaml', 'craft_dome.yaml')
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from prismpy.config.schema import ProjectConfig
from prismpy.config.dome_merger import DomeMerger

logger = logging.getLogger(__name__)


class ConfigFormat:
    """Detected configuration format."""
    LEGACY = "legacy"  # Single-file format
    DOME = "dome"      # Base + DOME format
    ACE = "ace"        # AgMIP ACE JSON format


def detect_format(config_path: Union[str, Path]) -> str:
    """
    Detect the format of a configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigFormat constant indicating detected format
    """
    config_path = Path(config_path)

    # Check file extension
    if config_path.suffix.lower() == '.json':
        return ConfigFormat.ACE

    # Load and inspect YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    # DOME format indicators
    if raw.get('_meta', {}).get('format') == 'icasa_ace':
        return ConfigFormat.DOME
    if raw.get('dome_type') in ('platform_overlay', 'field_overlay'):
        return ConfigFormat.DOME

    # ACE JSON indicators
    if 'experiments' in raw and 'soils' in raw:
        return ConfigFormat.ACE

    return ConfigFormat.LEGACY


def load_raw_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load raw YAML config without validation.

    Args:
        config_path: Path to YAML file

    Returns:
        Raw dictionary from YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_dome_config(
    base_path: Union[str, Path],
    dome_path: Union[str, Path],
    validate: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load and merge a base config with a platform DOME.

    Args:
        base_path: Path to ICASA-compliant base config
        dome_path: Path to platform DOME config
        validate: Whether to validate the merge

    Returns:
        Tuple of (merged_config, raw_base) for flexibility

    Raises:
        FileNotFoundError: If either file doesn't exist
        ValueError: If DOME validation fails
    """
    base_path = Path(base_path)
    dome_path = Path(dome_path)

    # Load raw configs
    base = load_raw_yaml(base_path)
    dome = load_raw_yaml(dome_path)

    # Add source metadata
    base.setdefault('_meta', {})['source'] = str(base_path)
    dome.setdefault('_meta', {})['source'] = str(dome_path)

    # Merge using DomeMerger
    merger = DomeMerger()
    merged = merger.merge(base, dome, validate=validate)

    logger.info(f"Merged base config '{base_path.name}' with DOME '{dome_path.name}'")
    logger.info(f"Target platform: {dome.get('platform', 'unknown')}")

    return merged, base


def load_auto_config(
    config_path: Union[str, Path],
    dome_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Auto-detect config format and load appropriately.

    Supports:
    - Legacy single-file YAML
    - DOME format (base + dome)
    - ACE JSON format

    Args:
        config_path: Path to config (base config if using DOME)
        dome_path: Optional path to DOME (if using DOME format)

    Returns:
        Configuration dictionary ready for translation
    """
    config_path = Path(config_path)

    # If DOME path provided, use DOME loading
    if dome_path:
        merged, _ = load_dome_config(config_path, dome_path)
        return merged

    # Auto-detect format
    fmt = detect_format(config_path)

    if fmt == ConfigFormat.ACE:
        # Import ACE JSON using AceConverter
        from prismpy.standards import AceConverter
        converter = AceConverter()
        return converter.import_ace_file(config_path)

    elif fmt == ConfigFormat.DOME:
        # This is a base config without DOME - return raw
        logger.warning(
            f"Config '{config_path.name}' is ICASA/DOME format but no DOME provided. "
            "Returning base config only."
        )
        return load_raw_yaml(config_path)

    else:
        # Legacy format - load as-is
        return load_raw_yaml(config_path)


def load_config(config_path: Union[str, Path]) -> ProjectConfig:
    """Load and validate a project configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated ProjectConfig object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        pydantic.ValidationError: If the config fails validation.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    return ProjectConfig.model_validate(raw_config)


def save_config(config: ProjectConfig, output_path: Union[str, Path]) -> None:
    """Save a project configuration to a YAML file.

    Args:
        config: ProjectConfig object to save.
        output_path: Path to save the YAML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, handling Path objects and enums
    config_dict = config.model_dump(mode="json")

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def create_example_config() -> ProjectConfig:
    """Create an example project configuration for reference.

    Returns:
        A fully populated ProjectConfig with example values.
    """
    from datetime import date
    from prismpy.config.schema import (
        ProjectInfo,
        RegionConfig,
        BoundaryConfig,
        BoundarySource,
        CropConfig,
        CropCalendarConfig,
        TemporalConfig,
        Platform,
    )

    return ProjectConfig(
        project=ProjectInfo(
            name="mali_maize_example",
            description="Example maize simulation for Koutiala, Mali",
            version="1.0",
            created=date.today(),
        ),
        region=RegionConfig(
            name="Koutiala",
            country="Mali",
            country_iso3="MLI",
            boundary=BoundaryConfig(
                source=BoundarySource.GADM,
                gadm_level=2,
                gadm_filter_field="NAME_2",
                gadm_filter_value="Koutiala",
            ),
        ),
        crop=CropConfig(
            name="Maize",
            name_short="mai",
            variety="improved_opv",
            calendar=CropCalendarConfig(
                planting_doy=166,  # June 15
                maturity_doy=285,  # October 12
                source="literature",
                reference="Traore et al. (2013)",
            ),
        ),
        temporal=TemporalConfig(
            start_year=2015,
            end_year=2020,
            spinup_years=2,
        ),
        targets=[
            Platform.SARRA_PY,
            Platform.CRAFT,
            Platform.PYTHIA,
            Platform.ACEA,
        ],
    )
