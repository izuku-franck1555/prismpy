"""
DOME Merger: Merge base configurations with platform-specific overlays.

This module implements the DOME (Data Overlay for Multi-model Export) pattern
from Porter et al. (2014), allowing separation of generic agronomic data from
platform-specific settings.

Architecture:
    Base Config (ICASA-compliant agronomic data)
         +
    Platform DOME (DSSAT codes, file paths, schema settings)
         =
    Merged Config (ready for platform translator)

Example:
    >>> merger = DomeMerger()
    >>> merged = merger.merge(base_config, craft_dome)
    >>> # merged config has all fields needed for CRAFT translation
"""

import copy
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger(__name__)


class DomeMerger:
    """
    Merges base configurations with platform-specific DOMEs.

    DOME (Data Overlay for Multi-model Export) provides a mechanism for
    separating generic agronomic parameters from platform-specific settings.
    This allows the same base config to be used with multiple platforms.
    """

    # Fields that can only appear in base config
    BASE_ONLY_FIELDS = {
        'crop', 'phenology', 'physiology', 'region', 'temporal',
        'management', 'soil', 'location'
    }

    # Fields that can only appear in DOME
    DOME_ONLY_FIELDS = {
        'dome_type', 'platform', 'cultivar', 'data_sources',
        'schema', 'soil_mask', 'output'
    }

    # Fields that are merged (DOME overrides base)
    MERGED_FIELDS = {
        'fertilizer', 'processing'
    }

    def __init__(self, expand_env_vars: bool = True):
        """
        Initialize the DOME merger.

        Args:
            expand_env_vars: Whether to expand ${VAR} environment variables in DOMEs
        """
        self.expand_env_vars = expand_env_vars

    # =========================================================================
    # Core Merge Operation
    # =========================================================================

    def merge(self, base: Dict[str, Any], dome: Dict[str, Any],
              validate: bool = True) -> Dict[str, Any]:
        """
        Merge a base config with a platform DOME.

        Args:
            base: Base configuration (ICASA-compliant agronomic data)
            dome: Platform DOME (platform-specific settings)
            validate: Whether to validate the merge result

        Returns:
            Merged configuration ready for platform translator

        Raises:
            ValueError: If validation fails
        """
        # Deep copy to avoid modifying originals
        merged = copy.deepcopy(base)
        dome_copy = copy.deepcopy(dome)

        # Expand environment variables in DOME
        if self.expand_env_vars:
            dome_copy = self._expand_env_vars(dome_copy)

        # Validate inputs
        if validate:
            self._validate_base(base)
            self._validate_dome(dome_copy)

        # Merge DOME into base
        merged = self._deep_merge(merged, dome_copy)

        # Add metadata about the merge
        merged['_merge_info'] = {
            'base_source': base.get('_meta', {}).get('source', 'unknown'),
            'dome_platform': dome_copy.get('platform', 'unknown'),
            'dome_type': dome_copy.get('dome_type', 'platform_overlay'),
        }

        # Build platform_config section from DOME
        platform = dome_copy.get('platform', 'unknown')

        # If DOME has explicit platform_config, use it directly (preferred)
        # Otherwise, extract from scattered DOME fields (legacy support)
        if 'platform_config' in dome_copy and platform in dome_copy['platform_config']:
            # Use explicit platform_config from DOME
            logger.info(f"Using explicit platform_config from DOME for {platform}")
            merged['platform_config'] = dome_copy['platform_config']
        else:
            # Extract from DOME fields (backward compatibility)
            logger.info(f"Extracting platform_config from DOME fields for {platform}")
            merged['platform_config'] = {
                platform: self._extract_platform_config(dome_copy, platform)
            }

        return merged

    def _deep_merge(self, base: Dict[str, Any],
                    overlay: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge overlay into base.

        Overlay values override base values. For nested dicts, merge recursively.
        For lists, overlay replaces base (no concatenation).
        """
        result = copy.deepcopy(base)

        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Overlay value replaces base value
                result[key] = copy.deepcopy(value)

        return result

    def _expand_env_vars(self, data: Any) -> Any:
        """Recursively expand ${VAR} environment variable references."""
        if isinstance(data, str):
            # Find all ${VAR} patterns
            pattern = r'\$\{([^}]+)\}'

            def replace_var(match):
                var_name = match.group(1)
                value = os.environ.get(var_name)
                if value is None:
                    logger.warning(f"Environment variable not set: {var_name}")
                    return match.group(0)  # Keep original if not found
                return value

            return re.sub(pattern, replace_var, data)

        elif isinstance(data, dict):
            return {k: self._expand_env_vars(v) for k, v in data.items()}

        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]

        return data

    # =========================================================================
    # Platform Config Extraction
    # =========================================================================

    def _extract_platform_config(self, dome: Dict[str, Any],
                                 platform: str) -> Dict[str, Any]:
        """Extract platform_config section from DOME."""
        config = {'enabled': True}

        # Map DOME fields to platform_config structure
        if platform == 'craft':
            config.update(self._extract_craft_config(dome))
        elif platform == 'pythia':
            config.update(self._extract_pythia_config(dome))
        elif platform == 'acea':
            config.update(self._extract_acea_config(dome))
        elif platform == 'sarra_py':
            config.update(self._extract_sarra_py_config(dome))

        return config

    def _extract_craft_config(self, dome: Dict[str, Any]) -> Dict[str, Any]:
        """Extract CRAFT-specific platform config from DOME."""
        config = {}

        # Cultivar
        cultivar = dome.get('cultivar', {})
        if cultivar.get('dssat_code'):
            config['default_cultivar'] = cultivar['dssat_code']
        if cultivar.get('cultivar_file'):
            config['cultivar_file'] = cultivar['cultivar_file']

        # Fertilizer codes
        fertilizer = dome.get('fertilizer', {})
        if fertilizer.get('material_code'):
            config['fertilizer_material_code'] = fertilizer['material_code']
        if fertilizer.get('application_code'):
            config['fertilizer_application_code'] = fertilizer['application_code']
        if fertilizer.get('depth_cm'):
            config['fertilizer_depth_cm'] = fertilizer['depth_cm']

        # Data sources
        data_sources = dome.get('data_sources', {})

        hwsd = data_sources.get('hwsd', {})
        if hwsd.get('bil_path'):
            config['hwsd_bil_path'] = hwsd['bil_path']
        if hwsd.get('mdb_path'):
            config['hwsd_mdb_path'] = hwsd['mdb_path']

        gadm = data_sources.get('gadm', {})
        if gadm.get('shp_dir'):
            config['gadm_data_path'] = gadm['shp_dir']
        if gadm.get('country_iso3'):
            config['gadm_country_iso3'] = gadm['country_iso3']

        spam = data_sources.get('spam', {})
        if spam.get('raster_path'):
            config['spam_raster_path'] = spam['raster_path']
        config['spam_cap_at_100_percent'] = spam.get('cap_at_100_percent', True)

        # Schema
        schema = dome.get('schema', {})
        if schema.get('level'):
            config['schema_level'] = schema['level']
        if schema.get('admin_names'):
            admin_names = schema['admin_names']
            if isinstance(admin_names, dict):
                config['admin_level1_name'] = admin_names.get('level1')
                config['admin_level2_name'] = admin_names.get('level2')
        config['resolution_arcmin'] = schema.get('resolution_arcmin', 5)

        # Soil mask
        soil_mask = dome.get('soil_mask', {})
        config['include_soil_mask'] = soil_mask.get('enabled', True)
        config['soil_source'] = soil_mask.get('source', 'hwsd')

        # Output
        output = dome.get('output', {})
        config['include_weather_input_csv'] = output.get('include_weather_input_csv', True)

        return config

    def _extract_pythia_config(self, dome: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Pythia-specific platform config from DOME."""
        config = {}

        data_sources = dome.get('data_sources', {})

        # eGHR
        eghr = data_sources.get('eghr', {})
        if eghr.get('raster_path'):
            config['eghr_raster_path'] = eghr['raster_path']
        if eghr.get('database_path'):
            config['eghr_database_path'] = eghr['database_path']
        if eghr.get('sol_dir'):
            config['eghr_sol_dir'] = eghr['sol_dir']

        # SPAM
        spam = data_sources.get('spam', {})
        if spam.get('raster_dir'):
            config['spam_raster_dir'] = spam['raster_dir']
        if spam.get('version'):
            config['spam_version'] = spam['version']

        # Weather
        weather = dome.get('weather', {})
        config['weather_download_delay'] = weather.get('download_delay', 2.0)

        # DSSAT version
        config['dssat_version'] = dome.get('dssat_version', '4.8')

        return config

    def _extract_acea_config(self, dome: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ACEA-specific platform config from DOME."""
        config = {}

        data_sources = dome.get('data_sources', {})

        # HWSD
        hwsd = data_sources.get('hwsd', {})
        if hwsd.get('bil_path'):
            config['hwsd_bil_path'] = hwsd['bil_path']
        if hwsd.get('mdb_path'):
            config['hwsd_mdb_path'] = hwsd['mdb_path']

        # SPAM
        spam = data_sources.get('spam', {})
        if spam.get('data_dir'):
            config['spam_data_dir'] = spam['data_dir']

        # GAEZ
        gaez = data_sources.get('gaez', {})
        if gaez.get('data_dir'):
            config['gaez_data_dir'] = gaez['data_dir']
        config['gaez_auto_download'] = gaez.get('auto_download', True)

        # Settings
        settings = dome.get('settings', {})
        config['compute_et0'] = settings.get('compute_et0', True)
        config['download_climate'] = settings.get('download_climate', True)
        config['resolution'] = settings.get('resolution', '5arcmin')

        return config

    def _extract_sarra_py_config(self, dome: Dict[str, Any]) -> Dict[str, Any]:
        """Extract SARRA-Py-specific platform config from DOME."""
        config = {}

        # Climate sources
        climate_sources = dome.get('climate_sources', {})
        if climate_sources:
            config['climate_sources'] = climate_sources

        # Soil source
        config['soil_source'] = dome.get('soil_source', 'isda')

        # Resolution
        config['resolution'] = dome.get('resolution', 0.0375)

        # Sowing search
        sowing = dome.get('sowing_search', {})
        if sowing.get('month'):
            config['sowing_search_month'] = sowing['month']
        if sowing.get('day'):
            config['sowing_search_day'] = sowing['day']

        # Templates
        templates = dome.get('templates', {})
        if templates.get('variety'):
            config['variety_template'] = templates['variety']
        if templates.get('itk'):
            config['itk_template'] = templates['itk']
        if templates.get('soil'):
            config['soil_template'] = templates['soil']

        return config

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_base(self, base: Dict[str, Any]) -> None:
        """Validate base config structure."""
        # Check for platform-specific fields that shouldn't be in base
        for field in self.DOME_ONLY_FIELDS:
            if field in base:
                logger.warning(
                    f"DOME-only field '{field}' found in base config. "
                    "Consider moving to platform DOME."
                )

    def _validate_dome(self, dome: Dict[str, Any]) -> None:
        """Validate DOME structure."""
        if 'platform' not in dome:
            raise ValueError("DOME must specify 'platform' field")

        valid_platforms = {'craft', 'pythia', 'acea', 'sarra_py'}
        platform = dome.get('platform')
        if platform not in valid_platforms:
            raise ValueError(
                f"Invalid platform '{platform}'. "
                f"Valid platforms: {valid_platforms}"
            )

    # =========================================================================
    # Extraction (Reverse Operation)
    # =========================================================================

    def extract_base(self, merged: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract base config from a merged configuration.

        This removes all platform-specific fields, leaving only
        ICASA-compliant agronomic data.

        Args:
            merged: Merged configuration

        Returns:
            Base configuration (platform-agnostic)
        """
        base = copy.deepcopy(merged)

        # Remove DOME-only fields
        for field in self.DOME_ONLY_FIELDS:
            base.pop(field, None)

        # Remove platform_config
        base.pop('platform_config', None)

        # Remove merge info
        base.pop('_merge_info', None)

        # Remove data_sources paths (these are platform-specific)
        if 'data_sources' in base:
            # Keep only source types, not paths
            data_sources = base['data_sources']
            for key in list(data_sources.keys()):
                if isinstance(data_sources[key], dict):
                    # Remove path-like entries
                    for subkey in list(data_sources[key].keys()):
                        if 'path' in subkey.lower() or 'dir' in subkey.lower():
                            del data_sources[key][subkey]

        return base

    def extract_dome(self, merged: Dict[str, Any],
                     platform: str) -> Dict[str, Any]:
        """
        Extract platform DOME from a merged configuration.

        This extracts all platform-specific settings into a DOME structure.

        Args:
            merged: Merged configuration
            platform: Target platform

        Returns:
            Platform DOME configuration
        """
        dome = {
            'dome_type': 'platform_overlay',
            'platform': platform,
        }

        # Extract platform_config
        platform_config = merged.get('platform_config', {}).get(platform, {})
        if platform_config:
            dome.update(self._platform_config_to_dome(platform_config, platform))

        return dome

    def _platform_config_to_dome(self, config: Dict[str, Any],
                                 platform: str) -> Dict[str, Any]:
        """Convert platform_config back to DOME structure."""
        dome = {}

        if platform == 'craft':
            # Cultivar
            if config.get('default_cultivar'):
                dome['cultivar'] = {'dssat_code': config['default_cultivar']}

            # Data sources
            data_sources = {}
            if config.get('hwsd_bil_path') or config.get('hwsd_mdb_path'):
                data_sources['hwsd'] = {
                    'bil_path': config.get('hwsd_bil_path'),
                    'mdb_path': config.get('hwsd_mdb_path'),
                }
            if config.get('spam_raster_path'):
                data_sources['spam'] = {'raster_path': config['spam_raster_path']}
            if data_sources:
                dome['data_sources'] = data_sources

            # Schema
            if config.get('schema_level'):
                dome['schema'] = {
                    'level': config['schema_level'],
                    'admin_names': {
                        'level1': config.get('admin_level1_name'),
                        'level2': config.get('admin_level2_name'),
                    }
                }

        # Similar extraction for other platforms...

        return dome


# Convenience function
def merge_config(base: Dict[str, Any], dome: Dict[str, Any]) -> Dict[str, Any]:
    """Merge base config with DOME using default merger."""
    merger = DomeMerger()
    return merger.merge(base, dome)
