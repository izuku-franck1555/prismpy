"""
ACE Converter: Import/Export AgMIP Crop Experiment (ACE) JSON format.

This module provides conversion between prismpy configuration files and
the AgMIP ACE (AgMIP Crop Experiment) JSON format, enabling data exchange with
the broader AgMIP ecosystem.

ACE Format Reference:
- Porter et al. (2014): Harmonization and translation of crop modeling data
- AgMIP Data Interoperability: http://research.agmip.org/display/it/Data+Interoperability

Example:
    >>> converter = AceConverter()
    >>> ace_json = converter.export_ace(base_config)
    >>> base_config = converter.import_ace(ace_json)
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .icasa_mapper import IcasaMapper, get_mapper

logger = logging.getLogger(__name__)


class AceConverter:
    """
    Converter between prismpy configs and AgMIP ACE JSON format.

    ACE (AgMIP Crop Experiment) is a JSON-based format for storing agricultural
    experiment data using ICASA variable names. This converter enables:
    - Export: Convert prismpy base configs to ACE JSON
    - Import: Convert ACE JSON datasets to prismpy configs
    """

    # ACE JSON structure version
    ACE_VERSION = "1.1"

    def __init__(self, mapper: Optional[IcasaMapper] = None):
        """
        Initialize the ACE converter.

        Args:
            mapper: ICASA mapper instance. If None, uses default.
        """
        self.mapper = mapper or get_mapper()

    # =========================================================================
    # Export: prismpy → ACE JSON
    # =========================================================================

    def export_ace(self, config: Dict[str, Any],
                   include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export a prismpy base config to ACE JSON format.

        Args:
            config: prismpy configuration dictionary
            include_metadata: Whether to include export metadata

        Returns:
            ACE-formatted JSON dictionary
        """
        ace = {
            "experiments": [],
            "soils": [],
            "weathers": []
        }

        # Create experiment entry
        experiment = self._build_experiment(config)
        ace["experiments"].append(experiment)

        # Create soil entry if present
        soil = self._build_soil(config)
        if soil:
            ace["soils"].append(soil)
            # Link soil to experiment
            experiment["soil_id"] = soil["soil_id"]

        # Add metadata
        if include_metadata:
            ace["_meta"] = {
                "format": "ACE",
                "version": self.ACE_VERSION,
                "generator": "prismpy",
                "generated_at": datetime.now().isoformat(),
                "source_format": "prismpy_base_config"
            }

        return ace

    def _build_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build ACE experiment structure from config."""
        experiment = {}

        # Generate experiment ID from config hash
        config_str = json.dumps(config, sort_keys=True)
        exp_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        experiment["exname"] = config.get("project", {}).get("name", f"EXP_{exp_hash}")
        experiment["exp_id"] = f"CT_{exp_hash}"

        # Location
        region = config.get("region", {})
        location = config.get("location", {})
        experiment["fl_lat"] = location.get("latitude") or region.get("centroid_lat")
        experiment["fl_long"] = location.get("longitude") or region.get("centroid_lon")
        experiment["fl_loc_1"] = region.get("name", "")
        experiment["fl_loc_2"] = region.get("country", "")

        # Crop identification
        crop = config.get("crop", {})
        experiment["crid"] = self._get_crop_code(crop.get("name", ""))
        experiment["cul_name"] = crop.get("variety", "")

        # Management
        management = config.get("management", {})
        experiment["management"] = self._build_management(management, config)

        # Initial conditions (phenology/physiology as metadata)
        experiment["initial_conditions"] = self._build_initial_conditions(config)

        # Temporal
        temporal = config.get("temporal", {})
        experiment["sc_year"] = temporal.get("start_year")
        experiment["endsim"] = temporal.get("end_year")

        return experiment

    def _build_management(self, management: Dict[str, Any],
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """Build ACE management structure."""
        mgmt = {}

        # Planting
        planting = {}
        planting["pdate"] = management.get("planting_doy") or management.get("planting_date")

        # Convert planting density (plants/ha → plants/m²)
        density = management.get("planting_density")
        if density:
            planting["ppop"] = density / 10000.0 if density > 100 else density

        planting["plrs"] = management.get("row_spacing_cm")
        planting["pldp"] = management.get("planting_depth_cm", 5)
        planting["plme"] = management.get("planting_method", "S")
        planting["plds"] = management.get("plant_distribution", "R")

        mgmt["planting"] = [planting]

        # Fertilizer
        fertilizer_n = management.get("fertilizer_n_total")
        if fertilizer_n:
            fertilizer_events = self._build_fertilizer_events(management, fertilizer_n)
            mgmt["fertilizer"] = fertilizer_events

        # Irrigation
        if management.get("irrigation"):
            mgmt["irrigation"] = [{
                "irrig": "Y",
                "irop": management.get("irrigation_method", "IR001")
            }]
        else:
            mgmt["irrigation"] = [{"irrig": "N"}]

        return mgmt

    def _build_fertilizer_events(self, management: Dict[str, Any],
                                 total_n: float) -> List[Dict[str, Any]]:
        """Build ACE fertilizer event list."""
        events = []

        # Get splits
        splits = management.get("fertilizer", {}).get("splits", [])
        if not splits:
            # Default: single application at planting
            splits = [{"dap": 0, "fraction": 1.0}]

        n_splits = management.get("fertilizer_n_splits", [])
        n_fractions = management.get("fertilizer_n_fractions", [])

        if n_splits and n_fractions:
            splits = [
                {"dap": dap, "fraction": frac}
                for dap, frac in zip(n_splits, n_fractions)
            ]

        for i, split in enumerate(splits):
            event = {
                "fday": split.get("dap", 0),
                "fecd": management.get("fertilizer_material_code", "FE005"),
                "feacd": management.get("fertilizer_application_code", "AP002"),
                "feamn": total_n * split.get("fraction", 1.0),
                "fedep": management.get("fertilizer_depth_cm", 5)
            }

            # Add P and K if present
            p_total = management.get("fertilizer_p_total")
            if p_total:
                event["feamp"] = p_total * split.get("fraction", 1.0)

            k_total = management.get("fertilizer_k_total")
            if k_total:
                event["feamk"] = k_total * split.get("fraction", 1.0)

            events.append(event)

        return events

    def _build_initial_conditions(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build ACE initial conditions (includes phenology/physiology params)."""
        ic = {}

        # Phenology parameters (stored as metadata in ACE)
        phenology = config.get("phenology", {})
        if phenology:
            ic["phenology"] = {
                "p1": phenology.get("emergence_gdd") or phenology.get("P1"),
                "p2": phenology.get("photoperiod_sensitivity") or phenology.get("P2"),
                "p5": phenology.get("grain_filling_gdd") or phenology.get("P5"),
                "phint": phenology.get("phyllochron") or phenology.get("PHINT"),
            }
            # Remove None values
            ic["phenology"] = {k: v for k, v in ic["phenology"].items() if v is not None}

        # Physiology parameters
        physiology = config.get("physiology", {})
        if physiology:
            ic["physiology"] = {
                "tb": physiology.get("base_temperature") or physiology.get("TB"),
                "to": physiology.get("optimal_temperature") or physiology.get("TO"),
                "tm": physiology.get("max_temperature") or physiology.get("TM"),
                "rue": physiology.get("radiation_use_efficiency") or physiology.get("RUEF"),
                "hi": physiology.get("harvest_index") or physiology.get("HI"),
            }
            ic["physiology"] = {k: v for k, v in ic["physiology"].items() if v is not None}

        return ic

    def _build_soil(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ACE soil structure if soil info is present."""
        soil_config = config.get("soil", {})
        if not soil_config:
            return None

        soil = {}

        # Generate soil ID
        soil["soil_id"] = f"CT_SOIL_{hash(str(soil_config)) % 100000:05d}"

        # Basic soil properties
        depth = soil_config.get("soil_depth_m") or soil_config.get("default_depth_m")
        if depth:
            soil["sldp"] = depth * 100  # Convert m to cm

        soil["sltx"] = soil_config.get("texture", "")
        soil["salb"] = soil_config.get("albedo")
        soil["slro"] = soil_config.get("runoff_curve")

        # Source info
        soil["sl_source"] = soil_config.get("source", "unknown")

        return soil

    def _get_crop_code(self, crop_name: str) -> str:
        """Get ICASA crop code from crop name."""
        crop_codes = {
            "maize": "MZ",
            "corn": "MZ",
            "wheat": "WH",
            "rice": "RI",
            "soybean": "SB",
            "sorghum": "SG",
            "millet": "ML",
            "pearl millet": "ML",
            "peanut": "PN",
            "groundnut": "PN",
            "cotton": "CT",
            "cassava": "CS",
            "cowpea": "CP",
            "dry bean": "BN",
        }
        return crop_codes.get(crop_name.lower(), "XX")

    # =========================================================================
    # Import: ACE JSON → prismpy
    # =========================================================================

    def import_ace(self, ace_data: Union[Dict[str, Any], str, Path],
                   experiment_index: int = 0) -> Dict[str, Any]:
        """
        Import an ACE JSON dataset to prismpy config format.

        Args:
            ace_data: ACE JSON data (dict, JSON string, or file path)
            experiment_index: Which experiment to import if multiple

        Returns:
            prismpy base configuration dictionary
        """
        # Load data if string or path
        if isinstance(ace_data, (str, Path)):
            if Path(ace_data).exists():
                with open(ace_data, 'r') as f:
                    ace_data = json.load(f)
            else:
                ace_data = json.loads(ace_data)

        # Get experiment
        experiments = ace_data.get("experiments", [])
        if not experiments:
            raise ValueError("No experiments found in ACE data")

        if experiment_index >= len(experiments):
            raise ValueError(f"Experiment index {experiment_index} out of range")

        experiment = experiments[experiment_index]

        # Build config
        config = {}

        # Project
        config["project"] = {
            "name": experiment.get("exname", "Imported from ACE"),
            "description": f"Imported from ACE experiment {experiment.get('exp_id', 'unknown')}",
        }

        # Region
        config["region"] = {
            "name": experiment.get("fl_loc_1", "Unknown"),
            "country": experiment.get("fl_loc_2", "Unknown"),
        }

        # Location
        config["location"] = {
            "latitude": experiment.get("fl_lat"),
            "longitude": experiment.get("fl_long"),
        }

        # Crop
        config["crop"] = {
            "name": self._get_crop_name(experiment.get("crid", "")),
            "variety": experiment.get("cul_name", ""),
        }

        # Phenology from initial conditions
        ic = experiment.get("initial_conditions", {})
        phenology_data = ic.get("phenology", {})
        if phenology_data:
            config["phenology"] = {
                "emergence_gdd": phenology_data.get("p1"),
                "photoperiod_sensitivity": phenology_data.get("p2"),
                "grain_filling_gdd": phenology_data.get("p5"),
                "phyllochron": phenology_data.get("phint"),
            }
            # Remove None values
            config["phenology"] = {k: v for k, v in config["phenology"].items() if v is not None}

        # Physiology from initial conditions
        physiology_data = ic.get("physiology", {})
        if physiology_data:
            config["physiology"] = {
                "base_temperature": physiology_data.get("tb"),
                "optimal_temperature": physiology_data.get("to"),
                "max_temperature": physiology_data.get("tm"),
                "radiation_use_efficiency": physiology_data.get("rue"),
                "harvest_index": physiology_data.get("hi"),
            }
            config["physiology"] = {k: v for k, v in config["physiology"].items() if v is not None}

        # Management
        config["management"] = self._import_management(experiment.get("management", {}))

        # Temporal
        config["temporal"] = {
            "start_year": experiment.get("sc_year"),
            "end_year": experiment.get("endsim"),
        }

        # Soil
        soil_id = experiment.get("soil_id")
        if soil_id:
            soils = ace_data.get("soils", [])
            soil = next((s for s in soils if s.get("soil_id") == soil_id), None)
            if soil:
                config["soil"] = self._import_soil(soil)

        return config

    def _import_management(self, mgmt: Dict[str, Any]) -> Dict[str, Any]:
        """Import ACE management data to prismpy format."""
        management = {}

        # Planting
        plantings = mgmt.get("planting", [])
        if plantings:
            planting = plantings[0]  # Take first planting event
            management["planting_doy"] = planting.get("pdate")

            # Convert plant population (plants/m² → plants/ha)
            ppop = planting.get("ppop")
            if ppop:
                management["planting_density"] = ppop * 10000 if ppop < 100 else ppop

            management["row_spacing_cm"] = planting.get("plrs")
            management["planting_depth_cm"] = planting.get("pldp")
            management["planting_method"] = planting.get("plme")
            management["plant_distribution"] = planting.get("plds")

        # Fertilizer
        fertilizers = mgmt.get("fertilizer", [])
        if fertilizers:
            total_n = sum(f.get("feamn", 0) for f in fertilizers)
            management["fertilizer_n_total"] = total_n

            if len(fertilizers) > 1:
                management["fertilizer_n_splits"] = [f.get("fday", 0) for f in fertilizers]
                if total_n > 0:
                    management["fertilizer_n_fractions"] = [
                        f.get("feamn", 0) / total_n for f in fertilizers
                    ]

        # Irrigation
        irrigations = mgmt.get("irrigation", [])
        if irrigations:
            irrig = irrigations[0]
            management["irrigation"] = irrig.get("irrig", "N") == "Y"

        return management

    def _import_soil(self, soil: Dict[str, Any]) -> Dict[str, Any]:
        """Import ACE soil data to prismpy format."""
        soil_config = {}

        depth_cm = soil.get("sldp")
        if depth_cm:
            soil_config["default_depth_m"] = depth_cm / 100.0

        soil_config["texture"] = soil.get("sltx")
        soil_config["albedo"] = soil.get("salb")
        soil_config["runoff_curve"] = soil.get("slro")
        soil_config["source"] = soil.get("sl_source", "imported")

        # Remove None values
        return {k: v for k, v in soil_config.items() if v is not None}

    def _get_crop_name(self, crop_code: str) -> str:
        """Get crop name from ICASA code."""
        code_names = {
            "MZ": "Maize",
            "WH": "Wheat",
            "RI": "Rice",
            "SB": "Soybean",
            "SG": "Sorghum",
            "ML": "Millet",
            "PN": "Peanut",
            "CT": "Cotton",
            "CS": "Cassava",
            "CP": "Cowpea",
            "BN": "Dry bean",
        }
        return code_names.get(crop_code.upper(), crop_code)

    # =========================================================================
    # File I/O
    # =========================================================================

    def export_to_file(self, config: Dict[str, Any], output_path: Path,
                       indent: int = 2) -> Path:
        """
        Export config to ACE JSON file.

        Args:
            config: prismpy configuration
            output_path: Output file path
            indent: JSON indentation

        Returns:
            Path to created file
        """
        ace_data = self.export_ace(config)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(ace_data, f, indent=indent)

        logger.info(f"Exported ACE JSON to {output_path}")
        return output_path

    def import_from_file(self, input_path: Path,
                         experiment_index: int = 0) -> Dict[str, Any]:
        """
        Import config from ACE JSON file.

        Args:
            input_path: Input ACE JSON file
            experiment_index: Which experiment to import

        Returns:
            prismpy configuration dictionary
        """
        with open(input_path, 'r') as f:
            ace_data = json.load(f)

        config = self.import_ace(ace_data, experiment_index)
        logger.info(f"Imported ACE JSON from {input_path}")

        return config


# Convenience functions
def export_ace(config: Dict[str, Any]) -> Dict[str, Any]:
    """Export config to ACE JSON format using default converter."""
    converter = AceConverter()
    return converter.export_ace(config)


def import_ace(ace_data: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
    """Import ACE JSON to config format using default converter."""
    converter = AceConverter()
    return converter.import_ace(ace_data)
