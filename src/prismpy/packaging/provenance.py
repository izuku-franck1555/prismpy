"""
Provenance Tracking for prismpy packages.

Provides tools for tracking and recording data provenance,
including decisions made during data processing.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ProvenanceTracker:
    """
    Tracks provenance information across processing stages.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        workflow: str = "prismpy"
    ):
        """
        Initialize provenance tracker.

        Args:
            session_id: Unique session identifier (auto-generated if not provided)
            workflow: Name of the workflow being tracked
        """
        self.session_id = session_id or f"ct_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.workflow = workflow
        self.stages: List[Dict[str, Any]] = []
        self.global_decisions: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()

    def add_stage(
        self,
        stage_name: str,
        inputs: Dict[str, Any],
        outputs: List[str],
        decisions: Optional[List[Dict[str, str]]] = None,
        notes: Optional[str] = None
    ) -> None:
        """
        Record a processing stage.

        Args:
            stage_name: Name of the stage (e.g., "RETRIEVE", "TRANSLATE")
            inputs: Dictionary of input files/parameters
            outputs: List of output files produced
            decisions: List of decisions made (each with type, value, rationale)
            notes: Optional notes about the stage
        """
        stage_record = {
            "stage": stage_name,
            "executed_at": datetime.now().isoformat(),
            "inputs": inputs,
            "outputs": outputs,
            "decisions": decisions or [],
        }

        if notes:
            stage_record["notes"] = notes

        self.stages.append(stage_record)

    def add_decision(
        self,
        decision_type: str,
        value: Any,
        rationale: str,
        stage: Optional[str] = None
    ) -> None:
        """
        Record a decision.

        Args:
            decision_type: Type of decision (e.g., "BOUNDARY_SOURCE", "RAINFALL_SOURCE")
            value: The value/choice made
            rationale: Explanation for the decision
            stage: Optional stage to add the decision to
        """
        decision = {
            "type": decision_type,
            "value": value,
            "rationale": rationale,
            "recorded_at": datetime.now().isoformat()
        }

        if stage:
            for s in self.stages:
                if s["stage"] == stage:
                    s["decisions"].append(decision)
                    return

        self.global_decisions.append(decision)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert provenance to dictionary.

        Returns:
            Provenance dictionary
        """
        return {
            "session_id": self.session_id,
            "workflow": self.workflow,
            "created_at": self.created_at,
            "completed_at": datetime.now().isoformat(),
            "stages": self.stages,
            "global_decisions": self.global_decisions if self.global_decisions else None
        }

    def save(self, output_path: Union[str, Path]) -> Path:
        """
        Save provenance to JSON file.

        Args:
            output_path: Path to save provenance

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        return output_path


def create_stage_record(
    stage_name: str,
    inputs: Dict[str, Any],
    outputs: List[str],
    decisions: Optional[List[Dict[str, str]]] = None,
    executed_at: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a stage record dictionary.

    Args:
        stage_name: Name of the processing stage
        inputs: Dictionary of inputs
        outputs: List of output files
        decisions: List of decisions made
        executed_at: Timestamp (auto-generated if not provided)

    Returns:
        Stage record dictionary
    """
    return {
        "stage": stage_name,
        "executed_at": executed_at or datetime.now().isoformat(),
        "inputs": inputs,
        "outputs": outputs,
        "decisions": decisions or []
    }


def create_decision(
    decision_type: str,
    value: Any,
    rationale: str,
    alternatives: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a decision record.

    Args:
        decision_type: Type of decision
        value: The chosen value
        rationale: Reason for the choice
        alternatives: Optional list of alternatives considered

    Returns:
        Decision dictionary
    """
    decision = {
        "type": decision_type,
        "value": value,
        "rationale": rationale
    }

    if alternatives:
        decision["alternatives_considered"] = alternatives

    return decision


def load_provenance(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load provenance from JSON file.

    Args:
        path: Path to provenance.json

    Returns:
        Provenance dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


# Default data source decisions for SARRA-Py workflow
DEFAULT_DECISIONS = {
    "BOUNDARY_SOURCE": create_decision(
        "BOUNDARY_SOURCE",
        "GADM v4.1",
        "Standard global administrative boundaries with consistent naming"
    ),
    "RAINFALL_SOURCE": create_decision(
        "RAINFALL_SOURCE",
        "TAMSAT v3.1",
        "Best coverage and accuracy for West Africa rainfall estimation",
        alternatives=["CHIRPS", "GPM"]
    ),
    "TEMPERATURE_SOURCE": create_decision(
        "TEMPERATURE_SOURCE",
        "AgERA5",
        "High resolution reanalysis data from Copernicus Climate Data Store",
        alternatives=["NASA POWER", "CRU"]
    ),
    "SOIL_SOURCE": create_decision(
        "SOIL_SOURCE",
        "iSDA",
        "High resolution (30m) soil data for Africa from iSDA Africa",
        alternatives=["HWSD", "SoilGrids"]
    ),
    "CROP_PARAMETERS": create_decision(
        "CROP_PARAMETERS",
        "SARRA-Py defaults",
        "Validated parameters from SARRA-Py model templates"
    )
}
