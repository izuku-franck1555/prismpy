"""
Provenance models for prismpy.

These models track the complete lineage and transformation history
of all data artifacts, enabling full reproducibility and audit trails.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class OperationType(str, Enum):
    """Types of data operations tracked in provenance."""
    RETRIEVE = "retrieve"
    GAP_FILL = "gap_fill"
    INTERPOLATE = "interpolate"
    RESAMPLE = "resample"
    REPROJECT = "reproject"
    CONVERT_UNITS = "convert_units"
    QUALITY_CHECK = "quality_check"
    TRANSLATE = "translate"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    BUILD_GRID = "build_grid"  # V2-19: grid creation transformation


class DecisionType(str, Enum):
    """Types of decisions made during data processing."""
    SOURCE_SELECTION = "source_selection"
    GAP_FILL_METHOD = "gap_fill_method"
    DEFAULT_VALUE = "default_value"
    INTERPOLATION_METHOD = "interpolation_method"
    RESAMPLING_METHOD = "resampling_method"
    QUALITY_THRESHOLD = "quality_threshold"
    PARAMETER_OVERRIDE = "parameter_override"
    FORMAT_CHOICE = "format_choice"
    # V2-19 additions — additive, no breaking changes
    FALLBACK_SUBSTITUTION = "fallback_substitution"  # Silent substitution when primary source fails (distinct from DEFAULT_VALUE)
    UNIT_CONVERSION = "unit_conversion"              # Unit conversions (iSDA bd/100, pH/10, W/m² → MJ/m²/day)
    AGGREGATION_METHOD = "aggregation_method"        # Aggregation choice (mean/sum/majority) distinct from RESAMPLING


@dataclass
class DecisionRecord:
    """Documents a specific decision made during translation.

    Attributes:
        decision_type: Category of the decision
        description: What was decided
        rationale: Why this decision was made
        alternatives_considered: Other options that were available
        reference: Citation or documentation link
    """
    decision_type: DecisionType
    description: str
    rationale: str
    alternatives_considered: List[str] = field(default_factory=list)
    reference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_type": self.decision_type.value,
            "description": self.description,
            "rationale": self.rationale,
            "alternatives_considered": self.alternatives_considered,
            "reference": self.reference,
        }


@dataclass
class TransformationRecord:
    """Records a single data transformation.

    Attributes:
        operation: Type of operation performed
        timestamp: When the operation was performed
        parameters: All parameters used in the operation
        decisions: Decisions made during the operation
        input_hash: SHA256 hash of input data (for verification)
        output_hash: SHA256 hash of output data (for verification)
        warnings: Any warnings generated during the operation
        errors: Any errors encountered (operation may still succeed)
    """
    operation: OperationType
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    decisions: List[DecisionRecord] = field(default_factory=list)
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation.value,
            "timestamp": self.timestamp.isoformat(),
            "parameters": self.parameters,
            "decisions": [d.to_dict() for d in self.decisions],
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class DataLineage:
    """Tracks the complete lineage of a data artifact.

    Attributes:
        artifact_id: Unique identifier for this artifact
        artifact_type: Type of artifact (e.g., "climate", "soil", "crop_params")
        created_at: When the artifact was created
        source_artifacts: IDs of source artifacts used to create this one
        transformations: Sequence of transformations applied
    """
    artifact_id: str
    artifact_type: str
    created_at: datetime = field(default_factory=datetime.now)
    source_artifacts: List[str] = field(default_factory=list)
    transformations: List[TransformationRecord] = field(default_factory=list)

    def add_transformation(self, transformation: TransformationRecord) -> None:
        """Add a transformation to the lineage."""
        self.transformations.append(transformation)

    @property
    def n_transformations(self) -> int:
        """Number of transformations applied."""
        return len(self.transformations)

    @property
    def all_decisions(self) -> List[DecisionRecord]:
        """Get all decisions across all transformations."""
        decisions = []
        for t in self.transformations:
            decisions.extend(t.decisions)
        return decisions

    @property
    def all_warnings(self) -> List[str]:
        """Get all warnings across all transformations."""
        warnings = []
        for t in self.transformations:
            warnings.extend(t.warnings)
        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "created_at": self.created_at.isoformat(),
            "source_artifacts": self.source_artifacts,
            "n_transformations": self.n_transformations,
            "transformations": [t.to_dict() for t in self.transformations],
        }


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a translation session.

    This is the top-level container for all provenance information
    from a single execution of the translation pipeline.

    Attributes:
        session_id: Unique session identifier
        created_at: When the session started
        project_name: Name of the project being processed
        config_hash: Hash of the configuration used
        artifacts: All data artifacts and their lineages
        summary: Summary statistics
    """
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    project_name: Optional[str] = None
    config_hash: Optional[str] = None
    artifacts: Dict[str, DataLineage] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def add_artifact(self, lineage: DataLineage) -> None:
        """Add an artifact lineage to the record."""
        self.artifacts[lineage.artifact_id] = lineage

    def get_artifact(self, artifact_id: str) -> Optional[DataLineage]:
        """Get lineage for a specific artifact."""
        return self.artifacts.get(artifact_id)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        total_decisions = sum(
            len(lin.all_decisions) for lin in self.artifacts.values()
        )
        total_warnings = sum(
            len(lin.all_warnings) for lin in self.artifacts.values()
        )
        total_transformations = sum(
            lin.n_transformations for lin in self.artifacts.values()
        )

        # Count decisions by type
        decision_counts = {}
        for lineage in self.artifacts.values():
            for decision in lineage.all_decisions:
                dtype = decision.decision_type.value
                decision_counts[dtype] = decision_counts.get(dtype, 0) + 1

        self.summary = {
            "n_artifacts": len(self.artifacts),
            "n_transformations": total_transformations,
            "n_decisions": total_decisions,
            "n_warnings": total_warnings,
            "decisions_by_type": decision_counts,
            "artifact_types": list(set(
                lin.artifact_type for lin in self.artifacts.values()
            )),
        }
        return self.summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        self.compute_summary()
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "project_name": self.project_name,
            "config_hash": self.config_hash,
            "artifacts": {
                aid: lin.to_dict() for aid, lin in self.artifacts.items()
            },
            "summary": self.summary,
        }

    def save_json(self, path: str) -> None:
        """Save provenance record to JSON file."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
