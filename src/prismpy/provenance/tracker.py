"""
Provenance tracking system for prismpy.

This module provides the central tracking system for all data handling
decisions, implementing the 'formalized methodology' requirement.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from prismpy.models.provenance import (
    DataLineage,
    DecisionRecord,
    DecisionType,
    OperationType,
    ProvenanceRecord,
    TransformationRecord,
)


class ProvenanceTracker:
    """Central tracking system for all data handling decisions.

    This class implements the formalized methodology by:
    1. Recording every data retrieval operation
    2. Documenting all transformation parameters
    3. Logging decision rules and their rationale
    4. Computing content hashes for reproducibility verification

    Attributes:
        enabled: Whether provenance tracking is enabled
        include_hashes: Whether to compute SHA256 hashes
        include_parameters: Whether to include all parameters
        storage_format: Storage format ("json", "sqlite", or "both")
    """

    def __init__(
        self,
        enabled: bool = True,
        include_hashes: bool = True,
        include_parameters: bool = True,
        storage_format: str = "json",
        output_dir: Optional[Union[str, Path]] = None,
        project_name: Optional[str] = None,
    ):
        """Initialize the provenance tracker.

        Args:
            enabled: Whether to enable tracking
            include_hashes: Whether to compute file hashes
            include_parameters: Whether to include full parameters
            storage_format: Storage format for provenance records
            output_dir: Directory for provenance output files
            project_name: Name of the project being tracked
        """
        self.enabled = enabled
        self.include_hashes = include_hashes
        self.include_parameters = include_parameters
        self.storage_format = storage_format
        self.output_dir = Path(output_dir) if output_dir else Path("provenance")
        self.project_name = project_name

        self.logger = logging.getLogger(__name__)

        # Generate unique session ID
        self.session_id = self._generate_session_id()

        # Initialize provenance record
        self.record = ProvenanceRecord(
            session_id=self.session_id,
            project_name=project_name,
        )

        # Track current artifact being processed
        self._current_artifact_id: Optional[str] = None

    def _generate_session_id(self) -> str:
        """Generate a unique session identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"tr_{timestamp}_{unique_id}"

    def _compute_hash(self, path: Union[str, Path]) -> Optional[str]:
        """Compute SHA256 hash of file contents."""
        if not self.include_hashes:
            return None

        path = Path(path)
        if not path.exists():
            return None

        try:
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not compute hash for {path}: {e}")
            return None

    def _compute_data_hash(self, data: Any) -> Optional[str]:
        """Compute hash of in-memory data."""
        if not self.include_hashes:
            return None

        try:
            # Convert to JSON string for hashing
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Could not compute data hash: {e}")
            return None

    def start_artifact(
        self,
        artifact_type: str,
        source_artifacts: Optional[List[str]] = None,
        artifact_id: Optional[str] = None,
    ) -> str:
        """Start tracking a new data artifact.

        Args:
            artifact_type: Type of artifact (e.g., "climate", "soil")
            source_artifacts: IDs of source artifacts
            artifact_id: Optional custom artifact ID

        Returns:
            The artifact ID
        """
        if not self.enabled:
            return ""

        if artifact_id is None:
            artifact_id = f"{artifact_type}_{self.session_id}_{len(self.record.artifacts)}"

        lineage = DataLineage(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            source_artifacts=source_artifacts or [],
        )

        self.record.add_artifact(lineage)
        self._current_artifact_id = artifact_id

        self.logger.debug(f"Started tracking artifact: {artifact_id}")
        return artifact_id

    def record_retrieval(
        self,
        source: str,
        parameters: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        decisions: Optional[List[DecisionRecord]] = None,
        artifact_id: Optional[str] = None,
    ) -> None:
        """Record a data retrieval operation.

        Args:
            source: Data source identifier (e.g., "NASA_POWER", "GADM")
            parameters: Retrieval parameters
            output_path: Path where data was saved
            decisions: Decisions made during retrieval
            artifact_id: Artifact ID (uses current if not specified)
        """
        if not self.enabled:
            return

        artifact_id = artifact_id or self._current_artifact_id
        if not artifact_id:
            self.logger.warning("No artifact ID for retrieval record")
            return

        lineage = self.record.get_artifact(artifact_id)
        if not lineage:
            self.logger.warning(f"Unknown artifact: {artifact_id}")
            return

        # Prepare parameters
        record_params = {"source": source}
        if self.include_parameters:
            record_params.update(parameters)
        else:
            # Include only key parameters
            record_params["parameters_count"] = len(parameters)

        # Compute output hash
        output_hash = None
        if output_path:
            output_hash = self._compute_hash(output_path)

        transformation = TransformationRecord(
            operation=OperationType.RETRIEVE,
            parameters=record_params,
            decisions=decisions or [],
            output_hash=output_hash,
        )

        lineage.add_transformation(transformation)
        self.logger.debug(f"Recorded retrieval from {source}")

    def record_transformation(
        self,
        operation: OperationType,
        parameters: Dict[str, Any],
        decisions: Optional[List[DecisionRecord]] = None,
        input_hash: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        warnings: Optional[List[str]] = None,
        artifact_id: Optional[str] = None,
    ) -> None:
        """Record a data transformation.

        Args:
            operation: Type of operation
            parameters: Operation parameters
            decisions: Decisions made during transformation
            input_hash: Hash of input data
            output_path: Path where output was saved
            warnings: Any warnings generated
            artifact_id: Artifact ID (uses current if not specified)
        """
        if not self.enabled:
            return

        artifact_id = artifact_id or self._current_artifact_id
        if not artifact_id:
            self.logger.warning("No artifact ID for transformation record")
            return

        lineage = self.record.get_artifact(artifact_id)
        if not lineage:
            self.logger.warning(f"Unknown artifact: {artifact_id}")
            return

        # Prepare parameters
        record_params = parameters if self.include_parameters else {}

        # Compute output hash
        output_hash = None
        if output_path:
            output_hash = self._compute_hash(output_path)

        transformation = TransformationRecord(
            operation=operation,
            parameters=record_params,
            decisions=decisions or [],
            input_hash=input_hash,
            output_hash=output_hash,
            warnings=warnings or [],
        )

        lineage.add_transformation(transformation)
        self.logger.debug(f"Recorded {operation.value} transformation")

    def record_decision(
        self,
        decision_type: DecisionType,
        description: str,
        rationale: str,
        alternatives: Optional[List[str]] = None,
        reference: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> DecisionRecord:
        """Create and optionally record a decision.

        This is a convenience method that creates a DecisionRecord.
        The decision will be attached to the next transformation.

        Args:
            decision_type: Type of decision
            description: What was decided
            rationale: Why this decision was made
            alternatives: Other options considered
            reference: Citation or documentation link
            artifact_id: Artifact ID (for context)

        Returns:
            The created DecisionRecord
        """
        decision = DecisionRecord(
            decision_type=decision_type,
            description=description,
            rationale=rationale,
            alternatives_considered=alternatives or [],
            reference=reference,
        )

        self.logger.debug(f"Created decision: {description}")
        return decision

    def add_warning(
        self,
        message: str,
        artifact_id: Optional[str] = None,
    ) -> None:
        """Add a warning to the current transformation.

        Args:
            message: Warning message
            artifact_id: Artifact ID (uses current if not specified)
        """
        if not self.enabled:
            return

        artifact_id = artifact_id or self._current_artifact_id
        if not artifact_id:
            return

        lineage = self.record.get_artifact(artifact_id)
        if lineage and lineage.transformations:
            lineage.transformations[-1].warnings.append(message)
            self.logger.warning(message)

    def set_config_hash(self, config: Any) -> None:
        """Set the configuration hash for the session.

        Args:
            config: Configuration object to hash
        """
        if not self.enabled:
            return

        self.record.config_hash = self._compute_data_hash(config)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session.

        Returns:
            Summary dictionary
        """
        return self.record.compute_summary()

    def save(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Save provenance record to file.

        Args:
            output_path: Optional custom output path

        Returns:
            Path where the record was saved
        """
        if not self.enabled:
            return Path()

        # Determine output path
        if output_path:
            save_path = Path(output_path)
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.output_dir / f"{self.session_id}_provenance.json"

        # Save based on storage format
        if self.storage_format in ("json", "both"):
            self.record.save_json(save_path)
            self.logger.info(f"Saved provenance to {save_path}")

        return save_path

    def get_report(self) -> str:
        """Generate a human-readable provenance report.

        Returns:
            Formatted report string
        """
        summary = self.get_summary()

        lines = [
            "=" * 60,
            "PROVENANCE REPORT",
            "=" * 60,
            f"Session ID: {self.session_id}",
            f"Project: {self.project_name or 'N/A'}",
            f"Created: {self.record.created_at.isoformat()}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Artifacts processed: {summary.get('n_artifacts', 0)}",
            f"Transformations: {summary.get('n_transformations', 0)}",
            f"Decisions made: {summary.get('n_decisions', 0)}",
            f"Warnings: {summary.get('n_warnings', 0)}",
            "",
        ]

        # Decisions by type
        if summary.get("decisions_by_type"):
            lines.append("DECISIONS BY TYPE")
            lines.append("-" * 40)
            for dtype, count in summary["decisions_by_type"].items():
                lines.append(f"  {dtype}: {count}")
            lines.append("")

        # Artifact details
        lines.append("ARTIFACTS")
        lines.append("-" * 40)
        for artifact_id, lineage in self.record.artifacts.items():
            lines.append(f"\n{artifact_id} ({lineage.artifact_type})")
            for i, transform in enumerate(lineage.transformations, 1):
                lines.append(f"  {i}. {transform.operation.value}")
                if transform.decisions:
                    for decision in transform.decisions:
                        lines.append(f"     - {decision.description}")
                if transform.warnings:
                    for warning in transform.warnings:
                        lines.append(f"     ! {warning}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# Convenience function for creating common decisions
def decision_source_selection(
    selected: str,
    alternatives: List[str],
    rationale: str,
    reference: Optional[str] = None,
) -> DecisionRecord:
    """Create a source selection decision record."""
    return DecisionRecord(
        decision_type=DecisionType.SOURCE_SELECTION,
        description=f"Selected {selected} as data source",
        rationale=rationale,
        alternatives_considered=alternatives,
        reference=reference,
    )


def decision_gap_fill(
    method: str,
    max_gap: int,
    gaps_filled: int,
    reference: Optional[str] = None,
) -> DecisionRecord:
    """Create a gap-filling decision record."""
    return DecisionRecord(
        decision_type=DecisionType.GAP_FILL_METHOD,
        description=f"Used {method} to fill {gaps_filled} gaps (max {max_gap} days)",
        rationale=f"Standard approach for short meteorological gaps",
        alternatives_considered=["linear", "nearest", "climatology"],
        reference=reference,
    )


def decision_default_value(
    parameter: str,
    value: Any,
    reason: str,
) -> DecisionRecord:
    """Create a default value decision record."""
    return DecisionRecord(
        decision_type=DecisionType.DEFAULT_VALUE,
        description=f"Used default value {value} for {parameter}",
        rationale=reason,
        alternatives_considered=[],
    )
