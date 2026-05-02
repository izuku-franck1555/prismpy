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
    QUALITY_CHECK = "quality_check"                  # Validation quality check outcome (Phase 2a)


@dataclass
class AlternativeConsidered:
    """A rejected alternative in a decision.

    Phase 4: structured alternative with name + reason_rejected,
    replacing the flat string list for richer UI display.
    """
    name: str
    reason_rejected: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name}
        if self.reason_rejected:
            d["reason_rejected"] = self.reason_rejected
        return d


@dataclass
class DecisionRecord:
    """Documents a specific decision made during translation.

    Attributes:
        decision_type: Category of the decision
        description: What was decided
        rationale: Why this decision was made
        alternatives_considered: Other options (structured or flat strings)
        reference: Citation or documentation link
        severity: info (standard), warning (fallback/substitution),
            error (known-incorrect outcome like TP-06 CROPGRO→CERES)
        label: Short one-liner for collapsed timeline view
            (distinct from the longer description)
        timestamp: When the decision was recorded (enables timeline)
    """
    decision_type: DecisionType
    description: str
    rationale: str
    alternatives_considered: List = field(default_factory=list)
    reference: Optional[str] = None
    severity: str = "info"
    label: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Serialize alternatives — support both flat strings (backward
        # compat) and structured AlternativeConsidered objects
        alts = []
        for alt in self.alternatives_considered:
            if isinstance(alt, AlternativeConsidered):
                alts.append(alt.to_dict())
            elif isinstance(alt, dict):
                alts.append(alt)
            else:
                alts.append({"name": str(alt)})

        return {
            "decision_type": self.decision_type.value,
            "description": self.description,
            "rationale": self.rationale,
            "alternatives_considered": alts,
            "reference": self.reference,
            "severity": self.severity,
            "label": self.label or self.description[:80],
            "timestamp": self.timestamp.isoformat(),
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
        stage: Pipeline stage this artifact belongs to (retrieve/harmonize/
            translate/validate). Phase 4 addition for timeline grouping.
    """
    artifact_id: str
    artifact_type: str
    created_at: datetime = field(default_factory=datetime.now)
    source_artifacts: List[str] = field(default_factory=list)
    transformations: List[TransformationRecord] = field(default_factory=list)
    stage: Optional[str] = None

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
        d = {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "created_at": self.created_at.isoformat(),
            "source_artifacts": self.source_artifacts,
            "n_transformations": self.n_transformations,
            "transformations": [t.to_dict() for t in self.transformations],
        }
        if self.stage:
            d["stage"] = self.stage
        return d


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
        boundary: F-R AC-4 — boundary section recording the
            cell-inclusion-rule decisions that produced the
            simulation domain. Empty dict ``{}`` for legacy
            records (pre-F-R) so reads stay non-throwing;
            ``ProvenanceTracker.set_boundary(...)`` populates
            8 fields (source / version / inclusion_rule /
            min_share_percent / 4 cell-count fields) once
            HARMONIZE finishes.
        texture_renormalize_details: Sprint D.1 AC-6 — per-layer
            entries for every texture-fraction renormalization
            applied at the harmonize stage. Each entry is the
            serialized form of
            :class:`prismpy.harmonize.texture_renormalize.TextureRenormalizationProvenance`.
            Empty list when no renormalization fired.
        rh_clip_details: Sprint D.1 AC-6 — per-record entries for
            every rh value clipped from (100, 102] to 100 at the
            harmonize stage. Each entry is the serialized form of
            :class:`prismpy.harmonize.rh_clip.RHClipProvenance`.
            Empty list when no clip fired.
        cells_unavailable_details: Sprint D.1 AC-6 — per-cell
            entries for every cell routed to
            ``data_availability='unavailable'`` at the harmonize
            stage with the reason / cause taxonomy populated. Each
            entry is a dict with ``cell_id`` plus the matching
            ``unavailable_reason`` and ``unavailable_cause``
            literals.
        pythia_misdat_replacements: Sprint D.1 AC-6 — per-translator
            count of records that wrote the DSSAT MISDAT sentinel
            (-99.0) for missing rain. Currently the value is the
            count keyed by translator name; PYTHIA is the only
            target today. Empty dict when no missing rain occurred.
    """
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    project_name: Optional[str] = None
    config_hash: Optional[str] = None
    artifacts: Dict[str, DataLineage] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    # F-R AC-4: boundary block populated by ProvenanceTracker.set_boundary().
    # Dict[str, Any] (NOT Optional[Dict]) — empty {} is the legacy compat
    # sentinel; readers can do ``record.boundary.get("inclusion_rule")``
    # without None-checks.
    boundary: Dict[str, Any] = field(default_factory=dict)
    # Sprint D.1 AC-6 — additive harmonize-stage detail lists.
    # Each list is empty by default for legacy compat; the
    # corresponding setter methods on ProvenanceTracker populate
    # them as the pipeline runs.
    texture_renormalize_details: List[Dict[str, Any]] = field(default_factory=list)
    rh_clip_details: List[Dict[str, Any]] = field(default_factory=list)
    cells_unavailable_details: List[Dict[str, Any]] = field(default_factory=list)
    pythia_misdat_replacements: Dict[str, int] = field(default_factory=dict)

    def add_artifact(self, lineage: DataLineage) -> None:
        """Add an artifact lineage to the record."""
        self.artifacts[lineage.artifact_id] = lineage

    def get_artifact(self, artifact_id: str) -> Optional[DataLineage]:
        """Get lineage for a specific artifact."""
        return self.artifacts.get(artifact_id)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics.

        Sprint D.1 AC-6: aggregate counts for the harmonize-stage
        transformations are merged into the summary alongside the
        existing artifact / decision / warning aggregates so
        consumers reading the summary block get every per-sprint
        invariant in one place.
        """
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

        # Sprint D.1 AC-6 — aggregate per-cause counts for
        # unavailable cells. The detail list carries each entry's
        # ``unavailable_cause`` (the schema-pinned cause field on
        # CellSummary); the summary surfaces the count keyed by
        # cause so a consumer reads the headline number without
        # iterating the detail list. Entries that pre-date the
        # cause field (or come from a writer that doesn't set
        # one) bucket into ``"unknown"``.
        cells_unavailable_by_cause: Dict[str, int] = {}
        for entry in self.cells_unavailable_details:
            cause = (
                entry.get("unavailable_cause")
                or entry.get("cause")
                or "unknown"
            )
            cells_unavailable_by_cause[cause] = (
                cells_unavailable_by_cause.get(cause, 0) + 1
            )

        self.summary = {
            "n_artifacts": len(self.artifacts),
            "n_transformations": total_transformations,
            "n_decisions": total_decisions,
            "n_warnings": total_warnings,
            "decisions_by_type": decision_counts,
            "artifact_types": list(set(
                lin.artifact_type for lin in self.artifacts.values()
            )),
            # Sprint D.1 AC-6 additions — additive aggregates from
            # the harmonize-stage detail lists. Empty / zero
            # values for runs without the corresponding fix
            # firing.
            "texture_renormalize_count": len(
                self.texture_renormalize_details
            ),
            "rh_clip_count": len(self.rh_clip_details),
            "cells_unavailable_by_cause": cells_unavailable_by_cause,
            "pythia_misdat_replacements": dict(
                self.pythia_misdat_replacements
            ),
        }
        return self.summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Sprint D.1 AC-6: harmonize-stage detail lists serialize
        alongside the existing top-level keys. JSON insertion
        order matches the field order on the dataclass so a
        diff-friendly serialization stays stable across runs.
        """
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
            # F-R AC-4: boundary block always emitted (empty dict
            # for legacy reads). Per-platform copies via shutil
            # carry the same block automatically.
            "boundary": self.boundary,
            # Sprint D.1 AC-6 — additive harmonize-stage detail
            # lists. Empty lists / dicts for legacy reads (the
            # corresponding setter methods on ProvenanceTracker
            # populate the lists; consumers should ``.get(key,
            # [])`` to stay backward-compat with old payloads).
            "texture_renormalize_details": list(
                self.texture_renormalize_details
            ),
            "rh_clip_details": list(self.rh_clip_details),
            "cells_unavailable_details": list(
                self.cells_unavailable_details
            ),
        }

    def save_json(self, path: str) -> None:
        """Save provenance record to JSON file."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
