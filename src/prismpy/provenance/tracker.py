"""
Provenance tracking system for prismpy.

This module provides the central tracking system for all data handling
decisions, implementing the 'formalized methodology' requirement.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from prismpy.models.provenance import (
    DataLineage,
    DecisionRecord,
    DecisionType,
    OperationType,
    ProvenanceRecord,
    TransformationRecord,
)


class ProvenanceStateError(RuntimeError):
    """Raised when provenance state is inconsistent.

    Fires when record_retrieval/record_transformation is called with
    pending decisions but no active artifact — indicates a bug in
    caller ordering (decision recorded before start_artifact).
    """


class ProvenanceTracker:
    """Central tracking system for all data handling decisions.

    Implements the formalized methodology by:
    1. Recording every data retrieval operation
    2. Documenting all transformation parameters
    3. Logging decision rules and their rationale
    4. Computing content hashes for reproducibility verification

    Thread safety (V2-19):
        This tracker is thread-safe for concurrent record_decision() calls
        via an internal `threading.RLock`. However, the tracker is designed
        as a PER-PIPELINE-RUN INSTANCE — it is created in
        ``TranslationPipeline.__init__()`` and is NOT a shared singleton.
        Do NOT reuse a tracker instance across multiple
        ``TranslationPipeline.execute()`` calls — pending-decision state
        would carry over. The web app achieves per-run isolation because
        ``prismweb/core/tasks.py`` creates a fresh ``TranslationPipeline``
        (and therefore a fresh tracker) for every pipeline run.

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

        # V2-19: pending decisions list — decisions accumulate here until
        # a record_retrieval/record_transformation call flushes them into
        # the transformation it's recording. Each entry is (decision,
        # artifact_id_at_record_time) so misattribution is impossible when
        # the current-artifact pointer moves between decisions.
        self._pending_decisions: List[Tuple[DecisionRecord, Optional[str]]] = []

        # V2-19: thread-safety lock for pending_decisions list mutations.
        # RLock (not Lock) because finalize() calls record_transformation()
        # which acquires the same lock — RLock permits re-entrance from
        # the same thread.
        self._lock = threading.RLock()

        # V2-19: incremental checkpoint save path (set by caller if
        # belt-and-suspenders stage-boundary saves are desired)
        self._checkpoint_path: Optional[Path] = None

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

        V2-19: pending decisions bound to this artifact are flushed into
        the resulting ``TransformationRecord`` (pending-first, explicit-
        second). Raises ``ProvenanceStateError`` if there are pending
        decisions but no active artifact (caller ordering bug).

        Args:
            source: Data source identifier (e.g., "NASA_POWER", "GADM")
            parameters: Retrieval parameters
            output_path: Path where data was saved
            decisions: Decisions made during retrieval (explicit)
            artifact_id: Artifact ID (uses current if not specified)
        """
        if not self.enabled:
            return

        artifact_id = artifact_id or self._current_artifact_id
        if not artifact_id:
            # Improvement 6: fail-fast if decisions are pending but no
            # artifact exists to attach them to
            with self._lock:
                if self._pending_decisions:
                    raise ProvenanceStateError(
                        f"record_retrieval called with no active artifact "
                        f"but {len(self._pending_decisions)} pending decisions. "
                        f"Caller must call start_artifact() before "
                        f"record_decision()."
                    )
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

        # V2-19: drain pending decisions bound to this artifact + merge
        # caller's explicit decisions
        merged_decisions = self._drain_pending_for_artifact(artifact_id, decisions)

        transformation = TransformationRecord(
            operation=OperationType.RETRIEVE,
            parameters=record_params,
            decisions=merged_decisions,
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

        V2-19: pending decisions bound to this artifact are flushed into
        the resulting ``TransformationRecord``. Raises
        ``ProvenanceStateError`` if there are pending decisions but no
        active artifact (caller ordering bug).

        Args:
            operation: Type of operation
            parameters: Operation parameters
            decisions: Decisions made during transformation (explicit)
            input_hash: Hash of input data
            output_path: Path where output was saved
            warnings: Any warnings generated
            artifact_id: Artifact ID (uses current if not specified)
        """
        if not self.enabled:
            return

        artifact_id = artifact_id or self._current_artifact_id
        if not artifact_id:
            # Improvement 6: fail-fast if decisions are pending but no
            # artifact exists to attach them to
            with self._lock:
                if self._pending_decisions:
                    raise ProvenanceStateError(
                        f"record_transformation called with no active artifact "
                        f"but {len(self._pending_decisions)} pending decisions. "
                        f"Caller must call start_artifact() before "
                        f"record_decision()."
                    )
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

        # V2-19: drain pending decisions bound to this artifact + merge
        # caller's explicit decisions
        merged_decisions = self._drain_pending_for_artifact(artifact_id, decisions)

        transformation = TransformationRecord(
            operation=operation,
            parameters=record_params,
            decisions=merged_decisions,
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
        """Record a decision and bind it to the correct artifact.

        Two attachment paths exist depending on the artifact's lifecycle
        state at record-time:

        1. **Direct-attach (V2-19b-fix Finding 4)** — when an explicit
           ``artifact_id`` is provided AND the named artifact already
           exists AND has at least one transformation, the decision is
           attached directly to the most recent transformation. This
           handles the "trailing decision" case where a decision is
           recorded AFTER the artifact's last transformation completes
           (e.g., effective-resolution warning recorded after BUILD_GRID),
           which the forward-only pending-list mechanism cannot handle.

        2. **Pending-list (V2-19 A1)** — when no explicit ``artifact_id``
           is provided, OR the named artifact does not yet exist, OR the
           artifact exists but has no transformations yet, the decision is
           queued in the pending list along with its bound artifact ID
           (bound at record-time). The next ``record_retrieval`` /
           ``record_transformation`` call against the bound artifact
           drains the pending list into that transformation. This handles
           the "leading decision" case (e.g., iSDA cascade source
           selection recorded before the soil retrieval transformation).

        Both paths preserve the V2-19 A1 invariant: a decision's binding
        is determined at record-time, never silently reassigned later.

        Args:
            decision_type: Type of decision
            description: What was decided
            rationale: Why this decision was made
            alternatives: Other options considered
            reference: Citation or documentation link
            artifact_id: Explicit artifact ID. When provided and the
                artifact has at least one transformation, attaches
                directly to the most recent transformation. Otherwise
                falls back to the pending-list path with this ID as
                the binding (or ``_current_artifact_id`` if no explicit
                ID is given).

        Returns:
            The created DecisionRecord (for backward compatibility —
            callers no longer need to do anything with the return value).
        """
        # Improvement 1: early-return when tracking is disabled
        if not self.enabled:
            return DecisionRecord(
                decision_type=decision_type,
                description=description,
                rationale=rationale,
                alternatives_considered=alternatives or [],
                reference=reference,
            )

        decision = DecisionRecord(
            decision_type=decision_type,
            description=description,
            rationale=rationale,
            alternatives_considered=alternatives or [],
            reference=reference,
        )

        # V2-19b-fix Finding 4: EXPLICIT path (direct-attach or fail-fast).
        #
        # When the caller passes an explicit artifact_id, the semantic is
        # STRICT: land the decision or fail loudly. Silent fallback to
        # the pending list is the exact soft-failure mode that hid
        # Finding 4 in V2-19a — never repeat it for the explicit case.
        #
        # (a) Artifact exists + has ≥1 transformation → direct-attach to
        #     the most recent transformation.
        # (b) Artifact exists + has 0 transformations → ProvenanceStateError
        #     (caller should record_transformation first, or use the
        #     implicit pending-list path instead).
        # (c) Artifact doesn't exist → ProvenanceStateError (caller
        #     should start_artifact first).
        #
        # The lock protects both the existence-check and the append from
        # races with concurrent record_transformation calls.
        if artifact_id is not None:
            with self._lock:
                existing = self.record.get_artifact(artifact_id)
                if existing is None:
                    # Case (c): artifact doesn't exist — true caller error.
                    raise ProvenanceStateError(
                        f"record_decision called with explicit artifact_id="
                        f"'{artifact_id}' but no artifact with that ID exists. "
                        f"Call start_artifact('{artifact_id}') first, or omit "
                        f"the artifact_id parameter to use the implicit "
                        f"pending-list path."
                    )
                if existing.transformations:
                    # Case (a): trailing decision — direct-attach to the
                    # most recent transformation. No future transformation
                    # is expected to drain this from the pending list.
                    existing.transformations[-1].decisions.append(decision)
                    self.logger.info(
                        "Decision recorded (direct-attach to %s): %s",
                        artifact_id,
                        description,
                    )
                    return decision
                # Case (b): leading decision — artifact exists but has no
                # transformations yet. Fall through to pending-list path
                # with the explicit artifact_id as the binding. The next
                # record_retrieval / record_transformation against this
                # artifact will drain it. This is the normal forward-
                # binding pattern (e.g., climate SOURCE_SELECTION recorded
                # after start_artifact("climate") but before the climate
                # retrieval transformation).

        # IMPLICIT path (V2-19 A1 pending-list, unchanged):
        # No explicit artifact_id → bind to _current_artifact_id at
        # record-time to prevent misattribution if the pointer moves.
        bound_artifact = artifact_id or self._current_artifact_id

        with self._lock:
            self._pending_decisions.append((decision, bound_artifact))
            # Sanity cap: warn ONCE when crossing the threshold (not per append)
            if len(self._pending_decisions) == 51:
                self.logger.warning(
                    "Pending decisions list has 50+ entries — is a "
                    "record_transformation call missing?"
                )

        self.logger.info("Decision recorded: %s", description)
        return decision

    def _drain_pending_for_artifact(
        self,
        artifact_id: str,
        caller_decisions: Optional[List[DecisionRecord]] = None,
    ) -> List[DecisionRecord]:
        """Drain pending decisions bound to ``artifact_id``.

        Decisions bound to a different artifact remain in the pending list.
        The caller's explicit ``decisions`` parameter (if any) is merged
        with the drained pending decisions — explicit decisions come AFTER
        pending ones (pending-first, explicit-second).

        Improvement 6: fail-fast on unattached decisions mid-stream.
        If there are pending decisions with a None artifact binding AND
        the current artifact is also None, something is wrong (decision
        recorded before any ``start_artifact`` call).
        """
        with self._lock:
            drained: List[DecisionRecord] = []
            remaining: List[Tuple[DecisionRecord, Optional[str]]] = []
            orphan_count = 0
            for decision, bound_id in self._pending_decisions:
                if bound_id is None or bound_id == artifact_id:
                    drained.append(decision)
                    if bound_id is None:
                        orphan_count += 1
                else:
                    remaining.append((decision, bound_id))
            self._pending_decisions = remaining

        # Improvement 6: strict enforcement — if we drained orphans while
        # there was no current artifact when the decision was recorded,
        # the caller forgot to start_artifact() before recording. Fail
        # fast so the bug is visible, not silently buried in output.
        #
        # Exception: finalize() intentionally drains orphans into the
        # synthetic "pipeline" artifact — it passes a sentinel caller
        # flag by invoking this method only via _flush_all_for_finalize.
        # That path is handled in finalize() itself.

        result = list(drained)
        if caller_decisions:
            result.extend(caller_decisions)
        return result

    def finalize(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Finalize the tracker — flush any remaining pending decisions.

        Improvement 3 + 4: creates a synthetic ``pipeline`` artifact for
        decisions that were recorded but never attached to a retrieval or
        transformation (tail stragglers — e.g., a decision recorded AFTER
        the last transformation completes but BEFORE the pipeline exits).

        This is a complementary catch to improvement 6 (which enforces
        strict ordering DURING normal stage execution). finalize()
        catches the tail, improvement 6 catches the middle.

        The caller should call this in a ``try/finally`` block wrapping
        ``pipeline.execute()`` so it fires even on exception paths.

        Args:
            output_path: Optional path for an incremental checkpoint save
                at finalize time (belt-and-suspenders).
        """
        if not self.enabled:
            return

        with self._lock:
            if self._pending_decisions:
                # Create a synthetic pipeline artifact to home the stragglers
                synthetic_id = f"pipeline_{self.session_id}_finalize"
                if synthetic_id not in self.record.artifacts:
                    synthetic = DataLineage(
                        artifact_id=synthetic_id,
                        artifact_type="pipeline",
                    )
                    self.record.add_artifact(synthetic)

                synthetic_lineage = self.record.get_artifact(synthetic_id)
                straggler_decisions = [d for d, _ in self._pending_decisions]
                self._pending_decisions = []  # drained

                synthetic_lineage.add_transformation(
                    TransformationRecord(
                        operation=OperationType.VALIDATE,
                        parameters={
                            "stage": "finalize",
                            "n_stragglers": len(straggler_decisions),
                        },
                        decisions=straggler_decisions,
                    )
                )
                self.logger.info(
                    "finalize(): flushed %d straggler decisions into synthetic "
                    "pipeline artifact",
                    len(straggler_decisions),
                )

        if output_path:
            self.save(output_path)

    def checkpoint_save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Incremental checkpoint save at a stage boundary.

        Improvement 7: belt-and-suspenders for SIGKILL/Django timeout
        protection. Callers (typically the pipeline executor) invoke this
        at each stage boundary so partial provenance survives a crash.

        The save is a no-op if ``enabled`` is False. Errors are logged
        but not re-raised — checkpoint saves must never crash the pipeline.
        """
        if not self.enabled:
            return
        try:
            path = output_path or self._checkpoint_path
            if path:
                self.save(path)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Checkpoint save failed (non-fatal): %s", exc)

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
        """Save provenance record to file(s).

        V2-19: emits BOTH formats side-by-side:
        - ``<name>.json`` — native System A artifact-lineage format (rich)
        - ``<name>_stages.json`` — auto-derived System B compat format
          (flat stages by OperationType), for legacy consumers

        Safety net: if there are unattached pending decisions at save time
        (e.g., caller forgot to call finalize()), they are dumped into an
        ``unattached_decisions`` top-level field in the output JSON with
        a WARNING log. This catches bugs in future code that records
        decisions without a subsequent transform.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to the primary (rich) provenance file
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
            # Safety net: if there are pending decisions, dump them into
            # the output as unattached_decisions so nothing is silently lost
            rich_dict = self.record.to_dict()
            with self._lock:
                if self._pending_decisions:
                    self.logger.warning(
                        "save() called with %d unflushed pending decisions — "
                        "dumping to 'unattached_decisions' field. Caller "
                        "likely forgot to call finalize().",
                        len(self._pending_decisions),
                    )
                    rich_dict["unattached_decisions"] = [
                        {**d.to_dict(), "bound_artifact_id": aid}
                        for d, aid in self._pending_decisions
                    ]

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(rich_dict, f, indent=2, default=str)
            self.logger.info(f"Saved provenance to {save_path}")

            # V2-19: emit stages compat file alongside
            stages_path = save_path.with_name(
                save_path.stem + "_stages" + save_path.suffix
            )
            stages_dict = self._derive_stages_format(rich_dict)
            with open(stages_path, "w", encoding="utf-8") as f:
                json.dump(stages_dict, f, indent=2, default=str)
            self.logger.info(f"Saved stages-compat provenance to {stages_path}")

        return save_path

    def _derive_stages_format(self, rich_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-derive the System B stages format from System A artifacts.

        Walks all artifacts' transformations, buckets them by
        ``OperationType`` → System B stage name, and emits a flat
        stages-list. Decisions recorded on artifact X during a RETRIEVE
        transformation land in the "retrieve" stage regardless of
        which artifact they came from.

        Stage mapping (OperationType → stage name):
        - RETRIEVE → "retrieve"
        - AGGREGATE, BUILD_GRID, RESAMPLE, REPROJECT, CONVERT_UNITS,
          INTERPOLATE, GAP_FILL, QUALITY_CHECK → "harmonize"
        - TRANSLATE → "translate"
        - VALIDATE → "validate"

        The output shape matches what ``packaging/provenance.py`` wrote
        historically, so VS-01 and other legacy readers continue to work.
        """
        stage_mapping = {
            OperationType.RETRIEVE: "retrieve",
            OperationType.AGGREGATE: "harmonize",
            OperationType.BUILD_GRID: "harmonize",
            OperationType.RESAMPLE: "harmonize",
            OperationType.REPROJECT: "harmonize",
            OperationType.CONVERT_UNITS: "harmonize",
            OperationType.INTERPOLATE: "harmonize",
            OperationType.GAP_FILL: "harmonize",
            OperationType.QUALITY_CHECK: "harmonize",
            OperationType.TRANSLATE: "translate",
            OperationType.VALIDATE: "validate",
        }

        # Accumulate per-stage decisions + artifact references
        stage_buckets: Dict[str, Dict[str, Any]] = {}
        for artifact_id, artifact_dict in rich_dict.get("artifacts", {}).items():
            artifact_type = artifact_dict.get("artifact_type", "unknown")
            for transform in artifact_dict.get("transformations", []):
                op_str = transform.get("operation", "")
                try:
                    op = OperationType(op_str)
                except ValueError:
                    op = OperationType.VALIDATE  # unknown ops go to validate bucket
                stage_name = stage_mapping.get(op, "harmonize")

                bucket = stage_buckets.setdefault(
                    stage_name,
                    {
                        "stage": stage_name,
                        "inputs": [],
                        "outputs": [],
                        "decisions": [],
                        "warnings": [],
                    },
                )
                # Track which artifacts participated in this stage
                if artifact_type not in bucket["inputs"]:
                    bucket["inputs"].append(artifact_type)
                bucket["decisions"].extend(transform.get("decisions", []))
                bucket["warnings"].extend(transform.get("warnings", []))

        # Emit stages in canonical order
        canonical_order = ["retrieve", "harmonize", "translate", "validate"]
        stages_list = [
            stage_buckets[name]
            for name in canonical_order
            if name in stage_buckets
        ]

        return {
            "workflow": "prismpy",
            "session_id": rich_dict.get("session_id"),
            "created_at": rich_dict.get("created_at"),
            "project_name": rich_dict.get("project_name"),
            "stages": stages_list,
            "summary": rich_dict.get("summary", {}),
        }

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
