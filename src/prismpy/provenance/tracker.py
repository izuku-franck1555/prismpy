"""
Provenance tracking system for prismpy.

This module provides the central tracking system for all data handling
decisions, implementing the 'formalized methodology' requirement.
"""

import copy
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
        stage: Optional[str] = None,
    ) -> str:
        """Start tracking a new data artifact.

        Args:
            artifact_type: Type of artifact (e.g., "climate", "soil")
            source_artifacts: IDs of source artifacts
            artifact_id: Optional custom artifact ID
            stage: Pipeline stage (retrieve/harmonize/translate/validate)
                for timeline grouping in the UI

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
            stage=stage,
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
        alternatives: Optional[List] = None,
        reference: Optional[str] = None,
        artifact_id: Optional[str] = None,
        severity: str = "info",
        label: Optional[str] = None,
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
                severity=severity,
                label=label,
            )

        decision = DecisionRecord(
            decision_type=decision_type,
            description=description,
            rationale=rationale,
            alternatives_considered=alternatives or [],
            reference=reference,
            severity=severity,
            label=label,
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

    def set_boundary(
        self,
        *,
        source: str,
        version: Optional[str],
        inclusion_rule: str,
        min_share_percent: float,
        n_cells_full_extent: int,
        n_cells_excluded_by_inclusion_rule: int,
        n_cells_excluded_by_min_share_percent: int,
        n_cells_admitted: int,
    ) -> None:
        """F-R AC-4: record the boundary-rule decisions that
        produced the simulation domain.

        Called once per run by the HARMONIZE-stage filter
        (AC-2 Stage 5) once the canonical filtered grid is
        finalized. Writes 8 fields onto ``record.boundary``;
        the per-platform ``shutil.copy2`` step at
        ``executor.py:2997-3011`` carries them into each
        platform_dir/provenance.json copy automatically.

        Keyword-only signature so the call-site reads as a
        named-fields contract (mirrors the cell-count
        arithmetic invariant `n_cells_full_extent =
        n_cells_excluded_by_inclusion_rule +
        n_cells_excluded_by_min_share_percent +
        n_cells_admitted + n_cells_user_excluded`; the
        last term is computed at AC-7 invariant test time
        from ``config.region.exclude_cells``).

        Args:
            source: ``BoundaryConfig.source.value`` (e.g.,
                ``"gadm"``, ``"manual"``, ``"shapefile"``).
            version: e.g., ``"GADM v4.1"``; ``None`` when the
                source has no version concept.
            inclusion_rule: ``Literal['bbox_intersects',
                'centroid_strict']`` from BoundaryConfig.
            min_share_percent: float ∈ [0.0, 100.0] from
                BoundaryConfig.
            n_cells_full_extent: cells in raw extent before
                any filter (informational; supports the AC-7
                arithmetic invariant).
            n_cells_excluded_by_inclusion_rule: cells trimmed
                by the inclusion_rule filter (always 0 when
                ``inclusion_rule='bbox_intersects'``).
            n_cells_excluded_by_min_share_percent: cells
                trimmed by the SP threshold filter (always 0
                when ``min_share_percent=0.0``).
            n_cells_admitted: final cell count after
                inclusion_rule + threshold + user-skip;
                equals ``cell_summary.n_cells`` at
                run-finalization.
        """
        if not self.enabled:
            return

        self.record.boundary = {
            "source": source,
            "version": version,
            "inclusion_rule": inclusion_rule,
            "min_share_percent": min_share_percent,
            "n_cells_full_extent": n_cells_full_extent,
            "n_cells_excluded_by_inclusion_rule": n_cells_excluded_by_inclusion_rule,
            "n_cells_excluded_by_min_share_percent": n_cells_excluded_by_min_share_percent,
            "n_cells_admitted": n_cells_admitted,
        }

    def record_texture_renormalization(
        self,
        provenance_entry: Any,
    ) -> None:
        """Sprint D.1 AC-6 — record one texture-fraction
        renormalization at the harmonize stage. The argument is
        either a
        :class:`prismpy.harmonize.texture_renormalize.TextureRenormalizationProvenance`
        Pydantic instance or an already-serialized dict; the
        record stores the dict form so the provenance JSON
        round-trips cleanly.
        """
        if not self.enabled:
            return
        entry = (
            provenance_entry.model_dump()
            if hasattr(provenance_entry, "model_dump")
            else dict(provenance_entry)
        )
        self.record.texture_renormalize_details.append(entry)

    def record_rh_clip(
        self,
        provenance_entry: Any,
    ) -> None:
        """Sprint D.1 AC-6 — record one rh clip at the harmonize
        stage. The argument is either a
        :class:`prismpy.harmonize.rh_clip.RHClipProvenance`
        Pydantic instance or an already-serialized dict.
        """
        if not self.enabled:
            return
        entry = (
            provenance_entry.model_dump()
            if hasattr(provenance_entry, "model_dump")
            else dict(provenance_entry)
        )
        self.record.rh_clip_details.append(entry)

    def record_cell_unavailable(
        self,
        cell_id: int,
        unavailable_reason: str,
        unavailable_cause: Optional[str] = None,
    ) -> None:
        """Sprint D.1 AC-6 — record one cell routed to
        ``data_availability='unavailable'`` at the harmonize
        stage. The detail list captures the cell id plus the
        axis (``unavailable_reason``) and cause
        (``unavailable_cause``) so the consumer can replay the
        routing per cell.
        """
        if not self.enabled:
            return
        self.record.cells_unavailable_details.append(
            {
                "cell_id": cell_id,
                "unavailable_reason": unavailable_reason,
                "unavailable_cause": unavailable_cause,
            }
        )

    def record_pythia_misdat_replacement(
        self,
        translator: str = "pythia",
        count: int = 1,
    ) -> None:
        """Sprint D.1 AC-6 — track records that wrote the DSSAT
        MISDAT sentinel for missing rain. Currently the count
        is keyed by translator name; PYTHIA is the only target
        for the AC-2 fix today, but the keyed shape leaves room
        for a future translator to surface its own MISDAT
        statistics without changing the field shape.
        """
        if not self.enabled:
            return
        self.record.pythia_misdat_replacements[translator] = (
            self.record.pythia_misdat_replacements.get(translator, 0) + count
        )

    def record_stage_1_verdicts(
        self,
        snapshot: Dict[str, Any],
    ) -> None:
        """Sprint F AC-F-7 — attach a Stage 1 verdict snapshot
        to the provenance record at pipeline start.

        The wizard caller computes the snapshot at project
        creation (the cached ``Project.stage_1_verdicts``
        JSONField shape) and passes it here so the run's
        provenance.json carries the per-(crop, zone) verdict
        plus the substrate-version stamps the verdict was
        computed against. The snapshot is the same shape AC-F-5
        defines for the cache + AC-F-9 cockpit-readiness pin
        consumes.

        Args:
            snapshot: Dict matching the AC-F-5 schema
                (``{"schema_version", "cache_key", "created_at",
                "entries", "substrate_versions"}``). Pass the
                FULL dict, not just the entries list, so a
                consumer can read the substrate stamps without
                also reading the cached entries.

        Stored on
        :attr:`ProvenanceRecord.stage_1_verdicts_snapshot`;
        serialized into ``provenance.json`` per
        :meth:`ProvenanceRecord.to_dict`.
        """
        if not self.enabled:
            return
        # Defensive deep-copy so a downstream caller mutating
        # the passed-in dict (or any of its nested lists /
        # dicts) cannot retroactively alter the recorded
        # snapshot. AC-F-7 + codex Gate A #14 — the snapshot
        # is the audit trail of which verdict shape produced
        # the run; a shallow copy would let entries[] mutation
        # leak through.
        self.record.stage_1_verdicts_snapshot = copy.deepcopy(snapshot)

    def set_eghr_substrate_decision(
        self,
        decision: str,
        reason: str,
    ) -> None:
        """Sprint S Gate-B-FIX — record the eGHR substrate dispatch.

        Called by :class:`prismpy.translators.pythia.translator.PythiaTranslator`
        from inside ``_include_eghr_data(data)`` after the canonical-
        vs-legacy dispatch decision is taken. Writes both the
        machine-readable decision (``"canonical"`` or
        ``"legacy_bundled"``) and the reason code (``"ok"``,
        ``"disabled_via_flag"``, ``"disabled_via_env"``,
        ``"inputs_unavailable"``) to dedicated top-level fields on
        :class:`prismpy.models.provenance.ProvenanceRecord` so
        downstream consumers — the AC-8 reproduction snippet, the
        evaluator's Gate B verifier, and Dr. Kofi's grep-the-package
        workflow — read an unambiguous binary signal.

        Per durable §24 canonical-source-or-pin: the field IS the
        source of truth for "did the canonical substrate path run
        on this package". Consumers MUST NOT re-derive the
        dispatch decision from secondary signals (e.g., presence
        of ``eGHR/{CC}.SOL``, absence of "SOL file not found"
        warnings, raster-vs-database row count); inferring is
        exactly the detective work that produced the false-PASS
        on the b5fb6538 evaluator run.

        Args:
            decision: ``"canonical"`` or ``"legacy_bundled"``. The
                receiver MUST validate this enum on read; a value
                outside the two accepted strings indicates a
                producer/consumer drift (durable §24 / Sprint S
                two-vocabulary observational memo).
            reason: One of ``"ok"`` / ``"disabled_via_flag"`` /
                ``"disabled_via_env"`` / ``"inputs_unavailable"``.

        Returns:
            None. The record is mutated in place. Callers should
            not rely on ordering guarantees with respect to other
            ``record_*`` / ``set_*`` calls; idempotent overwrites
            are safe (last write wins).
        """
        if not self.enabled:
            return
        # Validate the enum at the schema boundary per durable §6.4
        # schema-layer discipline. A bad call site fails loud here
        # rather than persisting a junk value that the AC-8 snippet
        # later struggles to interpret.
        accepted_decisions = {"canonical", "legacy_bundled"}
        if decision not in accepted_decisions:
            raise ValueError(
                f"eghr_substrate_decision must be one of {sorted(accepted_decisions)}; "
                f"got {decision!r}"
            )
        accepted_reasons = {
            "ok",
            "disabled_via_flag",
            "disabled_via_env",
            "inputs_unavailable",
        }
        if reason not in accepted_reasons:
            raise ValueError(
                f"eghr_substrate_reason must be one of {sorted(accepted_reasons)}; "
                f"got {reason!r}"
            )
        self.record.eghr_substrate_decision = decision
        self.record.eghr_substrate_reason = reason

    def record_wizard_decision(
        self,
        record_payload: Union[Dict[str, Any], "WizardOverrideRecord"],
        artifact_id: Optional[str] = None,
    ) -> None:
        """Sprint F AC-F-6 — replay a saved wizard-time
        override into the run's provenance trail.

        The wizard-time override is captured BEFORE the
        pipeline starts (no tracker exists yet) and persisted
        by prismweb in ``Project.wizard_decisions``. At
        pipeline-start, the saved payload is replayed here:
        the helper validates the payload via
        :class:`prismpy.provenance.WizardOverrideRecord` (or
        accepts an already-typed record), then routes it
        through :meth:`record_decision` with
        ``DecisionType.USER_OVERRIDE``.

        A single validated entry-point keeps the prismweb
        caller side simple — the form-validation + the
        pipeline-start replay both go through the same
        :class:`WizardOverrideRecord` shape rather than
        duplicating field rules across repos.

        Args:
            record_payload: Either a
                :class:`WizardOverrideRecord` instance or the
                JSON-mode dict it serializes to. The dict is
                validated through Pydantic on entry so a
                malformed payload fails-loud instead of
                drifting into provenance.
            artifact_id: Optional artifact binding. When
                provided + the artifact already has at least
                one transformation, the decision attaches
                directly; otherwise it lands on the pending
                list per the existing :meth:`record_decision`
                semantics.
        """
        if not self.enabled:
            return
        # Local import keeps the module-import surface lean;
        # WizardOverrideRecord is an opt-in consumer per the
        # existing record_bound_gen_provenance pattern.
        from prismpy.provenance.wizard_decisions import (
            WizardOverrideRecord,
            build_wizard_override_payload,
        )
        if isinstance(record_payload, WizardOverrideRecord):
            record = record_payload
        else:
            # Will raise pydantic.ValidationError on malformed
            # payload — caller's responsibility to catch +
            # surface to the user.
            record = WizardOverrideRecord.model_validate(record_payload)
        payload = build_wizard_override_payload(record)
        # The structured payload lands in description /
        # rationale / reference per the existing DecisionRecord
        # shape; the verdict_hash + evidence_type + zones land
        # in the rationale free-text per builder Adj-12 (V2-23
        # polish extends DecisionRecord with first-class
        # structured fields).
        zones = ", ".join(record.affected_zones)
        rationale_full = (
            f"User override on Stage 1 verdict for zones "
            f"[{zones}]. Evidence type: {record.evidence_type}. "
            f"Rationale: {record.rationale} "
            f"verdict_hash={record.verdict_hash}"
        )
        if record.evidence_url:
            rationale_full += f" evidence_url={record.evidence_url}"
        if record.methodology_paper_doi:
            rationale_full += (
                f" methodology_paper_doi="
                f"{record.methodology_paper_doi}"
            )
        # Sprint E.1 codex MEDIUM 2 absorption — stamp the
        # wizard-time discriminator on the rationale verbatim
        # so audit-grep against ``override_at_pre_pipeline=True``
        # surfaces every wizard-time entry. The cockpit-time
        # path stamps ``override_at_pre_pipeline=False`` in
        # ``record_cockpit_decision`` below; the two markers
        # are byte-disjoint so a regex grep splits the audit
        # cleanly.
        rationale_full += " override_at_pre_pipeline=True"
        self.record_decision(
            decision_type=DecisionType.USER_OVERRIDE,
            description=(
                f"Wizard-time override for affected zones "
                f"[{zones}]"
            ),
            rationale=rationale_full,
            reference=(
                record.evidence_url or record.methodology_paper_doi
            ),
            artifact_id=artifact_id,
            severity="warning",
            label="user_override",
        )

    def record_cockpit_decision(
        self,
        decision_type: "DecisionType",
        category: str,
        bucket: int,
        affected_cells: Optional[List[str]] = None,
        affected_zones: Optional[List[str]] = None,
        rationale: Optional[str] = None,
        evidence_type: Optional[str] = None,
        evidence_type_other_specify: Optional[str] = None,
        evidence_url: Optional[str] = None,
        methodology_paper_doi: Optional[str] = None,
        verdict_hash: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> None:
        """Sprint E.1 AC-E1-0 — replay a cockpit-time decision
        into the run's provenance trail.

        Mirror of :meth:`record_wizard_decision` (Sprint F AC-F-6)
        for the three cockpit-time decision-types per the
        :data:`WARNING_BUCKET_MAP` semantics:

        * :data:`DecisionType.USER_ACKNOWLEDGE` — Bucket 2 INFO
          "I've read this and I'm proceeding" affirmation.
        * :data:`DecisionType.USER_SKIP` — Bucket 3 EXCLUDE
          "exclude these cells from the next run" decision.
        * :data:`DecisionType.USER_OVERRIDE` — Bucket 5 cockpit-
          time override. Distinct from the wizard-time path
          (which uses :meth:`record_wizard_decision`) by the
          ``override_at_pre_pipeline=False`` discriminator
          stamped onto the rationale; the wizard path stamps
          ``override_at_pre_pipeline=True``.

        The single validated entry-point matches the
        :meth:`record_wizard_decision` precedent — caller +
        replay path both route through this helper rather than
        constructing :meth:`record_decision` payloads ad-hoc.

        Args:
            decision_type: One of ``USER_ACKNOWLEDGE``,
                ``USER_SKIP``, ``USER_OVERRIDE``. Other values
                raise :class:`ValueError` so a typo'd caller
                fails loud rather than landing as a
                generic decision.
            category: Lowercase :class:`WarningCategory` enum
                value (e.g., ``"climate_envelope_tail"``,
                ``"crop_region_mismatch"``). Pinned at the
                canonical-string level per F25.
            bucket: Bucket number from
                :data:`WARNING_BUCKET_MAP` — 2, 3, or 5. Other
                values raise :class:`ValueError`.
            affected_cells: Cell-id list for
                ``USER_SKIP`` / ``USER_ACKNOWLEDGE``. Required
                non-empty for those types.
            affected_zones: Zone-code list for
                ``USER_OVERRIDE``. Required non-empty for that
                type per :class:`WizardOverrideRecord` parity.
            rationale: Required free-form text for
                ``USER_OVERRIDE``; optional for
                ``USER_ACKNOWLEDGE``/``USER_SKIP``. The
                ``WizardOverrideRecord`` ≥50-char filler-
                rejection enforced for ``USER_OVERRIDE`` per
                wizard-parity (CC-33).
            evidence_type: Required for ``USER_OVERRIDE``; one
                of the 6 :data:`EvidenceType` values
                (``local_trial`` / ``irrigation`` /
                ``cultivar_specific`` / ``citation`` /
                ``field_observation`` / ``other``).
            evidence_type_other_specify: Required (non-empty
                after trim) when ``evidence_type == "other"``;
                must be ``None`` otherwise. Mirrors the
                :class:`WizardOverrideRecord` conditional-
                required validator.
            evidence_url: Optional citation URL (must be
                ``https://``).
            methodology_paper_doi: Optional DOI.
            verdict_hash: 64-char SHA-256 hex pinning the
                Stage 1 verdict snapshot at decision time —
                required for ``USER_OVERRIDE``.
            artifact_id: Optional artifact binding mirroring
                :meth:`record_decision` semantics.

        The cockpit-time discriminator (``override_at_pre_pipeline=False``)
        is hardcoded on the rationale here — callers cannot
        override it. The wizard-time
        :meth:`record_wizard_decision` stamps the
        complementary ``True`` marker, so audit-grep against
        either pattern surfaces only the matching path. Per
        codex MEDIUM 2.

        Raises:
            ValueError: On unsupported decision_type, bad
                bucket, or missing required field per the
                decision-type contract above.
        """
        if not self.enabled:
            return
        # Local import keeps the module-import surface lean;
        # WarningCategory + WizardOverrideRecord are opt-in
        # consumers per the existing pattern.
        from prismpy.warnings.categories import (
            WARNING_BUCKET_MAP,
            WarningCategory,
            WarningBucket,
        )
        # Validate the decision_type early — typed-enum on the
        # caller side is the contract; if a stringly-typed
        # caller slips through, fail loud.
        allowed = (
            DecisionType.USER_ACKNOWLEDGE,
            DecisionType.USER_SKIP,
            DecisionType.USER_OVERRIDE,
        )
        if decision_type not in allowed:
            raise ValueError(
                f"record_cockpit_decision only accepts "
                f"USER_ACKNOWLEDGE / USER_SKIP / USER_OVERRIDE; "
                f"got {decision_type!r}."
            )
        # Codex MEDIUM 1 absorption — cross-check that the
        # category resolves to a WarningCategory enum value
        # AND that ``WARNING_BUCKET_MAP[category]`` matches
        # the supplied bucket integer. Without this, a typo'd
        # caller could silently record an acknowledge against
        # a category whose canonical bucket is 3 (TRUE_EXCLUDE),
        # corrupting the audit-trail's bucket-affordance
        # invariant.
        try:
            category_enum = WarningCategory(category)
        except ValueError as exc:
            raise ValueError(
                f"category {category!r} is not a "
                f"WarningCategory enum value; reconcile with "
                f"prismpy.warnings.categories per F25."
            ) from exc
        canonical_bucket_enum = WARNING_BUCKET_MAP[category_enum]
        # Map the WarningBucket enum member to the integer the
        # caller supplied. The integer surface is the cockpit-
        # facing convention (bucket 2 / 3 / 5); WarningBucket
        # is the substrate enum.
        _bucket_int_map = {
            WarningBucket.AUTO_FIXABLE: 0,
            WarningBucket.INFORMATIONAL: 2,
            WarningBucket.TRUE_EXCLUDE: 3,
            WarningBucket.INTERPOLATABLE: 4,
            WarningBucket.MANUAL_OVERRIDE_WITH_EVIDENCE: 5,
        }
        canonical_bucket_int = _bucket_int_map[canonical_bucket_enum]
        if bucket not in (2, 3, 5):
            raise ValueError(
                f"bucket must be 2 / 3 / 5 per "
                f"WARNING_BUCKET_MAP; got {bucket!r}."
            )
        if bucket != canonical_bucket_int:
            raise ValueError(
                f"bucket {bucket} does not match the canonical "
                f"bucket for category {category!r} "
                f"(WARNING_BUCKET_MAP resolves to "
                f"{canonical_bucket_int}). The cockpit caller "
                f"must pass the bucket integer matching the "
                f"category's WARNING_BUCKET_MAP entry."
            )
        affected_cells = list(affected_cells or [])
        affected_zones = list(affected_zones or [])
        if decision_type is DecisionType.USER_OVERRIDE:
            # Codex HIGH 2 absorption — route override
            # validation through WizardOverrideRecord so the
            # cockpit-time path inherits the same Pydantic
            # invariants the wizard-time path enforces:
            # filler-rejection, https-only evidence_url,
            # 64-char hex verdict_hash, conditional
            # evidence_type_other_specify pairing. Without
            # this, the cockpit caller could persist
            # malformed-rationale or non-hex-verdict_hash
            # decisions that the wizard would reject.
            from prismpy.provenance.wizard_decisions import (
                WizardOverrideRecord,
            )
            try:
                # The Pydantic validators raise
                # ValidationError; surface it as ValueError
                # so the caller's exception-handling stays
                # uniform. WizardOverrideRecord is the
                # single source of truth for override-shape
                # invariants.
                _validated = WizardOverrideRecord(
                    rationale=rationale or "",
                    evidence_type=evidence_type or "",
                    evidence_type_other_specify=evidence_type_other_specify,
                    affected_zones=affected_zones,
                    verdict_hash=verdict_hash or "",
                    evidence_url=evidence_url,
                    methodology_paper_doi=methodology_paper_doi,
                )
            except Exception as exc:
                raise ValueError(
                    f"USER_OVERRIDE rejected by "
                    f"WizardOverrideRecord validators: {exc}"
                ) from exc
            # Mirror the validated values — the model may have
            # normalized whitespace (e.g., the
            # ``_strip_other_specify`` validator).
            evidence_type = _validated.evidence_type
            evidence_type_other_specify = (
                _validated.evidence_type_other_specify
            )
            rationale = _validated.rationale
            verdict_hash = _validated.verdict_hash
            evidence_url = _validated.evidence_url
            methodology_paper_doi = _validated.methodology_paper_doi
            affected_zones = list(_validated.affected_zones)
        else:
            # USER_ACKNOWLEDGE / USER_SKIP require
            # affected_cells.
            if not affected_cells:
                raise ValueError(
                    f"{decision_type.value} requires "
                    f"non-empty affected_cells."
                )
        # Build the rationale free-text payload per the existing
        # DecisionRecord shape. The cockpit-time discriminator
        # ``override_at_pre_pipeline=False`` is stamped verbatim
        # so audit-grep can split wizard-time vs cockpit-time.
        cells_part = (
            f"affected_cells={len(affected_cells)} "
            if affected_cells
            else ""
        )
        zones_part = (
            f"affected_zones=[{', '.join(affected_zones)}] "
            if affected_zones
            else ""
        )
        rationale_full = (
            f"Cockpit-time {decision_type.value} on bucket "
            f"{bucket} category {category}. {cells_part}{zones_part}"
        )
        if rationale:
            rationale_full += f"Rationale: {rationale} "
        if evidence_type:
            rationale_full += f"evidence_type={evidence_type} "
        if evidence_type_other_specify:
            rationale_full += (
                f"evidence_type_other_specify={evidence_type_other_specify} "
            )
        if verdict_hash:
            rationale_full += f"verdict_hash={verdict_hash} "
        if evidence_url:
            rationale_full += f"evidence_url={evidence_url} "
        if methodology_paper_doi:
            rationale_full += (
                f"methodology_paper_doi={methodology_paper_doi} "
            )
        # Codex MEDIUM 2 — discriminator hardcoded ``False``
        # for the cockpit-time path; the wizard-time
        # ``record_wizard_decision`` stamps ``True`` so
        # audit-grep splits cleanly.
        rationale_full += "override_at_pre_pipeline=False"
        self.record_decision(
            decision_type=decision_type,
            description=(
                f"Cockpit decision: {decision_type.value} on "
                f"bucket {bucket} ({category})"
            ),
            rationale=rationale_full,
            reference=evidence_url or methodology_paper_doi,
            artifact_id=artifact_id,
            severity=(
                "warning"
                if decision_type is DecisionType.USER_OVERRIDE
                else "info"
            ),
            label=decision_type.value,
        )

    def record_bound_gen_provenance(
        self,
        provenance: "BoundGenProvenance",
        output_path: Union[str, Path],
    ) -> Path:
        """Sprint E.0.5 AC-Q2-A1-c — record a bound-gen run.

        Writes a sidecar JSON file at ``output_path`` describing
        the environment + ERA5 archive metadata + dependency +
        thread-pin configuration of the bound-gen run that
        produced the bounds file. Sprint-level provenance — not
        attached to a per-artifact lineage — so the bound-gen
        management command can call this once per run regardless
        of whether the prismpy session has an active artifact.

        See :class:`prismpy.bounds.BoundGenProvenance` for the
        full schema (including the deposit-status conjunction
        validator) and
        :func:`prismpy.bounds.write_bound_gen_provenance` for
        the underlying serialization helper.

        Returns the absolute path of the written file.
        """
        # Local import keeps prismpy.provenance free of a hard
        # dependency on prismpy.bounds at module import time;
        # the bound-gen substrate is an opt-in consumer.
        from prismpy.bounds import write_bound_gen_provenance
        if not self.enabled:
            return Path(output_path).resolve()
        return write_bound_gen_provenance(provenance, output_path)

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
                json.dump(rich_dict, f, indent=2, default=str, ensure_ascii=False)
            self.logger.info(f"Saved provenance to {save_path}")

            # V2-19: emit stages compat file alongside
            stages_path = save_path.with_name(
                save_path.stem + "_stages" + save_path.suffix
            )
            stages_dict = self._derive_stages_format(rich_dict)
            with open(stages_path, "w", encoding="utf-8") as f:
                json.dump(stages_dict, f, indent=2, default=str, ensure_ascii=False)
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
