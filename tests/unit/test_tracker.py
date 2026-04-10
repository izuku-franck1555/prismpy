"""
Unit tests for ProvenanceTracker (V2-19 API repair).

These tests verify the 7 pattern improvements in A1:
1. enabled=False early-return
2. Artifact-id binding at record-time (not flush-time)
3. try/finally finalize() behavior
4. Synthetic pipeline artifact for stragglers
5. (Thread-safety docstring — not unit-testable, verified via code-read)
6. Fail-fast ProvenanceStateError on unattached decisions mid-stream
7. Incremental checkpoint saves

Plus enum completeness + dual-output format verification.
"""

import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prismpy.models.provenance import DecisionType, OperationType
from prismpy.provenance.tracker import ProvenanceStateError, ProvenanceTracker


class TestRecordDecisionPersists:
    """Test 1: record_decision persists via pending list."""

    def test_decision_flushes_on_next_transformation(self):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("climate")
        t.record_decision(
            decision_type=DecisionType.SOURCE_SELECTION,
            description="NASA POWER",
            rationale="best coverage",
            alternatives=["AgERA5"],
        )
        t.record_retrieval(source="NASA_POWER", parameters={"lat": 12.0})

        summary = t.get_summary()
        assert summary["n_decisions"] == 1
        assert summary["decisions_by_type"]["source_selection"] == 1

    def test_multiple_decisions_flush_to_same_transformation(self):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("soil")
        t.record_decision(DecisionType.SOURCE_SELECTION, "HWSD", "primary")
        t.record_decision(DecisionType.FALLBACK_SUBSTITUTION, "DEFAULT_SOIL", "missing SMU")
        t.record_transformation(OperationType.RETRIEVE, parameters={})

        summary = t.get_summary()
        assert summary["n_decisions"] == 2
        assert summary["n_transformations"] == 1

    def test_decision_return_value_backward_compat(self):
        """Callers that still use the return value should not break."""
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("region")
        decision = t.record_decision(
            DecisionType.DEFAULT_VALUE, "test", "test"
        )
        assert decision is not None
        assert decision.description == "test"


class TestFinalizeFlushesStragglers:
    """Test 2: finalize() creates synthetic pipeline artifact."""

    def test_finalize_with_stragglers(self):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("region")
        t.record_retrieval(source="GADM", parameters={})
        # Record decisions after the last transformation — stragglers
        t.record_decision(DecisionType.DEFAULT_VALUE, "straggler1", "test")
        t.record_decision(DecisionType.DEFAULT_VALUE, "straggler2", "test")

        t.finalize()

        summary = t.get_summary()
        assert summary["n_decisions"] == 2
        assert summary["n_artifacts"] == 2  # region + synthetic pipeline

        # Verify the synthetic pipeline artifact exists
        pipeline_ids = [
            aid for aid, lin in t.record.artifacts.items()
            if lin.artifact_type == "pipeline"
        ]
        assert len(pipeline_ids) == 1

    def test_finalize_no_stragglers_is_noop(self):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("region")
        t.record_decision(DecisionType.DEFAULT_VALUE, "attached", "test")
        t.record_retrieval(source="GADM", parameters={})

        t.finalize()

        summary = t.get_summary()
        assert summary["n_decisions"] == 1
        assert summary["n_artifacts"] == 1  # no synthetic pipeline added


class TestUnattachedDecisionsRaise:
    """Test 3: fail-fast on decisions with no active artifact."""

    def test_transformation_without_artifact_raises(self):
        t = ProvenanceTracker(project_name="test")
        t.record_decision(DecisionType.SOURCE_SELECTION, "test", "test")

        with pytest.raises(ProvenanceStateError, match="no active artifact"):
            t.record_transformation(OperationType.RETRIEVE, parameters={})

    def test_retrieval_without_artifact_raises(self):
        t = ProvenanceTracker(project_name="test")
        t.record_decision(DecisionType.SOURCE_SELECTION, "test", "test")

        with pytest.raises(ProvenanceStateError, match="no active artifact"):
            t.record_retrieval(source="test", parameters={})


class TestThreadSafety:
    """Test 4: concurrent record_decision() calls — 4 threads × 25 decisions."""

    def test_concurrent_decisions_no_loss(self):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("test")

        n_threads = 4
        n_per_thread = 25
        expected_total = n_threads * n_per_thread

        def worker(idx):
            for i in range(n_per_thread):
                t.record_decision(
                    DecisionType.SOURCE_SELECTION,
                    f"thread-{idx}-decision-{i}",
                    "test",
                )

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(n_threads)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        t.record_retrieval(source="test", parameters={})

        summary = t.get_summary()
        assert summary["n_decisions"] == expected_total, (
            f"Expected {expected_total} decisions after flush, "
            f"got {summary['n_decisions']}"
        )


class TestEnabledShortCircuit:
    """Test 5: enabled=False short-circuits all writes."""

    def test_disabled_tracker_noop(self):
        t = ProvenanceTracker(project_name="test", enabled=False)
        t.start_artifact("climate")
        t.record_decision(DecisionType.SOURCE_SELECTION, "x", "y")
        t.record_retrieval(source="NASA_POWER", parameters={})
        t.record_transformation(OperationType.TRANSLATE, parameters={})
        t.finalize()

        summary = t.get_summary()
        assert summary["n_artifacts"] == 0
        assert summary["n_decisions"] == 0
        assert summary["n_transformations"] == 0


class TestEnumCompleteness:
    """Test 6: new enum values exist."""

    def test_decision_type_additions(self):
        assert DecisionType.FALLBACK_SUBSTITUTION.value == "fallback_substitution"
        assert DecisionType.UNIT_CONVERSION.value == "unit_conversion"
        assert DecisionType.AGGREGATION_METHOD.value == "aggregation_method"

    def test_operation_type_additions(self):
        assert OperationType.BUILD_GRID.value == "build_grid"

    def test_no_spatial_alignment(self):
        """SPATIAL_ALIGNMENT was intentionally NOT added (dead code reference)."""
        assert not hasattr(DecisionType, "SPATIAL_ALIGNMENT")


class TestExceptionMidStage:
    """Test 7: finalize still flushes pending decisions on exception paths."""

    def test_exception_does_not_lose_decisions(self):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("climate")
        t.record_decision(DecisionType.SOURCE_SELECTION, "before-exception", "test")

        # Simulate a pipeline stage raising mid-execution
        try:
            raise RuntimeError("simulated pipeline failure")
        except RuntimeError:
            # finalize() should still flush the pending decision
            t.finalize()

        summary = t.get_summary()
        assert summary["n_decisions"] == 1

    def test_try_finally_pattern(self):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("region")
        t.record_decision(DecisionType.DEFAULT_VALUE, "straggler", "test")

        try:
            raise ValueError("simulated")
        except ValueError:
            pass
        finally:
            t.finalize()

        summary = t.get_summary()
        assert summary["n_decisions"] == 1


class TestDualOutputFormat:
    """Test 8: save() emits both provenance.json and provenance_stages.json."""

    def test_dual_output_both_files_created(self, tmp_path):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("climate")
        t.record_decision(DecisionType.SOURCE_SELECTION, "NASA POWER", "test")
        t.record_retrieval(source="NASA_POWER", parameters={"lat": 12.0})

        out = tmp_path / "prov.json"
        t.save(out)

        assert out.exists()
        stages_path = out.with_name("prov_stages.json")
        assert stages_path.exists()

    def test_stages_format_structure(self, tmp_path):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("climate")
        t.record_decision(DecisionType.SOURCE_SELECTION, "NASA POWER", "test")
        t.record_retrieval(source="NASA_POWER", parameters={})
        t.start_artifact("soil")
        t.record_decision(DecisionType.FALLBACK_SUBSTITUTION, "HWSD default", "test")
        t.record_retrieval(source="HWSD", parameters={})
        t.record_transformation(OperationType.TRANSLATE, parameters={"platform": "craft"})

        out = tmp_path / "prov.json"
        t.save(out)
        stages_path = out.with_name("prov_stages.json")

        with open(stages_path) as f:
            stages_dict = json.load(f)

        assert "stages" in stages_dict
        assert "workflow" in stages_dict
        assert stages_dict["workflow"] == "prismpy"

        stage_names = [s["stage"] for s in stages_dict["stages"]]
        assert "retrieve" in stage_names
        assert "translate" in stage_names

        retrieve_stage = next(s for s in stages_dict["stages"] if s["stage"] == "retrieve")
        # Both the NASA POWER decision and HWSD fallback decision should be in retrieve
        assert len(retrieve_stage["decisions"]) == 2

    def test_rich_format_has_artifacts(self, tmp_path):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("region")
        t.record_retrieval(source="GADM", parameters={})

        out = tmp_path / "prov.json"
        t.save(out)

        with open(out) as f:
            rich_dict = json.load(f)

        assert "artifacts" in rich_dict
        assert "session_id" in rich_dict
        assert len(rich_dict["artifacts"]) == 1

    def test_unattached_decisions_safety_net(self, tmp_path):
        """If caller forgets to finalize, unflushed pending decisions land
        in the output's unattached_decisions field, not silently lost."""
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("climate")
        t.record_retrieval(source="NASA_POWER", parameters={})
        # Record AFTER retrieval — pending, not attached
        t.record_decision(DecisionType.DEFAULT_VALUE, "orphan", "test")

        out = tmp_path / "prov.json"
        t.save(out)  # no finalize() called!

        with open(out) as f:
            rich_dict = json.load(f)

        assert "unattached_decisions" in rich_dict
        assert len(rich_dict["unattached_decisions"]) == 1
        assert rich_dict["unattached_decisions"][0]["description"] == "orphan"


class TestCheckpointSaves:
    """Test 9: incremental checkpoint saves at stage boundaries."""

    def test_checkpoint_save_writes_file(self, tmp_path):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("region")
        t.record_retrieval(source="GADM", parameters={})

        out = tmp_path / "checkpoint.json"
        t.checkpoint_save(out)

        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert data["session_id"] == t.session_id

    def test_checkpoint_save_with_configured_path(self, tmp_path):
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("region")
        t.record_retrieval(source="GADM", parameters={})

        # Set path once, then call checkpoint_save() with no arg
        t._checkpoint_path = tmp_path / "auto_checkpoint.json"
        t.checkpoint_save()

        assert (tmp_path / "auto_checkpoint.json").exists()

    def test_checkpoint_save_non_fatal_on_error(self, tmp_path):
        """Checkpoint save failures must NEVER crash the pipeline."""
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("region")

        # Point to an impossible path
        impossible = tmp_path / "does_not_exist" / "sub" / "prov.json"
        # Mock save_json to raise, verify checkpoint_save swallows
        with patch.object(t, "save", side_effect=OSError("disk full")):
            t.checkpoint_save(impossible)  # should NOT raise

    def test_checkpoint_save_disabled_tracker_noop(self, tmp_path):
        t = ProvenanceTracker(project_name="test", enabled=False)
        out = tmp_path / "checkpoint.json"
        t.checkpoint_save(out)
        assert not out.exists()


class TestArtifactIdBinding:
    """Verify improvement 2: artifact-id bound at record-time, not flush-time."""

    def test_decision_bound_to_artifact_at_record_time(self):
        t = ProvenanceTracker(project_name="test")

        # Start climate artifact, record decision
        climate_id = t.start_artifact("climate")
        t.record_decision(DecisionType.SOURCE_SELECTION, "climate-decision", "test")

        # Move current artifact pointer to soil — climate decision should
        # still be bound to climate, not reassigned to soil
        soil_id = t.start_artifact("soil")
        t.record_decision(DecisionType.FALLBACK_SUBSTITUTION, "soil-decision", "test")

        # Flush soil transformation — should drain ONLY the soil decision
        t.record_retrieval(source="HWSD", parameters={})
        # Climate decision should still be pending
        assert len(t._pending_decisions) == 1
        assert t._pending_decisions[0][1] == climate_id

        # Flush climate transformation — should drain the climate decision
        t.record_retrieval(source="NASA_POWER", parameters={}, artifact_id=climate_id)

        summary = t.get_summary()
        assert summary["n_decisions"] == 2
        # Verify climate artifact has exactly 1 decision
        climate_lineage = t.record.get_artifact(climate_id)
        assert len(climate_lineage.all_decisions) == 1
        assert climate_lineage.all_decisions[0].description == "climate-decision"
        # Verify soil artifact has exactly 1 decision
        soil_lineage = t.record.get_artifact(soil_id)
        assert len(soil_lineage.all_decisions) == 1
        assert soil_lineage.all_decisions[0].description == "soil-decision"


class TestTrailingDecisionDirectAttach:
    """V2-19b-fix Finding 4: trailing decisions direct-attach to existing transformations.

    The pending-list mechanism is forward-only — it only drains decisions
    on FUTURE transformations against the bound artifact. When a decision
    is recorded AFTER the artifact's last transformation, no future
    transformation comes to drain it, and the decision sits in pending
    forever (eventually dumped to ``unattached_decisions`` at save time).

    The fix: when explicit ``artifact_id`` references an artifact that
    already has at least one transformation, attach the decision directly
    to its most recent transformation. This handles the trailing-decision
    case while leaving the leading-decision case (V2-19 A1 pending list)
    unchanged.

    These tests use STRUCTURAL binding assertions (e.g.,
    ``assert decision in artifacts[X].decisions``), not global counts —
    per the team-lead's binding-verification methodology rule.
    """

    def test_trailing_decision_attaches_to_existing_grid_artifact(self):
        """The exact bug from the V2-19b canonical-flow capture.

        Reproduce the executor's BUILD_GRID + effective-resolution sequence:
          1. start_artifact("grid")
          2. record_decision(AGGREGATION) — leading, drained by next transform
          3. record_transformation(BUILD_GRID, artifact_id="grid")
          4. record_decision(SOURCE_SELECTION, artifact_id="grid") — TRAILING
          5. start_artifact("soil") — current pointer moves away from grid
          6. record_retrieval(soil)

        Expected: BOTH decisions land on the grid artifact's BUILD_GRID
        transformation. ``unattached_decisions`` is empty.

        Pre-fix behaviour: the trailing decision sat in pending forever
        and was dumped to ``unattached_decisions`` with
        ``bound_artifact_id="grid"`` at save time.
        """
        t = ProvenanceTracker(project_name="test")

        # Step 1-3: leading-decision case (still works post-fix)
        t.start_artifact("grid", artifact_id="grid")
        t.record_decision(
            DecisionType.AGGREGATION_METHOD,
            "5-arcmin uniform grid",
            "canonical resolution",
        )
        t.record_transformation(
            OperationType.BUILD_GRID,
            parameters={"resolution": "5arcmin", "n_cells": 114},
            artifact_id="grid",
        )

        # Step 4: trailing decision — explicit artifact_id="grid" but the
        # grid artifact's transformation already happened
        t.record_decision(
            DecisionType.SOURCE_SELECTION,
            "Effective resolution WARNING: ...",
            "NASA POWER 0.5° coarser than 5-arcmin target",
            artifact_id="grid",
        )

        # Step 5: current pointer moves to soil
        t.start_artifact("soil")
        t.record_retrieval(source="iSDA", parameters={})

        # STRUCTURAL ASSERTIONS — the binding-verification methodology
        grid_lineage = t.record.get_artifact("grid")
        assert grid_lineage is not None, "grid artifact must exist"
        assert grid_lineage.transformations, (
            "grid artifact must have at least one transformation"
        )

        # Both grid decisions land on the BUILD_GRID transformation
        grid_decisions = grid_lineage.transformations[-1].decisions
        descriptions = [d.description for d in grid_decisions]
        assert "5-arcmin uniform grid" in descriptions, (
            f"AGGREGATION decision (leading) must land on grid; got {descriptions}"
        )
        assert "Effective resolution WARNING: ..." in descriptions, (
            f"SOURCE_SELECTION decision (trailing) must land on grid; "
            f"got {descriptions}"
        )

        # Pending list is empty — no orphans waiting at save time
        assert not t._pending_decisions, (
            f"pending list must be empty after both decisions land; "
            f"got {[(d.description, aid) for d, aid in t._pending_decisions]}"
        )

        # The trailing decision did NOT leak to the soil artifact
        soil_lineage = next(
            (
                lin for lin in t.record.artifacts.values()
                if lin.artifact_type == "soil"
            ),
            None,
        )
        assert soil_lineage is not None
        soil_descriptions = [
            d.description
            for tx in soil_lineage.transformations
            for d in tx.decisions
        ]
        assert "Effective resolution WARNING: ..." not in soil_descriptions, (
            "trailing decision must NOT contaminate the soil artifact"
        )

    def test_save_dump_does_not_emit_unattached_for_trailing_decision(self):
        """Save-time safety net: ``unattached_decisions`` is empty when
        the only decisions are trailing decisions properly attached.

        This is the canonical-flow capture's smoking-gun test. Pre-fix,
        the saved JSON had ``unattached_decisions`` with the effective-
        resolution decision and ``bound_artifact_id="grid"`` preserved.
        Post-fix, that field must be absent (or empty list).
        """
        with tempfile.TemporaryDirectory() as td:
            t = ProvenanceTracker(
                project_name="test",
                output_dir=td,
            )
            t.start_artifact("grid", artifact_id="grid")
            t.record_transformation(
                OperationType.BUILD_GRID,
                parameters={"resolution": "5arcmin"},
                artifact_id="grid",
            )
            # Trailing decision against grid (post-transform)
            t.record_decision(
                DecisionType.SOURCE_SELECTION,
                "trailing-grid",
                "test",
                artifact_id="grid",
            )

            # Save and inspect the rich JSON
            save_path = t.save()
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # The trailing decision must NOT appear in unattached_decisions
            unattached = data.get("unattached_decisions", [])
            assert not unattached, (
                f"unattached_decisions must be empty, got {unattached}"
            )

            # The trailing decision must appear on grid's BUILD_GRID transform
            grid = data["artifacts"]["grid"]
            grid_descriptions = [
                d["description"]
                for tx in grid["transformations"]
                for d in tx["decisions"]
            ]
            assert "trailing-grid" in grid_descriptions

    def test_leading_decision_implicit_still_uses_pending_list(self):
        """Regression guard: the V2-19 A1 leading-decision IMPLICIT case
        must NOT be broken by the Finding 4 fix. When no explicit
        ``artifact_id`` is given and the artifact has zero transformations
        at record time, the decision goes to pending via the implicit
        path and is drained by the next transformation.

        Note: the EXPLICIT variant (artifact_id="soil" when soil has no
        transformations) now raises ProvenanceStateError per Refinement 2.
        The implicit path is for callers who don't know which artifact
        they're targeting, or who are recording before start_artifact.
        """
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("soil", artifact_id="soil")
        # Leading decision: NO explicit artifact_id — uses _current_artifact_id
        t.record_decision(
            DecisionType.SOURCE_SELECTION,
            "iSDA primary",
            "Africa-wide 30m",
            # NO artifact_id= parameter → implicit pending-list path
        )

        # Decision is in pending, bound to current artifact ("soil")
        assert len(t._pending_decisions) == 1
        assert t._pending_decisions[0][1] == "soil"

        # Now record the soil retrieve transform — drains pending
        t.record_retrieval(source="iSDA", parameters={"n_profiles": 114})

        # Pending is empty, decision landed on soil artifact
        assert not t._pending_decisions
        soil_decisions = t.record.get_artifact("soil").transformations[-1].decisions
        assert len(soil_decisions) == 1
        assert soil_decisions[0].description == "iSDA primary"

    def test_trailing_decision_with_no_explicit_artifact_id_uses_pending(self):
        """Regression guard: a trailing decision without explicit
        ``artifact_id`` cannot direct-attach (we don't know which
        artifact it belongs to without explicit intent). It must go to
        pending bound to ``_current_artifact_id`` and be drained by the
        next transformation, OR end up unattached at save time.
        """
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("grid", artifact_id="grid")
        t.record_transformation(
            OperationType.BUILD_GRID,
            parameters={},
            artifact_id="grid",
        )
        # Trailing decision WITHOUT explicit artifact_id — falls back to
        # pending-list path with _current_artifact_id="grid"
        t.record_decision(
            DecisionType.SOURCE_SELECTION,
            "trailing no-explicit",
            "test",
        )

        # Decision is in pending, bound to current ("grid")
        assert len(t._pending_decisions) == 1
        assert t._pending_decisions[0][1] == "grid"
        # Did NOT direct-attach (no explicit artifact_id was provided)
        grid_decisions = t.record.get_artifact("grid").transformations[-1].decisions
        assert "trailing no-explicit" not in [d.description for d in grid_decisions]

    def test_explicit_artifact_id_to_nonexistent_artifact_raises(self):
        """V2-19b-fix Refinement 2: explicit ``artifact_id`` referencing
        a non-existent artifact raises ``ProvenanceStateError`` (fail-fast).

        Silent fallback to pending was the soft-failure mode that hid
        Finding 4. When the caller is explicit, the system is strict.
        Callers who need the pending-list path (leading decisions before
        start_artifact) should omit the artifact_id parameter.
        """
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("climate")

        with pytest.raises(ProvenanceStateError, match="no artifact with that ID"):
            t.record_decision(
                DecisionType.SOURCE_SELECTION,
                "deferred-soil",
                "test",
                artifact_id="soil",  # "soil" doesn't exist yet
            )

    def test_explicit_artifact_id_to_empty_artifact_uses_pending(self):
        """Case (b): explicit ``artifact_id`` referencing an artifact with
        zero transformations falls through to the pending-list path.

        This is the normal leading-decision pattern: the caller knows
        the artifact is started but hasn't been transformed yet. The
        pending entry will be drained by the next transformation against
        that artifact (e.g., record_retrieval for climate).
        """
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("soil", artifact_id="soil")
        # soil artifact exists but has no transformations yet

        # Leading decision with explicit artifact_id — should go to
        # pending, NOT raise
        t.record_decision(
            DecisionType.SOURCE_SELECTION,
            "leading-explicit",
            "test",
            artifact_id="soil",
        )

        # Decision is in pending, bound to "soil"
        assert len(t._pending_decisions) == 1
        assert t._pending_decisions[0][1] == "soil"
        assert t._pending_decisions[0][0].description == "leading-explicit"

        # Now record the retrieval — drains pending
        t.record_retrieval(source="iSDA", parameters={}, artifact_id="soil")

        assert not t._pending_decisions
        soil_decisions = t.record.get_artifact("soil").transformations[-1].decisions
        assert len(soil_decisions) == 1
        assert soil_decisions[0].description == "leading-explicit"

    def test_trailing_decision_attaches_to_most_recent_transformation(self):
        """When the artifact has multiple transformations, direct-attach
        targets the MOST RECENT one (not the first). This mirrors the
        natural intent: the user is recording a decision related to the
        most recently completed work on that artifact.
        """
        t = ProvenanceTracker(project_name="test")
        t.start_artifact("soil", artifact_id="soil")
        t.record_retrieval(source="iSDA", parameters={"phase": "first"})
        t.record_retrieval(source="HWSD", parameters={"phase": "second"})

        # Trailing decision against soil — should attach to the SECOND
        # transformation (HWSD), not the first (iSDA)
        t.record_decision(
            DecisionType.FALLBACK_SUBSTITUTION,
            "fallback-after-iSDA-failed",
            "test",
            artifact_id="soil",
        )

        soil_lineage = t.record.get_artifact("soil")
        assert len(soil_lineage.transformations) == 2

        first_decisions = [d.description for d in soil_lineage.transformations[0].decisions]
        second_decisions = [d.description for d in soil_lineage.transformations[1].decisions]

        assert "fallback-after-iSDA-failed" not in first_decisions, (
            "trailing decision must not attach to the first transformation"
        )
        assert "fallback-after-iSDA-failed" in second_decisions, (
            "trailing decision must attach to the most recent transformation"
        )
