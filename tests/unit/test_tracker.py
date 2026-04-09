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
