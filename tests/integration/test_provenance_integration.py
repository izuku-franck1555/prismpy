"""
V2-19 C13 — CI integration test for provenance system.

Verifies that a pipeline run produces structurally correct provenance
output with all expected decision types, artifacts, and stages.

Uses a deterministic mock-based approach: constructs a ProvenanceTracker
and exercises the same call patterns that the real pipeline executor
uses, then verifies the output JSON structure.

Assertions use structural binding (not just counts) per the
binding-verification methodology rule from V2-19b-fix.
"""

import json
import tempfile
from pathlib import Path

import pytest

from prismpy.models.provenance import DecisionType, OperationType
from prismpy.provenance.tracker import ProvenanceTracker


class TestProvenanceIntegration:
    """C13: provenance system produces correct output from pipeline call patterns."""

    def _simulate_pipeline_provenance(self, output_dir: Path) -> Path:
        """Simulate a CRAFT Koutiala pipeline's provenance calls.

        Exercises the same call patterns as executor.py's 5-stage
        pipeline, producing a provenance.json that mirrors a real run.
        Returns path to the saved provenance file.
        """
        t = ProvenanceTracker(
            project_name="C13 integration test",
            output_dir=str(output_dir),
        )

        # === RETRIEVE stage ===
        # Region artifact (with pygadm fallback — Finding 3)
        t.start_artifact("region", stage="retrieve")
        t.record_decision(
            DecisionType.SOURCE_SELECTION,
            "Boundary source: pygadm fallback (MLI level 2 'Koutiala')",
            "GADM standard path unavailable, used pygadm fallback.",
        )
        t.record_retrieval(source="config", parameters={"region_name": "Koutiala"})

        # Climate artifact
        t.start_artifact("climate", artifact_id="climate", stage="retrieve")
        t.record_decision(
            DecisionType.SOURCE_SELECTION,
            "Climate source for craft: NASA POWER v9",
            "CRAFT uses NASA POWER for all variables.",
            artifact_id="climate",
        )
        t.record_retrieval(
            source="climate_sources",
            parameters={"n_locations": 0, "enabled_platforms": ["craft"]},
            artifact_id="climate",
        )

        # Soil artifact
        t.start_artifact("soil", artifact_id="soil", stage="retrieve")
        t.record_retrieval(
            source="soil_sources",
            parameters={"n_profiles": 0},
            artifact_id="soil",
        )

        # === HARMONIZE stage ===
        # Grid artifact
        t.start_artifact("grid", artifact_id="grid", stage="harmonize")
        t.record_decision(
            DecisionType.AGGREGATION_METHOD,
            "5-arcmin uniform grid (114 cells)",
            "Canonical grid resolution for all platforms.",
        )
        t.record_transformation(
            OperationType.BUILD_GRID,
            parameters={"resolution": "5arcmin", "n_cells": 114},
            artifact_id="grid",
        )
        # Effective-resolution warning (trailing decision — Finding 4)
        t.record_decision(
            DecisionType.SOURCE_SELECTION,
            "Effective resolution WARNING: target 5-arcmin finer than 1 source(s)",
            "NASA POWER 0.5° coarser than 5-arcmin target.",
            artifact_id="grid",
            severity="warning",
        )

        # iSDA soil cascade decisions
        t.record_decision(
            DecisionType.RESAMPLING_METHOD,
            "iSDA point sampling: nearest-neighbour (1x1 window)",
            "Cell coordinates transformed, 1x1 Window selects one pixel.",
            artifact_id="soil",
        )
        t.record_decision(
            DecisionType.SOURCE_SELECTION,
            "Soil source: iSDA 1km resampled from 30m COG",
            "iSDA 30m resampled to 1km via COG overview.",
            artifact_id="soil",
        )
        t.record_retrieval(
            source="iSDA",
            parameters={"cascade": "iSDA→HWSD", "final_source": "iSDA", "n_profiles": 114},
            artifact_id="soil",
        )

        # === TRANSLATE stage ===
        t.start_artifact("output_craft", artifact_id="output_craft", stage="translate")
        t.record_decision(
            DecisionType.FORMAT_CHOICE,
            "Generated CRAFT inputs for Koutiala",
            "CRAFT requires tab-separated files with DSSAT soil format.",
        )
        t.record_transformation(
            OperationType.TRANSLATE,
            parameters={"platform": "craft", "region": "Koutiala", "success": True},
            artifact_id="output_craft",
        )

        # === VALIDATE stage ===
        t.start_artifact("validation", artifact_id="validation", stage="validate")
        t.record_transformation(
            OperationType.VALIDATE,
            parameters={"n_checks": 12, "overall_result": "pass"},
            artifact_id="validation",
        )
        t.record_decision(
            DecisionType.QUALITY_CHECK,
            "Scientific validation: PASS (12 checks)",
            "6 Tier 1 scientific quality checks per manuscript Section 2.5.",
            artifact_id="validation",
        )

        # Finalize + save
        t.finalize()
        return t.save(output_path=output_dir / "provenance.json")

    def test_provenance_structure(self, tmp_path):
        """C13 core: provenance output has required structure."""
        prov_path = self._simulate_pipeline_provenance(tmp_path)

        with open(prov_path, "r") as f:
            data = json.load(f)

        # Top-level keys
        assert "session_id" in data
        assert "artifacts" in data
        assert "summary" in data

        summary = data["summary"]

        # n_decisions >= 8
        assert summary["n_decisions"] >= 8, (
            f"n_decisions={summary['n_decisions']}, expected >= 8"
        )

        # n_transformations >= 3
        assert summary["n_transformations"] >= 3, (
            f"n_transformations={summary['n_transformations']}, expected >= 3"
        )

        # n_artifacts >= 3
        assert summary["n_artifacts"] >= 3, (
            f"n_artifacts={summary['n_artifacts']}, expected >= 3"
        )

        # >= 5 distinct decision types
        decision_types = set(summary.get("decisions_by_type", {}).keys())
        assert len(decision_types) >= 5, (
            f"distinct types={decision_types} (count={len(decision_types)}), "
            f"expected >= 5"
        )

    def test_provenance_has_four_stages(self, tmp_path):
        """C13: provenance artifacts span 4 pipeline stages."""
        prov_path = self._simulate_pipeline_provenance(tmp_path)

        with open(prov_path, "r") as f:
            data = json.load(f)

        stages = set()
        for artifact in data["artifacts"].values():
            stage = artifact.get("stage")
            if stage:
                stages.add(stage)

        expected_stages = {"retrieve", "harmonize", "translate", "validate"}
        assert expected_stages.issubset(stages), (
            f"stages={stages}, expected at least {expected_stages}"
        )

    def test_decision_binding_structural(self, tmp_path):
        """C13 + AC1: decisions are bound to the correct artifacts.

        Structural binding assertions — not just "decision exists"
        but "decision is on the right artifact".
        """
        prov_path = self._simulate_pipeline_provenance(tmp_path)

        with open(prov_path, "r") as f:
            data = json.load(f)

        artifacts = data["artifacts"]

        # Grid artifact has AGGREGATION_METHOD + effective-resolution
        grid = artifacts.get("grid", {})
        grid_decisions = [
            d["description"]
            for tx in grid.get("transformations", [])
            for d in tx.get("decisions", [])
        ]
        assert any("5-arcmin" in d for d in grid_decisions), (
            f"grid must have AGGREGATION decision; got {grid_decisions}"
        )
        assert any("Effective resolution" in d for d in grid_decisions), (
            f"grid must have effective-resolution decision; got {grid_decisions}"
        )

        # Soil artifact has iSDA SOURCE_SELECTION + RESAMPLING_METHOD
        soil = artifacts.get("soil", {})
        soil_decisions = [
            d["decision_type"]
            for tx in soil.get("transformations", [])
            for d in tx.get("decisions", [])
        ]
        assert "source_selection" in soil_decisions
        assert "resampling_method" in soil_decisions

        # Validation artifact has QUALITY_CHECK
        validation = artifacts.get("validation", {})
        val_decisions = [
            d["decision_type"]
            for tx in validation.get("transformations", [])
            for d in tx.get("decisions", [])
        ]
        assert "quality_check" in val_decisions

        # No unattached_decisions
        unattached = data.get("unattached_decisions", [])
        assert not unattached, (
            f"unattached_decisions must be empty; got {unattached}"
        )

    def test_decision_phase4_fields(self, tmp_path):
        """Phase 4: decisions have severity, label, timestamp fields."""
        prov_path = self._simulate_pipeline_provenance(tmp_path)

        with open(prov_path, "r") as f:
            data = json.load(f)

        for artifact in data["artifacts"].values():
            for tx in artifact.get("transformations", []):
                for decision in tx.get("decisions", []):
                    assert "severity" in decision, (
                        f"decision missing 'severity': {decision['description']}"
                    )
                    assert decision["severity"] in ("info", "warning", "error")
                    assert "label" in decision
                    assert "timestamp" in decision
                    # alternatives are structured objects
                    for alt in decision.get("alternatives_considered", []):
                        assert isinstance(alt, dict), (
                            f"alternative must be dict, got {type(alt)}: {alt}"
                        )
                        assert "name" in alt

    def test_stages_compat_format(self, tmp_path):
        """Dual-output: stages-format file exists and has correct structure."""
        prov_path = self._simulate_pipeline_provenance(tmp_path)
        stages_path = prov_path.with_name(
            prov_path.stem + "_stages" + prov_path.suffix
        )

        assert stages_path.exists(), f"stages file missing at {stages_path}"

        with open(stages_path, "r") as f:
            data = json.load(f)

        assert "stages" in data
        assert "workflow" in data
        assert data["workflow"] == "prismpy"

        stage_names = [s["stage"] for s in data["stages"]]
        assert "retrieve" in stage_names
        assert "harmonize" in stage_names
        assert "translate" in stage_names

    def test_required_decision_types_present(self, tmp_path):
        """C13: all 5+ required decision types are present."""
        prov_path = self._simulate_pipeline_provenance(tmp_path)

        with open(prov_path, "r") as f:
            data = json.load(f)

        all_types = set()
        for artifact in data["artifacts"].values():
            for tx in artifact.get("transformations", []):
                for d in tx.get("decisions", []):
                    all_types.add(d["decision_type"])

        required = {
            "source_selection",
            "aggregation_method",
            "resampling_method",
            "format_choice",
            "quality_check",
        }
        assert required.issubset(all_types), (
            f"missing types: {required - all_types}; present: {all_types}"
        )
