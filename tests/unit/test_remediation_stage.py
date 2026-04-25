"""V2-22c-PRE.4 — REMEDIATION pipeline stage scaffolding +
Veto #4 server enforcement.

10-test matrix per the contract §6 verification plan:

| Test # | Scenario                                  | Expected behavior                              |
|--------|-------------------------------------------|------------------------------------------------|
| 1      | No-op path (no spec)                      | StageResult(success=True) returned             |
| 2      | silent tier (all neighbors same class)    | No exception                                   |
| 3      | warn tier acknowledged                    | No exception                                   |
| 4      | warn tier unacknowledged                  | RemediationBlocked(reason=warn_unacked) raised |
| 5      | BLOCK target_null + ack=true              | STILL raises (D33 NON-acknowledgeable)         |
| 6      | BLOCK neighbor_null + ack=true            | STILL raises                                   |
| 7      | BLOCK all_cross_class + ack=true          | STILL raises                                   |
| 8      | D31 carve-out (method='direct_user_supplied')| ('silent', None); no exception             |
| 9      | RemediationBlocked(reason='other')        | ValueError raised                              |
| 10     | HIGH 5: 1 of 4 neighbors absent (Δ.6)     | RemediationBlocked(reason=neighbor_null) raised|

Tests #5/#6/#7 are the load-bearing adversarial-bypass guards: they
catch a refactor that demotes BLOCK to WARN under acknowledgment
(the F2 fix). Test #10 is the HIGH 5 closure — without preserve-
nulls-first ordering in `_veto_4_tier`, a missing-from-soil-dict
neighbor silently filters out and the tier downgrades. Test #9 is
the §12.4 binding — REASON_VALID frozenset enforcement at the
RemediationBlocked constructor.

Plus: structural / AST tests for PRE.4.1 (enum order), PRE.4.3
(narrow except RemediationBlocked + call order), PRE.4.5
(STAGE_ORDER 3-copy parity will be in cross-repo tests; this
module covers the prismpy-side enum).
"""
from __future__ import annotations

import ast
import inspect
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from prismpy.models.region import Region, BoundingBox
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.pipeline._remediation import (
    RemediationBlocked,
    _block_message,
    _veto_4_tier,
)
from prismpy.pipeline.executor import (
    PipelineStage,
    TranslationPipeline,
)
from prismpy.translators.base import UnifiedData


# =====================================================================
# Helpers
# =====================================================================


def _make_pipeline():
    return TranslationPipeline.__new__(TranslationPipeline)


def _make_grid_with_target_and_neighbors():
    """Build a 5-cell grid: target at (0, 0) + 4 neighbors at the
    cardinal directions one row/col away. Cell IDs:

        ... 1 ...
        ... 0 4 ...   (0=target at row=0,col=0)
        ... 2 ...
        ... 3 ...

    Used for the Veto #4 tier matrix — _idw_neighbor_cells with
    radius=2 finds exactly these 4 neighbors.
    """
    cells = [
        GridCell(cell_id=0, lat=14.0, lon=0.0, row=10, col=10, resolution="5arcmin"),
        GridCell(cell_id=1, lat=14.1, lon=0.0, row=9,  col=10, resolution="5arcmin"),
        GridCell(cell_id=2, lat=13.9, lon=0.0, row=11, col=10, resolution="5arcmin"),
        GridCell(cell_id=3, lat=13.8, lon=0.0, row=12, col=10, resolution="5arcmin"),
        GridCell(cell_id=4, lat=14.0, lon=0.1, row=10, col=11, resolution="5arcmin"),
    ]
    return SpatialGrid(
        bounds=BoundingBox(minx=0, miny=13, maxx=1, maxy=15),
        resolution="5arcmin", cells=cells,
    )


def _profile(profile_id: str, sand: float, clay: float):
    """Build a SoilProfile whose surface_texture derives from the
    USDA classifier on the (sand, clay) pair."""
    return SoilProfile(
        profile_id=profile_id, lat=0.5, lon=0.5, source="iSDA",
        layers=[SoilLayer(
            depth_top=0, depth_bottom=0.2,
            sand=sand, clay=clay,
        )],
    )


def _empty_profile(profile_id: str):
    """SoilProfile with no layers → surface_texture returns None,
    triggering the BLOCK target_null / neighbor_null paths."""
    return SoilProfile(
        profile_id=profile_id, lat=0.5, lon=0.5, source="iSDA",
        layers=[],
    )


def _unified(soil_dict: Dict[int, Any]):
    """UnifiedData with the 5-cell test grid + the supplied soil
    dict. Climate is empty (Veto #4 ignores climate)."""
    return UnifiedData(
        region=Region(
            name="t", country="t", country_iso3="TST",
            bounds=BoundingBox(minx=0, miny=13, maxx=1, maxy=15),
        ),
        grid=_make_grid_with_target_and_neighbors(),
        soil=soil_dict,
        climate={},
    )


def _imp(*, cell_id=0, method="idw", radius=2, **extra):
    """Build a synthetic imputation_action dict."""
    spec = {
        "cell_id": cell_id, "method": method,
        "parameters": {"max_radius_cells": radius},
    }
    spec.update(extra)
    return spec


# =====================================================================
# Test #1 — No-op path (no spec)
# =====================================================================


class TestVetoNoOp:
    def test_no_spec_yields_success_stage_result(self):
        """Test #1 — when self.config has no remediation_spec
        (originals, retries — the most common path), the method
        returns success=True with no exceptions."""
        pipeline = _make_pipeline()
        pipeline.config = SimpleNamespace(remediation_spec=None)
        result = pipeline._execute_remediation({}, None)
        assert result.success
        assert result.stage == PipelineStage.REMEDIATION
        assert result.errors == []


# =====================================================================
# Test #2 — silent tier (all neighbors share target's class)
# =====================================================================


class TestSilentTier:
    def test_all_neighbors_same_class_no_exception(self):
        """Test #2 — target sand=80 clay=10 → Loamy Sand. All 4
        neighbors also Loamy Sand. _veto_4_tier returns ('silent',
        None); no exception."""
        soil = {
            0: _profile("p0", 80, 10),
            1: _profile("p1", 80, 10),
            2: _profile("p2", 80, 10),
            3: _profile("p3", 80, 10),
            4: _profile("p4", 80, 10),
        }
        unified = _unified(soil)
        tier, reason = _veto_4_tier(_imp(cell_id=0), unified)
        assert tier == "silent"
        assert reason is None


# =====================================================================
# Tests #3, #4 — warn tier acknowledged / unacknowledged
# =====================================================================


class TestWarnTier:
    def _mixed_neighborhood(self):
        """Target Loamy Sand; 3 neighbors Loamy Sand, 1 Clay → warn."""
        return _unified({
            0: _profile("p0", 80, 10),  # target — Loamy Sand
            1: _profile("p1", 80, 10),  # Loamy Sand
            2: _profile("p2", 80, 10),  # Loamy Sand
            3: _profile("p3", 20, 50),  # Clay (different class)
            4: _profile("p4", 80, 10),  # Loamy Sand
        })

    def test_warn_unacknowledged_raises_warn_unacked(self):
        """Test #4 — warn tier + ack=False → executor raises
        RemediationBlocked with reason='veto_4_warn_unacknowledged'."""
        pipeline = _make_pipeline()
        pipeline.config = SimpleNamespace(remediation_spec={
            "imputations": [_imp(cell_id=0, veto_4_acknowledged=False)],
        })
        with pytest.raises(RemediationBlocked) as exc_info:
            pipeline._execute_remediation({}, self._mixed_neighborhood())
        assert exc_info.value.reason == RemediationBlocked.REASON_WARN_UNACKED

    def test_warn_acknowledged_no_exception(self):
        """Test #3 — warn tier + ack=True → no exception."""
        pipeline = _make_pipeline()
        pipeline.config = SimpleNamespace(remediation_spec={
            "imputations": [_imp(cell_id=0, veto_4_acknowledged=True)],
        })
        result = pipeline._execute_remediation(
            {}, self._mixed_neighborhood(),
        )
        assert result.success
        assert result.errors == []


# =====================================================================
# Tests #5, #6, #7 — BLOCK adversarial-bypass matrix
# =====================================================================


class TestBlockAdversarialBypass:
    """Load-bearing adversarial-bypass tests for the F2 fix —
    tests #5/#6/#7 catch the refactor where someone reorders the
    `if tier == 'block': raise` clause behind the
    `veto_4_acknowledged` flag check, silently demoting BLOCK to
    WARN."""

    def test_block_target_null_still_raises_with_ack(self):
        """Test #5 — target cell has no surface_texture →
        RemediationBlocked(reason='target_null') raised even with
        ack=True (D33 NON-acknowledgeable)."""
        pipeline = _make_pipeline()
        soil = {
            0: _empty_profile("p0"),  # target null
            1: _profile("p1", 80, 10),
            2: _profile("p2", 80, 10),
            3: _profile("p3", 80, 10),
            4: _profile("p4", 80, 10),
        }
        unified = _unified(soil)
        pipeline.config = SimpleNamespace(remediation_spec={
            "imputations": [
                _imp(cell_id=0, veto_4_acknowledged=True),
            ],
        })
        with pytest.raises(RemediationBlocked) as exc_info:
            pipeline._execute_remediation({}, unified)
        assert exc_info.value.reason == RemediationBlocked.REASON_TARGET_NULL

    def test_block_neighbor_null_still_raises_with_ack(self):
        """Test #6 — 1 of 4 neighbors has no surface_texture →
        RemediationBlocked(reason='neighbor_null') raised even with
        ack=True."""
        pipeline = _make_pipeline()
        soil = {
            0: _profile("p0", 80, 10),
            1: _empty_profile("p1"),  # null neighbor
            2: _profile("p2", 80, 10),
            3: _profile("p3", 80, 10),
            4: _profile("p4", 80, 10),
        }
        unified = _unified(soil)
        pipeline.config = SimpleNamespace(remediation_spec={
            "imputations": [
                _imp(cell_id=0, veto_4_acknowledged=True),
            ],
        })
        with pytest.raises(RemediationBlocked) as exc_info:
            pipeline._execute_remediation({}, unified)
        assert exc_info.value.reason == RemediationBlocked.REASON_NEIGHBOR_NULL

    def test_block_all_cross_class_still_raises_with_ack(self):
        """Test #7 — all 4 neighbors in different soil class →
        RemediationBlocked(reason='all_cross_class') raised even
        with ack=True."""
        pipeline = _make_pipeline()
        soil = {
            0: _profile("p0", 80, 10),  # target — Loamy Sand
            1: _profile("p1", 20, 50),  # Clay
            2: _profile("p2", 20, 50),  # Clay
            3: _profile("p3", 20, 50),  # Clay
            4: _profile("p4", 20, 50),  # Clay
        }
        unified = _unified(soil)
        pipeline.config = SimpleNamespace(remediation_spec={
            "imputations": [
                _imp(cell_id=0, veto_4_acknowledged=True),
            ],
        })
        with pytest.raises(RemediationBlocked) as exc_info:
            pipeline._execute_remediation({}, unified)
        assert exc_info.value.reason == RemediationBlocked.REASON_ALL_CROSS_CLASS


# =====================================================================
# Test #8 — D31 carve-out (direct user-supplied Override bypasses)
# =====================================================================


class TestD31CarveOut:
    def test_direct_user_supplied_method_returns_silent(self):
        """Test #8 — method without 'idw' or 'neighbor' substring
        → ('silent', None) regardless of soil_class state. Direct
        user-supplied Override values bypass Veto #4 entirely
        (D31)."""
        soil = {
            0: _empty_profile("p0"),  # target null — would normally BLOCK
        }
        unified = _unified(soil)
        tier, reason = _veto_4_tier(
            _imp(cell_id=0, method="direct_user_supplied"),
            unified,
        )
        assert tier == "silent"
        assert reason is None


# =====================================================================
# Test #9 — RemediationBlocked constructor enforcement (§12.4)
# =====================================================================


class TestReasonValidEnforcement:
    """Evaluator §12.4 binding — `RemediationBlocked.REASON_VALID`
    is an exact frozenset; constructing with an unrecognized
    reason raises ValueError."""

    def test_unknown_reason_raises_value_error(self):
        """Test #9 — schema-drift guard at the boundary."""
        with pytest.raises(ValueError) as exc_info:
            RemediationBlocked(reason="other_unknown_reason")
        assert "Invalid RemediationBlocked.reason" in str(exc_info.value)

    def test_reason_valid_is_exact_frozenset(self):
        """The 4 canonical reasons MUST match the cockpit AC-9.3
        copy variants exactly. Synonym drift (e.g.,
        'cross_class_all' instead of 'all_cross_class') breaks the
        cockpit copy mapping."""
        assert RemediationBlocked.REASON_VALID == frozenset({
            "target_null",
            "neighbor_null",
            "all_cross_class",
            "veto_4_warn_unacknowledged",
        })

    def test_known_reasons_construct_cleanly(self):
        for reason in RemediationBlocked.REASON_VALID:
            exc = RemediationBlocked(reason=reason, cell_id=42)
            assert exc.reason == reason
            assert exc.cell_id == 42


# =====================================================================
# Test #10 — HIGH 5 / Δ.6 closure (preserve-nulls-first ordering)
# =====================================================================


class TestNeighborNullPreserveOrder:
    """Test #10 — HIGH 5 fix from Δ.6. Without preserve-nulls-first
    ordering in `_veto_4_tier`, a neighbor missing from
    `unified_data.soil` would silently filter out before the
    null-check. The fix builds the FULL profile list FIRST, then
    checks for None entries — a missing neighbor surfaces as a
    BLOCK neighbor_null, not silent or warn."""

    def test_neighbor_missing_from_soil_dict_yields_block_not_silent(self):
        """Critical adversarial test: cell 3 (one of the 4
        neighbors) is NOT in the soil dict at all. soil.get(3)
        returns None. Without the fix, this gets filtered out and
        only cells 1, 2, 4 are checked — all Loamy Sand → silent
        tier (BUG). With the fix, the None entry remains in the
        list, the `any(p is None ...)` check fires, and the tier
        is BLOCK neighbor_null."""
        soil = {
            0: _profile("p0", 80, 10),
            1: _profile("p1", 80, 10),
            2: _profile("p2", 80, 10),
            # 3 is intentionally absent from the soil dict
            4: _profile("p4", 80, 10),
        }
        unified = _unified(soil)
        tier, reason = _veto_4_tier(_imp(cell_id=0), unified)
        assert tier == "block", (
            "HIGH 5 / Δ.6 regression: missing-from-dict neighbor "
            "silently filtered out, demoting BLOCK to silent. "
            "Refactor must preserve `None` entries in "
            "`neighbor_profiles = [soil.get(n) for n in neighbor_ids]` "
            "BEFORE the null check."
        )
        assert reason == RemediationBlocked.REASON_NEIGHBOR_NULL


# =====================================================================
# PRE.4.1 — PipelineStage.REMEDIATION enum extension
# =====================================================================


class TestPipelineStageRemediation:
    """V2-22c-PRE.4.1 — enum order: insertion at index 3 between
    TRANSLATE (2) and VALIDATE (4). The cockpit's processing-page
    progress bar reads this order."""

    def test_remediation_value_is_remediation(self):
        assert PipelineStage.REMEDIATION.value == "remediation"

    def test_remediation_at_index_3_between_translate_and_validate(self):
        order = list(PipelineStage)
        idx_remediation = order.index(PipelineStage.REMEDIATION)
        idx_translate = order.index(PipelineStage.TRANSLATE)
        idx_validate = order.index(PipelineStage.VALIDATE)
        assert idx_translate < idx_remediation < idx_validate
        assert idx_remediation == 3

    def test_string_membership(self):
        """Stage labels read from `Enum.__members__` use the string
        value; the cockpit's STAGE_LABELS dict reads from this."""
        assert "remediation" in [s.value for s in PipelineStage]


# =====================================================================
# PRE.4.3 — narrow except RemediationBlocked binding (§12.5)
# =====================================================================


class TestNarrowExceptBinding:
    """Evaluator §12.5 binding — the catch block in `execute()`
    around the remediation stage MUST be `except RemediationBlocked
    as block:`, NOT `except Exception`. A broad catch would swallow
    the structured `reason` field and break the cockpit's AC-9.3
    BLOCK copy specificity."""

    def test_execute_method_catches_remediation_blocked_narrowly(self):
        executor_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "prismpy" / "pipeline" / "executor.py"
        )
        tree = ast.parse(executor_path.read_text())

        # Find the `def execute(self, ...)` method.
        execute_funcs = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "execute"
        ]
        # Multiple `execute` matches possible (other classes); filter
        # to the one inside class TranslationPipeline.
        execute_in_pipeline = None
        for cls in ast.walk(tree):
            if isinstance(cls, ast.ClassDef) and cls.name == "TranslationPipeline":
                for member in cls.body:
                    if (
                        isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and member.name == "execute"
                    ):
                        execute_in_pipeline = member
                        break
        assert execute_in_pipeline is not None, (
            "TranslationPipeline.execute method not found in AST walk"
        )

        # Find the Try block that DIRECTLY wraps the
        # `_execute_remediation` call. The execute() body has an
        # outer `try ... except PipelineCancelled / except Exception`
        # that envelops every stage; the REMEDIATION stage adds its
        # own narrow inner try. We want the innermost try whose body
        # mentions `_execute_remediation` — pick the one with the
        # smallest body.
        candidate_tries = []
        for node in ast.walk(execute_in_pipeline):
            if isinstance(node, ast.Try):
                body_src = "\n".join(ast.unparse(b) for b in node.body)
                if "_execute_remediation" in body_src:
                    candidate_tries.append(node)
        assert candidate_tries, (
            "execute() AST: Try block wrapping _execute_remediation "
            "call not found — has the stage gate been removed?"
        )
        # The INNER try has the smallest body — it directly wraps
        # the _execute_remediation call.
        rem_try = min(
            candidate_tries,
            key=lambda t: sum(
                len(ast.unparse(b)) for b in t.body
            ),
        )

        # The handlers list MUST contain `except RemediationBlocked`
        # and MUST NOT contain a bare `except Exception:` for this
        # try (a broad catch would defeat §12.5).
        handler_types = []
        for h in rem_try.handlers:
            if h.type is None:
                handler_types.append("BARE")
            else:
                handler_types.append(ast.unparse(h.type))
        assert "RemediationBlocked" in handler_types, (
            "execute() Try around _execute_remediation MUST have an "
            "`except RemediationBlocked as block:` handler. §12.5 "
            "binding regression — catching broadly would swallow the "
            "structured reason field."
        )
        # Defensive: the only handler should be RemediationBlocked.
        # If a future refactor adds `except Exception:`, the cockpit
        # AC-9.3 BLOCK copy specificity is at risk.
        assert "Exception" not in handler_types, (
            "execute() Try around _execute_remediation has a broad "
            "`except Exception` handler — §12.5 forbids it (broad "
            "catch swallows the structured reason field)."
        )


# =====================================================================
# AC PRE.4.2 — _block_message strings (cockpit AC-9.3 mirror)
# =====================================================================


class TestBlockMessageMirrors:
    """Server-side _block_message renders the same string the
    cockpit will show, so provenance.json carries user-visible
    text verbatim."""

    def test_target_null_message_mentions_substitute_or_exclude(self):
        msg = _block_message({"cell_id": 14}, RemediationBlocked.REASON_TARGET_NULL)
        assert "14" in msg
        assert "Substitute" in msg or "Exclude" in msg

    def test_neighbor_null_message_mentions_substitute(self):
        msg = _block_message({"cell_id": 7}, RemediationBlocked.REASON_NEIGHBOR_NULL)
        assert "7" in msg
        assert "Substitute" in msg

    def test_all_cross_class_message_mentions_agronomic(self):
        msg = _block_message({"cell_id": 22}, RemediationBlocked.REASON_ALL_CROSS_CLASS)
        assert "22" in msg
        assert "agronomic" in msg.lower() or "different" in msg.lower()
