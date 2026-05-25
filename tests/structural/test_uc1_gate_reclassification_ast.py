"""UC1 gate-reclassification structural pins (UC-GEN-S1-UC1-CONTRACT §3.1/§5/§10).

PR2 producer-side mirror of the prism-runner PR1 reclassification. Pins, against
the COMMITTED ``packaging/manifest.py``:

- MUST-1: UC1 hard gen set = {n_years_gte_4, manifest_cells_populated,
  manifest_crops_populated}; the dispatch-only forecast_or_analog_mode_resolved
  is removed from the catalog; n_years_gte_3 → n_years_gte_4 (BL-1).
- MUST-1b (OQ-2): the orphan gen evaluator + its _dispatch_gate branch removed.
- MUST-5 + SF-1: the >=30 adequacy ADVISORY gate + the templated flag carrying
  the actual N.
- §10: prismpy catalog references 14 unique wire strings (no enum on the
  producer side).

Placement note: this lives in ``tests/structural/`` (NOT the root
``tests/test_manifest_emit_invariants_ast.py`` that §10 names "primary")
because prismpy CI runs ``pytest tests/structural tests/unit`` explicitly —
the root tests/ files are committed but outside the CI collection scope, so a
pin there would not gate in CI. Co-located with the sibling
``*_emit_invariants_ast.py`` structural pins.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_MANIFEST_SRC_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "prismpy" / "packaging" / "manifest.py"
)


def _load_manifest_ast() -> ast.Module:
    return ast.parse(
        _MANIFEST_SRC_PATH.read_text(encoding="utf-8"),
        filename=str(_MANIFEST_SRC_PATH),
    )


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _string_constants(node: ast.AST) -> set[str]:
    return {
        n.value for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


# ── MUST-1: UC1 hard gate set ───────────────────────────────────────────────


def test_uc1_per_uc_gates_reclassified_hard_set() -> None:
    from prismpy.packaging.manifest import ADVISORY_GATES, PER_UC_GATES

    uc1 = PER_UC_GATES["yield_forecast"]
    hard = uc1 - ADVISORY_GATES
    assert hard == {
        "n_years_gte_4",
        "manifest_cells_populated",
        "manifest_crops_populated",
    }, f"UC1 hard set drift: {sorted(hard)}"
    assert "forecast_or_analog_mode_resolved" not in uc1, (
        "forecast_or_analog_mode_resolved must be REMOVED from the UC1 gen "
        "catalog (MUST-1b / OQ-2) — dispatch-time check only."
    )
    assert "n_years_gte_3" not in uc1, (
        "UC1 must use n_years_gte_4 (BL-1 analog floor), not n_years_gte_3."
    )
    assert "n_years_gte_30_for_forecast_adequacy" in uc1, (
        "UC1 must carry the >=30 adequacy advisory (MUST-5)."
    )


def test_uc1_adequacy_gate_is_advisory() -> None:
    from prismpy.packaging.manifest import ADVISORY_GATES

    assert "n_years_gte_30_for_forecast_adequacy" in ADVISORY_GATES


# ── §10: catalog-unique count ───────────────────────────────────────────────


def test_catalog_unique_wire_strings_is_14() -> None:
    from prismpy.packaging.manifest import PER_UC_GATES

    unique = {g for gates in PER_UC_GATES.values() for g in gates}
    assert len(unique) == 14, (
        f"expected 14 unique catalog wire strings, got {len(unique)}: "
        f"{sorted(unique)}"
    )
    assert "forecast_or_analog_mode_resolved" not in unique


# ── MUST-1b (OQ-2): orphan evaluator + dispatch branch removed ──────────────


def test_forecast_or_analog_evaluator_removed() -> None:
    from prismpy.packaging import manifest as _m

    assert not hasattr(_m, "_eval_gate_forecast_or_analog_mode_resolved"), (
        "_eval_gate_forecast_or_analog_mode_resolved must be REMOVED (MUST-1b)."
    )
    assert hasattr(_m, "_eval_gate_n_years_gte_4")
    assert hasattr(_m, "_eval_gate_n_years_gte_30_for_forecast_adequacy")

    tree = _load_manifest_ast()
    dispatch = _find_function(tree, "_dispatch_gate")
    assert dispatch is not None, "_dispatch_gate not found in manifest.py"
    dispatch_strings = _string_constants(dispatch)
    assert "forecast_or_analog_mode_resolved" not in dispatch_strings, (
        "_dispatch_gate must not reference forecast_or_analog_mode_resolved."
    )
    assert "n_years_gte_4" in dispatch_strings
    assert "n_years_gte_30_for_forecast_adequacy" in dispatch_strings


# ── MUST-5 / SF-1: templated adequacy flag carrying actual N ────────────────


def test_uc1_adequacy_template_present_and_carries_n() -> None:
    tree = _load_manifest_ast()
    string_constants = _string_constants(tree)
    anchor = "forecast_adequacy:n_years_"
    matches = [c for c in string_constants if c.startswith(anchor)]
    assert matches, (
        f"UC1 adequacy advisory template ({anchor}...) not found in "
        f"manifest.py string constants."
    )
    assert any("{n}" in c for c in matches), (
        "the adequacy template must interpolate the actual N via ``{n}`` "
        "(SF-1: a 4-yr and a 28-yr package must be distinguishable)."
    )


def test_uc1_gte_4_evaluator_floor() -> None:
    """The gte_4 evaluator passes on >=4-year spans, fails below (BL-1)."""
    from prismpy.packaging import manifest as _m

    assert _m._eval_gate_n_years_gte_4({"start_year": 2017, "end_year": 2020})
    assert not _m._eval_gate_n_years_gte_4({"start_year": 2018, "end_year": 2020})


# ── Interface: end-to-end emit (no gates_failed + adequacy flag carries N) ──


def test_uc1_4yr_emits_no_gates_failed_and_adequacy_flag_with_n() -> None:
    from prismpy.packaging.manifest import canonical_uc_readiness_emitter

    project_config = {
        "start_year": 2017,
        "end_year": 2020,  # 4 years
        "use_case_config": {"yield_forecast": {}},
    }
    manifest_so_far = {
        "cells": [{"id": 1}],
        "crops": [{"name": "Maize"}],
        "crop": {"name": "Maize"},
        "region": {"name": "Sahel"},
    }
    out = canonical_uc_readiness_emitter(project_config, "acea", manifest_so_far)
    uc1 = out["yield_forecast"]
    assert "n_years_gte_4" in uc1["gates_passed"]
    assert not uc1.get("gates_failed"), (
        f"4-yr UC1 must have NO hard gates_failed; got {uc1.get('gates_failed')}"
    )
    assert (
        "forecast_adequacy:n_years_4_lt_30_wide_sampling_uncertainty"
        in uc1["advisory_flags"]
    ), f"adequacy flag must carry N=4; got {uc1['advisory_flags']}"


def test_uc1_3yr_lands_in_gates_failed() -> None:
    """V-6 (BL-1 false-ready closed at gen): a 3-year UC1 package FAILS
    n_years_gte_4 → lands in gates_failed → NOT advertised UC1-ready."""
    from prismpy.packaging.manifest import canonical_uc_readiness_emitter

    project_config = {
        "start_year": 2018,
        "end_year": 2020,  # 3 years
        "use_case_config": {"yield_forecast": {}},
    }
    manifest_so_far = {
        "cells": [{"id": 1}],
        "crops": [{"name": "Maize"}],
        "crop": {"name": "Maize"},
        "region": {"name": "Sahel"},
    }
    out = canonical_uc_readiness_emitter(project_config, "acea", manifest_so_far)
    uc1 = out["yield_forecast"]
    failed_ids = {g["gate_id"] for g in uc1.get("gates_failed", [])}
    assert "n_years_gte_4" in failed_ids, (
        f"3-yr UC1 must fail n_years_gte_4 (BL-1 false-ready closed at gen); "
        f"gates_failed={uc1.get('gates_failed')}"
    )
