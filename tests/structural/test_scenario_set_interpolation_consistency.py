"""Structural pin: AC-E2-22 cross-document validator integration.

Asserts ``validate_scenario_set`` invokes
``validate_manifest_cell_summary_consistency`` for each package's
manifest + cell_summary pair. The integration runs at scenario-set
validation time so a manifest flag drift can't slip past the
validator boundary.
"""

from __future__ import annotations

import ast
from pathlib import Path

from prismpy.validators import scenario_set


def _scenario_set_module_path() -> Path:
    return Path(scenario_set.__file__)


def test_validator_module_calls_interpolation_consistency() -> None:
    """AST walker over scenario_set.py: assert
    ``_validate_interpolation_consistency`` is called inside
    ``validate_scenario_set``. Drift (someone deletes the call
    accidentally) fires this pin."""
    src = _scenario_set_module_path().read_text(encoding="utf-8")
    tree = ast.parse(src)

    found_in_validate_scenario_set = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "validate_scenario_set"
        ):
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "_validate_interpolation_consistency"
                ):
                    found_in_validate_scenario_set = True
                    break
    assert found_in_validate_scenario_set, (
        "validate_scenario_set must call "
        "_validate_interpolation_consistency for cross-document "
        "manifest/cell_summary checking per AC-E2-22."
    )


def test_helper_imports_canonical_validator() -> None:
    """The integration helper imports
    ``validate_manifest_cell_summary_consistency`` from the
    canonical module (per §0.2 #4 cross-document validator)."""
    src = _scenario_set_module_path().read_text(encoding="utf-8")
    assert "from prismpy.validators.manifest_consistency import" in src
    assert "validate_manifest_cell_summary_consistency" in src


def test_helper_handles_missing_cell_summary_gracefully() -> None:
    """Forward-compat: legacy packages without cell_summary.json
    must not raise. The helper is a no-op when the file is absent."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # No cell_summary.json present.
        # Should NOT raise — legacy package compat.
        scenario_set._validate_interpolation_consistency(
            Path(tmp), {}, "test_label"
        )
