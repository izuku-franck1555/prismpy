"""Pin scenario_helpers production callsites — orphan-helper guard.

Sprint G + UC2 fix #277 introduced scenario-helper functions in
``prismpy.packaging.scenario_helpers``:

* :func:`build_baseline_scenario_block` — constructs a "baseline"-role
  ScenarioBlock for observed-climate baseline packages.
* :func:`build_baseline_scenario_block_for_period` — convenience
  wrapper that auto-derives label + CO₂ from period + region + crop.
* :func:`rewrite_pythia_config_for_scenario` — overwrites
  pythia_config.json year fields to match a scenario's time slice.
* :func:`estimate_observed_co2_ppm` — piecewise-linear CO₂ ppm lookup.

Per durable §24 canonical-source-or-pin: a canonical helper is
useful only when production code routes through it. A helper that
ships in the library but has zero non-test callsites is an
"orphan" — it advertises a behavior contract that nothing in
production exercises. Evaluator's UC2 round-1 verdict caught
exactly this class on commit ``2ecc0a9``: the helpers existed in
``scenario_helpers.py`` but the PYTHIA translator's manifest
emission path didn't import or call them.

This test pins the production callsites so future regressions
(refactor that drops the import; renaming that breaks the call
chain) fail loud at structural-test time.

Scope of the AST walker:
- ``src/prismpy/`` only — production code.
- Excludes ``tests/`` — test files use the helpers heavily, that's
  expected.
- Excludes ``packaging/scenario_helpers.py`` itself — the helper
  module's own self-reference is uninteresting.

For helpers consumed by external scripts (e.g.,
``rewrite_pythia_config_for_scenario`` is called by prismweb's
gitignored ``generate_pythia_climate_change_pkg.py``), this test
pins **export-from-library** rather than in-tree callsite. The
canonical-source contract is satisfied when the helper is
importable from a stable public path; the consumer wiring is the
consumer's lane.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


def _prismpy_src_dir() -> Path:
    """Resolve the on-disk ``src/prismpy`` directory."""
    import prismpy
    return Path(prismpy.__file__).parent


def _walk_production_calls(target_function_name: str) -> list[tuple[Path, int]]:
    """Find every call site of ``target_function_name`` in prismpy production code.

    AST walker scoped to ``src/prismpy/**/*.py`` excluding the
    helper-module-self-reference and any test paths. Returns a list
    of ``(file_path, line_number)`` tuples for inspection.
    """
    src = _prismpy_src_dir()
    callsites: list[tuple[Path, int]] = []

    for py_path in src.rglob("*.py"):
        # Exclude helper module's self-reference + any test paths.
        if "scenario_helpers" in py_path.name:
            continue
        if "tests" in py_path.parts:
            continue
        if "__pycache__" in py_path.parts:
            continue
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match `target(...)` (direct name).
                if isinstance(func, ast.Name) and func.id == target_function_name:
                    callsites.append((py_path, node.lineno))
                # Match `module.target(...)` (attribute access).
                elif (
                    isinstance(func, ast.Attribute)
                    and func.attr == target_function_name
                ):
                    callsites.append((py_path, node.lineno))
    return callsites


# ── CA-3 pins: helpers MUST have at least one production callsite ──


def test_build_baseline_scenario_block_for_period_called_in_production() -> None:
    """The PYTHIA translator's manifest emission path MUST call the
    period-aware baseline scenario builder.

    Without this callsite, the wired helper is orphaned: the
    library publishes a baseline-block builder, but production code
    never invokes it, and every PYTHIA package ships with
    ``manifest.scenario = null``. That's the regression evaluator
    caught on round 1 (commit ``2ecc0a9``); the round-2 fixup
    (CA-1) wires the call into ``_generate_manifest`` and this pin
    locks the wiring.
    """
    callsites = _walk_production_calls("build_baseline_scenario_block_for_period")
    assert len(callsites) >= 1, (
        "build_baseline_scenario_block_for_period has no production "
        "callsites. The helper is orphaned — the PYTHIA translator's "
        "manifest emission path must invoke it so every package "
        "carries a non-null manifest.scenario block. Re-wire the "
        "call in `translator.py:_generate_manifest`."
    )
    # Belt-and-suspenders: at least one of the callsites MUST live
    # in the PYTHIA translator's manifest emission path. The CA-1
    # wiring puts it there; future refactors that move the call
    # out of the translator surface this assertion.
    pythia_translator_callsites = [
        cs for cs in callsites
        if "translators/pythia" in str(cs[0]).replace("\\", "/")
    ]
    assert len(pythia_translator_callsites) >= 1, (
        "build_baseline_scenario_block_for_period must be called "
        f"from prismpy/translators/pythia/. Found callsites at: "
        f"{[(str(p), n) for p, n in callsites]}"
    )


# ── Export-from-library pin: helpers consumed by external scripts ──


def test_rewrite_pythia_config_for_scenario_is_publicly_importable() -> None:
    """The pythia_config year-rewriter is importable via the canonical path.

    The function is consumed by prismweb's gitignored
    ``generate_pythia_climate_change_pkg.py`` (CA-2 wiring), which
    this AST walker cannot see. The canonical-source contract is
    that the helper IS exported from a stable public path so the
    external consumer can reliably ``import`` it. This pin asserts
    the public-path contract; the consumer wiring is the
    consumer's lane.
    """
    module = importlib.import_module("prismpy.packaging.scenario_helpers")
    assert hasattr(module, "rewrite_pythia_config_for_scenario"), (
        "rewrite_pythia_config_for_scenario is no longer importable "
        "from prismpy.packaging.scenario_helpers. External consumers "
        "(prismweb climate-change script) rely on this stable public "
        "path; do NOT rename or move without coordinated update."
    )
    # Belt-and-suspenders: the function is callable.
    fn = module.rewrite_pythia_config_for_scenario
    assert callable(fn), (
        f"rewrite_pythia_config_for_scenario is not callable; "
        f"got {type(fn).__name__}."
    )


def test_build_baseline_scenario_block_is_publicly_importable() -> None:
    """The lower-level baseline-block builder stays importable.

    External consumers may use ``build_baseline_scenario_block``
    directly (full control over fields) or
    ``build_baseline_scenario_block_for_period`` (auto-derive from
    period + region + crop). This pin asserts the lower-level
    helper stays exported as the canonical building block.
    """
    module = importlib.import_module("prismpy.packaging.scenario_helpers")
    assert hasattr(module, "build_baseline_scenario_block")
    assert callable(module.build_baseline_scenario_block)


def test_estimate_observed_co2_ppm_is_publicly_importable() -> None:
    """The CO₂ piecewise-linear estimator stays importable.

    Used by the period-aware baseline-block builder; external
    consumers may also use it directly to look up observed-period
    CO₂ for non-PYTHIA workflows. Pin the public path.
    """
    module = importlib.import_module("prismpy.packaging.scenario_helpers")
    assert hasattr(module, "estimate_observed_co2_ppm")
    assert callable(module.estimate_observed_co2_ppm)
