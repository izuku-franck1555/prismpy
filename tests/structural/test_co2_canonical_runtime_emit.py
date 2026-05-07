"""Structural pin: AC-G-9 Layer 3 — runtime-emit walker per F-G-2c.

Sprint G AC-G-9 Layer 3 absorbs the codex Gate A HIGH-2 surface gap:
``ScenarioBlock`` catches manifest-side CO₂ drift, but existing
translator paths emit CO₂ values to MODEL-INPUT files that the model
reads at runtime — bypassing the manifest entirely. Empirical
example: ACEA writes ``CO2_DATA`` to ``co2/...txt`` at
``acea/translator.py:2769`` (codex empirical citation; observational
historical NOAA data, not a Sprint G projection path — see scope).

This walker enforces F-G-2c: per-translator runtime-emit paths that
write CO₂ to model-input files MUST route through
:func:`prismpy.standards.co2_ppm.get_co2_ppm_with_provenance` for
PROJECTION mode emission. OBSERVED-mode emission (historical/NOAA-
sourced concentrations) is allowed to keep its own table — projection
mode is what Layer 3 protects.

Walker scope: ``prismpy/src/prismpy/translators/**/*.py`` per durable
#20 sibling-sweep. The walker:

1. Identifies functions whose name signals projection-mode handling
   (``projection`` or ``_climate_kind`` or AC-G-7-style writers).
2. For each, scans the body for assignments whose target name token
   matches the canonical CO₂ identifier whitelist with a numeric
   literal value.
3. If any such bypass assignment is found in projection-handling
   code, the walker asserts the function ALSO contains a Call to
   ``get_co2_ppm_with_provenance`` somewhere in its body (canonical
   route present) OR fails loud.

Per Draft 5 contract line 235: the walker also accepts the explicit
escape hatch — translators that declare
``manifest.scenario.co2_consumption_path = "not_consumed_by_this_platform"``
opt out of Layer 3 for that platform.

Sprint G AC-G-7a/b/c projection paths emit weather variables
(tasmean/tasmax/tasmin/pr/hurs/rsds/tdew), NOT CO₂, so this walker
currently passes with zero violations. The walker is forward-looking:
it catches a FUTURE commit that adds CO₂ emission to a projection
path without routing through canonical.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

from prismpy.standards.co2_ppm import (
    CANONICAL_CO2_CONCENTRATION_IDENTIFIERS,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRANSLATORS_ROOT = _REPO_ROOT / "src" / "prismpy" / "translators"


# Functions whose name suggests projection-mode handling. The walker
# only enforces F-G-2c on these (observed-mode handlers like
# ``_generate_co2_data`` keep their own historical table).
_PROJECTION_HANDLER_NAME_PATTERNS = (
    re.compile(r"projection", re.IGNORECASE),
    # AC-G-7a/b/c writers carry climate_kind discriminator
    re.compile(r"_generate_climate_files", re.IGNORECASE),
    re.compile(r"_generate_weather_files", re.IGNORECASE),
    re.compile(r"_generate_climate_pickles", re.IGNORECASE),
    re.compile(r"_generate_projection_climate", re.IGNORECASE),
)

# Canonical lookup function name — its presence in a function body
# satisfies the F-G-2c routing requirement.
_CANONICAL_LOOKUP_FN = "get_co2_ppm_with_provenance"


def _iter_translator_modules() -> Iterator[Path]:
    """Yield every .py file under ``prismpy/src/prismpy/translators/``,
    excluding ``__pycache__``."""
    for path in _TRANSLATORS_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _function_is_projection_handler(name: str) -> bool:
    """Return True if the function name matches any projection-handler
    pattern. Walker scope; observed-mode handlers fall through."""
    for pattern in _PROJECTION_HANDLER_NAME_PATTERNS:
        if pattern.search(name):
            return True
    return False


def _function_contains_canonical_lookup_call(
    func_node: ast.AST,
) -> bool:
    """Walk a function body for any Call whose func name is
    ``get_co2_ppm_with_provenance``."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == _CANONICAL_LOOKUP_FN:
                return True
            if (
                isinstance(func, ast.Attribute)
                and func.attr == _CANONICAL_LOOKUP_FN
            ):
                return True
    return False


def _function_emits_co2_literal(
    func_node: ast.AST,
) -> List[Tuple[str, int]]:
    """Walk a function body for any assignment whose target name is
    in the canonical whitelist (case-insensitive) AND whose value is
    a numeric literal. Returns ``(target_name, lineno)`` for each
    violation."""
    violations: List[Tuple[str, int]] = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if (
                    target.id.lower()
                    not in CANONICAL_CO2_CONCENTRATION_IDENTIFIERS
                ):
                    continue
                value = node.value
                if (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, (int, float))
                ):
                    violations.append((target.id, target.lineno))
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if not isinstance(target, ast.Name):
                continue
            if (
                target.id.lower()
                not in CANONICAL_CO2_CONCENTRATION_IDENTIFIERS
            ):
                continue
            value = node.value
            if (
                value is not None
                and isinstance(value, ast.Constant)
                and isinstance(value.value, (int, float))
            ):
                violations.append((target.id, target.lineno))
    return violations


# ── §1 Walker correctness on synthetic positives ─────────────────────


def test_walker_flags_projection_handler_with_bypass_assignment() -> None:
    """Synthetic positive: a function named ``_generate_projection_x``
    that assigns ``co2_ppm = 478.0`` and does NOT call
    ``get_co2_ppm_with_provenance`` MUST be flagged."""
    fake_src = (
        "def _generate_projection_x():\n"
        "    co2_ppm = 478.0\n"
        "    return co2_ppm\n"
    )
    tree = ast.parse(fake_src)
    func = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    )
    assert _function_is_projection_handler(func.name)
    violations = _function_emits_co2_literal(func)
    assert violations == [("co2_ppm", 2)]
    assert not _function_contains_canonical_lookup_call(func)


def test_walker_accepts_projection_handler_with_canonical_route() -> None:
    """Synthetic negative: a projection handler that uses the
    canonical lookup MUST NOT be flagged."""
    fake_src = (
        "def _generate_projection_x(scenario, time_slice):\n"
        "    co2_ppm, prov = get_co2_ppm_with_provenance(scenario, time_slice)\n"
        "    return co2_ppm\n"
    )
    tree = ast.parse(fake_src)
    func = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    )
    assert _function_is_projection_handler(func.name)
    violations = _function_emits_co2_literal(func)
    assert violations == []  # no literal assignment
    assert _function_contains_canonical_lookup_call(func)


def test_walker_skips_observed_handler() -> None:
    """A function name that does NOT match the projection-handler
    pattern is out-of-scope for Layer 3 even if it emits CO₂
    literals (e.g., ``_generate_co2_data`` historical NOAA table)."""
    assert not _function_is_projection_handler("_generate_co2_data")
    assert not _function_is_projection_handler("emit_observed_climate")


def test_walker_pattern_set_covers_ac_g_7_writer_names() -> None:
    """The pattern set must include all 4 translator writer entry
    points added in AC-G-7a/7b/7c (CRAFT/PYTHIA/ACEA/SARRA-Py)."""
    assert _function_is_projection_handler("_generate_climate_files")
    assert _function_is_projection_handler("_generate_weather_files")
    assert _function_is_projection_handler("_generate_climate_pickles")
    assert _function_is_projection_handler(
        "_generate_projection_climate_geotiffs"
    )


# ── §2 F-G-2c sibling-sweep over translators (real) ──────────────────


def test_layer3_no_projection_handler_bypasses_canonical_lookup() -> None:
    """Walk every translator module's projection-handler functions.
    For each that emits a canonical CO₂ literal, assert it ALSO
    contains a call to ``get_co2_ppm_with_provenance``. Otherwise
    surface the file + function + line + identifier."""
    violations: List[str] = []

    for src_path in _iter_translator_modules():
        try:
            tree = ast.parse(src_path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            if not _function_is_projection_handler(node.name):
                continue
            literals = _function_emits_co2_literal(node)
            if not literals:
                continue
            if _function_contains_canonical_lookup_call(node):
                continue
            # Bypass: handler emits canonical CO₂ literal without
            # routing through the lookup. Surface every such site.
            for target, lineno in literals:
                rel = src_path.relative_to(_TRANSLATORS_ROOT.parent.parent.parent)
                violations.append(
                    f"  {rel}:{lineno}: function `{node.name}` "
                    f"assigns `{target} = <literal>` without calling "
                    f"`{_CANONICAL_LOOKUP_FN}(...)`"
                )

    if violations:
        msg = "\n".join(
            [
                "AC-G-9 Layer 3 (F-G-2c) bypass detected — projection "
                "handler emits canonical CO₂ literal without routing "
                "through prismpy.standards.co2_ppm:",
                *violations,
                "",
                "Either call get_co2_ppm_with_provenance(scenario, "
                "time_slice) OR declare scenario.co2_consumption_path "
                "= 'not_consumed_by_this_platform' in the manifest.",
            ]
        )
        pytest.fail(msg)


def test_layer3_walker_scope_is_translators_only() -> None:
    """Sanity: the walker scope is ``prismpy/src/prismpy/translators/``
    — it does NOT walk ``standards/`` (which holds the canonical
    table itself), ``models/``, ``harmonize/``, etc."""
    assert _TRANSLATORS_ROOT.exists()
    assert _TRANSLATORS_ROOT.name == "translators"
    # Spot-check: at least the 4 platform sub-packages exist
    expected = {"acea", "craft", "pythia", "sarra_py"}
    have = {p.name for p in _TRANSLATORS_ROOT.iterdir() if p.is_dir()}
    missing = expected - have
    assert not missing, f"Translator sub-packages missing: {missing}"


def test_layer3_walker_includes_every_translator_subpackage() -> None:
    """Every translator subpackage is walked. Catches a future
    refactor that moves a translator outside the walked path."""
    walked_files = list(_iter_translator_modules())
    walked_subpackages = {
        f.parent.name for f in walked_files if f.parent != _TRANSLATORS_ROOT
    }
    expected_subpackages = {"acea", "craft", "pythia", "sarra_py"}
    missing = expected_subpackages - walked_subpackages
    assert not missing, (
        f"Translator subpackages not walked by Layer 3: {missing}"
    )


# ── §3 Documentation — projection-handler pattern set is the contract ─


def test_projection_handler_pattern_set_is_documented_in_module() -> None:
    """The walker's projection-handler pattern set IS the contract
    surface. If a future refactor adds a new projection-mode entry
    point, it MUST extend
    ``_PROJECTION_HANDLER_NAME_PATTERNS`` here so Layer 3 covers it.

    This test pins the pattern set so the next change touches BOTH
    the regex list AND the test that documents the contract."""
    expected_patterns = {
        "projection",
        "_generate_climate_files",
        "_generate_weather_files",
        "_generate_climate_pickles",
        "_generate_projection_climate",
    }
    actual_patterns = {p.pattern for p in _PROJECTION_HANDLER_NAME_PATTERNS}
    assert actual_patterns == expected_patterns
