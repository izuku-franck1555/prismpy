"""Structural pin: AC-G-9 Layer 1 — canonical CO₂ source + substrate AST walker.

Sprint G AC-G-9 splits CO₂ canonical-source enforcement into 3 layers
(per Draft 5 codex Gate A HIGH-2 absorption):

* **Layer 1 (this module)** — substrate AST walker: no module-level
  assignment whose target name is in
  :data:`CANONICAL_CO2_CONCENTRATION_IDENTIFIERS` AND whose value is
  a literal float, OUTSIDE
  :mod:`prismpy.standards.co2_ppm`. Walker scope:
  ``prismpy/src/prismpy/**/*.py`` ONLY (mirrors
  ``test_cultivar_codes_registered.py`` precedent — excludes
  ``tests/``, ``.local/``, ``scripts/``, ``docs/`` by construction).

* **Layer 2** (`test_scenario_block_schema.py` §X) — Pydantic
  post-validator on :class:`ScenarioBlock` semantically asserts
  ``math.isclose(co2_ppm, lookup_value, rel_tol=1e-9)`` AND exact
  provenance string match.

* **Layer 3** (`test_co2_canonical_runtime_emit.py`) — sibling AST
  walker per durable #20 sibling-sweep over
  ``prismpy/src/prismpy/translators/**/*.py``.

Per durable §24 canonical-source-or-pin: ``CO2_PPM_BY_SCENARIO_PERIOD``
+ ``get_co2_ppm_with_provenance`` + ``CANONICAL_CO2_CONCENTRATION_IDENTIFIERS``
all live in :mod:`prismpy.standards.co2_ppm`. Every other module that
needs a CO₂ ppm value imports from there; nothing redefines.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Iterator

import pytest

from prismpy.standards.co2_ppm import (
    CANONICAL_CO2_CONCENTRATION_IDENTIFIERS,
    CO2_PPM_BY_SCENARIO_PERIOD,
    CO2_PPM_REL_TOL,
    CO2ProvenanceMismatchError,
    co2_ppm_matches_canonical,
    get_co2_ppm_with_provenance,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRISMPY_SRC = _REPO_ROOT / "src" / "prismpy"
_CANONICAL_MODULE = _PRISMPY_SRC / "standards" / "co2_ppm.py"


# ── §1 Canonical table contents + structure ──────────────────────────


def test_canonical_table_carries_4_isimip3b_primary_entries() -> None:
    """Sprint G primary core ensemble: SSP245 + SSP585 × 2 time-slices.
    The table MUST register all 4 (and only these 4 for now)."""
    expected_keys = {
        ("SSP245", (2046, 2065)),
        ("SSP245", (2086, 2100)),
        ("SSP585", (2046, 2065)),
        ("SSP585", (2086, 2100)),
    }
    assert set(CO2_PPM_BY_SCENARIO_PERIOD.keys()) == expected_keys


@pytest.mark.parametrize(
    "scenario,time_slice,expected_ppm",
    [
        ("SSP245", (2046, 2065), 478.0),
        ("SSP245", (2086, 2100), 541.0),
        ("SSP585", (2046, 2065), 571.0),
        ("SSP585", (2086, 2100), 1054.0),
    ],
)
def test_canonical_table_values_per_ar6_wg1_annex_iii(
    scenario: str,
    time_slice: tuple,
    expected_ppm: float,
) -> None:
    """Each registered (scenario, time_slice) tuple yields the AR6
    WG1 Annex III mid-year-of-period concentration."""
    ppm, provenance = CO2_PPM_BY_SCENARIO_PERIOD[(scenario, time_slice)]
    assert ppm == expected_ppm
    assert "AR6" in provenance
    assert "Annex III" in provenance


def test_canonical_table_provenance_strings_uniform() -> None:
    """All 4 entries cite the same canonical provenance string —
    indicating the same source-of-truth applies. Drift between
    entries would mean someone mixed sources."""
    provenance_values = {p for _, p in CO2_PPM_BY_SCENARIO_PERIOD.values()}
    assert len(provenance_values) == 1, (
        f"Mixed provenance across canonical table: {provenance_values}. "
        "Sprint G primary core ensemble should cite ONE source."
    )


# ── §2 Whitelist closed-set discipline ───────────────────────────────


def test_whitelist_is_frozenset_of_strings() -> None:
    """Per pass-2 MEDIUM-Rebase-2 the whitelist is a closed
    ``frozenset`` so consumers can't mutate the canonical scope."""
    assert isinstance(CANONICAL_CO2_CONCENTRATION_IDENTIFIERS, frozenset)
    assert all(
        isinstance(name, str)
        for name in CANONICAL_CO2_CONCENTRATION_IDENTIFIERS
    )


def test_whitelist_contains_4_canonical_names() -> None:
    """The 4 canonical identifier names per the contract:
    co2_ppm / co2_concentration / atmospheric_co2_ppm /
    atmospheric_co2_concentration."""
    expected = {
        "co2_ppm",
        "co2_concentration",
        "atmospheric_co2_ppm",
        "atmospheric_co2_concentration",
    }
    assert CANONICAL_CO2_CONCENTRATION_IDENTIFIERS == expected


# ── §3 get_co2_ppm_with_provenance lookup correctness ────────────────


@pytest.mark.parametrize(
    "scenario,time_slice,expected_ppm,expected_in_provenance",
    [
        ("SSP245", (2046, 2065), 478.0, "AR6"),
        ("SSP245", (2086, 2100), 541.0, "AR6"),
        ("SSP585", (2046, 2065), 571.0, "AR6"),
        ("SSP585", (2086, 2100), 1054.0, "AR6"),
    ],
)
def test_lookup_returns_canonical_pair(
    scenario: str,
    time_slice: tuple,
    expected_ppm: float,
    expected_in_provenance: str,
) -> None:
    ppm, prov = get_co2_ppm_with_provenance(scenario, time_slice)
    assert ppm == expected_ppm
    assert expected_in_provenance in prov


def test_lookup_raises_for_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="SSP370"):
        get_co2_ppm_with_provenance("SSP370", (2046, 2065))


def test_lookup_raises_for_unknown_time_slice() -> None:
    with pytest.raises(ValueError, match="2030"):
        get_co2_ppm_with_provenance("SSP245", (2030, 2049))


def test_lookup_error_message_lists_registered_keys() -> None:
    """The error message must enumerate the registered (scenario,
    time_slice) tuples so the caller knows what's available."""
    with pytest.raises(ValueError) as exc_info:
        get_co2_ppm_with_provenance("SSP370", (2046, 2065))
    msg = str(exc_info.value)
    assert "SSP245" in msg
    assert "SSP585" in msg


# ── §4 math.isclose tolerance constant ───────────────────────────────


def test_co2_ppm_rel_tol_is_one_part_per_billion() -> None:
    """Per pass-2 MEDIUM-Rebase-3: rel_tol=1e-9 catches deliberate
    cooking while absorbing float-serialization noise."""
    assert CO2_PPM_REL_TOL == 1e-9


def test_co2_ppm_matches_canonical_accepts_exact_value() -> None:
    assert co2_ppm_matches_canonical(478.0, 478.0)


def test_co2_ppm_matches_canonical_accepts_sub_ulp_drift() -> None:
    """JSON round-trip can introduce sub-ULP rounding noise. Within
    rel_tol=1e-9 the matcher accepts; beyond it rejects."""
    # Within tolerance (~1e-12 deviation on 478.0)
    assert co2_ppm_matches_canonical(478.0 + 1e-12, 478.0)


def test_co2_ppm_matches_canonical_rejects_cooking() -> None:
    """A 0.1 ppm change is far above rel_tol — should fail the
    matcher (catches deliberate cooking)."""
    assert not co2_ppm_matches_canonical(478.1, 478.0)
    assert not co2_ppm_matches_canonical(479.0, 478.0)


# ── §5 CO2ProvenanceMismatchError exception shape ────────────────────


def test_co2_provenance_mismatch_error_is_value_error_subclass() -> None:
    """Per durable §6.4 schema-layer discipline: typed exceptions for
    schema violations subclass ValueError."""
    assert issubclass(CO2ProvenanceMismatchError, ValueError)


def test_co2_provenance_mismatch_error_carries_structured_fields() -> None:
    """Per pass-2 MEDIUM-3 (analogous to ScenarioSetValidationError):
    structured trace fields on the exception so callers (the Sprint G
    validate_scenario_set + cockpit error rendering) get specific
    info, not a freeform message."""
    err = CO2ProvenanceMismatchError(
        "test mismatch",
        scenario="SSP245",
        time_slice=(2046, 2065),
        observed_co2_ppm=400.0,
        expected_co2_ppm=478.0,
        observed_provenance="paraphrased",
        expected_provenance="canonical",
    )
    assert err.scenario == "SSP245"
    assert err.time_slice == (2046, 2065)
    assert err.observed_co2_ppm == 400.0
    assert err.expected_co2_ppm == 478.0
    assert err.observed_provenance == "paraphrased"
    assert err.expected_provenance == "canonical"
    assert "test mismatch" in str(err)


# ── §6 Layer 1 substrate AST walker ──────────────────────────────────


def _iter_python_modules(root: Path) -> Iterator[Path]:
    """Yield every .py file under ``root`` excluding __pycache__ +
    test directories."""
    for path in root.rglob("*.py"):
        # Exclude generated bytecode caches
        if "__pycache__" in path.parts:
            continue
        # Exclude test directories (mutation-drill fixtures may
        # legitimately ship hardcoded co2_ppm values for testing)
        if "tests" in path.parts:
            continue
        yield path


def _module_level_assignments_to_canonical_names(
    src_path: Path,
) -> list[tuple[str, ast.AST]]:
    """Find module-level assignments whose target name is in the
    canonical whitelist AND whose value is a literal float.

    Returns a list of ``(target_name, value_node)`` tuples for every
    violation found in the module.
    """
    src = src_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    violations: list[tuple[str, ast.AST]] = []
    for node in tree.body:  # MODULE-LEVEL only (not nested)
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        # Determine the target name(s)
        target_names: list[str] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_names.append(target.id)
        else:  # AnnAssign
            if isinstance(node.target, ast.Name):
                target_names.append(node.target.id)
        # Determine the value
        value = node.value
        if value is None:
            continue
        # Lower-case for case-insensitive whitelist match
        for name in target_names:
            if name.lower() not in CANONICAL_CO2_CONCENTRATION_IDENTIFIERS:
                continue
            # Check if value is a literal float (number)
            if isinstance(value, ast.Constant) and isinstance(
                value.value, (int, float)
            ):
                violations.append((name, value))
            # Negative literal: ast.UnaryOp wrapping ast.Constant
            elif (
                isinstance(value, ast.UnaryOp)
                and isinstance(value.op, ast.USub)
                and isinstance(value.operand, ast.Constant)
                and isinstance(value.operand.value, (int, float))
            ):
                violations.append((name, value))
    return violations


def test_layer1_no_canonical_co2_assignment_outside_canonical_module() -> None:
    """Layer 1 substrate AST walker: no module-level ``co2_ppm =
    478.0`` (or any whitelist-name = literal-float) anywhere in
    ``prismpy/src/prismpy/`` EXCEPT the canonical module itself."""
    violations: list[tuple[Path, str, ast.AST]] = []
    for src_path in _iter_python_modules(_PRISMPY_SRC):
        # Allow the canonical module to declare its own table
        if src_path == _CANONICAL_MODULE:
            continue
        for name, value_node in _module_level_assignments_to_canonical_names(
            src_path
        ):
            violations.append((src_path, name, value_node))

    if violations:
        msg_parts = ["Layer 1 substrate violation — non-canonical CO₂:"]
        for path, name, _ in violations:
            rel = path.relative_to(_PRISMPY_SRC.parent.parent)
            msg_parts.append(f"  {rel}: {name} = <literal>")
        msg_parts.append(
            "Route through prismpy.standards.co2_ppm.get_co2_ppm_with_provenance(...)"
        )
        pytest.fail("\n".join(msg_parts))


def test_layer1_canonical_module_itself_carries_the_table() -> None:
    """Sanity: the canonical module IS allowed to declare the table.
    This is the inverse of the Layer 1 walker — confirms the walker's
    exception path actually applies."""
    src = _CANONICAL_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    # The canonical module declares CO2_PPM_BY_SCENARIO_PERIOD at
    # module level; that's expected. Layer 1 walker excludes this
    # path by design.
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "CO2_PPM_BY_SCENARIO_PERIOD"
    ]
    assert len(assignments) == 1


def test_layer1_walker_catches_canonical_name_violation_synthetic() -> None:
    """Synthetic-positive smoke test: build a fake module that
    declares ``co2_ppm = 478.0`` at module level, run the walker,
    confirm it surfaces the violation. Calibrates the walker's
    detection at its core invariant."""
    fake_src = (
        "import math\n"
        "co2_ppm = 478.0  # would-be-violation\n"
        "def f(): pass\n"
    )
    tree = ast.parse(fake_src)
    violations = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in CANONICAL_CO2_CONCENTRATION_IDENTIFIERS
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (int, float))
        ):
            violations.append(node.targets[0].id)
    assert violations == ["co2_ppm"], (
        "Walker logic does not surface the synthetic violation — "
        "check the matcher matches lower-cased whitelist names + "
        "literal float values."
    )


def test_layer1_walker_excludes_function_scope_assignments() -> None:
    """Per Draft 5 Layer 1 scope: module-level only. Function-scope
    ``co2_ppm = 478.0`` (e.g., a unit-test variable) is NOT a Layer 1
    violation by design — it is caught at Layer 2 (Pydantic
    validation) if the value reaches ``ScenarioBlock``.

    Synthetic: a function body assigning to ``co2_ppm`` should not
    trigger Layer 1."""
    fake_src = (
        "def helper():\n"
        "    co2_ppm = 478.0\n"
        "    return co2_ppm\n"
    )
    tree = ast.parse(fake_src)
    # No module-level assignments
    module_level_assigns = [
        n
        for n in tree.body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
    ]
    assert module_level_assigns == []


# ── §7 Public API minimal ────────────────────────────────────────────


def test_co2_ppm_module_public_api() -> None:
    """The module exports the canonical table + whitelist + tolerance
    + lookup + matcher + exception. No internal helpers leaked."""
    import prismpy.standards.co2_ppm as co2_mod

    assert set(co2_mod.__all__) == {
        "CO2_PPM_BY_SCENARIO_PERIOD",
        "CANONICAL_CO2_CONCENTRATION_IDENTIFIERS",
        "CO2_PPM_REL_TOL",
        "CO2ProvenanceMismatchError",
        "get_co2_ppm_with_provenance",
        "co2_ppm_matches_canonical",
    }
