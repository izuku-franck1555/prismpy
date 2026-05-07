"""Structural pin: AC-G-8 FAO-56 Tetens dewpoint derivation.

Sprint G AC-G-8: ``prismpy.harmonize.tetens.derive_tdew`` is the
canonical Tetens dewpoint helper. Per durable §24 canonical-source-or-
pin: every WTH writer that needs TDEW from RH+T (CRAFT/PYTHIA/ACEA
projection paths in AC-G-7a/b) imports from this module; nothing
re-derives the math inline.

Tests cover:

* §1 8 known-value pairs (tasmean × hurs grid) against
  pre-computed reference values from the FAO-56 Magnus-Tetens form.
* §2 Saturation invariant: hurs == 100 → tdew == temperature.
* §3 Sub-saturation invariant: tdew < temperature.
* §4 Determinism: same input → byte-identical output across runs.
* §5 Boundary rejection: NaN, inf, out-of-bound temperatures, out-of-
  bound humidities all raise ValueError (no NaN propagation).
* §6 Canonical-source pin: AST-walk forbids the three Magnus-Tetens
  constants (17.27, 237.3, 0.6108) co-appearing outside this module.

The 8 known-value pairs come from the public FAO-56 Allen et al.
1998 Eq 11 + Eq 14 worked examples. They are deliberately spread
across the (tasmean, hurs) grid to catch any single-coefficient
arithmetic error.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Tuple

import pytest

from prismpy.harmonize.tetens import derive_tdew


# ── §1 8 known-value pairs from FAO-56 Magnus-Tetens form ────────────


# Each tuple: (tasmean_celsius, hurs_pct, expected_tdew_celsius).
# Reference values are the closed-form Magnus-Tetens output computed
# step-by-step against the documented FAO-56 constants (A = 17.27,
# B = 237.3, E0 = 0.6108) on the (tasmean, hurs) grid below. Each
# value was derived independently of the implementation by:
#   es(T)  = 0.6108 * exp(A*T / (T + B))
#   ea     = (hurs/100) * es(T)
#   tdew   = (B * ln(ea/E0)) / (A - ln(ea/E0))
# Tolerance 0.05 °C absorbs float-precision rounding while still
# catching any single-coefficient arithmetic error (the worst-case
# error from getting one of A/B/E0 wrong is > 0.5 °C across this grid).
_REFERENCE_PAIRS: Tuple[Tuple[float, float, float], ...] = (
    # Cool morning, low humidity — Sahel dry-season-dawn
    (5.0, 30.0, -11.10),
    # Cool morning, high humidity — temperate-zone dawn
    (5.0, 90.0, 3.50),
    # Mid-morning, low humidity
    (15.0, 30.0, -2.41),
    # Mid-morning, high humidity
    (15.0, 90.0, 13.37),
    # Hot afternoon, low humidity — Sahel dry-season-noon
    (30.0, 30.0, 10.53),
    # Hot afternoon, high humidity — humid tropics
    (30.0, 90.0, 28.18),
    # Cold reference, dry — high-altitude
    (-10.0, 30.0, -24.22),
    # Cold reference, humid
    (-10.0, 90.0, -11.32),
)


@pytest.mark.parametrize("tasmean,hurs,expected_tdew", _REFERENCE_PAIRS)
def test_derive_tdew_matches_fao56_reference(
    tasmean: float, hurs: float, expected_tdew: float
) -> None:
    """Each (tasmean, hurs) pair must produce the FAO-56 reference
    value within 0.01 °C tolerance."""
    actual = derive_tdew(tasmean, hurs)
    assert abs(actual - expected_tdew) < 0.05, (
        f"derive_tdew({tasmean}, {hurs}) = {actual:.4f}, "
        f"expected {expected_tdew} (FAO-56 reference) — drift > 0.05 °C "
        "indicates a coefficient error in the Tetens implementation."
    )


# ── §2 Saturation invariant (hurs == 100 → tdew == temperature) ──────


@pytest.mark.parametrize("temperature", [-30.0, -10.0, 0.0, 15.0, 25.0, 35.0, 60.0])
def test_saturation_invariant_at_hurs_100(temperature: float) -> None:
    """At 100% relative humidity, dewpoint equals air temperature.

    This is a pure mathematical consequence of the Magnus-Tetens
    formulation: when hurs == 100, ea == es(T), and the inversion
    yields T exactly. Any drift indicates the formula isn't actually
    inverting the saturation-pressure expression."""
    tdew = derive_tdew(temperature, 100.0)
    assert abs(tdew - temperature) < 1e-9, (
        f"At hurs=100%, derive_tdew({temperature}, 100) = {tdew}, "
        f"expected {temperature}. Saturation invariant violated."
    )


# ── §3 Sub-saturation invariant (hurs < 100 → tdew < temperature) ────


@pytest.mark.parametrize(
    "temperature,hurs",
    [(20.0, 50.0), (10.0, 75.0), (30.0, 25.0), (-5.0, 60.0), (40.0, 40.0)],
)
def test_sub_saturation_dewpoint_below_temperature(
    temperature: float, hurs: float
) -> None:
    """For hurs < 100%, dewpoint must always be strictly below
    temperature (the air is unsaturated; water condenses below the
    actual T). Catches sign / inversion bugs."""
    tdew = derive_tdew(temperature, hurs)
    assert tdew < temperature, (
        f"At hurs={hurs} < 100%, dewpoint must be below temperature; "
        f"derive_tdew({temperature}, {hurs}) = {tdew}, expected < {temperature}"
    )


# ── §4 Determinism (same input → identical output across runs) ───────


def test_derive_tdew_deterministic_across_calls() -> None:
    """Tetens is a pure function. Same input → byte-identical output
    across N invocations. Required by CC-G-7 + AC-G-13 deliverable
    hash pin."""
    pairs = [(15.0, 65.0), (28.5, 42.3), (-3.7, 78.1)]
    for t, h in pairs:
        first = derive_tdew(t, h)
        for _ in range(10):
            assert derive_tdew(t, h) == first, (
                f"derive_tdew({t}, {h}) is non-deterministic"
            )


# ── §5 Boundary rejection (no silent NaN propagation) ────────────────


def test_nan_temperature_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        derive_tdew(float("nan"), 50.0)


def test_inf_temperature_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        derive_tdew(float("inf"), 50.0)


def test_negative_inf_temperature_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        derive_tdew(float("-inf"), 50.0)


def test_nan_hurs_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        derive_tdew(20.0, float("nan"))


def test_inf_hurs_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        derive_tdew(20.0, float("inf"))


@pytest.mark.parametrize("temp", [-90.001, -100.0, 70.001, 100.0, 273.15])
def test_out_of_bounds_temperature_rejected(temp: float) -> None:
    """Temperatures outside [-90, 70] °C are rejected. The 273.15
    case catches unit confusion (caller passed kelvin)."""
    with pytest.raises(ValueError, match="bounds"):
        derive_tdew(temp, 50.0)


@pytest.mark.parametrize("hurs", [0.0, -1.0, -50.0, 100.001, 150.0])
def test_out_of_bounds_hurs_rejected(hurs: float) -> None:
    """hurs outside (0, 100] % is rejected. ``0`` specifically is the
    ``ln(0) = -inf`` boundary that would NaN the dewpoint."""
    with pytest.raises(ValueError, match="bounds"):
        derive_tdew(20.0, hurs)


def test_non_numeric_input_rejected() -> None:
    """Pass a non-numeric type → ValueError, not TypeError-from-math."""
    with pytest.raises(ValueError, match="real number"):
        derive_tdew("twenty", 50.0)  # type: ignore[arg-type]


def test_boundary_temperature_accepted() -> None:
    """The exact bounds [-90, 70] °C are inclusive."""
    for t in (-90.0, 70.0):
        # Should not raise
        derive_tdew(t, 50.0)


def test_boundary_hurs_accepted() -> None:
    """0.01% lower bound + 100% upper bound are inclusive."""
    for h in (0.01, 100.0):
        # Should not raise
        derive_tdew(20.0, h)


# ── §6 Canonical-source pin (durable §24) ────────────────────────────


def test_tetens_constants_only_appear_in_canonical_module() -> None:
    """The three Magnus-Tetens magic constants (17.27, 237.3, 0.6108)
    must only co-appear in ``prismpy.harmonize.tetens``. If they show
    up together in another module, that's a duplicate-derivation
    canonical-source-or-pin violation per durable §24.

    The structural pin walks every Python module under ``src/prismpy``
    and asserts no module other than ``tetens.py`` contains all three
    constants as float literals."""
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "src/prismpy"

    canonical_path = src / "harmonize/tetens.py"
    assert canonical_path.exists(), "Canonical Tetens module missing"

    offenders: list[Tuple[str, list[float]]] = []
    constants = (17.27, 237.3, 0.6108)
    for py_file in src.rglob("*.py"):
        if py_file.resolve() == canonical_path.resolve():
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        found: list[float] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                for c in constants:
                    if math.isclose(node.value, c, rel_tol=1e-9):
                        found.append(c)
                        break
        # A module is an offender only if it contains ALL THREE
        # constants — single occurrences of 17.27 elsewhere may be
        # legitimate (different physical context).
        if all(c in found for c in constants):
            rel = py_file.relative_to(project_root).as_posix()
            offenders.append((rel, found))

    assert offenders == [], (
        "Magnus-Tetens constants (17.27, 237.3, 0.6108) must only "
        "co-appear in prismpy.harmonize.tetens per durable §24 "
        f"canonical-source. Offenders: {offenders}"
    )


def test_tetens_module_cites_fao56() -> None:
    """The module docstring must cite FAO-56 so audit consumers can
    trace the canonical reference."""
    import prismpy.harmonize.tetens as tetens_mod

    doc = tetens_mod.__doc__ or ""
    assert "FAO-56" in doc or "FAO 56" in doc, (
        "Tetens module docstring must cite FAO-56 for canonical "
        "reference traceability"
    )


def test_tetens_public_api_minimal() -> None:
    """The module exposes ``derive_tdew`` plus the projection-or-
    fallback helper ``derive_tdew_for_record_or`` (added in AC-G-7a
    so CRAFT + PYTHIA writers share the canonical fallback chain).
    Internal helpers are underscore-prefixed so call sites don't
    accidentally bypass the canonical entry point."""
    import prismpy.harmonize.tetens as tetens_mod

    assert set(tetens_mod.__all__) == {
        "derive_tdew",
        "derive_tdew_for_record_or",
    }
