"""Sprint E.2 AC-E2-3 ext + Codex Gate A MEDIUM A1 + Builder
Sub-CA #6 — :func:`prismpy.validators.affordance_routing.route_affordance`
callsites all pass the ``cell_failure_context`` kwarg.

The Sprint E.2 extension to :func:`route_affordance` added a
required ``cell_failure_context: Dict[str, Any]`` parameter so
the cockpit's per-cell routing engine + the future
``bucket_for`` triple consumer share the same per-cell metric
dict. Without a structural pin, a future caller could silently
omit the kwarg by relying on a TypeError at runtime — which
breaks late, in production, on real-data runs only.

Per durable §24 canonical-source-or-pin: every callsite passes
the kwarg explicitly, even if the value is an empty dict
(``{}``) — the empty dict is the canonical "no per-cell
context" sentinel. Drift detection: a future caller that uses
positional args + drops the trailing dict, OR that wraps
``route_affordance`` in a helper that doesn't surface the
parameter, fails this pin at CI time.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "prismpy"
TESTS_ROOT = REPO_ROOT / "tests"


def _find_route_affordance_callsites(
    source: str,
) -> list[tuple[int, ast.Call]]:
    """Walk the AST + return ``(line_no, ast.Call)`` for every
    invocation that resolves to ``route_affordance(...)``."""
    tree = ast.parse(source)
    callsites: list[tuple[int, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match plain ``route_affordance(...)`` + module-prefixed
        # ``affordance_routing.route_affordance(...)`` +
        # ``foo.route_affordance(...)`` (defensive — covers
        # imports that alias the module but keep the name).
        if isinstance(func, ast.Name) and func.id == "route_affordance":
            callsites.append((node.lineno, node))
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "route_affordance"
        ):
            callsites.append((node.lineno, node))
    return callsites


def _has_cell_failure_context_kwarg(call_node: ast.Call) -> bool:
    """True iff the call passes ``cell_failure_context=`` as a
    keyword argument OR a 6th positional argument."""
    for kw in call_node.keywords:
        if kw.arg == "cell_failure_context":
            return True
    # Positional arg #6 (index 5) covers the 6-positional form
    # ``route_affordance(check_id, platform, zone, elevation_m,
    # n_candidates_in_radius, cell_failure_context)``. Sprint
    # E.2 callers all use keyword form per the canonical
    # invocation pattern, but a positional pass is structurally
    # equivalent so the pin tolerates it.
    return len(call_node.args) >= 6


@pytest.fixture(scope="module")
def callsites() -> list[tuple[Path, int, ast.Call]]:
    """Discover every ``route_affordance(...)`` callsite across
    prismpy src + tests. Returns ``[(path, lineno, ast.Call)]``."""
    out: list[tuple[Path, int, ast.Call]] = []
    for root in (SRC_ROOT, TESTS_ROOT):
        for py_path in root.rglob("*.py"):
            try:
                text = py_path.read_text(encoding="utf-8")
            except OSError:
                continue
            try:
                hits = _find_route_affordance_callsites(text)
            except SyntaxError:
                continue
            for line_no, call_node in hits:
                out.append((py_path, line_no, call_node))
    # Skip the def itself in affordance_routing.py (one
    # ``ast.FunctionDef`` not an ``ast.Call``) — the AST walk
    # already filters via Call-only iteration above. Skip
    # imports + ``__all__`` references via the same filter.
    return out


def test_route_affordance_has_callsites_to_check(callsites) -> None:
    """Sanity floor — at least the test module's 10 callsites
    + the prismweb side gets discovered. Without callsites the
    pin would silently pass even with all-broken signatures."""
    assert len(callsites) >= 10, (
        f"route_affordance callsite discovery returned {len(callsites)}; "
        f"expected >= 10. Did the AST walker break? Or did all "
        f"callsites move outside prismpy src/tests? Check the "
        f"pin's coverage."
    )


def test_every_callsite_passes_cell_failure_context(callsites) -> None:
    """Every ``route_affordance(...)`` invocation MUST pass
    ``cell_failure_context`` (positional 6th OR keyword). A
    future caller that drops the kwarg fails this pin at CI."""
    offenders: list[str] = []
    for path, line_no, call_node in callsites:
        if not _has_cell_failure_context_kwarg(call_node):
            offenders.append(
                f"{path.relative_to(REPO_ROOT)}:{line_no}"
            )
    assert not offenders, (
        "route_affordance callsites missing cell_failure_context "
        "kwarg (Sprint E.2 AC-E2-3 ext + Codex Gate A MEDIUM A1 "
        "+ Builder Sub-CA #6 — every caller MUST pass the "
        "context dict, empty {} is the canonical 'no context' "
        "sentinel):\n  " + "\n  ".join(offenders)
    )


# ── HIGH-2 reachability pair — coverage bucket-4 routing ────────────


def test_coverage_high_pct_routes_to_interpolate_and_bucket_4() -> None:
    """Sprint E.2 AC-E2-3 ext + Codex round 1 HIGH-2 absorption —
    when ``coverage_pct >= 80`` the routing pipeline closes
    cleanly: producer emits ``coverage_climate_cells`` →
    :func:`route_affordance` returns ``interpolate`` →
    :func:`bucket_for` returns ``bucket=4`` with the
    ``climate-dual-scale`` diagnostic variant.

    Pre-fix, branch 11 of ``bucket_for`` was structurally
    unreachable in real-data runs (producer IDs didn't match
    the literal ``coverage_per_cell`` check + the affordance
    was hard-pinned to ``rerun_full_sources``); this pin
    locks the contract-specified path open.
    """
    from prismpy.cockpit.routing_decision import bucket_for
    from prismpy.config.schema import Platform
    from prismpy.validators.affordance_routing import route_affordance

    context = {"coverage_pct": 85.0}
    affordance = route_affordance(
        check_id="coverage_climate_cells",
        platform=Platform.PYTHIA,
        zone="BSh",
        elevation_m=300.0,
        n_candidates_in_radius=4,
        cell_failure_context=context,
    )
    assert affordance == "interpolate", (
        f"coverage_climate_cells with coverage_pct=85 should "
        f"route to 'interpolate' (≥80%% threshold + ≥1 "
        f"neighbour); got {affordance!r}"
    )

    decision = bucket_for(
        check_id="coverage_climate_cells",
        cell_failure_context=context,
        routed_affordance=affordance,
    )
    assert decision.bucket == 4, (
        f"coverage_climate_cells with affordance='interpolate' "
        f"should route to bucket 4 INTERPOLATABLE; got bucket "
        f"{decision.bucket}"
    )
    assert decision.diagnostic_variant == "climate-dual-scale", (
        f"climate-side coverage check should dispatch to "
        f"'climate-dual-scale' Variant B; got "
        f"{decision.diagnostic_variant!r}"
    )


def test_coverage_low_pct_routes_to_rerun_and_bucket_3() -> None:
    """Anti-mutation drill paired with the above — when
    ``coverage_pct < 80`` the same producer + same routing
    inputs MUST route to ``rerun_full_sources`` + bucket 3
    TRUE_EXCLUDE (substantial gap; rerun with full source set).

    Symmetric to the high-coverage case so a future routing-
    engine refactor that breaks one direction can't quietly
    pass the other.
    """
    from prismpy.cockpit.routing_decision import bucket_for
    from prismpy.config.schema import Platform
    from prismpy.validators.affordance_routing import route_affordance

    context = {"coverage_pct": 70.0}
    affordance = route_affordance(
        check_id="coverage_climate_cells",
        platform=Platform.PYTHIA,
        zone="BSh",
        elevation_m=300.0,
        n_candidates_in_radius=4,
        cell_failure_context=context,
    )
    assert affordance == "rerun_full_sources", (
        f"coverage_climate_cells with coverage_pct=70 should "
        f"route to 'rerun_full_sources' (below 80%% threshold); "
        f"got {affordance!r}"
    )

    decision = bucket_for(
        check_id="coverage_climate_cells",
        cell_failure_context=context,
        routed_affordance=affordance,
    )
    assert decision.bucket == 3, (
        f"coverage_climate_cells with affordance='rerun_full_sources' "
        f"should route to bucket 3 TRUE_EXCLUDE; got bucket "
        f"{decision.bucket}"
    )


def test_coverage_soil_high_pct_routes_to_soil_layered_variant() -> None:
    """Soil-side coverage with high coverage_pct dispatches to
    ``soil-layered`` Variant C diagnostic_variant (per Draft 6.2
    spec + team-lead disposition: 'or soil-layered if check_id
    contains soil')."""
    from prismpy.cockpit.routing_decision import bucket_for
    from prismpy.config.schema import Platform
    from prismpy.validators.affordance_routing import route_affordance

    context = {"coverage_pct": 92.0}
    affordance = route_affordance(
        check_id="coverage_soil_cells",
        platform=Platform.PYTHIA,
        zone="BSh",
        elevation_m=300.0,
        n_candidates_in_radius=4,
        cell_failure_context=context,
    )
    decision = bucket_for(
        check_id="coverage_soil_cells",
        cell_failure_context=context,
        routed_affordance=affordance,
    )
    assert affordance == "interpolate"
    assert decision.bucket == 4
    assert decision.diagnostic_variant == "soil-layered"
