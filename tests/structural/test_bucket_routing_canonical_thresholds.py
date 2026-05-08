"""Sprint E.2 AC-E2-3 ext + Codex Gate A MEDIUM A1 — canonical
thresholds live only in :mod:`prismpy.cockpit.bucket_thresholds`.

The per-cell bucket routing introduced at AC-E2-3 ext refines
the legacy per-category :data:`prismpy.cockpit.manifest._DIMENSION_BUCKET_MAP`
(every dimension-toggle category mapped to bucket 3 unconditionally)
into per-cell-aware logic via :func:`prismpy.cockpit.routing_decision.bucket_for`.
The split thresholds (14-day gap / 80% coverage / 0.20m profile depth)
must live ONLY in :mod:`prismpy.cockpit.bucket_thresholds`; every
caller (manifest builder, cockpit display surfaces, structural tests
themselves) imports the same constant.

Per durable §24 canonical-source-or-pin: the F25-style "bare
threshold literal" hazard surfaces fastest as a numeric drift —
e.g., a refactor sets the gap cutoff to 7 in the manifest but
to 14 elsewhere; the cockpit's filter-fade math diverges from the
producer's bucket-int. This pin walks the prismpy source tree +
asserts no other module hard-codes the threshold constants on
the active code path.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Union

import pytest

from prismpy.cockpit.bucket_thresholds import (
    COVERAGE_PER_CELL_BUCKET_4_MIN_PCT,
    PROFILE_DEPTH_BUCKET_3_MIN_M,
    TEMPORAL_GAP_BUCKET_4_MAX_DAYS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "prismpy"
THRESHOLDS_FILE = SRC_ROOT / "cockpit" / "bucket_thresholds.py"


def test_canonical_thresholds_match_expected_values() -> None:
    """Lock the canonical threshold values themselves so a
    drift in the contract (Sprint S 14-day gap / Sprint F
    80% coverage / DSSAT 0.20m minimum) surfaces at CI time."""
    assert TEMPORAL_GAP_BUCKET_4_MAX_DAYS == 14
    assert COVERAGE_PER_CELL_BUCKET_4_MIN_PCT == 80.0
    assert PROFILE_DEPTH_BUCKET_3_MIN_M == 0.20


@pytest.mark.parametrize(
    "value,name",
    [
        (14, "TEMPORAL_GAP_BUCKET_4_MAX_DAYS"),
        (80.0, "COVERAGE_PER_CELL_BUCKET_4_MIN_PCT"),
        (0.20, "PROFILE_DEPTH_BUCKET_3_MIN_M"),
    ],
)
def test_threshold_literal_appears_only_in_canonical_module(
    value: Union[int, float], name: str,
) -> None:
    """Within the cockpit module hierarchy + the affordance_routing
    call site, assert the threshold value appears ONLY as a
    numeric literal inside the canonical
    ``prismpy/cockpit/bucket_thresholds.py`` module.

    Per durable §24 canonical-source-or-pin + §6.3 redesign-
    when-class-repeats: a regex literal-walk over .py files
    was the first design + hit two false-positive classes:
    (a) coincidental matches in unrelated modules (column
    counts / DSSAT SOL params), and (b) the literal embedded
    in module docstring narrative (matched even with
    triple-quote heuristics). The AST-based redesign walks
    :class:`ast.Constant` nodes of int / float type only —
    str Constants (docstrings + comments) are categorically
    excluded from the walk.

    Scoped to ``src/prismpy/cockpit/`` + the affordance_routing
    module (the routing engine that calls ``bucket_for``).
    """
    allowed_files = {
        THRESHOLDS_FILE.resolve(),
        Path(__file__).resolve(),
    }
    in_scope_files = list((SRC_ROOT / "cockpit").rglob("*.py")) + [
        SRC_ROOT / "validators" / "affordance_routing.py",
    ]
    offenders: list[str] = []
    for py_path in in_scope_files:
        if py_path.resolve() in allowed_files:
            continue
        try:
            text = py_path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, (int, float)):
                continue
            # Exclude bool (subclass of int) — True/False
            # literals don't represent threshold drift.
            if isinstance(node.value, bool):
                continue
            # Match int with int (14), float with float (80.0
            # / 0.20). Strict type-equality avoids 14.0 == 14
            # collision and similar.
            if type(node.value) is not type(value):
                continue
            if node.value != value:
                continue
            offenders.append(
                f"{py_path.relative_to(REPO_ROOT)}:{node.lineno}: "
                f"literal {node.value!r}"
            )
    assert not offenders, (
        f"Threshold literal {value!r} (canonical name {name!r}) "
        f"appears as a numeric AST Constant in cockpit-scoped "
        f".py files outside the canonical module. Per durable "
        f"§24 every cockpit consumer must import from "
        f"``prismpy.cockpit.bucket_thresholds``:\n  "
        + "\n  ".join(offenders)
    )


def test_routing_decision_imports_canonical_thresholds() -> None:
    """``routing_decision.py`` must import the threshold
    constants by name from ``bucket_thresholds.py`` — the
    expected canonical consumer pattern. Pinning the import
    surface ensures a future refactor that breaks the import
    doesn't silently fall back to inline literals."""
    routing_path = SRC_ROOT / "cockpit" / "routing_decision.py"
    text = routing_path.read_text(encoding="utf-8")
    assert "from prismpy.cockpit.bucket_thresholds import (" in text
    assert "TEMPORAL_GAP_BUCKET_4_MAX_DAYS" in text
    assert "COVERAGE_PER_CELL_BUCKET_4_MIN_PCT" in text
    assert "PROFILE_DEPTH_BUCKET_3_MIN_M" in text
