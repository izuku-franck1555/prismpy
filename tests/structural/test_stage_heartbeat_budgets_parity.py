"""F-DE RC1 Pin F-DE-RC1-1 — STAGE_HEARTBEAT_BUDGETS canonical pin.

Asserts the per-stage heartbeat budget dict at
``prismpy/pipeline/stage_budgets.py`` covers every
:class:`prismpy.pipeline.executor.PipelineStage` enum value and that
budget values are bounded + lowercase + identifier-shaped — the
prismweb watchdog reaper relies on each of these invariants.

Also exercises the wheel-install path per F-DE RC1 §K.8: the package
re-export at ``prismpy.pipeline`` is imported as a smoke check so a
future ``setuptools`` discovery regression that drops the new module
from the wheel trips this pin loudly.

Per F-DE RC1 2026-05-13 cycle-3 LOCKED + builder grounding pass.
"""
from __future__ import annotations

import typing

# K.8 inline wheel-import smoke — exercises the package surface that
# the wheel build must publish. Module-level import (not function-
# local) so a missing wheel entry fails at collection rather than at
# the assertion.
from prismpy.pipeline import STAGE_HEARTBEAT_BUDGETS as _PKG_EXPORT
from prismpy.pipeline import stage_budgets as _SUBMODULE
from prismpy.pipeline.executor import PipelineStage
from prismpy.pipeline.stage_budgets import STAGE_HEARTBEAT_BUDGETS


def test_canonical_key_set_matches_pipelinestage_enum() -> None:
    """Every key in the canonical budget dict is exactly one of the
    ``PipelineStage`` enum values. A future stage rename / addition
    in the executor must be paired with an entry here OR this pin
    fires loud.
    """
    canonical_values = {member.value for member in PipelineStage}
    budget_keys = set(STAGE_HEARTBEAT_BUDGETS.keys())

    assert budget_keys == canonical_values, (
        f"STAGE_HEARTBEAT_BUDGETS keys {sorted(budget_keys)} do not "
        f"match PipelineStage enum values {sorted(canonical_values)}. "
        f"Add the missing stage's budget OR remove the orphan entry."
    )


def test_budget_values_are_reasonable_bounds() -> None:
    """Each budget is a positive int between 60s (minimum watchdog
    poll) and 1800s (30 min — well past any healthy substage). A
    value outside that band is almost certainly a typo.
    """
    for stage, budget in STAGE_HEARTBEAT_BUDGETS.items():
        assert isinstance(budget, int), (
            f"Budget for {stage!r} is {type(budget).__name__}; "
            f"expected int."
        )
        assert 60 <= budget <= 1800, (
            f"Budget for {stage!r} = {budget}s is outside the "
            f"[60, 1800] safety band. If a stage genuinely needs "
            f"a longer or shorter budget, document the rationale in "
            f"``stage_budgets.py`` and update this bound."
        )


def test_canonical_keys_are_lowercase_identifiers() -> None:
    """Keys must be lowercase Python identifiers so the watchdog's
    reap-reason format ``no_progress_for_<stage>_<budget>s`` is a
    clean regex match (no spaces / no hyphens / no UPPERCASE).
    """
    for stage in STAGE_HEARTBEAT_BUDGETS:
        assert stage.islower(), (
            f"Stage key {stage!r} is not lowercase."
        )
        assert stage.isidentifier(), (
            f"Stage key {stage!r} is not a valid Python identifier."
        )


def test_documented_profile_values_match() -> None:
    """The cycle-3 contract pins the exact budget profile per stage.
    Drift either side (contract edit without code change, or vice
    versa) fires this pin so the profile stays auditable.
    """
    expected = {
        "retrieve":    600,
        "harmonize":   240,
        "translate":   600,
        "remediation": 300,
        "validate":    180,
        "package":     120,
    }
    assert STAGE_HEARTBEAT_BUDGETS == expected, (
        f"STAGE_HEARTBEAT_BUDGETS drifted from documented profile.\n"
        f"  expected: {expected}\n"
        f"  actual:   {dict(STAGE_HEARTBEAT_BUDGETS)}\n"
        f"If the profile was intentionally retuned, update both the "
        f"contract (``F-DE-RC1-CONTRACT.md`` §A.4) and this assertion."
    )


def test_package_export_matches_submodule() -> None:
    """K.8 wheel-import smoke: the package-level re-export at
    ``prismpy.pipeline`` resolves to the same dict object as the
    submodule's. A missing wheel entry (e.g., setuptools
    ``find_packages()`` regression) would surface as an ImportError
    at collection; this assertion catches a wheel that shipped the
    submodule but forgot the re-export.
    """
    assert _PKG_EXPORT is STAGE_HEARTBEAT_BUDGETS, (
        "``from prismpy.pipeline import STAGE_HEARTBEAT_BUDGETS`` "
        "did not return the canonical dict. Check "
        "``prismpy/pipeline/__init__.py`` re-export."
    )
    assert _SUBMODULE.STAGE_HEARTBEAT_BUDGETS is STAGE_HEARTBEAT_BUDGETS
