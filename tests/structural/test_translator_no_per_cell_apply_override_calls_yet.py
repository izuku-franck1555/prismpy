"""Phase 1 → Phase 2 boundary marker pin.

Sprint E.3 AC-E3-9 substrate-vs-integration split per team-lead
Draft 5 mini-amendment + builder honest-flag at B5 close.

**State at Phase 1 close** (this pin's current assertion):
no translator's per-cell write site CALLS ``apply_override(...)``.
The substrate is in place — the helper module exists, the
imports are wired into each translator, the
``_PLATFORM_OVERRIDE_EXEMPTIONS`` registry is canonical, and
``test_translator_platform_coverage.py`` enforces the import-
level coverage. But the actual per-cell call-site dispatch
is Phase 2 work where the prismweb-side caller threads a real
sidecar through to the orchestrator.

**State at Phase 2 close** (this pin INVERTS):
this assertion will flip to "every climate / soil / management
writer in each translator MUST call apply_override at least
once". The Phase 2 absorption commit MUST update this pin's
direction (from "0 calls" to "≥ N calls per translator").
A failure to update this pin at Phase 2 close fires structurally
loud — the marker test catches the substrate-vs-integration
boundary slippage class.

Per ``feedback_synthetic_alpine_injection.md`` precedent: this
pin is the Phase 1 substrate-flow exercise that surfaces the
no-op-on-no-sidecar contract empirically before real-data
integration lands at Phase 2. It complements
``test_translator_platform_coverage.py`` (import-level) with
call-site-level state tracking.
"""

from __future__ import annotations

import ast
from pathlib import Path

from prismpy.config.schema import Platform
from prismpy.translators._shared.cockpit_overrides import (
    _PLATFORM_OVERRIDE_EXEMPTIONS,
)


def _prismpy_src_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent.parent / "src" / "prismpy"


_TRANSLATOR_DIR_BY_PLATFORM = {
    Platform.SARRA_PY: "sarra_py",
    Platform.CRAFT: "craft",
    Platform.PYTHIA: "pythia",
    Platform.ACEA: "acea",
}


def _count_apply_override_calls(translator_path: Path) -> int:
    """Count direct ``apply_override(...)`` invocations in the
    translator module's body (NOT in docstrings)."""
    if not translator_path.exists():
        return 0
    tree = ast.parse(translator_path.read_text())
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Direct ``apply_override(...)`` call (no attribute access).
        if isinstance(func, ast.Name) and func.id == "apply_override":
            count += 1
    return count


def test_phase_1_no_translator_calls_apply_override_yet() -> None:
    """At Phase 1 close, each non-exempt translator imports
    ``apply_override`` (per
    ``test_translator_platform_coverage.py``) but does NOT yet
    invoke it from a per-cell write site. This pin tracks the
    state until Phase 2 absorption flips the assertion direction.

    A premature call-site landing fires this pin loud + signals
    the Phase 2 absorption commit landed early without the
    inversion update. Per ``feedback_workflow_first_design.md``
    discipline: contract states the boundary; pin enforces it.
    """
    src_root = _prismpy_src_root()
    offenders: list[str] = []

    for platform in Platform:
        if platform.value in _PLATFORM_OVERRIDE_EXEMPTIONS:
            continue
        sub_dir = _TRANSLATOR_DIR_BY_PLATFORM.get(platform)
        if sub_dir is None:
            continue
        translator_path = src_root / "translators" / sub_dir / "translator.py"
        n_calls = _count_apply_override_calls(translator_path)
        if n_calls > 0:
            offenders.append(
                f"{platform.value}: {n_calls} apply_override call(s)"
            )

    assert not offenders, (
        f"Phase 1 marker pin violated — translator(s) carry "
        f"per-cell ``apply_override`` calls before the Phase 2 "
        f"absorption is supposed to land them: {offenders}. "
        f"Either (a) the call sites landed early, in which case "
        f"the Phase 2 absorption commit must INVERT this pin's "
        f"assertion direction (from \"0 calls\" to \"≥ N calls "
        f"per translator\") atomically with the call-site "
        f"insertions, OR (b) the test needs an updated marker "
        f"comment because we're past the Phase 1 / Phase 2 "
        f"boundary. Either way: an explicit absorption signal "
        f"is required."
    )


def test_phase_1_marker_pin_documents_inversion_contract() -> None:
    """This sentinel test pins the existence of the absorption
    contract in this module's docstring — a casual deletion of
    the inversion wording would otherwise let a future
    contributor flip the assertion above without updating the
    accompanying narrative."""
    pin_path = Path(__file__)
    text = pin_path.read_text()
    expected_phrases = [
        "Phase 1 → Phase 2 boundary marker pin",
        "INVERTS",
        "Phase 2 absorption commit MUST update this pin",
    ]
    missing = [p for p in expected_phrases if p not in text]
    assert not missing, (
        f"Marker pin docstring drifted from canonical absorption "
        f"contract phrasing. Missing: {sorted(missing)}. The "
        f"docstring documents the inversion direction; a future "
        f"refactor MUST keep the contract phrasing intact OR "
        f"update both the docstring + test logic together."
    )
