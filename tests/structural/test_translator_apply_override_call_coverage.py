"""Per-translator ``apply_override(...)`` call-coverage pin.

Sprint E.3 fixup +15 (F-BN Boundary 3) — Phase 2 absorption of the
Phase 1 marker pin originally at
``test_translator_no_per_cell_apply_override_calls_yet.py``. The
former pin asserted "every non-exempt translator imports
``apply_override`` but does NOT yet call it" — that was the
substrate-only state of Phase 1. The PYTHIA-only ship in fixup +15
inverts the assertion for PYTHIA (which is now fully wired) while
keeping the deferred-translator class (CRAFT / SARRA-Py / ACEA) at
"0 calls + deferred-warning emission" until Phase 4.6 expansion.

**Current state at fixup +15**:

* PYTHIA — fully wired. Per-day WTH writes route through
  ``apply_override`` for tmax / tmin / precip / srad; eGHR
  substrate builder routes soil values through
  ``_apply_soil_overrides_to_assignment`` (which is the per-cell
  pre-write dispatch). The pin requires PYTHIA's translator
  module body to carry ≥1 direct ``apply_override(...)`` call
  (the soil path lives in ``translators/_shared/eghr_substrate.py``,
  which the eghr-substrate companion pin asserts; this pin
  catches the translator-side import-AND-call invariant).
* CRAFT, SARRA-Py, ACEA — deferred per the fixup +15 scope guard.
  Each translator emits a runtime warning when a non-empty cockpit
  sidecar is present (verified at the translator-deferral pin
  below) but does NOT yet invoke ``apply_override(...)``. A future
  Phase 4.6 expansion lands those wires + atomically promotes the
  translator from the ``_DEFERRED_PHASE_4_6`` set below to
  ``_FULLY_WIRED``.

**Phase 4.6 expansion contract**: when CRAFT / SARRA-Py / ACEA
land their value-replacement wiring, move the platform value
from ``_DEFERRED_PHASE_4_6`` to ``_FULLY_WIRED``. The pin then
flips that platform's assertion from "0 calls + deferred-warning
emission" to "≥1 calls". A premature wire-without-set-update fires
this pin loud; a set-update-without-wire fires the other
direction. The two-state contract makes the boundary explicit and
caught structurally. The atomically promotes the platform contract
documented here keeps the migration explicit at PR review time.

Per durable §24 canonical-source-or-pin: the platform deferral set
lives once in this pin; downstream consumers (deferred-warning
logging at translator init in each module) cite this set in their
warning copy so the persona's audit trail names the same set as
the structural pin.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import FrozenSet

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


# Phase 2 (fixup +15) — PYTHIA fully wired; remaining translators
# are deferred per scope-guard for Phase 4.6 expansion. The pin
# enforces the partition without forcing a single ship cycle to
# wire all four.
_FULLY_WIRED: FrozenSet[str] = frozenset({Platform.PYTHIA.value})
_DEFERRED_PHASE_4_6: FrozenSet[str] = frozenset({
    Platform.CRAFT.value,
    Platform.SARRA_PY.value,
    Platform.ACEA.value,
})


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


def _has_deferred_warning_marker(translator_path: Path) -> bool:
    """True iff the translator module's body contains a
    deferred-warning log mentioning the canonical phase marker.

    The deferred-translator class emits a runtime warning when a
    non-empty cockpit sidecar is present. The marker phrase
    ``"Phase 4.6"`` is the canonical ship-target token per the
    fixup +15 dispatch's scope-guard wording; the
    ``cockpit_override_sidecar`` attribute reference confirms the
    warning fires on the right condition. A casual deletion of
    either fires the pin loud rather than silently letting the
    persona's overrides vanish without an audit trail.
    """
    if not translator_path.exists():
        return False
    text = translator_path.read_text()
    if "Phase 4.6" not in text:
        return False
    if "cockpit_override_sidecar" not in text:
        return False
    return True


def test_partition_is_total_and_disjoint() -> None:
    """Every non-exempt platform belongs to exactly one of the two
    sets (fully wired vs deferred). A platform missing from both
    OR appearing in both fires the pin loud."""
    all_platforms = {p.value for p in Platform}
    non_exempt = all_platforms - _PLATFORM_OVERRIDE_EXEMPTIONS
    union = _FULLY_WIRED | _DEFERRED_PHASE_4_6
    intersection = _FULLY_WIRED & _DEFERRED_PHASE_4_6
    assert union == non_exempt, (
        f"Translator partition non-total: non-exempt platforms = "
        f"{sorted(non_exempt)}; union(_FULLY_WIRED, "
        f"_DEFERRED_PHASE_4_6) = {sorted(union)}. Every translator "
        f"must be classified."
    )
    assert not intersection, (
        f"Translator partition non-disjoint: same platform appears in "
        f"BOTH _FULLY_WIRED and _DEFERRED_PHASE_4_6 — "
        f"{sorted(intersection)}. A translator is either wired or "
        f"deferred, not both."
    )


def test_fully_wired_translators_call_apply_override() -> None:
    """Every translator in ``_FULLY_WIRED`` MUST contain at least
    one direct ``apply_override(...)`` call in its translator
    module. A wire-without-call drift (e.g., a refactor that
    extracted the call to a helper but left the import) fires this
    pin loud."""
    src_root = _prismpy_src_root()
    missing: list[str] = []
    for platform_value in sorted(_FULLY_WIRED):
        sub_dir = _TRANSLATOR_DIR_BY_PLATFORM[Platform(platform_value)]
        translator_path = src_root / "translators" / sub_dir / "translator.py"
        n_calls = _count_apply_override_calls(translator_path)
        if n_calls < 1:
            missing.append(
                f"{platform_value}: 0 apply_override calls in "
                f"{translator_path.relative_to(src_root)}"
            )

    assert not missing, (
        f"Fully-wired translator(s) lost their ``apply_override`` "
        f"call site(s): {missing}. The Phase 2 wiring contract "
        f"requires every translator in ``_FULLY_WIRED`` to invoke "
        f"the canonical helper at least once per its per-cell write "
        f"path. Either restore the call OR explicitly move the "
        f"platform back into ``_DEFERRED_PHASE_4_6`` if the value-"
        f"replacement coverage has been intentionally rolled back."
    )


def test_deferred_translators_emit_deferred_warning() -> None:
    """Every translator in ``_DEFERRED_PHASE_4_6`` MUST emit a
    runtime warning when a non-empty cockpit sidecar is present —
    the marker phrase ``"Phase 4.6"`` + reference to
    ``cockpit_override_sidecar`` are required in the translator
    module body so the persona's audit log carries an honest
    signal that the override was NOT applied."""
    src_root = _prismpy_src_root()
    silent: list[str] = []
    for platform_value in sorted(_DEFERRED_PHASE_4_6):
        sub_dir = _TRANSLATOR_DIR_BY_PLATFORM[Platform(platform_value)]
        translator_path = src_root / "translators" / sub_dir / "translator.py"
        if not _has_deferred_warning_marker(translator_path):
            silent.append(
                f"{platform_value}: no deferred-Phase-4.6 warning marker "
                f"in {translator_path.relative_to(src_root)}"
            )

    assert not silent, (
        f"Deferred translator(s) silently swallow cockpit overrides "
        f"without an honest-signal warning: {silent}. Each translator "
        f"in ``_DEFERRED_PHASE_4_6`` MUST emit a runtime warning "
        f"naming the Phase 4.6 ship target when the sidecar carries "
        f"entries — silent-skip class per "
        f"``feedback_no_data_cooking.md``."
    )


def test_marker_pin_documents_phase_4_6_contract() -> None:
    """Sentinel pin — the inversion contract phrasing in this
    module's docstring is the absorption record. A casual deletion
    of the inversion wording would let a future contributor drift
    the partition without updating the narrative."""
    pin_path = Path(__file__)
    text = pin_path.read_text()
    expected_phrases = [
        "Phase 4.6 expansion contract",
        "_FULLY_WIRED",
        "_DEFERRED_PHASE_4_6",
        "atomically promotes the",
    ]
    missing = [p for p in expected_phrases if p not in text]
    assert not missing, (
        f"Marker pin docstring drifted from canonical Phase 4.6 "
        f"expansion contract phrasing. Missing: {sorted(missing)}."
    )
