"""Sprint S Gate-B-FIX — provenance.eghr_substrate_decision tracker contract.

Pins the dedicated top-level field that surfaces the PYTHIA eGHR
substrate dispatch decision (``"canonical"`` vs ``"legacy_bundled"``)
in ``provenance.json``. Per durable §24 canonical-source-or-pin: this
field IS the source of truth that downstream consumers (the AC-8
reproduction snippet, the evaluator's Gate B verifier, Dr. Kofi's
grep-the-package workflow) read; consumers MUST NOT re-derive the
dispatch decision from secondary signals (presence of CM.SOL,
absence of fallback warnings, raster-vs-database row counts).

The b5fb6538 evaluator real-data run produced a false-PASS exactly
because the dispatch decision was inferred from secondary signals
under stale-runserver conditions. This test net pins the canonical
field's behaviour at the schema layer so the failure mode cannot
silently recur.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from prismpy.provenance.tracker import ProvenanceTracker


@pytest.fixture
def tracker(tmp_path: Path) -> Iterator[ProvenanceTracker]:
    """Fresh ProvenanceTracker per test (per-pipeline-run isolation)."""
    yield ProvenanceTracker(
        enabled=True,
        output_dir=tmp_path,
        project_name="sprint_s_eghr_substrate_decision",
    )


def test_eghr_substrate_decision_defaults_to_none(tracker: ProvenanceTracker) -> None:
    """A fresh tracker has ``eghr_substrate_decision is None``.

    Legacy compat — packages built by translators without Sprint S
    canonical-substrate dispatch leave the field as ``None``. The
    AC-8 snippet treats absence as a FAIL (the package was built
    by stale code), but the tracker itself must initialise the
    field cleanly so the dataclass construction doesn't raise.
    """
    assert tracker.record.eghr_substrate_decision is None
    assert tracker.record.eghr_substrate_reason is None


def test_set_eghr_substrate_decision_canonical_ok(tracker: ProvenanceTracker) -> None:
    """``set_eghr_substrate_decision('canonical', 'ok')`` writes both fields."""
    tracker.set_eghr_substrate_decision(decision="canonical", reason="ok")
    assert tracker.record.eghr_substrate_decision == "canonical"
    assert tracker.record.eghr_substrate_reason == "ok"


@pytest.mark.parametrize(
    "reason",
    ["disabled_via_flag", "disabled_via_env", "inputs_unavailable"],
)
def test_set_eghr_substrate_decision_legacy_each_reason(
    tracker: ProvenanceTracker,
    reason: str,
) -> None:
    """Every accepted ``reason`` code stores correctly on the legacy branch."""
    tracker.set_eghr_substrate_decision(decision="legacy_bundled", reason=reason)
    assert tracker.record.eghr_substrate_decision == "legacy_bundled"
    assert tracker.record.eghr_substrate_reason == reason


def test_set_eghr_substrate_decision_rejects_unknown_decision(
    tracker: ProvenanceTracker,
) -> None:
    """Unknown decision strings raise ``ValueError`` at the schema boundary.

    Per durable §6.4 schema-layer discipline: validate the enum at
    the boundary so a bad call site fails loud rather than persisting
    a junk value that the AC-8 snippet later struggles to interpret.
    """
    with pytest.raises(ValueError, match="eghr_substrate_decision must be one of"):
        tracker.set_eghr_substrate_decision(
            decision="canonical_with_typo", reason="ok",
        )
    with pytest.raises(ValueError, match="eghr_substrate_decision must be one of"):
        tracker.set_eghr_substrate_decision(decision="", reason="ok")


def test_set_eghr_substrate_decision_rejects_unknown_reason(
    tracker: ProvenanceTracker,
) -> None:
    """Unknown reason codes raise ``ValueError`` at the schema boundary."""
    with pytest.raises(ValueError, match="eghr_substrate_reason must be one of"):
        tracker.set_eghr_substrate_decision(
            decision="legacy_bundled", reason="unknown_cause",
        )


def test_set_eghr_substrate_decision_idempotent_overwrite(
    tracker: ProvenanceTracker,
) -> None:
    """Sequential calls overwrite cleanly (last write wins).

    The dispatcher may be re-entered (e.g., translator re-instantiated
    in a long-running runserver process across multiple pipeline runs).
    The tracker is per-run, but an overwrite must not raise — the call
    site is idempotent.
    """
    tracker.set_eghr_substrate_decision(decision="legacy_bundled", reason="inputs_unavailable")
    tracker.set_eghr_substrate_decision(decision="canonical", reason="ok")
    assert tracker.record.eghr_substrate_decision == "canonical"
    assert tracker.record.eghr_substrate_reason == "ok"


def test_eghr_substrate_decision_surfaces_in_to_dict(tracker: ProvenanceTracker) -> None:
    """The dedicated fields surface at the top level of ``to_dict()``.

    The AC-8 reproduction snippet (Sprint S durable §25 user-snippet
    canonical Gate B) reads ``provenance.json["eghr_substrate_decision"]``
    via ``json.load(...)``. This test pins the serialization path so
    a refactor of the to_dict shape cannot silently break the snippet.
    """
    tracker.set_eghr_substrate_decision(decision="canonical", reason="ok")
    serialized = tracker.record.to_dict()
    assert serialized.get("eghr_substrate_decision") == "canonical"
    assert serialized.get("eghr_substrate_reason") == "ok"


def test_eghr_substrate_decision_omitted_when_unset(tracker: ProvenanceTracker) -> None:
    """``to_dict()`` omits the keys when the decision is unset (legacy compat).

    Pre-Sprint-S consumers reading ``provenance.json.get("eghr_substrate_decision")``
    must see ``None`` (the absence-of-key value), not a literal ``None``
    string or empty payload. This test pins the omit-when-None
    serialization shape that closes the legacy-compat half of durable §24.
    """
    serialized = tracker.record.to_dict()
    assert "eghr_substrate_decision" not in serialized
    assert "eghr_substrate_reason" not in serialized


def test_set_eghr_substrate_decision_no_op_when_disabled(tmp_path: Path) -> None:
    """A disabled tracker silently ignores the setter call.

    The tracker honours ``enabled=False`` for every other write
    method; the eGHR substrate setter must follow the same contract
    so production runs that disable provenance for performance
    reasons don't crash on the new call site.
    """
    disabled = ProvenanceTracker(
        enabled=False, output_dir=tmp_path, project_name="disabled",
    )
    disabled.set_eghr_substrate_decision(decision="canonical", reason="ok")
    assert disabled.record.eghr_substrate_decision is None
    assert disabled.record.eghr_substrate_reason is None
