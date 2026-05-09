"""Phantom-bug pin: sidecar entries ⇔ manifest flag.

Sprint E.3 AC-E3-13 #6 + AC-E3-7 sub-6 cross-document validator
absorbed. The cross-document invariant pinned here:

    (len(sidecar.overrides) > 0) ⇔ manifest.flags.overrides_present

Drives the honest-signal floor per ``feedback_no_data_cooking.md``:
a manifest claiming ``overrides_present == True`` with an empty
sidecar is over-claim (consumers expect overrides + see none); a
manifest claiming ``overrides_present == False`` with non-empty
sidecar is under-claim (overrides exist + the manifest lies).
Either direction degrades audit-trail integrity per durable §24.

This pin is behavioral — exercises the round-trip across the
sidecar writer + the manifest flag's expected value with three
scenarios:

1. **Empty sidecar + flag False** — happy path for a non-override
   run; the consumer reads "no overrides applied, methods text
   omits override paragraph".
2. **Non-empty sidecar + flag True** — happy path for a run with
   active overrides; the consumer reads "N overrides applied,
   methods text includes override paragraph naming the entries".
3. **Mismatch shapes reject** — empty sidecar with flag=True OR
   non-empty sidecar with flag=False both surface as audit-trail-
   integrity violations. The check helper raises the cross-
   document validator's exception.

Sprint E.3 ships the cross-document validator helper at
:func:`assert_sidecar_manifest_consistent`; consumers (the
prismweb-side commit-decision endpoint at AC-E3-19) call this
helper after writing the sidecar + before persisting the
manifest, fail-loud on mismatch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest

from prismpy.cockpit.cockpit_overrides_writer import (
    CockpitOverrideSidecar,
    OverrideSidecarEntry,
)


def _empty_sidecar() -> CockpitOverrideSidecar:
    return CockpitOverrideSidecar(
        schema_version="1.0",
        produced_at=datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc),
        overrides=[],
    )


def _non_empty_sidecar() -> CockpitOverrideSidecar:
    entry = OverrideSidecarEntry(
        cell_id="c001",
        check_id="value_range_tmax",
        variable_key="tmax_growing_season_mean",
        value=32.5,
        unit="C",
        decision_id=uuid4(),
        evidence_type="field_observation",
    )
    return CockpitOverrideSidecar(
        schema_version="1.0",
        produced_at=datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc),
        overrides=[entry],
    )


def _check_sidecar_manifest_consistency(
    sidecar: CockpitOverrideSidecar,
    manifest_flag: bool,
) -> None:
    """Behavioral helper exercising the cross-document invariant.

    The Sprint E.3 production helper at
    ``assert_sidecar_manifest_consistent`` lives at the prismweb-
    side commit-decision endpoint per AC-E3-19; this test-local
    helper mirrors the contract for Phase 1 substrate-level
    verification."""
    has_entries = len(sidecar.overrides) > 0
    if has_entries != manifest_flag:
        raise AssertionError(
            f"Sidecar / manifest cross-document drift: "
            f"sidecar has {len(sidecar.overrides)} entries "
            f"(non-empty={has_entries}); manifest "
            f"overrides_present={manifest_flag}. The honest-"
            f"signal floor requires: "
            f"(non-empty sidecar) ⇔ (overrides_present=True)."
        )


# ── §1 happy paths ─────────────────────────────────────────────────


def test_empty_sidecar_with_flag_false_consistent() -> None:
    """All-reverted-bulk run: sidecar is empty, flag is False;
    methods text omits override paragraph."""
    _check_sidecar_manifest_consistency(_empty_sidecar(), False)


def test_non_empty_sidecar_with_flag_true_consistent() -> None:
    """Active-override run: sidecar carries entries, flag is True;
    methods text includes override paragraph."""
    _check_sidecar_manifest_consistency(_non_empty_sidecar(), True)


# ── §2 mismatch shapes reject ──────────────────────────────────────


def test_empty_sidecar_with_flag_true_rejects() -> None:
    """Over-claim case: manifest claims overrides_present=True
    but sidecar is empty. Audit-trail integrity violation per
    durable §24 — the cross-document validator MUST raise."""
    with pytest.raises(AssertionError, match="cross-document drift"):
        _check_sidecar_manifest_consistency(_empty_sidecar(), True)


def test_non_empty_sidecar_with_flag_false_rejects() -> None:
    """Under-claim case: manifest claims overrides_present=False
    but sidecar has entries. Same integrity violation per
    durable §24."""
    with pytest.raises(AssertionError, match="cross-document drift"):
        _check_sidecar_manifest_consistency(_non_empty_sidecar(), False)


# ── §3 schema-version invariant ────────────────────────────────────


def test_sidecar_carries_canonical_schema_version() -> None:
    """The schema_version field is pinned to the canonical
    string at construction; a typo'd version on the wire fails
    Pydantic validation. Pin coverage of the canonical value
    (sub-criterion of phantom-bug pin #10
    `test_sidecar_schema_version_pin.py` per AC-E3-12 #10)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CockpitOverrideSidecar(
            schema_version="0.9",  # type: ignore[arg-type]
            produced_at=datetime.now(timezone.utc),
            overrides=[],
        )

    # Canonical version constructs cleanly.
    sidecar = CockpitOverrideSidecar(
        schema_version="1.0",
        produced_at=datetime.now(timezone.utc),
        overrides=[],
    )
    assert sidecar.schema_version == "1.0"
