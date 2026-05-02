"""Sprint D.1 AC-3 — relative-humidity clip-to-100 helper.

Pins the harmonize-stage decision function + the clip
transformation. Boundary inclusivity (100.0 is pass; 102.0 is
clip) and the tagged-dataclass return shape (per builder ADD-2)
are exercised explicitly so a regression on either edge surfaces
here rather than at the executor wiring layer.
"""
from __future__ import annotations

import pytest

from prismpy.harmonize.rh_clip import (
    RHClipProvenance,
    RhAction,
    clip_rh,
    rh_action_for,
)


# ---------------------------------------------------------------------------
# AC-3 — rh_action_for boundaries + bands
# ---------------------------------------------------------------------------


def test_action_pass_for_below_100():
    """A clearly sub-saturated rh passes through unchanged."""
    action = rh_action_for(85.0)
    assert action.action == "pass"
    assert action.clipped_rh is None


def test_action_pass_at_100_boundary_inclusive():
    """100.0 exactly is the saturation boundary — passes
    unchanged (no clip needed)."""
    action = rh_action_for(100.0)
    assert action.action == "pass"
    assert action.clipped_rh is None


def test_action_clip_in_rounding_window():
    """A value in (100, 102] is the rounding-tolerance window
    NASA POWER / AgERA5 emit; clip to exactly 100."""
    action = rh_action_for(100.7)
    assert action.action == "clip"
    assert action.clipped_rh == 100.0


def test_action_clip_at_102_boundary_inclusive():
    """102.0 exactly is on the clip side — boundary inclusive."""
    action = rh_action_for(102.0)
    assert action.action == "clip"
    assert action.clipped_rh == 100.0


def test_action_exclude_above_102():
    """A value above the rounding-tolerance window cannot be
    defended as a rounding artifact and is routed to exclude."""
    action = rh_action_for(103.0)
    assert action.action == "exclude"
    assert action.clipped_rh is None


def test_action_exclude_just_above_102():
    """The 102 threshold is strictly inclusive — 102.001 is
    exclude."""
    action = rh_action_for(102.001)
    assert action.action == "exclude"


def test_action_returns_tagged_dataclass():
    """Per builder ADD-2, the return shape is a tagged
    :class:`RhAction` so the call site can read ``clipped_rh``
    directly without re-deriving it."""
    action = rh_action_for(101.0)
    assert isinstance(action, RhAction)
    assert action.action == "clip"
    assert action.clipped_rh == 100.0


# ---------------------------------------------------------------------------
# AC-3 — clip_rh transformation + provenance
# ---------------------------------------------------------------------------


def test_clip_rh_pass_returns_value_unchanged():
    """For a pass action, the value rides through with provenance None."""
    rh, prov = clip_rh(85.5, cell_id=1, date="2024-06-15")
    assert rh == 85.5
    assert prov is None


def test_clip_rh_clip_returns_100_with_provenance():
    """For a clip action, the value is exactly 100 + a
    provenance entry captures the original."""
    rh, prov = clip_rh(101.4, cell_id=1, date="2024-06-15")
    assert rh == 100.0
    assert isinstance(prov, RHClipProvenance)
    assert prov.original_rh == 101.4
    assert prov.clipped_rh == 100.0
    assert prov.cell_id == 1
    assert prov.date == "2024-06-15"
    assert prov.category == "rh_clip"


def test_clip_rh_exclude_raises():
    """``clip_rh`` is the transformation-only path — exclude cases
    must be routed at the executor level via :func:`rh_action_for`
    before calling ``clip_rh``. Calling clip_rh on an exclude value
    is a caller bug; the helper raises."""
    with pytest.raises(ValueError, match="exclude"):
        clip_rh(150.0, cell_id=1, date="2024-06-15")


def test_clip_rh_provenance_is_pydantic():
    """Provenance entries are Pydantic models so they serialize
    cleanly into the provenance summary."""
    _, prov = clip_rh(101.0, cell_id=2, date="2024-07-04")
    dumped = prov.model_dump()
    assert dumped["category"] == "rh_clip"
    assert dumped["cell_id"] == 2
    assert dumped["original_rh"] == 101.0
    assert dumped["clipped_rh"] == 100.0


# ---------------------------------------------------------------------------
# AC-3 — anti-mutation pin
# ---------------------------------------------------------------------------


def test_clip_threshold_change_flips_boundary_behavior():
    """A value of 102.5 sits in the exclude band today. If the
    clip threshold drifts upward to 103, this test fails first."""
    action = rh_action_for(102.5)
    assert action.action == "exclude"
    # Sanity: a value at 101.5 stays in the clip band
    action_clip = rh_action_for(101.5)
    assert action_clip.action == "clip"
