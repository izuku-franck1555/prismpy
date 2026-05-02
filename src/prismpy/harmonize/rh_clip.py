"""Relative-humidity clip-to-100 at the harmonize stage.

NASA POWER and AgERA5 reanalysis pipelines occasionally emit
daily relative-humidity values slightly above 100% near a
saturated-air situation due to rounding in the upstream
post-processing. The harmonize stage clips values within a
defensible rounding tolerance ``(100, 102]`` to exactly 100 with a
provenance entry; values above 102 fall outside any defensible
window and route the cell's climate axis to
``unavailable`` with cause ``climate_rh_invalid``.

The thresholds live in :mod:`prismpy.harmonize.constants`
alongside the texture-renormalization thresholds so a single
edit moves both the harmonize-stage transformation and the
validator's accept-band guards in lock-step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from pydantic import BaseModel

from prismpy.harmonize.constants import (
    RH_CLIP_THRESHOLD_PCT,
    RH_PHYSICAL_MAX_PCT,
)


class RHClipProvenance(BaseModel):
    """Per-record provenance entry written each time a daily rh
    value is clipped from ``(100, 102]`` to exactly 100. The
    original value is preserved so a downstream auditor can replay
    the transformation.
    """

    category: str = "rh_clip"
    cell_id: int
    date: str
    original_rh: float
    clipped_rh: float = 100.0


@dataclass
class RhAction:
    """Tagged outcome from :func:`rh_action_for`.

    ``action`` is one of ``'pass'``, ``'clip'``, or ``'exclude'``;
    ``clipped_rh`` carries the post-clip value when the action is
    ``'clip'`` and ``None`` otherwise. The tagged shape removes a
    re-derivation step at the call site — the caller writes
    ``rh = action.clipped_rh`` directly when the action is clip.
    """

    action: str
    clipped_rh: Optional[float] = None


def rh_action_for(rh: float) -> RhAction:
    """Decide what to do with a daily rh value.

    Args:
        rh: The raw rh value as emitted by the source. Already
            assumed to be in percent (NASA POWER / AgERA5
            convention); a fractional value (0-1) is out of scope
            and would be caught by the validator.

    Returns:
        :class:`RhAction` tagging the outcome. ``rh <= 100`` returns
        ``action='pass'`` with ``clipped_rh=None``;
        ``rh in (100, 102]`` returns ``action='clip'`` with
        ``clipped_rh=100.0``; ``rh > 102`` returns
        ``action='exclude'`` with ``clipped_rh=None``.

    Boundary inclusivity: 100.0 is ``pass``; 102.0 is ``clip``.
    """
    if rh <= RH_PHYSICAL_MAX_PCT:
        return RhAction(action="pass", clipped_rh=None)
    if rh <= RH_CLIP_THRESHOLD_PCT:
        return RhAction(action="clip", clipped_rh=RH_PHYSICAL_MAX_PCT)
    return RhAction(action="exclude", clipped_rh=None)


def clip_rh(
    rh: float, cell_id: int, date: str,
) -> Tuple[float, Optional[RHClipProvenance]]:
    """Apply the rh decision and return both the post-clip value
    and an optional provenance entry. Returns ``(rh, None)``
    unchanged for pass cases; ``(100.0, RHClipProvenance(...))``
    for clip cases. The exclude case is handled at the executor
    level — the caller checks the action explicitly via
    :func:`rh_action_for` before invoking ``clip_rh``.

    Pre-condition: :func:`rh_action_for` returned ``'pass'`` or
    ``'clip'`` for this rh value.
    """
    action = rh_action_for(rh)
    if action.action == "pass":
        return rh, None
    if action.action == "clip":
        provenance = RHClipProvenance(
            cell_id=cell_id,
            date=date,
            original_rh=rh,
        )
        return RH_PHYSICAL_MAX_PCT, provenance
    raise ValueError(
        f"clip_rh called for rh={rh!r}; action='exclude'. The "
        f"caller must route exclude cases to the cell-unavailable "
        f"path before invoking clip_rh."
    )
