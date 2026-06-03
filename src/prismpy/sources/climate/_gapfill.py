"""Short-gap recovery for NASA POWER daily climate.

Real NASA POWER solar radiation (``ALLSKY_SFC_SW_DWN``, satellite-derived)
carries sporadic single-day ``-999`` inside an otherwise-published window. A
strict "zero interior gaps" check would reject the whole cell for one such day,
so a smooth variable with a short interior gap is recovered by linear
interpolation from its bracketing real days before coverage is assessed.

Rain is never interpolated — a missing rain day is not the mean of its
neighbours, and rainfall is the most discontinuous, highest-leverage model
input. A gap longer than ``MAX_INTERP_GAP_DAYS``, or one without a real day on
both sides, is left missing for the coverage check to reject honestly (no
fabrication). Every fill is recorded so a filled cell is never silently treated
as clean — the record is surfaced as provenance.
"""
from __future__ import annotations

from typing import Dict, List

# Longest run of consecutive missing days a smooth variable is interpolated
# across; a longer run is left for the coverage check to reject. Single source
# of this bound — callers and tests read it here, never a bare literal.
MAX_INTERP_GAP_DAYS = 5

# Smooth variables recovered by interpolation. Precipitation is deliberately
# excluded (uninterpolable); the required vars not listed here stay strict.
_INTERPOLATED_FIELDS = ("srad", "tmin", "tmax")


def normalize_missing(records: List) -> None:
    """Map physically-impossible values to missing, in place, so they cannot
    reach the model: negative solar radiation, negative precipitation, and an
    inverted temperature pair (``tmax < tmin``, both nulled). A real zero solar
    value (a high-latitude polar-night day) is legitimate and kept."""
    for record in records:
        if record.srad is not None and record.srad < 0:
            record.srad = None
        if record.precip is not None and record.precip < 0:
            record.precip = None
        if (record.tmax is not None and record.tmin is not None
                and record.tmax < record.tmin):
            record.tmax = None
            record.tmin = None


def fill_short_gaps(records: List) -> Dict[str, Dict[str, object]]:
    """Linearly interpolate short interior gaps of the smooth variables in
    ``records`` (assumed date-sorted), in place.

    Returns a provenance map ``{field: {"n_filled_days": int, "method":
    "linear-interp"}}`` for the fields that were filled — empty when nothing
    was filled, so a clean cell carries an empty record.
    """
    provenance: Dict[str, Dict[str, object]] = {}
    for field in _INTERPOLATED_FIELDS:
        n_filled = _fill_field(records, field)
        if n_filled:
            provenance[field] = {
                "n_filled_days": n_filled,
                "method": "linear-interp",
            }
    return provenance


def recompute_means(records: List) -> None:
    """Recompute the derived mean temperature, in place, for any day whose
    ``tmin`` and ``tmax`` are present but whose mean is missing, so a day with
    a recovered temperature keeps a mean consistent with its endpoints."""
    for record in records:
        if (record.tmean is None and record.tmin is not None
                and record.tmax is not None):
            record.tmean = (record.tmin + record.tmax) / 2


def _fill_field(records: List, field: str) -> int:
    """Fill short, both-sides-bracketed gaps of one field; return the count of
    days filled. A run longer than the cap, or missing a real day on either
    side (a boundary gap), is left untouched."""
    filled = 0
    n = len(records)
    i = 0
    while i < n:
        if getattr(records[i], field) is not None:
            i += 1
            continue
        # A missing run spans the indices [i, j), with j exclusive.
        j = i
        while j < n and getattr(records[j], field) is None:
            j += 1
        run_len = j - i
        left, right = i - 1, j  # bracketing indices (real days)
        if left >= 0 and right < n and run_len <= MAX_INTERP_GAP_DAYS:
            _interpolate_run(records, field, left, right)
            filled += run_len
        i = j
    return filled


def _interpolate_run(records: List, field: str, left: int, right: int) -> None:
    """Linearly interpolate ``field`` for the days strictly between ``left``
    and ``right`` (both real), weighted by calendar-date distance so a gap that
    straddles a cross-year stitch is filled correctly."""
    d_left, d_right = records[left].date, records[right].date
    v_left, v_right = getattr(records[left], field), getattr(records[right], field)
    span_days = (d_right - d_left).days
    for k in range(left + 1, right):
        frac = (records[k].date - d_left).days / span_days
        setattr(records[k], field, v_left + (v_right - v_left) * frac)
