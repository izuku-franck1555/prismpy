"""Climate-envelope Stage 1 wizard-time compatibility logic.

Per Sprint E.0.5 AC-Q3-A-a/b/c. Compares per-zone climate
aggregates against ECOCROP envelopes (TMIN/TMAX/RMIN/RMAX)
and emits one of four verdicts per (zone × variable):

* ``COMPATIBLE`` — full IQR within envelope (precip), or no
  extremes-kill (thermal)
* ``MARGINAL_HETEROGENEOUS`` — precip P50 in envelope but
  the IQR straddles RMIN or RMAX
* ``MARGINAL_THERMAL_SEASONAL`` — both cold-kill (P10
  extreme tmin < crop TMIN) AND heat-kill (P90 extreme tmax
  > crop TMAX); seasonal-window refinement deferred to
  Sprint F per AC-Q3-A-c
* ``INCOMPATIBLE`` — precip P50 outside envelope, OR
  cold-kill alone, OR heat-kill alone

Stage 1 scope per AC-Q3-A-d + probe-1-A: precip + tmin +
tmax only. ALTMX, pH, photoperiod, GMIN/GMAX, latitude are
out of scope (Sprint F / V3 territory). A subsequent Sprint
E.0.5 commit lands an F27 AST walker that pins this scope
discipline at module-code time.

Algorithmic discipline:

* **AC-Q3-A-a**: precip uses three-state IQR distribution
  (P25/P50/P75 across cells). Reverting to single-point zone-
  mean masks fringe heterogeneity; F-pattern equivalent at
  the bound-gen layer.
* **AC-Q3-A-b**: no buffer multipliers. Strict envelope
  comparison. The IQR distribution carries the marginal-zone
  signal natively; ±X% buffers are forbidden in this sprint
  per Stage 0.5 §13 refusal #6.
* **AC-Q3-A-c**: thermal uses extremes-aware aggregation.
  Per-cell extremes FIRST (single coldest day per cell across
  30 yrs), then zone P10/P90 across cells. Reverting to zone-
  mean-of-extremes lets a Cfa Corn Belt zone-mean tmin = 10°C
  satisfy maize TMIN = 10°C as a false-positive.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


class CompatibilityVerdict(str, Enum):
    """Stage 1 wizard-time crop-region compatibility verdict.

    Subclasses :class:`str` (Python 3.10 compat) so the
    values are JSON-serializable directly. The four states
    cover the AC-Q3-A-a/c verdict matrix.
    """
    COMPATIBLE = "compatible"
    MARGINAL_HETEROGENEOUS = "marginal_heterogeneous"
    MARGINAL_THERMAL_SEASONAL = "marginal_thermal_seasonal"
    INCOMPATIBLE = "incompatible"


# Worst-case-wins ordering for verdict aggregation across
# variables (precip + thermal). INCOMPATIBLE overrides any
# marginal; MARGINAL_THERMAL_SEASONAL is a stronger marginal
# than MARGINAL_HETEROGENEOUS because it implies both extremes
# fire (just maybe rehabilitated by seasonal window).
_VERDICT_RANK: Dict[CompatibilityVerdict, int] = {
    CompatibilityVerdict.COMPATIBLE: 0,
    CompatibilityVerdict.MARGINAL_HETEROGENEOUS: 1,
    CompatibilityVerdict.MARGINAL_THERMAL_SEASONAL: 2,
    CompatibilityVerdict.INCOMPATIBLE: 3,
}


# Numpy quantile method pinned per the substrate's determinism
# contract (research doc §Q2.X.4 + AC-Q2-B1 thread-pin set).
# 'linear' equals the WMO No. 1203 climatological-normal
# percentile convention, byte-identical across runs given
# byte-identical input and a thread-pinned BLAS backend.
_NP_QUANTILE_METHOD: str = "linear"


def compare_precip_iqr(
    p25: float, p50: float, p75: float,
    rmin: float, rmax: float,
) -> CompatibilityVerdict:
    """Three-state precip verdict per AC-Q3-A-a + AC-Q3-A-b.

    The IQR distribution carries the heterogeneity signal
    natively: a zone whose median is in-envelope but bottom
    quartile is below RMIN gets ``MARGINAL_HETEROGENEOUS``,
    not ``COMPATIBLE``. Per AC-Q3-A-b: no buffer multipliers;
    strict envelope comparison only.

    Predicates are inclusive at the boundary
    (``P50 == RMIN`` is in-envelope, ``P75 == RMAX`` is
    in-envelope) per Option α LEAN locked in Draft 1.

    Args:
        p25, p50, p75: zone P25/P50/P75 of per-cell annual
            mean precip across cells (mm/yr).
        rmin, rmax: ECOCROP RMIN/RMAX envelope (mm/yr).

    Returns:
        :class:`CompatibilityVerdict.COMPATIBLE` when full
        IQR is within envelope; ``MARGINAL_HETEROGENEOUS``
        when P50 in but IQR straddles; ``INCOMPATIBLE`` when
        P50 outside envelope.

    Raises:
        ValueError: if any input is non-finite, if percentile
            ordering breaks (P25 > P50 > P75), or if envelope
            ordering breaks (RMIN >= RMAX).
    """
    _check_finite("p25", p25)
    _check_finite("p50", p50)
    _check_finite("p75", p75)
    _check_finite("rmin", rmin)
    _check_finite("rmax", rmax)
    if not (p25 <= p50 <= p75):
        raise ValueError(
            f"IQR percentiles must satisfy P25 <= P50 <= P75; "
            f"got P25={p25}, P50={p50}, P75={p75}."
        )
    if not rmin < rmax:
        raise ValueError(
            f"Envelope must satisfy RMIN < RMAX; got "
            f"RMIN={rmin}, RMAX={rmax}."
        )
    # Median outside envelope -> incompatible. The strict <
    # makes P50 == RMIN inclusive (boundary-in-envelope per
    # Option α LEAN).
    if p50 < rmin or p50 > rmax:
        return CompatibilityVerdict.INCOMPATIBLE
    # Median in, IQR straddles edge -> marginal heterogeneous.
    if p25 < rmin or p75 > rmax:
        return CompatibilityVerdict.MARGINAL_HETEROGENEOUS
    return CompatibilityVerdict.COMPATIBLE


def compare_thermal_extremes(
    zone_p10_extreme_tmin: float,
    zone_p90_extreme_tmax: float,
    crop_tmin: float,
    crop_tmax: float,
) -> CompatibilityVerdict:
    """Extremes-aware thermal verdict per AC-Q3-A-c.

    Aggregation order is enforced by the caller (see
    :func:`compute_zone_thermal_extremes`): per-cell extremes
    FIRST (single coldest tmin / hottest tmax per cell across
    30 yrs), then zone P10 / P90 across cells. Reverting to
    zone-mean-of-extremes lets a Cfa Corn Belt zone-mean tmin
    = 10°C satisfy maize TMIN = 10°C as a false-positive.

    Predicates are inclusive at the boundary
    (``P10_extreme == TMIN`` is in-envelope) per the
    AC-Q3-A-a Option α LEAN extension to thermal.

    Args:
        zone_p10_extreme_tmin: zone P10 of per-cell minimum-
            of-daily-tmin across 30 years (°C).
        zone_p90_extreme_tmax: zone P90 of per-cell maximum-
            of-daily-tmax across 30 years (°C).
        crop_tmin, crop_tmax: ECOCROP TMIN/TMAX envelope (°C).

    Returns:
        :class:`CompatibilityVerdict.COMPATIBLE` when no
        kill; ``MARGINAL_THERMAL_SEASONAL`` when both cold-
        kill AND heat-kill (Sprint F refines via crop's
        seasonal window); ``INCOMPATIBLE`` when either cold-
        kill or heat-kill alone fires.

    Raises:
        ValueError: if any input is non-finite, or if envelope
            ordering breaks (TMIN >= TMAX).
    """
    _check_finite("zone_p10_extreme_tmin", zone_p10_extreme_tmin)
    _check_finite("zone_p90_extreme_tmax", zone_p90_extreme_tmax)
    _check_finite("crop_tmin", crop_tmin)
    _check_finite("crop_tmax", crop_tmax)
    if not crop_tmin < crop_tmax:
        raise ValueError(
            f"Envelope must satisfy TMIN < TMAX; got "
            f"TMIN={crop_tmin}, TMAX={crop_tmax}."
        )
    # Aggregate sanity: P10 of cold extremes must not exceed
    # P90 of hot extremes. An inverted aggregate indicates
    # swapped variables, unit corruption, or broken upstream
    # aggregation; without this guard, inverted inputs slip
    # through as a silent COMPATIBLE because both kill checks
    # are False (cold-kill compares cold-tail to TMIN; heat-
    # kill compares hot-tail to TMAX).
    if zone_p10_extreme_tmin > zone_p90_extreme_tmax:
        raise ValueError(
            f"compare_thermal_extremes: zone aggregates must "
            f"satisfy P10_extreme_tmin <= P90_extreme_tmax; "
            f"got P10={zone_p10_extreme_tmin}, "
            f"P90={zone_p90_extreme_tmax}. An inverted "
            f"aggregate indicates swapped variables, unit "
            f"corruption, or broken upstream aggregation; "
            f"silent COMPATIBLE on this input would mask the "
            f"upstream bug."
        )
    cold_kill = zone_p10_extreme_tmin < crop_tmin
    heat_kill = zone_p90_extreme_tmax > crop_tmax
    if cold_kill and heat_kill:
        # Both extremes fire annually but seasonal-window
        # refinement (Sprint F) may rehabilitate this
        # combination if the crop's growing window is in a
        # moderate sub-season (e.g., Sahel maize JJAS only).
        # Stage 1 surfaces marginal_thermal_seasonal; Sprint
        # F resolves to compatible / incompatible.
        return CompatibilityVerdict.MARGINAL_THERMAL_SEASONAL
    if cold_kill or heat_kill:
        return CompatibilityVerdict.INCOMPATIBLE
    return CompatibilityVerdict.COMPATIBLE


# ── Sprint F G-c-4 reason helpers ────────────────────────────────────
#
# Per Sprint F AC-F-2 + builder grounding Adj-3 Path B: the
# verdict functions above stay pure (return enum only); these
# sibling helpers compose a short data-only reason string the
# caller can wrap with crop + zone + ECOCROP URL when emitting
# the wizard banner. Reason copy is intentionally data-only —
# zone-specific editorial gloss ("typical for arid climate")
# moves to the banner template per AC-F-10. Returning ``None``
# for non-INCOMPATIBLE verdicts keeps the contract honest:
# only INCOMPATIBLE drives the Bucket 3 EXCLUDE emit, and
# MARGINAL_* verdicts already surface via Bucket 2 INFO from
# :class:`ClimateEnvelopeValidator` so a marginal reason here
# would produce a noisy double-emit.


def precip_verdict_reason(
    verdict: CompatibilityVerdict,
    p25: float, p50: float, p75: float,
    rmin: float, rmax: float,
) -> Optional[str]:
    """Return a short data-only reason string for a precip
    INCOMPATIBLE verdict, or ``None`` for any non-INCOMPATIBLE
    state.

    The reason names which IQR percentile drove the verdict
    (P50 below RMIN, or P50 above RMAX) and reports the
    crossing magnitude. Caller is responsible for wrapping with
    crop name + zone label + ECOCROP source URL per AC-F-2's
    ≤120-char banner-copy budget; this helper deliberately
    omits those fields so the substrate stays free of free-text
    formatting concerns.

    Args:
        verdict: result of :func:`compare_precip_iqr` for the
            same (p25, p50, p75, rmin, rmax) tuple.
        p25, p50, p75: zone P25/P50/P75 of per-cell annual mean
            precip (mm/yr).
        rmin, rmax: ECOCROP RMIN/RMAX envelope (mm/yr).

    Returns:
        ``None`` if verdict is COMPATIBLE / MARGINAL_*. For
        INCOMPATIBLE, a string of the form
        ``"P50 = 280mm/yr below RMIN = 1000mm/yr"`` (low side)
        or ``"P50 = 4500mm/yr above RMAX = 4000mm/yr"`` (high
        side). Inclusive boundary semantics match
        :func:`compare_precip_iqr` (P50 == RMIN is in-envelope,
        so an INCOMPATIBLE on that boundary cannot be reached).
    """
    if verdict is not CompatibilityVerdict.INCOMPATIBLE:
        return None
    # The verdict-fn marks INCOMPATIBLE only when P50 falls
    # outside the envelope; one of the two branches must fire.
    # The third "shouldn't happen" branch returns None defensively
    # rather than raising — a substrate change in the verdict
    # function should be caught by the substrate's own tests, not
    # by the reason helper crashing the wizard.
    if p50 < rmin:
        return (
            f"P50 = {p50:.0f}mm/yr below RMIN = {rmin:.0f}mm/yr"
        )
    if p50 > rmax:
        return (
            f"P50 = {p50:.0f}mm/yr above RMAX = {rmax:.0f}mm/yr"
        )
    return None


def thermal_verdict_reason(
    verdict: CompatibilityVerdict,
    zone_p10_extreme_tmin: float,
    zone_p90_extreme_tmax: float,
    crop_tmin: float,
    crop_tmax: float,
) -> Optional[str]:
    """Return a short data-only reason string for a thermal
    INCOMPATIBLE verdict, or ``None`` for any non-INCOMPATIBLE
    state.

    Names which extreme drove the verdict (cold-kill = P10
    extreme tmin below crop TMIN, or heat-kill = P90 extreme
    tmax above crop TMAX) and reports the crossing magnitude.
    Per :func:`compare_thermal_extremes` semantics, INCOMPATIBLE
    fires only on cold-kill OR heat-kill alone; the both-kills
    case routes to MARGINAL_THERMAL_SEASONAL and returns
    ``None`` here because Sprint F does not emit on marginal
    verdicts (already covered by ClimateEnvelopeValidator's
    Bucket 2 INFO emit).

    Args:
        verdict: result of :func:`compare_thermal_extremes` for
            the same input tuple.
        zone_p10_extreme_tmin: zone P10 of per-cell minimum-of-
            daily-tmin across the substrate window (°C).
        zone_p90_extreme_tmax: zone P90 of per-cell maximum-of-
            daily-tmax across the substrate window (°C).
        crop_tmin, crop_tmax: ECOCROP TMIN/TMAX envelope (°C).

    Returns:
        ``None`` if verdict is COMPATIBLE / MARGINAL_*. For
        INCOMPATIBLE, either ``"P10 extreme tmin = -5°C below
        crop TMIN = 10°C"`` (cold-kill) or ``"P90 extreme tmax
        = 48°C above crop TMAX = 40°C"`` (heat-kill).
    """
    if verdict is not CompatibilityVerdict.INCOMPATIBLE:
        return None
    cold_kill = zone_p10_extreme_tmin < crop_tmin
    heat_kill = zone_p90_extreme_tmax > crop_tmax
    # INCOMPATIBLE fires only on cold-kill OR heat-kill (not
    # both — the both-kill case routes to MARGINAL_THERMAL_*).
    if cold_kill:
        return (
            f"P10 extreme tmin = {zone_p10_extreme_tmin:.0f}°C "
            f"below crop TMIN = {crop_tmin:.0f}°C"
        )
    if heat_kill:
        return (
            f"P90 extreme tmax = {zone_p90_extreme_tmax:.0f}°C "
            f"above crop TMAX = {crop_tmax:.0f}°C"
        )
    return None


def precip_verdict_explanation(
    verdict: CompatibilityVerdict,
    p25: float, p50: float, p75: float,
    rmin: float, rmax: float,
    *,
    crop_name: str,
    zone_label: Optional[str] = None,
) -> Optional[str]:
    """Return a 2-sentence plain-language explanation for a precip
    INCOMPATIBLE verdict, or ``None`` for any non-INCOMPATIBLE
    state.

    Sibling to :func:`precip_verdict_reason`. Where the reason
    helper produces a data-bound technical string ("P50 = 400mm/yr
    below RMIN = 1000mm/yr"), this helper produces persona-readable
    copy that names the crop's typical water need + the region's
    realized precip, then explains why that gap matters in plain
    language. The wizard banner surfaces both: technical reason
    in the disclosed details, plain-language explanation visible
    by default per ux-expert verdict.

    Args:
        verdict: result of :func:`compare_precip_iqr` for the
            same (p25, p50, p75, rmin, rmax) tuple.
        p25, p50, p75: zone P25/P50/P75 of per-cell annual mean
            precip (mm/yr).
        rmin, rmax: ECOCROP RMIN/RMAX envelope (mm/yr).
        crop_name: human-readable crop label (e.g. "Rice").
            Used verbatim in the output sentence.
        zone_label: optional human-readable zone label
            (e.g. "Hot semi-arid"). When provided, the
            explanation names the zone explicitly; when None,
            falls back to "this region".

    Returns:
        ``None`` if verdict is COMPATIBLE / MARGINAL_*. For
        INCOMPATIBLE, a 2-sentence string like:

        precip-too-dry::

            "Rice typically needs at least 1000mm of annual
             rainfall to grow well. The Hot semi-arid climate
             zone in your region averages around 400mm/year —
             too dry for rice without irrigation."

        precip-too-wet::

            "Rice tolerates up to 4000mm of annual rainfall.
             The Tropical rainforest climate zone in your
             region averages around 4500mm/year, which exceeds
             the crop's tolerance and risks waterlogging."
    """
    if verdict is not CompatibilityVerdict.INCOMPATIBLE:
        return None
    label = zone_label or "this region"
    if p50 < rmin:
        return (
            f"{crop_name} typically needs at least {rmin:.0f}mm of "
            f"annual rainfall to grow well. The {label} climate "
            f"zone in your region averages around {p50:.0f}mm/year "
            f"— too dry for {crop_name.lower()} without "
            f"irrigation."
        )
    if p50 > rmax:
        return (
            f"{crop_name} tolerates up to {rmax:.0f}mm of annual "
            f"rainfall. The {label} climate zone in your region "
            f"averages around {p50:.0f}mm/year, which exceeds the "
            f"crop's tolerance and risks waterlogging."
        )
    return None


def thermal_verdict_explanation(
    verdict: CompatibilityVerdict,
    zone_p10_extreme_tmin: float,
    zone_p90_extreme_tmax: float,
    crop_tmin: float,
    crop_tmax: float,
    *,
    crop_name: str,
    zone_label: Optional[str] = None,
) -> Optional[str]:
    """Return a 2-sentence plain-language explanation for a thermal
    INCOMPATIBLE verdict, or ``None`` for any non-INCOMPATIBLE
    state.

    Sibling to :func:`thermal_verdict_reason`. The wizard banner
    pairs both — technical reason in the disclosed details
    block, plain-language explanation visible by default — so
    Aminata, Moussa, and Ibrahim see WHY the crop will struggle,
    not just the substrate's diagnostic line.

    Args:
        verdict: result of :func:`compare_thermal_extremes`.
        zone_p10_extreme_tmin: zone P10 of per-cell minimum-of-
            daily-tmin across the substrate window (°C).
        zone_p90_extreme_tmax: zone P90 of per-cell maximum-of-
            daily-tmax across the substrate window (°C).
        crop_tmin, crop_tmax: ECOCROP TMIN/TMAX envelope (°C).
        crop_name: human-readable crop label.
        zone_label: optional human-readable zone label.

    Returns:
        ``None`` if verdict is COMPATIBLE / MARGINAL_*. For
        INCOMPATIBLE, either:

        thermal-heat-kill::

            "Maize tolerates daytime highs up to 47°C, but the
             Hot desert climate zone sees peaks above 49°C
             during the hottest days of the year. Heat stress
             will reduce grain-fill significantly."

        thermal-cold-kill::

            "Rice needs minimum temperatures above 10°C to
             grow, but the Subarctic climate zone drops below
             -5°C during the coldest days of the year. Cold
             damage will kill the crop."
    """
    if verdict is not CompatibilityVerdict.INCOMPATIBLE:
        return None
    label = zone_label or "this region"
    cold_kill = zone_p10_extreme_tmin < crop_tmin
    heat_kill = zone_p90_extreme_tmax > crop_tmax
    if cold_kill:
        return (
            f"{crop_name} needs minimum temperatures above "
            f"{crop_tmin:.0f}°C to grow, but the {label} "
            f"climate zone drops below "
            f"{zone_p10_extreme_tmin:.0f}°C during the "
            f"coldest days of the year. Cold damage will kill "
            f"the crop."
        )
    if heat_kill:
        return (
            f"{crop_name} tolerates daytime highs up to "
            f"{crop_tmax:.0f}°C, but the {label} climate "
            f"zone sees peaks above "
            f"{zone_p90_extreme_tmax:.0f}°C during the "
            f"hottest days of the year. Heat stress will reduce "
            f"grain-fill significantly."
        )
    return None


def aggregate_verdicts(
    verdicts: Iterable[CompatibilityVerdict],
) -> CompatibilityVerdict:
    """Aggregate per-variable verdicts to a single overall
    verdict via worst-case-wins.

    Used to combine precip + thermal per (zone × crop) into a
    single signal for the wizard banner. Order:
    INCOMPATIBLE > MARGINAL_THERMAL_SEASONAL >
    MARGINAL_HETEROGENEOUS > COMPATIBLE.

    Raises:
        ValueError: if ``verdicts`` is empty.
    """
    materialized = list(verdicts)
    if not materialized:
        raise ValueError(
            "aggregate_verdicts requires at least one verdict."
        )
    return max(materialized, key=_VERDICT_RANK.__getitem__)


def compute_zone_precip_iqr(
    cell_annual_precips: Sequence[float],
) -> Dict[str, float]:
    """Compute zone P25/P50/P75 of per-cell annual mean precip.

    Per AC-Q3-A-a: each cell carries a single multi-year
    mean of annual precip (mm/yr); the zone aggregate is the
    IQR distribution across those per-cell values.

    Returns ``{"p25": ..., "p50": ..., "p75": ...}``.

    Raises:
        ValueError: if ``cell_annual_precips`` is empty or
            contains a non-finite value.
    """
    arr = _to_finite_float64(cell_annual_precips, "cell_annual_precips")
    return {
        "p25": float(np.quantile(arr, 0.25, method=_NP_QUANTILE_METHOD)),
        "p50": float(np.quantile(arr, 0.50, method=_NP_QUANTILE_METHOD)),
        "p75": float(np.quantile(arr, 0.75, method=_NP_QUANTILE_METHOD)),
    }


def compute_zone_thermal_extremes(
    cell_extreme_tmins: Sequence[float],
    cell_extreme_tmaxs: Sequence[float],
) -> Dict[str, float]:
    """Compute zone P10 + P90 across per-cell thermal extremes.

    Per AC-Q3-A-c aggregation order: each cell carries the
    single coldest daily tmin (across 30 yrs) and the single
    hottest daily tmax. The zone aggregate is the
    P10 of cell extreme tmins (cold-tail) and the P90 of cell
    extreme tmaxs (hot-tail).

    Returns ``{"p10_extreme_tmin": ..., "p90_extreme_tmax": ...}``.

    Raises:
        ValueError: if either sequence is empty, lengths
            differ, or any value is non-finite.
    """
    if len(cell_extreme_tmins) != len(cell_extreme_tmaxs):
        raise ValueError(
            f"compute_zone_thermal_extremes: tmin/tmax sequence "
            f"lengths must match; got "
            f"{len(cell_extreme_tmins)} vs {len(cell_extreme_tmaxs)}."
        )
    tmin_arr = _to_finite_float64(
        cell_extreme_tmins, "cell_extreme_tmins",
    )
    tmax_arr = _to_finite_float64(
        cell_extreme_tmaxs, "cell_extreme_tmaxs",
    )
    # Per-cell ordering: each cell's extreme tmin must be at
    # most its extreme tmax. An inverted pair indicates
    # swapped variables at the caller; without this guard
    # the aggregate (P10 of "tmin", P90 of "tmax") could
    # silently pass into compare_thermal_extremes and
    # produce a meaningless verdict.
    inverted_mask = tmin_arr > tmax_arr
    if bool(np.any(inverted_mask)):
        bad_indices = [int(i) for i in np.where(inverted_mask)[0][:5]]
        raise ValueError(
            f"compute_zone_thermal_extremes: per-cell extremes "
            f"must satisfy tmin <= tmax; first inverted cells "
            f"(up to 5): {bad_indices}. An inverted pair "
            f"indicates swapped variables at the caller."
        )
    return {
        "p10_extreme_tmin": float(
            np.quantile(tmin_arr, 0.10, method=_NP_QUANTILE_METHOD),
        ),
        "p90_extreme_tmax": float(
            np.quantile(tmax_arr, 0.90, method=_NP_QUANTILE_METHOD),
        ),
    }


# --- Internal helpers ---


def _check_finite(name: str, value: float) -> None:
    """Reject non-finite numeric inputs fail-loud."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"climate_envelope: {name}={value!r} must be a number."
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            f"climate_envelope: {name}={value!r} must be finite "
            f"(NaN / inf are rejected)."
        )


def _to_finite_float64(
    seq: Sequence[float], name: str,
) -> "np.ndarray":
    """Materialize a sequence as a finite float64 numpy array.

    Empty sequences and non-finite values are rejected fail-
    loud per the substrate's honest-signal contract.
    """
    if len(seq) == 0:
        raise ValueError(
            f"climate_envelope: {name} must contain at least one "
            f"value; got an empty sequence."
        )
    arr = np.asarray(seq, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"climate_envelope: {name} contains non-finite values "
            f"(NaN / inf)."
        )
    return arr


# --- ClimateEnvelopeValidator (commit 8 — InputValidator subclass) ---


from prismpy.validators.base import ValidationIssue
from prismpy.validators.input_base import (
    InputValidationContext,
    InputValidationResult,
    InputValidator,
)
from prismpy.warnings.categories import WarningCategory


class ClimateEnvelopeValidator(InputValidator):
    """Wizard-time climate-envelope check per Sprint E.0.5.

    For each Köppen-Geiger zone in the input state, the
    validator (a) flags the zone INSUFFICIENTLY_SAMPLED if
    ``n_cell_days`` < ``min_cell_days_per_zone`` and (b) for
    sufficient zones, runs the AC-Q3-A-a/b/c precip + thermal
    verdict logic and emits a CLIMATE_ENVELOPE_TAIL issue
    when the verdict is marginal. INCOMPATIBLE verdicts are
    NOT emitted by this validator; per the team-lead Decision
    2 framing, the crop-incompatibility surface is
    :class:`CropPhysiologicalValidator` (skeleton in E.0.5;
    populated in Sprint F).

    EMITS is the canonical pin for the F25-shape walker:
    every WarningCategory emitted by :meth:`validate` MUST
    appear in this frozenset. INCOMPATIBLE verdicts produce
    no issue, so CROP_REGION_MISMATCH stays out of the
    EMITS frozenset for this validator.
    """

    EMITS = frozenset({
        WarningCategory.CLIMATE_ENVELOPE_TAIL,
        WarningCategory.INSUFFICIENTLY_SAMPLED,
    })

    def validate(
        self, input_state: InputValidationContext,
    ) -> InputValidationResult:
        """Run per-zone envelope checks on the input state.

        The crop envelope and zone aggregates are typed
        Pydantic models (``CropEnvelope`` /
        ``ZoneAggregate``) by the time they reach this method,
        so missing-key / typo'd-field / mutated-mid-check
        defects have already failed at context construction.
        """
        crop_envelope = input_state.crop_envelope
        crop_tmin = crop_envelope.TMIN
        crop_tmax = crop_envelope.TMAX
        crop_rmin = crop_envelope.RMIN
        crop_rmax = crop_envelope.RMAX

        issues: list[ValidationIssue] = []
        per_zone_verdicts: dict[str, dict[str, str]] = {}

        for zone, aggs in input_state.zone_aggregates.items():
            if aggs.n_cell_days < input_state.min_cell_days_per_zone:
                issues.append(ValidationIssue(
                    severity="warning",
                    category=WarningCategory.INSUFFICIENTLY_SAMPLED.value,
                    message=(
                        f"Zone {zone!r} has insufficient sample "
                        f"({aggs.n_cell_days:,} cell-days < "
                        f"{input_state.min_cell_days_per_zone:,} "
                        f"threshold); skipping envelope verdict."
                    ),
                    details={
                        "zone": zone,
                        "n_cell_days": aggs.n_cell_days,
                        "threshold": input_state.min_cell_days_per_zone,
                    },
                ))
                per_zone_verdicts[zone] = {
                    "precip": "skipped_insufficient_sample",
                    "thermal": "skipped_insufficient_sample",
                }
                continue

            precip_verdict = compare_precip_iqr(
                p25=aggs.p25,
                p50=aggs.p50,
                p75=aggs.p75,
                rmin=crop_rmin,
                rmax=crop_rmax,
            )
            thermal_verdict = compare_thermal_extremes(
                zone_p10_extreme_tmin=aggs.p10_extreme_tmin,
                zone_p90_extreme_tmax=aggs.p90_extreme_tmax,
                crop_tmin=crop_tmin,
                crop_tmax=crop_tmax,
            )
            per_zone_verdicts[zone] = {
                "precip": precip_verdict.value,
                "thermal": thermal_verdict.value,
            }

            for variable, verdict in (
                ("precip", precip_verdict),
                ("thermal", thermal_verdict),
            ):
                if verdict in (
                    CompatibilityVerdict.MARGINAL_HETEROGENEOUS,
                    CompatibilityVerdict.MARGINAL_THERMAL_SEASONAL,
                ):
                    issues.append(ValidationIssue(
                        severity="warning",
                        category=WarningCategory.CLIMATE_ENVELOPE_TAIL.value,
                        message=(
                            f"Zone {zone!r} {variable} verdict is "
                            f"{verdict.value!r} for crop "
                            f"{input_state.crop_name!r}; the "
                            f"climate is at the tail of the crop's "
                            f"envelope (Bucket 2 informational)."
                        ),
                        details={
                            "zone": zone,
                            "variable": variable,
                            "verdict": verdict.value,
                            "crop": input_state.crop_name,
                        },
                    ))
                # COMPATIBLE -> no issue.
                # INCOMPATIBLE -> no issue from this validator;
                # CropPhysiologicalValidator (Sprint F populates)
                # owns the CROP_REGION_MISMATCH emission.

        return InputValidationResult(
            valid=all(i.severity != "error" for i in issues),
            issues=issues,
            metadata={"per_zone_verdicts": per_zone_verdicts},
        )
