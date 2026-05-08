"""HARMONIZE-stage writer for ``cockpit_observed_values.json``.

Sprint E.2 AC-E2-28 + Codex Gate A HIGH 5 + CMS schema input
#302 (Hybrid A 17-key) + CMS-DOMAIN-CA-1 absorption (OC counted
as chemistry; soil count 11 → 10) + Builder Sub-CA #3 (reuse
:meth:`SoilProfile.get_weighted_average` per durable §24).

Per builder probe #301 architectural verdict (Option B
harmonize-stash) — the cockpit's IDW orchestrator (consumed at
prismweb's Phase 2 cycle) needs per-cell aggregate observed
values for climate + soil variables to feed
:func:`prismpy.harmonize.idw_interpolation.interpolate_idw`.
The producer side (:class:`UnifiedData` in-memory at HARMONIZE
stage) holds the raw values; the consumer side runs
post-completion (cockpit opens days later on a finished
package). This writer persists the per-cell aggregates to a
canonical JSON sidecar at HARMONIZE finalization so the
cockpit reader path is cross-platform agnostic + survives the
in-memory-data-gone-after-pipeline-finish lifecycle.

Schema shape (Hybrid A flat scalars + sidecar metadata per
CMS §2):

```json
{
    "schema_version": "1.0",
    "growing_season_window": {
        "planting_doy": 152,
        "harvest_doy": 304,
        "n_days_in_window": 153,
        "convention": "doy_inclusive_planting_through_harvest"
    },
    "aggregation_method": {
        "<key>": "<method-string>"
    },
    "units": {
        "<key>": "<unit>"
    },
    "soil_substrate_note": "...",
    "cells": {
        "<cell_id>": {
            "lat": ...,
            "lon": ...,
            "n_layers_in_substrate": 3,
            "soil_aggregation_substrate": "in_memory_layers" |
                "eghr_no_in_memory_layers",
            "tmax_growing_season_mean": ...,
            ... (17 aggregate keys total: 7 climate + 10 soil)
        }
    }
}
```

Per durable §24 canonical-source-or-pin: this module is the
single producer of the schema; the consumer (cockpit IDW
orchestrator) imports the canonical key list from
:data:`OBSERVED_VALUES_CLIMATE_KEYS` +
:data:`OBSERVED_VALUES_SOIL_KEYS` so adding a key here
auto-propagates to the consumer via the structural pin at
``tests/structural/test_observed_values_writer_schema_parity.py``.

Per durable §27 two-vocabulary substrate-drift: producer +
consumer agree on the 17-key vocabulary AND the
``soil_aggregation_substrate`` enum values
(``in_memory_layers`` vs ``eghr_no_in_memory_layers``) AND
the units strings.

CMS §9 follow-on math contracts:

* Depth-weighting (§9.2) — delegates to
  :meth:`SoilProfile.get_weighted_average` per durable §24
  Sub-CA #3 (does NOT reinvent depth-weighted-mean math).
* 1-layer correction (§9.1) — natural via the existing
  helper's ``effective_thickness = min(L.depth_bottom, depth_max)
  - L.depth_top`` semantics.
* DOY contract (§9.4) — ``ValueError`` on missing
  ``planting_doy`` (mirrors translator strictness at
  ``translators/pythia/translator.py:961-963``).
* PYTHIA-skip-soil (§9.6 Concern A) — cells with no
  ``SoilProfile.layers`` emit ``null`` for soil keys + set
  ``soil_aggregation_substrate: "eghr_no_in_memory_layers"``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from prismpy.models.climate import ClimateRecord
from prismpy.models.crop import CropCalendar
from prismpy.models.soil import SoilProfile
from prismpy.translators.base import UnifiedData


# ── Canonical schema vocabulary ─────────────────────────────────────


SCHEMA_VERSION: str = "1.0"


# 7 climate aggregate keys per CMS §1. All computed over the
# growing-season window (planting_doy → harvest_doy inclusive)
# from per-day :class:`ClimateRecord` values; ``rh_mean`` +
# ``wind_mean`` are ``None`` when the field is absent on the
# record (Optional in :class:`ClimateRecord`).
OBSERVED_VALUES_CLIMATE_KEYS: Tuple[str, ...] = (
    "tmax_growing_season_mean",
    "tmin_growing_season_mean",
    "precip_growing_season_total",
    "precip_growing_season_n_rainy_days",
    "srad_growing_season_mean",
    "rh_growing_season_mean",
    "wind_growing_season_mean",
)


# 10 soil aggregate keys per CMS §1 + CMS-DOMAIN-CA-1 (OC counted
# as chemistry per absorption: 11 → 10 keys).
#
# Group 1 — texture (3): rootzone-weighted-mean over [0, 1.0m].
# Group 2 — chemistry (3): top-30cm-weighted-mean over [0, 0.30m]
#   (organic_carbon counted as chemistry per CMS-DOMAIN-CA-1; not
#   a standalone OC key).
# Group 3 — hydraulic (3): rootzone-weighted-mean over [0, 1.0m].
# Group 4 — scalar (1): total profile depth in cm (NOT m).
OBSERVED_VALUES_SOIL_KEYS: Tuple[str, ...] = (
    # Texture rootzone-weighted-mean (0-100cm).
    "sand_rootzone_mean",
    "clay_rootzone_mean",
    "silt_rootzone_mean",
    # Chemistry top-30cm-weighted-mean. ``organic_carbon`` is
    # one of the chemistry keys per CMS-DOMAIN-CA-1; not a
    # standalone OC key.
    "ph_top30cm_mean",
    "organic_carbon_top30cm_mean",
    "bulk_density_top30cm_mean",
    # Hydraulic rootzone-weighted-mean (0-100cm).
    "field_capacity_rootzone_mean",
    "wilting_point_rootzone_mean",
    "saturated_wc_rootzone_mean",
    # Scalar profile depth (cm; ``SoilProfile.total_depth`` is in
    # m per soil.py:136 — converted via *100 in the writer).
    "soil_profile_depth_cm",
)


# Method-string per aggregate key. Read by consumers to render
# the methods text + provenance label for each aggregate.
AGGREGATION_METHOD: Dict[str, str] = {
    "tmax_growing_season_mean": "growing_season_mean",
    "tmin_growing_season_mean": "growing_season_mean",
    "precip_growing_season_total": "growing_season_sum",
    "precip_growing_season_n_rainy_days": "growing_season_count_gt_1mm",
    "srad_growing_season_mean": "growing_season_mean",
    "rh_growing_season_mean": "growing_season_mean",
    "wind_growing_season_mean": "growing_season_mean",
    "sand_rootzone_mean": "rootzone_weighted_mean_0_100cm",
    "clay_rootzone_mean": "rootzone_weighted_mean_0_100cm",
    "silt_rootzone_mean": "rootzone_weighted_mean_0_100cm",
    "ph_top30cm_mean": "top30cm_weighted_mean",
    "organic_carbon_top30cm_mean": "top30cm_weighted_mean",
    "bulk_density_top30cm_mean": "top30cm_weighted_mean",
    "field_capacity_rootzone_mean": "rootzone_weighted_mean_0_100cm",
    "wilting_point_rootzone_mean": "rootzone_weighted_mean_0_100cm",
    "saturated_wc_rootzone_mean": "rootzone_weighted_mean_0_100cm",
    "soil_profile_depth_cm": "scalar_total_depth",
}


# Units string per aggregate key. ``count`` for the rainy-day
# tally (dimensionless); ``cm`` for the profile depth scalar
# (converted from :attr:`SoilProfile.total_depth` which is in m).
AGGREGATION_UNITS: Dict[str, str] = {
    "tmax_growing_season_mean": "°C",
    "tmin_growing_season_mean": "°C",
    "precip_growing_season_total": "mm",
    "precip_growing_season_n_rainy_days": "count",
    "srad_growing_season_mean": "MJ/m²/day",
    "rh_growing_season_mean": "%",
    "wind_growing_season_mean": "m/s",
    "sand_rootzone_mean": "%",
    "clay_rootzone_mean": "%",
    "silt_rootzone_mean": "%",
    "ph_top30cm_mean": "pH",
    "organic_carbon_top30cm_mean": "%",
    "bulk_density_top30cm_mean": "g/cm³",
    "field_capacity_rootzone_mean": "cm³/cm³",
    "wilting_point_rootzone_mean": "cm³/cm³",
    "saturated_wc_rootzone_mean": "cm³/cm³",
    "soil_profile_depth_cm": "cm",
}


# Canonical soil-substrate sentinel values per CMS §9.6 +
# Codex MEDIUM A3 absorption. Cells whose soil profile is
# present + has layers emit ``in_memory_layers``; cells from
# the eGHR / PYTHIA path that lack ``SoilProfile.layers``
# emit ``eghr_no_in_memory_layers`` AND null soil aggregates.
SOIL_AGGREGATION_IN_MEMORY: str = "in_memory_layers"
SOIL_AGGREGATION_EGHR_SKIP: str = "eghr_no_in_memory_layers"


# Documentation note rendered into the JSON file's
# ``soil_substrate_note`` field. Surfaces the PYTHIA-skip-soil
# disclosure for human consumers (cockpit-side reader, audit
# log) so the ``null`` soil aggregates aren't silently
# misinterpreted as "missing data" when they're actually
# "substrate not in-memory at HARMONIZE stage".
SOIL_SUBSTRATE_NOTE: str = (
    "Soil aggregates compute from in-memory SoilProfile.layers "
    "at HARMONIZE stage. PYTHIA cells that flow through the "
    "eGHR substrate path do not populate layers in unified_data; "
    "their soil keys emit null with soil_aggregation_substrate "
    "= 'eghr_no_in_memory_layers'. The cockpit IDW orchestrator "
    "treats these as climate-only interpolation candidates."
)


# Growing-season convention identifier. Documented in the JSON
# file so consumers know how the window was framed (DOY-inclusive
# from planting through harvest). Sub-fortnight + multi-year
# windows are out-of-scope for Phase 1.5.
GROWING_SEASON_CONVENTION: str = (
    "doy_inclusive_planting_through_harvest"
)


# Rainy-day threshold per WMO + crop-modeling-specialist guidance:
# a day with precip >= 1.0 mm counts as "rainy" for cumulative
# rainfall + count statistics. Dispatch uses this for
# ``precip_growing_season_n_rainy_days``.
RAINY_DAY_PRECIP_THRESHOLD_MM: float = 1.0


# ── Internal: window helpers ────────────────────────────────────────


def _growing_season_window(
    crop_calendar: Optional[Dict[int, CropCalendar]],
    cell_id: int,
) -> Tuple[int, int, int]:
    """Resolve the growing-season window for ``cell_id``.

    Returns ``(planting_doy, harvest_doy, n_days_in_window)``.
    Per CMS §9.4 + dispatch directive: raises :class:`ValueError`
    when ``planting_doy`` is None or the calendar entry is
    missing — mirrors translator strictness at
    ``translators/pythia/translator.py:961-963``.

    The harvest_doy is taken via :class:`CropCalendar`'s
    ``__post_init__`` default (falls back to maturity_doy when
    not explicitly set), matching the per-cell semantic the
    translators consume.
    """
    if not crop_calendar or cell_id not in crop_calendar:
        raise ValueError(
            f"observed_values_writer: cell_id={cell_id} has no "
            f"CropCalendar entry; cannot resolve growing-season "
            f"window. Per CMS §9.4 the calendar is required for "
            f"every cell in the climate substrate."
        )
    cal = crop_calendar[cell_id]
    if cal.planting_doy is None:
        raise ValueError(
            f"observed_values_writer: cell_id={cell_id} has "
            f"planting_doy=None; the calendar must carry an "
            f"explicit planting day per CMS §9.4."
        )
    if cal.harvest_doy is None:
        raise ValueError(
            f"observed_values_writer: cell_id={cell_id} has "
            f"harvest_doy=None even after CropCalendar "
            f"__post_init__ defaults; the calendar is malformed."
        )
    return cal.planting_doy, cal.harvest_doy, cal.growing_season_days + 1


def _in_window(doy: int, planting_doy: int, harvest_doy: int) -> bool:
    """True iff ``doy`` falls inside the growing-season window
    (inclusive at both boundaries). Handles year-boundary wrap
    (planting_doy > harvest_doy when the season spans Dec 31)."""
    if harvest_doy >= planting_doy:
        return planting_doy <= doy <= harvest_doy
    # Wrap: e.g., planting=300, harvest=60 (DJF maize season).
    return doy >= planting_doy or doy <= harvest_doy


# ── Public: climate aggregates ──────────────────────────────────────


def compute_climate_aggregates(
    records: List[ClimateRecord],
    planting_doy: int,
    harvest_doy: int,
) -> Dict[str, Optional[float]]:
    """Compute the 7 climate aggregates over the growing-season window.

    Per CMS §1 + §9.4: filters ``records`` by DOY against
    ``[planting_doy, harvest_doy]`` (inclusive; wrap-aware) +
    averages the within-window values. Optional fields (``rh``,
    ``wind``) emit ``None`` when no record carries them.

    Returns a dict with all 7 :data:`OBSERVED_VALUES_CLIMATE_KEYS`
    populated (None values are valid when the underlying field
    is absent — the consumer treats None as "data not available
    on this platform's substrate").
    """
    in_window = [
        r for r in records
        if hasattr(r, "doy")
        and _in_window(r.doy, planting_doy, harvest_doy)
    ]

    def _mean_of(field_name: str) -> Optional[float]:
        vals = [
            getattr(r, field_name)
            for r in in_window
            if getattr(r, field_name, None) is not None
        ]
        if not vals:
            return None
        return sum(vals) / len(vals)

    tmax_vals = [r.tmax for r in in_window if getattr(r, "tmax", None) is not None]
    tmin_vals = [r.tmin for r in in_window if getattr(r, "tmin", None) is not None]
    precip_vals = [r.precip for r in in_window if getattr(r, "precip", None) is not None]
    srad_vals = [r.srad for r in in_window if getattr(r, "srad", None) is not None]

    return {
        "tmax_growing_season_mean": (
            sum(tmax_vals) / len(tmax_vals) if tmax_vals else None
        ),
        "tmin_growing_season_mean": (
            sum(tmin_vals) / len(tmin_vals) if tmin_vals else None
        ),
        "precip_growing_season_total": (
            sum(precip_vals) if precip_vals else None
        ),
        "precip_growing_season_n_rainy_days": (
            sum(
                1 for v in precip_vals
                if v >= RAINY_DAY_PRECIP_THRESHOLD_MM
            )
            if precip_vals else None
        ),
        "srad_growing_season_mean": (
            sum(srad_vals) / len(srad_vals) if srad_vals else None
        ),
        "rh_growing_season_mean": _mean_of("rh"),
        "wind_growing_season_mean": _mean_of("wind"),
    }


# ── Public: soil aggregates ─────────────────────────────────────────


def compute_soil_aggregates(
    profile: Optional[SoilProfile],
) -> Tuple[Dict[str, Optional[float]], str, int]:
    """Compute the 10 soil aggregates from a :class:`SoilProfile`.

    Per CMS §1 + CMS-DOMAIN-CA-1 + Sub-CA #3: every depth-weighted
    aggregate routes through :meth:`SoilProfile.get_weighted_average`
    (canonical source at :file:`prismpy/models/soil.py:198`); this
    function does NOT reinvent the math — it composes the helper
    over the 9 layered keys + reads ``profile.total_depth`` for
    the scalar.

    PYTHIA-skip-soil (§9.6 Concern A): when ``profile`` is None
    OR ``profile.layers`` is empty, all 10 keys emit ``None`` +
    the substrate sentinel returns
    :data:`SOIL_AGGREGATION_EGHR_SKIP`. The cockpit IDW
    orchestrator treats these cells as climate-only candidates.

    Returns ``(aggregates, soil_aggregation_substrate, n_layers)``.
    """
    if profile is None or not getattr(profile, "layers", []):
        return (
            {key: None for key in OBSERVED_VALUES_SOIL_KEYS},
            SOIL_AGGREGATION_EGHR_SKIP,
            0,
        )

    n_layers = profile.n_layers

    def _rootzone(prop: str) -> Optional[float]:
        """Rootzone-weighted-mean over [0, 1.0m] via the canonical
        helper. Returns None when no contributing layer carries the
        property (consumer-side renders 'no data' rather than 0)."""
        return profile.get_weighted_average(prop, max_depth=1.0)

    def _top30cm(prop: str) -> Optional[float]:
        """Top-30cm-weighted-mean over [0, 0.30m] via the canonical
        helper."""
        return profile.get_weighted_average(prop, max_depth=0.30)

    aggregates: Dict[str, Optional[float]] = {
        "sand_rootzone_mean": _rootzone("sand"),
        "clay_rootzone_mean": _rootzone("clay"),
        "silt_rootzone_mean": _rootzone("silt"),
        "ph_top30cm_mean": _top30cm("ph"),
        "organic_carbon_top30cm_mean": _top30cm("organic_carbon"),
        "bulk_density_top30cm_mean": _top30cm("bulk_density"),
        "field_capacity_rootzone_mean": _rootzone("field_capacity"),
        "wilting_point_rootzone_mean": _rootzone("wilting_point"),
        "saturated_wc_rootzone_mean": _rootzone("saturated_wc"),
        # Scalar — ``SoilProfile.total_depth`` is in m per soil.py:136
        # (auto-computed in __post_init__ as max layer depth_bottom).
        # Multiply by 100 for cm so the cockpit drawer renders
        # "85 cm profile depth" (operator-friendly) rather than
        # "0.85 m" (model-input-style).
        "soil_profile_depth_cm": (
            profile.total_depth * 100.0
            if profile.total_depth is not None
            else None
        ),
    }
    return aggregates, SOIL_AGGREGATION_IN_MEMORY, n_layers


# ── Public: writer orchestrator ─────────────────────────────────────


def write_observed_values_json(
    unified_data: UnifiedData,
    crop_calendar: Optional[Dict[int, CropCalendar]],
    output_path: Union[str, Path],
) -> Path:
    """Persist the per-cell observed-values JSON sidecar.

    Called at HARMONIZE-stage finalization in the pipeline
    executor (per Sprint G sibling-sweep, fires for both
    baseline + projection runs uniformly).

    Args:
        unified_data: HARMONIZE-stage container with per-cell
            climate + soil substrates.
        crop_calendar: Per-cell ``CropCalendar`` map. Required
            for every cell in ``unified_data.climate``;
            :class:`ValueError` is raised when an entry is
            missing per CMS §9.4.
        output_path: Path to write the JSON to. Parent directory
            must exist (the executor wires this; the writer does
            not mkdir defensively).

    Returns:
        :class:`pathlib.Path` of the written JSON (echo of
        ``output_path`` for chaining).

    Raises:
        ValueError: per :func:`_growing_season_window` when a
            cell's calendar is malformed.
    """
    output_path = Path(output_path)

    climate = unified_data.climate or {}
    soil = unified_data.soil or {}

    # Resolve a project-level growing-season window from the
    # first calendar entry. Per-cell growing-season-days vary
    # only at the maturity_doy of the calendar; planting +
    # harvest are project-wide for the Sprint E.2 substrate
    # (a future per-cell-window expansion is V2-19.5 territory).
    # The ``growing_season_window`` block in the JSON file
    # surfaces the window the writer used for ALL cells; a
    # heterogeneous calendar dict surfaces the first cell's
    # window so the metadata stays well-formed (the cockpit
    # renders the window for context, not enforcement).
    project_planting_doy: Optional[int] = None
    project_harvest_doy: Optional[int] = None
    project_n_days: Optional[int] = None
    if crop_calendar:
        first_cell = next(iter(crop_calendar))
        project_planting_doy, project_harvest_doy, project_n_days = (
            _growing_season_window(crop_calendar, first_cell)
        )

    # Per-cell payload assembly.
    cells_block: Dict[str, Dict[str, Any]] = {}
    all_cell_ids = set(climate.keys()) | set(soil.keys())
    for cell_id in sorted(all_cell_ids):
        ts = climate.get(cell_id)
        profile = soil.get(cell_id)

        # Climate aggregates — only when in-memory records exist
        # for this cell. SARRA-Py path-dict shape (no records
        # attribute) drops to all-None climate aggregates; the
        # cockpit-side IDW orchestrator currently runs only
        # against in-memory record paths (Phase 1.5 scope; SARRA
        # observed-values is a Phase 2+ extension).
        if ts is not None and hasattr(ts, "records"):
            planting_doy, harvest_doy, _n_days = (
                _growing_season_window(crop_calendar, cell_id)
            )
            climate_aggregates = compute_climate_aggregates(
                ts.records, planting_doy, harvest_doy,
            )
        else:
            climate_aggregates = {
                key: None for key in OBSERVED_VALUES_CLIMATE_KEYS
            }

        # Soil aggregates — handles PYTHIA-skip via
        # ``compute_soil_aggregates`` returning the EGHR_SKIP
        # sentinel for None-or-empty profiles.
        soil_aggregates, soil_substrate, n_layers = (
            compute_soil_aggregates(profile)
        )

        # Coordinates from the climate timeseries metadata, or
        # from the soil profile, or None when neither carries
        # them. The cockpit IDW orchestrator reads lat/lon
        # for the haversine distance calculation; ``None``
        # signals the cell can't enter the IDW candidate pool.
        lat: Optional[float] = None
        lon: Optional[float] = None
        if profile is not None:
            lat = getattr(profile, "lat", None)
            lon = getattr(profile, "lon", None)
        if (lat is None or lon is None) and ts is not None:
            ts_meta = getattr(ts, "metadata", None) or {}
            lat = lat if lat is not None else ts_meta.get("lat")
            lon = lon if lon is not None else ts_meta.get("lon")

        cell_payload: Dict[str, Any] = {
            "lat": lat,
            "lon": lon,
            "n_layers_in_substrate": n_layers,
            "soil_aggregation_substrate": soil_substrate,
        }
        cell_payload.update(climate_aggregates)
        cell_payload.update(soil_aggregates)
        cells_block[str(cell_id)] = cell_payload

    payload = {
        "schema_version": SCHEMA_VERSION,
        "growing_season_window": {
            "planting_doy": project_planting_doy,
            "harvest_doy": project_harvest_doy,
            "n_days_in_window": project_n_days,
            "convention": GROWING_SEASON_CONVENTION,
        },
        "aggregation_method": dict(AGGREGATION_METHOD),
        "units": dict(AGGREGATION_UNITS),
        "soil_substrate_note": SOIL_SUBSTRATE_NOTE,
        "cells": cells_block,
    }

    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "AGGREGATION_METHOD",
    "AGGREGATION_UNITS",
    "GROWING_SEASON_CONVENTION",
    "OBSERVED_VALUES_CLIMATE_KEYS",
    "OBSERVED_VALUES_SOIL_KEYS",
    "RAINY_DAY_PRECIP_THRESHOLD_MM",
    "SCHEMA_VERSION",
    "SOIL_AGGREGATION_EGHR_SKIP",
    "SOIL_AGGREGATION_IN_MEMORY",
    "SOIL_SUBSTRATE_NOTE",
    "compute_climate_aggregates",
    "compute_soil_aggregates",
    "write_observed_values_json",
]
