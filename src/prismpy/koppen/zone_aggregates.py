"""Synthetic zone-aggregates fixture loader.

Sprint F AC-F-3 ships a synthetic placeholder fixture at
:data:`ZONE_AGGREGATES_PATH` so the Stage 1 wizard surface can
be exercised end-to-end without blocking on the V2-19.5 Data
Bootstrapper that ratchets in real AgERA5-derived aggregates.
The fixture follows the same per-zone schema the validator
consumes via :class:`prismpy.validators.input_base.ZoneAggregate`,
so the wizard caller can read the JSON, walk the per-zone
percentiles, and construct a typed
:class:`ZoneAggregate` instance per zone in the user's region.

Schema versioning + license posture:

* The substrate version is the ``version`` field at the top
  of the JSON. The Sprint F AC-F-5 cache key composes against
  this string so a future fixture update (or ratchet to real
  data) invalidates per-project Stage 1 verdicts safely.
* The license footer is explicit about synthetic shape; do
  not derive scientific claims from these values until V2-19.5
  ships the real bound-gen output.
* Schema mirrors :class:`ZoneAggregate` plus an optional
  ``label`` for the wizard banner / cockpit drawer copy.

Provenance-aware loaders:

* :func:`load_zone_aggregates` returns the parsed top-level
  dict. The ``zones`` sub-dict keys are the canonical KG zone
  codes (e.g. ``"BSh"`` / ``"Aw"`` / ``"Cfa"`` / ``"Cwa"`` /
  ``"Af"``) per the contract's 5-zone canonical set.
* :func:`build_zone_aggregate` returns a typed
  :class:`ZoneAggregate` for one zone code. Raises
  :class:`KeyError` on unknown zone code; raises
  :class:`pydantic.ValidationError` on shape drift.

Anti-mutation drills:

* Drop a percentile field for any zone → loader raises
  ``pydantic.ValidationError`` from :class:`ZoneAggregate`'s
  required-field validators.
* Add a forbidden ECOCROP field on any zone → not relevant
  here (the JSON schema is per-zone climate, not per-crop
  envelope); the F27 walker still polices the
  ``climate_envelope`` substrate.
* Introduce a NaN value → :class:`ZoneAggregate` field
  ordering validator catches at load time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from prismpy.validators.input_base import ZoneAggregate


# Substrate path matches the package-data declaration in
# pyproject.toml so an installed wheel ships the JSON next to
# the loader. The path is fixed at import time; callers that
# need a different file (e.g., V2-19.5 ratcheted output) pass
# an explicit path to :func:`load_zone_aggregates`.
ZONE_AGGREGATES_PATH: Path = (
    Path(__file__).parent / "data" / "zone_aggregates_v1.json"
)


def load_zone_aggregates(
    path: Path | None = None,
) -> Dict[str, Any]:
    """Read + parse the synthetic zone-aggregates JSON.

    Returns the entire top-level dict so the caller can read
    the ``version`` (for the AC-F-5 cache-key composition),
    the ``license`` footer, and walk ``zones[<code>]`` per the
    user's region.

    Args:
        path: Optional override; defaults to
            :data:`ZONE_AGGREGATES_PATH`.

    Raises:
        FileNotFoundError: If the path does not exist.
        json.JSONDecodeError: If the JSON is malformed.
    """
    target = Path(path) if path is not None else ZONE_AGGREGATES_PATH
    with open(target, encoding="utf-8") as fp:
        return json.load(fp)


def build_zone_aggregate(
    zone_code: str,
    payload: Dict[str, Any] | None = None,
    *,
    min_cell_days_per_zone: int | None = None,
) -> ZoneAggregate:
    """Construct a typed :class:`ZoneAggregate` for one zone.

    Reads the JSON if ``payload`` is None; flattens the
    nested ``precip{p25,p50,p75}`` + ``thermal{p10_extreme_tmin,
    p90_extreme_tmax}`` schema to the flat
    :class:`ZoneAggregate` Pydantic shape.

    Args:
        zone_code: Canonical KG zone code (e.g. ``"BSh"``).
            Case-sensitive — the substrate JSON uses the
            classifier's canonical keys.
        payload: Optional parsed top-level dict. When None, the
            substrate JSON is loaded fresh from disk. Pass an
            in-memory payload to test alternative fixture
            shapes without touching the filesystem.
        min_cell_days_per_zone: Reserved for future use.
            Currently unused; the validator threshold lives on
            :class:`InputValidationContext`, not on
            :class:`ZoneAggregate`. Kept as a keyword-only
            extension point so a later Sprint can attach a
            per-zone threshold without a breaking signature
            change.

    Raises:
        KeyError: If ``zone_code`` is not present in the
            JSON's ``zones`` dict.
    """
    if payload is None:
        payload = load_zone_aggregates()
    zones = payload.get("zones", {})
    if zone_code not in zones:
        raise KeyError(
            f"Zone code {zone_code!r} not in zone-aggregates "
            f"fixture; available codes: {sorted(zones.keys())}"
        )
    entry = zones[zone_code]
    precip = entry["precip"]
    thermal = entry["thermal"]
    return ZoneAggregate(
        p25=precip["p25"],
        p50=precip["p50"],
        p75=precip["p75"],
        p10_extreme_tmin=thermal["p10_extreme_tmin"],
        p90_extreme_tmax=thermal["p90_extreme_tmax"],
        n_cell_days=entry["n_cell_days"],
    )
