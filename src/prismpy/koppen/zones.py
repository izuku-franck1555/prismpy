"""Canonical ``KoppenZone`` Literal — single source of truth for the
five Köppen-Geiger zone codes the cockpit's affordance routing
recognises.

Sprint E.2 AC-E2-20 + Builder Sub-CA-A (Draft 2.1 mini-amendment):

* The Literal is **manually maintained** in this module — Python
  ``typing.Literal`` requires statically-known string values, so a
  JSON-derived runtime expression cannot be used. Instead, the canonical
  source of truth remains the registry at
  ``prismpy/koppen/data/zone_aggregates_v1.json``; this module's
  Literal mirrors the registry keys.

* A **mirror pin** at
  ``tests/structural/test_koppen_zone_literal_mirrors_registry.py``
  asserts ``set(typing.get_args(KoppenZone)) == set(registry["zones"].keys())``.
  Adding a zone to the JSON registry without extending the Literal
  here (or vice versa) fires loud at CI time per durable §24
  canonical-source-or-pin discipline. The pin is the structural
  enforcement that the manual mirror is faithful.

* Module load fails loud (``ImportError``) if the registry JSON file
  is missing. A silent fallback to "no zones" would let an
  empty-Literal slip into production where every zone-typed input
  would be rejected at runtime — that's the silent-skip class
  ``feedback_no_data_cooking.md`` warns against.

* East African Highland zones (``Cwb``, ``Cwc``) are intentionally NOT
  in the Sprint E.2 scope — Sprint F shipped the West Africa
  registry (Sahel / Sudan-Savanna / Guinea / forest / highland) per
  the four personas' geography (Bamako / Dakar / ICRISAT-Niger /
  rural West Africa). Cwb / Cwc zone-aggregate substrate is V2-19.5
  Data Bootstrapper future-explore (task #131); when those zones
  land in the registry, both the Literal here AND the affordance-
  routing rules at AC-E2-3 extend together, gated by the mirror
  pin.

Consumers (per Sprint E.2 §0.2 canonical-source #7 + AC-E2-1 /
AC-E2-3 / AC-E2-5 / AC-E2-6):

* ``prismpy.models.interpolated_cell.InterpolatedCellRecord.affected_zone_code``
  validates against this Literal at construction time.
* ``prismpy.validators.affordance_routing.route_affordance``'s ``zone``
  parameter is typed ``KoppenZone``; the Highland-precip exclusion
  rule matches ``KoppenZone.Cwa`` per Draft 5.2.
* ``prismpy.standards.interpolation_caveats.caveats_for`` uses
  ``KoppenZone`` as its zone-discrimination key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal


# ── Registry path discovery ──────────────────────────────────────────


# The registry lives alongside this module under ``data/`` to keep
# the canonical-source surface tightly colocated. ``data/zone_aggregates_v1.json``
# is the bundled package data per ``pyproject.toml``'s
# ``[tool.setuptools.package-data]`` glob; importing this module from
# the wheel-installed package resolves the path the same way a
# source-tree run does.
_KOPPEN_REGISTRY_PATH: Final[Path] = (
    Path(__file__).resolve().parent / "data" / "zone_aggregates_v1.json"
)


def _load_registry_zone_codes() -> tuple[str, ...]:
    """Read the registry once at module-load time. Raises ImportError
    on missing file or malformed shape so a broken substrate fails
    LOUD rather than silently producing an empty Literal that rejects
    every zone input at runtime.
    """
    if not _KOPPEN_REGISTRY_PATH.exists():
        raise ImportError(
            f"prismpy.koppen.zones: canonical Köppen-zone registry "
            f"missing at {_KOPPEN_REGISTRY_PATH}. The registry is the "
            f"single source of truth for the KoppenZone Literal; a "
            f"missing file means the bundled-data ship is broken "
            f"(check pyproject.toml [tool.setuptools.package-data] "
            f"globs + wheel build round-trip pin)."
        )
    try:
        payload = json.loads(_KOPPEN_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImportError(
            f"prismpy.koppen.zones: cannot parse Köppen-zone registry "
            f"at {_KOPPEN_REGISTRY_PATH}: {exc!r}"
        ) from exc
    zones = payload.get("zones") if isinstance(payload, dict) else None
    if not isinstance(zones, dict):
        raise ImportError(
            f"prismpy.koppen.zones: registry at {_KOPPEN_REGISTRY_PATH} "
            f"has unexpected shape (expected top-level 'zones' dict); "
            f"got top-level type {type(payload).__name__!r}."
        )
    if not zones:
        raise ImportError(
            f"prismpy.koppen.zones: registry at {_KOPPEN_REGISTRY_PATH} "
            f"declares an empty 'zones' dict; the canonical-source "
            f"layer cannot operate on an empty zone roster."
        )
    return tuple(sorted(zones.keys()))


# Module-load side-effect: the import statement above runs the loader,
# which raises ImportError if the registry is missing or malformed.
_REGISTRY_ZONE_CODES: Final[tuple[str, ...]] = _load_registry_zone_codes()


# ── Canonical Literal ────────────────────────────────────────────────


# Manual mirror of the registry per Sub-CA-A (Draft 2.1 mini-amendment):
# the ``Literal[...]`` arguments must be statically-known string
# values for the type checker (mypy / pyright) to use them. A
# runtime ``Literal[*tuple]`` unpacking is not supported. The
# structural mirror pin at
# ``tests/structural/test_koppen_zone_literal_mirrors_registry.py``
# enforces ``set(typing.get_args(KoppenZone)) == set(_REGISTRY_ZONE_CODES)``
# so any drift between the manual list and the registry fires loud.
KoppenZone = Literal["Af", "Aw", "BSh", "Cfa", "Cwa"]


__all__ = [
    "KoppenZone",
]
