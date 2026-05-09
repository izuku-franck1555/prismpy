"""PACKAGE-end writer for ``cell_roster_snapshot.json``.

Sprint E.3 AC-E3-10 + post-Draft 2 codex CA-4 timing absorption.
The cockpit's State C-newcells variant trigger at AC-E3-19 fires
when:

* The persona has a prior ``OverrideRecord`` with
  ``applied_to_scope == "zone"``.
* Today's run added new cells to the same zone.
* The new cells route to a bucket overlapping the prior override.

Detecting the new cells requires comparing today's cell roster
against the prior run's roster. This module persists today's
roster to a canonical JSON sidecar at PACKAGE-end so the cockpit's
cross-run comparator at
:func:`prismweb.core.services.cockpit_decisions.compare_cell_rosters`
(Phase 2) can read prior + current snapshots and return the
set-difference of new cells per zone.

**Stage placement** (codex CA-4 absorbed): the writer fires at
**PACKAGE-end / post-validation** rather than HARMONIZE-end
because ``check_ids_failed`` is assembled from the validation
report during PACKAGE / cell-summary construction
(``prismpy/pipeline/executor.py:3425-3431`` +
``_build_cell_summary`` at ``:3587-3633``); HARMONIZE-end has no
failed-checks data yet. This mirrors the
``cockpit_observed_values.json`` writer pattern's "fire when the
data is fully assembled" rule.

Schema shape::

    {
        "schema_version": "1.0",
        "run_id": "<UUID>",
        "produced_at": "<ISO-8601>",
        "cells": [
            {
                "cell_id": "12345",
                "lat": 8.234,
                "lon": 13.567,
                "koppen_code": "Cwa",
                "elevation_m": 1820.5,
                "check_ids_failed": ["value_range_precip", "coverage_climate_cells"]
            }
        ]
    }

Per durable §24 canonical-source-or-pin: the schema lives once
here; the round-trip Pydantic validator
:class:`CellRosterSnapshot` at the bottom of this module is
the reader's contract. The Köppen code field type-validates
against the canonical
:data:`prismpy.koppen.zones.KoppenZone` Literal so a typo'd
zone code on the producer side rejects at construction time
per durable §27 two-vocabulary substrate-drift discipline.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from prismpy.koppen.zones import KoppenZone


SCHEMA_VERSION: Literal["1.0"] = "1.0"


class CellRosterEntry(BaseModel):
    """One cell entry in the snapshot.

    The ``check_ids_failed`` list carries the cell's flagged
    warnings as of validation completion at PACKAGE-end. Empty
    list = unflagged cell (still in the roster for cross-run
    presence-tracking)."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )

    cell_id: str
    """Canonical cell-id reference per
    :class:`prismpy.cells.schema.CellID`."""

    lat: float
    """Cell-center latitude (decimal degrees, WGS84)."""

    lon: float
    """Cell-center longitude (decimal degrees, WGS84)."""

    koppen_code: KoppenZone
    """Köppen-Geiger climate-zone code per the canonical
    :data:`prismpy.koppen.zones.KoppenZone` Literal. A typo'd
    zone code rejects at construction time."""

    elevation_m: Optional[float] = None
    """Cell-center elevation (meters above sea level). Optional —
    cells without DEM coverage emit None. The cockpit's State
    C-newcells variant filter at AC-E3-19 may use elevation for
    tilt-aware comparison; absent elevation falls back to
    coordinate-only comparison."""

    check_ids_failed: List[str]
    """Per-cell check_ids that flagged this cell at validation
    completion. Drawn from the validator output during
    ``_build_cell_summary`` at executor.py:3587-3633. Empty list
    = cell passed all checks."""


class CellRosterSnapshot(BaseModel):
    """Top-level snapshot payload — round-trip contract for
    ``cell_roster_snapshot.json``."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )

    schema_version: Literal["1.0"]
    """Pinned to :data:`SCHEMA_VERSION`."""

    run_id: UUID
    """The ``PipelineRun.id`` this snapshot describes. Lets the
    cross-run comparator key by run rather than by file path."""

    produced_at: datetime
    """Wall-clock at writer invocation (ISO 8601)."""

    cells: List[CellRosterEntry]
    """Per-cell roster entries. Order is canonical ascending by
    ``cell_id`` for byte-stable JSON output."""


def write_cell_roster_snapshot(
    *,
    run_id: UUID,
    cells: List[CellRosterEntry],
    output_path: Union[str, Path],
    produced_at: Optional[datetime] = None,
) -> Path:
    """Write the cell_roster_snapshot.json sidecar to disk.

    Args:
        run_id: The :class:`PipelineRun` UUID the snapshot
            describes.
        cells: Per-cell roster entries assembled from the
            validation report. Caller is responsible for
            populating ``check_ids_failed`` from the
            validator-output report.
        output_path: Where to write the JSON.
        produced_at: Optional override for the timestamp; defaults
            to ``datetime.now(timezone.utc)``.

    Returns:
        Path of the written JSON.

    Atomicity: writes to a sibling temp file + atomic rename so a
    mid-write failure cannot leave a torn payload visible to
    consumers (mirrors
    :func:`cockpit_overrides_writer.write_cockpit_overrides_json`).

    The writer fires loud — no broad-except swallowing per the
    executor.py:3325 lesson. Caller wraps in higher-level
    error handling if non-fatal-on-failure semantics are needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if produced_at is None:
        produced_at = datetime.now(timezone.utc)

    # Order by cell_id ascending for byte-stable output.
    sorted_cells = sorted(cells, key=lambda c: c.cell_id)

    snapshot = CellRosterSnapshot(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        produced_at=produced_at,
        cells=sorted_cells,
    )

    payload = snapshot.model_dump(mode="json")

    # Atomic write — temp file in same dir + os.replace.
    fd, temp_name = tempfile.mkstemp(
        prefix=".cell_roster_snapshot.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
        os.replace(temp_name, str(output_path))
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    return output_path


def new_cells_in_zone(
    *,
    prior_snapshot: CellRosterSnapshot,
    current_snapshot: CellRosterSnapshot,
    zone_code: KoppenZone,
) -> Tuple[str, ...]:
    """Cross-run comparator — return cells in ``zone_code`` that
    appear in ``current_snapshot`` but NOT in ``prior_snapshot``.

    Powers the State C-newcells variant trigger per AC-E3-19 +
    Drill-E3-L absorbed: if the persona's prior decision had
    ``applied_to_scope == "zone"`` over zone X with snapshot
    ``[c1, c2, c3]``, and today's run added c4 to zone X, this
    helper returns ``("c4",)`` so the cockpit can surface the
    State C-newcells panel rather than silently extending the
    prior override.

    Args:
        prior_snapshot: A previously-written
            :class:`CellRosterSnapshot`.
        current_snapshot: Today's snapshot.
        zone_code: The Köppen-zone discriminator. Comparison is
            scoped to cells with ``koppen_code == zone_code`` on
            BOTH sides.

    Returns:
        Tuple of cell_ids present in ``current_snapshot`` for
        ``zone_code`` but absent from ``prior_snapshot`` (also
        filtered to ``zone_code``). Tuple ordered by cell_id
        ascending for deterministic comparison output. Empty
        tuple = no new cells in zone (the cockpit's State
        C-newcells panel does NOT fire).
    """
    prior_zone_cells = {
        entry.cell_id
        for entry in prior_snapshot.cells
        if entry.koppen_code == zone_code
    }
    current_zone_cells = {
        entry.cell_id
        for entry in current_snapshot.cells
        if entry.koppen_code == zone_code
    }
    new_ids = current_zone_cells - prior_zone_cells
    return tuple(sorted(new_ids))


__all__ = [
    "CellRosterEntry",
    "CellRosterSnapshot",
    "SCHEMA_VERSION",
    "new_cells_in_zone",
    "write_cell_roster_snapshot",
]
