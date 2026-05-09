"""Behavioral tests for cell_roster_snapshot.json writer + cross-run
comparator.

Sprint E.3 AC-E3-10 sub-criteria + Drill-E3-L (snapshot semantics
State C-newcells variant trigger).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from prismpy.cockpit.cell_roster_snapshot import (
    CellRosterEntry,
    CellRosterSnapshot,
    SCHEMA_VERSION,
    new_cells_in_zone,
    write_cell_roster_snapshot,
)


# ── Fixture builders ───────────────────────────────────────────────


def _entry(
    *,
    cell_id: str = "c001",
    lat: float = 8.5,
    lon: float = 13.2,
    koppen_code: str = "BSh",
    elevation_m: float = 250.0,
    check_ids_failed: list = None,
) -> CellRosterEntry:
    return CellRosterEntry(
        cell_id=cell_id,
        lat=lat,
        lon=lon,
        koppen_code=koppen_code,
        elevation_m=elevation_m,
        check_ids_failed=check_ids_failed or [],
    )


# ── §1 schema round-trip + Köppen validation ───────────────────────


def test_snapshot_pydantic_round_trip(tmp_path: Path) -> None:
    """AC-E3-10 sub-1+4: schema validates Köppen zones via the
    canonical Literal; writer fires + round-trips through the
    Pydantic schema."""
    output_path = tmp_path / "cell_roster_snapshot.json"
    run_id = uuid4()
    write_cell_roster_snapshot(
        run_id=run_id,
        cells=[
            _entry(cell_id="c001", check_ids_failed=["value_range_tmax"]),
            _entry(cell_id="c002", koppen_code="Cwa", elevation_m=1820.5),
        ],
        output_path=output_path,
    )
    raw = json.loads(output_path.read_text())
    snapshot = CellRosterSnapshot(**raw)
    assert snapshot.schema_version == SCHEMA_VERSION
    assert snapshot.run_id == run_id
    assert len(snapshot.cells) == 2


def test_invalid_koppen_code_rejects() -> None:
    """A typo'd zone code rejects at construction time per
    durable §27 two-vocabulary substrate-drift discipline. The
    canonical Literal at prismpy/koppen/zones.py is the single
    source."""
    with pytest.raises(ValidationError):
        CellRosterEntry(
            cell_id="c001",
            lat=8.5,
            lon=13.2,
            koppen_code="ZZZ",  # type: ignore[arg-type]
            elevation_m=250.0,
            check_ids_failed=[],
        )


def test_optional_elevation_accepts_none() -> None:
    """Cells without DEM coverage emit None for elevation."""
    entry = _entry(elevation_m=None)
    assert entry.elevation_m is None


def test_check_ids_failed_empty_list_accepts() -> None:
    """An unflagged cell (passed all checks) has empty
    check_ids_failed but is still in the roster for cross-run
    presence-tracking."""
    entry = _entry(check_ids_failed=[])
    assert entry.check_ids_failed == []


# ── §2 byte-stable output + sorting ────────────────────────────────


def test_cells_sorted_by_cell_id_for_byte_stable_output(
    tmp_path: Path,
) -> None:
    """Per the AC-E3-10 PACKAGE-end pattern matching
    cockpit_observed_values.json byte-stable contract."""
    output_path = tmp_path / "cell_roster_snapshot.json"
    write_cell_roster_snapshot(
        run_id=uuid4(),
        cells=[
            _entry(cell_id="c003"),
            _entry(cell_id="c001"),
            _entry(cell_id="c002"),
        ],
        output_path=output_path,
    )
    snapshot = CellRosterSnapshot(**json.loads(output_path.read_text()))
    cell_ids = [entry.cell_id for entry in snapshot.cells]
    assert cell_ids == sorted(cell_ids)
    assert cell_ids == ["c001", "c002", "c003"]


# ── §3 atomicity drill ─────────────────────────────────────────────


def test_no_torn_artifact_on_simulated_mid_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """AC-E3-10 sub-2 — writer fails loud; mid-write failure
    leaves NO torn artifact at output_path."""
    import os as os_module

    output_path = tmp_path / "cell_roster_snapshot.json"

    def _replace_raises(*args, **kwargs):
        raise OSError("simulated mid-write failure")

    monkeypatch.setattr(os_module, "replace", _replace_raises)

    with pytest.raises(OSError, match="simulated mid-write failure"):
        write_cell_roster_snapshot(
            run_id=uuid4(),
            cells=[_entry()],
            output_path=output_path,
        )

    assert not output_path.exists()


# ── §4 cross-run comparator (Drill-E3-L) ───────────────────────────


def test_new_cells_in_zone_returns_diff() -> None:
    """AC-E3-10 sub-3 + Drill-E3-L: prior snapshot has [c1, c2,
    c3] in zone BSh; current snapshot adds c4 in zone BSh.
    Comparator returns ('c4',)."""
    prior = CellRosterSnapshot(
        schema_version="1.0",
        run_id=uuid4(),
        produced_at=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
        cells=[
            _entry(cell_id="c001"),
            _entry(cell_id="c002"),
            _entry(cell_id="c003"),
        ],
    )
    current = CellRosterSnapshot(
        schema_version="1.0",
        run_id=uuid4(),
        produced_at=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
        cells=[
            _entry(cell_id="c001"),
            _entry(cell_id="c002"),
            _entry(cell_id="c003"),
            _entry(cell_id="c004"),
        ],
    )
    new_ids = new_cells_in_zone(
        prior_snapshot=prior,
        current_snapshot=current,
        zone_code="BSh",
    )
    assert new_ids == ("c004",)


def test_new_cells_in_zone_filters_by_zone() -> None:
    """The comparator scopes both sides by zone_code. A c4 added
    to a different zone (Cwa) does NOT appear in the BSh result."""
    prior = CellRosterSnapshot(
        schema_version="1.0",
        run_id=uuid4(),
        produced_at=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
        cells=[_entry(cell_id="c001", koppen_code="BSh")],
    )
    current = CellRosterSnapshot(
        schema_version="1.0",
        run_id=uuid4(),
        produced_at=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
        cells=[
            _entry(cell_id="c001", koppen_code="BSh"),
            _entry(cell_id="c004", koppen_code="Cwa"),  # different zone
        ],
    )
    new_ids = new_cells_in_zone(
        prior_snapshot=prior,
        current_snapshot=current,
        zone_code="BSh",
    )
    assert new_ids == ()


def test_new_cells_in_zone_empty_when_no_change() -> None:
    """Identical rosters → empty tuple."""
    cells = [_entry(cell_id=f"c{i:03d}") for i in range(1, 4)]
    prior = CellRosterSnapshot(
        schema_version="1.0",
        run_id=uuid4(),
        produced_at=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
        cells=cells,
    )
    current = CellRosterSnapshot(
        schema_version="1.0",
        run_id=uuid4(),
        produced_at=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
        cells=cells,
    )
    assert (
        new_cells_in_zone(
            prior_snapshot=prior,
            current_snapshot=current,
            zone_code="BSh",
        )
        == ()
    )


def test_new_cells_returned_in_sorted_order() -> None:
    """Pin: returned tuple is sorted by cell_id ascending for
    deterministic comparator output."""
    prior = CellRosterSnapshot(
        schema_version="1.0",
        run_id=uuid4(),
        produced_at=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
        cells=[_entry(cell_id="c001")],
    )
    current = CellRosterSnapshot(
        schema_version="1.0",
        run_id=uuid4(),
        produced_at=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc),
        cells=[
            _entry(cell_id="c001"),
            _entry(cell_id="c099"),
            _entry(cell_id="c004"),
            _entry(cell_id="c042"),
        ],
    )
    new_ids = new_cells_in_zone(
        prior_snapshot=prior,
        current_snapshot=current,
        zone_code="BSh",
    )
    assert new_ids == ("c004", "c042", "c099")
    assert list(new_ids) == sorted(new_ids)
