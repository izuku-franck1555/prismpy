"""Structural pin: ``lookup_zone_and_elevation`` substrate.

Sprint E.2 AC-E2-5. Asserts:

* The function fails LOUD via ``LookupSkipped`` when paths are
  None / missing / unreadable — NEVER silently returns a
  placeholder per ``feedback_no_data_cooking.md`` honest-signal
  contract.
* The ``ZoneElevationLookup`` result type carries the canonical
  fields (``koppen_code`` + ``elevation_m``).
* Module exports + the canonical export surface are stable.

Real-rasterio integration tests live in the evaluator's pre-commit
Gate B framework (Phase 1 verification-strategy doc); this module
covers the substrate-level invariants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prismpy.harmonize.idw_interpolation import Cell
from prismpy.preprocess.zone_elevation_lookup import (
    LookupSkipped,
    ZoneElevationLookup,
    lookup_zone_and_elevation,
)


_TEST_CELL = Cell(cell_id="test", lat=13.5417, lon=2.1250, value=0.0)


# ── §1 fail-loud on missing inputs ──────────────────────────────────


def test_none_paths_raise_lookup_skipped() -> None:
    with pytest.raises(LookupSkipped, match="requires both"):
        lookup_zone_and_elevation(_TEST_CELL, hwsd2_path=None, koppen_path=None)


def test_missing_hwsd2_path_raises_lookup_skipped(tmp_path: Path) -> None:
    missing_hwsd = tmp_path / "missing_hwsd.tif"
    koppen = tmp_path / "koppen.tif"
    koppen.write_bytes(b"")  # exists but doesn't matter for this assertion
    with pytest.raises(LookupSkipped, match="HWSD2 raster missing"):
        lookup_zone_and_elevation(
            _TEST_CELL, hwsd2_path=missing_hwsd, koppen_path=koppen
        )


def test_missing_koppen_path_raises_lookup_skipped(tmp_path: Path) -> None:
    hwsd = tmp_path / "hwsd.tif"
    hwsd.write_bytes(b"")
    missing_koppen = tmp_path / "missing_koppen.tif"
    with pytest.raises(LookupSkipped, match="Köppen raster missing"):
        lookup_zone_and_elevation(
            _TEST_CELL, hwsd2_path=hwsd, koppen_path=missing_koppen
        )


# ── §2 result type shape ────────────────────────────────────────────


def test_result_type_has_canonical_fields() -> None:
    """Smoke test: the dataclass has koppen_code + elevation_m."""
    result = ZoneElevationLookup(koppen_code="BSh", elevation_m=300.0)
    assert result.koppen_code == "BSh"
    assert result.elevation_m == 300.0


def test_result_type_is_frozen() -> None:
    """Frozen dataclass: post-construction mutation is forbidden so
    the lookup result can't be edited downstream by accident."""
    result = ZoneElevationLookup(koppen_code="BSh", elevation_m=300.0)
    with pytest.raises(Exception):  # FrozenInstanceError or similar
        result.koppen_code = "Aw"  # type: ignore[misc]


# ── §3 dunder-all ───────────────────────────────────────────────────


def test_module_exports_canonical_surface() -> None:
    from prismpy.preprocess import zone_elevation_lookup
    expected = sorted([
        "LookupSkipped",
        "ZoneElevationLookup",
        "lookup_zone_and_elevation",
    ])
    assert sorted(zone_elevation_lookup.__all__) == expected
