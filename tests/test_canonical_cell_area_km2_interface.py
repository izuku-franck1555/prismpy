"""Interface compliance tests for :mod:`prismpy.cells.canonical_cell_area_km2`.

Per the canonical-helper convention shipped in PRs #48 / #50 / #51 and
the test-authorship discipline (builder writes AST guards + interface
compliance; eval-2 writes spec-driven behavior probes from contract
only): this file covers the helper's signature, type behavior, frozen-
dataclass invariants, default-constant binding, and obvious error
paths.

Specialist greenlight (2026-05-19) verified the geodesic formula is
``area_km2 = resolution_deg² × DEG2_TO_KM2 × cos(latitude_radians)`` with
``DEG2_TO_KM2 = 12364.0`` (spherical-mean Earth radius constant). The
tests below pin the magnitudes the specialist named so future formula
edits cannot drift without surfacing a failure here first.
"""

from __future__ import annotations

import math
import pytest

from prismpy.cells.canonical_cell_area_km2 import (
    DEG2_TO_KM2_DEFAULT,
    SpatialRef,
    canonical_cell_area_km2,
)


# ── Signature + return-type compliance ──────────────────────────────────────


def test_helper_returns_float() -> None:
    sr = SpatialRef(
        resolution_deg=1.0 / 12.0,
        cell_centroid_latitude=lambda cid: 0.0,
    )
    result = canonical_cell_area_km2(1, sr)
    assert isinstance(result, float)


def test_helper_accepts_int_cell_id() -> None:
    sr = SpatialRef(
        resolution_deg=1.0 / 12.0,
        cell_centroid_latitude=lambda cid: 14.0,
    )
    for cid in (0, 1, 12345, -1, -999):
        result = canonical_cell_area_km2(cid, sr)
        assert isinstance(result, float)
        assert result > 0.0


# ── SpatialRef dataclass invariants ─────────────────────────────────────────


def test_spatial_ref_is_frozen() -> None:
    """Specialist spec: ``SpatialRef`` is a frozen dataclass — mutation
    must raise to prevent silent per-cell-area drift mid-package.
    """
    sr = SpatialRef(
        resolution_deg=1.0 / 12.0,
        cell_centroid_latitude=lambda cid: 0.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        sr.resolution_deg = 5.0


def test_spatial_ref_default_deg2_to_km2_is_specialist_constant() -> None:
    sr = SpatialRef(
        resolution_deg=1.0 / 12.0,
        cell_centroid_latitude=lambda cid: 0.0,
    )
    assert sr.deg2_to_km2 == 12364.0
    assert DEG2_TO_KM2_DEFAULT == 12364.0


def test_spatial_ref_accepts_custom_deg2_to_km2() -> None:
    sr = SpatialRef(
        resolution_deg=1.0 / 12.0,
        cell_centroid_latitude=lambda cid: 0.0,
        deg2_to_km2=12392.5,
    )
    assert sr.deg2_to_km2 == 12392.5


def test_spatial_ref_callable_cell_centroid_latitude_resolves_per_cell() -> None:
    """The ``cell_centroid_latitude`` field is a callable per specialist
    spec; passing a dict-lookup closure produces per-cell variation in
    the emitted area.
    """
    lat_by_cell = {100: 0.0, 101: 25.0, 102: -10.0}
    sr = SpatialRef(
        resolution_deg=1.0 / 12.0,
        cell_centroid_latitude=lat_by_cell.__getitem__,
    )
    area_100 = canonical_cell_area_km2(100, sr)
    area_101 = canonical_cell_area_km2(101, sr)
    area_102 = canonical_cell_area_km2(102, sr)
    assert area_100 > area_101
    assert pytest.approx(area_100, rel=1e-6) == canonical_cell_area_km2(102, sr) * (
        math.cos(math.radians(0.0)) / math.cos(math.radians(-10.0))
    )
    assert area_102 < area_100


# ── Specialist-verified magnitudes (Sahel band 5-arcmin) ────────────────────


@pytest.mark.parametrize(
    ("latitude_deg", "expected_km2"),
    [
        (0.0, 85.86),
        (14.0, 83.31),
        (25.0, 77.82),
    ],
)
def test_helper_matches_specialist_magnitudes_at_5arcmin(
    latitude_deg: float, expected_km2: float,
) -> None:
    sr = SpatialRef(
        resolution_deg=1.0 / 12.0,
        cell_centroid_latitude=lambda cid: latitude_deg,
    )
    actual = canonical_cell_area_km2(1, sr)
    assert actual == pytest.approx(expected_km2, abs=0.05), (
        f"5-arcmin cell area at {latitude_deg}° expected ~{expected_km2} "
        f"km² per specialist 2026-05-19 sanity; got {actual:.4f}"
    )


def test_helper_area_strictly_decreasing_with_absolute_latitude() -> None:
    """cos(lat) compresses the longitude axis; area declines as |lat|
    increases. Pin the monotonicity so a sign-flip or radians/degrees
    bug surfaces immediately.
    """
    lats = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0]
    areas = []
    for lat in lats:
        sr = SpatialRef(
            resolution_deg=1.0 / 12.0,
            cell_centroid_latitude=lambda cid, lat=lat: lat,
        )
        areas.append(canonical_cell_area_km2(1, sr))
    for prev, current in zip(areas, areas[1:]):
        assert current < prev, (
            f"area must decrease with |latitude|; got {areas}"
        )


def test_helper_zero_at_poles() -> None:
    """At ±90°, cos(lat) = 0 → area_km2 = 0.0. Pin the boundary."""
    sr_north = SpatialRef(
        resolution_deg=1.0 / 12.0,
        cell_centroid_latitude=lambda cid: 90.0,
    )
    sr_south = SpatialRef(
        resolution_deg=1.0 / 12.0,
        cell_centroid_latitude=lambda cid: -90.0,
    )
    assert canonical_cell_area_km2(1, sr_north) == pytest.approx(0.0, abs=1e-10)
    assert canonical_cell_area_km2(1, sr_south) == pytest.approx(0.0, abs=1e-10)


def test_helper_symmetric_in_latitude_sign() -> None:
    """Geodesic area depends on |lat| (cos is even); ±lat must produce
    the same area.
    """
    for lat in (10.0, 25.0, 45.0):
        sr_pos = SpatialRef(
            resolution_deg=1.0 / 12.0,
            cell_centroid_latitude=lambda cid, lat=lat: lat,
        )
        sr_neg = SpatialRef(
            resolution_deg=1.0 / 12.0,
            cell_centroid_latitude=lambda cid, lat=lat: -lat,
        )
        assert canonical_cell_area_km2(1, sr_pos) == pytest.approx(
            canonical_cell_area_km2(1, sr_neg), rel=1e-12,
        )


# ── Resolution scaling ──────────────────────────────────────────────────────


def test_helper_quadratic_in_resolution() -> None:
    """area ∝ resolution_deg² → doubling resolution quadruples area."""
    sr_5arcmin = SpatialRef(
        resolution_deg=1.0 / 12.0,
        cell_centroid_latitude=lambda cid: 0.0,
    )
    sr_10arcmin = SpatialRef(
        resolution_deg=2.0 / 12.0,
        cell_centroid_latitude=lambda cid: 0.0,
    )
    area_5 = canonical_cell_area_km2(1, sr_5arcmin)
    area_10 = canonical_cell_area_km2(1, sr_10arcmin)
    assert area_10 == pytest.approx(4.0 * area_5, rel=1e-9)


# ── Obvious error paths ─────────────────────────────────────────────────────


def test_helper_propagates_callable_exceptions() -> None:
    """When the cell_centroid_latitude callable raises (unknown cell_id
    against a dict lookup, downstream service unavailable, etc.) the
    error surfaces verbatim — the helper does NOT silently swallow.
    """
    def _bad_lookup(cid: int) -> float:
        raise KeyError(f"cell_id {cid} not in registry")

    sr = SpatialRef(
        resolution_deg=1.0 / 12.0,
        cell_centroid_latitude=_bad_lookup,
    )
    with pytest.raises(KeyError, match="cell_id 999"):
        canonical_cell_area_km2(999, sr)


def test_helper_rejects_non_numeric_resolution() -> None:
    """resolution_deg coerced via ``float(...)``: non-numeric input
    surfaces as ``ValueError`` / ``TypeError`` (Python coercion).
    """
    sr = SpatialRef(
        resolution_deg="not-a-number",  # type: ignore[arg-type]
        cell_centroid_latitude=lambda cid: 0.0,
    )
    with pytest.raises((ValueError, TypeError)):
        canonical_cell_area_km2(1, sr)
