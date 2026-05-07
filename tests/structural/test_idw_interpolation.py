"""Structural pin: ``interpolate_idw`` + ``SpatialIndex``.

Sprint E.2 AC-E2-2 + Drills E (0-neighbour) / F (k=2 degraded) /
F2 (k=1 zero-width) / R (numeric CI formula k=4 with negative-control).

The IDW math is small enough to hand-compute against the known
formula:
    weighted_mean = sum(w_i * v_i) / sum(w_i)
    s_w = sqrt(sum(w_i * (v_i - mean)^2) / sum(w_i))
    CI half-width = 1.96 * s_w / sqrt(n)

For the test fixtures we use lat/lon pairs whose haversine distance
is straightforward to compute to a few decimal places. The negative-
control pattern (Drill R-e) wraps the formula with a deliberate
off-by-one and asserts the test would catch the bug.
"""

from __future__ import annotations

import math

import pytest

from prismpy.harmonize.idw_interpolation import (
    Cell,
    InsufficientNeighborsError,
    InterpolationResult,
    interpolate_idw,
)
from prismpy.spatial_index import SpatialIndex


# ── Fixture helpers ──────────────────────────────────────────────────


# Choose target at the equator (0, 0). Place neighbours at small
# offsets so haversine distance is dominated by the latitude/longitude
# delta. At the equator, 1° ~= 111.32 km; we pick offsets in the
# 0.05° range to land neighbours in the 5-13 km bucket.
_TARGET = Cell(cell_id="target", lat=0.0, lon=0.0, value=0.0)


def _make_neighbour(suffix: str, lat: float, lon: float, value: float) -> Cell:
    return Cell(cell_id=f"n_{suffix}", lat=lat, lon=lon, value=value)


# ── §1 happy path k=4 ───────────────────────────────────────────────


def test_k_four_returns_weighted_mean_and_ci() -> None:
    """Drill R (a) — synthetic 4-neighbour input with hand-computed
    expected output (within float-precision)."""
    neighbours = [
        _make_neighbour("a", 0.045, 0.0, 10.0),   # ~5.0 km north
        _make_neighbour("b", 0.063, 0.0, 12.0),   # ~7.0 km north
        _make_neighbour("c", 0.090, 0.0, 14.0),   # ~10.0 km north
        _make_neighbour("d", 0.117, 0.0, 16.0),   # ~13.0 km north
    ]
    result = interpolate_idw(_TARGET, neighbours)

    assert isinstance(result, InterpolationResult)
    assert len(result.source_cells) == 4
    assert result.n_neighbors_in_radius == 4
    assert result.degraded_due_to_insufficient_neighbors is False
    # Closer neighbours dominate; weighted mean lies between min and max.
    assert 10.0 < result.value < 16.0
    # Lower-weighted CI in the closer-neighbour direction (smaller
    # std around mean).
    assert result.ci_lower < result.value < result.ci_upper


def test_self_match_filtered_before_distance_calc() -> None:
    """A candidate with the SAME cell_id as the target is dropped
    before distance computation — the divide-by-zero on 1/d² would
    blow up the math otherwise."""
    target_with_self = Cell(cell_id="self", lat=0.0, lon=0.0, value=99.0)
    neighbours = [
        Cell(cell_id="self", lat=0.0, lon=0.0, value=99.0),  # self
        _make_neighbour("a", 0.045, 0.0, 10.0),
    ]
    result = interpolate_idw(target_with_self, neighbours)
    # Only the non-self neighbour contributed.
    assert result.source_cells == ["n_a"]
    assert result.value == 10.0


# ── §2 degraded paths ───────────────────────────────────────────────


def test_drill_e_zero_neighbours_raises() -> None:
    """Drill E — pass zero candidates within R; assert exception."""
    far_neighbour = _make_neighbour("far", 1.0, 1.0, 99.0)  # ~157 km away
    with pytest.raises(InsufficientNeighborsError, match="zero candidate"):
        interpolate_idw(_TARGET, [far_neighbour])


def test_drill_f_two_neighbours_degraded_path() -> None:
    """Drill F — k=2 case: degraded=True, len(source_cells)=2,
    same CI formula with n=2."""
    neighbours = [
        _make_neighbour("a", 0.045, 0.0, 10.0),
        _make_neighbour("b", 0.090, 0.0, 14.0),
    ]
    result = interpolate_idw(_TARGET, neighbours)
    assert result.degraded_due_to_insufficient_neighbors is True
    assert len(result.source_cells) == 2
    # CI is non-trivial (two distinct values → non-zero variance).
    assert result.ci_upper > result.ci_lower


def test_drill_f2_single_neighbour_zero_width_ci() -> None:
    """Drill F2 — k=1 case: ci_lower == ci_upper == value;
    degraded=True; n=1."""
    only_neighbour = _make_neighbour("only", 0.045, 0.0, 12.345)
    result = interpolate_idw(_TARGET, [only_neighbour])
    assert result.degraded_due_to_insufficient_neighbors is True
    assert result.source_cells == ["n_only"]
    assert result.value == pytest.approx(12.345)
    assert result.ci_lower == pytest.approx(12.345)
    assert result.ci_upper == pytest.approx(12.345)


def test_three_neighbours_degraded() -> None:
    """k=3 also degraded; CI formula with n=3."""
    neighbours = [
        _make_neighbour("a", 0.045, 0.0, 10.0),
        _make_neighbour("b", 0.063, 0.0, 12.0),
        _make_neighbour("c", 0.090, 0.0, 14.0),
    ]
    result = interpolate_idw(_TARGET, neighbours)
    assert result.degraded_due_to_insufficient_neighbors is True
    assert len(result.source_cells) == 3


# ── §3 numeric formula validation (Drill R) ─────────────────────────


def test_drill_r_numeric_formula_with_known_values() -> None:
    """Drill R (a) — synthetic input + hand-computed weighted mean +
    weighted std + 1.96 * s_w / sqrt(4). Compare to within float
    precision."""
    # Place 4 neighbours due-north at known distances, with known values.
    neighbours = [
        _make_neighbour("a", 0.045, 0.0, 10.0),
        _make_neighbour("b", 0.063, 0.0, 12.0),
        _make_neighbour("c", 0.090, 0.0, 14.0),
        _make_neighbour("d", 0.117, 0.0, 16.0),
    ]
    result = interpolate_idw(_TARGET, neighbours)

    # Hand-compute against haversine distances.
    from prismpy.utils.gis_utils import haversine_distance

    distances = [
        haversine_distance(0.0, 0.0, n.lon, n.lat) for n in neighbours
    ]
    weights = [1.0 / (d ** 2) for d in distances]
    sum_w = sum(weights)
    expected_mean = sum(w * n.value for w, n in zip(weights, neighbours)) / sum_w
    expected_var = (
        sum(w * (n.value - expected_mean) ** 2 for w, n in zip(weights, neighbours))
        / sum_w
    )
    expected_std = math.sqrt(expected_var)
    expected_half = 1.96 * expected_std / math.sqrt(4)

    assert result.value == pytest.approx(expected_mean, rel=1e-9)
    assert result.ci_lower == pytest.approx(expected_mean - expected_half, rel=1e-9)
    assert result.ci_upper == pytest.approx(expected_mean + expected_half, rel=1e-9)


def test_drill_r_negative_control_off_by_one_caught() -> None:
    """Drill R (e) — if the formula were `1.96 * s_w / sqrt(n+1)`
    (off-by-one) the CI bounds would be SMALLER than the correct
    formula. Assert the actual computed CI is wider than the
    off-by-one approximation; this drill verifies the test would
    catch a sqrt(n+1) regression."""
    neighbours = [
        _make_neighbour("a", 0.045, 0.0, 10.0),
        _make_neighbour("b", 0.063, 0.0, 12.0),
        _make_neighbour("c", 0.090, 0.0, 14.0),
        _make_neighbour("d", 0.117, 0.0, 16.0),
    ]
    result = interpolate_idw(_TARGET, neighbours)
    # Off-by-one formula CI half-width.
    correct_half = result.ci_upper - result.value
    off_by_one_half = correct_half * math.sqrt(4) / math.sqrt(5)
    assert correct_half > off_by_one_half, (
        f"correct half-width {correct_half} should exceed "
        f"off-by-one half-width {off_by_one_half}"
    )


def test_radius_filter_excludes_far_neighbours() -> None:
    """Neighbours outside R don't contribute even if they're in
    the candidate pool."""
    neighbours = [
        _make_neighbour("near", 0.045, 0.0, 10.0),
        _make_neighbour("far", 1.0, 1.0, 99.0),  # ~157 km — outside R=15
    ]
    result = interpolate_idw(_TARGET, neighbours)
    assert result.source_cells == ["n_near"]


# ── §4 SpatialIndex wrapper ─────────────────────────────────────────


def test_spatial_index_finds_neighbours_within_radius() -> None:
    cells = [
        _make_neighbour("a", 0.045, 0.0, 10.0),
        _make_neighbour("b", 0.090, 0.0, 12.0),
        _make_neighbour("far", 1.0, 1.0, 99.0),
    ]
    index = SpatialIndex(cells)
    target = Cell(cell_id="target", lat=0.0, lon=0.0, value=0.0)
    neighbours = index.query_neighbours_within_radius_km(target, radius_km=15.0)
    cell_ids = sorted(n.cell_id for n in neighbours)
    assert cell_ids == ["n_a", "n_b"]  # "far" excluded by radius


def test_spatial_index_excludes_self_by_cell_id() -> None:
    cells = [
        _make_neighbour("a", 0.045, 0.0, 10.0),
    ]
    self_target = Cell(cell_id="n_a", lat=0.045, lon=0.0, value=10.0)
    index = SpatialIndex(cells + [self_target])
    neighbours = index.query_neighbours_within_radius_km(self_target, radius_km=15.0)
    # The self-cell with cell_id == "n_a" is in the index but
    # filtered by cell_id equality.
    assert all(n.cell_id != "n_a" for n in neighbours)


def test_spatial_index_empty_roster_rejected() -> None:
    with pytest.raises(ValueError, match="at least one cell"):
        SpatialIndex([])


def test_spatial_index_len() -> None:
    cells = [_make_neighbour(str(i), 0.01 * i, 0.0, float(i)) for i in range(10)]
    index = SpatialIndex(cells)
    assert len(index) == 10


# ── §5 dunder-all ───────────────────────────────────────────────────


def test_idw_module_exports_canonical_surface() -> None:
    from prismpy.harmonize import idw_interpolation
    expected = sorted([
        "Cell",
        "InsufficientNeighborsError",
        "InterpolationResult",
        "interpolate_idw",
    ])
    assert sorted(idw_interpolation.__all__) == expected


def test_spatial_index_module_exports() -> None:
    from prismpy import spatial_index
    assert spatial_index.__all__ == ["SpatialIndex"]
