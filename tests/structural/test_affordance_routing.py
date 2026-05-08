"""Structural pin: ``route_affordance`` routing rules.

Sprint E.2 AC-E2-3 + Drill G (Highland-precip boundary trio).
Asserts every routing rule in the contract returns the documented
``AffordanceType`` for representative inputs.
"""

from __future__ import annotations

import pytest

from prismpy.config.schema import Platform
from prismpy.validators.affordance_routing import route_affordance


# ── §1 value_range_X with neighbours → "interpolate" ────────────────


@pytest.mark.parametrize(
    "check_id",
    [
        "value_range_tmax",
        "value_range_tmin",
        "value_range_srad",
        "value_range_rh",
        "value_range_wind",
    ],
)
def test_value_range_with_neighbours_routes_to_interpolate(check_id: str) -> None:
    result = route_affordance(
        check_id=check_id,
        platform=Platform.PYTHIA,
        zone="BSh",
        elevation_m=300.0,
        n_candidates_in_radius=4,
        cell_failure_context={},
    )
    assert result == "interpolate"


@pytest.mark.parametrize(
    "check_id",
    [
        "value_range_tmax",
        "value_range_tmin",
        "value_range_srad",
        "value_range_rh",
        "value_range_wind",
        "value_range_precip",
        "region_specific_bounds",
    ],
)
def test_value_range_with_zero_neighbours_routes_to_skip(check_id: str) -> None:
    """Per WA CA-1: 0-neighbour case is caught BEFORE the IDW
    engine runs; routing returns "skip" defensively."""
    result = route_affordance(
        check_id=check_id,
        platform=Platform.PYTHIA,
        zone="BSh",
        elevation_m=300.0,
        n_candidates_in_radius=0,
        cell_failure_context={},
    )
    assert result == "skip"


# ── §2 Drill G — Highland-precip boundary trio ──────────────────────


@pytest.mark.parametrize(
    "elevation_m,expected",
    [
        (1499.99, "interpolate"),  # below threshold → interpolate
        (1500.00, "interpolate"),  # AT threshold → interpolate (>1500 only)
        (1500.01, "skip"),         # above threshold → skip (orographic exclusion)
    ],
)
def test_drill_g_highland_precip_boundary_trio(
    elevation_m: float, expected: str
) -> None:
    result = route_affordance(
        check_id="value_range_precip",
        platform=Platform.PYTHIA,
        zone="Cwa",
        elevation_m=elevation_m,
        n_candidates_in_radius=4,
        cell_failure_context={},
    )
    assert result == expected, (
        f"Cwa precip at elevation_m={elevation_m} should route to "
        f"{expected!r}; got {result!r}"
    )


def test_non_highland_precip_routes_to_interpolate() -> None:
    """Non-Cwa zones don't trigger Highland-precip exclusion even at
    high elevation — orographic effects per Daly 2006 are zone-
    classified, not just elevation-classified."""
    result = route_affordance(
        check_id="value_range_precip",
        platform=Platform.PYTHIA,
        zone="BSh",
        elevation_m=2000.0,
        n_candidates_in_radius=4,
        cell_failure_context={},
    )
    assert result == "interpolate"


# ── §3 soil + cross-variable + completeness rules ───────────────────


@pytest.mark.parametrize(
    "check_id",
    [
        "value_range_soil_clay",
        "value_range_soil_sand",
        "value_range_soil_silt",
        "value_range_soil_bulk_density",
        "value_range_soil_ph",
        "value_range_soil_texture_sum",
    ],
)
def test_soil_check_ids_route_to_skip(check_id: str) -> None:
    """Soil profile gaps don't follow climate gradients; never
    interpolate per Decision 5 Stage-0 §11.5."""
    result = route_affordance(
        check_id=check_id,
        platform=Platform.PYTHIA,
        zone="BSh",
        elevation_m=300.0,
        n_candidates_in_radius=4,
        cell_failure_context={},
    )
    assert result == "skip"


def test_cross_variable_routes_to_skip() -> None:
    result = route_affordance(
        check_id="cross_variable",
        platform=Platform.PYTHIA,
        zone="BSh",
        elevation_m=300.0,
        n_candidates_in_radius=4,
        cell_failure_context={},
    )
    assert result == "skip"


def test_temporal_completeness_sarra_py_routes_to_rerun() -> None:
    result = route_affordance(
        check_id="temporal_completeness",
        platform=Platform.SARRA_PY,
        zone="BSh",
        elevation_m=300.0,
        n_candidates_in_radius=4,
        cell_failure_context={},
    )
    assert result == "rerun_full_sources"


def test_coverage_per_cell_routes_to_rerun() -> None:
    result = route_affordance(
        check_id="coverage_per_cell",
        platform=Platform.PYTHIA,
        zone="BSh",
        elevation_m=300.0,
        n_candidates_in_radius=4,
        cell_failure_context={},
    )
    assert result == "rerun_full_sources"


def test_crop_region_mismatch_routes_to_override() -> None:
    result = route_affordance(
        check_id="crop_region_mismatch",
        platform=Platform.PYTHIA,
        zone="BSh",
        elevation_m=300.0,
        n_candidates_in_radius=4,
        cell_failure_context={},
    )
    assert result == "override"


def test_unknown_check_id_routes_to_acknowledge() -> None:
    """Default-fallback for unknown check_ids: acknowledge (Bucket-2)."""
    result = route_affordance(
        check_id="value_range_unknown_future_check",
        platform=Platform.PYTHIA,
        zone="BSh",
        elevation_m=300.0,
        n_candidates_in_radius=4,
        cell_failure_context={},
    )
    assert result == "acknowledge"


def test_dunder_all_lists_canonical_exports() -> None:
    from prismpy.validators import affordance_routing
    assert sorted(affordance_routing.__all__) == [
        "AFFORDANCE_TO_ACTION_MAP",
        "AffordanceType",
        "route_affordance",
    ]
