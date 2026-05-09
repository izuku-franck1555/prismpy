"""Structural pin: ``IDW_RADIUS_BY_PLATFORM`` registry coverage.

Sprint E.3 AC-E3-11 sub-5 + Stage 1 §9 #6 + Draft 2 CMS CA-1
BLOCKING absorbed. Three invariants close the silent-drift class
per durable §24 canonical-source-or-pin + durable §27 two-
vocabulary substrate-drift:

§1 Coverage — the registry built by
``_build_idw_radius_by_platform`` covers EVERY ``Platform.*`` enum
member. A new platform addition without a registry entry fires
this pin loud rather than silently routing to a missing-key
KeyError at runtime.

§2 ACEA radius is 100 km (CMS CA-1 BLOCKING resolution) — pinned
explicitly so a future refactor that drifts the value back to
the universal 15 km default fires loud. ACEA at 50 km grid
spacing needs ≥75 km radius to capture decorrelation envelope;
the prior 15 km default produced 0 candidates per cell →
production-blocking class.

§3 Helper round-trip — :func:`get_idw_radius_for_platform` returns
the right value for each known platform AND raises ``KeyError``
on an unknown platform. The helper is the canonical dispatch
surface; the IDW orchestrator + future per-platform IDW callers
route through it (no inline registry lookups outside the helper).
"""

from __future__ import annotations

import pytest

from prismpy.config.schema import Platform
from prismpy.standards.idw_methods import (
    _build_idw_radius_by_platform,
    get_idw_radius_for_platform,
)


# ── §1 coverage ────────────────────────────────────────────────────


def test_registry_covers_every_platform_enum_member() -> None:
    """A new ``Platform.*`` member without a registry entry is the
    silent-drift case the canonical-source-or-pin discipline
    catches. Pin coverage of every enum member."""
    registry = _build_idw_radius_by_platform()
    enum_values = {p.value for p in Platform}
    registry_values = set(registry.keys())
    missing = enum_values - registry_values
    assert not missing, (
        f"IDW_RADIUS_BY_PLATFORM registry missing entries for "
        f"Platform.* members: {sorted(missing)}. Add a radius "
        f"entry in ``prismpy/standards/idw_methods.py`` per "
        f"CMS CA-1 BLOCKING discipline."
    )


def test_registry_has_no_phantom_platform_keys() -> None:
    """The registry should not carry keys that aren't in the enum
    (typo / stale rename caught loud)."""
    registry = _build_idw_radius_by_platform()
    enum_values = {p.value for p in Platform}
    phantom = set(registry.keys()) - enum_values
    assert not phantom, (
        f"IDW_RADIUS_BY_PLATFORM registry has phantom keys not in "
        f"Platform enum: {sorted(phantom)}."
    )


# ── §2 ACEA radius is 100 km ───────────────────────────────────────


def test_acea_radius_is_one_hundred_km() -> None:
    """CMS CA-1 BLOCKING absorption — ACEA at 50 km grid spacing
    needs 100 km radius (≈2× cell size) to capture decorrelation
    envelope. The prior universal 15 km default produced 0
    candidates per cell, a production-blocking failure class."""
    radius = get_idw_radius_for_platform(Platform.ACEA.value)
    assert radius == 100.0, (
        f"ACEA radius drifted from 100 km canonical: got {radius}. "
        f"Sprint E.3 AC-E3-11 + CMS CA-1 BLOCKING pinned 100 km. "
        f"A drift to 15 km re-introduces the 0-candidate failure "
        f"class on the Sorghum-Maradi-2020-2020 fixture."
    )


def test_canonical_radii_match_contract_spec() -> None:
    """Pin all 4 platform radii against the AC-E3-11 contract
    table. A drift on any one fires the pin loud — domain
    rationale per CMS Boulanger 2018 + AGRHYMET + Lebel & Ali
    2009."""
    expected = {
        Platform.SARRA_PY.value: 15.0,
        Platform.CRAFT.value:    15.0,
        Platform.PYTHIA.value:   25.0,
        Platform.ACEA.value:     100.0,
    }
    for platform_value, expected_radius in expected.items():
        actual = get_idw_radius_for_platform(platform_value)
        assert actual == expected_radius, (
            f"Platform={platform_value!r} radius drifted: got "
            f"{actual}, expected {expected_radius} per CMS CA-1 "
            f"BLOCKING table"
        )


# ── §3 helper round-trip ──────────────────────────────────────────


def test_helper_returns_radius_for_each_platform() -> None:
    """Round-trip: helper returns the value for every Platform.*
    member."""
    for platform in Platform:
        radius = get_idw_radius_for_platform(platform.value)
        assert isinstance(radius, float)
        assert radius > 0.0


def test_helper_raises_keyerror_on_unknown_platform() -> None:
    """An unknown platform value fails loud (KeyError) per the
    helper's docstring contract; per ``feedback_no_data_cooking
    .md`` honest-signal floor, silent-skip-and-default is
    forbidden."""
    with pytest.raises(KeyError):
        get_idw_radius_for_platform("not_a_real_platform")
