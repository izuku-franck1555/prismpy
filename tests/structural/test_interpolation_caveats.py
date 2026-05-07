"""Structural pin: ``caveats_for(zone, check_id)`` domain-rule mapping.

Sprint E.2 AC-E2-6. The function is the canonical source for which
``CaveatCode`` values apply to a (zone, check_id) pair. This pin
covers:

* The three Sprint E.2 rules — (BSh, precip), (BSh, wind),
  (Cwa, precip) — return the expected caveat codes.
* All other (zone, check_id) pairs return ``[]`` (absence of caveat
  is its own honest signal).
* Returned codes are a subset of the canonical ``CaveatCode``
  Literal (every output is a recognised code).
* The function is deterministic — same arguments produce same
  output across calls.
"""

from __future__ import annotations

import typing

from prismpy.standards.caveat_codes import CaveatCode
from prismpy.standards.interpolation_caveats import caveats_for


# ── §1 the three Sprint E.2 rules ───────────────────────────────────


def test_sahel_precip_returns_sahel_precip_convective() -> None:
    assert caveats_for("BSh", "value_range_precip") == [
        "sahel-precip-convective"
    ]


def test_sahel_wind_returns_sahel_wind_convective() -> None:
    assert caveats_for("BSh", "value_range_wind") == [
        "sahel-wind-convective"
    ]


def test_highland_precip_returns_highland_orographic_excluded() -> None:
    assert caveats_for("Cwa", "value_range_precip") == [
        "highland-orographic-excluded"
    ]


# ── §2 absent-of-caveat returns empty list ──────────────────────────


def test_sahel_with_unrelated_check_id_returns_empty() -> None:
    """Sahel BSh + (anything other than precip / wind) → no caveat."""
    for check_id in ("value_range_tmax", "value_range_tmin", "value_range_srad"):
        assert caveats_for("BSh", check_id) == [], (
            f"BSh + {check_id!r} should return [] but got "
            f"{caveats_for('BSh', check_id)}"
        )


def test_non_sahel_non_highland_zones_return_empty() -> None:
    """Tropical (Af / Aw / Cfa) zones have no Sprint E.2 caveats."""
    for zone in ("Af", "Aw", "Cfa"):
        for check_id in (
            "value_range_precip",
            "value_range_wind",
            "value_range_tmax",
            "value_range_tmin",
        ):
            assert caveats_for(zone, check_id) == [], (
                f"{zone} + {check_id!r} should return [] but got "
                f"{caveats_for(zone, check_id)}"
            )


def test_highland_with_unrelated_check_id_returns_empty() -> None:
    """Cwa Highland + (anything other than precip) → no caveat."""
    for check_id in ("value_range_tmax", "value_range_tmin", "value_range_wind"):
        assert caveats_for("Cwa", check_id) == [], (
            f"Cwa + {check_id!r} should return [] but got "
            f"{caveats_for('Cwa', check_id)}"
        )


def test_unknown_check_id_returns_empty() -> None:
    """Unknown check_id → empty list (forward-compat: a future
    check_id added to the validators surface should default to no
    caveat unless explicitly extended here)."""
    assert caveats_for("BSh", "value_range_unknown_future_check") == []


# ── §3 every returned code is a valid CaveatCode Literal value ──────


def test_every_returned_code_is_a_canonical_caveat_code() -> None:
    """Defensive: a future contributor extending ``caveats_for``
    might introduce a typo in a returned literal. This pin asserts
    every code the function ever returns is a member of the
    canonical ``CaveatCode`` Literal."""
    canonical_codes = set(typing.get_args(CaveatCode))
    sweep_inputs = [
        ("BSh", "value_range_precip"),
        ("BSh", "value_range_wind"),
        ("BSh", "value_range_tmax"),
        ("Aw", "value_range_precip"),
        ("Cfa", "value_range_precip"),
        ("Cwa", "value_range_precip"),
        ("Cwa", "value_range_tmax"),
        ("Af", "value_range_precip"),
    ]
    for zone, check_id in sweep_inputs:
        codes = caveats_for(zone, check_id)
        for code in codes:
            assert code in canonical_codes, (
                f"caveats_for({zone!r}, {check_id!r}) returned "
                f"{code!r} which is NOT in CaveatCode Literal "
                f"{sorted(canonical_codes)!r}"
            )


# ── §4 determinism (same input → same output across calls) ──────────


def test_function_is_deterministic() -> None:
    """The function has no global state or random-number generator;
    repeated calls with the same args MUST produce identical output."""
    for _ in range(5):
        assert caveats_for("BSh", "value_range_precip") == [
            "sahel-precip-convective"
        ]
        assert caveats_for("Cwa", "value_range_precip") == [
            "highland-orographic-excluded"
        ]
        assert caveats_for("Cfa", "value_range_precip") == []


# ── §5 dunder-all is the canonical export surface ───────────────────


def test_module_exports_only_caveats_for() -> None:
    from prismpy.standards import interpolation_caveats
    assert interpolation_caveats.__all__ == ["caveats_for"]
