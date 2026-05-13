"""F-DJ Pin F-DJ-1 — ManagementConfig.sowing_mode alias normalization.

Asserts the prismpy schema accepts ``'fixed'`` as an alias for
``'fixed_date'`` (backward-compat for the wizard producer at
``prismweb/templates/wizard/crop.html:285-286``). Canonical
``'fixed_date'`` + ``'opportunistic'`` pass through unchanged;
invalid inputs still rejected per ``Literal`` constraint.

Per F-DJ 2026-05-13 cycle-2 amendment 2 LOCKED + durable §27
producer-consumer parity discipline.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from prismpy.config.schema import ManagementConfig


def test_sowing_mode_alias_fixed_normalizes_to_fixed_date() -> None:
    """``'fixed'`` alias normalizes to canonical ``'fixed_date'``."""
    mgmt = ManagementConfig.model_validate(
        {"planting_density": 100000.0, "sowing_mode": "fixed"}
    )
    assert mgmt.sowing_mode == "fixed_date", (
        f"Expected 'fixed' to normalize to 'fixed_date'; got "
        f"{mgmt.sowing_mode!r}"
    )


def test_sowing_mode_canonical_fixed_date_passthrough() -> None:
    """Canonical ``'fixed_date'`` accepted unchanged."""
    mgmt = ManagementConfig.model_validate(
        {"planting_density": 100000.0, "sowing_mode": "fixed_date"}
    )
    assert mgmt.sowing_mode == "fixed_date"


def test_sowing_mode_opportunistic_passthrough() -> None:
    """Canonical ``'opportunistic'`` accepted unchanged (no normalization)."""
    mgmt = ManagementConfig.model_validate(
        {"planting_density": 100000.0, "sowing_mode": "opportunistic"}
    )
    assert mgmt.sowing_mode == "opportunistic"


def test_sowing_mode_invalid_rejected() -> None:
    """Invalid sowing_mode values still rejected per Literal constraint
    (e.g., future ``'planting_window'`` UI option would need an explicit
    alias entry + validator branch — not silent acceptance)."""
    with pytest.raises(ValidationError):
        ManagementConfig.model_validate(
            {"planting_density": 100000.0, "sowing_mode": "invalid_value"}
        )
