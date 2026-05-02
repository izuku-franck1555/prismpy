"""Sprint D.1 AC-5 — UnavailableCause Pydantic Literal + axis-cause coherence.

Pins the cell-summary schema's new cause discriminator + the
fourth ``model_validator`` invariant. The cause taxonomy is
intentionally narrow — only the literals a producer in this
repo emits today are accepted; an unregistered string is a
schema violation so a future writer cannot silently introduce
a divergent vocabulary.

Backward-compat: Pydantic Optional default None tolerates field
absence in records produced before this field landed; the
backward-compat fixture below pins that path against a v2.1
record sourced from prior session output.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from prismpy.cells.schema import (
    CellSummary,
    UnavailableCause,
)


# ---------------------------------------------------------------------------
# AC-5 — cause Literal validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cause,axis",
    [
        ("soil_no_hwsd_coverage", "soil"),
        ("soil_texture_invalid", "soil"),
        ("climate_rh_invalid", "climate"),
    ],
)
def test_known_cause_literal_loads_with_compatible_axis(cause, axis):
    """Each cause Literal value loads cleanly when paired with its
    axis-compatible ``unavailable_reason``."""
    cell = CellSummary(
        cell_id="c0",
        data_availability="unavailable",
        unavailable_reason=axis,
        unavailable_cause=cause,
    )
    assert cell.unavailable_cause == cause


def test_unregistered_cause_string_rejected_by_pydantic():
    """A cause value not in the Literal taxonomy raises ValidationError
    so an undocumented vocabulary cannot reach the consumer."""
    with pytest.raises(ValidationError):
        CellSummary(
            cell_id="c1",
            data_availability="unavailable",
            unavailable_reason="soil",
            unavailable_cause="invented_cause_not_in_taxonomy",
        )


def test_cause_default_is_none_for_complete_cells():
    """``unavailable_cause`` defaults to None and ``data_availability=
    'complete'`` cells round-trip with the field absent."""
    cell = CellSummary(cell_id="c2")
    assert cell.unavailable_cause is None
    assert cell.data_availability == "complete"


def test_cause_taxonomy_lists_only_three_reachable_literals():
    """Per the Sprint D.1 builder subcounter, ``climate_no_data`` and
    ``soil_default_retired`` were dropped because no producer in
    the current repo emits them. The taxonomy enumerates exactly
    the three reachable literals."""
    legal = set(UnavailableCause.__args__)  # type: ignore[attr-defined]
    assert legal == {
        "soil_no_hwsd_coverage",
        "soil_texture_invalid",
        "climate_rh_invalid",
    }


# ---------------------------------------------------------------------------
# AC-5 — axis-cause coherence (the fourth model_validator)
# ---------------------------------------------------------------------------


def test_cause_without_axis_raises():
    """Setting a cause without an axis cannot be valid — the
    consumer needs the axis to route the cell."""
    with pytest.raises(ValidationError) as excinfo:
        CellSummary(
            cell_id="c3",
            # ``data_availability='unavailable'`` would already be
            # rejected by invariant 1 without a reason; verify the
            # cause-without-axis case explicitly by leaving the cell
            # in ``complete`` so invariant 1 is not the trigger.
            data_availability="complete",
            unavailable_cause="soil_no_hwsd_coverage",
        )
    msg = str(excinfo.value)
    assert "unavailable_cause" in msg
    assert "unavailable_reason" in msg


@pytest.mark.parametrize(
    "cause,wrong_axis",
    [
        ("soil_no_hwsd_coverage", "climate"),
        ("soil_texture_invalid", "climate"),
        ("climate_rh_invalid", "soil"),
    ],
)
def test_cause_axis_mismatch_raises(cause, wrong_axis):
    """A soil cause cannot ride on a climate-only axis (and vice
    versa). The mismatched-axis pairing is rejected at schema
    layer so the consumer never sees a contradictory record."""
    with pytest.raises(ValidationError) as excinfo:
        CellSummary(
            cell_id="c4",
            data_availability="unavailable",
            unavailable_reason=wrong_axis,
            unavailable_cause=cause,
        )
    msg = str(excinfo.value)
    assert "incompatible" in msg or "does not include" in msg


@pytest.mark.parametrize(
    "cause",
    ["soil_no_hwsd_coverage", "soil_texture_invalid", "climate_rh_invalid"],
)
def test_cause_compatible_with_climate_and_soil_axis(cause):
    """``climate_and_soil`` is a superset axis — every cause
    (soil-side or climate-side) is compatible with it because
    both substrates failed."""
    cell = CellSummary(
        cell_id="c5",
        data_availability="unavailable",
        unavailable_reason="climate_and_soil",
        unavailable_cause=cause,
    )
    assert cell.unavailable_cause == cause


# ---------------------------------------------------------------------------
# AC-5 — backward compatibility (legacy v2.1 record without cause)
# ---------------------------------------------------------------------------


def test_legacy_record_without_cause_loads_cleanly():
    """A v2.1 record produced before the cause field landed loads
    with ``unavailable_cause`` defaulting to None — the consumer
    does not have to handle a KeyError or a missing-field path."""
    legacy_dict = {
        "cell_id": "c6",
        "lat": 12.4,
        "lon": -5.4,
        "data_availability": "unavailable",
        "unavailable_reason": "soil",
        "cell_summary_version": "2.1",
        "failed_checks": [],
    }
    cell = CellSummary.model_validate(legacy_dict)
    assert cell.unavailable_cause is None
    assert cell.unavailable_reason == "soil"


def test_round_trip_preserves_cause():
    """A cause value survives ``model_dump`` → ``model_validate``."""
    cell = CellSummary(
        cell_id="c7",
        data_availability="unavailable",
        unavailable_reason="climate",
        unavailable_cause="climate_rh_invalid",
    )
    dumped = cell.model_dump()
    reloaded = CellSummary.model_validate(dumped)
    assert reloaded.unavailable_cause == "climate_rh_invalid"


# ---------------------------------------------------------------------------
# AC-1.2 — harmonize/constants.py docstring rationale grep
# ---------------------------------------------------------------------------


def test_harmonize_constants_docstring_cites_validator_alignment():
    """``harmonize/constants.py`` docstring documents the alignment
    between the renormalize-eligible band (+/- 3%) and the
    validator's accept band (+/- 5%) at
    ``validators/scientific.py:1477``."""
    from prismpy import harmonize
    docstring = harmonize.constants.__doc__ or ""
    # Validator alignment cite
    assert "validators/scientific.py:1477" in docstring
    # Tolerance band cite
    assert "[95, 105]" in docstring
    # Jones citation
    assert "Jones" in docstring and "2003" in docstring


def test_harmonize_constants_match_contract_values():
    """Threshold literals match the Draft 3 LOCKED FINAL contract:
    3.0 / 5.0 / 100.0 / 102.0."""
    from prismpy.harmonize.constants import (
        RH_CLIP_THRESHOLD_PCT,
        RH_PHYSICAL_MAX_PCT,
        TEXTURE_RENORMALIZE_THRESHOLD_PCT,
        TEXTURE_WARN_THRESHOLD_PCT,
    )
    assert TEXTURE_RENORMALIZE_THRESHOLD_PCT == 3.0
    assert TEXTURE_WARN_THRESHOLD_PCT == 5.0
    assert RH_PHYSICAL_MAX_PCT == 100.0
    assert RH_CLIP_THRESHOLD_PCT == 102.0
