"""Cell-summary schema tests (v2.1).

Six structural test cases per crop-modeling-specialist's §1.5
spec. Three round-trip cases verify forward/backward compat
across the v2.0 → v2.1 reader/writer matrix; three reject
cases verify the cross-field invariants fire on contradictions
the consumer would otherwise have to silently absorb.

The matrix:

  Producer  Consumer  Behavior
  --------  --------  ------------------------------------------
  v2.1      v2.1      round-trip preserves data_availability +
                       unavailable_reason
  v2.1      v2.0      record loads; new fields silently drop
                       (extra="ignore" shim)
  v2.0      v2.1      record loads; new fields default to
                       complete/None/2.0 (legacy assumption)

The three reject cases pin the invariants — without them, a
downstream consumer would have to guard every read against the
"unavailable but no reason" / "complete with reason" /
"unavailable but failed_checks present" combinations the spec
explicitly forbids.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from prismpy.cells import (
    CELL_SUMMARY_VERSION_LATEST,
    CELL_SUMMARY_VERSIONS,
    CellSummary,
)


# A minimal v2.0-shape Pydantic reader — used to exercise the
# "v2.1 producer + v2.0 consumer" backward-compat scenario. The
# extra="ignore" config is the shim audit point: if a future
# v2.0 reader landed without it, this stand-in test would pass
# but production v2.0 readers would crash on v2.1 records.
class _V20Reader(BaseModel):
    """Stand-in for any v2.0-era cell-summary reader. Mirrors
    the original v2.0 baseline fields and treats unknown fields
    as silently-drop per the §1.4 backward-compat shim."""

    model_config = ConfigDict(extra="ignore")
    cell_id: str | None = None
    lat: float | None = None
    lon: float | None = None
    failed_checks: list = []


class TestRoundTripCompat:
    """The v2.0 / v2.1 reader-writer matrix must preserve
    semantics in both directions."""

    def test_v21_round_trip_preserves_new_fields(self):
        """Case 1: v2.1 producer + v2.1 consumer → new
        fields preserved verbatim."""
        record = {
            "cell_id": "n12_e34",
            "lat": 12.5,
            "lon": -7.3,
            "failed_checks": [],
            "data_availability": "unavailable",
            "unavailable_reason": "climate",
            "cell_summary_version": "2.1",
        }
        loaded = CellSummary.model_validate(record)
        assert loaded.data_availability == "unavailable"
        assert loaded.unavailable_reason == "climate"
        assert loaded.cell_summary_version == "2.1"

    def test_v21_record_loads_in_v20_reader_with_extra_ignore(self):
        """Case 2: v2.1 producer + v2.0 consumer with
        extra="ignore" → record loads; new fields silently
        drop. The shim audit relies on v2.0 readers using
        ConfigDict(extra="ignore"); this test pins that shim
        with a stand-in v2.0 model."""
        v21_record = {
            "cell_id": "n12_e34",
            "lat": 12.5,
            "lon": -7.3,
            "failed_checks": [],
            "data_availability": "unavailable",
            "unavailable_reason": "climate",
            "cell_summary_version": "2.1",
        }
        loaded = _V20Reader.model_validate(v21_record)
        # The v2.0 reader sees only its own fields. The v2.1
        # additions silently drop.
        assert loaded.cell_id == "n12_e34"
        assert not hasattr(loaded, "data_availability")
        assert not hasattr(loaded, "unavailable_reason")

    def test_v20_record_loads_in_v21_reader_with_defaults(self):
        """Case 3: v2.0 producer + v2.1 consumer → defaults
        applied. data_availability defaults to "complete",
        unavailable_reason defaults to None — the implicit
        pre-v2.1 assumption made explicit."""
        v20_record = {
            "cell_id": "n12_e34",
            "lat": 12.5,
            "lon": -7.3,
            "failed_checks": [],
            # No data_availability, unavailable_reason, or
            # cell_summary_version fields — pure v2.0 record.
        }
        loaded = CellSummary.model_validate(v20_record)
        assert loaded.data_availability == "complete"
        assert loaded.unavailable_reason is None
        # The default cell_summary_version is "2.1" (the
        # latest); a v2.0 producer that explicitly stamps
        # the field reads back as "2.0", but a v2.0 record
        # missing the field promotes to the latest schema
        # since that is the consumer's effective contract.
        assert loaded.cell_summary_version == CELL_SUMMARY_VERSION_LATEST

    def test_v20_explicit_version_stays_v20(self):
        """Belt-and-braces — when the v2.0 producer DID stamp
        cell_summary_version='2.0' the consumer must echo that
        exact string, not silently upgrade. The string is the
        provenance marker; rewriting it would lose the
        producer's actual schema version."""
        v20_record = {
            "cell_id": "n12_e34",
            "data_availability": "complete",
            "cell_summary_version": "2.0",
        }
        loaded = CellSummary.model_validate(v20_record)
        assert loaded.cell_summary_version == "2.0"


class TestCrossFieldInvariants:
    """The three §1.3 invariants must fire on contradictory
    field combinations. Pydantic raises ValidationError; tests
    pin both the rejection AND the error-message keyword so a
    future schema change can rewrite the message but the
    rejection contract stays observable."""

    def test_unavailable_without_reason_is_rejected(self):
        """Invariant 1 — data_availability='unavailable' with
        unavailable_reason=None must reject."""
        with pytest.raises(ValidationError) as excinfo:
            CellSummary.model_validate({
                "data_availability": "unavailable",
                "unavailable_reason": None,
            })
        assert "unavailable_reason" in str(excinfo.value)

    def test_complete_with_reason_is_rejected(self):
        """Invariant 2 — data_availability='complete' with a
        non-None unavailable_reason must reject."""
        with pytest.raises(ValidationError) as excinfo:
            CellSummary.model_validate({
                "data_availability": "complete",
                "unavailable_reason": "climate",
            })
        assert "complete" in str(excinfo.value).lower()

    def test_climate_unavailable_rejects_climate_check_fail(self):
        """Invariant 3 (per-axis) — a cell with climate
        unavailable cannot carry a climate check in
        failed_checks. The category discriminator is the
        explicit `category` field on the failed_check entry."""
        with pytest.raises(ValidationError) as excinfo:
            CellSummary.model_validate({
                "data_availability": "unavailable",
                "unavailable_reason": "climate",
                "failed_checks": [{
                    "check_id": "value_range_tmax",
                    "result": "fail",
                    "category": "climate",
                }],
            })
        assert "climate" in str(excinfo.value).lower()

    def test_soil_unavailable_rejects_soil_check_fail(self):
        """Mirror — soil unavailable cannot carry a soil
        check in failed_checks."""
        with pytest.raises(ValidationError) as excinfo:
            CellSummary.model_validate({
                "data_availability": "unavailable",
                "unavailable_reason": "soil",
                "failed_checks": [{
                    "check_id": "value_range_soil_pH",
                    "result": "fail",
                    "category": "soil",
                }],
            })
        assert "soil" in str(excinfo.value).lower()

    def test_full_unavailability_rejects_any_failed_checks(self):
        """When BOTH axes are unavailable, neither climate nor
        soil checks could have run — failed_checks must be
        empty regardless of which axis the entry references."""
        for category in ("climate", "soil"):
            with pytest.raises(ValidationError):
                CellSummary.model_validate({
                    "data_availability": "unavailable",
                    "unavailable_reason": "climate_and_soil",
                    "failed_checks": [{
                        "check_id": (
                            "value_range_tmax" if category == "climate"
                            else "value_range_soil_pH"
                        ),
                        "result": "fail",
                        "category": category,
                    }],
                })

    def test_climate_unavailable_accepts_soil_check_fail(self):
        """The mixed-availability accept path — climate
        unavailable + a legitimate soil check fail loads
        cleanly. The validator did run on the available soil
        axis, so the fail entry is honest information."""
        loaded = CellSummary.model_validate({
            "data_availability": "unavailable",
            "unavailable_reason": "climate",
            "failed_checks": [{
                "check_id": "value_range_soil_pH",
                "result": "fail",
                "category": "soil",
            }],
        })
        assert loaded.data_availability == "unavailable"
        assert len(loaded.failed_checks) == 1
        assert loaded.failed_checks[0]["category"] == "soil"

    def test_soil_unavailable_accepts_climate_check_fail(self):
        """Mirror — soil unavailable + climate fail loads
        cleanly."""
        loaded = CellSummary.model_validate({
            "data_availability": "unavailable",
            "unavailable_reason": "soil",
            "failed_checks": [{
                "check_id": "value_range_tmax",
                "result": "fail",
                "category": "climate",
            }],
        })
        assert loaded.data_availability == "unavailable"
        assert loaded.failed_checks[0]["category"] == "climate"

    def test_per_axis_falls_back_to_name_prefix_when_category_absent(self):
        """Legacy records that pre-date the explicit `category`
        field must still be enforced via the check-id name
        prefix. ``value_range_tmax`` matches the climate
        taxonomy without an explicit category."""
        with pytest.raises(ValidationError):
            CellSummary.model_validate({
                "data_availability": "unavailable",
                "unavailable_reason": "climate",
                "failed_checks": [{
                    "check_id": "value_range_tmax",
                    "result": "fail",
                    # No `category` field — legacy shape.
                }],
            })

    @pytest.mark.parametrize("check_id,reason", [
        ("coverage_climate_cells", "climate"),
        ("region_specific_bounds", "climate"),
        ("value_range_climate", "climate"),
        ("coverage_soil_cells", "soil"),
        ("value_range_soil", "soil"),
        ("value_range_texture_sum", "soil"),
    ])
    def test_axis_aggregator_check_ids_enforced_by_invariant(
        self, check_id, reason,
    ):
        """G7 §2 — the schema taxonomy now classifies the per-cell
        coverage check_ids, region-bounds, and axis-level value-
        range aggregators so a manual fixture or non-executor
        producer cannot smuggle a climate fail past an
        unavailable-climate cell. The executor's per-cell pivot
        already filters these; this test pins the schema-side
        safety net for every classification."""
        with pytest.raises(ValidationError):
            CellSummary.model_validate({
                "data_availability": "unavailable",
                "unavailable_reason": reason,
                "failed_checks": [{
                    "check_id": check_id,
                    "result": "fail",
                    # No explicit category — name-based classification.
                }],
            })

    def test_per_axis_ignores_uncategorised_checks(self):
        """Checks whose name carries neither a climate nor a soil
        affiliation fall outside the per-axis invariant — they
        operate at a different scope (e.g., a delegated marker
        like ``format_compliance`` that has no substrate axis,
        or future post-translate / cross-platform checks). The
        invariant only forbids fails on the unavailable axis;
        truly axis-agnostic check_ids must round-trip cleanly.

        NOTE (G7 §2) — the prior fixture used
        ``coverage_climate_cells`` as the canonical uncategorised
        example, but the taxonomy now classifies that id as
        climate-axis in lockstep with the executor's per-cell
        pivot filter. The new fixture uses
        ``format_compliance`` — a delegated marker whose
        check_id does not encode either substrate."""
        loaded = CellSummary.model_validate({
            "data_availability": "unavailable",
            "unavailable_reason": "climate",
            "failed_checks": [{
                "check_id": "format_compliance",
                "result": "fail",
                # No category, name does not match either prefix.
            }],
        })
        assert len(loaded.failed_checks) == 1


class TestVersionConstants:
    """The exported version constants must match the Literal
    union; if a contributor adds "2.2" without updating the
    Literal, the test catches it before runtime drift."""

    def test_constants_match_literal(self):
        # The CellSummary annotation is a Literal type at the
        # class level. We don't introspect the annotation
        # itself (Pydantic's model_fields does that); we check
        # the constants line up with the test matrix.
        assert "2.0" in CELL_SUMMARY_VERSIONS
        assert "2.1" in CELL_SUMMARY_VERSIONS
        assert CELL_SUMMARY_VERSION_LATEST == "2.1"
        assert CELL_SUMMARY_VERSION_LATEST in CELL_SUMMARY_VERSIONS
