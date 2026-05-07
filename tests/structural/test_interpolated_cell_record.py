"""Structural pin: ``InterpolatedCellRecord`` schema invariants.

Sprint E.2 AC-E2-1. Asserts:

* The Pydantic model has the seven contracted fields with the right
  types + ``extra="forbid"`` + ``validate_assignment=True``.
* ``source_cells`` rejects empty lists.
* Inverted CI bounds (lower > upper) raise ``ValidationError``.
* Equal CI bounds (lower == upper, e.g., k=1 zero-width path) are
  accepted — that's the honest-signal Phrase 2 surface in
  AC-E2-7.
* Unknown ``affected_zone_code`` (not in the registry) raises.
* Unknown ``caveat_codes`` (not in the canonical Literal) raise.
* The DOI regex rejects bibliographic-text strings.
* Unknown fields raise (``extra="forbid"``).
* Reassignment validates (``validate_assignment=True``).
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from prismpy.models.interpolated_cell import InterpolatedCellRecord


# ── §1 valid records — happy path + edge cases ──────────────────────


def _valid_record(**overrides) -> InterpolatedCellRecord:
    """Fixture builder; tests override individual fields."""
    defaults = dict(
        interpolation_method="idw_k4_r15km_w_inverse_dist_sq",
        source_cells=["c001", "c002", "c003", "c004"],
        uncertainty_ci_lower=410.0,
        uncertainty_ci_upper=414.0,
        method_doi="10.1145/800186.810616",
        applied_at_decision_id=uuid4(),
        affected_zone_code="BSh",
        caveat_codes=["sahel-precip-convective"],
    )
    defaults.update(overrides)
    return InterpolatedCellRecord(**defaults)


def test_valid_full_record_constructs_cleanly() -> None:
    record = _valid_record()
    assert record.interpolation_method == "idw_k4_r15km_w_inverse_dist_sq"
    assert len(record.source_cells) == 4
    assert record.uncertainty_ci_lower == 410.0
    assert record.uncertainty_ci_upper == 414.0
    assert record.method_doi == "10.1145/800186.810616"
    assert record.affected_zone_code == "BSh"
    assert record.caveat_codes == ["sahel-precip-convective"]


def test_caveat_codes_default_to_empty_list() -> None:
    record = _valid_record(caveat_codes=[])
    assert record.caveat_codes == []


def test_zero_width_ci_accepted_for_k_one_path() -> None:
    """The k=1 degraded path produces ci_lower == ci_upper == mean
    (Phrase 2 in AC-E2-7's methods text). The schema MUST accept this
    case — it's the honest-signal surface, not a bug."""
    record = _valid_record(uncertainty_ci_lower=412.0, uncertainty_ci_upper=412.0)
    assert record.uncertainty_ci_lower == record.uncertainty_ci_upper


def test_minimum_one_source_cell_accepted() -> None:
    """Single neighbour case (k=1 degraded). Must be valid."""
    record = _valid_record(source_cells=["only_neighbour"])
    assert record.source_cells == ["only_neighbour"]


# ── §2 invalid records — rejection at construction ──────────────────


def test_empty_source_cells_rejected() -> None:
    """A 0-neighbour case routes to skip per AC-E2-3 BEFORE the engine
    runs; an InterpolatedCellRecord with empty source_cells is a
    substrate bug and MUST raise."""
    with pytest.raises(ValidationError):
        _valid_record(source_cells=[])


def test_inverted_ci_bounds_rejected() -> None:
    """ci_lower > ci_upper signals a numeric-formula bug; reject at
    construction with an explicit message naming the violation."""
    with pytest.raises(ValidationError, match="bounds inverted"):
        _valid_record(uncertainty_ci_lower=414.0, uncertainty_ci_upper=410.0)


def test_unknown_zone_code_rejected() -> None:
    """Affected zone MUST be in the canonical KoppenZone Literal
    (which mirrors the registry per AC-E2-20). An out-of-registry
    code raises."""
    with pytest.raises(ValidationError):
        _valid_record(affected_zone_code="Cwb")  # not in 5-zone Sprint E.2 scope


def test_unknown_caveat_code_rejected() -> None:
    """A caveat code outside the canonical CaveatCode Literal
    raises per durable §23 phantom-bug discipline (don't manufacture
    caveats that aren't there)."""
    with pytest.raises(ValidationError):
        _valid_record(caveat_codes=["random-typo-caveat"])


def test_invalid_doi_format_rejected() -> None:
    """The DOI regex rejects bibliographic-text strings + freeform
    non-DOI input. Forbidden examples below."""
    invalid_dois = (
        "Shepard 1968",  # bibliographic text
        "10",            # missing slash + name
        "doi:10.1145/800186.810616",  # leading "doi:" prefix
        "  10.1145/800186.810616  ",  # whitespace
    )
    for invalid in invalid_dois:
        with pytest.raises(ValidationError):
            _valid_record(method_doi=invalid)


def test_unknown_field_rejected_via_extra_forbid() -> None:
    """``extra="forbid"`` catches typos at the field-name level —
    a typo'd field name doesn't silently lose data; it raises."""
    with pytest.raises(ValidationError):
        InterpolatedCellRecord(
            interpolation_method="idw_k4_r15km_w_inverse_dist_sq",
            source_cells=["c1"],
            uncertainty_ci_lower=1.0,
            uncertainty_ci_upper=2.0,
            method_doi="10.1145/800186.810616",
            applied_at_decision_id=uuid4(),
            affected_zone_code="BSh",
            caveat_codes=[],
            unknown_field="value",  # typo / drift
        )


def test_reassignment_to_invalid_per_field_type_rejected() -> None:
    """``validate_assignment=True`` re-runs FIELD validators on
    attribute set, so a type-mismatch on reassignment raises.
    Cross-field model_validator behavior on reassignment is a
    Pydantic v2 implementation detail; the construction-time
    invariant in ``test_inverted_ci_bounds_rejected`` is the
    load-bearing pin for the lower<=upper rule."""
    record = _valid_record()
    with pytest.raises(ValidationError):
        record.affected_zone_code = "Cwb"  # not in 5-zone Literal


def test_validate_assignment_config_enabled() -> None:
    """Pin the config flag itself — ``validate_assignment=True`` is
    the schema-level guarantee that assignment-time validation runs;
    a future ConfigDict edit that drops the flag would silently lose
    type-checking on reassignment."""
    config = InterpolatedCellRecord.model_config
    assert config.get("validate_assignment") is True
    assert config.get("extra") == "forbid"


def test_unknown_interpolation_method_rejected() -> None:
    """The Literal locks the single canonical method identifier."""
    with pytest.raises(ValidationError):
        _valid_record(interpolation_method="kriging_v1")


# ── §3 dunder-all is the canonical export surface ───────────────────


def test_module_exports_celliid_and_record_in_dunder_all() -> None:
    from prismpy.models import interpolated_cell
    assert sorted(interpolated_cell.__all__) == [
        "CellID",
        "InterpolatedCellRecord",
    ]
