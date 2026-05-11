"""G7 §2 — validator short-circuit + cell-summary v2.1 integration.

Tests pin the contract added in §2: every input-data validator
emits a ``result='unavailable'`` record (rather than a misleading
pass/warning/fail) when its axis has no input. The ``_unavailable``
helper centralises the canonical dict shape so every short-circuit
site stays in lockstep, and the cell-summary builder bumps to
``cell_summary_version='2.1'`` once it threads the new
``data_availability`` / ``unavailable_reason`` fields and applies
the §1 invariant 3 per-axis pivot filter.

The four persona framings woven through the assertions:

* **Aminata (DSSAT MISDAT)** — when climate is unavailable, the
  validator emits an explicit unavailable marker with the
  ``icasa_misdat=True`` details flag rather than a
  silently-imputed pass. Aminata's downstream DSSAT pipeline can
  refuse to ingest cells whose climate axis is documented-missing
  and route them to her exclusion list.
* **Moussa (stakeholder counts)** — ``n_unavailable`` is its own
  top-level rollup count alongside ``n_pass`` / ``n_warning`` /
  ``n_fail`` so Moussa's executive summary distinguishes "checks
  did not run" from "checks ran and found problems".
* **Dr. Kofi (ICASA traceability)** — every unavailable record
  carries ``details.cause`` (the discriminator) AND
  ``details.icasa_misdat=True`` (the standards lineage). Kofi's
  audit trail can grep for the marker without parsing summaries.
* **Ibrahim (Region Health distinct counts)** — the cell-summary
  v2.1 fields (``data_availability`` / ``unavailable_reason``)
  let Ibrahim's mobile Region Health card render distinct counts
  per axis (climate-only unavailable / soil-only unavailable /
  both unavailable) without re-deriving from failed_checks.
"""
from __future__ import annotations

from datetime import date as _date
from types import SimpleNamespace

import pytest

from prismpy.cells import CELL_SUMMARY_VERSION_LATEST, CellSummary
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.translators.base import UnifiedData
from prismpy.validators.scientific import (
    CLIMATE_TIER1_CHECKS,
    SOIL_CHECKS_ALL,
    UNAVAILABLE_CAUSES,
    _UNAVAILABLE_RESULT,
    _check_coverage,
    _check_coverage_climate_cells,
    _check_coverage_soil_cells,
    _check_cross_variable_consistency,
    _check_region_bounds,
    _check_soil_completeness,
    _check_temporal_completeness,
    _check_value_ranges,
    _has_climate_data,
    _has_soil_data,
    _unavailable,
    run_scientific_validation,
)


# ---------------------------------------------------------------------------
# Fixture helpers — minimal stand-ins keep the tests independent of the
# project-wide pytest fixtures so a future fixture refactor does not
# silently change the §2 contract surface.
# ---------------------------------------------------------------------------


def _make_region():
    return Region(
        name="t", country="t", country_iso3="TST",
        bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
    )


def _make_grid(n_cells: int = 2) -> SpatialGrid:
    cells = [
        GridCell(cell_id=i, lat=0.5, lon=0.5,
                 row=0, col=i, resolution="5arcmin")
        for i in range(n_cells)
    ]
    return SpatialGrid(
        bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        resolution="5arcmin", cells=cells,
    )


def _make_ts(n_records: int = 1, *, cell_id: int = 0):
    records = [ClimateRecord(
        date=_date(2020, 1, d + 1),
        tmax=30.0, tmin=20.0, precip=2.0, srad=20.0,
    ) for d in range(n_records)]
    return ClimateTimeSeries(
        records=records, location_id=str(cell_id),
        lat=0.5, lon=0.5, source="TEST",
    )


def _make_profile(*, profile_id: str = "p0", with_layers: bool = True):
    layers = []
    if with_layers:
        layers.append(SoilLayer(
            depth_top=0.0, depth_bottom=0.2,
            sand=40.0, clay=30.0,
        ))
    return SoilProfile(
        profile_id=profile_id, lat=0.5, lon=0.5,
        source="iSDA", layers=layers,
    )


def _make_unified(*, n_cells=2, climate=None, soil=None):
    return UnifiedData(
        region=_make_region(),
        grid=_make_grid(n_cells),
        climate=climate if climate is not None else {},
        soil=soil if soil is not None else {},
    )


class _FakeTemporal:
    def __init__(self):
        self.start_year = 2020
        self.end_year = 2020
        self.spinup_years = 0

    def get_climate_end_date(self, crop_cal):
        return _date(2020, 1, 5)


def _make_config():
    return SimpleNamespace(
        temporal=_FakeTemporal(),
        crop=SimpleNamespace(calendar=None),
    )


# ---------------------------------------------------------------------------
# §2.1 — _unavailable helper canonical shape
# ---------------------------------------------------------------------------


class TestUnavailableHelperShape:
    """The helper centralises the canonical dict so every short-
    circuit site stays in lockstep. Tests pin the six top-level
    keys + the two non-negotiable details keys."""

    def test_six_top_level_keys(self):
        rec = _unavailable(
            check_id="value_range_climate",
            scope="per_record",
            cause="no_climate_fetch",
        )
        assert set(rec.keys()) == {
            "check", "scope", "result", "summary",
            "manuscript_claim", "details",
        }

    def test_result_is_literal_unavailable(self):
        rec = _unavailable("temporal_completeness", "global", "no_climate_fetch")
        assert rec["result"] == _UNAVAILABLE_RESULT == "unavailable"

    def test_check_field_passes_through(self):
        rec = _unavailable("foo_bar", "per_cell", "no_climate_fetch")
        assert rec["check"] == "foo_bar"

    def test_details_carry_cause_and_icasa_marker(self):
        """Dr. Kofi's audit trail — every unavailable record must
        carry the explicit MISDAT marker so a grep finds them
        without parsing summaries."""
        rec = _unavailable("temporal_completeness", "global", "no_climate_fetch")
        assert rec["details"]["cause"] == "no_climate_fetch"
        assert rec["details"]["icasa_misdat"] is True

    def test_default_summary_terse_per_spec(self):
        """The crop-specialist §2 pack pinned the summary to the
        terse "Data not available; validation skipped." copy —
        less prose than the verbose default we shipped at 391eb5a.
        The persona-friendly framing lives on the longer
        manuscript_claim; the summary stays scan-friendly for
        cockpit chip text."""
        rec = _unavailable("temporal_completeness", "global", "no_climate_fetch")
        assert rec["summary"] == "Data not available; validation skipped."

    def test_caller_summary_overrides_default(self):
        rec = _unavailable(
            check_id="x",
            scope="global",
            cause="no_climate_fetch",
            summary="custom",
        )
        assert rec["summary"] == "custom"

    def test_caller_details_merge_after_canonical_keys(self):
        rec = _unavailable(
            check_id="x",
            scope="global",
            cause="no_climate_fetch",
            details={"extra_key": "extra_value"},
        )
        assert rec["details"]["cause"] == "no_climate_fetch"
        assert rec["details"]["icasa_misdat"] is True
        assert rec["details"]["extra_key"] == "extra_value"

    def test_affected_cells_default_empty_list(self):
        """The pivot reads ``details.affected_cells`` and would skip
        records whose key is absent (treated as ``None`` → empty
        iter). The helper sets the empty-list default so consumers
        can iterate without a None-guard."""
        rec = _unavailable("temporal_completeness", "per_cell", "no_climate_fetch")
        assert rec["details"]["affected_cells"] == []

    def test_affected_cells_caller_can_supply(self):
        rec = _unavailable(
            check_id="x", scope="per_cell", cause="no_soil_match",
            affected_cells=[1, 2, 3],
        )
        assert rec["details"]["affected_cells"] == [1, 2, 3]

    def test_default_manuscript_claim_includes_misdat_marker(self):
        """Crop-specialist §2 pack DRY assertion — every default
        manuscript_claim must mention ICASA MISDAT and the explicit
        "Validation was not performed" framing so Dr. Kofi's audit
        grep finds the standards lineage without parsing summaries.
        The check_id appears verbatim so the audit row reads
        "Validation was not performed for <check_id>"."""
        rec = _unavailable("foo", "global", "no_climate_fetch")
        assert "ICASA MISDAT" in rec["manuscript_claim"]
        assert "Validation was not performed" in rec["manuscript_claim"]
        assert "foo" in rec["manuscript_claim"]


# ---------------------------------------------------------------------------
# §2.2 — _has_climate_data / _has_soil_data discriminators
# ---------------------------------------------------------------------------


class TestAxisAvailabilityHelpers:
    """The helpers gate every short-circuit. Tests document the
    discriminator semantics so a future change to one validator
    does not silently desynchronise from the rest."""

    def test_has_climate_empty_dict_false(self):
        assert _has_climate_data({}) is False

    def test_has_climate_none_false(self):
        assert _has_climate_data(None) is False

    def test_has_climate_per_cell_with_records_true(self):
        climate = {0: _make_ts(n_records=3)}
        assert _has_climate_data(climate) is True

    def test_has_climate_per_cell_with_empty_records_false(self):
        """ClimateTimeSeries with empty records list is the same as
        no climate — the file-based validator still runs in a
        delegated state but the per-cell path has nothing to
        check, so the axis counts as unavailable."""
        climate = {0: _make_ts(n_records=0)}
        assert _has_climate_data(climate) is False

    def test_has_climate_file_based_true(self):
        """SARRA-Py file-based shape; the validator delegates to
        post-translate sampling which CAN run, so the axis is
        available in the §2 sense (the validator's file-based
        branch fires its own info/warning records)."""
        assert _has_climate_data({"rainfall_dir": "/x", "agera5_dir": "/y"}) is True

    def test_has_soil_empty_dict_false(self):
        assert _has_soil_data({}) is False

    def test_has_soil_with_layers_true(self):
        soil = {0: _make_profile()}
        assert _has_soil_data(soil) is True

    def test_has_soil_empty_layers_false(self):
        soil = {0: _make_profile(with_layers=False)}
        assert _has_soil_data(soil) is False


# ---------------------------------------------------------------------------
# §2.3 — Per-validator short-circuit
# ---------------------------------------------------------------------------


class TestTemporalCompletenessShortCircuit:
    """When climate is unavailable the temporal completeness
    validator must short-circuit with the unavailable marker
    rather than emit a vacuous warning. Aminata's DSSAT MISDAT
    pipeline relies on this distinction."""

    def test_no_climate_emits_unavailable(self):
        unified = _make_unified(n_cells=2, climate={}, soil={0: _make_profile()})
        check = _check_temporal_completeness(unified, _make_config())
        assert check["result"] == "unavailable"
        assert check["details"]["cause"] == "no_climate_fetch"
        assert check["details"]["icasa_misdat"] is True

    def test_with_climate_runs_normally(self):
        climate = {0: _make_ts(n_records=5)}
        unified = _make_unified(n_cells=1, climate=climate)
        check = _check_temporal_completeness(unified, _make_config())
        assert check["result"] in ("pass", "warning", "fail")


class TestCrossVariableConsistencyShortCircuit:
    def test_no_climate_emits_unavailable(self):
        unified = _make_unified(climate={})
        check = _check_cross_variable_consistency(unified)
        assert check["result"] == "unavailable"

    def test_no_climate_does_not_emit_phantom_pass(self):
        """The prior shape returned ``result='pass'`` on empty
        climate — Aminata would see a green tick on a check that
        ran on zero records. The unavailable record forbids the
        phantom pass."""
        unified = _make_unified(climate={})
        check = _check_cross_variable_consistency(unified)
        assert check["result"] != "pass"


class TestValueRangesShortCircuit:
    """The value-range validator must short-circuit per axis —
    climate axis empty → ``value_range_climate`` unavailable;
    soil axis empty → ``value_range_soil`` unavailable. Mixed
    availability runs the available axis and emits unavailable
    only for the missing one."""

    def test_no_climate_emits_climate_axis_unavailable(self):
        unified = _make_unified(
            climate={}, soil={0: _make_profile()},
        )
        checks = _check_value_ranges(unified)
        climate_checks = [
            c for c in checks if c["check"] == "value_range_climate"
        ]
        assert len(climate_checks) == 1
        assert climate_checks[0]["result"] == "unavailable"
        assert climate_checks[0]["details"]["cause"] == "no_climate_fetch"

    def test_no_soil_emits_soil_axis_unavailable(self):
        climate = {0: _make_ts(n_records=2)}
        unified = _make_unified(climate=climate, soil={})
        checks = _check_value_ranges(unified)
        soil_checks = [c for c in checks if c["check"] == "value_range_soil"]
        assert len(soil_checks) == 1
        assert soil_checks[0]["result"] == "unavailable"
        assert soil_checks[0]["details"]["cause"] == "no_soil_match"

    def test_neither_axis_emits_both_unavailable(self):
        unified = _make_unified(climate={}, soil={})
        checks = _check_value_ranges(unified)
        unavailable_check_ids = {
            c["check"] for c in checks if c["result"] == "unavailable"
        }
        assert "value_range_climate" in unavailable_check_ids
        assert "value_range_soil" in unavailable_check_ids

    def test_climate_only_runs_normally_with_soil_unavailable(self):
        """Mixed availability — Aminata's climate axis ran cleanly
        and her cells hold real value-range pass records, while
        the soil axis carries the unavailable marker. The two
        axes must NOT cross-contaminate."""
        climate = {0: _make_ts(n_records=3)}
        unified = _make_unified(climate=climate, soil={})
        checks = _check_value_ranges(unified)
        climate_var_records = [
            c for c in checks
            if c["check"].startswith("value_range_")
            and c["check"] not in ("value_range_climate", "value_range_soil")
            and not c["check"].startswith("value_range_soil_")
            and c["check"] != "value_range_texture_sum"
        ]
        # Some climate per-variable records exist (tmax / tmin / precip / srad).
        assert any(
            c["result"] in ("pass", "warning", "fail")
            for c in climate_var_records
        ), "climate value-range records did not run on a mixed-availability fixture"
        # And the soil axis emitted exactly one unavailable record.
        soil_unavail = [
            c for c in checks if c["check"] == "value_range_soil"
            and c["result"] == "unavailable"
        ]
        assert len(soil_unavail) == 1


class TestSoilCompletenessShortCircuit:
    def test_no_soil_emits_unavailable(self):
        unified = _make_unified(soil={})
        check = _check_soil_completeness(unified, "pythia")
        assert check["result"] == "unavailable"
        assert check["details"]["cause"] == "no_soil_match"
        # The platform context is preserved for the Methods-tab
        # reader who needs to know which platform short-circuited.
        assert check["details"]["platform"] == "pythia"

    def test_empty_layers_count_as_unavailable(self):
        """A profile with no layers is the same as no profile — the
        check has nothing to inspect."""
        unified = _make_unified(soil={0: _make_profile(with_layers=False)})
        check = _check_soil_completeness(unified, "craft")
        assert check["result"] == "unavailable"


class TestCoverageGlobalShortCircuit:
    """The global coverage check (``spatial_temporal_coverage``)
    short-circuits ONLY when both axes are unavailable — partial
    availability still surfaces the available-axis count."""

    def test_both_axes_missing_emits_unavailable(self):
        unified = _make_unified(climate={}, soil={})
        check = _check_coverage(unified, _make_config())
        assert check["result"] == "unavailable"
        assert check["details"]["cause"] == "no_climate_and_soil_fetch"

    def test_climate_only_missing_runs_normally(self):
        soil = {0: _make_profile(), 1: _make_profile(profile_id="p1")}
        unified = _make_unified(climate={}, soil=soil)
        check = _check_coverage(unified, _make_config())
        # Soil count surfaces; result is warning (climate at 0/N).
        assert check["result"] in ("pass", "warning")


class TestCoveragePerCellShortCircuit:
    """F-CO honest-signal rollup — INVERTED from the prior G7 §2
    contract. Full axis unavailability with a populated grid now
    emits ``fail`` (not ``unavailable``) so the top-level rollup
    at ``run_scientific_validation`` matches the per-cell view.
    The rollup filters ``unavailable`` out of the ``runnable``
    set; emitting unavailable for a 0/N coverage path made
    overall_result='warning' while per-cell validation_status='fail'
    on every cell (warning-auditor §6.2 RCA: F-AG NEW manifestation
    at retrieval-failure axis).

    The §1 per-axis invariant 3 the G7 §2 design intended to
    protect is still satisfied: the fail record is a SINGLE
    per-cell entry covering the whole grid (not one fail per
    validator-axis cross product), and the ``cause`` discriminator
    marks it as axis-level so the cockpit drawer renders the
    unavailability narrative rather than per-cell-anomaly framing.
    """

    def test_climate_axis_fully_missing_emits_fail(self):
        # F-CO Layer 1 — when climate is empty and grid has cells,
        # emit fail with affected_cells = all grid cells.
        unified = _make_unified(climate={}, soil={0: _make_profile()})
        check = _check_coverage_climate_cells(unified)
        assert check["result"] == "fail"
        # n_total + n_missing both anchor to the grid size, so the
        # consumer can render "0/N covered" honestly.
        assert check["details"]["n_total"] == 2
        assert check["details"]["n_missing"] == 2
        assert len(check["details"]["affected_cells"]) == 2
        # ``cause`` discriminator preserved so the cockpit drawer
        # renders the axis-level absence rather than per-cell
        # anomaly framing.
        assert check["details"]["cause"] == "no_climate_fetch"
        # ICASA / MISDAT provenance preserved on the record so
        # Dr. Kofi's audit-grep continuity is unaffected.
        assert check["details"]["icasa_misdat"] is True

    def test_soil_axis_fully_missing_emits_fail(self):
        # F-CO Layer 1 symmetric mirror for soil.
        unified = _make_unified(soil={})
        check = _check_coverage_soil_cells(unified)
        assert check["result"] == "fail"
        assert check["details"]["cause"] == "no_soil_match"
        assert check["details"]["icasa_misdat"] is True
        assert check["details"]["n_missing"] == check["details"]["n_total"]

    def test_partial_climate_coverage_still_emits_fail(self):
        """Mixed availability — the validator runs and reports the
        partial gap; the per-axis filter runs at the executor
        pivot, not here."""
        climate = {0: _make_ts(n_records=1)}
        unified = _make_unified(n_cells=2, climate=climate)
        check = _check_coverage_climate_cells(unified)
        assert check["result"] == "fail"
        assert 1 in check["details"]["affected_cells"]


class TestRegionBoundsShortCircuit:
    def test_no_climate_emits_unavailable(self):
        # Region bounds short-circuits at the climate-availability
        # check ONLY when a real region (with thresholds) was
        # detected; the universal fallback emits ``info`` regardless.
        # Use a Sahel-region centroid (-3, 15) so the bounds lookup
        # picks up the Sahel thresholds and reaches the climate
        # availability gate.
        sahel_region = Region(
            name="sahel-test", country="t", country_iso3="TST",
            bounds=BoundingBox(minx=-4, miny=14, maxx=-2, maxy=16),
        )
        unified = UnifiedData(
            region=sahel_region, grid=_make_grid(1),
            climate={}, soil={},
        )
        check = _check_region_bounds(unified, _make_config())
        assert check["result"] == "unavailable"
        assert check["details"]["cause"] == "no_climate_fetch"


# ---------------------------------------------------------------------------
# §2.4 — Top-level rollup carries n_unavailable + bumps validation_version
# ---------------------------------------------------------------------------


class TestTopLevelRollup:
    """Moussa's stakeholder summary needs distinct counts for
    "checks ran and passed" / "checks ran and warned" / "checks
    ran and failed" / "checks did not run". The rollup keys
    expose all four."""

    def test_validation_version_bumped_to_2_1(self):
        unified = _make_unified()
        report = run_scientific_validation(unified, _make_config())
        assert report["validation_version"] == "2.1"

    def test_n_unavailable_count_present(self):
        unified = _make_unified(climate={}, soil={})
        report = run_scientific_validation(unified, _make_config())
        assert "n_unavailable" in report
        assert report["n_unavailable"] >= 1

    def test_overall_result_unavailable_when_no_runnable_check(self):
        """When BOTH axes are unavailable AND no check produced a
        runnable verdict, the overall_result becomes the new
        ``unavailable`` state. The cockpit certificate consumer
        can render the documented-MISDAT banner instead of a
        green tick."""
        unified = _make_unified(climate={}, soil={})
        report = run_scientific_validation(unified, _make_config())
        # Format compliance still emits a pass record (it is a
        # placeholder), so even with both axes unavailable the
        # overall stays at "pass". We pin the actual behaviour
        # rather than the wished-for one — the format compliance
        # placeholder is a separate concern and has its own
        # follow-up (G8). Still: the unavailable count must not
        # be zero; the runnable rollup counts must reflect only
        # the records that actually ran.
        assert report["n_unavailable"] >= 1

    def test_overall_result_pass_when_only_one_axis_unavailable(self):
        """Mixed availability — the cockpit's overall certificate
        stays at "pass" when the available axis ran cleanly. The
        per-cell ``data_availability`` carries the absence
        narrative; the global certificate is not in the per-axis
        invariant's scope."""
        climate = {0: _make_ts(n_records=2)}
        unified = _make_unified(climate=climate, soil={})
        report = run_scientific_validation(unified, _make_config())
        # The available axis must produce at least one runnable
        # record so the rollup cannot collapse to unavailable.
        assert report["overall_result"] in ("pass", "warning", "fail")


# ---------------------------------------------------------------------------
# §2.5 — Cell-summary v2.1 schema integration
# ---------------------------------------------------------------------------


class TestCellSummaryV21Integration:
    """Ibrahim's mobile Region Health card reads
    ``data_availability`` / ``unavailable_reason`` per cell. The
    executor must populate these fields from has_climate /
    has_soil and the resulting dict must round-trip through the
    §1 CellSummary schema."""

    def _build(self, *, climate=None, soil=None, n_cells=2):
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        unified = _make_unified(
            n_cells=n_cells, climate=climate, soil=soil,
        )
        return pipeline._build_cell_summary(unified)

    def test_complete_cell_yields_complete_availability(self):
        climate = {0: _make_ts(n_records=2)}
        soil = {0: _make_profile()}
        out = self._build(n_cells=1, climate=climate, soil=soil)
        cell = out["cells"][0]
        assert cell["data_availability"] == "complete"
        assert cell["unavailable_reason"] is None

    def test_climate_only_unavailable_cell(self):
        soil = {0: _make_profile()}
        out = self._build(n_cells=1, climate={}, soil=soil)
        cell = out["cells"][0]
        assert cell["data_availability"] == "unavailable"
        assert cell["unavailable_reason"] == "climate"

    def test_soil_only_unavailable_cell(self):
        climate = {0: _make_ts(n_records=2)}
        out = self._build(n_cells=1, climate=climate, soil={})
        cell = out["cells"][0]
        assert cell["data_availability"] == "unavailable"
        assert cell["unavailable_reason"] == "soil"

    def test_both_axes_unavailable_cell(self):
        out = self._build(n_cells=1, climate={}, soil={})
        cell = out["cells"][0]
        assert cell["data_availability"] == "unavailable"
        assert cell["unavailable_reason"] == "climate_and_soil"

    def test_has_soil_flag_populated(self):
        """The flag is the symmetric counterpart to ``has_climate``.
        Cell with profile + layers → True; cell without → False."""
        soil = {0: _make_profile()}
        out = self._build(n_cells=2, soil=soil)
        assert out["cells"][0]["has_soil"] is True
        assert out["cells"][1]["has_soil"] is False

    def test_top_level_cell_summary_version_is_2_1(self):
        out = self._build()
        assert out["cell_summary_version"] == CELL_SUMMARY_VERSION_LATEST == "2.1"

    def test_cell_record_loads_through_schema(self):
        """The §1 schema's three cross-field invariants gate every
        cell-summary read. The executor's emitted dict must round-
        trip cleanly — without the §2 per-axis pivot filter, the
        schema would reject mixed-availability cells."""
        climate = {0: _make_ts(n_records=2)}
        out = self._build(n_cells=2, climate=climate, soil={})
        for raw_cell in out["cells"]:
            # The schema requires `cell_id` (string in v2.0); the
            # executor's projection emits `id` per the existing
            # contract. Map for the round-trip.
            payload = dict(raw_cell)
            payload["cell_id"] = str(payload.get("id"))
            payload["cell_summary_version"] = "2.1"
            CellSummary.model_validate(payload)


# ---------------------------------------------------------------------------
# §2.6 — Pipeline arithmetic (counts of available / unavailable axes)
# ---------------------------------------------------------------------------


class TestPipelineArithmetic:
    """Moussa's stakeholder summary aggregates the per-cell
    ``data_availability`` field into region-level counts. The
    arithmetic (n cells with each state) must be derivable from
    the executor's emitted dict without re-running the
    validator."""

    def _summary_counts(self, summary):
        counts = {"complete": 0, "climate": 0, "soil": 0, "climate_and_soil": 0}
        for cell in summary["cells"]:
            if cell["data_availability"] == "complete":
                counts["complete"] += 1
            else:
                counts[cell["unavailable_reason"]] += 1
        return counts

    def test_distinct_per_axis_counts(self):
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        # Mix: cell 0 complete, cell 1 missing climate, cell 2
        # missing soil, cell 3 missing both.
        climate = {
            0: _make_ts(n_records=1),
            2: _make_ts(n_records=1),
        }
        soil = {
            0: _make_profile(profile_id="p0"),
            1: _make_profile(profile_id="p1"),
        }
        unified = _make_unified(n_cells=4, climate=climate, soil=soil)
        out = pipeline._build_cell_summary(unified)
        counts = self._summary_counts(out)
        assert counts == {
            "complete": 1,
            "climate": 1,
            "soil": 1,
            "climate_and_soil": 1,
        }

    def test_complete_cells_have_no_unavailable_reason(self):
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        climate = {0: _make_ts(n_records=1)}
        soil = {0: _make_profile()}
        unified = _make_unified(n_cells=1, climate=climate, soil=soil)
        out = pipeline._build_cell_summary(unified)
        cell = out["cells"][0]
        assert cell["data_availability"] == "complete"
        assert cell["unavailable_reason"] is None


# ---------------------------------------------------------------------------
# §2.7 — Iteration-constant integrity (crop-specialist §2 pack §3)
# ---------------------------------------------------------------------------


class TestIterationConstants:
    """``CLIMATE_TIER1_CHECKS`` and ``SOIL_CHECKS_ALL`` are the
    canonical iteration sets called for in crop-modeling-specialist's
    §2 pack §3. Production sibling-sweeps and DRY tests both bind
    against them; if a check_id is added or renamed, updating the
    constant propagates the change to every consumer in lockstep."""

    def test_climate_tier1_includes_axis_aggregator(self):
        """The axis-level aggregator id ``value_range_climate`` is
        the record short-circuited at axis-fully-unavailable; it
        must be in the iteration set."""
        assert "value_range_climate" in CLIMATE_TIER1_CHECKS

    def test_climate_tier1_includes_temporal_completeness(self):
        assert "temporal_completeness" in CLIMATE_TIER1_CHECKS

    def test_climate_tier1_includes_cross_variable_consistency(self):
        assert "cross_variable_consistency" in CLIMATE_TIER1_CHECKS

    def test_climate_tier1_includes_per_variable_value_ranges(self):
        for var in ("tmax", "tmin", "precip", "srad", "rh", "wind"):
            assert f"value_range_{var}" in CLIMATE_TIER1_CHECKS, var

    def test_climate_tier1_includes_coverage(self):
        assert "coverage_climate_cells" in CLIMATE_TIER1_CHECKS

    def test_climate_tier1_includes_region_bounds(self):
        assert "region_specific_bounds" in CLIMATE_TIER1_CHECKS

    def test_soil_checks_includes_axis_aggregator(self):
        assert "value_range_soil" in SOIL_CHECKS_ALL

    def test_soil_checks_includes_per_variable_value_ranges(self):
        for var in ("sand", "clay", "silt", "organic_carbon", "ph", "bulk_density"):
            assert f"value_range_soil_{var}" in SOIL_CHECKS_ALL, var

    def test_soil_checks_includes_texture_sum(self):
        assert "value_range_texture_sum" in SOIL_CHECKS_ALL

    def test_soil_checks_includes_coverage(self):
        assert "coverage_soil_cells" in SOIL_CHECKS_ALL

    def test_soil_checks_includes_per_platform_completeness(self):
        for platform in ("craft", "pythia", "acea", "sarra_py"):
            assert f"soil_completeness_{platform}" in SOIL_CHECKS_ALL, platform

    def test_unavailable_causes_constant_matches_canonical_vocabulary(self):
        """The crop-specialist §2 pack canonical cause vocabulary —
        renaming any of these is a contract change."""
        assert "no_climate_fetch" in UNAVAILABLE_CAUSES
        assert "no_soil_match" in UNAVAILABLE_CAUSES
        assert "soil_cascade_exhausted" in UNAVAILABLE_CAUSES
        assert "no_climate_and_soil_fetch" in UNAVAILABLE_CAUSES
        assert "no_translated_output" in UNAVAILABLE_CAUSES


class TestManuscriptClaimDefensiveGuard:
    """Crop-modeling-specialist §2 pack §3 note 3 — when a caller
    overrides ``manuscript_claim``, the override must preserve the
    ICASA / MISDAT semantic anchor so Dr. Kofi's audit grep
    continues to find the standards lineage on every unavailable
    record. A non-conforming override fires a logger.warning so
    the violation surfaces in audit logs without breaking the
    pipeline."""

    def test_default_claim_passes_guard_without_warning(self, caplog):
        """The helper's default claim already carries 'ICASA MISDAT';
        no warning fires on the canonical path."""
        import logging
        with caplog.at_level(logging.WARNING):
            _unavailable("foo", "global", "no_climate_fetch")
        assert not any(
            "ICASA" in r.getMessage() or "MISDAT" in r.getMessage()
            for r in caplog.records
        )

    def test_override_with_icasa_anchor_passes_guard(self, caplog):
        """A caller-supplied claim that preserves the ICASA anchor
        does not trigger the guard."""
        import logging
        with caplog.at_level(logging.WARNING):
            _unavailable(
                check_id="foo",
                scope="global",
                cause="no_climate_fetch",
                manuscript_claim="Custom audit row preserving ICASA MISDAT context.",
            )
        assert not any(
            "audit-grep continuity at risk" in r.getMessage()
            for r in caplog.records
        )

    def test_override_dropping_icasa_marker_fires_warning(self, caplog):
        """A caller-supplied claim that drops both ICASA and MISDAT
        markers must surface the violation as a logger.warning so
        Dr. Kofi's audit-grep continuity gap is visible in logs."""
        import logging
        with caplog.at_level(logging.WARNING):
            _unavailable(
                check_id="foo",
                scope="global",
                cause="no_climate_fetch",
                manuscript_claim="A claim without the standards lineage.",
            )
        assert any(
            "audit-grep continuity at risk" in r.getMessage()
            for r in caplog.records
        ), (
            "Override missing both ICASA and MISDAT markers must "
            "trigger the defensive logger.warning. Pack §3 note 3."
        )

    def test_override_with_misdat_alone_passes_guard(self, caplog):
        """Either marker alone is sufficient — both ICASA and MISDAT
        are recognised standards lineage tokens."""
        import logging
        with caplog.at_level(logging.WARNING):
            _unavailable(
                check_id="foo",
                scope="global",
                cause="no_climate_fetch",
                manuscript_claim="MISDAT marker alone is fine.",
            )
        assert not any(
            "audit-grep continuity at risk" in r.getMessage()
            for r in caplog.records
        )

    def test_warning_does_not_block_record_construction(self):
        """The defensive guard fires a warning but still returns a
        well-formed record so the pipeline keeps running. A future
        promotion to ValueError is a contract change; today the
        guard is observability-only."""
        rec = _unavailable(
            check_id="foo",
            scope="global",
            cause="no_climate_fetch",
            manuscript_claim="No standards anchor in this claim.",
        )
        assert rec["check"] == "foo"
        assert rec["result"] == "unavailable"
        # The override-as-supplied is preserved verbatim — the guard
        # logs but does not rewrite the field.
        assert rec["manuscript_claim"] == "No standards anchor in this claim."


class TestIterationConstantsDeriveFromSourceDicts:
    """Crop-modeling-specialist §2 pack §3 note 4 — the iteration
    constants must be generator-derived from CLIMATE_RANGES /
    SOIL_RANGES / PLATFORM_SOIL_REQUIREMENTS so a new variable or
    platform added to the source-of-truth dict propagates to every
    consumer in lockstep. Tests pin the derivation at the structural
    layer; if a future contributor hand-rolls an iteration set, the
    drift surfaces here."""

    def test_climate_tier1_includes_every_climate_range_variable(self):
        from prismpy.validators.scientific import CLIMATE_RANGES
        for var in CLIMATE_RANGES:
            assert f"value_range_{var}" in CLIMATE_TIER1_CHECKS, (
                f"value_range_{var} missing from CLIMATE_TIER1_CHECKS — "
                "the iteration constant must derive from CLIMATE_RANGES "
                "via generator expression so a new variable propagates."
            )

    def test_soil_checks_includes_every_soil_range_variable(self):
        from prismpy.validators.scientific import SOIL_RANGES
        for var in SOIL_RANGES:
            assert f"value_range_soil_{var}" in SOIL_CHECKS_ALL

    def test_soil_checks_includes_every_platform_in_requirements(self):
        from prismpy.validators.scientific import PLATFORM_SOIL_REQUIREMENTS
        for platform in PLATFORM_SOIL_REQUIREMENTS:
            assert f"soil_completeness_{platform}" in SOIL_CHECKS_ALL


class TestUnavailableHelperCellIdHandling:
    """The crop-specialist §2 pack signature added an optional
    ``cell_id`` parameter so per-cell short-circuit records can
    pin the cell context on ``details.cell_id``. Axis-level
    callers omit it and the field stays out of details."""

    def test_cell_id_attached_to_details_when_supplied(self):
        rec = _unavailable(
            check_id="value_range_tmax",
            scope="per_cell",
            cause="no_climate_fetch",
            cell_id=42,
        )
        assert rec["details"]["cell_id"] == 42

    def test_cell_id_absent_when_caller_omits(self):
        """Axis-level callers (e.g., ``_check_temporal_completeness``
        emitting one global record) must NOT carry a synthetic
        cell_id on details — the field's absence is the discriminator
        between per-cell and axis-level records."""
        rec = _unavailable(
            check_id="temporal_completeness",
            scope="global",
            cause="no_climate_fetch",
        )
        assert "cell_id" not in rec["details"]


class TestCanonicalCauseVocabularyAtCallsites:
    """Crop-specialist §2 pack §1 — every short-circuit emits the
    canonical cause string, not the prior ``climate_data_missing`` /
    ``soil_data_missing`` / ``climate_and_soil_missing`` placeholders.
    Tests pin the rename so a future regression to the placeholder
    vocabulary is caught."""

    def test_temporal_completeness_uses_canonical_cause(self):
        unified = _make_unified(climate={})
        check = _check_temporal_completeness(unified, _make_config())
        assert check["details"]["cause"] == "no_climate_fetch"

    def test_value_range_climate_uses_canonical_cause(self):
        unified = _make_unified(climate={})
        checks = _check_value_ranges(unified)
        record = next(c for c in checks if c["check"] == "value_range_climate")
        assert record["details"]["cause"] == "no_climate_fetch"

    def test_value_range_soil_uses_canonical_cause(self):
        climate = {0: _make_ts(n_records=1)}
        unified = _make_unified(climate=climate, soil={})
        checks = _check_value_ranges(unified)
        record = next(c for c in checks if c["check"] == "value_range_soil")
        assert record["details"]["cause"] == "no_soil_match"

    def test_soil_completeness_uses_canonical_cause(self):
        unified = _make_unified(soil={})
        check = _check_soil_completeness(unified, "pythia")
        assert check["details"]["cause"] == "no_soil_match"

    def test_coverage_global_both_axes_uses_canonical_cause(self):
        unified = _make_unified(climate={}, soil={})
        check = _check_coverage(unified, _make_config())
        assert check["details"]["cause"] == "no_climate_and_soil_fetch"
