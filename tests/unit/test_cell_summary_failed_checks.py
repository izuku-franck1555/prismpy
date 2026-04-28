"""V2-22c-PRE.1 — cell_summary_version + soil_class + structured failed_checks.

Covers ACs PRE.1.1 (cell_summary_version), PRE.1.6 (soil_class via existing
SoilProfile.surface_texture), and is the home for PRE.1.2 (failed_checks
structured shape) tests as those land in subsequent commits.

Aligned with the evaluator's pre-committed Gate B criteria (`prismweb/.local/
v2-22c-r5-evidence/evaluator/V2-22c-VERIFICATION-STRATEGY.md` §2 + §5 +
PRE-CONTRACT.md §12 tightenings):

* PRE.1.1 — string equality `"2.0"`, top-level placement, single emit
  site (sibling-sweep grep against `prismpy/src/`).
* PRE.1.6 — `soil_class` MUST exist on every cell. Valid profile → string.
  Empty layers / no profile → JSON `null` (NOT key elision). Uniform
  consumer interface so the cockpit Veto #4 preflight sees a deterministic
  string-vs-null instead of a tri-state with `undefined`.

The tests bypass TranslationPipeline.__init__ via __new__ — `_build_cell_summary`
is a pure data-projection method that doesn't touch any instance state on
self, so a synthetic instance is sufficient and avoids pulling in the
ProjectConfig + ProvenanceTracker dependency chain that init requires.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from prismpy.models.region import Region, BoundingBox
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.translators.base import UnifiedData


_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "prismpy"


def _make_pipeline():
    """Bypass __init__ — _build_cell_summary doesn't touch self.<>."""
    return TranslationPipeline.__new__(TranslationPipeline)


def _make_region():
    return Region(
        name="test", country="test", country_iso3="TST",
        bounds=BoundingBox(minx=0.0, miny=0.0, maxx=1.0, maxy=1.0),
    )


def _make_grid(n_cells: int = 2) -> SpatialGrid:
    cells = [
        GridCell(cell_id=i, lat=0.5 + i * 0.01, lon=0.5 + i * 0.01,
                 row=0, col=i, resolution="5arcmin")
        for i in range(n_cells)
    ]
    return SpatialGrid(
        bounds=BoundingBox(minx=0.0, miny=0.0, maxx=1.0, maxy=1.0),
        resolution="5arcmin",
        cells=cells,
    )


def _make_soil_profile(*, profile_id: str, sand: float, clay: float,
                       source: str = "iSDA",
                       with_layers: bool = True) -> SoilProfile:
    layers = []
    if with_layers:
        layers.append(SoilLayer(
            depth_top=0.0, depth_bottom=0.2,
            sand=sand, clay=clay,
        ))
    return SoilProfile(
        profile_id=profile_id, lat=0.5, lon=0.5,
        source=source, layers=layers,
    )


def _make_full_unified(n_cells: int = 2):
    """G7 §2 — construct a unified-data fixture where every cell has
    BOTH climate records and a soil profile. The §2 per-axis pivot
    filter drops failed-check entries from cells whose
    ``data_availability == "unavailable"``; tests that exercise the
    pivot under normal operation need ``complete`` cells so the
    filter does not mask the pivot behaviour they actually want to
    verify. Tests that specifically test the §2 short-circuit
    construct their own stripped-down fixtures inline."""
    from datetime import date as _date
    grid = _make_grid(n_cells=n_cells)
    climate = {}
    soil = {}
    for cid in range(n_cells):
        climate[cid] = ClimateTimeSeries(
            location_id=cid, lat=0.5, lon=0.5, source="TEST",
            records=[ClimateRecord(
                date=_date(2015, 1, 1), tmax=30.0, tmin=20.0,
                precip=2.0, srad=20.0,
            )],
        )
        soil[cid] = _make_soil_profile(
            profile_id=f"p{cid}", sand=40.0, clay=30.0,
        )
    return UnifiedData(
        region=_make_region(), grid=grid, climate=climate, soil=soil,
    )


class TestCellSummaryVersion:
    """V2-22c-PRE.1.1 (D5/D7) — cell_summary_version field on the
    top-level dict returned by _build_cell_summary."""

    def test_cell_summary_version_field_present(self):
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid())
        out = pipeline._build_cell_summary(unified)
        assert "cell_summary_version" in out, (
            "cell_summary_version is the loader-fallback signal at "
            "prismweb/core/views.py::_load_cell_summary; missing key "
            "would force every consumer into the pre-PRE.1 synthesis "
            "branch even on post-PRE.1 fixtures."
        )

    def test_cell_summary_version_value_is_2_1(self):
        """G7 §2 — version bumped to "2.1" once the executor populates
        the new ``data_availability`` / ``unavailable_reason`` fields
        and short-circuits the per-cell pivot per axis. The previous
        "2.0" pin remains a valid input shape for the schema (round-
        trip compat is covered in tests/test_cell_summary_schema.py),
        but the producer side now stamps the latest version."""
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid())
        out = pipeline._build_cell_summary(unified)
        assert out["cell_summary_version"] == "2.1"

    def test_cell_summary_version_present_alongside_existing_top_level_keys(self):
        """Regression guard — the new field must not displace existing
        top-level keys that consumers already read."""
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid())
        out = pipeline._build_cell_summary(unified)
        for key in ("n_cells", "resolution", "cells"):
            assert key in out, f"top-level key {key!r} regressed"

    def test_cell_summary_version_at_top_level_not_nested(self):
        """Evaluator §12.1 numeric criterion — field MUST be at the
        top level (sibling of `n_cells` / `resolution` / `cells`),
        NOT nested inside `meta` or any sub-object. The cockpit
        loader-fallback at `prismweb/core/views.py::_load_cell_summary`
        reads the top-level key directly; a nested form would force
        every consumer into the pre-PRE.1 synthesis branch."""
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid())
        out = pipeline._build_cell_summary(unified)
        # Top-level presence already covered above. Belt-and-braces:
        # confirm the field is NOT shadowed inside any nested dict.
        for nest_key in ("meta", "metadata", "header", "schema"):
            nested = out.get(nest_key)
            if isinstance(nested, dict):
                assert "cell_summary_version" not in nested, (
                    f"cell_summary_version leaked into nested {nest_key!r} "
                    "object — must live at top level only."
                )

    def test_single_build_cell_summary_emit_site_in_src(self):
        """Evaluator §12.1 sibling-sweep — only one site in
        `prismpy/src/` should construct the cell-summary dict (the
        canonical `_build_cell_summary` definition). A secondary
        emit site = contract regression because it would diverge
        from this method's schema discipline (e.g., a copy-paste
        helper that forgets the new `cell_summary_version` /
        `soil_class` fields).

        Pattern matched: a `def _build_cell_summary` definition.
        Module-level helpers and TranslationPipeline references in
        tests / docs are NOT counted (they're not emit sites). The
        method definition is canonical.
        """
        pat = re.compile(r"^\s*def\s+_build_cell_summary\b")
        emit_sites = []
        for py_file in _SRC_ROOT.rglob("*.py"):
            for lineno, line in enumerate(
                py_file.read_text().splitlines(), start=1,
            ):
                if pat.search(line):
                    emit_sites.append((str(py_file), lineno))
        assert len(emit_sites) == 1, (
            f"Expected single _build_cell_summary definition in "
            f"prismpy/src/; found {len(emit_sites)}: {emit_sites!r}. "
            "Multiple sites would diverge on schema additions; "
            "contract treats this as a regression."
        )


class TestSoilClassField:
    """V2-22c-PRE.1.6 (D26) — per-cell `soil_class` field via existing
    SoilProfile.surface_texture USDA classifier."""

    def test_soil_class_emitted_when_surface_layer_present(self):
        pipeline = _make_pipeline()
        # Loamy Sand: sand=80, clay=10 (per _get_texture_class thresholds)
        soil = {
            0: _make_soil_profile(profile_id="p0", sand=80.0, clay=10.0),
        }
        unified = UnifiedData(
            region=_make_region(), grid=_make_grid(n_cells=1), soil=soil,
        )
        out = pipeline._build_cell_summary(unified)
        cell = out["cells"][0]
        assert "soil_class" in cell
        assert cell["soil_class"] == "Loamy Sand"

    def test_soil_class_is_null_when_layers_empty(self):
        """Evaluator §2 numeric criterion — empty-profile edge surfaces
        as `None` (JSON null), NOT elision. Cockpit Veto #4 preflight
        depends on a deterministic string-or-null read; `undefined`
        from key elision would silently disable the cross-class block
        on no-soil cells."""
        pipeline = _make_pipeline()
        soil = {
            0: _make_soil_profile(
                profile_id="p0", sand=80.0, clay=10.0, with_layers=False,
            ),
        }
        unified = UnifiedData(
            region=_make_region(), grid=_make_grid(n_cells=1), soil=soil,
        )
        out = pipeline._build_cell_summary(unified)
        cell = out["cells"][0]
        assert "soil_class" in cell, (
            "soil_class key MUST exist on every cell per evaluator "
            "§12 / §2 binding; elision broke the cockpit JS contract."
        )
        assert cell["soil_class"] is None

    def test_soil_class_is_null_when_no_profile(self):
        """Same null-not-elide discipline when no SoilProfile exists
        for the cell at all — uniform consumer interface across all
        three paths (valid profile / empty layers / no profile)."""
        pipeline = _make_pipeline()
        unified = UnifiedData(
            region=_make_region(), grid=_make_grid(n_cells=1), soil={},
        )
        out = pipeline._build_cell_summary(unified)
        cell = out["cells"][0]
        assert "soil_class" in cell
        assert cell["soil_class"] is None
        assert cell["soil_source"] == "none"

    def test_soil_class_field_present_on_every_cell(self):
        """Evaluator §2 binding: 'Field MUST exist on every cell
        (even null).' This pins the schema invariant across mixed
        cell populations — every cell, regardless of profile status,
        carries the key."""
        pipeline = _make_pipeline()
        # Mixed: cell 0 has a valid profile, cell 1 has an empty
        # profile (no layers), cell 2 has no profile at all.
        soil = {
            0: _make_soil_profile(profile_id="p0", sand=80.0, clay=10.0),
            1: _make_soil_profile(
                profile_id="p1", sand=10.0, clay=80.0, with_layers=False,
            ),
            # cell 2 absent from soil dict on purpose
        }
        unified = UnifiedData(
            region=_make_region(), grid=_make_grid(n_cells=3), soil=soil,
        )
        out = pipeline._build_cell_summary(unified)
        for cell in out["cells"]:
            assert "soil_class" in cell, (
                f"cell {cell['id']} missing soil_class — schema "
                "invariant requires the key on every cell."
            )

    def test_soil_class_distinct_classes_for_distinct_textures(self):
        """Sanity check on the classifier mapping — Sand vs Clay vs Loam
        should NOT collide. Anchors the cockpit chip-strip rendering."""
        pipeline = _make_pipeline()
        soil = {
            0: _make_soil_profile(profile_id="p0", sand=90.0, clay=5.0),   # Sand
            1: _make_soil_profile(profile_id="p1", sand=20.0, clay=50.0),  # Clay
        }
        unified = UnifiedData(
            region=_make_region(), grid=_make_grid(n_cells=2), soil=soil,
        )
        out = pipeline._build_cell_summary(unified)
        classes = [cell.get("soil_class") for cell in out["cells"]]
        assert classes[0] != classes[1], (
            f"distinct sand/clay yielded same soil_class={classes[0]!r}; "
            "regression in SoilProfile._get_texture_class wiring."
        )
        assert classes[0] == "Sand"
        assert classes[1] == "Clay"


def _validation_report(*checks):
    """Wrap a list of synthetic check dicts in the validation_report
    envelope shape (`{validation_version, checks: [...]}`). Tests
    inject these straight into `_build_cell_summary` to exercise
    the PRE.1.2 / PRE.1.8 pivot without spinning up the full
    validate stage."""
    return {"validation_version": "2.0", "checks": list(checks)}


class TestFailedChecksEmptyByDefault:
    """V2-22c-PRE.1.2 evaluator §12.2 binding — cells with no
    failures emit `failed_checks: []` (empty list, NOT absent key).
    Without the empty-list guarantee, a cockpit JS that uses
    `cell.failed_checks.length === 0` for the pass-state branch
    breaks on cells that never had a chance to fail."""

    def test_no_validation_report_yields_empty_failed_checks_per_cell(self):
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid())
        out = pipeline._build_cell_summary(unified)
        for cell in out["cells"]:
            assert cell["failed_checks"] == [], (
                f"cell {cell['id']} expected failed_checks=[]; "
                f"got {cell['failed_checks']!r}"
            )

    def test_no_validation_report_yields_empty_top_level_details(self):
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid())
        out = pipeline._build_cell_summary(unified)
        assert out["cell_failed_check_details"] == []

    def test_passing_check_does_not_pivot_into_failed_checks(self):
        """A `result='pass'` check must NOT appear on any cell — only
        fail / warning land in failed_checks."""
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid())
        report = _validation_report({
            "check": "value_range_tmax",
            "scope": "per_record",
            "result": "pass",
            "details": {"affected_cells": [0, 1]},
        })
        out = pipeline._build_cell_summary(unified, report)
        for cell in out["cells"]:
            assert cell["failed_checks"] == []


class TestFailedChecksStructuredShape:
    """V2-22c-PRE.1.2 evaluator §12.2 — every entry has the exact
    3-key shape `{check_id, result, category}`. No extras, no
    missing, no bare check_id strings."""

    def test_per_cell_check_pivots_into_failed_checks(self):
        pipeline = _make_pipeline()
        # G7 §2 — full-availability fixture so the pivot's per-axis
        # filter does not drop the climate-axis ``value_range_tmax``
        # entry. The original bare-grid construction left every cell
        # at ``data_availability="unavailable"`` with reason
        # ``climate_and_soil``, so the §1 invariant 3 filter would
        # silently drop the test's expected entry.
        unified = _make_full_unified(n_cells=2)
        report = _validation_report({
            "check": "value_range_tmax",
            "scope": "per_record",
            "result": "fail",
            "details": {"affected_cells": [0]},
        })
        out = pipeline._build_cell_summary(unified, report)
        assert len(out["cells"][0]["failed_checks"]) == 1
        assert out["cells"][1]["failed_checks"] == []

    def test_failed_check_entry_has_exactly_three_keys(self):
        pipeline = _make_pipeline()
        unified = _make_full_unified(n_cells=1)
        report = _validation_report({
            "check": "temporal_completeness",
            "scope": "per_cell",
            "result": "warning",
            "details": {"affected_cells": [0]},
        })
        out = pipeline._build_cell_summary(unified, report)
        entry = out["cells"][0]["failed_checks"][0]
        assert set(entry.keys()) == {"check_id", "result", "category"}, (
            f"entry must have exactly 3 keys; got {set(entry.keys())!r}"
        )
        assert entry["check_id"] == "temporal_completeness"
        assert entry["result"] == "warning"
        assert entry["category"] == "temporal"

    def test_result_only_fail_or_warning(self):
        """Evaluator §12.2 — `result ∈ {"fail", "warning"}` literal.
        The pivot's pre-filter excludes `pass` and `info`."""
        pipeline = _make_pipeline()
        unified = _make_full_unified(n_cells=1)
        report = _validation_report(
            {
                "check": "value_range_tmax",
                "scope": "per_record",
                "result": "info",
                "details": {"affected_cells": [0]},
            },
            {
                "check": "value_range_tmin",
                "scope": "per_record",
                "result": "fail",
                "details": {"affected_cells": [0]},
            },
        )
        out = pipeline._build_cell_summary(unified, report)
        entries = out["cells"][0]["failed_checks"]
        assert len(entries) == 1
        assert entries[0]["check_id"] == "value_range_tmin"
        assert entries[0]["result"] == "fail"


class TestFailedChecksRegionScopeExcluded:
    """V2-22c-PRE.1.2 evaluator §12.2 — region-scoped checks
    (scope='global') MUST NOT appear in per-cell failed_checks.
    They live in the banner per Appendix H two-zone rendering;
    leaking them into per-cell would double-render the same
    failure on every cell."""

    def test_global_scope_check_excluded_from_per_cell_pivot(self):
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid(n_cells=1))
        report = _validation_report(
            {
                "check": "format_compliance",
                "scope": "global",
                "result": "fail",
                "details": {"affected_cells": [0]},
            },
            {
                "check": "spatial_temporal_coverage",
                "scope": "global",
                "result": "warning",
                "details": {"affected_cells": [0]},
            },
        )
        out = pipeline._build_cell_summary(unified, report)
        assert out["cells"][0]["failed_checks"] == [], (
            "global-scope checks leaked into per-cell failed_checks; "
            "Appendix H two-zone rendering is broken — banner + map "
            "would render the same failure twice."
        )

    def test_unknown_check_id_prefix_does_not_pivot(self):
        """A per-cell-scoped check whose check_id doesn't match a
        known `_CATEGORY_FROM_PREFIX` prefix is skipped. Surfaces
        as a sibling-sweep finding at evaluator §12.2 — the
        validator should never emit such a check_id."""
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid(n_cells=1))
        report = _validation_report({
            "check": "future_unknown_check_id_xyz",
            "scope": "per_cell",
            "result": "fail",
            "details": {"affected_cells": [0]},
        })
        out = pipeline._build_cell_summary(unified, report)
        assert out["cells"][0]["failed_checks"] == []


class TestFailedChecksTupleAffectedCells:
    """PRE.1.4/1.5 emit `(cell_id, layer_idx)` tuples; PRE.1.7 emits
    bare cell_ids; the pivot must duck-type-extract the cell_id from
    either shape."""

    def test_tuple_affected_cells_extract_cell_id(self):
        pipeline = _make_pipeline()
        unified = _make_full_unified(n_cells=2)
        report = _validation_report({
            "check": "value_range_soil_clay",
            "scope": "per_layer",
            "result": "warning",
            "details": {"affected_cells": [(0, 0), (1, 2)]},
        })
        out = pipeline._build_cell_summary(unified, report)
        assert len(out["cells"][0]["failed_checks"]) == 1
        assert len(out["cells"][1]["failed_checks"]) == 1


class TestCellFailedCheckDetailsFlatten:
    """V2-22c-PRE.1.8 (D35) — top-level `cell_failed_check_details`
    array carries per-violation context (cell, check_id, result,
    category, layer_idx, variable, value, unit, bounds). Cockpit
    drawer reads this directly without joining back to cells."""

    def test_violation_details_flattened_into_top_level(self):
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid(n_cells=1))
        report = _validation_report({
            "check": "value_range_soil_clay",
            "scope": "per_layer",
            "result": "warning",
            "details": {
                "affected_cells": [(0, 2)],
                "violation_details": [{
                    "cell_id": 0, "layer_idx": 2,
                    "variable": "clay", "value": 89.0,
                    "unit": "%", "bounds": [0, 100],
                }],
            },
        })
        out = pipeline._build_cell_summary(unified, report)
        details = out["cell_failed_check_details"]
        assert len(details) == 1
        d = details[0]
        assert d["cell_id"] == 0
        assert d["check_id"] == "value_range_soil_clay"
        assert d["result"] == "warning"
        assert d["category"] == "value_range"
        assert d["layer_idx"] == 2
        assert d["variable"] == "clay"
        assert d["value"] == pytest.approx(89.0)
        assert d["unit"] == "%"
        assert d["bounds"] == [0, 100]

    def test_violation_details_sorted_by_cell_then_check_then_layer(self):
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid(n_cells=3))
        report = _validation_report(
            {
                "check": "value_range_soil_clay",
                "scope": "per_layer",
                "result": "warning",
                "details": {
                    "affected_cells": [(2, 0), (0, 0), (1, 0)],
                    "violation_details": [
                        {"cell_id": 2, "layer_idx": 0, "variable": "clay",
                         "value": 89.0, "unit": "%", "bounds": [0, 100]},
                        {"cell_id": 0, "layer_idx": 0, "variable": "clay",
                         "value": 88.0, "unit": "%", "bounds": [0, 100]},
                        {"cell_id": 1, "layer_idx": 0, "variable": "clay",
                         "value": 92.0, "unit": "%", "bounds": [0, 100]},
                    ],
                },
            },
        )
        out = pipeline._build_cell_summary(unified, report)
        details = out["cell_failed_check_details"]
        cell_order = [d["cell_id"] for d in details]
        assert cell_order == sorted(cell_order)

    def test_per_cell_failed_checks_sorted_within_cell(self):
        """Within a single cell, multiple failed checks are sorted by
        (check_id, result) so cockpit chip rendering is stable."""
        pipeline = _make_pipeline()
        unified = _make_full_unified(n_cells=1)
        report = _validation_report(
            {
                "check": "temporal_completeness",
                "scope": "per_cell",
                "result": "warning",
                "details": {"affected_cells": [0]},
            },
            {
                "check": "value_range_tmax",
                "scope": "per_record",
                "result": "fail",
                "details": {"affected_cells": [0]},
            },
            {
                "check": "cross_variable_consistency",
                "scope": "per_record",
                "result": "warning",
                "details": {"affected_cells": [0]},
            },
        )
        out = pipeline._build_cell_summary(unified, report)
        check_ids = [e["check_id"] for e in out["cells"][0]["failed_checks"]]
        assert check_ids == sorted(check_ids)


class TestTemporalCompletenessAffectedCells:
    """V2-22c-PRE codex P2 #1 — `temporal_completeness` check
    emits `details.affected_cells` (sorted ASC) so the per-cell
    failed_checks pivot picks up cells with missing dates. Without
    this list, gap cells would silently lose the `temporal_completeness`
    chip on the cockpit Layer 2 chip strip."""

    def test_temporal_completeness_emits_affected_cells_for_gaps(self):
        from datetime import date
        from types import SimpleNamespace
        from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
        from prismpy.validators.scientific import _check_temporal_completeness

        # Build a tiny climate dict: cell 0 has a 2-day gap; cell 1
        # is complete. Expect affected_cells == [0].
        climate = {
            0: ClimateTimeSeries(
                records=[
                    ClimateRecord(
                        date=date(2020, 1, d),
                        tmax=25.0, tmin=15.0, precip=0.0, srad=20.0,
                    )
                    for d in range(1, 4)  # 3 days; expected 5 → gap
                ],
                location_id="0", lat=0.5, lon=0.5, source="test",
            ),
            1: ClimateTimeSeries(
                records=[
                    ClimateRecord(
                        date=date(2020, 1, d),
                        tmax=25.0, tmin=15.0, precip=0.0, srad=20.0,
                    )
                    for d in range(1, 6)  # 5 days; complete
                ],
                location_id="1", lat=0.5, lon=0.5, source="test",
            ),
        }
        unified = UnifiedData(
            region=_make_region(), climate=climate, soil={},
        )
        # `_check_temporal_completeness` reads
        # `config.temporal.{start_year,spinup_years}` plus
        # `config.crop.calendar` (passed to `temporal.get_climate_end_date`).
        # Synth a minimal SimpleNamespace tree that satisfies the
        # attribute path without booting the full ProjectConfig
        # schema (which has many unrelated required fields).
        from datetime import date as _date

        class _FakeTemporal:
            def __init__(self):
                self.start_year = 2020
                self.end_year = 2020
                self.spinup_years = 0

            def get_climate_end_date(self, crop_cal):
                return _date(2020, 1, 5)

        config = SimpleNamespace(
            temporal=_FakeTemporal(),
            crop=SimpleNamespace(calendar=None),
        )
        check = _check_temporal_completeness(unified, config)
        assert "affected_cells" in check["details"]
        assert 0 in check["details"]["affected_cells"]
        assert check["details"]["affected_cells"] == sorted(
            check["details"]["affected_cells"]
        )


class TestSoilCompletenessAffectedCells:
    """V2-22c-PRE codex P2 #2 — `soil_completeness_<platform>` emits
    `details.affected_cells` so the per-cell `failed_checks` pivot
    surfaces incomplete-soil cells correctly. Without this list, the
    pivot would skip the check (sample_missing dict doesn't drive
    the pivot)."""

    def test_soil_completeness_emits_affected_cells_for_incomplete(self):
        from prismpy.validators.scientific import _check_soil_completeness

        # Cell 0 has clay only; required props for craft include
        # sand/clay/silt/organic_carbon/ph/bulk_density → 5 missing.
        # Cell 1 is complete.
        soil = {
            0: SoilProfile(
                profile_id="p0", lat=0.5, lon=0.5, source="iSDA",
                layers=[SoilLayer(
                    depth_top=0, depth_bottom=0.2,
                    sand=40, clay=30, silt=30,
                    # organic_carbon, ph, bulk_density absent
                )],
            ),
            1: SoilProfile(
                profile_id="p1", lat=0.5, lon=0.5, source="iSDA",
                layers=[SoilLayer(
                    depth_top=0, depth_bottom=0.2,
                    sand=40, clay=30, silt=30,
                    organic_carbon=2.0, ph=6.5, bulk_density=1.4,
                )],
            ),
        }
        unified = UnifiedData(
            region=_make_region(), soil=soil, climate={},
        )
        check = _check_soil_completeness(unified, "craft")
        assert "affected_cells" in check["details"]
        # Cell 0 missing properties → in affected_cells.
        assert 0 in check["details"]["affected_cells"]
        # Cell 1 complete → NOT in affected_cells.
        assert 1 not in check["details"]["affected_cells"]
        assert check["details"]["affected_cells"] == sorted(
            check["details"]["affected_cells"]
        )


class TestCoverageCategoryMapping:
    """V2-22c-PRE codex P2 #3 — `coverage_climate_cells` and
    `coverage_soil_cells` (D36) classify under category 'coverage'
    in the validation report's category rollup. Without the mapping
    they fall through to the schema fallback and the cockpit's
    coverage-chip rendering disagrees with the validator's category
    summary."""

    def test_coverage_climate_cells_classified_as_coverage(self):
        from prismpy.validators.scientific import _get_check_category
        assert _get_check_category("coverage_climate_cells") == "coverage"

    def test_coverage_soil_cells_classified_as_coverage(self):
        from prismpy.validators.scientific import _get_check_category
        assert _get_check_category("coverage_soil_cells") == "coverage"


class TestValidationReportEnvelopeExtraction:
    """V2-22c-PRE codex P1 #1 — `_build_cell_summary` accepts the
    nested scientific report (the dict with `checks: [...]`), NOT the
    validation_summary envelope (`{'scientific': {...}, 'post_translate':
    {...}}`). The fix at the executor's package-stage call site
    extracts `validate_result.data['scientific']` before passing in.

    This test pins the read contract: `_build_cell_summary` reads
    `validation_report.get('checks', [])` directly. If a future
    refactor passes the envelope, the per-cell pivot returns empty
    arrays for every cell and the cockpit rendering goes silently
    blank. The fix at executor.py:3024-3027 extracts the nested
    report; the unit test below exercises the call signature
    contract."""

    def test_passing_envelope_yields_empty_failed_checks(self):
        """Negative regression: when the WRONG shape (the envelope)
        is passed, `_build_cell_summary` reads `checks` from the
        envelope (which is absent) and emits empty arrays. The
        executor-side fix extracts `data['scientific']` so the
        method receives the right shape; this test confirms the
        envelope-vs-report distinction matters at the read site."""
        pipeline = _make_pipeline()
        # G7 §2 — full-availability so the §1 invariant 3 filter
        # does not preempt the envelope-shape regression we are
        # testing here.
        unified = _make_full_unified(n_cells=1)
        envelope = {
            "scientific": {
                "validation_version": "2.0",
                "checks": [{
                    "check": "value_range_tmax",
                    "scope": "per_record",
                    "result": "fail",
                    "details": {"affected_cells": [0]},
                }],
            },
            "post_translate": {"checks": []},
        }
        # Passing the envelope (wrong shape) yields empty failed_checks.
        out = pipeline._build_cell_summary(unified, envelope)
        assert out["cells"][0]["failed_checks"] == [], (
            "envelope as `validation_report` produces empty failed_checks "
            "— this is the codex P1 #1 bug; the executor's package "
            "stage must extract `data['scientific']` before passing in"
        )
        # Passing the nested report (correct shape) yields populated.
        out = pipeline._build_cell_summary(unified, envelope["scientific"])
        assert len(out["cells"][0]["failed_checks"]) == 1
        assert out["cells"][0]["failed_checks"][0]["check_id"] == "value_range_tmax"


class TestProjectConfigRemediationSpec:
    """V2-22c-PRE codex P1 #2 — `ProjectConfig` carries an optional
    `remediation_spec: Dict[str, Any]` field. Without this, Pydantic
    v2's model_validate drops the extra input on real cockpit
    submissions and the REMEDIATION stage's getattr-from-config
    silently returns None → server-side Veto #4 enforcement never
    fires on any real run."""

    def test_remediation_spec_field_present_with_default_none(self):
        from prismpy.config.schema import ProjectConfig
        # Field exists on the model.
        assert "remediation_spec" in ProjectConfig.model_fields
        # Default is None — original runs and retries are no-op.
        field = ProjectConfig.model_fields["remediation_spec"]
        assert field.default is None

    def test_remediation_spec_survives_validation(self):
        from prismpy.config.schema import ProjectConfig
        spec = {
            "imputations": [
                {"cell_id": 4, "method": "idw"},
            ],
            "exclusions": {"cells": [], "days": []},
        }
        cfg = ProjectConfig.model_validate({
            "project": {"name": "t", "version": "1"},
            "region": {
                "name": "t", "country": "t", "country_iso3": "TST",
                "boundary": {
                    "source": "gadm", "gadm_level": 2,
                    "gadm_filter_field": "NAME_2",
                    "gadm_filter_value": "T",
                },
            },
            "crop": {
                "name": "maize", "name_short": "MZ",
                "calendar": {
                    "planting_doy": 90, "maturity_doy": 270,
                    "source": "wizard", "reference": "",
                },
            },
            "temporal": {
                "start_year": 2020, "end_year": 2020,
                "spinup_years": 0,
            },
            "remediation_spec": spec,
        })
        assert cfg.remediation_spec == spec, (
            "remediation_spec was dropped by Pydantic — codex P1 #2 "
            "regression. The field MUST be declared on ProjectConfig "
            "so cockpit-bulk-fix re-runs reach the Veto #4 "
            "enforcement instead of taking the no-op branch."
        )
