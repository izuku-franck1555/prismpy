"""Coverage-Honesty Phase A — α (validator) behavioral + structural pins.

Covers the producer-boundary substrate the results page renders from: a
single-source coverage diff, real SARRA-Py per-cell coverage (no silent
info-stub), and a ``remediable_findings`` list built from two derivation
paths — bucket-based for data-bearing checks, and SYNTHESIZED for absent
cells (which carry no per-cell RoutingDecision and would otherwise be
orphaned by the cell-summary pivot).
"""

import inspect
import types
import unittest

from prismpy.validators import scientific
from prismpy.validators.scientific import (
    _bucket_is_remediable,
    _build_remediable_findings,
    _check_coverage_climate_cells,
    _check_coverage_soil_cells,
    _resolve_warning_bucket,
    compute_coverage_diff,
)
from prismpy.validators.post_translate import sarra_py_climate_rasters_readable
from prismpy.warnings.categories import WarningBucket


def _cell(cell_id):
    return types.SimpleNamespace(cell_id=cell_id)


def _grid(cell_ids):
    cells = [_cell(c) for c in cell_ids]
    return types.SimpleNamespace(
        cells=cells, n_cells=len(cells),
        n_rows=len(cells), n_cols=1, resolution=0.5,
    )


def _series(records):
    return types.SimpleNamespace(records=list(records), source="agera5")


def _profile(layers):
    return types.SimpleNamespace(layers=list(layers))


def _ud(*, grid=None, climate=None, soil=None):
    return types.SimpleNamespace(
        grid=grid, climate=climate if climate is not None else {},
        soil=soil if soil is not None else {},
    )


# ── C.2 — compute_coverage_diff single-source helper units ──────────────


class TestComputeCoverageDiff(unittest.TestCase):
    def test_full_coverage_empty_diff(self):
        self.assertEqual(compute_coverage_diff([1, 2, 3], [1, 2, 3]), [])

    def test_missing_cells_returned(self):
        self.assertEqual(compute_coverage_diff([1, 2, 3, 4], [2, 4]), [1, 3])

    def test_order_deterministic(self):
        # Unsorted inputs → sorted output (stable JSON diffs).
        self.assertEqual(
            compute_coverage_diff([5, 1, 9, 3], [9]), [1, 3, 5],
        )

    def test_extra_data_cells_ignored(self):
        # Cells with data but not expected don't appear as missing.
        self.assertEqual(compute_coverage_diff([1, 2], [1, 2, 99]), [])


# ── C.1(a) — ACEA 1-missing-soil → synthesized absent finding (Path B) ──


class TestAbsentSoilSynthesis(unittest.TestCase):
    def test_one_missing_soil_cell_surfaces_as_remediable(self):
        grid = _grid([1, 2, 3])
        soil = {1: _profile([object()]), 2: _profile([object()])}  # cell 3 absent
        check = _check_coverage_soil_cells(_ud(grid=grid, soil=soil))
        self.assertEqual(check["result"], "fail")
        self.assertEqual(check["details"]["affected_cells"], [3])

        findings = _build_remediable_findings([check], None)
        self.assertEqual(len(findings), 1)
        f = findings[0]
        self.assertEqual(f["cell_ids"], [3])
        self.assertTrue(f["remediable"])
        self.assertEqual(f["remediation_kind"], "hwsd_fallback")
        self.assertEqual(f["cause"], "absent")

    def test_absent_climate_cell_routes_to_retry_not_cockpit(self):
        # D2 — climate-absent is a fetch failure (NASA POWER/AgERA5 cover
        # land), so it routes to RETRY and is NOT cockpit-remediable; never
        # interpolated (moral hazard).
        grid = _grid([1, 2])
        climate = {1: _series([object()])}  # cell 2 absent
        check = _check_coverage_climate_cells(_ud(grid=grid, climate=climate))
        findings = _build_remediable_findings([check], None)
        self.assertEqual(findings[0]["cell_ids"], [2])
        self.assertFalse(findings[0]["remediable"])
        self.assertEqual(findings[0]["remediation_kind"], "retry")


# ── C.1(b) — SARRA-Py per-cell climate: real diff, never silent info ────


class TestSarraPyCoverage(unittest.TestCase):
    def _file_based_ud(self, grid):
        # The file-based discriminator is a dict of *_dir keys.
        return _ud(
            grid=grid,
            climate={"rainfall_dir": "/tmp/r", "agera5_dir": "/tmp/a"},
        )

    def test_unreadable_no_sample_is_unverifiable_not_info(self):
        """No sample + rasters not readable → unverifiable, never a silent
        info pass (the #166 regression)."""
        grid = _grid([1, 2, 3])
        check = _check_coverage_climate_cells(
            self._file_based_ud(grid),
            sarra_climate_per_cell=None, sarra_climate_readable=False,
        )
        self.assertEqual(check["result"], "unavailable")
        self.assertEqual(check["details"]["cause"], "coverage_unverifiable")
        self.assertNotEqual(check["result"], "info")

    def test_unreadable_empty_sample_is_unverifiable(self):
        """Empty ``{}`` sample with UNREADABLE rasters → unverifiable: the
        sampler couldn't read any GeoTIFF, so coverage is unknown."""
        grid = _grid([1, 2, 3])
        check = _check_coverage_climate_cells(
            self._file_based_ud(grid),
            sarra_climate_per_cell={}, sarra_climate_readable=False,
        )
        self.assertEqual(check["result"], "unavailable")
        self.assertEqual(check["details"]["cause"], "coverage_unverifiable")
        self.assertEqual(check["details"]["affected_cells"], [])

    def test_readable_empty_sample_is_measured_all_missing(self):
        """Empty ``{}`` sample with READABLE rasters → a MEASURED gap: the
        rasters opened fine but no cell fell on covered data, so every cell
        is missing (fail), NOT unverifiable. (codex F-1 / team-lead arb.)"""
        grid = _grid([1, 2, 3])
        check = _check_coverage_climate_cells(
            self._file_based_ud(grid),
            sarra_climate_per_cell={}, sarra_climate_readable=True,
        )
        self.assertEqual(check["result"], "fail")
        self.assertEqual(check["details"]["affected_cells"], [1, 2, 3])
        self.assertEqual(check["details"]["n_missing"], 3)
        # These measured-absent climate cells reach Path B as a finding, but
        # climate-absent routes to RETRY (not cockpit-remediable, D2).
        findings = _build_remediable_findings([check], None)
        assert findings and findings[0]["remediable"] is False
        self.assertEqual(findings[0]["remediation_kind"], "retry")
        self.assertEqual(findings[0]["cell_ids"], [1, 2, 3])

    def test_partial_sample_yields_real_diff(self):
        grid = _grid([1, 2, 3])
        sample = {
            1: {"rain": [1.0, 2.0]},
            2: {"rain": []},   # empty → no data
            # cell 3 absent from sample entirely
        }
        check = _check_coverage_climate_cells(
            self._file_based_ud(grid), sarra_climate_per_cell=sample,
        )
        self.assertEqual(check["result"], "fail")
        self.assertEqual(check["details"]["affected_cells"], [2, 3])

    def test_full_sample_passes(self):
        grid = _grid([1, 2])
        sample = {1: {"rain": [1.0]}, 2: {"rain": [2.0]}}
        check = _check_coverage_climate_cells(
            self._file_based_ud(grid), sarra_climate_per_cell=sample,
        )
        self.assertEqual(check["result"], "pass")
        self.assertEqual(check["details"]["affected_cells"], [])

    def test_readability_probe_false_when_no_climate_dir(self):
        """The readability probe returns False when there is no climate
        raster tree to open (so the coverage check reports unverifiable)."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            assert sarra_py_climate_rasters_readable(Path(d)) is False


# ── C.1(c)/(d) + C.3 — substrate shape, empty-list, unknown-bucket ─────


class TestRemediableFindingsShape(unittest.TestCase):
    _KEYS = {
        "category", "check", "scope", "result", "cell_ids",
        "cause", "remediable", "remediation_kind", "detail",
    }

    def test_all_complete_yields_empty_list(self):
        passing = {
            "check": "coverage_soil_cells", "scope": "per_cell",
            "result": "pass", "details": {"affected_cells": []},
        }
        self.assertEqual(_build_remediable_findings([passing], None), [])

    def test_findings_carry_full_key_set(self):
        check = {
            "check": "coverage_soil_cells", "scope": "per_cell",
            "result": "fail", "category": "completeness",
            "summary": "1 missing", "details": {"affected_cells": [9]},
        }
        f = _build_remediable_findings([check], None)[0]
        self.assertEqual(set(f.keys()), self._KEYS)

    def test_unknown_bucket_data_bearing_surfaced_not_dropped(self):
        # A data-bearing failing check with no resolvable WarningCategory
        # is surfaced as non-remediable, never silently dropped (CR-7).
        check = {
            "check": "value_range_tmax", "scope": "per_cell",
            "result": "warning", "category": "ranges",
            "summary": "out of range", "details": {"affected_cells": [4]},
        }
        findings = _build_remediable_findings([check], None)
        self.assertEqual(len(findings), 1)
        self.assertFalse(findings[0]["remediable"])
        self.assertIsNone(findings[0]["remediation_kind"])

    def test_unavailable_surfaces_with_coverage_unverifiable(self):
        check = {
            "check": "coverage_climate_cells", "scope": "per_cell",
            "result": "unavailable", "category": "completeness",
            "summary": "u", "details": {
                "affected_cells": [], "cause": "coverage_unverifiable",
            },
        }
        f = _build_remediable_findings([check], None)[0]
        self.assertFalse(f["remediable"])
        self.assertEqual(f["cause"], "coverage_unverifiable")

    def test_passing_checks_excluded(self):
        checks = [
            {"check": "x", "result": "pass", "details": {}},
            {"check": "y", "result": "info", "details": {}},
        ]
        self.assertEqual(_build_remediable_findings(checks, None), [])


# ── C.3 — bucket→remediable mapping pin (V2 = {INTERPOLATABLE}) ─────────


class TestBucketRemediableMapping(unittest.TestCase):
    def test_only_interpolatable_is_remediable(self):
        self.assertTrue(_bucket_is_remediable(WarningBucket.INTERPOLATABLE))
        for b in (
            WarningBucket.AUTO_FIXABLE, WarningBucket.INFORMATIONAL,
            WarningBucket.TRUE_EXCLUDE,
            WarningBucket.MANUAL_OVERRIDE_WITH_EVIDENCE,
        ):
            self.assertFalse(
                _bucket_is_remediable(b), f"{b} must NOT be V2-remediable",
            )

    def test_unknown_bucket_not_remediable(self):
        self.assertFalse(_bucket_is_remediable(None))

    def test_resolve_uses_canonical_map(self):
        self.assertEqual(
            _resolve_warning_bucket("short_gap_interpolatable"),
            WarningBucket.INTERPOLATABLE,
        )
        self.assertIsNone(_resolve_warning_bucket("not_a_category"))
        self.assertIsNone(_resolve_warning_bucket(None))


# ── C.4 — coverage ↔ single-source parity (READ-ONLY pin, CR-6) ─────────


class TestCoverageSingleSourceParity(unittest.TestCase):
    def test_soil_affected_cells_derive_from_compute_coverage_diff(self):
        grid = _grid([1, 2, 3, 4])
        soil = {1: _profile([object()]), 3: _profile([object()])}
        check = _check_coverage_soil_cells(_ud(grid=grid, soil=soil))
        cells_with_data = {1, 3}
        self.assertEqual(
            check["details"]["affected_cells"],
            compute_coverage_diff({1, 2, 3, 4}, cells_with_data),
        )

    def test_both_coverage_checks_call_the_single_source(self):
        # Single-source enforcement (#24): neither coverage check may
        # re-inline its own grid set-diff.
        for fn in (_check_coverage_climate_cells, _check_coverage_soil_cells):
            src = inspect.getsource(fn)
            self.assertIn(
                "compute_coverage_diff(", src,
                f"{fn.__name__} must derive missing cells from the helper",
            )


# ── End-to-end — remediable_findings present in the rollup ──────────────


def test_report_carries_remediable_findings_list(sample_project_config):
    """The rollup return dict carries an absent-cell finding end-to-end,
    using a real ProjectConfig so the config-dependent checks run."""
    grid = _grid([1, 2])
    soil = {1: _profile([object()])}  # cell 2 absent → a finding
    report = scientific.run_scientific_validation(
        _ud(grid=grid, soil=soil), sample_project_config, enabled_platforms=[],
    )
    assert "remediable_findings" in report
    assert isinstance(report["remediable_findings"], list)
    absent = [
        f for f in report["remediable_findings"]
        if f["check"] == "coverage_soil_cells" and 2 in f["cell_ids"]
    ]
    assert absent, "absent soil cell must reach remediable_findings"
    assert absent[0]["remediable"]


if __name__ == "__main__":
    unittest.main()
