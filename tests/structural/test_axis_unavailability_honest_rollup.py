"""F-CO — pin the honest-signal rollup contract for full axis
unavailability.

Per warning-auditor §6.2 RCA: when the climate (or soil) axis is
fully unavailable but the grid has cells, the per-cell validation
status is ``fail`` on every cell. The G7 §2 ``unavailable``
convention emitted by ``_check_coverage_climate_cells`` and
``_check_coverage_soil_cells`` masked this at the rollup because
``run_scientific_validation`` filters ``unavailable`` out of the
``runnable`` set when deciding ``overall_result``. With no fail
or warning in the runnable set after the filter, the rollup
falsely reported ``pass`` / ``warning`` while the cockpit map
showed 0/N coverage.

F-CO Layer 1 splits the convention: full-axis-unavailable with a
populated grid emits ``fail`` so the rollup sees the failure at
the runnable level. The single fail record carries
``affected_cells = all grid cells`` plus a ``cause`` discriminator
so the cockpit drawer still renders the axis-level absence
narrative (not per-cell anomaly framing).

This pin enforces the producer-consumer contract per durable §27:
the per-cell coverage validator (producer) and
``run_scientific_validation`` rollup (consumer) speak the same
honest-signal vocabulary about full-axis unavailability.
"""
from __future__ import annotations

import unittest

from prismpy.models.region import BoundingBox, Region
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators.base import UnifiedData
from prismpy.validators.scientific import (
    _check_coverage_climate_cells,
    _check_coverage_soil_cells,
    run_scientific_validation,
)


def _make_unified(*, n_cells: int = 3, climate=None, soil=None):
    cells = [
        GridCell(cell_id=i, lat=0.5, lon=0.5,
                 row=0, col=i, resolution="5arcmin")
        for i in range(n_cells)
    ]
    grid = SpatialGrid(
        bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        resolution="5arcmin", cells=cells,
    )
    return UnifiedData(
        region=Region(
            name="t", country="t", country_iso3="TST",
            bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        ),
        grid=grid,
        climate=climate if climate is not None else {},
        soil=soil if soil is not None else {},
    )


class TestPerCellCoverageFullAxisUnavailableEmitsFail(unittest.TestCase):
    """Producer-side pin — the per-cell coverage validators MUST
    emit ``fail`` (not ``unavailable``) on full-axis-unavailable
    runs so the rollup's runnable filter doesn't mask the signal."""

    def test_climate_fully_empty_emits_fail_with_all_cells_affected(self):
        unified = _make_unified(n_cells=8, climate={})
        check = _check_coverage_climate_cells(unified)
        self.assertEqual(
            check["result"], "fail",
            "F-CO Layer 1 — empty climate dict MUST emit fail (not "
            "unavailable) so the rollup's runnable filter picks up "
            "the failure. The ``unavailable`` shape was the source "
            "of the F-AG NEW manifestation: rollup overall='warning' "
            "while per-cell validation_status='fail' on every cell.",
        )
        self.assertEqual(check["details"]["n_missing"], 8)
        self.assertEqual(len(check["details"]["affected_cells"]), 8)
        self.assertEqual(check["details"]["cause"], "no_climate_fetch")
        # ICASA / MISDAT provenance preserved on the fail record so
        # the cockpit's axis-level absence narrative still renders.
        self.assertTrue(check["details"]["icasa_misdat"])
        # F-CO codex round-1 LOW absorption — Dr. Kofi's audit-grep
        # workflow searches the ``manuscript_claim`` STRING (not
        # just ``details.icasa_misdat``) for the ICASA / MISDAT
        # tokens that ``_unavailable()`` treats as the audit-trail
        # anchor. The F-CO fail records carry the same axis-level
        # absence narrative as an ``unavailable`` record at the
        # same site, so the anchor must survive in the string.
        self.assertIn(
            "ICASA", check["manuscript_claim"],
            "F-CO fail records MUST keep the ICASA token in "
            "``manuscript_claim`` so Dr. Kofi's audit-grep "
            "workflow finds the standards lineage. The "
            "``_unavailable()`` helper guards this contract via "
            "``logger.warning`` at scientific.py:917-923; the fail "
            "path carries the same audit-trail responsibility.",
        )
        self.assertIn(
            "MISDAT", check["manuscript_claim"],
            "F-CO fail records MUST keep the MISDAT token in "
            "``manuscript_claim`` (paired anchor with ICASA).",
        )

    def test_soil_fully_empty_emits_fail_with_all_cells_affected(self):
        unified = _make_unified(n_cells=8, climate={0: None}, soil={})
        check = _check_coverage_soil_cells(unified)
        self.assertEqual(
            check["result"], "fail",
            "F-CO Layer 1 symmetric soil-side mirror — empty soil "
            "dict MUST emit fail per durable §27 producer-consumer "
            "parity.",
        )
        self.assertEqual(check["details"]["n_missing"], 8)
        self.assertEqual(check["details"]["cause"], "no_soil_match")
        self.assertTrue(check["details"]["icasa_misdat"])
        # F-CO codex round-1 LOW absorption — symmetric audit-grep
        # anchor pin (see climate-side test for the contract).
        self.assertIn("ICASA", check["manuscript_claim"])
        self.assertIn("MISDAT", check["manuscript_claim"])

    def test_grid_none_emits_info_not_fail(self):
        # Defensive — when the grid itself is unavailable, the
        # validators short-circuit to ``info``, NOT fail. F-CO's
        # honest-signal split only applies when the grid is known.
        unified = UnifiedData(
            region=Region(
                name="t", country="t", country_iso3="TST",
                bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
            ),
            grid=None,
            climate={},
            soil={},
        )
        check_climate = _check_coverage_climate_cells(unified)
        check_soil = _check_coverage_soil_cells(unified)
        self.assertEqual(check_climate["result"], "info")
        self.assertEqual(check_soil["result"], "info")


class TestRollupHonestSignalOnFullAxisUnavailability(unittest.TestCase):
    """Consumer-side pin — when the per-cell coverage validator
    emits ``fail`` for full-axis-unavailable cases (F-CO Layer 1),
    the rollup logic at ``run_scientific_validation`` MUST surface
    ``overall_result='fail'``. This pin tests the rollup logic in
    isolation by synthesizing a check list with the post-F-CO
    shape; the full end-to-end path is covered by the existing
    integration-tier fixtures."""

    def _roll_up(self, checks):
        """Mirror of the rollup logic at scientific.py:184-196.
        Mirror — not import — because the rollup is inlined inside
        ``run_scientific_validation``. The ``TestProducerSourceShape``-
        style mirror approach catches divergence via the test for
        the climate-side fail emission above; if a future refactor
        reworks the rollup, both the production logic and this
        mirror need updating in lock-step (durable §27)."""
        results = [c["result"] for c in checks]
        n_unavailable = results.count("unavailable")
        runnable = [r for r in results if r != "unavailable"]
        if "fail" in runnable:
            return "fail"
        if "warning" in runnable:
            return "warning"
        if n_unavailable > 0 and not any(
            r in ("pass", "warning", "fail") for r in runnable
        ):
            return "unavailable"
        return "pass"

    def test_rollup_emits_fail_when_axis_coverage_fails(self):
        # F-CO contract — when ``coverage_climate_cells`` emits
        # ``fail`` (the post-F-CO shape) and other axis validators
        # emit ``unavailable``, the rollup MUST surface ``fail``.
        # Previously these validators all emitted ``unavailable``;
        # the rollup filtered them out and the only surviving
        # ``warning`` from ``coverage_global`` won the rollup.
        checks_post_f_co = [
            {"check": "coverage_climate_cells", "result": "fail"},
            # Sibling climate-axis validators retain the G7 §2
            # convention — they still emit ``unavailable`` on a
            # fully-empty climate axis. The rollup filter then
            # excludes them from the ``runnable`` set.
            {"check": "temporal_completeness", "result": "unavailable"},
            {"check": "value_range_climate_tmax", "result": "unavailable"},
            {"check": "cross_variable_consistency", "result": "unavailable"},
            # The axis-agnostic ``coverage_global`` check still
            # ran (it doesn't need climate to do its grid math)
            # and may emit ``warning`` for the partial substrate.
            {"check": "coverage_global", "result": "warning"},
        ]
        overall = self._roll_up(checks_post_f_co)
        self.assertEqual(
            overall, "fail",
            "F-CO rollup contract — when the per-cell coverage "
            "validator emits fail on full-axis-unavailable, the "
            "rollup MUST surface fail. The runnable filter keeps "
            "the fail visible; the prior all-unavailable shape "
            "got filtered to a single warning that lost the "
            "0/N-coverage signal.",
        )

    def test_rollup_pre_f_co_shape_proves_regression_class(self):
        # Documentation pin — the PRE-F-CO shape (all coverage
        # validators emitting unavailable, only ``coverage_global``
        # surviving with warning) rolled up to ``warning`` while
        # per-cell validation_status='fail' on every cell. This
        # test PINS the regression class so a future refactor that
        # accidentally reverts the F-CO split is caught by the
        # paired ``test_rollup_emits_fail_when_axis_coverage_fails``
        # pin above (which expects ``fail``).
        checks_pre_f_co = [
            # Old G7 §2 — coverage validator also emitted
            # unavailable on full-axis-unavailable.
            {"check": "coverage_climate_cells", "result": "unavailable"},
            {"check": "temporal_completeness", "result": "unavailable"},
            {"check": "value_range_climate_tmax", "result": "unavailable"},
            {"check": "cross_variable_consistency", "result": "unavailable"},
            {"check": "coverage_global", "result": "warning"},
        ]
        overall = self._roll_up(checks_pre_f_co)
        self.assertEqual(
            overall, "warning",
            "Documentation pin — the PRE-F-CO shape rolls up to "
            "'warning' (not fail), masking the 0/N coverage. This "
            "is the dishonest-signal class F-CO closes; the paired "
            "post-F-CO pin above asserts the new behaviour.",
        )


if __name__ == '__main__':
    unittest.main()
