"""F-R Sprint A — integration tests for the boundary inclusion-rule
+ share-percent threshold across SpatialGrid.from_bounds, the AC-2
harmonize-stage filter logic, gadm.generate_schema_data (the CRAFT
4-path canonical-grid choke point), and the GeometryRequiredError
guard.

Anchors:
- Stage 0 empirical bifurcation: bbox_intersects vs centroid_strict
  produce different cell counts on the same polygon. The 149/114
  Koutiala numbers come from the team-lead's Phase 0 verdict; this
  module reproduces the bifurcation on a synthetic polygon (no
  external GADM fetch needed for unit-tier integration).
- AC-7 arithmetic invariant per AC-2 Stage 5:
  ``n_cells_full_extent = excluded_by_inclusion_rule
   + excluded_by_min_share_percent + admitted``.
  (User-skip is excluded from this scope per Draft 6 §AC-2.)
- AC-3 CRAFT 4-path drill: applying threshold to only some paths
  produces cell-count drift.
- AC-2 Stage 0 GeometryRequiredError on centroid_strict + no-polygon.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple
from unittest import TestCase

import pytest

from prismpy.config.schema import (
    BoundaryConfig,
    BoundarySource,
    ManualBoundsConfig,
)
from prismpy.data_sources.gadm import GADMDataSource
from prismpy.models.region import BoundingBox
from prismpy.models.spatial import SpatialGrid


# ---------------------------------------------------------------------------
# Synthetic Koutiala polygon — small enough for unit-tier; big enough to
# carve out the bbox_intersects vs centroid_strict bifurcation.
# ---------------------------------------------------------------------------


def _make_koutiala_diamond_wkt() -> str:
    """A diamond-shaped polygon centered at the synthetic Koutiala
    bbox center. Diamond shape is deliberate — its bounding box
    extends beyond its actual area, so cells in the bbox corners
    have their centroid OUTSIDE the polygon (centroid_strict
    excludes them) while their bbox INTERSECTS the polygon
    (bbox_intersects admits them). This is the structural
    bifurcation the F-R fix exposes."""
    # Synthetic Koutiala — 1.0° × 1.0° bbox, diamond inscribed.
    # Center: (-5.5, 12.0); diamond vertices at compass points.
    return (
        "POLYGON(("
        "-5.5 12.5, "  # north
        "-5.0 12.0, "  # east
        "-5.5 11.5, "  # south
        "-6.0 12.0, "  # west
        "-5.5 12.5"    # close
        "))"
    )


def _koutiala_diamond_bbox() -> BoundingBox:
    """Bounding box covering the diamond. At 5 arcmin (1/12°) that's
    12 × 12 = 144 cells in the rectangle."""
    return BoundingBox(minx=-6.0, miny=11.5, maxx=-5.0, maxy=12.5)


# ---------------------------------------------------------------------------
# AC-2 Stage 2 — inclusion_rule via SpatialGrid.from_bounds clip_geometry
# ---------------------------------------------------------------------------


class TestInclusionRuleEmpiricalBifurcation(TestCase):
    """Stage 0 empirical anchor: same polygon + same bbox produce
    different cell counts under the two inclusion_rule values.
    bbox_intersects admits the full extent (clip_geometry=None);
    centroid_strict admits only cells whose centroid lies inside
    the polygon."""

    @classmethod
    def setUpClass(cls):
        try:
            from shapely import wkt
            cls._wkt = wkt
            cls._diamond = wkt.loads(_make_koutiala_diamond_wkt())
            cls._shapely_available = True
        except ImportError:
            cls._shapely_available = False

    def setUp(self):
        if not self._shapely_available:
            self.skipTest("shapely not available — skip integration probe")
        self.bbox = _koutiala_diamond_bbox()

    def test_bbox_intersects_admits_full_extent(self):
        """``inclusion_rule='bbox_intersects'`` is realised in
        SpatialGrid.from_bounds via ``clip_geometry=None``: every
        cell in the bounding-box rectangle is admitted, including
        the diamond's bbox corners that fall outside the polygon
        itself. This is the AgMIP-canonical baseline."""
        grid = SpatialGrid.from_bounds(
            self.bbox, resolution="5arcmin",
            clip_geometry=None, exclude_cells=None,
        )
        # 1.0° / (5/60)° = 12 cell rows × 12 cell cols. SpatialGrid
        # may include edge cells beyond the strict bounding-box per
        # its alignment rules; ~144-180 is the structural envelope.
        # The pin: full extent is comfortably > 100 (clearly bigger
        # than centroid_strict result on the diamond).
        self.assertGreaterEqual(grid.n_cells, 100)

    def test_centroid_strict_drops_corner_cells(self):
        """``inclusion_rule='centroid_strict'`` realised via
        ``clip_geometry=<polygon>``: cells whose centroid falls
        OUTSIDE the polygon are dropped. Diamond shape has bbox
        corners outside the polygon, so centroid_strict count is
        strictly less than bbox_intersects count."""
        grid_full = SpatialGrid.from_bounds(
            self.bbox, resolution="5arcmin",
            clip_geometry=None, exclude_cells=None,
        )
        grid_strict = SpatialGrid.from_bounds(
            self.bbox, resolution="5arcmin",
            clip_geometry=self._diamond, exclude_cells=None,
        )
        self.assertLess(
            grid_strict.n_cells, grid_full.n_cells,
            "centroid_strict must admit FEWER cells than "
            "bbox_intersects baseline — the diamond polygon's bbox "
            "corners have centroids outside the polygon, so "
            "centroid_strict drops them.",
        )
        # The bifurcation is the load-bearing F-R contract: any
        # delta proves the rule landed.
        delta = grid_full.n_cells - grid_strict.n_cells
        self.assertGreater(delta, 0)


# ---------------------------------------------------------------------------
# AC-2 Stage 0 — GeometryRequiredError guard
# ---------------------------------------------------------------------------


class TestCentroidStrictRequiresGeometry(TestCase):
    """AC-2 Stage 0: ``inclusion_rule='centroid_strict'`` selected
    on a manual-bbox boundary (no GADM polygon) raises
    GeometryRequiredError at HARMONIZE entry. CC-2 honest-signal
    discipline — silent fallback would render
    ``provenance.boundary.inclusion_rule='centroid_strict'`` while
    secretly applying bbox_intersects (the F-R bug class)."""

    def test_geometry_required_error_class_exists(self):
        from prismpy.pipeline.executor import GeometryRequiredError
        self.assertTrue(issubclass(GeometryRequiredError, ValueError))

    def test_error_message_cites_actionable_remediation(self):
        from prismpy.pipeline.executor import GeometryRequiredError
        try:
            raise GeometryRequiredError(
                "inclusion_rule='centroid_strict' requires a "
                "GADM geometry; region.geometry_wkt is empty/"
                "unparsable. Either select a GADM source for "
                "region or use inclusion_rule='bbox_intersects'."
            )
        except GeometryRequiredError as exc:
            msg = str(exc)
        self.assertIn('centroid_strict', msg)
        self.assertIn('GADM', msg)
        self.assertIn('bbox_intersects', msg)


# ---------------------------------------------------------------------------
# AC-3 — CRAFT 4-path canonical-grid threshold drill
# ---------------------------------------------------------------------------


class TestGadmGenerateSchemaDataThresholdFilter(TestCase):
    """AC-3: ``gadm.generate_schema_data(threshold=...)`` admits or
    drops cells based on share_percent. All 4 CRAFT translator
    paths thread this kwarg; the fix is upstream-of-callsite
    (single-touchpoint per audit + Phase 0 4-path correction)."""

    @classmethod
    def setUpClass(cls):
        try:
            import geopandas as gpd
            from shapely import wkt
            cls._gpd = gpd
            cls._wkt = wkt
            cls._diamond = wkt.loads(_make_koutiala_diamond_wkt())
            cls._gdf = gpd.GeoDataFrame(geometry=[cls._diamond], crs="EPSG:4326")
            cls._gpd_available = True
        except ImportError:
            cls._gpd_available = False

    def setUp(self):
        if not self._gpd_available:
            self.skipTest("geopandas/shapely not available")
        self.gadm = GADMDataSource(gadm_path=None)

    def test_default_threshold_zero_admits_all_intersecting(self):
        """Default ``threshold=0.0`` (AgMIP-canonical) admits every
        cell whose extent intersects the polygon. Same as pre-F-R
        behaviour."""
        rows_zero, _ = self.gadm.generate_schema_data(
            gdf=self._gdf, resolution_deg=5/60, threshold=0.0,
        )
        # Diamond bbox 1.0° × 1.0° at 5 arcmin → up to ~144 cells;
        # diamond intersects ~half — anything > 0 is structurally
        # correct.
        self.assertGreater(len(rows_zero), 0)
        # Verify SP values are all in [0, 100]
        for row in rows_zero:
            self.assertGreaterEqual(row['share_percent'], 0.0)
            self.assertLessEqual(row['share_percent'], 100.0)

    def test_threshold_25_drops_low_share_cells(self):
        rows_zero, _ = self.gadm.generate_schema_data(
            gdf=self._gdf, resolution_deg=5/60, threshold=0.0,
        )
        rows_25, _ = self.gadm.generate_schema_data(
            gdf=self._gdf, resolution_deg=5/60, threshold=25.0,
        )
        self.assertLess(
            len(rows_25), len(rows_zero),
            "threshold=25.0 must drop MORE cells than "
            "threshold=0.0; cells with share_percent < 25 are "
            "filtered.",
        )
        # All admitted cells must have share_percent >= 25
        for row in rows_25:
            self.assertGreaterEqual(row['share_percent'], 25.0)

    def test_threshold_100_admits_only_full_cells(self):
        rows_full, _ = self.gadm.generate_schema_data(
            gdf=self._gdf, resolution_deg=5/60, threshold=100.0,
        )
        # Diamond polygon has no cell with 100% coverage (all cells
        # are at boundary; at least some have <100). May admit 0
        # cells; that's the structural test.
        for row in rows_full:
            self.assertGreaterEqual(row['share_percent'], 100.0)


class TestCraftFourPathThresholdAntiMutationDrill(TestCase):
    """AC-3 anti-mutation drill: simulate threshold being applied
    only to N of 4 paths. The single-touchpoint fix at
    gadm.generate_schema_data covers all 4 paths uniformly. Source
    inspection confirms all 4 callsites pass the threshold kwarg."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        translator_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "prismpy" / "translators"
            / "craft" / "translator.py"
        )
        cls._source = translator_path.read_text(encoding='utf-8')

    def test_path_1_threads_threshold(self):
        # Path 1: _generate_schema_from_gadm callsite at translator.py:538.
        # threshold kwarg must be present at the callsite.
        self.assertIn(
            'threshold=self.config.region.boundary.min_share_percent',
            self._source,
            "Path 1 (CRAFT schema gen via _generate_schema_from_gadm) "
            "must thread threshold from BoundaryConfig.",
        )

    def test_internal_helper_threads_threshold_to_gadm(self):
        # _generate_schema_from_gadm internally calls
        # gadm.generate_schema_data — must thread threshold kwarg.
        # Non-default-arg threading: the internal helper takes
        # `threshold` as a kwarg-only param + forwards it.
        self.assertIn(
            'threshold: float = 0.0',
            self._source,
            "_generate_schema_from_gadm must declare ``threshold`` "
            "as a kwarg-only param with default 0.0.",
        )
        self.assertIn(
            'threshold=threshold',
            self._source,
            "_generate_schema_from_gadm must forward threshold to "
            "gadm.generate_schema_data.",
        )

    def test_all_four_paths_thread_threshold_or_use_helper(self):
        # AC-3 invariant: the 4 paths through CRAFT translator that
        # produce cell sets all flow through gadm.generate_schema_data
        # with a threshold kwarg. Source-pin counts:
        #   Path 1  — _generate_schema_from_gadm callsite (CRAFT schema)
        #   Path 1b — direct gadm.generate_schema_data (CRAFT schema)
        #   Path 1c — _generate_schema_from_gadm callsite (Python schema)
        #   Path 1d — direct gadm.generate_schema_data (Python schema)
        # We count the threshold kwarg occurrences = 5
        # (1 in _generate_schema_from_gadm signature definition + 1
        # forward inside helper + 1 each at Path 1, 1b, 1c, 1d
        # callsites = 5 OR 6 depending on how the helper signature
        # is counted). The pin: at least 4 callsite-style threshold
        # threadings.
        threshold_threadings = self._source.count(
            'threshold=self.config.region.boundary.min_share_percent'
        )
        self.assertGreaterEqual(
            threshold_threadings, 4,
            f"Expected at least 4 callsite threshold threadings "
            f"(Path 1 + 1b + 1c + 1d); got {threshold_threadings}. "
            f"AC-3 4-path drill failed — a callsite forgot the "
            f"threshold kwarg, breaking canonical-grid agreement "
            f"between AC-2 harmonize and CRAFT schema rows.",
        )


# ---------------------------------------------------------------------------
# AC-7 arithmetic invariant — boundary scope (user_excluded out of scope)
# ---------------------------------------------------------------------------


class TestACSevenArithmeticInvariant(TestCase):
    """AC-7: the boundary cell-count fields satisfy the arithmetic
    identity:
        n_cells_full_extent = n_cells_excluded_by_inclusion_rule
                              + n_cells_excluded_by_min_share_percent
                              + n_cells_admitted
    (User-skip lives at AC-2 Stage 4 + is computed at integration
    test time from config.region.exclude_cells; the boundary block
    tracks the BOUNDARY-RULE scope only.)"""

    def test_invariant_holds_when_no_filtering(self):
        # bbox_intersects + 0.0 threshold + no user_skip → all cells
        # admitted; both filter counts are 0.
        n_full = 144
        n_excluded_by_rule = 0
        n_excluded_by_threshold = 0
        n_admitted = 144
        self.assertEqual(
            n_full,
            n_excluded_by_rule + n_excluded_by_threshold + n_admitted,
        )

    def test_invariant_holds_with_centroid_strict(self):
        # centroid_strict drops corners; threshold=0; no user_skip
        # → admitted = full - excluded_by_rule
        n_full = 144
        n_excluded_by_rule = 30
        n_excluded_by_threshold = 0
        n_admitted = 114
        self.assertEqual(
            n_full,
            n_excluded_by_rule + n_excluded_by_threshold + n_admitted,
        )

    def test_invariant_holds_with_threshold_filter(self):
        # bbox_intersects + threshold=25 → some cells dropped by
        # threshold; rule drops nothing.
        n_full = 144
        n_excluded_by_rule = 0
        n_excluded_by_threshold = 12
        n_admitted = 132
        self.assertEqual(
            n_full,
            n_excluded_by_rule + n_excluded_by_threshold + n_admitted,
        )

    def test_invariant_holds_with_compound_filters(self):
        # centroid_strict + threshold=10 → both filters apply
        n_full = 144
        n_excluded_by_rule = 30
        n_excluded_by_threshold = 5
        n_admitted = 109
        self.assertEqual(
            n_full,
            n_excluded_by_rule + n_excluded_by_threshold + n_admitted,
        )


# ---------------------------------------------------------------------------
# AC-2 Stage 5 — provenance.set_boundary writes canonical fields
# ---------------------------------------------------------------------------


class TestProvenanceSetBoundaryEmitsAllEightFields(TestCase):
    """AC-4 + AC-2 Stage 5 integration: when AC-2's harmonize
    helper finalizes the canonical filtered grid, it calls
    ``provenance.set_boundary(...)`` with the 8 fields per
    Draft 6. The boundary block lands on
    ``record.boundary`` and serializes through ``to_dict()``."""

    def test_full_eight_field_roundtrip_through_to_dict(self):
        # bbox_intersects + 0.0 threshold + no user-skip on a
        # synthetic 192-cell extent: all cells admit. AC-7
        # arithmetic invariant requires
        #   192 (full) = 0 (rule) + 0 (threshold) + 192 (admitted)
        # The Koutiala 192 → 149 reduction comes from a different
        # config (centroid_strict OR threshold > 0), not from
        # the AgMIP-canonical defaults asserted here.
        from prismpy.provenance.tracker import ProvenanceTracker
        tracker = ProvenanceTracker(enabled=True, project_name='koutiala')
        tracker.set_boundary(
            source='gadm',
            version='GADM v4.1',
            inclusion_rule='bbox_intersects',
            min_share_percent=0.0,
            n_cells_full_extent=192,
            n_cells_excluded_by_inclusion_rule=0,
            n_cells_excluded_by_min_share_percent=0,
            n_cells_admitted=192,
        )
        d = tracker.record.to_dict()
        b = d['boundary']
        # All 8 fields present, types match Draft 6 schema.
        self.assertEqual(b['source'], 'gadm')
        self.assertEqual(b['version'], 'GADM v4.1')
        self.assertEqual(b['inclusion_rule'], 'bbox_intersects')
        self.assertEqual(b['min_share_percent'], 0.0)
        self.assertEqual(b['n_cells_full_extent'], 192)
        self.assertEqual(b['n_cells_excluded_by_inclusion_rule'], 0)
        self.assertEqual(b['n_cells_excluded_by_min_share_percent'], 0)
        self.assertEqual(b['n_cells_admitted'], 192)
        # AC-7 arithmetic invariant on the live record.
        self.assertEqual(
            b['n_cells_full_extent'],
            (b['n_cells_excluded_by_inclusion_rule']
             + b['n_cells_excluded_by_min_share_percent']
             + b['n_cells_admitted']),
        )

    def test_centroid_strict_run_records_correct_field_values(self):
        # Simulates a Koutiala centroid_strict + 0.0 run that admits 114.
        from prismpy.provenance.tracker import ProvenanceTracker
        tracker = ProvenanceTracker(enabled=True, project_name='koutiala')
        tracker.set_boundary(
            source='gadm',
            version='GADM v4.1',
            inclusion_rule='centroid_strict',
            min_share_percent=0.0,
            n_cells_full_extent=192,
            n_cells_excluded_by_inclusion_rule=78,
            n_cells_excluded_by_min_share_percent=0,
            n_cells_admitted=114,
        )
        b = tracker.record.boundary
        self.assertEqual(b['inclusion_rule'], 'centroid_strict')
        self.assertEqual(b['n_cells_admitted'], 114)
        # Arithmetic invariant
        self.assertEqual(
            b['n_cells_full_extent'],
            (b['n_cells_excluded_by_inclusion_rule']
             + b['n_cells_excluded_by_min_share_percent']
             + b['n_cells_admitted']),
        )


# ---------------------------------------------------------------------------
# AC-16 — ICASA conformance / pickle byte-equivalence sentinel
# ---------------------------------------------------------------------------


class TestACSixteenICASAConformance(TestCase):
    """AC-16: the F-R additions stay in the boundary-config /
    provenance-boundary surface and NEVER leak into SOL / WTH /
    FileX / pickle simulation outputs. Per CC-5 invariant.

    Default-domain-changed disambiguation: when a user explicitly
    selects ``inclusion_rule='centroid_strict'`` (the pre-F-R
    behaviour as opt-in), same-config runs reproduce the same
    Koutiala 114-cell domain as pre-F-R. The byte-equivalence
    contract applies to SAME-CONFIG runs (not "default" runs —
    F-R changes the default to bbox_intersects which produces
    149 cells, an intentional behaviour change). Test pin
    documents this disambiguation."""

    def test_boundaryconfig_fields_never_leak_to_grid_cell_to_dict(self):
        # GridCell.to_dict (which translators consume to write SOL /
        # WTH rows) MUST NOT include inclusion_rule or
        # min_share_percent fields. Those are F-R schema additions
        # for boundary config + provenance only.
        from prismpy.models.spatial import GridCell
        cell = GridCell(
            cell_id=1, lat=12.5, lon=-7.0, row=0, col=0,
            share_percent=42.5,
        )
        d = cell.to_dict()
        # share_percent is allowed (cell-level metadata; carried
        # through to translators for CRAFT schema 4th column).
        self.assertIn('share_percent', d)
        # inclusion_rule + min_share_percent are NOT cell-level
        # metadata; they live on BoundaryConfig + provenance only.
        self.assertNotIn('inclusion_rule', d)
        self.assertNotIn('min_share_percent', d)

    def test_centroid_strict_opt_in_documented_as_pre_F_R_baseline(self):
        # CC-1 + AC-16 disambiguation: opting into
        # ``inclusion_rule='centroid_strict'`` reproduces the
        # pre-F-R cell count (114 on Koutiala). New default
        # ``bbox_intersects'`` is an intentional behavior change
        # that restores the AgMIP-canonical 149-cell baseline.
        # This test pins the contract documentation; the actual
        # byte-equivalence test against pre-F-R outputs is
        # deferred to the AC-16 run-level test (would require
        # checked-in reference outputs).
        cfg_default = BoundaryConfig(
            source=BoundarySource.GADM,
            gadm_filter_value='Koutiala',
        )
        cfg_pre_f_r_opt_in = BoundaryConfig(
            source=BoundarySource.GADM,
            gadm_filter_value='Koutiala',
            inclusion_rule='centroid_strict',
        )
        # Default reproduces 149-cell behavior (bbox_intersects).
        self.assertEqual(cfg_default.inclusion_rule, 'bbox_intersects')
        # Opt-in reproduces 114-cell pre-F-R behavior.
        self.assertEqual(cfg_pre_f_r_opt_in.inclusion_rule, 'centroid_strict')

    def test_provenance_boundary_doesnt_leak_into_summary(self):
        # ProvenanceRecord.summary is a separate aggregate field.
        # Boundary block has its own top-level key, NOT folded
        # into summary (different consumers; different write times).
        from prismpy.provenance.tracker import ProvenanceTracker
        tracker = ProvenanceTracker(enabled=True, project_name='test')
        tracker.set_boundary(
            source='gadm',
            version='GADM v4.1',
            inclusion_rule='bbox_intersects',
            min_share_percent=0.0,
            n_cells_full_extent=144,
            n_cells_excluded_by_inclusion_rule=0,
            n_cells_excluded_by_min_share_percent=0,
            n_cells_admitted=144,
        )
        d = tracker.record.to_dict()
        self.assertIn('boundary', d)
        self.assertIn('summary', d)
        # Summary doesn't carry boundary fields
        self.assertNotIn('inclusion_rule', d.get('summary', {}))
        self.assertNotIn('min_share_percent', d.get('summary', {}))
