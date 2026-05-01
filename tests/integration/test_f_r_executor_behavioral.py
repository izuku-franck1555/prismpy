"""F-R Sprint A — executor-driven behavioral integration tests.

These tests drive ``TranslationPipeline._execute_harmonize`` end-to-end
so that the AC-2 5-stage filter is exercised as a whole, not piecewise.
The existing ``test_f_r_boundary_pipeline.py`` covers structural pins
on isolated surfaces (``SpatialGrid.from_bounds``, the
``ProvenanceTracker.set_boundary`` setter, source-text inspection of
the CRAFT 4-path threshold threading). Those tests cannot detect a
contributor removing executor branch logic — the executor never runs
in them.

Pin map (executor lines reference HEAD ``9195f85``):
- T1 — Test 1 (centroid_strict + GADM polygon admits the canonical
  filtered grid). Pins:
  * line 2139-2142: conditional ``clip_geometry`` assignment when
    ``inclusion_rule == 'centroid_strict'``. If removed, every run
    behaves like ``bbox_intersects`` and admits the full bbox extent.
  * line 2114 (negative anchor): the Stage 0 guard does NOT fire
    when geometry IS present.
- T2 — Test 2 (centroid_strict + missing/unparsable geometry produces
  distinct fingerprinted error messages). Two scenarios via
  ``self.subTest`` so each raise site is independently pinned in one
  test method:
  * 'no_geometry' — line 2114: Stage 0 guard. Error must contain
    "requires a GADM geometry".
  * 'bad_geometry' — line 2147: Stage 2 wkt.loads except handler.
    Error must contain "could not be parsed by shapely".
- T3 — Test 3 (bbox_intersects + irregular polygon counts the cells
  whose bbox does not intersect the polygon). Pins:
  * line 2210: the ``n_cells_excluded_by_inclusion_rule += 1``
    increment inside Stage 3's bbox-intersect early-out branch.
    Without it, the AC-7 arithmetic identity breaks for irregular
    polygons.

Note on the broad ``except Exception`` at executor.py:2472:
``_execute_harmonize`` catches every exception the F-R fixes raise
(including ``GeometryRequiredError``) and rewrites them into
``StageResult(success=False, errors=[…])``. T2 therefore asserts on
``result.errors`` content rather than wrapping the call in
``pytest.raises(GeometryRequiredError)`` — adding a re-raise carve-out
to the outer try/except would alter executor behavior outside this
fixup's scope (Sprint A's Gate B verdict is locked at HEAD ``9195f85``).
The pin remains tight: each raise site has a distinct error-message
fingerprint that the test asserts on; removing either raise changes
the message and fails the corresponding subTest.

The downstream stages (per-cell soil retrieval, climate gap-fill)
fail in this test environment because no real data sources are wired
up. T1 and T3 therefore inspect ``pipeline.provenance.record.boundary``
(populated by the executor's Stage 5 BEFORE the downstream failure)
rather than ``result.success`` / ``result.data``.
"""
from __future__ import annotations

from unittest import TestCase

from prismpy.config.schema import (
    BoundaryConfig,
    BoundarySource,
    CropCalendarConfig,
    CropConfig,
    ManualBoundsConfig,
    OutputConfig,
    Platform,
    ProjectConfig,
    ProjectInfo,
    RegionConfig,
    TemporalConfig,
)
from prismpy.models.region import BoundingBox, Region
from prismpy.models.spatial import SpatialGrid
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.provenance.tracker import ProvenanceTracker


# ---------------------------------------------------------------------------
# Synthetic Koutiala diamond polygon — same fixture as
# test_f_r_boundary_pipeline.py. Diamond shape produces the
# bbox/centroid bifurcation needed to exercise both inclusion_rule
# branches.
# ---------------------------------------------------------------------------

_DIAMOND_WKT = (
    "POLYGON(("
    "-5.5 12.5, "  # north
    "-5.0 12.0, "  # east
    "-5.5 11.5, "  # south
    "-6.0 12.0, "  # west
    "-5.5 12.5"    # close
    "))"
)


def _make_bbox() -> BoundingBox:
    return BoundingBox(minx=-6.0, miny=11.5, maxx=-5.0, maxy=12.5)


def _make_pipeline(*, inclusion_rule: str,
                   min_share_percent: float = 0.0) -> TranslationPipeline:
    """Build a minimal TranslationPipeline whose ``BoundaryConfig``
    carries the requested inclusion rule + share threshold. All other
    fields are taken from the conftest sample project config defaults
    so the executor's Stage 1 — Stage 5 surface is the only varying
    piece."""
    cfg = ProjectConfig(
        project=ProjectInfo(
            name='f_r_executor_behavioral',
            description='F-R Sprint A executor-driven behavioral test',
        ),
        region=RegionConfig(
            name='Koutiala',
            country='Mali',
            country_iso3='MLI',
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=-6.0, miny=11.5, maxx=-5.0, maxy=12.5,
                ),
                inclusion_rule=inclusion_rule,
                min_share_percent=min_share_percent,
            ),
        ),
        crop=CropConfig(
            name='Maize', name_short='mai', variety='Medium-duration',
            calendar=CropCalendarConfig(
                planting_doy=166, maturity_doy=285,
            ),
        ),
        temporal=TemporalConfig(
            start_year=2015, end_year=2020, spinup_years=2,
        ),
        targets=[Platform.CRAFT],
        output=OutputConfig(
            base_dir='outputs', structure='by_platform',
        ),
    )
    return TranslationPipeline(
        cfg,
        provenance=ProvenanceTracker(
            enabled=True, project_name='f_r_executor_behavioral',
        ),
    )


def _shapely_or_skip(testcase: TestCase):
    try:
        from shapely import wkt  # noqa: F401
    except ImportError:
        testcase.skipTest("shapely not available — skip executor probe")


# ---------------------------------------------------------------------------
# Test 1 — centroid_strict + GADM polygon: the executor's filtered
# grid count matches an independent ``SpatialGrid.from_bounds`` call.
# ---------------------------------------------------------------------------


class TestPipelineHarmonizeCentroidStrictAdmitsCanonicalGrid(TestCase):
    """Drives ``_execute_harmonize`` with ``inclusion_rule='centroid_strict'``
    and a non-trivial polygon; asserts that
    ``provenance.record.boundary['n_cells_admitted']`` equals an
    independent canonical-grid count from ``SpatialGrid.from_bounds``.

    Pins:
    - executor.py:2139-2142 (conditional ``clip_geometry`` assignment)
      — without it, Stage 2 admits the full extent and admitted
      becomes the bbox count, not the polygon-clipped count.
    - executor.py:2114 (negative anchor) — the Stage 0 guard MUST NOT
      fire when ``geometry_wkt`` is present; if a future contributor
      broadens the guard, this test crashes at Stage 0.
    """

    def setUp(self):
        _shapely_or_skip(self)

    def test_admitted_count_matches_independent_canonical_grid(self):
        from shapely import wkt
        diamond = wkt.loads(_DIAMOND_WKT)

        pipe = _make_pipeline(inclusion_rule='centroid_strict')
        region = Region(
            name='Koutiala', country='Mali', country_iso3='MLI',
            bounds=_make_bbox(),
            geometry_wkt=_DIAMOND_WKT,
        )
        # Drive the harmonize stage end-to-end. Downstream stages may
        # fail (no real data sources) but Stage 5 fires first and the
        # provenance.boundary is populated before the failure.
        pipe._execute_harmonize({'region': region})

        boundary = pipe.provenance.record.boundary
        self.assertIsNotNone(
            boundary,
            "executor must populate ``provenance.boundary`` at "
            "Stage 5 even when downstream stages fail. Missing "
            "boundary block means Stage 5 never fired or the Stage 0 "
            "guard tripped erroneously.",
        )
        admitted = boundary['n_cells_admitted']

        # Independent canonical-grid count — same parameters the
        # executor's Stage 2 uses internally. The two MUST agree.
        canonical_grid = SpatialGrid.from_bounds(
            _make_bbox(),
            resolution="5arcmin",
            clip_geometry=diamond,
            exclude_cells=None,
        )
        self.assertEqual(
            admitted, canonical_grid.n_cells,
            f"Executor's centroid_strict path admitted {admitted} "
            f"cells; an independent SpatialGrid.from_bounds call with "
            f"the same polygon admits {canonical_grid.n_cells}. The "
            f"two must agree — disagreement indicates the executor "
            f"is NOT applying the centroid_strict clip-geometry "
            f"(line 2139-2142 likely removed/bypassed).",
        )
        # Stage 0 guard must NOT fire when geometry is present.
        # If it did, errors[0] would contain the Stage-0 marker
        # AND boundary would be unset (negative anchor for line 2114).
        self.assertEqual(
            boundary['inclusion_rule'], 'centroid_strict',
            "Stage 5 must record the inclusion_rule as configured; "
            "any other value indicates Stage 0 short-circuited or "
            "Stage 2 silently degraded.",
        )


# ---------------------------------------------------------------------------
# Test 2 — centroid_strict without resolvable geometry: each raise
# site (Stage 0 guard at 2114; Stage 2 parse-failure at 2147) produces
# a distinct fingerprinted error message. Two scenarios via
# self.subTest so the test method counts as one but pins both sites.
# ---------------------------------------------------------------------------


class TestPipelineHarmonizeCentroidStrictRequiresGeometry(TestCase):
    """One test method with two scenarios via ``self.subTest`` —
    pins the two GeometryRequiredError raise sites independently.

    Note on assertion form: ``_execute_harmonize``'s outer
    ``except Exception`` (line 2472) rewrites every raised exception
    into ``StageResult(success=False, errors=[str(e)])``. The
    error-class identity is therefore preserved only inside the
    message string. We assert on message content, not on
    ``pytest.raises``. Adding a ``except GeometryRequiredError:
    raise`` carve-out to the outer try/except would propagate the
    error class but is a behavior change that is out-of-scope for
    this test fixup (Sprint A's Gate B verdict is locked at HEAD
    ``9195f85``).
    """

    def setUp(self):
        _shapely_or_skip(self)

    def test_each_raise_site_emits_distinct_fingerprint(self):
        # Scenario 1 (no_geometry → line 2114 / Stage 0 guard):
        # geometry_wkt is None. Without the raise, Stage 0 falls
        # through to Stage 2 where ``shapely.wkt.loads(None)`` returns
        # None silently (does NOT raise), so Stage 2's parse-failure
        # path at 2147 never fires either. Stage 0 is the load-bearing
        # guard for this scenario.
        # Scenario 2 (bad_geometry → line 2147 / Stage 2 parse): a
        # malformed but truthy WKT bypasses Stage 0; Stage 2's
        # ``wkt.loads`` raises which is re-raised as
        # GeometryRequiredError with the parse-failure fingerprint.
        scenarios = [
            ('no_geometry', None,
             'requires a GADM geometry', 2114),
            ('bad_geometry', 'NOT_VALID_WKT_GARBAGE',
             'could not be parsed by shapely', 2147),
        ]
        for case_id, geom, expected_marker, line_pinned in scenarios:
            with self.subTest(scenario=case_id, pins=line_pinned):
                pipe = _make_pipeline(inclusion_rule='centroid_strict')
                region = Region(
                    name='Koutiala', country='Mali',
                    country_iso3='MLI',
                    bounds=_make_bbox(),
                    geometry_wkt=geom,
                )
                result = pipe._execute_harmonize({'region': region})

                self.assertFalse(
                    result.success,
                    f"[{case_id}] centroid_strict + bad geometry "
                    f"must fail the stage. If success=True, the "
                    f"raise at line {line_pinned} was bypassed.",
                )
                first_error = result.errors[0]
                self.assertIn(
                    'centroid_strict', first_error,
                    f"[{case_id}] error must name the inclusion_rule "
                    f"so an operator can act on it. Pinned at "
                    f"executor.py:{line_pinned}.",
                )
                self.assertIn(
                    expected_marker, first_error,
                    f"[{case_id}] error must contain the fingerprint "
                    f"{expected_marker!r} produced ONLY by the raise "
                    f"at executor.py:{line_pinned}. A different "
                    f"fingerprint indicates that raise has been "
                    f"removed and a different code path fired "
                    f"instead (or silently fell through, the bug "
                    f"codex P2 #2 caught).",
                )


# ---------------------------------------------------------------------------
# Test 3 — bbox_intersects + irregular polygon: the Stage 3 increment
# at line 2210 keeps the AC-7 arithmetic invariant intact.
# ---------------------------------------------------------------------------


class TestPipelineHarmonizeBBoxIntersectsCountsExclusions(TestCase):
    """Drives ``_execute_harmonize`` with ``inclusion_rule='bbox_intersects'``
    and an irregular polygon. The diamond's bbox extends beyond the
    polygon, so cells in the bbox corners have boxes that do not
    intersect the polygon. Stage 3's bbox early-out at line 2210
    must increment ``n_cells_excluded_by_inclusion_rule`` for those
    cells — otherwise the AC-7 arithmetic identity ``full_extent =
    excluded_by_rule + excluded_by_threshold + admitted`` breaks.

    Pins executor.py:2210. Removing ``n_cells_excluded_by_inclusion_rule
    += 1`` keeps the Stage 2 count of 0 even though Stage 3 actively
    drops cells; admitted falls but the rule-exclusion counter does
    not rise, breaking the identity.
    """

    def setUp(self):
        _shapely_or_skip(self)

    def test_irregular_polygon_increments_inclusion_rule_counter(self):
        # Codex MEDIUM #1 absorption — fixture self-validation.
        # The pin at executor.py:2210 only fires when at least one
        # cell's bbox lies outside the admin polygon. If a future
        # refactor changed grid origin / resolution / cell size in a
        # way that shifted every cell's bbox to overlap the diamond,
        # the increment would not fire — and the assertGreater below
        # would fail with a misleading "increment not firing" message
        # when the real cause is fixture drift. Pre-validate the
        # fixture from outside the executor: build the same canonical
        # grid and check that at least one cell-box does not intersect
        # the diamond. If this guard fails, the diamond fixture has
        # drifted and the test must be updated; the pin at line 2210
        # is not the failure cause.
        from shapely import wkt
        from shapely.geometry import box as _shp_box
        diamond = wkt.loads(_DIAMOND_WKT)
        canonical_full_grid = SpatialGrid.from_bounds(
            _make_bbox(), resolution="5arcmin",
            clip_geometry=None, exclude_cells=None,
        )
        halfres = (5.0 / 60.0) / 2.0
        out_of_polygon_cells = [
            c for c in canonical_full_grid.cells
            if not _shp_box(
                c.lon - halfres, c.lat - halfres,
                c.lon + halfres, c.lat + halfres,
            ).intersects(diamond)
        ]
        self.assertGreater(
            len(out_of_polygon_cells), 0,
            "Diamond fixture must produce at least one cell whose "
            "bbox does not intersect the polygon — that is the "
            "scenario the line-2210 increment exists to count. "
            "Zero out-of-polygon cells means the fixture drifted "
            "(grid origin / resolution / cell-size change) and the "
            "test must be updated; the pin at line 2210 cannot be "
            "exercised under this fixture.",
        )

        pipe = _make_pipeline(inclusion_rule='bbox_intersects')
        region = Region(
            name='Koutiala', country='Mali', country_iso3='MLI',
            bounds=_make_bbox(),
            geometry_wkt=_DIAMOND_WKT,
        )
        pipe._execute_harmonize({'region': region})

        boundary = pipe.provenance.record.boundary
        self.assertIsNotNone(
            boundary,
            "Stage 5 must populate provenance.boundary even when "
            "downstream stages fail; missing boundary indicates Stage 5 "
            "did not fire or Stage 3 raised an unexpected exception.",
        )

        # Load-bearing assertion: Stage 3's increment at line 2210
        # must have fired for the diamond's out-of-polygon cells.
        self.assertGreater(
            boundary['n_cells_excluded_by_inclusion_rule'], 0,
            f"bbox_intersects + diamond polygon must exclude at "
            f"least one cell whose bbox does not intersect the "
            f"polygon. Got "
            f"n_cells_excluded_by_inclusion_rule="
            f"{boundary['n_cells_excluded_by_inclusion_rule']}; "
            f"the increment at executor.py:2210 is not firing.",
        )

        # AC-7 arithmetic invariant on the live executor output.
        full = boundary['n_cells_full_extent']
        excl_rule = boundary['n_cells_excluded_by_inclusion_rule']
        excl_thr = boundary['n_cells_excluded_by_min_share_percent']
        admitted = boundary['n_cells_admitted']
        self.assertEqual(
            full, excl_rule + excl_thr + admitted,
            f"AC-7 arithmetic identity broken on a live executor "
            f"run: {full} != {excl_rule} + {excl_thr} + {admitted} "
            f"(={excl_rule + excl_thr + admitted}). The Stage 3 "
            f"bbox early-out at line 2210 likely skipped its "
            f"counter increment.",
        )
