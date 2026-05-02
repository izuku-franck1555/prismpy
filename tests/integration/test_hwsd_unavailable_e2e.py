"""Sprint D.1 AC-4 — HWSD unavailable_cells end-to-end propagation pin.

The contract (AC-4): cells without HWSD coverage carry
``data_availability='unavailable'`` with ``unavailable_reason='soil'``
and ``unavailable_cause='soil_no_hwsd_coverage'``. The HWSD source
records each missed cell on its instance attribute ``unavailable_cells``;
the cascade orchestrator inside ``_execute_harmonize`` must surface
that list to the harmonize-stage helper so the routing fires.

This pin closes the propagation gap MEDIUM #2 surfaced in the GB-ready
self-check: ``_retrieve_hwsd_for_grid`` previously returned only a
profiles dict, dropping the loader's unavailable list before it
reached ``apply_harmonize_transformations``. Commit 9 changed the
return type to ``Tuple[Optional[Dict], List[Dict]]`` and the cascade
orchestrator now stashes the list into
``retrieved_data['hwsd_unavailable_cells']``.

The test exercises the executor end-to-end: it monkeypatches the
iSDA path to None (forcing the HWSD branch) and the HWSD path to
return a contract-shaped fixture (no profiles + a populated
unavailable list keyed by grid cell IDs). It then asserts that
``apply_harmonize_transformations`` produced the expected
``cells_unavailable`` entries on the unified-data harmonize_stats
metadata block + that the ProvenanceTracker recorded each one with
the contract-named cause.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
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
from prismpy.models.soil import SoilProfile
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.provenance.tracker import ProvenanceTracker


def _make_bbox() -> BoundingBox:
    """Tiny bbox so the executor's Stage-2 canonical grid generates
    a small, predictable number of cells."""
    return BoundingBox(minx=-5.10, miny=11.90, maxx=-5.00, maxy=12.00)


def _make_pipeline() -> TranslationPipeline:
    """Build a minimal CRAFT-targeted pipeline with bbox_intersects
    boundary so Stage 0-5 admit the canonical grid without needing
    a GADM polygon. CRAFT is HWSD-aware so the cascade will exercise
    the HWSD branch when iSDA returns None."""
    cfg = ProjectConfig(
        project=ProjectInfo(
            name='hwsd_unavailable_e2e',
            description='Sprint D.1 AC-4 HWSD unavailable_cells E2E pin',
        ),
        region=RegionConfig(
            name='Koutiala', country='Mali', country_iso3='MLI',
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=-5.10, miny=11.90, maxx=-5.00, maxy=12.00,
                ),
                inclusion_rule='bbox_intersects',
                min_share_percent=0.0,
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
            enabled=True, project_name='hwsd_unavailable_e2e',
        ),
    )


def _make_region() -> Region:
    return Region(
        name='Koutiala', country='Mali', country_iso3='MLI',
        bounds=_make_bbox(),
        # geometry_wkt absent — bbox_intersects + manual bbox does
        # not require a polygon.
    )


# ---------------------------------------------------------------------------
# AC-4 end-to-end — _retrieve_hwsd_for_grid returns a tuple, the
# cascade orchestrator stashes the list, and apply_harmonize_transformations
# routes each cell to data_availability='unavailable'.
# ---------------------------------------------------------------------------


class TestHWSDUnavailableEndToEndPropagation(TestCase):
    """The single load-bearing pin for AC-4 across the cascade."""

    def setUp(self):
        try:
            from shapely import wkt  # noqa: F401
        except ImportError:
            self.skipTest("shapely not available — skip executor probe")

    def _run_with_stubs(
        self,
        hwsd_unavailable: List[Dict[str, Any]],
        hwsd_profiles: Optional[Dict[int, SoilProfile]] = None,
    ) -> TranslationPipeline:
        """Drive ``_execute_harmonize`` with stubbed iSDA + HWSD
        cascade arms so the test exercises the wiring path without
        requiring real raster / API access."""
        pipe = _make_pipeline()

        # Force iSDA to return None so the cascade falls through to
        # HWSD (the contract-named branch for AC-4).
        pipe._retrieve_isda_api_for_grid = (  # type: ignore[assignment]
            lambda grid, region: None
        )

        # Stub HWSD to return the AC-4 contract shape: a tuple of
        # (profiles_or_None, unavailable_cells_list).
        def _stub_hwsd(
            grid, region,
        ) -> Tuple[Optional[Dict[int, SoilProfile]], List[Dict[str, Any]]]:
            return hwsd_profiles, list(hwsd_unavailable)

        pipe._retrieve_hwsd_for_grid = _stub_hwsd  # type: ignore[assignment]

        region = _make_region()
        pipe._execute_harmonize({'region': region})
        return pipe

    def test_loader_unavailable_list_routes_through_harmonize(self):
        """The contract-shaped HWSD-unavailable list lands on the
        harmonize-stage stats with the matching cause."""
        unavailable = [
            {'cell_id': 0, 'cause': 'soil_no_hwsd_coverage'},
            {'cell_id': 1, 'cause': 'soil_no_hwsd_coverage'},
        ]
        pipe = self._run_with_stubs(hwsd_unavailable=unavailable)

        # Pull the harmonize-stats block off the provenance record.
        # apply_harmonize_transformations writes a cells-unavailable
        # detail per entry via record_cell_unavailable.
        details = pipe.provenance.record.cells_unavailable_details
        self.assertGreaterEqual(
            len(details), 2,
            "AC-4 contract: each loader-level unavailable cell must "
            "be surfaced as a cells_unavailable_details entry on the "
            "ProvenanceTracker. <2 entries means the cascade "
            "orchestrator at executor.py:2354 dropped the list "
            "before the harmonize helper could route it.",
        )

        causes = [d.get('unavailable_cause') for d in details]
        self.assertIn(
            'soil_no_hwsd_coverage', causes,
            "AC-4 contract: the cause string must be "
            "'soil_no_hwsd_coverage' for HWSD-coverage-miss cells. "
            "Missing cause means the harmonize helper saw the entry "
            "but routed it under a different cause string.",
        )

        # Each surfaced HWSD-coverage entry MUST carry axis 'soil'.
        soil_axis_entries = [
            d for d in details
            if d.get('unavailable_cause') == 'soil_no_hwsd_coverage'
        ]
        for entry in soil_axis_entries:
            self.assertEqual(
                entry.get('unavailable_reason'), 'soil',
                "AC-4 contract: HWSD-coverage-miss cells route to "
                "axis 'soil'. A different reason indicates the "
                "harmonize helper mis-classified the entry.",
            )

    def test_empty_unavailable_list_does_not_inject_phantom_entries(self):
        """When HWSD returns no unavailable cells (e.g., full
        coverage path), the harmonize stage must not invent any
        cell-unavailable records. Negative anchor — guards against
        a future refactor that defaults to a non-empty list."""
        pipe = self._run_with_stubs(hwsd_unavailable=[])
        details = pipe.provenance.record.cells_unavailable_details
        # No HWSD-coverage-miss entries should appear.
        coverage_miss = [
            d for d in details
            if d.get('unavailable_cause') == 'soil_no_hwsd_coverage'
        ]
        self.assertEqual(
            len(coverage_miss), 0,
            "Negative anchor: empty HWSD-unavailable list must not "
            "produce phantom soil_no_hwsd_coverage entries on the "
            "harmonize stats. >0 means the cascade orchestrator "
            "synthesised cells the loader never recorded.",
        )

    def test_cell_ids_preserved_through_the_cascade(self):
        """The cell IDs the loader recorded are preserved through
        the cascade — the harmonize helper's cells_unavailable
        entries reference the same cell IDs the loader put on the
        list. Without this, downstream cell-summary builders would
        attach the routing to the wrong cells."""
        unavailable = [
            {'cell_id': 0, 'cause': 'soil_no_hwsd_coverage'},
            {'cell_id': 2, 'cause': 'soil_no_hwsd_coverage'},
        ]
        pipe = self._run_with_stubs(hwsd_unavailable=unavailable)

        details = pipe.provenance.record.cells_unavailable_details
        recorded_ids = sorted(
            d.get('cell_id') for d in details
            if d.get('unavailable_cause') == 'soil_no_hwsd_coverage'
        )
        self.assertEqual(
            recorded_ids, [0, 2],
            f"AC-4 contract: cell IDs must be preserved end-to-end. "
            f"Loader recorded [0, 2] but harmonize stats surfaced "
            f"{recorded_ids}. The cascade orchestrator is rewriting "
            f"the cell IDs.",
        )


# ---------------------------------------------------------------------------
# Cascade-orchestrator wiring — the source-level signature pin is the
# anti-revert anchor. The behavioral tests above already crash if the
# tuple-return contract is reverted (Python raises TypeError on
# unpacking a single value), so the signature pin is the only
# additional guard worth adding without duplicating coverage.
# ---------------------------------------------------------------------------


class TestRetrieveHWSDSignaturePin(TestCase):
    """Pins ``_retrieve_hwsd_for_grid``'s tuple-return signature so
    a future contributor cannot quietly revert to single-value
    return without tripping this pin."""

    def test_retrieve_hwsd_signature_returns_tuple(self):
        """The method's source must declare the tuple-of-(profiles,
        unavailable_cells) return type. Without the signature
        update, mypy / type checkers would not catch a future
        revert to single-value return."""
        import inspect
        from prismpy.pipeline.executor import TranslationPipeline
        sig = inspect.signature(
            TranslationPipeline._retrieve_hwsd_for_grid
        )
        ret_annotation = str(sig.return_annotation)
        self.assertIn(
            'Tuple', ret_annotation,
            f"AC-4 signature pin: _retrieve_hwsd_for_grid must "
            f"return Tuple[Optional[Dict], List[Dict]]. Current "
            f"annotation: {ret_annotation!r}. A non-tuple return "
            f"signals the unavailable_cells propagation gap is "
            f"reopened.",
        )
