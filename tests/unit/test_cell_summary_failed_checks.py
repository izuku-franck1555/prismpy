"""V2-22c-PRE.1 — cell_summary_version + soil_class + structured failed_checks.

Covers ACs PRE.1.1 (cell_summary_version), PRE.1.6 (soil_class via existing
SoilProfile.surface_texture), and is the home for PRE.1.2 (failed_checks
structured shape) tests as those land in subsequent commits.

The tests bypass TranslationPipeline.__init__ via __new__ — `_build_cell_summary`
is a pure data-projection method that doesn't touch any instance state on
self, so a synthetic instance is sufficient and avoids pulling in the
ProjectConfig + ProvenanceTracker dependency chain that init requires.
"""
from __future__ import annotations

import pytest

from prismpy.models.region import Region, BoundingBox
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.translators.base import UnifiedData


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

    def test_cell_summary_version_value_is_2_0(self):
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid())
        out = pipeline._build_cell_summary(unified)
        assert out["cell_summary_version"] == "2.0"

    def test_cell_summary_version_present_alongside_existing_top_level_keys(self):
        """Regression guard — the new field must not displace existing
        top-level keys that consumers already read."""
        pipeline = _make_pipeline()
        unified = UnifiedData(region=_make_region(), grid=_make_grid())
        out = pipeline._build_cell_summary(unified)
        for key in ("n_cells", "resolution", "cells"):
            assert key in out, f"top-level key {key!r} regressed"


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

    def test_soil_class_elided_when_layers_empty(self):
        """Edge case from contract: empty profile → surface_texture
        returns None → soil_class key absent (matches tmax_range elision)."""
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
        assert "soil_class" not in cell, (
            "empty layers list yields surface_texture=None; the "
            "soil_class key must be elided so the cockpit JS renders "
            "'No soil data' instead of 'soil_class=None'."
        )

    def test_soil_class_elided_when_no_profile(self):
        """No SoilProfile for this cell at all → no soil_class."""
        pipeline = _make_pipeline()
        unified = UnifiedData(
            region=_make_region(), grid=_make_grid(n_cells=1), soil={},
        )
        out = pipeline._build_cell_summary(unified)
        cell = out["cells"][0]
        assert "soil_class" not in cell
        assert cell["soil_source"] == "none"

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
