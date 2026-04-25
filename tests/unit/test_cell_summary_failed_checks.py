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
