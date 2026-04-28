"""F16 — texture-fraction validator narrative metadata + plain-
language summary.

The pre-F16 summary read "abnormal totals" — a shorthand that
left the physical invariant implicit. The cockpit chip + drawer
cell-inspection surfaces need an explicit physical-invariant
sentence so a researcher (Aminata first-time, Moussa stakeholder)
reads the check name + summary as a single complete claim.

The validator emit also adds per-cell vs per-layer metadata so
the chip rendering ("7 cells flagged") and the drawer detail
("8 of 136 layers") derive from one canonical source. The
metadata convention generalizes to any future per-layer
validator: emit per-layer counts alongside affected_cells /
violation_details so the cockpit / drawer can decide which
granularity to surface.

The ``LAYERS_PER_PLATFORM`` constant exposes each platform's
conventional per-cell layer count for downstream consumer
rendering (the chip / drawer can describe the chip-vs-drawer
hierarchy consistently across platforms).

Persona reach: F16 lands for ALL FOUR researchers — Aminata
first-time confusion (texture chip text); Moussa stakeholder
narrative (chip vs drawer two-tier hierarchy); Dr. Kofi audit
lineage (texture metadata greppable on details.*); Ibrahim
mobile (Region Health card reads chip text consistently).
"""

from __future__ import annotations

import unittest
from typing import Dict

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators.base import UnifiedData
from prismpy.validators.scientific import (
    LAYERS_PER_PLATFORM,
    _check_value_ranges,
)


# ---------------------------------------------------------------------------
# Fixture helpers — build a synthetic UnifiedData with N profiles, each
# carrying M layers, with controlled texture-sum violations so the
# validator's metadata can be exercised at the unit-test layer.
# ---------------------------------------------------------------------------


def _make_layer(*, sand: float, clay: float, silt: float,
                depth_top: float = 0.0, depth_bottom: float = 0.2):
    return SoilLayer(
        depth_top=depth_top, depth_bottom=depth_bottom,
        sand=sand, clay=clay, silt=silt,
    )


def _make_profile(*, profile_id: str, layers):
    return SoilProfile(
        profile_id=profile_id, lat=0.5, lon=0.5,
        source="iSDA", layers=layers,
    )


def _make_unified(*, soil: Dict[int, SoilProfile]):
    cells = [
        GridCell(cell_id=cid, lat=0.5, lon=0.5,
                 row=0, col=cid, resolution="5arcmin")
        for cid in soil
    ]
    return UnifiedData(
        region=Region(
            name="t", country="t", country_iso3="TST",
            bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        ),
        grid=SpatialGrid(
            bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
            resolution="5arcmin", cells=cells,
        ),
        climate={},
        soil=soil,
    )


def _texture_check(unified):
    checks = _check_value_ranges(unified)
    return next(
        (c for c in checks if c["check"] == "value_range_texture_sum"),
        None,
    )


class TestLayersPerPlatformConstant(unittest.TestCase):
    """The ``LAYERS_PER_PLATFORM`` constant exposes each
    platform's conventional per-cell layer count so consumer
    rendering can describe the chip / drawer hierarchy
    consistently. Adding a new platform only needs an entry
    here; existing consumers pick it up automatically."""

    def test_constant_covers_all_four_platforms(self):
        for platform in ("craft", "pythia", "acea", "sarra_py"):
            self.assertIn(platform, LAYERS_PER_PLATFORM)

    def test_canonical_layer_counts(self):
        """The values match the per-platform translator emit
        conventions: CRAFT ships a 6-layer profile, PYTHIA
        surface + subsurface (2), ACEA 4-layer, SARRA-Py 2."""
        self.assertEqual(LAYERS_PER_PLATFORM["craft"], 6)
        self.assertEqual(LAYERS_PER_PLATFORM["pythia"], 2)
        self.assertEqual(LAYERS_PER_PLATFORM["acea"], 4)
        self.assertEqual(LAYERS_PER_PLATFORM["sarra_py"], 2)


class TestTextureSumSummaryWording(unittest.TestCase):
    """The summary string the consumer renders carries the
    physical-invariant phrasing rather than the prior 'abnormal
    totals' shorthand. The new wording reads as a complete
    sentence next to the check name."""

    def test_violation_summary_uses_invariant_phrasing(self):
        # One profile, one layer, deliberately broken: 50/50/50
        # = 150 sums above the [95, 105] band → flagged.
        soil = {0: _make_profile(
            profile_id="p0",
            layers=[_make_layer(sand=50, clay=50, silt=50)],
        )}
        check = _texture_check(_make_unified(soil=soil))
        self.assertIsNotNone(check)
        self.assertIn(
            "texture fractions that don't sum to 100%",
            check["summary"],
        )
        self.assertNotIn("abnormal totals", check["summary"])

    def test_pass_summary_unchanged(self):
        """The pass branch stays at the prior wording — the
        F16 narrative redesign only touches the violation case
        where the chip / drawer surface need the explicit
        physical-invariant sentence."""
        soil = {0: _make_profile(
            profile_id="p0",
            layers=[_make_layer(sand=50, clay=30, silt=20)],
        )}
        check = _texture_check(_make_unified(soil=soil))
        self.assertIsNotNone(check)
        self.assertEqual(check["result"], "pass")
        self.assertIn("for all", check["summary"])


class TestTextureSumNarrativeMetadata(unittest.TestCase):
    """The validator's emit now carries per-cell vs per-layer
    counts so the chip rendering ('7 cells flagged') and the
    drawer detail ('8 of 136 layers') derive from one canonical
    source. The metadata generalizes to any future per-layer
    validator."""

    def test_single_cell_single_layer_violation(self):
        """One cell with one bad layer → ``n_flagged_cells = 1``,
        ``n_flagged_layers = 1``, no multi-flagged cells."""
        soil = {0: _make_profile(
            profile_id="p0",
            layers=[
                _make_layer(sand=50, clay=30, silt=20),  # OK
                _make_layer(sand=80, clay=80, silt=80),  # broken
            ],
        )}
        check = _texture_check(_make_unified(soil=soil))
        details = check["details"]
        self.assertEqual(details["n_flagged_cells"], 1)
        self.assertEqual(details["n_flagged_layers"], 1)
        self.assertEqual(details["n_total_layers"], 2)
        self.assertEqual(details["n_multi_flagged_cells"], 0)

    def test_multi_flagged_cell_counted(self):
        """A cell with TWO broken layers contributes to both
        ``n_flagged_cells`` (counted once) and
        ``n_multi_flagged_cells`` (counted once because >1
        layer flagged)."""
        soil = {0: _make_profile(
            profile_id="p0",
            layers=[
                _make_layer(sand=80, clay=80, silt=80),
                _make_layer(sand=70, clay=70, silt=70),
            ],
        )}
        check = _texture_check(_make_unified(soil=soil))
        details = check["details"]
        self.assertEqual(details["n_flagged_cells"], 1)
        self.assertEqual(details["n_flagged_layers"], 2)
        self.assertEqual(details["n_multi_flagged_cells"], 1)

    def test_layers_per_cell_typical_is_mode(self):
        """``n_layers_per_cell_typical`` is the most-common
        layers-per-cell across profiles — robust to a single
        outlier cell with a different layer count."""
        soil = {
            0: _make_profile(
                profile_id="p0",
                layers=[_make_layer(sand=50, clay=30, silt=20)],
            ),  # 1 layer
            1: _make_profile(
                profile_id="p1",
                layers=[
                    _make_layer(sand=50, clay=30, silt=20),
                    _make_layer(sand=50, clay=30, silt=20),
                    _make_layer(sand=50, clay=30, silt=20),
                ],
            ),  # 3 layers
            2: _make_profile(
                profile_id="p2",
                layers=[
                    _make_layer(sand=50, clay=30, silt=20),
                    _make_layer(sand=50, clay=30, silt=20),
                    _make_layer(sand=50, clay=30, silt=20),
                ],
            ),  # 3 layers
        }
        # Deliberately introduce one violation so the texture
        # check fires (the metadata block is computed only when
        # the check runs).
        soil[0].layers[0] = _make_layer(sand=80, clay=80, silt=80)
        check = _texture_check(_make_unified(soil=soil))
        details = check["details"]
        self.assertEqual(details["n_layers_per_cell_typical"], 3)

    def test_depth_range_description_present(self):
        """Depth-range-description renders the spans the soil
        profile carries; consumers append the unit at render time
        (m vs cm ambiguity is loader-side)."""
        soil = {0: _make_profile(
            profile_id="p0",
            layers=[
                _make_layer(sand=80, clay=80, silt=80,
                            depth_top=0.0, depth_bottom=0.2),
                _make_layer(sand=50, clay=30, silt=20,
                            depth_top=0.2, depth_bottom=0.5),
            ],
        )}
        check = _texture_check(_make_unified(soil=soil))
        details = check["details"]
        self.assertEqual(details["depth_range_description"], "0.0–0.5")

    def test_metadata_keys_complete(self):
        """The F16 contract pins six new keys on details. A future
        regression dropping any of them surfaces here rather than
        as a silent consumer null."""
        soil = {0: _make_profile(
            profile_id="p0",
            layers=[_make_layer(sand=80, clay=80, silt=80)],
        )}
        check = _texture_check(_make_unified(soil=soil))
        for key in (
            "n_flagged_cells",
            "n_flagged_layers",
            "n_total_layers",
            "n_multi_flagged_cells",
            "n_layers_per_cell_typical",
            "depth_range_description",
        ):
            self.assertIn(key, check["details"])


if __name__ == "__main__":
    unittest.main()
