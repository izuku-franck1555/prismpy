"""F-DL Pin DL-2 — SARRA-Py path-dict + int-keyed soil regression.

Reproduces the production bug pattern: SARRA-Py's
``_load_climate_data`` populates ``unified_data.climate`` as a
path-dict (``{"rainfall_dir": Path(...), "agera5_dir": Path(...),
"metadata": {...}}``) while the harmonize / placeholder layer
populates ``unified_data.soil`` with int keys. Without the
F-DL filter, ``write_observed_values_json`` does
``sorted(set(climate.keys()) | set(soil.keys()))`` on a mixed-
type set and crashes with ``TypeError: '<' not supported between
instances of 'int' and 'str'``.

The two scenarios pin the writer's two-shape semantics:

Scenario A — mixed: path-dict climate + int-keyed soil (the
production bug surface). After the F-DL fix, the writer skips
the path-dict keys and emits a sidecar whose ``cells_block``
contains the int-keyed soil cells with null climate aggregates.

Scenario B — pure path-dict + empty soil. The writer's
``cells_block`` is empty (no real-integral cell IDs anywhere)
but the sidecar's metadata block is still populated, preserving
the honest-signal floor.

Per F-DL contract §D Pin DL-2.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismpy.config.schema import Platform
from prismpy.cockpit.observed_values_writer import write_observed_values_json
from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators.base import UnifiedData


def _real_soil_profile(profile_id: str, lat: float, lon: float) -> SoilProfile:
    """Build a real SoilProfile with one realistic layer. Helper keeps
    the test bodies focused on the path-dict / int-key mix-and-match
    rather than the per-layer chemistry."""
    layer = SoilLayer(
        depth_top=0.0,
        depth_bottom=0.20,
        sand=42.0,
        clay=18.0,
        silt=40.0,
        organic_carbon=0.6,
        bulk_density=1.45,
        ph=6.4,
        field_capacity=0.27,
        wilting_point=0.11,
        saturated_wc=0.42,
    )
    return SoilProfile(
        profile_id=profile_id,
        lat=lat,
        lon=lon,
        source="iSDA",
        layers=[layer],
        total_depth=0.20,
        metadata={},
    )


def _minimal_region() -> Region:
    return Region(
        name="Test-Maradi",
        country="Niger",
        country_iso3="NER",
        bounds=BoundingBox(minx=7.0, miny=13.0, maxx=8.0, maxy=14.0),
    )


def _minimal_grid() -> SpatialGrid:
    """Trivial 1-cell grid sufficient for the writer's emit path.
    The writer reads cell IDs from climate / soil, not the grid; the
    grid is only needed because UnifiedData carries it as a non-
    optional attribute."""
    return SpatialGrid(
        resolution="5arcmin",
        cells=[
            GridCell(
                cell_id=0,
                lat=13.5,
                lon=7.5,
                row=0,
                col=0,
            ),
        ],
        bounds=BoundingBox(minx=7.0, miny=13.0, maxx=8.0, maxy=14.0),
    )


def _unified_data(climate: dict, soil: dict) -> UnifiedData:
    return UnifiedData(
        region=_minimal_region(),
        grid=_minimal_grid(),
        climate=climate,
        soil=soil,
        crop_params=None,
        crop_calendar={},
        metadata={"platform": Platform.SARRA_PY.value},
    )


def test_sarra_py_path_dict_climate_plus_int_keyed_soil_does_not_crash(
    tmp_path: Path,
) -> None:
    """Scenario A — the production bug surface. Climate is a SARRA-Py
    path-dict with str keys; soil carries two int-keyed
    ``SoilProfile`` entries. The pre-F-DL writer crashed here with
    ``TypeError: '<' not supported between instances of 'int' and
    'str'``. Post-F-DL: writer skips the str keys, emits a sidecar
    with the two int-keyed cells filled in soil-only (climate
    aggregates are null because SARRA-Py path-dicts carry no
    in-memory records)."""
    climate = {
        "rainfall_dir": Path("/tmp/synthetic-rainfall"),
        "agera5_dir": Path("/tmp/synthetic-agera5"),
        "metadata": {"source": "synthetic"},
    }
    soil = {
        0: _real_soil_profile("p0", 13.5, 7.5),
        1: _real_soil_profile("p1", 13.6, 7.6),
    }
    data = _unified_data(climate, soil)
    output_path = tmp_path / "cockpit_observed_values.json"
    result = write_observed_values_json(
        unified_data=data,
        crop_calendar=None,
        output_path=output_path,
    )
    assert result == output_path
    assert output_path.exists(), (
        "Writer must persist the sidecar even when the climate side "
        "is a path-dict; pre-F-DL the writer crashed before reaching "
        "the JSON dump"
    )
    payload = json.loads(output_path.read_text())
    cells_block = payload.get("cells", {})
    # cells_block keys are str-coerced (writer's L530-538) so we
    # match on the stringified int keys.
    assert set(cells_block.keys()) == {"0", "1"}, (
        f"Expected cells_block keyed by stringified int soil keys; "
        f"got {sorted(cells_block.keys())}"
    )
    for cell_key in ("0", "1"):
        cell = cells_block[cell_key]
        # Soil aggregates: present (real SoilProfile populated). The
        # writer emits a flat schema — soil keys land at top level
        # alongside the climate keys. We pin on a representative
        # soil aggregate that the writer always emits when soil is
        # not empty.
        assert cell.get("sand_rootzone_mean") == 42.0, (
            f"cell[{cell_key!r}] should have soil aggregates populated; "
            f"got {cell}"
        )
        assert cell.get("clay_rootzone_mean") == 18.0
        # Climate aggregates: null (path-dict carries no in-memory
        # records). Writer at L490-491 documents this fallback.
        # Pin on representative climate keys.
        assert cell.get("precip_growing_season_total") is None, (
            "Climate aggregates must be null when climate is a "
            "path-dict (no in-memory records)"
        )
        assert cell.get("tmax_growing_season_mean") is None


def test_pure_path_dict_climate_plus_empty_soil_emits_empty_cells_block(
    tmp_path: Path,
) -> None:
    """Scenario B — pure path-dict, no soil. Writer's ``cells_block``
    is empty (no real-integral cell IDs from either source), but the
    sidecar is still written with metadata so downstream consumers
    distinguish "empty by design" from "writer crashed silently"."""
    climate = {
        "rainfall_dir": Path("/tmp/synthetic-rainfall"),
        "agera5_dir": Path("/tmp/synthetic-agera5"),
        "metadata": {"source": "synthetic"},
    }
    soil: dict = {}
    data = _unified_data(climate, soil)
    output_path = tmp_path / "cockpit_observed_values.json"
    write_observed_values_json(
        unified_data=data,
        crop_calendar=None,
        output_path=output_path,
    )
    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload.get("cells", {}) == {}, (
        "cells_block must be empty when neither source produces real-"
        "integral cell IDs"
    )
    # Honest-signal floor: schema_version + metadata are still
    # present even when cells_block is empty.
    assert "schema_version" in payload
