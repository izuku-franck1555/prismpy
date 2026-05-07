"""Sprint S AC-8 — canonical Gate B for the Bénoué-Cameroon-Sorghum scenario.

This is the load-bearing acceptance test of Sprint S per durable
lesson §25. Bénoué (northern Cameroon, ~8.5°N 13.8°E) is the
country that failed under the legacy bundled-eGHR flow because
``data/eghr/CM.SOL`` was never bundled — the fix is the per-package
canonical substrate produced by :func:`build_eghr_substrate`.

The four contract criteria from Sprint S AC-8 (the locked acceptance
list in the dispatch):

1. ``eGHR/CM.SOL`` exists with ≥ 1 ``*<name>`` profile block.
2. ``eGHR/GHR.db`` exists with ≥ 1 row in ``profile_map``.
3. ``raster/soil.tif`` exists with non-zero pixel IDs (every cell
   resolves to a profile id; no orphan pixels).
4. Zero "SOL file not found" / "raster path not configured" /
   fallback-warning entries in the run log (the canonical path
   was used; no legacy global-DB fallback fired).

Companion reproduction snippet at
``prismpy/.local/AC-8-BENOUE-REPRODUCTION-SNIPPET.py`` (gitignored)
runs the same four assertions against any delivered package
directory; that snippet is the user-runnable Gate B per durable
lesson §25.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Dict, List

import pytest
import rasterio

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
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.provenance.tracker import ProvenanceTracker
from prismpy.translators.base import UnifiedData
from prismpy.translators.pythia.translator import PythiaTranslator


# Patterns that flag the legacy bundled-file fallback path. The
# canonical substrate run must NOT emit any of these.
_FALLBACK_WARNING_PATTERNS: List[re.Pattern] = [
    re.compile(r"SOL file not found"),
    re.compile(r"eGHR raster path not configured"),
    re.compile(r"GHR\.db not found"),
    re.compile(r"Soil raster not generated"),
    re.compile(r"falling back to region country"),
    re.compile(r"Per-package eGHR substrate missing"),
    re.compile(r"Canonical eGHR substrate inputs unavailable"),
    re.compile(r"PRISMPY_DISABLE_CANONICAL_EGHR"),
]


def _build_benoue_config(output_dir: Path) -> ProjectConfig:
    """Project config for a Bénoué-Cameroon-Sorghum run."""
    return ProjectConfig(
        project=ProjectInfo(
            name="ac8_benoue_canonical_gate_b",
            description=(
                "AC-8 Bénoué-Cameroon-Sorghum canonical Gate B (durable §25 "
                "user-snippet acceptance). The country that failed today."
            ),
        ),
        region=RegionConfig(
            name="Bénoué",
            country="Cameroon",
            country_iso3="CMR",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=13.5,
                    miny=8.0,
                    maxx=14.5,
                    maxy=9.0,
                ),
            ),
        ),
        crop=CropConfig(
            name="Sorghum",
            name_short="sgh",
            variety="Medium-duration",
            calendar=CropCalendarConfig(
                planting_doy=166,
                maturity_doy=285,
            ),
        ),
        temporal=TemporalConfig(
            start_year=2015,
            end_year=2015,
            spinup_years=0,
        ),
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir=str(output_dir), structure="by_platform"),
    )


def _build_benoue_grid() -> SpatialGrid:
    """A 6 × 6 grid covering the Bénoué bounding box at 10-arcmin resolution.

    The Bénoué bbox (13.5-14.5 E, 8.0-9.0 N) is 1° × 1°. A 6 × 6 grid
    at increment 1/6 ° ≈ 10 arcmin gives 36 cells, enough to
    exercise the substrate writer's per-cell + per-profile dedup
    logic while staying lightweight in the test suite.
    """
    cells: List[GridCell] = []
    cell_id = 0
    increment = 1.0 / 6.0
    for row in range(6):
        for col in range(6):
            cells.append(
                GridCell(
                    cell_id=cell_id,
                    lat=8.0 + (row + 0.5) * increment,
                    lon=13.5 + (col + 0.5) * increment,
                    row=row,
                    col=col,
                    resolution="custom",
                )
            )
            cell_id += 1
    return SpatialGrid(
        resolution="custom",
        cells=cells,
        increment_deg=increment,
        bounds=BoundingBox(minx=13.5, miny=8.0, maxx=14.5, maxy=9.0),
    )


def _build_benoue_profiles() -> Dict[int, SoilProfile]:
    """Per-cell Bénoué profiles — three distinct soil types over 36 cells.

    The grid has 36 cells; this fixture assigns profiles such that
    most cells share a profile (typical for HWSD2-derived inputs
    where soil polygons span multiple cells). Three distinct
    profiles let the substrate exercise dedup, raster id assignment,
    and SOL block emission in one fixture.
    """
    # Three Bénoué-relevant soil types: clayey ferruginous, sandy
    # loam, and a mid-range loam — representative of the West-African
    # savanna soils that Bénoué Sorghum production runs on.
    clayey = SoilLayer(
        depth_top=0.0,
        depth_bottom=0.2,
        sand=20.0,
        clay=50.0,
        silt=30.0,
        organic_carbon=0.6,
        bulk_density=1.35,
        ph=6.4,
        field_capacity=0.32,
        wilting_point=0.20,
    )
    sandy = SoilLayer(
        depth_top=0.0,
        depth_bottom=0.2,
        sand=78.0,
        clay=10.0,
        silt=12.0,
        organic_carbon=0.2,
        bulk_density=1.55,
        ph=6.2,
        field_capacity=0.16,
        wilting_point=0.06,
    )
    loamy = SoilLayer(
        depth_top=0.0,
        depth_bottom=0.2,
        sand=42.0,
        clay=25.0,
        silt=33.0,
        organic_carbon=0.4,
        bulk_density=1.45,
        ph=6.5,
        field_capacity=0.24,
        wilting_point=0.12,
    )

    profiles: Dict[int, SoilProfile] = {}
    for cell_id in range(36):
        if cell_id < 12:
            layer = clayey
        elif cell_id < 24:
            layer = sandy
        else:
            layer = loamy
        profiles[cell_id] = SoilProfile(
            profile_id=f"BNU_{cell_id}",
            lat=8.0 + (cell_id // 6 + 0.5) / 6.0,
            lon=13.5 + (cell_id % 6 + 0.5) / 6.0,
            source="hwsd2",
            layers=[layer],
        )
    return profiles


def _build_benoue_unified_data() -> UnifiedData:
    region = Region(
        name="Bénoué",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=13.5, miny=8.0, maxx=14.5, maxy=9.0),
    )
    return UnifiedData(
        region=region,
        grid=_build_benoue_grid(),
        soil=_build_benoue_profiles(),
    )


def test_ac8_benoue_canonical_gate_b(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Sprint S AC-8 — Bénoué-Cameroon-Sorghum canonical Gate B end-to-end smoke.

    Builds the eGHR substrate via the PYTHIA translator's canonical
    path and asserts the four acceptance criteria from the Sprint S
    AC-8 contract:

    1. ``eGHR/CM.SOL`` exists with ≥ 1 ``*<name>`` profile block.
    2. ``eGHR/GHR.db`` exists with ≥ 1 row in ``profile_map``.
    3. ``raster/soil.tif`` exists with non-zero pixel IDs.
    4. Zero fallback-warning entries in the run log.

    Each criterion is asserted with an explicit failure message
    naming the contract clause so a regression chase localizes
    directly.
    """
    config = _build_benoue_config(tmp_path)
    provenance = ProvenanceTracker(
        enabled=True,
        output_dir=tmp_path,
        project_name="ac8_benoue_canonical_gate_b",
    )
    translator = PythiaTranslator(
        config=config,
        output_dir=str(tmp_path),
        provenance=provenance,
        prefer_canonical_substrate=True,
    )
    data = _build_benoue_unified_data()

    with caplog.at_level(logging.WARNING):
        eghr_dir = translator._include_eghr_data(data)

    # ------------------------------------------------------------------
    # Contract criterion 1: eGHR/CM.SOL exists with ≥ 1 *<name> block.
    # ------------------------------------------------------------------
    sol_path = tmp_path / "eGHR" / "CM.SOL"
    assert sol_path.exists(), (
        "AC-8 contract criterion 1: eGHR/CM.SOL must exist after canonical "
        f"substrate build for Bénoué-Cameroon-Sorghum. Path: {sol_path}"
    )
    sol_blocks = [
        line
        for line in sol_path.read_text().splitlines()
        if line.startswith("*") and not line.startswith("*SOILS:")
    ]
    assert len(sol_blocks) >= 1, (
        "AC-8 contract criterion 1: eGHR/CM.SOL must contain at least one "
        f"'*<name>' profile block; found 0 in {sol_path}."
    )

    # ------------------------------------------------------------------
    # Contract criterion 2: eGHR/GHR.db exists with ≥ 1 profile_map row.
    # ------------------------------------------------------------------
    db_path = tmp_path / "eGHR" / "GHR.db"
    assert db_path.exists(), (
        "AC-8 contract criterion 2: eGHR/GHR.db must exist after canonical "
        f"substrate build. Path: {db_path}"
    )
    with sqlite3.connect(db_path) as conn:
        row_count = int(conn.execute("SELECT COUNT(*) FROM profile_map").fetchone()[0])
    assert row_count >= 1, (
        "AC-8 contract criterion 2: profile_map must contain at least one "
        f"row; found {row_count} in {db_path}."
    )

    # ------------------------------------------------------------------
    # Contract criterion 3: raster/soil.tif exists with non-zero pixel IDs.
    # ------------------------------------------------------------------
    raster_path = tmp_path / "raster" / "soil.tif"
    assert raster_path.exists(), (
        "AC-8 contract criterion 3: raster/soil.tif must exist after "
        f"canonical substrate build. Path: {raster_path}"
    )
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        nodata = src.nodata
    nonzero_pixels = int((band != nodata).sum()) if nodata is not None else int((band != 0).sum())
    total_pixels = int(band.size)
    assert nonzero_pixels >= 1, (
        "AC-8 contract criterion 3: raster/soil.tif must carry non-zero "
        f"pixel IDs; got {nonzero_pixels}/{total_pixels} non-zero pixels."
    )
    # Stronger pin — every cell that had a SoilProfile must resolve to a non-nodata pixel.
    assert nonzero_pixels >= len(_build_benoue_profiles()), (
        "AC-8 contract criterion 3: every input cell must resolve to a "
        f"non-nodata pixel; got {nonzero_pixels} non-nodata vs "
        f"{len(_build_benoue_profiles())} cells with profiles."
    )

    # ------------------------------------------------------------------
    # Contract criterion 4: zero fallback-warning entries in the run log.
    # ------------------------------------------------------------------
    fallback_warnings = [
        record
        for record in caplog.records
        if any(pat.search(record.getMessage()) for pat in _FALLBACK_WARNING_PATTERNS)
    ]
    assert not fallback_warnings, (
        "AC-8 contract criterion 4: canonical run must emit zero "
        "fallback-warning entries (no SOL file not found / raster path not "
        "configured / falling-back-to-region-country / etc). Captured:\n"
        + "\n".join(f"  [{r.levelname}] {r.getMessage()}" for r in fallback_warnings)
    )

    # Sanity: substrate dispatcher returned the eGHR directory.
    assert eghr_dir == tmp_path / "eGHR", (
        f"_include_eghr_data should return the eGHR directory; got {eghr_dir!r}."
    )

    # ------------------------------------------------------------------
    # Contract criterion 5 (Sprint S Gate-B-FIX): provenance.json carries
    # the dedicated eghr_substrate_decision field set to "canonical". This
    # is the load-bearing source-of-truth signal added to close the
    # b5fb6538 false-PASS loop — downstream consumers (the AC-8
    # reproduction snippet, the evaluator's Gate B verifier) read this
    # field directly rather than inferring the dispatch decision from
    # secondary signals like presence-of-CM.SOL or absence-of-fallback-
    # warnings (durable §24 canonical-source-or-pin).
    # ------------------------------------------------------------------
    assert provenance.record.eghr_substrate_decision == "canonical", (
        "AC-8 contract criterion 5 (Sprint S Gate-B-FIX): "
        "provenance.record.eghr_substrate_decision must be 'canonical' "
        f"after the canonical-path dispatch; got "
        f"{provenance.record.eghr_substrate_decision!r}."
    )
    assert provenance.record.eghr_substrate_reason == "ok", (
        "AC-8 contract criterion 5 (Sprint S Gate-B-FIX): "
        "provenance.record.eghr_substrate_reason must be 'ok' on the "
        f"canonical happy path; got {provenance.record.eghr_substrate_reason!r}."
    )

    # Belt-and-suspenders: serialize the record to JSON via to_dict()
    # and re-read the keys; the AC-8 reproduction snippet reads the
    # serialized JSON, so the field MUST survive serialization.
    serialized = provenance.record.to_dict()
    assert serialized.get("eghr_substrate_decision") == "canonical", (
        "AC-8 contract criterion 5 serialization: provenance.json must "
        "carry 'eghr_substrate_decision' == 'canonical' at top level "
        f"after to_dict(); got {serialized.get('eghr_substrate_decision')!r}."
    )
    assert serialized.get("eghr_substrate_reason") == "ok", (
        "AC-8 contract criterion 5 serialization: provenance.json must "
        "carry 'eghr_substrate_reason' == 'ok' at top level after "
        f"to_dict(); got {serialized.get('eghr_substrate_reason')!r}."
    )
