"""Sprint S AC-5 — `_get_required_country_codes` reads from the per-package eGHR substrate.

The PYTHIA translator's country-code resolution previously read the
clipped soil raster from the package output directory but the GHR.db
from a globally-bundled path on disk. When the global path was a
broken symlink (the Bénoué/Cameroon failure mode), the resolution
fell through to the ``region.country_iso3 -> iso2`` fallback, which
returned a country whose ``.SOL`` was never bundled, producing
"Copied 0 .SOL files for countries: ['CM']" downstream.

After AC-5, both reads target the same per-package substrate (raster
at ``output_dir/raster/soil.tif`` plus database at
``output_dir/eGHR/GHR.db``), so the country-code resolution can no
longer drift away from the substrate the package actually carries.

The fallback branch survives for edge cases (partial runs, tests
that exercise the helper without a built substrate); the happy path
where the substrate builder has run does not trigger it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pytest

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
from prismpy.translators._shared import build_eghr_substrate


def _build_project_config(output_dir: Path) -> ProjectConfig:
    """Minimal ``ProjectConfig`` wiring PYTHIA against the Bénoué bbox.

    The country is Cameroon (``CMR`` -> ``CM``) so a happy-path
    substrate-driven resolution returns ``{"CM"}`` and a fallback
    resolution also returns ``{"CM"}``; the differentiator between
    the two paths is the warning emission, not the return value.
    """
    return ProjectConfig(
        project=ProjectInfo(
            name="ac5_country_codes",
            description="AC-5 _get_required_country_codes substrate-driven test",
        ),
        region=RegionConfig(
            name="Test Region",
            country="Cameroon",
            country_iso3="CMR",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=2.0,
                    miny=11.5,
                    maxx=3.5,
                    maxy=12.5,
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


def _build_grid_2x3() -> SpatialGrid:
    cells = []
    cell_id = 0
    for row in range(2):
        for col in range(3):
            cells.append(
                GridCell(
                    cell_id=cell_id,
                    lat=12.0 - row * 0.5,
                    lon=2.0 + col * 0.5,
                    row=row,
                    col=col,
                    resolution="custom",
                )
            )
            cell_id += 1
    return SpatialGrid(
        resolution="custom",
        cells=cells,
        increment_deg=0.5,
        bounds=BoundingBox(minx=2.0, miny=11.0, maxx=3.5, maxy=12.5),
    )


def _build_profiles_5_cells_3_distinct() -> Dict[int, SoilProfile]:
    return {
        0: SoilProfile(
            profile_id="P0",
            lat=12.0,
            lon=2.0,
            source="hwsd2",
            layers=[
                SoilLayer(
                    depth_top=0.0,
                    depth_bottom=0.2,
                    sand=30.0,
                    clay=45.0,
                    silt=25.0,
                    organic_carbon=0.5,
                    bulk_density=1.4,
                    ph=6.5,
                    field_capacity=0.30,
                    wilting_point=0.18,
                ),
            ],
        ),
        1: SoilProfile(
            profile_id="P1",
            lat=12.0,
            lon=2.5,
            source="hwsd2",
            layers=[
                SoilLayer(
                    depth_top=0.0,
                    depth_bottom=0.3,
                    sand=82.0,
                    clay=8.0,
                    silt=10.0,
                    organic_carbon=0.2,
                    bulk_density=1.55,
                    ph=7.0,
                    field_capacity=0.15,
                    wilting_point=0.05,
                ),
            ],
        ),
        2: SoilProfile(
            profile_id="P2",
            lat=12.0,
            lon=3.0,
            source="hwsd2",
            layers=[
                SoilLayer(
                    depth_top=0.0,
                    depth_bottom=0.25,
                    sand=40.0,
                    clay=20.0,
                    silt=40.0,
                    organic_carbon=0.4,
                    bulk_density=1.45,
                    ph=6.7,
                    field_capacity=0.22,
                    wilting_point=0.12,
                ),
            ],
        ),
    }


def _build_translator(tmp_path: Path):
    from prismpy.translators.pythia.translator import PythiaTranslator

    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(config=config, output_dir=str(tmp_path))
    return translator


def test_country_codes_resolved_from_local_substrate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Happy path: per-package substrate exists -> resolver reads it cleanly.

    Builds the substrate under the translator's output directory then
    calls the helper; the resolver must return the substrate's country
    codes AND emit no "falling back to region country" warning.
    """
    region = Region(
        name="Test Region",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=2.0, miny=11.5, maxx=3.5, maxy=12.5),
    )
    build_eghr_substrate(
        grid=_build_grid_2x3(),
        profiles_by_cell=_build_profiles_5_cells_3_distinct(),
        country_code="CM",
        region=region,
        output_dir=tmp_path,
    )

    translator = _build_translator(tmp_path)

    with caplog.at_level(logging.WARNING):
        result = translator._get_required_country_codes()

    assert result == {"CM"}
    fallback_warnings = [
        record
        for record in caplog.records
        if "falling back to region country" in record.getMessage()
    ]
    assert not fallback_warnings, (
        "Happy-path substrate resolution emitted the fallback warning: "
        f"{[r.getMessage() for r in fallback_warnings]}. "
        "The fallback branch must not fire when the substrate is present."
    )


def test_country_codes_fallback_when_substrate_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Edge case: no substrate yet -> fallback emits a warning and returns
    the region's ISO 3166 alpha-2 code derived from ``country_iso3``.

    The fallback branch is preserved for partial-run / early-test
    edge cases. Asserting it still works keeps the helper functional
    in those contexts even after the canonical happy path moved off
    the bundled global GHR.db.
    """
    translator = _build_translator(tmp_path)

    with caplog.at_level(logging.WARNING):
        result = translator._get_required_country_codes()

    assert result == {"CM"}, (
        f"Fallback should derive 'CM' from region.country_iso3='CMR'; got {result}"
    )
    fallback_warnings = [
        record
        for record in caplog.records
        if "Per-package eGHR substrate missing" in record.getMessage()
    ]
    assert fallback_warnings, (
        "Substrate-missing path did not emit the expected fallback warning. "
        f"Captured warnings: {[r.getMessage() for r in caplog.records]}"
    )


def test_country_codes_does_not_consult_global_pythia_eghr_database_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The resolver no longer reads the bundled global ``eghr_database_path``.

    Building the per-package substrate gives the resolver everything
    it needs at the local path. A non-existent global path on the
    config must not cause a failure or an extra warning, and the
    resolver must still return the substrate's country codes.

    This pin protects against a regression that re-introduces the
    global-DB lookup as a "belt and braces" extra read; the local
    substrate is now the canonical source per durable §24.
    """
    region = Region(
        name="Test Region",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=2.0, miny=11.5, maxx=3.5, maxy=12.5),
    )
    build_eghr_substrate(
        grid=_build_grid_2x3(),
        profiles_by_cell=_build_profiles_5_cells_3_distinct(),
        country_code="CM",
        region=region,
        output_dir=tmp_path,
    )

    translator = _build_translator(tmp_path)
    # Plant a deliberately-broken sentinel on the (legacy) global path
    # so any code that tries to consult it would raise visibly.
    if (
        translator.config.platform_config is not None
        and getattr(translator.config.platform_config, "pythia", None) is not None
    ):
        translator.config.platform_config.pythia.eghr_database_path = (
            "/dev/null/this-path-must-not-be-read.db"
        )

    with caplog.at_level(logging.WARNING):
        result = translator._get_required_country_codes()

    assert result == {"CM"}
    # No "GHR.db not found at /dev/null/..." style warnings should appear,
    # because the resolver never consults the global path.
    bad_globals = [
        record
        for record in caplog.records
        if "this-path-must-not-be-read" in record.getMessage()
    ]
    assert not bad_globals, (
        "Resolver consulted the legacy global eghr_database_path: "
        f"{[r.getMessage() for r in bad_globals]}"
    )
