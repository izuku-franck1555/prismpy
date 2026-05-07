"""Sprint S AC-6 — ``prefer_canonical_substrate`` flag wires the substrate builder into the translator.

The PYTHIA translator now exposes a ``prefer_canonical_substrate``
keyword on construction; ``True`` (default) routes
``_include_eghr_data`` through :func:`build_eghr_substrate` to
synthesize the per-package eGHR triple from the upstream-resolved
per-cell soil profiles, while ``False`` runs the legacy bundled-file
flow that copies the global GHR.db and per-country .SOL files. The
canonical path raises :class:`BuildEghrSubstrateError` when the
inputs the builder needs (grid + soil profiles) are absent rather
than silently dropping back to the legacy flow — matching the
project's honest-signal contract for substrate failures.
"""

from __future__ import annotations

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
from prismpy.translators.base import UnifiedData
from prismpy.translators.pythia.translator import (
    BuildEghrSubstrateError,
    PythiaTranslator,
)


def _build_project_config(output_dir: Path) -> ProjectConfig:
    return ProjectConfig(
        project=ProjectInfo(
            name="ac6_canonical_substrate_flag",
            description="AC-6 prefer_canonical_substrate dispatch test",
        ),
        region=RegionConfig(
            name="Bénoué",
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


def _build_profiles() -> Dict[int, SoilProfile]:
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
    }


def _build_unified_data() -> UnifiedData:
    """Minimal UnifiedData wiring the substrate builder's required inputs."""
    region = Region(
        name="Bénoué",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=2.0, miny=11.5, maxx=3.5, maxy=12.5),
    )
    return UnifiedData(
        region=region,
        grid=_build_grid_2x3(),
        soil=_build_profiles(),
    )


def test_default_prefer_canonical_substrate_is_true(tmp_path: Path) -> None:
    """The translator constructs with ``prefer_canonical_substrate=True`` by default."""
    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(config=config, output_dir=str(tmp_path))
    assert translator.prefer_canonical_substrate is True


def test_canonical_path_writes_full_substrate_triple_via_build_eghr_substrate(
    tmp_path: Path,
) -> None:
    """``prefer_canonical_substrate=True`` routes through ``build_eghr_substrate``.

    Asserts the three artifact files exist on disk after
    ``_include_eghr_data`` returns and that the returned directory
    points at the package's ``eGHR/`` subdirectory.
    """
    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(
        config=config,
        output_dir=str(tmp_path),
        prefer_canonical_substrate=True,
    )

    eghr_dir = translator._include_eghr_data(_build_unified_data())

    assert eghr_dir == tmp_path / "eGHR"
    assert (tmp_path / "raster" / "soil.tif").exists()
    assert (tmp_path / "eGHR" / "GHR.db").exists()
    assert (tmp_path / "eGHR" / "CM.SOL").exists()


def test_canonical_path_raises_typed_error_when_data_is_none(tmp_path: Path) -> None:
    """Substrate-build failures under canonical mode raise ``BuildEghrSubstrateError``.

    No ``data`` argument means no grid + no soil profiles to pass to
    the builder. The canonical path must fail loud rather than
    silently fall back to the legacy bundled flow.
    """
    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(
        config=config,
        output_dir=str(tmp_path),
        prefer_canonical_substrate=True,
    )

    with pytest.raises(BuildEghrSubstrateError):
        translator._include_eghr_data(None)


def test_canonical_path_raises_typed_error_when_grid_missing(tmp_path: Path) -> None:
    """``data.grid is None`` under canonical mode -> ``BuildEghrSubstrateError``."""
    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(
        config=config,
        output_dir=str(tmp_path),
        prefer_canonical_substrate=True,
    )

    region = Region(
        name="Bénoué",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=2.0, miny=11.5, maxx=3.5, maxy=12.5),
    )
    data = UnifiedData(region=region, grid=None, soil=_build_profiles())

    with pytest.raises(BuildEghrSubstrateError, match="populated grid"):
        translator._include_eghr_data(data)


def test_canonical_path_raises_typed_error_when_soil_profiles_missing(
    tmp_path: Path,
) -> None:
    """``data.soil`` empty under canonical mode -> ``BuildEghrSubstrateError``."""
    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(
        config=config,
        output_dir=str(tmp_path),
        prefer_canonical_substrate=True,
    )

    region = Region(
        name="Bénoué",
        country="Cameroon",
        country_iso3="CMR",
        bounds=BoundingBox(minx=2.0, miny=11.5, maxx=3.5, maxy=12.5),
    )
    data = UnifiedData(region=region, grid=_build_grid_2x3(), soil={})

    with pytest.raises(BuildEghrSubstrateError, match="non-empty dict"):
        translator._include_eghr_data(data)


def test_canonical_path_does_not_consult_global_eghr_database_path(
    tmp_path: Path,
) -> None:
    """Synthesized substrate ignores the legacy global ``eghr_database_path``.

    The canonical path builds the GHR.db from per-cell profiles; a
    deliberately broken sentinel on the legacy global path must not
    affect the produced artifacts.
    """
    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(
        config=config,
        output_dir=str(tmp_path),
        prefer_canonical_substrate=True,
    )
    if (
        translator.config.platform_config is not None
        and getattr(translator.config.platform_config, "pythia", None) is not None
    ):
        translator.config.platform_config.pythia.eghr_database_path = (
            "/dev/null/this-path-must-not-be-read.db"
        )

    eghr_dir = translator._include_eghr_data(_build_unified_data())

    assert eghr_dir == tmp_path / "eGHR"
    assert (tmp_path / "eGHR" / "GHR.db").exists()
    # The synthesized GHR.db must carry profile_map rows derived from
    # the per-cell profiles, not from any global database the legacy
    # path would have copied.
    import sqlite3

    with sqlite3.connect(tmp_path / "eGHR" / "GHR.db") as conn:
        rows = conn.execute("SELECT id, profile FROM profile_map").fetchall()
    assert len(rows) >= 1
    for _id, profile in rows:
        assert profile.startswith("CM"), (
            f"profile_map row {profile!r} should carry the CM country prefix "
            "the substrate built from region.country_iso3='CMR'."
        )


def test_legacy_path_skips_canonical_substrate_build(
    tmp_path: Path,
) -> None:
    """``prefer_canonical_substrate=False`` runs the legacy bundled flow.

    The legacy path is the existing copy-bundled-file flow. It does
    NOT call ``build_eghr_substrate`` — even when ``data`` is fully
    populated. Asserts via the absence of artifact files that
    ``build_eghr_substrate`` exclusively produces (the legacy flow
    needs ``pythia.eghr_database_path`` and ``pythia.eghr_sol_dir``
    set; with neither configured here, it returns None without
    side-effecting the substrate triple).
    """
    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(
        config=config,
        output_dir=str(tmp_path),
        prefer_canonical_substrate=False,
    )

    result = translator._include_eghr_data(_build_unified_data())

    # Legacy path with no eghr_database_path / eghr_sol_dir configured
    # returns None and writes no artifacts; the canonical path under
    # the same inputs would have raised BuildEghrSubstrateError because
    # it never silently falls back to the legacy flow.
    assert result is None
    assert not (tmp_path / "eGHR" / "GHR.db").exists()
    assert not (tmp_path / "eGHR" / "CM.SOL").exists()
