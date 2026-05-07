"""Sprint S AC-9 — ``PRISMPY_DISABLE_CANONICAL_EGHR=1`` environment-variable opt-out.

The PYTHIA translator's substrate-mode dispatch checks the
``PRISMPY_DISABLE_CANONICAL_EGHR`` environment variable and treats
``"1"`` as an explicit operator escape hatch: when set, the
dispatcher always routes to the legacy bundled-eGHR flow regardless
of the ``prefer_canonical_substrate`` constructor parameter.

Per the project's no-data-cooking contract, the dispatcher emits
a WARNING when the env-var is honored — operators must always see
an explicit log line indicating that the legacy path was taken
instead of the canonical default. The warning is the structural
pin asserting honest-signal behavior.
"""

from __future__ import annotations

import logging
import os
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
from prismpy.translators.pythia.translator import PythiaTranslator


_ENV_VAR = "PRISMPY_DISABLE_CANONICAL_EGHR"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the env-var starts unset for every test."""
    monkeypatch.delenv(_ENV_VAR, raising=False)


def _build_project_config(output_dir: Path) -> ProjectConfig:
    return ProjectConfig(
        project=ProjectInfo(
            name="ac9_disable_env_var",
            description="AC-9 PRISMPY_DISABLE_CANONICAL_EGHR escape-hatch test",
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
    }


def _build_unified_data() -> UnifiedData:
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


def test_env_var_unset_canonical_runs(tmp_path: Path) -> None:
    """With the env-var unset, the canonical dispatcher chooses the canonical path."""
    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(
        config=config,
        output_dir=str(tmp_path),
        prefer_canonical_substrate=True,
    )
    assert translator._canonical_substrate_will_run(_build_unified_data()) is True


def test_env_var_set_to_1_forces_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With ``PRISMPY_DISABLE_CANONICAL_EGHR=1``, the dispatcher refuses the canonical path.

    Even when ``prefer_canonical_substrate=True`` and every canonical
    input is present, the env-var takes precedence and routes to
    the legacy bundled flow. A WARNING log fires per the no-data-
    cooking contract — operators must always see explicit signal
    that the legacy path was taken.
    """
    monkeypatch.setenv(_ENV_VAR, "1")

    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(
        config=config,
        output_dir=str(tmp_path),
        prefer_canonical_substrate=True,
    )

    with caplog.at_level(logging.WARNING):
        result = translator._canonical_substrate_will_run(_build_unified_data())

    assert result is False
    disable_warnings = [
        record
        for record in caplog.records
        if "PRISMPY_DISABLE_CANONICAL_EGHR" in record.getMessage()
    ]
    assert disable_warnings, (
        "Dispatcher did not emit the PRISMPY_DISABLE_CANONICAL_EGHR=1 "
        "WARNING; the no-data-cooking contract requires explicit signal "
        "when the canonical path is bypassed."
    )


def test_env_var_other_values_do_not_disable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the literal string ``"1"`` triggers the escape hatch.

    Other non-empty values (``"true"``, ``"yes"``, ``"on"``) do NOT
    disable the canonical path; the contract is intentionally
    strict to prevent typos from silently dropping operators back
    to the legacy bundled flow.
    """
    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(
        config=config,
        output_dir=str(tmp_path),
        prefer_canonical_substrate=True,
    )

    for value in ("true", "yes", "on", "TRUE", "0", ""):
        monkeypatch.setenv(_ENV_VAR, value)
        assert translator._canonical_substrate_will_run(_build_unified_data()) is True, (
            f"Env-var value {value!r} should NOT disable the canonical path "
            "(only the literal '1' does)."
        )


def test_env_var_takes_precedence_over_constructor_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``PRISMPY_DISABLE_CANONICAL_EGHR=1`` overrides ``prefer_canonical_substrate=True``.

    The env-var is the operator escape hatch; the constructor
    parameter is the production default. When they conflict, the
    runtime escape hatch wins so a deployed binary can be coaxed
    to legacy mode without a code change.
    """
    monkeypatch.setenv(_ENV_VAR, "1")

    config = _build_project_config(tmp_path)
    translator = PythiaTranslator(
        config=config,
        output_dir=str(tmp_path),
        prefer_canonical_substrate=True,
    )

    with caplog.at_level(logging.WARNING):
        result = translator._canonical_substrate_will_run(_build_unified_data())

    assert result is False
    assert any(
        "PRISMPY_DISABLE_CANONICAL_EGHR" in record.getMessage()
        for record in caplog.records
    )
