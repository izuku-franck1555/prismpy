"""CRAFT cropland-vintage honesty.

CRAFT reads a verbatim ``spam_raster_path`` (it never calls ``resolve_spam_raster``), so:
  * **Silent-uniform fail-loud:** a SELECTED SPAM mask that cannot be honored must FAIL LOUD — never
    silently degrade to a uniform (all-100%) mask (the masquerade). Uniform stays legitimate
    ONLY when no mask was requested (``spam_raster_path is None``).
  * **Honest label:** derived from the ACTUAL raster basename via ``identify_vintage`` — never a
    hardcoded ``"SPAM 2020"``; an unrecognized basename fails loud rather than mislabel.

These pins are RED on the parent, which warned-and-uniformed on a missing raster and hardcoded
``crop_mask_source = "SPAM 2020"`` regardless of the actual file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prismpy.config.schema import (
    BoundaryConfig,
    BoundarySource,
    CraftConfig,
    CropCalendarConfig,
    CropConfig,
    ManualBoundsConfig,
    OutputConfig,
    Platform,
    PlatformConfigGroup,
    ProjectConfig,
    ProjectInfo,
    RegionConfig,
    TemporalConfig,
)
from prismpy.models.region import BoundingBox, Region
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.sources.crop_areas.spam_vintage import (
    SpamVintageError,
    VintageNotRegisteredError,
    VintageRasterAbsentError,
    identify_vintage,
)
from prismpy.translators.base import UnifiedData
from prismpy.translators.craft.translator import CraftTranslator

V2R2_NAME = "spam2020_V2r2_global_H_MAIZ_A.tif"
V2010_NAME = "spam2010V2r0_global_H_MAIZ_R.tif"


def _cfg(tmp, *, spam_raster_path=None) -> ProjectConfig:
    kwargs = dict(
        project=ProjectInfo(name="craft_vintage", description="vintage honesty"),
        region=RegionConfig(
            name="Koutiala", country="Mali", country_iso3="MLI",
            grid_resolution="5arcmin",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(minx=-6.0, miny=11.0, maxx=-4.0, maxy=13.0),
                inclusion_rule="bbox_intersects",
            ),
        ),
        crop=CropConfig(
            name="Maize", name_short="mai", variety="M",
            calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285),
        ),
        temporal=TemporalConfig(start_year=2015, end_year=2020, spinup_years=2),
        targets=[Platform.CRAFT],
        output=OutputConfig(base_dir=str(tmp), structure="by_platform"),
    )
    if spam_raster_path is not None:
        kwargs["platform_config"] = PlatformConfigGroup(
            craft=CraftConfig(spam_raster_path=Path(spam_raster_path))
        )
    return ProjectConfig(**kwargs)


def _grid() -> SpatialGrid:
    return SpatialGrid(
        resolution="5arcmin",
        cells=[
            GridCell(cell_id=0, lat=12.0, lon=-5.0, row=1, col=1),
            GridCell(cell_id=1, lat=12.1, lon=-5.1, row=2, col=2),
        ],
    )


# --------------------------------------------------------------------------- #
# identify_vintage — the registry-level basename→(year, release) identity      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "filename,expected",
    [
        ("spam2020_V2r2_global_H_MAIZ_A.tif", ("2020", "V2r2")),
        ("spam2010V2r0_global_H_ACOF_H.tif", ("2010", "V2r0")),
        ("/abs/dir/spam2020_V2r2_global_H_COFF_I.tif", ("2020", "V2r2")),
        ("spam2020_V2r0_global_H_MAIZ_A.tif", None),   # 2020/V2r0 is NOT a registered vintage
        ("spam2020_V2r2_global_H_ZZZZ_A.tif", None),   # code not in the 2020 inventory
        ("spam2020_V2r2_global_H_MAIZ_H.tif", None),   # H stratum absent in 2020 {A,I,R}
        ("spam2010V2r0_global_H_MAIZ_H.tif", ("2010", "V2r0")),  # H valid in 2010
        ("harvest_area_2020_V2r2.tif", None),          # clipped-mask name, not a source raster
        ("random.tif", None),
    ],
)
def test_identify_vintage(filename, expected):
    assert identify_vintage(filename) == expected


# --------------------------------------------------------------------------- #
# Silent-uniform fail-loud — a selected-but-unhonorable SPAM mask FAILS LOUD    #
# --------------------------------------------------------------------------- #
def test_selected_but_absent_raster_fails_loud(tmp_path):
    cfg = _cfg(tmp_path, spam_raster_path=tmp_path / "missing" / V2R2_NAME)
    tr = CraftTranslator(cfg)
    with pytest.raises(VintageRasterAbsentError, match="not on disk"):
        tr._generate_crop_mask(_grid(), None)


def test_unrecognized_basename_fails_loud(tmp_path):
    bad = tmp_path / "not_a_spam_vintage.tif"
    bad.write_bytes(b"stub")  # exists, but the basename matches no registered vintage
    cfg = _cfg(tmp_path, spam_raster_path=bad)
    tr = CraftTranslator(cfg)
    with pytest.raises(VintageNotRegisteredError, match="no registered cropland"):
        tr._generate_crop_mask(_grid(), None)


def test_spam_named_but_unregistered_basename_fails_loud(tmp_path):
    # The masquerade case: a file that LOOKS like SPAM (spam20xx_… naming) but is not a
    # registered provisioned vintage (2020/V2r0 here — only 2020/V2r2 + 2010/V2r0 are
    # registered) must fail loud, never be silently mislabelled. spam_raster_path is SPAM-only
    # by contract, so there is no custom-raster path to preserve.
    masq = tmp_path / "spam2020_V2r0_global_H_MAIZ_A.tif"
    masq.write_bytes(b"stub")  # exists + SPAM-named, but an unregistered vintage
    cfg = _cfg(tmp_path, spam_raster_path=masq)
    tr = CraftTranslator(cfg)
    with pytest.raises(VintageNotRegisteredError, match="no registered cropland"):
        tr._generate_crop_mask(_grid(), None)


def test_no_spam_path_keeps_uniform_mode(tmp_path):
    # The legitimate uniform path (no SPAM mask requested) MUST still work — no raise, and every
    # cell gets the default coverage. Regression pin for the not-a-masquerade case.
    cfg = _cfg(tmp_path, spam_raster_path=None)
    tr = CraftTranslator(cfg)
    (tr.output_dir / "crop_mask").mkdir(parents=True, exist_ok=True)  # pipeline pre-creates this
    mask_path = tr._generate_crop_mask(_grid(), None)
    content = Path(mask_path).read_text()
    data_lines = [ln for ln in content.splitlines() if ln and not ln.startswith("CellId")]
    assert len(data_lines) == 2                       # both cells present
    assert all(ln.endswith("\t1.0") for ln in data_lines)  # uniform default coverage


# --------------------------------------------------------------------------- #
# Honest label — derived from the basename, NOT a hardcoded "SPAM 2020"         #
# --------------------------------------------------------------------------- #
def _find_value(obj, key):
    """First value for `key` anywhere in a nested manifest dict/list, else None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_value(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_value(v, key)
            if found is not None:
                return found
    return None


def _manifest(tmp_path, spam_name):
    # Bind an APPLIED raster that exists on disk, then parse the emitted manifest JSON structurally.
    raster = tmp_path / spam_name
    raster.write_bytes(b"stub")
    tr = CraftTranslator(_cfg(tmp_path, spam_raster_path=raster))
    data = UnifiedData(
        region=Region(
            name="Koutiala", country="Mali", country_iso3="MLI",
            bounds=BoundingBox(minx=-6.0, miny=11.0, maxx=-4.0, maxy=13.0),
        ),
        grid=_grid(),
    )
    tr._generate_package_metadata(data, [])
    return json.loads((tr.output_dir / "manifest.json").read_text())


def test_honest_label_v2r2(tmp_path):
    # Exact manifest JSON field (not a text substring) for an applied V2r2 raster.
    assert _find_value(_manifest(tmp_path, V2R2_NAME), "crop_mask") == "SPAM 2020 V2r2"


def test_honest_label_2010_v2r0(tmp_path):
    # A 2010 raster is labelled 2010 — never the parent's hardcoded "SPAM 2020".
    assert _find_value(_manifest(tmp_path, V2010_NAME), "crop_mask") == "SPAM 2010 V2r0"


# --------------------------------------------------------------------------- #
# Extraction fail-loud — a selected mask that cannot be READ, or that covers    #
# fewer cells than requested, must RAISE (never silently default-fill 100%).    #
# RED on parent 714abec: it returned {} on rasterio-missing and let the         #
# truncating zip drop cells.                                                    #
# --------------------------------------------------------------------------- #
class _FakeShortSrc:
    """rasterio-dataset stand-in whose .sample() returns FEWER values than coords."""

    height = 10
    width = 10
    res = (0.0833, 0.0833)

    def __init__(self, keep):
        self._keep = keep

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sample(self, coords):
        coords = list(coords)
        return [np.array([1.0], dtype=float) for _ in coords[: self._keep]]


def test_rasterio_missing_fails_loud(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "rasterio", None)  # `import rasterio` -> ImportError
    tr = CraftTranslator(_cfg(tmp_path, spam_raster_path=tmp_path / V2R2_NAME))
    with pytest.raises(SpamVintageError, match="rasterio is not installed"):
        tr._extract_crop_mask_from_spam(list(_grid().cells), tmp_path / V2R2_NAME)


def test_partial_extraction_fails_loud(tmp_path, monkeypatch):
    import rasterio
    cells = list(_grid().cells)  # 2 cells; sample returns 1 -> partial
    monkeypatch.setattr(rasterio, "open", lambda *a, **k: _FakeShortSrc(keep=len(cells) - 1))
    tr = CraftTranslator(_cfg(tmp_path, spam_raster_path=tmp_path / V2R2_NAME))
    with pytest.raises(SpamVintageError, match="partially"):
        tr._extract_crop_mask_from_spam(cells, tmp_path / V2R2_NAME)


def test_empty_extraction_fails_loud(tmp_path, monkeypatch):
    import rasterio
    cells = list(_grid().cells)  # sample returns 0 -> empty
    monkeypatch.setattr(rasterio, "open", lambda *a, **k: _FakeShortSrc(keep=0))
    tr = CraftTranslator(_cfg(tmp_path, spam_raster_path=tmp_path / V2R2_NAME))
    with pytest.raises(SpamVintageError, match="partially"):
        tr._extract_crop_mask_from_spam(cells, tmp_path / V2R2_NAME)
