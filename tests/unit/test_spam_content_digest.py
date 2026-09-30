"""The SPAM registry's identity layer: the canonical content digest of a harvested-area layer, the
pinned digest of every registered source, each crop's code in each vintage, and the crop triples an
ACEA package may declare. Expectations are computed here from the canonical encoding's definition."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from prismpy.sources.crop_areas import spam_vintage as sv
from prismpy.translators.acea.translator import ACEA_FAO_CODE_MAP, SPAM_CODE_MAP

ROOT = Path(__file__).resolve().parents[2]
ZIPS = {("2020", "V2r2"): Path.home() / "Downloads/spam2020V2r2_global_harvested_area.geotiff.zip",
        ("2010", "V2r0"): Path.home() / "Downloads/spam2010v2r0_global_harv_area.geotiff.zip"}


def _layer(path, band, *, nodata=np.nan, crs="EPSG:4326", count=1, **creation):
    transform = from_origin(-180.0, 90.0, 1 / 12, 1 / 12)
    with rasterio.open(path, "w", driver="GTiff", height=band.shape[0], width=band.shape[1], count=count,
                       dtype="float32", crs=crs, transform=transform, nodata=nodata, **creation) as ds:
        for i in range(1, count + 1):
            ds.write(band, i)
    return path


def _digest(path):
    with rasterio.open(path) as ds:
        return sv.content_digest(ds)


def _expected(path):
    """The canonical encoding, as defined: sha256(header ‖ 0x0A ‖ band as little-endian float32)."""
    with rasterio.open(path) as ds:
        band, nodata = ds.read(1), ds.nodata
        header = {"nodata": None if nodata is None else repr(float(nodata)),
                  "crs": "EPSG:4326", "transform": [repr(float(x)) for x in tuple(ds.transform)[0:6]],
                  "byteorder": "<", "dtype": "float32", "band_index": 1, "count": 1,
                  "shape": [ds.height, ds.width], "schema": "spam-content-digest/1"}
    text = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8") + b"\n" + band.astype("<f4").tobytes(order="C")).hexdigest()


def _band(seed=0, shape=(32, 32)):
    band = np.random.default_rng(seed).uniform(0, 500, shape).astype("float32")
    band[0, :3] = np.nan
    return band


# ── the canonical content digest ─────────────────────────────────────────────


@pytest.mark.parametrize("nodata", [np.nan, -3.4028234663852886e38, -1.0, None])
def test_the_digest_is_the_canonical_encoding(tmp_path, nodata):
    path = _layer(tmp_path / "a.tif", _band(), nodata=nodata)
    assert _digest(path) == _expected(path)


def test_the_digest_does_not_depend_on_how_the_file_is_encoded(tmp_path):
    band = _band()
    plain = _layer(tmp_path / "plain.tif", band)
    packed = _layer(tmp_path / "packed.tif", band, compress="deflate", predictor=3, tiled=True,
                    blockxsize=16, blockysize=16)
    assert plain.read_bytes() != packed.read_bytes()
    assert _digest(plain) == _digest(packed)


def test_the_sign_of_a_zero_and_a_nan_payload_change_the_digest(tmp_path):
    band = _band()
    band[5, 5] = 0.0
    negative_zero, other_nan = band.copy(), band.copy()
    negative_zero[5, 5] = -0.0
    other_nan.view(np.uint32)[0, 0] = 0x7FC00001            # a NaN with another payload
    assert np.isnan(other_nan[0, 0]) and other_nan.view(np.uint32)[0, 0] != band.view(np.uint32)[0, 0]
    digests = {_digest(_layer(tmp_path / f"{i}.tif", b)) for i, b in enumerate((band, negative_zero, other_nan))}
    assert len(digests) == 3


def test_a_value_change_and_a_shifted_origin_change_the_digest(tmp_path):
    band = _band()
    moved = tmp_path / "moved.tif"
    with rasterio.open(moved, "w", driver="GTiff", height=32, width=32, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(-179.5, 90.0, 1 / 12, 1 / 12), nodata=np.nan) as ds:
        ds.write(band, 1)
    changed = band.copy()
    changed[10, 10] += 1.0
    base = _digest(_layer(tmp_path / "base.tif", band))
    assert base != _digest(moved)
    assert base != _digest(_layer(tmp_path / "changed.tif", changed))


def test_a_layer_with_more_than_one_band_is_refused(tmp_path):
    with pytest.raises(sv.SpamVintageError, match="one band"):
        _digest(_layer(tmp_path / "two.tif", _band(), count=2))


def test_a_crs_without_an_authority_code_is_refused(tmp_path):
    crs = "+proj=longlat +a=6371000 +b=6371000 +no_defs"
    with pytest.raises(sv.SpamVintageError, match="authority"):
        _digest(_layer(tmp_path / "custom.tif", _band(), crs=crs))


# ── the pinned registry ──────────────────────────────────────────────────────


def test_every_registered_source_has_exactly_one_pinned_digest():
    expected = {(year, release, code, tech) for (year, release), spec in sv.SPAM_VINTAGES.items()
                for code in spec.crops for tech in "RIA"}
    assert len(expected) == 46 * 3 + 42 * 3 == 264
    assert set(sv.SPAM_CONTENT_DIGESTS) == set(sv.SPAM_CONTENT_HEADERS) == expected
    assert all(len(d) == 64 and int(d, 16) >= 0 for d in sv.SPAM_CONTENT_DIGESTS.values())
    for header in sv.SPAM_CONTENT_HEADERS.values():
        assert (header["schema"], header["count"], header["band_index"], header["byteorder"]) == (
            "spam-content-digest/1", 1, 1, "<")
    assert sv.SPAM_CONTENT_HEADERS[("2020", "V2r2", "MILL", "R")]["nodata"] == "nan"


def test_the_default_vintage_is_2020_v2r2():
    assert sv.DEFAULT_SPAM_VINTAGE == ("2020", "V2r2") and sv.DEFAULT_SPAM_VINTAGE in sv.SPAM_VINTAGES


@pytest.mark.parametrize("vintage, code, tech", [(("2020", "V2r2"), "MILL", "R"), (("2010", "V2r0"), "MAIZ", "A")])
def test_the_pinned_digest_is_the_provisioned_source(vintage, code, tech):
    archive = ZIPS[vintage]
    if not archive.is_file():
        pytest.skip(f"the provisioned archive {archive.name} is not on this machine")
    name = sv.SPAM_VINTAGES[vintage].pattern.format(code=code, tech=tech)
    (member,) = [m for m in zipfile.ZipFile(archive).namelist() if Path(m).name == name]
    with rasterio.open(f"/vsizip/{archive}/{member}") as ds:
        assert sv.content_digest(ds) == sv.SPAM_CONTENT_DIGESTS[(*vintage, code, tech)]


# ── crop codes and the canonical triples ─────────────────────────────────────


@pytest.mark.parametrize("code, vintage, spelled", [
    ("SMIL", ("2020", "V2r2"), "MILL"), ("MILL", ("2010", "V2r0"), "SMIL"),
    ("ACOF", ("2020", "V2r2"), "COFF"), ("COFF", ("2010", "V2r0"), "ACOF"),
    ("MAIZ", ("2010", "V2r0"), "MAIZ"), ("maiz", ("2020", "V2r2"), "MAIZ"),
    ("ZZZZ", ("2020", "V2r2"), "ZZZZ"),
])
def test_each_crop_code_is_spelled_as_the_vintage_spells_it(code, vintage, spelled):
    assert sv.code_for_vintage(code, *vintage) == spelled


def test_a_code_in_an_unregistered_vintage_is_refused():
    with pytest.raises(sv.VintageNotRegisteredError):
        sv.code_for_vintage("MAIZ", "2020", "V2r0")


def test_a_code_the_vintage_does_not_map_is_refused_when_resolved(tmp_path):
    code = sv.code_for_vintage("ZZZZ", "2020", "V2r2")
    with pytest.raises(sv.CropNotInVintageError):
        sv.resolve_spam_raster(tmp_path, "2020", "V2r2", code, "R")


@pytest.mark.parametrize("vintage", sorted(sv.SPAM_VINTAGES))
def test_the_canonical_triples_are_every_acea_crop_with_a_spam_code(vintage):
    triples = sv.acea_canonical_triples(*vintage)
    assert set(triples) == {(name, ACEA_FAO_CODE_MAP[name], sv.code_for_vintage(SPAM_CODE_MAP[name], *vintage))
                            for name in ACEA_FAO_CODE_MAP if name in SPAM_CODE_MAP}
    assert len(triples) == len(set(triples)) == len(set(SPAM_CODE_MAP) & set(ACEA_FAO_CODE_MAP))
    finger = "MILL" if vintage == ("2020", "V2r2") else "SMIL"
    assert {t for t in triples if t[1] == 79} == {("Millet", 79, "PMIL"), ("Pearl Millet", 79, "PMIL"),
                                                ("Finger Millet", 79, finger)}
    assert ("Maize", 56, "MAIZ") in triples and not any(t[0] == "Maize" and t[1] != 56 for t in triples)


# ── the registry ships in the wheel ──────────────────────────────────────────


def test_an_installed_prismpy_loads_the_pinned_digests(tmp_path):
    # Built from a copy of the sources alone, so a build/ directory left by an earlier build cannot
    # supply the data file the package data must declare.
    source = tmp_path / "source"
    shutil.copytree(ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, source / name)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir",
                    "--wheel-dir", str(tmp_path / "wheel"), str(source)],
                   check=True, capture_output=True, text=True, timeout=240)
    (wheel,) = (tmp_path / "wheel").glob("prismpy-*.whl")
    site = tmp_path / "site"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir", "--target", str(site),
                    str(wheel)], check=True, capture_output=True, text=True, timeout=240)
    probe = ("import json, sys; from importlib import resources; "
             "from prismpy.sources.crop_areas import spam_vintage as sv; "
             "text = resources.files('prismpy.sources.crop_areas').joinpath('spam_content_digests.json').read_text(); "
             "print(json.dumps([sv.__file__, len(sv.SPAM_CONTENT_DIGESTS), text]))")
    out = subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, text=True, timeout=120,
                         cwd=tmp_path, env={"PYTHONPATH": str(site), "PATH": "/usr/bin:/bin"})
    module, n, text = json.loads(out.stdout.strip().splitlines()[-1])
    assert Path(module).resolve().is_relative_to(site.resolve())
    assert n == 264
    assert text == (ROOT / "src/prismpy/sources/crop_areas/spam_content_digests.json").read_text()
