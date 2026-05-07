"""Structural pin: AC-G-7c SARRA-Py GeoTIFF projection-climate path.

Sprint G AC-G-7c: SARRA-Py's per-variable climate writer accepts the
``ClimateKind`` discriminator. The OBSERVED path (preexisting) copies
TAMSAT/AgERA5 GeoTIFFs OR writes per-cell NetCDFs from
ClimateTimeSeries data. The PROJECTION path takes a
``Dict[variable_name, xr.DataArray]`` (output of AC-G-2 cached_cutout
+ AC-G-7d calendar conversion) and writes per-variable GeoTIFF
directories with one sidecar ``.meta.json`` per directory.

Per CC-G-7 + AC-G-13 deliverable hash precondition: GeoTIFF byte-
identity requires explicit pinning of crs (EPSG:4326), dtype
(float32), nodata (-9999.0), compress ('lzw'), tiled=True with
explicit blockxsize/blockysize, and GDAL_PAM_ENABLED=NO via
``rasterio.Env`` scope.

Tests:

* §1 ``ProjectionClimateMeta`` `variable` field exercise (already
  covered partially in test_acea_projection_pickle; this file adds
  the SARRA-Py-side pins).
* §2 SARRA-Py PROJECTION emits per-variable GeoTIFF directories.
* §3 Sidecar .meta.json contents per directory.
* §4 GeoTIFF byte-identity determinism.
* §5 PROJECTION without projection_meta raises ValueError.
* §6 OBSERVED path unchanged (default kind preserves backward compat).
* §7 Sibling-sweep — SARRA-Py imports canonical helpers.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import xarray as xr

from prismpy.harmonize.climate_kind import ClimateKind
from prismpy.models.scenario import (
    BiasCorrectionMethod,
    ProjectionClimateMeta,
)


# ── Fixture builders ─────────────────────────────────────────────────


def _make_projection_data_array(
    variable: str = "tasmax",
    n_days: int = 3,
    n_lat: int = 16,
    n_lon: int = 16,
) -> xr.DataArray:
    """Build a synthetic xarray DataArray with time/lat/lon dims and
    ISIMIP3b-shape values."""
    import datetime as dt

    dates = [
        dt.date(2046, 6, 1) + dt.timedelta(days=i)
        for i in range(n_days)
    ]
    # Niamey area: roughly 13-15°N, 1-3°E at 0.5° resolution; the
    # 16×16 default lets the writer's tiled-output branch fire (tiled
    # output requires block dims as multiples of 16; smaller fixtures
    # would route to stripped output instead).
    lats = np.linspace(13.0, 14.0, n_lat)  # ascending
    lons = np.linspace(1.5, 3.0, n_lon)
    # Synthetic deterministic values
    values = np.arange(n_days * n_lat * n_lon, dtype="float32").reshape(
        n_days, n_lat, n_lon
    )
    return xr.DataArray(
        values,
        coords={
            "time": np.array([np.datetime64(d, "D") for d in dates]),
            "lat": lats,
            "lon": lons,
        },
        dims=("time", "lat", "lon"),
        name=variable,
    )


def _instantiate_minimal_sarra_py(tmp_path: Path):
    from prismpy.translators.sarra_py.translator import SarraPyTranslator

    inst = SarraPyTranslator.__new__(SarraPyTranslator)
    inst.output_dir = tmp_path / "pkg"
    (inst.output_dir / "data" / "climate").mkdir(parents=True, exist_ok=True)
    inst.provenance = None
    return inst


def _valid_meta_kwargs() -> dict:
    return {
        "gcm_source": "gfdl-esm4",
        "bias_correction_method": BiasCorrectionMethod.QUANTILE_MAPPING,
        "time_slice_start": 2046,
        "time_slice_end": 2065,
    }


# ── §1 ProjectionClimateMeta.variable field exercise ─────────────────


def test_projection_meta_variable_field_optional() -> None:
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())
    assert meta.variable is None


def test_projection_meta_variable_field_accepts_climate_var_name() -> None:
    meta = ProjectionClimateMeta(**_valid_meta_kwargs(), variable="tasmax")
    assert meta.variable == "tasmax"


def test_projection_meta_variable_field_rejects_empty_string() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ProjectionClimateMeta(**_valid_meta_kwargs(), variable="")


# ── §2 SARRA-Py PROJECTION emits per-variable GeoTIFF directories ────


def test_sarra_py_projection_emits_per_variable_directories(
    tmp_path: Path,
) -> None:
    inst = _instantiate_minimal_sarra_py(tmp_path)
    climate_by_var = {
        "tasmax": _make_projection_data_array("tasmax"),
        "pr": _make_projection_data_array("pr"),
    }
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())

    files = inst._generate_projection_climate_geotiffs(climate_by_var, meta)

    # Per-variable directories exist
    tasmax_dir = inst.output_dir / "data" / "climate" / "tasmax"
    pr_dir = inst.output_dir / "data" / "climate" / "pr"
    assert tasmax_dir.is_dir()
    assert pr_dir.is_dir()

    # Each directory has 3 days worth of GeoTIFFs + 1 sidecar
    tasmax_tifs = sorted(tasmax_dir.glob("*.tif"))
    pr_tifs = sorted(pr_dir.glob("*.tif"))
    assert len(tasmax_tifs) == 3
    assert len(pr_tifs) == 3

    tasmax_meta = tasmax_dir / ".meta.json"
    pr_meta = pr_dir / ".meta.json"
    assert tasmax_meta.exists()
    assert pr_meta.exists()


def test_sarra_py_projection_geotiff_filenames_are_dates(
    tmp_path: Path,
) -> None:
    inst = _instantiate_minimal_sarra_py(tmp_path)
    climate_by_var = {"tasmax": _make_projection_data_array("tasmax", n_days=2)}
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())

    inst._generate_projection_climate_geotiffs(climate_by_var, meta)

    tifs = sorted((inst.output_dir / "data" / "climate" / "tasmax").glob("*.tif"))
    assert tifs[0].stem == "2046-06-01"
    assert tifs[1].stem == "2046-06-02"


# ── §3 Sidecar contents per directory ────────────────────────────────


def test_sarra_py_projection_sidecar_carries_canonical_fields(
    tmp_path: Path,
) -> None:
    inst = _instantiate_minimal_sarra_py(tmp_path)
    climate_by_var = {"tasmax": _make_projection_data_array("tasmax")}
    meta = ProjectionClimateMeta(
        **_valid_meta_kwargs(), scenario_label="niamey-millet"
    )

    inst._generate_projection_climate_geotiffs(climate_by_var, meta)

    sidecar = inst.output_dir / "data" / "climate" / "tasmax" / ".meta.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["gcm_source"] == "gfdl-esm4"
    assert payload["bias_correction_method"] == "quantile_mapping"
    assert payload["time_slice_start"] == 2046
    assert payload["time_slice_end"] == 2065
    assert payload["variable"] == "tasmax"  # per-variable, NOT cell_id
    assert payload["scenario_label"] == "niamey-millet"
    # cell_id is omitted (None) for SARRA-Py per-variable sidecars
    assert "cell_id" not in payload


def test_sarra_py_projection_sidecar_validates_against_schema(
    tmp_path: Path,
) -> None:
    """Round-trip: write sidecar → re-parse → must validate against
    ProjectionClimateMeta."""
    inst = _instantiate_minimal_sarra_py(tmp_path)
    climate_by_var = {"pr": _make_projection_data_array("pr")}
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())

    inst._generate_projection_climate_geotiffs(climate_by_var, meta)

    sidecar = inst.output_dir / "data" / "climate" / "pr" / ".meta.json"
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    re_validated = ProjectionClimateMeta.model_validate(raw)
    assert re_validated.variable == "pr"


# ── §4 GeoTIFF byte-identity determinism ─────────────────────────────


def test_sarra_py_projection_geotiff_deterministic(tmp_path: Path) -> None:
    """CC-G-7 + AC-G-13: same input + same projection_meta → byte-
    identical GeoTIFF across two writer invocations."""
    inst_a = _instantiate_minimal_sarra_py(tmp_path / "a")
    inst_b = _instantiate_minimal_sarra_py(tmp_path / "b")
    climate_by_var = {"tasmax": _make_projection_data_array("tasmax")}
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())

    inst_a._generate_projection_climate_geotiffs(climate_by_var, meta)
    inst_b._generate_projection_climate_geotiffs(climate_by_var, meta)

    tifs_a = sorted((inst_a.output_dir / "data" / "climate" / "tasmax").glob("*.tif"))
    tifs_b = sorted((inst_b.output_dir / "data" / "climate" / "tasmax").glob("*.tif"))
    assert len(tifs_a) == len(tifs_b)
    for tif_a, tif_b in zip(tifs_a, tifs_b):
        assert tif_a.read_bytes() == tif_b.read_bytes(), (
            f"GeoTIFF byte drift: {tif_a.name} vs {tif_b.name}"
        )


def test_sarra_py_projection_sidecar_deterministic(tmp_path: Path) -> None:
    inst_a = _instantiate_minimal_sarra_py(tmp_path / "a")
    inst_b = _instantiate_minimal_sarra_py(tmp_path / "b")
    climate_by_var = {"tasmax": _make_projection_data_array("tasmax")}
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())

    inst_a._generate_projection_climate_geotiffs(climate_by_var, meta)
    inst_b._generate_projection_climate_geotiffs(climate_by_var, meta)

    sidecar_a = inst_a.output_dir / "data" / "climate" / "tasmax" / ".meta.json"
    sidecar_b = inst_b.output_dir / "data" / "climate" / "tasmax" / ".meta.json"
    assert sidecar_a.read_bytes() == sidecar_b.read_bytes()


def test_sarra_py_projection_geotiff_no_aux_xml_sidecar(
    tmp_path: Path,
) -> None:
    """``GDAL_PAM_ENABLED=NO`` env scope means no `.aux.xml` files
    appear next to the GeoTIFFs. Without the env-pin, GDAL would
    write `.aux.xml` sidecars on some platforms which would break
    byte-identity."""
    inst = _instantiate_minimal_sarra_py(tmp_path)
    climate_by_var = {"tasmax": _make_projection_data_array("tasmax")}
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())

    inst._generate_projection_climate_geotiffs(climate_by_var, meta)

    var_dir = inst.output_dir / "data" / "climate" / "tasmax"
    aux_files = list(var_dir.glob("*.aux.xml"))
    assert aux_files == [], (
        f"GDAL .aux.xml sidecars must not appear in projection output: {aux_files}"
    )


def test_sarra_py_projection_geotiff_metadata_pinned(tmp_path: Path) -> None:
    """Read the emitted GeoTIFF's metadata + assert canonical pins:
    crs=EPSG:4326, dtype=float32, nodata=-9999, compress=LZW, tiled=True
    when image dims are large enough (32x32 fixture lets the 16x16 tile
    grid actually subdivide; 16x16 fixture would report tiled=False on
    read-back because a single 16x16 tile in a 16x16 image is the same
    as stripped layout)."""
    import rasterio

    inst = _instantiate_minimal_sarra_py(tmp_path)
    climate_by_var = {
        "tasmax": _make_projection_data_array("tasmax", n_lat=32, n_lon=32)
    }
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())

    inst._generate_projection_climate_geotiffs(climate_by_var, meta)

    tif = next((inst.output_dir / "data" / "climate" / "tasmax").glob("*.tif"))
    with rasterio.open(tif) as src:
        assert src.crs.to_epsg() == 4326
        assert str(src.dtypes[0]) == "float32"
        assert src.nodata == -9999.0
        # Compression check
        compression = src.profile.get("compress", "").lower()
        assert compression == "lzw"
        # Tiled output (32x32 image with 16x16 blocks → 4 tiles)
        assert src.profile.get("tiled") is True


# ── §5 PROJECTION without projection_meta raises ─────────────────────


def test_sarra_py_projection_without_meta_raises(tmp_path: Path) -> None:
    inst = _instantiate_minimal_sarra_py(tmp_path)
    climate_by_var = {"tasmax": _make_projection_data_array("tasmax")}
    with pytest.raises(ValueError, match="projection_meta"):
        inst._generate_climate_files(
            climate_by_var,
            region=None,
            climate_kind=ClimateKind.PROJECTION,
            projection_meta=None,
        )


# ── §6 OBSERVED path unchanged + default kind preserves backward-compat ─


def test_sarra_py_default_kind_dispatches_to_observed(tmp_path: Path) -> None:
    """Default kind = OBSERVED; the existing path-dict / ClimateTimeSeries
    handling fires (NOT the projection helper)."""
    inst = _instantiate_minimal_sarra_py(tmp_path)
    # Pass an empty dict — observed dispatch returns empty list (no
    # rainfall_dir / agera5_dir keys; not a ClimateTimeSeries dict
    # either since values aren't TS objects).
    files = inst._generate_climate_files({}, region=None)
    assert files == []


def test_sarra_py_observed_path_dict_dispatch_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When OBSERVED + climate_data has 'rainfall_dir' key, the
    existing _copy_climate_geotiffs path is invoked (NOT the
    projection helper)."""
    inst = _instantiate_minimal_sarra_py(tmp_path)
    called: Dict[str, bool] = {"copy": False, "projection": False}

    def fake_copy(self_, climate):
        called["copy"] = True
        return []

    def fake_projection(self_, climate, meta):
        called["projection"] = True
        return []

    monkeypatch.setattr(
        type(inst), "_copy_climate_geotiffs", fake_copy
    )
    monkeypatch.setattr(
        type(inst),
        "_generate_projection_climate_geotiffs",
        fake_projection,
    )

    inst._generate_climate_files(
        {"rainfall_dir": str(tmp_path)},
        region=None,
        climate_kind=ClimateKind.OBSERVED,
    )
    assert called["copy"] is True
    assert called["projection"] is False


# ── §7 Sibling-sweep — SARRA-Py imports canonical helpers ────────────


def test_sarra_py_translator_imports_canonical_helpers() -> None:
    """Per durable §24: SARRA-Py writer must import ClimateKind +
    ProjectionClimateMeta from canonical sources."""
    import ast

    project_root = Path(__file__).resolve().parents[2]
    src = (
        project_root / "src/prismpy/translators/sarra_py/translator.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = {"ClimateKind": False, "ProjectionClimateMeta": False}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "prismpy.harmonize.climate_kind" and any(
                a.name == "ClimateKind" for a in node.names
            ):
                found["ClimateKind"] = True
            if node.module == "prismpy.models.scenario" and any(
                a.name == "ProjectionClimateMeta" for a in node.names
            ):
                found["ProjectionClimateMeta"] = True
    missing = [k for k, v in found.items() if not v]
    assert not missing, (
        f"SARRA-Py translator must import {missing} from canonical sources "
        "(durable §24 canonical-source-or-pin)."
    )


def test_sarra_py_writer_signature_accepts_climate_kind_keyword() -> None:
    """SARRA-Py's ``_generate_climate_files`` must accept the
    ``climate_kind`` + ``projection_meta`` keywords."""
    from prismpy.translators.sarra_py.translator import SarraPyTranslator

    sig = inspect.signature(SarraPyTranslator._generate_climate_files)
    assert "climate_kind" in sig.parameters
    assert "projection_meta" in sig.parameters
