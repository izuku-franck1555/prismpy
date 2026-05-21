"""Producer-boundary nodata gate on iSDA per-cell reads.

`_retrieve_isda_api_for_grid` MUST consult `src.nodata` before
scaling raster values so the uint8 sentinel (commonly 255) never
leaks through `× scale` as physically-impossible substrate
(e.g. pH = 25.5). Downstream None routes the cell through the
HWSD fallback cascade.

Negative-case mutations: drop the `nodata is not None` guard OR
remove the `== nodata` comparison on either band → tests FAIL.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prismpy.pipeline.executor import TranslationPipeline

_EXECUTOR = (
    Path(__file__).resolve().parents[2]
    / "src" / "prismpy" / "pipeline" / "executor.py"
)

# Mirrors the gate expression from `_retrieve_isda_api_for_grid` so
# the behavioural cases below evaluate the exact contract. Pinned by
# `test_gate_expression_matches_source` against the live source.
_GATE_EXPR = (
    "(None if (nodata is not None and raw == nodata) "
    "else float(raw) * scale)"
)


def _function_body_source(text: str, name: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"function {name!r} not found in executor.py")


def _nodata_comparisons(body: str) -> list[ast.Compare]:
    return [
        node for node in ast.walk(ast.parse(body))
        if isinstance(node, ast.Compare)
        and any(
            isinstance(comp, ast.Name) and comp.id == "nodata"
            for comp in [node.left, *node.comparators]
        )
        and any(isinstance(op, ast.Eq) for op in node.ops)
    ]


def test_isda_retrieval_reads_src_nodata() -> None:
    body = _function_body_source(
        _EXECUTOR.read_text(encoding="utf-8"),
        "_retrieve_isda_api_for_grid",
    )
    assert "src.nodata" in body, (
        "_retrieve_isda_api_for_grid MUST read src.nodata before "
        "scaling raster values."
    )


def test_isda_retrieval_gates_both_band_reads_against_nodata() -> None:
    body = _function_body_source(
        _EXECUTOR.read_text(encoding="utf-8"),
        "_retrieve_isda_api_for_grid",
    )
    assert "nodata is not None" in body, (
        "Missing `nodata is not None` guard."
    )
    comparisons = _nodata_comparisons(body)
    assert len(comparisons) >= 2, (
        "Expected both band reads (b1_raw + b2_raw) to be gated "
        f"against nodata via `== nodata`; found {len(comparisons)} "
        "comparison(s). Asymmetric gating leaks the unguarded band."
    )


def test_gate_expression_matches_source() -> None:
    """The behavioural cases below evaluate `_GATE_EXPR`; this pin
    anchors that expression to the live source so a future
    expression edit either updates the test or fails fast."""
    body = _function_body_source(
        _EXECUTOR.read_text(encoding="utf-8"),
        "_retrieve_isda_api_for_grid",
    )
    canonical_b1 = (
        "b1 = None if nodata is not None and b1_raw == nodata "
        "else float(b1_raw) * scale"
    )
    canonical_b2 = (
        "b2 = None if nodata is not None and b2_raw == nodata "
        "else float(b2_raw) * scale"
    )
    assert canonical_b1 in body, (
        "iSDA b1 gate expression drifted from canonical "
        "`None if nodata is not None and b1_raw == nodata "
        "else float(b1_raw) * scale`."
    )
    assert canonical_b2 in body, (
        "iSDA b2 gate expression drifted from canonical pattern."
    )


@pytest.mark.parametrize(
    "b1_raw, b2_raw, want_b1, want_b2",
    [
        # band-1 nodata: b1 returns None, b2 scales normally.
        (255, 50, None, 5.0),
        # band-2 nodata: b1 scales normally, b2 returns None.
        (60, 255, 6.0, None),
        # both nodata: both return None.
        (255, 255, None, None),
    ],
    ids=["band-1-nodata", "band-2-nodata", "both-nodata"],
)
def test_gate_behaviour_for_synthetic_raster(
    b1_raw, b2_raw, want_b1, want_b2,
):
    """Evaluate the gate expression with `nodata=255` and `scale=0.1`
    (mirrors iSDA pH per `_retrieve_isda_api_for_grid` PROPERTIES
    table). Asserts each band's sentinel value yields None while
    non-sentinel values scale through normally."""
    nodata = 255
    scale = 0.1
    b1 = eval(  # noqa: S307 — gate-expression contract verification
        _GATE_EXPR, {"nodata": nodata, "raw": b1_raw, "scale": scale},
    )
    b2 = eval(  # noqa: S307 — see above
        _GATE_EXPR, {"nodata": nodata, "raw": b2_raw, "scale": scale},
    )
    assert b1 == want_b1
    assert b2 == want_b2


def test_gate_passes_through_when_src_nodata_is_none():
    """`nodata is not None` short-circuits so rasters without a
    declared nodata value bypass the gate and scale every read.
    Covers tier-1 1km caches that may omit nodata metadata."""
    b1 = eval(  # noqa: S307
        _GATE_EXPR, {"nodata": None, "raw": 50, "scale": 0.1},
    )
    assert b1 == 5.0


@patch("rasterio.open")
def test_mocked_raster_drops_sentinel_cell_from_profiles(mock_open):
    """Belt-and-braces integration: with a fake rasterio context
    that returns 255 (nodata sentinel) for every band, the cell's
    sand_content stays None and the cell drops out of the iSDA
    profiles dict — routing it to the HWSD fallback cascade."""
    pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")

    fake_src = MagicMock()
    fake_src.nodata = 255
    fake_src.index = MagicMock(return_value=(0, 0))
    fake_src.read = MagicMock(return_value=[[255]])
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=fake_src)
    ctx.__exit__ = MagicMock(return_value=False)
    mock_open.return_value = ctx

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.logger = MagicMock()
    pipeline.provenance = MagicMock()
    pipeline.provenance.enabled = False
    pipeline._region_in_isda_coverage = MagicMock(return_value=True)
    pipeline._ensure_isda_1km_cache = MagicMock(return_value=None)
    pipeline.config = MagicMock()
    pipeline.config.data_sources.cache_dir = None

    cell = MagicMock(cell_id=42, lat=10.0, lon=5.0)
    grid = MagicMock(cells=[cell])
    region = MagicMock(name="TestRegion")
    region.bounds.to_gis_format = MagicMock(return_value="bbox")

    result = pipeline._retrieve_isda_api_for_grid(grid, region)
    # Every band returns the sentinel → profiles dict is empty.
    assert result is None or 42 not in (result or {}), (
        f"Sentinel-only raster MUST NOT produce an iSDA profile "
        f"for the cell; got {result!r}."
    )
