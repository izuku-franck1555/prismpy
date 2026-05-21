"""Producer-boundary nodata gate on iSDA per-cell reads.

`_retrieve_isda_api_for_grid` MUST consult `src.nodata` before
scaling raster values so the uint8 sentinel (commonly 255) never
leaks through `× scale` as physically-impossible substrate
(e.g. pH = 25.5). Downstream None routes the cell through the
HWSD fallback cascade.

Negative-case: revert the gate (drop the `nodata is not None`
comparison) → this test FAILS.
"""
from __future__ import annotations

import ast
from pathlib import Path

_EXECUTOR = (
    Path(__file__).resolve().parents[2]
    / "src" / "prismpy" / "pipeline" / "executor.py"
)


def _function_body_source(text: str, name: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"function {name!r} not found in executor.py")


def test_isda_retrieval_reads_src_nodata() -> None:
    body = _function_body_source(
        _EXECUTOR.read_text(encoding="utf-8"),
        "_retrieve_isda_api_for_grid",
    )
    assert "src.nodata" in body, (
        "_retrieve_isda_api_for_grid MUST read src.nodata before "
        "scaling raster values (iSDA uint8 sentinel × scale leak fix)."
    )


def test_isda_retrieval_gates_band_reads_against_nodata() -> None:
    body = _function_body_source(
        _EXECUTOR.read_text(encoding="utf-8"),
        "_retrieve_isda_api_for_grid",
    )
    assert "nodata is not None" in body, (
        "_retrieve_isda_api_for_grid MUST guard band reads with "
        "`nodata is not None` so the per-cell scale only runs on "
        "non-sentinel values."
    )
    assert body.count("== nodata") >= 1, (
        "_retrieve_isda_api_for_grid MUST compare each band read "
        "against the nodata sentinel before applying scale."
    )
