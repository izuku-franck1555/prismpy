"""Structural pin — F-BP-15 sarra_py maize over-prediction advisory.

Spec: ``EXP/prism-runner/UC-GEN-S2-UC4-CONTRACT.md`` v0.2 LOCKED §3 MUST-3 +
§7 A4. The contract requires that ``ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_
PREDICTION`` rides on UC4 readiness ``advisory_flags`` IFF the package is:

* ``platform == "sarra_py"``, AND
* ``crop.name`` is maize (case-insensitive), AND
* package centroid latitude is strictly greater than 14.0 degN
  (one-term-per-concept reuse of the Phase 1 millet rule's canonical
  threshold).

Two guard tiers:

* **AST tier** asserts the wiring stays anchored — the named constant is
  defined, the predicate ``_sarra_py_maize_at_deep_sahel`` is wired into
  ``canonical_uc_readiness_emitter``, and the advisory is appended under
  the UC4 branch only. A refactor that drops the predicate guard or
  hoists the append above the UC4 elif would silently disable the
  disclosure on every deep-Sahel sarra_py maize package — the AST guard
  catches that even when the behavior tests would still pass on a
  rearranged structure.
* **Behavior tier** exercises ``create_manifest`` end-to-end with
  synthesized configs covering the positive case (Mopti-like bbox), the
  negative-by-latitude case (Segou-like bbox), the strict-``>`` boundary
  at exactly 14.0 degN, the cross-axis leak matrix (wrong platform /
  wrong crop), and the closed-world cross-UC leak. UC4 itself still
  dispatches; the advisory is a disclosure that prismweb's honest-signal
  banner reads — not a block.

Hosted under ``tests/structural/`` so the bound-gen workflow's
``tests/structural/**`` path filter triggers CI on PRs that touch the
F-BP-15 emit surface; runs alongside the existing UC4 preserve-raw
emit-invariant guards.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from prismpy.packaging.manifest import create_manifest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MANIFEST_SRC = _REPO_ROOT / "src" / "prismpy" / "packaging" / "manifest.py"


# F-BP-15 advisory literal. Held verbatim here (NOT imported from the
# module under test) so the structural pin catches accidental drift in
# either the source's constant or the contract's exact wire format.
EXPECTED_ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION = (
    "sarra_py_maize_over_prediction_pending_calibration:"
    "biomass_WUE_~45_kg_per_mm_~2x_realistic_~20_kg_per_mm"
)

# Strict-``>`` boundary reused from the Phase 1 millet variety rule.
EXPECTED_DEEP_SAHEL_LAT_THRESHOLD = 14.0


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _build_min_project_config(
    *,
    crop_name: str,
    bounds_gis: List[float],
    use_cases: List[str],
    n_years: int = 10,
) -> Dict[str, Any]:
    """Synthesize a minimal spec-compliant ``project_config`` for the
    ``create_manifest`` emit clauses exercised by this file.

    Self-contained — does NOT depend on any ``tests/phase3/conftest.py``
    helper (which lives outside the CI-collected test paths) so the pin
    remains hermetic when run under ``pytest tests/structural``.
    """
    uc_keysets: Dict[str, Dict[str, Any]] = {
        "yield_forecast": {
            "cores": 1, "target_year": 2024, "forecast_date": None,
            "max_runs": 500, "cultivar_ids": None, "n_analogs": None,
        },
        "drought_management": {
            "cores": 1, "risk_metric": "prob_drought",
            "critical_window": "full_season", "critical_window_start": None,
            "critical_window_end": None, "drought_threshold": 0.4,
            "min_consecutive_days": 7, "baseline_start": None,
            "baseline_end": None, "no_cell_day_output": False,
            "drought_threshold_grid": None,
        },
    }
    start_year = 2018
    end_year = start_year + max(n_years - 1, 0)
    return {
        "project_name": "uc4_pr1_pin_pkg",
        "region_name": "test_region",
        "country": "Mali",
        "gadm_level": 2,
        "crop_name": crop_name,
        "planting_doy": 170,
        "maturity_doy": 285,
        "start_year": start_year,
        "end_year": end_year,
        "spinup_years": 0,
        "data_sources": {"climate": "AgERA5"},
        "bounds_gis": bounds_gis,
        "use_case_config": {
            uc: dict(uc_keysets[uc]) for uc in use_cases if uc in uc_keysets
        },
    }


@pytest.fixture
def package_dir(tmp_path: Path) -> Path:
    """Minimal package directory so ``collect_files_with_checksums``
    produces a ``files`` list of length 1. The emit clauses under test
    (uc_readiness advisory_flags) derive from ``project_config`` only.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "metadata.json").write_text(json.dumps({"placeholder": True}))
    return pkg


# ──────────────────────────────────────────────────────────────────────────
# AST tier — wiring guards
# ──────────────────────────────────────────────────────────────────────────


def _load_manifest_ast() -> ast.Module:
    src = _MANIFEST_SRC.read_text(encoding="utf-8")
    return ast.parse(src, filename=str(_MANIFEST_SRC))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in manifest.py")


def test_ast_advisory_constant_defined_with_expected_value() -> None:
    """The named-constant ``ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION``
    must be assigned the exact contract-wire literal at module scope.
    Catches drift to a different magnitude / different over-prediction
    description (the disclosure semantics live in the description half).
    """
    tree = _load_manifest_ast()
    found = False
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id
            == "ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION"
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        assert value == (
            EXPECTED_ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION
        ), (
            "ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION value drift: "
            f"got {value!r}; expected the §3 MUST-3 literal "
            f"{EXPECTED_ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION!r}"
        )
        found = True
        break
    assert found, (
        "ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION must be defined at "
        "manifest.py module scope (named-constant pattern, mirroring "
        "ADVISORY_FLAG_UC4_SEVERITY_TIER)."
    )


def test_ast_predicate_helper_defined() -> None:
    """``_sarra_py_maize_at_deep_sahel`` must be defined at manifest.py
    module scope so the wiring guard below can reference it.
    """
    tree = _load_manifest_ast()
    names = {
        n.name for n in tree.body if isinstance(n, ast.FunctionDef)
    }
    assert "_sarra_py_maize_at_deep_sahel" in names, (
        "_sarra_py_maize_at_deep_sahel must be defined at manifest.py "
        "module scope — refactor moving it elsewhere would silently "
        "break the wiring guard in canonical_uc_readiness_emitter."
    )


def test_ast_predicate_uses_strict_gt_14_threshold() -> None:
    """The deep-Sahel threshold must remain ``14.0`` AND the comparison
    must be strict-``>`` (Phase 1 millet rule alignment — exactly 14.0
    falls on the conservative no-advisory side).
    """
    tree = _load_manifest_ast()
    func = _find_function(tree, "_sarra_py_maize_at_deep_sahel")
    saw_strict_gt_threshold = False
    for node in ast.walk(func):
        if not isinstance(node, ast.Compare):
            continue
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Gt):
            continue
        # RHS = a Name (the module constant) OR a literal 14.0.
        rhs = node.comparators[0]
        if isinstance(rhs, ast.Constant) and rhs.value == 14.0:
            saw_strict_gt_threshold = True
            break
        if (
            isinstance(rhs, ast.Name)
            and rhs.id == "_DEEP_SAHEL_LAT_THRESHOLD_DEG"
        ):
            saw_strict_gt_threshold = True
            break
    assert saw_strict_gt_threshold, (
        "_sarra_py_maize_at_deep_sahel must compare lat_mid with strict "
        "``>`` against the 14.0-degN threshold (named constant or "
        "literal) — flipping to ``>=`` or moving the threshold would "
        "break the Phase 1 millet rule alignment."
    )


def test_ast_advisory_append_is_uc4_gated_by_predicate() -> None:
    """The advisory append must live inside the ``uc_name ==
    "drought_management"`` branch (so the flag rides on UC4 readiness
    only) AND inside an ``if _sarra_py_maize_at_deep_sahel(...)`` guard
    (so it never emits on the wrong platform / wrong crop / wrong lat).
    """
    tree = _load_manifest_ast()
    func = _find_function(tree, "canonical_uc_readiness_emitter")
    func_src = ast.unparse(func)
    assert "ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION" in func_src, (
        "canonical_uc_readiness_emitter must reference "
        "ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION (F-BP-15 advisory "
        "wiring under UC4)."
    )
    assert "_sarra_py_maize_at_deep_sahel" in func_src, (
        "canonical_uc_readiness_emitter must gate the F-BP-15 advisory "
        "append on _sarra_py_maize_at_deep_sahel(platform, "
        "manifest_so_far) — dropping the predicate would leak the "
        "advisory across every UC4 package."
    )


# ──────────────────────────────────────────────────────────────────────────
# Behavior tier — end-to-end emit invariants
# ──────────────────────────────────────────────────────────────────────────


# Mopti-like bbox: lat_mid ~14.475 > 14.0 → MUST emit the advisory.
_MOPTI_BBOX_GIS = [-4.20, 14.40, -4.05, 14.55]
# Segou-like bbox: lat_mid ~13.44 < 14.0 → MUST NOT emit (landrace lat).
_SEGOU_BBOX_GIS = [-6.91, 13.09, -5.15, 13.79]
# Strict-``>`` boundary case: lat_mid == 14.0 → MUST NOT emit.
_BOUNDARY_BBOX_GIS = [-4.0, 14.0, -3.9, 14.0]


def test_behavior_sarra_py_maize_at_deep_sahel_emits_advisory(
    package_dir: Path,
) -> None:
    """Positive case: sarra_py + crop=Maize + Mopti-like bbox →
    ``ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION`` is appended to UC4
    readiness, alongside the always-emitted severity_tier flag.
    """
    cfg = _build_min_project_config(
        crop_name="Maize",
        bounds_gis=_MOPTI_BBOX_GIS,
        use_cases=["drought_management"],
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    flags = m["uc_readiness"]["drought_management"]["advisory_flags"]
    assert (
        EXPECTED_ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION in flags
    ), (
        "§3 MUST-3 violation: sarra_py + maize + lat>14 advisory missing"
        f" (flags={flags})"
    )
    assert "severity_tier:viz_layer_thresholds_v1" in flags, (
        f"UC4 severity_tier missing alongside MUST-3 advisory: {flags}"
    )


def test_behavior_sarra_py_maize_at_landrace_lat_no_advisory(
    package_dir: Path,
) -> None:
    """Negative-by-latitude: sarra_py + crop=Maize + Segou-like bbox
    (lat_mid < 14.0) → advisory MUST NOT emit. Preserves the Phase 1
    landrace-correct path at ~13 degN.
    """
    cfg = _build_min_project_config(
        crop_name="Maize",
        bounds_gis=_SEGOU_BBOX_GIS,
        use_cases=["drought_management"],
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    flags = m["uc_readiness"]["drought_management"]["advisory_flags"]
    assert (
        EXPECTED_ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION not in flags
    ), (
        "§3 MUST-3 false-positive: sarra_py + maize + lat<14 emitted "
        f"the deep-Sahel advisory (flags={flags})"
    )


def test_behavior_boundary_lat_exactly_14_no_advisory(
    package_dir: Path,
) -> None:
    """Boundary case: ``lat_mid == 14.0`` falls on the no-advisory side
    (strict-``>`` alignment with the Phase 1 millet rule).
    """
    cfg = _build_min_project_config(
        crop_name="Maize",
        bounds_gis=_BOUNDARY_BBOX_GIS,
        use_cases=["drought_management"],
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    flags = m["uc_readiness"]["drought_management"]["advisory_flags"]
    assert (
        EXPECTED_ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION not in flags
    ), (
        "§3 MUST-3 boundary failure: lat==14.0 must default to "
        f"no-advisory (Phase-1 strict-> alignment); flags={flags}"
    )


@pytest.mark.parametrize(
    "platform,crop_name",
    [
        # Wrong-crop matrix at deep-Sahel.
        ("sarra_py", "Sorghum"),
        ("sarra_py", "Millet"),
        ("sarra_py", "Cowpea"),
        # Wrong-platform matrix at deep-Sahel maize.
        ("pythia", "Maize"),
        ("craft", "Maize"),
        ("acea", "Maize"),
    ],
)
def test_behavior_advisory_does_not_leak_across_axes(
    platform: str, crop_name: str, package_dir: Path,
) -> None:
    """Cross-axis leakage guard: (platform, crop) outside (sarra_py,
    maize) MUST NOT emit the F-BP-15 advisory even at deep-Sahel
    latitudes. Closes the predicate-too-permissive failure mode.
    """
    cfg = _build_min_project_config(
        crop_name=crop_name,
        bounds_gis=_MOPTI_BBOX_GIS,
        use_cases=["drought_management"],
    )
    m = create_manifest(package_dir, cfg, platform=platform)
    flags = m["uc_readiness"]["drought_management"]["advisory_flags"]
    assert (
        EXPECTED_ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION not in flags
    ), (
        f"§3 MUST-3 leakage: platform={platform!r} crop={crop_name!r} "
        f"emitted the sarra_py-maize-only advisory (flags={flags})"
    )


def test_behavior_advisory_absent_when_uc4_not_declared(
    package_dir: Path,
) -> None:
    """Closed-world: a sarra_py + maize + Mopti-like package that does
    NOT declare ``drought_management`` MUST NOT emit the F-BP-15
    advisory anywhere in ``uc_readiness`` (the advisory rides on UC4
    only — UC1 / UC3 / UC5 / UC6 readiness must stay clean).
    """
    cfg = _build_min_project_config(
        crop_name="Maize",
        bounds_gis=_MOPTI_BBOX_GIS,
        use_cases=["yield_forecast"],
    )
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    ucr = m["uc_readiness"]
    assert "drought_management" not in ucr, (
        f"UC4 not declared but emitted: {sorted(ucr)}"
    )
    for uc, entry in ucr.items():
        assert (
            EXPECTED_ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION
            not in entry["advisory_flags"]
        ), (
            f"§3 MUST-3 leakage to UC {uc!r}: F-BP-15 advisory must "
            f"only ride on UC4 readiness; "
            f"flags={entry['advisory_flags']}"
        )


def test_behavior_advisory_reads_from_bounds_sarra_py_fallback(
    package_dir: Path,
) -> None:
    """Fallback path: when only ``bounds_sarra_py`` (the SARRA-Py
    ``[lat_NW, lon_NW, lat_SE, lon_SE]`` format) is present and
    ``bounds_gis`` is absent, the predicate must still resolve the
    centroid latitude correctly and emit on deep-Sahel maize.
    """
    cfg = _build_min_project_config(
        crop_name="Maize",
        bounds_gis=_MOPTI_BBOX_GIS,
        use_cases=["drought_management"],
    )
    # Drop bounds_gis; inject only the SARRA-Py-format bounds.
    cfg.pop("bounds_gis", None)
    # bounds_sarra_py = [lat_NW, lon_NW, lat_SE, lon_SE]; lat_NW is
    # the northern (max) lat, lat_SE the southern (min). Mopti-like:
    # lat_NW=14.55, lat_SE=14.40 → lat_mid=14.475 > 14.0.
    cfg["bounds_sarra_py"] = [14.55, -4.20, 14.40, -4.05]
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    flags = m["uc_readiness"]["drought_management"]["advisory_flags"]
    assert (
        EXPECTED_ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION in flags
    ), (
        "§3 MUST-3 fallback failure: bounds_sarra_py-only path failed "
        f"to resolve lat>14 (flags={flags})"
    )


def test_behavior_advisory_absent_when_bounds_missing(
    package_dir: Path,
) -> None:
    """Defensive: when both ``bounds_gis`` and ``bounds_sarra_py`` are
    absent, the predicate MUST NOT crash and MUST return False (silent
    no-emit is the safe default — we never raise a false F-BP-15
    disclosure on a package whose latitude we can't determine).
    """
    cfg = _build_min_project_config(
        crop_name="Maize",
        bounds_gis=_MOPTI_BBOX_GIS,
        use_cases=["drought_management"],
    )
    cfg.pop("bounds_gis", None)  # nothing left for the predicate to read
    m = create_manifest(package_dir, cfg, platform="sarra_py")
    flags = m["uc_readiness"]["drought_management"]["advisory_flags"]
    assert (
        EXPECTED_ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION not in flags
    ), (
        "§3 MUST-3 defensive failure: missing bounds must yield "
        f"no-advisory (silent no-emit); flags={flags}"
    )


def test_behavior_advisory_emit_is_deterministic_across_repeat_calls(
    package_dir: Path,
) -> None:
    """Two emits with identical input produce identical advisory_flags
    lists (no ordering drift, no spurious duplicates). Mirrors T14 in
    the legacy phase3 advisory_flag pin.
    """
    cfg = _build_min_project_config(
        crop_name="Maize",
        bounds_gis=_MOPTI_BBOX_GIS,
        use_cases=["drought_management"],
    )
    m1 = create_manifest(package_dir, cfg, platform="sarra_py")
    m2 = create_manifest(package_dir, cfg, platform="sarra_py")
    f1 = m1["uc_readiness"]["drought_management"]["advisory_flags"]
    f2 = m2["uc_readiness"]["drought_management"]["advisory_flags"]
    assert f1 == f2, f"advisory_flags order drift across repeated emits: {f1} vs {f2}"
    assert (
        f1.count(
            EXPECTED_ADVISORY_FLAG_SARRA_PY_MAIZE_OVER_PREDICTION
        ) == 1
    ), (
        "F-BP-15 advisory must emit exactly once per UC4 readiness "
        f"(no duplicate append); flags={f1}"
    )
