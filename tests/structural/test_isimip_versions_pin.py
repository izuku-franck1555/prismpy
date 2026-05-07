"""Structural pin: ISIMIP3b 12-dimension version constants.

Sprint G CC-G-6 closes the byte-identical-reproducibility gap by
pinning every substrate dimension that could silently change under
us between an ISIMIP point release and our next regeneration. This
module is the source-side pin; the live drift-detection test (real
``client.datasets()`` API + mocked-fixture parity) lands alongside
``cached_cutout`` in the AC-G-2 commit.

For now we assert:

* The 12 named module-level constants exist with the expected types.
* ``CALENDAR_BY_GCM`` enumerates the 5 GCMs in the ISIMIP3b primary
  core ensemble with the right calendar string.
* ``VARIABLE_UNITS`` enumerates the 6 CF-1.x variables Sprint G ships.
* ``SCENARIO_PRODUCT_MAP`` covers ssp585 → InputData / ssp245 →
  SecondaryInputData.
* The derived rosters (``PRIMARY_GCMS``, ``SUPPORTED_VARIABLES``)
  agree with the master tables.
* ``ISIMIP3bClient`` and ``data_sources.isimip3b`` import the rosters
  from this module rather than redefining them — durable #24
  canonical-source-or-pin.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Mapping

from prismpy.standards import isimip_versions as iv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ── §1 12 named dimensions exist ─────────────────────────────────────


_EXPECTED_NAMES = (
    "SIMULATION_ROUND",
    "SCENARIO_PRODUCT_MAP",
    "BIAS_CORRECTION_VERSION",
    "REFERENCE_DATASET_VERSION",
    "CALENDAR_BY_GCM",
    "CF_VARIABLE_NAMING_CONVENTION",
    "VARIABLE_UNITS",
    "GRID_RESOLUTION_DEG",
    "EXTRACTION_METHOD",
    "MASK_FILL_CONVENTION",
    "LONGITUDE_CONVENTION",
    "NETCDF_FORMAT_ENGINE",
    "PRIMARY_GCMS",
    "SUPPORTED_VARIABLES",
    "SCENARIO_TIME_SLICES",
)


def test_isimip_versions_exposes_all_required_constants() -> None:
    missing = [name for name in _EXPECTED_NAMES if not hasattr(iv, name)]
    assert not missing, f"isimip_versions missing: {missing}"


def test_dunder_all_matches_required_set() -> None:
    declared = set(getattr(iv, "__all__", ()))
    missing = set(_EXPECTED_NAMES) - declared
    assert not missing, f"__all__ missing names: {sorted(missing)}"


# ── §2 simulation_round ──────────────────────────────────────────────


def test_simulation_round_pinned_to_isimip3b() -> None:
    assert iv.SIMULATION_ROUND == "ISIMIP3b"


# ── §3 scenario → product mapping ────────────────────────────────────


def test_scenario_product_map_covers_ssp585_and_ssp245() -> None:
    assert iv.SCENARIO_PRODUCT_MAP["ssp585"] == "InputData"
    assert iv.SCENARIO_PRODUCT_MAP["ssp245"] == "SecondaryInputData"


def test_scenario_product_map_does_not_silently_cover_other_ssps() -> None:
    # Sprint G scopes ssp245 + ssp585 only; adding ssp126/ssp370 is
    # an explicit Sprint H+ concern. A drift here would silently
    # widen scope without contract amendment.
    assert "ssp126" not in iv.SCENARIO_PRODUCT_MAP
    assert "ssp370" not in iv.SCENARIO_PRODUCT_MAP


# ── §4 bias-correction + reference dataset versions ──────────────────


def test_bias_correction_version_pinned() -> None:
    assert iv.BIAS_CORRECTION_VERSION == "ISIMIP3BASD v2.5.0"


def test_reference_dataset_version_pinned() -> None:
    assert iv.REFERENCE_DATASET_VERSION == "W5E5 v2.0"


# ── §5 calendar by GCM (5 GCMs in primary core ensemble) ─────────────


def test_calendar_by_gcm_enumerates_all_five_primary_gcms() -> None:
    expected = {
        "gfdl-esm4": "noleap",
        "ipsl-cm6a-lr": "gregorian",
        "mpi-esm1-2-hr": "gregorian",
        "mri-esm2-0": "gregorian",
        "ukesm1-0-ll": "360_day",
    }
    assert dict(iv.CALENDAR_BY_GCM) == expected


def test_primary_gcms_derived_from_calendar_table() -> None:
    """``PRIMARY_GCMS`` must be exactly the keys of ``CALENDAR_BY_GCM``
    — adding a 6th GCM requires updating both atomically."""
    assert iv.PRIMARY_GCMS == frozenset(iv.CALENDAR_BY_GCM.keys())
    assert len(iv.PRIMARY_GCMS) == 5


# ── §6 CF variable naming + per-variable units ───────────────────────


def test_cf_variable_naming_convention_pinned() -> None:
    assert iv.CF_VARIABLE_NAMING_CONVENTION == "CF-1.x"


def test_variable_units_enumerates_six_sprint_g_variables() -> None:
    expected = {
        "rsds": "W m-2",
        "tasmax": "K",
        "tasmin": "K",
        "pr": "kg m-2 s-1",
        "hurs": "%",
        "sfcWind": "m s-1",
    }
    assert dict(iv.VARIABLE_UNITS) == expected


def test_supported_variables_derived_from_units_table() -> None:
    assert iv.SUPPORTED_VARIABLES == frozenset(iv.VARIABLE_UNITS.keys())
    assert len(iv.SUPPORTED_VARIABLES) == 6


# ── §7 grid + extraction + mask + longitude ──────────────────────────


def test_grid_resolution_pinned() -> None:
    assert iv.GRID_RESOLUTION_DEG == "0.5 deg"


def test_extraction_method_pinned() -> None:
    assert iv.EXTRACTION_METHOD == "cutout_bbox"


def test_mask_fill_convention_pinned() -> None:
    assert "FillValue" in iv.MASK_FILL_CONVENTION
    assert "sea_mask" in iv.MASK_FILL_CONVENTION


def test_longitude_convention_pinned_to_minus_180_to_180() -> None:
    """West Africa bboxes use west < east in the -180..180 convention.
    A drift to 0..360 would silently mis-translate every Africa-region
    bbox at the boundary."""
    assert iv.LONGITUDE_CONVENTION == "-180_to_180"


# ── §8 netCDF format engine (xarray + netCDF4 + cftime) ──────────────


def test_netcdf_format_engine_includes_three_required_libraries() -> None:
    """LOW-Pass4-3 explicit pin: cftime version pinned alongside
    netcdf4 and xarray so AC-G-7d's calendar conversion semantics are
    deterministic across regenerations."""
    engine: Mapping[str, str] = iv.NETCDF_FORMAT_ENGINE
    for required in ("netcdf4", "xarray", "cftime"):
        assert required in engine, (
            f"NETCDF_FORMAT_ENGINE must pin {required!r}: got {dict(engine)!r}"
        )


# ── §9 scenario time-slice ensemble ──────────────────────────────────


def test_scenario_time_slices_covers_mid_century_and_end_century() -> None:
    assert (2046, 2065) in iv.SCENARIO_TIME_SLICES
    assert (2086, 2100) in iv.SCENARIO_TIME_SLICES
    assert len(iv.SCENARIO_TIME_SLICES) == 2


# ── §10 canonical-source discipline (durable #24) ────────────────────


def test_isimip3b_client_imports_rosters_from_canonical_source() -> None:
    """``prismpy.data_sources.isimip3b`` MUST import the GCM and variable
    rosters from ``prismpy.standards.isimip_versions`` rather than
    redefining them. A second, parallel definition is a durable #24
    canonical-source-or-pin violation."""
    src = (
        _project_root()
        / "src/prismpy/data_sources/isimip3b.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)

    imported_from_versions: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "prismpy.standards.isimip_versions"
        ):
            for alias in node.names:
                imported_from_versions.add(alias.name)

    required = {"PRIMARY_GCMS", "SCENARIO_PRODUCT_MAP", "SUPPORTED_VARIABLES"}
    missing = required - imported_from_versions
    assert not missing, (
        f"data_sources.isimip3b must import these from isimip_versions: "
        f"{sorted(missing)}"
    )

    # And the redefinitions MUST be gone — only the alias from the
    # ImportFrom may exist.
    redefined: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "_PRIMARY_GCMS",
                    "_SUPPORTED_VARIABLES",
                    "_SCENARIO_PRODUCT_MAP",
                }:
                    # Only allow the alias-rename pattern from
                    # ImportFrom (which uses ast.ImportFrom, not Assign).
                    redefined.append(target.id)

    assert redefined == [], (
        "data_sources/isimip3b.py must not redefine "
        f"the imported rosters at module scope; offenders: {redefined}"
    )
