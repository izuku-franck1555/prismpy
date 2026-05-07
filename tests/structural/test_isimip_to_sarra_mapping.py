"""Structural pins: ISIMIP → SARRA-Py canonical mapping + unit conversions.

Sprint G boundary 7/7 codex round 2 absorption per team-lead's
authorization 2026-05-07. Four pins enumerated:

a. Mapping completeness — every ISIMIP variable in the writer's
   emit set ⊆ ``ISIMIP_TO_SARRA_VAR_MAPPING`` keys.
b. Unit conversion correctness — K → K passthrough, kg m⁻² s⁻¹ →
   mm/day × 86400, W m⁻² → J m⁻²/day × 86400.
c. **Two-vocabulary structural pin** per
   ``feedback_two_vocabulary_substrate_drift.md`` + #227 pattern —
   AST walker assertion that the writer's POST-MAPPING output
   directory name set ⊆ the SARRA-Py consumer's expected directory
   set (lines 121-168 + 1755-1762 of ``sarra_py/translator.py``).
   This is the bug-class-closing pin: the same producer-vs-consumer
   vocabulary divergence that surfaced in #227 cockpit
   ``failed_checks[].category`` (Sprint E.0 enum vs V2-22c-PRE.1.2
   prefix taxonomy) now has a structural guard for the
   ISIMIP-vs-SARRA-Py case.
d. Sibling-sweep per durable §20 — every place in the SARRA-Py
   translator that reads / asserts climate directory names is
   walked + cross-referenced against the canonical mapping's
   target values.

Per durable §24 canonical-source-or-pin: the mapping itself lives at
:data:`prismpy.standards.isimip_versions.ISIMIP_TO_SARRA_VAR_MAPPING`
+ the conversion math at
:mod:`prismpy.harmonize.isimip_unit_conversions`. Every consumer
routes through these; the pins below assert no inline restatement +
no producer-vs-consumer drift.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Set

import numpy as np
import pytest

from prismpy.harmonize.isimip_unit_conversions import (
    convert_to_sarra_py_units,
    pr_kg_m2_s_to_mm_day,
    rsds_w_m2_to_j_m2_day,
    sarra_py_directory_for_isimip,
    temperature_passthrough_k,
)
from prismpy.standards.isimip_versions import (
    ISIMIP_TO_SARRA_VAR_MAPPING,
    SARRA_PY_DERIVED_VARIABLE_DIRECTORIES,
    SUPPORTED_VARIABLES,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SARRA_TRANSLATOR_SRC = (
    _REPO_ROOT / "src" / "prismpy" / "translators" / "sarra_py" / "translator.py"
)


# ── Pin (a) Mapping completeness ─────────────────────────────────────


def test_pin_a_mapping_keys_are_subset_of_supported_variables() -> None:
    """Every ISIMIP variable in ``ISIMIP_TO_SARRA_VAR_MAPPING`` MUST
    be in :data:`SUPPORTED_VARIABLES` — the canonical 6 CF-1.x
    variables Sprint G ships. A mapping entry for an unsupported
    variable would imply the writer can emit something the ISIMIP3b
    client can't fetch."""
    mapping_keys = set(ISIMIP_TO_SARRA_VAR_MAPPING.keys())
    assert mapping_keys.issubset(SUPPORTED_VARIABLES), (
        f"Mapping keys not in SUPPORTED_VARIABLES: "
        f"{mapping_keys - SUPPORTED_VARIABLES}"
    )


def test_pin_a_mapping_covers_all_directly_mappable_isimip_vars() -> None:
    """Of the 6 SUPPORTED_VARIABLES (rsds / tasmax / tasmin / pr /
    hurs / sfcWind), 4 have direct SARRA-Py consumers. The remaining
    2 (``hurs`` + ``sfcWind``) have no SARRA-Py directory mapping
    — they're emitted by other translators (CRAFT 8-col WTH /
    PYTHIA 8-col WTH / ACEA pickle). Pin the 4 directly-mapped
    entries so a future deletion fails loud."""
    expected_directly_mapped = {"pr", "tasmax", "tasmin", "rsds"}
    actual = set(ISIMIP_TO_SARRA_VAR_MAPPING.keys())
    assert actual == expected_directly_mapped, (
        f"Directly-mapped ISIMIP variables drifted: "
        f"expected {expected_directly_mapped}, got {actual}"
    )


def test_pin_a_mapping_carries_three_field_tuples() -> None:
    """Each mapping value is a ``(sarra_dir, source_unit, target_unit)``
    tuple — the unit fields are self-documenting + structural."""
    for isimip_var, value in ISIMIP_TO_SARRA_VAR_MAPPING.items():
        assert isinstance(value, tuple), (
            f"Mapping[{isimip_var!r}] is not a tuple: {value}"
        )
        assert len(value) == 3, (
            f"Mapping[{isimip_var!r}] tuple should be (sarra_dir, "
            f"source_unit, target_unit), got {len(value)} fields"
        )
        sarra_dir, source_unit, target_unit = value
        assert isinstance(sarra_dir, str) and sarra_dir
        assert isinstance(source_unit, str) and source_unit
        assert isinstance(target_unit, str) and target_unit


# ── Pin (b) Unit conversion correctness ──────────────────────────────


def test_pin_b_pr_conversion_kg_m2_s_to_mm_day() -> None:
    """1 kg m⁻² s⁻¹ × 86400 s/day = 86400 mm/day. A mid-Africa
    typical heavy daily rainfall of 50 mm corresponds to ~5.787e-4
    kg m⁻² s⁻¹ (50 / 86400). Pin both directions."""
    # 1 kg m⁻² s⁻¹ → 86400 mm/day
    assert pr_kg_m2_s_to_mm_day(1.0) == pytest.approx(86400.0)
    # 5.787e-4 kg m⁻² s⁻¹ → ~50 mm/day
    typical_heavy_rain_kgm2s = 50.0 / 86400.0
    assert pr_kg_m2_s_to_mm_day(typical_heavy_rain_kgm2s) == pytest.approx(50.0)
    # Array input
    arr = np.array([0.0, 1.0, typical_heavy_rain_kgm2s])
    out = pr_kg_m2_s_to_mm_day(arr)
    np.testing.assert_array_almost_equal(out, [0.0, 86400.0, 50.0])


def test_pin_b_rsds_conversion_w_m2_to_j_m2_day() -> None:
    """1 W m⁻² × 86400 s/day = 86400 J m⁻²/day. A typical sunny-day
    average solar flux of 250 W m⁻² → 21,600,000 J m⁻²/day."""
    assert rsds_w_m2_to_j_m2_day(1.0) == pytest.approx(86400.0)
    assert rsds_w_m2_to_j_m2_day(250.0) == pytest.approx(21600000.0)


def test_pin_b_temperature_passthrough_k() -> None:
    """Temperature is K passthrough — AgERA5 + SARRA-Py both consume
    Kelvin, so no conversion. A future SARRA-Py upgrade that expects
    °C would replace this helper; the pin asserts the current
    contract."""
    assert temperature_passthrough_k(300.0) == 300.0
    assert temperature_passthrough_k(273.15) == 273.15
    arr = np.array([250.0, 273.15, 320.0])
    np.testing.assert_array_equal(temperature_passthrough_k(arr), arr)


def test_pin_b_dispatcher_routes_each_variable_correctly() -> None:
    """The :func:`convert_to_sarra_py_units` dispatcher routes by
    ISIMIP variable name. Pin each of the 4 entries."""
    # pr → mm/day × 86400
    assert convert_to_sarra_py_units("pr", 1.0) == pytest.approx(86400.0)
    # tasmax / tasmin → K passthrough
    assert convert_to_sarra_py_units("tasmax", 300.0) == 300.0
    assert convert_to_sarra_py_units("tasmin", 280.0) == 280.0
    # rsds → J/m²/day × 86400
    assert convert_to_sarra_py_units("rsds", 250.0) == pytest.approx(21600000.0)


def test_pin_b_dispatcher_rejects_unknown_isimip_variable() -> None:
    """Unknown variable raises ValueError with registered keys
    enumerated."""
    with pytest.raises(ValueError, match="hurs"):
        convert_to_sarra_py_units("hurs", 50.0)
    with pytest.raises(ValueError, match="ssp_invented"):
        convert_to_sarra_py_units("ssp_invented", 1.0)


def test_pin_b_directory_lookup_returns_canonical_sarra_name() -> None:
    """:func:`sarra_py_directory_for_isimip` returns the canonical
    SARRA-Py directory name."""
    assert (
        sarra_py_directory_for_isimip("pr") == "rainfall"
    )
    assert (
        sarra_py_directory_for_isimip("tasmax")
        == "2m_temperature_24_hour_maximum"
    )
    assert (
        sarra_py_directory_for_isimip("tasmin")
        == "2m_temperature_24_hour_minimum"
    )
    assert (
        sarra_py_directory_for_isimip("rsds")
        == "solar_radiation_flux_daily"
    )


# ── Pin (c) Two-vocabulary structural pin ────────────────────────────
# (per feedback_two_vocabulary_substrate_drift.md + #227 pattern)


def _extract_sarra_consumer_directory_names() -> Set[str]:
    """Walk the SARRA-Py translator source to extract every
    string literal that names a climate directory expected by the
    consumer side. Two structured surfaces:

    1. ``standard_subdirs`` list at lines 160-170 in ``translate()``
       (subdirectory creation list)
    2. ``climate_vars`` list at lines 1755-1762 in ``validate_outputs``
       (per-directory existence check)

    Returns the union of every literal directory name found.
    """
    src = _SARRA_TRANSLATOR_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    consumer_names: Set[str] = set()

    for node in ast.walk(tree):
        # Match string-list patterns like
        #   ["data/climate/foo", ...] or ["foo", "bar", ...]
        if isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(
                    elt.value, str
                ):
                    s = elt.value
                    # The "data/climate/<name>" form
                    if s.startswith("data/climate/"):
                        name = s[len("data/climate/") :]
                        if name and "/" not in name:
                            consumer_names.add(name)
                    # The bare-name form in climate_vars list
                    # (heuristic: matches a known SARRA prefix)
                    elif re.match(
                        r"^(rainfall|2m_temperature_|"
                        r"solar_radiation_|ET0)",
                        s,
                    ):
                        consumer_names.add(s)
    return consumer_names


def test_pin_c_writer_emits_only_consumer_recognized_directories() -> None:
    """Two-vocabulary structural pin per
    ``feedback_two_vocabulary_substrate_drift.md``:
    the SARRA-Py projection writer's POST-MAPPING output directory
    names (the values in :data:`ISIMIP_TO_SARRA_VAR_MAPPING`) MUST
    be a subset of the SARRA-Py consumer's expected directory set
    (extracted from ``translator.py``'s subdir-creation list +
    validation list).

    A future drift where the writer emits a directory the consumer
    doesn't recognize fires this pin. Same class as #227 cockpit
    ``failed_checks[].category`` enum-vs-prefix vocab drift; this
    pin is the canonical guard against the ISIMIP-vs-SARRA-Py case.
    """
    consumer_dirs = _extract_sarra_consumer_directory_names()
    writer_post_mapping_dirs = {
        sarra_dir for sarra_dir, _, _ in ISIMIP_TO_SARRA_VAR_MAPPING.values()
    }
    drift = writer_post_mapping_dirs - consumer_dirs
    assert not drift, (
        f"Two-vocabulary substrate drift detected: writer emits "
        f"directories {drift} that the SARRA-Py consumer does not "
        f"recognize. Consumer expected set: {consumer_dirs}. "
        "Per feedback_two_vocabulary_substrate_drift.md, the writer's "
        "POST-MAPPING output vocabulary MUST be a subset of the "
        "consumer's expected vocabulary. Either extend the consumer "
        "to recognize the new directory OR fix the mapping target name."
    )


def test_pin_c_consumer_known_dirs_include_canonical_targets() -> None:
    """Sanity inverse: every value in
    :data:`ISIMIP_TO_SARRA_VAR_MAPPING` IS a name the consumer-side
    extractor recognizes. Catches a refactor that renames a consumer
    constant (e.g., ``2m_temperature_24_hour_maximum`` →
    ``2m_temp_max_daily``) without updating the mapping in lock-step."""
    consumer_dirs = _extract_sarra_consumer_directory_names()
    assert "rainfall" in consumer_dirs
    assert "2m_temperature_24_hour_maximum" in consumer_dirs
    assert "2m_temperature_24_hour_minimum" in consumer_dirs
    assert "solar_radiation_flux_daily" in consumer_dirs


# ── Pin (d) Sibling-sweep over SARRA-Py translator ───────────────────


def test_pin_d_no_raw_isimip_cf_directory_string_in_writer() -> None:
    """Sibling-sweep per durable §20: the SARRA-Py translator's
    ``_generate_projection_climate_geotiffs`` body must NOT contain
    a literal string ``"data/climate/tasmax"`` /
    ``"data/climate/pr"`` / ``"data/climate/rsds"`` etc — those
    would imply a bypass of the canonical mapping. Walk the source
    looking for any literal under ``data/climate/<isimip_cf_name>``.
    """
    src = _SARRA_TRANSLATOR_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Raw ISIMIP CF variable names that should NEVER appear in the
    # consumer-facing ``data/climate/<name>`` paths inside the SARRA-Py
    # translator — they are the writer-side input vocabulary, not the
    # post-mapping output vocabulary.
    isimip_only_names: Set[str] = set(SUPPORTED_VARIABLES)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if s.startswith("data/climate/"):
                name = s[len("data/climate/") :]
                if name in isimip_only_names:
                    violations.append((s, getattr(node, "lineno", 0)))
    assert not violations, (
        "Sibling-sweep violation: SARRA-Py translator contains raw "
        "ISIMIP CF directory paths (bypassing the canonical mapping). "
        f"Found: {violations}. Route through "
        "sarra_py_directory_for_isimip() per durable §24."
    )


def test_pin_d_writer_imports_canonical_mapping_helpers() -> None:
    """Sibling-sweep continued: the SARRA-Py translator MUST import
    ``convert_to_sarra_py_units`` + ``sarra_py_directory_for_isimip``
    from the canonical helper module. AST-walks the import statements."""
    src = _SARRA_TRANSLATOR_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    expected = {
        "convert_to_sarra_py_units",
        "sarra_py_directory_for_isimip",
        "ISIMIP_TO_SARRA_VAR_MAPPING",
    }
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in expected:
                    found.add(alias.name)
    missing = expected - found
    assert not missing, (
        f"SARRA-Py translator missing canonical-mapping imports: "
        f"{missing}. Add `from prismpy.harmonize.isimip_unit_conversions "
        "import convert_to_sarra_py_units, sarra_py_directory_for_isimip` "
        "and `from prismpy.standards.isimip_versions import "
        "ISIMIP_TO_SARRA_VAR_MAPPING` per durable §24."
    )


def test_pin_d_no_inline_unit_conversion_constants_outside_canonical() -> None:
    """Sibling-sweep per durable §20 — the magic numbers ``86400.0``
    (seconds-per-day for unit conversion) MUST appear ONLY inside
    ``prismpy/standards/isimip_versions.py`` (canonical declaration)
    + ``prismpy/harmonize/isimip_unit_conversions.py`` (math
    surface) + ``prismpy/harmonize/calendar_conversion.py`` (older
    Sprint G usage of 86400 for missing-day fill — but that's a
    different conversion). Translator-level unit math would be a
    canonical-source-bypass."""
    sarra_src = _SARRA_TRANSLATOR_SRC.read_text(encoding="utf-8")
    # 86400.0 OR 86400 (with optional trailing decimal) standalone
    # numeric literal — exclude string contexts.
    tree = ast.parse(sarra_src)
    forbidden_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float)
        ):
            if node.value == 86400 or node.value == 86400.0:
                forbidden_lines.append(getattr(node, "lineno", 0))
    assert not forbidden_lines, (
        f"SARRA-Py translator contains literal 86400 unit-conversion "
        f"constant at lines {forbidden_lines}. Route through "
        "prismpy.harmonize.isimip_unit_conversions per durable §24."
    )


# ── Public-API surface pins ──────────────────────────────────────────


def test_isimip_unit_conversions_module_public_api() -> None:
    """The unit-conversions module exposes the 5 canonical helpers."""
    import prismpy.harmonize.isimip_unit_conversions as mod

    assert set(mod.__all__) == {
        "pr_kg_m2_s_to_mm_day",
        "rsds_w_m2_to_j_m2_day",
        "temperature_passthrough_k",
        "convert_to_sarra_py_units",
        "sarra_py_directory_for_isimip",
    }


def test_isimip_versions_module_exports_mapping_and_derived_set() -> None:
    """The standards module exposes the canonical mapping +
    derived-variables set."""
    import prismpy.standards.isimip_versions as mod

    assert "ISIMIP_TO_SARRA_VAR_MAPPING" in mod.__all__
    assert "SARRA_PY_DERIVED_VARIABLE_DIRECTORIES" in mod.__all__


def test_derived_variable_directories_set_contains_2_entries() -> None:
    """SARRA_PY_DERIVED_VARIABLE_DIRECTORIES enumerates the 2
    SARRA-Py expected directories that require derivation
    (tasmean = avg(tasmax, tasmin); ET0Hargeaves)."""
    assert SARRA_PY_DERIVED_VARIABLE_DIRECTORIES == frozenset(
        {
            "2m_temperature_24_hour_mean",
            "ET0Hargeaves",
        }
    )
