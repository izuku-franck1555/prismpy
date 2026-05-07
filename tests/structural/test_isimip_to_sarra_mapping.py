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
    rsds_w_m2_to_kj_m2_day,
    sarra_py_directory_for_isimip,
    temperature_kelvin_to_celsius,
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


def test_pin_b_rsds_conversion_w_m2_to_kj_m2_day() -> None:
    """Codex round 2 P1 absorption: 1 W m⁻² × 86400 s/day ÷ 1000 J/kJ
    = 86.4 kJ m⁻²/day matches AgERA5 ``version="SARRA-Py"``
    vendored-library output. A typical sunny-day average solar
    flux of 250 W m⁻² → 21,600 kJ m⁻²/day."""
    assert rsds_w_m2_to_kj_m2_day(1.0) == pytest.approx(86.4)
    assert rsds_w_m2_to_kj_m2_day(250.0) == pytest.approx(21600.0)


def test_pin_b_temperature_kelvin_to_celsius() -> None:
    """Codex round 2 P1 absorption: SARRA-Py consumes °C per
    ``post_translate.SARRA_PY_VAR_MAPPING`` noop ops (comment:
    "already °C"). Temperature conversion subtracts 273.15."""
    assert temperature_kelvin_to_celsius(273.15) == pytest.approx(0.0)
    assert temperature_kelvin_to_celsius(300.0) == pytest.approx(26.85)
    arr = np.array([250.0, 273.15, 320.0])
    np.testing.assert_array_almost_equal(
        temperature_kelvin_to_celsius(arr),
        np.array([-23.15, 0.0, 46.85]),
    )


def test_pin_b_dispatcher_routes_each_variable_correctly() -> None:
    """The :func:`convert_to_sarra_py_units` dispatcher routes by
    ISIMIP variable name. Pin each of the 4 entries (codex round 2
    P1 absorption: tasmax/tasmin → °C; rsds → kJ/m²/day)."""
    # pr → mm/day × 86400
    assert convert_to_sarra_py_units("pr", 1.0) == pytest.approx(86400.0)
    # tasmax / tasmin → °C (K - 273.15)
    assert convert_to_sarra_py_units("tasmax", 300.0) == pytest.approx(26.85)
    assert convert_to_sarra_py_units("tasmin", 280.0) == pytest.approx(6.85)
    # rsds → kJ/m²/day × 86.4
    assert convert_to_sarra_py_units("rsds", 250.0) == pytest.approx(21600.0)


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
        "rsds_w_m2_to_kj_m2_day",
        "temperature_kelvin_to_celsius",
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


# ── §10 Sibling-sweep evidence — producer ⇆ consumer unit alignment ──
#
# Per team-lead 2026-05-07 boundary 7/7 absorption authorization point
# 2: verify EVERY ISIMIP variable's producer target unit against the
# consumer-side normalization metadata. Cited file:line per variable
# below; the round-trip behavioural pin (§11) + declarative pin (§12)
# enforce the alignment structurally.
#
# Cross-repo evidence for the 5 ISIMIP CF variables in
# :data:`SUPPORTED_VARIABLES`:
#
# 1. ``pr`` (precipitation) — producer maps to ``rainfall`` directory
#    in mm/day.
#    Consumer: ``validators/post_translate.py:573`` declares
#    ``"rainfall": ("rain", "noop", 0.0)`` → no further conversion;
#    research-friendly unit = mm/day. The ``post_translate.py:564``
#    comment confirms the SARRA_data_download library "already
#    converts AgERA5 to researcher-friendly units before writing its
#    .tif files (tmax / tmin in °C, srad in kJ/m²/day, rain in
#    mm/day)" — so consumer EXPECTS mm/day on disk.
#    ✓ Producer kg m⁻² s⁻¹ × 86400 → mm/day matches.
#
# 2. ``tasmax`` (max temperature) — producer maps to
#    ``2m_temperature_24_hour_maximum`` in °C.
#    Consumer: ``validators/post_translate.py:574`` declares
#    ``"2m_temperature_24_hour_maximum": ("tmax", "noop", 0.0)`` with
#    explicit comment ``# already °C`` → consumer EXPECTS °C.
#    ✓ Producer K - 273.15 → °C matches.
#
# 3. ``tasmin`` (min temperature) — producer maps to
#    ``2m_temperature_24_hour_minimum`` in °C.
#    Consumer: ``validators/post_translate.py:575`` declares
#    ``"2m_temperature_24_hour_minimum": ("tmin", "noop", 0.0)``
#    with explicit comment ``# already °C``.
#    ✓ Producer K - 273.15 → °C matches.
#
# 4. ``rsds`` (surface downwelling shortwave) — producer maps to
#    ``solar_radiation_flux_daily`` in kJ/m²/day.
#    Consumer: ``validators/post_translate.py:576`` declares
#    ``"solar_radiation_flux_daily": ("srad", "mul", 1e-3)`` with
#    comment ``# kJ/m²/d → MJ/m²/d`` — consumer EXPECTS kJ/m²/day on
#    disk + applies × 1e-3 at validation time to reach MJ/m²/day.
#    ✓ Producer W m⁻² × 86.4 → kJ/m²/day matches.
#
# 5. ``hurs`` (relative humidity) — NOT in producer mapping; SKIPPED.
#    Consumer: SARRA-Py expected directories list at
#    ``translators/sarra_py/translator.py:1755-1761`` does NOT include
#    any humidity directory. AgERA5 vendored library at
#    ``sources/climate/agera5.py:800`` ships ``vapour_pressure_24_hour_mean``
#    (vapour pressure in hPa, NOT relative humidity). Per producer's
#    ``ISIMIP_TO_SARRA_VAR_MAPPING`` exclusion logic + writer's
#    skip-with-warning at ``translators/sarra_py/translator.py:794-803``,
#    no SARRA-Py-facing emission for ``hurs``.
#    ✓ Correct exclusion; no producer ⇆ consumer mismatch.
#
# 6. ``sfcWind`` (10m wind speed) — NOT in producer mapping; SKIPPED.
#    Consumer: SARRA-Py expected directories don't include a wind
#    directory. AgERA5 ships ``10m_wind_speed_24_hour_mean`` per
#    ``sources/climate/agera5.py:801`` but that's
#    consumed only by other translators (CRAFT 8-col WTH), not
#    SARRA-Py.
#    ✓ Correct exclusion.


# ── §11 Round-trip behavioural pin (producer → consumer chain) ───────


def _apply_consumer_normalization(operation: str, operand: float, value: float) -> float:
    """Mirror of ``validators/post_translate.py`` consumer-side
    normalization op + operand. Pure function for testability."""
    if operation == "noop":
        return value
    if operation == "mul":
        return value * operand
    if operation == "add":
        return value + operand
    raise ValueError(f"Unsupported consumer op: {operation!r}")


def test_pin_round_trip_pr_kg_m2_s_to_canonical_mm_day() -> None:
    """Round-trip pin per team-lead boundary 7/7 absorption authorization
    point 3: realistic ISIMIP ``pr`` input (typical Sahel daily rainfall
    50 mm = 5.787e-4 kg m⁻² s⁻¹) → producer conversion → consumer
    normalization → final canonical unit (mm/day).

    Closes the producer⇆consumer unit-alignment class structurally:
    declarative completeness pins guarantee the mapping exists,
    behavioural round-trip pin guarantees the chain delivers the
    consumer's expected canonical SARRA-Py unit."""
    from prismpy.validators.post_translate import SARRA_PY_VAR_MAPPING

    isimip_input_kg_m2_s = 50.0 / 86400.0  # 50 mm/day → kg m⁻² s⁻¹
    # Step 1: producer conversion
    producer_output = convert_to_sarra_py_units("pr", isimip_input_kg_m2_s)
    sarra_dir = sarra_py_directory_for_isimip("pr")
    assert sarra_dir == "rainfall"
    # Step 2: consumer normalization
    var_internal, op, operand = SARRA_PY_VAR_MAPPING[sarra_dir]
    final_value = _apply_consumer_normalization(op, operand, producer_output)
    # Final canonical SARRA-Py unit: mm/day
    assert var_internal == "rain"
    assert op == "noop"  # consumer expects mm/day on disk → no further conversion
    assert final_value == pytest.approx(50.0, abs=1e-3), (
        f"pr round-trip: 50 mm/day input → ISIMIP CF kg m⁻² s⁻¹ → "
        f"producer mm/day → consumer noop → expected 50.0 mm/day, "
        f"got {final_value}"
    )


@pytest.mark.parametrize(
    "isimip_var,kelvin_input,expected_celsius",
    [
        ("tasmax", 300.0, 26.85),
        ("tasmin", 280.0, 6.85),
        ("tasmax", 273.15, 0.0),
    ],
)
def test_pin_round_trip_temp_kelvin_to_canonical_celsius(
    isimip_var: str, kelvin_input: float, expected_celsius: float
) -> None:
    """Round-trip: ISIMIP K → producer K - 273.15 → °C → consumer
    noop → final °C. Catches a future drift where producer reverts to
    K passthrough (codex round 2 P1 regression class) — round-trip pin
    fires because final value is then 273.15 K too high."""
    from prismpy.validators.post_translate import SARRA_PY_VAR_MAPPING

    producer_output = convert_to_sarra_py_units(isimip_var, kelvin_input)
    sarra_dir = sarra_py_directory_for_isimip(isimip_var)
    var_internal, op, operand = SARRA_PY_VAR_MAPPING[sarra_dir]
    final_value = _apply_consumer_normalization(op, operand, producer_output)
    assert op == "noop", (
        f"Consumer for {sarra_dir!r} should be noop (already °C); "
        f"got {op!r}"
    )
    assert final_value == pytest.approx(expected_celsius, abs=1e-3), (
        f"{isimip_var} round-trip: {kelvin_input} K input → producer "
        f"°C → consumer noop → expected {expected_celsius} °C, got "
        f"{final_value}"
    )


def test_pin_round_trip_rsds_w_m2_to_canonical_mj_m2_day() -> None:
    """Round-trip: ISIMIP rsds W m⁻² → producer × 86.4 → kJ/m²/day →
    consumer × 1e-3 → final MJ/m²/day (canonical SARRA-Py
    research-friendly unit). 250 W m⁻² → 21600 kJ/m²/day → 21.6
    MJ/m²/day."""
    from prismpy.validators.post_translate import SARRA_PY_VAR_MAPPING

    producer_output = convert_to_sarra_py_units("rsds", 250.0)
    sarra_dir = sarra_py_directory_for_isimip("rsds")
    var_internal, op, operand = SARRA_PY_VAR_MAPPING[sarra_dir]
    final_value = _apply_consumer_normalization(op, operand, producer_output)
    assert var_internal == "srad"
    assert op == "mul"
    assert operand == pytest.approx(1e-3)  # kJ → MJ
    assert final_value == pytest.approx(21.6, abs=1e-3), (
        f"rsds round-trip: 250 W m⁻² → producer kJ/m²/day → consumer "
        f"× 1e-3 → expected 21.6 MJ/m²/day, got {final_value}"
    )


# ── §12 Declarative pin — producer ⇆ consumer unit alignment ─────────


# Mapping: SARRA-Py directory name → (consumer's pre-normalize unit
# expected on disk, canonical SARRA-Py "research-friendly" final unit
# after consumer normalization). Sourced from
# ``validators/post_translate.py:564-577`` SARRA_PY_VAR_MAPPING +
# its file-unit-verification comment block.
_CONSUMER_EXPECTED_UNITS: dict = {
    # sarra_dir: (pre_normalize_unit, post_normalize_canonical_unit)
    "rainfall": ("mm/day", "mm/day"),  # noop
    "2m_temperature_24_hour_maximum": ("degC", "degC"),  # noop
    "2m_temperature_24_hour_minimum": ("degC", "degC"),  # noop
    "solar_radiation_flux_daily": (
        "kJ m-2 day-1",
        "MJ m-2 day-1",
    ),  # × 1e-3
}


def test_pin_declarative_producer_target_matches_consumer_expected() -> None:
    """Declarative pin per team-lead boundary 7/7 absorption point 4:
    every entry in :data:`ISIMIP_TO_SARRA_VAR_MAPPING`'s target unit
    matches the consumer-side expected pre-normalize unit per
    :data:`_CONSUMER_EXPECTED_UNITS` (sourced from
    ``validators/post_translate.py``).

    A future drift where the producer's target unit changes (e.g.,
    revert °C → K, or shift kJ → J) without updating the consumer
    side fires this pin. This IS the durable §24 canonical-source-
    or-pin substrate that round-1's "completeness" pin should have
    been (round 1 only checked "does mapping exist"; this checks
    "does target unit match what consumer expects"). Two-vocabulary
    drift bug-class-closer per ``feedback_two_vocabulary_substrate_drift.md``."""
    drifts = []
    for isimip_var, (
        sarra_dir,
        source_unit,
        target_unit,
    ) in ISIMIP_TO_SARRA_VAR_MAPPING.items():
        if sarra_dir not in _CONSUMER_EXPECTED_UNITS:
            drifts.append(
                f"{isimip_var}: producer maps to SARRA dir "
                f"{sarra_dir!r} but consumer "
                "validators/post_translate.SARRA_PY_VAR_MAPPING does NOT "
                "register that directory."
            )
            continue
        consumer_expected, _canonical = _CONSUMER_EXPECTED_UNITS[sarra_dir]
        if target_unit != consumer_expected:
            drifts.append(
                f"{isimip_var}: producer target unit {target_unit!r} "
                f"!= consumer expected pre-normalize unit "
                f"{consumer_expected!r} for {sarra_dir!r}. Either "
                "fix producer's ISIMIP_TO_SARRA_VAR_MAPPING target_unit "
                "OR update the canonical _CONSUMER_EXPECTED_UNITS table "
                "in this test (with cited consumer file:line evidence)."
            )
    assert not drifts, "\n".join(drifts)


def test_pin_declarative_consumer_normalization_metadata_in_sync() -> None:
    """Inverse declarative pin: every entry in the consumer's
    ``SARRA_PY_VAR_MAPPING`` corresponds to a SARRA-Py expected
    directory listed in :data:`_CONSUMER_EXPECTED_UNITS`. Catches a
    refactor that adds a new consumer entry (e.g., wind speed support)
    without updating this test's metadata."""
    from prismpy.validators.post_translate import SARRA_PY_VAR_MAPPING

    consumer_dirs = set(SARRA_PY_VAR_MAPPING.keys())
    pinned_dirs = set(_CONSUMER_EXPECTED_UNITS.keys())
    missing_in_pin = consumer_dirs - pinned_dirs
    extra_in_pin = pinned_dirs - consumer_dirs
    assert not missing_in_pin, (
        f"SARRA_PY_VAR_MAPPING entries without canonical-unit pin: "
        f"{missing_in_pin}. Add entries to _CONSUMER_EXPECTED_UNITS "
        "with cited validators/post_translate.py file:line evidence."
    )
    assert not extra_in_pin, (
        f"Stale _CONSUMER_EXPECTED_UNITS entries (no consumer "
        f"counterpart): {extra_in_pin}. Drop them."
    )


def test_pin_declarative_no_producer_mapping_for_directories_consumer_doesnt_recognize() -> None:
    """The producer should NOT emit to a SARRA-Py directory that the
    consumer's ``SARRA_PY_VAR_MAPPING`` doesn't register. Catches a
    future producer addition that ships a new directory the consumer
    can't process."""
    from prismpy.validators.post_translate import SARRA_PY_VAR_MAPPING

    producer_targets = {
        sarra_dir
        for sarra_dir, _, _ in ISIMIP_TO_SARRA_VAR_MAPPING.values()
    }
    consumer_keys = set(SARRA_PY_VAR_MAPPING.keys())
    orphans = producer_targets - consumer_keys
    assert not orphans, (
        f"Producer ISIMIP_TO_SARRA_VAR_MAPPING emits to SARRA dirs "
        f"{orphans} that the consumer SARRA_PY_VAR_MAPPING does NOT "
        "recognize. Either drop the producer entry OR extend the "
        "consumer's SARRA_PY_VAR_MAPPING."
    )
