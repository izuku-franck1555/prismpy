"""Prep-wiring (#214): the PYTHIA translator EMITS the DSSAT-native planting fields the prism-runner
reads — ``ppop``/``ppoe``/``plrs``/``pldp`` in ``pythia_config`` ``default_setup`` AND as fixed-width
jinja placeholders in the SNX ``@P`` row. Unit-aware (plants/ha is divided by 10000 ONCE; a plants/m²
``plant_population`` override is used AS-IS — the two-unit trap); per-crop absent-edge defaults, never
-99; the DSSAT 6-char ``@P`` columns preserved on a real render.
"""
from __future__ import annotations

import re

from prismpy.config.schema import ManagementConfig
from prismpy.translators.pythia.translator import PythiaTranslator
from tests.unit.test_pythia_plant_mode import _cfg, _generated_pythia_json

_FIELDS = ("ppop", "ppoe", "plrs", "pldp")

# jinja2 is NOT a prismpy dependency (the prism-runner renders the SNX template), so simulate the
# runner's render of the @P row's fixed-width fields faithfully: ``{{ "%5.1f"|format(ppop|default(
# 5.3)) }}`` renders exactly ``"%5.1f" % (ctx.ppop if present else 5.3)`` — a lost leading space or a
# wrong width shifts the compared substring and fails the test.
_JINJA_FIELD = re.compile(r'\{\{ "(%[0-9.]+f)"\|format\((\w+)\|default\(([-0-9.]+)\)\) \}\}')


def _render(row: str, **ctx) -> str:
    row = _JINJA_FIELD.sub(lambda m: m.group(1) % ctx.get(m.group(2), float(m.group(3))), row)
    return row.replace("{{ pdate | default(-99) }}", str(ctx.get("pdate", -99)))


# ── value pin — both unit paths + the #27 runner-contract field names ────────

def test_default_setup_emits_the_four_fields_with_runner_names(tmp_path):
    ds = _generated_pythia_json(tmp_path, ManagementConfig(planting_density=62500.0))["default_setup"]
    for name in _FIELDS:                          # #27 format-match: exact runner-contract names
        assert name in ds, f"{name} missing from pythia_config default_setup"


def test_density_ha_is_converted_once_to_ppop_m2(tmp_path):
    # 62500 plants/ha -> 6.25 plants/m² (the single /10000). NOT 62500 (unconverted), NOT 0.000625
    # (double-converted -> a dead stand).
    ds = _generated_pythia_json(tmp_path, ManagementConfig(planting_density=62500.0))["default_setup"]
    assert ds["ppop"] == 6.25
    assert ds["ppoe"] == 6.25
    assert ds["ppop"] != 62500.0
    assert ds["ppop"] != 0.000625


def test_pythia_has_no_m2_override_so_ppop_is_single_unit(tmp_path):
    # The plants/m² `plant_population` override is a CRAFT-config field, NOT PythiaConfig — so the
    # PYTHIA path has a SINGLE unit source (planting_density, plants/ha) and no phantom-m²-override
    # double-convert path. Guard the premise: PythiaConfig must not grow a plant_population field
    # without the resolver learning to use it as-is (else a re-/10000 dead-stands the crop).
    from prismpy.config.schema import PythiaConfig
    assert "plant_population" not in PythiaConfig.model_fields
    t = PythiaTranslator(config=_cfg(tmp_path, ManagementConfig(planting_density=55000.0)),
                         output_dir=str(tmp_path))
    assert t._resolve_planting_params()["ppop"] == 5.5              # 55000 ha -> 5.5 m² (the one /10000)


def test_absent_config_uses_per_crop_default_never_zero(tmp_path):
    # No management config -> the per-crop maize literal (crop-modeling), never -99/0.
    t = PythiaTranslator(config=_cfg(tmp_path, None), output_dir=str(tmp_path))
    assert t._resolve_planting_params() == {"ppop": 5.3, "ppoe": 5.3, "plrs": 75.0, "pldp": 5.0}


# ── template pin — jinja placeholders + fixed-width 6-char @P columns ────────

def _maize_snx(tmp_path) -> str:
    t = PythiaTranslator(config=_cfg(tmp_path, ManagementConfig(planting_density=62500.0)),
                         output_dir=str(tmp_path))
    return t._build_snx_content(
        exp_id="MLMZ8001", region_name="Wami", country="Tanzania", crop_name="Maize",
        cultivar=t._map_generic_to_cultivar(), fertilizer=t._map_generic_to_fertilizer(),
        config=t._map_generic_to_pythia_config(),
    )


def _at_p_data_row(snx: str) -> str:
    lines = snx.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("@P PDATE"):
            return lines[i + 1]
    raise AssertionError("@P planting row not found in generated SNX")


def test_at_p_row_is_runner_filled_jinja_not_prep_baked(tmp_path):
    row = _at_p_data_row(_maize_snx(tmp_path))
    for f in _FIELDS:
        assert f"format({f}|default(" in row, f"@P {f} is not a runner-filled jinja placeholder"


def test_at_p_columns_stay_6_wide_when_the_runner_fills_them(tmp_path):
    row = _at_p_data_row(_maize_snx(tmp_path))
    filled = _render(row, ppop=6.25, ppoe=6.25, plrs=70.0, pldp=5.0, pdate="2020001")
    assert "{{" not in filled and "}}" not in filled             # fully rendered, no leftover jinja
    assert "   6.2   6.2" in filled                              # PPOP + PPOE: 1-sep + %5.1f each
    assert "    70" in filled                                    # PLRS: 1-sep + %5.0f
    assert "     5" in filled                                    # PLDP: 1-sep + %5.0f


def test_at_p_absent_edge_renders_the_per_crop_literal_never_99(tmp_path):
    row = _at_p_data_row(_maize_snx(tmp_path))
    default_filled = _render(row)                                # no ctx -> the default() must fire
    assert "{{" not in default_filled and "}}" not in default_filled
    assert "   5.3   5.3" in default_filled                      # maize PPOP/PPOE default, not -99
    assert "    75" in default_filled                            # maize PLRS default, not -99


# ── #27 producer-consumer format-match: the runner reads cultivar_id from the manifest ──

def test_manifest_carries_the_cultivar_id_field_name(tmp_path):
    # The runner reads the resolved cultivar as `cultivar_id` from the manifest — NOT the
    # PYTHIA-internal pythia_config `ingeno` (renaming that would break PYTHIA's own SNX render).
    import prismpy.packaging.manifest as m
    from pathlib import Path
    assert '"cultivar_id"' in Path(m.__file__).read_text()
