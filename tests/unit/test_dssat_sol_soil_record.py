"""The .SOL soil record line: written directly after ``*SOILS:`` (DSSAT ignores
``!`` lines), it counts what the file holds and the grid cells on each kind of
soil; the generic profile's depth and plant-available water are derived from
the values written. Plus the other consumers of layered HWSD profiles."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.translators._shared import dssat_sol_writer as writer

REGION = Region(name="Koutiala", country="Mali", country_iso3="MLI",
                bounds=BoundingBox(minx=-6.0, miny=12.0, maxx=-5.0, maxy=13.0))


def _layer(top, bottom, **kw):
    values = dict(sand=40.0, clay=30.0, silt=30.0, organic_carbon=1.0,
                  bulk_density=1.4, ph=6.5)
    values.update(kw)
    return SoilLayer(depth_top=top, depth_bottom=bottom, **values)


def _profile(layers, source="hwsd", **metadata):
    return SoilProfile(profile_id="p", lat=12.5, lon=-5.5, source=source, layers=layers,
                       total_depth=layers[-1].depth_bottom, metadata=metadata)


def _default(lower_fc=0.28, bottom=1.0):
    return _profile([
        _layer(0.0, 0.2, sand=60.0, clay=18.0, silt=22.0, organic_carbon=0.5,
               bulk_density=1.4, ph=6.5, field_capacity=0.25, wilting_point=0.10),
        _layer(0.2, bottom, sand=55.0, clay=22.0, silt=23.0, organic_carbon=0.3,
               bulk_density=1.5, ph=6.3, field_capacity=lower_fc, wilting_point=0.12),
    ], source="default")


def _write(tmp_path, profiles, counts=None, causes=None, record=True, out=None):
    path = tmp_path / "ML.SOL"
    writer.write_dssat_sol(path, profiles, "ML", REGION, soil_record=record,
                           profile_cell_counts=counts, default_causes=causes,
                           default_declaration_out=out)
    return path.read_text().splitlines()


def _record(lines):
    records = [ln for ln in lines if ln.startswith("! prismpy soil record:")]
    assert len(records) == 1
    return dict(field.split("=", 1) for field in records[0].split(": ", 1)[1].split())


def test_record_line_counts_and_placement(tmp_path):
    profiles = {
        1469: _profile([_layer(0.0, 0.2, organic_carbon=40.4)], ptf_domain_flags={0: "organic"}),
        1470: _profile([_layer(0.0, 0.2, bulk_density=0.76)], ptf_domain_flags={0: "andic"}),
        1471: _profile([_layer(0.0, 0.2, bulk_density=None)]),
        0: _default(),
    }
    lines = _write(tmp_path, profiles, {1469: 3, 1470: 2, 1471: 4, 0: 1},
                   {"no_hwsd_soil_at_cell_centre": 1})
    assert lines[0].startswith("*SOILS: Koutiala")
    assert lines[1] == (
        "! prismpy soil record: profiles=4 layers=5 organic_layers=1 andic_layers=1"
        " chem_defaulted=bulk_density:1,organic_carbon:0,ph:0"
        " chem_default_values=bulk_density:1.40,organic_carbon:0.50,ph:6.5"
        " cells_with_flagged_layers=5 cells_with_defaulted_chemistry=4 cells=10"
        " default_cells=1 default_cause=no_hwsd_soil_at_cell_centre:1"
        " default_fraction=0.1000 default_warning=1 default_profile=ML00000000"
        " default_depth_cm=0-100 default_paw_mm=158")
    assert lines[2] == ""
    assert lines[3].startswith("*ML00000000")


def test_record_line_follows_the_d4_grammar(tmp_path):
    lines = _write(tmp_path, {1: _profile([_layer(0.0, 0.2)]), 0: _default()},
                   {1: 19, 0: 1}, {"no_hwsd_soil_at_cell_centre": 1})
    grammar = (
        r"! prismpy soil record: profiles=\d+ layers=\d+ organic_layers=\d+ andic_layers=\d+"
        r" chem_defaulted=bulk_density:\d+,organic_carbon:\d+,ph:\d+"
        r" chem_default_values=bulk_density:1\.40,organic_carbon:0\.50,ph:6\.5"
        r" cells_with_flagged_layers=\d+ cells_with_defaulted_chemistry=\d+ cells=\d+"
        r" default_cells=\d+ default_cause=(-|[a-z_]+:\d+(,[a-z_]+:\d+)*)"
        r" default_fraction=\d\.\d{4} default_warning=[01] default_profile=(-|[A-Z]{2}\d{8})"
        r" default_depth_cm=(-|\d+-\d+) default_paw_mm=(-|\d+)")
    assert re.fullmatch(grammar, lines[1])


@pytest.mark.parametrize("k0, n, fraction, warning", [
    (1001, 20001, "0.0500", "1"),
    (1000, 20000, "0.0500", "0"),
    (2, 20, "0.1000", "1"),
    (1, 20, "0.0500", "0"),
])
def test_warning_is_the_exact_ratio_not_the_rounded_one(tmp_path, k0, n, fraction, warning):
    out = []
    record = _record(_write(tmp_path, {1: _profile([_layer(0.0, 0.2)]), 0: _default()},
                            {1: n - k0, 0: k0}, {"no_hwsd_soil_at_cell_centre": k0}, out=out))
    assert (record["default_fraction"], record["default_warning"]) == (fraction, warning)
    assert out[0].warning is (warning == "1") and (out[0].default_cells, out[0].cells) == (k0, n)


@pytest.mark.parametrize("lower_fc, bottom, depth, paw", [
    (0.28, 1.0, "0-100", "158"),
    (0.24, 1.0, "0-100", "126"),
    (0.28, 0.8, "0-80", "126"),
])
def test_default_depth_and_water_are_derived_from_the_written_layers(
        tmp_path, lower_fc, bottom, depth, paw):
    out = []
    record = _record(_write(tmp_path, {0: _default(lower_fc, bottom)}, {0: 4},
                            {"no_soil_source": 4}, out=out))
    assert (record["default_depth_cm"], record["default_paw_mm"]) == (depth, paw)
    assert (out[0].depth_cm, out[0].paw_mm) == (depth, int(paw))


def test_water_uses_the_written_value_not_the_input(tmp_path):
    """A field capacity of 0.0 is written as the writer's 0.25 fallback."""
    profile = _default()
    profile.layers[0].field_capacity = 0.0
    record = _record(_write(tmp_path, {0: profile}, {0: 1}, {"no_soil_source": 1}))
    assert record["default_paw_mm"] == "158"


def test_no_generic_cells_renders_dashes(tmp_path):
    record = _record(_write(tmp_path, {1: _profile([_layer(0.0, 0.2)])}, {1: 7}, {}))
    assert {k: record[k] for k in ("cells", "default_cells", "default_cause", "default_fraction",
                                   "default_warning", "default_profile", "default_depth_cm",
                                   "default_paw_mm")} == {
        "cells": "7", "default_cells": "0", "default_cause": "-", "default_fraction": "0.0000",
        "default_warning": "0", "default_profile": "-", "default_depth_cm": "-",
        "default_paw_mm": "-"}


def test_the_placeholder_is_the_generic_profile(tmp_path):
    placeholder = _default()
    placeholder.source = "placeholder"
    record = _record(_write(tmp_path, {90000001: placeholder}, {90000001: 5},
                            {"retrieve_stage_placeholder": 5}))
    assert (record["default_cause"], record["default_profile"], record["default_paw_mm"]) == (
        "retrieve_stage_placeholder:5", "ML90000001", "158")


def test_record_is_off_by_default_and_the_file_is_otherwise_unchanged(tmp_path):
    profiles = {1: _profile([_layer(0.0, 0.2)]), 0: _default()}
    plain = _write(tmp_path, profiles, record=False)
    recorded = _write(tmp_path, profiles, {1: 1, 0: 1}, {"no_soil_source": 1})
    assert not any(ln.startswith("!") for ln in plain)
    assert [ln for ln in recorded if not ln.startswith("! prismpy soil record:")] == plain
    with pytest.raises(ValueError):
        _write(tmp_path, profiles, None, {})


def test_acea_reads_the_top_layer_of_a_layered_profile(tmp_path):
    from prismpy.translators.acea.translator import AceaTranslator

    layers = [_layer(i / 5, (i + 1) / 5, sand=40.0 + i, clay=30.0 - i) for i in range(5)]
    translator = AceaTranslator.__new__(AceaTranslator)
    translator.output_dir = tmp_path
    (tmp_path / "soil").mkdir()
    path = translator._generate_soil_csv({7: _profile(layers)}, [7])
    row = next(csv.DictReader(Path(path).open()))
    assert (float(row["sand"]), float(row["clay"])) == (40.0, 30.0)


def test_pythia_substrate_writes_every_layer_and_no_record(tmp_path):
    from prismpy.models.spatial import GridCell, SpatialGrid
    from prismpy.translators._shared.eghr_substrate import build_eghr_substrate

    layers = [_layer(i / 5, (i + 1) / 5, sand=40.0 + i, clay=30.0 - i) for i in range(5)]
    cell = GridCell(cell_id=1, lat=12.5, lon=-5.5, row=0, col=0, resolution="5arcmin")
    grid = SpatialGrid(resolution="5arcmin", cells=[cell])
    result = build_eghr_substrate(grid, {1: _profile(layers)}, "ML", REGION, tmp_path)
    sol = Path(result.sol_path).read_text().splitlines()
    assert not any(ln.startswith("!") for ln in sol)
    depths = [int(ln.split()[0]) for ln in sol if re.match(r"^\s+\d+\s+-9", ln)]
    assert depths == [20, 40, 60, 80, 100]
