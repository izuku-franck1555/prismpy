"""HWSD profiles: the dominant component's layers D1..D5 to 100 cm as a contiguous
prefix with valid texture, per-layer chemistry (organic/andic flagged, absent or
out-of-bounds values left to the writer's declared defaults), the real soil
mapping unit id, and no profile for a unit that is not a soil."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from prismpy.models.region import BoundingBox, Region
from prismpy.sources.soil.hwsd import HWSD_LAYER_CODES, HWSDConfig, HWSDSource

TOPS = (0, 20, 40, 60, 80)


def _region() -> Region:
    return Region(name="Koutiala", country="Mali", country_iso3="MLI",
                  bounds=BoundingBox(minx=-6.0, miny=12.0, maxx=-5.0, maxy=13.0))


def _row(smu, layer, top, bottom, sand=40.0, silt=30.0, clay=30.0, oc=1.0, ph=6.5,
         bd=1.4, seq=1, share=100):
    return {"ID": 200000 + smu, "HWSD2_SMU_ID": smu, "SEQUENCE": seq, "SHARE": share,
            "LAYER": layer, "TOPDEP": top, "BOTDEP": bottom, "SAND": sand, "SILT": silt,
            "CLAY": clay, "ORG_CARBON": oc, "PH_WATER": ph, "BULK": bd}


def _unit(smu, seq=1, share=100, **by_layer):
    """A full D1..D5 component; layer i gets distinct values unless overridden."""
    rows = []
    for i, top in enumerate(TOPS):
        values = dict(sand=40.0 + i, silt=30.0, clay=30.0 - i, oc=1.0 + i / 10,
                      ph=6.0 + i / 10, bd=1.3 + i / 100)
        values.update(by_layer.get(f"D{i + 1}", {}))
        rows.append(_row(smu, f"D{i + 1}", top, top + 20, seq=seq, share=share, **values))
    return rows


def _extract(rows, smu_ids):
    source = HWSDSource(HWSDConfig(bil_path=Path("x.bil"), mdb_path=Path("x.mdb")))
    coords = [(12.25 + i * 0.1, -5.5 + i * 0.1) for i in range(len(smu_ids))]
    with patch.object(source, "_sample_bil_raster", return_value=list(smu_ids)), \
            patch.object(source, "_export_mdb_table", return_value=pd.DataFrame(rows)):
        profiles = source._extract_from_bil_mdb(region=_region(), cell_coords=coords)
    return profiles, source


def _layer_values(layer):
    return (layer.depth_top, layer.depth_bottom, layer.sand, layer.silt, layer.clay,
            layer.organic_carbon, layer.ph, layer.bulk_density)


def test_complete_unit_gives_five_layers_each_from_its_own_row():
    profiles, _ = _extract(_unit(1469), [1469])
    profile = profiles[0]
    assert profile.total_depth == 1.0 and len(profile.layers) == 5
    for i, layer in enumerate(profile.layers):
        assert _layer_values(layer) == pytest.approx((
            TOPS[i] / 100, (TOPS[i] + 20) / 100, 40.0 + i, 30.0, 30.0 - i,
            1.0 + i / 10, 6.0 + i / 10, 1.3 + i / 100))


def test_dominant_component_by_share_then_lowest_sequence():
    minor = _unit(1469, seq=1, share=30, D1=dict(sand=10.0, silt=60.0, clay=30.0))
    major = _unit(1469, seq=2, share=70, D1=dict(sand=70.0, silt=10.0, clay=20.0))
    profiles, _ = _extract(minor + major, [1469])
    assert profiles[0].layers[0].sand == 70.0
    tied = _unit(1469, seq=2, share=50, D1=dict(sand=70.0, silt=10.0, clay=20.0)) + \
        _unit(1469, seq=1, share=50, D1=dict(sand=10.0, silt=60.0, clay=30.0))
    profiles, _ = _extract(tied, [1469])
    assert profiles[0].layers[0].sand == 10.0


def test_missing_duplicate_or_gapped_layer_stops_the_prefix():
    missing_d2 = [r for r in _unit(1469) if r["LAYER"] != "D2"]
    duplicate_d2 = _unit(1469) + [_row(1469, "D2", 20, 40, sand=50.0, silt=25.0, clay=25.0)]
    gapped_d2 = [dict(r, TOPDEP=25) if r["LAYER"] == "D2" else r for r in _unit(1469)]
    for rows in (missing_d2, duplicate_d2, gapped_d2):
        profiles, _ = _extract(rows, [1469])
        assert len(profiles[0].layers) == 1 and profiles[0].total_depth == 0.2
    d1_not_at_surface = [dict(r, TOPDEP=5) if r["LAYER"] == "D1" else r for r in _unit(1469)]
    profiles, source = _extract(d1_not_at_surface, [1469])
    assert profiles == {} and [e["cell_id"] for e in source.unavailable_cells] == [0]


def test_invalid_middle_layer_drops_every_layer_below_it():
    rows = _unit(1469, D3=dict(sand=-9.0, silt=-9.0, clay=-9.0))
    profiles, _ = _extract(rows, [1469])
    assert [layer.depth_bottom for layer in profiles[0].layers] == [0.2, 0.4]


@pytest.mark.parametrize("sentinel", [-9.0, -4.0, -7.0])
def test_texture_sentinel_truncates_before_the_layer(sentinel):
    rows = _unit(1469, D3=dict(sand=sentinel, silt=sentinel, clay=sentinel))
    profiles, _ = _extract(rows, [1469])
    assert len(profiles[0].layers) == 2


def test_extreme_but_plausible_chemistry_is_kept_and_flagged():
    """Real HWSD rows: SMU 9240 (BD 0.76, andic) and SMU 857 (OC 40.4, organic)."""
    rows = [
        _row(9240, "D1", 0, 20, sand=32, silt=51, clay=17, oc=8.7870007, ph=5.1999998, bd=0.75999999, share=90),
        _row(9240, "D2", 20, 40, sand=38, silt=46, clay=16, oc=4.0310001, ph=5.4000001, bd=0.86000001, share=90),
        _row(857, "D1", 0, 20, sand=30, silt=40, clay=30, oc=40.400002, ph=5.0, bd=0.27000001, share=60),
        _row(857, "D2", 20, 40, sand=33, silt=41, clay=26, oc=39.056999, ph=5.0999999, bd=0.31999999, share=60),
        _row(1469, "D1", 0, 20),
    ]
    profiles, _ = _extract(rows, [9240, 857, 1469])
    andic, organic, mineral = profiles[0], profiles[1], profiles[2]
    assert andic.layers[0].bulk_density == pytest.approx(0.76)
    assert organic.layers[0].organic_carbon == pytest.approx(40.4)
    assert andic.metadata["ptf_domain_flags"] == {0: "andic", 1: "andic"}
    assert organic.metadata["ptf_domain_flags"] == {0: "organic", 1: "organic"}
    assert "ptf_domain_flags" not in mineral.metadata


def test_flag_boundaries_are_organic_at_20_percent_and_andic_below_0_9():
    rows = [_row(1, "D1", 0, 20, oc=20.0, bd=1.2), _row(2, "D1", 0, 20, oc=1.0, bd=0.9),
            _row(3, "D1", 0, 20, oc=19.99, bd=0.89)]
    profiles, _ = _extract(rows, [1, 2, 3])
    assert profiles[0].metadata["ptf_domain_flags"] == {0: "organic"}
    assert "ptf_domain_flags" not in profiles[1].metadata
    assert profiles[2].metadata["ptf_domain_flags"] == {0: "andic"}


def test_absent_or_out_of_bounds_chemistry_is_left_to_the_writer_default():
    rows = _unit(1469, D1=dict(bd=-9.0), D2=dict(oc=float("nan")), D3=dict(bd=2.4))
    profiles, _ = _extract(rows, [1469])
    profile = profiles[0]
    assert len(profile.layers) == 5
    assert profile.layers[0].bulk_density is None
    assert profile.layers[1].organic_carbon is None
    assert profile.layers[2].bulk_density is None
    assert profile.metadata["chem_defaulted"] == {
        0: ["bulk_density"], 1: ["organic_carbon"], 2: ["bulk_density"]}


def test_a_non_soil_or_absent_unit_has_no_profile_and_is_recorded_once():
    non_soil = [_row(7001, "D1", 0, 20, sand=-9.0, silt=-9.0, clay=-9.0),
                _row(7001, "D2", 20, 40, sand=-9.0, silt=-9.0, clay=-9.0)]
    profiles, source = _extract(_unit(1469) + non_soil, [1469, 7001, 999999])
    assert set(profiles) == {0}
    assert sorted(e["cell_id"] for e in source.unavailable_cells) == [1, 2]
    all_miss = HWSDSource(HWSDConfig(bil_path=Path("x.bil"), mdb_path=Path("x.mdb")))
    with patch.object(all_miss, "_sample_bil_raster", return_value=[7001, 999999]), \
            patch.object(all_miss, "_export_mdb_table", return_value=pd.DataFrame(non_soil)):
        result = all_miss.retrieve(region=_region(), cell_coords=[(12.25, -5.5), (12.35, -5.4)])
    assert result.success is False
    assert sorted(e["cell_id"] for e in all_miss.unavailable_cells) == [0, 1]


def test_the_profile_carries_the_real_mapping_unit_id():
    profiles, _ = _extract(_unit(1469), [1469])
    assert profiles[0].metadata["hwsd_smu_id"] == 1469


def test_config_accepts_only_the_top_layer_name():
    assert HWSDConfig().layers == HWSD_LAYER_CODES == ("D1", "D2", "D3", "D4", "D5")
    HWSDConfig(layer="D1")
    with pytest.raises(ValueError):
        HWSDConfig(layer="D2")


def test_texture_sum_tolerance_is_two_percent():
    rows = _unit(1, D2=dict(sand=45.0, silt=30.0, clay=30.0)) + \
        _unit(2, D2=dict(sand=41.5, silt=30.0, clay=30.0))
    profiles, _ = _extract(rows, [1, 2])
    assert len(profiles[0].layers) == 1
    assert len(profiles[1].layers) == 5
    derived = _unit(3, D2=dict(sand=70.0, silt=None, clay=40.0))
    profiles, _ = _extract(derived, [3])
    assert len(profiles[0].layers) == 1
