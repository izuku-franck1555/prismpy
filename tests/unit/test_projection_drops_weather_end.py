"""A projection package covers other years than its base package, so the projection's manifest
rewrite drops the base package's declared weather end."""
from __future__ import annotations

import json
from types import SimpleNamespace

from prismpy.packaging import scenario_set_generator as ssg


def test_a_projection_rewrite_drops_the_base_packages_weather_end(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "temporal": {"start_year": 1999, "end_year": 2026, "spinup_years": 1,
                     "weather_end": "2026-10-10"},
        "data_sources": {"climate": "NASA POWER"}}))
    ssg._rewrite_projection_manifest(
        tmp_path, scenario_block=SimpleNamespace(model_dump=lambda: {}), gcm_source="GFDL-ESM4",
        rcp_or_ssp="ssp585", start_year=2041, end_year=2050, projection_climate={})
    assert json.loads((tmp_path / "manifest.json").read_text())["temporal"] == {
        "start_year": 2041, "end_year": 2050, "spinup_years": 1}
