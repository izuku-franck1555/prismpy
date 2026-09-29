"""The crop-presence roster rule on real SPAM 2020 crops: its config identity, whole-grid byte
identity, the harmonize-stage cut with exact counts at 5' and 30', and the typed errors it raises."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from prismpy.cells.admission import cell_id_5arcmin_to_30arcmin_parent as parent
from prismpy.pipeline.executor import TranslationPipeline
from tests.unit._crop_presence_fixtures import (
    COAST_BOX, COAST_MAIZ, INLAND_BOX, INLAND_POTA, StopAfterGrid, box_cells, identity_of, make_config,
    mapped, region_of, run_grid_stages,
)

# The whole-grid config's hash input and YAML dump at the pre-rule base 4e763a1.
_BASE_DUMP_SHA256 = "1e43bdfeb6607a60cd2c18b9f013959cf0d5b32880136ce2cafe216a295d40e9"
_BASE_YAML_SHA256 = "f20f32d312296408692960ed0dc7c6db114271c6efa529d43e54877ee9106aef"
_LEGACY_BOUNDARY_KEYS = {
    "source", "version", "inclusion_rule", "min_share_percent", "n_cells_full_extent",
    "n_cells_excluded_by_inclusion_rule", "n_cells_excluded_by_min_share_percent", "n_cells_admitted",
}
_RULE_FIELDS = {"year", "release", "crop_code", "stratum", "sha256", "crs", "transform", "band",
                "width", "height", "nodata"}


def _rule(path=COAST_MAIZ, **override):
    from prismpy.config.schema import CropPresenceRule

    return CropPresenceRule(**{**identity_of(path), **override})


def _restricted(tmp_path, **kwargs):
    return make_config(tmp_path, rule=_rule(), rule_path=COAST_MAIZ, **kwargs)


def _digest(ids):
    from prismpy.cells.roster_digest import roster_id_digest

    return roster_id_digest(ids)


def _mapped_ids(ids):
    by_id = {c.cell_id: c for c in box_cells(COAST_BOX)}
    return {cid for cid in ids if mapped(COAST_MAIZ, by_id[cid].lat, by_id[cid].lon)}


# --- the config identity (F6) ------------------------------------------------------------------- #
def test_the_rule_is_an_identity_never_a_host_path():
    from prismpy.config.schema import CropPresenceRule

    ident = identity_of(COAST_MAIZ)
    assert set(CropPresenceRule.model_fields) == _RULE_FIELDS
    assert CropPresenceRule(**ident).sha256 == ident["sha256"]
    for extra in ({"raster_path": str(COAST_MAIZ)}, {"path": str(COAST_MAIZ)}):
        with pytest.raises(ValidationError):
            CropPresenceRule(**ident, **extra)
    for bad in ({"stratum": "R"}, {"sha256": "not-a-digest"}, {"band": 0}):
        with pytest.raises(ValidationError):
            CropPresenceRule(**{**ident, **bad})


def test_a_resolved_layer_needs_a_rule(tmp_path):
    with pytest.raises(ValidationError):
        make_config(tmp_path, rule_path=COAST_MAIZ)


def test_the_rule_needs_the_5_arcmin_grid_and_a_restricting_engine(tmp_path):
    with pytest.raises(ValidationError):
        make_config(tmp_path, rule=_rule(), targets=("pythia",), grid_resolution="30arcmin")
    with pytest.raises(ValidationError):
        make_config(tmp_path, rule=_rule(), targets=("sarra_py",))


def test_the_frozen_rule_resolves_on_another_data_root(tmp_path, monkeypatch):
    moved = tmp_path / "elsewhere" / COAST_MAIZ.name
    moved.parent.mkdir()
    shutil.copyfile(COAST_MAIZ, moved)
    _, _, ids, _ = run_grid_stages(make_config(tmp_path, rule=_rule(), rule_path=moved), monkeypatch)
    assert ids and set(ids) == _mapped_ids(ids)


# --- whole-grid byte identity (FY-24) ---------------------------------------------------------- #
def test_the_whole_grid_config_is_byte_identical(tmp_path):
    from prismpy.config.loader import save_config
    from prismpy.utils.sanitization import region_cache_key_from_config, region_cache_key_from_region

    cfg = make_config(Path("/golden"))
    for dump in (cfg.model_dump(), cfg.model_dump(mode="json")):
        assert not {"crop_presence", "crop_presence_path"} & set(dump["region"]["boundary"])
    digest = hashlib.sha256(json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()).hexdigest()
    assert digest == _BASE_DUMP_SHA256
    save_config(cfg, tmp_path / "config.yaml")
    assert hashlib.sha256((tmp_path / "config.yaml").read_bytes()).hexdigest() == _BASE_YAML_SHA256
    assert region_cache_key_from_config(cfg.region) == "manual_5.750000_7.250000_3.750000_5.250000"
    assert region_cache_key_from_region(region_of(cfg)) == "manual_5.750000_7.250000_3.750000_5.250000"


def test_the_identity_enters_the_hash_and_the_rule_stays_out_of_the_cache_key(tmp_path):
    from prismpy.utils.sanitization import region_cache_key_from_config

    restricted = _restricted(tmp_path)
    boundary = restricted.model_dump(mode="json")["region"]["boundary"]
    assert boundary["crop_presence"]["sha256"] == identity_of(COAST_MAIZ)["sha256"]
    assert region_cache_key_from_config(restricted.region) == region_cache_key_from_config(
        make_config(tmp_path).region)


# --- the harmonize-stage cut and its record (FY-1, FY-2, FY-3, FY-4) --------------------------- #
def test_the_rule_keeps_exactly_the_mapped_cells_with_exact_counts(tmp_path, monkeypatch):
    _, _, whole_ids, _ = run_grid_stages(make_config(tmp_path), monkeypatch)
    _, boundary, ids, _ = run_grid_stages(_restricted(tmp_path), monkeypatch)
    kept = _mapped_ids(whole_ids)

    assert set(ids) == kept and 0 < len(kept) < len(whole_ids)
    record = boundary["crop_presence"]
    assert (record["n5_before_rule"], record["n5_after_rule"], record["n5_final"]) == (
        len(whole_ids), len(kept), len(kept))
    assert record["n5_before_rule_digest"] == _digest(whole_ids)
    assert record["n5_after_rule_digest"] == record["n5_final_digest"] == _digest(kept)
    assert record["identity_expected"] == {k: v for k, v in identity_of(COAST_MAIZ).items()}
    assert record["identity_observed"]["sha256"] == record["identity_expected"]["sha256"]
    assert boundary["n_cells_admitted"] == len(kept)
    assert not any(key.startswith("n30_") for key in record)


def test_acea_records_the_30_arcmin_parents_any_child(tmp_path, monkeypatch):
    _, _, whole_ids, _ = run_grid_stages(make_config(tmp_path, targets=("acea",)), monkeypatch)
    _, boundary, ids, _ = run_grid_stages(_restricted(tmp_path, targets=("acea",)), monkeypatch)
    children = defaultdict(set)
    for cid in _mapped_ids(whole_ids):
        children[parent(cid)].add(cid)
    before = {parent(cid) for cid in whole_ids}
    one_child = {p for p, kids in children.items() if len(kids) == 1}
    emptied = before - set(children)
    assert one_child and emptied  # the fixture holds both shapes

    record = boundary["crop_presence"]
    after = {parent(cid) for cid in ids}
    assert one_child <= after and not emptied & after
    for stage, parents in (("before_rule", before), ("after_rule", after), ("final", after)):
        assert record[f"n30_{stage}"] == len(parents)
        assert record[f"n30_{stage}_digest"] == _digest(parents)


def test_user_exclusions_apply_after_the_rule_and_never_blame_spam(tmp_path, monkeypatch):
    _, _, whole_ids, _ = run_grid_stages(make_config(tmp_path), monkeypatch)
    kept = sorted(_mapped_ids(whole_ids))
    zero = next(cid for cid in whole_ids if cid not in kept)
    _, boundary, ids, _ = run_grid_stages(_restricted(tmp_path, exclude_cells=[kept[0], zero]), monkeypatch)
    record = boundary["crop_presence"]
    assert set(ids) == set(kept[1:])
    assert (record["n5_after_rule"], record["n5_final"]) == (len(kept), len(kept) - 1)
    assert record["n5_final_digest"] == _digest(kept[1:])


def test_a_roster_emptied_by_user_exclusions_keeps_its_route(tmp_path, monkeypatch):
    _, _, whole_ids, _ = run_grid_stages(make_config(tmp_path), monkeypatch)
    kept = sorted(_mapped_ids(whole_ids))
    result, boundary, ids, _ = run_grid_stages(_restricted(tmp_path, exclude_cells=kept), monkeypatch)
    assert ids == [] and result.error_events == []
    assert boundary["crop_presence"]["n5_final"] == 0


def test_the_boundary_record_is_unchanged_without_the_rule(tmp_path, monkeypatch):
    _, boundary, _, _ = run_grid_stages(make_config(tmp_path), monkeypatch)
    assert set(boundary) == _LEGACY_BOUNDARY_KEYS


def test_the_record_is_reproducible_and_digest_bound(tmp_path, monkeypatch):
    from prismpy.pipeline.crop_presence import crop_presence_record_digest

    _, first, _, _ = run_grid_stages(_restricted(tmp_path), monkeypatch)
    _, again, _, _ = run_grid_stages(_restricted(tmp_path / "again"), monkeypatch)
    record = first["crop_presence"]
    assert record == again["crop_presence"]
    assert str(COAST_MAIZ.parent) not in json.dumps(record)
    digest = crop_presence_record_digest(record)
    assert digest == crop_presence_record_digest(dict(reversed(list(record.items()))))
    assert digest != crop_presence_record_digest({**record, "n5_final": record["n5_final"] + 1})


# --- the typed errors (FY-7, FY-8, FY-9) ------------------------------------------------------- #
def _no_soil_fetch(self, grid, region):
    raise StopAfterGrid("the region reached the soil fetch")


def test_zero_mapped_cells_fail_the_real_pipeline_with_a_classified_event(tmp_path, monkeypatch):
    from prismpy.errors import classify_to_event_dict
    from prismpy.sources.crop_areas.presence import CropPresenceEmptyError

    monkeypatch.setattr(TranslationPipeline, "_load_climate_data", lambda self, region: {})
    monkeypatch.setattr(TranslationPipeline, "_load_soil_data", lambda self, region: None)
    monkeypatch.setattr(TranslationPipeline, "_retrieve_isda_api_for_grid", _no_soil_fetch)
    rule = _rule(INLAND_POTA, crop_code="POTA")
    cfg = make_config(tmp_path, rule=rule, rule_path=INLAND_POTA, box=INLAND_BOX,
                      targets=("pythia",), crop=("Potato", "pot"))
    result = TranslationPipeline(cfg).execute()

    harmonize = result.stages["harmonize"]
    message = "no cell of this region has Potato harvested area > 0 in SPAM 2020 V2r2 POTA_A"
    assert result.success is False and harmonize.success is False
    assert harmonize.errors == [f"Harmonization failed: {message}"]
    assert harmonize.error_events == [classify_to_event_dict(CropPresenceEmptyError(message))]
    assert "translate" not in result.stages


@pytest.mark.parametrize("override", [{"sha256": "0" * 64}, {"width": 25}])
def test_identity_drift_fails_loud_as_a_classified_event(tmp_path, monkeypatch, override):
    result, _, ids, _ = run_grid_stages(
        make_config(tmp_path, rule=_rule(**override), rule_path=COAST_MAIZ), monkeypatch)
    (event,) = result.error_events
    assert ids is None and event["error_class"] == "CropPresenceIdentityError"


def test_a_rule_without_its_resolved_layer_fails_loud(tmp_path, monkeypatch):
    result, _, ids, _ = run_grid_stages(make_config(tmp_path, rule=_rule()), monkeypatch)
    (event,) = result.error_events
    assert ids is None and event["error_class"] == "CropPresenceIdentityError"


def test_other_harmonize_failures_keep_their_route(tmp_path, monkeypatch):
    result, _, _, _ = run_grid_stages(make_config(tmp_path), monkeypatch)
    assert result.success is False and result.error_events == []
    assert result.errors == ["Harmonization failed: stopped after the grid stages"]
