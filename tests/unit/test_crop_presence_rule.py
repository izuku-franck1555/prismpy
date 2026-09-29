"""The crop-presence roster rule: its config identity, whole-grid byte identity, the harmonize-stage
cut with exact per-resolution counts, and the typed errors it raises."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from prismpy.models.spatial import SpatialGrid
from tests.unit._crop_presence_fixtures import (
    NORTH, V2R2_NODATA, WEST, INC, identity_of, make_config, run_grid_stages, write_layer,
)

# The whole-grid config's hash input and YAML dump at the pre-rule base 4e763a1.
_BASE_DUMP_SHA256 = "8d1c7d68cfefc7ef3b0d5730205648f865d2e8f4d76911cc31a1172cd4348ecf"
_BASE_YAML_SHA256 = "cd7e759591bc517d14a99ed2ac579d6a255d14be9fb219cbabcbc6ea17fdcfe2"
_LEGACY_BOUNDARY_KEYS = {
    "source", "version", "inclusion_rule", "min_share_percent", "n_cells_full_extent",
    "n_cells_excluded_by_inclusion_rule", "n_cells_excluded_by_min_share_percent", "n_cells_admitted",
}
# Fixture pixels (row, col): four mapped cells inside the region, one outside it, and no-area pixels.
_MAPPED = {(4, 4): 12.0, (5, 9): 0.5, (10, 12): 3.0, (15, 15): 1.0}
_OTHERS = {(1, 1): 9.0, (6, 6): V2R2_NODATA, (7, 7): -5.0, (8, 8): np.nan}


def _layer(tmp_path, pixels=None):
    arr = np.zeros((18, 18), dtype="float32")
    for (row, col), value in (pixels if pixels is not None else {**_MAPPED, **_OTHERS}).items():
        arr[row, col] = value
    return write_layer(tmp_path / "spam2020_V2r2_global_H_MAIZ_A.tif", arr)


def _rule(path, **override):
    from prismpy.config.schema import CropPresenceRule

    return CropPresenceRule(**{**identity_of(path), **override})


def _present(path, cell_id):
    """Independent read of the fixture: harvested area > 0 at the cell's pixel."""
    import rasterio

    row, col = divmod(cell_id, SpatialGrid.GLOBAL_COLS_5ARCMIN)
    lat, lon = SpatialGrid.latlon_from_rowcol_5arcmin(row, col)
    with rasterio.open(path) as src:
        value = float(src.read(1)[int((NORTH - lat) / INC), int((lon - WEST) / INC)])
    return math.isfinite(value) and value != V2R2_NODATA and value > 0


def _digest(ids):
    from prismpy.cells.roster_digest import roster_id_digest

    return roster_id_digest(ids)


# --- the config identity ---------------------------------------------------------------------- #
def test_the_rule_is_an_identity_never_a_host_path(tmp_path):
    from prismpy.config.schema import CropPresenceRule

    ident = identity_of(write_layer(tmp_path / "layer.tif"))
    assert CropPresenceRule(**ident).sha256 == ident["sha256"]
    for extra in ({"raster_path": "/data/layer.tif"}, {"path": "/data/layer.tif"}):
        with pytest.raises(ValidationError):
            CropPresenceRule(**ident, **extra)
    for bad in ({"stratum": "R"}, {"sha256": "not-a-digest"}, {"band": 0}):
        with pytest.raises(ValidationError):
            CropPresenceRule(**{**ident, **bad})


def test_a_resolved_layer_needs_a_rule(tmp_path):
    with pytest.raises(ValidationError):
        make_config(tmp_path, rule_path=tmp_path / "layer.tif")


def test_the_rule_needs_the_5_arcmin_grid_and_a_restricting_engine(tmp_path):
    rule = _rule(write_layer(tmp_path / "layer.tif"))
    with pytest.raises(ValidationError):
        make_config(tmp_path, rule=rule, targets=("pythia",), grid_resolution="30arcmin")
    with pytest.raises(ValidationError):
        make_config(tmp_path, rule=rule, targets=("sarra_py",))


def test_the_whole_grid_config_is_byte_identical(tmp_path):
    from prismpy.config.loader import save_config

    cfg = make_config(Path("/golden"))
    for dump in (cfg.model_dump(), cfg.model_dump(mode="json")):
        assert not {"crop_presence", "crop_presence_path"} & set(dump["region"]["boundary"])
    digest = hashlib.sha256(json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()).hexdigest()
    assert digest == _BASE_DUMP_SHA256
    save_config(cfg, tmp_path / "config.yaml")
    assert hashlib.sha256((tmp_path / "config.yaml").read_bytes()).hexdigest() == _BASE_YAML_SHA256


def test_the_identity_enters_the_hash_and_the_rule_stays_out_of_the_cache_key(tmp_path):
    from prismpy.utils.sanitization import region_cache_key_from_config

    path = write_layer(tmp_path / "layer.tif")
    restricted = make_config(tmp_path, rule=_rule(path), rule_path=path)
    boundary = restricted.model_dump(mode="json")["region"]["boundary"]
    assert boundary["crop_presence"]["sha256"] == identity_of(path)["sha256"]
    whole = make_config(tmp_path)
    assert region_cache_key_from_config(restricted.region) == region_cache_key_from_config(whole.region)


# --- the harmonize-stage cut and its record ---------------------------------------------------- #
def test_the_rule_keeps_exactly_the_mapped_cells_with_exact_counts(tmp_path, monkeypatch):
    path = _layer(tmp_path)
    _, _, whole_ids = run_grid_stages(make_config(tmp_path), monkeypatch)
    _, boundary, ids = run_grid_stages(make_config(tmp_path, rule=_rule(path), rule_path=path), monkeypatch)
    mapped = {cid for cid in whole_ids if _present(path, cid)}

    assert set(ids) == mapped and len(mapped) == len(_MAPPED)
    record = boundary["crop_presence"]
    assert (record["n5_before_rule"], record["n5_after_rule"], record["n5_final"]) == (
        len(whole_ids), len(mapped), len(mapped))
    assert record["n5_before_rule_digest"] == _digest(whole_ids)
    assert record["n5_after_rule_digest"] == record["n5_final_digest"] == _digest(mapped)
    assert record["identity_observed"]["sha256"] == record["identity_expected"]["sha256"]
    assert boundary["n_cells_admitted"] == len(mapped)
    assert not any(key.startswith("n30_") for key in record)


def test_acea_records_the_30_arcmin_parents(tmp_path, monkeypatch):
    from prismpy.cells.admission import cell_id_5arcmin_to_30arcmin_parent as parent

    path = _layer(tmp_path)
    _, _, whole_ids = run_grid_stages(make_config(tmp_path, targets=("acea",)), monkeypatch)
    _, boundary, ids = run_grid_stages(
        make_config(tmp_path, rule=_rule(path), rule_path=path, targets=("acea",)), monkeypatch)
    record = boundary["crop_presence"]
    for stage, cells in (("before_rule", whole_ids), ("after_rule", ids), ("final", ids)):
        parents = {parent(cid) for cid in cells}
        assert record[f"n30_{stage}"] == len(parents)
        assert record[f"n30_{stage}_digest"] == _digest(parents)


def test_user_exclusions_still_apply_after_the_rule(tmp_path, monkeypatch):
    path = _layer(tmp_path)
    _, _, whole_ids = run_grid_stages(make_config(tmp_path), monkeypatch)
    mapped = sorted(cid for cid in whole_ids if _present(path, cid))
    unmapped = next(cid for cid in whole_ids if cid not in mapped)
    _, boundary, ids = run_grid_stages(
        make_config(tmp_path, rule=_rule(path), rule_path=path, exclude_cells=[mapped[0], unmapped]),
        monkeypatch)
    record = boundary["crop_presence"]
    assert set(ids) == set(mapped[1:])
    assert (record["n5_after_rule"], record["n5_final"]) == (len(mapped), len(mapped) - 1)
    assert record["n5_final_digest"] == _digest(mapped[1:])


def test_the_boundary_record_is_unchanged_without_the_rule(tmp_path, monkeypatch):
    _, boundary, _ = run_grid_stages(make_config(tmp_path), monkeypatch)
    assert set(boundary) == _LEGACY_BOUNDARY_KEYS


def test_the_record_is_reproducible_and_digest_bound(tmp_path, monkeypatch):
    from prismpy.pipeline.crop_presence import crop_presence_record_digest

    path = _layer(tmp_path)
    _, first, _ = run_grid_stages(make_config(tmp_path, rule=_rule(path), rule_path=path), monkeypatch)
    _, again, _ = run_grid_stages(
        make_config(tmp_path / "again", rule=_rule(path), rule_path=path), monkeypatch)
    record = first["crop_presence"]
    assert record == again["crop_presence"]
    assert str(path) not in json.dumps(record)
    digest = crop_presence_record_digest(record)
    assert digest == crop_presence_record_digest(dict(reversed(list(record.items()))))
    assert digest != crop_presence_record_digest({**record, "n5_final": record["n5_final"] + 1})


# --- the typed errors reach the stage's error events ------------------------------------------- #
def _single_event(result):
    assert result.success is False
    (event,) = result.error_events
    return event


def test_zero_mapped_cells_fail_loud_as_a_classified_event(tmp_path, monkeypatch):
    path = _layer(tmp_path, {(1, 1): 9.0})
    result, _, ids = run_grid_stages(make_config(tmp_path, rule=_rule(path), rule_path=path), monkeypatch)
    event = _single_event(result)
    assert ids is None and event["error_class"] == "CropPresenceEmptyError"
    assert "Maize" in event["message"] and "V2r2" in event["message"]


@pytest.mark.parametrize("override", [{"sha256": "0" * 64}, {"width": 19}])
def test_identity_drift_fails_loud_as_a_classified_event(tmp_path, monkeypatch, override):
    path = _layer(tmp_path)
    result, _, ids = run_grid_stages(
        make_config(tmp_path, rule=_rule(path, **override), rule_path=path), monkeypatch)
    assert ids is None and _single_event(result)["error_class"] == "CropPresenceIdentityError"


def test_a_rule_without_its_resolved_layer_fails_loud(tmp_path, monkeypatch):
    path = _layer(tmp_path)
    result, _, ids = run_grid_stages(make_config(tmp_path, rule=_rule(path)), monkeypatch)
    assert ids is None and _single_event(result)["error_class"] == "CropPresenceIdentityError"


def test_other_harmonize_failures_keep_their_route(tmp_path, monkeypatch):
    result, _, _ = run_grid_stages(make_config(tmp_path), monkeypatch)
    assert result.success is False and result.error_events == []
    assert any("stopped after the grid stages" in error for error in result.errors)
