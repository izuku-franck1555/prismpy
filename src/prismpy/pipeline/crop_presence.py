"""The crop-presence roster rule: after the boundary rule and min_share_percent, keep only the cells
whose harvested area in the rule's SPAM layer is > 0 ha. It is a roster definition, like
min_share_percent, so the cells it drops never enter the grid.

The rule records itself for the boundary block: the frozen and observed layer identity, and the
roster before the rule, after it and after user exclusions (stage 4), as counts and id digests at
5 arc-minutes and, when ACEA is a target, over the 30 arc-minute parents ACEA simulates.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from prismpy.cells.admission import cell_id_5arcmin_to_30arcmin_parent
from prismpy.cells.roster_digest import ROSTER_DIGEST_ENCODING, roster_id_digest
from prismpy.sources.crop_areas.presence import (
    CropPresenceEmptyError,
    CropPresenceIdentityError,
    cell_presence,
    crop_area_present,
)

CROP_PRESENCE_RULE = "spam_harvested_area_gt_0"


@dataclass
class CropPresenceOutcome:
    kept: List[Any]
    record: Dict[str, Any]


def _record_roster(record: Dict[str, Any], stage: str, ids: Iterable[int], acea_target: bool) -> None:
    ids = list(ids)
    record[f"n5_{stage}"] = len(ids)
    record[f"n5_{stage}_digest"] = roster_id_digest(ids)
    if acea_target:
        parents = {cell_id_5arcmin_to_30arcmin_parent(cid) for cid in ids}
        record[f"n30_{stage}"] = len(parents)
        record[f"n30_{stage}_digest"] = roster_id_digest(parents)


def apply_crop_presence_rule(cells, rule, raster_path, *, crop_name: str, acea_target: bool,
                             grid_resolution: str = "5arcmin") -> CropPresenceOutcome:
    """The cells with harvested area > 0 ha in the rule's layer, and the rule's record.

    Refuses (typed) a grid other than 5', a layer not resolved for this run or one that does not
    match the frozen identity; raises CropPresenceEmptyError when the rule leaves no cell of a
    non-empty grid."""
    if grid_resolution != "5arcmin":
        raise CropPresenceIdentityError(
            f"the crop-presence rule judges SPAM's 5-arcmin cells, not a {grid_resolution} grid")
    if not raster_path:
        raise CropPresenceIdentityError(f"the {rule.layer_label} layer was not resolved for this run")
    cells = list(cells)
    expected = rule.model_dump(mode="json")
    presence = cell_presence(cells, raster_path, expected=expected)
    kept = [c for c in cells if crop_area_present(presence.areas[c.cell_id])]
    if cells and not kept:  # a grid that arrives empty was emptied upstream, never by the layer
        raise CropPresenceEmptyError(
            f"no cell of this region has {crop_name} harvested area > 0 in {rule.layer_label}")
    record: Dict[str, Any] = {
        "rule": CROP_PRESENCE_RULE,
        "grid": "5arcmin",
        "digest_encoding": ROSTER_DIGEST_ENCODING,
        "identity_expected": expected,
        "identity_observed": presence.observed.as_dict(),
    }
    _record_roster(record, "before_rule", (c.cell_id for c in cells), acea_target)
    _record_roster(record, "after_rule", (c.cell_id for c in kept), acea_target)
    return CropPresenceOutcome(kept=kept, record=record)


def finalize_crop_presence_record(record: Dict[str, Any], final_cells, *, acea_target: bool) -> Dict[str, Any]:
    """Add the roster after user exclusions (stage 4) to the rule's record."""
    _record_roster(record, "final", (c.cell_id for c in final_cells), acea_target)
    return record


def crop_presence_record_digest(record: Dict[str, Any]) -> str:
    """A canonical digest of the record: every field in it is reproducible from the same layer and
    configuration (no path, run id or time), so a re-derived record must digest identically."""
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
