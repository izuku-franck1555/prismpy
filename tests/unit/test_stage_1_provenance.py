"""Sprint F AC-F-7 — Stage 1 verdict snapshot in provenance.

Pins:

* :attr:`ProvenanceRecord.stage_1_verdicts_snapshot` defaults
  to ``None``.
* :meth:`ProvenanceTracker.record_stage_1_verdicts` populates
  the field with a defensive copy.
* :meth:`ProvenanceRecord.to_dict` omits the key when ``None``
  (legacy back-compat) and includes it when populated.

Anti-mutation drills:

* Drop the field-omission guard so ``None`` always serializes
  → legacy provenance.json consumers reading via
  ``.get("stage_1_verdicts_snapshot", default)`` may get
  ``None`` where they expected a missing key →
  ``test_to_dict_omits_snapshot_when_none`` fails.
* Drop the defensive copy → mutation of the passed-in dict
  retroactively alters the recorded snapshot →
  ``test_record_stores_defensive_copy`` fails.
"""
from __future__ import annotations

import unittest

from prismpy.models.provenance import ProvenanceRecord
from prismpy.provenance import ProvenanceTracker


_STAGE_1_SNAPSHOT_FIXTURE = {
    "schema_version": "stage_1_v1",
    "cache_key": "abc123",
    "created_at": "2026-05-04T12:00:00Z",
    "substrate_versions": {
        "bounds_version": "frozen_v1",
        "zone_classifier_version": "beck_2023_v1",
        "ecocrop_envelope_version": "ecocrop_v2_2026-05-04",
    },
    "entries": [
        {
            "category": "crop_region_mismatch",
            "zone": "BSh",
            "zone_label": "Hot semi-arid",
            "crop": "rice",
            "verdict": "incompatible",
            "reason": (
                "Rice requires ≥1000mm/yr per FAO ECOCROP "
                "(...); BSh P50 = 280mm/yr."
            ),
            "n_cell_days_in_zone": 4_614_540,
            "stage_1_verdict_id": "abc-def",
            "override_decision_id": None,
            "override_status": "none",
        },
    ],
}


class TestStage1VerdictsField(unittest.TestCase):
    """Pin the additive field defaulting to ``None``."""

    def test_default_is_none(self):
        record = ProvenanceRecord(session_id="s")
        self.assertIsNone(record.stage_1_verdicts_snapshot)

    def test_assignable(self):
        record = ProvenanceRecord(session_id="s")
        record.stage_1_verdicts_snapshot = {"k": "v"}
        self.assertEqual(record.stage_1_verdicts_snapshot, {"k": "v"})


class TestProvenanceTrackerRecordStage1(unittest.TestCase):
    """Pin :meth:`ProvenanceTracker.record_stage_1_verdicts`."""

    def setUp(self):
        self.tracker = ProvenanceTracker(project_name="test")

    def test_record_populates_snapshot_field(self):
        self.tracker.record_stage_1_verdicts(_STAGE_1_SNAPSHOT_FIXTURE)
        self.assertEqual(
            self.tracker.record.stage_1_verdicts_snapshot,
            _STAGE_1_SNAPSHOT_FIXTURE,
        )

    def test_record_stores_defensive_copy(self):
        # Mutating the passed-in dict after recording must NOT
        # alter the recorded snapshot. AC-F-7 + codex Gate A
        # #14 audit-trail pin.
        snapshot = {
            "schema_version": "stage_1_v1",
            "entries": [],
        }
        self.tracker.record_stage_1_verdicts(snapshot)
        snapshot["entries"].append({"injected": True})
        self.assertEqual(
            self.tracker.record.stage_1_verdicts_snapshot["entries"],
            [],
            "record_stage_1_verdicts must store a defensive "
            "copy; mutation leaked through.",
        )

    def test_record_disabled_tracker_is_noop(self):
        # Mirrors the existing setter pattern — disabled tracker
        # silently skips the write, leaving the field at None.
        disabled = ProvenanceTracker(enabled=False)
        disabled.record_stage_1_verdicts(_STAGE_1_SNAPSHOT_FIXTURE)
        self.assertIsNone(disabled.record.stage_1_verdicts_snapshot)


class TestProvenanceToDictBackCompat(unittest.TestCase):
    """Pin the to_dict serialization back-compat (codex Gate A
    #14 — explicit field, omit when None)."""

    def test_to_dict_omits_snapshot_when_none(self):
        record = ProvenanceRecord(session_id="s")
        out = record.to_dict()
        self.assertNotIn("stage_1_verdicts_snapshot", out)

    def test_to_dict_includes_snapshot_when_populated(self):
        record = ProvenanceRecord(session_id="s")
        record.stage_1_verdicts_snapshot = _STAGE_1_SNAPSHOT_FIXTURE
        out = record.to_dict()
        self.assertIn("stage_1_verdicts_snapshot", out)
        self.assertEqual(
            out["stage_1_verdicts_snapshot"],
            _STAGE_1_SNAPSHOT_FIXTURE,
        )

    def test_to_dict_serializes_defensive_copy(self):
        # Mutating the dict returned by to_dict() must not
        # alter the underlying record's stored snapshot.
        record = ProvenanceRecord(session_id="s")
        record.stage_1_verdicts_snapshot = dict(
            _STAGE_1_SNAPSHOT_FIXTURE
        )
        out = record.to_dict()
        out["stage_1_verdicts_snapshot"]["mutated"] = True
        self.assertNotIn(
            "mutated",
            record.stage_1_verdicts_snapshot,
            "to_dict() must defensively copy the snapshot so "
            "downstream serializers cannot mutate the record.",
        )


if __name__ == "__main__":
    unittest.main()
