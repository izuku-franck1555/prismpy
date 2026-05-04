"""Sprint F AC-F-6 + AC-F-7 — wizard override replay coupling.

Pins the end-to-end stale-override detection contract:

* Recording a Stage 1 verdict snapshot via
  :meth:`ProvenanceTracker.record_stage_1_verdicts` produces
  a deterministic hash via :func:`compute_verdict_hash`.
* An override created with that hash and replayed via
  :meth:`ProvenanceTracker.record_wizard_decision` lands in
  the run's provenance trail with
  :data:`DecisionType.USER_OVERRIDE`.
* A subsequent verdict snapshot that differs (e.g., bumped
  substrate version) produces a different hash, demonstrating
  the stale-override detection signal that prismweb wires
  into its cache-invalidation logic per AC-F-5.

Codex Gate A second-round Dim 11 (HIGH) flagged this end-to-
end test as missing; this file closes the gap.
"""
from __future__ import annotations

import unittest

from prismpy.models.provenance import DecisionType
from prismpy.provenance import (
    ProvenanceTracker,
    WizardOverrideRecord,
    compute_verdict_hash,
)


_SNAPSHOT_BSh_RICE_INCOMPATIBLE = {
    "schema_version": "stage_1_v1",
    "cache_key": "test-coupling-1",
    "substrate_versions": {
        "bounds_version": "frozen_v1",
        "zone_classifier_version": "beck_2023_v1",
        "ecocrop_envelope_version": "ecocrop_v2_2026-05-04",
    },
    "entries": [
        {
            "category": "crop_region_mismatch",
            "zone": "BSh",
            "crop": "rice",
            "verdict": "incompatible",
        },
    ],
}

_SNAPSHOT_BSh_RICE_AFTER_SUBSTRATE_BUMP = {
    "schema_version": "stage_1_v1",
    "cache_key": "test-coupling-2",
    "substrate_versions": {
        "bounds_version": "frozen_v2",  # BUMPED
        "zone_classifier_version": "beck_2023_v1",
        "ecocrop_envelope_version": "ecocrop_v2_2026-05-04",
    },
    "entries": [
        {
            "category": "crop_region_mismatch",
            "zone": "BSh",
            "crop": "rice",
            "verdict": "incompatible",
        },
    ],
}


class TestWizardOverrideHashCoupling(unittest.TestCase):
    """Pin the end-to-end coupling between
    :meth:`record_stage_1_verdicts` and
    :func:`compute_verdict_hash` — the contract Sprint F
    relies on for stale-override detection."""

    def setUp(self):
        self.tracker = ProvenanceTracker(project_name="test")

    def test_recorded_snapshot_hashes_consistently(self):
        # The hash computed at override-creation time MUST
        # equal the hash recomputed against the snapshot
        # serialized into provenance.json. Anything else
        # breaks the stale-override detection.
        self.tracker.record_stage_1_verdicts(
            _SNAPSHOT_BSh_RICE_INCOMPATIBLE,
        )
        recorded = self.tracker.record.stage_1_verdicts_snapshot
        self.assertIsNotNone(recorded)

        hash_at_record_time = compute_verdict_hash(
            _SNAPSHOT_BSh_RICE_INCOMPATIBLE,
        )
        hash_after_record = compute_verdict_hash(recorded)

        self.assertEqual(
            hash_at_record_time, hash_after_record,
            "Hash drift between override-creation snapshot "
            "and persisted snapshot — stale-override "
            "detection contract broken.",
        )

    def test_substrate_version_bump_invalidates_hash(self):
        # AC-F-5 cache invalidation: when bounds_version (or
        # any substrate-version stamp) bumps, the verdict
        # snapshot recomputes and the hash changes. An
        # override stored against the OLD hash must surface
        # as stale.
        old_hash = compute_verdict_hash(
            _SNAPSHOT_BSh_RICE_INCOMPATIBLE,
        )
        new_hash = compute_verdict_hash(
            _SNAPSHOT_BSh_RICE_AFTER_SUBSTRATE_BUMP,
        )
        self.assertNotEqual(
            old_hash, new_hash,
            "Substrate-version bump must produce a different "
            "verdict hash so stale-override detection fires.",
        )


class TestRecordWizardDecisionReplay(unittest.TestCase):
    """Pin the prismpy-side replay helper that takes a saved
    wizard override payload and records it via
    :data:`DecisionType.USER_OVERRIDE` in the run's
    provenance trail."""

    def setUp(self):
        self.tracker = ProvenanceTracker(project_name="test")
        self._VALID_HASH = compute_verdict_hash(
            _SNAPSHOT_BSh_RICE_INCOMPATIBLE,
        )
        self._VALID_RATIONALE = (
            "We have a documented local trial proving the "
            "cultivar performs in this zone — proving with "
            "field data. Specifically yields exceeded 4 t/ha."
        )

    def _build_record(self) -> WizardOverrideRecord:
        return WizardOverrideRecord(
            rationale=self._VALID_RATIONALE,
            evidence_type="local_trial",
            affected_zones=["BSh"],
            verdict_hash=self._VALID_HASH,
        )

    def _collect_decisions(self):
        """Walk both attached + pending decisions, returning
        the underlying ``DecisionRecord`` objects.

        ``_pending_decisions`` items are
        ``(DecisionRecord, bound_artifact_id)`` tuples per
        ``tracker.py:474``; this helper unpacks them so the
        tests assert on the record directly regardless of
        which surface the replay landed on.
        """
        records = []
        for lineage in self.tracker.record.artifacts.values():
            records.extend(lineage.all_decisions)
        for entry in self.tracker._pending_decisions:
            decision = entry[0] if isinstance(entry, tuple) else entry
            records.append(decision)
        return records

    def test_replay_accepts_typed_record(self):
        record = self._build_record()
        self.tracker.record_wizard_decision(record)
        records = self._collect_decisions()
        self.assertGreaterEqual(len(records), 1)
        latest = records[-1]
        self.assertEqual(
            latest.decision_type, DecisionType.USER_OVERRIDE,
        )

    def test_replay_accepts_dict_payload(self):
        # Prismweb persists the dict shape; the replay path
        # accepts that directly without forcing the caller to
        # re-construct the typed record.
        record = self._build_record()
        from prismpy.provenance import build_wizard_override_payload
        payload = build_wizard_override_payload(record)
        self.tracker.record_wizard_decision(payload)
        records = self._collect_decisions()
        self.assertGreaterEqual(len(records), 1)

    def test_replay_rejects_malformed_dict(self):
        from pydantic import ValidationError
        bad_payload = {
            "rationale": "too short",  # below 50-char floor
            "evidence_type": "local_trial",
            "affected_zones": ["BSh"],
            "verdict_hash": self._VALID_HASH,
        }
        with self.assertRaises(ValidationError):
            self.tracker.record_wizard_decision(bad_payload)

    def test_replay_carries_zones_in_decision(self):
        record = WizardOverrideRecord(
            rationale=self._VALID_RATIONALE,
            evidence_type="local_trial",
            affected_zones=["BSh", "Aw"],
            verdict_hash=self._VALID_HASH,
        )
        self.tracker.record_wizard_decision(record)
        records = self._collect_decisions()
        latest = records[-1]
        # Zones land in the rationale free-text per builder
        # Adj-12 (V2-23 polish extends DecisionRecord with
        # first-class structured fields).
        self.assertIn("BSh", latest.rationale)
        self.assertIn("Aw", latest.rationale)
        self.assertIn("local_trial", latest.rationale)

    def test_replay_disabled_tracker_is_noop(self):
        disabled = ProvenanceTracker(enabled=False)
        record = self._build_record()
        # Should silently skip — disabled tracker stays empty.
        disabled.record_wizard_decision(record)
        self.assertEqual(disabled.record.artifacts, {})


if __name__ == "__main__":
    unittest.main()
