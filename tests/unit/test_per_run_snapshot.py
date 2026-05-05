"""Sprint E.1 AC-E1-2 — pin the per-run Stage 1 verdict
snapshot helper at :mod:`prismpy.cockpit.per_run_snapshot`.

The Sprint F invariant CC-26 promised this snapshot shape but
left the actual write-site helper unimplemented. This module
ships the helper + pins behavior.
"""
from __future__ import annotations

import unittest

from prismpy.cockpit import (
    compute_snapshot_at_launch_hash,
    is_snapshot_unavailable,
    snapshot_for_pipeline_run,
)


class TestSnapshotForPipelineRun(unittest.TestCase):
    """Deep-copy the project-level snapshot for the per-run
    frozen field."""

    def test_none_input_returns_empty_dict(self):
        self.assertEqual(snapshot_for_pipeline_run(None), {})

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(snapshot_for_pipeline_run({}), {})

    def test_non_empty_input_deep_copies(self):
        src = {
            "schema_version": "stage_1_v1",
            "entries": [{"category": "crop_region_mismatch", "zone": "BSh"}],
        }
        snap = snapshot_for_pipeline_run(src)
        # Identity differs; equality holds.
        self.assertIsNot(snap, src)
        self.assertEqual(snap, src)
        # Mutating the snapshot doesn't leak back to the
        # source — frozen-at-launch contract.
        snap["entries"][0]["zone"] = "MUTATED"
        self.assertEqual(src["entries"][0]["zone"], "BSh")


class TestIsSnapshotUnavailable(unittest.TestCase):
    """Operational-failure path discriminator."""

    def test_clean_snapshot_is_not_unavailable(self):
        self.assertFalse(is_snapshot_unavailable({
            "schema_version": "stage_1_v1",
            "entries": [],
        }))

    def test_unavailable_snapshot_returns_true(self):
        self.assertTrue(is_snapshot_unavailable({
            "schema_version": "stage_1_v1",
            "stage_1_unavailable": True,
            "unavailable_reason": "classifier_error",
            "entries": [],
        }))

    def test_none_input_returns_false(self):
        self.assertFalse(is_snapshot_unavailable(None))

    def test_non_dict_returns_false(self):
        self.assertFalse(is_snapshot_unavailable("not a dict"))
        self.assertFalse(is_snapshot_unavailable([]))


class TestComputeSnapshotAtLaunchHash(unittest.TestCase):
    """Byte-stable hash for drift detection."""

    def test_hash_is_64_char_sha256_hex(self):
        h = compute_snapshot_at_launch_hash({
            "schema_version": "stage_1_v1",
            "entries": [],
        })
        self.assertEqual(len(h), 64)
        int(h, 16)

    def test_hash_stable_across_key_order(self):
        a = compute_snapshot_at_launch_hash({
            "schema_version": "stage_1_v1", "entries": [],
        })
        b = compute_snapshot_at_launch_hash({
            "entries": [], "schema_version": "stage_1_v1",
        })
        self.assertEqual(a, b)

    def test_empty_snapshot_hashes_consistently(self):
        a = compute_snapshot_at_launch_hash({})
        b = compute_snapshot_at_launch_hash(None)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
