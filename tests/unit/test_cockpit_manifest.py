"""Sprint E.1 AC-E1-1 — pin the cockpit warning manifest
shape + builder semantics + byte-stable hash.

Producer (pipeline finish) and consumer (cockpit template +
Alpine state) read the same payload; this module pins the
shape so a drift on either side surfaces at fast tier rather
than at runtime.
"""
from __future__ import annotations

import json
import unittest

from prismpy.cockpit import (
    CockpitManifestEntry,
    UnknownCategoryError,
    build_cockpit_warning_manifest,
    compute_manifest_hash,
)


class TestCockpitManifestEntryShape(unittest.TestCase):
    """Frozen dataclass + canonical projection."""

    def test_frozen_dataclass_rejects_mutation(self):
        e = CockpitManifestEntry(
            entry_id="0" * 64, bucket=2,
            category="climate_envelope_tail",
            affected_cells=("c1",),
            count=1,
        )
        with self.assertRaises(Exception):
            e.bucket = 3  # type: ignore[misc]

    def test_to_dict_carries_canonical_keys(self):
        e = CockpitManifestEntry(
            entry_id="abc", bucket=2, category="x",
            affected_cells=("c1",),
            affected_zones=tuple(),
            count=1,
        )
        d = e.to_dict()
        self.assertEqual(
            set(d.keys()),
            {"entry_id", "bucket", "category",
             "affected_cells", "affected_zones", "count"},
        )

    def test_default_factories_yield_empty_tuples(self):
        e = CockpitManifestEntry(
            entry_id="x", bucket=2, category="y",
        )
        self.assertEqual(e.affected_cells, tuple())
        self.assertEqual(e.affected_zones, tuple())


class TestBuildManifestFromCellSummary(unittest.TestCase):
    """Per-cell warnings (Bucket 2 INFO + Bucket 3 EXCLUDE)."""

    def test_climate_envelope_tail_routes_to_bucket_2(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={
                "climate_envelope_tail": {"cells": ["c1", "c2", "c3"]},
            },
        )
        self.assertEqual(len(manifest), 1)
        entry = manifest[0]
        self.assertEqual(entry.bucket, 2)
        self.assertEqual(entry.category, "climate_envelope_tail")
        self.assertEqual(entry.count, 3)
        self.assertEqual(
            entry.affected_cells, ("c1", "c2", "c3"),
        )

    def test_soil_no_hwsd_coverage_routes_to_bucket_3(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={
                "soil_no_hwsd_coverage": {"cells": ["c1"]},
            },
        )
        self.assertEqual(manifest[0].bucket, 3)

    def test_unknown_category_raises(self):
        with self.assertRaises(UnknownCategoryError) as cm:
            build_cockpit_warning_manifest(
                parent_run_id="r1",
                cell_summary={"not_a_real_category": {"cells": ["c1"]}},
            )
        self.assertIn("not_a_real_category", str(cm.exception))

    def test_empty_cell_list_is_skipped(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={
                "climate_envelope_tail": {"cells": []},
            },
        )
        self.assertEqual(manifest, [])

    def test_cells_are_sorted_in_output(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={
                "climate_envelope_tail": {
                    "cells": ["z9", "a1", "m5"],
                },
            },
        )
        self.assertEqual(
            manifest[0].affected_cells,
            ("a1", "m5", "z9"),
        )


class TestBuildManifestFromStage1Verdicts(unittest.TestCase):
    """Per-zone wizard-time entries (Bucket 5 OVERRIDE)."""

    def test_crop_region_mismatch_routes_to_bucket_5(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            stage_1_verdicts={
                "entries": [
                    {
                        "category": "crop_region_mismatch",
                        "zone": "BSh",
                        "crop": "rice",
                    },
                ],
            },
        )
        self.assertEqual(len(manifest), 1)
        entry = manifest[0]
        self.assertEqual(entry.bucket, 5)
        self.assertEqual(entry.category, "crop_region_mismatch")
        self.assertEqual(entry.affected_zones, ("BSh",))
        self.assertEqual(entry.count, 1)

    def test_multi_zone_collapses_to_single_entry(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            stage_1_verdicts={
                "entries": [
                    {"category": "crop_region_mismatch", "zone": "BSh"},
                    {"category": "crop_region_mismatch", "zone": "BWh"},
                ],
            },
        )
        self.assertEqual(len(manifest), 1)
        # Sorted; the cockpit's match-disjoint algorithm reads
        # this as one row with affected_zones = (BSh, BWh).
        self.assertEqual(
            manifest[0].affected_zones, ("BSh", "BWh"),
        )

    def test_zones_dedupe(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            stage_1_verdicts={
                "entries": [
                    {"category": "crop_region_mismatch", "zone": "BSh"},
                    {"category": "crop_region_mismatch", "zone": "BSh"},
                ],
            },
        )
        self.assertEqual(manifest[0].affected_zones, ("BSh",))


class TestEntryIdStability(unittest.TestCase):
    """Stable hash over join key — re-renders dedupe cleanly."""

    def test_entry_id_is_64_char_sha256_hex(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={"climate_envelope_tail": {"cells": ["c1"]}},
        )
        entry_id = manifest[0].entry_id
        self.assertEqual(len(entry_id), 64)
        int(entry_id, 16)

    def test_entry_id_stable_across_call(self):
        first = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={"climate_envelope_tail": {"cells": ["c1", "c2"]}},
        )
        second = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={"climate_envelope_tail": {"cells": ["c2", "c1"]}},
        )
        # Cell-order should not affect entry_id.
        self.assertEqual(first[0].entry_id, second[0].entry_id)

    def test_different_parent_run_id_yields_different_entry_id(self):
        first = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={"climate_envelope_tail": {"cells": ["c1"]}},
        )
        second = build_cockpit_warning_manifest(
            parent_run_id="r2",
            cell_summary={"climate_envelope_tail": {"cells": ["c1"]}},
        )
        self.assertNotEqual(first[0].entry_id, second[0].entry_id)


class TestManifestOrdering(unittest.TestCase):
    """Sorted output for byte-stable hash."""

    def test_sorted_by_bucket_then_category(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={
                "soil_no_hwsd_coverage": {"cells": ["c1"]},  # B3
                "climate_envelope_tail": {"cells": ["c2"]},  # B2
            },
            stage_1_verdicts={
                "entries": [
                    {"category": "crop_region_mismatch", "zone": "BSh"},  # B5
                ],
            },
        )
        # B2 < B3 < B5
        buckets = [e.bucket for e in manifest]
        self.assertEqual(buckets, sorted(buckets))


class TestComputeManifestHash(unittest.TestCase):
    """Byte-stable JSON roundtrip — drift detection."""

    def test_hash_is_64_char_sha256_hex(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={"climate_envelope_tail": {"cells": ["c1"]}},
        )
        h = compute_manifest_hash(manifest)
        self.assertEqual(len(h), 64)
        int(h, 16)

    def test_hash_stable_across_calls(self):
        m1 = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={"climate_envelope_tail": {"cells": ["c1", "c2"]}},
        )
        m2 = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={"climate_envelope_tail": {"cells": ["c2", "c1"]}},
        )
        self.assertEqual(
            compute_manifest_hash(m1),
            compute_manifest_hash(m2),
        )

    def test_hash_sensitive_to_bucket_change(self):
        m1 = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={"climate_envelope_tail": {"cells": ["c1"]}},
        )
        m2 = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={"soil_no_hwsd_coverage": {"cells": ["c1"]}},
        )
        self.assertNotEqual(
            compute_manifest_hash(m1),
            compute_manifest_hash(m2),
        )

    def test_hash_roundtrips_through_json_serialization(self):
        # Persisting through PipelineRun.cell_validation_status
        # JSONField goes through json.dumps + json.loads; the
        # hash computed on the deserialized list must match.
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary={"climate_envelope_tail": {"cells": ["c1", "c2"]}},
        )
        original_hash = compute_manifest_hash(manifest)
        roundtripped = [
            CockpitManifestEntry(
                entry_id=d["entry_id"],
                bucket=d["bucket"],
                category=d["category"],
                affected_cells=tuple(d["affected_cells"]),
                affected_zones=tuple(d["affected_zones"]),
                count=d["count"],
            )
            for d in json.loads(
                json.dumps([e.to_dict() for e in manifest])
            )
        ]
        self.assertEqual(
            compute_manifest_hash(roundtripped),
            original_hash,
        )


class TestEmptyManifestPaths(unittest.TestCase):
    """Cockpit empty-state — no flagged warnings."""

    def test_no_inputs_yields_empty_manifest(self):
        self.assertEqual(
            build_cockpit_warning_manifest(parent_run_id="r1"),
            [],
        )

    def test_empty_inputs_yield_empty_manifest(self):
        self.assertEqual(
            build_cockpit_warning_manifest(
                parent_run_id="r1",
                cell_summary={},
                stage_1_verdicts={"entries": []},
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
