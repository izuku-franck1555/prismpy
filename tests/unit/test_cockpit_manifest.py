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


class TestProducerShapeCellSummary(unittest.TestCase):
    """Codex HIGH 1 absorption — accept the actual producer
    shape from :func:`prismpy.pipeline.executor._build_cell_summary`:
    ``{"cells": [{cell_id, failed_checks: [{check_id,
    category, ...}, ...]}, ...]}``."""

    def test_producer_shape_pivots_per_category(self):
        cell_summary = {
            "cells": [
                {
                    "cell_id": "c1",
                    "failed_checks": [
                        {
                            "check_id": "climate_envelope_tail",
                            "category": "climate_envelope_tail",
                        },
                    ],
                },
                {
                    "cell_id": "c2",
                    "failed_checks": [
                        {
                            "check_id": "climate_envelope_tail",
                            "category": "climate_envelope_tail",
                        },
                        {
                            "check_id": "soil_no_hwsd_coverage",
                            "category": "soil_no_hwsd_coverage",
                        },
                    ],
                },
            ],
        }
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary=cell_summary,
        )
        # Two manifest rows — Bucket 2 (climate_envelope_tail
        # with c1+c2) + Bucket 3 (soil_no_hwsd_coverage with c2).
        by_category = {e.category: e for e in manifest}
        self.assertEqual(set(by_category), {
            "climate_envelope_tail",
            "soil_no_hwsd_coverage",
        })
        self.assertEqual(
            by_category["climate_envelope_tail"].affected_cells,
            ("c1", "c2"),
        )
        self.assertEqual(
            by_category["soil_no_hwsd_coverage"].affected_cells,
            ("c2",),
        )
        # Buckets correctly resolved per WARNING_BUCKET_MAP.
        self.assertEqual(
            by_category["climate_envelope_tail"].bucket, 2,
        )
        self.assertEqual(
            by_category["soil_no_hwsd_coverage"].bucket, 3,
        )

    def test_producer_shape_dedupes_cells_within_category(self):
        # If a cell carries duplicate failed_checks for the
        # same category (e.g., per-variable emit collapse), the
        # manifest dedupes the cell_id.
        cell_summary = {
            "cells": [
                {
                    "cell_id": "c1",
                    "failed_checks": [
                        {"category": "climate_envelope_tail"},
                        {"category": "climate_envelope_tail"},
                    ],
                },
            ],
        }
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary=cell_summary,
        )
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0].affected_cells, ("c1",))

    def test_producer_shape_empty_failed_checks_skipped(self):
        cell_summary = {
            "cells": [
                {"cell_id": "c1", "failed_checks": []},
                {"cell_id": "c2"},  # no failed_checks key
            ],
        }
        self.assertEqual(
            build_cockpit_warning_manifest(
                parent_run_id="r1",
                cell_summary=cell_summary,
            ),
            [],
        )

    def test_producer_shape_missing_category_skipped(self):
        # A failed_check without category is malformed; the
        # builder skips it rather than failing loud — the
        # check_id-only paths are documented as legacy and
        # shouldn't break the manifest builder.
        cell_summary = {
            "cells": [
                {
                    "cell_id": "c1",
                    "failed_checks": [
                        {"check_id": "climate_envelope_tail"},
                    ],
                },
            ],
        }
        # No category in the failed_check means we can't bucket
        # it; the builder skips. Empty manifest is the
        # documented behavior.
        self.assertEqual(
            build_cockpit_warning_manifest(
                parent_run_id="r1",
                cell_summary=cell_summary,
            ),
            [],
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


class TestDimensionToggleVocabulary(unittest.TestCase):
    """The pipeline's per-cell pivot at
    ``executor.py:_category_for_check_id`` populates
    ``cell_summary.cells[i].failed_checks[j].category`` with
    the six dimension-toggle category strings declared in
    ``executor.py::_CATEGORY_FROM_PREFIX``. Those strings are
    NOT members of :class:`prismpy.warnings.WarningCategory`;
    the cockpit's ``_category_to_bucket_integer`` lookup must
    recognize them via :data:`_DIMENSION_BUCKET_MAP` (and
    route to bucket 3 TRUE_EXCLUDE) instead of raising
    :class:`UnknownCategoryError`. Pre-fix, every per-cell
    warning on a fresh project tripped the unknown-category
    path; the user saw the pre-E.0 fallback banner on a
    valid project.
    """

    def _producer_shape(self, category: str, cells: list[str]) -> dict:
        """Build a ``cell_summary`` payload that mirrors the
        executor's per-cell pivot output: each cell carries a
        ``failed_checks`` list, each check carries the
        dimension-toggle ``category`` string."""
        return {
            "cells": [
                {
                    "cell_id": cid,
                    "failed_checks": [
                        {
                            "check_id": f"{category}_check",
                            "category": category,
                        },
                    ],
                }
                for cid in cells
            ],
        }

    def test_value_range_routes_to_bucket_3(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary=self._producer_shape(
                "value_range", ["c1", "c2"],
            ),
        )
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0].bucket, 3)
        self.assertEqual(manifest[0].category, "value_range")
        self.assertEqual(
            manifest[0].affected_cells, ("c1", "c2"),
        )

    def test_cross_variable_routes_to_bucket_3(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary=self._producer_shape(
                "cross_variable", ["c1"],
            ),
        )
        self.assertEqual(manifest[0].bucket, 3)
        self.assertEqual(manifest[0].category, "cross_variable")

    def test_temporal_routes_to_bucket_3(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary=self._producer_shape(
                "temporal", ["c1"],
            ),
        )
        self.assertEqual(manifest[0].bucket, 3)
        self.assertEqual(manifest[0].category, "temporal")

    def test_soil_completeness_routes_to_bucket_3(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary=self._producer_shape(
                "soil_completeness", ["c1"],
            ),
        )
        self.assertEqual(manifest[0].bucket, 3)
        self.assertEqual(
            manifest[0].category, "soil_completeness",
        )

    def test_region_specific_bounds_routes_to_bucket_3(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary=self._producer_shape(
                "region_specific_bounds", ["c1"],
            ),
        )
        self.assertEqual(manifest[0].bucket, 3)
        self.assertEqual(
            manifest[0].category, "region_specific_bounds",
        )

    def test_coverage_per_cell_routes_to_bucket_3(self):
        manifest = build_cockpit_warning_manifest(
            parent_run_id="r1",
            cell_summary=self._producer_shape(
                "coverage_per_cell", ["c1"],
            ),
        )
        self.assertEqual(manifest[0].bucket, 3)
        self.assertEqual(
            manifest[0].category, "coverage_per_cell",
        )

    def test_unknown_category_diagnostic_names_both_vocabularies(self):
        """Negative case: a category string that is neither a
        :class:`WarningCategory` member NOR a
        :data:`_DIMENSION_BUCKET_MAP` key MUST still raise
        :class:`UnknownCategoryError`. The diagnostic must
        name BOTH vocabularies so a substrate-drift fix
        knows which side to update."""
        with self.assertRaises(UnknownCategoryError) as cm:
            build_cockpit_warning_manifest(
                parent_run_id="r1",
                cell_summary=self._producer_shape(
                    "snowpack_completeness_BOGUS",
                    ["c1"],
                ),
            )
        diag = str(cm.exception)
        self.assertIn("snowpack_completeness_BOGUS", diag)
        # Diagnostic must reference both vocabularies so
        # the contributor knows the surfaces to reconcile.
        self.assertIn("WarningCategory", diag)
        self.assertIn("dimension-toggle", diag)


if __name__ == "__main__":
    unittest.main()
