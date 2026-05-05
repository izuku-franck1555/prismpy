"""Sprint E.1 AC-E1-4 — pin the cockpit-readiness schema
reference declared in
:mod:`prismpy.koppen.cockpit_readiness_contract`.

The reference is the canonical authority for the Stage 1
verdict snapshot shape. Producer (prismweb wizard write-path)
and consumer (cockpit drawer Bucket 5 panel + run-lineage
rail + cross-repo consumer test) both read from this module
rather than embedding the field sets inline.
"""
from __future__ import annotations

import unittest

from prismpy.koppen.cockpit_readiness_contract import (
    ENTRY_FIELDS,
    REQUIRED_ENTRY_FIELDS,
    SCHEMA_VERSION_CURRENT,
    SUBSTRATE_VERSION_FIELDS,
    TOP_LEVEL_FIELDS,
    UNAVAILABLE_TOP_LEVEL_FIELDS,
)


class TestCockpitReadinessContractTopLevel(unittest.TestCase):
    """Pin the canonical top-level fields."""

    def test_schema_version_current_is_v1(self):
        self.assertEqual(SCHEMA_VERSION_CURRENT, "stage_1_v1")

    def test_top_level_fields_match_sprint_f_reference(self):
        # Sprint F's _STAGE_1_SNAPSHOT_FIXTURE pins these 7 keys
        # at tests/unit/test_stage_1_provenance.py:32+. The
        # contract reference must mirror that fixture exactly.
        self.assertEqual(
            TOP_LEVEL_FIELDS,
            frozenset({
                "schema_version",
                "cache_key",
                "region_cache_key",
                "crop",
                "created_at",
                "substrate_versions",
                "entries",
            }),
        )

    def test_unavailable_extension_is_2_keys(self):
        self.assertEqual(
            UNAVAILABLE_TOP_LEVEL_FIELDS,
            frozenset({"stage_1_unavailable", "unavailable_reason"}),
        )

    def test_unavailable_extension_disjoint_from_top_level(self):
        # The unavailable shape ADDS keys; top-level keys
        # remain. A stricter overlap check guards a future
        # refactor that might collapse the extension into a
        # top-level field.
        self.assertEqual(
            TOP_LEVEL_FIELDS & UNAVAILABLE_TOP_LEVEL_FIELDS,
            frozenset(),
        )


class TestCockpitReadinessContractEntries(unittest.TestCase):
    """Pin the per-entry fields."""

    def test_entry_fields_carry_sprint_f_base_set(self):
        # Sprint F base set — 10 keys per the
        # _STAGE_1_SNAPSHOT_FIXTURE entry shape.
        sprint_f_base = {
            "category", "zone", "zone_label", "crop", "verdict",
            "reason", "n_cell_days_in_zone", "stage_1_verdict_id",
            "override_decision_id", "override_status",
        }
        self.assertTrue(sprint_f_base.issubset(ENTRY_FIELDS))

    def test_entry_fields_carry_bucket_5_extension(self):
        # Path β step 4 (1668763) added explanation +
        # verbatim_source_url for the wizard banner +
        # cockpit drawer plain-language UX.
        self.assertIn("explanation", ENTRY_FIELDS)
        self.assertIn("verbatim_source_url", ENTRY_FIELDS)

    def test_required_entry_fields_subset_of_full_set(self):
        # Required ⊂ full. The remaining fields are optional
        # (e.g., stage_1_verdict_id is None until a future
        # verdict-table persistence layer ships).
        self.assertTrue(REQUIRED_ENTRY_FIELDS.issubset(ENTRY_FIELDS))

    def test_required_entry_fields_carry_persona_facing_fields(self):
        # The cockpit drawer Bucket 5 panel needs every field
        # the persona reads off (zone label + plain-language
        # explanation + per-crop ECOCROP URL). Missing any
        # would break the AC-E1-6 panel render.
        for required in (
            "zone_label", "explanation",
            "verbatim_source_url", "category",
        ):
            self.assertIn(
                required, REQUIRED_ENTRY_FIELDS,
                f"{required!r} is persona-facing; required-set "
                f"must include it.",
            )


class TestSubstrateVersionFields(unittest.TestCase):
    """Pin the substrate_versions block."""

    def test_substrate_version_block_carries_4_pins(self):
        self.assertEqual(
            SUBSTRATE_VERSION_FIELDS,
            frozenset({
                "bounds_version",
                "zone_classifier_version",
                "ecocrop_envelope_version",
                "zone_aggregates_version",
            }),
        )


if __name__ == "__main__":
    unittest.main()
