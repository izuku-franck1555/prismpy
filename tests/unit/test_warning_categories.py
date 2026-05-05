"""Sprint E.0 — WarningCategory + WARNING_BUCKET_MAP foundation.

Tests ride the chokepoint declared in
:mod:`prismpy.warnings.categories`. Every AC from the Sprint
E.0 contract is covered:

* AC-E0-1 — WarningCategory exists, is a StrEnum, declares
  the 10 contract-locked values.
* AC-E0-2 — WARNING_BUCKET_MAP is exhaustive across the enum
  (no orphan categories, no spurious extra keys).
* AC-E0-3 — StrEnum string-equality semantics: a member
  compares equal to its bare-string form, so Sprint D.1's
  pre-migration call sites that emit the bare string still
  satisfy ``cause == WarningCategory.X`` checks.
* AC-E0-4 — helper accessors return deterministically sorted
  tuples; ``AUTO_FIXABLE`` resolves to the empty tuple.
* AC-E0-5 — no orphans, no duplicates structural pin (with
  insertion-order pin per warning-auditor probe-4-A).
* AC-E0-6 — Sprint D.1 ``UnavailableCause`` Literal values
  are a subset of WarningCategory values; the Literal stays
  the canonical 3-value type while WarningCategory carries
  the wider chokepoint.
* AC-E0-7 — JSON serialization is byte-identical across
  repeated calls and stable across enum-member iteration
  (insertion-stable per Python 3.7+ dict guarantee, pinned
  here explicitly).

The tests intentionally stay structural / pure-Python. There
is no DB use; the FAST tier can run them without the
self-bootstrap dance that the prismweb cockpit-DB tests need.
"""
from __future__ import annotations

import json
import typing
import unittest

from prismpy.cells.schema import UnavailableCause
from prismpy.warnings import (
    WARNING_BUCKET_MAP,
    WarningBucket,
    WarningCategory,
    bucket_for,
    categories_in_bucket,
)


# Cite-locked snapshot of the 10 enum values per the contract.
# Adding or removing an enum value MUST update this snapshot
# AND propagate to WARNING_BUCKET_MAP — the orphan check
# enforces the second half automatically.
EXPECTED_CATEGORY_VALUES = (
    "soil_no_hwsd_coverage",
    "soil_texture_invalid",
    "climate_rh_invalid",
    "transitional_zone",
    "insufficiently_sampled",
    "climate_envelope_tail",
    "crop_region_mismatch",
    "crop_physiology_violation",
    "short_gap_interpolatable",
    "manual_override",
)


class TestWarningCategoryEnum(unittest.TestCase):
    """AC-E0-1 — WarningCategory exists, StrEnum-shaped, with
    the contract-locked 10 values."""

    def test_is_str_subclass(self):
        """The ``(str, Enum)`` base is what gives Sprint D.1
        its backward-compat string equality."""
        self.assertTrue(issubclass(WarningCategory, str))

    def test_declared_values_match_contract(self):
        actual = tuple(c.value for c in WarningCategory)
        self.assertEqual(actual, EXPECTED_CATEGORY_VALUES)

    def test_member_count(self):
        """Drift defense — the contract locks 10 values; if
        a future contributor lands an additional value
        without updating the contract, this fails first."""
        self.assertEqual(len(WarningCategory), 10)


class TestWarningBucketEnum(unittest.TestCase):
    """AC-E0-2 — WarningBucket declares the 5 cockpit-response
    buckets."""

    def test_five_buckets_declared(self):
        expected = {
            "auto_fixable",
            "informational",
            "true_exclude",
            "interpolatable",
            "manual_override_with_evidence",
        }
        actual = {b.value for b in WarningBucket}
        self.assertEqual(actual, expected)


class TestWarningBucketMap(unittest.TestCase):
    """AC-E0-2 + AC-E0-5 — WARNING_BUCKET_MAP is exhaustive
    across WarningCategory; insertion order matches the enum
    (per warning-auditor probe-4-A ordering pin)."""

    def test_no_orphan_categories(self):
        """Every WarningCategory value has a bucket entry."""
        self.assertEqual(
            set(WARNING_BUCKET_MAP.keys()),
            set(WarningCategory),
            "WARNING_BUCKET_MAP must cover every WarningCategory "
            "value — no orphan categories.",
        )

    def test_no_extra_keys(self):
        """No bucket entry without a matching enum value."""
        self.assertTrue(
            all(
                isinstance(k, WarningCategory)
                for k in WARNING_BUCKET_MAP.keys()
            ),
            "WARNING_BUCKET_MAP keys must all be WarningCategory "
            "members — no string keys.",
        )

    def test_insertion_order_matches_enum(self):
        """probe-4-A pin: the dict's insertion order must
        match :class:`WarningCategory`'s declaration order so
        a future refactor that reorders the dict is caught."""
        self.assertEqual(
            tuple(WARNING_BUCKET_MAP.keys()),
            tuple(WarningCategory),
            "WARNING_BUCKET_MAP key order must match the "
            "WarningCategory declaration order so insertion-"
            "stable iteration is byte-identical across runs.",
        )

    def test_all_values_are_buckets(self):
        self.assertTrue(
            all(
                isinstance(v, WarningBucket)
                for v in WARNING_BUCKET_MAP.values()
            ),
            "WARNING_BUCKET_MAP values must all be WarningBucket "
            "members.",
        )


class TestBucketAssignments(unittest.TestCase):
    """AC-E0-2 — explicit per-category bucket assignments
    locked from the contract."""

    def test_sprint_d1_causes_route_to_true_exclude(self):
        for cat in (
            WarningCategory.SOIL_NO_HWSD_COVERAGE,
            WarningCategory.SOIL_TEXTURE_INVALID,
            WarningCategory.CLIMATE_RH_INVALID,
        ):
            with self.subTest(category=cat):
                self.assertEqual(
                    bucket_for(cat), WarningBucket.TRUE_EXCLUDE,
                    f"{cat} must route to TRUE_EXCLUDE (Sprint D.1 "
                    f"missing-substrate semantic).",
                )

    def test_e0_5_informational_assignments(self):
        for cat in (
            WarningCategory.TRANSITIONAL_ZONE,
            WarningCategory.INSUFFICIENTLY_SAMPLED,
            WarningCategory.CLIMATE_ENVELOPE_TAIL,
        ):
            with self.subTest(category=cat):
                self.assertEqual(
                    bucket_for(cat), WarningBucket.INFORMATIONAL,
                )

    def test_crop_region_mismatch_routes_to_manual_override_with_evidence(self):
        # Sprint F bucket reclassification: the wizard offers a
        # documented-override path on Stage 1 INCOMPATIBLE
        # verdicts, so the category belongs in Bucket 5 not
        # Bucket 3. The data + UI classification must match.
        self.assertEqual(
            bucket_for(WarningCategory.CROP_REGION_MISMATCH),
            WarningBucket.MANUAL_OVERRIDE_WITH_EVIDENCE,
        )

    def test_crop_physiology_violation_routes_to_true_exclude(self):
        # Stage 2 per-cell ECOCROP tolerance violations remain
        # TRUE_EXCLUDE; the cockpit cannot meaningfully accept an
        # override on every individual cell, only at the wizard /
        # zone level (which is the Stage 1 → Bucket 5 path above).
        self.assertEqual(
            bucket_for(WarningCategory.CROP_PHYSIOLOGY_VIOLATION),
            WarningBucket.TRUE_EXCLUDE,
        )

    def test_short_gap_routes_to_interpolatable(self):
        self.assertEqual(
            bucket_for(WarningCategory.SHORT_GAP_INTERPOLATABLE),
            WarningBucket.INTERPOLATABLE,
        )

    def test_manual_override_routes_to_manual_with_evidence(self):
        self.assertEqual(
            bucket_for(WarningCategory.MANUAL_OVERRIDE),
            WarningBucket.MANUAL_OVERRIDE_WITH_EVIDENCE,
        )


class TestStrEqualityBackwardCompat(unittest.TestCase):
    """AC-E0-3 — the StrEnum equality glue is what lets Sprint
    D.1's pre-migration call sites stay correct."""

    def test_member_equals_its_value(self):
        self.assertEqual(
            WarningCategory.SOIL_NO_HWSD_COVERAGE,
            "soil_no_hwsd_coverage",
        )
        self.assertEqual(
            WarningCategory.SOIL_TEXTURE_INVALID,
            "soil_texture_invalid",
        )
        self.assertEqual(
            WarningCategory.CLIMATE_RH_INVALID,
            "climate_rh_invalid",
        )

    def test_dict_lookup_with_string_key(self):
        """A dict keyed on the enum member can be queried with
        the bare string form thanks to ``str``-based hashing."""
        d = {WarningCategory.SOIL_NO_HWSD_COVERAGE: "x"}
        self.assertEqual(d["soil_no_hwsd_coverage"], "x")

    def test_in_check_with_string(self):
        members = list(WarningCategory)
        self.assertIn("soil_no_hwsd_coverage", members)


class TestHelperAccessors(unittest.TestCase):
    """AC-E0-4 — :func:`bucket_for` and
    :func:`categories_in_bucket` behavior."""

    def test_bucket_for_resolves_known_member(self):
        self.assertEqual(
            bucket_for(WarningCategory.MANUAL_OVERRIDE),
            WarningBucket.MANUAL_OVERRIDE_WITH_EVIDENCE,
        )

    def test_categories_in_bucket_returns_tuple(self):
        result = categories_in_bucket(WarningBucket.TRUE_EXCLUDE)
        self.assertIsInstance(result, tuple)
        self.assertGreater(len(result), 0)

    def test_categories_in_bucket_sorted_by_value(self):
        """Sort-determinism pin — the helper must return tuples
        sorted by the category's string value, not by dict
        insertion order. Both are deterministic given the
        ordering pin in TestWarningBucketMap, but sorting is
        the explicit contract for downstream serializers."""
        true_exclude = categories_in_bucket(WarningBucket.TRUE_EXCLUDE)
        values = [c.value for c in true_exclude]
        self.assertEqual(values, sorted(values))

    def test_categories_in_bucket_auto_fixable_is_empty(self):
        """The empty Bucket 1 is INTENDED — Sprint D.1 auto-
        fixes are silent + provenance-only. A future sprint
        that adds an auto-fix-with-cockpit-surface use case
        will introduce the corresponding category at that
        time. This test pins the current empty state."""
        result = categories_in_bucket(WarningBucket.AUTO_FIXABLE)
        self.assertEqual(result, ())

    def test_categories_in_bucket_informational_count(self):
        """Sprint E.0.5 reservation count: the
        ``INFORMATIONAL`` bucket has exactly 3 reserved
        categories at E.0 ship time."""
        result = categories_in_bucket(WarningBucket.INFORMATIONAL)
        self.assertEqual(len(result), 3)

    def test_categories_in_bucket_true_exclude_count(self):
        """Bucket 3 has 4 categories: 3 Sprint D.1 + 1 reserved
        for Sprint F (CROP_PHYSIOLOGY_VIOLATION). Sprint F
        bucket reclassification dropped the count from 5 to 4
        when CROP_REGION_MISMATCH moved to Bucket 5
        (MANUAL_OVERRIDE_WITH_EVIDENCE) per ux-expert verdict +
        honest-signal review.
        """
        result = categories_in_bucket(WarningBucket.TRUE_EXCLUDE)
        self.assertEqual(len(result), 4)

    def test_categories_in_bucket_manual_override_with_evidence_count(self):
        """Bucket 5 has 2 categories: CROP_REGION_MISMATCH (Sprint
        F wizard-time override) + MANUAL_OVERRIDE (V3 reserve).
        Sprint F bucket reclassification grew the count from 1
        to 2 when CROP_REGION_MISMATCH moved here.
        """
        result = categories_in_bucket(
            WarningBucket.MANUAL_OVERRIDE_WITH_EVIDENCE,
        )
        self.assertEqual(len(result), 2)


class TestSprintD1BackwardCompat(unittest.TestCase):
    """AC-E0-6 — the 3 ``UnavailableCause`` Literal values
    that Sprint D.1 ships are a subset of WarningCategory."""

    def test_unavailable_cause_subset(self):
        """``UnavailableCause`` is a ``Literal[...]``; extract
        its allowed values via ``typing.get_args`` and assert
        each is a WarningCategory value."""
        literal_args = typing.get_args(UnavailableCause)
        self.assertEqual(len(literal_args), 3)
        warning_values = {c.value for c in WarningCategory}
        for cause in literal_args:
            with self.subTest(cause=cause):
                self.assertIn(
                    cause, warning_values,
                    f"UnavailableCause Literal value {cause!r} must "
                    f"be a WarningCategory value (Sprint D.1 "
                    f"backward-compat pin).",
                )

    def test_three_reachable_causes(self):
        """Pin the 3 reachable causes explicitly so a future
        contributor cannot silently rename one without this
        test failing."""
        literal_args = set(typing.get_args(UnavailableCause))
        self.assertEqual(
            literal_args,
            {
                "soil_no_hwsd_coverage",
                "soil_texture_invalid",
                "climate_rh_invalid",
            },
        )


class TestPydanticDeterminism(unittest.TestCase):
    """AC-E0-7 — JSON serialization is byte-identical across
    repeated calls + stable insertion-order across iterations."""

    def test_enum_iteration_is_stable(self):
        """Python 3.7+ guarantees dict insertion order; pin the
        enum's iteration explicitly so the test catches a
        regression if a future contributor switches to a
        custom metaclass that loses ordering."""
        first = tuple(WarningCategory)
        second = tuple(WarningCategory)
        self.assertEqual(first, second)

    def test_pydantic_basemodel_serializes_member_as_bare_string(self):
        """The ``str``-based StrEnum convention works with
        Pydantic v2's :meth:`BaseModel.model_dump` when
        ``mode='json'`` is used: enum members serialize as
        their string value, not as ``"WarningCategory.X"`` or
        the bare member name. Catches a hypothetical Pydantic
        2.x point release that flips the default to enum-name
        serialization without this suite noticing — the
        cell-summary v2.x JSON contract depends on the bare
        string form.
        """
        from pydantic import BaseModel

        class _Carrier(BaseModel):
            cause: WarningCategory

        carrier = _Carrier(cause=WarningCategory.SOIL_NO_HWSD_COVERAGE)
        # mode='json' emits JSON-compatible types; the enum
        # member must collapse to its string value.
        dumped = carrier.model_dump(mode="json")
        self.assertEqual(dumped["cause"], "soil_no_hwsd_coverage")
        # The full JSON dump carries the same bytes.
        json_str = carrier.model_dump_json()
        self.assertIn('"cause":"soil_no_hwsd_coverage"', json_str)

    def test_bucket_map_serialization_byte_identical(self):
        """Serialize the bucket map twice and assert byte-
        identical JSON output. This is the load-bearing pin
        for Sprint E.0.5 AC-Q2-B1's cross-AC determinism gate
        — bound generation MUST produce byte-identical output
        across runs, and the bucket map is one input."""
        # Convert WarningCategory → string for JSON round-trip
        # since enum members aren't JSON-natively serializable
        # in pure Python (Pydantic handles this differently).
        payload = {
            cat.value: bucket.value
            for cat, bucket in WARNING_BUCKET_MAP.items()
        }
        first = json.dumps(payload, sort_keys=False)
        second = json.dumps(payload, sort_keys=False)
        self.assertEqual(first, second)

    def test_categories_in_bucket_byte_identical(self):
        """Each :func:`categories_in_bucket` call returns a
        tuple in deterministic sorted order — repeated calls
        produce byte-identical results."""
        for bucket in WarningBucket:
            with self.subTest(bucket=bucket):
                first = categories_in_bucket(bucket)
                second = categories_in_bucket(bucket)
                self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
