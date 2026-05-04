"""Sprint E.0.5 AC-Q2-E — per-zone sample-quality threshold + assessment.

Pins the :data:`MIN_CELL_DAYS_PER_ZONE` constant at exactly
1,000,000 cell-days (AC-Q2-E-a) plus the Pydantic schema for
the :class:`ZoneSampleQuality` provenance record (AC-Q2-E-b)
plus the assessment function semantics.

Anti-mutation drills:

- Revert the threshold to 10,000 (a "literature floor" but
  much less conservative) → ``test_threshold_pinned_at_1m``
  fails.
- Revert from cell-days to plain cell count → the Pydantic
  field name changes from ``n_cell_days`` to ``n_cells``-only
  and ``test_record_has_both_n_cells_and_n_cell_days`` fails.
- Drop a per-zone provenance field (e.g.
  ``sample_quality_reason``) → ``test_required_fields`` fails.
- Allow incoherent verdicts (sample_quality='sufficient' with
  n_cell_days < threshold) → the model_validator-pinned
  consistency check breaks; a unit test catches it.
- Re-introduce the bound-gen-time check at provenance-write
  only (not at bound-gen call site) → an integration test in
  a future commit catches insufficient zones being silently
  written into the bounds file.
"""
from __future__ import annotations

import json
import unittest

from prismpy.bounds import (
    MIN_CELL_DAYS_PER_ZONE,
    SampleQuality,
    ZoneSampleQuality,
    assess_zone_sample_quality,
)


class TestThresholdConstant(unittest.TestCase):
    """The :data:`MIN_CELL_DAYS_PER_ZONE` constant pin per
    AC-Q2-E-a anti-mutation drill."""

    def test_threshold_pinned_at_1m(self):
        self.assertEqual(MIN_CELL_DAYS_PER_ZONE, 1_000_000)

    def test_threshold_is_int(self):
        self.assertIsInstance(MIN_CELL_DAYS_PER_ZONE, int)


class TestSampleQualityEnum(unittest.TestCase):
    """The two-state verdict enum."""

    def test_two_members(self):
        self.assertEqual(len(list(SampleQuality)), 2)

    def test_values_are_canonical_strings(self):
        self.assertEqual(SampleQuality.SUFFICIENT.value, "sufficient")
        self.assertEqual(
            SampleQuality.INSUFFICIENT.value, "insufficient",
        )


class TestZoneSampleQualityFields(unittest.TestCase):
    """Pin the Pydantic schema for the per-zone provenance
    record (AC-Q2-E-b: 5 required fields)."""

    _REQUIRED_FIELDS = (
        "n_cells",
        "n_cell_days",
        "sample_quality",
        "sample_quality_reason",
        "threshold",
    )

    def test_required_fields(self):
        model_fields = set(ZoneSampleQuality.model_fields.keys())
        for field in self._REQUIRED_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, model_fields)

    def test_no_unexpected_fields(self):
        model_fields = set(ZoneSampleQuality.model_fields.keys())
        unexpected = model_fields - set(self._REQUIRED_FIELDS)
        self.assertEqual(unexpected, set())

    def test_extra_fields_forbidden(self):
        # ConfigDict(extra='forbid'): typo'd / extra field fails.
        with self.assertRaises(ValueError):
            ZoneSampleQuality(
                n_cells=10,
                n_cell_days=1_000_000,
                sample_quality=SampleQuality.SUFFICIENT,
                sample_quality_reason="ok",
                threshold=1_000_000,
                extra_field="should_not_pass",
            )

    def test_negative_n_cells_rejected(self):
        with self.assertRaises(ValueError):
            ZoneSampleQuality(
                n_cells=-1,
                n_cell_days=10,
                sample_quality=SampleQuality.INSUFFICIENT,
                sample_quality_reason="bad",
                threshold=1_000_000,
            )

    def test_negative_n_cell_days_rejected(self):
        with self.assertRaises(ValueError):
            ZoneSampleQuality(
                n_cells=10,
                n_cell_days=-1,
                sample_quality=SampleQuality.INSUFFICIENT,
                sample_quality_reason="bad",
                threshold=1_000_000,
            )

    def test_empty_reason_rejected(self):
        with self.assertRaises(ValueError):
            ZoneSampleQuality(
                n_cells=10,
                n_cell_days=10,
                sample_quality=SampleQuality.INSUFFICIENT,
                sample_quality_reason="",
                threshold=1_000_000,
            )

    def test_record_has_both_n_cells_and_n_cell_days(self):
        # AC-Q2-E decoupled cell count from temporal window;
        # both fields ship in the record (for human audit), but
        # the threshold is applied to n_cell_days alone.
        model_fields = set(ZoneSampleQuality.model_fields.keys())
        self.assertIn("n_cells", model_fields)
        self.assertIn("n_cell_days", model_fields)


class TestZoneSampleQualityImmutability(unittest.TestCase):
    """Per codex Gate-A HIGH on commit 7b: the record is
    frozen after construction so a downstream caller cannot
    mutate ``sample_quality`` or ``n_cell_days`` and serialize
    an incoherent JSON that bypasses the verdict-vs-count
    consistency model_validator."""

    def test_sample_quality_is_immutable(self):
        record = assess_zone_sample_quality(
            n_cells=10, n_cell_days=500_000,
        )
        # Initial verdict: insufficient.
        self.assertEqual(
            record.sample_quality, SampleQuality.INSUFFICIENT,
        )
        # Attempting to mutate must raise.
        with self.assertRaises(Exception):  # ValidationError or AttributeError
            record.sample_quality = SampleQuality.SUFFICIENT

    def test_n_cell_days_is_immutable(self):
        record = assess_zone_sample_quality(
            n_cells=10, n_cell_days=500_000,
        )
        with self.assertRaises(Exception):
            record.n_cell_days = 2_000_000

    def test_threshold_is_immutable(self):
        record = assess_zone_sample_quality(
            n_cells=10, n_cell_days=500_000,
        )
        with self.assertRaises(Exception):
            record.threshold = 1


class TestZoneSampleQualityVerdictConsistency(unittest.TestCase):
    """The model validator pins ``sample_quality`` to the
    threshold check on ``n_cell_days``. An incoherent record
    fails fail-loud."""

    def test_sufficient_below_threshold_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ZoneSampleQuality(
                n_cells=10,
                n_cell_days=500_000,  # below threshold
                sample_quality=SampleQuality.SUFFICIENT,  # incoherent
                sample_quality_reason="bad",
                threshold=1_000_000,
            )
        self.assertIn("inconsistent", str(ctx.exception))

    def test_insufficient_above_threshold_rejected(self):
        with self.assertRaises(ValueError):
            ZoneSampleQuality(
                n_cells=10,
                n_cell_days=2_000_000,  # above threshold
                sample_quality=SampleQuality.INSUFFICIENT,  # incoherent
                sample_quality_reason="bad",
                threshold=1_000_000,
            )

    def test_at_threshold_is_sufficient(self):
        # n_cell_days == threshold counts as sufficient (>= predicate).
        record = ZoneSampleQuality(
            n_cells=10,
            n_cell_days=1_000_000,  # at threshold
            sample_quality=SampleQuality.SUFFICIENT,
            sample_quality_reason="at threshold",
            threshold=1_000_000,
        )
        self.assertEqual(record.sample_quality, SampleQuality.SUFFICIENT)


class TestAssessZoneSampleQuality(unittest.TestCase):
    """The :func:`assess_zone_sample_quality` helper builds a
    valid record for the caller's cell counts."""

    def test_above_threshold_returns_sufficient(self):
        record = assess_zone_sample_quality(
            n_cells=100, n_cell_days=2_000_000,
        )
        self.assertEqual(record.sample_quality, SampleQuality.SUFFICIENT)
        self.assertEqual(record.threshold, MIN_CELL_DAYS_PER_ZONE)
        self.assertEqual(record.n_cells, 100)
        self.assertEqual(record.n_cell_days, 2_000_000)

    def test_below_threshold_returns_insufficient(self):
        record = assess_zone_sample_quality(
            n_cells=10, n_cell_days=500_000,
        )
        self.assertEqual(
            record.sample_quality, SampleQuality.INSUFFICIENT,
        )
        # Reason text references the threshold + the delta
        # (so the cockpit can render the gap to the operator).
        self.assertIn("1,000,000", record.sample_quality_reason)
        self.assertIn("INSUFFICIENTLY_SAMPLED", record.sample_quality_reason)

    def test_zero_cell_days_returns_insufficient(self):
        record = assess_zone_sample_quality(n_cells=0, n_cell_days=0)
        self.assertEqual(
            record.sample_quality, SampleQuality.INSUFFICIENT,
        )

    def test_at_threshold_returns_sufficient(self):
        record = assess_zone_sample_quality(
            n_cells=10, n_cell_days=1_000_000,
        )
        self.assertEqual(record.sample_quality, SampleQuality.SUFFICIENT)

    def test_threshold_override_for_tests(self):
        # Caller can override the threshold for test scenarios
        # or future ratchet evaluation; the record carries the
        # override so the verdict stays auditable.
        record = assess_zone_sample_quality(
            n_cells=10, n_cell_days=10_000, threshold=5_000,
        )
        self.assertEqual(record.threshold, 5_000)
        self.assertEqual(record.sample_quality, SampleQuality.SUFFICIENT)

    def test_negative_inputs_rejected(self):
        with self.assertRaises(ValueError):
            assess_zone_sample_quality(n_cells=-1, n_cell_days=10)

    def test_round_trip_via_json(self):
        record = assess_zone_sample_quality(
            n_cells=100, n_cell_days=2_000_000,
        )
        text = record.model_dump_json()
        data = json.loads(text)
        for field in ("n_cells", "n_cell_days", "sample_quality",
                      "sample_quality_reason", "threshold"):
            self.assertIn(field, data)
        # Round-trip via model_validate_json reconstructs equal record.
        rehydrated = ZoneSampleQuality.model_validate_json(text)
        self.assertEqual(rehydrated, record)


if __name__ == "__main__":
    unittest.main()
