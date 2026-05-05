"""Sprint E.1 AC-E1-0 — pin the ``record_cockpit_decision``
wrapper on :class:`ProvenanceTracker`.

The helper mirrors :meth:`record_wizard_decision` (Sprint F
AC-F-6) for the three cockpit-time decision-types:

* :data:`DecisionType.USER_ACKNOWLEDGE` — Bucket 2 INFO ack.
* :data:`DecisionType.USER_SKIP` — Bucket 3 EXCLUDE skip.
* :data:`DecisionType.USER_OVERRIDE` — Bucket 5 cockpit-time
  override (distinct from wizard-time via the
  ``override_at_pre_pipeline=False`` discriminator stamped
  into the rationale).

Every supported decision-type goes through one validated
entry-point so the cockpit caller in prismweb can stay narrow
(no ad-hoc :meth:`record_decision` payloads). Failure modes
are explicit ``ValueError`` so a typo'd caller fails loud.
"""
from __future__ import annotations

import unittest

from prismpy.models.provenance import DecisionType
from prismpy.provenance import ProvenanceTracker


class TestRecordCockpitDecisionAcknowledge(unittest.TestCase):
    """Pin the Bucket 2 acknowledge path."""

    def setUp(self):
        self.tracker = ProvenanceTracker(project_name="t")

    def test_acknowledge_writes_user_acknowledge_decision(self):
        self.tracker.record_cockpit_decision(
            decision_type=DecisionType.USER_ACKNOWLEDGE,
            category="climate_envelope_tail",
            bucket=2,
            affected_cells=["c1", "c2"],
        )
        decisions = [d for d, _ in self.tracker._pending_decisions]
        self.assertEqual(len(decisions), 1)
        self.assertEqual(
            decisions[0].decision_type,
            DecisionType.USER_ACKNOWLEDGE,
        )
        self.assertIn("bucket 2", decisions[0].description)
        self.assertIn("climate_envelope_tail", decisions[0].description)

    def test_acknowledge_requires_affected_cells(self):
        with self.assertRaises(ValueError) as cm:
            self.tracker.record_cockpit_decision(
                decision_type=DecisionType.USER_ACKNOWLEDGE,
                category="climate_envelope_tail",
                bucket=2,
                affected_cells=[],
            )
        self.assertIn("affected_cells", str(cm.exception))

    def test_acknowledge_severity_is_info(self):
        self.tracker.record_cockpit_decision(
            decision_type=DecisionType.USER_ACKNOWLEDGE,
            category="climate_envelope_tail",
            bucket=2,
            affected_cells=["c1"],
        )
        # Acknowledge is a non-blocking affirmation; severity
        # should not amplify to "warning" (which is reserved
        # for override). The ``info`` level keeps the lineage
        # rail's audit-grep clean.
        decisions = [d for d, _ in self.tracker._pending_decisions]
        self.assertEqual(decisions[0].severity, "info")


class TestRecordCockpitDecisionSkip(unittest.TestCase):
    """Pin the Bucket 3 skip path."""

    def setUp(self):
        self.tracker = ProvenanceTracker(project_name="t")

    def test_skip_writes_user_skip_decision(self):
        self.tracker.record_cockpit_decision(
            decision_type=DecisionType.USER_SKIP,
            category="coverage_climate_cells",
            bucket=3,
            affected_cells=["c1", "c2", "c3"],
        )
        decisions = [d for d, _ in self.tracker._pending_decisions]
        self.assertEqual(len(decisions), 1)
        self.assertEqual(
            decisions[0].decision_type, DecisionType.USER_SKIP,
        )
        self.assertIn("affected_cells=3", decisions[0].rationale)

    def test_skip_requires_affected_cells(self):
        with self.assertRaises(ValueError):
            self.tracker.record_cockpit_decision(
                decision_type=DecisionType.USER_SKIP,
                category="coverage_climate_cells",
                bucket=3,
                affected_cells=None,
            )


class TestRecordCockpitDecisionOverride(unittest.TestCase):
    """Pin the Bucket 5 cockpit-time override path. Mirrors
    :class:`WizardOverrideRecord` parity per CC-33."""

    def setUp(self):
        self.tracker = ProvenanceTracker(project_name="t")

    def _valid_override_kwargs(self, **overrides):
        kwargs = {
            "decision_type": DecisionType.USER_OVERRIDE,
            "category": "crop_region_mismatch",
            "bucket": 5,
            "affected_zones": ["BSh"],
            "rationale": (
                "Local trial with cultivar ITA-150 yielded "
                "4.2 t/ha under irrigated 600 mm/yr regime."
            ),
            "evidence_type": "local_trial",
            "verdict_hash": "0" * 64,
        }
        kwargs.update(overrides)
        return kwargs

    def test_override_writes_user_override_decision(self):
        self.tracker.record_cockpit_decision(
            **self._valid_override_kwargs(),
        )
        decisions = [d for d, _ in self.tracker._pending_decisions]
        self.assertEqual(len(decisions), 1)
        self.assertEqual(
            decisions[0].decision_type,
            DecisionType.USER_OVERRIDE,
        )
        self.assertEqual(decisions[0].severity, "warning")
        # Rationale carries the cockpit-time discriminator
        # explicitly so audit-grep can split wizard-time vs
        # cockpit-time entries cleanly.
        self.assertIn(
            "override_at_pre_pipeline=False",
            decisions[0].rationale,
        )

    def test_override_requires_affected_zones(self):
        with self.assertRaises(ValueError) as cm:
            self.tracker.record_cockpit_decision(
                **self._valid_override_kwargs(affected_zones=[]),
            )
        self.assertIn("affected_zones", str(cm.exception))

    def test_override_rationale_floor_is_50_chars(self):
        # Wizard-parity per CC-33: the cockpit-time rationale
        # must clear the same ≥50-char floor the wizard
        # banner enforces.
        with self.assertRaises(ValueError) as cm:
            self.tracker.record_cockpit_decision(
                **self._valid_override_kwargs(rationale="too short"),
            )
        self.assertIn("rationale", str(cm.exception).lower())

    def test_override_requires_evidence_type(self):
        with self.assertRaises(ValueError) as cm:
            self.tracker.record_cockpit_decision(
                **self._valid_override_kwargs(evidence_type=None),
            )
        self.assertIn("evidence_type", str(cm.exception))

    def test_override_requires_verdict_hash(self):
        with self.assertRaises(ValueError) as cm:
            self.tracker.record_cockpit_decision(
                **self._valid_override_kwargs(verdict_hash=None),
            )
        self.assertIn("verdict_hash", str(cm.exception))

    def test_override_other_evidence_type_requires_specify(self):
        # Sprint E.1 codex BLOCKER 6 — when the cockpit caller
        # picks ``evidence_type='other'``, the companion
        # specify field must accompany.
        with self.assertRaises(ValueError) as cm:
            self.tracker.record_cockpit_decision(
                **self._valid_override_kwargs(
                    evidence_type="other",
                    evidence_type_other_specify=None,
                ),
            )
        self.assertIn(
            "evidence_type_other_specify",
            str(cm.exception),
        )

    def test_override_named_bucket_rejects_specify(self):
        with self.assertRaises(ValueError):
            self.tracker.record_cockpit_decision(
                **self._valid_override_kwargs(
                    evidence_type="local_trial",
                    evidence_type_other_specify="should be None",
                ),
            )

    def test_override_other_with_specify_is_accepted(self):
        self.tracker.record_cockpit_decision(
            **self._valid_override_kwargs(
                evidence_type="other",
                evidence_type_other_specify=(
                    "Cultural-knowledge basis from local agronomist."
                ),
            ),
        )
        decisions = [d for d, _ in self.tracker._pending_decisions]
        self.assertEqual(len(decisions), 1)
        self.assertIn(
            "evidence_type=other", decisions[0].rationale,
        )
        self.assertIn(
            "Cultural-knowledge", decisions[0].rationale,
        )

    def test_override_at_pre_pipeline_true_stamps_distinct_marker(self):
        # The discriminator stamps verbatim into the rationale.
        self.tracker.record_cockpit_decision(
            **self._valid_override_kwargs(
                override_at_pre_pipeline=True,
            ),
        )
        decisions = [d for d, _ in self.tracker._pending_decisions]
        self.assertIn(
            "override_at_pre_pipeline=True",
            decisions[0].rationale,
        )


class TestRecordCockpitDecisionGuards(unittest.TestCase):
    """Pin the type / bucket discriminator guards."""

    def setUp(self):
        self.tracker = ProvenanceTracker(project_name="t")

    def test_unsupported_decision_type_raises(self):
        with self.assertRaises(ValueError) as cm:
            self.tracker.record_cockpit_decision(
                decision_type=DecisionType.SOURCE_SELECTION,
                category="x",
                bucket=2,
                affected_cells=["c1"],
            )
        self.assertIn(
            "USER_ACKNOWLEDGE",
            str(cm.exception),
        )

    def test_invalid_bucket_raises(self):
        for bad_bucket in (0, 1, 4, 6, -1, 99):
            with self.assertRaises(ValueError, msg=f"bucket={bad_bucket}"):
                self.tracker.record_cockpit_decision(
                    decision_type=DecisionType.USER_ACKNOWLEDGE,
                    category="x",
                    bucket=bad_bucket,
                    affected_cells=["c1"],
                )

    def test_disabled_tracker_silently_returns(self):
        disabled = ProvenanceTracker(project_name="t", enabled=False)
        # No decisions written; no error raised.
        disabled.record_cockpit_decision(
            decision_type=DecisionType.USER_ACKNOWLEDGE,
            category="x",
            bucket=2,
            affected_cells=["c1"],
        )
        self.assertEqual(disabled._pending_decisions, [])


if __name__ == "__main__":
    unittest.main()
