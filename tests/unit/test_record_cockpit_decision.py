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
            category="soil_no_hwsd_coverage",
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
                category="soil_no_hwsd_coverage",
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

    def test_cockpit_path_hardcodes_false_discriminator(self):
        # Codex review MEDIUM 2 absorption — the cockpit-time
        # discriminator is hardcoded ``False``; callers cannot
        # override it. The wizard-time path stamps ``True`` so
        # audit-grep against ``override_at_pre_pipeline=True``
        # surfaces only the wizard entries (and ``=False``
        # surfaces only the cockpit entries).
        self.tracker.record_cockpit_decision(
            **self._valid_override_kwargs(),
        )
        decisions = [d for d, _ in self.tracker._pending_decisions]
        self.assertIn(
            "override_at_pre_pipeline=False",
            decisions[0].rationale,
        )
        self.assertNotIn(
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

    def test_unknown_category_raises(self):
        # Codex MEDIUM 1 absorption — category MUST resolve
        # to a WarningCategory enum value. A typo'd caller
        # ("crop_region_mismatch_typo") fails loud here
        # rather than landing as a generic decision.
        with self.assertRaises(ValueError) as cm:
            self.tracker.record_cockpit_decision(
                decision_type=DecisionType.USER_ACKNOWLEDGE,
                category="not_a_real_category",
                bucket=2,
                affected_cells=["c1"],
            )
        self.assertIn(
            "WarningCategory", str(cm.exception),
        )

    def test_bucket_must_match_category_canonical_bucket(self):
        # Codex MEDIUM 1 absorption — even when both bucket
        # and category are valid, they must agree per
        # WARNING_BUCKET_MAP. A caller passing
        # category=climate_envelope_tail (canonical bucket 2)
        # with bucket=3 is mis-classifying the warning; the
        # cockpit's bucket-affordance invariant breaks if this
        # is allowed.
        with self.assertRaises(ValueError) as cm:
            self.tracker.record_cockpit_decision(
                decision_type=DecisionType.USER_SKIP,
                category="climate_envelope_tail",
                bucket=3,  # canonical is 2
                affected_cells=["c1"],
            )
        self.assertIn(
            "does not match the canonical bucket",
            str(cm.exception),
        )

    def test_override_routes_through_wizard_override_record(self):
        # Codex HIGH 2 absorption — cockpit-time overrides
        # MUST inherit the same Pydantic invariants the
        # wizard-time path enforces. A non-hex verdict_hash
        # would slip through the previous implementation;
        # WizardOverrideRecord rejects it.
        tracker = ProvenanceTracker(project_name="t")
        with self.assertRaises(ValueError) as cm:
            tracker.record_cockpit_decision(
                decision_type=DecisionType.USER_OVERRIDE,
                category="crop_region_mismatch",
                bucket=5,
                affected_zones=["BSh"],
                rationale=(
                    "Local trial with cultivar ITA-150 yielded "
                    "4.2 t/ha under irrigated 600 mm/yr regime."
                ),
                evidence_type="local_trial",
                verdict_hash="not-a-valid-hex-hash",
            )
        self.assertIn(
            "WizardOverrideRecord",
            str(cm.exception),
        )

    def test_override_rejects_filler_rationale(self):
        # Codex HIGH 2 absorption — single-char-repeat
        # rationale that clears the 50-char floor must reject
        # via the WizardOverrideRecord filler-detector.
        tracker = ProvenanceTracker(project_name="t")
        with self.assertRaises(ValueError) as cm:
            tracker.record_cockpit_decision(
                decision_type=DecisionType.USER_OVERRIDE,
                category="crop_region_mismatch",
                bucket=5,
                affected_zones=["BSh"],
                rationale="a" * 60,  # filler — single-char repeat
                evidence_type="local_trial",
                verdict_hash="0" * 64,
            )
        # WizardOverrideRecord's filler validator surfaces
        # via the wrapper's HIGH 2 routing.
        self.assertIn(
            "WizardOverrideRecord",
            str(cm.exception),
        )

    def test_override_rejects_http_evidence_url(self):
        # Codex HIGH 2 absorption — http:// evidence URL must
        # reject via WizardOverrideRecord's https-only
        # validator.
        tracker = ProvenanceTracker(project_name="t")
        with self.assertRaises(ValueError):
            tracker.record_cockpit_decision(
                decision_type=DecisionType.USER_OVERRIDE,
                category="crop_region_mismatch",
                bucket=5,
                affected_zones=["BSh"],
                rationale=(
                    "Local trial with cultivar ITA-150 yielded "
                    "4.2 t/ha under irrigated 600 mm/yr regime."
                ),
                evidence_type="local_trial",
                verdict_hash="0" * 64,
                evidence_url="http://insecure.invalid/paper",
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
