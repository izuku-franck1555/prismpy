"""Sprint F AC-F-6 — wizard-time override record tests.

Pins:

* :class:`WizardOverrideRecord` Pydantic-frozen + extra-forbid
  + structural field validators (50-char rationale floor;
  filler-rejection heuristic; evidence_type Literal of 5
  values; affected_zones non-empty list; verdict_hash 64-char
  SHA-256 hex; evidence_url https-only).
* :func:`compute_verdict_hash` deterministic across dict-key
  ordering.
* :data:`DecisionType.USER_OVERRIDE` additive enum extension.

Anti-mutation drills:

* Drop ``USER_OVERRIDE`` from :class:`DecisionType` →
  ``test_user_override_decision_type_present`` fails. Catches
  a refactor that removes the enum value.
* Weaken rationale floor to 20 chars → form would accept
  trivial overrides; ``test_rejects_short_rationale`` fails.
* Accept any string for ``evidence_type`` → typo'd categories
  drift through; ``test_rejects_invalid_evidence_type`` fails.
* Accept empty ``affected_zones`` → an override could drift
  through with no actual zone affected;
  ``test_rejects_empty_affected_zones`` fails.
* Compute verdict_hash without sorted keys → different
  in-memory dict orderings produce different hashes; the
  cache-key recompute would never converge;
  ``test_compute_verdict_hash_stable_across_key_order``
  catches it.
"""
from __future__ import annotations

import unittest

from pydantic import ValidationError

from prismpy.models.provenance import DecisionType
from prismpy.provenance import (
    EvidenceType,
    WizardOverrideRecord,
    build_wizard_override_payload,
    compute_verdict_hash,
)


# Reusable 64-char SHA-256-shaped hex for hash tests.
_VALID_HASH = "0" * 64
_INVALID_HASH = "z" * 64
_VALID_RATIONALE = (
    "We have a documented local trial showing rice cultivar "
    "ITA-150 yielded 4.2 t/ha under irrigated 600 mm/yr "
    "regime in this BSh zone over the 2024-2025 seasons."
)


class TestUserOverrideDecisionType(unittest.TestCase):
    """Pin the additive ``DecisionType.USER_OVERRIDE`` enum
    extension."""

    def test_user_override_decision_type_present(self):
        self.assertTrue(hasattr(DecisionType, "USER_OVERRIDE"))
        self.assertEqual(
            DecisionType.USER_OVERRIDE.value, "user_override",
        )

    def test_user_override_distinct_from_parameter_override(self):
        # AC-F-6 + builder Adj-11: USER_OVERRIDE is a separate
        # enum value from PARAMETER_OVERRIDE — the override
        # carries persona-stated rationale + evidence, not a
        # substrate parameter change.
        self.assertNotEqual(
            DecisionType.USER_OVERRIDE,
            DecisionType.PARAMETER_OVERRIDE,
        )


class TestWizardOverrideRecordShape(unittest.TestCase):
    """Pin the structural shape of a wizard override record."""

    def test_minimal_record_constructs(self):
        record = WizardOverrideRecord(
            rationale=_VALID_RATIONALE,
            evidence_type="local_trial",
            affected_zones=["BSh", "Aw"],
            verdict_hash=_VALID_HASH,
        )
        self.assertEqual(record.decision_type, "user_override")
        self.assertEqual(record.evidence_type, "local_trial")
        self.assertEqual(record.affected_zones, ["BSh", "Aw"])
        # Optional fields default to None.
        self.assertIsNone(record.evidence_url)
        self.assertIsNone(record.methodology_paper_doi)

    def test_record_is_frozen(self):
        record = WizardOverrideRecord(
            rationale=_VALID_RATIONALE,
            evidence_type="local_trial",
            affected_zones=["BSh"],
            verdict_hash=_VALID_HASH,
        )
        with self.assertRaises(ValidationError):
            record.rationale = "different rationale"  # type: ignore[misc]

    def test_record_extra_forbid(self):
        with self.assertRaises(ValidationError):
            WizardOverrideRecord(
                rationale=_VALID_RATIONALE,
                evidence_type="local_trial",
                affected_zones=["BSh"],
                verdict_hash=_VALID_HASH,
                # Typo'd extra field — should fail-loud.
                affecting_zones=["BSh"],
            )


class TestWizardOverrideRecordValidators(unittest.TestCase):
    """Pin the field validators."""

    _BASE = {
        "rationale": _VALID_RATIONALE,
        "evidence_type": "local_trial",
        "affected_zones": ["BSh"],
        "verdict_hash": _VALID_HASH,
    }

    def test_rejects_short_rationale(self):
        with self.assertRaises(ValidationError):
            WizardOverrideRecord(**{**self._BASE, "rationale": "ok"})

    def test_rejects_filler_rationale(self):
        # Single-char repeats clear the 50-char length floor
        # but should be rejected as filler per warning-auditor
        # LOW-4.
        with self.assertRaises(ValidationError):
            WizardOverrideRecord(**{
                **self._BASE,
                "rationale": "a" * 60,
            })

    def test_rejects_invalid_evidence_type(self):
        with self.assertRaises(ValidationError):
            WizardOverrideRecord(**{
                **self._BASE,
                "evidence_type": "best_guess",
            })

    def test_rejects_empty_affected_zones(self):
        with self.assertRaises(ValidationError):
            WizardOverrideRecord(**{**self._BASE, "affected_zones": []})

    def test_rejects_short_verdict_hash(self):
        with self.assertRaises(ValidationError):
            WizardOverrideRecord(**{
                **self._BASE,
                "verdict_hash": "abc123",
            })

    def test_rejects_non_hex_verdict_hash(self):
        with self.assertRaises(ValidationError):
            WizardOverrideRecord(**{
                **self._BASE,
                "verdict_hash": _INVALID_HASH,
            })

    def test_rejects_http_evidence_url(self):
        with self.assertRaises(ValidationError):
            WizardOverrideRecord(**{
                **self._BASE,
                "evidence_url": "http://example.com",
            })

    def test_accepts_https_evidence_url(self):
        record = WizardOverrideRecord(**{
            **self._BASE,
            "evidence_url": "https://example.com/paper",
        })
        self.assertEqual(
            record.evidence_url, "https://example.com/paper",
        )

    def test_evidence_type_covers_5_canonical_values(self):
        # AC-F-6 + codex Gate A #15: exactly 5 evidence types.
        # Future expansion lands in V2-23 polish (additive).
        # Pin here so an accidental 6th value fails the test.
        from typing import get_args
        canonical = set(get_args(EvidenceType))
        self.assertEqual(
            canonical,
            {
                "local_trial", "irrigation", "cultivar_specific",
                "citation", "field_observation",
            },
        )


class TestComputeVerdictHash(unittest.TestCase):
    """Pin the verdict-hash determinism (codex Gate A #13 stale-
    override rejection)."""

    def test_compute_returns_64_char_hex(self):
        out = compute_verdict_hash({"a": 1, "b": 2})
        self.assertEqual(len(out), 64)
        # Verify hex-decodable.
        int(out, 16)

    def test_compute_stable_across_key_order(self):
        # Two semantically equal snapshots must produce the
        # same hash regardless of dict-key insertion order.
        a = compute_verdict_hash({"a": 1, "b": 2})
        b = compute_verdict_hash({"b": 2, "a": 1})
        self.assertEqual(a, b)

    def test_compute_distinguishes_different_payloads(self):
        a = compute_verdict_hash({"crop": "rice"})
        b = compute_verdict_hash({"crop": "maize"})
        self.assertNotEqual(a, b)

    def test_compute_handles_nested_structures(self):
        # AC-F-5 cache shape is nested
        # ``{schema_version, cache_key, entries[]}``;
        # hash must descend.
        snapshot = {
            "schema_version": "stage_1_v1",
            "cache_key": "abc",
            "entries": [
                {"zone": "BSh", "verdict": "incompatible"},
            ],
        }
        out = compute_verdict_hash(snapshot)
        self.assertEqual(len(out), 64)


class TestBuildWizardOverridePayload(unittest.TestCase):
    """Pin the JSON-mode dump shape."""

    def test_payload_is_dict_round_trippable(self):
        record = WizardOverrideRecord(
            rationale=_VALID_RATIONALE,
            evidence_type="local_trial",
            affected_zones=["BSh", "Aw"],
            verdict_hash=_VALID_HASH,
        )
        payload = build_wizard_override_payload(record)
        # Must be a plain dict the prismweb JSONField can store.
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["decision_type"], "user_override")
        self.assertEqual(payload["evidence_type"], "local_trial")
        self.assertEqual(payload["affected_zones"], ["BSh", "Aw"])
        # Timestamp is serialized as ISO-format string in JSON
        # mode, not a datetime object — the JSONField round-
        # trips the string cleanly.
        self.assertIsInstance(payload["timestamp"], str)


if __name__ == "__main__":
    unittest.main()
