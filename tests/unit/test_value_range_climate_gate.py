"""V2-22b/P.2 AC-B6b — post-merge gate on `value_range_climate` info.

Tests `_gate_value_range_climate_delegation`, the helper that runs
at the post-translate merge site in the executor. It escalates the
delegating info record to a warning when no delegated-to records
(`post_translate_range_sarra_py_<var>`) landed — which happens when
translation failed, the platform validator skipped, or rasterio
errored.

Direct helper tests here rather than a full executor integration
test — the helper is mutation-in-place on a plain dict, so the
three decision paths (happy / empty / partial) each map to one
small fixture.
"""

from __future__ import annotations

import unittest

from prismpy.pipeline.executor import (
    _gate_value_range_climate_delegation,
)


def _info_record() -> dict:
    """The exact shape `_check_value_ranges` emits for SARRA-Py
    file-based climate, AFTER `_restructure_to_categories` has
    enriched it. The enrichment adds `passed` (True for pass/info),
    `category`, and `unit` fields. Kept as a fixture so a copy-edit
    in scientific.py that changes the shape forces this test to
    update alongside — catching drift between emitter and gate."""
    return {
        "check": "value_range_climate",
        "scope": "per_record",
        "result": "info",
        "passed": True,  # set by _restructure_to_categories; escalation must flip this
        "category": "range_checks",
        "summary": (
            "Climate value ranges for SARRA-Py are computed "
            "from a stratified sample of 10 output files per "
            "variable — first, last, and eight evenly-spaced "
            "interior files. When available, the per-variable "
            "ranges appear below."
        ),
        "manuscript_claim": (
            "Section 2.5: value range verification (delegated)"
        ),
        "details": {
            "data_format": "geotiff_per_day",
            "delegated_to": "post_translate._validate_sarra_py_geotiffs",
            "sample_policy": (
                "stratified first/stride/last, 10 files per variable"
            ),
            "coverage_kind": "delegated",
        },
    }


def _delegated_record(var: str) -> dict:
    """Minimal stand-in for a `post_translate_range_sarra_py_<var>`
    record — only the `check` field matters for the gate's
    `startswith` predicate."""
    return {
        "check": f"post_translate_range_sarra_py_{var}",
        "result": "pass",
    }


class TestGateHappyPath(unittest.TestCase):
    """Post-translate ran and produced per-variable records → the
    info stays info. No side effects, no summary rewrite."""

    def test_all_four_variables_landed_keeps_info(self):
        sci = {"checks": [_info_record()] + [
            _delegated_record(v) for v in ('rain', 'tmax', 'tmin', 'srad')
        ]}
        before = dict(sci["checks"][0])
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        # Info record identity preserved byte-for-byte.
        self.assertEqual(sci["checks"][0], before)
        self.assertEqual(sci["checks"][0]["result"], "info")

    def test_single_variable_landed_still_keeps_info(self):
        """AC-B6b partial-path — even ONE delegated record proves
        the platform validator ran. Don't penalize a partial run
        by escalating as if nothing happened."""
        sci = {"checks": [_info_record(), _delegated_record('rain')]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        self.assertEqual(sci["checks"][0]["result"], "info")


class TestGateEmptyDelegatePath(unittest.TestCase):
    """Post-translate didn't run OR errored before producing any
    per-variable records → the info MUST escalate to a warning
    with the honest "did not run" summary."""

    def test_no_delegated_records_escalates_to_warning(self):
        sci = {"checks": [_info_record()]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        record = sci["checks"][0]
        self.assertEqual(record["result"], "warning")
        self.assertIn("did not run", record["summary"])
        self.assertIn("no per-variable records", record["summary"])

    def test_escalation_preserves_check_identity(self):
        """The check-record key (`"check": "value_range_climate"`)
        must survive the escalation so cross-run diffing tools
        still match the same record. The in-place mutation touches
        only `result`, `passed`, and `summary` — identity, scope,
        category, and details remain exactly as emitted."""
        sci = {"checks": [_info_record()]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        record = sci["checks"][0]
        self.assertEqual(record["check"], "value_range_climate")
        self.assertEqual(record["scope"], "per_record")
        self.assertEqual(record["category"], "range_checks")
        self.assertEqual(
            record["details"]["coverage_kind"], "delegated",
        )

    def test_escalation_flips_passed_to_false(self):
        """Codex self-check R2 MEDIUM — on escalation, `result` goes
        from "info" to "warning" AND `passed` goes from True to
        False in the same mutation. If they drift, downstream
        consumers keying off `passed` still render the skipped
        check as successful. The two fields must agree per the
        `result in ("pass", "info")` invariant set at emission
        time in `_restructure_to_categories`."""
        record = _info_record()
        self.assertTrue(record["passed"])
        self.assertEqual(record["result"], "info")
        sci = {"checks": [record]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        self.assertEqual(sci["checks"][0]["result"], "warning")
        self.assertFalse(sci["checks"][0]["passed"])

    def test_result_passed_invariant_holds_across_all_checks(self):
        """Broader regression guard — after gating, no check in
        `sci["checks"]` may carry `result == "warning" or "fail"`
        together with `passed == True`. Asserts the invariant
        codex R2 identified as missing from downstream consumer
        contracts."""
        record = _info_record()
        sibling_pass = {
            "check": "temporal_completeness", "result": "pass",
            "passed": True,
        }
        sibling_fail = {
            "check": "unit_consistency", "result": "fail",
            "passed": False,
        }
        sci = {"checks": [sibling_pass, record, sibling_fail]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        for c in sci["checks"]:
            if c.get("result") in ("warning", "fail"):
                self.assertFalse(
                    c.get("passed"),
                    msg=(
                        f'check {c["check"]!r} has result='
                        f'{c["result"]!r} but passed=True'
                    ),
                )

    def test_unrelated_checks_untouched(self):
        """Only the delegating info record is mutated. Sibling
        checks (structural, cross-variable, etc.) must be left
        exactly as they were."""
        unrelated = {
            "check": "unit_consistency",
            "result": "pass",
            "summary": "Units internally consistent.",
        }
        sci = {"checks": [unrelated, _info_record()]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        self.assertEqual(sci["checks"][0], unrelated)


class TestGateDiscriminatorGuard(unittest.TestCase):
    """The gate only acts on records tagged with
    `details.coverage_kind == "delegated"`. Any other `info` record
    that happens to share the `value_range_climate` check name but
    is NOT the delegation pattern must be left alone — protects
    the gate from firing on a future record reuse that isn't a
    delegation."""

    def test_non_delegated_coverage_kind_not_touched(self):
        r = _info_record()
        r["details"]["coverage_kind"] = "computed"
        sci = {"checks": [r]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        self.assertEqual(sci["checks"][0]["result"], "info")

    def test_missing_coverage_kind_not_touched(self):
        r = _info_record()
        r["details"].pop("coverage_kind", None)
        sci = {"checks": [r]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        self.assertEqual(sci["checks"][0]["result"], "info")


class TestGateEmptyInput(unittest.TestCase):
    """Defensive — an `sci` dict without a `checks` key, or with an
    empty list, must not raise. Exercise path for validators that
    short-circuit before producing any records."""

    def test_missing_checks_key_is_noop(self):
        sci = {}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        self.assertEqual(sci, {})

    def test_empty_checks_list_is_noop(self):
        sci = {"checks": []}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        self.assertEqual(sci, {"checks": []})


class TestGatePostTranslateRaisePath(unittest.TestCase):
    """Codex self-check R2 HIGH — when post-translate validation
    raises before it can merge its records, the executor formerly
    skipped the gate (it lived inside the post-translate
    try/except). The V2-22b/P.2 refactor moved the gate OUTSIDE
    the try, so the escalation fires on the exception path too —
    which is exactly the path the check is meant to expose.

    This test simulates that path directly: `validation_summary`
    carries a scientific report WITHOUT any merged post-translate
    records, as happens when `run_post_translate_validation()`
    raises at import time, hits a platform-specific validator
    crash, or encounters a missing output dir before any record
    is produced."""

    def test_exception_path_still_escalates_info(self):
        info = _info_record()
        # Pre-merge state: scientific checks exist, no
        # post_translate_range_sarra_py_* records have been merged
        # in — the exact state after post-translate raises.
        sci = {
            "checks": [
                {
                    "check": "temporal_completeness",
                    "result": "pass",
                    "passed": True,
                },
                info,
            ],
            "n_checks": 2,
            "n_pass": 2,
            "n_warning": 0,
            "n_fail": 0,
            "overall_result": "pass",
            "passed": True,
        }
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        # The info record is now a warning with `passed=False`.
        escalated = sci["checks"][1]
        self.assertEqual(escalated["result"], "warning")
        self.assertFalse(escalated["passed"])
        self.assertIn("did not run", escalated["summary"])

    def test_exception_path_preserves_other_checks(self):
        """Sibling checks — including unrelated passing ones — are
        untouched by the exception-path gating. Only the
        specifically-identified info record is escalated."""
        info = _info_record()
        pass_check = {
            "check": "temporal_completeness",
            "result": "pass",
            "passed": True,
            "summary": "All daily records present.",
        }
        sci = {"checks": [pass_check, info]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        self.assertEqual(sci["checks"][0], pass_check)


class TestGatePlatformAwareness(unittest.TestCase):
    """Codex self-check R3 MEDIUM — the scientific validator emits
    the delegation info record based on climate-dict shape
    (`rainfall_dir` / `agera5_dir`), not on whether SARRA-Py is
    actually in the pipeline's enabled platforms. A run that's
    ACEA-only but has those climate paths populated would emit
    the info; without the platform guard, the gate would then
    manufacture a bogus SARRA-Py warning. `sarra_py_enabled`
    short-circuits the escalation so the warning only fires on
    runs that legitimately needed SARRA-Py range records."""

    def test_sarra_py_disabled_keeps_info_even_with_no_delegated(self):
        """ACEA-only run path — climate dict has
        `rainfall_dir`/`agera5_dir` so scientific.py emits the
        delegation info, but SARRA-Py isn't enabled so no
        `post_translate_range_sarra_py_*` records ever land. The
        gate must NOT escalate: that would be a false-positive
        warning."""
        sci = {"checks": [_info_record()]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=False)
        self.assertEqual(sci["checks"][0]["result"], "info")
        self.assertTrue(sci["checks"][0]["passed"])

    def test_sarra_py_disabled_preserves_existing_warnings(self):
        """Defensive — if for some reason the info is already a
        warning (not possible in real flow, but defend against
        misuse), the disabled-platform short-circuit still keeps
        it untouched."""
        r = _info_record()
        r["result"] = "warning"
        r["passed"] = False
        sci = {"checks": [r]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=False)
        # Stays whatever it was. The gate's disabled-case early
        # return guarantees it doesn't re-write fields.
        self.assertEqual(sci["checks"][0]["result"], "warning")


class TestGateIdempotence(unittest.TestCase):
    """Codex self-check R2 HIGH implication — the gate is called
    unconditionally at the post-merge surfacing point, so it may
    run twice in some code paths (once inside the successful
    merge branch, once in the outer call). Running twice must be
    a no-op on the second call."""

    def test_second_call_does_not_re_escalate(self):
        sci = {"checks": [_info_record()]}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        state_after_first = {k: dict(v) if isinstance(v, dict) else v
                             for k, v in sci["checks"][0].items()}
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        state_after_second = {k: dict(v) if isinstance(v, dict) else v
                              for k, v in sci["checks"][0].items()}
        self.assertEqual(state_after_first, state_after_second)

    def test_second_call_when_already_warning_and_delegated_is_noop(self):
        """Variant — if the record is already a warning AND
        delegated records exist (the pattern B.6 explicitly does
        NOT escalate), a second call stays a no-op. The gate reads
        `details.coverage_kind == "delegated"` AND checks for
        absence of delegated records, so this combination simply
        falls through."""
        info = _info_record()
        info["result"] = "warning"  # simulate some prior escalation
        info["passed"] = False
        sci = {
            "checks": [
                info,
                {"check": "post_translate_range_sarra_py_rain", "result": "pass"},
            ],
        }
        _gate_value_range_climate_delegation(sci, sarra_py_enabled=True)
        # Unchanged — presence of the delegated record blocks
        # escalation at the `not delegated` predicate.
        self.assertEqual(sci["checks"][0]["result"], "warning")
        self.assertFalse(sci["checks"][0]["passed"])
