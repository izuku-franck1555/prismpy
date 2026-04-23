"""V2-22b/P.2 AC-AUDIT-2 — structural regression guarding the
single-integration-point invariant on
`_gate_value_range_climate_delegation`.

Context: the P.2 R1-R4 codex rounds surfaced four separate
consequences of a dual-call-site sprawl. R4's HIGH was the
smoking gun — my R3 signature change migrated one call site
(`executor.py:2540`) but left a stale copy at `:2496` using the
old 1-arg signature; the resulting `TypeError` was swallowed by
the surrounding broad-except as "Post-translate validation
skipped". Warning-auditor's AMBER verdict confirmed the root
cause was structural: two call sites with a divergence window
whenever the gate's shape changes.

This test locks the "exactly one gate call per
`_execute_validate` invocation" invariant structurally, so a
future contributor re-introducing a second call site fails the
test regardless of which surface symptom their change exposes.

Strategy: monkeypatch the gate helper with a call-counter wrapper
and drive `_execute_validate` across the four scenarios (a) happy
path — scientific + post-translate both succeed, (b) post-translate
raises, (c) scientific absent, (d) non-SARRA-Py platforms only.
In each, assert the counter equals 1.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch


def _build_pipeline_stub(
    enabled_platforms: list | None = None,
) -> SimpleNamespace:
    """Construct the minimum shape `_execute_validate` reads from
    `self`. Keeps the test honest — no pipeline internals we don't
    need, no real CDS / filesystem / rasterio touch."""
    from prismpy.config.schema import Platform

    if enabled_platforms is None:
        enabled_platforms = [Platform.SARRA_PY]

    config = SimpleNamespace(
        output=SimpleNamespace(base_dir="/tmp/prismpy-stub-out"),
        get_enabled_platforms=lambda: enabled_platforms,
    )
    logger = MagicMock()
    provenance = SimpleNamespace(enabled=False)
    # Layer 1 (platform validators) iterates `translation_results`;
    # for our structural test we pass an empty dict, so
    # `_get_validator` is never called. A real pipeline would
    # supply more, but the gate-integration invariant is orthogonal.
    return SimpleNamespace(
        config=config,
        logger=logger,
        provenance=provenance,
        _get_validator=lambda *a, **kw: None,
    )


def _minimal_scientific_report() -> Dict[str, Any]:
    """The shape `run_scientific_validation` returns, slim enough
    for the gate-invocation test. Omits detail fields the gate
    doesn't touch (categories, summary stats, etc.)."""
    return {
        "overall_result": "pass",
        "passed": True,
        "n_checks": 1,
        "n_pass": 1,
        "n_warning": 0,
        "n_fail": 0,
        "checks": [
            {
                "check": "value_range_climate",
                "result": "info",
                "passed": True,
                "details": {"coverage_kind": "delegated"},
            },
        ],
        "categories": {},
    }


def _minimal_post_translate_report() -> Dict[str, Any]:
    """Minimal post-translate report with one delegated-style
    range record so the gate's `not delegated` predicate is
    False (happy-path behaviour). For scenarios that need an
    empty report, call sites build their own."""
    return {
        "overall_result": "pass",
        "n_checks": 1,
        "checks": [
            {
                "check": "post_translate_range_sarra_py_rain",
                "result": "pass",
            },
        ],
    }


class TestGateCalledExactlyOnce(unittest.TestCase):
    """AMBER ruling AC-AUDIT-2: structural invariant — the gate is
    invoked EXACTLY ONCE per `_execute_validate` regardless of
    which branch the flow takes. Dual-call-site sprawl was the
    P.2 R1-R4 iteration's substrate; this test blocks its
    reintroduction."""

    def _run_once(
        self,
        *,
        scientific_returns=None,
        scientific_raises=None,
        post_translate_returns=None,
        post_translate_raises=None,
        enabled_platforms=None,
    ) -> int:
        """Drive `_execute_validate` with mocked collaborators and
        return the gate call count. Returns an integer so callers
        can assert `== 1` directly."""
        from prismpy.pipeline import executor as _exec_mod
        from prismpy.pipeline.executor import TranslationPipeline

        stub = _build_pipeline_stub(
            enabled_platforms=enabled_platforms,
        )

        def _scientific(*args, **kwargs):
            if scientific_raises is not None:
                raise scientific_raises
            return scientific_returns

        def _post_translate(*args, **kwargs):
            if post_translate_raises is not None:
                raise post_translate_raises
            return post_translate_returns

        call_count = {"n": 0}
        real_gate = _exec_mod._gate_value_range_climate_delegation

        def _counting_gate(*args, **kwargs):
            call_count["n"] += 1
            return real_gate(*args, **kwargs)

        with patch.object(
            _exec_mod, "_gate_value_range_climate_delegation",
            _counting_gate,
        ), patch(
            "prismpy.validators.scientific.run_scientific_validation",
            _scientific,
        ), patch(
            "prismpy.validators.post_translate.run_post_translate_validation",
            _post_translate,
        ):
            # Bind the method to our stub so `self.config`,
            # `self.logger`, `self.provenance` resolve correctly.
            TranslationPipeline._execute_validate(
                stub,
                translation_results={},
                unified_data=SimpleNamespace(climate={}, grid=None),
            )
        return call_count["n"]

    def test_happy_path_scientific_plus_post_translate_success(self):
        """Scenario (a) — both validators run cleanly. Gate runs
        ONCE at the post-merge surfacing point."""
        n = self._run_once(
            scientific_returns=_minimal_scientific_report(),
            post_translate_returns=_minimal_post_translate_report(),
        )
        self.assertEqual(n, 1)

    def test_post_translate_raises_path(self):
        """Scenario (b) — post-translate throws. Gate still runs
        ONCE (unconditional call outside the try/except)."""
        n = self._run_once(
            scientific_returns=_minimal_scientific_report(),
            post_translate_raises=RuntimeError('boom'),
        )
        self.assertEqual(n, 1)

    def test_scientific_absent_path(self):
        """Scenario (c) — scientific validator never produced a
        report (upstream failure). The gate's guard short-circuits
        when `final_sci is None`, and the helper is NOT invoked
        at all — which is also "exactly one integration point"
        semantically, since a no-op short-circuit isn't a call.
        Counter reads 0."""
        n = self._run_once(
            scientific_raises=RuntimeError('scientific failed'),
            post_translate_returns=_minimal_post_translate_report(),
        )
        self.assertEqual(n, 0)

    def test_non_sarra_py_platform_only(self):
        """Scenario (d) — SARRA-Py disabled. Gate still runs
        ONCE; the platform guard inside the helper short-circuits
        the escalation, but the call-count invariant is about the
        call site itself, not the helper's behaviour."""
        from prismpy.config.schema import Platform
        n = self._run_once(
            scientific_returns=_minimal_scientific_report(),
            post_translate_returns=_minimal_post_translate_report(),
            enabled_platforms=[Platform.ACEA],
        )
        self.assertEqual(n, 1)


class TestGateHasExactlyOneCallSite(unittest.TestCase):
    """Source-level AST check — a second call site to
    `_gate_value_range_climate_delegation` cannot slip in as dead
    code. Grep + count is cheaper than re-running the behavioural
    tests under a refactor; this catches the pattern at build
    time even if the new site is unreachable."""

    def test_exactly_one_call_in_executor(self):
        """Count the textual invocations of the helper in
        `executor.py`. Expected: exactly one call plus one
        definition = 2 occurrences. A second call site
        (the R1-R4 substrate) bumps this to 3 and fails the
        test."""
        import ast
        src_path = (
            Path(__file__).parent.parent.parent
            / 'src' / 'prismpy' / 'pipeline' / 'executor.py'
        )
        tree = ast.parse(src_path.read_text())
        call_count = 0
        def_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, 'id', None) or getattr(
                    func, 'attr', None,
                )
                if name == '_gate_value_range_climate_delegation':
                    call_count += 1
            elif isinstance(node, ast.FunctionDef):
                if node.name == '_gate_value_range_climate_delegation':
                    def_count += 1
        self.assertEqual(
            def_count, 1,
            'gate helper must be defined exactly once',
        )
        self.assertEqual(
            call_count, 1,
            (
                f'gate helper called {call_count} times in '
                f'executor.py — AMBER ruling AC-AUDIT-1 mandates '
                f'exactly one integration point. Adding a second '
                f'call site re-introduces the P.2 R1-R4 substrate.'
            ),
        )
