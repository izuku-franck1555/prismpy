"""V2-22b Group L — cooperative cancellation across climate + translator loops.

Covers ACs L.1 (TAMSAT), L.2 (NASA POWER), L.3 (translators), L.4 (AgERA5),
L.5 (pre-lock), L.8 (F-5 legacy warning on no-force path), L.9
(PipelineCancelled at pipeline boundary), L.10 (cleanup idempotency),
L.12 (structural carve-out regression).

Unit-scoped: most tests use direct method invocation with mocked
downloaders so the ``cancel_check`` plumbing is exercised without any
real network / filesystem latency. Integration-scoped live-cancel
smoke lives in prismweb/tests/test_task_thread_integration.py.
"""
from __future__ import annotations

import ast
import json
import logging
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import patch

import pytest

from prismpy.models.region import BoundingBox, Region
from prismpy.sources.climate._cancel import PipelineCancelled, raise_if_cancelled


# ── Unit coverage for the _cancel module itself ──────────────────────


class TestRaiseIfCancelled:
    """Helper surface — small but load-bearing. Unit tests for every
    branch of the 3-line function."""

    def test_none_is_noop(self):
        raise_if_cancelled(None, "anywhere")  # does not raise

    def test_returns_false_is_noop(self):
        raise_if_cancelled(lambda: False, "noop")  # does not raise

    def test_returns_true_raises_with_where(self):
        with pytest.raises(PipelineCancelled) as excinfo:
            raise_if_cancelled(lambda: True, "tamsat.phase1.as_completed")
        assert excinfo.value.where == "tamsat.phase1.as_completed"
        assert str(excinfo.value) == "tamsat.phase1.as_completed"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def maradi_region() -> Region:
    return Region(
        name="Maradi",
        country="Niger",
        country_iso3="NER",
        bounds=BoundingBox(minx=6.5, miny=13.0, maxx=8.0, maxy=14.5),
        gadm_level=1,
    )


# ── AC L.1 — TAMSAT cancel hooks at 3 sites + pre-lock ──────────────


class TestTAMSATCancelHooks:
    """AC L.1: cancel_check observed at (a) pre-lock, (b) pre-submit,
    (c) per future.result in the as_completed loop, (d) Phase 2 per-file.
    cancel_futures=True verified via spy on ThreadPoolExecutor.shutdown.
    """

    def test_pre_lock_cancel_raises_before_lock_acquire(
        self, tmp_path: Path, maradi_region: Region, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC L.5: cancel observed pre-lock raises BEFORE FileLock.acquire.
        Locks are 7200 s uninterruptible; this hook is the primary
        cancel-correctness signal for queued runs on a contested region."""
        from prismpy.sources.climate.tamsat import TAMSATSource

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        lock_acquire_count = {"n": 0}
        import filelock as _flmod

        real_acquire = _flmod.FileLock.acquire

        def spy_acquire(self, *a, **kw):
            lock_acquire_count["n"] += 1
            return real_acquire(self, *a, **kw)

        monkeypatch.setattr(_flmod.FileLock, "acquire", spy_acquire)

        source = TAMSATSource(cache_dir=cache_dir)
        with pytest.raises(PipelineCancelled) as excinfo:
            source.retrieve(
                region=maradi_region,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 1),
                download=True,
                cancel_check=lambda: True,  # cancel already observed
            )
        assert excinfo.value.where == "tamsat.before_lock"

        # AC L.5: the retrieve() `try/except PipelineCancelled: raise`
        # carve-out re-raises, and pre-lock fires BEFORE the lock is
        # ever requested.
        assert lock_acquire_count["n"] == 0, (
            f"AC L.5: pre-lock cancel must fire before acquire; "
            f"recorded {lock_acquire_count['n']} acquires"
        )

    def test_pre_submit_cancel_raises_zero_http_calls(
        self, tmp_path: Path, maradi_region: Region, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC L.1: cancel observed AFTER lock but BEFORE executor.submit
        must raise before any JASMIN HTTP call fires."""
        from prismpy.sources.climate.tamsat import TAMSATSource

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        http_calls = {"n": 0}

        def fake_requests_get(*args, **kwargs):
            http_calls["n"] += 1
            raise AssertionError("HTTP should not fire on pre-submit cancel")

        # `requests` is imported inside _download_tamsat — inject a fake
        # via sys.modules so the lazy import picks it up.
        import sys, types as _types
        fake_requests = _types.SimpleNamespace(
            get=fake_requests_get,
            exceptions=_types.SimpleNamespace(
                Timeout=Exception, RequestException=Exception,
            ),
        )
        monkeypatch.setitem(sys.modules, "requests", fake_requests)

        # Toggle: first cancel_check call returns False (pre-lock passes),
        # subsequent calls return True so we raise after the lock is
        # acquired but BEFORE pre-submit.
        call_count = {"n": 0}

        def cancel_toggle() -> bool:
            call_count["n"] += 1
            return call_count["n"] > 1  # first call False, then True

        source = TAMSATSource(cache_dir=cache_dir)
        with pytest.raises(PipelineCancelled) as excinfo:
            source.retrieve(
                region=maradi_region,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 2),
                download=True,
                cancel_check=cancel_toggle,
            )

        # `where` identifies the hook site.
        assert excinfo.value.where in (
            "tamsat.after_lock",
            "tamsat.phase1.before_submit",
        )
        # No HTTP fired.
        assert http_calls["n"] == 0

    def test_cancel_inside_as_completed_shuts_down_executor(
        self, tmp_path: Path, maradi_region: Region, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC L.1: when a _download_nc worker observes cancel, the
        PipelineCancelled unwinds through future.result into the
        as_completed loop. The carve-out calls
        ``executor.shutdown(wait=False, cancel_futures=True)`` before
        re-raising so pending dates never touch JASMIN."""
        from prismpy.sources.climate.tamsat import TAMSATSource

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        http_calls = {"n": 0}

        def fake_requests_get(*args, **kwargs):
            http_calls["n"] += 1
            # Slow response so queued dates stay queued behind first wave
            time.sleep(0.05)

            class _Resp:
                status_code = 404

                @staticmethod
                def raise_for_status():
                    pass

            return _Resp()

        import sys, types as _types
        fake_requests = _types.SimpleNamespace(
            get=fake_requests_get,
            exceptions=_types.SimpleNamespace(
                Timeout=Exception, RequestException=Exception,
            ),
        )
        monkeypatch.setitem(sys.modules, "requests", fake_requests)

        shutdown_calls: List[Dict[str, Any]] = []
        from concurrent import futures as _fut_mod

        real_shutdown = _fut_mod.ThreadPoolExecutor.shutdown

        def spy_shutdown(self, wait=True, **kwargs):
            shutdown_calls.append({"wait": wait, **kwargs})
            return real_shutdown(self, wait=wait, **kwargs)

        monkeypatch.setattr(
            _fut_mod.ThreadPoolExecutor, "shutdown", spy_shutdown,
        )

        # Cancel observation toggle: first 2 calls False (pre-lock,
        # post-lock, pre-submit), then True inside the workers + the
        # as_completed loop top.
        call_count = {"n": 0}

        def cancel_toggle() -> bool:
            call_count["n"] += 1
            return call_count["n"] > 3

        source = TAMSATSource(cache_dir=cache_dir)
        with pytest.raises(PipelineCancelled) as excinfo:
            source.retrieve(
                region=maradi_region,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 10),  # 10 days → several workers
                download=True,
                cancel_check=cancel_toggle,
            )
        assert excinfo.value.where.startswith("tamsat.")

        # AC L.1 #3: executor.shutdown(wait=False, cancel_futures=True)
        # was called BEFORE the context manager's default shutdown(wait=True).
        forced = [s for s in shutdown_calls if s.get("cancel_futures") is True]
        assert len(forced) >= 1, (
            f"expected cancel_futures=True shutdown; got {shutdown_calls}"
        )
        # Pending futures not-yet-started get 0 HTTP calls; already-
        # running workers may have fired their requests.get before
        # cancel observation. Either way, http_calls should be bounded
        # WELL below the total 10 days.
        assert http_calls["n"] < 10


# ── AC L.9 — PipelineCancelled.where attribute ──────────────────────


class TestPipelineCancelledWhere:
    def test_exc_where_preserved_through_raise(self):
        try:
            raise_if_cancelled(lambda: True, "agera5.year=2022")
        except PipelineCancelled as exc:
            assert exc.where == "agera5.year=2022"
            return
        pytest.fail("PipelineCancelled should have raised")


# ── AC L.12 — structural carve-out regression meta-test ─────────────


class TestCarveOutRegression:
    """AC L.12 (Group L Gate B round 2 / F-3 rewrite): structural
    regression lock for the ``except PipelineCancelled: raise`` carve-outs.

    The PREVIOUS version of this test hardcoded a set of expected broad-
    except line numbers per file and only fired if a Try node's broad-
    except was exactly at one of those lines. That was vacuous — line
    numbers drift, and a new broad-except added at a NEW line (as in
    nasa_power.py:330, agera5.py:544, executor.py:985 — all three
    surfaced by Gate B) would be completely invisible to the check.

    The rewrite replaces hardcoded line numbers with a structural rule:

      For every ``Try`` node in a cancel-sensitive module, IF the try
      body transitively calls any function known to raise
      ``PipelineCancelled`` AND the Try has an ``except Exception`` /
      ``except BaseException`` handler, THEN the Try MUST also have an
      ``except PipelineCancelled`` handler.

    The "known to raise PipelineCancelled" set is the cancel-primitive
    helper (``raise_if_cancelled``) plus every method that calls it
    transitively — source ``retrieve`` / ``_download_*`` /
    ``_fetch_from_api``. This list is deliberately small and central:
    adding a new cancel-aware method means adding its name here, which
    is a commit the reviewer must SEE.
    """

    # Cancel-sensitive modules. F-3 expansion: nasa_power.py + agera5.py
    # were NOT in the previous list; that omission let F-1 and F-2
    # through evaluator without surfacing here.
    CANCEL_SENSITIVE_FILES: List[str] = [
        "src/prismpy/sources/climate/tamsat.py",
        "src/prismpy/sources/climate/agera5.py",
        "src/prismpy/sources/climate/nasa_power.py",
        "src/prismpy/pipeline/executor.py",
        "src/prismpy/translators/craft/translator.py",
        "src/prismpy/translators/pythia/translator.py",
        "src/prismpy/translators/acea/translator.py",
    ]

    # Function/method names that are known transitive raisers of
    # PipelineCancelled. A ``Try`` whose body contains a ``Call`` to any
    # of these must have an ``except PipelineCancelled`` carve-out when
    # it also has a broad except handler.
    CANCEL_RAISING_CALLABLES: set = {
        # The cancel primitive itself
        "raise_if_cancelled",
        # Climate-source PRIVATE helpers — unique names, no collision
        # with soil/GADM/GAEZ sources.
        "_download_tamsat",
        "_download_nc",
        "_download_agera5",
        "_fetch_from_api",
        # Generic method names like ``retrieve`` and ``translate`` are
        # DELIBERATELY excluded — they collide with cancel-inert soil /
        # GADM / GAEZ retrieve()s and with translator internals that
        # aren't on the cancel hot path. The orchestrator-convention
        # backstop below catches whole-stage wrappers (the F-9 pattern).
    }

    # Try blocks that call ``retrieve()`` or similar but whose receiver
    # is a soil / GADM / GAEZ source that is OUT OF SCOPE per contract
    # §11 (local reads, no HTTP, no cancel hooks). The ``retrieve``
    # method name collides with cancel-raising climate sources but the
    # call does not actually raise ``PipelineCancelled``. Each entry
    # is (file, try_lineno, reason).
    #
    # This whitelist is a REVIEWED decision (not drift): adding an
    # entry requires explaining why the Try's receiver cannot raise
    # ``PipelineCancelled``. Deleting an entry without adding a carve-
    # out immediately re-triggers the meta-test.
    OUT_OF_SCOPE_TRY_BLOCKS: set = {
        # V2-22b L Gate B round 3 F-9B: the previous `(executor.py, 288)`
        # entry whitelisted the entire `_execute_retrieve` wrapper —
        # exactly the site F-9 surfaced as a cancel-swallow. Removed.
        # The remaining entries are local-only (no HTTP, no cancel
        # hooks per §11) and documented below.
        #
        # iSDA soil source — local-only reads per §11
        # Line shifted 1104 → 1229 after F-R added GeometryRequiredError
        # class at module top (~9 lines) plus harmonize-stage 5-stage
        # filter (~120 lines). Then 1229 → 1230 after Sprint E.0
        # imported WarningCategory at module top (+1 import line).
        # Same try block, new line number.
        ("src/prismpy/pipeline/executor.py", 1230),
        # HWSD soil source — local-only reads per §11
        # Line shifted 1741 → 1866 after F-R additions. Then
        # 1866 → 1878 after Sprint D.1 commit 9 extended the
        # ``_retrieve_hwsd_for_grid`` docstring with the AC-4
        # tuple-return contract. Then 1878 → 1879 after Sprint
        # E.0 imported WarningCategory at module top. Then
        # 1879 → 1899 after the F-AL substrate-hardening sweep
        # added ``except (ImportError, ModuleNotFoundError)``
        # carve-outs to the TAMSAT + AgERA5 download blocks in
        # ``_load_climate_data`` (~10 lines apiece). Then
        # 1899 → 1910 after the F-AL scope-extension added an
        # ``except (ImportError, ModuleNotFoundError)`` carve-out
        # to the pygadm fallback in ``_execute_retrieve`` (~11 lines).
        ("src/prismpy/pipeline/executor.py", 1910),
        # HWSD per-cell sampling inside CRAFT translator
        ("src/prismpy/translators/craft/translator.py", 1692),
        # pygadm fallback inside _execute_retrieve: local pygadm import
        # + pygadm.Names/Items calls; no HTTP, cancel-inert.
        # Line shifted from 496 → 510 after V2-22c-PRE.4.1 added the
        # REMEDIATION enum value + multi-line comment to the
        # PipelineStage class definition. Then 510 → 519 after F-R
        # added GeometryRequiredError class at module top (~9 lines).
        # Then 519 → 520 after Sprint E.0 added WarningCategory import.
        ("src/prismpy/pipeline/executor.py", 520),
        # Provenance-flush inside the translator-dispatch except handler
        # in _execute_translate: writes decision records, cancel-inert.
        # Line shifted from 2338 → 2346 (PRE.3.3 thread-through)
        # → 2360 (PRE.4.1 enum) → 2404 after V2-22c-PRE.1.10
        # cascade orchestrator + climate metadata backstop added
        # ~40 lines to `_execute_harmonize`. Then 2404 → 2535 after
        # F-R AC-2 5-stage filter added ~120 lines to _execute_harmonize.
        # Then 2535 → 2570 after F-R Sprint A codex Gate B absorption
        # added ~35 lines to _execute_harmonize (Stage 3 exclusion-counter
        # increment, Stage 5 boundary_source fallback resolution, and
        # GeometryRequiredError re-raise on shapely parse failure for
        # centroid_strict). Then 2570 → 2617 after Sprint D.1 wired the
        # apply_harmonize_transformations call + harmonize-stats
        # metadata into _execute_harmonize (~47 lines). Then 2617 →
        # 2666 after Sprint D.1 commit 9 wired the HWSD
        # ``unavailable_cells`` propagation through the cascade
        # orchestrator (~49 lines: extended docstring + tuple-return
        # capture/remap + caller unpacking + retrieved_data stash).
        # Then 2666 → 2671 after Sprint D.1 commit 10 absorbed
        # codex LOW Q5 (replaced defensive ``getattr(..., [])`` with
        # direct attribute access + 5-line comment documenting the
        # rationale). Then 2671 → 2675 after Sprint E.0
        # commit 2 site-migrated the HWSD remap default-cause
        # to ``WarningCategory.SOIL_NO_HWSD_COVERAGE.value``
        # (multi-line argument + WarningCategory import on top).
        # Then 2675 → 2695 after the F-AL substrate-hardening
        # sweep added ``except (ImportError, ModuleNotFoundError)``
        # carve-outs to the TAMSAT + AgERA5 download blocks in
        # ``_load_climate_data`` (~10 lines apiece, ~20 lines
        # cumulative shift forward). Then 2695 → 2706 after the
        # F-AL scope-extension added an ``except (ImportError,
        # ModuleNotFoundError)`` carve-out to the pygadm fallback
        # in ``_execute_retrieve`` (~11 lines). Then 2706 → 2733
        # after Sprint E.3 fixup +15 (F-BN Boundary 2) added the
        # ``cockpit_override_sidecar = self._load_cockpit_override_sidecar()``
        # call + ``translator.cockpit_override_sidecar = ...`` assignment
        # into ``_execute_translate`` (~27 lines: load call + comment +
        # threading assignment + comment + closing context). Then
        # 2733 → 2774 after F-CK hot-fix +17 fan-out replaced the
        # single-key ``{0: CropCalendar(...)}`` producer at line 752
        # with an explicit ``climate_cell_ids`` filter + dict
        # comprehension over the climate cell roster (~41 lines:
        # 25-line F-CK contract docstring + path-dict ``isinstance``
        # filter + comprehension + closing). Then 2774 → 2788 after
        # the F-CK round-3 codex absorption replaced the ``cid >= 0``
        # filter with ``isinstance(cid, int)``-only and rewrote the
        # surrounding comment to document the sentinel-retention
        # contract (~14 lines: extended F-CK round-3 docstring
        # explaining why ``-1`` sentinels must survive the executor
        # fan-out so PYTHIA / ACEA pass ``validate_input_data``).
        # Then 2788 → 2799 after the Sprint F-CP fixup (AC-F-CP-14)
        # routed ``_create_placeholder_climate`` through the canonical
        # ``PLACEHOLDER_CLIMATE_SENTINEL_ID`` constant import and added
        # a 5-line ``_real_n_locations`` helper at the
        # ``_load_climate_data`` provenance emit (~11 lines cumulative
        # shift forward in ``executor.py``). Same provenance try block,
        # new line number per durable §27 producer-consumer parity.
        ("src/prismpy/pipeline/executor.py", 2799),
    }

    # V2-22b L Gate B round 3 F-9B: methods whose bodies are allowed
    # to wrap whole-stage work in ``except Exception`` without a
    # ``PipelineCancelled`` carve-out. Only orchestrator methods that
    # provably do NOT reach any cancel-raising call. Each entry is
    # (file, function_name, reason). The interprocedural walk below
    # uses this set as an explicit exemption — new orchestrators that
    # fit the pattern but don't reach cancel must be listed here.
    ORCHESTRATOR_EXEMPTIONS: set = {
        # _execute_harmonize doesn't call download/fetch paths
        ("src/prismpy/pipeline/executor.py", "_execute_harmonize"),
        # _execute_validate doesn't call download/fetch paths
        ("src/prismpy/pipeline/executor.py", "_execute_validate"),
        # _execute_package only writes archive files
        ("src/prismpy/pipeline/executor.py", "_execute_package"),
        # _load_soil_data reaches iSDA/HWSD which are cancel-inert
        ("src/prismpy/pipeline/executor.py", "_load_soil_data"),
        # _load_crop_params is config-only
        ("src/prismpy/pipeline/executor.py", "_load_crop_params"),
        # CRAFT _load_schema_areas: loads spatial schema geometries
        # from local GeoDataFrames; cancel-inert.
        ("src/prismpy/translators/craft/translator.py", "_load_schema_areas"),
    }

    @staticmethod
    def _extract_call_name(call_node: ast.Call) -> str:
        """Return the shortest identifier that matches what the caller
        wrote: ``foo()`` → 'foo', ``obj.foo()`` → 'foo',
        ``module.pkg.foo()`` → 'foo'. We match on the terminal
        identifier so ``source.retrieve(...)`` and
        ``tamsat.retrieve(...)`` both resolve to ``retrieve``."""
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return ""

    @staticmethod
    def _handler_label(handler: ast.ExceptHandler) -> str:
        """Return the exception-class identifier in the handler:
        ``except Exception`` → 'Exception', ``except Foo.Bar`` → 'Bar'.
        Bare ``except:`` returns ''."""
        if handler.type is None:
            return ""
        if isinstance(handler.type, ast.Name):
            return handler.type.id
        if isinstance(handler.type, ast.Attribute):
            return handler.type.attr
        if isinstance(handler.type, ast.Tuple):
            # `except (PipelineCancelled, Exception):` — treat as
            # containing PipelineCancelled if that name appears
            names = []
            for elt in handler.type.elts:
                if isinstance(elt, ast.Name):
                    names.append(elt.id)
                elif isinstance(elt, ast.Attribute):
                    names.append(elt.attr)
            return ",".join(names)
        return ast.unparse(handler.type)

    @classmethod
    def _collect_calls_in_body(cls, stmts: List[ast.AST]) -> set:
        """Return names of all functions called in ``stmts``, without
        descending into nested ``def`` / ``async def`` / ``lambda`` /
        ``class`` bodies. Scope-bounded — a nested function's calls
        don't count as the outer function's calls until invoked."""
        names: set = set()
        pending: List[ast.AST] = list(stmts)
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.Lambda, ast.ClassDef)):
                continue  # don't descend
            if isinstance(node, ast.Call):
                name = cls._extract_call_name(node)
                if name:
                    names.add(name)
            for child in ast.iter_child_nodes(node):
                pending.append(child)
        return names

    @classmethod
    def _compute_cancel_reachable_functions(
        cls, tree: ast.Module,
    ) -> set:
        """Fixed-point — return the set of function NAMES in the module
        whose body can transitively raise ``PipelineCancelled``.

        Seed set: functions whose body directly calls a
        ``CANCEL_RAISING_CALLABLES`` member (``raise_if_cancelled``,
        ``_download_*``, ``_fetch_from_api``, ``retrieve``, etc.).
        Fixed-point expansion: a function whose body calls any function
        already in the reachable set is itself added to the set. Loops
        until no new functions are added.

        Non-descending scope — a nested def inside a function counts as
        that nested function's calls, not the outer's.
        """
        # func_name → set of callee names found in that function's body
        func_calls: Dict[str, set] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_calls[node.name] = cls._collect_calls_in_body(
                    list(node.body),
                )

        # Seed: any function that directly calls a cancel-raising name
        reachable: set = set()
        for fname, callees in func_calls.items():
            if callees & cls.CANCEL_RAISING_CALLABLES:
                reachable.add(fname)

        # Fixed-point: propagate reachability
        changed = True
        while changed:
            changed = False
            for fname, callees in func_calls.items():
                if fname in reachable:
                    continue
                if callees & reachable:
                    reachable.add(fname)
                    changed = True
        return reachable

    @staticmethod
    def _try_enclosing_function(
        tree: ast.Module, try_node: ast.Try,
    ) -> str:
        """Return the name of the FunctionDef that immediately encloses
        ``try_node`` (skipping nested defs — return the innermost def
        whose direct body contains the try). Returns '' if the try is
        at module scope."""
        # Walk with parent tracking
        best_name = ""
        best_lineno = -1
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Is try_node lexically inside this function's body?
                for sub in ast.walk(node):
                    if sub is try_node:
                        if node.lineno > best_lineno:
                            best_lineno = node.lineno
                            best_name = node.name
                        break
        return best_name

    @classmethod
    def _collect_violations(
        cls, rel_path: str, source: str,
    ) -> List[str]:
        """Core meta-test rule — extracted as a helper so F-9C's
        AST-surgery regression test can call it with modified source
        contents in memory."""
        import re

        tree = ast.parse(source)
        reachable_funcs = cls._compute_cancel_reachable_functions(tree)
        violations: List[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue

            # Explicit out-of-scope whitelist per §11 — soil / GADM /
            # GAEZ sources don't raise PipelineCancelled even though
            # the method name ``retrieve`` collides with climate
            # sources that do.
            if (rel_path, node.lineno) in cls.OUT_OF_SCOPE_TRY_BLOCKS:
                continue

            # Whole-function exemption: if the enclosing function is
            # declared cancel-inert, skip ALL rules for this Try
            # regardless of what it calls. The exemption is a REVIEWED
            # declaration — adding a cancel path to an exempt function
            # requires removing the exemption first.
            _enclosing_name = cls._try_enclosing_function(tree, node)
            if (
                _enclosing_name
                and (rel_path, _enclosing_name) in cls.ORCHESTRATOR_EXEMPTIONS
            ):
                continue

            # F-9B interprocedural reach rule: Try body is cancel-hot
            # if any Call in its body targets a function name that is
            # in ``CANCEL_RAISING_CALLABLES`` OR in the module-local
            # reachable set. Catches cases where the Try wraps a local
            # helper (``_load_climate_data``) that itself calls
            # ``source.retrieve`` — the pattern F-9 surfaced in
            # ``_execute_retrieve``.
            cancel_hot = False
            body_calls = cls._collect_calls_in_body(list(node.body))
            if body_calls & (cls.CANCEL_RAISING_CALLABLES | reachable_funcs):
                cancel_hot = True

            # F-9B orchestrator-convention backstop: any broad except
            # inside a method named ``_execute_*`` or ``_load_*`` is
            # treated as cancel-hot unless explicitly exempted.
            enclosing = cls._try_enclosing_function(tree, node)
            if (
                enclosing
                and re.match(r"^(_execute_|_load_)", enclosing)
                and (rel_path, enclosing) not in cls.ORCHESTRATOR_EXEMPTIONS
            ):
                cancel_hot = True

            if not cancel_hot:
                continue

            broad_handler_idx = None
            cancel_handler_idx = None
            for idx, handler in enumerate(node.handlers):
                label = cls._handler_label(handler)
                if label in ("Exception", "BaseException"):
                    if broad_handler_idx is None:
                        broad_handler_idx = idx
                if "PipelineCancelled" in label:
                    cancel_handler_idx = idx

            if broad_handler_idx is None:
                continue

            context = f"in {enclosing}()" if enclosing else "(module scope)"
            if cancel_handler_idx is None:
                violations.append(
                    f"{rel_path}:{node.lineno} {context} Try body is cancel-hot "
                    f"+ has `except Exception` at position "
                    f"{broad_handler_idx} but NO `except PipelineCancelled` "
                    f"carve-out"
                )
                continue

            if cancel_handler_idx > broad_handler_idx:
                violations.append(
                    f"{rel_path}:{node.lineno} {context} `except PipelineCancelled` "
                    f"at position {cancel_handler_idx} comes AFTER "
                    f"`except Exception` at {broad_handler_idx} — Python "
                    f"evaluates handlers in order, so cancel would be "
                    f"swallowed by the broad handler. Move the cancel "
                    f"carve-out BEFORE the broad except."
                )
        return violations

    def test_cancel_sensitive_try_blocks_have_pipelinecancelled_carve_out(
        self,
    ) -> None:
        """Structural rule — a Try in a cancel-sensitive module that
        is cancel-hot (body calls a cancel-raising function, directly
        OR transitively via a local cancel-reachable helper, OR is
        inside an ``_execute_*`` / ``_load_*`` orchestrator method)
        AND has a broad ``except Exception`` handler MUST also have
        an ``except PipelineCancelled`` handler BEFORE it in handler
        order. Python evaluates handlers in order, so a misplaced
        carve-out is as bad as a missing one."""
        import pathlib as _pl

        prismpy_root = _pl.Path(__file__).resolve().parents[2]
        violations: List[str] = []
        for rel_path in self.CANCEL_SENSITIVE_FILES:
            full_path = prismpy_root / rel_path
            source = full_path.read_text(encoding="utf-8")
            violations.extend(self._collect_violations(rel_path, source))

        assert not violations, (
            "\n\n".join(
                ["Cancel-sensitive Try blocks missing carve-outs:"] + violations
            )
        )

    def test_f9_regression_ast_surgery_demonstrates_meta_test_binds_executor_retrieve(
        self,
    ) -> None:
        """F-9C regression (Gate B round 3): demonstrate that the
        meta-test's interprocedural rule + removed whitelist now
        BINDS the ``_execute_retrieve`` broad-except site. Same
        two-way proof pattern as F-1: read the real executor source,
        REMOVE the F-9 carve-out via string substitution (in memory,
        not on disk), re-run the meta-test rule, assert the F-9 site
        is flagged. Restoring the carve-out (i.e., using the real
        source) produces zero violations."""
        import pathlib as _pl

        prismpy_root = _pl.Path(__file__).resolve().parents[2]
        rel_path = "src/prismpy/pipeline/executor.py"
        full_path = prismpy_root / rel_path
        original = full_path.read_text(encoding="utf-8")

        # Sanity: real source passes the rule
        violations_before = self._collect_violations(rel_path, original)
        assert not violations_before, (
            f"F-9C: real executor.py unexpectedly has violations "
            f"({violations_before}). Test setup invalid."
        )

        # Remove the F-9 carve-out in memory. Match on the distinctive
        # ``V2-22b L Gate B round 3 (F-9)`` comment so a later refactor
        # won't accidentally corrupt.
        needle = (
            "        except PipelineCancelled:\n"
            "            # V2-22b L Gate B round 3 (F-9): whole-stage wrapper was\n"
        )
        assert needle in original, (
            "F-9C: canonical F-9A block not found in executor.py — "
            "either the comment drifted (update the needle) or the "
            "carve-out was removed entirely (which is itself a regression)."
        )
        start_idx = original.index(needle)
        end_marker = "        except Exception as e:"
        end_idx = original.index(end_marker, start_idx)
        mutated = original[:start_idx] + original[end_idx:]

        violations_after = self._collect_violations(rel_path, mutated)

        # The meta-test must flag the _execute_retrieve Try.
        f9_violations = [
            v for v in violations_after if "_execute_retrieve" in v
        ]
        assert f9_violations, (
            f"F-9C: meta-test should flag _execute_retrieve after "
            f"carve-out removal, but violations are: {violations_after}"
        )

    def test_f9_orchestrator_exemptions_are_real_functions(self) -> None:
        """F-9B coverage probe: every method name in
        ``ORCHESTRATOR_EXEMPTIONS`` MUST exist as a real ``def`` in its
        listed file. Catches drift when a method is renamed and the
        exemption silently becomes dead."""
        import pathlib as _pl

        prismpy_root = _pl.Path(__file__).resolve().parents[2]
        for rel_path, func_name in self.ORCHESTRATOR_EXEMPTIONS:
            full_path = prismpy_root / rel_path
            source = full_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            found = any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func_name
                for node in ast.walk(tree)
            )
            assert found, (
                f"F-9B: ORCHESTRATOR_EXEMPTIONS lists "
                f"({rel_path}, {func_name}) but no such function exists. "
                f"If the method was renamed, update the exemption OR "
                f"delete it and add a carve-out to prove the renamed "
                f"method is still cancel-inert."
            )


# ── Gate B round 2 regression tests — carve-out escape assertions ──


class TestBroadExceptCarveOutEscape:
    """Gate B round 2 rule: pair counter-based assertions ("mock hit 0
    times") with ``assertRaises(PipelineCancelled)`` so a future broad-
    except-swallow regression is caught — counters alone prove the
    raise happened but NOT that it escaped the broad except.
    F-1/F-2/F-4 all slipped past the Group L round 1 tests because
    counter assertions were the only binding; they pass even when the
    cancel is silently rewritten."""

    def test_f1_nasa_power_cancel_escapes_retrieve(
        self, maradi_region: Region, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """F-1 regression (Gate B round 2 polish P-1): must bind the
        F-1 carve-out at ``nasa_power.py:332-336`` specifically —
        NOT the year-top `raise_if_cancelled` which escapes BEFORE
        control enters the try-block the carve-out protects.

        Test shape: patch ``_fetch_from_api`` so it raises
        ``PipelineCancelled`` from INSIDE the try-block. ``cancel_check``
        returns ``False`` so the year-top hook doesn't short-circuit.
        The raise path is: ``_fetch_from_api()`` → inside the try at
        ``:296`` → past the carve-out at ``:332-336`` → out of
        ``retrieve()``. If the carve-out is removed, the broad
        ``except Exception`` at ``:337`` would rewrite the raise as
        ``RetrievalResult(success=False, errors=["API request failed..."])``
        and the test's ``pytest.raises(PipelineCancelled)`` would fail."""
        from prismpy.sources.climate.nasa_power import (
            NASAPowerSource, NASAPowerConfig,
        )

        source = NASAPowerSource(config=NASAPowerConfig())

        # Make _fetch_from_api raise PipelineCancelled — this is the
        # exact path the F-1 carve-out exists to let propagate.
        def fake_fetch(*args, **kwargs):
            raise PipelineCancelled("nasa_power.fetch.attempt=1")

        monkeypatch.setattr(source, "_fetch_from_api", fake_fetch)

        with pytest.raises(PipelineCancelled) as excinfo:
            source.retrieve(
                lat=14.0, lon=7.5,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                use_cache=False,  # force API path, skip cache
                # cancel_check=False so year-top hook is a no-op and the
                # PipelineCancelled comes from _fetch_from_api inside
                # the try — exercising the F-1 carve-out specifically.
                cancel_check=lambda: False,
            )
        assert excinfo.value.where == "nasa_power.fetch.attempt=1"

    @pytest.mark.parametrize(
        "arm_after_stage,expected_where",
        [
            ("retrieve", "executor.stage.harmonize"),
            ("harmonize", "executor.stage.translate"),
            # V2-22c-PRE.4.3 — REMEDIATION inserts between TRANSLATE
            # and VALIDATE, so the cancel-check that used to fire at
            # the top of validate now fires at the top of
            # remediation. The validate→package transition stays as
            # before; same for the earlier transitions.
            ("translate", "executor.stage.remediation"),
            ("remediation", "executor.stage.validate"),
            ("validate", "executor.stage.package"),
        ],
        ids=[
            "retrieve→harmonize", "harmonize→translate",
            "translate→remediation", "remediation→validate",
            "validate→package",
        ],
    )
    def test_f6_inter_stage_cancel_fires_at_stage_top(
        self, monkeypatch: pytest.MonkeyPatch,
        arm_after_stage: str, expected_where: str,
    ) -> None:
        """F-6 regression (Gate B round 2 polish P-3): the inter-stage
        cancel hook must fire at the top of EVERY stage iteration, not
        just retrieve→harmonize. Parametrized across all four
        transitions so an asymmetric bug (e.g., a hook commented out
        on one stage) is caught on its specific transition.

        Test shape: arm cancel at the boundary between the stage named
        ``arm_after_stage`` and the next one. Assert the NEXT stage's
        top-of-iteration ``raise_if_cancelled`` raises with
        ``where=expected_where`` AND that stage's ``_execute_*``
        never runs."""
        from prismpy.pipeline.executor import (
            TranslationPipeline, StageResult, PipelineStage,
        )
        from prismpy.translators.base import TranslationResult, UnifiedData
        from prismpy.config.schema import Platform

        class _FakeConfig:
            class project:
                name = "test-f6"
            class region:
                name = "Maradi"
                country = "Niger"
            class output:
                base_dir = "/tmp/test-f6"
            def get_enabled_platforms(self):
                return []

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.config = _FakeConfig()
        pipeline.logger = logging.getLogger("test.f6")
        pipeline.provenance = None
        pipeline.translators = {}

        cancel_armed = {"flag": False}

        def cancel_check() -> bool:
            return cancel_armed["flag"]

        # Each stage helper: succeeds quickly, arms cancel if this is
        # the arm-after stage, and records it ran.
        stages_ran: List[str] = []

        def _make_stage_runner(stage_name: str, result_data: Any):
            def _runner(self, *args, **kwargs):
                stages_ran.append(stage_name)
                if stage_name == arm_after_stage:
                    cancel_armed["flag"] = True
                return StageResult(
                    stage=getattr(PipelineStage, stage_name.upper()),
                    success=True, data=result_data,
                )
            return _runner

        # Translate + validate + package have special return shapes;
        # mock them to succeed.
        monkeypatch.setattr(
            TranslationPipeline,
            "_execute_retrieve",
            _make_stage_runner("retrieve", {}),
        )
        # _execute_harmonize must return truthy data so the TRANSLATE
        # stage enters (executor guards translate on ``if unified_data:``).
        def fake_harmonize(self, retrieved_data):
            stages_ran.append("harmonize")
            if arm_after_stage == "harmonize":
                cancel_armed["flag"] = True
            return StageResult(
                stage=PipelineStage.HARMONIZE, success=True,
                data=UnifiedData(region=None),
            )
        monkeypatch.setattr(
            TranslationPipeline, "_execute_harmonize", fake_harmonize,
        )

        # _execute_translate returns Dict[str, TranslationResult]; need
        # at least one success so VALIDATE stage enters.
        def fake_translate(self, unified_data):
            stages_ran.append("translate")
            if arm_after_stage == "translate":
                cancel_armed["flag"] = True
            return {
                "craft": TranslationResult(
                    success=True, platform=Platform.CRAFT,
                    output_dir=Path("/tmp/test-f6/craft"),
                    output_files=[], errors=[], warnings=[], metadata={},
                ),
            }
        monkeypatch.setattr(
            TranslationPipeline, "_execute_translate", fake_translate,
        )

        # V2-22c-PRE.4.1 — REMEDIATION fake. The stage runs between
        # TRANSLATE and VALIDATE; needs a fake that returns a
        # success StageResult so the test reaches VALIDATE.
        def fake_remediation(self, translation_results, unified_data=None):
            stages_ran.append("remediation")
            if arm_after_stage == "remediation":
                cancel_armed["flag"] = True
            return StageResult(
                stage=PipelineStage.REMEDIATION, success=True, data={},
            )
        monkeypatch.setattr(
            TranslationPipeline, "_execute_remediation", fake_remediation,
        )

        def fake_validate(self, translation_results, unified_data=None):
            stages_ran.append("validate")
            if arm_after_stage == "validate":
                cancel_armed["flag"] = True
            return StageResult(
                stage=PipelineStage.VALIDATE, success=True, data={},
            )
        monkeypatch.setattr(
            TranslationPipeline, "_execute_validate", fake_validate,
        )

        def fake_package(self, unified_data, translation_results, validate_result):
            stages_ran.append("package")
            return StageResult(
                stage=PipelineStage.PACKAGE, success=True, data={},
            )
        monkeypatch.setattr(
            TranslationPipeline, "_execute_package", fake_package,
        )

        with pytest.raises(PipelineCancelled) as excinfo:
            pipeline.execute(cancel_check=cancel_check)

        assert excinfo.value.where == expected_where, (
            f"expected {expected_where}; got {excinfo.value.where}"
        )
        # The stage that arms cancel DID run; the next stage must NOT
        next_stage = expected_where.rsplit(".", 1)[-1]
        assert arm_after_stage in stages_ran
        assert next_stage not in stages_ran, (
            f"AC F-6: {next_stage} stage ran despite inter-stage cancel; "
            f"stages_ran = {stages_ran}"
        )

    def test_f2_agera5_cancel_escapes_retrieve(
        self, tmp_path: Path, maradi_region: Region,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """F-2 regression: AgERA5 ``retrieve()`` with cancel observed
        inside the download branch must RAISE past the broad
        ``except Exception`` at agera5.py:564. The carve-out at
        :559 (``except PipelineCancelled: raise``) is what makes
        this work. Pre-fix behavior: cancel was rewritten as
        ``RetrievalResult(success=False, errors=["Download failed: ..."])``.

        F-AB env-divergence fix: ``SARRA_data_download`` is not in
        ``pyproject.toml`` dependencies, so CI / evaluator
        environments do NOT have it installed. The previous form of
        this test let ``_download_agera5`` run, which immediately
        executes ``from SARRA_data_download.get_AgERA5_data import
        download_AgERA5_year`` at agera5.py:727. In environments
        without the optional library, that import raised
        ``ImportError`` BEFORE any cancel hook fired, and the broad
        ``except Exception`` rewrote it as ``Download failed:
        ...`` — exactly the swallow path the carve-out exists to
        prevent, but for an unrelated reason (missing optional dep).
        Patching ``_download_agera5`` to raise ``PipelineCancelled``
        directly (mimicking the year-top
        ``raise_if_cancelled`` at agera5.py:759) decouples this
        test from the optional SARRA_data_download install footprint
        while still binding the F-2 retrieve()-level carve-out
        — removing the carve-out at :559 still rewrites the raise
        as ``RetrievalResult`` because ``PipelineCancelled``
        inherits from ``Exception`` and would be caught by :564.
        """
        from prismpy.sources.climate.agera5 import AgERA5Source

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Make the SARRA library probe return True so retrieve enters
        # the download branch (and the filelock + broad except block).
        monkeypatch.setattr(
            AgERA5Source, "sarra_download_available",
            property(lambda self: True),
        )

        # F-AB env-divergence fix: stub ``_download_agera5`` so the
        # cancel surfaces from inside the protected try-block without
        # touching the optional ``SARRA_data_download`` import. Mimics
        # the year-top ``raise_if_cancelled`` site at agera5.py:759.
        def fake_download_raises_cancel(
            self, *, bounds, start_date, end_date,
            output_dir: Path, region_name,
            progress_callback=None, cancel_check=None,
        ):
            raise_if_cancelled(
                cancel_check, f"agera5.year={start_date.year}",
            )

        monkeypatch.setattr(
            AgERA5Source,
            "_download_agera5",
            fake_download_raises_cancel,
        )

        source = AgERA5Source(cache_dir=cache_dir)
        with pytest.raises(PipelineCancelled) as excinfo:
            source.retrieve(
                region=maradi_region,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 1),
                download=True,
                cancel_check=lambda: True,
            )
        # Tighter than ``startswith("agera5.")``: pin the canonical
        # year-top cancel site so a future hook at, e.g.,
        # ``agera5.checksum`` or ``agera5.lock`` cannot satisfy this
        # assertion vacuously. Year value left wild so the test can be
        # parametrized over different ``start_date`` years without
        # touching this assertion.
        assert excinfo.value.where.startswith("agera5.year=")


# ── AC L.6 — post-cancel B2 invariant preserved ─────────────────────


class TestPostCancelB2Invariant:
    """AC L.6: after a cancel mid-download, the B2 `_manifest.writing`
    marker must persist on disk. The next `retrieve()` call observes
    the marker, treats the cache as cold, and force-redownloads —
    no half-state the B2 reader would treat as valid.

    This test exercises the end-to-end invariant at the retrieve()
    level: plant a marker + partial data, trigger cancel mid-download,
    verify marker persistence, then invoke a fresh retrieve() and
    observe the `marker_present` state fire."""

    def test_marker_persists_after_cancel_and_triggers_force_on_retry(
        self, tmp_path: Path, maradi_region: Region, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from prismpy.sources.climate.tamsat import (
            TAMSATSource, MARKER_FILENAME, MANIFEST_FILENAME,
        )

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = cache_dir / "tamsat" / "maradi"

        # First retrieve: cancel fires inside the download (after marker
        # is written but before manifest + marker_delete). Mirrors a
        # real mid-download cancel.
        cancel_armed = {"now": False}

        def fake_download_tamsat_cancelled(
            self, *, bounds, start_date, end_date,
            output_dir: Path, region_name,
            progress_callback=None, cancel_check=None,
        ):
            # Pretend we're mid-download — arm cancel and then raise
            # via the helper to mimic a Phase-1 as_completed cancel.
            cancel_armed["now"] = True
            raise_if_cancelled(cancel_check, "tamsat.phase1.as_completed")

        monkeypatch.setattr(
            "prismpy.sources.climate.tamsat.TAMSATSource._download_tamsat",
            fake_download_tamsat_cancelled,
        )

        source = TAMSATSource(cache_dir=cache_dir)
        with pytest.raises(PipelineCancelled):
            source.retrieve(
                region=maradi_region,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 1),
                download=True,
                cancel_check=lambda: cancel_armed["now"],
            )

        # AC L.6 part 1: marker persists on disk post-cancel
        marker_path = data_dir / MARKER_FILENAME
        manifest_path = data_dir / MANIFEST_FILENAME
        assert marker_path.exists(), (
            "AC L.6: _manifest.writing marker should persist after "
            "a cancelled download"
        )
        assert not manifest_path.exists(), (
            "AC L.6: _manifest.json must NOT exist on the cancelled "
            "path — only successful downloads write the manifest"
        )

        # AC L.6 part 2: next retrieve() observes marker_present →
        # force_redownload → successful fresh download.
        observed_force: Dict[str, Any] = {}

        def fake_download_tamsat_success(
            self, *, bounds, start_date, end_date,
            output_dir: Path, region_name,
            progress_callback=None, cancel_check=None,
        ):
            observed_force["data_dir_exists"] = output_dir.exists()
            observed_force["called"] = True
            (output_dir / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif").write_bytes(b"NEW")

        monkeypatch.setattr(
            "prismpy.sources.climate.tamsat.TAMSATSource._download_tamsat",
            fake_download_tamsat_success,
        )

        source = TAMSATSource(cache_dir=cache_dir)
        result = source.retrieve(
            region=maradi_region,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 1),
            download=True,
        )

        assert result.success, f"retry after cancel failed: {result.errors}"
        assert result.metadata.get("cache_state") == "marker_present", (
            "AC L.6: next retrieve must observe 'marker_present' state"
        )
        assert observed_force["called"] is True
        # Marker removed after successful retry; manifest written
        assert not marker_path.exists()
        assert manifest_path.exists()
