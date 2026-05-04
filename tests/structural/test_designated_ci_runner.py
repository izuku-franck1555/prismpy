"""Sprint E.0.5 F26 — designated-CI-runner walker.

Per AC-Q2-B1 + the contract Draft 4 F26 codification: bound-
gen MUST run only on the designated Linux + OpenBLAS CI
runner with the 5-thread pin set. Running on macOS Apple
Accelerate / Windows MKL / arbitrary dev machines is
forbidden because non-OpenBLAS BLAS backends drift on
parallel reduction order and produce non-byte-identical
percentile output.

This walker pins the ``bound-gen.yml`` workflow's runs-on
+ env block. Per codex Gate-A HIGH on commit 9, the walker
no longer permits a permanent skip — the workflow file is
required, and absence is a hard failure (a delete or rename
must trip the gate, not silently disable it).

Anti-mutation drills:

- Drop ``runs-on: ubuntu-*`` from the workflow → walker fires
  on the runs-on assertion.
- Drop any of the 5 thread-pin env vars from the workflow →
  walker fires on the env-vars assertion.
- Set any of the 5 thread vars to a value other than 1 →
  walker fires.
- Delete the workflow file → walker fires on the existence
  assertion (NOT a skip).
"""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOUND_GEN_YML = _REPO_ROOT / ".github" / "workflows" / "bound-gen.yml"


# 5-thread-pin set per the contract Draft 4 §AC-Q2-B1
# extended thread-pin set + warning-auditor probe-4-A
# (macOS Apple Accelerate doesn't honor OPENBLAS_NUM_THREADS).
_THREAD_PIN_VARS = frozenset({
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
})


def _load_workflow():
    """Load the bound-gen.yml workflow.

    Per codex Gate-A HIGH on commit 9, absence of the
    workflow is a HARD FAILURE rather than a skip. The
    workflow file is part of the F26 contract; deleting or
    renaming it must trip this gate.

    PyYAML 1.1 quirk: ``on:`` parses to the Python ``True``
    boolean rather than the string ``"on"`` (YAML 1.1
    treats ``on``/``off``/``yes``/``no`` as booleans). The
    loader normalizes this back to the string key so the
    rest of this module reads naturally.
    """
    if not _BOUND_GEN_YML.is_file():
        raise AssertionError(
            f"F26 violation: required workflow file missing at "
            f"{_BOUND_GEN_YML}. The bound-gen.yml workflow is "
            f"part of the AC-Q2-B1 contract; a delete or "
            f"rename must trip this gate, not silently disable "
            f"the designated-CI-runner walker."
        )
    with open(_BOUND_GEN_YML, encoding="utf-8") as fp:
        workflow = yaml.safe_load(fp)
    if isinstance(workflow, dict) and True in workflow:
        # Normalize PyYAML 1.1 boolean conversion of ``on:``.
        workflow["on"] = workflow.pop(True)
    return workflow


class TestBoundGenWorkflowExists(unittest.TestCase):
    """The bound-gen workflow file must exist; absence is a
    hard failure per codex Gate-A HIGH on commit 9."""

    def test_bound_gen_yml_exists(self):
        self.assertTrue(
            _BOUND_GEN_YML.is_file(),
            f"F26 violation: required workflow file missing at "
            f"{_BOUND_GEN_YML}.",
        )


class TestBoundGenRunsOnLinux(unittest.TestCase):
    """F26: bound-gen MUST run on a designated Linux runner.
    Apple Accelerate / MKL on Windows / dev-machine paths
    are forbidden."""

    def test_runs_on_ubuntu(self):
        workflow = _load_workflow()
        jobs = workflow.get("jobs", {})
        self.assertGreater(
            len(jobs), 0,
            "bound-gen.yml must declare at least one job.",
        )
        for job_name, job in jobs.items():
            with self.subTest(job=job_name):
                runs_on = job.get("runs-on", "")
                # Accept ubuntu-latest, ubuntu-22.04, ubuntu-20.04,
                # etc. Reject macos-* + windows-* + arbitrary.
                if isinstance(runs_on, list):
                    runs_on = runs_on[0] if runs_on else ""
                self.assertTrue(
                    str(runs_on).startswith("ubuntu"),
                    f"F26 violation: bound-gen job {job_name!r} "
                    f"runs-on {runs_on!r}; must be Linux "
                    f"(ubuntu-*).",
                )


class TestBoundGenThreadPins(unittest.TestCase):
    """F26: ALL 5 thread-pin env vars MUST be set to '1' on
    the bound-gen workflow."""

    def test_5_thread_pin_envs_set(self):
        workflow = _load_workflow()
        # Env can be at workflow level or per-job; check both.
        workflow_env = workflow.get("env", {}) or {}
        jobs = workflow.get("jobs", {})
        for job_name, job in jobs.items():
            with self.subTest(job=job_name):
                job_env = job.get("env", {}) or {}
                merged = {**workflow_env, **job_env}
                missing = _THREAD_PIN_VARS - merged.keys()
                self.assertFalse(
                    missing,
                    f"F26 violation: job {job_name!r} missing "
                    f"thread-pin env vars: {sorted(missing)}.",
                )

    def test_thread_pin_values_are_1(self):
        workflow = _load_workflow()
        workflow_env = workflow.get("env", {}) or {}
        jobs = workflow.get("jobs", {})
        for job_name, job in jobs.items():
            with self.subTest(job=job_name):
                job_env = job.get("env", {}) or {}
                merged = {**workflow_env, **job_env}
                for var in _THREAD_PIN_VARS:
                    if var not in merged:
                        continue
                    value = merged[var]
                    self.assertIn(
                        str(value), ("1",),
                        f"F26 violation: job {job_name!r} env "
                        f"{var}={value!r}; thread pins must "
                        f"be 1 for byte-identical bound-gen.",
                    )


class TestBoundGenTriggerPolicy(unittest.TestCase):
    """Per warning-auditor probe-A2 LOW counter-add: the
    workflow trigger filter must be paths-scoped on the
    bound-gen substrate (not every push) plus
    workflow_dispatch for explicit ratchet runs."""

    def test_paths_filter_includes_bound_gen_scope(self):
        workflow = _load_workflow()
        on = workflow.get("on")
        # Workflows can use 'on:' as a string, list, or dict.
        # Pin the dict form (paths-filter requires dict).
        # `on: workflow_dispatch` may be parsed to True by
        # PyYAML 1.1-style boolean conversion; tolerate both.
        self.assertIsInstance(
            on, dict,
            "bound-gen.yml 'on:' must be a dict for paths-filter.",
        )
        # Either push or pull_request must paths-filter on
        # bounds/ or koppen/.
        scoped = False
        for trigger in ("push", "pull_request"):
            cfg = on.get(trigger, {}) or {}
            if not isinstance(cfg, dict):
                continue
            paths = cfg.get("paths") or []
            joined = " ".join(paths)
            if "bounds" in joined or "koppen" in joined:
                scoped = True
                break
        self.assertTrue(
            scoped,
            "bound-gen.yml must paths-filter on bounds/ or "
            "koppen/ scope (else every push retriggers the "
            "designated-runner job).",
        )

    def test_workflow_dispatch_present(self):
        workflow = _load_workflow()
        on = workflow.get("on")
        self.assertIsInstance(on, dict)
        # ``workflow_dispatch:`` with no value parses to None.
        # The key presence is what matters.
        self.assertIn(
            "workflow_dispatch", on,
            "bound-gen.yml must support workflow_dispatch for "
            "explicit ratchet runs.",
        )


class TestBoundGenPreFlightAudit(unittest.TestCase):
    """Per warning-auditor probe-B-2 LOW counter-add: the
    workflow's first substantive step reports the runner's
    BLAS backend, env-var values, dep versions, and runner OS
    so a reviewer can audit F26 compliance from CI logs."""

    def test_pre_flight_audit_step_present(self):
        workflow = _load_workflow()
        jobs = workflow.get("jobs", {})
        # At least one job must have a step that reports
        # numpy.show_config (the canonical BLAS-backend probe).
        found = False
        for job in jobs.values():
            for step in job.get("steps", []):
                run = step.get("run", "")
                if "numpy.show_config" in run:
                    found = True
                    break
            if found:
                break
        self.assertTrue(
            found,
            "bound-gen.yml must run numpy.show_config() in a "
            "pre-flight audit step so the BLAS backend is "
            "visible in CI logs.",
        )


if __name__ == "__main__":
    unittest.main()
