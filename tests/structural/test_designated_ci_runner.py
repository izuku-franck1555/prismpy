"""Sprint E.0.5 F26 — designated-CI-runner walker.

Per AC-Q2-B1 + the contract Draft 4 F26 codification: bound-
gen MUST run only on the designated Linux + OpenBLAS CI
runner with the 5-thread pin set. Running on macOS Apple
Accelerate / Windows MKL / arbitrary dev machines is
forbidden because non-OpenBLAS BLAS backends drift on
parallel reduction order and produce non-byte-identical
percentile output.

This walker pins the ``bound-gen.yml`` workflow's runs-on
+ env block. The YAML ships in commit 10; while the YAML is
absent the walker SkipTests with a clear message so the test
file lights up automatically once the workflow lands.

Anti-mutation drills:

- Drop ``runs-on: ubuntu-*`` from the workflow → walker fires
  on the runs-on assertion.
- Drop any of the 5 thread-pin env vars from the workflow →
  walker fires on the env-vars assertion.
- Set any of the 5 thread vars to a value other than 1 →
  walker fires.
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
    if not _BOUND_GEN_YML.is_file():
        raise unittest.SkipTest(
            f"F26 walker pre-existing the YAML: "
            f"{_BOUND_GEN_YML} does not exist yet. Sprint "
            f"E.0.5 commit 10 lands the workflow file; once "
            f"it does, this walker pins runs-on + thread "
            f"vars."
        )
    with open(_BOUND_GEN_YML, encoding="utf-8") as fp:
        return yaml.safe_load(fp)


class TestBoundGenRunsOnLinux(unittest.TestCase):
    """F26: bound-gen MUST run on a designated Linux runner.
    Apple Accelerate / MKL on Windows / dev-machine paths
    are forbidden."""

    def test_bound_gen_yml_exists(self):
        if not _BOUND_GEN_YML.is_file():
            self.skipTest(
                f"F26 walker pre-existing the YAML: "
                f"{_BOUND_GEN_YML} lands in commit 10."
            )
        self.assertTrue(_BOUND_GEN_YML.is_file())

    def test_runs_on_ubuntu(self):
        workflow = _load_workflow()
        jobs = workflow.get("jobs", {})
        self.assertGreater(
            len(jobs), 0, "bound-gen.yml must declare at least one job.",
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


if __name__ == "__main__":
    unittest.main()
