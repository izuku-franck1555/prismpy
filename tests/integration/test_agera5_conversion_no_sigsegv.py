"""F-DP AC-DP-4 / Pin DP-4 — real-CDS Maradi regression fixture.

Running the AgERA5 conversion loop over a full 2-year × 6-variable
Maradi download (~4380 iterations) is the empirical reproduction of
the SIGSEGV class root-caused at
``prismpy/src/prismpy/vendor/sarra_data_download/get_AgERA5_data.py:273``.
This test invokes the production caller surface via subprocess so
any segfault surfaces as ``returncode in (-11, 139)`` to the parent
pytest process rather than crashing pytest itself.

## Markers

* ``@pytest.mark.slow`` — excluded from the FAST CI tier.
* ``@pytest.mark.skipif(not _has_cds_credentials())`` — needs
  ``~/.cdsapirc`` OR both ``CDSAPI_URL`` + ``CDSAPI_KEY`` env vars.

Run on-demand or nightly. The deployment-engineer canary post-deploy
is the production-side equivalent (real user-facing UI run).

## Why subprocess

A SIGSEGV in the libhdf5 close path inside the conversion loop
would kill the whole Python process. Spawning the conversion in a
subprocess + checking ``returncode`` lets pytest report a clean
test failure with the subprocess stderr tail instead of a pytest
collector crash.

Per F-DP contract LOCKED cycle-4 §Z.1 + §X.6 pytest wrapper +
§Y.3 corrected subprocess entry-point.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _has_cds_credentials() -> bool:
    """Return True when CDS API credentials are available, either via
    the user-level ``~/.cdsapirc`` file (the cdsapi library default)
    or via the ``CDSAPI_URL`` + ``CDSAPI_KEY`` env var pair (the
    CI / container override)."""
    if (Path.home() / ".cdsapirc").exists():
        return True
    return bool(
        os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY")
    )


# Expected output count for Maradi 2-year × 6-variable run:
#   2 years × 6 variables × 365 days = 4380 .tif files.
# The threshold is 4000 (not 4380) to absorb the ~3% slack the
# vendor sometimes produces when one variable has a missing day in
# the CDS archive — the SIGSEGV signature is "0 successful files
# past iteration ~700-800", not "missing one day".
EXPECTED_TIF_FLOOR = 4000

# Subprocess wall-clock ceiling (~60 min). Real-CDS Maradi 2-year run
# is typically 20-40 min depending on CDS queue depth + bandwidth;
# 60 min absorbs queue contention without masking genuine hangs.
SUBPROCESS_TIMEOUT_SECONDS = 3600


@pytest.mark.slow
@pytest.mark.skipif(
    not _has_cds_credentials(),
    reason=(
        "needs ~/.cdsapirc or CDSAPI_URL+CDSAPI_KEY env vars; "
        "skipping real-CDS regression fixture"
    ),
)
def test_agera5_two_year_conversion_no_sigsegv(tmp_path: Path) -> None:
    """Pin DP-4 / AC-DP-4: a full 2-year × 6-variable AgERA5 download
    on Maradi MUST complete without a worker SIGSEGV in the conversion
    C-layer.

    Pre-AC-DP-1a, this consistently SIGSEGVed around iteration
    ~700-800 of the inner conversion loop because xarray's
    ``CachingFileManager`` retained HDF5 handles in a 128-entry LRU
    and cumulative state collapsed in libhdf5 1.14.6 at LRU
    eviction time. AC-DP-1a wrapped the per-iteration
    ``xr.open_dataset`` in a ``with`` block; each iteration's handle
    is released synchronously at the iteration boundary so LRU
    eviction never fires.

    The subprocess wrapper catches the returncode shape that
    SIGSEGV produces:

    * ``-11`` on Unix (signal -SIGSEGV)
    * ``139`` on shells that surface signal+128 (128 + 11)

    Either form causes a pytest failure with the subprocess stderr
    tail attached so the operator can read the libhdf5 traceback.
    """
    fixture_script = Path(__file__).parent / "_agera5_two_year_subprocess.py"
    assert fixture_script.is_file(), (
        f"Subprocess entry-point script missing at {fixture_script!r}; "
        "AC-DP-4 cannot run."
    )

    cp = subprocess.run(
        [sys.executable, str(fixture_script), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )

    if cp.returncode in (-11, 139):
        pytest.fail(
            "AC-DP-4 SIGSEGV regression detected. The AgERA5 "
            "conversion subprocess returned "
            f"{cp.returncode} (SIGSEGV signature). Worker died in "
            "the libhdf5 close path — AC-DP-1a `with` block is "
            "missing or broken in the production AgERA5 "
            "conversion path.\n\n"
            "stderr tail:\n"
            f"{cp.stderr[-3000:]}"
        )
    if cp.returncode != 0:
        pytest.fail(
            "AC-DP-4 subprocess failed with non-SIGSEGV non-zero "
            f"returncode={cp.returncode}. Inspect the stderr below "
            "for the root cause.\n\n"
            "stderr tail:\n"
            f"{cp.stderr[-3000:]}\n\n"
            "stdout tail:\n"
            f"{cp.stdout[-2000:]}"
        )

    # Output verification: per §X.5 cycle-2 deployment-engineer
    # canary path correction, the finished cache lives at
    # ``<save_path>/AgERA5_<selected_area>/`` after vendor relocates
    # the staged files. The intermediate ``2_conversion/`` stage
    # directories are wiped by the vendor after a successful
    # relocate.
    relocated_dir = tmp_path / "AgERA5_maradi"
    assert relocated_dir.is_dir(), (
        f"AC-DP-4: final relocated cache directory missing at "
        f"{relocated_dir!r}; vendor likely failed mid-pipeline. "
        f"Subprocess stderr tail:\n{cp.stderr[-2000:]}"
    )

    total_tifs = list(relocated_dir.rglob("*.tif"))
    assert len(total_tifs) >= EXPECTED_TIF_FLOOR, (
        f"AC-DP-4: only {len(total_tifs)} .tif files found under "
        f"{relocated_dir!r}; expected >= {EXPECTED_TIF_FLOOR} (2 "
        f"years × 6 vars × ~365 days = ~4380). Partial completion "
        "signals a non-SIGSEGV failure mode — likely a CDS request "
        "failure for one or more (year, variable) pairs."
    )
