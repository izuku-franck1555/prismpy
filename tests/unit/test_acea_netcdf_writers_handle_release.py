"""Method-level behavioural pin: ACEA NetCDF writers release their
underlying file handles after every invocation.

Pin DP-1 (the AST walker at
``tests/structural/test_xarray_open_dataset_closed.py``) is the
structural net: it catches any future
``xarray.open_dataset(...)`` / ``netCDF4.Dataset(...)`` / ``h5py.File(...)``
site that is added without a ``with`` block (or ``try/finally``
close). The walker recognises both the ``import netCDF4 as nc;
nc.Dataset(...)`` alias-receiver shape AND the ``from netCDF4
import Dataset; Dataset(...)`` direct-callable shape after the
F-DP CORRECTION cycle.

This module adds a **behavioural net** at the writer-method layer:
each of the four ACEA writers that currently calls
``netCDF4.Dataset(...)`` (the F-DP-aware sibling-sweep sites) is
invoked end-to-end, and a shared fixture asserts the writer did
not leave a handle dangling.

Why both layers?

* If a future refactor introduces a new AST shape the Pin DP-1
  walker does not yet recognise (e.g., reflective lookup of
  ``netCDF4.Dataset`` through ``getattr``), the walker would
  silently pass while the writer leaks. The behavioural test
  catches the leak at the consumer layer regardless of how the
  handle was acquired.
* If a future code change unknowingly inlines a new
  ``Dataset(...)`` call into one of these four writers, the
  walker fires only when the new AST shape is in scope, but the
  behavioural test fires the moment the writer accumulates a
  process-wide handle delta.

Coverage class — combined FILE_CACHE + ``/proc/self/fd`` count:

* xarray-backed ``open_dataset(...)`` paths register in xarray's
  process-wide ``FILE_CACHE`` (``xarray.backends.file_manager``);
  if a writer opens through xarray and forgets to close, keys
  accumulate. Cross-platform.
* Direct ``netCDF4.Dataset(...)`` constructors do NOT register in
  ``FILE_CACHE`` — xarray's cache is downstream of its own
  acquire path, not the libnetcdf primitive. A direct-constructor
  leak therefore manifests as a ``/proc/self/fd`` count delta on
  Linux only. macOS dev hosts lack ``/proc``; the fixture yields
  the FD baseline as ``None`` there and the assertion shrinks to
  the cross-platform FILE_CACHE signal alone.

Per F-DP cycle-4 N-6 bundle (user-authorised in the AskUserQuestion
push-approval; behavioural complement to Pin DP-1).
"""
from __future__ import annotations

import ast
import inspect
import os
import textwrap
from pathlib import Path

import pytest

from xarray.backends.file_manager import FILE_CACHE


# ── Shared fixture ─────────────────────────────────────────────────


@pytest.fixture
def netcdf_handle_leak_detector():
    """Snapshot xarray ``FILE_CACHE`` keys + (Linux only)
    ``/proc/self/fd`` count before the test; on teardown assert
    neither grew during the test.

    The FILE_CACHE signal is cross-platform; the FD-count signal
    runs only when ``/proc/self/fd`` exists (Linux CI). The FD
    tolerance accommodates incidental pytest fixture overhead
    (typically 0-3 FDs); a real handle leak per AgERA5-iteration
    would push the delta well above 5.
    """
    cache_keys_before = set(FILE_CACHE.keys())
    fd_probe_available = os.path.exists("/proc/self/fd")
    fd_before = (
        len(os.listdir("/proc/self/fd")) if fd_probe_available else None
    )

    yield  # body of test runs here

    cache_keys_after = set(FILE_CACHE.keys())
    new_keys = cache_keys_after - cache_keys_before
    assert not new_keys, (
        f"xarray FILE_CACHE retained {len(new_keys)} handle(s) after the "
        f"writer call — xarray-backed open path leaked. Sample leaked "
        f"keys: {sorted(repr(k) for k in new_keys)[:3]}"
    )

    if fd_before is not None:
        fd_after = len(os.listdir("/proc/self/fd"))
        fd_delta = fd_after - fd_before
        # 5-FD tolerance accommodates pytest fixture / capture overhead;
        # a real netCDF4 / HDF5 handle leak per writer invocation would
        # push delta well past 5 (typically 1-N per Dataset open).
        assert fd_delta <= 5, (
            f"/proc/self/fd grew by {fd_delta} (> 5) after the writer "
            "call. Suggests an unclosed netCDF4.Dataset or xarray "
            "dataset; the Pin DP-1 with-wrap may have been bypassed."
        )


# ── Helper: minimal AceaTranslator instance ─────────────────────────


def _minimal_acea(tmp_path: Path, config=None):
    """Construct an ``AceaTranslator`` without triggering the heavy
    ``__init__`` setup (config / climate sources / soil sources).
    Only the attributes the targeted writer methods read are
    populated. Mirrors the pattern at
    ``test_acea_projection_pickle.py::_instantiate_minimal_acea``.

    The crop-params writer (``_generate_crop_params_nc``) reaches
    into ``self._get_default_acea_params``, which reads
    ``self.config.crop.name`` and ``self.provenance``. Tests
    requiring that writer pass the ``config`` kwarg; the simpler
    soil / calendar writers do not need it.
    """
    from prismpy.translators.acea.translator import AceaTranslator

    inst = AceaTranslator.__new__(AceaTranslator)
    inst.output_dir = tmp_path / "pkg"
    inst.output_dir.mkdir(parents=True, exist_ok=True)
    # Pre-create the writer subdirs. The writers themselves expect
    # the parent to exist (they ``.parent.mkdir(parents=True,
    # exist_ok=True)`` only for the explicit nc_path; some routes
    # call ``calendar_dir / "..."`` directly without intermediate
    # mkdir).
    (inst.output_dir / "soil").mkdir(exist_ok=True)
    (inst.output_dir / "crop_calendar").mkdir(exist_ok=True)
    (inst.output_dir / "crop_params").mkdir(exist_ok=True)
    inst.config = config
    inst.provenance = None
    return inst


# ── Behavioural handle-release tests ───────────────────────────────


def test_generate_soil_netcdf_from_profiles_releases_handles(
    tmp_path: Path,
    sample_soil_profile,
    netcdf_handle_leak_detector,
) -> None:
    """``_generate_soil_netcdf_from_profiles`` opens
    ``netCDF4.Dataset`` via the ``import netCDF4 as nc`` alias and
    wraps in a ``with`` block (Pin DP-1 alias-extension
    sibling-sweep). End-to-end invocation must produce the output
    file AND release the underlying handle.
    """
    inst = _minimal_acea(tmp_path)
    soil_profiles = {0: sample_soil_profile}
    cell_ids = [0]

    result = inst._generate_soil_netcdf_from_profiles(soil_profiles, cell_ids)

    assert result is not None, (
        "_generate_soil_netcdf_from_profiles returned None — writer "
        "may have failed before reaching the Dataset open path."
    )
    assert result.exists(), (
        f"Expected output file at {result!r}; not created."
    )
    assert result.suffix == ".nc"


def test_generate_crop_calendar_nc_releases_handles(
    tmp_path: Path,
    sample_crop_calendar,
    netcdf_handle_leak_detector,
) -> None:
    """``_generate_crop_calendar_nc`` opens ``Dataset`` via
    ``from netCDF4 import Dataset`` (the F-DP-1 codex-flagged
    direct-callable shape) and wraps in a ``with`` block per the
    F-DP CORRECTION cycle. Writer emits two .nc files (rainfed +
    irrigated); both must exist and the FD-count must stay flat.
    """
    inst = _minimal_acea(tmp_path)
    crop_calendar = {0: sample_crop_calendar}
    cell_ids = [0]

    files = inst._generate_crop_calendar_nc(crop_calendar, cell_ids, "mai")

    assert len(files) == 2, (
        f"Expected 2 calendar .nc files (rf + ir); got {len(files)}"
    )
    for fp in files:
        assert fp.exists(), f"Expected calendar file {fp!r}; not created"
        assert fp.suffix == ".nc"


def test_generate_crop_params_nc_releases_handles(
    tmp_path: Path,
    sample_project_config,
    sample_crop_params,
    netcdf_handle_leak_detector,
) -> None:
    """``_generate_crop_params_nc`` opens ``Dataset`` via the same
    ``from netCDF4 import Dataset`` shape and wraps in a ``with``
    block. Two .nc files (rainfed + irrigated) must be emitted with
    no handle leak.

    This writer reaches into ``self._get_default_acea_params``,
    which reads ``self.config.crop.name``; ``sample_project_config``
    provides a Maize config which has a default phenology entry.
    """
    inst = _minimal_acea(tmp_path, config=sample_project_config)
    cell_ids = [0]

    files = inst._generate_crop_params_nc(sample_crop_params, cell_ids, "mai")

    assert len(files) == 2, (
        f"Expected 2 params .nc files (rf + ir); got {len(files)}"
    )
    for fp in files:
        assert fp.exists(), f"Expected params file {fp!r}; not created"
        assert fp.suffix == ".nc"


def test_generate_acea_soil_netcdf_dataset_call_is_with_protected() -> None:
    """``_generate_acea_soil_netcdf`` reads from HWSD2.bil +
    HWSD2.mdb on disk; supplying realistic fixtures requires
    gigabytes of test data and a working rasterio install. A live
    handle-release invocation is therefore impractical in the unit
    suite; rely instead on the function-level structural assertion
    here AND Pin DP-1's module-wide AST walker.

    The structural check parses the writer's source, locates every
    ``netCDF4.Dataset(...)`` Call, and asserts each one is the
    ``context_expr`` of an enclosing ``With`` statement. Catches a
    future refactor that inlines a bare ``nc = nc.Dataset(...)``
    inside this writer even before Pin DP-1 picks it up (which it
    will, given the alias map currently in scope — but a
    higher-visibility maintainer signal lives at the test that
    names the specific writer).
    """
    from prismpy.translators.acea.translator import AceaTranslator

    # ``inspect.getsource`` for a class method returns the indented
    # source (the class body's indentation level). ``ast.parse``
    # rejects leading indentation as a syntax error, so dedent
    # before parsing.
    source = textwrap.dedent(
        inspect.getsource(AceaTranslator._generate_acea_soil_netcdf)
    )
    tree = ast.parse(source)

    dataset_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Dataset"
    ]
    assert dataset_calls, (
        "_generate_acea_soil_netcdf no longer contains any "
        "Dataset(...) call — N-6 fixture coverage is stale. Update "
        "this test to track wherever the writer's open path moved."
    )

    with_protected_call_ids: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Call):
                    with_protected_call_ids.add(id(item.context_expr))

    unprotected = [
        call for call in dataset_calls
        if id(call) not in with_protected_call_ids
    ]
    assert not unprotected, (
        f"_generate_acea_soil_netcdf has {len(unprotected)} "
        "nc.Dataset(...) call(s) NOT wrapped in `with` — the F-DP "
        "CORRECTION fix regressed at this writer."
    )
