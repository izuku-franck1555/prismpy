"""Pin DP-2 amendment (multi-threaded variant): assert FILE_CACHE
invariance under CONCURRENT conversion-loop execution from 2 threads
sharing the same process-global xarray cache.

Production threading model: gunicorn worker = 1 process; xarray
FILE_CACHE is module-level singleton; background scheduler / Channels
threads may incidentally touch xarray paths concurrent with pipeline
executor's conversion loop in the worker's main thread. libhdf5 1.14.6
is not thread-safe in close() paths; without AC-DP-1a's `with` block,
FILE_CACHE LRU eviction races with concurrent libhdf5 reads → SIGSEGV.

This test exercises the same convert function under 2-thread concurrent
load and asserts per-workspace cache-key invariance (no FILE_CACHE key
references either thread's tmp_path workspace post-run). Mirrors the
single-thread sibling's `_cache_keys_referencing(path)` pattern at
test_agera5_conversion_handle_release.py:172 (cycle-1.5 SF-1 absorption)
to avoid the vacuous-pass class that raw `len(FILE_CACHE)` is prone to
when prior tests filled the LRU to maxsize=128.
"""
from __future__ import annotations

import datetime
import threading
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr
from xarray.backends.file_manager import FILE_CACHE

from prismpy.vendor.sarra_data_download.get_AgERA5_data import (
    convert_AgERA5_netcdf_to_geotiff,
)

# Concurrent open count per thread. 2 threads × 200 = 400 cumulative opens
# > xarray default file_cache_maxsize=128 → guarantees eviction PRESSURE
# (cumulative opens > maxsize). Interleaving depends on GIL scheduling
# but FILE_CACHE per-workspace-key invariance assertion is correct under
# any interleaving. Anti-mutation pin asserts the threshold relationship.
_FILES_PER_THREAD = 200
_CACHE_MAXSIZE_DEFAULT = 128  # xarray default per AM-A2 verified probe


# ── Synthetic fixture (mirrors ST sibling shape; CF time units) ─────


_SYNTHETIC_VARIABLE: tuple = ("2m_temperature", "24_hour_mean")
_SYNTHETIC_VARIABLE_PAIR = (
    _SYNTHETIC_VARIABLE[0] + "_" + _SYNTHETIC_VARIABLE[1]
)
_SYNTHETIC_SELECTED_AREA = "synth_mt"
_SYNTHETIC_YEAR = 2022


def _build_synthetic_nc(path: Path, base_day: int) -> None:
    """Build CF-conformant single-variable netCDF mirroring AgERA5 shape."""
    with nc.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("lat", 2)
        ds.createDimension("lon", 2)
        t = ds.createVariable("time", "f8", ("time",))
        t.setncattr("units", "days since 1900-01-01 00:00:00")
        t.setncattr("calendar", "standard")
        t[:] = [float(44000 + base_day)]
        lat = ds.createVariable("lat", "f4", ("lat",))
        lat[:] = [12.99, 13.0]
        lon = ds.createVariable("lon", "f4", ("lon",))
        lon[:] = [6.28, 6.29]
        v = ds.createVariable(
            "Temperature_Air_2m_Mean_Daily", "f4", ("time", "lat", "lon")
        )
        v.setncattr("units", "K")
        v[:] = np.full((1, 2, 2), 273.15 + base_day, dtype="f4")


def _seed_thread_workspace(root: Path, n_files: int) -> Path:
    """Create extraction_path + n_files .nc files; return extraction_path."""
    extraction_path = (
        root
        / "1_extraction"
        / f"AgERA5_{_SYNTHETIC_SELECTED_AREA}"
        / str(_SYNTHETIC_YEAR)
        / _SYNTHETIC_VARIABLE_PAIR
    )
    extraction_path.mkdir(parents=True)
    for j in range(n_files):
        _build_synthetic_nc(extraction_path / f"{j:04d}.nc", base_day=j)
    return extraction_path


def _cache_keys_referencing(path: Path) -> list:
    """Return every FILE_CACHE key whose serialised form contains
    ``str(path)``. Mirrors the ST sibling's pattern (cycle-1.5 SF-1) —
    stable across xarray versions even if internal key shape rebases."""
    target = str(path)
    return [k for k in FILE_CACHE.keys() if target in repr(k)]


# ── Anti-vacuous guards (mirrors ST sibling guards) ─────────────────


def test_file_cache_attribute_importable_mt() -> None:
    """Fail-fast guard: if xarray renames or removes FILE_CACHE, the
    main MT assertion's import would crash with a cryptic ImportError
    at test-collection time. This dedicated guard surfaces the
    underlying issue instead. Mirrors the ST sibling's
    test_file_cache_attribute_importable (lines 146-167)."""
    from xarray.backends import file_manager

    assert hasattr(file_manager, "FILE_CACHE"), (
        "xarray.backends.file_manager.FILE_CACHE missing — likely an "
        "xarray version upgrade renamed the cache attribute. The MT "
        "variant of Pin DP-2 cannot assert handle-release semantics "
        "until the new attribute name is identified."
    )
    cache = file_manager.FILE_CACHE
    assert hasattr(cache, "__len__"), (
        f"xarray FILE_CACHE is {type(cache).__name__}; expected "
        "supports-len. The threshold-anchor anti-mutation pin "
        "cannot run."
    )
    assert hasattr(cache, "keys"), (
        f"xarray FILE_CACHE is {type(cache).__name__}; expected "
        "supports .keys(). The per-workspace assertion cannot run."
    )


def test_threshold_anti_mutation_pin() -> None:
    """Anti-mutation pin (per evaluator K-9 disposition + ST sibling
    structural-pin shape): _FILES_PER_THREAD must exceed xarray's
    default file_cache_maxsize so the LRU eviction path is forced.
    If a refactor lowers the constant below the threshold, the MT
    invariant test would vacuously pass under both fix and regression
    variants. This pin fails at refactor time, not at Gate B."""
    assert _FILES_PER_THREAD > _CACHE_MAXSIZE_DEFAULT, (
        f"_FILES_PER_THREAD={_FILES_PER_THREAD} must exceed xarray "
        f"file_cache_maxsize={_CACHE_MAXSIZE_DEFAULT} to force LRU "
        "eviction during the concurrent test. Production xarray "
        "2024.11.0 default per AM-A2 verified probe; verify at "
        "Gate B's K-6 production-version re-check."
    )
    # Also verify against live xarray's current default (catches a
    # future xarray default change that we forgot to mirror here).
    live_default = xr.get_options()["file_cache_maxsize"]
    assert _FILES_PER_THREAD > live_default, (
        f"_FILES_PER_THREAD={_FILES_PER_THREAD} must exceed live "
        f"xarray file_cache_maxsize={live_default}. Either bump the "
        "constant or investigate the xarray default change."
    )


def test_no_pre_existing_thread_workspace_in_cache(tmp_path: Path) -> None:
    """Anti-vacuous guard for the per-workspace tmp-key diff assertion
    below. Before the conversion loop runs, no FILE_CACHE key should
    reference either thread's workspace (the fixture writes via
    netCDF4.Dataset, NOT via xarray). Mirrors ST sibling's
    test_no_pre_existing_tmp_path_in_cache (lines 249-263) extended to
    cover both per-thread workspaces."""
    thread_roots = [tmp_path / f"thread_{i}" for i in range(2)]
    for root in thread_roots:
        _seed_thread_workspace(root, n_files=2)
    for root in thread_roots:
        assert _cache_keys_referencing(root) == [], (
            f"{root.name}-referencing keys exist in FILE_CACHE before "
            "the conversion loop runs; the per-workspace snapshot "
            "logic assumes a clean slate. Either the fixture is "
            "opening via xarray (it should be using netCDF4.Dataset "
            "directly) or tmp_path is being recycled across tests."
        )


# ── Main multi-threaded invariant ───────────────────────────────────


def test_concurrent_conversion_no_per_workspace_handle_leak(
    tmp_path: Path,
) -> None:
    """Pin DP-2 MT variant PRIMARY: two threads run
    convert_AgERA5_netcdf_to_geotiff on disjoint synthetic workspaces
    sharing process-global FILE_CACHE. Per AC-DP-1a `with` block, each
    iteration's HDF5 handle releases synchronously, so per-workspace
    cache-key snapshots show ZERO new keys referencing either workspace.

    Under WITHOUT-AC-DP-1a regression: each thread leaks handles into
    FILE_CACHE; the per-workspace diff surfaces the exact leaked paths
    (not vulnerable to LRU-saturation vacuous-pass that raw
    len(FILE_CACHE) suffers per codex BL-1 + builder SHOULD-FIX-1).
    """
    thread_roots = [tmp_path / f"thread_{i}" for i in range(2)]
    for root in thread_roots:
        _seed_thread_workspace(root, _FILES_PER_THREAD)

    # Per-workspace pre-call cache-key snapshots (should be empty
    # post-fixture per anti-vacuous guard above; recorded so the diff
    # isolates handles leaked by the conversion loops).
    before_per_workspace = {
        root.name: _cache_keys_referencing(root) for root in thread_roots
    }

    errors: list[BaseException] = []
    thread_ids_observed: list[int] = []
    barrier = threading.Barrier(2)

    def worker(workspace: Path) -> None:
        try:
            thread_ids_observed.append(threading.get_ident())
            barrier.wait(timeout=30)  # sync entry to maximize concurrency
            convert_AgERA5_netcdf_to_geotiff(
                area=None,
                selected_area=_SYNTHETIC_SELECTED_AREA,
                variables=[_SYNTHETIC_VARIABLE],
                query=datetime.date(_SYNTHETIC_YEAR, 1, 1),
                save_path=str(workspace),
            )
        except BaseException as exc:  # noqa: BLE001 — cross-thread visibility
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(root,), name=f"agera5-mt-{i}")
        for i, root in enumerate(thread_roots)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)
    for t in threads:
        assert not t.is_alive(), f"Thread {t.name} did not finish in 180s"

    # PRIMARY: per-workspace cache-key leak detection
    leaked_per_workspace = {}
    for root in thread_roots:
        after = _cache_keys_referencing(root)
        before = before_per_workspace[root.name]
        leaked = [k for k in after if k not in before]
        if leaked:
            leaked_per_workspace[root.name] = leaked
    assert not leaked_per_workspace, (
        f"xarray FILE_CACHE retains handles referencing "
        f"{len(leaked_per_workspace)} thread workspace(s) after "
        "concurrent conversion. AC-DP-1a `with` block must be "
        "missing or broken under concurrent execution. Leaks:\n"
        + "\n".join(
            f"  {name}: {len(keys)} keys; sample: {keys[0]!r}"
            for name, keys in leaked_per_workspace.items()
        )
    )

    # SECONDARY: no exception from either thread
    assert errors == [], (
        f"Concurrent conversion raised {len(errors)} exception(s): "
        f"{[type(e).__name__ + ': ' + str(e) for e in errors]}"
    )

    # TERTIARY: both threads actually ran (no early-exit / barrier timeout)
    assert len(set(thread_ids_observed)) == 2, (
        f"Expected 2 distinct thread IDs, got {thread_ids_observed}"
    )

    # QUATERNARY: all expected .tif outputs landed per workspace
    for root in thread_roots:
        tif_dir = (
            root
            / "2_conversion"
            / f"AgERA5_{_SYNTHETIC_SELECTED_AREA}"
            / _SYNTHETIC_VARIABLE_PAIR
        )
        tif_count = len(list(tif_dir.glob("*.tif")))
        assert tif_count == _FILES_PER_THREAD, (
            f"{root.name}: expected {_FILES_PER_THREAD} .tif files, "
            f"got {tif_count}"
        )
