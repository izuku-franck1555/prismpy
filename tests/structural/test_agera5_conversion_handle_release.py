"""F-DP Pin DP-2 — source-coupled handle-release assertion.

Pin DP-1 is an AST walker — it catches REGRESSION introduced via
removing the `with` block from `get_AgERA5_data.py:273`. Pin DP-2
catches the RUNTIME signal that motivated the `with` block in the
first place: after the real ``convert_AgERA5_netcdf_to_geotiff``
runs over a batch of synthetic netCDF files, the **xarray process-
wide FILE_CACHE MUST be empty** (every Dataset closed per iteration
via the AC-DP-1a `with` block).

This pin invokes the REAL vendor function — not a parallel
implementation. If the source drifts (e.g., someone removes the
`with` block, or wraps in a try/except that swallows the close
exception, or rebinds nc_file_content to a global), the FILE_CACHE
assertion fires because handles accumulate.

## FILE_CACHE attribute

xarray 2024.11.0 exposes the process-wide LRU file manager cache at
``xarray.backends.file_manager.FILE_CACHE`` per
`xarray/backends/file_manager.py:17 + :141`. The cache is an
``LRUCache`` keyed by file path; entries are added when xarray opens
a file and evicted when the cache reaches ``file_cache_maxsize`` or
when the holder explicitly closes the dataset. With AC-DP-1a's `with`
block, each iteration's open + close is a paired transaction →
FILE_CACHE returns to its pre-call size after the conversion loop
completes.

## Cross-platform FD probe (Pin DP-2 secondary signal)

Per F-DP cycle-2 builder GAP-2: ``len(os.listdir('/proc/self/fd'))``
is Linux-only. The skipif gate keeps the FD test cross-platform-
collectable: it runs on Linux (CI) and skips on macOS (dev hosts).
The primary FILE_CACHE assertion is cross-platform and is the
structural signal per codex Q1 + NICE-1.

## Synthetic dataset

The fixture builds 10 minimal netCDF4 files at the path layout that
the converter's `extraction_path = save_path/1_extraction/AgERA5_
{selected_area}/{year}/{variable[0]}_{variable[1]}/` formula
demands. Variable tuple `('2m_temperature', '24_hour_minimum')`
concatenates to `2m_temperature_24_hour_minimum` per the §Y.2
cycle-3 corrected single-tuple convention (builder cycle-4
AMBIGUITY-1 resolution).

CF time units (per VEN-7 + §Y.2 correction): each file's `time`
variable carries `units = 'days since 1900-01-01 00:00:00'` so the
converter's `pd.to_datetime(time.values[0])` resolves to distinct
calendar days; without CF units, all 10 timesteps would collapse to
the same date in the output filename and the 10-tif output check
would false-fail through overwrites.

Per F-DP contract LOCKED cycle-4 §Z.1 + §X.2 + §Y.2 corrected
fixture + builder GAP-2 cross-platform gating + AMBIGUITY-1
single-tuple convention.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import netCDF4
import numpy as np
import pytest

# Per codex BL-3: actual cache attribute in xarray 2024.11.0 lives at
# xarray.backends.file_manager.FILE_CACHE (NOT CachingFileManager._cache
# as the cycle-1 contract drafted). Verified at
# xarray/backends/file_manager.py:17 + :141 in the installed wheel.
from xarray.backends.file_manager import FILE_CACHE


# ── Synthetic fixture ──────────────────────────────────────────────


# §Y.2 cycle-3 + AMBIGUITY-1 cycle-4 resolution: variable tuple
# uses the single-tuple convention. variable[0]='2m_temperature',
# variable[1]='24_hour_minimum' → variable_pair (what the converter
# concatenates into the extraction_path subdir) ==
# '2m_temperature_24_hour_minimum'.
SYNTHETIC_VARIABLE: tuple = ("2m_temperature", "24_hour_minimum")
SYNTHETIC_VARIABLE_PAIR = (
    SYNTHETIC_VARIABLE[0] + "_" + SYNTHETIC_VARIABLE[1]
)
SYNTHETIC_SELECTED_AREA = "synth"
SYNTHETIC_YEAR = 2022
SYNTHETIC_N_FILES = 10


def _build_synthetic_netcdf_dataset(
    tmp_path: Path,
    n_files: int = SYNTHETIC_N_FILES,
) -> Path:
    """Construct ``n_files`` minimal AgERA5-shaped netCDF files at the
    extraction layout the converter expects. Returns the
    ``extraction_path`` directory."""
    extraction_path = (
        tmp_path
        / "1_extraction"
        / f"AgERA5_{SYNTHETIC_SELECTED_AREA}"
        / str(SYNTHETIC_YEAR)
        / SYNTHETIC_VARIABLE_PAIR
    )
    extraction_path.mkdir(parents=True, exist_ok=True)

    for i in range(n_files):
        nc_path = (
            extraction_path
            / f"AgERA5_{SYNTHETIC_SELECTED_AREA}_test_{i + 1:02d}.nc"
        )
        with netCDF4.Dataset(str(nc_path), "w", format="NETCDF4") as ds:
            ds.createDimension("lat", 2)
            ds.createDimension("lon", 2)
            ds.createDimension("time", 1)
            lat = ds.createVariable("lat", "f8", ("lat",))
            lon = ds.createVariable("lon", "f8", ("lon",))
            time = ds.createVariable("time", "f8", ("time",))
            # F-DP cycle-3 §Y.2 fix: CF time units enable
            # pd.to_datetime() to resolve each timestep to a unique
            # calendar day per VEN-7 converter requirement. Without
            # 'units' + 'calendar' attrs, every timestep collapses to
            # the same date in the output filename → outputs
            # overwrite → 10-file assertion false-fails.
            time.setncattr("units", "days since 1900-01-01 00:00:00")
            time.setncattr("calendar", "standard")
            tmin = ds.createVariable(
                "Temperature_Air_2m_Min_24h",
                "f4",
                ("time", "lat", "lon"),
            )
            tmin.setncattr("units", "K")
            lat[:] = [0.0, 0.1]
            lon[:] = [0.0, 0.1]
            # Day-offset i since 1900-01-01 → distinct calendar day per file
            time[:] = [float(44000 + i)]
            tmin[:] = np.full((1, 2, 2), 273.15 + i, dtype="f4")

    return extraction_path


# ── Anti-vacuous guards ────────────────────────────────────────────


def test_file_cache_attribute_importable() -> None:
    """The cycle-2 BL-3 fix relies on
    ``xarray.backends.file_manager.FILE_CACHE`` being the public-ish
    attribute the production xarray uses for handle caching. If a
    future xarray upgrade renames the attribute, this fail-fast
    assertion makes the regression obvious (instead of the main
    cache-empty test silently passing against a stale import)."""
    from xarray.backends import file_manager

    assert hasattr(file_manager, "FILE_CACHE"), (
        "xarray.backends.file_manager.FILE_CACHE missing — likely an "
        "xarray version upgrade renamed the cache attribute. Pin DP-2 "
        "cannot assert handle-release semantics until the new "
        "attribute name is identified."
    )
    cache = file_manager.FILE_CACHE
    assert hasattr(cache, "__len__"), (
        f"xarray FILE_CACHE is {type(cache).__name__}; expected "
        "supports-len. The size assertion in the main test "
        "cannot run."
    )


# ── Main invariant ─────────────────────────────────────────────────


def _cache_keys_referencing(path: Path) -> list:
    """Return every FILE_CACHE key whose serialised form contains
    ``str(path)``. xarray's LRU keys are nested tuples/lists where
    index 1 is a ``(file_path,)`` tuple; rather than depend on that
    internal shape, we serialise each key with ``repr`` and substring-
    match the test's tmp_path. Stable across xarray versions.
    """
    target = str(path)
    return [k for k in FILE_CACHE.keys() if target in repr(k)]


def test_convert_agera5_releases_handles_per_iteration(
    tmp_path: Path,
) -> None:
    """Pin DP-2 primary (cross-platform): after the real
    ``convert_AgERA5_netcdf_to_geotiff`` runs over 10 synthetic
    netCDF inputs, the xarray process-wide ``FILE_CACHE`` MUST NOT
    contain any handle that references this test's ``tmp_path``
    (every Dataset closed per iteration via the AC-DP-1a `with`
    block, popped from the LRU before the iteration ends).

    The assertion is scoped to keys referencing ``tmp_path``
    (cycle-1.5 SF-1 codex absorption) rather than a global
    ``len(FILE_CACHE)`` comparison. The latter can false-pass when a
    prior unrelated test has filled xarray's LRU and a mutated
    no-`with` conversion happens to evict OLD entries while
    inserting NEW ones — the global size stays the same but a leak
    is present. Scoping by tmp_path avoids that class.

    Anti-mutation: removing the `with` block from
    ``get_AgERA5_data.py:273`` leaves 5-6 ``tmp_path``-referencing
    entries in the cache (one per file the GC hasn't reclaimed
    yet); the assertion fires citing each leaked path.
    """
    from prismpy.vendor.sarra_data_download import get_AgERA5_data

    _build_synthetic_netcdf_dataset(tmp_path, n_files=SYNTHETIC_N_FILES)

    # Snapshot pre-call cache keys that already reference tmp_path
    # (should be empty; the fixture's own writes don't go through
    # xarray). Recorded so the diff isolates handles leaked by the
    # conversion loop, not by the fixture.
    before_tmp_keys = _cache_keys_referencing(tmp_path)

    get_AgERA5_data.convert_AgERA5_netcdf_to_geotiff(
        area=None,
        selected_area=SYNTHETIC_SELECTED_AREA,
        variables=[SYNTHETIC_VARIABLE],
        query=date(SYNTHETIC_YEAR, 1, 1),
        save_path=str(tmp_path),
    )

    after_tmp_keys = _cache_keys_referencing(tmp_path)
    leaked = [k for k in after_tmp_keys if k not in before_tmp_keys]
    assert not leaked, (
        f"xarray FILE_CACHE retains {len(leaked)} handle(s) "
        f"referencing {str(tmp_path)!r} after the conversion loop. "
        "The `with` block in convert_AgERA5_netcdf_to_geotiff must "
        "be missing or broken (AC-DP-1a regression). Leaked keys:\n"
        + "\n".join(f"  {k!r}" for k in leaked[:5])
    )

    # Output verification — 10 .tif files at the converter's output path
    conversion_path = (
        tmp_path
        / "2_conversion"
        / f"AgERA5_{SYNTHETIC_SELECTED_AREA}"
        / SYNTHETIC_VARIABLE_PAIR
    )
    output_tifs = sorted(conversion_path.glob("*.tif"))
    assert len(output_tifs) == SYNTHETIC_N_FILES, (
        f"Expected {SYNTHETIC_N_FILES} output .tif files at "
        f"{conversion_path!s}; got {len(output_tifs)}. "
        f"Files: {[p.name for p in output_tifs]}"
    )


def test_no_pre_existing_tmp_path_in_cache(tmp_path: Path) -> None:
    """Anti-vacuous probe for the tmp_path-scoped assertion above.
    Before the conversion loop runs, no FILE_CACHE key should
    reference this test's ``tmp_path`` (the fixture writes via
    ``netCDF4.Dataset`` directly, NOT via xarray). If a prior test
    has somehow seeded keys referencing this path, the snapshot
    logic would mis-classify them as leaks."""
    _build_synthetic_netcdf_dataset(tmp_path, n_files=2)
    assert _cache_keys_referencing(tmp_path) == [], (
        "tmp_path-referencing keys exist in FILE_CACHE before the "
        "conversion loop runs; the cycle-1.5 SF-1 snapshot logic "
        "assumes a clean slate. Either the fixture is opening via "
        "xarray (it should be using netCDF4.Dataset directly) or "
        "tmp_path is being recycled across tests."
    )


# ── Cross-platform FD probe (Linux-only; skipped on macOS) ─────────


@pytest.mark.skipif(
    not os.path.exists("/proc/self/fd"),
    reason=(
        "/proc/self/fd is Linux-only; FD probe skipped on macOS dev "
        "hosts per F-DP cycle-2 builder GAP-2. The cross-platform "
        "FILE_CACHE assertion in "
        "test_convert_agera5_releases_handles_per_iteration is the "
        "primary structural signal."
    ),
)
def test_convert_agera5_no_fd_leak(tmp_path: Path) -> None:
    """Pin DP-2 secondary (Linux-only): file-descriptor count must
    not grow by more than 5 across the conversion loop. The cap
    accommodates rasterio output handles + log + GDAL ephemera; a
    per-iteration leak would push delta well above 10. Skip on
    non-Linux (no `/proc`); the primary cross-platform signal lives
    in ``test_convert_agera5_releases_handles_per_iteration``."""
    from prismpy.vendor.sarra_data_download import get_AgERA5_data

    _build_synthetic_netcdf_dataset(tmp_path, n_files=SYNTHETIC_N_FILES)

    fds_before = len(os.listdir("/proc/self/fd"))

    get_AgERA5_data.convert_AgERA5_netcdf_to_geotiff(
        area=None,
        selected_area=SYNTHETIC_SELECTED_AREA,
        variables=[SYNTHETIC_VARIABLE],
        query=date(SYNTHETIC_YEAR, 1, 1),
        save_path=str(tmp_path),
    )

    fds_after = len(os.listdir("/proc/self/fd"))
    fd_delta = fds_after - fds_before
    assert fd_delta <= 5, (
        f"File-descriptor delta {fd_delta} (> 5) across "
        f"{SYNTHETIC_N_FILES}-file conversion loop — handle leak "
        "suspected. Cross-check with FILE_CACHE assertion in "
        "test_convert_agera5_releases_handles_per_iteration."
    )
