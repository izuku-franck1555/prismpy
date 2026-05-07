"""ISIMIP3b bias-adjusted daily climate data client.

Wraps the public ISIMIP3b data API (https://data.isimip.org/) for the
scenario-package generator. Every call goes through the upstream
``isimip-client`` library (https://github.com/ISI-MIP/isimip-client),
which the prismpy ``pyproject.toml`` declares as a required dependency.
No API key is required; the data is publicly available under the
ISIMIP Terms of Use (https://www.isimip.org/about/terms-of-use/) which
require attribution and forbid commercial redistribution of derived
products without separate agreement. Consumers of prismpy-generated
ISIMIP3b-derived packages inherit those terms and should cite the
underlying GCM, the ISIMIP3b bias-correction protocol (Lange et al.,
2021; ISIMIP3BASD v2.5.0), and the W5E5 reference dataset.

Public API:

* :class:`ISIMIP3bClient` — thin wrapper around
  :class:`isimip_client.client.ISIMIPClient` so prismpy code talks to
  a stable internal facade.
* :func:`discover_datasets` — resolve a (gcm × scenario × variable ×
  time_slice) tuple to the right ISIMIP product (``InputData`` for
  ``ssp585``, ``SecondaryInputData`` for ``ssp245``).
* :func:`cached_cutout` — TTL/version/DOI-aware bbox cutout that
  persists netCDF responses to a local cache so subsequent runs read
  from disk instead of hitting the API.

Typed exception hierarchy (all rooted at the :class:`Exception` family
so ``except Exception`` still catches them but the specific paths can
be discriminated):

* :class:`IsimipFetchError` — server-side or network-side fetch
  failure (HTTP non-200, timeout, connection reset).
* :class:`IsimipDatasetNotFoundError` — the requested
  (gcm × scenario × variable × time_slice) tuple has no matching
  dataset on ISIMIP. Carries a ``specifiers`` field so callers can
  surface the missing tuple unambiguously.
* :class:`InvalidIsimipResponseError` — ISIMIP returned 200 but the
  response body is malformed netCDF / unreadable / wrong shape.
* :class:`CacheDirectoryError` — cache directory cannot be created /
  read / written (permission denied, missing parent, etc.).
* :class:`CacheWriteError` — atomic-write staging-rename-meta sequence
  failed mid-flight (disk full, IO error, race).

The cache adversarial mutation drills (network failure mid-fetch /
disk full mid-write / permission denied / concurrent race / malformed
netCDF) calibrate that each path raises the correct typed exception.
Drill discipline: the drills MUST invoke real adversarial conditions
(monkey-patch ``os.makedirs`` to raise PermissionError; fill a tmpfs
to disk-full; close a socket mid-download; serve truncated netCDF
bytes) — NEVER ``mock.side_effect = CacheWriteError()`` style mock-raise.
A drill that just raises the typed exception directly does not exercise
the underlying failure path the exception is supposed to wrap.
"""

from __future__ import annotations

import json
import os
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from filelock import FileLock, Timeout

from prismpy.sources._cache_base import (
    DOWNLOAD_LOCK_TIMEOUT_SECONDS,
    TMPFILE_PREFIX,
    cache_lock_path,
    cleanup_orphan_tmpfiles,
    write_atomic_json,
)

# Import-time guard: missing isimip-client must fail loud, never silent
# skip. The library is declared required in pyproject.toml; absence of
# the package indicates a broken install (e.g., a venv that bypassed
# pip install -e .) and the scenario-package generator cannot run.
# This mirrors the cdsapi/SARRA_data_download / pygadm fail-loud
# discipline the F-AL train codified.
try:
    from isimip_client.client import ISIMIPClient
except ImportError as exc:  # pragma: no cover - exercised by structural test
    raise ImportError(
        "isimip-client is required to run the ISIMIP3b scenario-package "
        "generator. Install it with `pip install isimip-client>=2.0,<3.0` "
        "or, if you ran `pip install -e .` from the prismpy repo, "
        "verify the install completed without errors. The dependency is "
        "declared in prismpy/pyproject.toml [project.dependencies]."
    ) from exc

# Canonical 12-dimension version pin lives in
# ``prismpy.standards.isimip_versions``. Importing the GCM and variable
# rosters from there keeps this module from going out of sync with the
# pin module under future ISIMIP point releases.
from prismpy.standards.isimip_versions import (
    PRIMARY_GCMS as _PRIMARY_GCMS,
    SCENARIO_PRODUCT_MAP as _SCENARIO_PRODUCT_MAP,
    SIMULATION_ROUND as _SIMULATION_ROUND,
    SUPPORTED_VARIABLES as _SUPPORTED_VARIABLES,
)


# ── Typed exception hierarchy ────────────────────────────────────────


class IsimipFetchError(Exception):
    """Server-side or network-side fetch failure.

    Wraps HTTP non-200 responses, timeouts, connection resets, and any
    other network-level failure that prevents a successful download.
    Cache discipline: when this fires mid-download, the cache directory
    must be left clean — no partial ``.nc`` / ``.tmp`` lingering as
    readable.
    """


class IsimipDatasetNotFoundError(IsimipFetchError):
    """The (gcm × scenario × variable × time_slice) tuple has no match.

    Carries the ``specifiers`` field so the calling layer can surface
    the missing tuple in error messages without re-deriving it.
    """

    def __init__(self, message: str, specifiers: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.specifiers: Dict[str, Any] = dict(specifiers or {})


class InvalidIsimipResponseError(IsimipFetchError):
    """ISIMIP returned 200 but the response body is malformed.

    Examples: truncated netCDF, wrong magic header, unreadable shape,
    non-CF-compliant attributes that prismpy's xarray parser rejects.
    """


class CacheDirectoryError(Exception):
    """Cache directory cannot be created / read / written.

    Permission denied, missing parent that we cannot create, read-only
    filesystem mount, etc. Surfaces early before any download begins.
    """


class CacheWriteError(CacheDirectoryError):
    """Atomic-write staging → rename → meta sequence failed mid-flight.

    Disk full, IO error, race against another writer. Cache discipline:
    when this fires, partial state must be cleaned (no half-written
    ``.nc`` left readable) so the next reader gets a cold-cache miss
    and re-fetches cleanly.
    """


# ── Product / scenario resolution ────────────────────────────────────
#
# ``_PRIMARY_GCMS``, ``_SCENARIO_PRODUCT_MAP``, ``_SUPPORTED_VARIABLES``
# are imported above from ``prismpy.standards.isimip_versions``. They
# are the authoritative source for the 5-GCM × 6-variable × 2-scenario
# enumeration; this module does NOT redefine them.


def _product_for_scenario(scenario: str) -> str:
    """Return the ISIMIP product name for a given scenario."""
    try:
        return _SCENARIO_PRODUCT_MAP[scenario]
    except KeyError as exc:
        raise IsimipDatasetNotFoundError(
            f"Unsupported ISIMIP3b scenario: {scenario!r}. "
            f"Sprint G supports {sorted(_SCENARIO_PRODUCT_MAP)}.",
            specifiers={"scenario": scenario},
        ) from exc


# ── ISIMIP3bClient: thin facade over isimip_client.ISIMIPClient ──────


class ISIMIP3bClient:
    """Thin facade over :class:`isimip_client.client.ISIMIPClient`.

    Exposes only the methods the scenario-package generator needs and
    decorates them with typed exceptions so callers can discriminate
    failure modes without inspecting upstream HTTP machinery.
    """

    def __init__(self, *, data_url: Optional[str] = None) -> None:
        kwargs: Dict[str, Any] = {}
        if data_url is not None:
            kwargs["data_url"] = data_url
        self._inner = ISIMIPClient(**kwargs)

    # The three methods the generator actually calls; everything else
    # the upstream client exposes is intentionally unwrapped — the
    # facade is a deliberate narrowing.

    def datasets(self, **kwargs: Any) -> Dict[str, Any]:
        """List datasets matching the given query terms.

        Wraps :meth:`isimip_client.client.ISIMIPClient.datasets`. The
        response shape is the upstream's pass-through dict so callers
        that already accept that shape do not need to translate.
        """
        return self._inner.datasets(**kwargs)

    def dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Fetch a single dataset by id."""
        return self._inner.dataset(dataset_id)

    def cutout_bbox(
        self,
        paths: List[str],
        *,
        west: float,
        east: float,
        south: float,
        north: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Submit a server-side bbox cutout job and return its handle.

        The upstream ``cutout_bbox`` signature is
        ``cutout_bbox(paths, west, east, south, north, ...)`` — four
        SEPARATE float arguments in the canonical W/E/S/N order, NOT a
        single bbox list. The wrapper enforces keyword-only edges so
        callers cannot accidentally swap the order. Callers wait via
        ``isimip_client.client.ISIMIPClient.get_job(...)``; the cached
        cutout helper threads the wait + download + cache write
        together.
        """
        return self._inner.cutout_bbox(paths, west, east, south, north, **kwargs)


# ── Dataset discovery ────────────────────────────────────────────────


def discover_datasets(
    client: ISIMIP3bClient,
    *,
    gcm: str,
    scenario: str,
    variable: str,
    time_slice: Tuple[int, int],
) -> Dict[str, Any]:
    """Resolve a single (gcm × scenario × variable × time_slice) tuple.

    Returns the upstream dataset dict (with at minimum ``id``,
    ``version``, ``doi``, and ``files`` fields). The caller passes the
    result into :func:`cached_cutout` to fetch a bbox-trimmed netCDF.

    Raises:
        IsimipDatasetNotFoundError: if the tuple has no matching
            dataset on ISIMIP. Carries the failing specifiers in the
            ``specifiers`` attribute so the audit trail is unambiguous.
        IsimipFetchError: if the discovery API call itself fails
            (network error, HTTP non-200).

    The product (``InputData`` for ssp585, ``SecondaryInputData`` for
    ssp245) is derived inside this function so the calling layer never
    has to track that mapping by hand.
    """
    if gcm not in _PRIMARY_GCMS:
        raise IsimipDatasetNotFoundError(
            f"Unsupported GCM {gcm!r}. Sprint G primary core ensemble: "
            f"{sorted(_PRIMARY_GCMS)}.",
            specifiers={"gcm": gcm},
        )
    if variable not in _SUPPORTED_VARIABLES:
        raise IsimipDatasetNotFoundError(
            f"Unsupported variable {variable!r}. Sprint G allowlist: "
            f"{sorted(_SUPPORTED_VARIABLES)}.",
            specifiers={"variable": variable},
        )
    if time_slice[0] > time_slice[1]:
        raise IsimipDatasetNotFoundError(
            f"Invalid time_slice {time_slice!r} (start > end).",
            specifiers={"time_slice": time_slice},
        )

    product = _product_for_scenario(scenario)

    query: Dict[str, Any] = {
        "simulation_round": _SIMULATION_ROUND,
        "product": product,
        "climate_forcing": gcm,
        "climate_scenario": scenario,
        "climate_variable": variable,
    }

    try:
        response = client.datasets(**query)
    except Exception as exc:  # noqa: BLE001 — wrap upstream into typed
        raise IsimipFetchError(
            f"ISIMIP discovery call failed for {query!r}: {exc!r}"
        ) from exc

    # Default ``isimip_client.ISIMIPClient.list()`` returns the raw
    # ``results`` list, NOT the paginated dict. Pagination wraps the
    # list as ``{"results": [...], "next": ..., "previous": ...}``
    # only when the caller passes ``paginate=True``. Handle both shapes
    # so the discovery helper does not regress depending on whether
    # callers ever flip the pagination flag.
    if isinstance(response, list):
        results: List[Dict[str, Any]] = list(response)
    elif isinstance(response, dict):
        results = list(response.get("results") or [])
    else:
        raise IsimipFetchError(
            "Unexpected ISIMIP datasets() response shape "
            f"{type(response).__name__!r}; expected list or dict."
        )
    if not results:
        raise IsimipDatasetNotFoundError(
            f"No ISIMIP3b dataset matched {query!r} for time_slice "
            f"{time_slice!r}.",
            specifiers={**query, "time_slice": time_slice},
        )

    # ISIMIP3b commonly publishes multiple matching dataset entries
    # spanning different time periods; the caller filters down to the
    # requested ``time_slice`` using the dataset's ``files`` metadata.
    # The discovery surface returns the first match for now; AC-G-2's
    # ``cached_cutout`` is responsible for the time-slice filtering on
    # the dataset's per-file timestamps.
    #
    # Consumer contract (per durable lesson #24 canonical-source-or-pin
    # + F-AV regression net): the returned dict carries the upstream
    # API shape verbatim — ``product`` / ``climate_forcing`` /
    # ``climate_scenario`` / ``climate_variable`` live nested under
    # ``specifiers`` in the live API and at the top level in older
    # responses. Consumers MUST route extraction through the
    # ``_*_from_dataset`` helper family (or :func:`cached_cutout`,
    # which uses them); reading ``dataset["climate_scenario"]``
    # directly silently reintroduces F-AV against live data. A future
    # boundary-flatten redesign (see Sprint H+ ISIMIP3b client
    # redesign) is the canonical-source alternative; until that lands
    # the helpers are the only sanctioned consumer path.
    return results[0]


# ── Cached cutout (AC-G-2) ───────────────────────────────────────────


# Default TTL for cached netCDF cutouts. Cache invalidation per CC-G-4:
# stale on TTL expiry OR dataset version mismatch OR DOI mismatch OR
# missing meta. Caller may override via the ``$PRISMPY_ISIMIP_CACHE_TTL_DAYS``
# env var or the ``ttl_days`` keyword argument.
_DEFAULT_TTL_DAYS = 7
_TTL_ENV_VAR = "PRISMPY_ISIMIP_CACHE_TTL_DAYS"
_CACHE_DIR_ENV_VAR = "PRISMPY_ISIMIP_CACHE_DIR"
_DEFAULT_CACHE_DIR_LITERAL = "~/.cache/prismpy/isimip3b"

# bbox-key 4-decimal precision per AC-G-2 §2.4. Float jitter below this
# rounding must NOT fragment the cache: queries that differ by 5e-6 in
# any edge resolve to the same cache key.
_BBOX_KEY_DECIMALS = 4

# Cutout-job poll interval and total timeout. The upstream ISIMIP API
# is server-cutout-async: submit, poll, download. The poll interval is
# kept short enough to feel responsive for small bboxes (~30s typical
# job duration for a 1×1° cutout) and the total timeout is wide enough
# to absorb large bboxes plus rare server contention.
_CUTOUT_POLL_INTERVAL_SECONDS = 5.0
_CUTOUT_TOTAL_TIMEOUT_SECONDS = 60 * 30  # 30 minutes
_DOWNLOAD_CHUNK_BYTES = 64 * 1024


def _resolve_cache_dir(explicit: Optional[Path]) -> Path:
    """Pick the cache root — explicit argument wins; env var second;
    user-cache default last."""
    if explicit is not None:
        return Path(explicit).expanduser()
    env_value = os.environ.get(_CACHE_DIR_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return Path(_DEFAULT_CACHE_DIR_LITERAL).expanduser()


def _resolve_ttl_days(explicit: Optional[int]) -> int:
    """Pick the TTL — explicit argument wins; env var second; default last."""
    if explicit is not None:
        return int(explicit)
    env_value = os.environ.get(_TTL_ENV_VAR)
    if env_value:
        try:
            return int(env_value)
        except ValueError as exc:
            raise CacheDirectoryError(
                f"Invalid {_TTL_ENV_VAR} value {env_value!r}: must be integer days"
            ) from exc
    return _DEFAULT_TTL_DAYS


def _bbox_key(bbox: Dict[str, float]) -> str:
    """Compose the canonical 4-decimal-rounded bbox string per AC-G-2 §2.4.

    Format: ``S{south:+.4f}_N{north:+.4f}_W{west:+.4f}_E{east:+.4f}``.
    Float jitter at the 5th decimal does NOT fragment the cache.
    """
    try:
        south = float(bbox["south"])
        north = float(bbox["north"])
        west = float(bbox["west"])
        east = float(bbox["east"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheDirectoryError(
            f"bbox must include south/north/west/east as floats; got {bbox!r}"
        ) from exc
    fmt = f"%+.{_BBOX_KEY_DECIMALS}f"
    return (
        f"S{fmt % south}_N{fmt % north}_W{fmt % west}_E{fmt % east}"
    )


def _cache_paths(
    cache_root: Path,
    *,
    product: str,
    scenario: str,
    gcm: str,
    variable: str,
    bbox: Dict[str, float],
) -> Tuple[Path, Path]:
    """Compute the (nc_path, meta_path) pair for a given query."""
    bk = _bbox_key(bbox)
    nc = (
        cache_root
        / "ISIMIP3b"
        / product
        / scenario
        / gcm
        / variable
        / f"{bk}.nc"
    )
    meta = nc.with_suffix(".meta.json")
    return nc, meta


def _read_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    """Load a cache meta sidecar; return ``None`` on missing / unreadable.

    Missing meta is treated as cold-cache (re-fetch), NOT crash, per
    AC-G-2 §2.10.
    """
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_cache_fresh(
    nc_path: Path,
    meta_path: Path,
    *,
    expected_version: Any,
    expected_doi: Any,
    ttl_seconds: float,
) -> bool:
    """Return True iff the cache entry is usable as-is.

    Triggers re-fetch when:
    * netCDF file is missing
    * meta sidecar is missing or unreadable (treated as cold; AC-G-2 §2.10)
    * file mtime is older than TTL
    * meta.version disagrees with the live dataset's version
    * meta.dataset_doi disagrees with the live dataset's DOI
    """
    if not nc_path.exists():
        return False
    meta = _read_meta(meta_path)
    if meta is None:
        return False

    age_seconds = time.time() - nc_path.stat().st_mtime
    if age_seconds > ttl_seconds:
        return False

    if meta.get("version") != expected_version:
        return False
    if meta.get("dataset_doi") != expected_doi:
        return False
    return True


def _dataset_specifiers(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Return the dataset's ``specifiers`` sub-dict, or ``{}`` if absent.

    The live ISIMIP3b API nests the (product / climate_forcing /
    climate_scenario / climate_variable) fields under a ``specifiers``
    sub-dict (e.g., ``dataset["specifiers"]["climate_scenario"] =
    "ssp585"``). Older response shapes — and the synthetic dicts the
    internal harness constructs — carry the same fields at the top
    level. Each accessor below checks both layers so the helper family
    is shape-tolerant.
    """
    nested = dataset.get("specifiers")
    if isinstance(nested, dict):
        return nested
    return {}


def _scenario_from_dataset(dataset: Dict[str, Any]) -> str:
    """Extract the climate scenario from the upstream dataset dict.

    Uses ``climate_scenario`` (the ISIMIP API field) with a fallback to
    ``scenario`` for older response shapes. Each name is tried at the
    top level first, then under ``specifiers`` to cover the live API's
    nested response shape.
    """
    specs = _dataset_specifiers(dataset)
    return str(
        dataset.get("climate_scenario")
        or specs.get("climate_scenario")
        or dataset.get("scenario")
        or specs.get("scenario")
        or ""
    )


def _gcm_from_dataset(dataset: Dict[str, Any]) -> str:
    specs = _dataset_specifiers(dataset)
    return str(
        dataset.get("climate_forcing")
        or specs.get("climate_forcing")
        or dataset.get("gcm")
        or specs.get("gcm")
        or ""
    )


def _variable_from_dataset(dataset: Dict[str, Any]) -> str:
    specs = _dataset_specifiers(dataset)
    return str(
        dataset.get("climate_variable")
        or specs.get("climate_variable")
        or dataset.get("variable")
        or specs.get("variable")
        or ""
    )


def _product_from_dataset(dataset: Dict[str, Any]) -> str:
    specs = _dataset_specifiers(dataset)
    return str(
        dataset.get("product")
        or specs.get("product")
        or ""
    )


def _dataset_paths(dataset: Dict[str, Any]) -> List[str]:
    """Extract dataset path identifiers for upstream cutout submission.

    Returns the explicit ``path``/``paths`` field if present; otherwise
    derives from the dataset's ``id``. The upstream cutout_bbox helper
    accepts a list of dataset paths.
    """
    if "paths" in dataset and dataset["paths"]:
        return list(dataset["paths"])
    if "path" in dataset and dataset["path"]:
        return [str(dataset["path"])]
    if dataset.get("id"):
        return [str(dataset["id"])]
    raise IsimipDatasetNotFoundError(
        f"Dataset has no path / paths / id field for cutout submission: {dataset!r}",
        specifiers={"dataset_keys": sorted(dataset.keys())},
    )


def _submit_and_wait_cutout_job(
    client: ISIMIP3bClient,
    *,
    dataset: Dict[str, Any],
    bbox: Dict[str, float],
    poll_interval_seconds: float = _CUTOUT_POLL_INTERVAL_SECONDS,
    total_timeout_seconds: float = _CUTOUT_TOTAL_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Submit a server-side cutout job and block until it finishes.

    Wraps every upstream failure in :class:`IsimipFetchError`. Returns
    the finished job dict; the caller pulls ``file_url`` (or equivalent)
    out of it for the actual file download.
    """
    paths = _dataset_paths(dataset)

    try:
        # Upstream ``cutout_bbox`` takes 4 separate W/E/S/N floats; the
        # wrapper enforces keyword-only edges so caller-side ordering
        # cannot drift silently.
        job = client.cutout_bbox(
            paths,
            west=float(bbox["west"]),
            east=float(bbox["east"]),
            south=float(bbox["south"]),
            north=float(bbox["north"]),
        )
    except Exception as exc:  # noqa: BLE001 — wrap upstream
        raise IsimipFetchError(
            f"ISIMIP cutout_bbox submission failed: {exc!r}"
        ) from exc

    deadline = time.monotonic() + total_timeout_seconds
    while True:
        status = str(job.get("status") or "").lower()
        if status in {"finished", "complete", "completed", "success"}:
            return job
        if status in {"failed", "error", "cancelled"}:
            raise IsimipFetchError(
                f"ISIMIP cutout job ended in {status!r}: {job!r}"
            )
        if time.monotonic() >= deadline:
            raise IsimipFetchError(
                f"ISIMIP cutout job exceeded {total_timeout_seconds}s; last status={status!r}"
            )
        time.sleep(poll_interval_seconds)
        # Re-poll. If the upstream wraps polling for us, the job dict
        # holds a refreshed status; otherwise the upstream client
        # exposes ``get_job(job_url)`` which we re-invoke.
        try:
            job_url = job.get("job_url") or job.get("url") or job.get("id")
            refreshed = client._inner.get_job(job_url) if job_url else job  # type: ignore[attr-defined]
            if isinstance(refreshed, dict):
                job = refreshed
        except Exception as exc:  # noqa: BLE001
            raise IsimipFetchError(
                f"ISIMIP cutout job poll failed: {exc!r}"
            ) from exc


def _download_to_staging(
    file_url: str,
    staging_path: Path,
    *,
    chunk_bytes: int = _DOWNLOAD_CHUNK_BYTES,
) -> None:
    """Stream the cutout result to a staging file.

    Network failures surface as :class:`IsimipFetchError`. The staging
    path is a sibling of the final cache path so a subsequent
    ``os.replace`` is atomic on the same filesystem.
    """
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(file_url, stream=True, timeout=120)
        response.raise_for_status()
        with staging_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_bytes):
                if chunk:
                    fh.write(chunk)
    except requests.exceptions.RequestException as exc:
        # Best-effort cleanup of any partial bytes — half-written cache
        # entries must not be readable per AC-G-2 §2.11.
        try:
            staging_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise IsimipFetchError(
            f"ISIMIP cutout download from {file_url!r} failed: {exc!r}"
        ) from exc
    except OSError as exc:
        # Disk-full / IO error / permission denied during the staging
        # write loop. The ``requests`` exception path above only covers
        # network-side failures; the file system can also fail during
        # ``fh.write(chunk)`` and that has to surface as the typed
        # ``CacheWriteError`` per the AC-G-1 typed exception contract.
        try:
            staging_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise CacheWriteError(
            f"Cache write to staging path {staging_path} failed: {exc!r}"
        ) from exc


def _unwrap_zip_cutout_if_needed(staging_path: Path) -> None:
    """Replace ``staging_path`` with the inner NetCDF when the body is ZIP.

    The live ISIMIP3b cutout API wraps the result in a ZIP archive
    (first 4 bytes ``PK\\x03\\x04``). Upstream ``isimip_client``'s
    ``download(extract=True)`` helper unzips for the caller, but
    :func:`cached_cutout` uses its own streaming download path so the
    unwrap has to live here. After this returns, ``staging_path``
    carries the extracted NetCDF body — :func:`_validate_netcdf_magic`
    runs against the inner file, not the ZIP container.

    A ZIP body that contains zero or more than one NetCDF entry, or
    that has the ZIP magic prefix but is not a readable archive,
    surfaces as :class:`InvalidIsimipResponseError` so the caller can
    discriminate this from a network failure.
    """
    try:
        with staging_path.open("rb") as fh:
            magic = fh.read(4)
    except OSError as exc:
        raise InvalidIsimipResponseError(
            f"Cannot read staging file {staging_path}: {exc!r}"
        ) from exc
    if magic != b"PK\x03\x04":
        return  # Not a ZIP body — leave for the NetCDF magic check.

    try:
        with zipfile.ZipFile(staging_path, "r") as zf:
            members = [
                name
                for name in zf.namelist()
                if name.lower().endswith((".nc", ".nc4"))
            ]
            if not members:
                raise InvalidIsimipResponseError(
                    f"ISIMIP cutout response at {staging_path} is a ZIP "
                    f"but contains no NetCDF entries (members: "
                    f"{zf.namelist()!r})."
                )
            if len(members) > 1:
                raise InvalidIsimipResponseError(
                    f"ISIMIP cutout response at {staging_path} is a ZIP "
                    f"with multiple NetCDF entries; expected exactly 1: "
                    f"{members!r}."
                )
            extracted = zf.read(members[0])
    except zipfile.BadZipFile as exc:
        raise InvalidIsimipResponseError(
            f"ISIMIP cutout response at {staging_path} has ZIP magic "
            f"but is not a readable ZIP archive: {exc!r}"
        ) from exc

    # Atomic-replace the staging file with the inner NetCDF so the
    # downstream ``os.replace(staging, cache)`` step lands a NetCDF in
    # the cache, not a ZIP. Per AC-G-2 §2.11 the staging-rename-meta
    # discipline must hold; an unwrap-write failure mid-flight is a
    # cache-write error, not an upstream invalid-response error.
    #
    # The tmp file uses the canonical ``.writing-*.tmp`` naming pattern
    # so a SIGKILL between ``write_bytes`` and ``os.replace`` leaves an
    # orphan that the next caller's ``cleanup_orphan_tmpfiles`` sweep
    # removes — the cleanup glob in ``_cache_base.py`` matches this
    # exact prefix.
    tmp_path = staging_path.with_name(
        f"{TMPFILE_PREFIX}unwrap-{uuid.uuid4().hex}.tmp"
    )
    try:
        tmp_path.write_bytes(extracted)
        os.replace(str(tmp_path), str(staging_path))
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise CacheWriteError(
            f"Cannot write unzipped NetCDF over staging path "
            f"{staging_path}: {exc!r}"
        ) from exc


def _validate_netcdf_magic(staging_path: Path) -> None:
    """Reject malformed netCDF responses early.

    A 200 OK with body that is not a netCDF file (truncated bytes,
    HTML error page, JSON error payload) surfaces as
    :class:`InvalidIsimipResponseError`. The check inspects the first
    bytes for one of the documented netCDF magic numbers.
    """
    try:
        with staging_path.open("rb") as fh:
            head = fh.read(8)
    except OSError as exc:
        raise InvalidIsimipResponseError(
            f"Cannot read staging file {staging_path}: {exc!r}"
        ) from exc
    # netCDF classic / 64-bit-offset / cdf-5: ``CDF\x01`` / ``CDF\x02`` /
    # ``CDF\x05``. NetCDF-4 / HDF5: ``\x89HDF``.
    if head[:3] == b"CDF" and head[3:4] in (b"\x01", b"\x02", b"\x05"):
        return
    if head[:4] == b"\x89HDF":
        return
    raise InvalidIsimipResponseError(
        f"Response body at {staging_path} is not a recognised netCDF "
        f"format (first 8 bytes: {head!r})."
    )


def _file_url_from_job(job: Dict[str, Any]) -> str:
    """Extract the download URL from a finished cutout job dict."""
    for key in ("file_url", "download_url", "url"):
        url = job.get(key)
        if isinstance(url, str) and url:
            return url
    raise IsimipFetchError(
        f"Finished cutout job has no file URL field: keys={sorted(job.keys())}"
    )


def cached_cutout(
    client: ISIMIP3bClient,
    dataset: Dict[str, Any],
    bbox: Dict[str, float],
    *,
    cache_dir: Optional[Path] = None,
    ttl_days: Optional[int] = None,
) -> Path:
    """Persist an ISIMIP3b bbox cutout to a local cache.

    Cache discipline per AC-G-2 + CC-G-4:

    * Cache root: ``$PRISMPY_ISIMIP_CACHE_DIR`` env var; default
      ``~/.cache/prismpy/isimip3b/``.
    * TTL: ``$PRISMPY_ISIMIP_CACHE_TTL_DAYS`` env var; default 7 days.
    * Cache key: ``<root>/ISIMIP3b/<product>/<scenario>/<gcm>/<variable>/<bbox_key>.nc``
      where ``bbox_key`` is the 4-decimal-rounded edge string. Float
      jitter at the 5th decimal does NOT fragment the cache.
    * Sidecar ``.meta.json`` records ``version`` + ``dataset_doi`` +
      ``fetch_time``; missing meta is treated as cold-cache (re-fetch),
      NOT a crash.
    * Invalidation triggers: file mtime > TTL, ``meta.version !=
      dataset['version']``, ``meta.dataset_doi != dataset['doi']``.
    * Atomic-write discipline: download to staging path → rename into
      final cache path → write meta. Every failure path raises a typed
      exception (``IsimipFetchError`` / ``InvalidIsimipResponseError``
      / ``CacheDirectoryError`` / ``CacheWriteError``).

    Args:
        client: ISIMIP3bClient instance.
        dataset: Result of :func:`discover_datasets`. Must carry
            ``id`` (or ``path``/``paths``), ``version``, ``doi``.
        bbox: Dict with ``south`` / ``north`` / ``west`` / ``east``
            keys (degrees, WGS84).
        cache_dir: Override for the cache root.
        ttl_days: Override for the TTL.

    Returns:
        Path to the cached netCDF file on disk.
    """
    cache_root = _resolve_cache_dir(cache_dir)
    ttl_seconds = _resolve_ttl_days(ttl_days) * 86400

    product = _product_from_dataset(dataset)
    scenario = _scenario_from_dataset(dataset)
    gcm = _gcm_from_dataset(dataset)
    variable = _variable_from_dataset(dataset)
    if not all([product, scenario, gcm, variable]):
        raise IsimipDatasetNotFoundError(
            "Dataset is missing required fields "
            "(product / climate_scenario / climate_forcing / climate_variable). "
            f"Dataset keys: {sorted(dataset.keys())}.",
            specifiers={"dataset_keys": sorted(dataset.keys())},
        )

    expected_version = dataset.get("version")
    expected_doi = dataset.get("doi")

    nc_path, meta_path = _cache_paths(
        cache_root,
        product=product,
        scenario=scenario,
        gcm=gcm,
        variable=variable,
        bbox=bbox,
    )

    # Best-effort cache root creation; surface as typed exception so the
    # caller can discriminate this from a download failure.
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CacheDirectoryError(
            f"Cannot create cache root {cache_root}: {exc!r}"
        ) from exc

    # Per-source-per-bbox-key lock. Concurrent callers serialize on the
    # same key; different keys overlap freely. Lock key includes
    # product/scenario/gcm/variable/bbox so two queries that differ in
    # any of those fields don't block each other.
    lock_key = f"{product}-{scenario}-{gcm}-{variable}-{_bbox_key(bbox)}"
    lock_path = cache_lock_path(cache_root, source="isimip3b", key=lock_key)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path))

    try:
        with lock.acquire(timeout=DOWNLOAD_LOCK_TIMEOUT_SECONDS):
            # Re-check fresh-cache inside the lock so a thread that
            # waited on an earlier writer's download skips its own.
            if _is_cache_fresh(
                nc_path,
                meta_path,
                expected_version=expected_version,
                expected_doi=expected_doi,
                ttl_seconds=ttl_seconds,
            ):
                return nc_path

            # Cache miss — fetch a fresh copy.
            nc_path.parent.mkdir(parents=True, exist_ok=True)
            staging_dir = nc_path.parent
            cleanup_orphan_tmpfiles(staging_dir)

            job = _submit_and_wait_cutout_job(
                client, dataset=dataset, bbox=bbox
            )
            file_url = _file_url_from_job(job)

            staging_path = staging_dir / f"{nc_path.name}.partial"
            try:
                _download_to_staging(file_url, staging_path)
                # Live ISIMIP cutout responses arrive ZIP-wrapped; the
                # unwrap is a no-op when the body is already a plain
                # NetCDF (synthetic test fixtures + older API shapes).
                _unwrap_zip_cutout_if_needed(staging_path)
                _validate_netcdf_magic(staging_path)
                # Atomic rename. ``os.replace`` is atomic on the same
                # filesystem; staging is a sibling of nc_path so the
                # rename does not cross filesystems.
                try:
                    os.replace(str(staging_path), str(nc_path))
                except OSError as exc:
                    raise CacheWriteError(
                        f"Cannot move staging cutout to cache path "
                        f"{nc_path}: {exc!r}"
                    ) from exc

                # Meta sidecar last so a missing meta on the next read
                # is the only signal of a half-written entry. Per
                # AC-G-2 §2.10 missing meta = cold-cache, not crash.
                meta_payload = {
                    "version": expected_version,
                    "dataset_doi": expected_doi,
                    "fetch_time": time.time(),
                    "product": product,
                    "scenario": scenario,
                    "gcm": gcm,
                    "variable": variable,
                    "bbox": {
                        "south": float(bbox["south"]),
                        "north": float(bbox["north"]),
                        "west": float(bbox["west"]),
                        "east": float(bbox["east"]),
                    },
                }
                try:
                    write_atomic_json(meta_path, meta_payload)
                except OSError as exc:
                    # Half-state recovery: drop the .nc so the next
                    # reader sees no readable cache and re-fetches
                    # cleanly. Per AC-G-2 §2.13 cache-discipline.
                    try:
                        nc_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise CacheWriteError(
                        f"Cannot write meta sidecar {meta_path}: {exc!r}"
                    ) from exc
            finally:
                # Best-effort cleanup of any leftover staging fragment
                # whatever the path through the try block.
                try:
                    if staging_path.exists():
                        staging_path.unlink()
                except OSError:
                    pass
    except Timeout as exc:
        raise IsimipFetchError(
            f"Cache lock acquisition timed out after "
            f"{DOWNLOAD_LOCK_TIMEOUT_SECONDS}s on {lock_path}"
        ) from exc

    return nc_path


__all__ = [
    "IsimipFetchError",
    "IsimipDatasetNotFoundError",
    "InvalidIsimipResponseError",
    "CacheDirectoryError",
    "CacheWriteError",
    "ISIMIP3bClient",
    "discover_datasets",
    "cached_cutout",
]
