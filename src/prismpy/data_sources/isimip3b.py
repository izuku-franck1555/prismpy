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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# ISIMIP3b primary core ensemble: 5 GCMs.
_PRIMARY_GCMS = frozenset(
    {"gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"}
)

# Scenario → ISIMIP "product" mapping. ssp585 is shipped under the
# primary "InputData" product; ssp245 under "SecondaryInputData". This
# is a publication convention of the ISIMIP3b release, not a free
# parameter — passing the wrong product silently returns an empty
# dataset list.
_SCENARIO_PRODUCT_MAP: Dict[str, str] = {
    "ssp585": "InputData",
    "ssp245": "SecondaryInputData",
}

# Variable allowlist. Sprint G ships these six CF-1.x daily variables
# from the ISIMIP3b primary core ensemble. Adding a seventh requires
# CC-G-6 dim 7 (variable_units) to gain a row + the AC-G-7 conversion
# table to gain a row.
_SUPPORTED_VARIABLES = frozenset(
    {"rsds", "tasmax", "tasmin", "pr", "hurs", "sfcWind"}
)


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
        bbox: List[float],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Submit a server-side bbox cutout job and return its handle.

        The upstream ``cutout_bbox`` accepts a list of dataset paths
        plus a 4-element bbox in ``[south, north, west, east]`` order
        and submits an asynchronous job. Callers wait via
        ``isimip_client.client.ISIMIPClient.get_job(...)``; the cached
        cutout helper threads the wait + download + cache write
        together.
        """
        return self._inner.cutout_bbox(paths, bbox, **kwargs)


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
        "simulation_round": "ISIMIP3b",
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

    results: List[Dict[str, Any]] = list(response.get("results") or [])
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
    return results[0]


# ── Cached cutout (skeleton — full body lands in AC-G-2) ─────────────


def cached_cutout(
    client: ISIMIP3bClient,
    dataset: Dict[str, Any],
    bbox: Dict[str, float],
    *,
    cache_dir: Optional[Path] = None,
    ttl_days: Optional[int] = None,
) -> Path:
    """Persist an ISIMIP3b bbox cutout to a local cache (AC-G-2).

    The full implementation arrives in the AC-G-2 commit alongside the
    12-dimension version pin in :mod:`prismpy.standards.isimip_versions`
    and the cache adversarial mutation drills (F-G-10). For AC-G-1 the
    function is declared so callers + tests can import it; the body
    raises ``NotImplementedError`` until AC-G-2 lands.

    Args:
        client: ISIMIP3bClient instance.
        dataset: Result of :func:`discover_datasets`.
        bbox: Dict with ``south`` / ``north`` / ``west`` / ``east``
            keys (degrees, WGS84). Float jitter at the 5th decimal is
            normalized at the cache key layer so cold/warm decisions
            are stable.
        cache_dir: Override for the cache root. Default resolves from
            ``$PRISMPY_ISIMIP_CACHE_DIR`` and falls back to
            ``~/.cache/prismpy/isimip3b/``.
        ttl_days: Override for the TTL. Default resolves from
            ``$PRISMPY_ISIMIP_CACHE_TTL_DAYS`` and falls back to 7.

    Returns:
        Path to the cached netCDF file on disk.
    """
    raise NotImplementedError(
        "cached_cutout is implemented in the Sprint G AC-G-2 commit. "
        "Returning a placeholder path here would make the silent-skip "
        "class regress (feedback_no_data_cooking.md) — fail loud "
        "instead."
    )


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
