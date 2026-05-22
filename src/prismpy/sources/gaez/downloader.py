"""
GAEZ (Global Agro-Ecological Zones) data downloader.

Fetches GAEZ v4 crop-suitability / potential-yield rasters from FAO's
production Esri Image Service (see ``esri_client.py`` for the canonical
host URL) for ACEA's downstream raster reader. The deprecated FAO dev S3
bucket is restricted upstream (Sorghum + the Intermediate input level
return 403 universally; remaining crops partially restricted); the Esri
ImageServer publishes the same res02 HIST baseline data under an
attribute-filter schema that this module translates from ACEA's
downloader vocabulary.

Reference:
  FAO GAEZ Data Portal — https://gaez.fao.org/
  Esri ImageServer REST API — /server/rest/services/res02/ImageServer

Module layout (per durable §6.4):
  - ``errors.py``         : Typed exceptions + GAEZFetchSummary
  - ``esri_schemas.py``   : Pydantic /exportImage request + error envelope
  - ``raster_mapping.py`` : Vocabulary translator + EsriQuerySpec
  - ``esri_client.py``    : Raw REST wrapper (SOLE owner of host string)
  - ``downloader.py``     : This module — thin orchestrator over the above

Public surface preserved (per AC-F-CP-1):
  - Module constants: ``GAEZ_CROP_CODES``, ``GAEZ_CULTIVAR_MAP``,
    ``GAEZ_VARIABLES``, ``INPUT_LEVELS``
  - ``GAEZDownloader``: ``__init__``, ``get_cultivars_for_crop``,
    ``_download_with_retry``, ``download_cultivar``, ``download_for_crop``,
    ``copy_to_output``, ``check_cached``
  - ``max_workers`` added as additive kwarg to ``__init__`` with
    ``default=10`` (per Stage C empirical baseline); back-compat preserved
    so ``GAEZDownloader()`` and ``GAEZDownloader(retries=5)`` still work.

Fail-loud propagation (per AC-F-CP-10 + F-AG-class status precedent):
  - Per-raster fetch failures surface as ``EsriFetchError`` from the client.
  - ``download_cultivar`` accumulates failures into ``GAEZFetchSummary`` and
    raises ``GAEZDownloadError`` if ``failed > 0`` (no silent ``return {}``).
  - The ACEA translator catches the typed exception and downgrades the
    ``TranslationResult`` to ``success=False``, which propagates to
    ``PipelineRun.status='error'``.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from prismpy.sources.climate._cancel import PipelineCancelled, raise_if_cancelled
from prismpy.sources.gaez.errors import (
    EsriFetchError,
    GAEZDownloadError,
    GAEZFetchSummary,
    UnknownCombination,
)
from prismpy.sources.gaez.esri_client import EsriImageServiceClient
from prismpy.sources.gaez.raster_mapping import resolve_esri_query

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level public constants — preserved for backward compatibility.
# ---------------------------------------------------------------------------

# GAEZ cultivar name to 4-letter code mapping. Kept as the canonical roster
# of cultivar identifiers ACEA's downstream filename / install-script
# consumers depend on. The dual-form aliasing (space + underscore) per
# builder DELTA-CA-3 mirrors the historical filename normalization at
# ``cultivar.replace(' ', '_')``.
GAEZ_CROP_CODES = {
    # Maize cultivars
    'Highland maize': 'hmze',
    'Lowland maize': 'lmze',
    'Temperate maize': 'tmze',
    'Maize': 'maiz',
    'Highland_maize': 'hmze',
    'Lowland_maize': 'lmze',
    'Temperate_maize': 'tmze',
    # Wheat cultivars
    'Spring wheat': 'swhe',
    'Winter wheat': 'wwhe',
    'Spring_wheat': 'swhe',
    'Winter_wheat': 'wwhe',
    # Rice cultivars (wetland = paddy, dryland = upland)
    'Wetland rice': 'ricw',
    'Dryland rice': 'ricd',
    'Wetland_rice': 'ricw',
    'Dryland_rice': 'ricd',
    # Sorghum cultivars
    'Highland sorghum': 'hsor',
    'Lowland sorghum': 'lsor',
    'Highland_sorghum': 'hsor',
    'Lowland_sorghum': 'lsor',
    # Millet
    'Pearl millet': 'pmlt',
    'Foxtail millet': 'fmlt',
    'Pearl_millet': 'pmlt',
    'Foxtail_millet': 'fmlt',
}


# Crop name -> GAEZ cultivar list. Canonical source for the dimension; the
# AST-walker structural pin ``test_no_inline_cultivar_map_redefinition``
# enforces that consumers import this constant rather than redefining the
# mapping locally (per durable §24 + WA CA-2).
GAEZ_CULTIVAR_MAP = {
    'Maize': ['Highland_maize', 'Lowland_maize', 'Temperate_maize', 'Maize'],
    'Corn': ['Highland_maize', 'Lowland_maize', 'Temperate_maize', 'Maize'],
    'Wheat': ['Spring_wheat', 'Winter_wheat'],
    'Rice': ['Wetland_rice', 'Dryland_rice'],
    'Sorghum': ['Highland_sorghum', 'Lowland_sorghum'],
    'Millet': ['Pearl_millet', 'Foxtail_millet'],
}


# GAEZ variable codes — ACEA's expected `{var}/` subdirectory roster.
GAEZ_VARIABLES = {
    'LUT': 'idx',              # Suitability LUT sequence number
    'PotentialYield': 'yld',   # Potential yield (kg/ha)
    'F3': 'fc3',               # Constraint factor
}


# Input levels — ACEA's default is High + Low (Intermediate is not
# carried in the res02 HIST baseline service).
INPUT_LEVELS = {
    'High': '8110H',
    'Intermediate': '8110I',
    'Low': '8110L',
}


# Per-path lock cache for atomic-write contention guard (AC-F-CP-4).
# Concurrent workers fetching the same ``output_path`` use the per-path
# threading.Lock to serialize writes. The lock cache itself is guarded by
# its own meta-lock so two threads racing on cache initialization for the
# same path don't get split locks. In practice the fan-out is per unique
# (cultivar, water, level, var) so same-path collision is structurally
# unlikely, but the lock is the defensive belt that turns a flaky probability
# into a regression test we can assert.
_PATH_LOCKS: Dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    """Return the per-path lock; lazily construct under the meta-lock."""
    key = str(path)
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[key] = lock
        return lock


class GAEZDownloader:
    """Thin orchestrator over the Esri ImageService client.

    Fetches per-cultivar rasters with retry/backoff, writes them atomically
    into the per-variable cache layout ACEA reads, and surfaces failures
    honestly via typed exceptions.
    """

    DEFAULT_CACHE_DIR = Path.home() / ".prismpy" / "cache" / "gaez"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        retries: int = 3,
        backoff: float = 1.5,
        max_workers: int = 10,
        client: Optional[EsriImageServiceClient] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries
        self.backoff = backoff
        self.max_workers = max_workers
        # Client is dependency-injectable for test stubs; defaults to a
        # production-config instance so the existing API surface stays
        # zero-argument-callable.
        self.client = client or EsriImageServiceClient(
            retries=retries, backoff=backoff
        )

    # ------------------------------------------------------------------
    # Public helpers — preserved surface.
    # ------------------------------------------------------------------

    def get_cultivars_for_crop(self, crop_name: str) -> List[str]:
        """Get GAEZ cultivar names for a crop."""
        return GAEZ_CULTIVAR_MAP.get(crop_name, [crop_name])

    def _compute_cache_path(
        self,
        cultivar: str,
        water_supply: str,
        input_level: str,
        gaez_var: str,
    ) -> Path:
        """Canonical cache-path constructor per AC-F-CP-4.

        Filename format: ``{var}/{var}_{cultivar_fname}_{water}_{level}_1981_2010.tif``
        where ``cultivar_fname = cultivar.replace(' ', '_')``.

        Preserved from the S3-era contract so old cached files remain valid.
        """
        cultivar_fname = cultivar.replace(' ', '_')
        filename = (
            f"{gaez_var}_{cultivar_fname}_{water_supply}_{input_level}_1981_2010.tif"
        )
        return self.cache_dir / gaez_var / filename

    # ------------------------------------------------------------------
    # Atomic write helper — temp file + os.replace under per-path lock.
    # ------------------------------------------------------------------

    def _atomic_write(self, output_path: Path, payload: bytes) -> None:
        """Write ``payload`` to ``output_path`` atomically per AC-F-CP-4.

        Strategy: write to ``output_path.with_suffix('.tif.tmp')``, fsync,
        then ``os.replace(tmp, output_path)``. Under the per-path lock so a
        concurrent writer's tmp file can't collide with ours. The replace
        is the atomicity primitive: POSIX guarantees the rename is atomic
        on the same filesystem so an interrupted process either leaves the
        prior payload intact OR the new payload — never a partial-write
        garbage stream.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')

        lock = _path_lock(output_path)
        with lock:
            try:
                with open(tmp_path, 'wb') as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, output_path)
            except Exception:
                # Clean up tmp on failure so a future retry has a clean slate.
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:  # pragma: no cover - best-effort cleanup
                    pass
                raise

    # ------------------------------------------------------------------
    # Legacy retry helper — kept for back-compat (callers outside the
    # download_cultivar fan-out path may still invoke it). The new path
    # uses EsriImageServiceClient.fetch_image directly.
    # ------------------------------------------------------------------

    def _download_with_retry(
        self,
        url: str,
        output_path: Path,
        overwrite: bool = False,
    ) -> Tuple[bool, str]:
        """Legacy file-by-URL retry helper.

        Retained for back-compat with any caller that still uses the URL
        path (no production callers in prismpy reach this anymore — the
        Esri migration routes through ``EsriImageServiceClient.fetch_image``
        instead). Kept as a thin wrapper around ``requests`` so the public
        surface remains stable and the honest-signal HTTP status surfacing
        (AC-F-CP-6) is testable without rebuilding the Esri client mock.

        Returns:
            Tuple of (success, message).
        """
        try:
            import requests
        except ImportError as e:
            raise ModuleNotFoundError(
                "requests is required for GAEZ downloads but did not "
                "import. The package is declared in pyproject.toml "
                "[project] dependencies; reinstall prismpy with "
                "`pip install -e .` to refresh the venv."
            ) from e

        if output_path.exists() and not overwrite:
            logger.debug(f"Using cached: {output_path}")
            return True, "cached"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        attempt = 0
        wait = 1.0

        while attempt <= self.retries:
            msg = "unknown"
            try:
                response = requests.get(url.strip(), timeout=30, stream=True)
                response.raise_for_status()

                # Buffer + atomic write so a mid-stream failure doesn't
                # leave a half-written file at ``output_path``.
                buf = bytearray()
                for chunk in response.iter_content(chunk_size=8192):
                    buf.extend(chunk)
                self._atomic_write(output_path, bytes(buf))
                logger.info(
                    f"Downloaded {output_path.name} ({len(buf) / 1024:.0f} KB)"
                )
                return True, f"{len(buf)} bytes"

            except requests.exceptions.HTTPError as e:
                # Per codex M3 + builder CA-1: use ``is not None`` for the
                # response-presence check; the empty-response truthy check
                # is a defensive belt against a rare-but-real failure mode.
                if e.response is not None:
                    status = e.response.status_code
                    msg = f"HTTP {status}"
                else:
                    msg = "HTTP unknown (no response)"
            except requests.exceptions.Timeout:
                msg = "TIMEOUT"
            except requests.exceptions.ConnectionError:
                msg = "CONNECTION_ERROR"
            except requests.exceptions.RequestException as e:
                msg = f"REQUEST_ERROR: {e}"
            except Exception as e:  # pragma: no cover
                msg = f"ERROR: {e}"

            attempt += 1
            if attempt > self.retries:
                logger.error(
                    f"Failed to download {url}: {msg} (after {self.retries} retries)"
                )
                return False, msg

            logger.warning(
                f"Retry {attempt}/{self.retries} for {output_path.name} ({msg})"
            )
            time.sleep(wait)
            wait *= self.backoff

        return False, "max retries exceeded"

    # ------------------------------------------------------------------
    # Per-cultivar fetch fan-out — the Esri migration core.
    # ------------------------------------------------------------------

    def _fetch_one(
        self,
        cultivar: str,
        water_supply: str,
        input_level: str,
        acea_var: str,
        cache_path: Path,
        overwrite: bool,
        cancel_check: Optional[Callable[[], bool]] = None,
        progress_callback: Any = None,
    ) -> Tuple[str, Path]:
        """Fetch a single raster, write atomically, return the cache path.

        Raises ``EsriFetchError`` (or ``UnknownCombination``) on failure;
        the caller catches and accumulates into ``GAEZFetchSummary``.

        ``cancel_check`` + ``progress_callback`` thread through to
        ``fetch_image`` (the innermost retry surface) for cooperative
        cancellation + producer-side retry-attempt progress (Ship 1' 5-level
        cancel-wire / PRI-6).
        """
        if cache_path.exists() and not overwrite:
            logger.debug(f"GAEZ cache hit: {cache_path}")
            return f"{acea_var}_{cultivar}_{water_supply}_{input_level}", cache_path

        query = resolve_esri_query(cultivar, water_supply, input_level, acea_var)
        start = time.monotonic()
        payload = self.client.fetch_image(
            query,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        self._atomic_write(cache_path, payload)
        logger.info(
            f"Esri fetch ok ({elapsed_ms:.0f}ms, {len(payload) / 1024:.0f} KB): "
            f"{cache_path.name}"
        )
        return f"{acea_var}_{cultivar}_{water_supply}_{input_level}", cache_path

    def download_cultivar(
        self,
        cultivar: str,
        water_supply: str = 'irr',
        input_levels: Optional[List[str]] = None,
        overwrite: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
        progress_callback: Any = None,
    ) -> Dict[str, Path]:
        """Download every GAEZ variable for one cultivar.

        Raises ``GAEZDownloadError`` if any raster fails — the F-AG-class
        status propagation contract requires fail-loud on partial failure.

        ``cancel_check`` is observed before fan-out, per single-threaded
        raster, and (via ``fetch_image``) inside each worker; on cancel the
        ThreadPoolExecutor is shut down with ``cancel_futures=True`` so
        queued rasters never touch the FAO service (TAMSAT precedent).
        """
        if input_levels is None:
            input_levels = ['High', 'Low']

        raise_if_cancelled(cancel_check, f"gaez.download_cultivar.start={cultivar}")

        crop_code = GAEZ_CROP_CODES.get(cultivar)
        if not crop_code:
            logger.warning(f"No GAEZ code for cultivar: {cultivar}")
            return {}

        # Build the per-raster work list before fan-out so the
        # ThreadPoolExecutor batch is bounded by the input-levels Cartesian
        # product (typically 2 levels × 3 vars = 6 rasters per cultivar).
        tasks: List[Tuple[str, str, str, Path]] = []
        for input_level in input_levels:
            if input_level not in INPUT_LEVELS:
                logger.warning(
                    f"Skipping unknown input_level={input_level!r} (not in {list(INPUT_LEVELS.keys())})"
                )
                continue
            for acea_var in GAEZ_VARIABLES.keys():
                cache_path = self._compute_cache_path(
                    cultivar, water_supply, input_level, acea_var
                )
                tasks.append((input_level, acea_var, water_supply, cache_path))

        downloaded: Dict[str, Path] = {}
        errors: List[EsriFetchError] = []

        # Single-threaded path when max_workers <= 1 (eases drill / debugging).
        if self.max_workers <= 1:
            for input_level, acea_var, ws, cache_path in tasks:
                # Per-raster cancel check covers the cache-hit sweep too,
                # where fetch_image (and its per-attempt check) never runs.
                raise_if_cancelled(
                    cancel_check, f"gaez.download_cultivar.loop={cultivar}"
                )
                try:
                    key, path = self._fetch_one(
                        cultivar, ws, input_level, acea_var, cache_path, overwrite,
                        cancel_check=cancel_check,
                        progress_callback=progress_callback,
                    )
                    downloaded[key] = path
                except UnknownCombination as e:
                    errors.append(EsriFetchError(0, f"UNKNOWN_COMBINATION: {e}"))
                except EsriFetchError as e:
                    errors.append(e)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="gaez-esri",
            ) as pool:
                futures = {
                    pool.submit(
                        self._fetch_one,
                        cultivar,
                        ws,
                        input_level,
                        acea_var,
                        cache_path,
                        overwrite,
                        cancel_check=cancel_check,
                        progress_callback=progress_callback,
                    ): (input_level, acea_var)
                    for input_level, acea_var, ws, cache_path in tasks
                }
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        key, path = fut.result()
                        downloaded[key] = path
                    except PipelineCancelled:
                        # A worker observed cancel mid-retry. Stop queued
                        # rasters immediately so they never touch FAO, then
                        # propagate the cancel (carve-out before the
                        # EsriFetchError accumulation so cancel is never
                        # rewritten as a fetch error).
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise
                    except UnknownCombination as e:
                        errors.append(EsriFetchError(0, f"UNKNOWN_COMBINATION: {e}"))
                    except EsriFetchError as e:
                        errors.append(e)

        if errors:
            summary = GAEZFetchSummary(
                succeeded=len(downloaded),
                failed=len(errors),
                errors=errors,
            )
            raise GAEZDownloadError(summary, cultivar=cultivar)

        return downloaded

    # ------------------------------------------------------------------
    # Crop-level fan-out — iterates over cultivars in the GAEZ_CULTIVAR_MAP
    # entry for the crop.
    # ------------------------------------------------------------------

    def download_for_crop(
        self,
        crop_name: str,
        water_supply: str = 'irr',
        input_levels: Optional[List[str]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        progress_callback: Any = None,
    ) -> Dict[str, Path]:
        """Download GAEZ data for every cultivar of a crop.

        Re-raises ``GAEZDownloadError`` on any cultivar's failure so the
        ACEA translator can surface ``PipelineRun.status='error'``.
        ``cancel_check`` + ``progress_callback`` thread down to
        ``download_cultivar`` (Ship 1' 5-level cancel-wire).
        """
        all_downloaded: Dict[str, Path] = {}
        cultivars = self.get_cultivars_for_crop(crop_name)

        for cultivar in cultivars:
            raise_if_cancelled(cancel_check, f"gaez.download_for_crop={crop_name}")
            downloaded = self.download_cultivar(
                cultivar,
                water_supply=water_supply,
                input_levels=input_levels,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
            all_downloaded.update(downloaded)

        return all_downloaded

    def copy_to_output(
        self,
        crop_name: str,
        output_dir: Path,
        water_supplies: Optional[List[str]] = None,
        input_levels: Optional[List[str]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        progress_callback: Any = None,
    ) -> List[Path]:
        """Download GAEZ data and copy to output directory.

        Raises ``GAEZDownloadError`` if any underlying ``download_cultivar``
        fails (per F-AG-class fail-loud contract). ``cancel_check`` +
        ``progress_callback`` are the top of the Ship 1' 5-level cancel-wire
        (ACEA passes ``self.cancel_check`` here); they thread down to
        ``fetch_image``.
        """
        if water_supplies is None:
            water_supplies = ['irr', 'rf']
        if input_levels is None:
            input_levels = ['High', 'Low']

        output_files: List[Path] = []

        for ws in water_supplies:
            downloaded = self.download_for_crop(
                crop_name,
                water_supply=ws,
                input_levels=input_levels,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )

            for key, src_path in downloaded.items():
                if not src_path.exists():
                    continue

                subdir = "other"
                for var in GAEZ_VARIABLES.keys():
                    if key.startswith(var):
                        subdir = var
                        break

                dst_dir = output_dir / subdir
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                output_files.append(dst_path)
                logger.debug(f"Copied {src_path} to {dst_path}")

        return output_files

    def check_cached(self, crop_name: str) -> Dict[str, bool]:
        """Check which GAEZ files are already cached."""
        cached: Dict[str, bool] = {}
        cultivars = self.get_cultivars_for_crop(crop_name)

        for cultivar in cultivars:
            for ws in ['irr', 'rf']:
                for level in ['High', 'Low']:
                    for var in GAEZ_VARIABLES.keys():
                        cache_path = self._compute_cache_path(cultivar, ws, level, var)
                        key = f"{var}_{cultivar}_{ws}_{level}"
                        cached[key] = cache_path.exists()

        return cached
