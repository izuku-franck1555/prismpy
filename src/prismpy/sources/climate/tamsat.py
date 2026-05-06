"""
TAMSAT rainfall data source retriever.

This module provides functionality to access TAMSAT (Tropical Applications of
Meteorology using SATellite data) rainfall estimates for Africa.

TAMSAT provides daily rainfall estimates at ~4km (0.0375°) resolution,
available from 1983-present, primarily used by SARRA-Py.

Reference: SARRA-Py/02-WEATHER-PREPARATION/ implementation patterns.
"""

import glob
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from filelock import FileLock, Timeout

from prismpy.models.region import BoundingBox, Region
from prismpy.provenance.tracker import DecisionType, ProvenanceTracker
from prismpy.sources.base import DataSource, RetrievalResult
from prismpy.sources.climate._cancel import PipelineCancelled, raise_if_cancelled


logger = logging.getLogger(__name__)


# ── V2-22a B2: cache-manifest + filelock helpers ─────────────────────
# Shared by tamsat.py and agera5.py for cache-isolation correctness.
# Closes C1 (bbox-blind cache contamination) and C2 (concurrent download
# race) per V2-22a-B2-CONTRACT. agera5.py imports the public names below.

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "_manifest.json"
MARKER_FILENAME = "_manifest.writing"
TMPFILE_PREFIX = ".writing-"
# 0.01° per edge ≈ 1 km — well above sub-pixel noise on TAMSAT (0.0375°)
# and AgERA5 (0.1°) grids, matches wizard map-click precision.
BBOX_TOLERANCE_DEG = 0.01
# 7200 s (2 h) — covers worst-case 3-year SARRA-Py runs under CDS contention.
DOWNLOAD_LOCK_TIMEOUT_SECONDS = 7200

# One WARNING per cache path per process lifetime for legacy (no-manifest)
# caches — keeps server logs from drowning when many requests hit a region
# whose cache predates B2.
_legacy_warned_cache_paths: set = set()
_legacy_warned_lock = threading.Lock()


class CacheManifestState(NamedTuple):
    """Outcome of inspecting a cache directory for a valid manifest.

    Attributes:
        cache_hit: True if the caller may use the cached data as-is (still
            subject to the caller's own completeness/date-coverage check).
        force_redownload: True if the caller MUST wipe stale data files
            and re-download. Set whenever the manifest check detected a
            cold signal: bbox_mismatch, manifest_corrupt, marker_present,
            or file_count_drift.
        reason: Short tag for logs.
        prior_bbox: Prior manifest's bbox when reason='bbox_mismatch'.
        marker_started_at: Marker's started_at ISO when reason='marker_present'.
        expected_file_count: Manifest's recorded file_count when
            reason='file_count_drift' (the "expected" side of the drift).
    """
    cache_hit: bool
    force_redownload: bool
    reason: str
    prior_bbox: Optional[Dict[str, float]] = None
    marker_started_at: Optional[str] = None
    expected_file_count: Optional[int] = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bbox_to_dict(bounds: BoundingBox) -> Dict[str, float]:
    """Extract bbox in canonical {north,south,east,west} keys.

    Avoids the GIS-vs-SARRA-Py tuple-order drift risk (Rev 10 correction
    #3) by always using explicit edge names.
    """
    return {
        "north": float(bounds.maxy),
        "south": float(bounds.miny),
        "east": float(bounds.maxx),
        "west": float(bounds.minx),
    }


def _bbox_matches(
    a: Dict[str, float],
    b: Dict[str, float],
    tolerance_deg: float = BBOX_TOLERANCE_DEG,
) -> bool:
    for edge in ("north", "south", "east", "west"):
        if abs(float(a.get(edge, 0.0)) - float(b.get(edge, 0.0))) > tolerance_deg:
            return False
    return True


def count_tif_files(data_dir: Path) -> int:
    """Recursive .tif count under data_dir (0 if absent).

    Works for TAMSAT (flat) and AgERA5 (per-variable subdirs). Excludes
    .nc fragments in TAMSAT's _raw_nc/ staging area.
    """
    if not data_dir.exists():
        return 0
    return sum(1 for _ in data_dir.rglob("*.tif"))


def _cleanup_orphan_tmpfiles(target_dir: Path) -> None:
    """Best-effort removal of leftover atomic-write tempfiles.

    A SIGKILL between os.fsync and os.replace can leave a .writing-XXXX.tmp
    fragment in the target dir. Cleanup is opportunistic — failures are
    swallowed because the next manifest write will overwrite the target
    regardless.
    """
    if not target_dir.exists():
        return
    for stale in target_dir.glob(f"{TMPFILE_PREFIX}*.tmp"):
        try:
            stale.unlink()
        except OSError:
            pass


def write_marker(
    marker_path: Path,
    *,
    source: str,
    region_name: str,
    run_id: Optional[str],
) -> None:
    """Create _manifest.writing — the in-progress signal that survives SIGKILL.

    Persists if the writer process is killed mid-download. The next reader
    sees the marker and treats the cache as cold even when data files
    look complete (AC 1.7.3c). Deleted by delete_marker after a successful
    manifest replace.
    """
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": _utcnow_iso(),
        "run_id": run_id or "",
        "pid": os.getpid(),
        "source": source,
        "region_name": region_name,
    }
    marker_path.write_text(json.dumps(payload), encoding="utf-8")


def delete_marker(marker_path: Path) -> None:
    try:
        marker_path.unlink()
    except FileNotFoundError:
        pass


def write_cache_manifest(
    manifest_path: Path,
    *,
    source: str,
    region_name: str,
    bbox: Dict[str, float],
    start_date: date,
    end_date: date,
    run_id: Optional[str],
    file_count: int,
) -> None:
    """Write _manifest.json atomically; called inside the per-source lock.

    Uses tempfile in the SAME directory as the target so os.replace stays
    atomic on the same filesystem (cross-fs rename degrades to copy+unlink
    and loses atomicity). fsync on the file ensures the bytes hit disk;
    fsync on the directory ensures the rename itself is durable across a
    crash.

    Crash semantics (AC 1.7.1):
      * SIGKILL between tempfile creation and os.replace → orphan
        .writing-XXXX.tmp may persist (best-effort cleanup on writer
        re-entry). Target manifest is either absent or contains a prior
        successful manifest — never partial/corrupt.
      * SIGKILL between os.replace and dir-fsync → target is written but
        the rename may not be durable. Acceptable: next reader either sees
        the new manifest or its absence (which the marker still flags as
        cold).
    """
    target_dir = manifest_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_orphan_tmpfiles(target_dir)

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": source,
        "region_name": region_name,
        "bbox": bbox,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "created_at": _utcnow_iso(),
        "run_id": run_id or "",
        "file_count": file_count,
    }
    body = json.dumps(payload, indent=2).encode("utf-8")

    tf = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=TMPFILE_PREFIX,
        suffix=".tmp",
        dir=str(target_dir),
        delete=False,
    )
    tmp_path = Path(tf.name)
    try:
        tf.write(body)
        tf.flush()
        os.fsync(tf.fileno())
        tf.close()
        os.replace(str(tmp_path), str(manifest_path))
    except Exception:
        try:
            tf.close()
        except Exception:
            pass
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    # Crash-durable directory rename — fsync the dir entry itself.
    try:
        dir_fd = os.open(str(target_dir), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def check_cache_manifest(
    manifest_path: Path,
    marker_path: Path,
    *,
    expected_bbox: Dict[str, float],
    actual_file_count: int,
    data_files_present: bool,
) -> CacheManifestState:
    """Inspect a cache dir and return the action the caller should take.

    Evaluation order is significant: marker presence overrides everything
    because a prior writer may have crashed AFTER replacing the manifest
    but BEFORE deleting the marker (or partway through producing data).

    States:
      * marker_present       → cold, force re-download
      * no_data              → cold, no force needed
      * legacy_assume_valid  → cache hit (warn once per cache path)
      * manifest_corrupt     → cold, force re-download
      * bbox_mismatch        → cold, force re-download (prior bbox in log)
      * file_count_drift     → cold, force re-download
      * valid                → cache hit
    """
    if marker_path.exists():
        try:
            marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
            started_at = marker_data.get("started_at") if isinstance(marker_data, dict) else None
        except (OSError, json.JSONDecodeError):
            started_at = None
        return CacheManifestState(
            cache_hit=False,
            force_redownload=True,
            reason="marker_present",
            marker_started_at=started_at,
        )

    if not manifest_path.exists():
        if not data_files_present:
            return CacheManifestState(
                cache_hit=False,
                force_redownload=False,
                reason="no_data",
            )
        return CacheManifestState(
            cache_hit=True,
            force_redownload=False,
            reason="legacy_assume_valid",
        )

    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return CacheManifestState(
            cache_hit=False,
            force_redownload=True,
            reason="manifest_corrupt",
        )

    if (not isinstance(manifest, dict)
            or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION):
        return CacheManifestState(
            cache_hit=False,
            force_redownload=True,
            reason="manifest_corrupt",
        )

    prior_bbox = manifest.get("bbox") if isinstance(manifest.get("bbox"), dict) else None
    if not prior_bbox or not _bbox_matches(prior_bbox, expected_bbox):
        return CacheManifestState(
            cache_hit=False,
            force_redownload=True,
            reason="bbox_mismatch",
            prior_bbox=prior_bbox,
        )

    manifest_file_count = manifest.get("file_count")
    if isinstance(manifest_file_count, int) and manifest_file_count != actual_file_count:
        return CacheManifestState(
            cache_hit=False,
            force_redownload=True,
            reason="file_count_drift",
            expected_file_count=manifest_file_count,
        )

    return CacheManifestState(
        cache_hit=True,
        force_redownload=False,
        reason="valid",
    )


def warn_legacy_cache_once(cache_path: Path, source_logger: logging.Logger) -> None:
    """One WARNING per cache path per process lifetime."""
    try:
        key = str(cache_path.resolve())
    except OSError:
        key = str(cache_path)
    with _legacy_warned_lock:
        if key in _legacy_warned_cache_paths:
            return
        _legacy_warned_cache_paths.add(key)
    source_logger.warning(
        "Legacy cache (no manifest) at %s — assuming valid for backward "
        "compatibility. Will be replaced with a manifest after the next "
        "successful download of this region.",
        key,
    )


def bbox_field_for_log(bbox: Optional[Dict[str, float]]) -> str:
    if not bbox:
        return "<none>"
    return (
        f"N={bbox.get('north')} S={bbox.get('south')} "
        f"E={bbox.get('east')} W={bbox.get('west')}"
    )


def cache_lock_path(cache_dir: Path, source: str, region_name) -> Path:
    """Per-source, per-region lock path: .{source}-{key}.lock.

    Different sources on the same region use different lock files so
    a single SARRA-Py run can progress through TAMSAT then AgERA5
    without self-blocking, and two users hitting the same region
    overlap on different sources but serialize on the same source.

    `region_name` must be a `Region` dataclass (post-resolution).
    `cache_lock_path` is a source-internal helper for prismpy
    retrieval flows — every live caller passes a `Region`. The
    parameter keeps the `region_name` label for call-site
    symmetry; the helper delegates to
    `region_cache_key_from_region` so any non-`Region` shape
    fails loudly at `getattr` on `.boundary_source` / `.bounds`.
    """
    from prismpy.utils.sanitization import region_cache_key_from_region
    safe = region_cache_key_from_region(region_name)
    return cache_dir / f".{source}-{safe}.lock"


@dataclass
class TAMSATConfig:
    """Configuration for TAMSAT data access."""

    base_url: str = "https://www.tamsat.org.uk/"
    version: str = "v3.1"
    resolution: float = 0.0375  # ~4km
    data_dir: Optional[Path] = None  # Local data directory
    use_sarra_download: bool = True  # Try to use SARRA_data_download library
    timeout: int = 300  # Download timeout


@dataclass
class TAMSATData:
    """Container for TAMSAT rainfall data.

    Attributes:
        region_name: Name of the region
        bounds: Bounding box (GIS format)
        bounds_sarra_py: Bounding box (SARRA-Py format)
        start_date: Start of data coverage
        end_date: End of data coverage
        resolution: Spatial resolution in degrees
        data_dir: Directory containing the GeoTIFF files
        file_count: Number of daily files available
        variables: List of available variables (typically just 'rain')
    """

    region_name: str
    bounds: List[float]
    bounds_sarra_py: List[float]
    start_date: date
    end_date: date
    resolution: float
    data_dir: Path
    file_count: int
    variables: List[str]


class TAMSATSource(DataSource):
    """Data source for TAMSAT rainfall estimates.

    TAMSAT (Tropical Applications of Meteorology using SATellite data)
    provides daily rainfall estimates for Africa at ~4km resolution.

    The data can be accessed in two ways:
    1. Direct download from the JASMIN TAMSAT server (crop + GeoTIFF)
    2. Loading from pre-downloaded GeoTIFF files

    Attributes:
        NAME: Data source identifier
        EARLIEST_DATE: Earliest available date
        RESOLUTION: Native resolution in degrees
        CRS: Coordinate reference system
    """

    NAME = "tamsat"
    EARLIEST_DATE = date(1983, 1, 1)
    RESOLUTION = 0.0375  # ~4km
    CRS = "EPSG:4326"

    # File naming pattern
    FILE_PATTERN = "TAMSAT_{version}_{region}_rfe_filled_{year}_{month:02d}_{day:02d}.tif"

    def __init__(
        self,
        config: Optional[TAMSATConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ):
        """Initialize the TAMSAT data source.

        Args:
            config: TAMSAT configuration
            cache_dir: Directory for caching data
            provenance: Provenance tracker
        """
        super().__init__(cache_dir=cache_dir, provenance=provenance)
        self.config = config or TAMSATConfig()

    @property
    def sarra_download_available(self) -> bool:
        """Check if TAMSAT download capability is available.

        Always True — uses direct HTTP download to JASMIN server,
        no external library dependency.
        """
        return True

    def retrieve(
        self,
        region: Region,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        download: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
        **kwargs,
    ) -> RetrievalResult:
        """Retrieve TAMSAT rainfall data for a region.

        Args:
            region: Region with bounding box
            start_date: Start date (YYYY-MM-DD or date object)
            end_date: End date (YYYY-MM-DD or date object)
            data_dir: Directory containing/for TAMSAT data
            download: Whether to download data if not available locally
            **kwargs: Additional parameters

        Returns:
            RetrievalResult containing TAMSATData object
        """
        errors = []
        warnings = []
        run_id = kwargs.get('run_id')
        # V2-22b L F-8: ``cancel_check`` is now an explicit kwarg (see
        # method signature). Previously read from ``kwargs.get``; that
        # meant a caller typo (e.g., ``cancel_chck``) silently disabled
        # cancellation. Explicit signature makes typos a TypeError.
        metadata = {
            "source": self.NAME,
            "version": self.config.version,
            "resolution": self.config.resolution,
        }

        # Parse dates
        start_date = self._parse_date(start_date) if start_date else self.EARLIEST_DATE
        end_date = self._parse_date(end_date) if end_date else date.today() - timedelta(days=1)

        metadata["start_date"] = start_date.isoformat()
        metadata["end_date"] = end_date.isoformat()

        # Codex Path A follow-up — compute the cache key ONCE for
        # this run and thread it through every identity use
        # downstream (filenames, existence checks, validation, lock,
        # download). Using `region.name` for some and `cache_key`
        # for others produces a split brain where manual-unnamed
        # runs re-download dates that are already on disk under a
        # different filename.
        from prismpy.utils.sanitization import region_cache_key_from_region
        region_key = region_cache_key_from_region(region)

        # Determine data directory
        if data_dir:
            data_dir = Path(data_dir)
        elif self.config.data_dir:
            data_dir = self.config.data_dir
        else:
            # Default: cache_dir/tamsat/{region_key}/ — manual
            # regions key by bbox so unnamed-manual projects don't
            # collide on "Unnamed study area".
            data_dir = self.cache_dir / "tamsat" / region_key

        # Get bounds in both formats
        bounds_gis = region.bounds.to_gis_format()
        bounds_sarra_py = region.bounds.to_sarra_py_format()
        bbox_dict = bbox_to_dict(region.bounds)

        metadata["bounds_gis"] = bounds_gis
        metadata["bounds_sarra_py"] = bounds_sarra_py
        metadata["data_dir"] = str(data_dir)

        # Cache-isolation paths (V2-22a B2)
        manifest_path = data_dir / MANIFEST_FILENAME
        marker_path = data_dir / MARKER_FILENAME
        force_redownload = False
        # V2-22b L F-5 (Gate A round 1 MEDIUM 1): `state` is only
        # bound inside the `if data_dir.exists():` branch below, so
        # the legacy-warning emission downstream must guard with
        # `state is not None`. Initialize here to avoid NameError.
        state = None

        # V2-22a B2 + Gate B BLOCKER fix: the manifest check must fire
        # INDEPENDENT of file_info.complete. Otherwise a partial cache
        # from a crashed writer (e.g., 200 out of 1000 .tif files) whose
        # manifest carries the stale-bbox would fall through the "else"
        # branch of the old code with force_redownload=False, and the
        # downloader's per-date .exists() short-circuit at tamsat.py
        # ':514' would preserve the stale .tif files while a fresh
        # manifest got written over them. Silent contamination.
        #
        # New flow: always consult the manifest first. Any cold signal
        # (bbox_mismatch / manifest_corrupt / marker_present /
        # file_count_drift) sets force_redownload=True regardless of
        # completeness; the "cache hit" short-circuit only applies when
        # BOTH the manifest says valid/legacy AND file_info says complete.
        if data_dir.exists():
            file_info = self._validate_local_files(
                data_dir=data_dir,
                region_name=region_key,
                start_date=start_date,
                end_date=end_date,
            )
            actual_count = count_tif_files(data_dir)
            state = check_cache_manifest(
                manifest_path,
                marker_path,
                expected_bbox=bbox_dict,
                actual_file_count=actual_count,
                data_files_present=actual_count > 0,
            )

            # Invalidation signals fire independent of completeness —
            # this is the BLOCKER fix.
            force_redownload = state.force_redownload
            metadata["cache_state"] = state.reason

            if state.reason == "bbox_mismatch":
                self.logger.info(
                    "TAMSAT cache bbox mismatch for %s — prior=%s "
                    "requested=%s — re-downloading",
                    region.name,
                    bbox_field_for_log(state.prior_bbox),
                    bbox_field_for_log(bbox_dict),
                )
            elif state.reason == "manifest_corrupt":
                self.logger.warning(
                    "TAMSAT manifest at %s is corrupt/unreadable — "
                    "treating as cold",
                    manifest_path,
                )
            elif state.reason == "marker_present":
                self.logger.warning(
                    "TAMSAT marker present at %s (started_at=%s) — "
                    "prior download interrupted; re-downloading",
                    marker_path,
                    state.marker_started_at,
                )
            elif state.reason == "file_count_drift":
                # Gate B LOW 1: include expected + actual in the log
                self.logger.warning(
                    "TAMSAT manifest file_count drift at %s — expected=%s "
                    "(from manifest) actual=%d (disk count) — treating as "
                    "cold",
                    manifest_path,
                    state.expected_file_count,
                    actual_count,
                )

            # Cache hit only when BOTH the manifest says OK AND the
            # completeness check says we have every date we need.
            if state.cache_hit and file_info["complete"]:
                if state.reason == "legacy_assume_valid":
                    warn_legacy_cache_once(data_dir, self.logger)

                self.logger.info(
                    f"Found {file_info['file_count']} TAMSAT files for {region.name}"
                )

                tamsat_data = TAMSATData(
                    region_name=region_key,
                    bounds=bounds_gis,
                    bounds_sarra_py=bounds_sarra_py,
                    start_date=start_date,
                    end_date=end_date,
                    resolution=self.config.resolution,
                    data_dir=data_dir,
                    file_count=file_info["file_count"],
                    variables=["rain"],
                )

                metadata["from_local"] = True
                metadata["file_count"] = file_info["file_count"]
                metadata["missing_dates"] = [d.isoformat() for d in file_info["missing_dates"][:10]]

                if file_info["missing_dates"]:
                    warnings.append(
                        f"{len(file_info['missing_dates'])} dates missing from local files"
                    )

                return self.create_result(
                    success=True,
                    data=tamsat_data,
                    output_path=data_dir,
                    warnings=warnings,
                    metadata=metadata,
                )

            if not file_info["complete"]:
                warnings.append(
                    f"Local data incomplete: {file_info['file_count']} files found, "
                    f"{len(file_info['missing_dates'])} dates missing"
                )

        # Download if requested and library available
        if download:
            if not self.sarra_download_available:
                return self.create_result(
                    success=False,
                    errors=[
                        "SARRA_data_download library not available. "
                        "Install with: pip install SARRA-data-download"
                    ],
                    warnings=warnings,
                    metadata=metadata,
                )

            # V2-22a B2: serialize concurrent downloads on the same
            # (source, region) via a per-source-per-region filelock.
            # Different sources on the same region run concurrently
            # (separate lock files); same source same region serializes.
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            lock_path = cache_lock_path(self.cache_dir, source=self.NAME, region_name=region)
            lock = FileLock(str(lock_path))

            # V2-22b L (AC L.5): pre-lock cancel observation. Lock-wait
            # itself is not interruptible — if cancel fires DURING the
            # 7200 s wait, observation is delayed up to that ceiling
            # (documented operator-tier guidance; architectural fix is
            # V2-22c work). Pre-lock + post-lock checks catch the most
            # common cases.
            raise_if_cancelled(cancel_check, "tamsat.before_lock")
            try:
                with lock.acquire(timeout=DOWNLOAD_LOCK_TIMEOUT_SECONDS):
                    raise_if_cancelled(cancel_check, "tamsat.after_lock")
                    # Marker BEFORE any data write so a SIGKILL during the
                    # download leaves a "this cache may be partial" signal
                    # the next reader will respect (AC 1.7.3c).
                    write_marker(
                        marker_path,
                        source=self.NAME,
                        region_name=region_key,
                        run_id=run_id,
                    )

                    # V2-22b L F-5: emit legacy-cache warning when we
                    # reach the download branch on a pre-B2 cache —
                    # UNCONDITIONAL of `force_redownload`. The bug F-5
                    # closes is the "legacy_assume_valid + incomplete
                    # (force=False)" path silently re-downloading with
                    # no operator-visible signal. Module-level dedup
                    # in warn_legacy_cache_once ensures at-most-once
                    # per cache path per process.
                    if state is not None and state.reason == "legacy_assume_valid":
                        warn_legacy_cache_once(data_dir, self.logger)

                    # AC 1.7.3e: force_redownload bypasses TAMSAT's per-file
                    # short-circuits at :514 (.tif partition skip) and :654
                    # (.tif conversion skip) by removing the stale .tif files
                    # outright. _raw_nc/*.nc is bbox-INDEPENDENT (full TAMSAT
                    # grid for that date; cropping happens at conversion) and
                    # is preserved so a bbox change doesn't trigger a multi-MB
                    # JASMIN refetch with no correctness benefit.
                    if force_redownload:
                        for tif in data_dir.glob("*.tif"):
                            try:
                                tif.unlink()
                            except OSError:
                                pass

                    self._download_tamsat(
                        bounds=bounds_sarra_py,
                        start_date=start_date,
                        end_date=end_date,
                        output_dir=data_dir,
                        region_name=region_key,
                        progress_callback=kwargs.get('progress_callback'),
                        cancel_check=cancel_check,
                    )

                    # Re-validate after download
                    file_info = self._validate_local_files(
                        data_dir=data_dir,
                        region_name=region_key,
                        start_date=start_date,
                        end_date=end_date,
                    )

                    tamsat_data = TAMSATData(
                        region_name=region_key,
                        bounds=bounds_gis,
                        bounds_sarra_py=bounds_sarra_py,
                        start_date=start_date,
                        end_date=end_date,
                        resolution=self.config.resolution,
                        data_dir=data_dir,
                        file_count=file_info["file_count"],
                        variables=["rain"],
                    )

                    metadata["downloaded"] = True
                    metadata["file_count"] = file_info["file_count"]
                    if force_redownload:
                        metadata["force_redownload"] = True

                    # Record provenance
                    if self.provenance:
                        self.provenance.record_retrieval(
                            source=self.NAME,
                            parameters={
                                "region": region.name,
                                "bounds": bounds_sarra_py,
                                "start_date": start_date.isoformat(),
                                "end_date": end_date.isoformat(),
                            },
                            output_path=data_dir,
                            decisions=[],
                        )

                    # AC 1.7.3d: manifest replace BEFORE marker delete.
                    # Marker stays on disk if any of these fail so the next
                    # reader sees the cache as cold.
                    write_cache_manifest(
                        manifest_path,
                        source=self.NAME,
                        region_name=region_key,
                        bbox=bbox_dict,
                        start_date=start_date,
                        end_date=end_date,
                        run_id=run_id,
                        file_count=count_tif_files(data_dir),
                    )
                    delete_marker(marker_path)

                    return self.create_result(
                        success=True,
                        data=tamsat_data,
                        output_path=data_dir,
                        warnings=warnings,
                        metadata=metadata,
                    )

            except Timeout:
                return self.create_result(
                    success=False,
                    errors=[
                        "Another run on this region is downloading data "
                        "(~90 min max). Please wait and retry."
                    ],
                    warnings=warnings,
                    metadata=metadata,
                )
            except PipelineCancelled:
                # V2-22b L: cooperative cancellation must unwind past
                # this broad except, not be rewritten as a download
                # failure (AC L.9). pipeline.execute catches at the
                # boundary and runs handler-local cleanup.
                raise
            except (ImportError, ModuleNotFoundError):
                # An undeclared transitive dependency (rioxarray,
                # rasterio plugin, pyproj data file binding, ...)
                # is a configuration error, not a runtime data
                # error. Letting it surface as ``Download failed:
                # {e}`` masked the gap until a fresh py312 venv
                # hit the missing-rioxarray case end-to-end. Per
                # durable lesson #6 (broad-except carve-out), let
                # the ImportError propagate so pip / CI / startup
                # surfaces the missing dep loudly.
                raise
            except Exception as e:
                return self.create_result(
                    success=False,
                    errors=[f"Download failed: {e}"],
                    warnings=warnings,
                    metadata=metadata,
                )

        # Data not available and download not requested
        return self.create_result(
            success=False,
            errors=[
                f"TAMSAT data not found at {data_dir}. "
                "Either provide existing data or set download=True"
            ],
            warnings=warnings,
            metadata=metadata,
        )

    def validate(self, data: Any) -> List[str]:
        """Validate TAMSAT data.

        Args:
            data: TAMSATData object to validate

        Returns:
            List of validation error/warning messages
        """
        warnings = []

        if not isinstance(data, TAMSATData):
            return [f"Expected TAMSATData, got {type(data)}"]

        # Check file count
        expected_days = (data.end_date - data.start_date).days + 1
        if data.file_count < expected_days * 0.95:  # Allow 5% missing
            warnings.append(
                f"Only {data.file_count}/{expected_days} daily files available "
                f"({100*data.file_count/expected_days:.1f}%)"
            )

        # Check data directory exists
        if not data.data_dir.exists():
            warnings.append(f"Data directory does not exist: {data.data_dir}")

        # Validate bounds
        if data.bounds[0] >= data.bounds[2]:  # minx >= maxx
            warnings.append("Invalid bounds: minx >= maxx")
        if data.bounds[1] >= data.bounds[3]:  # miny >= maxy
            warnings.append("Invalid bounds: miny >= maxy")

        return warnings

    def load_daily_rainfall(
        self,
        data_dir: Union[str, Path],
        target_date: date,
        region_name: str,
    ) -> Optional[np.ndarray]:
        """Load rainfall data for a single day.

        Args:
            data_dir: Directory containing TAMSAT files
            target_date: Date to load
            region_name: Region name for file pattern

        Returns:
            2D numpy array of rainfall values, or None if file not found
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required for loading TAMSAT files")

        data_dir = Path(data_dir)
        filename = self.FILE_PATTERN.format(
            version=self.config.version,
            region=region_name,
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
        )

        file_path = data_dir / filename

        if not file_path.exists():
            # Try alternative patterns
            patterns = [
                f"TAMSAT*{target_date.year}_{target_date.month:02d}_{target_date.day:02d}.tif",
                f"*rfe*{target_date.year}_{target_date.month:02d}_{target_date.day:02d}.tif",
            ]
            for pattern in patterns:
                matches = list(data_dir.glob(pattern))
                if matches:
                    file_path = matches[0]
                    break
            else:
                return None

        with rasterio.open(file_path) as src:
            data = src.read(1)  # First band
            return data

    def load_timeseries(
        self,
        data_dir: Union[str, Path],
        start_date: date,
        end_date: date,
        region_name: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Load rainfall time series from TAMSAT files.

        Args:
            data_dir: Directory containing TAMSAT files
            start_date: Start date
            end_date: End date
            region_name: Region name
            lat: Optional latitude for point extraction
            lon: Optional longitude for point extraction

        Returns:
            Dictionary with dates and rainfall values
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio required for loading TAMSAT files")

        data_dir = Path(data_dir)
        dates = []
        rainfall = []

        current = start_date
        while current <= end_date:
            data = self.load_daily_rainfall(data_dir, current, region_name)

            if data is not None:
                if lat is not None and lon is not None:
                    # Extract point value (would need transform info)
                    # For now, use center value as placeholder
                    value = float(np.nanmean(data))
                else:
                    # Use spatial mean
                    value = float(np.nanmean(data))

                dates.append(current)
                rainfall.append(value)
            else:
                dates.append(current)
                rainfall.append(np.nan)

            current += timedelta(days=1)

        return {
            "dates": dates,
            "rainfall": rainfall,
            "unit": "mm/day",
            "source": self.NAME,
        }

    def _download_tamsat(
        self,
        bounds: List[float],
        start_date: date,
        end_date: date,
        output_dir: Path,
        region_name: str,
        progress_callback=None,
        max_workers: int = 4,
        cancel_check=None,
    ) -> None:
        """Download TAMSAT daily rainfall and crop to region bounds.

        Two-phase architecture to avoid SIGSEGV from concurrent
        rasterio/PROJ writes:

        Phase 1 — Parallel HTTP download (thread-safe):
            4 threads fetch raw .nc files from JASMIN. Pure HTTP,
            no GDAL/PROJ loaded. 4x network speedup.

        Phase 2 — Sequential crop + convert (rasterio-safe):
            Single-threaded xarray crop + rioxarray GeoTIFF write.
            No concurrency on PROJ/GDAL, no SIGSEGV.

        V2-22b L (cooperative cancellation, ≤180 s cancel-to-exit
        worst case): ``cancel_check`` is checked before ``executor.submit``
        (blocks queued dates from starting), after each ``future.result``
        in the ``as_completed`` loop, and at the top of every Phase-2
        iteration before ``rio.to_raster``. On cancel observation, the
        Phase-1 executor is shut down with ``cancel_futures=True`` so
        pending dates don't touch JASMIN at all; already-running
        workers complete at their 90-s HTTP ceiling (2× for 5xx
        retry → 180 s theoretical max per in-flight worker).

        Args:
            bounds: Bounding box in SARRA-Py format [lat_NW, lon_NW, lat_SE, lon_SE]
            start_date: Start date
            end_date: End date
            output_dir: Output directory for cropped GeoTIFFs
            region_name: Region name for file naming
            progress_callback: Optional callback(current, total, detail)
            max_workers: Number of parallel download threads (default 4)
        """
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        TAMSAT_URL = (
            "https://gws-access.jasmin.ac.uk/public/tamsat/rfe/data/"
            "v3.1/daily/{year}/{month:02d}/"
            "rfe{year}_{month:02d}_{day:02d}.v3.1.nc"
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse SARRA-Py bounds: [lat_NW, lon_NW, lat_SE, lon_SE]
        lat_nw, lon_nw, lat_se, lon_se = bounds
        lat_min = min(lat_nw, lat_se)
        lat_max = max(lat_nw, lat_se)
        lon_min = min(lon_nw, lon_se)
        lon_max = max(lon_nw, lon_se)

        # Partition dates across three disk states so the plan log
        # reflects actual HTTP work instead of tif-partition state:
        #   - final_tif_cached: the .tif already exists, no work to do
        #   - nc_cached: the .tif is missing but the raw .nc is
        #     already in `_raw_nc/`, so Phase 1 skips HTTP and only
        #     Phase 2 runs
        #   - http_needed: neither the .tif nor the .nc exists, so
        #     Phase 1 will issue an HTTP request
        #
        # Issue 4 (warning-auditor MEDIUM) fix — the previous log said
        # "{dates_to_download} to fetch, {cached} cached, total=..."
        # which conflated the two cache tiers. A run that wiped the
        # marker file but kept the .nc cache reported "1096 to fetch,
        # 0 cached" even though only ~366 HTTP requests actually
        # fired. The new log reports HTTP count honestly.
        nc_dir = output_dir / "_raw_nc"
        dates_to_download = []
        final_tif_cached = 0
        nc_cached = 0
        current_date = start_date
        while current_date <= end_date:
            tif_name = self.FILE_PATTERN.format(
                version=self.config.version, region=region_name,
                year=current_date.year, month=current_date.month,
                day=current_date.day,
            )
            if (output_dir / tif_name).exists():
                final_tif_cached += 1
            else:
                dates_to_download.append(current_date)
                nc_name = (
                    f"rfe{current_date.year}_{current_date.month:02d}"
                    f"_{current_date.day:02d}.nc"
                )
                if (nc_dir / nc_name).exists():
                    nc_cached += 1
            current_date += timedelta(days=1)

        total_days = (end_date - start_date).days + 1
        http_needed = len(dates_to_download) - nc_cached
        # Retained for downstream callers that referenced the old
        # local name; the log no longer uses it.
        already_have = final_tif_cached

        self.logger.info(
            f"TAMSAT plan: {total_days} total files for {region_name}, "
            f"{http_needed} need HTTP download, "
            f"{nc_cached} already in .nc cache, "
            f"{final_tif_cached} already have final .tif "
            f"(workers={max_workers})"
        )

        if not dates_to_download:
            if progress_callback:
                progress_callback(total_days, total_days, '')
            return

        # Ensure the raw-nc dir exists before Phase 1 writes into it.
        # Partition loop above only read it for existence checks —
        # safe to delay mkdir until we know we actually have work.
        nc_dir.mkdir(exist_ok=True)

        # ── Phase 1: Parallel HTTP download (thread-safe, no GDAL) ──

        def _download_nc(target_date):
            """Download a single raw .nc file. Pure HTTP, no rasterio."""
            # V2-22b L: per-worker cancel check — raises PipelineCancelled
            # into the future, caught by the `except PipelineCancelled:
            # raise` carve-out at the as_completed loop (tamsat.py:1106).
            raise_if_cancelled(
                cancel_check, f"tamsat._download_nc.{target_date}"
            )
            nc_name = f"rfe{target_date.year}_{target_date.month:02d}_{target_date.day:02d}.nc"
            nc_path = nc_dir / nc_name

            if nc_path.exists():
                return "cached"

            url = TAMSAT_URL.format(
                year=target_date.year,
                month=target_date.month,
                day=target_date.day,
            )

            try:
                resp = requests.get(url, timeout=(30, 60))

                if resp.status_code == 404:
                    return "skipped"

                if resp.status_code >= 500:
                    self.logger.warning(
                        f"TAMSAT server {resp.status_code} for "
                        f"{target_date}, retrying..."
                    )
                    resp = requests.get(url, timeout=(30, 60))
                    if resp.status_code != 200:
                        return f"HTTP {resp.status_code}"

                resp.raise_for_status()
                nc_path.write_bytes(resp.content)
                return "ok"

            except requests.exceptions.Timeout:
                return "timeout"
            except requests.exceptions.RequestException as e:
                return f"download error: {e}"

        # Codex self-check MEDIUM — track HTTP fetches and .nc-cache
        # hits SEPARATELY so Phase 1 progress + completion messages
        # report actual network work, not the conflated "ok or cached"
        # count. Extends the plan-log honesty (32a9ada) to the runtime
        # path: a run whose .nc cache survived a marker wipe must not
        # look like it downloaded N files when it actually served
        # them from cache.
        dl_http_fetched = 0
        dl_nc_cache_served = 0
        dl_skipped = 0
        errors = []

        self.logger.info(
            f"TAMSAT Phase 1: downloading {len(dates_to_download)} "
            f".nc files ({max_workers} threads)..."
        )

        # V2-22b L: pre-submit check — fires BEFORE any JASMIN call.
        # Operator clicks cancel before the first future starts and
        # cancel_to_exit is sub-second.
        raise_if_cancelled(cancel_check, "tamsat.phase1.before_submit")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_download_nc, d): d
                for d in dates_to_download
            }
            for future in as_completed(futures):
                target_date = futures[future]
                try:
                    result = future.result(timeout=90)
                    if result == "ok":
                        dl_http_fetched += 1
                    elif result == "cached":
                        dl_nc_cache_served += 1
                    elif result == "skipped":
                        dl_skipped += 1
                    else:
                        self.logger.warning(
                            f"TAMSAT {target_date}: {result}"
                        )
                        errors.append(f"{target_date}: {result}")
                        dl_skipped += 1
                except PipelineCancelled:
                    # V2-22b L: a _download_nc worker observed cancel
                    # and raised PipelineCancelled in its future.
                    # future.result re-raised it here. Shut down the
                    # executor with cancel_futures=True so queued-but-
                    # not-started dates don't touch JASMIN, then
                    # propagate past the as_completed loop and unwind
                    # through _download_tamsat.
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as e:
                    self.logger.warning(
                        f"TAMSAT {target_date}: {e}"
                    )
                    errors.append(f"{target_date}: {e}")
                    dl_skipped += 1

                # V2-22b L: top-of-iteration check — catches cancel
                # observed between worker completions, NOT via future
                # propagation. Finer-grained than waiting for every
                # future to resolve.
                #
                # F-5 fix (Group L Gate B round 2): this poll path
                # raises DIRECTLY via ``raise_if_cancelled`` without
                # the worker-future wrapping of the inner try/except.
                # Without a pre-raise shutdown, the ThreadPoolExecutor
                # context-manager ``__exit__`` calls
                # ``shutdown(wait=True)`` and the queued futures run
                # to completion — AC L.1 / BLOCKER 3 reopens. Force
                # shutdown with ``cancel_futures=True`` BEFORE raising
                # so pending dates are cancelled; in-flight workers
                # still drain at the 180 s HTTP ceiling (expected).
                if cancel_check is not None and cancel_check():
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise PipelineCancelled("tamsat.phase1.as_completed")

                processed = dl_http_fetched + dl_nc_cache_served + dl_skipped
                if progress_callback and processed % 10 == 0:
                    progress_callback(
                        already_have + processed // 2,
                        total_days,
                        # Honest two-signal message: HTTP fetches are
                        # the slow path; .nc cache hits are fast and
                        # shouldn't be labelled "downloading".
                        f'TAMSAT rainfall: {dl_http_fetched} HTTP fetched + '
                        f'{dl_nc_cache_served} from cache ({processed}/'
                        f'{len(dates_to_download)} processed)',
                    )

        self.logger.info(
            f"TAMSAT Phase 1 complete: {dl_http_fetched} HTTP fetched, "
            f"{dl_nc_cache_served} nc-cache served, "
            f"{dl_skipped} skipped"
        )

        # ── Phase 2: Sequential crop + GeoTIFF (single-threaded, rasterio-safe) ──

        import xarray as xr
        import rioxarray  # noqa: F401

        converted = 0
        nc_files = sorted(nc_dir.glob("*.nc"))

        self.logger.info(
            f"TAMSAT Phase 2: converting {len(nc_files)} "
            f".nc → .tif (sequential)..."
        )

        for nc_path in nc_files:
            # V2-22b L: per-file cancel check — ~0.5-2 s granularity,
            # matches AC L.1 target. The raise happens BEFORE both the
            # xr.open_dataset call (slow on cold PROJ) and the
            # rio.to_raster write, so cancel observed here means this
            # file is neither started nor partially written.
            raise_if_cancelled(cancel_check, "tamsat.phase2.convert")

            # Parse date from filename: rfe{Y}_{M}_{D}.nc
            stem = nc_path.stem  # rfe2020_01_15
            parts = stem.replace("rfe", "").split("_")
            try:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            except (ValueError, IndexError):
                continue

            tif_name = self.FILE_PATTERN.format(
                version=self.config.version, region=region_name,
                year=y, month=m, day=d,
            )
            tif_path = output_dir / tif_name

            if tif_path.exists():
                converted += 1
                nc_path.unlink()
                continue

            try:
                ds = xr.open_dataset(str(nc_path))
                try:
                    ds_cropped = ds.where(
                        (ds.lat >= lat_min) & (ds.lat <= lat_max)
                        & (ds.lon >= lon_min) & (ds.lon <= lon_max),
                        drop=True,
                    )
                    rfe = ds_cropped["rfe"]
                    rfe = rfe.rio.set_spatial_dims(
                        x_dim="lon", y_dim="lat"
                    )
                    rfe = rfe.rio.write_crs("EPSG:4326")
                    # V2-22b L F-7: second per-file check right before
                    # the rio.to_raster disk write. Catches cancel fired
                    # during the xr.open_dataset + crop operations
                    # (which can take seconds on a cold PROJ cache).
                    raise_if_cancelled(
                        cancel_check, f"tamsat.phase2.to_raster={nc_path.name}",
                    )
                    rfe.rio.to_raster(str(tif_path))
                    converted += 1
                except PipelineCancelled:
                    # V2-22b L Gate B round 2: inner-try carve-out so
                    # the F-7 pre-to_raster cancel doesn't get rewritten
                    # as "Failed to convert {name}" (which also tries
                    # to unlink the tif_path — we want the cancel to
                    # skip both, and the `ds.close()` finally to still run).
                    raise
                except Exception as e:
                    self.logger.warning(
                        f"Failed to convert {nc_path.name}: {e}"
                    )
                    if tif_path.exists():
                        tif_path.unlink()
                finally:
                    ds.close()
            except PipelineCancelled:
                # V2-22b L Gate B round 2: outer-try carve-out so the
                # inner-try's re-raised cancel propagates past the outer
                # except-Exception log line.
                raise
            except Exception as e:
                self.logger.warning(
                    f"Failed to open {nc_path.name}: {e}"
                )

            # Clean up raw .nc after conversion
            try:
                nc_path.unlink()
            except OSError:
                pass

            if progress_callback and converted % 10 == 0:
                progress_callback(
                    already_have + converted,
                    total_days,
                    f'TAMSAT rainfall: converting {converted}/'
                    f'{len(nc_files)} files',
                )

        # Clean up temp dir
        try:
            nc_dir.rmdir()
        except OSError:
            pass

        # Final progress
        if progress_callback:
            progress_callback(total_days, total_days, '')

        total_done = already_have + converted
        # Completion-level honesty (Codex self-check MEDIUM): mirror
        # the plan log's three-tier partition so operators can
        # reconstruct where the wall-time went.
        self.logger.info(
            f"TAMSAT download complete: {total_done}/{total_days} files "
            f"({dl_http_fetched} HTTP fetched, "
            f"{dl_nc_cache_served} nc-cache served, "
            f"{already_have} .tif already cached), "
            f"{dl_skipped} skipped, {len(errors)} errors"
        )
        if errors:
            self.logger.warning(
                f"TAMSAT download errors (first 10): {errors[:10]}"
            )

    def _validate_local_files(
        self,
        data_dir: Path,
        region_name: str,
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:
        """Validate local TAMSAT files for completeness.

        Args:
            data_dir: Directory containing files
            region_name: Region name
            start_date: Expected start date
            end_date: Expected end date

        Returns:
            Dictionary with validation results
        """
        # Find all TAMSAT files
        patterns = [
            "TAMSAT*.tif",
            "*rfe*.tif",
        ]

        files = []
        for pattern in patterns:
            files.extend(data_dir.glob(pattern))

        # Parse dates from filenames
        found_dates = set()
        for f in files:
            basename = f.name
            # Try to extract date from filename
            # Pattern: ..._YYYY_MM_DD.tif
            parts = basename.replace(".tif", "").split("_")
            if len(parts) >= 3:
                try:
                    year = int(parts[-3])
                    month = int(parts[-2])
                    day = int(parts[-1])
                    found_dates.add(date(year, month, day))
                except (ValueError, IndexError):
                    pass

        # Calculate expected dates
        expected_dates = set()
        current = start_date
        while current <= end_date:
            expected_dates.add(current)
            current += timedelta(days=1)

        # Find missing dates
        missing_dates = sorted(expected_dates - found_dates)

        return {
            "file_count": len(found_dates),
            "expected_count": len(expected_dates),
            "missing_dates": missing_dates,
            "complete": len(missing_dates) == 0,
            "coverage_pct": 100 * len(found_dates) / max(len(expected_dates), 1),
        }

    def _parse_date(self, date_input: Union[str, date]) -> date:
        """Parse date from string or date object."""
        if isinstance(date_input, date):
            return date_input
        return datetime.strptime(date_input, "%Y-%m-%d").date()

    def get_expected_file_path(
        self,
        data_dir: Path,
        region_name: str,
        target_date: date,
    ) -> Path:
        """Get expected file path for a given date.

        Args:
            data_dir: Base data directory
            region_name: Region name
            target_date: Target date

        Returns:
            Expected file path
        """
        filename = self.FILE_PATTERN.format(
            version=self.config.version,
            region=region_name,
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
        )
        return data_dir / filename
