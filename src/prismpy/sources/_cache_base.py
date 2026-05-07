"""
Generic cache-isolation primitives shared across data sources.

This module owns the substrate that every prismpy data source uses to
keep cache reads, writes, and locks safe under crashes, concurrent
processes, and bbox/version drift. It accepts a generic ``key: str``
lock identifier so callers with different cache-key shapes can share
one implementation:

* TAMSAT and AgERA5 cache per (source, region) — they pre-convert the
  ``Region`` dataclass to a sanitized string via
  ``region_cache_key_from_region``. ``tamsat.cache_lock_path`` is the
  Region-coupled compatibility wrapper.

* ISIMIP3b (Sprint G) caches per (product, scenario, gcm, variable,
  bbox_key) — it derives its own string key and calls the generic
  ``cache_lock_path`` directly from this module.

The split codifies the canonical-source-or-structural-pin discipline
(durable lesson #24): ONE atomic-write helper, ONE lock-path helper,
ONE manifest writer/reader, three callers. Sprint G AC-G-2.0 drove the
extraction; the existing tamsat/agera5 contract is preserved
byte-for-byte through the compatibility shim in tamsat.py.

Crash semantics for the manifest writer (preserved from V2-22a B2):

* SIGKILL between tempfile creation and ``os.replace`` → orphan
  ``.writing-XXXX.tmp`` may persist (best-effort cleanup on writer
  re-entry). Target manifest is either absent or contains a prior
  successful manifest — never partial/corrupt.
* SIGKILL between ``os.replace`` and dir-fsync → target is written but
  the rename may not be durable. Acceptable: next reader either sees
  the new manifest or its absence (which the marker still flags as
  cold).
"""

import json
import logging
import os
import tempfile
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional


# ── Constants ────────────────────────────────────────────────────────

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "_manifest.json"
MARKER_FILENAME = "_manifest.writing"
TMPFILE_PREFIX = ".writing-"

# 0.01° per edge ≈ 1 km — well above sub-pixel noise on TAMSAT (0.0375°)
# and AgERA5 (0.1°) grids, matches wizard map-click precision.
BBOX_TOLERANCE_DEG = 0.01

# 7200 s (2 h) — covers worst-case 3-year SARRA-Py runs under CDS contention.
DOWNLOAD_LOCK_TIMEOUT_SECONDS = 7200


# ── Module-level state for legacy-cache warner ───────────────────────

# One WARNING per cache path per process lifetime for legacy (no-manifest)
# caches — keeps server logs from drowning when many requests hit a region
# whose cache predates B2.
_legacy_warned_cache_paths: set = set()
_legacy_warned_lock = threading.Lock()


# ── Cache state NamedTuple ───────────────────────────────────────────


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


# ── Generic helpers ──────────────────────────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bbox_matches(
    a: Dict[str, float],
    b: Dict[str, float],
    tolerance_deg: float = BBOX_TOLERANCE_DEG,
) -> bool:
    for edge in ("north", "south", "east", "west"):
        if abs(float(a.get(edge, 0.0)) - float(b.get(edge, 0.0))) > tolerance_deg:
            return False
    return True


def bbox_to_dict(bounds: Any) -> Dict[str, float]:
    """Extract bbox in canonical {north,south,east,west} keys.

    Avoids the GIS-vs-SARRA-Py tuple-order drift risk by always using
    explicit edge names. Accepts any bounds-like object exposing
    ``maxx/minx/maxy/miny`` attributes (typically
    ``prismpy.models.region.BoundingBox``).
    """
    return {
        "north": float(bounds.maxy),
        "south": float(bounds.miny),
        "east": float(bounds.maxx),
        "west": float(bounds.minx),
    }


def count_tif_files(data_dir: Path) -> int:
    """Recursive .tif count under data_dir (0 if absent).

    Works for TAMSAT (flat) and AgERA5 (per-variable subdirs). Excludes
    .nc fragments in TAMSAT's ``_raw_nc/`` staging area.
    """
    if not data_dir.exists():
        return 0
    return sum(1 for _ in data_dir.rglob("*.tif"))


def cleanup_orphan_tmpfiles(target_dir: Path) -> None:
    """Best-effort removal of leftover atomic-write tempfiles.

    A SIGKILL between os.fsync and os.replace can leave a
    ``.writing-XXXX.tmp`` fragment in the target dir. Cleanup is
    opportunistic — failures are swallowed because the next manifest
    write will overwrite the target regardless.
    """
    if not target_dir.exists():
        return
    for stale in target_dir.glob(f"{TMPFILE_PREFIX}*.tmp"):
        try:
            stale.unlink()
        except OSError:
            pass


def bbox_field_for_log(bbox: Optional[Dict[str, float]]) -> str:
    if not bbox:
        return "<none>"
    return (
        f"N={bbox.get('north')} S={bbox.get('south')} "
        f"E={bbox.get('east')} W={bbox.get('west')}"
    )


# ── Generic atomic-write helper ──────────────────────────────────────


def write_atomic_json(target_path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write a JSON payload to ``target_path``.

    Uses tempfile + ``os.replace`` + dir-fsync so the rename is durable
    across crashes. Reusable by:

    * TAMSAT/AgERA5 ``write_cache_manifest`` (climate-Region cache shape).
    * TAMSAT/AgERA5 ``write_marker`` (in-progress marker).
    * ISIMIP3b ``cached_cutout`` ``.meta.json`` writer (Sprint G).

    Crash semantics: tempfile staging dir == target's parent so
    ``os.replace`` stays atomic on the same filesystem (cross-fs rename
    degrades to copy+unlink and loses atomicity). ``fsync`` on the file
    ensures bytes hit disk; ``fsync`` on the directory ensures the
    rename is durable across a crash.

    The payload is serialized with ``json.dumps(..., indent=2)`` for
    human inspection; consumers can override formatting by serializing
    upstream and bypassing this helper if byte-identical reproducibility
    is required (Sprint G CC-G-7 manifest writer rewrite).
    """
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    cleanup_orphan_tmpfiles(target_dir)

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
        os.replace(str(tmp_path), str(target_path))
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


# ── Generic lock-path primitive ──────────────────────────────────────


def cache_lock_path(cache_dir: Path, source: str, key: str) -> Path:
    """Per-source, per-key lock path: ``<cache_dir>/.<source>-<key>.lock``.

    Different sources on the same key use different lock files so a
    single SARRA-Py run can progress through TAMSAT then AgERA5 without
    self-blocking, and two users hitting the same key overlap on
    different sources but serialize on the same source.

    ``key`` is a generic string identifier — sanitized at call site.
    Climate-Region callers (TAMSAT, AgERA5) pre-convert their ``Region``
    via ``region_cache_key_from_region`` and pass the result here. The
    Region-coupled compatibility wrapper lives at
    ``tamsat.cache_lock_path``. ISIMIP3b (Sprint G) uses its own
    per-(product/scenario/gcm/variable/bbox) key derivation and calls
    this generic function directly.
    """
    return cache_dir / f".{source}-{key}.lock"


# ── Climate-domain marker + manifest helpers ─────────────────────────
#
# These functions carry climate-Region-shaped payload fields (bbox,
# region_name, start_date, end_date, run_id, file_count). They live in
# this module so TAMSAT and AgERA5 share one implementation per durable
# lesson #24. Sources with different cache shapes (e.g., ISIMIP3b)
# build their own equivalents on top of ``write_atomic_json`` rather
# than reusing these.


def write_marker(
    marker_path: Path,
    *,
    source: str,
    region_name: str,
    run_id: Optional[str],
) -> None:
    """Create ``_manifest.writing`` — the in-progress signal that survives SIGKILL.

    Persists if the writer process is killed mid-download. The next
    reader sees the marker and treats the cache as cold even when data
    files look complete (AC 1.7.3c). Deleted by ``delete_marker`` after
    a successful manifest replace.
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
    """Write ``_manifest.json`` atomically; called inside the per-source lock.

    Climate-Region-shaped manifest payload. Routes through
    ``write_atomic_json`` for the actual atomic-write so the
    crash-durability semantics live in ONE place.
    """
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
    write_atomic_json(manifest_path, payload)


def check_cache_manifest(
    manifest_path: Path,
    marker_path: Path,
    *,
    expected_bbox: Dict[str, float],
    actual_file_count: int,
    data_files_present: bool,
) -> CacheManifestState:
    """Inspect a cache dir and return the action the caller should take.

    Evaluation order is significant: marker presence overrides
    everything because a prior writer may have crashed AFTER replacing
    the manifest but BEFORE deleting the marker (or partway through
    producing data).

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
            started_at = (
                marker_data.get("started_at") if isinstance(marker_data, dict) else None
            )
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

    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
    ):
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


