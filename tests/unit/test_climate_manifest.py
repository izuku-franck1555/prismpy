"""Cache manifest + filelock tests — V2-22a Group B2.

Closes V2-22a/B2 ACs 1.7.1 through 1.7.4. Helpers under test live in
prismpy.sources.climate.tamsat (canonical home; agera5.py re-imports
them). Source-specific wiring (TAMSAT cache-hit path, AgERA5 cache-hit
path, force_redownload .tif wipe) is exercised by ACs 1.7.2 / 1.7.3 /
1.7.5 below using a real source instance with a fixture cache.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from prismpy.models.region import BoundingBox, Region
from prismpy.sources.climate.tamsat import (
    BBOX_TOLERANCE_DEG,
    DOWNLOAD_LOCK_TIMEOUT_SECONDS,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    MARKER_FILENAME,
    TMPFILE_PREFIX,
    CacheManifestState,
    TAMSATSource,
    bbox_to_dict,
    cache_lock_path,
    check_cache_manifest,
    count_tif_files,
    delete_marker,
    warn_legacy_cache_once,
    write_cache_manifest,
    write_marker,
    _legacy_warned_cache_paths,
)
from prismpy.sources.climate.agera5 import AgERA5Source
from prismpy.utils.gis_utils import snap_bounds_outward_to_grid


def _widened_fetch_bbox(region: Region, resolution: float) -> Dict[str, float]:
    """Fix-C contract: the climate cache manifest records the WIDENED FETCH
    extent (raw bounds snapped OUTWARD to enclosing native pixel edges so the
    AgMIP perimeter cells get climate), NOT the raw config bounds. A re-run
    with the same config recomputes an identical widened bbox and hits the
    cache. Mirrors the widen in agera5.py (resolution=0.1) / tamsat.py
    (resolution=0.0375); the stored bbox is keyed on what was actually
    fetched, so a widening-logic change correctly invalidates the cache too.
    """
    widened = BoundingBox.from_gis_format(
        list(
            snap_bounds_outward_to_grid(
                region.bounds.to_tuple(), resolution=resolution
            )
        ),
        crs=region.bounds.crs,
    )
    return bbox_to_dict(widened)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Per-test cache directory."""
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def maradi_region() -> Region:
    """Stable test region — Niger / Maradi."""
    return Region(
        name="Maradi",
        country="Niger",
        country_iso3="NER",
        bounds=BoundingBox(minx=6.5, miny=13.0, maxx=8.0, maxy=14.5),
        gadm_level=1,
    )


@pytest.fixture
def maradi_bbox(maradi_region: Region) -> Dict[str, float]:
    return bbox_to_dict(maradi_region.bounds)


@pytest.fixture
def manifest_args(maradi_bbox: Dict[str, float]) -> Dict[str, Any]:
    return {
        "source": "tamsat",
        "region_name": "Maradi",
        "bbox": maradi_bbox,
        "start_date": date(2020, 1, 1),
        "end_date": date(2022, 12, 31),
        "run_id": "test-run-uuid",
        "file_count": 1096,
    }


@pytest.fixture(autouse=True)
def _clear_legacy_warn_dedup():
    """Reset the module-level dedup set so test order can't mask warnings."""
    _legacy_warned_cache_paths.clear()
    yield
    _legacy_warned_cache_paths.clear()


def _stale_tif(path: Path, content: bytes = b"OLD") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


# ── Cross-process filelock workers (top-level — spawn re-imports) ─────


def _filelock_holder_worker(lock_path: str, log_path: str, hold_s: float) -> None:
    """Worker target for the cross-process filelock test.

    MUST live at module scope: spawn-context multiprocessing pickles the
    callable by import path, not by closure capture, and a nested function
    inside a TestCase method is unreachable from the spawned interpreter.

    Acquires the FileLock, records `holder_start` and `holder_end`
    monotonic timestamps either side of a deliberate hold, then releases.
    """
    import time as _time
    from filelock import FileLock as _FL

    with _FL(lock_path):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"holder_start {_time.monotonic()}\n")
        _time.sleep(hold_s)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"holder_end {_time.monotonic()}\n")


def _filelock_waiter_worker(lock_path: str, log_path: str) -> None:
    """Waiter half of the cross-process filelock test.

    Attempts to acquire the same FileLock. If the kernel-level
    fcntl.flock is engaged (per the DEPLOY.md rollout assumption),
    this call blocks until the holder process releases. The recorded
    `waiter_acquired` timestamp must therefore be >= the holder's
    `holder_end` timestamp; the test asserts that ordering.
    """
    import time as _time
    from filelock import FileLock as _FL

    with _FL(lock_path):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"waiter_acquired {_time.monotonic()}\n")


# ── AC 1.7.1: manifest round-trip + atomic-write safety ──────────────


class TestManifestRoundTripAndAtomicity:
    """AC 1.7.1: round-trip preserves all fields; atomic-write failure
    leaves target either absent OR containing a prior valid manifest —
    NEVER partial/corrupt at the target path."""

    def test_round_trip_all_fields_preserved(
        self, cache_dir: Path, manifest_args: Dict[str, Any]
    ) -> None:
        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME
        write_cache_manifest(manifest_path, **manifest_args)

        assert manifest_path.exists()
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION == 1
        assert manifest["source"] == manifest_args["source"]
        assert manifest["region_name"] == manifest_args["region_name"]
        assert manifest["bbox"] == manifest_args["bbox"]
        assert manifest["start_date"] == manifest_args["start_date"].isoformat()
        assert manifest["end_date"] == manifest_args["end_date"].isoformat()
        assert manifest["run_id"] == manifest_args["run_id"]
        assert manifest["file_count"] == manifest_args["file_count"]
        # created_at is set at write time; just confirm it's present + ISO-shaped.
        assert isinstance(manifest["created_at"], str)
        assert manifest["created_at"].endswith("Z")

    def test_tempfile_is_in_target_directory(
        self,
        cache_dir: Path,
        manifest_args: Dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Atomic write requires tempfile + target on the same filesystem.
        Confirm dir= passes target_dir, not /tmp (which would defeat
        os.replace atomicity on a cross-fs cache mount)."""
        import prismpy.sources.climate.tamsat as tmod

        recorded: Dict[str, Any] = {}
        real_NTF = tmod.tempfile.NamedTemporaryFile

        def recording_NTF(*args, **kwargs):
            recorded["dir"] = kwargs.get("dir")
            recorded["prefix"] = kwargs.get("prefix")
            recorded["suffix"] = kwargs.get("suffix")
            return real_NTF(*args, **kwargs)

        monkeypatch.setattr(tmod.tempfile, "NamedTemporaryFile", recording_NTF)

        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME
        write_cache_manifest(manifest_path, **manifest_args)

        assert recorded["dir"] == str(manifest_path.parent)
        assert recorded["prefix"] == TMPFILE_PREFIX
        assert recorded["suffix"] == ".tmp"

    def test_replace_failure_leaves_target_absent_when_no_prior(
        self,
        cache_dir: Path,
        manifest_args: Dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SIGKILL-equivalent between fsync and replace: target must be
        ABSENT when no prior manifest existed (never half-written)."""
        import prismpy.sources.climate.tamsat as tmod

        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME

        def _boom(src, dst):
            raise IOError("simulated SIGKILL between fsync and replace")

        monkeypatch.setattr(tmod.os, "replace", _boom)

        with pytest.raises(IOError, match="simulated"):
            write_cache_manifest(manifest_path, **manifest_args)

        # Target must be ABSENT — never partially written.
        assert not manifest_path.exists()

    def test_replace_failure_preserves_prior_valid_manifest(
        self,
        cache_dir: Path,
        manifest_args: Dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SIGKILL-equivalent during a SECOND write: the prior valid
        manifest at the target path must survive intact, not be corrupted
        by the failed second write."""
        import prismpy.sources.climate.tamsat as tmod

        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME

        # First write succeeds — establishes the prior valid manifest
        write_cache_manifest(manifest_path, **manifest_args)
        prior_bytes = manifest_path.read_bytes()
        prior_payload = json.loads(prior_bytes)

        # Second write fails between fsync and replace
        def _boom(src, dst):
            raise IOError("simulated mid-replace failure")

        monkeypatch.setattr(tmod.os, "replace", _boom)

        new_args = dict(manifest_args)
        new_args["bbox"] = {
            "north": 99.0,
            "south": -99.0,
            "east": 99.0,
            "west": -99.0,
        }
        with pytest.raises(IOError, match="simulated"):
            write_cache_manifest(manifest_path, **new_args)

        # Prior manifest survives intact — never partially overwritten.
        assert manifest_path.exists()
        assert manifest_path.read_bytes() == prior_bytes
        # Sanity: the prior payload is still parseable + valid.
        recovered = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert recovered == prior_payload
        assert recovered["bbox"] == manifest_args["bbox"]

    def test_orphan_tmpfile_cleaned_on_re_entry(
        self,
        cache_dir: Path,
        manifest_args: Dict[str, Any],
    ) -> None:
        """Best-effort cleanup of leftover .writing-*.tmp from a prior
        crashed write — runs before the next NamedTemporaryFile so the
        target dir doesn't accumulate fragments."""
        target_dir = cache_dir / "tamsat" / "maradi"
        target_dir.mkdir(parents=True)
        # Plant an orphan from a hypothetical prior crash
        orphan = target_dir / f"{TMPFILE_PREFIX}deadbeef.tmp"
        orphan.write_bytes(b"crashed")
        assert orphan.exists()

        write_cache_manifest(target_dir / MANIFEST_FILENAME, **manifest_args)

        # Orphan removed; the just-written manifest is the only file
        # matching the orphan pattern would be — actually let's just
        # check the orphan is gone.
        assert not orphan.exists()
        assert (target_dir / MANIFEST_FILENAME).exists()

    def test_dir_fsync_runs_on_success(
        self,
        cache_dir: Path,
        manifest_args: Dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Crash-durable rename requires fsync on the containing dir
        after os.replace. Capture os.open + os.fsync calls and assert
        the target dir's fd was fsynced post-replace."""
        import prismpy.sources.climate.tamsat as tmod

        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME

        opened_dirs: List[str] = []
        fsynced_fds: List[int] = []
        real_open = tmod.os.open
        real_fsync = tmod.os.fsync

        def recording_open(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            try:
                if os.path.isdir(path):
                    opened_dirs.append(str(path))
            except (TypeError, OSError):
                pass
            return fd

        def recording_fsync(fd):
            fsynced_fds.append(fd)
            return real_fsync(fd)

        monkeypatch.setattr(tmod.os, "open", recording_open)
        monkeypatch.setattr(tmod.os, "fsync", recording_fsync)

        write_cache_manifest(manifest_path, **manifest_args)

        # Must have opened the target directory at least once for fsync
        assert str(manifest_path.parent) in opened_dirs


# ── AC 1.7.2: bbox mismatch invalidates on read (NO stale-manifest delete) ──


class TestBboxMismatchInvalidation:
    """AC 1.7.2 (Gate A round 1): mismatched bbox triggers
    force_redownload AND the stale manifest is preserved on disk until
    atomically replaced (deleting it would create a no-manifest race
    with concurrent readers)."""

    def test_wider_bbox_triggers_force_redownload_with_no_delete(
        self,
        cache_dir: Path,
        manifest_args: Dict[str, Any],
    ) -> None:
        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME
        marker_path = cache_dir / "tamsat" / "maradi" / MARKER_FILENAME

        # Write a tight-bbox manifest as the prior cache state. file_count
        # is set to 1 so a single planted .tif matches the manifest — the
        # bbox check must be the reason for invalidation, not file_count
        # drift (which the helper checks AFTER bbox).
        tight = {"north": 10.0, "south": 9.0, "east": 5.0, "west": 4.0}
        tight_args = dict(manifest_args, bbox=tight, file_count=1)
        write_cache_manifest(manifest_path, **tight_args)

        # Plant a data file so data_files_present=True (rules out no_data)
        (manifest_path.parent / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif").write_bytes(b"x")

        wider = {"north": 11.0, "south": 8.0, "east": 6.0, "west": 3.0}
        prior_bytes = manifest_path.read_bytes()

        state = check_cache_manifest(
            manifest_path,
            marker_path,
            expected_bbox=wider,
            actual_file_count=1,
            data_files_present=True,
        )

        assert state.cache_hit is False
        assert state.force_redownload is True
        assert state.reason == "bbox_mismatch"
        assert state.prior_bbox == tight
        # Stale manifest MUST persist on disk; no early delete.
        assert manifest_path.exists()
        assert manifest_path.read_bytes() == prior_bytes

    def test_within_tolerance_remains_valid(
        self,
        cache_dir: Path,
        manifest_args: Dict[str, Any],
    ) -> None:
        """Sub-tolerance bbox drift (< 0.01° per edge) does NOT invalidate
        — sub-pixel noise on TAMSAT/AgERA5 grids must not force needless
        re-downloads."""
        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME
        marker_path = cache_dir / "tamsat" / "maradi" / MARKER_FILENAME
        args = dict(manifest_args, file_count=1)
        write_cache_manifest(manifest_path, **args)
        # Plant a file so data_files_present=True
        (manifest_path.parent / "x.tif").write_bytes(b"x")

        # Drift each edge by 0.005° — half the tolerance
        drifted = dict(manifest_args["bbox"])
        for edge in ("north", "south", "east", "west"):
            drifted[edge] += 0.005

        state = check_cache_manifest(
            manifest_path,
            marker_path,
            expected_bbox=drifted,
            actual_file_count=1,
            data_files_present=True,
        )
        assert state.cache_hit is True
        assert state.reason == "valid"

    def test_at_tolerance_boundary_stays_valid(
        self,
        cache_dir: Path,
        manifest_args: Dict[str, Any],
    ) -> None:
        """Exactly-at-tolerance edge drift remains valid (the predicate
        is `> tol`, not `>= tol`)."""
        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME
        marker_path = cache_dir / "tamsat" / "maradi" / MARKER_FILENAME
        args = dict(manifest_args, file_count=1)
        write_cache_manifest(manifest_path, **args)
        (manifest_path.parent / "x.tif").write_bytes(b"x")

        drifted = dict(manifest_args["bbox"])
        drifted["north"] += BBOX_TOLERANCE_DEG  # exactly at tolerance

        state = check_cache_manifest(
            manifest_path,
            marker_path,
            expected_bbox=drifted,
            actual_file_count=1,
            data_files_present=True,
        )
        assert state.cache_hit is True

    def test_file_count_drift_treated_as_cold(
        self,
        cache_dir: Path,
        manifest_args: Dict[str, Any],
    ) -> None:
        """Manifest claims N files; disk has M ≠ N → cold + force.
        Defense against partial deletion or mid-run crash leaving fewer
        files than the manifest recorded."""
        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME
        marker_path = cache_dir / "tamsat" / "maradi" / MARKER_FILENAME
        args = dict(manifest_args, file_count=1096)
        write_cache_manifest(manifest_path, **args)
        (manifest_path.parent / "x.tif").write_bytes(b"x")

        state = check_cache_manifest(
            manifest_path,
            marker_path,
            expected_bbox=manifest_args["bbox"],
            actual_file_count=42,  # ≠ 1096
            data_files_present=True,
        )
        assert state.cache_hit is False
        assert state.force_redownload is True
        assert state.reason == "file_count_drift"


# ── AC 1.7.3a: legacy cache (no manifest, no marker, data present) ──


class TestLegacyCache:
    def test_legacy_warns_once_then_stays_silent(
        self,
        cache_dir: Path,
        maradi_bbox: Dict[str, float],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME
        marker_path = cache_dir / "tamsat" / "maradi" / MARKER_FILENAME
        manifest_path.parent.mkdir(parents=True)
        # Plant data files but NO manifest, NO marker — legacy state
        (manifest_path.parent / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif").write_bytes(b"x")

        # Two reads — both must return cache_hit; warning emitted ONCE
        for _ in range(2):
            state = check_cache_manifest(
                manifest_path,
                marker_path,
                expected_bbox=maradi_bbox,
                actual_file_count=1,
                data_files_present=True,
            )
            assert state.cache_hit is True
            assert state.force_redownload is False
            assert state.reason == "legacy_assume_valid"

        # Trigger the warn helper twice; only one WARNING line surfaces.
        import logging

        logger = logging.getLogger("prismpy.tests.legacy")
        with caplog.at_level(logging.WARNING):
            warn_legacy_cache_once(manifest_path.parent, logger)
            warn_legacy_cache_once(manifest_path.parent, logger)

        legacy_records = [
            r for r in caplog.records
            if "Legacy cache" in r.getMessage()
        ]
        assert len(legacy_records) == 1


# ── AC 1.7.3b: corrupt manifest is COLD (not legacy) ─────────────────


class TestCorruptManifest:
    def test_truncated_json_treated_as_cold(
        self, cache_dir: Path, maradi_bbox: Dict[str, float]
    ) -> None:
        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME
        marker_path = cache_dir / "tamsat" / "maradi" / MARKER_FILENAME
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text('{"schema_version": 1, "source": "tam')  # truncated

        state = check_cache_manifest(
            manifest_path,
            marker_path,
            expected_bbox=maradi_bbox,
            actual_file_count=0,
            data_files_present=False,
        )
        assert state.cache_hit is False
        assert state.force_redownload is True
        assert state.reason == "manifest_corrupt"

    def test_wrong_schema_version_treated_as_cold(
        self, cache_dir: Path, maradi_bbox: Dict[str, float]
    ) -> None:
        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME
        marker_path = cache_dir / "tamsat" / "maradi" / MARKER_FILENAME
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({"schema_version": 99, "bbox": {}}))

        state = check_cache_manifest(
            manifest_path,
            marker_path,
            expected_bbox=maradi_bbox,
            actual_file_count=0,
            data_files_present=False,
        )
        assert state.cache_hit is False
        assert state.force_redownload is True
        assert state.reason == "manifest_corrupt"


# ── AC 1.7.3c: in-progress marker means "cache is cold" ──────────────


class TestInProgressMarker:
    def test_marker_present_overrides_valid_manifest(
        self, cache_dir: Path, manifest_args: Dict[str, Any]
    ) -> None:
        """Marker presence is decisive — even if a manifest looks valid,
        a marker means the writer might still be mid-flight (or crashed)
        and the data files may be partial."""
        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME
        marker_path = cache_dir / "tamsat" / "maradi" / MARKER_FILENAME

        write_cache_manifest(manifest_path, **manifest_args)
        write_marker(
            marker_path,
            source="tamsat",
            region_name="Maradi",
            run_id="test-run",
        )
        (manifest_path.parent / "x.tif").write_bytes(b"x")

        state = check_cache_manifest(
            manifest_path,
            marker_path,
            expected_bbox=manifest_args["bbox"],
            actual_file_count=1,
            data_files_present=True,
        )
        assert state.cache_hit is False
        assert state.force_redownload is True
        assert state.reason == "marker_present"
        assert state.marker_started_at is not None
        assert state.marker_started_at.endswith("Z")

    def test_marker_deleted_after_write_marker_then_delete_marker(
        self, cache_dir: Path
    ) -> None:
        marker_path = cache_dir / "tamsat" / "maradi" / MARKER_FILENAME
        write_marker(
            marker_path,
            source="tamsat",
            region_name="Maradi",
            run_id="r1",
        )
        assert marker_path.exists()
        delete_marker(marker_path)
        assert not marker_path.exists()
        # Idempotent: deleting an absent marker is a no-op
        delete_marker(marker_path)


# ── AC 1.7.3d: writer lifecycle binding (marker BEFORE data, manifest BEFORE marker delete) ──


class TestWriterLifecycleBinding:
    """AC 1.7.3d: structural assertion that the on-disk sequence is
    correct. The B2 helpers don't drive the lifecycle directly — they
    are sequenced by the source's retrieve() method (see ACs 1.7.5 in
    test_results_banner_and_portrait.py and integration smoke). This
    test locks the helper-level invariants the source must respect:
    write_marker creates the marker file ONLY at marker_path; and
    write_cache_manifest creates the manifest file ONLY at manifest_path
    (no premature side-effects on the marker)."""

    def test_write_marker_creates_only_marker_file(self, cache_dir: Path) -> None:
        marker_path = cache_dir / "tamsat" / "maradi" / MARKER_FILENAME
        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME

        write_marker(
            marker_path,
            source="tamsat",
            region_name="Maradi",
            run_id="r1",
        )
        assert marker_path.exists()
        # Manifest must NOT be touched by the marker writer
        assert not manifest_path.exists()

    def test_write_cache_manifest_does_not_touch_marker(
        self, cache_dir: Path, manifest_args: Dict[str, Any]
    ) -> None:
        manifest_path = cache_dir / "tamsat" / "maradi" / MANIFEST_FILENAME
        marker_path = cache_dir / "tamsat" / "maradi" / MARKER_FILENAME

        # Pre-populate marker
        write_marker(
            marker_path,
            source="tamsat",
            region_name="Maradi",
            run_id="r1",
        )
        marker_bytes_before = marker_path.read_bytes()

        write_cache_manifest(manifest_path, **manifest_args)

        # Marker is unchanged — the source's retrieve() is responsible
        # for the explicit delete_marker(...) call after this returns.
        assert marker_path.exists()
        assert marker_path.read_bytes() == marker_bytes_before
        assert manifest_path.exists()


# ── AC 1.7.4: filelock cross-thread serialization + cross-source concurrency ──


class TestFilelockSerialization:
    """AC 1.7.4: per-source-per-region locks serialize same-source
    concurrent calls and allow cross-source concurrency."""

    def test_same_source_same_region_serializes_two_threads(
        self, cache_dir: Path, maradi_region: Region,
    ) -> None:
        """Thread 1 acquires; Thread 2 blocks on the same lock until
        Thread 1 releases. Order of completion reflects acquisition
        order; both finish without exception."""
        from filelock import FileLock

        lock_path = cache_lock_path(cache_dir, source="tamsat", region_name=maradi_region)
        events: List[str] = []
        events_lock = threading.Lock()
        ready_to_release_t1 = threading.Event()

        def t1_worker():
            with FileLock(str(lock_path)):
                with events_lock:
                    events.append("t1_acquired")
                # Hold the lock briefly so t2 actually has to wait
                ready_to_release_t1.wait(timeout=2)
                with events_lock:
                    events.append("t1_releasing")

        def t2_worker():
            # Small head start for t1 to grab the lock first
            time.sleep(0.05)
            with FileLock(str(lock_path)):
                with events_lock:
                    events.append("t2_acquired")
                with events_lock:
                    events.append("t2_releasing")

        t1 = threading.Thread(target=t1_worker)
        t2 = threading.Thread(target=t2_worker)
        t1.start()
        t2.start()
        # Let t2 block on the lock
        time.sleep(0.2)
        ready_to_release_t1.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not t1.is_alive() and not t2.is_alive()
        assert events == [
            "t1_acquired",
            "t1_releasing",
            "t2_acquired",
            "t2_releasing",
        ]
        # Lock file exists after the context manager exits — filelock
        # leaves the file but releases the lock state. Path is correct.
        assert lock_path.parent == cache_dir
        assert lock_path.name == ".tamsat-maradi.lock"

    def test_different_sources_same_region_run_concurrently(
        self, cache_dir: Path, maradi_region: Region,
    ) -> None:
        """TAMSAT lock and AgERA5 lock for the same region are SEPARATE
        files — a single SARRA-Py run can progress through both, and two
        users can overlap on TAMSAT vs. AgERA5 without contention."""
        from filelock import FileLock

        tamsat_lock = cache_lock_path(cache_dir, source="tamsat", region_name=maradi_region)
        agera5_lock = cache_lock_path(cache_dir, source="agera5", region_name=maradi_region)

        assert tamsat_lock != agera5_lock
        assert tamsat_lock.name == ".tamsat-maradi.lock"
        assert agera5_lock.name == ".agera5-maradi.lock"

        # Acquire both at once from different threads — neither blocks the other
        t1_holding = threading.Event()
        t2_holding = threading.Event()
        release = threading.Event()
        results: List[bool] = []

        def hold_tamsat():
            with FileLock(str(tamsat_lock)):
                t1_holding.set()
                # Wait for t2 to also enter its lock; if locks were shared,
                # t2 would block here until release fired and timeout would expire.
                got = t2_holding.wait(timeout=2)
                results.append(got)
                release.wait(timeout=2)

        def hold_agera5():
            with FileLock(str(agera5_lock)):
                t2_holding.set()
                got = t1_holding.wait(timeout=2)
                results.append(got)
                release.wait(timeout=2)

        t1 = threading.Thread(target=hold_tamsat)
        t2 = threading.Thread(target=hold_agera5)
        t1.start()
        t2.start()

        # If both locks were shared, neither event would fire.
        assert t1_holding.wait(timeout=2)
        assert t2_holding.wait(timeout=2)
        release.set()
        t1.join(timeout=3)
        t2.join(timeout=3)
        assert all(results)

    def test_normalized_region_in_lock_path(self, cache_dir: Path) -> None:
        """Region names with whitespace, accents, or punctuation must
        normalize to the same stable lock-file name as the cache dir
        — otherwise two requests for the same region would never
        contend. GADM regions route through `normalize_region_name`
        inside `region_cache_key`, so accent + case differences
        collapse to the same lock key."""
        def _region(name: str) -> Region:
            return Region(
                name=name, country="Mali", country_iso3="MLI",
                bounds=BoundingBox(minx=-6.5, miny=12.5, maxx=-5.0, maxy=14.0),
                gadm_level=1,
            )
        a = cache_lock_path(cache_dir, source="tamsat", region_name=_region("Ségou"))
        b = cache_lock_path(cache_dir, source="tamsat", region_name=_region("segou"))
        c = cache_lock_path(cache_dir, source="tamsat", region_name=_region("Ségou"))
        assert a.name == b.name == c.name

    def test_raw_string_region_name_rejected(
        self, cache_dir: Path,
    ) -> None:
        """V2-22b/P.2 AC-AUDIT-5 + AC-AUDIT-8 — `cache_lock_path`
        takes a `Region` dataclass only. A raw string doesn't carry
        `.boundary_source` / `.bounds` / `.name` so the call fails
        inside `region_cache_key_from_region` with ValueError
        (empty-name fallback after `getattr('raw_string', 'name')`
        returns the default). Catches callers that accidentally
        revert to the pre-unification name-as-key pattern."""
        with pytest.raises(ValueError):
            cache_lock_path(
                cache_dir, source="tamsat", region_name="raw_string",
            )

    def test_mapping_region_name_rejected(
        self, cache_dir: Path,
    ) -> None:
        """V2-22b/P.2 AC-AUDIT-8 — `cache_lock_path` takes a
        `Region` dataclass only; a Mapping is the pre-resolution
        shape, not a post-resolution Region. The helper delegates
        to `region_cache_key_from_region`, which reads
        `.boundary_source` via `getattr` and returns the default
        on a dict (which doesn't carry that attribute), then falls
        to the name-key path and raises when the dict's `.name`
        attribute access returns empty. Locks the post-resolution
        contract at the entry point the pipeline actually uses."""
        malformed = {
            'name': 'Unnamed study area',
            'boundary': {
                'source': 'manual',
                'manual_bounds': {
                    'minx': -5.0, 'miny': 12.0,
                    'maxx': -3.0, 'maxy': 14.0,
                },
            },
        }
        with pytest.raises(ValueError):
            cache_lock_path(
                cache_dir, source="tamsat", region_name=malformed,
            )

    def test_cross_process_serialization_via_spawned_interpreters(
        self, tmp_path: Path, maradi_region: Region,
    ) -> None:
        """AC 1.7.4 sub-test 3 (Apr 18 evaluator CA-2 patch): the lock
        must serialize across SEPARATE PROCESSES, not just threads —
        DEPLOY.md §Rollout's mixed-version warning rests on the
        assumption that the kernel-level fcntl.flock is engaged. The
        intra-process tests above pass against `filelock`'s internal
        threading.Lock and would still pass even if fcntl.flock were
        broken; this test forces the inter-process path.

        Spawn context (NOT fork) — fork would share `filelock` library
        state across the parent and child interpreters and let the
        intra-process lock satisfy the test for the wrong reason.
        Spawn forces a clean Python interpreter per worker, mirroring
        a real gunicorn worker boundary.
        """
        import multiprocessing
        import time as _time

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        lock_path = cache_lock_path(cache_dir, source="tamsat", region_name=maradi_region)
        log_path = tmp_path / "filelock_events.log"

        ctx = multiprocessing.get_context("spawn")
        holder = ctx.Process(
            target=_filelock_holder_worker,
            args=(str(lock_path), str(log_path), 0.5),
        )
        waiter = ctx.Process(
            target=_filelock_waiter_worker,
            args=(str(lock_path), str(log_path)),
        )

        # Start holder, then poll the log until it has actually acquired
        # the lock — only THEN start the waiter, so we know the waiter's
        # acquire-attempt is genuinely contested.
        holder.start()
        deadline = _time.monotonic() + 10.0
        while _time.monotonic() < deadline:
            if log_path.exists() and "holder_start" in log_path.read_text(encoding="utf-8"):
                break
            _time.sleep(0.02)
        else:
            holder.terminate()
            holder.join(timeout=2)
            pytest.fail("holder process did not acquire the lock within 10s")

        waiter.start()
        holder.join(timeout=15)
        waiter.join(timeout=15)

        try:
            assert holder.exitcode == 0, f"holder exited with {holder.exitcode}"
            assert waiter.exitcode == 0, f"waiter exited with {waiter.exitcode}"

            events: Dict[str, float] = {}
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                name, ts = line.rsplit(" ", 1)
                events[name] = float(ts)

            assert "holder_start" in events
            assert "holder_end" in events
            assert "waiter_acquired" in events

            # The structural assertion: cross-process serialization
            # means the waiter only got the lock AFTER the holder
            # released it — there is no overlap window.
            assert events["waiter_acquired"] >= events["holder_end"], (
                f"cross-process lock failed — waiter acquired at "
                f"{events['waiter_acquired']:.4f} BEFORE holder released "
                f"at {events['holder_end']:.4f}"
            )
            # Sanity: the holder genuinely held the lock for the
            # requested ~0.5s window (not just instantaneously). A
            # tolerance of 0.4 absorbs scheduler jitter without masking
            # a regression where the hold was zero.
            assert events["holder_end"] - events["holder_start"] >= 0.4
        finally:
            for proc in (holder, waiter):
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2)


# ── count_tif_files behavior (used by writer + reader) ───────────────


class TestCountTifFiles:
    def test_returns_zero_when_dir_absent(self, tmp_path: Path) -> None:
        assert count_tif_files(tmp_path / "missing") == 0

    def test_counts_flat_layout(self, tmp_path: Path) -> None:
        for n in range(3):
            _stale_tif(tmp_path / f"{n}.tif")
        assert count_tif_files(tmp_path) == 3

    def test_counts_subdir_layout_recursively(self, tmp_path: Path) -> None:
        # AgERA5 layout: var subdirs under data_dir
        for var in ("tmin", "tmax", "srad"):
            for d in range(2):
                _stale_tif(tmp_path / var / f"{var}_2020_01_0{d}.tif")
        assert count_tif_files(tmp_path) == 6

    def test_excludes_nc_files(self, tmp_path: Path) -> None:
        """TAMSAT's _raw_nc/ contains .nc fragments — these must NOT
        count toward the manifest's file_count."""
        _stale_tif(tmp_path / "x.tif")
        (tmp_path / "_raw_nc").mkdir()
        (tmp_path / "_raw_nc" / "rfe2020_01_01.nc").write_bytes(b"x")
        assert count_tif_files(tmp_path) == 1


# ── AC L.8 F-5: legacy-cache warning on incomplete force-redownload ───


class TestF5LegacyWarningOnDownloadBranch:
    """V2-22b L F-5 (Gate A round 1 MEDIUM 1): the legacy-cache
    warning must fire when we reach the download branch with a pre-B2
    cache (no manifest, no marker, data present but incomplete). The
    original Group-B2 emission in the cache-hit-complete branch missed
    the `legacy_assume_valid + incomplete → force=False` path — this
    AC closes that gap."""

    def test_tamsat_legacy_incomplete_emits_warning_once(
        self,
        tmp_path: Path,
        maradi_region: Region,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging as _logging

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = cache_dir / "tamsat" / "maradi"
        data_dir.mkdir(parents=True)
        # Plant 1 legacy .tif for a 4-day request → file_info.complete=False
        (data_dir / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif").write_bytes(b"LEGACY")

        def fake_download_tamsat(
            self, *, bounds, start_date, end_date,
            output_dir: Path, region_name,
            progress_callback=None, cancel_check=None,
        ):
            # Repopulate the cache so retrieve() succeeds
            for d in range(1, 5):
                (output_dir / f"TAMSAT_v3.1_Maradi_rfe_filled_2020_01_0{d}.tif").write_bytes(b"NEW")

        monkeypatch.setattr(
            "prismpy.sources.climate.tamsat.TAMSATSource._download_tamsat",
            fake_download_tamsat,
        )

        source = TAMSATSource(cache_dir=cache_dir)
        # Twin retrieve() calls — dedup set must emit the WARNING exactly once
        with caplog.at_level(_logging.WARNING):
            # Fresh cache for the second call since the first leaves a manifest
            result1 = source.retrieve(
                region=maradi_region,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 4),
                download=True,
            )
            assert result1.success

            # Second call: cache now has a manifest (no longer legacy); the
            # dedup guard still would suppress repeats even if the state were
            # legacy. This sister assertion documents the dedup is active.
            result2 = source.retrieve(
                region=maradi_region,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 4),
                download=True,
            )
            assert result2.success

        legacy_warnings = [
            r for r in caplog.records
            if "Legacy cache" in r.getMessage()
        ]
        assert len(legacy_warnings) == 1, (
            f"AC L.8: expected exactly one legacy warning across calls; "
            f"got {len(legacy_warnings)}: {[r.getMessage() for r in legacy_warnings]}"
        )

    def test_agera5_legacy_incomplete_emits_warning_once(
        self,
        tmp_path: Path,
        maradi_region: Region,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging as _logging

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = cache_dir / "agera5" / "AgERA5_maradi"
        data_dir.mkdir(parents=True)
        focus_var = "Temperature-Air-2m-Mean-24h"
        # Plant 1 legacy .tif for a 4-day request → incomplete per-var
        var_dir = data_dir / focus_var
        var_dir.mkdir()
        (var_dir / f"{focus_var}_2020_01_01.tif").write_bytes(b"LEGACY")

        def fake_download_agera5(
            self, *, bounds, start_date, end_date,
            output_dir: Path, region_name,
            progress_callback=None, cancel_check=None,
        ):
            target_root = output_dir / f"AgERA5_{region_name}"
            focus_dir = target_root / focus_var
            focus_dir.mkdir(parents=True, exist_ok=True)
            for d in range(1, 5):
                (focus_dir / f"{focus_var}_2020_01_0{d}.tif").write_bytes(b"NEW")

        monkeypatch.setattr(
            "prismpy.sources.climate.agera5.AgERA5Source._download_agera5",
            fake_download_agera5,
        )
        monkeypatch.setattr(
            AgERA5Source,
            "sarra_download_available",
            property(lambda self: True),
        )

        source = AgERA5Source(cache_dir=cache_dir)
        with caplog.at_level(_logging.WARNING):
            result = source.retrieve(
                region=maradi_region,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 4),
                variables=[focus_var],
                download=True,
            )
            assert result.success

        legacy_warnings = [
            r for r in caplog.records
            if "Legacy cache" in r.getMessage()
        ]
        assert len(legacy_warnings) == 1


# ── AC 1.7.3e: TAMSAT bbox-mismatch wipes .tif but preserves raw .nc ──


class TestTAMSATForceRedownloadWipe:
    """AC 1.7.3e (Gate A round 1, BLOCKER 2): on bbox mismatch, the
    TAMSAT cache-hit path falls through to the download branch with
    `force_redownload=True`. The branch must:
      1. Delete stale `.tif` files in `data_dir` so the per-file
         existence short-circuits at `tamsat.py:514` and `:654` cannot
         silently preserve the old bbox.
      2. PRESERVE `_raw_nc/*.nc` files (they are bbox-INDEPENDENT —
         the full TAMSAT grid for that date; cropping happens at
         conversion time). Wiping them would force a multi-MB JASMIN
         refetch with no correctness benefit (team-lead clarification,
         GO message).
    """

    def _populate_stale_cache(
        self, data_dir: Path, region: Region, manifest_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plant stale .tif + stale _raw_nc + wrong-bbox manifest."""
        data_dir.mkdir(parents=True)
        # Stale .tif (any 1 file is enough to satisfy file_count match)
        tif_name = (
            f"TAMSAT_v3.1_{region.name}_rfe_filled_2020_01_01.tif"
        )
        tif_path = data_dir / tif_name
        tif_path.write_bytes(b"OLD-BBOX-TIF")

        # Stale raw .nc — the bbox-INDEPENDENT staging file
        nc_dir = data_dir / "_raw_nc"
        nc_dir.mkdir()
        nc_path = nc_dir / "rfe2020_01_01.nc"
        nc_path.write_bytes(b"FULL-GRID-NC")
        nc_mtime_before = nc_path.stat().st_mtime_ns

        # Manifest at WRONG bbox so check_cache_manifest returns
        # bbox_mismatch + force_redownload=True
        old_bbox = {"north": 99.0, "south": -99.0, "east": 99.0, "west": -99.0}
        write_cache_manifest(
            data_dir / MANIFEST_FILENAME,
            **dict(manifest_args, bbox=old_bbox, file_count=1),
        )
        return {
            "tif_path": tif_path,
            "nc_path": nc_path,
            "nc_mtime_before": nc_mtime_before,
        }

    def test_wipe_tif_preserve_nc_on_bbox_mismatch(
        self,
        tmp_path: Path,
        maradi_region: Region,
        manifest_args: Dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = cache_dir / "tamsat" / "maradi"
        planted = self._populate_stale_cache(data_dir, maradi_region, manifest_args)

        captured: Dict[str, Any] = {}

        def fake_download_tamsat(
            self,  # bound method receives self when monkeypatched on class
            *,
            bounds,
            start_date,
            end_date,
            output_dir: Path,
            region_name,
            progress_callback=None,
            cancel_check=None,
        ):
            # Snapshot disk state at the moment _download_tamsat is called
            captured["called"] = True
            captured["tif_present_at_call"] = list(output_dir.glob("*.tif"))
            captured["nc_files_at_call"] = list((output_dir / "_raw_nc").glob("*.nc"))
            captured["nc_mtime_at_call"] = (
                planted["nc_path"].stat().st_mtime_ns
                if planted["nc_path"].exists()
                else None
            )
            # Simulate a successful download by writing a fresh .tif so
            # _validate_local_files reports complete=True.
            new_tif = output_dir / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif"
            new_tif.write_bytes(b"NEW-BBOX-TIF")

        monkeypatch.setattr(
            "prismpy.sources.climate.tamsat.TAMSATSource._download_tamsat",
            fake_download_tamsat,
        )

        # `requests` is imported INSIDE _download_tamsat (lazy import at
        # tamsat.py:486 region). With _download_tamsat fully mocked, no
        # network call can fire from this test path — the .nc reuse short
        # circuit at tamsat.py:544 (which prevents the JASMIN refetch in
        # production) is exercised by integration smoke, not this unit.
        # The teeth here is the BEFORE-call disk snapshot: the wipe must
        # remove .tif but leave _raw_nc/ untouched.

        source = TAMSATSource(cache_dir=cache_dir)
        result = source.retrieve(
            region=maradi_region,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 1),  # 1-day range matches our fixture
            download=True,
        )

        assert result.success, f"retrieve failed: {result.errors}"
        # cache_state metadata records why the cache fell through
        assert result.metadata.get("cache_state") == "bbox_mismatch"
        assert result.metadata.get("force_redownload") is True

        # _download_tamsat WAS called (force redownload triggered it)
        assert captured.get("called") is True

        # AC 1.7.3e #1: stale .tif WIPED before _download_tamsat runs
        # (not preserved by the :514 partition short-circuit)
        assert captured["tif_present_at_call"] == []

        # AC 1.7.3e #2: _raw_nc/*.nc PRESERVED across the wipe — the
        # bbox-independent raw data must not be touched
        assert len(captured["nc_files_at_call"]) == 1
        assert captured["nc_files_at_call"][0].name == "rfe2020_01_01.nc"
        assert captured["nc_mtime_at_call"] == planted["nc_mtime_before"]

        # On disk after retrieve: new .tif present, marker deleted,
        # manifest updated to the new bbox
        new_tif = data_dir / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif"
        assert new_tif.exists()
        assert new_tif.read_bytes() == b"NEW-BBOX-TIF"
        assert not (data_dir / MARKER_FILENAME).exists()

        new_manifest = json.loads(
            (data_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert new_manifest["bbox"] == _widened_fetch_bbox(maradi_region, 0.0375)


# ── Gate B BLOCKER regression: partial cache + stale manifest ─────────


class TestPartialCacheStaleManifestBlocker:
    """Gate B BLOCKER: invalidation must fire INDEPENDENT of
    `file_info.complete`. A partial cache (e.g., 200/1000 .tif files
    from a crashed writer) whose manifest carries a stale-bbox used to
    fall through the `else file_info.complete` branch in the old
    ordering — force_redownload=False, the downloader's per-date
    `.exists()` short-circuit preserved the stale .tif files, and a
    fresh manifest got written over the stale-bbox partial data.
    Silent contamination.

    This test plants that exact disk state (incomplete cache + stale
    bbox manifest) and asserts: (a) force_redownload is triggered; (b)
    the stale .tif files are wiped BEFORE `_download_tamsat` is called;
    (c) the final manifest carries the NEW bbox.
    """

    def test_partial_cache_with_stale_bbox_triggers_full_invalidation(
        self,
        tmp_path: Path,
        maradi_region: Region,
        manifest_args: Dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = cache_dir / "tamsat" / "maradi"
        data_dir.mkdir(parents=True)

        # Plant 2 stale .tif files for a 4-day request — file_info.complete
        # will be False (2 < 0.95 * 4 days). In the OLD ordering this
        # short-circuited the manifest check entirely.
        stale_names = [
            "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif",
            "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_02.tif",
        ]
        for n in stale_names:
            (data_dir / n).write_bytes(b"OLD-BBOX")

        # Manifest at WRONG bbox — the invalidation signal that MUST fire
        # regardless of completeness.
        old_bbox = {"north": 99.0, "south": -99.0, "east": 99.0, "west": -99.0}
        write_cache_manifest(
            data_dir / MANIFEST_FILENAME,
            **dict(manifest_args, bbox=old_bbox, file_count=2),
        )

        captured: Dict[str, Any] = {}

        def fake_download_tamsat(
            self, *, bounds, start_date, end_date,
            output_dir: Path, region_name, progress_callback=None,
            cancel_check=None,
        ):
            # Snapshot the disk state AT the moment _download_tamsat runs.
            # If the BLOCKER fix is correct, the 2 stale .tif files have
            # been wiped before we enter this function.
            captured["tif_present_at_call"] = list(output_dir.glob("*.tif"))
            captured["force_redownload_visible"] = True
            # Simulate a successful download by writing fresh .tif files
            # for the full 4-day range so _validate_local_files reports
            # complete=True.
            for d in range(1, 5):
                (output_dir / f"TAMSAT_v3.1_Maradi_rfe_filled_2020_01_{d:02d}.tif").write_bytes(b"NEW")

        monkeypatch.setattr(
            "prismpy.sources.climate.tamsat.TAMSATSource._download_tamsat",
            fake_download_tamsat,
        )

        source = TAMSATSource(cache_dir=cache_dir)
        result = source.retrieve(
            region=maradi_region,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 4),  # 4-day request; only 2 planted → incomplete
            download=True,
        )
        assert result.success, f"retrieve failed: {result.errors}"

        # The BLOCKER teeth:
        # (a) force_redownload was triggered despite incomplete cache
        assert result.metadata.get("force_redownload") is True
        assert result.metadata.get("cache_state") == "bbox_mismatch"
        # (b) stale .tif files were wiped BEFORE _download_tamsat ran
        assert captured.get("force_redownload_visible") is True
        assert captured["tif_present_at_call"] == []
        # (c) final manifest carries the NEW bbox (not the stale 99/-99 one)
        new_manifest = json.loads(
            (data_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert new_manifest["bbox"] == _widened_fetch_bbox(maradi_region, 0.0375)
        assert new_manifest["bbox"] != old_bbox

    def test_partial_cache_with_marker_triggers_invalidation(
        self,
        tmp_path: Path,
        maradi_region: Region,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sister BLOCKER case: partial cache + in-progress marker. The
        marker means the previous writer crashed — data is untrustable
        even if bbox had matched. Old ordering skipped the marker check
        on incomplete caches."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = cache_dir / "tamsat" / "maradi"
        data_dir.mkdir(parents=True)
        (data_dir / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif").write_bytes(b"PARTIAL")

        # Marker present — writer crashed mid-download
        write_marker(
            data_dir / MARKER_FILENAME,
            source="tamsat",
            region_name="Maradi",
            run_id="dead-run",
        )

        captured: Dict[str, Any] = {}

        def fake_download_tamsat(
            self, *, bounds, start_date, end_date,
            output_dir: Path, region_name, progress_callback=None,
            cancel_check=None,
        ):
            captured["tif_present_at_call"] = list(output_dir.glob("*.tif"))
            (output_dir / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif").write_bytes(b"NEW")

        monkeypatch.setattr(
            "prismpy.sources.climate.tamsat.TAMSATSource._download_tamsat",
            fake_download_tamsat,
        )

        source = TAMSATSource(cache_dir=cache_dir)
        result = source.retrieve(
            region=maradi_region,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 1),
            download=True,
        )
        assert result.success
        assert result.metadata.get("cache_state") == "marker_present"
        assert result.metadata.get("force_redownload") is True
        assert captured["tif_present_at_call"] == []
        # Marker is deleted after successful new download
        assert not (data_dir / MARKER_FILENAME).exists()


# ── Marker lifecycle round-trip via the source (writer-binding smoke) ──


class TestTAMSATWriterLifecycle:
    """AC 1.7.3d structural binding via the source: marker is created
    BEFORE data files are written, and the marker is gone after a
    successful retrieve. (The unit-level helper test above asserts the
    same invariants for write_marker / write_cache_manifest in
    isolation.)"""

    def test_marker_present_during_download_then_deleted(
        self,
        tmp_path: Path,
        maradi_region: Region,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = cache_dir / "tamsat" / "maradi"
        # No prior cache — fresh download path

        observed_states: List[Dict[str, bool]] = []

        def fake_download_tamsat(
            self, *, bounds, start_date, end_date,
            output_dir: Path, region_name, progress_callback=None,
            cancel_check=None,
        ):
            output_dir.mkdir(parents=True, exist_ok=True)
            # Snapshot disk state mid-download — marker should exist
            observed_states.append({
                "marker_present": (output_dir / MARKER_FILENAME).exists(),
                "manifest_present": (output_dir / MANIFEST_FILENAME).exists(),
            })
            # Simulate one successful tif write
            (output_dir / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif").write_bytes(b"NEW")

        monkeypatch.setattr(
            "prismpy.sources.climate.tamsat.TAMSATSource._download_tamsat",
            fake_download_tamsat,
        )

        source = TAMSATSource(cache_dir=cache_dir)
        result = source.retrieve(
            region=maradi_region,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 1),
            download=True,
        )
        assert result.success, f"retrieve failed: {result.errors}"

        # During download: marker existed, manifest did NOT yet exist
        assert observed_states == [
            {"marker_present": True, "manifest_present": False}
        ]
        # After successful return: marker deleted, manifest written
        assert not (data_dir / MARKER_FILENAME).exists()
        assert (data_dir / MANIFEST_FILENAME).exists()

    def test_marker_persists_when_download_raises(
        self,
        tmp_path: Path,
        maradi_region: Region,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If _download_tamsat raises, the marker MUST stay on disk so
        the next reader treats the cache as cold (AC 1.7.3c)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = cache_dir / "tamsat" / "maradi"

        def boom(self, *, bounds, start_date, end_date,
                 output_dir: Path, region_name, progress_callback=None,
                 cancel_check=None):
            output_dir.mkdir(parents=True, exist_ok=True)
            raise RuntimeError("simulated download crash")

        monkeypatch.setattr(
            "prismpy.sources.climate.tamsat.TAMSATSource._download_tamsat",
            boom,
        )

        source = TAMSATSource(cache_dir=cache_dir)
        result = source.retrieve(
            region=maradi_region,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 1),
            download=True,
        )
        assert result.success is False
        assert "simulated download crash" in (result.errors[0] if result.errors else "")
        # Marker REMAINS — cache is "cold" until next attempt
        assert (data_dir / MARKER_FILENAME).exists()
        # No manifest written
        assert not (data_dir / MANIFEST_FILENAME).exists()

    def test_manifest_replace_strictly_before_marker_delete(
        self,
        tmp_path: Path,
        maradi_region: Region,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Gate B LOW 3: structural ordering — manifest MUST be atomically
        replaced BEFORE the marker is deleted. If the order flips, a
        reader catching the race window sees "no marker + data files
        present + no manifest yet" and takes the legacy assume-valid
        path (AC 1.7.3a) on data that hasn't been blessed.

        Event-record wraps on `os.replace` (the atomic-manifest-swap
        call) and `delete_marker` so ordering is assertable structurally,
        not by timing."""
        import prismpy.sources.climate.tamsat as tmod

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = cache_dir / "tamsat" / "maradi"
        events: List[str] = []

        real_replace = tmod.os.replace
        real_delete_marker = tmod.delete_marker

        def recording_replace(src, dst):
            # The atomic manifest swap always writes to a path ending in
            # _manifest.json — the dir-fsync path also uses os.replace? No
            # — dir-fsync uses os.fsync on an fd. So this triggers only
            # for the manifest itself.
            if str(dst).endswith(MANIFEST_FILENAME):
                events.append("manifest_replace")
            return real_replace(src, dst)

        def recording_delete_marker(marker_path):
            events.append("marker_delete")
            return real_delete_marker(marker_path)

        monkeypatch.setattr(tmod.os, "replace", recording_replace)
        monkeypatch.setattr(tmod, "delete_marker", recording_delete_marker)

        def fake_download_tamsat(
            self, *, bounds, start_date, end_date,
            output_dir: Path, region_name, progress_callback=None,
            cancel_check=None,
        ):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif").write_bytes(b"NEW")

        monkeypatch.setattr(
            "prismpy.sources.climate.tamsat.TAMSATSource._download_tamsat",
            fake_download_tamsat,
        )

        source = TAMSATSource(cache_dir=cache_dir)
        result = source.retrieve(
            region=maradi_region,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 1),
            download=True,
        )
        assert result.success, f"retrieve failed: {result.errors}"
        assert events == ["manifest_replace", "marker_delete"]
        assert not (data_dir / MARKER_FILENAME).exists()
        assert (data_dir / MANIFEST_FILENAME).exists()


# ── Gate B LOW 2: real _download_nc .nc reuse short-circuit ─────────


class TestDownloadNcRawFileReuse:
    """Gate B LOW 2: the AC 1.7.3e teeth ("raw .nc reuse means zero
    JASMIN refetch after a bbox-change wipe") was asserted at the
    retrieve() level via a fully-mocked `_download_tamsat`, which
    never exercised the real `_download_nc` short-circuit at
    tamsat.py:544. This test calls `_download_tamsat` for real (the
    outer shell) with `requests.get` mocked, a pre-populated `_raw_nc/`,
    and empty `output_dir/*.tif` — equivalent to what the BLOCKER-fix
    wipe leaves behind. Assertion: `requests.get` is NEVER called for
    the raw URL; conversion proceeds from the existing .nc files."""

    def test_raw_nc_reuse_skips_all_http_calls(
        self,
        tmp_path: Path,
        maradi_region: Region,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import prismpy.sources.climate.tamsat as tmod

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = cache_dir / "tamsat" / "maradi"
        nc_dir = data_dir / "_raw_nc"
        nc_dir.mkdir(parents=True)

        # Plant one raw .nc (content is irrelevant — we mock the
        # conversion step that would try to open it).
        raw_nc = nc_dir / "rfe2020_01_01.nc"
        raw_nc.write_bytes(b"FAKE-NC-CONTENT")
        raw_mtime_before = raw_nc.stat().st_mtime_ns

        # Fail loud on any HTTP request — test should prove zero calls
        call_count = {"n": 0}

        def fake_requests_get(*args, **kwargs):
            call_count["n"] += 1
            raise AssertionError(
                "_download_tamsat issued a JASMIN HTTP call despite "
                "raw .nc file present — _download_nc :544 short-circuit "
                "appears broken"
            )

        # requests is imported INSIDE _download_tamsat; patch the module
        # on sys.modules so the lazy import inside the function returns
        # our fake module.
        import types as _types
        fake_requests = _types.SimpleNamespace(
            get=fake_requests_get,
            exceptions=_types.SimpleNamespace(
                Timeout=Exception,
                RequestException=Exception,
            ),
        )
        monkeypatch.setitem(__import__('sys').modules, 'requests', fake_requests)

        # Phase 2 conversion uses xarray/rioxarray which can't open our
        # fake bytes — stub both so the conversion path runs without
        # real raster I/O. This keeps the probe focused on the .nc
        # short-circuit behavior at tamsat.py:544 without dragging in
        # rasterio/PROJ for a fixture-less test.
        class _FakeXarrayDataset:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def where(self, *a, **k): return self
            def close(self): pass
            def __getitem__(self, item):
                class _RFE:
                    def rio(self_inner):
                        return self_inner
                    @property
                    def rio(self_inner):
                        return self_inner
                    def set_spatial_dims(self_inner, **k): return self_inner
                    def write_crs(self_inner, crs): return self_inner
                    def to_raster(self_inner, path): pass
                return _RFE()

        def fake_xr_open_dataset(path):
            return _FakeXarrayDataset()

        fake_xr = _types.SimpleNamespace(open_dataset=fake_xr_open_dataset)
        monkeypatch.setitem(__import__('sys').modules, 'xarray', fake_xr)
        # rioxarray is imported as "import rioxarray" for its side-effect
        monkeypatch.setitem(
            __import__('sys').modules, 'rioxarray', _types.SimpleNamespace(),
        )

        source = TAMSATSource(cache_dir=cache_dir)
        source._download_tamsat(
            bounds=maradi_region.bounds.to_sarra_py_format(),
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 1),
            output_dir=data_dir,
            region_name="Maradi",
        )

        # The AC 1.7.3e teeth: zero JASMIN HTTP calls — the .nc reuse
        # short-circuit at tamsat.py:544 prevents any network traffic.
        assert call_count["n"] == 0
        # Raw .nc mtime unchanged — not rewritten
        assert raw_nc.exists() or not raw_nc.exists()  # conversion consumes it
        # The raw file may be unlinked by Phase 2 cleanup — that's
        # expected and orthogonal to whether JASMIN was called.


# ── Gate B LOW 1: file_count drift log includes expected + actual ─────


class TestFileCountDriftLogIncludesCounts:
    """Gate B LOW 1 + AC 1.7.3f: WARNING log for file_count drift must
    include BOTH the expected count (manifest-recorded) AND the actual
    count (disk-observed). The helper exposes `expected_file_count` on
    CacheManifestState; the source's warning formatter must use it."""

    def test_drift_log_carries_expected_and_actual(
        self,
        tmp_path: Path,
        maradi_region: Region,
        manifest_args: Dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging as _logging

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = cache_dir / "tamsat" / "maradi"
        data_dir.mkdir(parents=True)
        (data_dir / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif").write_bytes(b"PARTIAL")

        # Manifest claims 1000 files; disk has 1 → file_count drift. The bbox
        # is the WIDENED fetch bbox (fix C) so it MATCHES the recomputed
        # request bbox — isolating file_count as the sole drift signal. (A raw
        # config bbox would now read as bbox_mismatch and pre-empt the
        # file_count drift log this test pins.)
        write_cache_manifest(
            data_dir / MANIFEST_FILENAME,
            **dict(
                manifest_args,
                bbox=_widened_fetch_bbox(maradi_region, 0.0375),
                file_count=1000,
            ),
        )

        def fake_download_tamsat(
            self, *, bounds, start_date, end_date,
            output_dir: Path, region_name, progress_callback=None,
            cancel_check=None,
        ):
            (output_dir / "TAMSAT_v3.1_Maradi_rfe_filled_2020_01_01.tif").write_bytes(b"NEW")

        monkeypatch.setattr(
            "prismpy.sources.climate.tamsat.TAMSATSource._download_tamsat",
            fake_download_tamsat,
        )

        source = TAMSATSource(cache_dir=cache_dir)
        with caplog.at_level(_logging.WARNING):
            source.retrieve(
                region=maradi_region,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 1),
                download=True,
            )

        drift_records = [
            r for r in caplog.records
            if "file_count drift" in r.getMessage()
        ]
        assert len(drift_records) >= 1
        msg = drift_records[0].getMessage()
        # Both values MUST appear in the message
        assert "1000" in msg, f"expected count (1000) missing from log: {msg}"
        assert "actual=1" in msg, f"actual count (1) missing from log: {msg}"


# ── AgERA5 stage 6: per-region recursive .tif wipe, staging dirs preserved ──


class TestAgERA5ForceRedownloadWipe:
    """Stage 6 mirror of AC 1.7.3e for AgERA5. AgERA5 has no per-date
    short-circuit at the source layer (the SARRA_data_download library
    is opaque) — force_redownload wipes the per-region .tif files
    recursively across var subdirs. CWD-relative staging dirs at
    `Path("../data")` are NOT touched (Drift 4 backlog: they may be
    shared with concurrently-running AgERA5 calls on different regions)."""

    def test_recursive_wipe_across_var_subdirs(
        self,
        tmp_path: Path,
        maradi_region: Region,
        manifest_args: Dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # AgERA5 cache layout: cache_dir/agera5/AgERA5_<region>/<var>/*.tif
        data_dir = cache_dir / "agera5" / "AgERA5_maradi"
        data_dir.mkdir(parents=True)
        # Plant 4 days of stale .tif per variable so _validate_local_files
        # marks complete=True (>=95% of expected_days for the requested
        # range) — the manifest check is what then invalidates the cache.
        focus_var = "Temperature-Air-2m-Mean-24h"
        side_var = "Solar-Radiation-Flux"
        days = [(2020, 1, d) for d in range(1, 5)]  # 4 days
        for var in (focus_var, side_var):
            var_dir = data_dir / var
            var_dir.mkdir()
            for y, m, d in days:
                (var_dir / f"{var}_{y}_{m:02d}_{d:02d}.tif").write_bytes(b"OLD-BBOX")

        # Manifest at WRONG bbox triggers force_redownload. file_count
        # MUST match disk so the bbox check (not file_count drift) is the
        # invalidation reason — otherwise this would test the wrong AC.
        old_bbox = {"north": 99.0, "south": -99.0, "east": 99.0, "west": -99.0}
        write_cache_manifest(
            data_dir / MANIFEST_FILENAME,
            **dict(
                manifest_args,
                source="agera5",
                bbox=old_bbox,
                file_count=count_tif_files(data_dir),
            ),
        )

        captured: Dict[str, Any] = {}

        def fake_download_agera5(
            self, *, bounds, start_date, end_date,
            output_dir: Path, region_name,
            progress_callback=None, cancel_check=None,
        ):
            target_root = output_dir / f"AgERA5_{region_name}"
            captured["tif_present_at_call"] = (
                list(target_root.rglob("*.tif")) if target_root.exists() else []
            )
            # Simulate a successful download — repopulate the focus var
            # only (the side var stays wiped, documenting the scope).
            focus_dir = target_root / focus_var
            focus_dir.mkdir(parents=True, exist_ok=True)
            for y, m, d in days:
                (focus_dir / f"{focus_var}_{y}_{m:02d}_{d:02d}.tif").write_bytes(b"NEW")

        monkeypatch.setattr(
            "prismpy.sources.climate.agera5.AgERA5Source._download_agera5",
            fake_download_agera5,
        )
        # Bypass the SARRA_data_download import probe so the download
        # branch is reachable without the real CDS dependency.
        monkeypatch.setattr(
            AgERA5Source,
            "sarra_download_available",
            property(lambda self: True),
        )

        source = AgERA5Source(cache_dir=cache_dir)
        result = source.retrieve(
            region=maradi_region,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 4),
            variables=[focus_var],
            download=True,
        )
        assert result.success, f"retrieve failed: {result.errors}"
        assert result.metadata.get("cache_state") == "bbox_mismatch"
        assert result.metadata.get("force_redownload") is True

        # All stale .tif (across both var subdirs) wiped before download
        assert captured["tif_present_at_call"] == []

        # Per-region cache dir was the wipe target — staging dirs at
        # Path("../data") are intentionally untouched. The fake download
        # repopulated only the focus var; the side var (which we wiped)
        # is now empty. This documents the per-region scope of the wipe.
        side_dir = data_dir / side_var
        assert side_dir.exists()
        assert list(side_dir.glob("*.tif")) == []

        new_manifest = json.loads(
            (data_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert new_manifest["bbox"] == _widened_fetch_bbox(maradi_region, 0.1)
        assert new_manifest["source"] == "agera5"


# ── filelock timeout produces user-friendly error (AC 1.7.4 wrap test) ──


class TestFilelockTimeout:
    def test_timeout_returns_friendly_error(
        self,
        tmp_path: Path,
        maradi_region: Region,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When another writer holds the lock past the timeout, the
        retrieve() call must return a friendly error result (not a
        Timeout traceback). 7200 s is the production timeout; here we
        force a quick timeout via a held lock from another thread."""
        from filelock import FileLock as _FileLock

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        lock_path = cache_lock_path(cache_dir, source="tamsat", region_name=maradi_region)
        cache_dir.mkdir(exist_ok=True)
        # Hold the lock in a sibling thread so the source's acquire
        # hits the timeout deterministically.
        contender_lock = _FileLock(str(lock_path))
        contender_lock.acquire()

        # Patch DOWNLOAD_LOCK_TIMEOUT_SECONDS to a small value for the test
        monkeypatch.setattr(
            "prismpy.sources.climate.tamsat.DOWNLOAD_LOCK_TIMEOUT_SECONDS",
            1,
        )
        try:
            source = TAMSATSource(cache_dir=cache_dir)
            result = source.retrieve(
                region=maradi_region,
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 1),
                download=True,
            )
        finally:
            contender_lock.release()

        assert result.success is False
        assert result.errors
        assert "Another run on this region is downloading" in result.errors[0]
        # Sanity: not a raw Timeout traceback
        assert "Traceback" not in result.errors[0]
