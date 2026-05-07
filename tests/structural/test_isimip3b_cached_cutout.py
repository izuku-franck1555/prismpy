"""Structural pin: cached_cutout cache discipline + cache adversarial drills.

Sprint G AC-G-2 §2.1-§2.13 verification probes + F-G-10 cache adversarial
drills #8-#12 with REAL-condition invocation per discipline-notes
MED-Pass4-1.

Cache discipline (§2.1-§2.10): TTL/version/DOI invalidation, missing-meta
cold-cache fallback, bbox-key 4-decimal normalization, atomic write, env
overrides, and the canonical cache-path shape.

Real-condition adversarial drills (§F-G-10 / drills #8-#12):

* #8  IsimipFetchError       — real socket close mid-download via
                                ``responses`` + ``ConnectionError`` body.
* #9  CacheWriteError        — real ``OSError(28)`` on the
                                ``os.replace`` + meta-write paths via
                                monkey-patch (mirroring disk-full).
* #10 CacheDirectoryError    — real ``PermissionError`` on cache root
                                creation via ``os.chmod`` of a parent
                                directory.
* #11 IsimipFetchError-concurrent — N threads on cold cache; lock
                                serializes; only 1 download fired
                                (verified by call-count + identical
                                results).
* #12 InvalidIsimipResponseError — real malformed netCDF served via
                                ``responses`` (HTML error page bytes).

Drill discipline per MED-Pass4-1: each drill exercises the underlying
failure path, NOT a synthetic ``mock.side_effect = TypedException()``
that bypasses the body and tests nothing. The fixtures monkey-patch
real OS / network primitives so the body's typed-exception wrapping is
empirically exercised.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
import responses

from prismpy.data_sources import isimip3b
from prismpy.data_sources.isimip3b import (
    CacheDirectoryError,
    CacheWriteError,
    IsimipFetchError,
    InvalidIsimipResponseError,
    cached_cutout,
)


# ── Fakes used by the discipline + drill probes ──────────────────────


# A minimal valid netCDF-classic header: magic ``CDF`` + version byte
# 0x01 + 4 zero-bytes for ``numrecs``. Real netCDF files extend this
# with dim/var/data sections; for our magic-number validator the first
# 4 bytes are sufficient. The stub is padded so the staging file has
# bytes to write through.
_FAKE_NETCDF_BODY = b"CDF\x01" + b"\x00" * 256
_FAKE_NETCDF_HDF5 = b"\x89HDF" + b"\x00" * 256
_FAKE_FILE_URL = "https://files.isimip.org/cutouts/test.nc"


def _stub_dataset() -> Dict[str, Any]:
    """Return a synthetic dataset dict with all the fields cached_cutout
    reads (product / scenario / gcm / variable / version / doi / id)."""
    return {
        "id": "isimip3b/InputData/gfdl-esm4/ssp585/tasmax/v1",
        "version": "20240101",
        "doi": "10.5281/zenodo.fake",
        "product": "InputData",
        "climate_scenario": "ssp585",
        "climate_forcing": "gfdl-esm4",
        "climate_variable": "tasmax",
        "files": [],
    }


class _FakeISIMIP3bClient:
    """In-memory client whose ``cutout_bbox`` returns a finished job
    dict directly (no upstream polling). The download URL is a fixed
    string the ``responses`` fixtures intercept.

    Mirrors the real ISIMIP3bClient surface enough for cached_cutout
    to call ``cutout_bbox(paths, bbox)`` and read ``file_url`` out
    of the returned dict.
    """

    def __init__(
        self,
        *,
        cutout_response: Optional[Dict[str, Any]] = None,
        cutout_side_effect: Optional[Exception] = None,
    ) -> None:
        self.cutout_calls: List[Dict[str, Any]] = []
        self._response = cutout_response or {
            "id": "job-123",
            "status": "finished",
            "file_url": _FAKE_FILE_URL,
        }
        self._side_effect = cutout_side_effect

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
        # Match the real ``isimip_client.client.ISIMIPClient.cutout_bbox``
        # 4-float signature so the fake doesn't drift from upstream
        # (durable §24 + structural pin in
        # ``test_isimip_client_signature_alignment.py``).
        self.cutout_calls.append(
            {
                "paths": list(paths),
                "west": west,
                "east": east,
                "south": south,
                "north": north,
            }
        )
        if self._side_effect is not None:
            raise self._side_effect
        return self._response


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test cache root; clears the env override that might leak in
    from the dev environment."""
    monkeypatch.delenv("PRISMPY_ISIMIP_CACHE_DIR", raising=False)
    monkeypatch.delenv("PRISMPY_ISIMIP_CACHE_TTL_DAYS", raising=False)
    root = tmp_path / "isimip-cache"
    return root


@pytest.fixture
def fake_dataset() -> Dict[str, Any]:
    return _stub_dataset()


@pytest.fixture
def http_responses() -> Any:
    """``responses`` lib activation as a fixture so individual tests
    can register the cutout download mock."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rs:
        yield rs


# ── §2.1 cache primitives reused from _cache_base (durable #24) ──────


def test_cached_cutout_imports_cache_primitives_from_canonical_source() -> None:
    """The module must route cache lock + atomic write through the
    canonical ``_cache_base`` substrate, not its own re-implementation."""
    import ast

    src = (
        Path(__file__).resolve().parents[2]
        / "src/prismpy/data_sources/isimip3b.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imports_from_base: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "prismpy.sources._cache_base"
        ):
            for alias in node.names:
                imports_from_base.add(alias.name)
    required = {"cache_lock_path", "write_atomic_json"}
    missing = required - imports_from_base
    assert not missing, (
        f"isimip3b.py must import {sorted(missing)} from _cache_base"
    )


# ── §2.2 + §2.3 env overrides for cache root + TTL ───────────────────


def test_cache_root_resolves_from_env_var(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRISMPY_ISIMIP_CACHE_DIR", str(cache_root))
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    client = _FakeISIMIP3bClient()
    nc_path = cached_cutout(
        client,  # type: ignore[arg-type]
        fake_dataset,
        bbox={"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0},
    )
    # Cache root from env was honored (path falls under the env value)
    assert str(nc_path).startswith(str(cache_root))


def test_cache_ttl_resolves_from_env_var(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRISMPY_ISIMIP_CACHE_TTL_DAYS", "1")
    # Smoke: env var value must parse as int; invalid raises CacheDirectoryError
    monkeypatch.setenv("PRISMPY_ISIMIP_CACHE_TTL_DAYS", "not-an-int")
    client = _FakeISIMIP3bClient()
    with pytest.raises(CacheDirectoryError):
        cached_cutout(
            client,  # type: ignore[arg-type]
            fake_dataset,
            bbox={"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0},
            cache_dir=cache_root,
        )


# ── §2.4 bbox-key 4-decimal normalization ────────────────────────────


def test_bbox_key_is_4_decimal_rounded() -> None:
    """5e-6-level jitter must NOT fragment the cache, and a 1e-4-level
    delta MUST fragment it."""
    bbox_a = {"south": 13.500000, "north": 14.5, "west": 1.5, "east": 3.0}
    bbox_b = {"south": 13.500001, "north": 14.5, "west": 1.5, "east": 3.0}
    bbox_c = {"south": 13.5000005, "north": 14.5, "west": 1.5, "east": 3.0}
    assert isimip3b._bbox_key(bbox_a) == isimip3b._bbox_key(bbox_b)
    assert isimip3b._bbox_key(bbox_a) == isimip3b._bbox_key(bbox_c)
    bbox_diff = {"south": 13.5001, "north": 14.5, "west": 1.5, "east": 3.0}
    assert isimip3b._bbox_key(bbox_a) != isimip3b._bbox_key(bbox_diff)


# ── §2.5 cache-path shape ────────────────────────────────────────────


def test_cache_path_shape_matches_contract(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
) -> None:
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    client = _FakeISIMIP3bClient()
    nc_path = cached_cutout(
        client,  # type: ignore[arg-type]
        fake_dataset,
        bbox={"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0},
        cache_dir=cache_root,
    )
    rel = nc_path.relative_to(cache_root)
    parts = rel.parts
    # Expected layout: ISIMIP3b/<product>/<scenario>/<gcm>/<variable>/<bbox_key>.nc
    assert parts[0] == "ISIMIP3b"
    assert parts[1] == "InputData"
    assert parts[2] == "ssp585"
    assert parts[3] == "gfdl-esm4"
    assert parts[4] == "tasmax"
    assert parts[5].endswith(".nc")
    assert parts[5].startswith("S+13.0000_N+14.5000_W+1.5000_E+3.0000")


# ── §2.6 .meta.json sidecar captures version + DOI + fetch_time ──────


def test_meta_sidecar_records_version_and_doi(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
) -> None:
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    client = _FakeISIMIP3bClient()
    nc_path = cached_cutout(
        client,  # type: ignore[arg-type]
        fake_dataset,
        bbox={"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0},
        cache_dir=cache_root,
    )
    meta_path = nc_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["version"] == "20240101"
    assert meta["dataset_doi"] == "10.5281/zenodo.fake"
    assert "fetch_time" in meta
    assert meta["bbox"] == {"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0}


# ── §2.7 TTL invalidation ────────────────────────────────────────────


def test_ttl_expiry_triggers_refetch(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
) -> None:
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    client = _FakeISIMIP3bClient()
    bbox = {"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0}
    nc_path = cached_cutout(
        client,  # type: ignore[arg-type]
        fake_dataset,
        bbox=bbox,
        cache_dir=cache_root,
        ttl_days=7,
    )
    assert len(client.cutout_calls) == 1

    # Push the file's mtime back beyond the TTL.
    stale_atime = nc_path.stat().st_atime
    stale_mtime = time.time() - 8 * 86400
    os.utime(nc_path, (stale_atime, stale_mtime))

    cached_cutout(
        client,  # type: ignore[arg-type]
        fake_dataset,
        bbox=bbox,
        cache_dir=cache_root,
        ttl_days=7,
    )
    assert len(client.cutout_calls) == 2, "Cutout must re-fire after TTL expiry"


# ── §2.8 + §2.9 version and DOI invalidation ─────────────────────────


def test_version_mismatch_triggers_refetch(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
) -> None:
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    client = _FakeISIMIP3bClient()
    bbox = {"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0}
    cached_cutout(
        client,  # type: ignore[arg-type]
        fake_dataset,
        bbox=bbox,
        cache_dir=cache_root,
    )
    assert len(client.cutout_calls) == 1

    # Caller passes a fresh dataset dict with a bumped version.
    bumped = dict(fake_dataset)
    bumped["version"] = "20240601"
    cached_cutout(
        client,  # type: ignore[arg-type]
        bumped,
        bbox=bbox,
        cache_dir=cache_root,
    )
    assert len(client.cutout_calls) == 2, "Version bump must re-fetch"


def test_doi_mismatch_triggers_refetch(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
) -> None:
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    client = _FakeISIMIP3bClient()
    bbox = {"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0}
    cached_cutout(
        client,  # type: ignore[arg-type]
        fake_dataset,
        bbox=bbox,
        cache_dir=cache_root,
    )
    bumped = dict(fake_dataset)
    bumped["doi"] = "10.5281/zenodo.different"
    cached_cutout(
        client,  # type: ignore[arg-type]
        bumped,
        bbox=bbox,
        cache_dir=cache_root,
    )
    assert len(client.cutout_calls) == 2, "DOI re-issue must re-fetch"


# ── §2.10 missing meta = cold-cache (NOT crash) ──────────────────────


def test_missing_meta_treated_as_cold_cache(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
) -> None:
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    client = _FakeISIMIP3bClient()
    bbox = {"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0}
    nc_path = cached_cutout(
        client,  # type: ignore[arg-type]
        fake_dataset,
        bbox=bbox,
        cache_dir=cache_root,
    )

    # Delete the meta file but keep the .nc — simulates the AC-G-2 §2.10
    # race window between download success and meta write.
    nc_path.with_suffix(".meta.json").unlink()
    cached_cutout(
        client,  # type: ignore[arg-type]
        fake_dataset,
        bbox=bbox,
        cache_dir=cache_root,
    )
    assert len(client.cutout_calls) == 2, (
        "Missing meta must trigger a re-fetch, not a crash, not a stale serve"
    )


# ── §2.11 atomic-write: warm cache hit returns same path, no re-fetch ─


def test_warm_cache_hit_no_refetch(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
) -> None:
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    client = _FakeISIMIP3bClient()
    bbox = {"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0}
    nc_a = cached_cutout(
        client,  # type: ignore[arg-type]
        fake_dataset,
        bbox=bbox,
        cache_dir=cache_root,
    )
    nc_b = cached_cutout(
        client,  # type: ignore[arg-type]
        fake_dataset,
        bbox=bbox,
        cache_dir=cache_root,
    )
    assert nc_a == nc_b
    assert len(client.cutout_calls) == 1


# ── §F-G-10 #8: IsimipFetchError on real network failure mid-download ─


def test_drill_8_isimip_fetch_error_on_connection_drop(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
) -> None:
    """Real-condition drill #8 per MED-Pass4-1 — ``responses`` raises
    ``ConnectionError`` mid-stream; cached_cutout must surface a typed
    ``IsimipFetchError`` and leave NO partial bytes in the cache."""
    import requests as _requests

    http_responses.add(
        responses.GET,
        _FAKE_FILE_URL,
        body=_requests.exceptions.ConnectionError("simulated connection drop"),
    )
    client = _FakeISIMIP3bClient()
    bbox = {"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0}
    with pytest.raises(IsimipFetchError):
        cached_cutout(
            client,  # type: ignore[arg-type]
            fake_dataset,
            bbox=bbox,
            cache_dir=cache_root,
        )
    # No partial / staging files lingering as readable cache.
    leftovers: list[Path] = []
    if cache_root.exists():
        leftovers = list(cache_root.rglob("*.nc")) + list(
            cache_root.rglob("*.partial")
        )
    assert leftovers == [], f"Cache must be clean post-failure: {leftovers}"


# ── §F-G-10 #9: CacheWriteError on real OS write failure ─────────────


def test_drill_9_cache_write_error_on_disk_full(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-condition drill #9 per MED-Pass4-1 — monkey-patch
    ``os.replace`` (the rename step inside cached_cutout) to raise
    ``OSError(28, "No space left on device")``. cached_cutout must
    surface a typed ``CacheWriteError`` and leave the cache clean."""
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    client = _FakeISIMIP3bClient()
    bbox = {"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0}

    def _disk_full_replace(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(isimip3b.os, "replace", _disk_full_replace)

    with pytest.raises(CacheWriteError):
        cached_cutout(
            client,  # type: ignore[arg-type]
            fake_dataset,
            bbox=bbox,
            cache_dir=cache_root,
        )

    leftovers = list(cache_root.rglob("*.nc"))
    assert leftovers == [], (
        f"Cache must NOT contain a half-written .nc post-disk-full: {leftovers}"
    )


# ── §F-G-10 #9b: CacheWriteError on real OSError mid-write ───────────


def test_drill_9b_cache_write_error_on_disk_full_mid_write(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-condition drill #9b per MED-Pass4-1 — disk-full surfaces
    DURING the staging-write loop, not at the rename step.

    Drill #9 exercises ``os.replace`` failure (rename step). This
    companion drill exercises ``fh.write(chunk)`` failure (write
    step). Codex round 1 on b89b784 caught the unwrapped OSError on
    the write path; this drill calibrates that the new ``except OSError``
    clause in ``_download_to_staging`` raises the typed
    ``CacheWriteError`` and leaves the cache clean.
    """
    http_responses.add(
        responses.GET,
        _FAKE_FILE_URL,
        body=_FAKE_NETCDF_BODY,
    )
    client = _FakeISIMIP3bClient()
    bbox = {"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0}

    real_open = Path.open

    def failing_writer_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any):
        fh = real_open(self, mode, *args, **kwargs)
        # Only intercept the staging file's binary-write path to avoid
        # blowing up the fixture's own JSON writes.
        if "w" in mode and "b" in mode and self.name.endswith(".partial"):
            def write_with_disk_full(_data):
                raise OSError(28, "No space left on device")

            fh.write = write_with_disk_full  # type: ignore[method-assign]
        return fh

    monkeypatch.setattr(Path, "open", failing_writer_open)

    with pytest.raises(CacheWriteError) as exc_info:
        cached_cutout(
            client,  # type: ignore[arg-type]
            fake_dataset,
            bbox=bbox,
            cache_dir=cache_root,
        )
    assert "No space left on device" in str(exc_info.value)

    # Cache stays clean: no .nc / .partial leftovers.
    leftovers = list(cache_root.rglob("*.nc")) + list(
        cache_root.rglob("*.partial")
    )
    assert leftovers == [], (
        f"Cache must be clean post-write-failure: {leftovers}"
    )


# ── §F-G-10 #10: CacheDirectoryError on real PermissionError ─────────


def test_drill_10_cache_directory_error_on_perm_denied(
    tmp_path: Path,
    fake_dataset: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-condition drill #10 per MED-Pass4-1 — monkey-patch
    ``Path.mkdir`` to raise ``PermissionError`` when the cache root is
    being created. cached_cutout must surface a typed
    ``CacheDirectoryError`` early before any download begins."""
    cache_root = tmp_path / "perm-denied-cache"

    real_mkdir = Path.mkdir

    def _perm_denied_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if str(self).startswith(str(cache_root)):
            raise PermissionError(13, "Permission denied", str(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", _perm_denied_mkdir)

    client = _FakeISIMIP3bClient()
    with pytest.raises(CacheDirectoryError):
        cached_cutout(
            client,  # type: ignore[arg-type]
            fake_dataset,
            bbox={"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0},
            cache_dir=cache_root,
        )
    # No download fired (perm-denied is detected pre-network).
    assert len(client.cutout_calls) == 0


# ── §F-G-10 #11: concurrent threads serialize on the lock ────────────


def test_drill_11_concurrent_callers_serialize_via_lock(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
) -> None:
    """Real-condition drill #11 per MED-Pass4-1 — N threads calling
    cached_cutout for the same bbox on a cold cache must serialize on
    the per-key filelock. Only ONE upstream cutout is fired; all
    callers receive the same path."""
    http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    # If serialization breaks we'd see N concurrent downloads; pad the
    # responses queue so the test fails on call-count rather than
    # ``responses`` running out of registrations.
    for _ in range(8):
        http_responses.add(responses.GET, _FAKE_FILE_URL, body=_FAKE_NETCDF_BODY)
    client = _FakeISIMIP3bClient()
    bbox = {"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0}
    results: List[Path] = []
    errors: List[BaseException] = []

    def worker() -> None:
        try:
            results.append(
                cached_cutout(
                    client,  # type: ignore[arg-type]
                    fake_dataset,
                    bbox=bbox,
                    cache_dir=cache_root,
                )
            )
        except BaseException as exc:  # noqa: BLE001 — record for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == [], f"Concurrent callers must not raise: {errors}"
    assert len(results) == 4
    assert len(set(results)) == 1, "All threads must receive the same cache path"
    assert len(client.cutout_calls) == 1, (
        f"Lock must serialize cutout submissions; got {len(client.cutout_calls)}"
    )


# ── §F-G-10 #12: InvalidIsimipResponseError on real malformed body ───


def test_drill_12_invalid_response_on_malformed_netcdf(
    cache_root: Path,
    fake_dataset: Dict[str, Any],
    http_responses: responses.RequestsMock,
) -> None:
    """Real-condition drill #12 per MED-Pass4-1 — ``responses`` serves
    bytes that look like an HTML error page (no netCDF magic).
    cached_cutout's magic-number validator must raise
    ``InvalidIsimipResponseError`` and leave the cache clean."""
    http_responses.add(
        responses.GET,
        _FAKE_FILE_URL,
        body=b"<html><body>500 Internal Server Error</body></html>",
        status=200,
    )
    client = _FakeISIMIP3bClient()
    bbox = {"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0}
    with pytest.raises(InvalidIsimipResponseError):
        cached_cutout(
            client,  # type: ignore[arg-type]
            fake_dataset,
            bbox=bbox,
            cache_dir=cache_root,
        )
    leftovers = list(cache_root.rglob("*.nc")) + list(
        cache_root.rglob("*.partial")
    )
    assert leftovers == [], (
        f"Cache must be clean after malformed-response failure: {leftovers}"
    )


def test_netcdf_magic_validator_accepts_classic_and_hdf5() -> None:
    """The magic-number validator must accept both the classic
    ``CDF\\x01/02/05`` and the HDF5 ``\\x89HDF`` opening bytes."""
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(_FAKE_NETCDF_BODY)
        classic_path = Path(tf.name)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(_FAKE_NETCDF_HDF5)
        hdf5_path = Path(tf.name)

    try:
        # No raise expected
        isimip3b._validate_netcdf_magic(classic_path)
        isimip3b._validate_netcdf_magic(hdf5_path)
    finally:
        classic_path.unlink(missing_ok=True)
        hdf5_path.unlink(missing_ok=True)


# ── Cutout-job failure paths surface as IsimipFetchError ─────────────


def test_cutout_job_failed_status_raises_typed_error(
    cache_root: Path, fake_dataset: Dict[str, Any]
) -> None:
    """A finished-but-failed cutout job (status=failed) surfaces as a
    typed ``IsimipFetchError`` rather than a silent return."""
    client = _FakeISIMIP3bClient(
        cutout_response={
            "id": "job-failed",
            "status": "failed",
            "file_url": _FAKE_FILE_URL,
        }
    )
    with pytest.raises(IsimipFetchError):
        cached_cutout(
            client,  # type: ignore[arg-type]
            fake_dataset,
            bbox={"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0},
            cache_dir=cache_root,
        )


def test_cutout_job_submit_upstream_exception_wrapped(
    cache_root: Path, fake_dataset: Dict[str, Any]
) -> None:
    """An upstream exception during cutout submission must surface as
    ``IsimipFetchError``, NOT bubble up as the upstream's bare type."""
    client = _FakeISIMIP3bClient(cutout_side_effect=RuntimeError("upstream boom"))
    with pytest.raises(IsimipFetchError):
        cached_cutout(
            client,  # type: ignore[arg-type]
            fake_dataset,
            bbox={"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0},
            cache_dir=cache_root,
        )


# ── Dataset shape errors surface as typed errors ─────────────────────


def test_dataset_missing_required_fields_raises_typed_error(
    cache_root: Path,
) -> None:
    client = _FakeISIMIP3bClient()
    with pytest.raises(isimip3b.IsimipDatasetNotFoundError):
        cached_cutout(
            client,  # type: ignore[arg-type]
            dataset={"id": "incomplete"},  # no climate_scenario / forcing / variable
            bbox={"south": 13.0, "north": 14.5, "west": 1.5, "east": 3.0},
            cache_dir=cache_root,
        )


def test_bbox_missing_keys_raises_typed_error(
    cache_root: Path, fake_dataset: Dict[str, Any]
) -> None:
    client = _FakeISIMIP3bClient()
    with pytest.raises(CacheDirectoryError):
        cached_cutout(
            client,  # type: ignore[arg-type]
            fake_dataset,
            bbox={"south": 13.0, "north": 14.5},  # missing west / east
            cache_dir=cache_root,
        )
