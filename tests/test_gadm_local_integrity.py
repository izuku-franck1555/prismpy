"""Real mount-integrity pins for the LocalGADMAdapter (artifact SHA + size/mtime
+ GID_0 index at mount; per-serve re-stat; URL-cache disable). These run the
control UN-bypassed — a real indexed+manifest fixture, real stat/PRAGMA I/O, and
a real in-place artifact swap. No mock of the control under test.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

import prismpy.gadm_local as gl
from prismpy.gadm_local import GADM_HOST, LocalGADMAdapter, mount_local_gadm

_SUBSET = Path(__file__).parent / "fixtures" / "gadm_subset_NGA_MLI.gpkg"


def _sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write_manifest(gpkg, sha=None) -> None:
    st = os.stat(gpkg)
    Path(str(gpkg) + gl._MANIFEST_SUFFIX).write_text(json.dumps({
        "sha256": sha if sha is not None else _sha256(gpkg),
        "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}))


def _add_gid0_index(gpkg) -> None:
    con = sqlite3.connect(str(gpkg))
    con.execute('CREATE INDEX idx_gadm_gid0 ON gadm_410("GID_0")')
    con.commit()
    con.close()


@pytest.fixture
def valid_artifact(tmp_path, monkeypatch):
    """A subset copy that is GID_0-indexed + manifested + SHA-pinned → the mount
    integrity check passes for real."""
    gpkg = tmp_path / "gadm_410.gpkg"
    shutil.copy(_SUBSET, gpkg)
    _add_gid0_index(gpkg)              # the required GID_0 index
    sha = _sha256(gpkg)
    _write_manifest(gpkg, sha)         # manifest AFTER indexing (final size/mtime)
    monkeypatch.setattr(gl, "EXPECTED_GADM_ARTIFACT_SHA256", sha)
    return str(gpkg)


@pytest.fixture
def _session_guard():
    """Restore pygadm.session mount + URL-cache flag around a mount test."""
    import pygadm
    prior_adapter = pygadm.session.adapters.get(GADM_HOST)
    prior_disabled = pygadm.session.settings.disabled
    yield pygadm
    pygadm.session.adapters.pop(GADM_HOST, None)
    if prior_adapter is not None:
        pygadm.session.mount(GADM_HOST, prior_adapter)
    pygadm.session.settings.disabled = prior_disabled


# ── happy path (real) ────────────────────────────────────────────────────────

def test_valid_indexed_manifested_artifact_serves(valid_artifact):
    a = LocalGADMAdapter(valid_artifact)
    assert a._integrity_ok is True
    body = a._synthesize("NGA", 2)
    assert body is not None and len(json.loads(body)["features"]) == 775


# ── fail-closed: each invalid case → delegate-only (integrity False → None) ───

def test_missing_manifest_is_delegate_only(valid_artifact):
    os.remove(valid_artifact + gl._MANIFEST_SUFFIX)
    a = LocalGADMAdapter(valid_artifact)
    assert a._integrity_ok is False
    assert a._synthesize("NGA", 2) is None


def test_wrong_expected_sha_is_delegate_only(valid_artifact, monkeypatch):
    monkeypatch.setattr(gl, "EXPECTED_GADM_ARTIFACT_SHA256", "0" * 64)  # != manifest
    a = LocalGADMAdapter(valid_artifact)
    assert a._integrity_ok is False
    assert a._synthesize("NGA", 2) is None


def test_missing_gid0_index_is_delegate_only(tmp_path, monkeypatch):
    gpkg = tmp_path / "gadm_410.gpkg"
    shutil.copy(_SUBSET, gpkg)  # NO index
    sha = _sha256(gpkg)
    _write_manifest(gpkg, sha)
    monkeypatch.setattr(gl, "EXPECTED_GADM_ARTIFACT_SHA256", sha)
    a = LocalGADMAdapter(str(gpkg))
    assert a._integrity_ok is False
    assert a._synthesize("NGA", 2) is None


def test_size_mtime_mismatch_is_delegate_only(valid_artifact):
    os.utime(valid_artifact, ns=(0, 0))  # mtime_ns → 0 != manifest
    a = LocalGADMAdapter(valid_artifact)
    assert a._integrity_ok is False
    assert a._synthesize("NGA", 2) is None


# ── per-serve re-stat: a REAL in-place swap flips integrity off ──────────────

def test_in_place_swap_flips_integrity_off_and_new_miss_delegates(valid_artifact):
    a = LocalGADMAdapter(valid_artifact)
    assert a._integrity_ok is True
    x = a._synthesize("NGA", 2)  # cache X for NGA
    assert x is not None
    # REAL in-place mutation of the served artifact → stat (size+mtime) changes.
    with open(valid_artifact, "ab") as f:
        f.write(b"\x00")
    os.utime(valid_artifact, ns=(1, 987654321))
    # A cached HIT still serves the pre-swap bytes (no re-stat on a hit):
    assert a._synthesize("NGA", 2) == x
    # A NEW country MISS re-stats → identity changed → delegate + flip off:
    assert a._synthesize("MLI", 2) is None
    assert a._integrity_ok is False


# ── mount_local_gadm disables the URL cache (even delegate-only) ─────────────

def test_mount_valid_disables_url_cache(valid_artifact, _session_guard):
    pygadm = _session_guard
    pygadm.session.settings.disabled = False
    assert mount_local_gadm(valid_artifact) is True
    assert pygadm.session.settings.disabled is True
    mounted = pygadm.session.adapters.get(GADM_HOST)
    assert isinstance(mounted, LocalGADMAdapter) and mounted._integrity_ok is True


def test_mount_invalid_stays_mounted_delegate_only_and_disables_cache(tmp_path, _session_guard):
    gpkg = tmp_path / "gadm_410.gpkg"
    shutil.copy(_SUBSET, gpkg)  # no index, no manifest → integrity fails
    pygadm = _session_guard
    pygadm.session.settings.disabled = False
    assert mount_local_gadm(str(gpkg)) is True  # still returns True (mounted)
    assert pygadm.session.settings.disabled is True  # disabled even delegate-only
    mounted = pygadm.session.adapters.get(GADM_HOST)
    assert isinstance(mounted, LocalGADMAdapter) and mounted._integrity_ok is False


# ── remount idempotency: same artifact → same adapter kept ───────────────────

def test_remount_same_artifact_is_idempotent(valid_artifact, _session_guard):
    pygadm = _session_guard
    mount_local_gadm(valid_artifact)
    first = pygadm.session.adapters.get(GADM_HOST)
    mount_local_gadm(valid_artifact)  # same (path, layer, identity) → keep first
    assert pygadm.session.adapters.get(GADM_HOST) is first
