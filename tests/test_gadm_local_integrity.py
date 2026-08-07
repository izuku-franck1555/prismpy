"""Real mount-integrity pins for the LocalGADMAdapter (artifact SHA + size/mtime
+ GID_0 index at mount; per-serve re-stat; URL-cache disable). These run the
control UN-bypassed — a real indexed+manifest fixture, real stat/PRAGMA I/O, and
a real in-place artifact swap. No mock of the control under test.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

import prismpy.gadm_local as gl
from prismpy.gadm_local import GADM_HOST, LocalGADMAdapter, mount_local_gadm

_SUBSET = Path(__file__).parent / "fixtures" / "gadm_subset_NGA_MLI.gpkg"
_GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{}_{}.json"


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


def _add_index(gpkg, create_sql) -> None:
    """Create an arbitrary index (used to build wrong-column + non-leading-GID_0
    fixtures that must NOT satisfy the seek-serving index requirement)."""
    con = sqlite3.connect(str(gpkg))
    con.execute(create_sql)
    con.commit()
    con.close()


def _indexed_manifested(gpkg_dir, create_sql, monkeypatch):
    """A subset copy carrying ``create_sql``'s index + a matching manifest +
    SHA-pin — so only the index's SHAPE decides the mount outcome."""
    gpkg = gpkg_dir / "gadm_410.gpkg"
    shutil.copy(_SUBSET, gpkg)
    _add_index(gpkg, create_sql)
    sha = _sha256(gpkg)
    _write_manifest(gpkg, sha)
    monkeypatch.setattr(gl, "EXPECTED_GADM_ARTIFACT_SHA256", sha)
    return str(gpkg)


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


# ── per-serve re-stat: a REAL swap to a DISTINCT artifact flips integrity off ─

def test_in_place_swap_to_distinct_artifact_delegates_not_new_rows(valid_artifact, tmp_path):
    a = LocalGADMAdapter(valid_artifact)
    assert a._integrity_ok is True
    x = a._synthesize("NGA", 2)  # cache NGA from the VERIFIED artifact A
    assert x is not None
    # Atomically swap A -> a DISTINCT, independently-VALID artifact B (a real,
    # readable subset that DOES carry MLI rows). B is a separate file, so the
    # swap changes the stat identity (inode + size), not merely the mtime — and
    # B would serve MLI if the adapter ever read it.
    b = tmp_path / "artifact_B.gpkg"
    shutil.copy(_SUBSET, b)
    os.replace(str(b), valid_artifact)  # atomic in-place swap A -> B
    # A cached HIT still serves the pre-swap A bytes (no re-stat on a hit):
    assert a._synthesize("NGA", 2) == x
    # A NEW-country MISS re-stats -> identity changed -> delegate. It must return
    # None, NOT B's MLI rows: the swapped-in artifact is unverified, so the
    # adapter refuses to read it rather than serving the new file's data.
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


# ── the index must have GID_0 as its LEADING column (seek-serving) ────────────

def test_name0_only_index_is_delegate_only(tmp_path, monkeypatch):
    # An index on a DIFFERENT column (NAME_0, not GID_0) must NOT satisfy the
    # GID_0-index requirement — guards a "presence-only" regression that would
    # accept any index (only index-ABSENT was pinned before).
    gpkg = _indexed_manifested(
        tmp_path, 'CREATE INDEX idx_name0 ON gadm_410("NAME_0")', monkeypatch)
    a = LocalGADMAdapter(gpkg)
    assert a._integrity_ok is False
    assert a._synthesize("NGA", 2) is None


def test_non_leading_gid0_composite_index_is_delegate_only(tmp_path, monkeypatch):
    # A composite index with GID_0 in a NON-leading position — (NAME_0, GID_0) —
    # cannot serve the WHERE GID_0=? seek, so it must NOT count. A presence-only
    # "GID_0 in cols" check would wrongly accept it.
    gpkg = _indexed_manifested(
        tmp_path, 'CREATE INDEX idx_composite ON gadm_410("NAME_0","GID_0")',
        monkeypatch)
    a = LocalGADMAdapter(gpkg)
    assert a._integrity_ok is False
    assert a._synthesize("NGA", 2) is None


def test_leading_gid0_composite_index_is_accepted(tmp_path, monkeypatch):
    # Positive control — GID_0 as the LEADING column of a composite index
    # (GID_0, NAME_0) DOES serve the seek, so integrity passes (fix must not
    # over-reject a valid leading-GID_0 index).
    gpkg = _indexed_manifested(
        tmp_path, 'CREATE INDEX idx_lead ON gadm_410("GID_0","NAME_0")',
        monkeypatch)
    a = LocalGADMAdapter(gpkg)
    assert a._integrity_ok is True
    assert a._synthesize("NGA", 2) is not None


# ── mount trusts the manifest sha; it NEVER rehashes the (2.76 GB) artifact ───

def test_mount_uses_manifest_sha_and_does_not_rehash(valid_artifact, monkeypatch):
    # The mount check must trust the sidecar manifest's sha, never recompute a
    # digest of the whole artifact. Confirm the manifest IS consulted AND no
    # large byte-blob is fed to hashlib.sha256 during mount + a served request.
    big_hashes = []
    real_sha256 = hashlib.sha256

    def spy_sha256(data=b"", *a, **k):
        if isinstance(data, (bytes, bytearray)) and len(data) >= 1_000_000:
            big_hashes.append(len(data))
        return real_sha256(data, *a, **k)

    monkeypatch.setattr(hashlib, "sha256", spy_sha256)
    manifest_reads = []
    real_read_manifest = gl._read_manifest
    monkeypatch.setattr(
        gl, "_read_manifest",
        lambda p: (manifest_reads.append(p), real_read_manifest(p))[1])

    a = LocalGADMAdapter(valid_artifact)         # mount
    assert a._integrity_ok is True
    assert a._synthesize("NGA", 2) is not None   # serve
    assert manifest_reads, "the mount check must consult the sidecar manifest"
    assert not big_hashes, (
        "the artifact must NOT be rehashed at mount — the manifest sha is "
        f"trusted (saw a {max(big_hashes, default=0)}-byte hash)")


# ── fail-closed emits a boot WARNING and delegates through the FULL send() ────

def test_fail_closed_warns_at_mount_and_delegates_through_send(valid_artifact, caplog):
    os.remove(valid_artifact + gl._MANIFEST_SUFFIX)  # no manifest → integrity fails
    with caplog.at_level(logging.WARNING, logger="prismpy.gadm_local"):
        a = LocalGADMAdapter(valid_artifact)
    assert a._integrity_ok is False
    assert any("integrity check FAILED" in r.getMessage() for r in caplog.records), \
        "a WARNING must be emitted at mount when the artifact integrity fails"
    # The delegate must run through the FULL adapter path — send() sees the None
    # from _synthesize and delegates — not only the bare _synthesize helper.
    delegated = []
    a._delegate = lambda req, **k: (delegated.append(req.url), "DELEGATED")[1]
    req = types.SimpleNamespace(url=_GADM_URL.format("NGA", 2))
    assert a.send(req) == "DELEGATED"
    assert delegated == [req.url], \
        "send() must delegate a GADM request when the mount integrity failed"


# ── URL-cache disable is VERIFIED in effect, not silently swallowed ───────────

def test_url_cache_disable_returns_true_when_effective():
    class _Settings:
        disabled = False

    class _Session:
        def __init__(self):
            self.settings = _Settings()

    s = _Session()
    assert gl._disable_url_cache(s) is True
    assert s.settings.disabled is True


def test_url_cache_disable_warns_and_returns_false_when_flag_ignored(caplog):
    # A session whose `.settings.disabled` assignment is silently dropped (a
    # future requests_cache API change) must be caught LOUD — verified + WARN —
    # not swallowed, or a stale URL response could serve ahead of the adapter.
    class _StuckSettings:
        @property
        def disabled(self):
            return False

        @disabled.setter
        def disabled(self, _value):
            pass  # assignment ignored — the flag never takes effect

    class _Session:
        def __init__(self):
            self.settings = _StuckSettings()

    with caplog.at_level(logging.WARNING, logger="prismpy.gadm_local"):
        ok = gl._disable_url_cache(_Session())
    assert ok is False
    assert any("URL-cache disable failed" in r.getMessage() for r in caplog.records), \
        "a silently-ignored disable flag must emit a WARNING"


# ── the expected artifact SHA is env-overridable (DE sets it at staging) ──────

def _imported_expected_sha(env_value):
    """Import prismpy.gadm_local in a FRESH interpreter with the given
    EXPECTED_GADM_ARTIFACT_SHA256 env (None → unset) and return the resolved
    module constant. A subprocess so the import-time env read is exercised
    cleanly, without reloading the module into this test session."""
    env = {k: v for k, v in os.environ.items()
           if k != "EXPECTED_GADM_ARTIFACT_SHA256"}
    if env_value is not None:
        env["EXPECTED_GADM_ARTIFACT_SHA256"] = env_value
    proc = subprocess.run(
        [sys.executable, "-c",
         "import prismpy.gadm_local as gl; print(gl.EXPECTED_GADM_ARTIFACT_SHA256)"],
        capture_output=True, text=True, env=env, check=True)
    return proc.stdout.strip()


def test_expected_sha_reads_env_when_set():
    # DE sets the real indexed-gpkg digest via the env at staging (no code edit).
    sha = "a" * 64
    assert _imported_expected_sha(sha) == sha


def test_expected_sha_defaults_to_sentinel_when_env_unset():
    # Env unset → the "SET_AT_STAGING" sentinel, so an unstaged deploy fails every
    # integrity check and the adapter is delegate-only (unchanged fallback).
    assert _imported_expected_sha(None) == "SET_AT_STAGING"


@pytest.mark.parametrize("magic", [
    "SET_AT_STAGING", "", "   ", "not-a-sha256", "a" * 63, "a" * 65,
])
def test_non_sha_expected_fails_closed_even_on_matching_manifest(tmp_path, monkeypatch, magic):
    # A non-SHA expected digest (the sentinel, an empty/whitespace env value, or a
    # malformed string) must fail-close to delegate-only — EVEN against a bogus
    # manifest whose sha256 equals it — so an unstaged/misconfigured deploy can
    # never serve unverified data on a magic-string collision.
    gpkg = tmp_path / "gadm_410.gpkg"
    shutil.copy(_SUBSET, gpkg)
    _add_gid0_index(gpkg)
    st = os.stat(gpkg)
    Path(str(gpkg) + gl._MANIFEST_SUFFIX).write_text(json.dumps({
        "sha256": magic, "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}))
    monkeypatch.setattr(gl, "EXPECTED_GADM_ARTIFACT_SHA256", magic)
    a = LocalGADMAdapter(str(gpkg))
    assert a._integrity_ok is False
    assert a._synthesize("NGA", 2) is None
