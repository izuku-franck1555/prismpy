"""Pins for the local-first GADM transport (synth cache only, no frame cache):
the per-country FILTERED read (SPY the exact call — parity is green on the old
whole-read), the byte-bounded synth LRU (fixed literal ceiling, skip-oversized
through the real synth path), the injection guard, and the send() try-scope
fallbacks.
"""
from __future__ import annotations

import logging
import types
from pathlib import Path

import geopandas as gpd
import pytest

import prismpy.gadm_local as gl
from prismpy.gadm_local import (
    _SYNTH_CACHE_MAX_BYTES,
    _LRU,
    LocalGADMAdapter,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "gadm_subset_NGA_MLI.gpkg"
_GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{}_{}.json"


@pytest.fixture(autouse=True)
def _bypass_integrity(monkeypatch):
    # These pins exercise the serving / injection / fallback paths, not the mount
    # integrity control (tested in test_gadm_local_integrity.py) — so let the bare
    # subset fixture pass the mount check.
    monkeypatch.setattr(gl.LocalGADMAdapter, "_verify_artifact", lambda self: True)


def _req(iso3="NGA", level=2):
    return types.SimpleNamespace(url=_GADM_URL.format(iso3, level))


def _raise(exc):
    def _f(*_a, **_k):
        raise exc
    return _f


# ── SPY: filtered read, explicit engine, no unfiltered read ──────────────────

def test_read_is_filtered_per_country_with_pyogrio(monkeypatch):
    adapter = LocalGADMAdapter(str(_FIXTURE))
    calls = []
    real_read = gpd.read_file

    def spy(*a, **k):
        calls.append((a, k))
        return real_read(*a, **k)

    monkeypatch.setattr(gpd, "read_file", spy)
    assert adapter._synthesize("NGA", 2) is not None

    assert len(calls) == 1, f"one filtered read expected, got {len(calls)}"
    _args, kw = calls[0]
    assert kw.get("where") == "GID_0='NGA'", "must push the country filter to OGR"
    assert kw.get("engine") == "pyogrio", "engine must be explicit (0.14 → Fiona default)"
    assert kw.get("layer") == "gadm_410"
    assert all(c[1].get("where") for c in calls), "no unfiltered production read allowed"

    adapter._synthesize("NGA", 2)   # repeat SAME (iso3, level) → synth-cache hit
    assert len(calls) == 1, "the synth cache serves a repeat (iso3,level) with no re-read"
    adapter._synthesize("NGA", 1)   # NEW level → re-read (frames are not cached)
    assert len(calls) == 2, "a new level re-reads (a fast per-country filtered read)"


# ── EXACT synth ceiling (literal, not a self-derived count) ──────────────────

def test_synth_ceiling_is_locked_literal():
    assert _SYNTH_CACHE_MAX_BYTES == 96 * 1024 * 1024


# ── byte-bounded LRU: real LRU order, byte bound, skip-oversized ─────────────

def test_lru_evicts_untouched_not_touched():
    cache = _LRU(max_bytes=300, sizeof=lambda _v: 100)  # 3 slots of 100 bytes
    for k in ("A", "B", "C"):
        cache.put(k, k)
    assert cache.get("A") == "A"        # TOUCH the oldest → now MRU; LRU = B
    cache.put("D", "D")                 # over 300 → evict the LRU
    assert "B" not in cache, "the UNTOUCHED-oldest must be evicted"
    assert "A" in cache, "the touched entry survives (proves LRU, not FIFO)"
    assert "C" in cache and "D" in cache


def test_lru_bounds_by_bytes():
    cache = _LRU(max_bytes=100, sizeof=len)
    cache.put("a", b"x" * 60)
    cache.put("b", b"y" * 60)           # 120 > 100 → evict "a"
    assert "a" not in cache and "b" in cache


def test_lru_skips_oversized_single_entry():
    cache = _LRU(max_bytes=100, sizeof=len)
    cache.put("ok", b"z" * 50)
    cache.put("huge", b"Z" * 101)       # 101 > 100 → skip, never stored
    assert "huge" not in cache, "oversized value must not be cached"
    assert "ok" in cache, "the in-bound entry is untouched"
    assert cache._bytes <= 100, "the byte ceiling is never exceeded"


def test_synth_cache_skips_oversized_body_via_synthesize(monkeypatch):
    # Through the REAL prod path: lower the synth ceiling below NGA_2's ~11.5 MB
    # body → it is served but NOT cached (skip-oversized), so a 2nd identical
    # request re-reads + re-builds instead of returning a bloated cache.
    adapter = LocalGADMAdapter(str(_FIXTURE))
    adapter._synth_cache = _LRU(max_bytes=1024, sizeof=len)  # 1 KB ceiling
    reads = []
    real_read = gpd.read_file
    monkeypatch.setattr(gpd, "read_file",
                        lambda *a, **k: (reads.append(k), real_read(*a, **k))[1])
    b1 = adapter._synthesize("NGA", 2)
    assert b1 is not None
    assert ("NGA", 2) not in adapter._synth_cache, "oversized body must not be cached"
    b2 = adapter._synthesize("NGA", 2)
    assert b2 == b1
    assert len(reads) == 2, "an uncached oversized body → the 2nd call re-reads"
    assert adapter._synth_cache._bytes <= 1024, "the byte ceiling is never exceeded"


# ── injection guard at the read boundary (direct callers bypass send) ─────────

@pytest.mark.parametrize("bad", [
    "nga", "NGA' OR 1=1 --", "NG", "NGAA", "N;A", "NGA\n", "N'A", 123, None,
])
def test_bad_iso3_returns_none_and_reads_zero_times(monkeypatch, bad):
    adapter = LocalGADMAdapter(str(_FIXTURE))
    reads = []
    monkeypatch.setattr(gpd, "read_file",
                        lambda *a, **k: reads.append(k) or "SHOULD_NOT")
    assert adapter._synthesize(bad, 2) is None
    assert reads == [], f"read_file must be called 0x for bad iso3 {bad!r}"


# ── send() try wraps ONLY _synthesize: the 3 fallback pins ───────────────────

def test_synth_failure_warns_and_delegates_exactly_once(monkeypatch, caplog):
    adapter = LocalGADMAdapter(str(_FIXTURE))
    monkeypatch.setattr(adapter, "_synthesize",
                        _raise(RuntimeError("corrupt/wrong-layer")))
    delegated = []
    monkeypatch.setattr(adapter, "_delegate",
                        lambda req, **k: (delegated.append(req), "DELEGATED")[1])
    with caplog.at_level(logging.WARNING, logger="prismpy.gadm_local"):
        assert adapter.send(_req()) == "DELEGATED"
    assert len(delegated) == 1, "a local-serve failure delegates exactly once"
    assert any(r.levelno >= logging.WARNING for r in caplog.records), \
        "a WARNING must be emitted on the local-serve fallback"


def test_delegate_exception_propagates_after_one_call(monkeypatch):
    adapter = LocalGADMAdapter(str(_FIXTURE))
    monkeypatch.setattr(adapter, "_synthesize", lambda *a, **k: None)  # unbundled
    calls = []

    def boom(req, **k):
        calls.append(req)
        raise ConnectionError("network down")

    monkeypatch.setattr(adapter, "_delegate", boom)
    with pytest.raises(ConnectionError):
        adapter.send(_req())
    assert len(calls) == 1, "delegate is attempted once, then the error propagates"


def test_build_response_defect_propagates_with_zero_delegates(monkeypatch):
    adapter = LocalGADMAdapter(str(_FIXTURE))
    monkeypatch.setattr(
        adapter, "_synthesize",
        lambda *a, **k: b'{"type":"FeatureCollection","features":[]}')
    monkeypatch.setattr(adapter, "build_response",
                        _raise(RuntimeError("build defect")))
    delegated = []
    monkeypatch.setattr(adapter, "_delegate", lambda req, **k: delegated.append(req))
    with pytest.raises(RuntimeError, match="build defect"):
        adapter.send(_req())
    assert delegated == [], "a build_response defect must NOT be masked as a delegate"
