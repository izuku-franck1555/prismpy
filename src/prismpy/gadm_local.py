"""Local-first GADM transport: serve administrative boundaries from a pinned,
vendored GADM gpkg instead of the live UC Davis host.

pygadm fetches ``gadm41_{ISO3}_{level}.json`` from ``geodata.ucdavis.edu`` — a
runtime single point of failure (every region lookup 500s when that host is
down). This module mounts a ``requests`` transport adapter on the single
producer boundary, ``pygadm.session``, so EVERY consumer (prismweb wizard,
prismpy executor, the CRAFT translator) inherits the resilience with no
per-call-site patching. For ``gadm41_{ISO3}_{level}.json`` it synthesizes a
GeoJSON ``FeatureCollection`` from the local gpkg — schema-identical to the
network response (same features, GIDs, names); the gpkg carries full-precision
geometry, so the raw body is ~9x the network's coordinate-reduced JSON and
differs by ~0.08% area (immaterial for boundary display + study extents). A
missing or unset gpkg falls through to the real network path; non-GADM URLs and
unbundled countries delegate to the network too.

Only the requested country is read from the gpkg (an OGR attribute-filter
pushdown), so the whole-world gpkg is never materialized. Only the SYNTHESIZED
bodies are cached (a byte-bounded LRU keyed on the exactly-measurable
``len(bytes)``). Frames are NOT cached: a geometry-object GeoDataFrame's true
memory is not reliably measurable.

Integrity: the mounted artifact is verified ONCE at mount (metadata I/O, never a
2.76 GB rehash) — the sidecar manifest's SHA-256 must equal a pinned expected
digest, the file's size+mtime must equal the manifest's, and the layer must
carry a GID_0 index. Any failure leaves the adapter mounted but delegate-only.
When the adapter is authoritative, the requests_cache URL tier is disabled so no
stale cached response can serve ahead of the adapter.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import sqlite3
import threading
from collections import OrderedDict
from typing import Optional

from requests.adapters import HTTPAdapter
from urllib3.response import HTTPResponse

logger = logging.getLogger(__name__)

GADM_HOST = "https://geodata.ucdavis.edu"
DEFAULT_GPKG = "/data/gadm/gadm_410.gpkg"
_GADM_URL_RE = re.compile(r"/gadm41_(?P<iso3>[A-Z]{3})_(?P<level>\d)\.json")
# The ONLY shape allowed into the interpolated OGR predicate (a bare ISO3).
_ISO3_RE = re.compile(r"[A-Z]{3}")
# The dataset version is the single source for the layer name (and the gate).
EXPECTED_GADM_DATASET_VERSION = "410"  # == pygadm.__gadm_version__
_GPKG_LAYER = f"gadm_{EXPECTED_GADM_DATASET_VERSION}"
# SHA-256 of the FINAL (GID_0-indexed) artifact. DE sets this at staging to the
# indexed gpkg's digest; the staged sidecar manifest carries the same value, and
# the mount check requires manifest.sha256 == this (no 2.76 GB rehash). Until it
# is set, every artifact fails the check and the adapter is delegate-only.
EXPECTED_GADM_ARTIFACT_SHA256 = "SET_AT_STAGING"
_MANIFEST_SUFFIX = ".manifest.json"
# Bound delegated (non-local) requests (connect, read) so a black-hole GADM host
# fails fast instead of hanging the worker; pygadm itself passes timeout=None.
_DELEGATE_TIMEOUT = (5, 30)

# The synth cache is the ONLY retained state. Synth bodies are ``bytes`` so their
# size is exactly ``len()``; a body larger than the ceiling is served but NOT
# cached (skip-oversized), so no single country can push the cache past its bound.
_SYNTH_CACHE_MAX_BYTES = 96 * 1024 * 1024


class _LRU:
    """Minimal thread-safe byte-bounded LRU.

    ``sizeof(value)`` measures an entry. ``get`` marks the key most-recently-
    used; ``put`` skips a value larger than the ceiling (never cached as the sole
    over-bound entry) and otherwise evicts the least-recently-used until the
    bound holds (every stored entry <= ceiling, so the just-inserted survives).
    """

    def __init__(self, *, max_bytes, sizeof):
        self._d: "OrderedDict" = OrderedDict()
        self._mutex = threading.Lock()
        self._max_bytes = max_bytes
        self._sizeof = sizeof
        self._bytes = 0

    def get(self, key):
        with self._mutex:
            if key in self._d:
                self._d.move_to_end(key)  # most-recently-used
                return self._d[key]
            return None

    def put(self, key, value):
        with self._mutex:
            size = self._sizeof(value)
            # Skip-oversized: a value larger than the ceiling is NEVER cached
            # (it would exceed the bound as the sole entry). Drop any stale entry.
            if size > self._max_bytes:
                if key in self._d:
                    self._bytes -= self._sizeof(self._d.pop(key))
                return
            if key in self._d:
                self._bytes -= self._sizeof(self._d[key])
                self._d[key] = value
                self._d.move_to_end(key)
            else:
                self._d[key] = value
            self._bytes += size
            # Every stored entry is <= max_bytes, so the just-inserted (MRU)
            # survives: evicting the older entries brings _bytes to <= ceiling.
            while self._bytes > self._max_bytes and self._d:
                _, evicted = self._d.popitem(last=False)
                self._bytes -= self._sizeof(evicted)

    def __len__(self):
        with self._mutex:
            return len(self._d)

    def __contains__(self, key):
        with self._mutex:
            return key in self._d


def _read_manifest(gpkg_path: str) -> Optional[dict]:
    """Read the minimal sidecar manifest ``{sha256, size_bytes, mtime_ns}``
    beside the gpkg; None on any missing/malformed read (fail-closed upstream)."""
    try:
        with open(gpkg_path + _MANIFEST_SUFFIX, "r", encoding="utf-8") as f:
            m = json.load(f)
        return {"sha256": str(m["sha256"]),
                "size_bytes": int(m["size_bytes"]),
                "mtime_ns": int(m["mtime_ns"])}
    except Exception:  # noqa: BLE001 — missing/malformed manifest → fail-closed
        return None


def _artifact_id(gpkg_path: str):
    """Cheap stat identity ``(st_ino, st_size, st_mtime_ns)`` for detecting an
    in-place artifact swap; None if the path can't be stat'd."""
    try:
        st = os.stat(gpkg_path)
        return (st.st_ino, st.st_size, st.st_mtime_ns)
    except OSError:
        return None


def _gid0_index_present(gpkg_path: str, layer: str) -> bool:
    """True iff a SQLite index on ``layer`` covers GID_0 — metadata I/O, not a
    rehash. Requires ``PRAGMA index_info`` to list GID_0 (a bare index NAME is
    not enough). False on any error (unreadable gpkg → fail-closed)."""
    con = None
    try:
        con = sqlite3.connect(gpkg_path)
        for row in con.execute(f'PRAGMA index_list("{layer}")').fetchall():
            name = row[1]
            cols = [r[2] for r in con.execute(
                f'PRAGMA index_info("{name}")').fetchall()]
            if "GID_0" in cols:
                return True
        return False
    except Exception:  # noqa: BLE001 — unreadable gpkg / SQL error → fail-closed
        return False
    finally:
        if con is not None:
            con.close()


class LocalGADMAdapter(HTTPAdapter):
    """Serve ``gadm41_{ISO3}_{level}.json`` from a local gpkg; delegate every
    other request (non-GADM URLs, unbundled countries, integrity failure) to the
    real network."""

    def __init__(self, gpkg_path: str, layer: str = _GPKG_LAYER):
        super().__init__()
        self._gpkg_path = gpkg_path
        self._layer = layer
        # Only synthesized bodies are retained (bytes are exactly measurable).
        self._synth_cache = _LRU(max_bytes=_SYNTH_CACHE_MAX_BYTES, sizeof=len)
        # Mount-time artifact integrity: pin the stat identity + verify the
        # manifest digest, size/mtime, and the GID_0 index. Fail-closed leaves
        # the adapter mounted but delegate-only (every request delegates).
        self._artifact_id = _artifact_id(gpkg_path)
        self._integrity_ok = self._verify_artifact()
        if not self._integrity_ok:
            logger.warning(
                "GADM artifact integrity check FAILED for %s; the adapter is "
                "delegate-only (serving from UC-Davis).", gpkg_path)

    def _verify_artifact(self) -> bool:
        manifest = _read_manifest(self._gpkg_path)
        if manifest is None:
            return False
        if manifest["sha256"] != EXPECTED_GADM_ARTIFACT_SHA256:
            return False
        st = self._artifact_id
        if (st is None or st[1] != manifest["size_bytes"]
                or st[2] != manifest["mtime_ns"]):
            return False
        return _gid0_index_present(self._gpkg_path, self._layer)

    def _read_country(self, iso3: str):
        """Filtered read of ONLY ``iso3``'s rows from the gpkg — an OGR
        attribute-filter pushdown, so the whole-world gpkg never materializes.
        ``iso3`` is pre-validated by the caller. Not cached (see module doc)."""
        import geopandas as gpd
        # Explicit pyogrio engine — the geopandas 0.14 floor defaults to Fiona,
        # so the pushdown engine is not left to chance.
        return gpd.read_file(
            self._gpkg_path, layer=self._layer,
            where=f"GID_0='{iso3}'", engine="pyogrio")

    def send(self, request, **kwargs):
        match = _GADM_URL_RE.search(request.url or "")
        if match is None:
            return self._delegate(request, **kwargs)  # non-GADM → network
        # The try wraps ONLY the local-serve synthesis: a local read/schema
        # failure degrades to the network, but a delegate error or a
        # build_response defect must SURFACE, not be masked as "unavailable".
        try:
            body = self._synthesize(match.group("iso3"), int(match.group("level")))
        except Exception as exc:  # noqa: BLE001 — any local-serve failure → delegate
            logger.warning(
                "LocalGADM local serve failed for %s (%s); delegating to network.",
                request.url, exc)
            return self._delegate(request, **kwargs)
        if body is None:
            return self._delegate(request, **kwargs)  # unbundled / integrity → network
        raw = HTTPResponse(
            body=io.BytesIO(body),
            headers={"Content-Type": "application/json",
                     "Content-Length": str(len(body))},
            status=200,
            preload_content=False,
        )
        raw._request_url = request.url  # requests_cache reads this when caching
        return self.build_response(request, raw)

    def _delegate(self, request, **kwargs):
        # setdefault is a no-op here — pygadm passes timeout=None explicitly, so
        # the key is present; replace a None timeout with the bounded default.
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = _DELEGATE_TIMEOUT
        return super().send(request, **kwargs)

    def _synthesize(self, iso3: str, level: int) -> Optional[bytes]:
        """Build the GeoJSON FeatureCollection for one country+level, or None
        (caller delegates) if ``iso3`` is invalid, integrity has failed, or
        ``iso3`` is not in the local gpkg."""
        # Fail-closed injection guard AT the boundary: ``iso3`` is interpolated
        # into an OGR predicate, and direct callers bypass send()'s URL regex —
        # only a bare 3-letter code ever reaches _read_country's where-clause.
        if not isinstance(iso3, str) or not _ISO3_RE.fullmatch(iso3):
            return None
        if not self._integrity_ok:
            return None  # mount-time integrity failed → delegate
        key = (iso3, level)
        cached = self._synth_cache.get(key)
        if cached is not None:
            return cached
        # On a cache MISS, re-stat before reading: an in-place swap of the
        # artifact under the mounted adapter flips integrity off, so an
        # unchecked (post-mount) artifact is never read.
        if _artifact_id(self._gpkg_path) != self._artifact_id:
            self._integrity_ok = False
            logger.warning(
                "GADM artifact changed under the mounted adapter (%s); "
                "delegating.", self._gpkg_path)
            return None
        import pygadm
        from shapely.geometry import mapping
        sub = self._read_country(iso3)
        sub = sub[sub["GID_0"] == iso3]
        if sub.empty:
            return None
        keep = ([f"GID_{i}" for i in range(level + 1)]
                + [f"NAME_{i}" for i in range(level + 1)] + ["geometry"])
        if sub[f"GID_{level}"].nunique() == len(sub):
            derived = sub[keep]  # one row per unit already — exact, no dissolve
        else:
            # Precise union per GID group (no simplify/round); the GID/NAME
            # attrs are constant within a group, so 'first' is exact.
            derived = sub[keep].dissolve(
                by=f"GID_{level}", aggfunc="first").reset_index()
        by_gid = {row[f"GID_{level}"]: row for _, row in derived.iterrows()}
        # Emit rows in pygadm.Names(complete=True) order: pygadm overwrites
        # NAME_* BY POSITION from it, so any other order mis-pairs names.
        order = pygadm.Names(admin=iso3, content_level=level, complete=True)
        features = []
        for _, orow in order.iterrows():
            row = by_gid[orow[f"GID_{level}"]]
            props = {"COUNTRY": row["NAME_0"]}  # pygadm renames COUNTRY→NAME_0
            for i in range(level + 1):
                props[f"GID_{i}"] = row[f"GID_{i}"]
            for i in range(1, level + 1):
                props[f"NAME_{i}"] = row[f"NAME_{i}"]
            features.append({"type": "Feature", "properties": props,
                             "geometry": mapping(row["geometry"])})
        body = json.dumps(
            {"type": "FeatureCollection", "features": features}).encode()
        self._synth_cache.put(key, body)
        return body


def _disable_url_cache(session) -> None:
    """Persistently disable the requests_cache URL tier on ``session`` (thread-
    safe flag, NOT the per-call context) — it is a redundant stale surface once
    the adapter is authoritative. Best-effort; never raises."""
    try:
        session.settings.disabled = True
    except Exception:  # noqa: BLE001 — a non-CachedSession or API change → skip
        pass


def mount_local_gadm(gpkg_path: Optional[str] = None) -> bool:
    """Mount the local GADM adapter on ``pygadm.session`` (idempotent).

    Resolves the gpkg from ``gpkg_path`` else ``$GADM_LOCAL_GPKG`` else the
    default. Returns True if mounted (even delegate-only on integrity failure).
    Graceful: an unset/missing gpkg logs and leaves pygadm on its network path;
    never raises.
    """
    path = gpkg_path or os.environ.get("GADM_LOCAL_GPKG", DEFAULT_GPKG)
    if not path or not os.path.exists(path):
        logger.info(
            "GADM local gpkg unavailable (%s); pygadm uses the network path.",
            path)
        return False
    import pygadm
    # The adapter is authoritative for this host → disable the requests_cache URL
    # tier so no stale cached response serves ahead of it. Set for EVERY mount
    # (incl. a delegate-only integrity-failed mount) and before the idempotent
    # early return. The delegate path (UC-Davis) still works, just uncached.
    _disable_url_cache(pygadm.session)
    adapter = LocalGADMAdapter(path)
    existing = pygadm.session.adapters.get(GADM_HOST)
    if (isinstance(existing, LocalGADMAdapter)
            and existing._gpkg_path == adapter._gpkg_path
            and existing._layer == adapter._layer
            and existing._artifact_id == adapter._artifact_id):
        return True  # idempotent — same artifact (path, layer, identity)
    pygadm.session.mount(GADM_HOST, adapter)
    logger.info(
        "LocalGADMAdapter mounted on pygadm.session for %s (gpkg=%s, "
        "integrity_ok=%s).", GADM_HOST, path, adapter._integrity_ok)
    return True
