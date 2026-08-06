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
pushdown, ``where="GID_0='<iso3>'"``), so the whole-world 2.76 GB gpkg is never
materialized. Only the SYNTHESIZED bodies are cached (a byte-bounded LRU keyed
on the exactly-measurable ``len(bytes)``), so a repeat request for a country
skips the read and the rebuild. A NEW level of an already-seen country re-reads
— a fast per-country filtered read, well within the latency bar. Frames are NOT
cached: a geometry-object GeoDataFrame's true memory (the GEOS coordinate
buffers) is not reliably measurable, so a frame cache could not honestly bound
worker memory.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
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
_GPKG_LAYER = "gadm_410"
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


class LocalGADMAdapter(HTTPAdapter):
    """Serve ``gadm41_{ISO3}_{level}.json`` from a local gpkg; delegate every
    other request (non-GADM URLs, unbundled countries) to the real network."""

    def __init__(self, gpkg_path: str, layer: str = _GPKG_LAYER):
        super().__init__()
        self._gpkg_path = gpkg_path
        self._layer = layer
        # Only synthesized bodies are retained (bytes are exactly measurable).
        self._synth_cache = _LRU(max_bytes=_SYNTH_CACHE_MAX_BYTES, sizeof=len)

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
            return self._delegate(request, **kwargs)  # unbundled country → network
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
        if ``iso3`` is invalid or not in the local gpkg (caller delegates)."""
        # Fail-closed injection guard AT the boundary: ``iso3`` is interpolated
        # into an OGR predicate, and direct callers bypass send()'s URL regex —
        # only a bare 3-letter code ever reaches _read_country's where-clause.
        if not isinstance(iso3, str) or not _ISO3_RE.fullmatch(iso3):
            return None
        key = (iso3, level)
        cached = self._synth_cache.get(key)
        if cached is not None:
            return cached
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


def mount_local_gadm(gpkg_path: Optional[str] = None) -> bool:
    """Mount the local GADM adapter on ``pygadm.session`` (idempotent).

    Resolves the gpkg from ``gpkg_path`` else ``$GADM_LOCAL_GPKG`` else the
    default. Returns True if mounted. Graceful: an unset/missing gpkg logs and
    leaves pygadm on its network path; never raises.
    """
    path = gpkg_path or os.environ.get("GADM_LOCAL_GPKG", DEFAULT_GPKG)
    if not path or not os.path.exists(path):
        logger.info(
            "GADM local gpkg unavailable (%s); pygadm uses the network path.",
            path)
        return False
    import pygadm
    if isinstance(pygadm.session.adapters.get(GADM_HOST), LocalGADMAdapter):
        return True  # already mounted — idempotent
    pygadm.session.mount(GADM_HOST, LocalGADMAdapter(path))
    logger.info(
        "LocalGADMAdapter mounted on pygadm.session for %s (gpkg=%s).",
        GADM_HOST, path)
    return True
