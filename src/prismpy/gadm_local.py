"""Local-first GADM transport: serve administrative boundaries from a pinned,
vendored GADM gpkg instead of the live UC Davis host.

pygadm fetches ``gadm41_{ISO3}_{level}.json`` from ``geodata.ucdavis.edu`` — a
runtime single point of failure (every region lookup 500s when that host is
down). This module mounts a ``requests`` transport adapter on the single
producer boundary, ``pygadm.session``, so EVERY consumer (prismweb wizard,
prismpy executor, the CRAFT translator) inherits the resilience with no
per-call-site patching. For ``gadm41_{ISO3}_{level}.json`` it synthesizes the
same GeoJSON ``FeatureCollection`` from the local gpkg — byte/schema-identical
to the network response — so the hot path needs zero network. A missing or
unset gpkg falls through to the real network path; non-GADM URLs and unbundled
countries delegate to the network too.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
from typing import Optional

from requests.adapters import HTTPAdapter
from urllib3.response import HTTPResponse

logger = logging.getLogger(__name__)

GADM_HOST = "https://geodata.ucdavis.edu"
DEFAULT_GPKG = "/data/gadm/gadm_410.gpkg"
_GADM_URL_RE = re.compile(r"/gadm41_(?P<iso3>[A-Z]{3})_(?P<level>\d)\.json")
_GPKG_LAYER = "gadm_410"


class LocalGADMAdapter(HTTPAdapter):
    """Serve ``gadm41_{ISO3}_{level}.json`` from a local gpkg; delegate every
    other request (non-GADM URLs, unbundled countries) to the real network."""

    def __init__(self, gpkg_path: str, layer: str = _GPKG_LAYER):
        super().__init__()
        self._gpkg_path = gpkg_path
        self._layer = layer
        self._lock = threading.Lock()
        self._gdf = None  # lazy-loaded once, then read-only
        self._synth_cache: dict = {}

    def _frame(self):
        # Lazy-load the gpkg once under a lock; per-request derives are new
        # read-only frames, so concurrent requests never share a GDAL handle.
        if self._gdf is None:
            with self._lock:
                if self._gdf is None:
                    import geopandas as gpd
                    self._gdf = gpd.read_file(self._gpkg_path, layer=self._layer)
        return self._gdf

    def send(self, request, **kwargs):
        match = _GADM_URL_RE.search(request.url or "")
        if match is None:
            return super().send(request, **kwargs)  # non-GADM → network
        body = self._synthesize(match.group("iso3"), int(match.group("level")))
        if body is None:
            return super().send(request, **kwargs)  # unbundled country → network
        raw = HTTPResponse(
            body=io.BytesIO(body),
            headers={"Content-Type": "application/json",
                     "Content-Length": str(len(body))},
            status=200,
            preload_content=False,
        )
        raw._request_url = request.url  # requests_cache reads this when caching
        return self.build_response(request, raw)

    def _synthesize(self, iso3: str, level: int) -> Optional[bytes]:
        """Build the GeoJSON FeatureCollection for one country+level, or None
        if the country is not in the local gpkg (caller delegates to network)."""
        key = (iso3, level)
        cached = self._synth_cache.get(key)
        if cached is not None:
            return cached
        import pygadm
        from shapely.geometry import mapping
        sub = self._frame()
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
        with self._lock:
            self._synth_cache[key] = body
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
