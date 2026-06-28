"""AC-G4 battery for the local-first GADM transport (F-DR-21).

Proves the LocalGADMAdapter serves GADM boundaries from the pinned local gpkg
with zero network egress, byte/schema-identical to pygadm-from-network, across
the no-dissolve (NGA) and dissolve (MLI) source paths, under concurrency, with
graceful fallback for unbundled countries and a missing gpkg.

Fixture gpkg: tests/fixtures/gadm_subset_NGA_MLI.gpkg (merged GADM 4.1 subset;
NGA = 775 admin-2 finest [no-dissolve/exact], MLI = 704 finest / 50 admin-2
[dissolve path]). The independent "gpkg == official GADM" leg rests on the
deployment-engineer's 06-17 proof; the raw-from-network parity leg is deferred
to UC-Davis recovery (host is down — the reason this sprint exists).
"""
from __future__ import annotations

import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pytest

import pygadm
from prismpy.gadm_local import (
    GADM_HOST,
    LocalGADMAdapter,
    mount_local_gadm,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "gadm_subset_NGA_MLI.gpkg"
_GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{}_{}.json"


@pytest.fixture(scope="module")
def gpkg_truth():
    return gpd.read_file(_FIXTURE, layer="gadm_410")


class _CountingAdapter(LocalGADMAdapter):
    """Counts send invocations + locally-served responses (the positive
    no-egress confirmation: a request served here never hit the network)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.send_calls = 0
        self.served = 0

    def send(self, request, **kwargs):
        self.send_calls += 1
        resp = super().send(request, **kwargs)
        # 200 from our urllib3 body == locally synthesized (delegated network
        # calls under the egress block raise before returning).
        if getattr(resp, "status_code", None) == 200:
            self.served += 1
        return resp


@pytest.fixture
def mounted(monkeypatch):
    """Mount a counting LocalGADMAdapter on pygadm.session with a cold cache;
    restore on teardown so the mount never leaks across tests."""
    adapter = _CountingAdapter(str(_FIXTURE))
    prior = pygadm.session.adapters.get(GADM_HOST)
    pygadm.session.mount(GADM_HOST, adapter)
    pygadm.session.cache.clear()
    yield adapter
    pygadm.session.adapters.pop(GADM_HOST, None)
    if prior is not None:
        pygadm.session.mount(GADM_HOST, prior)
    pygadm.session.cache.clear()


@pytest.fixture
def block_ucdavis(monkeypatch):
    """Socket-level hard block: any DNS/connect to ucdavis RAISES, so a real
    egress attempt fails loud. Success under the block == proof of zero egress.
    Returns a dict whose 'count' is the number of egress attempts intercepted."""
    attempts = {"count": 0}
    real_gai = socket.getaddrinfo

    def guarded_getaddrinfo(host, *a, **k):
        if "ucdavis.edu" in str(host):
            attempts["count"] += 1
            raise ConnectionError(f"egress to {host} blocked by test")
        return real_gai(host, *a, **k)

    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, (tuple, list)) else address
        if "ucdavis" in str(host):
            attempts["count"] += 1
            raise ConnectionError("connect to ucdavis blocked by test")
        return real_connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    return attempts


# --- AC-G4(e) + AC-G4(a) synthesis fidelity: both source paths ---------------

@pytest.mark.parametrize(
    "iso3, level, expected, country, path",
    [
        ("NGA", 2, 775, "Nigeria", "no-dissolve"),
        ("NGA", 1, 37, "Nigeria", "no-dissolve"),
        ("NGA", 0, 1, "Nigeria", "no-dissolve"),
        ("MLI", 2, 50, "Mali", "dissolve"),
    ],
)
def test_synthesis_keyed_fidelity(mounted, gpkg_truth, iso3, level, expected,
                                  country, path):
    items = pygadm.Items(admin=iso3, content_level=level)
    assert len(items) == expected, (iso3, level)
    assert items[f"GID_{level}"].nunique() == expected
    # COUNTRY→NAME_0 rename reproduced (the gpkg column is NAME_0; the real
    # GeoJSON uses COUNTRY — the adapter emits COUNTRY so pygadm's :232 matches).
    assert (items["NAME_0"] == country).all()
    assert "COUNTRY" not in items.columns  # renamed away, like the network path
    # KEYED (never positional) GID→NAME pairing vs gpkg truth — the SF2 guard
    # against pygadm's by-position NAME_* overwrite mis-pairing names.
    truth = {r[f"GID_{level}"]: r[f"NAME_{level}"]
             for _, r in gpkg_truth[gpkg_truth.GID_0 == iso3].iterrows()}
    for _, row in items.iterrows():
        assert truth[row[f"GID_{level}"]] == row[f"NAME_{level}"]
    assert bool(items.geometry.is_valid.all())
    assert not items.geometry.is_empty.any()


def test_synthesized_json_body_carries_country_pre_rename(mounted):
    # Two-layer schema assertion: the adapter's JSON body has COUNTRY (pre
    # pygadm rename); the final frame surfaces NAME_0 (post rename, asserted
    # above). A body missing COUNTRY would silently drop the country name.
    body = mounted._synthesize("NGA", 2)
    fc = json.loads(body)
    props = fc["features"][0]["properties"]
    assert "COUNTRY" in props and props["COUNTRY"] == "Nigeria"
    assert "NAME_0" not in props  # NAME_0 is produced by pygadm's rename


# --- SF3: cold-cache served by adapter; warm-cache zero egress ---------------

def test_cold_cache_served_by_adapter_warm_cache_no_resend(mounted):
    pygadm.session.cache.clear()
    pygadm.Items(admin="NGA", content_level=2)
    assert mounted.served >= 1, "cold cache must be served by the local adapter"
    sends_after_cold = mounted.send_calls
    pygadm.Items(admin="NGA", content_level=2)  # warm
    assert mounted.send_calls == sends_after_cold, (
        "warm cache must serve from CachedSession without re-hitting the adapter"
    )


# --- AC-G4(a)/(b): zero boundary egress under a hard block -------------------

def test_bundled_lookup_zero_egress_under_block(mounted, block_ucdavis):
    for iso3, level, expected in [("NGA", 2, 775), ("MLI", 2, 50)]:
        items = pygadm.Items(admin=iso3, content_level=level)
        assert len(items) == expected
    assert mounted.served >= 2, "the adapter served the boundary requests"
    assert block_ucdavis["count"] == 0, (
        "ZERO egress: no DNS/connect to ucdavis was attempted for a bundled "
        "country (the adapter short-circuited the network)."
    )


# --- AC-G4(d): unbundled country degrades (delegates, no adapter crash) ------

def test_unbundled_country_delegates_to_network(mounted, block_ucdavis):
    # KEN is in pygadm's packaged name DB but NOT in the local gpkg → the
    # adapter returns None and delegates; under the block the network fails and
    # pygadm wraps it in its bare sentinel Exception (prismweb's D4 turns that
    # into a graceful None — covered in prismweb). The adapter must not crash.
    assert mounted._synthesize("KEN", 0) is None
    with pytest.raises(Exception) as ei:
        pygadm.Items(admin="KEN", content_level=1)
    assert "GADM server" in str(ei.value)
    assert block_ucdavis["count"] >= 1, "an unbundled country attempts egress"


# --- SF4: non-GADM URLs pass through (don't get synthesized) -----------------

def test_non_gadm_url_passes_through(mounted, block_ucdavis):
    # A non-gadm41 path on the ucdavis host must delegate to the network (here
    # blocked → raises), never be mis-served as a synthesized boundary.
    with pytest.raises(Exception):
        pygadm.session.get("https://geodata.ucdavis.edu/some/other/path.txt")
    assert block_ucdavis["count"] >= 1


# --- SF5: concurrent gpkg reads are stable -----------------------------------

def test_concurrent_synthesis_is_stable(mounted):
    # The adapter's gpkg read must be concurrency-safe (fresh derive per call,
    # lazy-load under a lock) — N parallel synths return identical bytes.
    ref = mounted._synthesize("NGA", 2)
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: mounted._synthesize("NGA", 2), range(24)))
    assert all(r == ref for r in results)
    assert all(len(json.loads(r)["features"]) == 775 for r in results)


def test_concurrent_items_through_session_stable(mounted):
    pygadm.session.cache.clear()
    errors, counts = [], []

    def _one(_):
        try:
            counts.append(len(pygadm.Items(admin="NGA", content_level=2)))
        except Exception as exc:  # noqa: BLE001 - record for the assertion
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(_one, range(12)))
    assert not errors, f"concurrent Items raised: {errors[:2]}"
    assert all(c == 775 for c in counts)


# --- AC-G2: mount idempotency + graceful-when-missing ------------------------

def test_mount_idempotent_and_logs(caplog):
    prior = pygadm.session.adapters.get(GADM_HOST)
    try:
        pygadm.session.adapters.pop(GADM_HOST, None)
        import logging
        with caplog.at_level(logging.INFO):
            assert mount_local_gadm(str(_FIXTURE)) is True
        assert any("LocalGADMAdapter mounted" in r.message for r in caplog.records)
        first = pygadm.session.adapters.get(GADM_HOST)
        assert isinstance(first, LocalGADMAdapter)
        assert mount_local_gadm(str(_FIXTURE)) is True  # idempotent
        assert pygadm.session.adapters.get(GADM_HOST) is first  # not re-mounted
    finally:
        pygadm.session.adapters.pop(GADM_HOST, None)
        if prior is not None:
            pygadm.session.mount(GADM_HOST, prior)


def test_mount_graceful_when_gpkg_missing(tmp_path, caplog):
    import logging
    missing = tmp_path / "nope.gpkg"
    before = dict(pygadm.session.adapters)
    with caplog.at_level(logging.INFO):
        assert mount_local_gadm(str(missing)) is False
    assert not isinstance(pygadm.session.adapters.get(GADM_HOST), LocalGADMAdapter)
    assert any("network path" in r.message for r in caplog.records)
    assert dict(pygadm.session.adapters) == before  # nothing mounted


# --- AC-G2: the exact pygadm pin (the coupling guard) ------------------------

def test_pygadm_pinned_exact():
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert '"pygadm==0.5.3"' in pyproject, (
        "pygadm must be pinned EXACT (==0.5.3): the adapter couples to its "
        "session/URL/by-position-NAME internals."
    )
    assert '"pygadm>=0.5,<1.0"' not in pyproject, "the loose range must be gone"
