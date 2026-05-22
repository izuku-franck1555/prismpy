"""GAEZ esri_client canonical-helper migration — behaviour tests.

Tests for ``EsriImageServiceClient.fetch_image`` after the bespoke retry
loop was replaced by ``retry_with_exponential_backoff``:

* In-band Esri error: HTTP 200 + a JSON ``{"error": {...}}`` envelope MUST
  raise ``EsriFetchError`` and be retried — never returned as raster bytes.
* Success short-circuit: 200 + image content returns on the first call.
* Exhaust parity: persistent failure raises ``EsriFetchError`` after exactly
  ``max_attempts`` calls.
* Cancel-during-retry: a cancel mid-storm raises ``PipelineCancelled`` at the
  next attempt boundary, well before the budget.
* Retry substage: a wired ``progress_callback`` receives
  ``retry_info={'kind': 'retry', ...}`` during the storm.
* ThreadPool cancel: a worker cancel propagates out of ``download_cultivar``.
"""
from __future__ import annotations

import functools

import pytest

from prismpy.sources.climate._cancel import PipelineCancelled
from prismpy.sources.common.retry import (
    retry_with_exponential_backoff as _REAL_HELPER,
)
from prismpy.sources.gaez.errors import EsriFetchError
from prismpy.sources.gaez.esri_client import EsriImageServiceClient


class _Query:
    """Minimal stand-in exposing the two methods fetch_image calls."""

    def to_mosaic_rule_json(self) -> str:
        return "{}"

    def to_where(self) -> str:
        return "WHERE 1=1"


class _Resp:
    def __init__(self, status, content_type, *, content=b"", json_data=None,
                 text=""):
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.content = content
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


class _RecordingCallback:
    def __init__(self):
        self.calls = []

    def on_substage_progress(self, stage, task, current, total, detail="",
                             retry_info=None):
        self.calls.append((stage, task, current, total, detail, retry_info))


@pytest.fixture()
def no_sleep(monkeypatch):
    """Force the helper used by esri_client to skip real backoff sleeps."""
    monkeypatch.setattr(
        "prismpy.sources.gaez.esri_client.retry_with_exponential_backoff",
        functools.partial(_REAL_HELPER, sleep_fn=lambda _s: None),
    )


def _patch_get(monkeypatch, fake_get):
    monkeypatch.setattr("requests.get", fake_get)


def test_success_first_attempt_returns_bytes(monkeypatch, no_sleep):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return _Resp(200, "image/tiff", content=b"TIFFBYTES")

    _patch_get(monkeypatch, fake_get)
    client = EsriImageServiceClient(retries=3)
    out = client.fetch_image(_Query())
    assert out == b"TIFFBYTES"
    assert len(calls) == 1, "success path must not retry"


def test_in_band_json_error_raises_and_retries(monkeypatch, no_sleep):
    """200 + JSON-error body MUST raise (not return the body) and the helper
    MUST retry it to exhaustion."""
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return _Resp(
            200, "application/json",
            json_data={"error": {"code": 499, "message": "token expired"}},
        )

    _patch_get(monkeypatch, fake_get)
    client = EsriImageServiceClient(retries=3)
    with pytest.raises(EsriFetchError) as ei:
        client.fetch_image(_Query())
    assert "ESRI_ERROR" in str(ei.value)
    # max_attempts = max(4, retries+1) = 4 → the in-band error was retried,
    # NOT silently returned as image data on the first 200.
    assert len(calls) == 4, (
        f"in-band error must be retried to exhaustion; got {len(calls)} calls"
    )


def test_persistent_5xx_exhausts_to_esri_fetch_error(monkeypatch, no_sleep):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return _Resp(503, "text/html", text="service unavailable")

    _patch_get(monkeypatch, fake_get)
    client = EsriImageServiceClient(retries=3)
    with pytest.raises(EsriFetchError) as ei:
        client.fetch_image(_Query())
    assert ei.value.status_code == 503
    assert len(calls) == 4


def test_cancel_during_retry_aborts_before_budget(monkeypatch, no_sleep):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return _Resp(503, "text/html", text="down")

    _patch_get(monkeypatch, fake_get)

    def cancel_check():
        # Trip once the first attempt has been made (i.e., in _on_retry).
        return len(calls) >= 1

    client = EsriImageServiceClient(retries=5)
    with pytest.raises(PipelineCancelled):
        client.fetch_image(_Query(), cancel_check=cancel_check)
    assert len(calls) == 1, (
        f"cancel must abort at the next attempt boundary, not run the full "
        f"budget; got {len(calls)} calls"
    )


def test_progress_callback_emits_retry_substage(monkeypatch, no_sleep):
    """A wired progress_callback receives the structured retry_info during
    the storm (translate stage, kind='retry')."""
    def fake_get(url, **kw):
        return _Resp(503, "text/html", text="down")

    _patch_get(monkeypatch, fake_get)
    cb = _RecordingCallback()
    client = EsriImageServiceClient(retries=3)
    with pytest.raises(EsriFetchError):
        client.fetch_image(_Query(), progress_callback=cb)

    retry_calls = [c for c in cb.calls if c[5] and c[5].get("kind") == "retry"]
    assert retry_calls, "expected at least one retry_info substage emit"
    for stage, task, current, total, detail, ri in retry_calls:
        assert stage == "translate"
        assert ri["provider"] == "GAEZ"
        assert ri["max_attempts"] == 4
        assert ri["attempt"] >= 1


def test_download_cultivar_early_cancel_propagates():
    """Cancel observed before fan-out raises PipelineCancelled (never
    rewritten as a GAEZ fetch error)."""
    from prismpy.sources.gaez.downloader import GAEZDownloader

    dl = GAEZDownloader(max_workers=1)
    with pytest.raises(PipelineCancelled):
        dl.download_cultivar("Highland maize", cancel_check=lambda: True)


def test_download_cultivar_threadpool_cancel_propagates(monkeypatch):
    """A worker observing cancel mid-retry propagates PipelineCancelled out
    of the ThreadPool path (carve-out before EsriFetchError accumulation)."""
    from prismpy.sources.gaez.downloader import (
        GAEZDownloader, INPUT_LEVELS, GAEZ_VARIABLES,
    )

    assert GAEZ_VARIABLES, "fixture precondition: GAEZ_VARIABLES non-empty"
    levels = list(INPUT_LEVELS.keys())[:1]
    assert levels, "fixture precondition: INPUT_LEVELS non-empty"

    dl = GAEZDownloader(max_workers=4)

    def boom(*a, **k):
        raise PipelineCancelled("worker-observed-cancel")

    monkeypatch.setattr(dl, "_fetch_one", boom)
    with pytest.raises(PipelineCancelled):
        dl.download_cultivar(
            "Highland maize", input_levels=levels, cancel_check=lambda: False
        )
