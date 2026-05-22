"""TAMSAT canonical-helper migration — behaviour tests.

Tests for the ``_download_nc`` HTTP path after the bespoke single 5xx retry
was replaced by ``retry_with_exponential_backoff``. Driven through
``_download_tamsat`` (``_download_nc`` is a closure). All scenarios return a
non-200 status so no ``.nc`` file is written and Phase 2 stays a no-op
(rasterio is imported but the conversion loop iterates an empty dir).

Preserved invariants under test:
* 404 → ``"skipped"`` fast-path is NON-retryable (exactly one HTTP call).
* 5xx is retried to ``max_attempts=3`` (initial + 2 retries).
* a cancel observed mid-retry raises ``PipelineCancelled`` at the next
  attempt boundary (does not run the full retry budget).
"""
from __future__ import annotations

import functools
from datetime import date

import pytest

# Phase 2 imports these unconditionally; skip gracefully where the rasterio
# stack is unavailable (CI has it; the full suite exercises the real path).
pytest.importorskip("xarray")
pytest.importorskip("rioxarray")

from prismpy.sources.climate._cancel import PipelineCancelled  # noqa: E402
from prismpy.sources.common.retry import (  # noqa: E402
    retry_with_exponential_backoff as _REAL_HELPER,
)
from prismpy.sources.climate.tamsat import TAMSATSource  # noqa: E402

_BOUNDS = [10.0, 0.0, 5.0, 5.0]  # [lat_NW, lon_NW, lat_SE, lon_SE]


class _Resp:
    def __init__(self, status, *, content=b"", text=""):
        self.status_code = status
        self.content = content
        self.text = text

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture()
def no_sleep(monkeypatch):
    monkeypatch.setattr(
        "prismpy.sources.climate.tamsat.retry_with_exponential_backoff",
        functools.partial(_REAL_HELPER, sleep_fn=lambda _s: None),
    )


def _src(tmp_path):
    return TAMSATSource(cache_dir=tmp_path / "cache")


def test_404_is_skipped_without_retry(tmp_path, monkeypatch, no_sleep):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return _Resp(404)

    monkeypatch.setattr("requests.get", fake_get)
    src = _src(tmp_path)
    src._download_tamsat(
        bounds=_BOUNDS,
        start_date=date(2020, 1, 15),
        end_date=date(2020, 1, 17),
        output_dir=tmp_path / "out",
        region_name="testreg",
        max_workers=1,
    )
    # 3 dates, one HTTP call each — 404 is a non-retryable fast-path.
    assert len(calls) == 3, f"404 must not retry; got {len(calls)} calls"


def test_5xx_retries_to_max_attempts(tmp_path, monkeypatch, no_sleep):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return _Resp(503, text="service unavailable")

    monkeypatch.setattr("requests.get", fake_get)
    src = _src(tmp_path)
    src._download_tamsat(
        bounds=_BOUNDS,
        start_date=date(2020, 1, 15),
        end_date=date(2020, 1, 15),
        output_dir=tmp_path / "out",
        region_name="testreg",
        max_workers=1,
    )
    # Single date, persistent 5xx → initial + 2 retries = 3 HTTP calls.
    assert len(calls) == 3, (
        f"5xx must retry to max_attempts=3; got {len(calls)} calls"
    )


def test_cancel_during_retry_aborts(tmp_path, monkeypatch, no_sleep):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return _Resp(503, text="down")

    monkeypatch.setattr("requests.get", fake_get)

    def cancel_check():
        # Trip once the first attempt has fired (i.e., inside _on_retry).
        return len(calls) >= 1

    src = _src(tmp_path)
    with pytest.raises(PipelineCancelled):
        src._download_tamsat(
            bounds=_BOUNDS,
            start_date=date(2020, 1, 15),
            end_date=date(2020, 1, 15),
            output_dir=tmp_path / "out",
            region_name="testreg",
            max_workers=1,
            cancel_check=cancel_check,
        )
    assert len(calls) == 1, (
        f"cancel must abort at the next attempt boundary; got {len(calls)}"
    )


def test_retry_observer_emits_substage(tmp_path, monkeypatch, no_sleep):
    """A wired retry_observer (the bridge closure) receives the structured
    retry payload during a TAMSAT 5xx storm."""
    def fake_get(url, **kw):
        return _Resp(503, text="down")

    monkeypatch.setattr("requests.get", fake_get)

    emits = []

    def observer(attempt, max_attempts, sleep_s):
        emits.append((attempt, max_attempts, sleep_s))

    src = _src(tmp_path)
    src._download_tamsat(
        bounds=_BOUNDS,
        start_date=date(2020, 1, 15),
        end_date=date(2020, 1, 15),
        output_dir=tmp_path / "out",
        region_name="testreg",
        max_workers=1,
        retry_observer=observer,
    )
    # max_attempts=3 → 2 retry-attempt emits.
    assert len(emits) == 2, f"expected 2 retry emits; got {emits}"
    assert all(m == 3 for _, m, _ in emits)
