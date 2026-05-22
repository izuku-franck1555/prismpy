"""Unit tests for ``classify_to_event_dict``.

The classifier is structural extraction only — no user-facing copy,
no Django / prismweb references. Every assertion here pins a piece of
the producer-boundary contract that the consumer (prismweb) relies on.
"""
from __future__ import annotations

import pytest

from prismpy.errors import ErrorEventDict, classify_to_event_dict


def test_basic_exception_carries_error_class_and_message() -> None:
    ev = classify_to_event_dict(ValueError("boom"))
    assert ev["error_class"] == "ValueError"
    assert ev["message"] == "boom"
    # absent attrs surface as None, not missing keys
    assert ev["source"] is None
    assert ev["missing_tiles"] is None
    assert ev["partial_progress"] is None
    assert ev["recoverable"] is None


def test_typed_exception_preserves_source_missing_tiles_recoverable() -> None:
    class _ClimateDownloadError(Exception):
        pass

    e = _ClimateDownloadError("NASA POWER incomplete: 4/4 tiles unfetched")
    e.source = "nasa_power"
    e.missing_tiles = [1, 2, 3, 4]
    e.recoverable = False
    ev = classify_to_event_dict(e)
    assert ev["source"] == "nasa_power"
    assert ev["missing_tiles"] == [1, 2, 3, 4]
    assert ev["recoverable"] is False


def test_partial_progress_derives_from_missing_tiles_and_grid_total() -> None:
    e = Exception("partial fail")
    e.missing_tiles = [10, 20]
    ev = classify_to_event_dict(e, {"grid_total": 5})
    assert ev["partial_progress"] == {
        "succeeded": 3, "failed": 2, "total": 5,
    }


def test_partial_progress_is_none_without_grid_total() -> None:
    e = Exception("x")
    e.missing_tiles = [1]
    ev = classify_to_event_dict(e, {"platform": "acea"})
    assert ev["partial_progress"] is None


def test_partial_progress_is_none_without_missing_tiles() -> None:
    ev = classify_to_event_dict(ValueError("x"), {"grid_total": 5})
    assert ev["partial_progress"] is None


def test_missing_assets_attr_is_a_supported_alias() -> None:
    """Translators that expose the failed-asset list under
    ``missing_assets`` (not ``missing_tiles``) still surface it."""
    e = Exception("y")
    e.missing_assets = ["a", "b", "c"]
    ev = classify_to_event_dict(e, {"grid_total": 10})
    assert ev["missing_tiles"] == ["a", "b", "c"]
    assert ev["partial_progress"] == {
        "succeeded": 7, "failed": 3, "total": 10,
    }


def test_returned_dict_has_canonical_keyset() -> None:
    """The payload always carries exactly the canonical six keys so the
    consumer can introspect without `KeyError` regardless of which
    fields the exception happened to carry."""
    ev = classify_to_event_dict(ValueError("x"))
    assert set(ev.keys()) == {
        "error_class", "message", "source",
        "missing_tiles", "partial_progress", "recoverable",
    }


def test_partial_progress_prefers_exc_total_over_context_grid_total() -> None:
    """When the exception carries ``total`` in the correct unit (typed
    raise-site contract), the classifier uses it even if context supplies
    a different ``grid_total`` — the catch-site's pixel-grid count would
    be a unit mismatch ("47,996 of 48,000" vs the honest "96 of 100")."""
    e = Exception("x")
    e.missing_tiles = [1, 2, 3, 4]
    e.total = 100
    ev = classify_to_event_dict(e, {"grid_total": 48000})  # wrong unit
    assert ev["partial_progress"] == {
        "succeeded": 96, "failed": 4, "total": 100,
    }


def test_acea_typed_exception_yields_correct_partial_progress() -> None:
    """Behavioural regression: the real ACEA raise constructor +
    classify produce a `partial_progress` in the right unit (count of
    30-arcmin cell IDs, NOT pixel-grid count)."""
    from prismpy.sources.climate.errors import ClimateDownloadError

    e = ClimateDownloadError(
        "NASA POWER incomplete",
        missing_tiles=[1, 2, 3, 4],
        source="nasa_power",
        total=100,
    )
    ev = classify_to_event_dict(e)
    assert ev["error_class"] == "ClimateDownloadError"
    assert ev["source"] == "nasa_power"
    assert ev["missing_tiles"] == [1, 2, 3, 4]
    assert ev["partial_progress"] == {
        "succeeded": 96, "failed": 4, "total": 100,
    }


def test_deferred_untyped_value_error_yields_no_partial_progress() -> None:
    """PYTHIA / SARRA-Py / CRAFT raise plain ``ValueError`` without
    ``missing_tiles`` or ``total``; ``partial_progress`` stays ``None`` —
    honest: no count to report rather than a fabricated number."""
    ev = classify_to_event_dict(ValueError("config error"))
    assert ev["error_class"] == "ValueError"
    assert ev["partial_progress"] is None


def test_keyset_matches_typed_dict_declaration() -> None:
    """Sanity: the runtime keyset is exactly the ``ErrorEventDict``
    declaration's keys — guards against drift between the producer
    classifier and the canonical TypedDict the consumer reads."""
    ev = classify_to_event_dict(ValueError("x"))
    declared = set(ErrorEventDict.__annotations__.keys())
    assert set(ev.keys()) == declared
