"""Structural pin: ISIMIP3b client substrate (Sprint G AC-G-1).

Per ``prismpy/.local/SPRINT-G-VERIFICATION-STRATEGY.md`` §2 AC-G-1:

* §1.1 ``isimip-client>=2.0,<3.0`` declared in pyproject ``[project.dependencies]``.
* §1.2 ``isimip_client.client.ISIMIPClient`` importable in the dev venv.
* §1.3 ``prismpy.data_sources.isimip3b`` exposes the required public API.
* §1.4 missing ``isimip-client`` raises ``ImportError`` loud (no silent skip).
* §1.5 module docstring cites the ISIMIP Terms of Use.
* §1.6 ``discover_datasets`` resolves ssp585 → ``InputData`` and ssp245 → ``SecondaryInputData``.

The §1.6 verification uses a fake ``ISIMIP3bClient`` so the structural
test does not depend on network availability — the upstream API call
itself is tested by AC-G-2's mocked-fixture cache discipline tests.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List

import pytest

from prismpy.data_sources import isimip3b


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ── §1.1 pyproject pin ───────────────────────────────────────────────


def test_isimip_client_declared_in_pyproject_with_pinned_range() -> None:
    """``isimip-client>=2.0,<3.0`` must appear in ``[project.dependencies]``.

    The upper bound caps below 3.0 so a future major-version refactor
    surfaces at sprint time rather than as silent contract drift.
    """
    pyproject = _project_root() / "pyproject.toml"
    cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = cfg["project"]["dependencies"]
    matches = [d for d in deps if d.startswith("isimip-client")]
    assert len(matches) == 1, (
        f"Expected exactly one isimip-client pin in [project.dependencies], got: {matches}"
    )
    pin = matches[0]
    assert pin == "isimip-client>=2.0,<3.0", (
        f"isimip-client pin must be exactly 'isimip-client>=2.0,<3.0' "
        f"per Sprint G AC-G-1; got {pin!r}"
    )


# ── §1.2 isimip_client importable ────────────────────────────────────


def test_isimip_client_library_is_importable() -> None:
    """The upstream library must be installed in the dev venv."""
    from isimip_client.client import ISIMIPClient  # noqa: F401


# ── §1.3 prismpy.data_sources.isimip3b public API ────────────────────


_REQUIRED_PUBLIC_NAMES = (
    "ISIMIP3bClient",
    "discover_datasets",
    "cached_cutout",
    "IsimipFetchError",
    "IsimipDatasetNotFoundError",
    "InvalidIsimipResponseError",
    "CacheDirectoryError",
    "CacheWriteError",
)


def test_isimip3b_module_exports_required_public_api() -> None:
    """Public API surface must match the contract."""
    missing = [name for name in _REQUIRED_PUBLIC_NAMES if not hasattr(isimip3b, name)]
    assert not missing, f"prismpy.data_sources.isimip3b missing: {missing}"


def test_isimip3b_module_lists_all_public_names_in_dunder_all() -> None:
    """``__all__`` should match the required public set so star imports
    do not accidentally miss a contract API."""
    declared = set(getattr(isimip3b, "__all__", ()))
    required = set(_REQUIRED_PUBLIC_NAMES)
    missing = required - declared
    assert not missing, (
        f"prismpy.data_sources.isimip3b.__all__ missing names: {sorted(missing)}"
    )


def test_typed_exception_hierarchy() -> None:
    """``IsimipDatasetNotFoundError`` and ``InvalidIsimipResponseError`` are
    subclasses of ``IsimipFetchError``; ``CacheWriteError`` is a subclass
    of ``CacheDirectoryError``. Discrimination via ``except`` works as
    contract specifies."""
    assert issubclass(isimip3b.IsimipDatasetNotFoundError, isimip3b.IsimipFetchError)
    assert issubclass(isimip3b.InvalidIsimipResponseError, isimip3b.IsimipFetchError)
    assert issubclass(isimip3b.CacheWriteError, isimip3b.CacheDirectoryError)
    # Cache and fetch hierarchies are intentionally distinct.
    assert not issubclass(isimip3b.CacheDirectoryError, isimip3b.IsimipFetchError)
    assert not issubclass(isimip3b.IsimipFetchError, isimip3b.CacheDirectoryError)


def test_dataset_not_found_carries_specifiers_field() -> None:
    """``IsimipDatasetNotFoundError`` must carry a ``specifiers`` dict."""
    err = isimip3b.IsimipDatasetNotFoundError(
        "msg", specifiers={"gcm": "fake-gcm", "scenario": "ssp585"}
    )
    assert err.specifiers == {"gcm": "fake-gcm", "scenario": "ssp585"}
    # Default to empty dict when no specifiers provided
    bare = isimip3b.IsimipDatasetNotFoundError("msg")
    assert bare.specifiers == {}


# ── §1.4 import-time guard fails loud on missing isimip-client ───────


def test_module_fails_loud_when_isimip_client_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reimporting the module with isimip_client absent raises a loud
    ImportError naming the dependency."""
    import importlib

    # Drop both the upstream library and our wrapper from sys.modules,
    # then mark the upstream as unavailable so the wrapper's import
    # path actually exercises the fail-loud branch.
    for name in list(sys.modules):
        if name.startswith("isimip_client") or name == "prismpy.data_sources.isimip3b":
            monkeypatch.delitem(sys.modules, name, raising=False)

    # Block the upstream import path. ``__init__`` is what
    # ``from isimip_client.client import ISIMIPClient`` resolves through.
    monkeypatch.setitem(sys.modules, "isimip_client", None)

    with pytest.raises(ImportError) as exc_info:
        importlib.import_module("prismpy.data_sources.isimip3b")
    assert "isimip-client" in str(exc_info.value), (
        "ImportError must name the dependency so the user can install it"
    )


# ── §1.5 module docstring cites ISIMIP Terms of Use ──────────────────


def test_module_docstring_cites_isimip_terms_of_use() -> None:
    """The module docstring must cite the ISIMIP Terms of Use URL so
    consumers of derived packages inherit the citation chain."""
    doc = isimip3b.__doc__ or ""
    assert "isimip.org" in doc.lower(), (
        "Module docstring must cite the ISIMIP project URL"
    )
    assert "terms of use" in doc.lower(), (
        "Module docstring must reference the ISIMIP Terms of Use"
    )


# ── §1.6 discover_datasets product mapping ───────────────────────────


class _FakeISIMIP3bClient:
    """Records the kwargs passed to ``datasets`` and returns canned
    fixture responses. Used by §1.6 to verify that ssp585 routes through
    ``InputData`` and ssp245 through ``SecondaryInputData`` without
    depending on real network calls."""

    def __init__(self, response: Dict[str, Any]) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._response = response

    def datasets(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        return self._response


def _stub_dataset(dataset_id: str) -> Dict[str, Any]:
    return {
        "id": dataset_id,
        "version": "20240101",
        "doi": "10.5281/zenodo.fake",
        "files": [],
    }


def test_discover_datasets_uses_input_data_product_for_ssp585() -> None:
    response = {"results": [_stub_dataset("ssp585-fake")]}
    fake = _FakeISIMIP3bClient(response)
    result = isimip3b.discover_datasets(
        fake,  # type: ignore[arg-type]
        gcm="gfdl-esm4",
        scenario="ssp585",
        variable="tasmax",
        time_slice=(2046, 2065),
    )
    assert result["id"] == "ssp585-fake"
    assert len(fake.calls) == 1
    assert fake.calls[0]["product"] == "InputData"


def test_discover_datasets_uses_secondary_input_data_product_for_ssp245() -> None:
    response = {"results": [_stub_dataset("ssp245-fake")]}
    fake = _FakeISIMIP3bClient(response)
    result = isimip3b.discover_datasets(
        fake,  # type: ignore[arg-type]
        gcm="ipsl-cm6a-lr",
        scenario="ssp245",
        variable="pr",
        time_slice=(2046, 2065),
    )
    assert result["id"] == "ssp245-fake"
    assert fake.calls[0]["product"] == "SecondaryInputData"


def test_discover_datasets_passes_climate_forcing_and_variable() -> None:
    response = {"results": [_stub_dataset("any")]}
    fake = _FakeISIMIP3bClient(response)
    isimip3b.discover_datasets(
        fake,  # type: ignore[arg-type]
        gcm="mpi-esm1-2-hr",
        scenario="ssp585",
        variable="hurs",
        time_slice=(2086, 2100),
    )
    call = fake.calls[0]
    assert call["climate_forcing"] == "mpi-esm1-2-hr"
    assert call["climate_variable"] == "hurs"
    assert call["climate_scenario"] == "ssp585"
    assert call["simulation_round"] == "ISIMIP3b"


def test_discover_datasets_raises_typed_error_when_no_results() -> None:
    """Empty result list → typed ``IsimipDatasetNotFoundError`` carrying
    the failing specifiers tuple."""
    fake = _FakeISIMIP3bClient({"results": []})
    with pytest.raises(isimip3b.IsimipDatasetNotFoundError) as exc_info:
        isimip3b.discover_datasets(
            fake,  # type: ignore[arg-type]
            gcm="gfdl-esm4",
            scenario="ssp585",
            variable="tasmax",
            time_slice=(3000, 3001),
        )
    spec = exc_info.value.specifiers
    assert spec.get("climate_forcing") == "gfdl-esm4"
    assert spec.get("climate_scenario") == "ssp585"
    assert spec.get("climate_variable") == "tasmax"
    assert spec.get("time_slice") == (3000, 3001)


def test_discover_datasets_rejects_unsupported_gcm() -> None:
    fake = _FakeISIMIP3bClient({"results": []})
    with pytest.raises(isimip3b.IsimipDatasetNotFoundError) as exc_info:
        isimip3b.discover_datasets(
            fake,  # type: ignore[arg-type]
            gcm="hadgem3-gc31-ll",  # not in the primary core ensemble
            scenario="ssp585",
            variable="tasmax",
            time_slice=(2046, 2065),
        )
    assert "hadgem3-gc31-ll" in str(exc_info.value)
    assert exc_info.value.specifiers.get("gcm") == "hadgem3-gc31-ll"


def test_discover_datasets_rejects_unsupported_variable() -> None:
    fake = _FakeISIMIP3bClient({"results": []})
    with pytest.raises(isimip3b.IsimipDatasetNotFoundError):
        isimip3b.discover_datasets(
            fake,  # type: ignore[arg-type]
            gcm="gfdl-esm4",
            scenario="ssp585",
            variable="psl",  # not in the supported allowlist
            time_slice=(2046, 2065),
        )


def test_discover_datasets_rejects_unsupported_scenario() -> None:
    fake = _FakeISIMIP3bClient({"results": []})
    with pytest.raises(isimip3b.IsimipDatasetNotFoundError) as exc_info:
        isimip3b.discover_datasets(
            fake,  # type: ignore[arg-type]
            gcm="gfdl-esm4",
            scenario="ssp126",  # not in Sprint G product map
            variable="tasmax",
            time_slice=(2046, 2065),
        )
    assert "ssp126" in str(exc_info.value)


def test_discover_datasets_rejects_inverted_time_slice() -> None:
    fake = _FakeISIMIP3bClient({"results": [_stub_dataset("any")]})
    with pytest.raises(isimip3b.IsimipDatasetNotFoundError):
        isimip3b.discover_datasets(
            fake,  # type: ignore[arg-type]
            gcm="gfdl-esm4",
            scenario="ssp585",
            variable="tasmax",
            time_slice=(2065, 2046),  # start > end
        )


def test_discover_datasets_wraps_upstream_failure_as_typed_error() -> None:
    """A network/HTTP failure inside the upstream client must surface
    as ``IsimipFetchError`` (NOT the bare upstream exception)."""

    class _BoomClient:
        def datasets(self, **_kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("simulated upstream network failure")

    with pytest.raises(isimip3b.IsimipFetchError):
        isimip3b.discover_datasets(
            _BoomClient(),  # type: ignore[arg-type]
            gcm="gfdl-esm4",
            scenario="ssp585",
            variable="tasmax",
            time_slice=(2046, 2065),
        )


# AC-G-2 cached_cutout body lives in tests/structural/test_isimip3b_cached_cutout.py.
# This file remains the AC-G-1 substrate-only structural pin.
