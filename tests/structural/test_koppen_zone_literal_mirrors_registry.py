"""Structural pin: ``KoppenZone`` Literal mirrors the JSON registry.

Sprint E.2 AC-E2-20 + Builder Sub-CA-A (Draft 2.1 mini-amendment).
The ``KoppenZone`` Literal at ``prismpy/koppen/zones.py`` is a
manually-maintained mirror of the Köppen-Geiger zone-codes the
``zone_aggregates_v1.json`` registry declares. This pin is the
canonical-source-or-pin enforcement (durable §24): adding a zone to
the JSON registry without extending the Literal here (or vice versa)
fails the pin loud at CI time, NOT silently at runtime.

Sprint S precedent: ``WarningCategory`` enum mirrors a registry-of-
truth via a similar AST + roster-comparison pin. Same discipline,
different surface.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path

import pytest

from prismpy.koppen import zones as koppen_zones


def _registry_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "prismpy"
        / "koppen"
        / "data"
        / "zone_aggregates_v1.json"
    )


def _registry_zone_codes() -> set[str]:
    """Read the JSON registry directly and return the set of zone
    codes. The pin compares this against the Literal's ``get_args``
    output; any drift surfaces as a clear set-difference message.
    """
    payload = json.loads(_registry_path().read_text(encoding="utf-8"))
    zones = payload["zones"]
    return set(zones.keys())


# ── §1 the manual Literal mirrors the registry exactly ──────────────


def test_koppen_zone_literal_args_match_registry_keys() -> None:
    """The set-equality pin: every Literal arg is a registry key, and
    every registry key is a Literal arg. Drift in either direction
    fails with a clear set-difference message that names the offending
    code(s)."""
    literal_args = set(typing.get_args(koppen_zones.KoppenZone))
    registry_codes = _registry_zone_codes()

    missing_in_literal = registry_codes - literal_args
    extra_in_literal = literal_args - registry_codes

    assert not missing_in_literal, (
        f"KoppenZone Literal is missing zones present in registry "
        f"{_registry_path().name}: {sorted(missing_in_literal)}. "
        f"Extend ``prismpy/koppen/zones.py::KoppenZone`` to include "
        f"these codes; the registry is the canonical source per "
        f"durable §24 canonical-source-or-pin."
    )
    assert not extra_in_literal, (
        f"KoppenZone Literal declares zones absent from the registry "
        f"{_registry_path().name}: {sorted(extra_in_literal)}. Either "
        f"add these zones to the JSON registry (with full per-zone "
        f"aggregates per AC-F-3 schema), or remove them from the "
        f"Literal. The registry is the canonical source."
    )


def test_koppen_zone_literal_carries_five_west_africa_zones() -> None:
    """Sprint E.2 ships with the West-Africa-relevant registry from
    Sprint F: Af / Aw / BSh / Cfa / Cwa. Cwb / Cwc (East African
    Highlands) are V2-19.5 Data Bootstrapper future-explore per
    task #131. This pin documents the current scope explicitly so a
    future expansion lands as an intentional contract change rather
    than an accidental Literal extension."""
    literal_args = set(typing.get_args(koppen_zones.KoppenZone))
    expected = {"Af", "Aw", "BSh", "Cfa", "Cwa"}
    assert literal_args == expected, (
        f"Sprint E.2 KoppenZone scope = {sorted(expected)}. Got "
        f"{sorted(literal_args)}. East African Highland zones (Cwb, "
        f"Cwc) belong to V2-19.5 Data Bootstrapper future-explore."
    )


# ── §2 module-load fail-loud discipline ─────────────────────────────


def test_module_exposes_only_koppen_zone_in_dunder_all() -> None:
    """The dunder-all is the canonical-source surface: only
    ``KoppenZone`` is exported. Internal helpers (loader, registry
    path constant) stay private to keep the import surface tight."""
    assert koppen_zones.__all__ == ["KoppenZone"]


def test_registry_zone_codes_constant_is_immutable_tuple() -> None:
    """The module's load-time discovery stores the registry codes as
    a tuple — immutability is the implicit contract that consumers
    can't accidentally mutate the canonical roster post-import."""
    codes = koppen_zones._REGISTRY_ZONE_CODES
    assert isinstance(codes, tuple)
    assert codes == tuple(sorted(codes)), (
        "Registry zone codes are stored sorted for deterministic "
        "test-comparison + diff-friendly review."
    )


def test_load_registry_raises_import_error_when_file_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module-load fail-loud discipline: a missing registry file
    surfaces as ``ImportError`` with an actionable message, NOT a
    silent empty Literal that rejects every input at runtime. Per
    Evaluator CA-1 contract sub-criterion + ``feedback_no_data_cooking.md``
    silent-skip-as-class avoidance."""
    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(koppen_zones, "_KOPPEN_REGISTRY_PATH", missing_path)
    with pytest.raises(ImportError, match="missing at"):
        koppen_zones._load_registry_zone_codes()


def test_load_registry_raises_import_error_on_empty_zones_dict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``zones`` dict is also fail-loud — an empty Literal
    would silently reject every zone-typed input."""
    empty_path = tmp_path / "empty_zones.json"
    empty_path.write_text(json.dumps({"zones": {}}), encoding="utf-8")
    monkeypatch.setattr(koppen_zones, "_KOPPEN_REGISTRY_PATH", empty_path)
    with pytest.raises(ImportError, match="empty 'zones' dict"):
        koppen_zones._load_registry_zone_codes()


def test_load_registry_raises_import_error_on_malformed_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry whose top-level shape isn't ``{'zones': dict}``
    surfaces as ``ImportError`` rather than silent ``KeyError`` /
    ``AttributeError`` blowing up at first import-site."""
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(json.dumps([{"code": "Af"}]), encoding="utf-8")
    monkeypatch.setattr(koppen_zones, "_KOPPEN_REGISTRY_PATH", malformed_path)
    with pytest.raises(ImportError, match="unexpected shape"):
        koppen_zones._load_registry_zone_codes()


def test_load_registry_raises_import_error_on_unparseable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrupted JSON in the registry file surfaces as
    ``ImportError`` with the upstream parse error chained — NOT a
    raw ``json.JSONDecodeError`` propagating up."""
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(koppen_zones, "_KOPPEN_REGISTRY_PATH", broken_path)
    with pytest.raises(ImportError, match="cannot parse"):
        koppen_zones._load_registry_zone_codes()


# ── §3 negative-control: walker catches synthetic registry drift ────


def test_drift_detection_when_literal_diverges_from_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative-control drill: simulate a registry that adds a new
    zone. Assert ``_load_registry_zone_codes()`` returns the new set
    — the mirror pin would then catch the Literal not having that
    zone (via the test_koppen_zone_literal_args_match_registry_keys
    set-equality assertion). This drill verifies the loader's
    extension behaviour without mutating the real registry."""
    extended_path = tmp_path / "extended.json"
    extended_path.write_text(
        json.dumps({
            "zones": {
                "Af": {},
                "Aw": {},
                "BSh": {},
                "Cfa": {},
                "Cwa": {},
                "Cwb": {},  # synthetic addition — Literal would not match
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(koppen_zones, "_KOPPEN_REGISTRY_PATH", extended_path)
    extended_codes = set(koppen_zones._load_registry_zone_codes())
    literal_args = set(typing.get_args(koppen_zones.KoppenZone))
    drift = extended_codes - literal_args
    assert drift == {"Cwb"}, (
        f"Loader must surface the synthetic Cwb addition; got drift {drift}"
    )
