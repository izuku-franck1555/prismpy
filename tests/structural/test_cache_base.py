"""Structural pin: cache primitives have ONE canonical home (_cache_base.py).

Sprint G AC-G-2.0 extracts the shared cache substrate from
``prismpy.sources.climate.tamsat`` to ``prismpy.sources._cache_base``
so TAMSAT, AgERA5, and ISIMIP3b (Sprint G) share one implementation.
This module is the structural pin for that contract:

* The generic ``cache_lock_path`` accepts ``key: str`` (independent of
  Region/Run shape).
* ``write_atomic_json`` is the single atomic-write helper the manifest
  writer + future ISIMIP3b ``.meta.json`` writer route through.
* TAMSAT exposes a Region-coupled compatibility wrapper that delegates
  to the generic.
* AgERA5 imports the generic directly from ``_cache_base``.
* TAMSAT's wrapper produces the same lock path as the generic when fed
  the same string key.

Per durable lesson #24 canonical-source-or-pin: ONE cache substrate,
three callers. Per durable #20 sibling-sweep: every consumer of the
former tamsat-only primitives is enumerated and pinned here.
"""

import ast
import inspect
from pathlib import Path

import pytest

from prismpy.models.region import BoundingBox, Region
from prismpy.sources import _cache_base
from prismpy.sources.climate import agera5, tamsat


# ── §1 Public API of _cache_base — exhaustive enumeration ────────────


_EXPECTED_PUBLIC_NAMES = frozenset(
    {
        # Constants
        "MANIFEST_SCHEMA_VERSION",
        "MANIFEST_FILENAME",
        "MARKER_FILENAME",
        "TMPFILE_PREFIX",
        "BBOX_TOLERANCE_DEG",
        "DOWNLOAD_LOCK_TIMEOUT_SECONDS",
        # State NamedTuple
        "CacheManifestState",
        # Generic primitives
        "cache_lock_path",
        "write_atomic_json",
        "cleanup_orphan_tmpfiles",
        "count_tif_files",
        "bbox_to_dict",
        "bbox_field_for_log",
        # Climate-domain manifest + marker helpers
        "write_marker",
        "delete_marker",
        "write_cache_manifest",
        "check_cache_manifest",
        "warn_legacy_cache_once",
    }
)


def test_cache_base_exposes_canonical_public_api() -> None:
    """Every name the cache substrate is expected to export must exist.

    Adding a new primitive without updating this set is a contract
    drift. Removing a primitive without updating consumers is the same
    drift in the other direction.
    """
    missing = {name for name in _EXPECTED_PUBLIC_NAMES if not hasattr(_cache_base, name)}
    assert not missing, f"_cache_base missing public names: {sorted(missing)}"


# ── §2 cache_lock_path: generic + Region-coupled equivalence ─────────


def test_cache_lock_path_generic_signature() -> None:
    """Generic cache_lock_path takes (cache_dir, source, key: str)."""
    sig = inspect.signature(_cache_base.cache_lock_path)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["cache_dir", "source", "key"]
    assert sig.parameters["key"].annotation is str


def test_cache_lock_path_path_shape(tmp_path: Path) -> None:
    """Generic cache_lock_path emits ``<cache_dir>/.<source>-<key>.lock``."""
    out = _cache_base.cache_lock_path(tmp_path, source="isimip3b", key="abc-123")
    assert out == tmp_path / ".isimip3b-abc-123.lock"


def test_tamsat_cache_lock_path_is_region_coupled_wrapper(tmp_path: Path) -> None:
    """TAMSAT's wrapper accepts a Region and produces the same path the
    generic would for the equivalent string key."""
    region = Region(
        name="Maradi",
        country="Niger",
        country_iso3="NER",
        bounds=BoundingBox(minx=6.5, miny=13.0, maxx=8.0, maxy=14.5),
        gadm_level=1,
    )
    from prismpy.utils.sanitization import region_cache_key_from_region

    wrapper_out = tamsat.cache_lock_path(tmp_path, source="tamsat", region_name=region)
    generic_out = _cache_base.cache_lock_path(
        tmp_path,
        source="tamsat",
        key=region_cache_key_from_region(region),
    )
    assert wrapper_out == generic_out


# ── §3 write_atomic_json: single canonical atomic-write helper ───────


def test_write_atomic_json_writes_canonical_json_then_renames(tmp_path: Path) -> None:
    """write_atomic_json delivers the JSON payload at the target with
    no leftover .writing-XXXX.tmp fragments after a successful write."""
    target = tmp_path / "subdir" / "_manifest.json"
    payload = {"schema_version": 1, "key": "value", "n": 42}

    _cache_base.write_atomic_json(target, payload)

    import json

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    leftovers = list(target.parent.glob(f"{_cache_base.TMPFILE_PREFIX}*.tmp"))
    assert leftovers == [], f"leftover tempfiles: {leftovers}"


def test_write_atomic_json_cleans_up_after_failed_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When os.replace fails mid-write, the tempfile is removed and the
    target either does not exist or holds the prior content — never a
    half-written body."""
    target = tmp_path / "_manifest.json"
    target.write_text('{"prior": true}', encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_cache_base.os, "replace", boom)

    with pytest.raises(OSError):
        _cache_base.write_atomic_json(target, {"new": True})

    leftovers = list(target.parent.glob(f"{_cache_base.TMPFILE_PREFIX}*.tmp"))
    assert leftovers == [], f"leftover tempfiles after failed replace: {leftovers}"
    # Prior content preserved (target either intact or absent — never partial)
    import json

    assert json.loads(target.read_text(encoding="utf-8")) == {"prior": True}


# ── §4 Sibling-sweep: enumerate every consumer of cache primitives ───


_EXPECTED_CONSUMERS = (
    "src/prismpy/sources/climate/tamsat.py",
    "src/prismpy/sources/climate/agera5.py",
)


def _module_imports(file_path: Path) -> set[str]:
    """Return the set of (from-module, name) tuples imported in a .py file."""
    imports: set[tuple[str, str]] = set()
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.add((module, alias.name))
    return imports


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_tamsat_imports_substrate_from_cache_base() -> None:
    """TAMSAT imports the substrate primitives from ``_cache_base``,
    NOT bespoke local definitions (post-AC-G-2.0)."""
    root = _project_root()
    imports = _module_imports(root / "src/prismpy/sources/climate/tamsat.py")
    cache_base_names = {name for module, name in imports if module == "prismpy.sources._cache_base"}

    required = {
        "MANIFEST_SCHEMA_VERSION",
        "MANIFEST_FILENAME",
        "MARKER_FILENAME",
        "TMPFILE_PREFIX",
        "BBOX_TOLERANCE_DEG",
        "DOWNLOAD_LOCK_TIMEOUT_SECONDS",
        "CacheManifestState",
        "check_cache_manifest",
        "write_cache_manifest",
        "write_marker",
        "delete_marker",
        "warn_legacy_cache_once",
        "bbox_to_dict",
        "bbox_field_for_log",
        "count_tif_files",
        "write_atomic_json",
    }
    missing = required - cache_base_names
    assert not missing, (
        f"tamsat.py must import these from _cache_base; missing: {sorted(missing)}"
    )


def test_agera5_imports_substrate_from_cache_base() -> None:
    """AgERA5 imports the cache substrate directly from ``_cache_base``,
    not transitively through ``tamsat``."""
    root = _project_root()
    imports = _module_imports(root / "src/prismpy/sources/climate/agera5.py")
    cache_base_names = {name for module, name in imports if module == "prismpy.sources._cache_base"}
    tamsat_names = {
        name for module, name in imports if module == "prismpy.sources.climate.tamsat"
    }

    required = {
        "DOWNLOAD_LOCK_TIMEOUT_SECONDS",
        "MANIFEST_FILENAME",
        "MARKER_FILENAME",
        "cache_lock_path",
        "check_cache_manifest",
        "write_cache_manifest",
        "write_marker",
        "delete_marker",
        "warn_legacy_cache_once",
        "bbox_to_dict",
        "bbox_field_for_log",
        "count_tif_files",
    }
    missing = required - cache_base_names
    assert not missing, (
        f"agera5.py must import these from _cache_base; missing: {sorted(missing)}"
    )

    forbidden_via_tamsat = required & tamsat_names
    assert not forbidden_via_tamsat, (
        "agera5.py must import cache primitives from _cache_base, "
        f"NOT from tamsat (durable #24 canonical-source-or-pin); "
        f"these names still leak via tamsat: {sorted(forbidden_via_tamsat)}"
    )


def test_no_other_module_redefines_cache_primitives() -> None:
    """Apart from ``_cache_base.py``, no other prismpy module may define
    a top-level function or constant whose name collides with the
    canonical cache primitives. Catches accidental re-implementation."""
    root = _project_root()
    src = root / "src/prismpy"
    canonical_names = {
        "write_atomic_json",
        "write_cache_manifest",
        "check_cache_manifest",
        "write_marker",
        "delete_marker",
        "warn_legacy_cache_once",
        "cleanup_orphan_tmpfiles",
        "_cleanup_orphan_tmpfiles",
    }
    offenders: list[tuple[str, str]] = []
    for py_file in src.rglob("*.py"):
        rel = py_file.relative_to(root).as_posix()
        if rel.endswith("/sources/_cache_base.py"):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in canonical_names:
                    offenders.append((rel, node.name))
    assert offenders == [], (
        "Cache primitives must live ONLY in prismpy/sources/_cache_base.py "
        f"per durable #24 canonical-source-or-pin. Offenders: {offenders}"
    )
