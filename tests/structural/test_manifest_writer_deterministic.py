"""Structural pin: package manifest writer produces deterministic bytes.

Sprint G CC-G-7 requires the package's ``manifest.json`` to be
byte-identical across re-runs on identical inputs and identical pinned
prismpy code. The Sprint G AC-G-13 deliverable hash pin (4 paired sets
× SHA-256 of each ``manifest.json``) depends on this invariant.

Forbidden in manifest content:
* Wall-clock stamps (``generated_at``, file ``modified`` mtime).
* CRLF newlines, BOM, ``\\uXXXX`` escapes for legitimate Unicode.
* Non-canonical key ordering (sort_keys must be applied).
* Random-seed-dependent values, build-host-specific paths, build-time
  stamps. (These rely on the source-data inputs not having those
  values; the writer cannot fix bad input but must not introduce
  any of its own.)

The pins below exercise ``create_manifest`` + ``save_manifest`` on a
synthetic package fixture and check the byte output meets every
invariant. A mutation that re-introduces ``datetime.now()`` into
either function fires this pin.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from prismpy.packaging.manifest import (
    create_manifest,
    get_file_info,
    save_manifest,
)


# ── §1 Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def synthetic_package(tmp_path: Path) -> Path:
    """Build a tiny on-disk package: a few files in a fresh dir.

    Used by the determinism tests so the manifest writer reads stable
    content.
    """
    pkg = tmp_path / "synthetic_package"
    pkg.mkdir()

    (pkg / "data").mkdir()
    (pkg / "data" / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    (pkg / "data" / "beta.txt").write_text("beta\n", encoding="utf-8")
    # Include a file with a non-ASCII path component to exercise the
    # ``ensure_ascii=False`` invariant on round-trip.
    (pkg / "data" / "ménoua.txt").write_text(
        "regional sample\n", encoding="utf-8"
    )

    return pkg


@pytest.fixture
def synthetic_config() -> Dict[str, Any]:
    """Minimal project config that ``create_manifest`` understands."""
    return {
        "project_name": "synthetic",
        "region_name": "Ménoua",
        "country": "Cameroon",
        "gadm_level": 1,
        "crop_name": "millet",
        "planting_doy": 152,
        "maturity_doy": 273,
        "start_year": 2020,
        "end_year": 2022,
        "spinup_years": 0,
        "data_sources": {"climate": "AgERA5", "boundaries": "GADM v4.1 admin level 1"},
    }


# ── §2 No wall-clock stamps in manifest content ──────────────────────


def test_manifest_has_no_generated_at_field(
    synthetic_package: Path, synthetic_config: Dict[str, Any]
) -> None:
    """``create_manifest`` MUST NOT emit a ``generated_at`` field.

    A wall-clock stamp at the top level breaks byte-identical
    regeneration: every run differs by ``datetime.now()``.
    """
    manifest = create_manifest(synthetic_package, synthetic_config)
    assert "generated_at" not in manifest, (
        "Manifest content must not include a wall-clock 'generated_at' field "
        "(CC-G-7 deterministic generation invariant)."
    )


def test_get_file_info_has_no_modified_field(synthetic_package: Path) -> None:
    """``get_file_info`` MUST NOT emit the filesystem mtime.

    Filesystem mtime is a filesystem concern, distinct from manifest
    content. Including it in the manifest breaks byte-identical
    regeneration (mtime advances every write).
    """
    target = synthetic_package / "data" / "alpha.txt"
    info = get_file_info(target, base_path=synthetic_package)
    assert "modified" not in info, (
        "File info must not include filesystem mtime in manifest content "
        "(CC-G-7 content-vs-filesystem separation)."
    )
    # The remaining identity fields are still required: path, sha256, size_bytes.
    assert {"path", "sha256", "size_bytes"} <= info.keys()


def test_no_file_entry_has_modified_field(
    synthetic_package: Path, synthetic_config: Dict[str, Any]
) -> None:
    """No file entry inside ``manifest.files[]`` carries an mtime."""
    manifest = create_manifest(synthetic_package, synthetic_config)
    offenders = [
        entry["path"] for entry in manifest["files"] if "modified" in entry
    ]
    assert offenders == [], (
        f"Manifest file entries must not carry mtime: {offenders}"
    )


# ── §3 Byte-identical regeneration ───────────────────────────────────


def test_save_manifest_is_byte_identical_across_two_runs(
    synthetic_package: Path,
    synthetic_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """Saving the same manifest twice produces byte-identical files.

    This is the load-bearing pin behind AC-G-13's deliverable hash
    pinning. A mutation that re-introduces ``datetime.now()`` or
    drops ``sort_keys=True`` fires this assertion.
    """
    manifest = create_manifest(synthetic_package, synthetic_config)

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    save_manifest(manifest, out_a)
    save_manifest(manifest, out_b)

    sha_a = hashlib.sha256(out_a.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(out_b.read_bytes()).hexdigest()
    assert sha_a == sha_b, (
        f"Manifest writer is not deterministic: {sha_a} vs {sha_b}"
    )


def test_create_manifest_is_byte_identical_across_two_runs(
    synthetic_package: Path,
    synthetic_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """Generating the manifest dict + saving twice round-trips identically.

    Catches regressions in either ``create_manifest`` (top-level
    fields, file inventory) or ``save_manifest`` (serialization).
    """
    manifest_a = create_manifest(synthetic_package, synthetic_config)
    manifest_b = create_manifest(synthetic_package, synthetic_config)

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    save_manifest(manifest_a, out_a)
    save_manifest(manifest_b, out_b)

    assert out_a.read_bytes() == out_b.read_bytes()


# ── §4 Canonical serialization invariants ────────────────────────────


def test_save_manifest_uses_sort_keys(
    synthetic_package: Path,
    synthetic_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """Top-level keys appear in lexicographic order in the file."""
    manifest = create_manifest(synthetic_package, synthetic_config)
    out = tmp_path / "manifest.json"
    save_manifest(manifest, out)

    parsed = json.loads(out.read_text(encoding="utf-8"))
    keys = list(parsed.keys())
    assert keys == sorted(keys), (
        f"Top-level keys not sorted: {keys}"
    )


def test_save_manifest_round_trips_unicode_unescaped(
    synthetic_package: Path,
    synthetic_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """Non-ASCII characters appear as UTF-8, NOT as ``\\uXXXX`` escapes.

    Region names like 'Ménoua' must appear unescaped. ``ensure_ascii=False``
    + UTF-8 encoding ensures the bytes are stable across Python versions.
    """
    manifest = create_manifest(synthetic_package, synthetic_config)
    out = tmp_path / "manifest.json"
    save_manifest(manifest, out)

    raw = out.read_bytes()
    assert "é".encode("utf-8") in raw, (
        "Unicode region name 'Ménoua' should appear as UTF-8 bytes, "
        "not as Unicode escape sequences."
    )
    assert b"\\u00e9" not in raw, (
        "Manifest must not contain \\uXXXX escapes for legitimate Unicode."
    )


def test_save_manifest_uses_lf_newlines(
    synthetic_package: Path,
    synthetic_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """Output uses LF newlines only, no CRLF.

    Even on platforms whose text mode auto-translates ``\\n`` to
    ``\\r\\n``, the writer's binary path keeps newlines as LF so the
    SHA-256 hash is stable across operating systems.
    """
    manifest = create_manifest(synthetic_package, synthetic_config)
    out = tmp_path / "manifest.json"
    save_manifest(manifest, out)

    raw = out.read_bytes()
    assert b"\r\n" not in raw, "Manifest must use LF newlines, not CRLF."
    assert b"\n" in raw, "Manifest must contain at least one LF newline."


def test_save_manifest_no_byte_order_mark(
    synthetic_package: Path,
    synthetic_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """Output starts with ``{`` (or ``[``) — no UTF-8 BOM."""
    manifest = create_manifest(synthetic_package, synthetic_config)
    out = tmp_path / "manifest.json"
    save_manifest(manifest, out)

    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), (
        "Manifest must not start with a UTF-8 BOM."
    )
    assert raw[:1] in (b"{", b"["), (
        f"Manifest should start with a JSON structural character; got {raw[:1]!r}"
    )


def test_save_manifest_round_trips_through_json_load(
    synthetic_package: Path,
    synthetic_config: Dict[str, Any],
    tmp_path: Path,
) -> None:
    """The bytes parse back into the same dict the writer was given."""
    manifest = create_manifest(synthetic_package, synthetic_config)
    out = tmp_path / "manifest.json"
    save_manifest(manifest, out)

    reparsed = json.loads(out.read_text(encoding="utf-8"))
    # Compare against the same manifest dict the writer received.
    # ``json.loads`` does not preserve insertion order on 3.7+ but
    # equality is content-based.
    assert reparsed == manifest


# ── §5 No datetime imports leak into manifest content (regression) ───


def test_manifest_module_does_not_emit_datetime_in_content() -> None:
    """The module's serialization path must not use ``datetime.now()``.

    AST walk over ``packaging.manifest``: any module-level call to
    ``datetime.now()`` outside ``validate_manifest`` (which produces
    a runtime validation REPORT, not manifest CONTENT) is a CC-G-7
    regression.
    """
    import ast

    source = (
        Path(__file__).resolve().parents[2]
        / "src/prismpy/packaging/manifest.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    offenders: list[tuple[str, int]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._enclosing_func: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._enclosing_func.append(node.name)
            self.generic_visit(node)
            self._enclosing_func.pop()

        def visit_Call(self, node: ast.Call) -> None:
            # Match datetime.now(...) and datetime.fromtimestamp(...)
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ) and node.func.value.id == "datetime":
                fn = node.func.attr
                if fn in {"now", "fromtimestamp"}:
                    enclosing = (
                        self._enclosing_func[-1]
                        if self._enclosing_func
                        else "<module>"
                    )
                    if enclosing not in {"validate_manifest"}:
                        offenders.append((enclosing, node.lineno))
            self.generic_visit(node)

    _Visitor().visit(tree)
    assert offenders == [], (
        "Manifest content writer must not call datetime.now() / "
        "datetime.fromtimestamp() (CC-G-7 deterministic generation). "
        f"Offenders: {offenders}"
    )
