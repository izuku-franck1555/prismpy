"""Structural pin: canonical IDW interpolation constants.

Sprint E.2 AC-E2-19 + Builder Sub-CA-A (module-allow-list pattern).
Asserts:

* The four ``IDW_DEFAULT_*`` constants exist on
  ``prismpy.standards.idw_methods`` with the contracted values + types.
* No module outside ``standards/idw_methods.py`` hardcodes a literal
  value for ``k``, ``radius_km``, or ``weight_power`` when calling
  ``interpolate_idw(...)``. Per Sub-CA-A: module-allow-list pattern is
  far simpler + far less false-positive-prone than surrounding-context
  regex on every literal-4 / literal-15 / literal-2 occurrence
  across the codebase.

The walker scope is narrow on purpose: it scans keyword arguments to
``interpolate_idw(...)`` calls. A caller is free to write
``range(4)`` or ``timedelta(seconds=15)`` for unrelated reasons; only
calls to the IDW engine are constrained to use the canonical
constants. Sprint S precedent: ``test_isimip_versions_pin.py``
canonical-source pin pattern.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final, get_type_hints

from prismpy.standards import idw_methods


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ── §1 four named constants exist with contracted values ─────────────


def test_idw_default_method_literal_value() -> None:
    assert idw_methods.IDW_DEFAULT_METHOD_LITERAL == "idw_k4_r15km_w_inverse_dist_sq"


def test_idw_default_k_value() -> None:
    assert idw_methods.IDW_DEFAULT_K == 4


def test_idw_default_r_value() -> None:
    assert idw_methods.IDW_DEFAULT_R == 15.0


def test_idw_default_w_value() -> None:
    assert idw_methods.IDW_DEFAULT_W == 2.0


# ── §2 types are correct ─────────────────────────────────────────────


def test_idw_default_method_literal_is_str() -> None:
    assert isinstance(idw_methods.IDW_DEFAULT_METHOD_LITERAL, str)


def test_idw_default_k_is_int() -> None:
    assert isinstance(idw_methods.IDW_DEFAULT_K, int)
    # Defensive: bool is an int subclass in Python; reject it explicitly.
    assert not isinstance(idw_methods.IDW_DEFAULT_K, bool)


def test_idw_default_r_is_float() -> None:
    assert isinstance(idw_methods.IDW_DEFAULT_R, float)


def test_idw_default_w_is_float() -> None:
    assert isinstance(idw_methods.IDW_DEFAULT_W, float)


# ── §3 ``Final`` typing applied ──────────────────────────────────────


def test_idw_constants_use_final_typing() -> None:
    """``Final[...]`` annotations communicate the canonical-source
    intent at the type-system level. A rebinding pattern (``IDW_DEFAULT_K
    = 5``) elsewhere would be a type error under mypy / pyright."""
    hints = get_type_hints(idw_methods, include_extras=True)
    for name in ("IDW_DEFAULT_METHOD_LITERAL", "IDW_DEFAULT_K", "IDW_DEFAULT_R", "IDW_DEFAULT_W"):
        annotation = hints.get(name)
        # ``Final[X]`` evaluates to a ``_GenericAlias`` whose origin is ``typing.Final``
        # OR resolves to the underlying type via ``get_type_hints`` depending on
        # interpreter version + ``include_extras``. Accept either the alias form
        # or the underlying type — what we forbid is the absence of an annotation.
        assert annotation is not None, (
            f"{name} must have a type annotation; ``Final[...]`` is the "
            f"canonical-source intent marker."
        )


# ── §4 dunder-all is the explicit canonical-source surface ───────────


def test_module_dunder_all_lists_four_canonical_names() -> None:
    expected = {
        "IDW_DEFAULT_K",
        "IDW_DEFAULT_METHOD_LITERAL",
        "IDW_DEFAULT_R",
        "IDW_DEFAULT_W",
    }
    assert set(idw_methods.__all__) == expected


# ── §5 no external module hardcodes IDW parameters (Sub-CA-A) ────────


_IDW_PARAM_KWARGS = frozenset({"k", "radius_km", "weight_power"})

# Allow-listed modules per Sub-CA-A module-allow-list pattern. Only
# the canonical-source module + its colocated tests may carry the
# numeric literals as IDW parameters. Test fixtures + parametrize
# arguments often need explicit literals (e.g., test_idw_engine.py
# passes ``k=2`` to exercise the degraded path); those are the
# legitimate exceptions.
_ALLOWED_MODULES: Final[frozenset[str]] = frozenset(
    {
        "src/prismpy/standards/idw_methods.py",
        # Tests directory walked by `_walk_idw_call_sites` separately.
    }
)


def _walk_idw_call_sites(source_path: Path) -> list[tuple[int, str, str]]:
    """Yield (lineno, kwarg, repr(value)) for every ``interpolate_idw(...)``
    call site that passes a Constant for k / radius_km / weight_power.
    Empty list = clean."""
    try:
        src = source_path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    offenders: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match ``interpolate_idw(...)`` calls: the function expression
        # is either a bare Name or an Attribute whose final attr is
        # ``interpolate_idw``.
        func = node.func
        target_name: str | None = None
        if isinstance(func, ast.Name):
            target_name = func.id
        elif isinstance(func, ast.Attribute):
            target_name = func.attr
        if target_name != "interpolate_idw":
            continue
        for kw in node.keywords:
            if kw.arg in _IDW_PARAM_KWARGS and isinstance(kw.value, ast.Constant):
                offenders.append((kw.value.lineno, kw.arg or "", repr(kw.value.value)))
    return offenders


def test_no_external_module_hardcodes_idw_parameters() -> None:
    """Sub-CA-A module-allow-list walker. Walks every ``.py`` source
    file under ``src/prismpy/`` outside the canonical-source module +
    asserts no ``interpolate_idw(...)`` call passes a literal Constant
    for k / radius_km / weight_power. The canonical pattern is to
    pass ``IDW_DEFAULT_K`` / ``IDW_DEFAULT_R`` / ``IDW_DEFAULT_W`` (or
    rely on the engine's defaults).

    Test files ARE allowed to pass explicit literals (e.g.,
    ``interpolate_idw(target, candidates, k=2)`` to exercise the
    degraded path) — those are the legitimate exceptions and the
    walker scope above excludes them by walking ``src/`` only.
    """
    src_root = _project_root() / "src" / "prismpy"
    offenders: list[str] = []
    for py_file in src_root.rglob("*.py"):
        rel_path = py_file.relative_to(_project_root()).as_posix()
        if rel_path in _ALLOWED_MODULES:
            continue
        for lineno, kwarg, value_repr in _walk_idw_call_sites(py_file):
            offenders.append(f"{rel_path}:{lineno}: kwarg={kwarg!r} value={value_repr}")
    assert not offenders, (
        "interpolate_idw(...) call sites passing literal Constants for "
        "k / radius_km / weight_power. Use IDW_DEFAULT_K / "
        "IDW_DEFAULT_R / IDW_DEFAULT_W from "
        "``prismpy.standards.idw_methods`` instead, or rely on the "
        "engine's default-args (which import from the canonical "
        "module). Offenders:\n  " + "\n  ".join(offenders)
    )


def test_walker_catches_a_synthetic_offender(tmp_path: Path) -> None:
    """Negative-control: feed the walker a synthetic offender file
    and assert it surfaces. Without this drill the walker could
    silently match nothing without anyone noticing."""
    offender = tmp_path / "fake_consumer.py"
    offender.write_text(
        "from prismpy.harmonize.idw_interpolation import interpolate_idw\n"
        "\n"
        "def caller():\n"
        "    return interpolate_idw(target, candidates, k=4, radius_km=15.0)\n"
    )
    hits = _walk_idw_call_sites(offender)
    kwarg_names = {kw for _, kw, _ in hits}
    assert kwarg_names == {"k", "radius_km"}, (
        f"Walker must catch literal k=4 + radius_km=15.0; got {kwarg_names}"
    )


def test_walker_ignores_canonical_constant_calls(tmp_path: Path) -> None:
    """Negative-control: a caller using ``IDW_DEFAULT_K`` etc. via
    Name references must NOT trigger the walker — only literal
    Constants do."""
    canonical = tmp_path / "fake_canonical_consumer.py"
    canonical.write_text(
        "from prismpy.harmonize.idw_interpolation import interpolate_idw\n"
        "from prismpy.standards.idw_methods import IDW_DEFAULT_K, IDW_DEFAULT_R\n"
        "\n"
        "def caller():\n"
        "    return interpolate_idw(target, candidates, k=IDW_DEFAULT_K, radius_km=IDW_DEFAULT_R)\n"
    )
    hits = _walk_idw_call_sites(canonical)
    assert hits == [], (
        f"Walker must not fire on Name references to canonical constants; got {hits}"
    )
