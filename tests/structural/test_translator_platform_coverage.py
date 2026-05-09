"""Structural pin: 4-translator override coverage.

Sprint E.3 AC-E3-9 sub-4 + AC-E3-12 #6 + builder grounding-pass
CA-5 absorbed. Three invariants close the silent-coverage-drift
class per durable §24 canonical-source-or-pin + durable §27 two-
vocabulary substrate-drift:

§1 Coverage — every ``Platform.*`` enum member NOT in
:data:`_PLATFORM_OVERRIDE_EXEMPTIONS` has a translator module that
imports ``apply_override`` from
``prismpy.translators._shared.cockpit_overrides``. A new
``Platform.*`` addition without override coverage (and without
explicit exemption) fires this pin loud — that's the
production-blocking class CMS CA-1 closed for ACEA radius and
this pin generalises to per-platform override wiring coverage.

§2 Exemption registry sanity — the registry contains exactly
``{"DSSAT"}`` per AC-E3-9 contract text (DSSAT is the family-
base used by PYTHIA + CRAFT internally, not a separate platform).
A future kernel-family extension that adds a new exempt platform
extends the registry intentionally.

§3 Translator-class registry consistency — the canonical
``_get_translator`` map at ``executor.py:292`` covers every
non-exempt ``Platform.*`` member. The pin reads the executor's
source via AST so a refactor that drops a translator binding
fires the pin loud.
"""

from __future__ import annotations

import ast
from pathlib import Path

from prismpy.config.schema import Platform
from prismpy.translators._shared.cockpit_overrides import (
    _PLATFORM_OVERRIDE_EXEMPTIONS,
)


# ── Helpers ────────────────────────────────────────────────────────


def _prismpy_src_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent.parent / "src" / "prismpy"


_TRANSLATOR_DIR_BY_PLATFORM = {
    Platform.SARRA_PY: "sarra_py",
    Platform.CRAFT: "craft",
    Platform.PYTHIA: "pythia",
    Platform.ACEA: "acea",
}


def _translator_imports_apply_override(platform: Platform) -> bool:
    """True iff the translator module for ``platform`` imports
    ``apply_override`` from
    ``prismpy.translators._shared.cockpit_overrides``."""
    src_root = _prismpy_src_root()
    sub_dir = _TRANSLATOR_DIR_BY_PLATFORM.get(platform)
    if sub_dir is None:
        return False
    translator_path = src_root / "translators" / sub_dir / "translator.py"
    if not translator_path.exists():
        return False
    text = translator_path.read_text()
    # AST-walk for the import; substring match is robust enough
    # since the import is a deterministic line-level pattern.
    return (
        "from prismpy.translators._shared.cockpit_overrides import apply_override"
        in text
    )


# ── §1 every non-exempt Platform.* has translator coverage ─────────


def test_every_non_exempt_platform_translator_imports_apply_override() -> None:
    """The structural-coverage pin: a new ``Platform.*`` member
    without override coverage (and without explicit exemption)
    fires this pin loud. Per AC-E3-9 sub-4 + builder CA-5 + CMS
    CA-1 generalised."""
    missing: list[str] = []
    for platform in Platform:
        if platform.value in _PLATFORM_OVERRIDE_EXEMPTIONS:
            continue
        if not _translator_imports_apply_override(platform):
            missing.append(platform.value)
    assert not missing, (
        f"Platform.* members missing apply_override import in their "
        f"translator: {sorted(missing)}. Per AC-E3-9 sub-4: every "
        f"non-exempt platform translator MUST import "
        f"``apply_override`` from "
        f"``prismpy.translators._shared.cockpit_overrides``. Add "
        f"the import OR add the platform value to "
        f"``_PLATFORM_OVERRIDE_EXEMPTIONS`` with a rationale "
        f"comment."
    )


def test_every_non_exempt_platform_in_translator_dir_map() -> None:
    """The pin's helper map covers every non-exempt ``Platform.*``
    member. A new platform addition that lacks a directory
    binding here would silently skip the coverage check; pin the
    map's coverage too."""
    missing = [
        p.value
        for p in Platform
        if p.value not in _PLATFORM_OVERRIDE_EXEMPTIONS
        and p not in _TRANSLATOR_DIR_BY_PLATFORM
    ]
    assert not missing, (
        f"Translator dir map missing entries for {sorted(missing)}. "
        f"Update ``_TRANSLATOR_DIR_BY_PLATFORM`` in this pin file."
    )


# ── §2 exemption registry sanity ───────────────────────────────────


def test_exemption_registry_contains_dssat_only() -> None:
    """Sprint E.3 v1 ships exactly DSSAT in the exemption set per
    AC-E3-9 contract text. A future kernel-family extension that
    adds a new exempt platform extends this assertion
    intentionally."""
    assert _PLATFORM_OVERRIDE_EXEMPTIONS == frozenset({"DSSAT"}), (
        f"Override-coverage exemption registry drifted: "
        f"{sorted(_PLATFORM_OVERRIDE_EXEMPTIONS)}. Sprint E.3 "
        f"contract scope is {{'DSSAT'}} only."
    )


def test_exemption_set_is_frozen() -> None:
    """Pin the immutability of the exemption set so a runtime
    mutation can't leak past the canonical-source contract."""
    assert isinstance(_PLATFORM_OVERRIDE_EXEMPTIONS, frozenset)


# ── §3 executor _get_translator map covers non-exempt platforms ────


def test_executor_translator_map_covers_non_exempt_platforms() -> None:
    """The ``_get_translator`` map at ``executor.py:292`` MUST
    cover every non-exempt ``Platform.*`` member. AST-walk the
    executor source to find the dict literal + assert coverage.

    The pin is scoped to Platform-keyed dict literals inside
    ``ExecutorClass._get_translator`` per AC-E3-9 sub-4
    canonical-derivation rule (NOT a hard-coded list)."""
    src_root = _prismpy_src_root()
    executor_path = src_root / "pipeline" / "executor.py"
    tree = ast.parse(executor_path.read_text())

    found_platform_values: set[str] = set()
    # Walk for dict literals whose keys are Platform.<NAME>
    # attributes within the _get_translator function.
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef)
            and node.name == "_get_translator"
        ):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Dict):
                for key_node in inner.keys:
                    if (
                        isinstance(key_node, ast.Attribute)
                        and isinstance(key_node.value, ast.Name)
                        and key_node.value.id == "Platform"
                    ):
                        found_platform_values.add(key_node.attr)
        break

    assert found_platform_values, (
        "Executor._get_translator map not located via AST walk. "
        "Has the executor refactored away from a Platform-keyed "
        "dict literal? Update this pin to follow."
    )

    # Compare against non-exempt Platform.* enum members. We map
    # the AST attribute names (``SARRA_PY``) to enum values
    # (``"sarra_py"``) via Platform.<NAME>.value.
    enum_names_to_values = {p.name: p.value for p in Platform}
    expected_names = {
        name
        for name, value in enum_names_to_values.items()
        if value not in _PLATFORM_OVERRIDE_EXEMPTIONS
    }

    missing_in_map = expected_names - found_platform_values
    assert not missing_in_map, (
        f"Executor._get_translator map missing entries for "
        f"{sorted(missing_in_map)}. Per AC-E3-9 sub-4 canonical-"
        f"derivation rule: the map covers every non-exempt "
        f"Platform.* member."
    )
