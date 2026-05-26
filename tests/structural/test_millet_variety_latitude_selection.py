"""Structural pin: latitude-aware millet variety selection (spec §2).

Exercises the REAL ``_resolve_millet_variety_template`` helper across the spec
branches plus the boundary tie-break (strict ``>`` picks landrace) and a
missing-bbox defensive case. Adds an AST guard so a refactor cannot silently
delete the wiring, and pins the YAML pair (byte-identical mirror + exactly two
cultivar-trait deltas vs the landrace).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from prismpy.translators.sarra_py.translator import SarraPyTranslator


def _h() -> SarraPyTranslator:
    # The resolver is a pure function of its arguments; __new__ is enough.
    return SarraPyTranslator.__new__(SarraPyTranslator)


def test_short_cycle_for_mopti_like_lat_mid_above_14() -> None:
    assert _h()._resolve_millet_variety_template(
        "Millet", None, {"lat_min": 14.40, "lat_max": 14.55}
    ) == "millet_sahel_short"


def test_landrace_for_sikasso_like_lat_mid_below_14() -> None:
    assert _h()._resolve_millet_variety_template(
        "Millet", None, {"lat_min": 11.2, "lat_max": 11.5}
    ) == "millet_west_africa"


def test_explicit_user_override_wins_at_any_latitude() -> None:
    # Returned verbatim regardless of latitude — researchers can experiment.
    assert _h()._resolve_millet_variety_template(
        "Millet", "custom_x", {"lat_min": 14.4, "lat_max": 14.55}
    ) == "custom_x"
    assert _h()._resolve_millet_variety_template(
        "Millet", "custom_x", {"lat_min": 11.0, "lat_max": 11.5}
    ) == "custom_x"


def test_boundary_lat_mid_equals_14_picks_landrace_strict_gt() -> None:
    # Strict ``>`` (not ``>=``): a tie defaults to the conservative landrace.
    assert _h()._resolve_millet_variety_template(
        "Millet", None, {"lat_min": 14.0, "lat_max": 14.0}
    ) == "millet_west_africa"


@pytest.mark.parametrize(
    "crop,bbox",
    [
        ("Sorghum", {"lat_min": 14.4, "lat_max": 14.55}),
        ("Maize", {"lat_min": 14.4, "lat_max": 14.55}),
        ("", {"lat_min": 14.4, "lat_max": 14.55}),
        (None, {"lat_min": 14.4, "lat_max": 14.55}),
        ("Millet", None),  # missing bbox -> defensive None, cascade unchanged
    ],
)
def test_returns_none_when_rule_does_not_apply(crop, bbox) -> None:
    assert _h()._resolve_millet_variety_template(crop, None, bbox) is None


def test_generate_variety_yaml_calls_the_resolver() -> None:
    """AST guard: the wiring must remain. Pins spec §3.2 call-site."""
    src = (
        Path(__file__).resolve().parents[2]
        / "src/prismpy/translators/sarra_py/translator.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    target = next(
        (
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_generate_variety_yaml"
        ),
        None,
    )
    assert target is not None, "_generate_variety_yaml not found in translator"
    assert any(
        isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "_resolve_millet_variety_template"
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "self"
        for c in ast.walk(target)
    ), "_generate_variety_yaml must call self._resolve_millet_variety_template(...)"


def test_yaml_pair_is_byte_identical_and_has_only_two_deltas() -> None:
    """Spec §1.0 + §1.2: mirror byte-identical, only SDJBVP+PPsens differ."""
    root = Path(__file__).resolve().parents[2]
    template = root / "templates/sarra_py/variety/millet_sahel_short.yaml"
    mirror = root / "data/sarra_py_varieties/millet_sahel_short.yaml"
    assert template.read_bytes() == mirror.read_bytes(), "mirror diverged"
    short = yaml.safe_load(template.read_text())
    landrace = yaml.safe_load(
        (root / "templates/sarra_py/variety/millet_west_africa.yaml").read_text()
    )
    diffs = {
        k: (landrace.get(k), short.get(k))
        for k in set(landrace) | set(short)
        if landrace.get(k) != short.get(k)
    }
    assert diffs == {
        "SDJBVP": (700.0, 400.0),
        "PPsens": (0.66, 3.0),
    }, f"unexpected param deltas vs landrace: {diffs}"
