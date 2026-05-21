"""Pattern #283 architectural-invariant guard — CRAFT UC4 preserve_raw emit.

CRAFT shares the DSSAT execution path with PYTHIA; the cycle-closure
patch adds a producer-side capability declaration mirroring the ACEA
shape from the previous PR. When a drought-management package
configures CRAFT, the translator must:

- declare ``adapter_version="1.0"`` at the TOP LEVEL of
  ``manifest_extra`` (the same shape the ACEA translator emits),
- declare ``adapter_capability.preserve_raw_supported`` with the 5
  base daily artifacts (CRAFT does not emit AquaCrop-native
  ``et_color`` / ``s_color`` arrays so green/blue water artifacts
  stay out of scope here; PYTHIA already covers the annual surface
  via Summary CWAM / HWAM).

A symmetric runtime probe exercises the CRAFT translator with a
synthetic UC4 package and asserts the captured
``create_manifest`` kwargs match the expected shape (the same fold
pattern used for the ACEA producer guard).
"""

from __future__ import annotations

import ast
import datetime
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CRAFT_TRANSLATOR = (
    _REPO_ROOT
    / "src" / "prismpy" / "translators" / "craft" / "translator.py"
)


_EXPECTED_BASE_DAILY = {
    "daily_eto_etc",
    "daily_ftsw",
    "daily_lai_or_phenology",
    "daily_precipitation",
    "daily_root_zone_moisture",
}


def _str_constants(node: ast.AST) -> set[str]:
    return {
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def _find_uc4_capability_if(tree: ast.Module) -> ast.If:
    """Locate the CRAFT translator's ``if 'drought_management' in ...:``
    gate whose body assigns ``manifest_extra['adapter_capability']``.

    Mirrors the ACEA AST guard: bind to the body content (not just the
    test expression) so unrelated ``'drought_management' in ...`` sites
    elsewhere in the file don't shadow the producer-emit gate.
    """
    for if_node in ast.walk(tree):
        if not isinstance(if_node, ast.If):
            continue
        if "drought_management" not in _str_constants(if_node.test):
            continue
        body_constants: set[str] = set()
        for child in if_node.body:
            body_constants |= _str_constants(child)
        if (
            "adapter_capability" in body_constants
            and "adapter_version" in body_constants
            and "preserve_raw_supported" in body_constants
        ):
            return if_node
    pytest.fail(
        "CRAFT translator must contain an `if 'drought_management' in ...:` "
        "gate whose body assigns adapter_capability + adapter_version"
    )


def _extract_subscript_assignments(
    if_node: ast.If, container_name: str,
) -> dict[str, ast.AST]:
    """Return ``{subscript_key: value_node}`` for the
    ``<container>[<str>] = <value>`` assignments inside ``if_node``."""
    out: dict[str, ast.AST] = {}
    for node in ast.walk(if_node):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Subscript):
            continue
        if not (isinstance(tgt.value, ast.Name) and tgt.value.id == container_name):
            continue
        key = tgt.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            out[key.value] = node.value
    return out


def test_craft_uc4_capability_gate_carries_5_daily_and_top_level_version(
) -> None:
    """Syntactic check: the CRAFT translator's UC4 gate uses the ``In``
    operator on ``'drought_management'`` and assigns
    ``manifest_extra['adapter_version'] = '1.0'`` at the top level
    alongside ``manifest_extra['adapter_capability']`` (a dict that
    carries the 5-base-daily list, no nested adapter_version).
    """
    tree = ast.parse(
        _CRAFT_TRANSLATOR.read_text(encoding="utf-8"),
        filename=str(_CRAFT_TRANSLATOR),
    )
    if_node = _find_uc4_capability_if(tree)

    # Gate uses ``In`` (not ``NotIn``)
    found_in = False
    for cmp in ast.walk(if_node.test):
        if isinstance(cmp, ast.Compare):
            for op in cmp.ops:
                if isinstance(op, ast.In):
                    found_in = True
                if isinstance(op, ast.NotIn):
                    pytest.fail(
                        "CRAFT UC4 capability gate must use the In operator "
                        "on 'drought_management'; found NotIn (inverted)."
                    )
    assert found_in

    assigns = _extract_subscript_assignments(if_node, "manifest_extra")
    assert "adapter_version" in assigns, (
        "manifest_extra must carry adapter_version at the top level "
        "(not nested under adapter_capability)."
    )
    av_node = assigns["adapter_version"]
    assert (
        isinstance(av_node, ast.Constant) and av_node.value == "1.0"
    ), (
        f"manifest_extra['adapter_version'] must be the literal '1.0'; "
        f"got {ast.dump(av_node)!r}"
    )
    assert "adapter_capability" in assigns
    cap_node = assigns["adapter_capability"]
    assert isinstance(cap_node, ast.Dict)
    cap_keys = {
        k.value for k in cap_node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert "adapter_version" not in cap_keys, (
        "CRAFT adapter_version must NOT be nested under "
        "adapter_capability; it must live at the top level."
    )

    preserve_raw_node = None
    for k, v in zip(cap_node.keys, cap_node.values):
        if (
            isinstance(k, ast.Constant)
            and k.value == "preserve_raw_supported"
        ):
            preserve_raw_node = v
            break
    assert isinstance(preserve_raw_node, ast.List)
    declared = {
        elt.value
        for elt in preserve_raw_node.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    }
    assert declared == _EXPECTED_BASE_DAILY, (
        f"CRAFT preserve_raw_supported must be EXACTLY the 5 base "
        f"daily artifacts (no annual / green-blue surface); got "
        f"{sorted(declared)}"
    )


def test_craft_translator_runtime_emits_adapter_capability_on_uc4() -> None:
    """Runtime probe: invoke the CRAFT translator on a synthetic UC4
    package config and capture the kwargs passed to
    ``create_manifest``. Asserts the top-level ``adapter_version`` +
    nested ``adapter_capability.preserve_raw_supported`` shape with the
    5 base daily artifacts.
    """
    from prismpy.translators.craft.translator import CraftTranslator

    config = SimpleNamespace(
        project=SimpleNamespace(name="craft-runtime-probe"),
        crop=SimpleNamespace(
            name="sorghum",
            calendar=SimpleNamespace(
                planting_doy=175, maturity_doy=305,
            ),
        ),
        temporal=SimpleNamespace(
            start_year=2018,
            end_year=2022,
            spinup_years=0,
            get_climate_end_date=lambda calendar: datetime.date(2022, 12, 31),
        ),
        region=SimpleNamespace(
            boundary=SimpleNamespace(
                source=SimpleNamespace(value="gadm"),
                gadm_level=2,
            ),
        ),
    )

    captured: dict[str, object] = {}

    def fake_create_manifest(*args, **kwargs):
        captured.update(kwargs)
        captured["_positional"] = list(args)
        return {"_captured": True, "uc_readiness": {}}

    with tempfile.TemporaryDirectory() as tmp:
        translator = CraftTranslator.__new__(CraftTranslator)
        translator.config = config
        translator.output_dir = Path(tmp)
        # CRAFT imports create_manifest / save_manifest / generate_readme
        # at MODULE TOP (translator.py:56-57) — patch the local bindings
        # in the translator module so the captured kwargs go through
        # ``fake_create_manifest`` rather than the real persisted-write
        # path.
        with patch(
            "prismpy.translators.craft.translator.create_manifest",
            side_effect=fake_create_manifest,
        ), patch(
            "prismpy.translators.craft.translator.save_manifest",
            return_value=Path(tmp) / "manifest.json",
        ), patch(
            "prismpy.translators.craft.translator.generate_readme",
            return_value=Path(tmp) / "README.md",
        ), patch.object(
            CraftTranslator, "get_platform_config", return_value=None,
        ):
            translator._generate_package_metadata(
                data=SimpleNamespace(
                    region=SimpleNamespace(
                        name="X", country="X", boundary_source="gadm",
                    ),
                    grid=SimpleNamespace(cells=[]),
                    soil={},
                ),
                output_files=[],
            )

    additional = captured.get("additional_metadata") or {}
    assert isinstance(additional, dict), (
        f"CRAFT translator must pass additional_metadata as a dict; "
        f"got {type(additional).__name__}"
    )
    assert additional.get("adapter_version") == "1.0", (
        f"adapter_version must be '1.0' at the TOP LEVEL of "
        f"additional_metadata; got {additional.get('adapter_version')!r}"
    )
    cap = additional.get("adapter_capability") or {}
    assert isinstance(cap, dict)
    assert "adapter_version" not in cap, (
        "CRAFT adapter_version must NOT be nested under "
        "adapter_capability."
    )
    declared = set(cap.get("preserve_raw_supported") or [])
    assert declared == _EXPECTED_BASE_DAILY, (
        f"CRAFT translator-emitted preserve_raw_supported must be the "
        f"5 base daily artifacts; got {sorted(declared)}"
    )


def test_craft_translator_omits_capability_when_uc4_absent() -> None:
    """When the package does NOT configure drought_management, the
    CRAFT translator must NOT carry adapter_capability —
    only the soil-fertility joint flag (kept verbatim)."""
    from prismpy.packaging.manifest import create_manifest
    import json

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg"
        pkg.mkdir()
        (pkg / "metadata.json").write_text(json.dumps({}))
        project_config = {
            "project_name": "craft-test",
            "region_name": "X",
            "country": "X",
            "gadm_level": 2,
            "crop_name": "sorghum",
            "planting_doy": 175,
            "maturity_doy": 305,
            "start_year": 2018,
            "end_year": 2022,
            "spinup_years": 0,
            "data_sources": {},
            "use_case_config": {"yield_forecast": {}},
        }
        manifest = create_manifest(
            pkg, project_config, platform="craft",
            additional_metadata=None,
        )
    assert "adapter_version" not in manifest
    assert "adapter_capability" not in manifest
