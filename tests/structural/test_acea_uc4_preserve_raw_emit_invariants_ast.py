"""Pattern #283 architectural-invariant guard — ACEA UC4 preserve_raw emit.

The ACEA translator declares the UC4 preserve_raw capability via
``additional_metadata`` blind-merge to ``create_manifest``. The
declaration must reach the persisted on-disk manifest as a top-level
``adapter_version`` plus a nested ``adapter_capability.preserve_raw_
supported`` list with the 13-artifact union: 5 base daily artifacts
(shared with PYTHIA / SARRA-Py), 2 annual artifacts (engine-consistency
with PYTHIA's 7-key map), and 6 green/blue/conditional-rainfall
water-partition scalars sourced from AquaCrop ``et_color`` / ``s_color``
arrays. Two guards:

- **A7-syntactic**: AST-walks ``translators/acea/translator.py`` and
  asserts that the ``manifest_extra`` build site assigns the EXACT
  13-key ``preserve_raw_supported`` list AND ``adapter_version =
  '1.0'`` AND the assignment is gated on ``drought_management`` in
  ``use_case_config``. Catches drift to a wrong artifact list or a
  missing version.
- **A7-runtime**: invokes ``create_manifest`` with the synthesised
  ``additional_metadata`` shape the translator builds and asserts
  the persisted manifest dict carries both keys with the expected
  shape after the blind-merge + ``_filter_additional_metadata``
  sanitisation.
"""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import pytest

from prismpy.packaging.manifest import create_manifest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ACEA_TRANSLATOR = (
    _REPO_ROOT
    / "src" / "prismpy" / "translators" / "acea" / "translator.py"
)


# F-BP-5 cycle closure: ACEA preserve_raw widens 5 → 13 artifacts.
# Base 5 daily artifacts + 2 annual (engine-consistency with PYTHIA) +
# 6 green/blue water-partition scalars (AquaCrop ``et_color`` /
# ``s_color`` arrays). Each subset stays addressable by name so a
# regression that drops one tier surfaces a specific failure.
_EXPECTED_BASE_DAILY = {
    "daily_eto_etc",
    "daily_ftsw",
    "daily_lai_or_phenology",
    "daily_precipitation",
    "daily_root_zone_moisture",
}
_EXPECTED_ANNUAL = {
    "annual_total_biomass",
    "annual_grain_yield",
}
_EXPECTED_GREEN_BLUE = {
    "daily_et_green",
    "daily_et_blue",
    "daily_et_cr",
    "daily_storage_green",
    "daily_storage_blue",
    "daily_storage_cr",
}
_EXPECTED_DAILY = _EXPECTED_BASE_DAILY | _EXPECTED_ANNUAL | _EXPECTED_GREEN_BLUE


def _str_constants(node: ast.AST) -> set[str]:
    return {
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def _find_uc4_capability_if(tree: ast.Module) -> ast.If:
    """Locate the translator's ``if 'drought_management' (not) in
    use_case_config`` gate inside ``_build_uc4_capability_extra``.

    Post the cycle-closure refactor the gate now lives inside the
    helper method as an early-return on the no-UC4 branch; the
    positive branch falls through to a ``return {dict-literal}`` that
    carries ``adapter_version`` and ``adapter_capability``. This
    finder returns the gate If so the existing operator / dict
    assertions still apply (the gate uses ``NotIn`` for the early-
    return — codified by the operator check below)."""
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        if func.name != "_build_uc4_capability_extra":
            continue
        for if_node in ast.walk(func):
            if not isinstance(if_node, ast.If):
                continue
            if "drought_management" not in _str_constants(if_node.test):
                continue
            return if_node
    pytest.fail(
        "ACEA translator must contain a ``_build_uc4_capability_extra`` "
        "helper with a ``drought_management`` gate."
    )


def _find_uc4_positive_return_dict(tree: ast.Module) -> ast.Dict:
    """Locate the positive-branch ``return {dict-literal}`` inside
    ``_build_uc4_capability_extra`` — the dict that carries
    ``adapter_version`` + ``adapter_capability`` + ``preserve_raw_
    supported``."""
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        if func.name != "_build_uc4_capability_extra":
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Return):
                continue
            val = node.value
            if not isinstance(val, ast.Dict):
                continue
            keys = {
                k.value for k in val.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            if "adapter_version" in keys and "adapter_capability" in keys:
                return val
    pytest.fail(
        "ACEA ``_build_uc4_capability_extra`` must return a dict literal "
        "with adapter_version + adapter_capability on the UC4-positive "
        "branch."
    )


def _extract_subscript_assignments(
    if_node: ast.If, container_name: str,
) -> dict[str, ast.AST]:
    """Return ``{subscript_key: value_node}`` for assignments of the form
    ``<container_name>[<str>] = <value>`` inside the if body.

    ``manifest_extra['adapter_version'] = '1.0'`` → ``{"adapter_version":
    Constant('1.0'), ...}``. Each key maps to the RHS AST node so the
    caller can inspect the value's shape (Constant / Dict / List).
    """
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


def test_a7_syntactic_exact_13_artifact_union_with_top_level_adapter_version() -> None:
    """Cycle-closure fold of the original 5-daily check — now binds to
    the 13-artifact union (5 base daily + 2 annual + 6 green/blue
    scalars) declared by the ACEA translator's
    ``_build_uc4_capability_extra`` helper post the cycle-closure
    refactor.

    Invariants:

    - The ``preserve_raw_supported`` list is the EXACT 13-key union
      (dropping any tier — base daily, annual, or green/blue — fails).
    - ``adapter_version`` is a TOP-LEVEL key of the returned dict
      (nesting it under ``adapter_capability`` fails).
    - The gate's early-return uses the ``NotIn`` operator (so the
      positive branch — UC4 in use_case_config — carries the
      capability surface).
    """
    tree = ast.parse(
        _ACEA_TRANSLATOR.read_text(encoding="utf-8"),
        filename=str(_ACEA_TRANSLATOR),
    )
    if_node = _find_uc4_capability_if(tree)

    # (1) Gate uses NotIn (early-return on no-UC4 branch).
    found_not_in = False
    for cmp in ast.walk(if_node.test):
        if isinstance(cmp, ast.Compare):
            for op in cmp.ops:
                if isinstance(op, ast.NotIn):
                    found_not_in = True
    assert found_not_in, (
        "UC4 capability gate inside ``_build_uc4_capability_extra`` "
        "must use the NotIn operator on 'drought_management' to "
        "early-return an empty dict on the no-UC4 branch."
    )

    # (2) Positive-branch return-dict carries adapter_version + cap.
    return_dict = _find_uc4_positive_return_dict(tree)
    cap_node = None
    av_node = None
    for k, v in zip(return_dict.keys, return_dict.values):
        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
            continue
        if k.value == "adapter_version":
            av_node = v
        elif k.value == "adapter_capability":
            cap_node = v
    assert av_node is not None and isinstance(av_node, ast.Constant), (
        "Helper's returned dict must carry adapter_version at the TOP "
        "LEVEL (not nested under adapter_capability)."
    )
    assert av_node.value == "1.0", (
        f"adapter_version must be the literal '1.0'; got {av_node.value!r}"
    )
    assert isinstance(cap_node, ast.Dict)
    cap_keys = {
        k.value for k in cap_node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert "adapter_version" not in cap_keys, (
        "adapter_version must NOT be nested under adapter_capability; "
        "it must live at the TOP LEVEL of the returned dict."
    )

    # (3) The preserve_raw_supported value is an EXACT 13-key List
    # (5 base daily + 2 annual + 6 green/blue scalars).
    preserve_raw_node = None
    for k, v in zip(cap_node.keys, cap_node.values):
        if (
            isinstance(k, ast.Constant)
            and k.value == "preserve_raw_supported"
        ):
            preserve_raw_node = v
            break
    assert isinstance(preserve_raw_node, ast.List), (
        "adapter_capability['preserve_raw_supported'] must be a list "
        "literal."
    )
    declared = {
        elt.value
        for elt in preserve_raw_node.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    }
    assert declared == _EXPECTED_DAILY, (
        f"preserve_raw_supported must be EXACTLY the 13-artifact union "
        f"(5 base daily + 2 annual + 6 green/blue scalars); got "
        f"{sorted(declared)}. Dropping a tier (e.g., omitting the 6 "
        f"green/blue scalars) or adding an unrelated key fails this guard."
    )


def test_a7_runtime_invokes_translator_and_captures_adapter_capability() -> None:
    """SH-2 codex+eval-2 fold — invoke the actual translator method on a
    synthetic UC4 ``package_config`` and capture the kwargs the
    translator passes to ``create_manifest``. Counterfactual-verifiable:
    if the translator emits a wrong shape (e.g., adapter_version nested
    under adapter_capability), the captured kwargs fail the asserts.
    """
    import datetime
    from types import SimpleNamespace
    from unittest.mock import patch

    from prismpy.translators.acea.translator import AceaTranslator

    # Minimal config: only the attributes _generate_package_metadata
    # actually reads (verified by reading the method body).
    config = SimpleNamespace(
        project=SimpleNamespace(name="acea-runtime-probe"),
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
    data = SimpleNamespace(
        region=SimpleNamespace(
            name="X", country="X", boundary_source="gadm",
        ),
        soil={},
    )

    captured: dict[str, object] = {}

    def fake_create_manifest(*args, **kwargs):
        captured.update(kwargs)
        captured["_positional"] = list(args)
        # Return a sentinel manifest dict so save_manifest succeeds.
        return {"_captured": True, "uc_readiness": {}}

    with tempfile.TemporaryDirectory() as tmp:
        translator = AceaTranslator.__new__(AceaTranslator)
        translator.config = config
        translator.output_dir = Path(tmp)
        translator._uc5_pythia_pk_silent_no_op_triggered = False

        # Monkeypatch the file-side helpers to keep this test
        # filesystem-light; the focus is the create_manifest kwargs.
        with patch(
            "prismpy.packaging.manifest.create_manifest",
            side_effect=fake_create_manifest,
        ), patch(
            "prismpy.packaging.manifest.save_manifest",
            return_value=Path(tmp) / "manifest.json",
        ), patch(
            "prismpy.packaging.readme_generator.generate_readme",
            return_value=Path(tmp) / "README.md",
        ), patch.object(
            AceaTranslator, "get_platform_config", return_value=None,
        ):
            translator._generate_package_metadata(
                data, [], "acea_test", [],
            )

    additional = captured.get("additional_metadata") or {}
    assert isinstance(additional, dict), (
        f"translator must pass additional_metadata as a dict; got "
        f"{type(additional).__name__}"
    )
    assert additional.get("adapter_version") == "1.0", (
        f"adapter_version must be the literal '1.0' at the TOP LEVEL of "
        f"additional_metadata; got {additional.get('adapter_version')!r}"
    )
    cap = additional.get("adapter_capability") or {}
    assert isinstance(cap, dict)
    # adapter_version must NOT be nested under adapter_capability.
    assert "adapter_version" not in cap, (
        "adapter_version must NOT be nested under adapter_capability; "
        "the translator-emitted shape places it at the top level."
    )
    declared = set(cap.get("preserve_raw_supported") or [])
    assert declared == _EXPECTED_DAILY, (
        f"translator-emitted preserve_raw_supported must be EXACTLY the "
        f"13-artifact union (5 base daily + 2 annual + 6 green/blue); "
        f"got {sorted(declared)}"
    )


def test_a7_acea_helper_emits_empty_extra_when_uc4_absent() -> None:
    """Translator-helper-level counterfactual (Gate-B fold of eval-2
    NICE-N2 + the same axis codex flagged for the CRAFT side): drive
    the actual ``AceaTranslator._build_uc4_capability_extra`` directly
    with a synthetic ``package_config`` whose ``use_case_config`` does
    NOT include ``drought_management`` and assert it returns an empty
    dict.

    Catches a regression that would unconditionally emit the modern
    capability surface — the manifest-write boundary test below would
    still pass because it bypasses the translator. Mirrors the UC4-
    POSITIVE runtime probe but exercises the no-UC4 branch.

    The accompanying positive-branch assertion (UC4 present →
    13-artifact union) anchors the symmetric expectation in one place.
    """
    from prismpy.translators.acea.translator import AceaTranslator

    translator = AceaTranslator.__new__(AceaTranslator)
    no_uc4 = {"use_case_config": {"yield_forecast": {}, "soil_fertility": {}}}
    assert translator._build_uc4_capability_extra(no_uc4) == {}, (
        "ACEA helper must return empty dict when drought_management "
        "is absent from use_case_config — capability surface is "
        "UC4-gated."
    )

    with_uc4 = {"use_case_config": {
        "yield_forecast": {}, "drought_management": {}, "soil_fertility": {},
    }}
    out = translator._build_uc4_capability_extra(with_uc4)
    assert out.get("adapter_version") == "1.0"
    cap = out.get("adapter_capability") or {}
    assert set(cap.get("preserve_raw_supported") or []) == _EXPECTED_DAILY, (
        "ACEA helper must emit the 13-artifact union when UC4 is "
        "present; mismatch indicates the gate-on-UC4 path drifted "
        "from the documented surface."
    )


def test_a7_runtime_acea_manifest_omits_capability_when_uc4_absent() -> None:
    """Counterfactual: when the translator does NOT supply
    ``additional_metadata`` (e.g., a future package that doesn't
    request UC4), the persisted manifest doesn't carry stray
    capability declarations. Bind to the blind-merge invariant: only
    keys actually passed in ``additional_metadata`` surface as
    top-level fields."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg"
        pkg.mkdir()
        (pkg / "metadata.json").write_text(json.dumps({}))
        project_config = {
            "project_name": "acea-test",
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
            "use_case_config": {
                "yield_forecast": {},
                "soil_fertility": {},
            },
        }
        manifest = create_manifest(
            pkg, project_config, platform="acea",
            additional_metadata=None,
        )
    assert "adapter_version" not in manifest, (
        "manifest must not carry adapter_version when additional_metadata "
        "is None — the translator only declares capability when UC4 is "
        "in use_case_config"
    )
    assert "adapter_capability" not in manifest


def test_a7_filter_strips_private_keys_keeps_capability() -> None:
    """The ``_filter_additional_metadata`` sanitisation strips
    ``_``-prefixed keys (private signals like the UC5 trigger) but
    keeps the public ``adapter_version`` + ``adapter_capability`` so
    they reach the persisted manifest."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "pkg"
        pkg.mkdir()
        (pkg / "metadata.json").write_text(json.dumps({}))
        project_config = {
            "project_name": "acea-test",
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
            "use_case_config": {
                "drought_management": {},
                "soil_fertility": {},
            },
        }
        additional_metadata = {
            # Private signal — must be stripped from the on-disk
            # manifest but available to the uc_readiness emitter via
            # the merged_project_config.
            "_acea_uc5_p_k_silent_no_op_triggered": True,
            # Public declarations — must survive to the manifest.
            "adapter_version": "1.0",
            "adapter_capability": {
                "preserve_raw_supported": sorted(_EXPECTED_DAILY),
            },
        }
        manifest = create_manifest(
            pkg, project_config, platform="acea",
            additional_metadata=additional_metadata,
        )
    assert "_acea_uc5_p_k_silent_no_op_triggered" not in manifest, (
        "private trigger keys must be stripped from the persisted manifest"
    )
    assert manifest.get("adapter_version") == "1.0"
    assert (
        set(manifest.get("adapter_capability", {})
            .get("preserve_raw_supported", []))
        == _EXPECTED_DAILY
    )
