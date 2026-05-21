"""Pattern #283 architectural-invariant guard — ACEA UC4 preserve_raw emit.

The ACEA translator declares the UC4 preserve_raw capability via
``additional_metadata`` blind-merge to ``create_manifest``. The
declaration must reach the persisted on-disk manifest as a top-level
``adapter_version`` plus a nested ``adapter_capability.preserve_raw_
supported`` list with the 5 daily artifacts. Two guards:

- **A7-syntactic**: AST-walks ``translators/acea/translator.py`` and
  asserts that the ``manifest_extra`` build site assigns the EXACT
  5-key ``preserve_raw_supported`` list AND ``adapter_version =
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


_EXPECTED_DAILY = {
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
    """Locate the translator's ``if 'drought_management' in ...`` gate
    whose body assigns ``manifest_extra['adapter_capability']``.

    The check uses the body content (not just the test expression) to
    avoid matching any unrelated `'drought_management' in ...` site.
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
        "ACEA translator must contain an `if 'drought_management' in ...:` "
        "gate whose body assigns adapter_capability + adapter_version"
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


def test_a7_syntactic_exact_5_daily_list_with_top_level_adapter_version() -> None:
    """SH-1 codex fold — tighten the syntactic check so the
    counterfactuals are actually verifiable:

    - The ``preserve_raw_supported`` list is the EXACT 5-key daily set
      (adding ``annual_total_biomass`` fails this).
    - ``adapter_version`` is assigned at the TOP LEVEL of
      ``manifest_extra`` (nesting it under ``adapter_capability`` fails
      this).
    - The gate uses the ``In`` operator (inverting to ``NotIn`` fails
      this).
    """
    tree = ast.parse(
        _ACEA_TRANSLATOR.read_text(encoding="utf-8"),
        filename=str(_ACEA_TRANSLATOR),
    )
    if_node = _find_uc4_capability_if(tree)

    # (1) Gate uses In, not NotIn.
    found_in_op = False
    for cmp in ast.walk(if_node.test):
        if isinstance(cmp, ast.Compare):
            for op in cmp.ops:
                if isinstance(op, ast.In):
                    found_in_op = True
                if isinstance(op, ast.NotIn):
                    pytest.fail(
                        "UC4 capability gate must use the In operator on "
                        "'drought_management'; found NotIn (inverted)."
                    )
    assert found_in_op, (
        "UC4 capability gate must use the In operator on "
        "'drought_management' (not equality / NotIn / Lt etc.)."
    )

    # (2) Subscript assignments on `manifest_extra` carry
    # ``adapter_version`` at the top level (not nested under
    # ``adapter_capability``).
    assigns = _extract_subscript_assignments(if_node, "manifest_extra")
    assert "adapter_version" in assigns, (
        "manifest_extra must carry adapter_version at the top level (not "
        "nested under adapter_capability)."
    )
    av_node = assigns["adapter_version"]
    assert (
        isinstance(av_node, ast.Constant) and av_node.value == "1.0"
    ), (
        f"manifest_extra['adapter_version'] must be the literal '1.0'; "
        f"got {ast.dump(av_node)!r}"
    )
    assert "adapter_capability" in assigns, (
        "manifest_extra must carry adapter_capability at the top level."
    )
    cap_node = assigns["adapter_capability"]
    assert isinstance(cap_node, ast.Dict), (
        "manifest_extra['adapter_capability'] must be a dict literal."
    )
    # The adapter_capability dict must NOT carry adapter_version
    # (nested-version counterfactual fails this).
    cap_keys = {
        k.value for k in cap_node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert "adapter_version" not in cap_keys, (
        "adapter_version must NOT be nested under adapter_capability; "
        "it must live at the top level of manifest_extra."
    )

    # (3) The preserve_raw_supported value is an EXACT 5-key List.
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
        f"preserve_raw_supported must be EXACTLY the 5 daily artifacts; "
        f"got {sorted(declared)}. Adding an annual artifact (e.g., "
        f"'annual_total_biomass') or any other key fails this guard."
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
        f"5 daily artifacts; got {sorted(declared)}"
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
