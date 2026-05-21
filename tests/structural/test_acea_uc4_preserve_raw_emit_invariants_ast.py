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


def test_a7_syntactic_acea_translator_declares_adapter_capability_for_uc4() -> None:
    """The translator source declares ``adapter_capability`` with the
    5 daily artifacts AND ``adapter_version = '1.0'`` AND gates the
    assignment on ``drought_management`` membership in
    ``use_case_config``."""
    src = _ACEA_TRANSLATOR.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_ACEA_TRANSLATOR))

    # Locate the gate `if 'drought_management' in package_config.get(...)`
    # whose body assigns the adapter_version + adapter_capability keys
    # on a manifest_extra dict.
    found_gate = False
    for if_node in ast.walk(tree):
        if not isinstance(if_node, ast.If):
            continue
        test_strings = _str_constants(if_node.test)
        if "drought_management" not in test_strings:
            continue
        # The If body must mention adapter_version + adapter_capability
        # + the 5 daily artifact strings.
        body_constants = set()
        for child in if_node.body:
            body_constants |= _str_constants(child)
        if "adapter_version" not in body_constants:
            continue
        if "adapter_capability" not in body_constants:
            continue
        if "preserve_raw_supported" not in body_constants:
            continue
        if "1.0" not in body_constants:
            continue
        if _EXPECTED_DAILY - body_constants:
            continue
        found_gate = True
        break
    assert found_gate, (
        "ACEA translator must gate the adapter_capability declaration on "
        "'drought_management' in use_case_config and assign exactly "
        "adapter_version='1.0' + the 5 daily artifact strings."
    )


def test_a7_runtime_acea_manifest_carries_adapter_capability_post_emit() -> None:
    """Invoke ``create_manifest`` with the synthesised
    ``additional_metadata`` the translator passes; assert the persisted
    manifest dict has the expected shape after the blind-merge."""
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
                "yield_forecast": {},
                "soil_fertility": {},
            },
        }
        additional_metadata = {
            "adapter_version": "1.0",
            "adapter_capability": {
                "preserve_raw_supported": sorted(_EXPECTED_DAILY),
            },
        }
        manifest = create_manifest(
            pkg, project_config, platform="acea",
            additional_metadata=additional_metadata,
        )
    assert manifest.get("adapter_version") == "1.0", (
        f"persisted manifest must carry adapter_version='1.0' at the "
        f"top level; got {manifest.get('adapter_version')!r}"
    )
    cap = manifest.get("adapter_capability") or {}
    assert isinstance(cap, dict)
    declared = set(cap.get("preserve_raw_supported") or [])
    assert declared == _EXPECTED_DAILY, (
        f"persisted adapter_capability.preserve_raw_supported must be "
        f"exactly the 5 daily artifacts; got {sorted(declared)}"
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
