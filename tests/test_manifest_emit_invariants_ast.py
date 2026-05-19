"""Pattern #283 architectural-invariant guard for the manifest emit surface.

Parses :mod:`prismpy.packaging.manifest` at the AST level and asserts the
structural contract that downstream consumers (prism-runner +
prismweb) rely on. Catches accidental key removal / refactor breakage
/ closed-world regressions long before integration tests would.

Test surface (per the canonical helper convention in
``prismpy/src/prismpy/`` post-PR48/50/51 — see module docstring of
``prismpy.packaging.manifest`` for the discipline rationale):

- **T1** emit cadence — every contract-bearing manifest key is present
  in the ``create_manifest`` body
- **T2** closed-world semantics — ``canonical_uc_readiness_emitter``
  iterates the emitted-UC source, NOT the global enum
- **T3** advisory_flag emit invariants — each spec'd flag string and
  emission condition is reachable from the source
- **T4** crops widening — ``canonical_crops_emitter`` returns
  ``list[dict]`` always and the legacy ``crop`` singleton key is also
  emitted from the body for backward-compatibility
- **T5** cells/cell_areas semantic source (STUB at this revision) — the
  full semantic assertion (path α reuse OR path β
  ``canonical_cell_area_km2`` helper call; NOT path γ uniform
  placeholder) is held until the crop-modeling-specialist Gate A
  approval lands. The stub asserts the helper exists and is reachable
  for the post-approval wire-in; full semantic check upgrades in the
  follow-up revision.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_MANIFEST_SRC_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "prismpy" / "packaging" / "manifest.py"
)


_TRANSLATOR_PLATFORMS = ("pythia", "sarra_py", "craft", "acea")
_TRANSLATOR_SRC_PATHS = tuple(
    Path(__file__).resolve().parent.parent
    / "src" / "prismpy" / "translators" / platform / "translator.py"
    for platform in _TRANSLATOR_PLATFORMS
)


def _load_manifest_ast() -> ast.Module:
    source = _MANIFEST_SRC_PATH.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(_MANIFEST_SRC_PATH))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"function {name!r} not found in {_MANIFEST_SRC_PATH}")


def _string_constants(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.add(node.value)
    return out


def _dict_literal_keys(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Dict):
            for k in child.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    return keys


def _subscript_assignment_keys(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
            slice_node = child.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                keys.add(slice_node.value)
    return keys


# ── T1: emit cadence ────────────────────────────────────────────────────────


REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "package_version",
    "generator",
    "platform",
    "project_name",
    "region",
    "crops",
    "crop",
    "temporal",
    "data_sources",
    "use_case_config",
    "summary",
    "cells",
    "files",
    "uc_readiness",
    "validation_status",
)


def test_t1_create_manifest_emits_all_contract_keys() -> None:
    tree = _load_manifest_ast()
    func = _find_function(tree, "create_manifest")

    literal_keys = _dict_literal_keys(func)
    subscript_keys = _subscript_assignment_keys(func)
    emitted = literal_keys | subscript_keys

    missing = sorted(set(REQUIRED_MANIFEST_KEYS) - emitted)
    assert not missing, (
        f"create_manifest body must emit every contract key per §2.7.6 "
        f"+ §2.7.3 + §2.7.6.1 MUST-7 + OQ-PR3-1 BL-3. Missing: {missing}. "
        f"Found: {sorted(emitted)}"
    )


# ── T2: closed-world semantics for uc_readiness emitter ─────────────────────


def _closed_world_iteration_present(func: ast.FunctionDef) -> bool:
    """Helper: detect closed-world iteration pattern.

    A function is "closed-world" iff:
      (a) it pulls a binding from ``project_config['use_case_config']``
          (subscript OR ``.get('use_case_config')`` call), AND
      (b) it loops over the keys / values / iter of that binding,
          NOT over a global enum like ``KNOWN_USE_CASE_NAMES``.

    We detect (a) via string-constant scan ('use_case_config' must
    appear as a constant somewhere) and (b) via for-loop analysis
    (no iter source containing 'KNOWN_USE_CASE_NAMES').
    """
    references_use_case_config_str = False
    iterates_known_enum = False
    has_for_loop = False
    for node in ast.walk(func):
        if isinstance(node, ast.Constant) and node.value == "use_case_config":
            references_use_case_config_str = True
        if isinstance(node, ast.For):
            has_for_loop = True
            iter_src = ast.unparse(node.iter)
            if "KNOWN_USE_CASE_NAMES" in iter_src:
                iterates_known_enum = True
    return (
        references_use_case_config_str
        and has_for_loop
        and not iterates_known_enum
    )


def test_t2_uc_readiness_emitter_iterates_emitted_ucs_not_all_six() -> None:
    tree = _load_manifest_ast()
    func = _find_function(tree, "canonical_uc_readiness_emitter")

    iterates_known_use_case_names = any(
        isinstance(node, ast.For)
        and "KNOWN_USE_CASE_NAMES" in ast.unparse(node.iter)
        for node in ast.walk(func)
    )
    assert not iterates_known_use_case_names, (
        "canonical_uc_readiness_emitter must NOT iterate "
        "KNOWN_USE_CASE_NAMES directly — closed-world contract per "
        "v0.3 BL-2 + parent §2.7.6 line ~2606 requires iteration over "
        "the emitted-UC source ('use_case_config' keyset). Iterating "
        "the global enum re-expands non-emitted UCs into the output "
        "dict and breaks the UI confirm-card HIDDEN-vs-DISABLED "
        "contract."
    )
    assert _closed_world_iteration_present(func), (
        "canonical_uc_readiness_emitter must reference "
        "'use_case_config' AND iterate over it (closed-world "
        "emitted-UC keyset)."
    )


def test_t2_use_case_config_serializer_iterates_emitted_ucs_not_all_six() -> None:
    tree = _load_manifest_ast()
    func = _find_function(tree, "canonical_use_case_config_serializer")

    iterates_known_use_case_names = any(
        isinstance(node, ast.For)
        and "KNOWN_USE_CASE_NAMES" in ast.unparse(node.iter)
        for node in ast.walk(func)
    )
    assert not iterates_known_use_case_names, (
        "canonical_use_case_config_serializer must NOT iterate "
        "KNOWN_USE_CASE_NAMES directly — same closed-world contract "
        "per v0.4 codex BL-2 residual."
    )
    assert _closed_world_iteration_present(func), (
        "canonical_use_case_config_serializer must reference "
        "'use_case_config' AND iterate over it."
    )


# ── T3: advisory_flag emit invariants ───────────────────────────────────────


SPECIFIED_ADVISORY_FLAG_LITERALS: tuple[str, ...] = (
    "sowing_rule_default_absent:falls_back_to_manifest_default",
    "pythia_pk_silent_no_op:fertility_stress_unmodeled_v3.1",
    "severity_tier:viz_layer_thresholds_v1",
    "roi_prices:viz_layer_regional_defaults",
    "herd_density:GLW_2020_default_supply_side_only",
)


def test_t3_advisory_flag_literal_strings_present() -> None:
    tree = _load_manifest_ast()
    string_constants = _string_constants(tree)

    missing = [
        flag for flag in SPECIFIED_ADVISORY_FLAG_LITERALS
        if flag not in string_constants
    ]
    assert not missing, (
        f"manifest.py must contain the §2.7.6.1 + §2.7.7 spec'd "
        f"advisory_flag literal strings. Missing: {missing}"
    )


def test_t3_uc1_shortfall_threshold_template_present() -> None:
    """UC1 advisory uses a per-package interpolated template
    (`<value>_kgha_<crop>_<region>`); assert the template skeleton is
    reachable as a string constant.
    """
    tree = _load_manifest_ast()
    string_constants = _string_constants(tree)
    template_anchor = "shortfall_threshold:viz_layer_default_"
    matches = [c for c in string_constants if c.startswith(template_anchor)]
    assert matches, (
        f"UC1 shortfall threshold advisory template "
        f"({template_anchor}...) not found in manifest.py string "
        f"constants."
    )


def test_t3_uc5_pythia_pk_advisory_conditional_on_acea_trigger() -> None:
    """The UC5 PYTHIA P+K advisory append must be conditional on the
    ACEA-translator-set trigger flag (closed-world: SARRA-Py / CRAFT /
    DSSAT translator paths do NOT set the flag → advisory absent).
    """
    tree = _load_manifest_ast()
    func = _find_function(tree, "canonical_uc_readiness_emitter")
    func_src = ast.unparse(func)
    assert "_acea_uc5_p_k_silent_no_op_triggered" in func_src, (
        "canonical_uc_readiness_emitter must gate the UC5 PYTHIA P+K "
        "advisory emission on the '_acea_uc5_p_k_silent_no_op_triggered' "
        "flag (the ACEA translator's :2915 side-effect) — keeps the "
        "advisory absent for non-ACEA translator paths per §2.7.6.1 "
        "MUST-6 + T2 contract."
    )


def test_t3_uc6_pythia_gates_failed_disclosure_present() -> None:
    """SH-4 disclosure: when platform='pythia' AND UC6 declared, the
    emitter appends a hard gates_failed entry for ``platform_supports_uc``.
    """
    tree = _load_manifest_ast()
    func = _find_function(tree, "canonical_uc_readiness_emitter")
    func_src = ast.unparse(func)
    assert "platform_supports_uc" in func_src, (
        "UC6+PYTHIA platform_supports_uc gates_failed disclosure per "
        "v0.3 SH-4 must be reachable from canonical_uc_readiness_emitter "
        "body."
    )


# ── T4: crops widening invariant ────────────────────────────────────────────


def test_t4_canonical_crops_emitter_returns_list_only() -> None:
    """The helper MUST always return ``list[dict]`` — no singleton
    code path. AST inspection: every Return statement's value
    expression is a list construction (either an ast.List literal or a
    list-comprehension).
    """
    tree = _load_manifest_ast()
    func = _find_function(tree, "canonical_crops_emitter")

    returns = [
        node for node in ast.walk(func)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    assert returns, "canonical_crops_emitter must have explicit Return statements"
    for ret in returns:
        node = ret.value
        is_list = isinstance(node, (ast.List, ast.ListComp))
        assert is_list, (
            f"canonical_crops_emitter Return at line {ret.lineno} must "
            f"return a list literal or list-comprehension (got "
            f"{type(node).__name__}). Singleton dict returns are "
            f"forbidden per §2.7.6.1 MUST-7."
        )


def test_t4_legacy_crop_singleton_key_still_emitted() -> None:
    """Backward-compat: ``manifest.crop`` legacy singleton key must
    still be emitted from create_manifest body alongside the new
    ``crops`` list.
    """
    tree = _load_manifest_ast()
    func = _find_function(tree, "create_manifest")
    keys = _dict_literal_keys(func) | _subscript_assignment_keys(func)
    assert "crop" in keys, (
        "Legacy 'crop' singleton key must still be emitted from "
        "create_manifest for pre-PR3 backward-compat consumers per "
        "§2.7.6.1 MUST-7."
    )
    assert "crops" in keys, (
        "New 'crops' list key must be emitted alongside legacy 'crop'."
    )


# ── T5: cells/cell_areas semantic source (FULL semantic check) ──────────────


_CELL_AREA_HELPER_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "prismpy" / "cells" / "canonical_cell_area_km2.py"
)


def test_t5_cells_and_cell_areas_keys_emitted_unconditionally() -> None:
    """OQ-PR3-1 BL-3 resolution path (a): both ``cells`` and
    ``cell_areas`` keys are emitted from ``create_manifest`` for ALL
    platforms — no platform-conditional gate that would re-introduce
    the prism-runner shim's manual-completion need.
    """
    tree = _load_manifest_ast()
    func = _find_function(tree, "create_manifest")
    keys = _dict_literal_keys(func) | _subscript_assignment_keys(func)
    assert "cells" in keys, "manifest dict literal must emit 'cells'"
    assert "cell_areas" in keys, "manifest dict literal must emit 'cell_areas'"


def test_t5_canonical_cell_area_km2_lives_in_canonical_module() -> None:
    """Per the canonical-helper convention (§3.5) + specialist 2026-05-19
    spec: the geodesic helper lives in
    ``prismpy/src/prismpy/cells/canonical_cell_area_km2.py`` as its own
    canonical-prefixed module — not buried inside
    ``packaging/manifest.py`` (where the v0.3 scaffold initially lived
    prior to specialist sign-off).
    """
    assert _CELL_AREA_HELPER_PATH.is_file(), (
        f"canonical_cell_area_km2 helper module not found at expected "
        f"path {_CELL_AREA_HELPER_PATH}; the canonical-helper convention "
        f"(§3.5) requires a dedicated module."
    )
    helper_src = _CELL_AREA_HELPER_PATH.read_text(encoding="utf-8")
    helper_tree = ast.parse(helper_src, filename=str(_CELL_AREA_HELPER_PATH))
    func_names = {
        node.name for node in ast.walk(helper_tree)
        if isinstance(node, ast.FunctionDef)
    }
    assert "canonical_cell_area_km2" in func_names, (
        "canonical_cell_area_km2 function must be defined in its "
        "canonical module."
    )


def test_t5_create_manifest_calls_canonical_cell_area_km2_helper() -> None:
    """Semantic source-method check per v0.4 codex SH-FIX-3 upgrade:
    the ``cell_areas`` emission MUST be backed by a call to
    ``canonical_cell_area_km2`` (path β: geodesic-computed). REJECTS
    path γ where ``cell_areas`` would be a uniform-placeholder
    constant detached from any per-cell geodesic source.
    """
    tree = _load_manifest_ast()
    func = _find_function(tree, "create_manifest")
    calls_helper = False
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == "canonical_cell_area_km2":
            calls_helper = True
            break
        if isinstance(target, ast.Attribute) and target.attr == "canonical_cell_area_km2":
            calls_helper = True
            break
    assert calls_helper, (
        "create_manifest must invoke canonical_cell_area_km2 to "
        "compute cell_areas[] per OQ-PR3-1 path β resolution. A "
        "uniform-default cell_areas emission (path γ — e.g., "
        "``[100.0] * len(cells)``) would satisfy key-presence but "
        "defeat the scientific-source requirement per §464 (Dr. Kofi "
        "paper-replication; Aminata thesis chapter)."
    )


def test_t5_spatial_ref_has_three_documented_fields() -> None:
    """Specialist 2026-05-19 spec: ``SpatialRef`` is a frozen dataclass
    with exactly three documented fields (``resolution_deg``,
    ``cell_centroid_latitude`` callable, ``deg2_to_km2`` constant).
    The AST guard asserts the dataclass exists + carries the three
    field names so future refactors can't silently drop a field.
    """
    helper_src = _CELL_AREA_HELPER_PATH.read_text(encoding="utf-8")
    helper_tree = ast.parse(helper_src, filename=str(_CELL_AREA_HELPER_PATH))
    spatial_ref_class = None
    for node in ast.walk(helper_tree):
        if isinstance(node, ast.ClassDef) and node.name == "SpatialRef":
            spatial_ref_class = node
            break
    assert spatial_ref_class is not None, (
        "SpatialRef dataclass not found in canonical_cell_area_km2.py"
    )
    field_names = {
        stmt.target.id
        for stmt in spatial_ref_class.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    for required in (
        "resolution_deg",
        "cell_centroid_latitude",
        "deg2_to_km2",
    ):
        assert required in field_names, (
            f"SpatialRef must declare '{required}' field per specialist "
            f"2026-05-19 spec; found fields: {sorted(field_names)}"
        )


# ── B1: translator call sites must populate use_case_config ─────────────────


def _find_create_manifest_call(tree: ast.Module) -> ast.Call:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id == "create_manifest":
                return node
            if isinstance(target, ast.Attribute) and target.attr == "create_manifest":
                return node
    pytest.fail("no create_manifest() call found in translator source")


def _project_config_for_call(call_node: ast.Call, tree: ast.Module) -> ast.AST:
    """Resolve the project_config argument of ``create_manifest`` to its
    dict-literal source. Accepts both positional and keyword
    invocations (``project_config=...``); the name (when not inline) is
    resolved by walking the module body for the most recent assignment.
    """
    arg: ast.AST | None = None
    if len(call_node.args) >= 2:
        arg = call_node.args[1]
    if arg is None:
        for kw in call_node.keywords:
            if kw.arg in ("project_config", "package_config"):
                arg = kw.value
                break
    if arg is None:
        pytest.fail("could not locate project_config argument on create_manifest()")
    if isinstance(arg, ast.Dict):
        return arg
    if isinstance(arg, ast.Name):
        target_name = arg.id
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for assign_target in node.targets:
                    if isinstance(assign_target, ast.Name) and assign_target.id == target_name:
                        if isinstance(node.value, ast.Dict):
                            return node.value
    pytest.fail(
        f"could not resolve create_manifest project_config arg to a "
        f"dict literal (got {type(arg).__name__})"
    )


@pytest.mark.parametrize(
    "translator_path", _TRANSLATOR_SRC_PATHS,
    ids=lambda p: p.parent.name,
)
def test_b1_translator_populates_use_case_config(translator_path: Path) -> None:
    """B1 closure: every translator's ``create_manifest`` call must
    pass a ``project_config`` (a.k.a. ``package_config``) dict that
    declares a non-empty ``use_case_config`` keyset.

    Pre-Phase-F-C the 4 translators built their package_config without
    ``use_case_config`` → ``canonical_use_case_config_serializer`` and
    ``canonical_uc_readiness_emitter`` iterated empty source → PR3's
    native-emission feature was non-functional in production
    (codex Gate B v1 BLOCKING-1). This test prevents regression by
    AST-asserting each translator's package_config dict literal
    contains the ``use_case_config`` key with at least one UC entry.
    """
    src = translator_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(translator_path))
    call = _find_create_manifest_call(tree)
    pc_dict = _project_config_for_call(call, tree)

    use_case_config_value = None
    for key, value in zip(pc_dict.keys, pc_dict.values):
        if isinstance(key, ast.Constant) and key.value == "use_case_config":
            use_case_config_value = value
            break

    assert use_case_config_value is not None, (
        f"{translator_path.parent.name} translator's package_config "
        f"dict literal must declare 'use_case_config' key — without it, "
        f"the manifest emitter sees an empty UC source and the "
        f"closed-world emit produces empty uc_readiness."
    )
    assert isinstance(use_case_config_value, ast.Dict), (
        f"{translator_path.parent.name} translator's 'use_case_config' "
        f"value must be a dict literal (got "
        f"{type(use_case_config_value).__name__})"
    )
    declared_ucs = [
        k.value for k in use_case_config_value.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    ]
    assert len(declared_ucs) >= 1, (
        f"{translator_path.parent.name} translator must declare at "
        f"least one UC in 'use_case_config' (got empty dict)."
    )
    for uc_name in declared_ucs:
        assert uc_name in {
            "yield_forecast",
            "climate_scenarios",
            "sowing_optimization",
            "drought_management",
            "soil_fertility",
            "livestock_feed",
        }, (
            f"{translator_path.parent.name} translator declares "
            f"unknown UC name {uc_name!r} in use_case_config; expected "
            f"one of the 6 KNOWN_USE_CASE_NAMES."
        )
