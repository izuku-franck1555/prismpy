"""Pattern #283 architectural-invariant guards for UC5 P+K disclosure emit.

F-BP-5 cycle R1 closes the CRAFT UC5 disclosure gap by mirroring the
cycle-4 PYTHIA producer fix on the CRAFT translator and widening the
prismpy emit gate from PYTHIA-only to the DSSAT-engine producer set
``{"pythia", "craft"}``. These AST guards lock the producer-side
invariants so a future refactor cannot silently revert either change
and return CRAFT UC5 packages to the pre-cycle-R1 silent state.

Two invariants (prismpy side; the consumer-side guards live in the
prism-runner repo):

- **P1 emit gate widening**: ``packaging/manifest.py`` emits the UC5
  P+K joint flag for ``platform in {"pythia", "craft"}`` — both
  literals must appear in the gate's set-membership check.
- **P2 CRAFT translator trigger setter**: ``translators/craft/translator.py``
  passes ``additional_metadata=manifest_extra`` to ``create_manifest``,
  and the ``manifest_extra`` dict contains
  ``_acea_uc5_p_k_silent_no_op_triggered: True``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MANIFEST = _REPO_ROOT / "src" / "prismpy" / "packaging" / "manifest.py"
_CRAFT_TRANSLATOR = (
    _REPO_ROOT / "src" / "prismpy" / "translators" / "craft" / "translator.py"
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"function {name!r} not found")


def _str_constants(node: ast.AST) -> set[str]:
    return {
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


# ── P1: emit gate widening ──────────────────────────────────────────────────


def test_p1_uc5_emit_gate_includes_pythia_and_craft() -> None:
    """The UC5 P+K joint-flag emit gate at ``packaging/manifest.py`` must
    test ``platform in {"pythia", "craft"}`` (or equivalent set with the
    same two literals). The cycle-R1 widening from PYTHIA-only to
    PYTHIA+CRAFT is the load-bearing producer-side change; a regression
    that reverts to ``platform == "pythia"`` (or that adds a non-DSSAT
    platform like ``"sarra_py"`` to the set) breaks the engine-axis
    invariant the test deck (T6 in the contract) relies on.

    Asserts EXACT set membership: ``"pythia"`` AND ``"craft"`` both
    appear inside the gate's test expression as string constants. Does
    NOT allow other platforms in the set — drift to a superset like
    ``{"pythia", "craft", "acea"}`` is also blocked.
    """
    tree = _parse(_MANIFEST)
    emitter = _find_function(tree, "canonical_uc_readiness_emitter")
    # Locate the If whose body appends the UC5 PYTHIA P+K flag, then
    # inspect its test expression for the platform-set membership.
    found_gate = False
    for if_node in ast.walk(emitter):
        if not isinstance(if_node, ast.If):
            continue
        body_constants = _str_constants(
            ast.Module(body=if_node.body, type_ignores=[])
        )
        if "pythia_pk_silent_no_op:fertility_stress_unmodeled_v3.1" not in body_constants:
            # Look at the names referenced in the body for the constant
            # name pattern as well (constant may be referenced by Name).
            body_names = {
                n.id for n in ast.walk(
                    ast.Module(body=if_node.body, type_ignores=[])
                ) if isinstance(n, ast.Name)
            }
            if "ADVISORY_FLAG_UC5_PYTHIA_PK_SILENT_NO_OP" not in body_names:
                continue
        test_constants = _str_constants(if_node.test)
        if "pythia" in test_constants and "craft" in test_constants:
            # Reject drift to a non-DSSAT platform in the set.
            unexpected = test_constants - {"pythia", "craft", "soil_fertility"}
            # ``soil_fertility`` is the uc_name guard and lives in the
            # same `and`-chained test; allow it to coexist. Anything else
            # is a drift signal.
            assert not unexpected, (
                f"emit gate platform-set drift: expected exactly "
                f"{{'pythia', 'craft'}}, got extra string constants "
                f"{sorted(unexpected)} in the gate test expression"
            )
            found_gate = True
            break
    assert found_gate, (
        "packaging/manifest.py UC5 emit gate must test "
        "`platform in {\"pythia\", \"craft\"}` — both literals must "
        "appear in the gate's test expression"
    )


# ── P2: CRAFT translator trigger setter ────────────────────────────────────


def test_p2_craft_translator_passes_trigger_to_create_manifest() -> None:
    """The CRAFT translator must build a ``manifest_extra`` dict
    containing ``_acea_uc5_p_k_silent_no_op_triggered: True`` and pass
    it as ``additional_metadata`` to the ``create_manifest`` call with
    ``platform='craft'``. Mirrors the ACEA pattern at
    ``acea/translator.py:2408-2417`` — same SSOT trigger key (Lesson
    #24 canonical-source-or-pin); single engine-axis trigger for all
    DSSAT-engine producer paths.
    """
    tree = _parse(_CRAFT_TRANSLATOR)
    # The ``manifest_extra`` dict must be assigned somewhere with the
    # trigger key set to True.
    saw_trigger_assignment = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "manifest_extra"
            and isinstance(node.value, ast.Dict)
        ):
            for k, v in zip(node.value.keys, node.value.values):
                if (
                    isinstance(k, ast.Constant)
                    and k.value == "_acea_uc5_p_k_silent_no_op_triggered"
                    and isinstance(v, ast.Constant)
                    and v.value is True
                ):
                    saw_trigger_assignment = True
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "manifest_extra"
            and isinstance(node.value, ast.Dict)
        ):
            for k, v in zip(node.value.keys, node.value.values):
                if (
                    isinstance(k, ast.Constant)
                    and k.value == "_acea_uc5_p_k_silent_no_op_triggered"
                    and isinstance(v, ast.Constant)
                    and v.value is True
                ):
                    saw_trigger_assignment = True
    assert saw_trigger_assignment, (
        "craft/translator.py must build a ``manifest_extra`` dict with "
        "``_acea_uc5_p_k_silent_no_op_triggered: True`` (boolean Constant; "
        "a runtime conditional that could evaluate to falsy silently "
        "disables the disclosure)"
    )
    # The create_manifest call with platform='craft' must receive
    # additional_metadata=manifest_extra.
    saw_call = False
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        if not (
            (isinstance(fn, ast.Name) and fn.id == "create_manifest")
            or (isinstance(fn, ast.Attribute) and fn.attr == "create_manifest")
        ):
            continue
        # Check platform kwarg or positional arg matches "craft".
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        platform_val = kwargs.get("platform")
        is_craft = (
            isinstance(platform_val, ast.Constant)
            and platform_val.value == "craft"
        )
        if not is_craft:
            continue
        # additional_metadata kwarg must be present and reference
        # manifest_extra (Name) — proves the trigger flows in.
        additional = kwargs.get("additional_metadata")
        if isinstance(additional, ast.Name) and additional.id == "manifest_extra":
            saw_call = True
            break
    assert saw_call, (
        "craft/translator.py must call ``create_manifest(..., "
        "platform='craft', additional_metadata=manifest_extra, ...)`` "
        "so the trigger reaches the emit gate"
    )
