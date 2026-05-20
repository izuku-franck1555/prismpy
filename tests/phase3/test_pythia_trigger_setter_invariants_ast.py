"""Pattern #283 architectural-invariant guards for the PYTHIA UC5 trigger.

AST-parses ``translators/pythia/translator.py`` and asserts the
verified-correct shape so a future refactor cannot silently remove the
trigger and return PYTHIA UC5 packages to the broken pre-fix state
where the joint advisory_flag was never emitted in production.

Two invariants (per the cycle-4 prismpy fix contract section 4):

- **T1 trigger setter exists**: the PYTHIA translator builds an
  ``additional_metadata`` dict containing
  ``"_acea_uc5_p_k_silent_no_op_triggered": True`` and passes it to the
  ``create_manifest`` call with ``platform="pythia"``. Mirrors the ACEA
  pattern at ``acea/translator.py:2408-2417``.
- **T2 cross-translator state-handoff documentation**: the trigger
  setter site carries an inline comment explaining the producer-chain
  semantics so future maintainers do not strip the "always-set" form
  thinking it is a no-op gate (the v0.2 audit N-V2-2 framing).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PYTHIA_TRANSLATOR = (
    _REPO_ROOT / "src" / "prismpy" / "translators" / "pythia" / "translator.py"
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"function {name!r} not found")


def _find_create_manifest_calls(scope: ast.AST) -> list[ast.Call]:
    out: list[ast.Call] = []
    for node in ast.walk(scope):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "create_manifest":
                out.append(node)
            elif isinstance(fn, ast.Attribute) and fn.attr == "create_manifest":
                out.append(node)
    return out


# ── T1: trigger setter exists + flows into create_manifest ──────────────────


def test_t1_pythia_translator_builds_additional_metadata_with_trigger() -> None:
    """The PYTHIA translator source declares an ``additional_metadata``
    dict whose keys include ``_acea_uc5_p_k_silent_no_op_triggered``.
    Without this key the manifest emit gate at ``manifest.py:644-650``
    cannot resolve ``uc5_pythia_pk_triggered`` to True, and the joint
    advisory_flag never lands on the ``soil_fertility`` uc_readiness
    entry (the cycle-4 BLOCKING-1 condition).
    """
    tree = _parse(_PYTHIA_TRANSLATOR)
    # Find the assignment that builds the ``additional_metadata`` dict.
    found_assign = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "additional_metadata"
            and isinstance(node.value, ast.Dict)
        ):
            keys = {
                k.value
                for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            if "_acea_uc5_p_k_silent_no_op_triggered" in keys:
                found_assign = True
                break
    assert found_assign, (
        "pythia/translator.py must assign an ``additional_metadata`` dict "
        "containing the ``_acea_uc5_p_k_silent_no_op_triggered`` key "
        "(mirrors the ACEA pattern at acea/translator.py:2408-2417)."
    )


def test_t1_create_manifest_called_with_additional_metadata_and_pythia() -> None:
    """The trigger-bearing ``additional_metadata`` reaches the
    ``create_manifest`` call with ``platform="pythia"`` (the gate at
    ``manifest.py:644-650`` requires both signals)."""
    tree = _parse(_PYTHIA_TRANSLATOR)
    create_calls = _find_create_manifest_calls(tree)
    assert create_calls, "no create_manifest call found in pythia/translator.py"
    matching = []
    for call in create_calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        platform_val = kwargs.get("platform")
        is_pythia = (
            isinstance(platform_val, ast.Constant)
            and platform_val.value == "pythia"
        )
        additional = kwargs.get("additional_metadata")
        has_additional = additional is not None
        if is_pythia and has_additional:
            matching.append(call)
    assert matching, (
        "pythia/translator.py must call create_manifest with BOTH "
        "platform='pythia' AND additional_metadata=... ; the trigger "
        "must reach the emitter through the create_manifest call site."
    )


def test_t1_trigger_value_is_true_not_none() -> None:
    """The trigger value must be ``True`` (boolean Constant), not None /
    a runtime conditional that could evaluate to falsy. The manifest
    gate at :604-606 reads ``bool(... .get(key))`` so a falsy value
    silently disables emission."""
    tree = _parse(_PYTHIA_TRANSLATOR)
    saw_true_assignment = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "additional_metadata"
            and isinstance(node.value, ast.Dict)
        ):
            for k, v in zip(node.value.keys, node.value.values):
                if (
                    isinstance(k, ast.Constant)
                    and k.value == "_acea_uc5_p_k_silent_no_op_triggered"
                    and isinstance(v, ast.Constant)
                    and v.value is True
                ):
                    saw_true_assignment = True
    assert saw_true_assignment, (
        "the trigger value must be the literal ``True``; a None / "
        "conditional / falsy expression silently disables the gate."
    )


# ── T2: cross-translator state-handoff documentation ────────────────────────


def test_t2_trigger_setter_carries_handoff_comment() -> None:
    """The trigger-setter site documents the cross-translator state-
    handoff: the per-package trigger is set from each translator's own
    UC5 detection (PYTHIA + ACEA each set it; no shared state).
    Mentions the downstream emit gate at ``manifest.py:644-650`` and
    the parallel ACEA pattern.

    Per audit NICE-2: future maintainers must understand that the
    "always-set-for-PYTHIA" form is correct (the v0.2 framing
    correction — a self-detect gate inside the translator would be a
    no-op because ``use_case_config`` unconditionally includes
    ``soil_fertility``).
    """
    src = _PYTHIA_TRANSLATOR.read_text(encoding="utf-8")
    # The comment must reference the ACEA mirror anchor + the
    # downstream emit gate so the maintainer can trace the handoff.
    assert "acea/translator.py" in src, (
        "trigger-setter comment must reference the parallel ACEA pattern "
        "(acea/translator.py:2408-2417) so the handoff intent is clear."
    )
    assert "manifest.py" in src, (
        "trigger-setter comment must reference the downstream emit gate "
        "(packaging/manifest.py:644-650) for the per-UC filter."
    )
    assert "soil_fertility" in src and "PHOSP" in src, (
        "trigger-setter comment must reference the trigger semantic "
        "(PYTHIA UC5 silently no-ops P+K; PHOSP=N + POTAS=N hardcode)."
    )
