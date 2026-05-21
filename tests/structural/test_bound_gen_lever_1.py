"""F-DS Phase 3 Pin PH3-prismpy-L1 — bound-gen workflow scope hygiene.

The bound-gen workflow `push` trigger MUST be scoped to the `main`
branch so feature-branch pushes (already gated by `pull_request`)
do not double-run the designated-runner job. Mirrors prismweb's
Phase 2 Lever-1 push-scope discipline.

The deprecated Node 20 actions (`actions/checkout@v4`,
`actions/setup-python@v5`) MUST be upgraded to their Node 22
successors (`@v5` and `@v6` respectively) per GitHub Actions
runtime-deprecation policy.

Negative-case mutations: drop `branches: [main]` from the push
trigger OR revert either action to its Node 20 version → pin FAILS.
"""
from __future__ import annotations

import re
from pathlib import Path

_BOUND_GEN = (
    Path(__file__).resolve().parents[2]
    / ".github" / "workflows" / "bound-gen.yml"
)

# Allowlist: explicit Node 22 versions for the actions bound-gen
# uses today. Adding a new action requires extending this map.
_ALLOWED_ACTION_VERSIONS = {
    "actions/checkout": ("v5",),
    "actions/setup-python": ("v6",),
}


def _yaml_text() -> str:
    return _BOUND_GEN.read_text(encoding="utf-8")


def test_push_trigger_scoped_to_main_branch() -> None:
    text = _yaml_text()
    # The push: block must declare `branches: [main]` (or YAML
    # list form) before the `paths:` filter so squash-merges
    # are the only push events that fire bound-gen.
    push_block = re.search(
        r"^\s*push:\s*\n((?:\s+.+\n)+?)(?=^\s*(?:pull_request|workflow_dispatch):)",
        text, re.MULTILINE,
    )
    assert push_block, "bound-gen.yml `push:` trigger block missing"
    body = push_block.group(1)
    assert re.search(r"branches:\s*\n\s*-\s*main\b", body) or \
        re.search(r"branches:\s*\[\s*main\s*\]", body), (
            "bound-gen.yml `push:` MUST scope `branches: [main]` so "
            "feature-branch pushes don't double-run alongside "
            "pull_request CI (F-DS Phase 3 Lever-1)."
        )


def test_actions_pinned_to_node_22_versions() -> None:
    text = _yaml_text()
    references = re.findall(r"uses:\s*([\w\-./]+)@(v\d+)", text)
    offenders = []
    for action, version in references:
        allowed = _ALLOWED_ACTION_VERSIONS.get(action)
        if allowed is None:
            offenders.append(
                f"{action}@{version} (not in allowlist; extend "
                f"_ALLOWED_ACTION_VERSIONS in this pin)"
            )
            continue
        if version not in allowed:
            offenders.append(
                f"{action}@{version} (allowed: {','.join(allowed)})"
            )
    assert not offenders, (
        "bound-gen.yml carries Node 20-deprecated action versions; "
        "upgrade to Node 22 successors:\n  - "
        + "\n  - ".join(offenders)
    )
