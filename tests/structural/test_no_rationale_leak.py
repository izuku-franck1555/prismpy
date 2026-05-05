"""Sprint E.1 AC-E1-12 / F33 — pin the no-rationale-leak
discipline across prismpy provenance + lineage + manifest
surfaces.

User-provided override rationale is sensitive content (PII +
free-form text + audit-trail integrity); per the privacy
invariant in :class:`WizardOverrideRecord` it stays in the
audit log only. Three load-bearing surfaces MUST NOT include
it in any methods-text-renderable / cockpit-manifest /
lineage-rail-renderable output:

* :class:`prismpy.cockpit.CockpitManifestEntry` — manifest is
  read by the cockpit's per-bucket grouping; must NOT carry
  ``rationale`` as a structural field. A future contributor
  cannot accidentally add ``rationale: str`` to the dataclass
  without F33 firing.
* :class:`prismpy.cockpit.compute_manifest_hash` /
  :class:`prismpy.cockpit.snapshot_for_pipeline_run` — both
  serialize structurally; their AST surface MUST NOT include
  any ``rationale`` access on a decision-shaped object.
* :class:`prismpy.koppen.cockpit_readiness_contract` schema
  reference — the field-set declarations MUST NOT include
  ``rationale`` (the cockpit drawer reads from the schema; if
  ``rationale`` lands here the wider Bucket 5 panel surfaces
  it visibly).

Codex Gate A SHOULD-FIX-4 requested the broader scope (not
just methods-text); the walker covers the three surfaces
where leakage would translate to visible PII drift in the
cockpit + lineage rail + manifest.

Walker pattern: AST scan for any ``Name`` / ``Attribute``
access on ``rationale`` inside the protected modules. The
:class:`WizardOverrideRecord` declaration is allowlisted —
that's the storage site, not a leak surface.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class _RationaleAccessFinder(ast.NodeVisitor):
    """Collect every `rationale` Name / Attribute access in a
    parse tree, with line numbers."""

    def __init__(self):
        self.hits: list[tuple[int, str]] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "rationale":
            self.hits.append((node.lineno, "Attribute"))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "rationale":
            self.hits.append((node.lineno, "Name"))
        self.generic_visit(node)


def _scan(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    finder = _RationaleAccessFinder()
    finder.visit(tree)
    return finder.hits


class TestNoRationaleLeakInCockpitManifest(unittest.TestCase):
    """Pin the manifest module."""

    def test_manifest_module_no_rationale_access(self):
        path = REPO_ROOT / "src" / "prismpy" / "cockpit" / "manifest.py"
        hits = _scan(path)
        self.assertEqual(
            hits, [],
            f"prismpy/cockpit/manifest.py must NOT access "
            f"`rationale` (the cockpit manifest is a "
            f"structural-only surface; rationale leak would "
            f"translate into visible PII in the per-bucket "
            f"grouping). Found at {hits!r}.",
        )

    def test_per_run_snapshot_module_no_rationale_access(self):
        path = (
            REPO_ROOT / "src" / "prismpy" / "cockpit"
            / "per_run_snapshot.py"
        )
        hits = _scan(path)
        self.assertEqual(
            hits, [],
            f"prismpy/cockpit/per_run_snapshot.py must NOT "
            f"access `rationale` (per-run snapshot is the "
            f"frozen-at-launch read of a structural payload; "
            f"rationale leak would surface in the lineage "
            f"rail). Found at {hits!r}.",
        )


class TestNoRationaleLeakInCockpitReadinessContract(unittest.TestCase):
    """Pin the schema reference module."""

    def test_contract_module_no_rationale_in_field_sets(self):
        path = (
            REPO_ROOT / "src" / "prismpy" / "koppen"
            / "cockpit_readiness_contract.py"
        )
        source = path.read_text(encoding="utf-8")
        # The schema reference module declares structural
        # field sets; ``rationale`` must NOT appear in any
        # frozenset member. AST scan for the literal string
        # would also catch a docstring mention; tighten via
        # parsed-tree scan of FrozenSet.elts.
        tree = ast.parse(source)
        # All Constant string nodes inside the file (keys of
        # the field-set declarations).
        forbidden = "rationale"
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Skip docstring / module docstring nodes;
                # those are not schema-declarations. Heuristic:
                # constants whose parent is a Set / FrozenSet /
                # Call to frozenset are the field-set members.
                # The simpler check: assert no field-set
                # constant matches "rationale".
                if node.value == forbidden:
                    self.fail(
                        f"cockpit_readiness_contract.py contains "
                        f"a string constant {forbidden!r} at "
                        f"line {node.lineno}; the schema "
                        f"reference must NOT include "
                        f"``rationale`` as a declared field "
                        f"(per F33 / privacy invariant). "
                        f"Rationale stays in the audit log "
                        f"only."
                    )


class TestRationaleAllowedInOverrideRecordStorage(unittest.TestCase):
    """Anti-mutation backstop — ``rationale`` IS allowed in
    the storage site (:class:`WizardOverrideRecord`) and the
    audit-log writer (:meth:`record_wizard_decision` /
    :meth:`record_cockpit_decision`). The walker's allowlist
    discipline keeps F33 narrow."""

    def test_wizard_decisions_module_uses_rationale(self):
        # Sanity check that the storage site DOES read
        # rationale — if a future refactor accidentally
        # removes the storage path, this catches it.
        path = (
            REPO_ROOT / "src" / "prismpy" / "provenance"
            / "wizard_decisions.py"
        )
        hits = _scan(path)
        self.assertGreater(
            len(hits), 0,
            "wizard_decisions.py is the storage site for "
            "rationale; AST scan must find at least one "
            "rationale access. If this assertion fails, the "
            "storage path was removed.",
        )


if __name__ == "__main__":
    unittest.main()
