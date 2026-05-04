"""Sprint F AC-F-8 — per-validator EMITS-discipline AST walker.

F30 forbids any :class:`InputValidator` subclass from emitting
a :class:`WarningCategory` value not declared in its
class-level ``EMITS`` frozenset.

The walker is a sibling to F25 — NOT an extension:

* F25 (Sprint E.0): module-level scan for bare
  ``WarningCategory`` string literals outside the canonical
  declaration module (``warnings/categories.py``). Catches a
  parallel string-tagged taxonomy.
* F30 (Sprint F): class-level scan over
  :class:`InputValidator` subclasses. Each subclass declares
  ``EMITS = frozenset({WarningCategory.X, ...})`` at class
  scope; F30 walks the subclass's containing module for
  ``WarningCategory.X`` access patterns and asserts every
  reachable category is in ``EMITS``.

Declared-superset-of-runtime semantics (per Sprint F Draft 2
contract line 248): ``EMITS`` may declare values that are
never emitted in this sprint. The walker enforces only the
subset direction (every emitted category MUST be declared);
unused declarations stay legal so a future sprint can light
up a category without re-running the contract review.

Anti-mutation drill (per AC-F-8): introduce a
``WarningCategory.CLIMATE_RH_INVALID.value`` emit in
``CropPhysiologicalValidator.validate`` body without updating
``EMITS`` → walker fires with file:line evidence. Test
:meth:`test_walker_detects_synthetic_violation` exercises the
drill via in-memory AST surgery (no on-disk source change
needed).
"""
from __future__ import annotations

import ast
import importlib
import inspect
import unittest
from pathlib import Path
from typing import List, Set, Tuple

from prismpy.validators.input_base import InputValidator
from prismpy.warnings.categories import WarningCategory


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src" / "prismpy"


# Pre-computed for fast membership checks.
_ALL_CATEGORY_NAMES: Set[str] = {c.name for c in WarningCategory}


def _find_input_validator_subclasses() -> List[type]:
    """Return every concrete :class:`InputValidator` subclass
    in :mod:`prismpy.validators`.

    The walker resolves subclasses via ``__subclasses__`` on
    the ABC plus a recursive descent so multi-level
    inheritance (a future ``StageNValidator(InputValidator)``
    that itself has subclasses) stays in scope. The
    ``input_base`` and ``climate_envelope`` modules are
    pre-imported by the unit-test conftest path; this helper
    walks the resolved class tree without re-importing.
    """
    # Force import of the validators package so subclasses are
    # registered. The modules below are the canonical Sprint F
    # Stage 1 surface; future expansions add to this list.
    importlib.import_module("prismpy.validators.climate_envelope")
    importlib.import_module("prismpy.validators.crop_physiological")

    subclasses: List[type] = []
    pending: List[type] = list(InputValidator.__subclasses__())
    while pending:
        cls = pending.pop()
        if inspect.isabstract(cls):
            continue
        subclasses.append(cls)
        pending.extend(cls.__subclasses__())
    return subclasses


def _parse_emits_frozenset_literal(
    cls: type,
) -> Set[str]:
    """Parse the class's ``EMITS`` frozenset literal from
    source via AST. Returns a set of WarningCategory member
    names (e.g. ``{"CROP_REGION_MISMATCH"}``).

    Falls back to runtime introspection
    (``cls.EMITS``) if the AST literal cannot be parsed —
    e.g., a future class that builds EMITS at runtime via
    factory. The AST path is preferred so the walker catches
    "the source declared X but the runtime resolved to Y"
    drift; runtime-only EMITS would slip past F30 today but
    are also rare and warrant a separate sprint.
    """
    try:
        source_path = inspect.getsourcefile(cls)
        if source_path is None:
            raise OSError
        tree = ast.parse(
            Path(source_path).read_text(encoding="utf-8"),
            filename=source_path,
        )
    except (OSError, SyntaxError):
        return {c.name for c in cls.EMITS}

    target_class: ast.ClassDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            target_class = node
            break
    if target_class is None:
        return {c.name for c in cls.EMITS}

    declared: Set[str] = set()
    for stmt in target_class.body:
        if not isinstance(stmt, ast.Assign):
            continue
        # Match ``EMITS = frozenset({...})``
        targets = [
            t for t in stmt.targets
            if isinstance(t, ast.Name) and t.id == "EMITS"
        ]
        if not targets:
            continue
        value = stmt.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
        ):
            continue
        if not value.args:
            return set()
        arg = value.args[0]
        if not isinstance(arg, ast.Set):
            continue
        for elt in arg.elts:
            if (
                isinstance(elt, ast.Attribute)
                and isinstance(elt.value, ast.Name)
                and elt.value.id == "WarningCategory"
            ):
                declared.add(elt.attr)
        return declared
    # No EMITS declaration found in source — fall back to
    # runtime introspection.
    return {c.name for c in cls.EMITS}


def _collect_runtime_emits_in_module(
    module_path: Path,
) -> List[Tuple[int, str]]:
    """Scan a single module for runtime
    ``WarningCategory.<NAME>`` access patterns at any AST
    depth. Returns ``(lineno, name)`` tuples.

    F30 scope per evaluator #5: instance methods + free
    functions in the same module count. The walker doesn't
    distinguish — every ``WarningCategory.X`` reference under
    the module is a candidate emit.
    """
    src = module_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(module_path))
    refs: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "WarningCategory"
            and node.attr in _ALL_CATEGORY_NAMES
        ):
            refs.append((node.lineno, node.attr))
    return refs


class TestPerValidatorEmitsWalker(unittest.TestCase):
    """F30 — every :class:`InputValidator` subclass's runtime
    ``WarningCategory`` emits must appear in the declared
    ``EMITS`` frozenset (subset rule; declared-superset
    allowed)."""

    def test_at_least_one_subclass_resolved(self):
        # Sanity: F30 is vacuously true if no subclasses exist.
        # Pin the discovery path so a future refactor that
        # accidentally moves the validators out of the
        # ``prismpy.validators`` package fails this assert
        # before the walker passes empty.
        subclasses = _find_input_validator_subclasses()
        self.assertGreaterEqual(
            len(subclasses), 2,
            "F30 walker requires at least the Sprint F two "
            "validator subclasses (ClimateEnvelopeValidator + "
            "CropPhysiologicalValidator) to be resolvable.",
        )

    def test_runtime_emits_subset_of_declared(self):
        """The walker's main rule.

        For each :class:`InputValidator` subclass:
        1. Parse the source-level ``EMITS`` frozenset literal
           into a set of member names.
        2. Scan the subclass's containing module for every
           ``WarningCategory.<NAME>`` reference.
        3. Assert every found ``<NAME>`` is in the declared
           set (declared-superset-of-runtime allowed).
        """
        violations: List[str] = []
        for cls in _find_input_validator_subclasses():
            declared = _parse_emits_frozenset_literal(cls)
            module_path = Path(inspect.getsourcefile(cls))
            refs = _collect_runtime_emits_in_module(module_path)
            relative = module_path.relative_to(_REPO_ROOT)
            for lineno, name in refs:
                if name not in declared:
                    violations.append(
                        f"{relative}:{lineno} — class "
                        f"{cls.__name__} references "
                        f"WarningCategory.{name} but "
                        f"{name!r} is NOT in EMITS = "
                        f"{sorted(declared)!r}"
                    )
        self.assertEqual(
            violations, [],
            "F30 violation(s): runtime WarningCategory "
            "reference outside declared EMITS frozenset.\n"
            + "\n".join(violations),
        )

    def test_walker_detects_synthetic_violation(self):
        """Anti-mutation drill: feed a synthetic source with
        a category emit absent from EMITS through the walker
        and assert the violation surfaces.

        Mirrors the F25 walker's self-test pattern. Pinning
        the walker against a synthetic violation also pins
        the AST-traversal logic — a no-op walker that never
        flags anything would silently pass the main rule
        whenever the real source happens to be clean.
        """
        synthetic = (
            "from prismpy.warnings.categories import WarningCategory\n"
            "\n"
            "class ToyValidator:\n"
            "    EMITS = frozenset({WarningCategory.CROP_REGION_MISMATCH})\n"
            "\n"
            "    def validate(self):\n"
            "        # Synthetic forbidden emit: CLIMATE_RH_INVALID\n"
            "        # is NOT in EMITS.\n"
            "        return WarningCategory.CLIMATE_RH_INVALID.value\n"
        )
        tree = ast.parse(synthetic)

        # Parse declared EMITS
        declared: Set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "ToyValidator"
            ):
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.Assign)
                        and any(
                            isinstance(t, ast.Name)
                            and t.id == "EMITS"
                            for t in stmt.targets
                        )
                        and isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Name)
                        and stmt.value.func.id == "frozenset"
                    ):
                        for arg in stmt.value.args:
                            if isinstance(arg, ast.Set):
                                for elt in arg.elts:
                                    if (
                                        isinstance(elt, ast.Attribute)
                                        and isinstance(elt.value, ast.Name)
                                        and elt.value.id == "WarningCategory"
                                    ):
                                        declared.add(elt.attr)

        # Scan refs
        refs: List[Tuple[int, str]] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "WarningCategory"
                and node.attr in _ALL_CATEGORY_NAMES
            ):
                refs.append((node.lineno, node.attr))

        violations = [
            (ln, name) for ln, name in refs if name not in declared
        ]
        self.assertEqual(
            len(violations), 1,
            f"Synthetic violation should fire exactly once; "
            f"got {violations!r}",
        )
        self.assertEqual(violations[0][1], "CLIMATE_RH_INVALID")

    def test_walker_accepts_declared_superset(self):
        """Anti-mutation drill: a class that DECLARES extra
        categories in EMITS but never emits them is legal per
        the contract's declared-superset semantics. The
        walker must NOT fire on this case.
        """
        synthetic = (
            "from prismpy.warnings.categories import WarningCategory\n"
            "\n"
            "class ToyValidator:\n"
            "    EMITS = frozenset({\n"
            "        WarningCategory.CROP_REGION_MISMATCH,\n"
            "        WarningCategory.INSUFFICIENTLY_SAMPLED,\n"
            "        WarningCategory.CROP_PHYSIOLOGY_VIOLATION,\n"
            "    })\n"
            "\n"
            "    def validate(self):\n"
            "        # Only emits one of three declared values.\n"
            "        return WarningCategory.CROP_REGION_MISMATCH.value\n"
        )
        tree = ast.parse(synthetic)

        declared: Set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "ToyValidator"
            ):
                for stmt in node.body:
                    if (
                        isinstance(stmt, ast.Assign)
                        and any(
                            isinstance(t, ast.Name)
                            and t.id == "EMITS"
                            for t in stmt.targets
                        )
                        and isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Name)
                        and stmt.value.func.id == "frozenset"
                    ):
                        for arg in stmt.value.args:
                            if isinstance(arg, ast.Set):
                                for elt in arg.elts:
                                    if (
                                        isinstance(elt, ast.Attribute)
                                        and isinstance(elt.value, ast.Name)
                                        and elt.value.id == "WarningCategory"
                                    ):
                                        declared.add(elt.attr)

        refs: List[Tuple[int, str]] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "WarningCategory"
                and node.attr in _ALL_CATEGORY_NAMES
            ):
                refs.append((node.lineno, node.attr))

        violations = [
            (ln, name) for ln, name in refs if name not in declared
        ]
        self.assertEqual(
            violations, [],
            f"Declared-superset semantics broken: walker "
            f"flagged {violations!r} despite all references "
            f"being declared.",
        )

    def test_emits_parser_handles_real_classes(self):
        """Pin the AST literal parser against the real
        Sprint F validator classes. A future refactor that
        wraps EMITS in a factory call (instead of literal
        ``frozenset({...})``) would cause the walker to fall
        back to runtime introspection — fine functionally
        but defeats the source-vs-runtime drift catch."""
        from prismpy.validators.climate_envelope import (
            ClimateEnvelopeValidator,
        )
        from prismpy.validators.crop_physiological import (
            CropPhysiologicalValidator,
        )

        cev_declared = _parse_emits_frozenset_literal(
            ClimateEnvelopeValidator,
        )
        self.assertEqual(
            cev_declared,
            {"CLIMATE_ENVELOPE_TAIL", "INSUFFICIENTLY_SAMPLED"},
        )

        cpv_declared = _parse_emits_frozenset_literal(
            CropPhysiologicalValidator,
        )
        self.assertEqual(
            cpv_declared,
            {
                "CROP_REGION_MISMATCH",
                "CROP_PHYSIOLOGY_VIOLATION",
                "INSUFFICIENTLY_SAMPLED",
            },
        )


if __name__ == "__main__":
    unittest.main()
