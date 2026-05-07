"""Sprint S Gate-B-FIX — pipeline executor PYTHIA canonical-default pin.

The PYTHIA orchestrator at
``prismpy/src/prismpy/pipeline/executor.py:_get_translator(Platform.PYTHIA)``
constructs ``PythiaTranslator(config=, output_dir=, provenance=)``
WITHOUT passing the ``prefer_canonical_substrate`` keyword. The
behaviour relies entirely on the constructor's default value
``True``.

This test pins that contract per durable §24 canonical-source-or-pin:
the orchestrator's construction site is the canonical entry point
for every prismweb-driven PYTHIA package run, so a regression
where the kwarg gets explicitly set to ``False`` (or where the
default flips) silently disables the canonical substrate path
for every real-data wizard run — exactly the failure shape that
the b5fb6538 evaluator Gate B surfaced (durable §25 user-snippet
canonical Gate B + `feedback_post_fix_vs_baseline_diff.md`).

Two structural assertions:

1. **Class-level default** — the constructor's
   ``prefer_canonical_substrate`` parameter has the default value
   ``True``. A regression flipping the default would silently
   re-enable the legacy path for every default-construction caller
   (the orchestrator + every test fixture).

2. **Orchestrator AST walk** — scan ``pipeline/executor.py`` for
   any explicit ``prefer_canonical_substrate=False`` kwargs in
   ``PythiaTranslator(...)`` calls. None should exist.

The pin is structural (parses the source AST) rather than a
behavioural integration test because:

- The orchestrator's ``_get_translator`` is a method that constructs
  the translator via a registry-mapped class. A behavioural test
  would need to mock the entire pipeline's registered platform set,
  which couples the test to changes in unrelated translator
  registrations. AST-level assertion has no such coupling.
- The pin needs to fire on REVIEW-time inspection of the executor
  diff — code review catches this earlier than test runtime.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from prismpy.translators.pythia.translator import PythiaTranslator


def test_pythia_translator_default_prefer_canonical_substrate_is_true() -> None:
    """The class signature default is ``True``.

    The orchestrator at ``pipeline/executor.py`` and every other
    default-construction caller relies on this. A flip to ``False``
    silently re-enables the legacy bundled-eGHR path, which in turn
    causes the F-AK-class country-mismatch bug (CM.SOL missing,
    NG.SOL substituted for a Cameroon project) — the exact bug
    Sprint S exists to fix.
    """
    sig = inspect.signature(PythiaTranslator.__init__)
    param = sig.parameters.get("prefer_canonical_substrate")
    assert param is not None, (
        "PythiaTranslator.__init__ must expose 'prefer_canonical_substrate' "
        "as a kwarg-only parameter; the orchestrator + every test fixture "
        "depends on the kwarg being available for explicit-True / "
        "explicit-False / default-True dispatch."
    )
    assert param.default is True, (
        "PythiaTranslator.__init__'s 'prefer_canonical_substrate' default "
        f"must be True (got {param.default!r}). The orchestrator at "
        "pipeline/executor.py:309-313 constructs the translator without "
        "passing the kwarg, so a default flip silently disables the "
        "canonical substrate path for every real-data wizard run."
    )


def _executor_path() -> Path:
    """Resolve the on-disk path to ``pipeline/executor.py``."""
    import prismpy.pipeline.executor as executor_mod
    return Path(executor_mod.__file__)


def test_pipeline_executor_does_not_force_legacy_path_for_pythia() -> None:
    """The orchestrator's PYTHIA construction site does not pass ``False``.

    Walks the AST of ``pipeline/executor.py`` and asserts that every
    ``PythiaTranslator(...)`` call site either omits the
    ``prefer_canonical_substrate`` kwarg (relying on the default
    ``True``) or passes ``=True`` explicitly. No call site should
    pass ``=False``.

    This is the structural enforcement leg of durable §24
    canonical-source-or-pin: the orchestrator IS the canonical entry
    point for prismweb-driven PYTHIA package runs; its construction
    site must consistently exercise the canonical substrate path.
    """
    src_path = _executor_path()
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    # Locate every PythiaTranslator(...) call in the executor.
    pythia_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Match both `PythiaTranslator(...)` and
            # `prismpy.translators.PythiaTranslator(...)` shapes.
            if isinstance(func, ast.Name) and func.id == "PythiaTranslator":
                pythia_calls.append(node)
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "PythiaTranslator"
            ):
                pythia_calls.append(node)

    # The orchestrator may not directly call PythiaTranslator(...) at all
    # — the construction is via a registry-mapped class
    # (translator_class(config=..., output_dir=..., provenance=...)),
    # which the AST walker can't statically resolve. In that case the
    # class-level default test above carries the contract; this test
    # is a defense-in-depth structural pin for any direct call site
    # that may be added in the future.
    if not pythia_calls:
        pytest.skip(
            "pipeline/executor.py has no direct PythiaTranslator(...) call "
            "sites (the construction goes via a registry-mapped class). "
            "The class-default test above carries the canonical-substrate "
            "default contract; this AST pin defends future direct call sites."
        )

    for call in pythia_calls:
        for kw in call.keywords:
            if kw.arg != "prefer_canonical_substrate":
                continue
            # Found an explicit kwarg — assert the value is not False.
            value_node = kw.value
            assert not (
                isinstance(value_node, ast.Constant)
                and value_node.value is False
            ), (
                "pipeline/executor.py PythiaTranslator(...) call at "
                f"line {call.lineno} passes prefer_canonical_substrate=False; "
                "this silently disables the canonical substrate path on "
                "every real-data wizard run and reproduces the F-AK-class "
                "country-mismatch bug (durable §24 + §25 + Sprint S "
                "contract). Remove the kwarg to rely on the True default, "
                "or change to =True for explicit opt-in."
            )


def test_pipeline_executor_translator_registry_includes_pythia() -> None:
    """The orchestrator's translator registry maps Platform.PYTHIA.

    Defense-in-depth pin: a refactor that drops PYTHIA from the
    registry would silently route every PYTHIA package run to a
    "no translator available" warning + skip path, masking the
    canonical-substrate question entirely. The b5fb6538 evaluator
    surfaced absence of the canonical decision in provenance.json;
    a no-translator path would surface absence of the entire
    PYTHIA package. Both are equally bad for the user.
    """
    src_path = _executor_path()
    src_text = src_path.read_text(encoding="utf-8")
    # The orchestrator's translator_map literal contains
    # ``Platform.PYTHIA: PythiaTranslator`` per the
    # ``_get_translator`` method. A textual grep is sufficient
    # for this defense; the import resolution check upstream
    # (`from prismpy.translators import PythiaTranslator`) is
    # covered by the "translator imports cleanly" pin elsewhere.
    assert "Platform.PYTHIA: PythiaTranslator" in src_text, (
        "pipeline/executor.py translator_map must include "
        "'Platform.PYTHIA: PythiaTranslator' so the orchestrator "
        "routes PYTHIA package generation through the Sprint S "
        "canonical-substrate-aware translator."
    )


def test_pipeline_get_translator_pythia_uses_canonical_default(tmp_path) -> None:
    """Behavioral pin (codex P3 round 1) — the registry-mapped construction
    path produces a translator with ``prefer_canonical_substrate=True``.

    Stronger than the AST walker above: this test instantiates a real
    :class:`prismpy.pipeline.executor.TranslationPipeline` and asks
    its ``_get_translator`` factory for the PYTHIA translator. The
    factory uses the registry-mapped class
    (``translator_map[Platform.PYTHIA] = PythiaTranslator``) and
    constructs it with ``(config=, output_dir=, provenance=)`` —
    identical to the production wizard-driven path.

    Asserts both:
    1. The returned object is a :class:`PythiaTranslator` (registry
       routes PYTHIA correctly).
    2. ``translator.prefer_canonical_substrate is True`` (the
       canonical default fires through the production construction
       site).

    Catches both regression classes in one assertion:
    - Class-level default flip (``prefer_canonical_substrate: bool = False``)
      → existing default-True unit test fails first; this test fails
      second as a backstop.
    - Orchestrator-explicit-False slip (someone refactors
      ``pipeline/executor.py:309-313`` to pass
      ``prefer_canonical_substrate=False``) → only this test catches
      it; the default-True unit test would still pass because the
      class signature is unchanged.

    Per durable §24 canonical-source-or-pin: the orchestrator IS
    the canonical entry point for prismweb-driven PYTHIA package
    runs, so its construction site MUST consistently exercise the
    canonical substrate path. This test catches future regressions
    where someone explicitly disables the canonical mode in the
    orchestrator without realizing the downstream impact.
    """
    from prismpy.config.schema import (
        BoundaryConfig,
        BoundarySource,
        CropCalendarConfig,
        CropConfig,
        ManualBoundsConfig,
        OutputConfig,
        Platform,
        ProjectConfig,
        ProjectInfo,
        RegionConfig,
        TemporalConfig,
    )
    from prismpy.pipeline.executor import TranslationPipeline
    from prismpy.provenance.tracker import ProvenanceTracker

    cfg = ProjectConfig(
        project=ProjectInfo(
            name="codex_p3_pythia_canonical_default_pin",
            description=(
                "Sprint S Gate-B-FIX behavioral pin per codex round 1 P3 — "
                "registry-mapped TranslationPipeline._get_translator(PYTHIA) "
                "must produce a translator with the canonical-substrate "
                "default firing."
            ),
        ),
        region=RegionConfig(
            name="Bénoué",
            country="Cameroon",
            country_iso3="CMR",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=13.5, miny=8.0, maxx=14.5, maxy=9.0,
                ),
            ),
        ),
        crop=CropConfig(
            name="Sorghum",
            name_short="sgh",
            variety="Medium-duration",
            calendar=CropCalendarConfig(
                planting_doy=166, maturity_doy=285,
            ),
        ),
        temporal=TemporalConfig(
            start_year=2015, end_year=2015, spinup_years=0,
        ),
        targets=[Platform.PYTHIA],
        output=OutputConfig(
            base_dir=str(tmp_path), structure="by_platform",
        ),
    )
    pipeline = TranslationPipeline(
        cfg,
        provenance=ProvenanceTracker(
            enabled=False, project_name="codex_p3_pythia_canonical_default_pin",
        ),
    )

    translator = pipeline._get_translator(Platform.PYTHIA)
    assert translator is not None, (
        "pipeline/_get_translator(Platform.PYTHIA) must return a translator "
        "instance (registry-mapped); got None — the registry refactor "
        "broke the PYTHIA mapping."
    )
    assert isinstance(translator, PythiaTranslator), (
        "pipeline/_get_translator(Platform.PYTHIA) must return a "
        f"PythiaTranslator instance; got {type(translator).__name__}. "
        "The registry mapping was changed to a non-PYTHIA class."
    )
    assert translator.prefer_canonical_substrate is True, (
        "pipeline/_get_translator(Platform.PYTHIA) constructed a translator "
        f"with prefer_canonical_substrate={translator.prefer_canonical_substrate!r}; "
        "the orchestrator's production construction site must exercise the "
        "canonical-substrate path by default. Either the class default "
        "flipped (covered by the default-True unit test) OR the "
        "orchestrator explicitly disabled canonical mode (covered ONLY "
        "by this test). Per durable §24 canonical-source-or-pin: the "
        "orchestrator IS the canonical entry point and MUST stay on the "
        "canonical-substrate path."
    )
