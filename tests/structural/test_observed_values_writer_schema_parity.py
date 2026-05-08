"""Sprint E.2 AC-E2-28 + Codex Gate A HIGH 5 + CMS-DOMAIN-CA-1
+ Builder Sub-CA #3 — :mod:`prismpy.cockpit.observed_values_writer`
schema-parity pin.

The cockpit's IDW orchestrator (consumed at prismweb's Phase 2)
reads per-cell aggregate observed values from
``cockpit_observed_values.json``. The producer
(:func:`prismpy.cockpit.observed_values_writer.write_observed_values_json`)
+ the consumer (cockpit-side reader) MUST agree on the 17-key
Hybrid A schema. Per durable §27 cross-stage two-vocabulary
substrate-drift the parity is enforced structurally so a
producer-side rename can't silently re-break the consumer.

Pins:

1. **Key-count invariant** — 7 climate + 10 soil = 17 total.
2. **Method-string completeness** — every aggregate key has a
   :data:`AGGREGATION_METHOD` entry.
3. **Units completeness** — every aggregate key has a
   :data:`AGGREGATION_UNITS` entry.
4. **No drift between key tuples + method/units dicts** — the
   set of keys equals the dict's key set in both directions.
5. **Soil substrate sentinel parity** — both
   :data:`SOIL_AGGREGATION_IN_MEMORY` +
   :data:`SOIL_AGGREGATION_EGHR_SKIP` are exercised by the
   writer's PYTHIA-skip path (verified via empirical call).
6. **Reuse of `SoilProfile.get_weighted_average`** per durable
   §24 + Builder Sub-CA #3 — the writer source must IMPORT the
   helper rather than re-implement depth-weighting math
   inline. AST-walks the writer source.
"""
from __future__ import annotations

import ast
from pathlib import Path

from prismpy.cockpit.observed_values_writer import (
    AGGREGATION_METHOD,
    AGGREGATION_UNITS,
    OBSERVED_VALUES_CLIMATE_KEYS,
    OBSERVED_VALUES_SOIL_KEYS,
    SOIL_AGGREGATION_EGHR_SKIP,
    SOIL_AGGREGATION_IN_MEMORY,
    compute_soil_aggregates,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WRITER_FILE = (
    REPO_ROOT / "src" / "prismpy" / "cockpit" / "observed_values_writer.py"
)


def test_key_count_invariant_17_total() -> None:
    """7 climate + 10 soil = 17 (Hybrid A schema per CMS §1 +
    CMS-DOMAIN-CA-1 absorption: OC counted as chemistry, soil
    count 11 → 10)."""
    assert len(OBSERVED_VALUES_CLIMATE_KEYS) == 7, (
        f"Climate key count drifted from 7 → "
        f"{len(OBSERVED_VALUES_CLIMATE_KEYS)}. CMS §1 specifies "
        f"7 climate aggregates; check the contract before adding."
    )
    assert len(OBSERVED_VALUES_SOIL_KEYS) == 10, (
        f"Soil key count drifted from 10 → "
        f"{len(OBSERVED_VALUES_SOIL_KEYS)}. CMS-DOMAIN-CA-1 "
        f"locked the count at 10 (3 texture + 3 chemistry "
        f"including OC + 3 hydraulic + 1 scalar depth)."
    )


def test_aggregation_method_covers_every_aggregate_key() -> None:
    """Every aggregate key has a method-string in
    :data:`AGGREGATION_METHOD`. A consumer rendering the methods
    text would silently elide an unknown key; the pin catches
    drift at CI time."""
    all_keys = set(OBSERVED_VALUES_CLIMATE_KEYS) | set(
        OBSERVED_VALUES_SOIL_KEYS
    )
    method_keys = set(AGGREGATION_METHOD.keys())
    missing = all_keys - method_keys
    extra = method_keys - all_keys
    assert not missing, (
        f"AGGREGATION_METHOD missing entries for: {sorted(missing)}"
    )
    assert not extra, (
        f"AGGREGATION_METHOD has phantom entries (no matching "
        f"aggregate key): {sorted(extra)}"
    )


def test_aggregation_units_covers_every_aggregate_key() -> None:
    """Every aggregate key has a units string in
    :data:`AGGREGATION_UNITS`. Drift detection mirrors the
    method-string pin above."""
    all_keys = set(OBSERVED_VALUES_CLIMATE_KEYS) | set(
        OBSERVED_VALUES_SOIL_KEYS
    )
    units_keys = set(AGGREGATION_UNITS.keys())
    missing = all_keys - units_keys
    extra = units_keys - all_keys
    assert not missing, (
        f"AGGREGATION_UNITS missing entries for: {sorted(missing)}"
    )
    assert not extra, (
        f"AGGREGATION_UNITS has phantom entries (no matching "
        f"aggregate key): {sorted(extra)}"
    )


def test_soil_substrate_sentinels_distinguishable() -> None:
    """The two canonical soil-substrate sentinels are
    distinguishable strings. A PYTHIA-skip cell must NOT collide
    with an in-memory cell; consumer-side filtering depends on
    the ``in {"in_memory_layers", "eghr_no_in_memory_layers"}``
    enum."""
    assert SOIL_AGGREGATION_IN_MEMORY != SOIL_AGGREGATION_EGHR_SKIP
    assert SOIL_AGGREGATION_IN_MEMORY == "in_memory_layers"
    assert SOIL_AGGREGATION_EGHR_SKIP == "eghr_no_in_memory_layers"


def test_pythia_skip_path_emits_eghr_sentinel() -> None:
    """Empirical pin — :func:`compute_soil_aggregates` returns
    the EGHR_SKIP sentinel + null aggregates for None or
    empty-layered profiles. CMS §9.6 Concern A."""
    soil_aggregates_none, substrate_none, n_layers_none = (
        compute_soil_aggregates(None)
    )
    assert substrate_none == SOIL_AGGREGATION_EGHR_SKIP
    assert n_layers_none == 0
    for key in OBSERVED_VALUES_SOIL_KEYS:
        assert soil_aggregates_none[key] is None, (
            f"PYTHIA-skip path leaked a non-null value for {key}; "
            f"cockpit IDW would silently treat the eGHR cell as "
            f"having soil data per CMS §9.6 Concern A."
        )


def test_writer_imports_get_weighted_average_canonical_helper() -> None:
    """Per durable §24 canonical-source-or-pin + Builder
    Sub-CA #3: the writer DELEGATES depth-weighting math to
    :meth:`SoilProfile.get_weighted_average` rather than
    re-implementing it inline. The pin walks the writer source
    + asserts ``profile.get_weighted_average(...)`` is the
    invocation pattern (NOT a local
    ``def _depth_weighted_mean(...)``).

    Drift detection: a future commit that adds a parallel
    helper inside :mod:`observed_values_writer` would create a
    second canonical source for depth-weighted-mean math —
    catches the same class of duplicate-canonical-source the
    F25 pin guards against for WarningCategory enum values.
    """
    text = WRITER_FILE.read_text(encoding="utf-8")
    tree = ast.parse(text)

    # Look for ``profile.get_weighted_average(...)`` call sites.
    found_helper_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "get_weighted_average":
            found_helper_calls += 1

    assert found_helper_calls >= 2, (
        f"observed_values_writer.py invokes "
        f"SoilProfile.get_weighted_average only "
        f"{found_helper_calls} time(s); expected >= 2 (one for "
        f"rootzone-mean, one for top-30cm-mean). Per Builder "
        f"Sub-CA #3 + durable §24, the writer MUST delegate to "
        f"the canonical helper at ``models/soil.py:198`` "
        f"rather than reinvent depth-weighting math inline."
    )

    # Defensive — also assert no local function in the writer
    # is named ``_depth_weighted_mean`` / ``_weighted_average``.
    forbidden_helper_names = {
        "_depth_weighted_mean",
        "_weighted_average",
        "_compute_weighted_mean",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert node.name not in forbidden_helper_names, (
                f"observed_values_writer.py defines forbidden "
                f"local helper {node.name!r} at line "
                f"{node.lineno}; reuse "
                f"SoilProfile.get_weighted_average instead "
                f"per Builder Sub-CA #3."
            )
