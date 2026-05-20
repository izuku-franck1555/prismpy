"""Structural pin for F-CX — producer-consumer soil vocab parity.

Closes the durable §27 producer-consumer vocabulary drift on the 5
soil variable keys: the override registry at
``prismpy/standards/override_value_shapes.py`` names soil keys with
the ``soil_*_*`` consumer vocab, while the writer at
``prismpy/cockpit/observed_values_writer.py`` emits the producer-side
``*_rootzone_mean`` / ``*_top30cm_mean`` / etc. ``resolve_observed_key``
is the single canonical hop between the two vocabularies; this pin
asserts that hop covers every soil entry, lands on a real producer-
emitted key, identity-passthroughs every climate key, and that the
canonical helper still lives exactly once at module level (§24 gold-
standard source-pin).

Per the F-CX cycle-1 + cycle-2 amendments (WA CA-1 + codex MED 1 +
WA LOW Note 1 + codex LOW 1).
"""
from __future__ import annotations

import ast
from pathlib import Path

from prismpy.cockpit.observed_values_writer import (
    OBSERVED_VALUES_CLIMATE_KEYS,
    OBSERVED_VALUES_SOIL_KEYS,
)
from prismpy.standards.override_value_shapes import (
    OVERRIDE_VALUE_SHAPES,
    _SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY,
    resolve_observed_key,
)


def _soil_override_keys() -> set[str]:
    # Substrate convention (cockpit_decisions:2061): soil vs climate
    # routing keys on ``shape.variable_key.startswith('soil_')`` — the
    # check_ids are ``value_range_soil_*`` but the variable_key prefix
    # is the canonical filter (WA CA-1 + codex MED 1).
    return {
        shape.variable_key
        for shape in OVERRIDE_VALUE_SHAPES.values()
        if shape.variable_key.startswith('soil_')
    }


def _climate_override_keys() -> set[str]:
    return {
        shape.variable_key
        for shape in OVERRIDE_VALUE_SHAPES.values()
        if not shape.variable_key.startswith('soil_')
    }


def test_soil_override_keys_map_completely() -> None:
    """Every soil entry in OVERRIDE_VALUE_SHAPES has a translation in
    the canonical mapping (no orphans). The explicit ``len == 5``
    closes codex MED 1's vacuous-empty-set anti-regression: a future
    registry shrink that emptied this set would otherwise pass set
    equality vacuously on two empty sides."""
    soil_keys = _soil_override_keys()
    assert len(soil_keys) == 5, (
        f'Expected exactly 5 soil variable_keys in OVERRIDE_VALUE_SHAPES; '
        f'got {len(soil_keys)} ({sorted(soil_keys)}). A registry shrink '
        'invalidates the F-CX vocabulary-parity contract.'
    )
    assert soil_keys == set(_SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY.keys()), (
        f'OVERRIDE_VALUE_SHAPES soil variable_keys {sorted(soil_keys)} != '
        f'_SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY keys '
        f'{sorted(_SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY.keys())}. Add the '
        'new soil entry to _SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY before '
        'shipping the new override shape, or the Override CURRENT field '
        'will be empty for that variable (the F-CX P0 vector).'
    )


def test_mapping_values_are_real_producer_keys() -> None:
    """Every translated value is a real producer-emitted observed-
    values-sidecar key. A typo or stale key would render the CURRENT
    field empty for that variable even after the mapping is added."""
    translated = set(_SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY.values())
    producer = set(OBSERVED_VALUES_SOIL_KEYS)
    assert translated <= producer, (
        f'_SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY values '
        f'{sorted(translated - producer)} are NOT emitted by the '
        f'producer writer (OBSERVED_VALUES_SOIL_KEYS='
        f'{sorted(producer)}). Typo or stale key — Override CURRENT '
        'would render empty.'
    )


def test_resolve_observed_key_translates_each_soil_key() -> None:
    """Direct call exercises every soil entry. Catches a regression
    in ``resolve_observed_key()`` itself (e.g. a refactor to a
    different lookup) independent of the mapping equality check."""
    assert resolve_observed_key('soil_ph') == 'ph_top30cm_mean'
    assert resolve_observed_key('soil_sand_pct') == 'sand_rootzone_mean'
    assert resolve_observed_key('soil_clay_pct') == 'clay_rootzone_mean'
    assert (
        resolve_observed_key('soil_organic_carbon_pct')
        == 'organic_carbon_top30cm_mean'
    )
    assert (
        resolve_observed_key('soil_bulk_density_g_cm3')
        == 'bulk_density_top30cm_mean'
    )


def test_climate_keys_identity_passthrough() -> None:
    """Identity-passthrough for the REAL registered climate keys
    (currently 4: tmax / tmin / precip / srad — codex LOW 1: no
    synthetic 'rainfall'). A future regression where the helper
    starts mis-translating climate keys fails here loudly."""
    climate_keys = _climate_override_keys()
    assert len(climate_keys) >= 1, (
        'Expected at least one climate variable_key in '
        'OVERRIDE_VALUE_SHAPES (currently 4); registry change?'
    )
    for key in climate_keys:
        assert resolve_observed_key(key) == key, (
            f'Climate key {key!r} MUST identity-passthrough; got '
            f'{resolve_observed_key(key)!r}. The helper started mis-'
            'translating climate keys.'
        )


def test_climate_override_keys_match_producer_byte_for_byte() -> None:
    """WA LOW Note 1 — symmetric coverage. Climate variable_keys MUST
    already match producer-emitted OBSERVED_VALUES_CLIMATE_KEYS byte-
    for-byte; identity-passthrough relies on this invariant. A future
    drift on either side trips this pin loudly + closes the §27
    producer-consumer symmetric coverage that pure soil-only parity
    left open."""
    climate_keys = _climate_override_keys()
    producer = set(OBSERVED_VALUES_CLIMATE_KEYS)
    assert climate_keys <= producer, (
        f'Climate variable_keys {sorted(climate_keys - producer)} are '
        f'NOT in OBSERVED_VALUES_CLIMATE_KEYS {sorted(producer)}. '
        'Either the producer writer changed key naming, or the '
        "registry adopted a new climate key the writer hasn't emitted "
        'yet. Identity-passthrough breaks for that key — add a mapping '
        'or align names at the source.'
    )


def test_canonical_source_ast_shape() -> None:
    """Gold-standard §24 AST source-pin — the canonical mapping +
    helper live exactly ONCE, at module level in
    ``override_value_shapes.py``. A refactor that moves either into a
    class / function / submodule (where a second source could appear
    and drift) fires this pin. Parses the module AST and asserts:

      * _SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY is a module-level
        annotated assignment.
      * resolve_observed_key is a module-level FunctionDef.
    """
    src = (
        Path(__file__).resolve().parents[2]
        / 'src' / 'prismpy' / 'standards' / 'override_value_shapes.py'
    )
    tree = ast.parse(src.read_text(encoding='utf-8'))
    module_level_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_level_names.add(node.target.id)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            module_level_names.add(node.targets[0].id)
    assert '_SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY' in module_level_names, (
        '_SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY MUST be a module-level '
        'constant in override_value_shapes.py (the canonical source). '
        'Moving it to a function / class / submodule re-opens the §24 '
        'canonical-source drift surface.'
    )
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert 'resolve_observed_key' in function_names, (
        'resolve_observed_key() MUST be defined at module level in '
        'override_value_shapes.py — the single hop between consumer and '
        'producer vocab. Moving it elsewhere re-opens the §24 canonical-'
        'source drift surface.'
    )
