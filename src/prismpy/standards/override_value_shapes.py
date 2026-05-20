"""Canonical per-check_id value-shape registry for cockpit Override.

Sprint E.3 AC-E3-3 + Draft 2 CMS CA-2 absorption. The registry maps
each value-replaceable ``check_id`` to a 5-tuple
``(variable_key, unit, numeric_type, override_min, override_max)`` that
two surfaces consume:

* The cockpit Override Edit form (Phase 2) reads ``unit`` for the
  unit-selector default + ``override_min`` / ``override_max`` for
  client-side range validation + ``numeric_type`` to pick text-field
  vs numeric-field rendering.
* The translator-side ``apply_override`` consumer (Phase 1 + AC-E3-9)
  reads ``variable_key`` to look up the override value in the sidecar
  by canonical key + ``unit`` to assert the persisted value matches
  the expected unit at write time.

**Physical-plausibility bounds vs validator thresholds**: the
``override_min`` / ``override_max`` here are STRICTLY WIDER than the
validator's ``CLIMATE_RANGES`` / ``SOIL_RANGES`` thresholds at
``prismpy/validators/scientific.py:61-77``. The validator's job is to
flag a cell as "outside the typical range we expect for credible
runs"; the override's job is to let a persona document a credible
value the validator flagged. So Override accepts the validator's
range plus a documented-anomaly margin (e.g., a Death-Valley extreme
on tmax) but rejects values outside physical plausibility — the
``override_max`` for tmax is 70°C (Earth-surface upper bound; Death
Valley record-day ≈57°C plus margin) not infinity.

Per durable lesson §24 canonical-source-or-pin: the bounds live here
once. Two structural pins close the drift class:

1. ``tests/structural/test_override_value_shape_per_check_id.py``
   AST-walks all consumers and asserts they reference
   ``OVERRIDE_VALUE_SHAPES`` directly rather than restating the
   bounds inline.

2. ``tests/structural/test_override_bounds_wider_than_validator.py``
   asserts the Override-bound-vs-validator-bound invariant:
   ``override_min <= validator_min AND override_max >= validator_max``
   for every key with a corresponding validator threshold. A future
   hardening of a validator threshold that drifted past the override
   bound would fire this pin loud rather than silently rejecting
   personas' documented anomalies.

**Coverage** (AC-E3-3 sub-1): the registry covers every ``check_id``
that admits value-replacement. Non-value-replacement check_ids
(``post_translate_*`` per-platform aggregators, ``format_compliance``
+ ``spatial_temporal_coverage`` axis-level, ``coverage_climate_cells``
+ ``coverage_soil_cells`` IDW-routed, ``soil_completeness_<platform>``
multi-field aggregators) are intentionally absent — the cockpit's
per-affordance routing surfaces Skip / Acknowledge / Interpolate for
those rather than Override. The structural pin at
``test_override_value_shape_per_check_id.py`` asserts the registry
keys are a SUBSET of ``enumerate_emitted_check_ids()``.

Future variable additions (e.g., ``value_range_wind`` or
``value_range_rh`` if a future sprint enables Override on those)
extend the registry here with bounds the crop-modeling-specialist
signs off on; until then, those check_ids stay non-Override.
"""

from __future__ import annotations

from typing import Final, Mapping, NamedTuple, Optional


class OverrideValueShape(NamedTuple):
    """Per-check_id value shape consumed by form + translator.

    ``variable_key`` — canonical key the sidecar entry uses (e.g.,
    ``"tmax_growing_season_mean"``); the translator reads the override
    value via ``sidecar.get(cell_id, variable_key)`` rather than
    re-deriving the key from the check_id. Mirrors
    ``OBSERVED_VALUES_CLIMATE_KEYS`` precedent at
    ``prismpy/cockpit/observed_values_writer.py``.

    ``unit`` — canonical unit string. Empty for unitless quantities
    (pH, bulk density coefficient). The form's unit selector defaults
    to this value; if the persona persists a different unit the form
    converts on submit before sidecar emission so all stored values
    are in canonical units (per durable §27 two-vocabulary substrate-
    drift discipline — value-and-unit pinned together at the source).

    ``numeric_type`` — ``float`` for continuous quantities,
    ``int`` for counts, ``str`` for categorical (currently no
    categorical Override targets ship; reserved for V3+).

    ``override_min`` / ``override_max`` — inclusive physical-
    plausibility bounds. ``None`` only on categorical fields where
    "skip-bounds" is the documented contract (currently no entries).
    """

    variable_key: str
    unit: str
    numeric_type: type
    override_min: Optional[float]
    override_max: Optional[float]


# ── Canonical registry — per check_id physical-plausibility bounds ───


# Per-check_id 5-tuple bounds. Each entry's ``(override_min,
# override_max)`` MUST be strictly wider than the corresponding
# ``CLIMATE_RANGES`` / ``SOIL_RANGES`` entry at
# ``prismpy/validators/scientific.py:61-77``; the structural pin at
# ``test_override_bounds_wider_than_validator.py`` enforces this
# invariant per AC-E3-3 sub-criterion 4.
#
# Domain rationale per crop-modeling-specialist Draft 2 CA-2:
#
# * tmax (-60, 70) °C — Earth-surface absolute max ≈57°C (Death
#   Valley 1913 official record); -60°C lower bound covers Antarctic
#   plateau extremes that no agricultural cell would actually hit but
#   leaves Override a documented-anomaly margin without sliding into
#   physically-impossible territory (e.g., -100°C).
# * tmin (-70, 50) °C — Antarctic Vostok station record -89.2°C (we
#   bound at -70 since no agricultural region reaches that extreme;
#   50°C upper accommodates documented hot-night extremes).
# * precip (0, 1000) mm/day — Cherrapunji record-day rainfall
#   ≈970 mm; 0 lower bound is physical (rain can't be negative).
# * srad (0, 50) MJ/m²/d — top-of-atmosphere ≈37 MJ/m²/d at equinox;
#   50 upper bound leaves slack for Holuhraun-style aerosol-amplified
#   surface measurements without going past physics.
# * Soil sand / clay (0, 100) % — physical bound (a percentage cannot
#   exceed 100%); identical to the validator's range (which is also
#   physical) so the override-strictly-wider invariant is satisfied
#   trivially (equal counts as ≥ at both endpoints; the structural pin
#   at ``test_override_bounds_wider_than_validator.py`` accepts
#   equality on physical-bound endpoints).
# * Soil organic carbon (0, 50) % — peat soils max ≈40% per IPCC
#   Tier-1 lookup; 50% upper accommodates well-documented histosol
#   extremes.
# * Soil pH (1, 12) — strong-acid mine drainage as low as ~1; alkali
#   soils up to ~11.5; 12 upper bound matches lab-titration scale.
# * Soil bulk density (0.1, 2.7) g/cm³ — peat ≈0.1 (dried); bedrock
#   ≈2.7 (granite/basalt limit); covers full pedological range.
OVERRIDE_VALUE_SHAPES: Final[dict[str, OverrideValueShape]] = {
    "value_range_tmax": OverrideValueShape(
        variable_key="tmax_growing_season_mean",
        unit="C",
        numeric_type=float,
        override_min=-60.0,
        override_max=70.0,
    ),
    "value_range_tmin": OverrideValueShape(
        variable_key="tmin_growing_season_mean",
        unit="C",
        numeric_type=float,
        override_min=-70.0,
        override_max=50.0,
    ),
    "value_range_precip": OverrideValueShape(
        variable_key="precip_growing_season_total",
        unit="mm/day",
        numeric_type=float,
        override_min=0.0,
        override_max=1000.0,
    ),
    "value_range_srad": OverrideValueShape(
        variable_key="srad_growing_season_mean",
        unit="MJ/m^2/d",
        numeric_type=float,
        override_min=0.0,
        override_max=50.0,
    ),
    "value_range_soil_sand": OverrideValueShape(
        variable_key="soil_sand_pct",
        unit="%",
        numeric_type=float,
        override_min=0.0,
        override_max=100.0,
    ),
    "value_range_soil_clay": OverrideValueShape(
        variable_key="soil_clay_pct",
        unit="%",
        numeric_type=float,
        override_min=0.0,
        override_max=100.0,
    ),
    "value_range_soil_organic_carbon": OverrideValueShape(
        variable_key="soil_organic_carbon_pct",
        unit="%",
        numeric_type=float,
        override_min=0.0,
        override_max=50.0,
    ),
    "value_range_soil_ph": OverrideValueShape(
        variable_key="soil_ph",
        unit="",
        numeric_type=float,
        override_min=1.0,
        override_max=12.0,
    ),
    "value_range_soil_bulk_density": OverrideValueShape(
        variable_key="soil_bulk_density_g_cm3",
        unit="g/cm^3",
        numeric_type=float,
        override_min=0.1,
        override_max=2.7,
    ),
}


# ── Canonical translation: consumer override vocab → producer vocab ──

# Per-substrate canonical translation from the consumer override
# registry's ``variable_key`` vocab to the producer observed-values
# sidecar's emitted-key vocab. The producer writer at
# ``prismpy/cockpit/observed_values_writer.py`` emits the right-hand-
# side keys (e.g. ``ph_top30cm_mean``); the override registry above
# names the same physical quantities with the left-hand-side
# (e.g. ``soil_ph``). Consumer-side translation through
# ``resolve_observed_key()`` lets the cockpit decisions service read
# the originating producer-emitted value when rendering the Override
# panel's CURRENT field, closing the producer-consumer vocabulary
# drift on the 5 soil keys. Climate keys already match byte-for-byte
# at the producer / consumer registries (4 keys identity-passthrough);
# the structural pin at
# ``tests/structural/test_soil_override_vocab_parity.py`` freezes
# that invariant and asserts every soil entry has a mapping that
# points to a real producer-emitted key.
_SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY: Final[Mapping[str, str]] = {
    'soil_ph':                  'ph_top30cm_mean',
    'soil_sand_pct':            'sand_rootzone_mean',
    'soil_clay_pct':            'clay_rootzone_mean',
    'soil_organic_carbon_pct':  'organic_carbon_top30cm_mean',
    'soil_bulk_density_g_cm3':  'bulk_density_top30cm_mean',
}


def resolve_observed_key(override_key: str) -> str:
    """Translate a consumer override-registry ``variable_key`` to the
    producer observed-values sidecar's emitted-key vocab.

    Soil keys translate via :data:`_SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY`;
    every other key (currently the 4 climate keys, which already match
    producer-emitted keys byte-for-byte) passes through unchanged.
    Future per-substrate additions extend the mapping; the structural
    pin at ``tests/structural/test_soil_override_vocab_parity.py``
    asserts the mapping stays complete + climate-symmetric so a
    future drift on either side fails fast.
    """
    return _SOIL_OVERRIDE_KEY_TO_OBSERVED_KEY.get(override_key, override_key)


def get_override_value_shape(check_id: str) -> Optional[OverrideValueShape]:
    """Return the value shape for ``check_id`` or ``None`` if Override
    is not defined for that check.

    The cockpit Override Edit form calls this helper when the persona
    selects Override on a flagged cell; ``None`` means the form should
    NOT surface a value-replacement input — the persona's affordance
    is Skip or Acknowledge instead. The translator-side
    ``apply_override`` helper does NOT call this (it reads from the
    sidecar by ``variable_key`` directly); the helper exists for the
    form's per-check_id branching.
    """
    return OVERRIDE_VALUE_SHAPES.get(check_id)


__all__ = [
    "OVERRIDE_VALUE_SHAPES",
    "OverrideValueShape",
    "get_override_value_shape",
    "resolve_observed_key",
]
