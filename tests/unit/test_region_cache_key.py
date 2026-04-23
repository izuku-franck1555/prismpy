"""V2-22b/P.2 AC-AUDIT-8 — narrow entry points for region identity.

The polymorphic `region_cache_key(region)` was replaced with two
narrow entry points so each caller is explicit about which shape
it's working with:

- `region_cache_key_from_region(region)` — POST-RESOLUTION. Takes a
  `Region` dataclass (or any object with `.boundary_source`,
  `.bounds`, `.name` attributes). Used by every prismpy caller.
- `region_cache_key_from_config(config)` — PRE-RESOLUTION. Takes a
  plain `Mapping` (prismweb's persisted JSON) or a Pydantic-like
  attribute-access object with `.boundary.source`,
  `.boundary.manual_bounds`, `.name`.

Both entry points are strict: malformed inputs raise `ValueError`
rather than silently degrading to a name-keyed identity.
"""

from __future__ import annotations

import types
import unittest

from prismpy.utils.sanitization import (
    normalize_region_name,
    region_cache_key_from_config,
    region_cache_key_from_region,
)


# =============================================================================
# region_cache_key_from_region — POST-RESOLUTION entry point
# =============================================================================


def _region_obj(
    *, name='Unnamed study area', boundary_source='manual',
    miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
):
    """Region-dataclass-shaped SimpleNamespace with boundary_source
    + bounds. Used in `from_region` tests."""
    bounds = types.SimpleNamespace(miny=miny, maxy=maxy, minx=minx, maxx=maxx)
    return types.SimpleNamespace(
        name=name, boundary_source=boundary_source, bounds=bounds,
    )


class TestFromRegionManual(unittest.TestCase):
    """Manual-source regions get bbox-keyed identity so unnamed
    manual projects sharing the `"Unnamed study area"` default
    display name each cache separately."""

    def test_same_bbox_same_key(self):
        r1 = _region_obj(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        r2 = _region_obj(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        self.assertEqual(
            region_cache_key_from_region(r1),
            region_cache_key_from_region(r2),
        )

    def test_different_bbox_different_key(self):
        r_mali = _region_obj(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        r_ethiopia = _region_obj(miny=6.0, maxy=9.0, minx=37.0, maxx=40.0)
        self.assertNotEqual(
            region_cache_key_from_region(r_mali),
            region_cache_key_from_region(r_ethiopia),
        )

    def test_key_is_bbox_prefixed_at_6_decimal(self):
        r = _region_obj(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        key = region_cache_key_from_region(r)
        self.assertTrue(key.startswith('manual_'))
        self.assertIn('12.000000', key)
        self.assertIn('14.000000', key)
        self.assertIn('-5.000000', key)
        self.assertIn('-3.000000', key)

    def test_nearby_boxes_produce_different_keys(self):
        """Sub-6-decimal differences distinguish bboxes — an earlier
        4-decimal format collapsed ~11 m differences onto the same
        key."""
        r1 = _region_obj(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        r2 = _region_obj(
            miny=12.00001, maxy=14.00001, minx=-5.00001, maxx=-3.00001,
        )
        self.assertNotEqual(
            region_cache_key_from_region(r1),
            region_cache_key_from_region(r2),
        )

    def test_negative_zero_canonicalizes(self):
        """A bbox edge on the prime meridian / equator can surface
        `-0.0` from JS / GeoJSON round-trips; `-0.0` and `0.0` must
        produce the same key."""
        r_nz = _region_obj(minx=-0.0, miny=0.0, maxx=1.0, maxy=1.0)
        r_pz = _region_obj(minx=0.0, miny=-0.0, maxx=1.0, maxy=1.0)
        self.assertEqual(
            region_cache_key_from_region(r_nz),
            region_cache_key_from_region(r_pz),
        )
        self.assertNotIn('-0', region_cache_key_from_region(r_nz))


class TestFromRegionGadm(unittest.TestCase):
    """GADM / shapefile / unknown-source regions use the normalized
    display name — admin names are already unique identifiers."""

    def test_gadm_region_name_keyed(self):
        r = _region_obj(name='Boulemane', boundary_source='gadm')
        self.assertEqual(
            region_cache_key_from_region(r),
            normalize_region_name('Boulemane'),
        )

    def test_missing_boundary_source_defaults_to_name_key(self):
        """Legacy Region constructor path with `boundary_source=None`
        falls to the name-key branch."""
        r = _region_obj(name='Legacy Region', boundary_source=None)
        self.assertEqual(
            region_cache_key_from_region(r),
            normalize_region_name('Legacy Region'),
        )

    def test_two_same_admin_regions_same_key(self):
        r1 = _region_obj(name='Maradi', boundary_source='gadm')
        r2 = _region_obj(name='Maradi', boundary_source='gadm')
        self.assertEqual(
            region_cache_key_from_region(r1),
            region_cache_key_from_region(r2),
        )


class TestFromRegionStrict(unittest.TestCase):
    """Strict guards — malformed post-resolution inputs raise
    `ValueError` instead of silently producing a degenerate key."""

    def test_manual_without_bounds_raises(self):
        r = types.SimpleNamespace(
            name='Malformed', boundary_source='manual', bounds=None,
        )
        with self.assertRaisesRegex(ValueError, 'requires .bounds'):
            region_cache_key_from_region(r)

    def test_manual_with_non_numeric_bounds_raises(self):
        """A bounds object with non-numeric coords fails fast, not
        silently falling through to the name-key."""
        bounds = types.SimpleNamespace(
            miny='not-a-number', maxy=1.0, minx=0.0, maxx=1.0,
        )
        r = types.SimpleNamespace(
            name='Bad region', boundary_source='manual', bounds=bounds,
        )
        with self.assertRaisesRegex(ValueError, 'numeric'):
            region_cache_key_from_region(r)

    def test_manual_bounds_missing_attribute_raises(self):
        """A bounds namespace with partial coordinates (missing
        attrs) raises via the numeric-parse guard."""
        bounds = types.SimpleNamespace(miny=1.0, maxy=2.0)
        # no minx / maxx
        r = types.SimpleNamespace(
            name='Partial', boundary_source='manual', bounds=bounds,
        )
        with self.assertRaisesRegex(ValueError, 'numeric'):
            region_cache_key_from_region(r)

    def test_non_manual_with_empty_name_raises(self):
        """GADM / shapefile regions must have a non-empty name —
        empty-name fallback used to produce a degenerate empty-key
        cache path that collided with every other empty-name input."""
        r = _region_obj(name='', boundary_source='gadm')
        with self.assertRaisesRegex(ValueError, 'non-empty .name'):
            region_cache_key_from_region(r)

    def test_non_manual_with_none_name_raises(self):
        r = _region_obj(name=None, boundary_source='gadm')
        with self.assertRaisesRegex(ValueError, 'non-empty .name'):
            region_cache_key_from_region(r)


class TestRegionRoundTripPreservesBoundarySource(unittest.TestCase):
    """`Region.to_dict()` / `from_dict()` must round-trip
    `boundary_source` so reloaded objects still route to the bbox
    cache key for manual regions. Locked structurally so a future
    serialization change can't drop the field."""

    def test_manual_region_round_trip_preserves_bbox_routing(self):
        from prismpy.models.region import BoundingBox, Region
        original = Region(
            name='Unnamed study area',
            country='Mali',
            country_iso3='MLI',
            bounds=BoundingBox(minx=-5.0, miny=12.0, maxx=-3.0, maxy=14.0),
            boundary_source='manual',
        )
        reloaded = Region.from_dict(original.to_dict())
        self.assertEqual(reloaded.boundary_source, 'manual')
        self.assertEqual(
            region_cache_key_from_region(original),
            region_cache_key_from_region(reloaded),
        )
        self.assertTrue(
            region_cache_key_from_region(reloaded).startswith('manual_'),
        )

    def test_gadm_region_round_trip_preserves_name_routing(self):
        from prismpy.models.region import BoundingBox, Region
        original = Region(
            name='Boulemane',
            country='Morocco',
            country_iso3='MAR',
            bounds=BoundingBox(minx=-5.0, miny=33.0, maxx=-4.0, maxy=34.0),
            boundary_source='gadm',
        )
        reloaded = Region.from_dict(original.to_dict())
        self.assertEqual(reloaded.boundary_source, 'gadm')
        self.assertEqual(
            region_cache_key_from_region(reloaded),
            normalize_region_name('Boulemane'),
        )


# =============================================================================
# region_cache_key_from_config — PRE-RESOLUTION entry point
# =============================================================================


def _build_manual_config(
    *, name='Unnamed study area', country='Mali', iso3='MLI',
    miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
):
    """Build a validated `RegionConfig` for a manual-boundary region
    through `RegionConfig.model_validate`. All Pydantic validators
    fire — ManualBoundsConfig checks ranges + ordering, BoundaryConfig
    enforces manual → manual_bounds, RegionConfig enforces name
    min_length=1, iso3 length==3. Callers that want to probe
    invalid inputs build the raw dict directly and assert the
    validate step fails."""
    from prismpy.config.schema import RegionConfig
    return RegionConfig.model_validate({
        'name': name, 'country': country, 'country_iso3': iso3,
        'boundary': {
            'source': 'manual',
            'manual_bounds': {
                'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy,
            },
        },
    })


def _build_gadm_config(name='Boulemane', country='Morocco', iso3='MAR'):
    from prismpy.config.schema import RegionConfig
    return RegionConfig.model_validate({
        'name': name, 'country': country, 'country_iso3': iso3,
        'boundary': {
            'source': 'gadm',
            'gadm_level': 2,
            'gadm_filter_field': 'NAME_2',
            'gadm_filter_value': name,
        },
    })


class TestFromConfigManual(unittest.TestCase):
    """Manual-source `RegionConfig` → bbox-keyed identity. The
    helper reads `config.boundary.source` (BoundarySource enum)
    and `config.boundary.manual_bounds` (ManualBoundsConfig
    Pydantic model) directly — no coercion, no fallback."""

    def test_manual_config_keys_by_bbox(self):
        rc = _build_manual_config(
            miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
        )
        key = region_cache_key_from_config(rc)
        self.assertTrue(key.startswith('manual_'))
        self.assertIn('12.000000', key)
        self.assertIn('14.000000', key)
        self.assertIn('-5.000000', key)
        self.assertIn('-3.000000', key)

    def test_same_bbox_same_key(self):
        rc1 = _build_manual_config(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        rc2 = _build_manual_config(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        self.assertEqual(
            region_cache_key_from_config(rc1),
            region_cache_key_from_config(rc2),
        )

    def test_different_bbox_different_key(self):
        rc1 = _build_manual_config(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        rc2 = _build_manual_config(miny=6.0, maxy=9.0, minx=37.0, maxx=40.0)
        self.assertNotEqual(
            region_cache_key_from_config(rc1),
            region_cache_key_from_config(rc2),
        )

    def test_nearby_boxes_produce_different_keys(self):
        rc1 = _build_manual_config(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        rc2 = _build_manual_config(
            miny=12.00001, maxy=14.00001,
            minx=-5.00001, maxx=-3.00001,
        )
        self.assertNotEqual(
            region_cache_key_from_config(rc1),
            region_cache_key_from_config(rc2),
        )

    def test_negative_zero_canonicalizes(self):
        """A bbox with a prime-meridian / equator edge produces
        `-0.0` from some JS / GeoJSON round-trips. Pydantic
        accepts `-0.0` as a valid float; the helper normalizes
        it to `0.0` so two mathematically-identical bboxes
        produce the same key."""
        rc_nz = _build_manual_config(
            minx=-0.0, miny=0.0, maxx=1.0, maxy=1.0,
        )
        rc_pz = _build_manual_config(
            minx=0.0, miny=-0.0, maxx=1.0, maxy=1.0,
        )
        self.assertEqual(
            region_cache_key_from_config(rc_nz),
            region_cache_key_from_config(rc_pz),
        )
        self.assertNotIn('-0', region_cache_key_from_config(rc_nz))


class TestFromConfigGadm(unittest.TestCase):
    """GADM-source `RegionConfig` → name-keyed identity."""

    def test_gadm_config_name_keyed(self):
        rc = _build_gadm_config(name='Boulemane')
        self.assertEqual(
            region_cache_key_from_config(rc),
            normalize_region_name('Boulemane'),
        )

    def test_same_admin_name_same_key(self):
        rc1 = _build_gadm_config(name='Maradi')
        rc2 = _build_gadm_config(name='Maradi')
        self.assertEqual(
            region_cache_key_from_config(rc1),
            region_cache_key_from_config(rc2),
        )

    def test_accented_admin_name_normalizes(self):
        rc = _build_gadm_config(name='Ségou')
        # normalize_region_name strips accents + lowercases.
        self.assertEqual(
            region_cache_key_from_config(rc),
            normalize_region_name('Ségou'),
        )


class TestFromConfigValidationBoundary(unittest.TestCase):
    """Pydantic validation happens at `RegionConfig.model_validate`
    — NOT inside `region_cache_key_from_config`. Malformed inputs
    never reach the helper; the tests here document that boundary
    by asserting Pydantic raises before the helper runs.

    Prior rounds (R3 / R6 / R7) tested imperative strict guards
    inside the helper. AC-AUDIT-9 collapsed those guards —
    validation lives in the schema models, and shipping the
    helper without duplicated validation is the point.
    """

    def test_empty_name_rejected_by_model(self):
        """`RegionConfig.name` is `Field(..., min_length=1)` — an
        empty name fails at model_validate, never reaches the
        helper."""
        from pydantic import ValidationError
        from prismpy.config.schema import RegionConfig
        with self.assertRaises(ValidationError):
            RegionConfig.model_validate({
                'name': '', 'country': 'Mali', 'country_iso3': 'MLI',
                'boundary': {
                    'source': 'manual',
                    'manual_bounds': {
                        'minx': -5.0, 'miny': 12.0,
                        'maxx': -3.0, 'maxy': 14.0,
                    },
                },
            })

    def test_unknown_source_rejected_by_model(self):
        """`BoundarySource` is a strict Enum — a typo like
        `'manual '` (trailing space) or unknown source string
        fails at the enum coercion step."""
        from pydantic import ValidationError
        from prismpy.config.schema import RegionConfig
        with self.assertRaises(ValidationError):
            RegionConfig.model_validate({
                'name': 'x', 'country': 'Mali', 'country_iso3': 'MLI',
                'boundary': {'source': 'manual '},
            })

    def test_manual_without_bounds_rejected_by_model(self):
        """`BoundaryConfig.validate_source_requirements` raises
        when `source == manual` but `manual_bounds` is absent."""
        from pydantic import ValidationError
        from prismpy.config.schema import RegionConfig
        with self.assertRaises(ValidationError):
            RegionConfig.model_validate({
                'name': 'x', 'country': 'Mali', 'country_iso3': 'MLI',
                'boundary': {'source': 'manual'},
            })

    def test_manual_bounds_wrong_ordering_rejected_by_model(self):
        """`ManualBoundsConfig.validate_bounds` raises when minx
        >= maxx or miny >= maxy."""
        from pydantic import ValidationError
        from prismpy.config.schema import RegionConfig
        with self.assertRaises(ValidationError):
            RegionConfig.model_validate({
                'name': 'x', 'country': 'Mali', 'country_iso3': 'MLI',
                'boundary': {
                    'source': 'manual',
                    'manual_bounds': {
                        'minx': 5.0, 'miny': 12.0,
                        'maxx': 3.0, 'maxy': 14.0,
                    },
                },
            })

    def test_manual_bounds_out_of_geographic_range_rejected_by_model(self):
        from pydantic import ValidationError
        from prismpy.config.schema import RegionConfig
        with self.assertRaises(ValidationError):
            RegionConfig.model_validate({
                'name': 'x', 'country': 'Mali', 'country_iso3': 'MLI',
                'boundary': {
                    'source': 'manual',
                    'manual_bounds': {
                        'minx': -5.0, 'miny': 12.0,
                        'maxx': -3.0, 'maxy': 95.0,  # latitude > 90
                    },
                },
            })


class TestRegionConfigNormalizableNameCountry(unittest.TestCase):
    """V2-22b/P.2 AC-AUDIT-10 — `RegionConfig.name` and
    `RegionConfig.country` must reject values whose normalized
    form is empty.

    Pre-AC-AUDIT-10 schema: `Field(..., min_length=1)` accepted
    strings like `'   '`, `'!!!'`, `'___'` — which validated
    upstream but collapsed to `''` when passed through
    `normalize_region_name` or `sanitize_admin_name` downstream.
    Result: two different malformed configs aliasing onto the
    same empty-string cache key / lock path / filename prefix.

    The schema-level validator catches those inputs at
    `model_validate` time so every downstream consumer gets the
    invariant for free. Covers both fields; both hit
    filename-normalization paths (name → `normalize_region_name`,
    country → `sanitize_admin_name` inside several translators).
    """

    @staticmethod
    def _build(**overrides):
        """Minimal valid config builder. Callers override name /
        country to exercise the new validator."""
        from prismpy.config.schema import RegionConfig
        payload = {
            'name': 'Koutiala', 'country': 'Mali', 'country_iso3': 'MLI',
            'boundary': {
                'source': 'manual',
                'manual_bounds': {
                    'minx': -5.0, 'miny': 12.0,
                    'maxx': -3.0, 'maxy': 14.0,
                },
            },
        }
        payload.update(overrides)
        return RegionConfig.model_validate(payload)

    def test_pure_whitespace_name_rejected(self):
        from pydantic import ValidationError
        with self.assertRaisesRegex(ValidationError, 'empty identifier'):
            self._build(name='   ')

    def test_pure_punctuation_name_rejected(self):
        from pydantic import ValidationError
        with self.assertRaisesRegex(ValidationError, 'empty identifier'):
            self._build(name='!!!')

    def test_pure_underscore_name_rejected(self):
        from pydantic import ValidationError
        with self.assertRaisesRegex(ValidationError, 'empty identifier'):
            self._build(name='___')

    def test_name_whitespace_trimmed(self):
        """`mode='before'` strips surrounding whitespace so a
        legitimate name with trailing/leading spaces doesn't
        fail downstream file-naming while preserving the UX of
        accepting copy-paste input."""
        rc = self._build(name='  Koutiala  ')
        self.assertEqual(rc.name, 'Koutiala')

    def test_pure_whitespace_country_rejected(self):
        from pydantic import ValidationError
        with self.assertRaisesRegex(ValidationError, 'empty identifier'):
            self._build(country='   ')

    def test_pure_punctuation_country_rejected(self):
        """Country strings feed translators' `sanitize_admin_name`
        path, which strips punctuation. `'!!!'` collapses to ''
        there too, producing malformed output paths."""
        from pydantic import ValidationError
        with self.assertRaisesRegex(ValidationError, 'empty identifier'):
            self._build(country='!!!')

    def test_country_whitespace_trimmed(self):
        rc = self._build(country='  Mali  ')
        self.assertEqual(rc.country, 'Mali')

    def test_name_country_parity_with_from_region(self):
        """Codex R9 observation — pre-AC-AUDIT-10, `from_config`
        silently returned '' for malformed names while
        `from_region` raised. The schema-level validator closes
        the divergence: both entry points now refuse the same
        logical input, just at different layers (validate vs
        helper-call)."""
        from pydantic import ValidationError
        # Pre-resolution path: RegionConfig rejects at validate.
        with self.assertRaises(ValidationError):
            self._build(name='   ')
        # Post-resolution path: Region with empty name raises
        # inside the helper (existing behaviour, unchanged).
        with self.assertRaises(ValueError):
            region_cache_key_from_region(
                types.SimpleNamespace(
                    name='   ',
                    boundary_source='gadm',
                    bounds=types.SimpleNamespace(
                        miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
                    ),
                )
            )


class TestRegionConfigUniversalInvariants(unittest.TestCase):
    """V2-22b/P.2 AC-AUDIT-11 — schema-level rejection of values
    that are pathological across every downstream consumer, not
    only the cache-key path.

    Scope boundary (from the sprint's stopping criterion): only
    UNIVERSAL invariants live here. Consumer-specific format
    requirements (CRAFT fixed-width column widths, DSSAT file
    structure rules) belong at the writer layer, not the schema.
    Codex R10's CRAFT-specific fixed-width concerns are deferred
    to V2-22b/S (platform-translator correctness sprint); this
    class covers only what's universal: embedded control
    characters never legitimate in any identifier, ISO3 codes
    must be three ASCII letters.
    """

    @staticmethod
    def _build(**overrides):
        from prismpy.config.schema import RegionConfig
        payload = {
            'name': 'Koutiala', 'country': 'Mali', 'country_iso3': 'MLI',
            'boundary': {
                'source': 'manual',
                'manual_bounds': {
                    'minx': -5.0, 'miny': 12.0,
                    'maxx': -3.0, 'maxy': 14.0,
                },
            },
        }
        payload.update(overrides)
        return RegionConfig.model_validate(payload)

    def test_name_rejects_embedded_newline(self):
        """Control chars inside the string survive
        `normalize_region_name` (turn into `_`) but corrupt
        structured outputs that write the raw name — log lines,
        JSON reports, fixed-width records."""
        from pydantic import ValidationError
        with self.assertRaisesRegex(
            ValidationError, 'control characters',
        ):
            self._build(name='Kou\ntiala')

    def test_name_rejects_embedded_tab(self):
        from pydantic import ValidationError
        with self.assertRaisesRegex(
            ValidationError, 'control characters',
        ):
            self._build(name='Kou\ttiala')

    def test_country_rejects_embedded_newline(self):
        from pydantic import ValidationError
        with self.assertRaisesRegex(
            ValidationError, 'control characters',
        ):
            self._build(country='Mali\nWest')

    def test_country_rejects_embedded_control_char(self):
        """Full ASCII control range \\x00-\\x1f + \\x7f."""
        from pydantic import ValidationError
        with self.assertRaisesRegex(
            ValidationError, 'control characters',
        ):
            self._build(country='Mali\x07West')  # bell

    def test_country_iso3_rejects_lowercase(self):
        """`validate_iso3` used to just uppercase; now rejects
        inputs that aren't exactly three ASCII letters. An
        already-uppercase code still passes; a lowercase one
        ALSO passes because the validator strips + uppercases
        first, then validates. A genuinely invalid code (digits,
        punctuation, length) is rejected."""
        # lowercase still passes (upper-cased + pattern-checked).
        rc = self._build(country_iso3='mli')
        self.assertEqual(rc.country_iso3, 'MLI')

    def test_country_iso3_rejects_digits(self):
        from pydantic import ValidationError
        with self.assertRaisesRegex(
            ValidationError, 'ISO 3166',
        ):
            self._build(country_iso3='ML2')

    def test_country_iso3_rejects_punctuation(self):
        from pydantic import ValidationError
        with self.assertRaisesRegex(
            ValidationError, 'ISO 3166',
        ):
            self._build(country_iso3='ML!')
