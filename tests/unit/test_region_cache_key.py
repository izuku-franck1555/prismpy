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


def _config_dict(
    *, name='Unnamed study area', source='manual',
    miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
):
    """Mapping-shaped config dict (prismweb's persisted shape)."""
    return {
        'name': name,
        'boundary': {
            'source': source,
            'manual_bounds': {
                'minx': minx, 'miny': miny,
                'maxx': maxx, 'maxy': maxy,
            },
        },
    }


def _config_ns(
    *, name='Unnamed study area', source='manual',
    miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
):
    """Pydantic-like SimpleNamespace shaped like RegionConfig. For
    assertions that attribute access works alongside Mapping access."""
    manual = types.SimpleNamespace(
        miny=miny, maxy=maxy, minx=minx, maxx=maxx,
    )
    boundary = types.SimpleNamespace(source=source, manual_bounds=manual)
    return types.SimpleNamespace(name=name, boundary=boundary)


class TestFromConfigManual(unittest.TestCase):
    """Manual-source configs — bbox-keyed via nested
    `boundary.manual_bounds`. Both Mapping and Pydantic-style
    inputs supported."""

    def test_dict_manual_keys_by_bbox(self):
        d = _config_dict(
            source='manual', miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
        )
        key = region_cache_key_from_config(d)
        self.assertTrue(key.startswith('manual_'))
        self.assertIn('12.000000', key)

    def test_pydantic_like_manual_keys_by_bbox(self):
        rc = _config_ns(
            source='manual', miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
        )
        self.assertTrue(region_cache_key_from_config(rc).startswith('manual_'))

    def test_enum_source_value_accepted(self):
        """Pydantic's real BoundarySource enum has `.value == 'manual'`."""
        source_enum = types.SimpleNamespace(value='manual')
        manual = types.SimpleNamespace(
            miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
        )
        boundary = types.SimpleNamespace(
            source=source_enum, manual_bounds=manual,
        )
        rc = types.SimpleNamespace(name='x', boundary=boundary)
        self.assertTrue(region_cache_key_from_config(rc).startswith('manual_'))

    def test_dict_and_pydantic_agree(self):
        """Mapping and attribute-style inputs with identical
        semantics produce identical keys — prismweb can persist
        via dict and prismpy can read via Pydantic without drift."""
        d = _config_dict(
            source='manual', miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
        )
        ns = _config_ns(
            source='manual', miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
        )
        self.assertEqual(
            region_cache_key_from_config(d),
            region_cache_key_from_config(ns),
        )

    def test_negative_zero_canonicalizes(self):
        d_nz = _config_dict(
            source='manual', minx=-0.0, miny=0.0, maxx=1.0, maxy=1.0,
        )
        d_pz = _config_dict(
            source='manual', minx=0.0, miny=-0.0, maxx=1.0, maxy=1.0,
        )
        self.assertEqual(
            region_cache_key_from_config(d_nz),
            region_cache_key_from_config(d_pz),
        )
        self.assertNotIn('-0', region_cache_key_from_config(d_nz))

    def test_nearby_boxes_produce_different_keys(self):
        d1 = _config_dict(
            source='manual', miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
        )
        d2 = _config_dict(
            source='manual',
            miny=12.00001, maxy=14.00001,
            minx=-5.00001, maxx=-3.00001,
        )
        self.assertNotEqual(
            region_cache_key_from_config(d1),
            region_cache_key_from_config(d2),
        )


class TestFromConfigGadm(unittest.TestCase):
    """GADM configs — name-keyed; no bbox fallback even if a
    manual_bounds dict is present under a gadm source."""

    def test_dict_gadm_name_keyed(self):
        d = {
            'name': 'Boulemane',
            'boundary': {'source': 'gadm'},
        }
        self.assertEqual(
            region_cache_key_from_config(d),
            normalize_region_name('Boulemane'),
        )

    def test_pydantic_gadm_name_keyed(self):
        rc = _config_ns(name='Maradi', source='gadm')
        self.assertEqual(
            region_cache_key_from_config(rc),
            normalize_region_name('Maradi'),
        )

    def test_missing_boundary_defaults_to_name_key(self):
        """Config without a `boundary` section at all — fall through
        to the name-key path (non-manual treatment)."""
        d = {'name': 'Plain name-only region'}
        self.assertEqual(
            region_cache_key_from_config(d),
            normalize_region_name('Plain name-only region'),
        )


class TestFromConfigStrict(unittest.TestCase):
    """Strict guards — pre-resolution inputs with malformed shapes
    raise `ValueError` instead of silently degrading. Closes the
    codex R3 / R6 / R7 surfaces:
    - dict-valued `boundary.source` silently stringified;
    - conflicting top-level vs nested source (removed — AC-AUDIT-8
      doesn't support top-level `boundary_source` on the config
      shape; it's a Region-dataclass field);
    - partial / stray `manual_bounds`;
    - top-level `bounds` recovery fallback for Mappings."""

    def test_raw_string_config_rejected(self):
        """A raw string as the whole config — no boundary, no name
        attr — should raise. Catches callers that pre-AC-AUDIT-5
        pattern-matched the old polymorphic signature."""
        with self.assertRaises(ValueError):
            region_cache_key_from_config('raw_string')

    def test_empty_dict_rejected(self):
        with self.assertRaises(ValueError):
            region_cache_key_from_config({})

    def test_manual_missing_manual_bounds_raises(self):
        d = {
            'name': 'Unnamed study area',
            'boundary': {'source': 'manual'},
            # no manual_bounds
        }
        with self.assertRaisesRegex(ValueError, 'manual_bounds'):
            region_cache_key_from_config(d)

    def test_manual_with_partial_manual_bounds_raises(self):
        d = {
            'name': 'Partial area',
            'boundary': {
                'source': 'manual',
                'manual_bounds': {'minx': -5.0, 'miny': 12.0},
                # maxx / maxy missing
            },
        }
        with self.assertRaisesRegex(ValueError, 'numeric'):
            region_cache_key_from_config(d)

    def test_manual_with_stray_top_level_bounds_raises(self):
        """Codex R7 scenario — a Mapping with partial
        `manual_bounds` AND a stray fully-populated top-level
        `bounds`. Under the polymorphic helper this returned the
        stray bbox's key. Under the strict `from_config` entry
        point, the stray `bounds` is NOT consulted — the partial
        `manual_bounds` fails its numeric-parse guard and
        surfaces as a ValueError."""
        d = {
            'name': 'Unnamed study area',
            'boundary': {
                'source': 'manual',
                'manual_bounds': {'minx': -5.0, 'miny': 12.0},
            },
            'bounds': {
                'minx': 1.0, 'miny': 2.0, 'maxx': 3.0, 'maxy': 4.0,
            },
        }
        with self.assertRaisesRegex(ValueError, 'numeric'):
            region_cache_key_from_config(d)

    def test_manual_no_manual_bounds_with_stray_top_level_bounds_raises(self):
        """Variant of the above — NO `manual_bounds` at all + stray
        top-level `bounds`. Previously would silently use the
        stray bounds; now raises because manual_bounds is missing."""
        d = {
            'name': 'Unnamed study area',
            'boundary': {'source': 'manual'},
            'bounds': {
                'minx': 1.0, 'miny': 2.0, 'maxx': 3.0, 'maxy': 4.0,
            },
        }
        with self.assertRaisesRegex(ValueError, 'manual_bounds'):
            region_cache_key_from_config(d)

    def test_dict_valued_source_rejected(self):
        """Non-string / non-enum source — e.g., a dict with a
        `'value'` key (codex R7 scenario: a version-skewed
        serializer that didn't unwrap the enum) — rejected with a
        clear message rather than stringified silently."""
        d = {
            'name': 'x',
            'boundary': {
                'source': {'value': 'manual'},
                'manual_bounds': {
                    'minx': -5.0, 'miny': 12.0,
                    'maxx': -3.0, 'maxy': 14.0,
                },
            },
        }
        with self.assertRaisesRegex(ValueError, 'invalid boundary source'):
            region_cache_key_from_config(d)

    def test_list_valued_source_rejected(self):
        d = {
            'name': 'x',
            'boundary': {
                'source': ['manual'],
                'manual_bounds': {
                    'minx': -5.0, 'miny': 12.0,
                    'maxx': -3.0, 'maxy': 14.0,
                },
            },
        }
        with self.assertRaisesRegex(ValueError, 'invalid boundary source'):
            region_cache_key_from_config(d)

    def test_enum_like_with_non_string_value_rejected(self):
        """Enum-like object whose `.value` is a non-string —
        guards against duck-typed enum impersonators that pass
        `hasattr(source, 'value')` but aren't the real enum."""
        fake = types.SimpleNamespace(value=42)
        rc = types.SimpleNamespace(
            name='x',
            boundary=types.SimpleNamespace(
                source=fake,
                manual_bounds=types.SimpleNamespace(
                    miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, 'invalid boundary source'):
            region_cache_key_from_config(rc)

    def test_non_manual_with_empty_name_raises(self):
        """Config without a manual source AND with an empty name
        has no identity to key off; raise instead of returning an
        empty-key cache path."""
        d = {'name': '', 'boundary': {'source': 'gadm'}}
        with self.assertRaisesRegex(ValueError, 'non-empty name'):
            region_cache_key_from_config(d)
