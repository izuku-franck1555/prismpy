"""Gate B / Path A — `region_cache_key()` routes cache/lock/path
identity by boundary source so unnamed-manual projects sharing the
`"Unnamed study area"` display name don't collide on a single
on-disk cache.

The function is polymorphic: it accepts either `Region` (post-
resolution dataclass with `boundary_source` + `bounds`) or
`RegionConfig` (pre-resolution pydantic model with
`boundary.source` + `boundary.manual_bounds`). Both paths are
tested below.
"""

from __future__ import annotations

import types
import unittest

from prismpy.utils.sanitization import normalize_region_name, region_cache_key


def _make_region(
    *, name='Unnamed study area', boundary_source='manual',
    miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
):
    """Post-resolution Region-shaped stub."""
    bounds = types.SimpleNamespace(miny=miny, maxy=maxy, minx=minx, maxx=maxx)
    return types.SimpleNamespace(
        name=name,
        boundary_source=boundary_source,
        bounds=bounds,
    )


def _make_region_config(
    *, name='Unnamed study area', source='manual',
    miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
):
    """Pre-resolution RegionConfig-shaped stub."""
    manual = types.SimpleNamespace(miny=miny, maxy=maxy, minx=minx, maxx=maxx)
    boundary = types.SimpleNamespace(source=source, manual_bounds=manual)
    return types.SimpleNamespace(name=name, boundary=boundary)


class TestRegionCacheKeyManual(unittest.TestCase):
    def test_same_bbox_same_key(self):
        """Manual regions at identical bboxes share a cache key
        (legitimate cache reuse — same climate data needed)."""
        r1 = _make_region(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        r2 = _make_region(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        self.assertEqual(region_cache_key(r1), region_cache_key(r2))

    def test_different_bbox_different_key(self):
        """Manual regions at different bboxes get different cache
        keys even if they share the `"Unnamed study area"` display
        name. This is the whole point of Path A."""
        r_mali = _make_region(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        r_ethiopia = _make_region(miny=6.0, maxy=9.0, minx=37.0, maxx=40.0)
        self.assertNotEqual(
            region_cache_key(r_mali), region_cache_key(r_ethiopia),
        )

    def test_key_is_bbox_prefixed_not_name(self):
        """The key format signals its source (`manual_` prefix +
        coord quadruple) so an operator seeing the cache dir can
        tell a manual-keyed entry from a GADM-keyed one."""
        r = _make_region(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        key = region_cache_key(r)
        self.assertTrue(key.startswith('manual_'))
        # 6-decimal precision keeps nearby-but-distinct boxes from
        # aliasing onto the same key (Gate B MEDIUM).
        self.assertIn('12.000000', key)
        self.assertIn('14.000000', key)
        self.assertIn('-5.000000', key)
        self.assertIn('-3.000000', key)

    def test_nearby_boxes_produce_different_keys(self):
        """Codex Path A follow-up MEDIUM — boxes that differ by
        more than 1e-6 degrees (~11 cm at the equator) must get
        distinct keys. The earlier 4-decimal precision collapsed
        differences finer than 1e-4 (~11 m) onto the same key."""
        r1 = _make_region(miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0)
        r2 = _make_region(
            miny=12.00001, maxy=14.00001, minx=-5.00001, maxx=-3.00001,
        )
        self.assertNotEqual(region_cache_key(r1), region_cache_key(r2))

    def test_regionconfig_manual_keyed_by_bbox(self):
        """RegionConfig path — pre-resolution pydantic-shaped input
        also routes to the bbox key."""
        rc = _make_region_config(
            source='manual', miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
        )
        self.assertTrue(region_cache_key(rc).startswith('manual_'))

    def test_regionconfig_enum_source(self):
        """When `boundary.source` is an Enum (as in the real
        `BoundarySource.MANUAL`), the helper extracts `.value`
        correctly and still routes manual → bbox."""
        source_enum = types.SimpleNamespace(value='manual')
        manual = types.SimpleNamespace(
            miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
        )
        boundary = types.SimpleNamespace(source=source_enum, manual_bounds=manual)
        rc = types.SimpleNamespace(name='Unnamed study area', boundary=boundary)
        self.assertTrue(region_cache_key(rc).startswith('manual_'))


class TestRegionCacheKeyGadm(unittest.TestCase):
    def test_gadm_region_falls_through_to_name_key(self):
        """GADM regions keep the name-keyed cache path because
        admin names are unique identifiers. The migration is
        backward-compatible — existing on-disk caches stay valid."""
        r = _make_region(
            name='Boulemane', boundary_source='gadm',
        )
        self.assertEqual(region_cache_key(r), normalize_region_name('Boulemane'))

    def test_regionconfig_gadm_falls_through_to_name_key(self):
        """RegionConfig GADM path — same name-keyed behavior."""
        rc = _make_region_config(name='Maradi', source='gadm')
        self.assertEqual(region_cache_key(rc), normalize_region_name('Maradi'))


class TestRegionRoundTripPreservesBoundarySource(unittest.TestCase):
    """Codex Path A MEDIUM — `Region.to_dict()` / `from_dict()`
    must round-trip `boundary_source` so reloaded objects still
    route to the bbox cache key for manual regions. Previously
    the field was dropped, causing reloaded manual regions to
    fall back to name-keyed caches and collide on the shared
    `UNNAMED_MANUAL_REGION_NAME`."""

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
        # Field survives the round-trip.
        self.assertEqual(reloaded.boundary_source, 'manual')
        # And the cache key still routes to bbox.
        self.assertEqual(region_cache_key(original), region_cache_key(reloaded))
        self.assertTrue(region_cache_key(reloaded).startswith('manual_'))

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
        self.assertEqual(region_cache_key(reloaded), normalize_region_name('Boulemane'))


class TestRegionCacheKeyFallbacks(unittest.TestCase):
    def test_missing_boundary_source_defaults_to_name_key(self):
        """If `boundary_source` is None (legacy Region constructor
        path), fall back to the name-based key so behavior doesn't
        regress."""
        r = types.SimpleNamespace(
            name='Legacy Region',
            boundary_source=None,
            bounds=types.SimpleNamespace(
                miny=10.0, maxy=12.0, minx=-5.0, maxx=-3.0,
            ),
        )
        self.assertEqual(
            region_cache_key(r), normalize_region_name('Legacy Region'),
        )

    def test_malformed_manual_bbox_raises_value_error(self):
        """V2-22b/P.2 AC-AUDIT-6 — when boundary_source says
        'manual' but bounds are missing / malformed, raise
        ValueError instead of falling through to the name key.
        The earlier fall-through quietly collapsed unrelated
        manual regions (two different bboxes both labeled
        "Unnamed study area" but with missing manual_bounds)
        onto the same cache/lock path, recreating the
        cross-tenant collision the unification was meant to
        eliminate. `source == 'manual'` is a hard contract —
        callers promising manual bounds and failing to provide
        them have a version-skew bug, not a graceful fallback
        case."""
        r = types.SimpleNamespace(
            name='Malformed',
            boundary_source='manual',
            bounds=None,  # missing
            boundary=None,
        )
        with self.assertRaisesRegex(ValueError, 'manual_bounds'):
            region_cache_key(r)

    def test_malformed_prismweb_mapping_raises(self):
        """Codex R6 HIGH — the V2-22b/P.2 Mapping-shape contract
        (AC-A1.c) made dicts first-class; paired with AC-AUDIT-5
        moving `cache_lock_path` onto `region_cache_key`, a
        version-skewed prismweb payload (e.g., `boundary.source`
        was persisted but `manual_bounds` was not) would have
        silently degraded to the display-name key, collapsing
        every "Unnamed study area" project onto the same lock.
        AC-AUDIT-6 makes that a hard failure; this test pins the
        prismweb-shaped payload that codex identified."""
        d = {
            'name': 'Unnamed study area',
            'boundary': {
                'source': 'manual',
                # manual_bounds omitted entirely (version-skew
                # scenario: older prismweb persisted the source
                # flag but not the bounds dict).
            },
        }
        with self.assertRaisesRegex(ValueError, 'manual_bounds'):
            region_cache_key(d)

    def test_malformed_manual_bounds_missing_keys_raises(self):
        """Partial bounds dict — source is manual and
        `manual_bounds` exists but is missing required keys.
        Must also raise; a partial dict cannot silently produce
        a numeric key."""
        d = {
            'name': 'Partial area',
            'boundary': {
                'source': 'manual',
                'manual_bounds': {'minx': -5.0, 'miny': 12.0},
                # maxx, maxy missing
            },
        }
        with self.assertRaisesRegex(ValueError, 'manual_bounds'):
            region_cache_key(d)


class TestRegionCacheKeyDictShape(unittest.TestCase):
    """V2-22b/P.2 AC-A1.c — accept Mapping/dict inputs directly.

    Prismweb persists `config.region` as JSON; the parsed form is a
    plain dict, not a Pydantic / dataclass. The cross-repo contract
    declares dict as a first-class input shape so prismweb's
    delegator stays a one-liner."""

    def test_dict_manual_keys_by_bbox(self):
        d = {
            'name': 'Unnamed study area',
            'boundary': {
                'source': 'manual',
                'manual_bounds': {
                    'minx': -5.0, 'miny': 12.0,
                    'maxx': -3.0, 'maxy': 14.0,
                },
            },
        }
        key = region_cache_key(d)
        self.assertTrue(key.startswith('manual_'))
        self.assertIn('12.000000', key)

    def test_dict_gadm_keys_by_name(self):
        d = {'name': 'Boulemane', 'boundary': {'source': 'gadm'}}
        self.assertEqual(
            region_cache_key(d), normalize_region_name('Boulemane'),
        )

    def test_dict_same_bbox_matches_simplenamespace(self):
        """A dict input and an object-shape input with identical
        semantics MUST produce the same key — otherwise prismweb
        and prismpy could compute different identities for the
        same persisted region."""
        ns = types.SimpleNamespace(
            name='Unnamed study area',
            boundary_source='manual',
            bounds=types.SimpleNamespace(
                miny=12.0, maxy=14.0, minx=-5.0, maxx=-3.0,
            ),
        )
        d = {
            'name': 'Unnamed study area',
            'boundary': {
                'source': 'manual',
                'manual_bounds': {
                    'minx': -5.0, 'miny': 12.0,
                    'maxx': -3.0, 'maxy': 14.0,
                },
            },
        }
        self.assertEqual(region_cache_key(ns), region_cache_key(d))


class TestRegionCacheKeyNegativeZero(unittest.TestCase):
    """V2-22b/P.2 AC-A1.b — `-0.0` canonicalizes to `0.0`.

    Python formats `-0.0` as `"-0.000000"` under `.6f`, which is
    text-distinct from `"0.000000"` even though the values are
    mathematically equal. Without this normalization, a user whose
    bbox edge sits on the prime meridian (Ghana / Togo / Benin)
    or the equator could produce two identical bboxes that compute
    different cache keys depending on how their GIS stack
    round-tripped the number through IEEE 754 and JSON."""

    def test_minx_negative_zero_matches_positive_zero(self):
        mk = lambda minx: {
            'name': '',
            'boundary': {
                'source': 'manual',
                'manual_bounds': {
                    'minx': minx, 'miny': 0.0,
                    'maxx': 1.0, 'maxy': 1.0,
                },
            },
        }
        self.assertEqual(
            region_cache_key(mk(-0.0)), region_cache_key(mk(0.0)),
        )

    def test_all_four_components_canonicalize(self):
        """All four bbox components must normalize independently —
        a bbox with multiple coordinates at 0 (e.g., spanning the
        equator/meridian intersection in the Gulf of Guinea) must
        not depend on which ones happened to carry the sign bit."""
        from prismpy.utils.sanitization import region_cache_key
        nz = {
            'name': '',
            'boundary': {
                'source': 'manual',
                'manual_bounds': {
                    'minx': -0.0, 'miny': -0.0,
                    'maxx': -0.0, 'maxy': -0.0,
                },
            },
        }
        pz = {
            'name': '',
            'boundary': {
                'source': 'manual',
                'manual_bounds': {
                    'minx': 0.0, 'miny': 0.0,
                    'maxx': 0.0, 'maxy': 0.0,
                },
            },
        }
        self.assertEqual(region_cache_key(nz), region_cache_key(pz))
        # And the canonical form never surfaces `-0`.
        self.assertNotIn('-0', region_cache_key(nz))
