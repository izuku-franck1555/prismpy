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

    def test_malformed_manual_bbox_falls_back_to_name_key(self):
        """If boundary_source says manual but bbox fields are
        missing, fall through to name-key rather than raising —
        upstream validation would reject this before cache paths
        are built, but the fallback keeps the helper itself
        robust."""
        r = types.SimpleNamespace(
            name='Malformed',
            boundary_source='manual',
            bounds=None,  # missing
            boundary=None,
        )
        # No exception, falls through to name-key.
        self.assertEqual(
            region_cache_key(r), normalize_region_name('Malformed'),
        )
