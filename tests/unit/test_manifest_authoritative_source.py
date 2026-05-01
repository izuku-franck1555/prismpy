"""F-W Sprint C — manifest authoritative-source unit tests.

Covers the structural pins that don't require running the full
pipeline: the ``derive_boundary_label`` helper's exhaustiveness +
return shape, the ``create_manifest`` default + None-preservation
contract, and line-grep / AST assertions on every translator's
manifest derivation site.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from unittest import TestCase

from prismpy.packaging.manifest import create_manifest, derive_boundary_label


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRANSLATORS = _REPO_ROOT / 'src' / 'prismpy' / 'translators'
_PACKAGING = _REPO_ROOT / 'src' / 'prismpy' / 'packaging'


# ---------------------------------------------------------------------------
# Helper exhaustiveness — every BoundarySource enum value is handled
# explicitly; unknown values raise ValueError (codex MEDIUM #1).
# ---------------------------------------------------------------------------


class TestDeriveBoundaryLabelExhaustiveness(TestCase):

    def test_gadm_returns_admin_level_label(self):
        label, description = derive_boundary_label('gadm', 2)
        self.assertEqual(label, 'GADM v4.1 admin level 2')
        self.assertEqual(description, 'Official administrative boundaries')

    def test_gadm_with_level_one_renders_level_one_label(self):
        label, _ = derive_boundary_label('gadm', 1)
        self.assertEqual(label, 'GADM v4.1 admin level 1')

    def test_gadm_with_none_level_falls_back_to_two(self):
        # Honors the BoundaryConfig schema default when the caller
        # forgot to thread the configured level. Pin so a future
        # refactor doesn't change this invariant silently.
        label, _ = derive_boundary_label('gadm', None)
        self.assertEqual(label, 'GADM v4.1 admin level 2')

    def test_manual_returns_bounding_box_label(self):
        label, description = derive_boundary_label('manual', None)
        self.assertEqual(label, 'Bounding box')
        self.assertEqual(description, 'Manual coordinate bounds')

    def test_manual_bounds_alias_returns_bounding_box_label(self):
        # The runtime alias produced by retrieve-stage fallback maps
        # to the same label as the explicit MANUAL config.
        label, _ = derive_boundary_label('manual_bounds', None)
        self.assertEqual(label, 'Bounding box')

    def test_shapefile_returns_custom_shapefile_label(self):
        label, description = derive_boundary_label('shapefile', None)
        self.assertEqual(label, 'Custom shapefile')
        self.assertEqual(description, 'User-provided boundary')

    def test_unknown_source_raises_value_error(self):
        # Anti-mutation drill: a future BoundarySource enum addition
        # (e.g., OSM, EUROSTAT_NUTS) MUST surface here as ValueError
        # so the helper gets updated explicitly. A silent else
        # fallthrough would mislabel the new source as 'manual'.
        with self.assertRaises(ValueError) as cm:
            derive_boundary_label('osm', None)
        self.assertIn("Unknown boundary source", str(cm.exception))
        self.assertIn("'osm'", str(cm.exception))

    def test_unknown_source_with_level_still_raises(self):
        with self.assertRaises(ValueError):
            derive_boundary_label('eurostat_nuts', 3)


# ---------------------------------------------------------------------------
# AC-7 — manifest.py:137 default + None-preservation contract.
# ---------------------------------------------------------------------------


class TestCreateManifestGadmLevelDefault(TestCase):
    """The manifest.py default must match the BoundaryConfig schema
    default (2). The default fires only when the translator omits
    the ``gadm_level`` key entirely; an explicit None from the new
    resolved-source-discriminator translators is preserved by
    ``dict.get`` because the key is present."""

    @staticmethod
    def _minimal_project_config(**overrides):
        base = {
            'project_name': 'pin-test',
            'region_name': 'Koutiala',
            'country': 'Mali',
            'crop_name': 'Maize',
            'start_year': 2010,
            'end_year': 2020,
        }
        base.update(overrides)
        return base

    def test_omitted_key_defaults_to_two(self, tmp_path=None):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix='manifest-pin-') as tmp:
            cfg = self._minimal_project_config()  # no gadm_level key
            manifest = create_manifest(
                Path(tmp), cfg, platform='craft',
            )
        self.assertEqual(
            manifest['region']['gadm_level'], 2,
            'Omitted gadm_level key must fall through to the schema '
            'default (2). The pre-Sprint-C value was 1; AC-7 raises '
            'it to match BoundaryConfig.gadm_level default.',
        )

    def test_explicit_none_is_preserved(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix='manifest-pin-') as tmp:
            cfg = self._minimal_project_config(gadm_level=None)
            manifest = create_manifest(
                Path(tmp), cfg, platform='craft',
            )
        self.assertIsNone(
            manifest['region']['gadm_level'],
            'Explicit None from a translator (the resolved-source-'
            'discriminator path for non-GADM sources) MUST be '
            'preserved. dict.get only returns the default when the '
            'key is absent; it returns None when the key is present '
            'with value None. This pin catches a refactor that '
            'rewrites the access as `dict.get(key) or 2`, which '
            'would convert None to 2 and lie about the package.',
        )

    def test_explicit_integer_is_preserved(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix='manifest-pin-') as tmp:
            cfg = self._minimal_project_config(gadm_level=1)
            manifest = create_manifest(
                Path(tmp), cfg, platform='craft',
            )
        self.assertEqual(manifest['region']['gadm_level'], 1)


# ---------------------------------------------------------------------------
# Translator source-line pins — every manifest derivation reads
# `getattr(data.region, 'boundary_source', None) or boundary.source.value`
# and feeds the helper. Anti-mutation drill: revert any translator
# back to the pre-Sprint-C pattern → these pins fail.
# ---------------------------------------------------------------------------


_DISCRIMINATOR_PATTERN = re.compile(
    r"getattr\(\s*data\.region\s*,\s*['\"]boundary_source['\"]\s*,\s*None\s*\)"
    r"\s*or\s*boundary_config\.source\.value"
)
_HELPER_IMPORT_PATTERN = re.compile(
    r"from\s+prismpy\.packaging\.manifest\s+import\s+[^\n]*derive_boundary_label"
)


def _read(path: Path) -> str:
    return path.read_text(encoding='utf-8')


class TestTranslatorResolvedSourcePins(TestCase):
    """Every translator's manifest derivation MUST read from the
    runtime-resolved boundary source via the documented expression.
    Anti-mutation drill: revert any one to `self.config.region.
    boundary.source.value` (skipping the runtime fallback) → this
    test fails.
    """

    def test_craft_uses_resolved_source_discriminator(self):
        source = _read(_TRANSLATORS / 'craft' / 'translator.py')
        self.assertRegex(
            source, _DISCRIMINATOR_PATTERN,
            'CRAFT manifest derivation must use the documented '
            'resolved-source discriminator pattern.',
        )
        self.assertRegex(source, _HELPER_IMPORT_PATTERN)

    def test_acea_uses_resolved_source_discriminator(self):
        source = _read(_TRANSLATORS / 'acea' / 'translator.py')
        self.assertRegex(source, _DISCRIMINATOR_PATTERN)
        self.assertRegex(source, _HELPER_IMPORT_PATTERN)

    def test_pythia_uses_resolved_source_discriminator(self):
        source = _read(_TRANSLATORS / 'pythia' / 'translator.py')
        self.assertRegex(source, _DISCRIMINATOR_PATTERN)
        self.assertRegex(source, _HELPER_IMPORT_PATTERN)

    def test_sarra_py_uses_resolved_source_discriminator(self):
        source = _read(_TRANSLATORS / 'sarra_py' / 'translator.py')
        self.assertRegex(source, _DISCRIMINATOR_PATTERN)
        self.assertRegex(source, _HELPER_IMPORT_PATTERN)


class TestTranslatorRuntimeRegionBypassEliminated(TestCase):
    """Codex Gate A HIGH #2: SARRA-Py's previous manifest gadm_level
    read at line 1443 went through ``data.region.gadm_level``, which
    the executor coerces to 2 even for MANUAL configs. AC-6 replaced
    that with the discriminator-derived value. Pin the absence of
    the previous shape so a revert surfaces as a test failure."""

    def test_sarra_py_does_not_read_runtime_region_gadm_level_in_manifest(self):
        source = _read(_TRANSLATORS / 'sarra_py' / 'translator.py')
        manifest_block_pattern = re.compile(
            r'project_config\s*=\s*\{[^}]*?'
            r'data\.region\.gadm_level\s+if\s+hasattr\(\s*data\.region\s*,'
            r'\s*[\'"]gadm_level[\'"]\s*\)',
            re.DOTALL,
        )
        self.assertNotRegex(
            source, manifest_block_pattern,
            "SARRA-Py's project_config (the manifest-side dict) "
            "must NOT read data.region.gadm_level. The runtime field "
            "is coerced to a numeric default for cache-path string "
            "formatting; reading it for the manifest bypasses the "
            "resolved-source discriminator and lies about MANUAL "
            "and GADM-failed-fallback runs.",
        )


class TestTranslatorManifestExpectedKeys(TestCase):
    """Each translator's manifest dict carries the keys AC-1/2/4/5
    pin. Catches a refactor that drops one accidentally."""

    def test_craft_package_config_carries_calendar_keys(self):
        source = _read(_TRANSLATORS / 'craft' / 'translator.py')
        self.assertIn("'planting_doy'", source)
        self.assertIn("'maturity_doy'", source)

    def test_acea_package_config_carries_all_four_new_keys(self):
        source = _read(_TRANSLATORS / 'acea' / 'translator.py')
        self.assertIn("'gadm_level'", source)
        self.assertIn("'planting_doy'", source)
        self.assertIn("'maturity_doy'", source)
        # AC-4 also adds 'boundaries' to data_sources.
        self.assertIn("'boundaries': boundary_label", source)

    def test_pythia_data_sources_carries_boundaries_key(self):
        source = _read(_TRANSLATORS / 'pythia' / 'translator.py')
        # Both the manifest project_config and the readme_config
        # data_sources blocks now thread boundary_label.
        occurrences = source.count('"boundaries": boundary_label')
        readme_occurrences = source.count("'boundaries': boundary_label")
        self.assertGreaterEqual(
            occurrences + readme_occurrences, 2,
            'PYTHIA must thread boundary_label into BOTH the manifest '
            "data_sources block (line 2316 region) AND the readme_config "
            'data_sources block (line 2410 region). Pre-Sprint-C the '
            "readme_config hardcoded 'GADM v4.1' even for manual configs.",
        )

    def test_sarra_py_does_not_hardcode_gadm_v4_1_in_project_config(self):
        # The hardcoded "GADM v4.1" still appears in module-level
        # docstrings + path strings; pin the absence specifically
        # inside the project_config dict by checking line 1453's
        # pre-Sprint-C shape.
        source = _read(_TRANSLATORS / 'sarra_py' / 'translator.py')
        self.assertNotIn(
            '"boundaries": "GADM v4.1"', source,
            'SARRA-Py project_config previously hardcoded "GADM v4.1" '
            'as the boundaries label. AC-6 replaced it with the '
            'helper-derived boundary_label.',
        )


# ---------------------------------------------------------------------------
# README — Admin Level: N/A rendering when gadm_level is None.
# Logic-only pin (no full README render); the integration test
# covers end-to-end rendering via generate_readme.
# ---------------------------------------------------------------------------


class TestReadmeAdminLevelLabel(TestCase):
    """Pin the conditional that produces 'Admin Level: N/A' when
    gadm_level is None. Anti-mutation drill: revert the conditional
    to `f"Admin Level {gadm_level}"` unconditional → a None
    config renders 'Admin Level None' which leaks the runtime
    sentinel into the README.
    """

    def test_readme_template_has_no_unconditional_admin_level(self):
        readme_source = _read(_PACKAGING / 'readme_generator.py')
        # The template string for the boundary cell uses the
        # template variable, not the unconditional 'Admin Level
        # {gadm_level}' shape.
        self.assertIn(
            '| Boundaries | {boundary_source} | JSON | {admin_level_label} |',
            readme_source,
            "README template must use the {admin_level_label} "
            "variable so the builder can render 'Admin Level: N/A' "
            "for non-GADM packages. The pre-Sprint-C template "
            "rendered 'Admin Level None' for None gadm_level.",
        )

    def test_readme_builder_maps_none_to_n_a_label(self):
        readme_source = _read(_PACKAGING / 'readme_generator.py')
        # The builder computes admin_level_label before populating
        # the values dict.
        self.assertIn('admin_level_label', readme_source)
        self.assertIn('"Admin Level: N/A"', readme_source)

    def test_readme_builder_renders_admin_level_for_gadm(self):
        # Direct unit test of the conditional shape.
        from prismpy.packaging.readme_generator import generate_readme
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix='readme-admin-') as tmp:
            output = Path(tmp) / 'README.md'
            generate_readme(
                output,
                {
                    'region_name': 'Koutiala', 'country': 'Mali',
                    'crop_name': 'Maize', 'gadm_level': 2,
                    'data_sources': {'boundaries': 'GADM v4.1 admin level 2'},
                },
                platform='sarra_py',
            )
            rendered = output.read_text(encoding='utf-8')
        self.assertIn('Admin Level 2', rendered)
        self.assertNotIn('Admin Level: N/A', rendered)
        self.assertNotIn('Admin Level None', rendered)

    def test_readme_builder_renders_n_a_for_none_gadm_level(self):
        from prismpy.packaging.readme_generator import generate_readme
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix='readme-admin-') as tmp:
            output = Path(tmp) / 'README.md'
            generate_readme(
                output,
                {
                    'region_name': 'Koutiala', 'country': 'Mali',
                    'crop_name': 'Maize', 'gadm_level': None,
                    'data_sources': {'boundaries': 'Bounding box'},
                },
                platform='sarra_py',
            )
            rendered = output.read_text(encoding='utf-8')
        self.assertIn('Admin Level: N/A', rendered)
        self.assertNotIn('Admin Level None', rendered)
        self.assertNotIn('Admin Level 2', rendered)
