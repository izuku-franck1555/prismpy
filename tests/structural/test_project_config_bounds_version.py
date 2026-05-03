"""Sprint E.0.5 AC-Q2-C2 — ProjectConfig bounds + classifier version pins.

Per CC-19, both ``bounds_version`` and
``zone_classifier_version`` are pinned per project at creation
time so re-runs reproduce the science. The values default to
the prismpy-shipped substrate at the moment the project is
created; a subsequent ratchet (new WMO normals window, new
Beck raster release, AgERA5 archive bump, numpy major-bump-
with-semantic-change) is independent of the prismpy version
the consumer is using.

This test pins both fields structurally and exercises the
default values via Pydantic instantiation. Anti-mutation
drill: drop either field → :func:`test_*_field_present` fails;
swap a default → :func:`test_*_default` fails.
"""
from __future__ import annotations

import unittest

from prismpy.config.schema import ProjectConfig


class TestProjectConfigBoundsVersion(unittest.TestCase):
    """``bounds_version`` is the per-project pin for the
    per-zone climate bounds substrate."""

    def test_field_present(self):
        self.assertIn("bounds_version", ProjectConfig.model_fields)

    def test_default_is_frozen_v1(self):
        field = ProjectConfig.model_fields["bounds_version"]
        self.assertEqual(field.default, "frozen_v1")

    def test_field_is_str(self):
        field = ProjectConfig.model_fields["bounds_version"]
        # Pydantic stores the annotation; ensure str.
        self.assertIs(field.annotation, str)


class TestProjectConfigZoneClassifierVersion(unittest.TestCase):
    """``zone_classifier_version`` is the per-project pin for
    the Köppen-Geiger raster substrate."""

    def test_field_present(self):
        self.assertIn(
            "zone_classifier_version", ProjectConfig.model_fields
        )

    def test_default_is_beck_2023_v1(self):
        field = ProjectConfig.model_fields["zone_classifier_version"]
        self.assertEqual(field.default, "beck_2023_v1")

    def test_field_is_str(self):
        field = ProjectConfig.model_fields["zone_classifier_version"]
        self.assertIs(field.annotation, str)


class TestBothFieldsArePinnedTogether(unittest.TestCase):
    """CC-19 — both fields must coexist on ProjectConfig.
    Neither can be dropped without breaking the contract."""

    def test_both_fields_declared(self):
        for field_name in ("bounds_version", "zone_classifier_version"):
            with self.subTest(field=field_name):
                self.assertIn(
                    field_name, ProjectConfig.model_fields,
                    f"ProjectConfig must declare '{field_name}' "
                    f"per CC-19. Both fields are pinned per "
                    f"project at creation; dropping one breaks "
                    f"the ratchet contract.",
                )


if __name__ == "__main__":
    unittest.main()
