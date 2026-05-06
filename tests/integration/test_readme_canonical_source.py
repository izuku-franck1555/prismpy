"""Pin the F-AF-v2 + F-AN canonical-source README behavior.

The CRAFT package README's management table reads from a config
dict whose values upstream may be ``None`` (when the wizard pins
the cultivar at ``platform_config.craft.default_cultivar`` while
``management.default_cultivar`` stays ``None``). Earlier the
config-get fallbacks returned the literal ``None``, which the
template formatted as the string ``"None"`` in the rendered
markdown — the user empirically observed every management cell
reading "None" on the four post-`b005046` packages.

The dir-tree placeholder ``{admin_names}`` similarly leaked a
``"_None"`` tail when the level-2 admin name was missing, so the
README depicted schema files as ``"5m_Madarounfa_None.txt"`` even
though the on-disk file was correctly named ``"5m_Madarounfa.txt"``.

Per durable lesson #24 (canonical-source for cross-boundary
invariants): the on-disk ``management/cultivar_data.txt`` is the
truth for what the package will run with; the README must inherit
from that file, not from one of N possible config paths. Per
durable lesson #25 (user-snippet canonical Gate B): the user's
"open the README and read the management table" workflow is the
load-bearing acceptance test.

Three invariants pinned:

1. The cultivar field reads from the on-disk
   ``management/cultivar_data.txt`` first, falling through to
   the config chain only if the file is absent.
2. The management table fields never render the literal ``"None"``
   — every field coalesces to a real value via the resolver chain.
3. The dir-tree ``{admin_names}`` placeholder never carries a
   trailing ``_None`` segment.

Anti-mutation drill: revert ``_resolve_management_table`` to a
direct ``config.get(...)`` call → the cultivar test below FIRES
with the diagnostic naming the empty-config-but-on-disk-truth
case the dispatch surfaced.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prismpy.packaging.readme_generator import (
    _coalesce,
    _resolve_admin_names,
    _resolve_cultivar_from_disk,
    _resolve_management_table,
    _resolve_planting_date,
    generate_readme,
)


def _write_cultivar_data(package_dir: Path, cultivar_id: str) -> None:
    """Write a synthetic CRAFT cultivar_data.txt with one data
    row carrying ``cultivar_id`` in the CultivarID column. The
    layout mirrors what ``_generate_cultivar_data`` writes on
    a real package."""
    mgmt_dir = package_dir / "management"
    mgmt_dir.mkdir(parents=True, exist_ok=True)
    text = "CellID\tCultivarID\tCultivarPercentage\n"
    text += f"3963625\t{cultivar_id}\t1\n"
    (mgmt_dir / "cultivar_data.txt").write_text(text, encoding="utf-8")


class TestCoalesceHelper(unittest.TestCase):
    """``_coalesce`` treats ``None`` and empty strings as
    "not set" so the fallback chain reaches the actual default
    even when a config key exists with a None value."""

    def test_returns_first_non_none(self):
        self.assertEqual(_coalesce(None, "first"), "first")
        self.assertEqual(_coalesce(None, None, "third"), "third")

    def test_treats_empty_string_as_missing(self):
        self.assertEqual(_coalesce("", "second"), "second")
        self.assertEqual(_coalesce("   ", "second"), "second")

    def test_falls_through_to_default(self):
        self.assertEqual(
            _coalesce(None, None, default="fallback"),
            "fallback",
        )

    def test_preserves_non_string_truthy_zero(self):
        # Numeric 0 is a real value, not "missing"; the
        # coalesce must return it.
        self.assertEqual(_coalesce(0, 5), 0)


class TestCultivarFromDisk(unittest.TestCase):
    """``_resolve_cultivar_from_disk`` reads the CultivarID
    column of ``management/cultivar_data.txt`` — the canonical
    source for the cultivar a package runs with."""

    def test_returns_cultivar_id_from_first_data_row(self):
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td)
            _write_cultivar_data(pkg, "990002")
            self.assertEqual(
                _resolve_cultivar_from_disk(pkg), "990002",
            )

    def test_returns_none_when_file_absent(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(_resolve_cultivar_from_disk(Path(td)))

    def test_returns_none_when_package_dir_is_none(self):
        self.assertIsNone(_resolve_cultivar_from_disk(None))


class TestPlantingDateResolver(unittest.TestCase):
    """``_resolve_planting_date`` computes a calendar date from
    ``planting_doy + start_year`` when the upstream MMDD string
    is the placeholder. The dispatch's example: doy=152,
    year=2018 → "Jun 01, 2018"."""

    def test_uses_real_mmdd_when_present(self):
        config = {"planting_date": "0531"}
        self.assertEqual(_resolve_planting_date(config), "0531")

    def test_skips_placeholder_mmdd(self):
        config = {
            "planting_date": "MMDD",
            "planting_doy": 152,
            "start_year": 2018,
        }
        self.assertEqual(
            _resolve_planting_date(config), "Jun 01, 2018",
        )

    def test_skips_string_none(self):
        config = {
            "planting_date": "None",
            "planting_doy": 152,
            "start_year": 2018,
        }
        self.assertEqual(
            _resolve_planting_date(config), "Jun 01, 2018",
        )

    def test_falls_back_to_text_when_doy_missing(self):
        self.assertEqual(
            _resolve_planting_date({}), "date not specified",
        )

    def test_reads_doy_from_nested_config(self):
        config = {
            "crop": {"calendar": {"planting_doy": 152}},
            "temporal": {"start_year": 2018},
        }
        self.assertEqual(
            _resolve_planting_date(config), "Jun 01, 2018",
        )


class TestManagementTableResolver(unittest.TestCase):
    """``_resolve_management_table`` is the F-AF-v2 entrypoint.
    Every field falls through a non-None chain so the README
    never renders the literal ``"None"``."""

    def test_reads_cultivar_from_disk_first(self):
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td)
            _write_cultivar_data(pkg, "990002")
            # config carries cultivar=None (the regression
            # state); the on-disk file wins anyway.
            config = {"cultivar": None}
            out = _resolve_management_table(config, pkg)
            self.assertEqual(
                out["cultivar"], "990002",
                "Cultivar must read from management/"
                "cultivar_data.txt when the file exists; "
                "config.None must NOT clobber the canonical "
                "source.",
            )

    def test_falls_through_to_config_when_file_missing(self):
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td)
            config = {"cultivar": "IB0149"}
            out = _resolve_management_table(config, pkg)
            self.assertEqual(out["cultivar"], "IB0149")

    def test_returns_default_when_nothing_set(self):
        config = {"cultivar": None}
        out = _resolve_management_table(config, None)
        self.assertEqual(out["cultivar"], "cultivar not specified")

    def test_management_fields_never_render_literal_none(self):
        # Regression scenario: every management field is None
        # in the config dict. Resolver must coalesce each to a
        # real default so the README never shows "None".
        config = {
            "cultivar": None,
            "plant_pop": None,
            "row_spacing": None,
            "total_n": None,
            "planting_date": None,
            "n_split_ratio": None,
        }
        out = _resolve_management_table(config, None)
        for key, value in out.items():
            self.assertIsNotNone(
                value,
                f"Management table field {key!r} resolved to "
                f"None; the README would render the literal "
                f"string \"None\" — F-AF-v2 violation.",
            )
            self.assertNotEqual(
                str(value), "None",
                f"Management table field {key!r} resolved to "
                f"the string {value!r}; the README must "
                f"display a real fallback value, not \"None\".",
            )


class TestAdminNamesResolver(unittest.TestCase):
    """``_resolve_admin_names`` strips the ``_None`` tail from
    the dir-tree placeholder so the README depicts the actual
    on-disk schema filenames."""

    def test_passes_through_real_admin_names(self):
        self.assertEqual(
            _resolve_admin_names({"admin_names": "Niamey_Sahel"}),
            "Niamey_Sahel",
        )

    def test_strips_trailing_none(self):
        # The CRAFT translator's f-string produces this when
        # admin_level2 is None.
        self.assertEqual(
            _resolve_admin_names({"admin_names": "Madarounfa_None"}),
            "Madarounfa",
        )

    def test_reconstructs_from_level1_when_level2_missing(self):
        config = {
            "admin_names": "Madarounfa_None",
            "admin_level1_name": "Madarounfa",
            "admin_level2_name": None,
        }
        self.assertEqual(_resolve_admin_names(config), "Madarounfa")

    def test_falls_back_to_region_name(self):
        config = {
            "admin_names": None,
            "admin_level1_name": None,
            "region_name": "Niamey-Niger",
        }
        self.assertEqual(_resolve_admin_names(config), "Niamey-Niger")

    def test_strips_interior_none_segment(self):
        # H3 absorption — level3-aware schema can produce
        # ``Region_None_District`` when the wizard only resolved
        # level1 + level3 (level2 missing). The dir-tree
        # filename on disk drops the None segment, so the
        # resolver must too.
        self.assertEqual(
            _resolve_admin_names({"admin_names": "Region_None_District"}),
            "Region_District",
        )

    def test_strips_multiple_none_segments(self):
        self.assertEqual(
            _resolve_admin_names({"admin_names": "Country_None_None_City"}),
            "Country_City",
        )

    def test_reconstructs_with_level3_when_present(self):
        config = {
            "admin_names": None,
            "admin_level1_name": "Region",
            "admin_level2_name": None,
            "admin_level3_name": "District",
        }
        self.assertEqual(
            _resolve_admin_names(config),
            "Region_District",
        )


class TestEndToEndReadmeRender(unittest.TestCase):
    """Integration: write a synthetic CRAFT package, render
    the README, assert the management table never carries
    ``"None"`` and the dir-tree filenames don't end in
    ``_None``. This is the user-snippet canonical Gate B per
    durable lesson #25 — open the README and read.
    """

    def test_readme_management_table_never_renders_none(self):
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td)
            _write_cultivar_data(pkg, "990002")

            # The regression config: every management field
            # is None; planting_date is the MMDD placeholder;
            # admin_names carries the _None tail. This is what
            # the wizard pinned at run #6 packages.
            config = {
                "package_name": "test_package",
                "project_name": "Test",
                "region_name": "Madarounfa",
                "country": "NER",
                "country_code": "NE",
                "crop_name": "Maize",
                "start_year": 2018,
                "end_year": 2020,
                "cultivar": None,
                "plant_pop": None,
                "row_spacing": None,
                "total_n": None,
                "planting_date": "MMDD",
                "planting_doy": 152,
                "n_split_ratio": None,
                "admin_names": "Madarounfa_None",
                "n_cells": 12,
                "n_soil_profiles": 3,
                "schema_level": 2,
                "craft_level": 2,
                "soil_source": "HWSD v2.0",
                "soil_description": "HWSD soil profiles",
                "crop_mask_source": "SPAM 2020",
                "crop_mask_description": "Harvested area",
                "boundary_source": "GADM v4.1",
                "boundary_description": "Admin boundaries",
                "data_sources": {},
            }

            readme_path = generate_readme(
                pkg / "README.md", config, platform="craft",
            )
            text = readme_path.read_text(encoding="utf-8")

            # The management table fields all have markdown
            # row prefixes like "**Cultivar** | ". Inspect a
            # window around each label and assert no "| None"
            # cells.
            for label in (
                "**Cultivar**", "**Plant population**",
                "**Row spacing**", "**Total N**",
                "**Planting date**", "**N split**",
            ):
                idx = text.find(label)
                self.assertGreater(
                    idx, 0,
                    f"README missing management label {label!r}; "
                    f"layout drift.",
                )
                row = text[idx: idx + 200].splitlines()[0]
                self.assertNotIn(
                    "| None |", row,
                    f"README management row for {label!r} "
                    f"renders the literal \"None\": {row!r}. "
                    f"F-AF-v2 violation — every field must "
                    f"resolve to a real value.",
                )

    def test_readme_dir_tree_drops_admin_none_tail(self):
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td)
            _write_cultivar_data(pkg, "990002")
            config = {
                "package_name": "test_package",
                "project_name": "Test",
                "region_name": "Madarounfa",
                "country": "NER",
                "country_code": "NE",
                "crop_name": "Maize",
                "start_year": 2018,
                "end_year": 2020,
                "planting_doy": 152,
                # The regression shape: admin_names carries
                # the _None tail because admin_level2 was
                # None upstream.
                "admin_names": "Madarounfa_None",
                "n_cells": 12,
                "n_soil_profiles": 3,
                "schema_level": 2,
                "craft_level": 2,
                "data_sources": {},
            }
            readme_path = generate_readme(
                pkg / "README.md", config, platform="craft",
            )
            text = readme_path.read_text(encoding="utf-8")
            # The dir-tree depicts schema filenames using the
            # admin_names placeholder. After F-AN absorption
            # the tail _None must not survive into the
            # rendered tree.
            self.assertNotIn(
                "_None.txt", text,
                "README dir-tree carries a filename ending in "
                "\"_None.txt\"; the placeholder leaked the "
                "Python None sentinel into the user-visible "
                "depiction. F-AN violation.",
            )
            # Positive pin: the rendered admin_names must
            # match the on-disk filename shape.
            self.assertIn(
                "5m_Madarounfa.txt", text,
                "README dir-tree must depict the actual on-"
                "disk schema filename pattern (5m_Madarounfa"
                ".txt) once the _None tail is stripped.",
            )


if __name__ == "__main__":
    unittest.main()
