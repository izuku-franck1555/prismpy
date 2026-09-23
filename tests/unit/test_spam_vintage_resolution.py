"""SPAM cropland-vintage: concrete registry + fail-loud resolver + AppliedVintage state +
vintage-honest README render + the boundary guard (resolver wired into PYTHIA only).

Self-contained: the per-vintage crop/stratum inventories are literal, so the four distinct
resolver failures are provable WITHOUT any external SPAM files on disk.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prismpy.sources.crop_areas.spam_vintage import (
    AppliedVintage,
    CropNotInVintageError,
    SPAM_VINTAGES,
    SpamVintageError,
    StratumNotInVintageError,
    VintageNotRegisteredError,
    VintageRasterAbsentError,
    resolve_spam_raster,
)


def _touch(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"")
    return path


# --------------------------------------------------------------------------- #
# Registry facts (self-contained; verified against Franck's provisioned files) #
# --------------------------------------------------------------------------- #
def test_registry_inventories_match_provisioned_reality():
    v2010 = SPAM_VINTAGES[("2010", "V2r0")]
    v2020 = SPAM_VINTAGES[("2020", "V2r2")]
    assert len(v2010.crops) == 42
    assert len(v2020.crops) == 46
    assert v2010.strata == frozenset({"A", "H", "I", "L", "R", "S"})
    assert v2020.strata == frozenset({"A", "I", "R"})
    # underscore-after-2020 vs none-after-2010 (the real per-vintage filename formats)
    assert v2020.pattern == "spam2020_V2r2_global_H_{code}_{tech}.tif"
    assert v2010.pattern == "spam2010V2r0_global_H_{code}_{tech}.tif"


def test_registry_crop_coverage_differs_between_vintages():
    v2010 = SPAM_VINTAGES[("2010", "V2r0")].crops
    v2020 = SPAM_VINTAGES[("2020", "V2r2")].crops
    # MAIZ (the Oromia crop) is in both; the coverage genuinely diverges at the edges.
    assert "MAIZ" in v2010 and "MAIZ" in v2020
    assert {"ACOF", "SMIL"} <= v2010 and not ({"ACOF", "SMIL"} & v2020)
    assert {"CITR", "COFF", "MILL", "ONIO", "RUBB", "TOMA"} <= v2020
    assert not ({"CITR", "COFF", "MILL", "ONIO", "RUBB", "TOMA"} & v2010)


# --------------------------------------------------------------------------- #
# The four distinct fail-loud exceptions, behavior-bound                       #
# --------------------------------------------------------------------------- #
def test_unregistered_vintage_raises(tmp_path):
    with pytest.raises(VintageNotRegisteredError, match="not a registered"):
        resolve_spam_raster(tmp_path, "2005", "V9r9", "MAIZ", "A")


def test_crop_not_in_vintage_raises(tmp_path):
    # TOMA exists in 2020/V2r2 but NOT in 2010/V2r0 → crop-not-mapped, not "absent file".
    with pytest.raises(CropNotInVintageError, match="not mapped"):
        resolve_spam_raster(tmp_path, "2010", "V2r0", "TOMA", "A")


def test_stratum_not_in_vintage_raises(tmp_path):
    # 'H' stratum exists in 2010 but NOT in 2020/V2r2 (A/I/R only).
    with pytest.raises(StratumNotInVintageError, match="stratum"):
        resolve_spam_raster(tmp_path, "2020", "V2r2", "MAIZ", "H")


def test_raster_absent_raises(tmp_path):
    # All valid, but the file is not provisioned → distinct "not found" error, never None.
    with pytest.raises(VintageRasterAbsentError, match="not found"):
        resolve_spam_raster(tmp_path, "2020", "V2r2", "MAIZ", "A")


def test_all_four_exceptions_subclass_the_base():
    for exc in (
        VintageNotRegisteredError,
        CropNotInVintageError,
        StratumNotInVintageError,
        VintageRasterAbsentError,
    ):
        assert issubclass(exc, SpamVintageError)


# --------------------------------------------------------------------------- #
# Success path — exact single path, both real vintages, no wildcard            #
# --------------------------------------------------------------------------- #
def test_resolve_success_2020_v2r2_case_insensitive(tmp_path):
    expected = _touch(tmp_path, "spam2020_V2r2_global_H_MAIZ_A.tif")
    got = resolve_spam_raster(tmp_path, "2020", "V2r2", "maiz", "a")  # lower-case in
    assert got == expected


def test_resolve_success_2010_v2r0(tmp_path):
    expected = _touch(tmp_path, "spam2010V2r0_global_H_MAIZ_A.tif")  # no underscore after 2010
    got = resolve_spam_raster(tmp_path, "2010", "V2r0", "MAIZ", "A")
    assert got == expected


def test_no_wildcard_cannot_smuggle_a_different_vintage(tmp_path):
    # A foreign-vintage file is present but the SELECTED vintage's file is absent. The removed
    # wildcard means the resolver STILL fails loud — it never clips the foreign 2010 file for 2020.
    _touch(tmp_path, "spam2010V2r0_global_H_MAIZ_A.tif")
    with pytest.raises(VintageRasterAbsentError):
        resolve_spam_raster(tmp_path, "2020", "V2r2", "MAIZ", "A")


# --------------------------------------------------------------------------- #
# AppliedVintage — the single canonical state object                          #
# --------------------------------------------------------------------------- #
def test_applied_vintage_label_and_manifest_dict():
    av = AppliedVintage(
        year="2020", release="V2r2",
        source_filename="spam2020_V2r2_global_H_MAIZ_A.tif",
        mask_filename="harvest_area_2020_V2r2.tif",
    )
    assert av.label == "SPAM 2020 V2r2"
    assert av.to_manifest_dict() == {
        "year": "2020",
        "release": "V2r2",
        "source_filename": "spam2020_V2r2_global_H_MAIZ_A.tif",
        "mask_filename": "harvest_area_2020_V2r2.tif",
    }


# --------------------------------------------------------------------------- #
# README renders the vintage from the STRUCTURED field (not parse/glob)        #
# --------------------------------------------------------------------------- #
def test_generate_readme_renders_vintage_from_structured_field(tmp_path):
    from prismpy.packaging.readme_generator import generate_readme

    config = {
        "project_name": "vintage_test",
        "region_name": "Oromia",
        "country": "Ethiopia",
        "crop_name": "Maize",
        "start_year": 2015,
        "end_year": 2019,
    }
    out = tmp_path / "README.md"
    generate_readme(
        out, config, platform="pythia", mask_present=True,
        crop_mask_vintage={
            "year": "2010", "release": "V2r0",
            "mask_filename": "harvest_area_2010_V2r0.tif",
        },
    )
    text = out.read_text()
    assert "SPAM 2010 V2r0" in text           # true applied vintage
    assert "harvest_area_2010_V2r0.tif" in text  # vintage-named file
    assert "2020 v2.0" not in text            # the old hardcoded table cell is gone
    assert "SPAM 2020" not in text            # regression guard: NO stale 2020 when 2010 applied


# --------------------------------------------------------------------------- #
# Boundary guard — resolver wired into PYTHIA ONLY (ACEA + CRAFT untouched) #
# --------------------------------------------------------------------------- #
def test_boundary_resolver_wired_into_pythia_only():
    import prismpy.translators.acea.translator as acea_mod
    import prismpy.translators.craft.translator as craft_mod
    import prismpy.translators.pythia.translator as pythia_mod

    for mod in (acea_mod, craft_mod):
        src = Path(mod.__file__).read_text()
        assert "spam_vintage" not in src, (
            f"{mod.__name__} must not reference the cropland-vintage module "
            "(boundary: ACEA/CRAFT vintage-honesty is a separate change)."
        )
        assert "resolve_spam_raster" not in src, (
            f"{mod.__name__} must not call resolve_spam_raster."
        )

    psrc = Path(pythia_mod.__file__).read_text()
    assert "resolve_spam_raster" in psrc, "PYTHIA must wire the fail-loud resolver."


def test_boundary_files_byte_identical_to_base():
    """Stronger boundary guard: the ACEA + CRAFT translators, the ACEA-only
    ``SPAMSource`` module (spam.py), and the provenance model + tracker must be BYTE-IDENTICAL to
    the merge-base with ``origin/main``. this change touches none of them, so no ACEA/CRAFT vintage behavior
    and no provenance schema can shift — identical source ⇒ identical runtime for these files. Skips
    where git history / ``origin/main`` is unavailable (the grep guard above still runs everywhere).
    """
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    boundary = (
        "src/prismpy/translators/acea/translator.py",
        "src/prismpy/translators/craft/translator.py",
        "src/prismpy/sources/crop_areas/spam.py",
        "src/prismpy/models/provenance.py",
        "src/prismpy/provenance/tracker.py",
    )

    def _git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)

    base = _git("merge-base", "HEAD", "origin/main")
    if base.returncode != 0 or not base.stdout.strip():
        pytest.skip("git merge-base / origin/main unavailable")
    base_sha = base.stdout.strip()

    breached = [
        rel for rel in boundary
        if _git("diff", "--quiet", base_sha, "--", rel).returncode != 0
    ]
    assert not breached, (
        "Boundary breached — these files must be byte-identical to the base "
        f"(this change touches no ACEA/CRAFT/provenance code): {breached}"
    )
