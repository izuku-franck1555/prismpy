"""Sprint D.1 AC-7 + AC-8 — preserve Sprint C honest-signal contract.

Pins that Sprint D.1's harmonize-stage + provenance additions
do NOT regress the F-W Sprint C honest-signal invariants:

* The manifest's ``data_sources.soil`` field stays a STRING in
  Sprint D.1 (the structured identity work is deferred to
  Sprint D.2). The 16-case parametrize spans 4 platforms × 4
  soil-source labels so a future commit that introduces a dict
  shape on soil surfaces here.
* Sprint C's boundary discriminator + null-XOR invariants are
  already pinned by the existing
  ``tests/integration/test_manifest_field_population_parity.py``
  and ``test_manifest_provenance_consistency.py`` files; this
  file does not duplicate that coverage.

The producer-vs-consumer pin parity (AC-6.1) lives in
``test_sprint_d_provenance_consumers.py``; this file is
strictly the F-W honest-signal regression net.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from prismpy.packaging.manifest import create_manifest, derive_boundary_label


def _build_project_config(
    platform: str,
    boundary_source: str,
    soil_label: str,
) -> dict:
    """Mirror the post-Sprint-C discriminator + helper pattern
    every translator follows."""
    if boundary_source == "gadm":
        gadm_level = 2
    else:
        gadm_level = None
    boundary_label, _ = derive_boundary_label(boundary_source, gadm_level)
    base = {
        "project_name": f"sprint-d-no-regress-{platform}",
        "package_name": f"sprint-d-no-regress-{platform}",
        "region_name": "Koutiala",
        "country": "Mali",
        "crop_name": "Maize",
        "planting_doy": 152,
        "maturity_doy": 304,
        "start_year": 2010,
        "end_year": 2020,
        "gadm_level": gadm_level,
        "data_sources": {
            "climate": "NASA POWER",
            "soil": soil_label,
            "boundaries": boundary_label,
        },
    }
    if platform == "craft":
        base["data_sources"]["crop_mask"] = "SPAM 2020"
    elif platform == "acea":
        base["data_sources"]["harvested_areas"] = "SPAM 2020"
        base["data_sources"]["crop_suitability"] = "FAO GAEZ v4"
    elif platform == "sarra_py":
        base["data_sources"]["rainfall"] = "TAMSAT v3.1"
        base["data_sources"]["temperature"] = "AgERA5"
    return base


# ---------------------------------------------------------------------------
# AC-7 — manifest.data_sources.soil stays a STRING (16 cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", ["craft", "pythia", "acea", "sarra_py"])
@pytest.mark.parametrize(
    "soil_label",
    ["iSDA Africa", "HWSD v2.0", "iSDA + HWSD fallback", "no_coverage"],
)
def test_manifest_data_sources_soil_is_string(platform, soil_label):
    """Sprint D.1 does NOT change the manifest soil field's
    shape; the existing prismweb consumers at ``views.py:982``
    and ``:2750`` call ``.lower()`` on the value, which only
    works if the value is a string. Sprint D.2 may introduce a
    structured identity ELSEWHERE (provenance / sidecar) but
    the manifest string shape must be preserved here."""
    project_config = _build_project_config(
        platform, boundary_source="manual", soil_label=soil_label,
    )
    with TemporaryDirectory(prefix=f"no-regress-{platform}-") as tmp:
        package_dir = Path(tmp) / platform
        package_dir.mkdir(parents=True)
        manifest = create_manifest(
            package_dir, project_config, platform=platform,
        )

    soil_value = manifest["data_sources"]["soil"]
    assert isinstance(soil_value, str), (
        f"{platform}/{soil_label}: manifest.data_sources.soil "
        f"is {type(soil_value).__name__}, expected str."
    )
    # The existing prismweb consumer pattern must still work.
    assert soil_value.lower() == soil_label.lower()


# ---------------------------------------------------------------------------
# AC-8 — F-W boundary string-pin sanity check (existing F-W tests carry the
# heavy lifting; this is a lightweight cross-reference)
# ---------------------------------------------------------------------------


def test_fw_boundary_invariants_still_locked():
    """Reflective check that Sprint C's load-bearing test
    modules are still on disk + collectible. The actual
    boundary invariant assertions live in those modules; this
    test catches accidental deletion of either pin file as part
    of Sprint D.1 churn."""
    fw_files = [
        "tests/integration/test_manifest_field_population_parity.py",
        "tests/integration/test_manifest_provenance_consistency.py",
        "tests/unit/test_manifest_authoritative_source.py",
    ]
    repo = Path(__file__).resolve().parents[2]
    for relpath in fw_files:
        path = repo / relpath
        assert path.exists(), (
            f"Sprint C honest-signal pin file {relpath!r} is "
            f"missing. Sprint D.1 must not delete F-W coverage."
        )
