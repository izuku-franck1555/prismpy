"""F-W Sprint C — cross-platform manifest parity (AC-8).

Verifies the manifest contract: identical-shape ``region.gadm_level``
+ ``data_sources.boundaries`` across CRAFT / PYTHIA / ACEA / SARRA-Py
for the same fixture. Both GADM and MANUAL boundary-source variants
exercised. The translator-side wiring (every translator using the
same resolved-source discriminator + helper) is pinned by the unit
test module ``test_manifest_authoritative_source.py``; this module
verifies the manifest WRITER layer end-to-end against the contract
that every translator passes the SAME dict shape to ``create_manifest``.

Approach: each translator builds a per-platform ``project_config``
dict and passes it to ``create_manifest``. AC-8 boils down to "every
translator's ``project_config`` shape produces a manifest with the
same fields when fed through ``create_manifest``." The fixture
factory below constructs the dict each translator would emit (after
the resolved-source discriminator + ``derive_boundary_label`` call)
for a Koutiala-Maize run, then exercises ``create_manifest`` for
each platform and asserts the output JSON matches.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from prismpy.packaging.manifest import (
    create_manifest,
    derive_boundary_label,
    save_manifest,
)


# ---------------------------------------------------------------------------
# Fixture factory — produces the canonical project_config each
# translator would build for a Koutiala-Maize run, parameterized on
# the resolved boundary source.
# ---------------------------------------------------------------------------


def _resolved_source_outcome(boundary_source: str):
    """Mirror the discriminator logic every translator now uses.
    Returns ``(resolved_source, manifest_gadm_level, boundary_label)``.

    For GADM, gadm_level is the configured level (2 for Koutiala
    fixtures); for non-GADM, gadm_level is ``None`` per AC-1 + AC-4
    + AC-5 + AC-6 honest-signal contract.
    """
    if boundary_source == 'gadm':
        gadm_level = 2
    else:
        gadm_level = None
    label, _ = derive_boundary_label(boundary_source, gadm_level)
    return boundary_source, gadm_level, label


def _project_config_for(platform: str, boundary_source: str):
    """Build the ``project_config`` dict each translator passes to
    ``create_manifest`` for a Koutiala-Maize fixture under the given
    resolved boundary source. Mirrors the post-Sprint-C structure of
    every translator's manifest derivation."""
    _, gadm_level, label = _resolved_source_outcome(boundary_source)
    base = {
        'project_name': f'koutiala-maize-{platform}',
        'package_name': f'koutiala-maize-{platform}',
        'region_name': 'Koutiala',
        'country': 'Mali',
        'crop_name': 'Maize',
        'planting_doy': 152,
        'maturity_doy': 304,
        'start_year': 2010,
        'end_year': 2020,
        'gadm_level': gadm_level,
        'data_sources': {
            'climate': 'NASA POWER',
            'soil': 'iSDA',
            'boundaries': label,
        },
    }
    if platform == 'craft':
        base['data_sources']['crop_mask'] = 'SPAM 2020'
    elif platform == 'acea':
        base['data_sources']['harvested_areas'] = 'SPAM 2020'
        base['data_sources']['crop_suitability'] = 'FAO GAEZ v4'
    elif platform == 'sarra_py':
        base['data_sources']['rainfall'] = 'TAMSAT v3.1'
        base['data_sources']['temperature'] = 'AgERA5'
    return base


# ---------------------------------------------------------------------------
# AC-8 cross-platform parity test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('platform', ['craft', 'pythia', 'acea', 'sarra_py'])
@pytest.mark.parametrize(
    'boundary_source,expected_gadm_level,expected_label',
    [
        ('gadm', 2, 'GADM v4.1 admin level 2'),
        ('manual', None, 'Bounding box'),
    ],
)
def test_manifest_fields_uniform_across_platforms(
    platform, boundary_source, expected_gadm_level, expected_label,
):
    """Every translator's manifest carries the same boundary +
    calendar shape for the same fixture. GADM and MANUAL variants
    both tested per the AC-8 anti-mutation drill matrix."""
    project_config = _project_config_for(platform, boundary_source)
    with TemporaryDirectory(prefix=f'parity-{platform}-') as tmp:
        package_dir = Path(tmp) / platform
        package_dir.mkdir(parents=True)
        manifest = create_manifest(
            package_dir, project_config, platform=platform,
        )

    assert manifest['region']['gadm_level'] == expected_gadm_level, (
        f"{platform} / {boundary_source}: expected "
        f"manifest.region.gadm_level={expected_gadm_level!r}, got "
        f"{manifest['region']['gadm_level']!r}. The cross-platform "
        f"parity invariant requires every translator to emit the "
        f"same gadm_level for the same boundary_source — non-GADM "
        f"resolves to None, GADM to the configured admin level."
    )
    assert manifest['region']['name'] == 'Koutiala'
    assert manifest['region']['country'] == 'Mali'
    assert manifest['crop']['planting_doy'] == 152
    assert manifest['crop']['maturity_doy'] == 304
    assert manifest['data_sources']['boundaries'] == expected_label, (
        f"{platform} / {boundary_source}: expected "
        f"manifest.data_sources.boundaries={expected_label!r}, got "
        f"{manifest['data_sources']['boundaries']!r}. The cross-"
        f"platform parity invariant requires every translator to "
        f"emit the same boundary label via derive_boundary_label."
    )


# ---------------------------------------------------------------------------
# AC-8 anti-mutation pin — the parametrize covers exactly 4 platforms × 2
# boundary_source variants = 8 cases. If a future contributor narrows the
# parametrize, this pin fires.
# ---------------------------------------------------------------------------


def test_parametrize_covers_all_platforms_and_boundary_sources():
    """Reflective check on the parametrize coverage. The cross-
    platform parity invariant requires 4 × 2 = 8 cases; any drop
    breaks the contract per the AC-8 anti-mutation drill."""
    import inspect
    src = inspect.getsource(test_manifest_fields_uniform_across_platforms)
    # Count platform values
    assert src.count("'craft'") >= 1
    assert src.count("'pythia'") >= 1
    assert src.count("'acea'") >= 1
    assert src.count("'sarra_py'") >= 1
    # Count boundary_source values
    assert src.count("'gadm'") >= 1
    assert src.count("'manual'") >= 1
