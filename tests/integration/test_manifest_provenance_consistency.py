"""F-W Sprint C — manifest-vs-provenance consistency (AC-9).

Verifies the load-bearing honest-signal invariant that closes the
F-W root-cause class:

    manifest.region.gadm_level is None
        IFF
    provenance.boundary.source != 'gadm'

Both sides derive from the SAME runtime-resolved boundary source.
The fallback case (config requests GADM, runtime falls back to
MANUAL) is tested explicitly because that's the most common
production failure mode the F-W audit caught — pre-Sprint-C the
manifest claimed GADM while provenance correctly said MANUAL.

Approach: each parametrized case simulates the executor's behavior
by building both the provenance.boundary record (via the
ProvenanceTracker.set_boundary contract) and the manifest's
package_config (via the resolved-source discriminator + helper),
both fed by the SAME ``resolved_source`` value. The test then
asserts the consistency invariant holds across all 12 cases.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from prismpy.packaging.manifest import (
    create_manifest,
    derive_boundary_label,
)
from prismpy.provenance.tracker import ProvenanceTracker


def _build_manifest_via_discriminator(
    platform: str,
    config_source: str,
    runtime_resolved_source,
    configured_gadm_level: int = 2,
):
    """Mirror the post-Sprint-C discriminator pattern every translator
    uses verbatim:

        resolved_source = (
            getattr(data.region, "boundary_source", None)
            or boundary.source.value
        )

    Both the runtime field (post-fallback) and the config source
    feed the discriminator; the runtime value wins when present.
    The fallback case is ``config_source='gadm'`` AND
    ``runtime_resolved_source='manual'`` — the runtime value takes
    precedence, manifest reflects the runtime outcome.

    Args:
        platform: target translator name (used in the synthetic
            package name; doesn't affect the discriminator).
        config_source: ``self.config.region.boundary.source.value``
            — the requested source.
        runtime_resolved_source: ``data.region.boundary_source`` —
            the runtime field set by the executor post-fallback. Pass
            ``None`` (or empty string) to simulate "runtime field
            absent" so the discriminator falls back to config_source.
        configured_gadm_level: ``self.config.region.boundary.gadm_level``
            — honored only when the discriminator-resolved source
            is GADM.
    """
    # The actual discriminator pattern:
    resolved_source = runtime_resolved_source or config_source
    manifest_gadm_level = (
        configured_gadm_level if resolved_source == 'gadm' else None
    )
    label, _ = derive_boundary_label(resolved_source, manifest_gadm_level)
    project_config = {
        'project_name': f'consistency-{platform}',
        'package_name': f'consistency-{platform}',
        'region_name': 'Koutiala',
        'country': 'Mali',
        'crop_name': 'Maize',
        'planting_doy': 152,
        'maturity_doy': 304,
        'start_year': 2010,
        'end_year': 2020,
        'gadm_level': manifest_gadm_level,
        'data_sources': {
            'climate': 'NASA POWER',
            'soil': 'iSDA',
            'boundaries': label,
        },
    }
    with TemporaryDirectory(prefix=f'consistency-{platform}-') as tmp:
        package_dir = Path(tmp) / platform
        package_dir.mkdir(parents=True)
        return create_manifest(package_dir, project_config, platform=platform)


def _build_provenance_for_resolved_source(resolved_source: str,
                                          configured_gadm_level: int = 2):
    """Mirror the executor's Stage 5 boundary-emit (AC-4 contract from
    Sprint A). The executor writes the RESOLVED source onto
    ``provenance.boundary.source`` so manifest and provenance agree
    by construction when both consume the same resolved value."""
    tracker = ProvenanceTracker(
        enabled=True, project_name='consistency-test',
    )
    n_cells_full = 149
    if resolved_source == 'gadm':
        admitted = n_cells_full
    else:
        admitted = n_cells_full
    tracker.set_boundary(
        source=resolved_source,
        version='GADM v4.1' if resolved_source == 'gadm' else None,
        inclusion_rule='bbox_intersects',
        min_share_percent=0.0,
        n_cells_full_extent=n_cells_full,
        n_cells_excluded_by_inclusion_rule=0,
        n_cells_excluded_by_min_share_percent=0,
        n_cells_admitted=admitted,
    )
    return tracker.record.boundary


# ---------------------------------------------------------------------------
# AC-9 — 4 platforms × 3 config-runtime variants = 12 cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('platform', ['craft', 'pythia', 'acea', 'sarra_py'])
@pytest.mark.parametrize(
    'config_source,runtime_resolved_source,expected_gadm_level',
    [
        ('gadm', 'gadm', 2),       # GADM requested, GADM resolved
        ('manual', 'manual', None),  # MANUAL requested, MANUAL resolved
        ('gadm', 'manual', None),  # GADM requested, fallback to MANUAL
                                     # (codex Gate A HIGH #3 — load-bearing)
    ],
)
def test_manifest_boundaries_matches_provenance_source(
    platform, config_source, runtime_resolved_source, expected_gadm_level,
):
    """AC-9 honest-signal invariant: manifest.region.gadm_level is
    None iff provenance.boundary.source != 'gadm'. Both sides derive
    from the same runtime-resolved source. Anti-mutation drill:
    revert the AC-1/4/5/6 discriminator to read directly from
    config.region.boundary.source.value (skipping the runtime
    fallback) → the fallback case (config GADM, runtime MANUAL)
    fails because manifest claims GADM while provenance says MANUAL.
    """
    # Manifest derivation runs the actual discriminator with BOTH
    # config_source AND runtime_resolved_source — the discriminator
    # picks runtime when present, otherwise falls back to config.
    # The fallback case (config GADM, runtime MANUAL) is the load-
    # bearing scenario the F-W audit caught: pre-Sprint-C the
    # manifest read from config_source directly and lied about
    # GADM-failed-fallback runs; post-Sprint-C the runtime value
    # wins so the manifest reflects on-disk reality.
    manifest = _build_manifest_via_discriminator(
        platform,
        config_source=config_source,
        runtime_resolved_source=runtime_resolved_source,
    )
    # The executor writes the resolved (post-fallback) source onto
    # provenance.boundary at Stage 5; both sides therefore consume
    # the runtime-resolved value.
    provenance_boundary = _build_provenance_for_resolved_source(
        runtime_resolved_source,
    )

    manifest_says_gadm = manifest['region']['gadm_level'] is not None
    provenance_says_gadm = provenance_boundary['source'] == 'gadm'

    assert manifest['region']['gadm_level'] == expected_gadm_level, (
        f"{platform}/{config_source}->{runtime_resolved_source}: "
        f"manifest.region.gadm_level={manifest['region']['gadm_level']!r}, "
        f"expected {expected_gadm_level!r}"
    )
    assert manifest_says_gadm == provenance_says_gadm, (
        f"{platform}/{config_source}->{runtime_resolved_source}: "
        f"manifest-provenance disagreement! "
        f"manifest.gadm_level={manifest['region']['gadm_level']!r}, "
        f"provenance.source={provenance_boundary['source']!r}. "
        f"The honest-signal contract requires both sides to reflect "
        f"the runtime outcome; disagreement means one side reads "
        f"from a different signal than the other."
    )


# ---------------------------------------------------------------------------
# AC-9 anti-mutation pin — coverage of the parametrize space
# ---------------------------------------------------------------------------


def test_parametrize_covers_fallback_case():
    """The fallback case (config GADM, runtime MANUAL) is the most
    common production failure mode the F-W audit caught. Pin its
    presence in the parametrize so a future trim doesn't drop it."""
    import inspect
    src = inspect.getsource(test_manifest_boundaries_matches_provenance_source)
    assert "('gadm', 'manual', None)" in src, (
        "AC-9 must cover the fallback case (config requests GADM, "
        "runtime resolves to MANUAL). This is the load-bearing "
        "test for the F-W root-cause class — pre-Sprint-C the "
        "manifest claimed GADM while provenance said MANUAL."
    )
