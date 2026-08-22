"""Soil-retrieval unblock (Slice-1) — HWSD provenance fail-safe pins.

Two contracts, both mockable (no BIL/MDB rasters needed — the raster/MDB readers
are stubbed, mirroring the existing unit-level soil tests):

  * AC-1: ``DecisionType`` is imported in hwsd.py, so the SMU-lookup-miss
    provenance decision is RECORDED. Previously the name was undefined, so a
    NameError raised at the record_decision call wiped every built profile.
  * AC-2 (structural): a provenance/tracker failure during that decision must
    NEVER discard the soil profiles already built — ``return profiles`` runs.

The bug detonated whenever a cell set contained >=1 SMU-lookup-miss (a water /
urban / gap cell, common on coastal or mixed terrain) and provenance was wired.
"""
from __future__ import annotations

import types

import pandas as pd

from prismpy.models.provenance import DecisionType
from prismpy.sources.soil.hwsd import HWSDConfig, HWSDSource


# _create_profile_from_hwsd only reads ``region.country_iso3``.
_REGION = types.SimpleNamespace(country_iso3="PRT")
# Two coords: the first samples to a valid SMU (10298, present in the table), the
# second to a NODATA miss (None) -> n_unavailable == 1 -> the SMU-miss provenance
# block runs. (Mafra, Portugal — the reported region.)
_COORDS = [(39.04, -9.375), (39.04, -9.458)]


def _mdb_frame() -> pd.DataFrame:
    """A minimal HWSD2_LAYERS table with the topsoil (D1, SEQUENCE 1) row for SMU 10298."""
    return pd.DataFrame({
        "HWSD2_SMU_ID": [10298],
        "LAYER": ["D1"],
        "SEQUENCE": [1],
        "SAND": [40.0], "CLAY": [30.0], "SILT": [30.0],
        "OC": [1.0], "PH": [6.5], "BULK_DENSITY": [1.4],
    })


def _source(provenance) -> HWSDSource:
    src = HWSDSource(
        config=HWSDConfig(bil_path="x.bil", mdb_path="x.mdb", use_defaults=True),
        provenance=provenance,
    )
    # Stub the raster + MDB readers so no BIL/MDB files are needed: one valid SMU,
    # one NODATA miss (None).
    src._sample_bil_raster = lambda coords: [10298, None]
    src._export_mdb_table = _mdb_frame
    return src


def test_provenance_error_does_not_discard_built_profiles():
    """AC-2 structural: a raising provenance tracker must NOT wipe the profiles
    already built. Unwrap the try/except in hwsd.py -> RED (the raise propagates
    out of _extract_from_bil_mdb and no profiles are returned)."""
    class RaisingProvenance:
        enabled = True

        def record_decision(self, **kwargs):
            raise RuntimeError("simulated tracker fault")

    src = _source(RaisingProvenance())
    profiles = src._extract_from_bil_mdb(region=_REGION, cell_coords=_COORDS)
    assert len(profiles) == 1        # the valid cell's profile survives the provenance fault
    assert 0 in profiles             # positional key of the valid coord


def test_smu_miss_decision_is_recorded_not_swallowed():
    """AC-1: with DecisionType imported, the SMU-miss FALLBACK_SUBSTITUTION
    decision is RECORDED. Remove the import in hwsd.py -> NameError at the
    record_decision call -> the AC-2 wrap swallows it -> the decision is never
    recorded -> RED. (This is the original prod bug's exact detonation site.)"""
    recorded = []

    class SpyProvenance:
        enabled = True

        def record_decision(self, **kwargs):
            recorded.append(kwargs.get("decision_type"))

    src = _source(SpyProvenance())
    profiles = src._extract_from_bil_mdb(region=_REGION, cell_coords=_COORDS)
    assert len(profiles) == 1                                   # the valid cell is built
    assert DecisionType.FALLBACK_SUBSTITUTION in recorded       # recorded, not NameError-swallowed
