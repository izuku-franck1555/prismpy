"""Per-zone climate bounds substrate (Sprint E.0.5 V2-22c-RESTART Phase 2).

Holds the substrate that pairs a Köppen-Geiger zone (per
:mod:`prismpy.koppen`) with a frozen-per-platform-release
climate envelope (per-zone P25/P50/P75 of annual precip and
P10/P90 of per-cell thermal extremes derived from AgERA5
1991-2020).

Sprint E.0.5 commit 7a ships:

* :class:`BoundGenProvenance` — Pydantic model that records
  the environment + ERA5 archive + dependency + thread-pin
  configuration of a bound-gen run (per AC-Q2-A1-c). Written
  as a sidecar JSON next to the bounds file so re-runs of
  the same ``bounds_version`` can verify byte-identical
  reproducibility against this record.

The actual per-zone bounds JSON files (``frozen_v1/...``) +
the bound-gen management command + the AgERA5 wiring land
in subsequent V2-22c sprint work; this package houses the
provenance model so the contract can be honored even before
the data substrate is generated.
"""
from __future__ import annotations

from prismpy.bounds.provenance import (
    BoundGenProvenance,
    DepositStatus,
    write_bound_gen_provenance,
)


__all__ = [
    "BoundGenProvenance",
    "DepositStatus",
    "write_bound_gen_provenance",
]
