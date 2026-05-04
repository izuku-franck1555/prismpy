"""Bound-generation public constants per Sprint E.0.5 AC-Q2-A1-a.

Public counterpart of the private cutoff offset used inside
:mod:`prismpy.bounds.provenance`. Keeps the substrate's
180-day cutoff math discoverable from the package root for
downstream consumers (bound-gen tooling, Methods-text
generation, future ratchet checks).

Per AC-Q2-A1-Reframe, the cutoff math is framed as
"up to 120-day AgERA5 lag accommodation; 90+ days margin
under pessimistic 30-day estimate" — see the field
description on
:attr:`prismpy.bounds.BoundGenProvenance.agera5_record_cutoff`.
"""
from __future__ import annotations

from typing import Final


# Number of days AgERA5 records must lag the bound-gen
# snapshot date to count for inclusion. Anti-mutation: a
# bound-gen change that lowers this floor breaks the
# AC-Q2-A1-a + AC-Q2-A1-Reframe lag-margin contract; the
# structural test pins the constant at exactly 180.
AGERA5_RECORD_CUTOFF_DAYS: Final[int] = 180
