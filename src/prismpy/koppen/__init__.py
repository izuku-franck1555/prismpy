"""Köppen-Geiger climate-zone substrate.

UNRELATED to the existing :mod:`prismpy.utils.zones` package,
which carries the management-zone vocabulary used by
translators when applying spatial management variability. This
package ships the Köppen-Geiger climate-zone classifier built
on the Beck et al. 2023 raster (Sci Data 10:724) and the
Stage 1 wizard-time crop-region compatibility check that
compares a project's per-cell climate distributions against
ECOCROP envelopes.

Sprint E.0.5 (V2-22c-RESTART Phase 2) ships:

* :data:`ECOCROP_ENVELOPE_PATH` — path to the
  ``ecocrop_envelopes.json`` data file (TMIN/TMAX/RMIN/RMAX
  per crop; no other ECOCROP fields per AC-Q3-A-d).
* :func:`load_ecocrop_envelopes` — reads + validates the
  JSON; raises on NaN values, RMIN ≥ RMAX / TMIN ≥ TMAX
  (AC-Q3-A-NaN strict-ordering pin), or missing /
  malformed per-crop provenance block (F28).
* (Sprint F populates) :class:`KGClassifier` — Beck 2023
  raster loader + zone classification + transitional-cell
  detection.

Out-of-scope ECOCROP fields (Sprint F or V3 territory):
ALTMX (max altitude), pH range, photoperiod, GMIN/GMAX
growing-day range, latitude range. A subsequent Sprint
E.0.5 commit lands an F27 AST walker at
``tests/structural/test_stage1_scope_walker.py`` to enforce
the precip-tmin-tmax-only discipline at module-code time;
this commit's bundled JSON enforces the discipline at the
data layer via ``test_no_out_of_scope_fields_in_any_crop``
in ``tests/structural/test_ecocrop_envelopes.py``.
"""
from __future__ import annotations

from prismpy.koppen.envelopes import (
    ECOCROP_ENVELOPE_PATH,
    EnvelopeValidationError,
    load_ecocrop_envelopes,
)


__all__ = [
    "ECOCROP_ENVELOPE_PATH",
    "EnvelopeValidationError",
    "load_ecocrop_envelopes",
]
