"""Canonical sentinel discipline for climate-cell IDs.

The pipeline executor at ``pipeline/executor.py::_create_placeholder_climate``
injects a single sentinel-keyed entry when no real climate data is available
at the retrieve stage. That sentinel lets ACEA / PYTHIA / CRAFT translators
pass the ``validate_input_data`` gate so their translate-time download paths
can run; it MUST NOT count as real climate coverage anywhere downstream.

Downstream consumers (translators, validators, manifest / provenance writers)
import ``is_real_climate_cell_id`` from this module and use it to filter the
sentinel entry out of real-coverage counts. The single canonical helper:

  - centralizes the sentinel convention (per durable §24 canonical-source);
  - is **non-int safe** (codex Draft 4 V5 catch) so it works on mixed-shape
    climate dicts — SARRA-Py's path-dict variants carry string keys like
    ``"rainfall_dir"`` / ``"agera5_dir"``; a raw ``cid >= 0`` filter would
    ``TypeError`` on those keys. Returning ``False`` for non-int keys keeps
    the helper safe at every metadata-writer site (AC-F-CP-13.5) and at the
    base translator's calendar fan-out (``translators/base.py``).

This module deliberately keeps imports to ``typing`` only so it can be
imported from inside heavy executor / translator modules without
introducing circular-import cascades — the
``test_circular_import_safety`` pin enforces that contract.
"""
from __future__ import annotations

from typing import Final


# Canonical sentinel ID for placeholder climate entries. The executor's
# ``_create_placeholder_climate`` returns a single-entry dict keyed by this
# constant; consumers filter it out via ``is_real_climate_cell_id``.
PLACEHOLDER_CLIMATE_SENTINEL_ID: Final[int] = -1


def is_real_climate_cell_id(key) -> bool:
    """Return True iff ``key`` is a real integer grid-cell ID.

    Real grid-cell IDs are non-negative ints (5-arcmin: 0..9_331_199;
    30-arcmin: 0..259_199). The placeholder sentinel
    ``PLACEHOLDER_CLIMATE_SENTINEL_ID`` (-1) returns ``False``; so does any
    non-int key (e.g., SARRA-Py path-dict shape ``{"rainfall_dir": ...,
    "agera5_dir": ...}``).

    Examples:
        >>> is_real_climate_cell_id(0)
        True
        >>> is_real_climate_cell_id(100)
        True
        >>> is_real_climate_cell_id(-1)  # PLACEHOLDER_CLIMATE_SENTINEL_ID
        False
        >>> is_real_climate_cell_id("rainfall_dir")  # SARRA-Py path-dict
        False
        >>> is_real_climate_cell_id(None)
        False
    """
    return isinstance(key, int) and key >= 0
