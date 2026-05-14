"""Canonical predicate for real-climate-cell IDs vs path-dict / placeholder keys.

The UnifiedData ``climate`` and ``soil`` dicts carry two different
vocabularies depending on the upstream pipeline:

* **Real-cell mode** (CRAFT / PYTHIA / ACEA / harmonize stage) — keys
  are non-negative integers (or ``numpy.int64``) that name actual
  grid cells, plus an occasional ``-1`` sentinel for the placeholder
  produced by the retrieve stage before real cells are surfaced.
* **Path-dict mode** (SARRA-Py ``_load_climate_data``) — keys are
  strings naming on-disk artefacts (``"rainfall_dir"``,
  ``"agera5_dir"``, ``"metadata"``) rather than grid cells.

Code that mixes the two vocabularies in a single ``sorted()`` /
``set | set`` / ``min`` operation crashes on Python 3 strict
comparison. The cockpit observed-values writer hit exactly this
mode in production (``int < str`` TypeError on the union of
``climate.keys()`` and ``soil.keys()``); the fix is to filter via
the helper below before any cross-type operation.

This module lives at the neutral ``prismpy.cells`` layer so both
``prismpy.cockpit`` and ``prismpy.translators`` can import it
without creating a cycle.

See F-DL contract + AC-DL-1 for the contract.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any


def is_real_climate_cell_id(cell_id: Any) -> bool:
    """Return ``True`` iff ``cell_id`` names a real grid cell.

    Accepts:

    * Native ``int`` ≥ 0 (including the ``0`` placeholder used by
      ``_create_placeholder_soil`` at ``executor.py:1387`` and by
      ``SpatialGrid.compute_cell_id_5arcmin`` for the first grid
      cell of a region).
    * Any subclass of ``numbers.Integral`` (covers ``numpy.int64``
      / ``numpy.int32`` that the pandas / xarray pipelines may
      surface). Production code may emit ``numpy.int64`` cell IDs
      that are NOT ``isinstance(cid, int)`` but ARE
      ``isinstance(cid, Integral)``.

    Rejects:

    * Path-dict ``str`` keys (``"rainfall_dir"``, ``"agera5_dir"``,
      ``"metadata"``) emitted by SARRA-Py's ``_load_climate_data``.
    * Stringified cell IDs (``"3799258"``) — those need explicit
      coercion before reaching this predicate.
    * Negative sentinels (e.g., the ``-1`` placeholder retained by
      ``executor.py:803-810`` and dropped at
      ``translators/base.py``).
    * ``None``, ``float``, ``bytes``, list/tuple, and any other
      non-integral type.
    * ``True`` / ``False``: in Python ``isinstance(True, int)``
      returns ``True``, so the explicit ``bool`` guard prevents
      booleans from masquerading as cell IDs.

    Used by:

    * ``prismpy.cockpit.observed_values_writer`` before the
      ``set(climate.keys()) | set(soil.keys())`` union → ``sorted``
      chain (F-DL AC-DL-2 fix site).
    * ``prismpy.translators.base`` in the per-cell climate
      surfacing helper and the ``crop_calendar`` fanout (F-DL
      AC-DL-3 sites; replaces the older inline
      ``isinstance(cid, int) and cid >= 0`` guards).
    """
    # ``bool(...)`` coerces the chained-AND result to Python ``True`` /
    # ``False``. Without it, ``cell_id >= 0`` on a ``numpy.int64``
    # returns ``numpy.True_`` (which compares equal to ``True`` but
    # is NOT ``True is True`` for downstream identity checks).
    return bool(
        not isinstance(cell_id, bool)
        and isinstance(cell_id, Integral)
        and cell_id >= 0
    )
