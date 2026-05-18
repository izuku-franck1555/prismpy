"""Back-compat re-export for ``is_real_climate_cell_id``.

The canonical home for the placeholder sentinel and the real-cell
predicate is now ``prismpy.sources.climate._sentinels``. This module
re-exports the predicate from that canonical location so existing
consumers (cockpit observed-values writer, base translator's per-cell
climate surfacing helper and crop-calendar fan-out) keep working
without changing their imports.

See ``prismpy.sources.climate._sentinels`` for the canonical
definition + the bool-guard / ``Integral`` semantics required by
production code that surfaces ``numpy.int64`` cell IDs.
"""
from __future__ import annotations

from prismpy.sources.climate._sentinels import is_real_climate_cell_id

__all__ = ["is_real_climate_cell_id"]
