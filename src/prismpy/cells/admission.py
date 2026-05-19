"""Producer-boundary canonical admission helper for shape-polymorphic
climate-data dicts.

Every translator's translate-time admission decision routes through
``canonical_climate_for_grid`` so admission discipline lives in ONE
site that every consumer trusts. Ad-hoc per-callsite filters in
consumer code are forbidden by the structural pin
(``TestAdmissionRoutingMetaPin`` in
``tests/structural/test_canonical_admission.py``): every production
``is_real_climate_cell_id`` callsite must either live in a file that
imports ``canonical_climate_for_grid`` from this module OR live in
an explicit allowlist of read-only metadata-counter sites.

The helper is the structural answer to a class of producer-consumer
admission drift bugs in which downstream consumers each apply slightly
different tightness bars to the same shape-polymorphic substrate
(``Dict[Any, ClimateTimeSeries]`` whose keys can be 5-arcmin grid
cell IDs, 30-arcmin parent tile IDs, the placeholder sentinel ``-1``,
or path-dict strings, and whose values may carry zero / one / many
records). Each consumer-side filter is a potential drift surface;
one canonical helper at the producer boundary closes the class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Mapping, Optional, Set, TYPE_CHECKING

from prismpy.cells.cell_id_validation import is_real_climate_cell_id

if TYPE_CHECKING:
    from prismpy.harmonize.unified import SpatialGrid
    from prismpy.sources.climate.types import ClimateTimeSeries


# 5-arcmin grid: 4320 cols x 2160 rows. 30-arcmin grid: 720 cols x 360
# rows. Reduction factor 6x per axis. Constants exported so tests +
# callers can reuse the canonical grid math without re-declaring it.
GRID_COLS_5ARCMIN = 4320
GRID_COLS_30ARCMIN = 720
_REDUCTION_FACTOR = 6


def cell_id_5arcmin_to_30arcmin_parent(cell_id_5: int) -> int:
    """Map a 5-arcmin cell ID to its parent 30-arcmin tile ID.

    Each 30-arcmin tile contains a 6x6 grid of 5-arcmin cells; the
    reduction is integer division by 6 on each axis. Moved here from
    ``acea/translator.py`` so the mapping is a module-level helper
    available to every translator + the canonical admission helper
    without bouncing through an ACEA-instance method.
    """
    row_5 = cell_id_5 // GRID_COLS_5ARCMIN
    col_5 = cell_id_5 % GRID_COLS_5ARCMIN
    row_30 = row_5 // _REDUCTION_FACTOR
    col_30 = col_5 // _REDUCTION_FACTOR
    return row_30 * GRID_COLS_30ARCMIN + col_30


def _admissible_series(
    key: Any,
    series: Any,
    *,
    require_multi_record: bool,
) -> bool:
    """Admissibility predicate for a single ``(key, series)`` pair.

    True iff the key is a real-int grid-cell ID AND the series has a
    ``records`` attribute AND the record list passes the caller's
    records-validity bar. Default ``require_multi_record=True`` admits
    only multi-record series (the strictest downstream consumer's bar
    for ``.WTH`` and pickle writers); ``False`` admits any non-empty
    records list (reserved for read-only metadata counters).
    """
    if not is_real_climate_cell_id(key):
        return False
    if not hasattr(series, "records"):
        return False
    records = series.records
    if records is None:
        return False
    if require_multi_record:
        return len(records) > 1
    return bool(records)


@dataclass(frozen=True)
class CanonicalClimate:
    """Canonical view of an admissible per-cell climate dict.

    ``per_cell`` holds the admissible entries keyed by 5-arcmin grid
    cell ID. ``covered_tile_ids`` is the lazily-computed set of
    30-arcmin parent tile IDs that have at least one admitted child.
    Frozen so callers cannot rebind the fields by accident; the
    cached property is fine under ``frozen=True`` because
    ``functools.cached_property`` writes the cache directly to
    ``instance.__dict__`` and bypasses the frozen ``__setattr__``
    guard. (See `the CPython implementation
    <https://docs.python.org/3/library/functools.html#functools.cached_property>`_:
    the descriptor mutates the instance dict, not via setattr.)
    """

    per_cell: Dict[int, "ClimateTimeSeries"] = field(default_factory=dict)

    @cached_property
    def covered_tile_ids(self) -> Set[int]:
        """30-arcmin parent tile IDs of every admitted cell.

        Computed once on first access and cached for the lifetime of
        the instance. Callers iterate ``per_cell.keys()`` and fold
        each through ``cell_id_5arcmin_to_30arcmin_parent`` without
        having to import the parent helper directly.
        """
        return {
            cell_id_5arcmin_to_30arcmin_parent(cell_id)
            for cell_id in self.per_cell.keys()
        }


def canonical_climate_for_grid(
    climate_data: Optional[Mapping[Any, "ClimateTimeSeries"]],
    grid: Optional["SpatialGrid"],
    *,
    require_multi_record: bool = True,
) -> CanonicalClimate:
    """Return the canonical per-grid-cell admission view of ``climate_data``.

    Admits only entries from known key spaces, both derived from
    ``grid.cells``:

    - **Direct per-cell admission**: a ``climate_data`` key matches a
      5-arcmin ``cell.cell_id`` from ``grid.cells``. The series writes
      straight to ``per_cell[cell_id]``.

    - **Tile-keyed broadcast admission**: a ``climate_data`` key
      matches a known 30-arcmin parent tile of some ``cell.cell_id``.
      The series fans out to every child cell of that tile, skipping
      cells already populated via the direct path so per-cell distinct
      series are preserved when both per-cell + tile-keyed inputs
      coexist in the same input.

    Foreign keys (not in either trusted space) are dropped WITHOUT
    folding. Dropping foreign keys at the boundary is the structural
    closure of the parent-fold-rescue class of admission bugs:
    ad-hoc parent-fold discriminators at downstream consumer sites
    can silently masquerade a foreign key as in-region coverage when
    the fold-by-coincidence lands on a target tile (``fold(600) ==
    100`` collides with target tile ``100``; the rescue intersection
    then "matches"). The helper rejects foreign keys at admission so
    the masquerade is structurally impossible.

    The records-validity bar applies the strictest downstream
    consumer's requirement by default (``len > 1`` — multi-record
    needed by ``.WTH`` and pickle writers). Single-record / empty /
    missing-records-attr entries are dropped. Pass
    ``require_multi_record=False`` (reserved for future read-only
    metadata counters) to admit any non-empty records list.

    Returns an empty ``CanonicalClimate`` when ``climate_data`` is
    None/empty OR when ``grid`` is None or has no cells; callers fall
    through cleanly (downstream coverage validators report
    missing-climate honestly).

    Args:
        climate_data: Producer-emitted climate dict (may carry mixed
            shapes: 5-arcmin / 30-arcmin / sentinel / path-dict).
        grid: Trusted source of admission key spaces (5-arcmin cells +
            their 30-arcmin parents).
        require_multi_record: If True (default), drop entries whose
            ``records`` list has fewer than two entries. If False,
            admit any non-empty records list.

    Returns:
        ``CanonicalClimate`` instance carrying ``per_cell`` keyed by
        ``cell.cell_id`` and ``covered_tile_ids`` as a cached set of
        30-arcmin parent tiles with at least one admitted child.
    """
    if not climate_data or grid is None or not grid.cells:
        return CanonicalClimate()

    # Build trusted admission spaces from grid.cells. The two spaces
    # are the only key shapes the helper will admit from climate_data.
    cell_id_to_tile: Dict[int, int] = {
        cell.cell_id: cell_id_5arcmin_to_30arcmin_parent(cell.cell_id)
        for cell in grid.cells
    }
    grid_cell_ids = set(cell_id_to_tile.keys())
    grid_tile_ids = set(cell_id_to_tile.values())

    # Two-pass admission. Direct per-cell first so per-cell-keyed input
    # preserves distinct series across sibling cells that share a
    # parent tile (an upstream producer emitting per-cell-distinct
    # 5-arcmin series would otherwise collapse into a last-write-wins
    # tile slot if we fed both passes into a single tile-keyed lookup).
    canonical: Dict[int, "ClimateTimeSeries"] = {}
    tile_lookup: Dict[int, "ClimateTimeSeries"] = {}
    for key, series in climate_data.items():
        if not _admissible_series(
            key, series, require_multi_record=require_multi_record
        ):
            continue
        if key in grid_cell_ids:
            canonical[key] = series
        elif key in grid_tile_ids:
            tile_lookup[key] = series
        # else: foreign key (not in either trusted space); drop without
        # folding. Foreign-fold-coincidence cannot enter the canonical
        # dict because admission requires explicit key-space membership.

    # Tile-keyed fan-out: every child cell of an admitted tile inherits
    # the tile's series unless Pass 1 already populated it.
    for cell in grid.cells:
        if cell.cell_id in canonical:
            continue
        tile = cell_id_to_tile[cell.cell_id]
        series = tile_lookup.get(tile)
        if series is not None:
            canonical[cell.cell_id] = series

    return CanonicalClimate(per_cell=canonical)


__all__ = [
    "GRID_COLS_5ARCMIN",
    "GRID_COLS_30ARCMIN",
    "CanonicalClimate",
    "canonical_climate_for_grid",
    "cell_id_5arcmin_to_30arcmin_parent",
]
