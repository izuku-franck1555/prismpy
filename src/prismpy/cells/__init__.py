"""Cell-summary schema layer.

The cell-summary record is the per-cell quality + provenance
payload prismpy emits for every grid cell. Until v2.1 the record
was a free-form dict assembled in `pipeline/executor.py`; this
module formalises the shape with a Pydantic model so downstream
consumers (prismweb /results/, /cockpit/, future analysis tools)
can rely on cross-field invariants rather than ad-hoc reads.

Re-exports the canonical schema model + version constants so a
caller does `from prismpy.cells import CellSummary` instead of
reaching into the schema submodule directly.
"""
from prismpy.cells.schema import (
    CELL_SUMMARY_VERSION_LATEST,
    CELL_SUMMARY_VERSIONS,
    CellSummary,
    DataAvailability,
    UnavailableReason,
)

__all__ = [
    "CELL_SUMMARY_VERSION_LATEST",
    "CELL_SUMMARY_VERSIONS",
    "CellSummary",
    "DataAvailability",
    "UnavailableReason",
]
