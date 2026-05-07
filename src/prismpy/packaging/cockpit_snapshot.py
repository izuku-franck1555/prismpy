"""Pure transformation: active CellDecisionRecord dict → JSON-stable
config snapshot.

Sprint E.2 §0.2 canonical-source #4 (prismpy half) + AC-E2-12. The
prismweb side at ``core/views/cockpit.py::cockpit_prepare()`` calls
this helper inside a ``@transaction.atomic`` block before creating
the derived ``PipelineRun`` whose ``config_snapshot`` carries the
serialized decisions.

The function is PURE — no Django, no I/O, no global state. Per the
I-DN-1 architecture-split locked at Draft 5.1 §0.2 #4, prismpy owns
serialization (Pydantic → JSON) and prismweb owns Django persistence
(transaction + ORM create + redirect).

Determinism: keys are sorted by ``str(cell_id)`` ascending so the
output JSON is byte-stable across calls. Drill O verifies the
rollback symmetry empirically; this module's pin verifies the
serialization itself.
"""

from __future__ import annotations

from typing import Any, Optional

from prismpy.models.decision_log import CellDecisionRecord
from prismpy.models.interpolated_cell import CellID


def serialize_decisions_to_config(
    active: dict[CellID, Optional[CellDecisionRecord]],
) -> dict[str, Any]:
    """Serialize the active-decisions dict to a JSON-stable config.

    Args:
        active: Output of ``current_decisions()`` — a dict mapping
            each cell with a decision in scope to its current
            active record (or ``None`` if all decisions on the cell
            were reverted).

    Returns:
        Dict with shape ``{"cockpit_decisions_at_launch": {<cell_id>:
        <serialized record or None>}}`` suitable for embedding into
        ``PipelineRun.config_snapshot``. Keys ordered by cell_id
        ascending for deterministic output across calls.

    The function calls ``record.model_dump(mode="json")`` on each
    non-None value so the output is JSON-serializable (UUIDs → strs,
    datetimes → ISO 8601). ``None`` values pass through unchanged.
    """
    return {
        "cockpit_decisions_at_launch": {
            str(cell_id): (
                record.model_dump(mode="json") if record is not None else None
            )
            for cell_id, record in sorted(active.items(), key=lambda kv: str(kv[0]))
        }
    }


__all__ = [
    "serialize_decisions_to_config",
]
