"""Pure transformation: active CellDecisionRecord dict → JSON-stable
config snapshot.

Sprint E.2 §0.2 canonical-source #4 (prismpy half) + AC-E2-12.
Sprint E.3 AC-E3-14 (post-Draft 3 codex HIGH-1) extends the input
shape to tuple-keyed ``dict[Tuple[CellID, str], ...]`` matching the
``current_decisions()`` reshape at AC-E3-6, and the output shape to
nested ``cockpit_decisions_at_launch[<cell_id>][<check_id>] = record_dict``
to preserve multi-check coexistence per cell without tuple-key
stringification ambiguity.

The prismweb side at ``core/services/pipeline_run_decisions.py
::commit_decision_snapshot`` calls this helper inside a
``transaction.atomic()`` block before creating the derived
``PipelineRun`` whose ``config_snapshot`` carries the serialized
decisions.

The function is PURE — no Django, no I/O, no global state. Per the
I-DN-1 architecture-split locked at Draft 5.1 §0.2 #4, prismpy owns
serialization (Pydantic → JSON) and prismweb owns Django persistence
(transaction + ORM create + redirect).

Determinism: outer keys (cell_id) sorted by ``str(cell_id)`` ascending
+ inner keys (check_id) sorted by ``str(check_id)`` ascending so the
output JSON is byte-stable across calls. Drill O verifies the
rollback symmetry empirically; this module's pin verifies the
serialization itself.

**Cat D filtering** (AC-E3-7 sidecar writer; lives at the writer not
here): the snapshot block carries every ``document_override``
decision regardless of category — the Cat D filter applies at the
HARMONIZE-end sidecar emission, not at config-snapshot
serialization, so the audit log retains the documentary basis for
methods-text use even though the translator sidecar omits it.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from prismpy.models.decision_log import CellDecisionRecord
from prismpy.models.interpolated_cell import CellID


def serialize_decisions_to_config(
    active: dict[Tuple[CellID, str], Optional[CellDecisionRecord]],
) -> dict[str, Any]:
    """Serialize the active-decisions dict to a JSON-stable config.

    Args:
        active: Output of ``current_decisions()`` — a dict mapping
            each ``(cell_id, check_id)`` pair with a decision in
            scope to its current active record (or ``None`` if all
            decisions on the pair were reverted). Sprint E.3
            AC-E3-6 reshape: keyed on tuples for multi-check
            coexistence per cell.

    Returns:
        Dict with shape::

            {
                "cockpit_decisions_at_launch": {
                    "<cell_id>": {
                        "<check_id>": <serialized record or None>,
                        ...
                    },
                    ...
                }
            }

        suitable for embedding into ``PipelineRun.config_snapshot``.
        Outer keys (cell_id) ordered by str ascending, inner keys
        (check_id) ordered by str ascending — both for byte-stable
        output across calls.

    The function calls ``record.model_dump(mode="json")`` on each
    non-None value so the output is JSON-serializable (UUIDs → strs,
    datetimes → ISO 8601). ``None`` values pass through unchanged.

    Empty ``active`` dict yields ``{"cockpit_decisions_at_launch":
    {}}``; consumers per AC-E3-14 sub-6 dual-shape loader (lives at
    Boundary 3) detect the nested vs flat shape on read.
    """
    # Group records by cell_id outer key. Pythonic + deterministic
    # (we sort within each cell's inner dict at emission time).
    grouped: dict[str, dict[str, Any]] = {}
    for (cell_id, check_id), record in active.items():
        cell_key = str(cell_id)
        check_key = str(check_id)
        record_dict = (
            record.model_dump(mode="json") if record is not None else None
        )
        grouped.setdefault(cell_key, {})[check_key] = record_dict

    # Sort outer + inner keys for deterministic byte-stable output.
    return {
        "cockpit_decisions_at_launch": {
            cell_key: {
                check_key: grouped[cell_key][check_key]
                for check_key in sorted(grouped[cell_key])
            }
            for cell_key in sorted(grouped)
        }
    }


__all__ = [
    "serialize_decisions_to_config",
]
