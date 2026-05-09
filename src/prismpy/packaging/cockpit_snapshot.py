"""Pure transformation: active CellDecisionRecord dict → JSON-stable
config snapshot.

Sprint E.2 §0.2 canonical-source #4 (prismpy half) + AC-E2-12.
Sprint E.3 AC-E3-14 (post-Draft 3 codex HIGH-1 + MEDIUM-1) extends
the input shape to tuple-keyed
``dict[Tuple[CellID, str], ...]`` matching the
``current_decisions()`` reshape at AC-E3-6, the output shape to
nested ``cockpit_decisions_at_launch[<cell_id>][<check_id>] =
record_dict`` to preserve multi-check coexistence per cell, and
adds a parallel ``cockpit_overrides_at_launch`` block (Extension
1: only ``document_override`` decisions) that the HARMONIZE-end
sidecar writer at
:mod:`prismpy.cockpit.cockpit_overrides_writer` consumes.

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

**Dual-shape loader** (AC-E3-14 sub-6 absorbed): legacy pre-E.3
snapshots emitted ``cockpit_decisions_at_launch`` in flat
``{cell_id: record}`` shape; post-E.3 snapshots use the nested
``{cell_id: {check_id: record}}`` shape. The
:func:`deserialize_decisions_from_config` reader detects the shape
on read and rehydrates either path into the canonical tuple-keyed
dict per AC-E3-6.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from prismpy.models.decision_log import CellDecisionRecord
from prismpy.models.interpolated_cell import CellID


_LEGACY_CHECK_ID_SENTINEL = "unknown_legacy"


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
                },
                "cockpit_overrides_at_launch": {
                    "<cell_id>": {
                        "<check_id>": <serialized override row>,
                        ...
                    },
                    ...
                }
            }

        suitable for embedding into
        ``PipelineRun.config_snapshot``. The
        ``cockpit_overrides_at_launch`` block (per AC-E3-14
        Extension 1) carries only the rows whose
        ``action == "document_override"`` so the HARMONIZE-end
        sidecar writer at
        :mod:`prismpy.cockpit.cockpit_overrides_writer` reads a
        focused subset rather than re-walking every decision.
        Both blocks share the same nested shape; the override
        block is structurally a filtered view of the decisions
        block (no Cat D filtering yet — that lives at the
        writer).

        Outer keys (cell_id) ordered by str ascending, inner
        keys (check_id) ordered by str ascending — both for
        byte-stable output across calls.

    The function calls ``record.model_dump(mode="json")`` on each
    non-None value so the output is JSON-serializable (UUIDs → strs,
    datetimes → ISO 8601). ``None`` values pass through unchanged.

    Empty ``active`` dict yields::

        {
            "cockpit_decisions_at_launch": {},
            "cockpit_overrides_at_launch": {}
        }

    consumers per :func:`deserialize_decisions_from_config` (the
    AC-E3-14 sub-6 dual-shape loader) detect the nested vs flat
    shape on read.
    """
    # Group by cell_id outer key into both blocks. The decisions
    # block carries every record; the overrides block carries
    # only the document_override subset.
    decisions_grouped: dict[str, dict[str, Any]] = {}
    overrides_grouped: dict[str, dict[str, Any]] = {}

    for (cell_id, check_id), record in active.items():
        cell_key = str(cell_id)
        check_key = str(check_id)
        record_dict = (
            record.model_dump(mode="json") if record is not None else None
        )
        decisions_grouped.setdefault(cell_key, {})[check_key] = record_dict
        if record is not None and record.action == "document_override":
            overrides_grouped.setdefault(cell_key, {})[check_key] = record_dict

    return {
        "cockpit_decisions_at_launch": _sort_nested(decisions_grouped),
        "cockpit_overrides_at_launch": _sort_nested(overrides_grouped),
    }


def _sort_nested(grouped: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Order outer + inner keys for byte-stable output."""
    return {
        cell_key: {
            check_key: grouped[cell_key][check_key]
            for check_key in sorted(grouped[cell_key])
        }
        for cell_key in sorted(grouped)
    }


def deserialize_decisions_from_config(
    config_snapshot_block: Any,
) -> dict[Tuple[CellID, str], Optional[dict[str, Any]]]:
    """Dual-shape loader — rehydrates the
    ``cockpit_decisions_at_launch`` block back into a tuple-keyed
    dict per AC-E3-6.

    Sprint E.3 AC-E3-14 sub-6 (codex MEDIUM-1 absorbed). Legacy
    pre-E.3 snapshots emitted the flat ``{cell_id: record}`` shape;
    post-E.3 snapshots use the nested ``{cell_id: {check_id:
    record}}`` shape. The detection rule uses the type of the
    first inner value:

    * **Flat shape** — inner value is a record-dict (has the
      ``decision_id`` key from CellDecisionRecord). Hydrates as
      ``(cell_id, record["check_id"] or "unknown_legacy")``;
      legacy rows without a ``check_id`` get the sentinel per
      AC-E3-6 backfill priority.
    * **Nested shape** — inner value is a ``dict`` of dicts
      (outer key is a check_id string). Hydrates directly via
      ``[cell_id][check_id] = record_dict``.

    Empty ``config_snapshot_block = {}`` returns an empty tuple-
    keyed dict (sentinel-free).

    Args:
        config_snapshot_block: The
            ``cockpit_decisions_at_launch`` value from
            ``PipelineRun.config_snapshot``. None / non-dict
            inputs return an empty dict (defensive — caller may
            pass a missing-key result without an explicit guard).

    Returns:
        Tuple-keyed dict mapping ``(cell_id, check_id)`` to the
        record-dict (or None for fully-reverted entries). The
        record-dict itself is NOT re-validated against
        :class:`CellDecisionRecord` here — callers that need
        Pydantic semantics call ``CellDecisionRecord.model_validate``
        on individual entries. Keeping this loader Pydantic-free
        avoids re-fetching the validator chain for callers that
        only need the raw shape (e.g., the sidecar writer at
        AC-E3-7).
    """
    if not isinstance(config_snapshot_block, dict):
        return {}

    if not config_snapshot_block:
        return {}

    # Detect shape via the first inner value. We grab any cell's
    # first inner value — the snapshot writer guarantees
    # homogeneous shape per snapshot (one writer fires once with
    # one shape choice; mixed-shape within a single snapshot is
    # impossible per atomicity at the writer).
    first_cell = next(iter(config_snapshot_block))
    first_value = config_snapshot_block[first_cell]

    if not isinstance(first_value, dict):
        # Defensive — non-dict inner value isn't a recognised
        # shape. Empty result so the consumer doesn't crash.
        return {}

    # Flat shape signal: the first_value carries CellDecisionRecord-
    # required keys (decision_id is the most discriminative since
    # all records have it; the nested shape's inner dict carries
    # check_ids as keys, not decision_id).
    is_flat = "decision_id" in first_value or first_value == {} or first_value is None

    # Refine flat-shape detection: a non-record dict-of-dicts
    # would fail the "decision_id key" test correctly, but an
    # empty inner dict is ambiguous. Treat empty inner dicts as
    # nested-shape (more conservative; flat shape wouldn't have
    # an empty record).
    if first_value == {}:
        is_flat = False

    out: dict[Tuple[CellID, str], Optional[dict[str, Any]]] = {}

    if is_flat:
        # Legacy flat shape — one record per cell_id. Recover
        # check_id from the record itself if present; else
        # fallback to ``unknown_legacy`` sentinel per AC-E3-6
        # backfill priority.
        for cell_id, record_dict in config_snapshot_block.items():
            if record_dict is None:
                out[(str(cell_id), _LEGACY_CHECK_ID_SENTINEL)] = None
                continue
            if not isinstance(record_dict, dict):
                continue
            check_id = record_dict.get("check_id") or _LEGACY_CHECK_ID_SENTINEL
            out[(str(cell_id), str(check_id))] = record_dict
    else:
        # Nested shape — direct hydration.
        for cell_id, inner in config_snapshot_block.items():
            if not isinstance(inner, dict):
                continue
            for check_id, record_dict in inner.items():
                out[(str(cell_id), str(check_id))] = record_dict

    return out


__all__ = [
    "deserialize_decisions_from_config",
    "serialize_decisions_to_config",
]
