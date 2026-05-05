"""Sprint E.1 AC-E1-2 — per-run Stage 1 verdict snapshot helper.

The Sprint F wizard-time snapshot lives on
``Project.stage_1_verdicts`` (mutable cross-run state — the
user can re-run the wizard and the field updates). The
cockpit + run-lineage rail need a frozen-at-launch copy on
each :class:`PipelineRun` so historical refinement runs render
the verdict the user actually launched against, not the
current Project-level state.

Sprint F's CC-26 invariant promised this snapshot but didn't
ship the write-site helper — it lived as an aspirational
``PipelineRun.config_snapshot.stage_1_verdicts_at_launch``
key referenced in
:func:`prismweb.core.services.wizard_validation.deepcopy_snapshot`'s
docstring without an actual writer. This module is the writer
the contract calls for.

The helper is deliberately minimal:

* :func:`snapshot_for_pipeline_run` deep-copies the active
  ``Project.stage_1_verdicts`` payload + freezes it under a
  byte-stable canonical form so the at-launch snapshot is
  comparable to a re-derivation at run-finish for drift
  detection.
* :func:`is_snapshot_unavailable` distinguishes the
  ``stage_1_unavailable`` operational-failure shape from a
  clean compatibility result with no Bucket 3 emit. The
  cockpit's per-run snapshot reader uses this to render
  "compatibility check did not run for this run" copy.

Per CC-30 the helper never re-derives the verdict — pure
read-from-snapshot consumer + deep-copy.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict


def snapshot_for_pipeline_run(
    project_stage_1_verdicts: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Deep-copy ``Project.stage_1_verdicts`` for a per-run
    frozen snapshot stored under
    ``PipelineRun.config_snapshot.stage_1_verdicts_at_launch``.

    Args:
        project_stage_1_verdicts: The current
            ``Project.stage_1_verdicts`` JSONField value, or
            ``None`` when the project predates Sprint F /
            never had a Stage 1 verdict stored.

    Returns:
        Deep copy of the input dict, or an empty dict when
        the input is None / empty. Empty-dict result keeps the
        downstream JSONField serialization clean (no
        ``null`` round-trip surprises).
    """
    if not project_stage_1_verdicts:
        return {}
    return copy.deepcopy(project_stage_1_verdicts)


def is_snapshot_unavailable(snapshot: Dict[str, Any] | None) -> bool:
    """Report whether the snapshot represents an operational-
    failure path (``stage_1_unavailable: True``) vs a clean
    compatibility result with no Bucket 3 emit.

    Per Sprint F codex review #DIM-1, the unavailable path
    persists an explicit flag rather than collapsing to an
    empty-entries dict — the cockpit reader uses this signal
    to render the "compatibility check did not run" copy.
    """
    if not isinstance(snapshot, dict):
        return False
    return bool(snapshot.get("stage_1_unavailable"))


def compute_snapshot_at_launch_hash(
    snapshot: Dict[str, Any],
) -> str:
    """Compute a byte-stable hash of the per-run snapshot.

    Used by the cockpit's drift detector — the manifest hash
    binds the cockpit-rendered manifest to the snapshot the
    pipeline launched against; a mismatch at re-render time
    surfaces the drift in the lineage rail.
    """
    payload = json.dumps(
        snapshot or {}, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
