"""Per-stage heartbeat budgets for the prismweb watchdog reaper.

Each :class:`prismpy.pipeline.executor.PipelineStage` gets a maximum
no-progress idle threshold tuned to its empirical workload profile.
The prismweb watchdog (``core/management/commands/watchdog_runs.py``)
computes idle from the newest progress signal (heartbeat thread /
substage update / stage ``started_at``) and reaps with reason
``no_progress_for_<stage>_<budget>s`` when ``idle > budget``.

**Canonical source** per durable §24 canonical-source-or-pin: this
module is the single source of truth for the stage list + per-stage
budgets. Four prismweb consumers (``WebProgressCallback.STAGE_ORDER`` +
``pipeline_recovery.STAGE_ORDER`` + ``processing.html`` JS-side
``STAGE_ORDER`` + ``views.py:6313`` cancel-path) MUST agree with this
canonical via the prismweb-side parity pin at
``tests/structural/test_stage_order_3_copy_parity.py``.

**Lightweight string keys** (per F-DE RC1 codex round-1 MED 3
absorption): the keys are plain ``str``, NOT
``prismpy.pipeline.executor.PipelineStage`` enum members. Importing
the executor stack into the prismweb watchdog at process boot would
bloat the watchdog dependency graph; the stage-name string is all
the watchdog actually needs. The structural pin
``tests/structural/test_stage_heartbeat_budgets_parity.py`` imports
``PipelineStage`` at test time only to validate the keys match the
canonical enum (so a future stage rename trips the pin).

**Profile rationale** (per F-DE RC1 cycle-3 contract §A.4 + user
production-pain context: SARRA-Py at 240s false-positive):

* ``retrieve`` 600s — NetCDF extraction + CDS queue + provider
  rate-limits. AgERA5 monitor's 8s-cadence preserves ``substage``
  timestamps only on byte-different detail; for unchanged-detail
  polls the heartbeat thread is the true safety signal (per F-DE RC1
  §K.4 + ``core/tasks.py:325-345`` ``_safe_pipeline_write``).
* ``harmonize`` 240s — orchestration; tight; minimal IO. The cycle-1
  240s legacy default was correct here; this is where the budget
  used to come from. Other stages need wider budgets.
* ``translate`` 600s — longest-platform (DSSAT compile). Per-platform
  tracking deferred to F-DE.B-AC-3; large-region tracking deferred
  per F-DE RC1 §K.10.
* ``remediation`` 300s — V2-22c-PRE.4.1 D25 stage between
  ``translate`` + ``validate``. Cockpit-bulk-fix re-runs apply the
  remediation spec; baseline runs no-op.
* ``validate`` 180s — post-translate validation; mostly IO + light
  compute.
* ``package`` 120s — fast IO; cockpit sidecar + provenance save.

Per F-DE RC1 2026-05-13 cycle-3 LOCKED + builder grounding pass.
"""
from __future__ import annotations

from typing import Final


# String keys (NOT ``PipelineStage`` enum) to keep the watchdog import
# path lightweight per codex round-1 MED 3 + F-DE RC1 grounding §L.4.
STAGE_HEARTBEAT_BUDGETS: Final[dict[str, int]] = {
    "retrieve":    600,
    "harmonize":   240,
    "translate":   600,
    "remediation": 300,
    "validate":    180,
    "package":     120,
}
