"""HARMONIZE-stage writer for ``cockpit_overrides.json``.

Sprint E.3 AC-E3-7 + Stage 1 §6 Override-with-values mental model.
The translator-side consumer (AC-E3-8 ``apply_override`` helper +
AC-E3-9 4-translator wiring) reads the per-cell value-replacement
overrides at write time so a documented Sahel hot-day extreme
tmax of 46 °C surfaces in the platform's per-cell weather file
instead of the validator-flagged raw value.

**Producer / consumer split** mirrors
``cockpit/observed_values_writer.py`` precedent:

* Producer side (this module) reads
  ``config_snapshot.cockpit_overrides_at_launch`` (output of
  :func:`prismpy.packaging.cockpit_snapshot.serialize_decisions_to_config`
  Extension 1) and persists a canonical JSON sidecar to disk.
* Consumer side (Phase 1 ``apply_override`` helper + Phase 2
  prismweb cockpit reader) loads the JSON and dispatches the per-
  cell value at write time.

**Cat D filtering** (codex CA-3 absorbed): the writer FILTERS Cat
D documentary-basis overrides from sidecar emission. Cat A/B/C
value-replacement overrides emit one entry per
``(cell_id, variable_key)`` pair; Cat D rows stay audit/methods-
only — the documentary basis IS the override per AC-E3-4
validator 3, so no value is dispatched to the translator. Filter
logic: skip rows where
``override_record.category_d_documentary_basis is not None``.

**All-reverted-bulk semantic** (codex MED-2 wording absorbed):
when every value-replacement override has been reverted, the
sidecar writes with ``overrides: []`` empty array; the manifest
flag ``overrides_present`` reflects "non-empty sidecar entries"
(False when array is empty); methods text omits the override
paragraph. The empty-array file SHOULD still emit so consumers
can distinguish "writer fired with no overrides" from "writer
never fired" (silent-skip class violation per
``feedback_no_data_cooking.md``).

**Atomicity** (WA CA-20 absorbed): the writer follows the
write-to-temp-file + atomic-rename pattern. A partial-write
failure mode (interrupt mid-write) does NOT leave torn artifacts
visible to consumers — the rename is the commit point. Pin at
``tests/structural/test_sidecar_writer_atomicity.py``.

Schema shape::

    {
        "schema_version": "1.0",
        "produced_at": "<ISO-8601>",
        "overrides": [
            {
                "cell_id": "12345",
                "check_id": "value_range_tmax",
                "variable_key": "tmax_growing_season_mean",
                "value": 32.5,
                "unit": "C",
                "decision_id": "<UUID>",
                "evidence_type": "field_observation"
            }
        ]
    }

Per durable §24 canonical-source-or-pin: the schema lives once
here; the round-trip Pydantic validator
:class:`CockpitOverrideSidecar` at the bottom of this module is
the reader's contract.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from prismpy.provenance.wizard_decisions import EvidenceType
from prismpy.standards.override_value_shapes import (
    OVERRIDE_VALUE_SHAPES,
    get_override_value_shape,
)


# ── Canonical schema constants ─────────────────────────────────────


SCHEMA_VERSION: Literal["1.0"] = "1.0"
"""Sidecar schema version. Bumped on incompatible shape change;
the reader's Pydantic validator at :class:`CockpitOverrideSidecar`
asserts the expected version on load. Per durable §24 canonical-
source-or-pin: the version lives once here."""


# ── Pydantic schemas — round-trip contract for the sidecar ─────────


class OverrideSidecarEntry(BaseModel):
    """One value-replacement override entry in the sidecar.

    A single :class:`prismpy.models.override.OverrideRecord` may
    yield multiple entries — one per variable_key in
    ``override_climate_values`` + ``override_soil_values``. The
    flat-entry shape lets the translator-side consumer dispatch
    on ``(cell_id, variable_key)`` directly without re-grouping.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )

    cell_id: str
    """Canonical cell-id reference per
    :class:`prismpy.cells.schema.CellID`."""

    check_id: str
    """The cockpit warning identifier the override answers (e.g.,
    ``value_range_tmax``). Carried through for audit + methods-text
    use; the translator dispatch uses ``variable_key`` not
    ``check_id``."""

    variable_key: str
    """Canonical variable_key the translator looks up in the
    sidecar (e.g., ``tmax_growing_season_mean``). Mirrors the
    ``OVERRIDE_VALUE_SHAPES`` registry entry's ``variable_key``
    field; the consumer-side reader at
    :func:`apply_override` looks up by ``(cell_id, variable_key)``
    directly."""

    value: Union[float, str, int]
    """The override value. ``float`` for continuous physical
    quantities (the dominant Sprint E.3 v1 case), ``int`` for
    counts, ``str`` for categorical (none currently). The
    consumer translator-side helper dispatches the raw value;
    no unit conversion happens at the consumer (the persona
    submitted in canonical units per the form's unit-conversion
    on submit per AC-E3-3 + the registry's ``unit`` field)."""

    unit: str
    """Canonical unit string per
    :data:`prismpy.standards.override_value_shapes.OVERRIDE_VALUE_SHAPES`.
    Empty string for unitless quantities (pH).  Same value as the
    registry; redundant on the wire but useful for human-readable
    sidecar inspection."""

    decision_id: UUID
    """UUID of the enclosing :class:`CellDecisionRecord`. Carried
    through so the translator-side consumer can correlate the
    override application back to the audit log entry."""

    evidence_type: EvidenceType
    """Categorical evidence basis per AC-E3-1 canonical Literal
    at ``provenance/wizard_decisions.py:119``. Persisted so the
    methods-text generator can emit per-evidence-type copy
    without re-loading the full OverrideRecord."""


class CockpitOverrideSidecar(BaseModel):
    """Top-level sidecar payload — round-trip contract for
    ``cockpit_overrides.json``."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )

    schema_version: Literal["1.0"]
    """Pinned to :data:`SCHEMA_VERSION`. A future shape change
    bumps the literal here + the reader's validator fires loud
    on legacy files until they're re-emitted."""

    produced_at: datetime
    """Wall-clock at writer invocation (ISO 8601). Surfaced for
    manifest-vs-sidecar drift detection — re-running the
    pipeline overwrites with a fresh timestamp."""

    overrides: List[OverrideSidecarEntry]
    """Per-(cell_id, variable_key) override entries. Empty list
    on the all-reverted-bulk path per the codex MED-2 absorption
    in this module's docstring."""


# ── Writer — extracts entries from config_snapshot block ───────────


def _is_cat_d_record(record_dict: Any) -> bool:
    """True iff the record is a Cat D documentary override row.

    The discriminator is
    ``override_record.category_d_documentary_basis is not None``
    per :class:`OverrideRecord` validator 3. Cat D rows skip
    sidecar emission per codex CA-3 absorption."""
    if not isinstance(record_dict, dict):
        return False
    if record_dict.get("action") != "document_override":
        return False
    override_record = record_dict.get("override_record")
    if not isinstance(override_record, dict):
        return False
    return override_record.get("category_d_documentary_basis") is not None


def _entries_from_record(record_dict: Any) -> List[OverrideSidecarEntry]:
    """Yield :class:`OverrideSidecarEntry` instances for one
    decision-record dict.

    Returns an empty list when the record is None, not a
    document_override, missing override_record, or a Cat D
    documentary row (filtered per codex CA-3)."""
    if not isinstance(record_dict, dict):
        return []
    if record_dict.get("action") != "document_override":
        return []
    if _is_cat_d_record(record_dict):
        return []
    override_record = record_dict.get("override_record")
    if not isinstance(override_record, dict):
        return []
    cell_id = record_dict.get("cell_id")
    check_id = record_dict.get("check_id")
    decision_id_str = record_dict.get("decision_id")
    evidence_type = override_record.get("evidence_type")
    if cell_id is None or check_id is None or decision_id_str is None:
        return []
    if evidence_type is None:
        return []

    decision_id = UUID(decision_id_str) if isinstance(decision_id_str, str) else decision_id_str

    entries: List[OverrideSidecarEntry] = []

    # Climate value-replacement entries — one per variable_key.
    climate_values = override_record.get("override_climate_values") or {}
    if isinstance(climate_values, dict):
        for variable_key, value in sorted(climate_values.items()):
            unit = _unit_for_variable_key(variable_key, fallback_check_id=check_id)
            entries.append(
                OverrideSidecarEntry(
                    cell_id=str(cell_id),
                    check_id=str(check_id),
                    variable_key=str(variable_key),
                    value=value,
                    unit=unit,
                    decision_id=decision_id,
                    evidence_type=evidence_type,
                )
            )

    # Soil value-replacement entries — same per-variable_key shape.
    soil_values = override_record.get("override_soil_values") or {}
    if isinstance(soil_values, dict):
        for variable_key, value in sorted(soil_values.items()):
            unit = _unit_for_variable_key(variable_key, fallback_check_id=check_id)
            entries.append(
                OverrideSidecarEntry(
                    cell_id=str(cell_id),
                    check_id=str(check_id),
                    variable_key=str(variable_key),
                    value=value,
                    unit=unit,
                    decision_id=decision_id,
                    evidence_type=evidence_type,
                )
            )

    return entries


def _unit_for_variable_key(
    variable_key: str,
    *,
    fallback_check_id: str,
) -> str:
    """Resolve the canonical unit for a sidecar entry.

    Looks up by ``check_id`` in
    :data:`OVERRIDE_VALUE_SHAPES`; the registry's
    ``variable_key`` field MUST match the sidecar entry's
    ``variable_key`` for a valid lookup. Returns empty string if
    the registry has no entry for the check_id (form-side
    validation should reject Override on such check_ids before
    reaching the writer; defensive empty-string fallback closes
    the no-data-cooking floor)."""
    shape = get_override_value_shape(fallback_check_id)
    if shape is None:
        return ""
    if shape.variable_key != variable_key:
        # Sidecar entry's variable_key drifted from the registry.
        # Defensive: walk the registry for a matching variable_key
        # to recover the correct unit.
        for registry_shape in OVERRIDE_VALUE_SHAPES.values():
            if registry_shape.variable_key == variable_key:
                return registry_shape.unit
        return ""
    return shape.unit


def write_cockpit_overrides_json(
    *,
    cockpit_overrides_at_launch: Any,
    output_path: Union[str, Path],
    produced_at: Optional[datetime] = None,
) -> Path:
    """Write the cockpit_overrides.json sidecar to disk.

    Args:
        cockpit_overrides_at_launch: The
            ``cockpit_overrides_at_launch`` block from the
            derived run's ``PipelineRun.config_snapshot``. Shape
            mirrors :func:`serialize_decisions_to_config`'s
            nested ``[<cell_id>][<check_id>] = record_dict``
            output. Empty dict / None is the all-reverted-bulk
            case — emits sidecar with empty overrides[].
        output_path: Where to write the JSON.
        produced_at: Optional override for the timestamp;
            defaults to ``datetime.now(timezone.utc)``. Tests
            pass a fixed value for deterministic comparison.

    Returns:
        Path of the written JSON (echo of ``output_path`` for
        chaining).

    Atomicity (WA CA-20): writes to a sibling temp file via
    :func:`tempfile.NamedTemporaryFile` + :func:`os.replace` so
    a mid-write failure cannot leave a torn payload visible to
    consumers. The rename is the commit point.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if produced_at is None:
        produced_at = datetime.now(timezone.utc)

    # Walk the nested ``cockpit_overrides_at_launch[<cell_id>][<check_id>] =
    # record_dict`` block; collect non-Cat-D value-replacement
    # entries.
    entries: List[OverrideSidecarEntry] = []
    if isinstance(cockpit_overrides_at_launch, dict):
        for cell_id in sorted(cockpit_overrides_at_launch):
            inner = cockpit_overrides_at_launch.get(cell_id)
            if not isinstance(inner, dict):
                continue
            for check_id in sorted(inner):
                record_dict = inner.get(check_id)
                entries.extend(_entries_from_record(record_dict))

    sidecar = CockpitOverrideSidecar(
        schema_version=SCHEMA_VERSION,
        produced_at=produced_at,
        overrides=entries,
    )

    payload = sidecar.model_dump(mode="json")

    # Atomic write — temp file in same dir + os.replace.
    fd, temp_name = tempfile.mkstemp(
        prefix=".cockpit_overrides.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
        os.replace(temp_name, str(output_path))
    except Exception:
        # Cleanup the temp file on any failure path; re-raise so
        # the caller surfaces the operational error.
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    return output_path


__all__ = [
    "CockpitOverrideSidecar",
    "OverrideSidecarEntry",
    "SCHEMA_VERSION",
    "write_cockpit_overrides_json",
]
