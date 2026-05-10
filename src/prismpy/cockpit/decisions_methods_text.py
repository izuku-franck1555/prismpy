"""Methods-text renderer for cockpit decision audit trail.

Sprint E.3 fixup +15 (F-BO Boundary 4 unified consumer point). The
post-execute prismweb hook calls
:func:`render_cockpit_decisions_text` once per derived run + writes
the result to ``<outdir>/<platform>/<region>/cockpit_decisions.txt``
in every platform package directory so the persona's audit trail
appears in every shipped artifact, regardless of which translator
processed which override.

The renderer reads from the derived run's
``config_snapshot.cockpit_decisions_at_launch`` block (the full
decision set, including Override Cat A/B/C value-replacement,
Override Cat D documentary-basis, and Acknowledge entries). Cat A/B/C
overrides ALSO emit a "(value applied to per-cell files)" suffix
note so the persona can correlate the audit row with the actual
canonical-file change in the package — closes the user-snippet
acceptance bar per durable §25 (persona sees evidence in plain
text).

**Producer / consumer split** (durable §27 two-vocabulary substrate
parity): the snapshot block produced by prismweb's
``commit_decision_snapshot`` carries serialized ``CellDecisionRecord``
dicts; the consumer here walks the nested dict shape without re-
validating via Pydantic (cheaper, since the snapshot was already
validated at write time). Schema-version drift would surface as a
missing field at the renderer; the helper degrades gracefully —
missing fields render as ``"(unspecified)"`` rather than crashing
the package emission.
"""
from __future__ import annotations

import logging
from typing import Any, List, Mapping, Optional

logger = logging.getLogger(__name__)


_TEXT_BANNER = "# Cockpit Decisions — User Audit Trail"
_TEXT_SUBHEADER_OVERRIDES_DOC = "## Documented Overrides (Cat D — documentary basis)"
_TEXT_SUBHEADER_OVERRIDES_VAL = "## Value-Replacement Overrides (Cat A / B / C)"
_TEXT_SUBHEADER_ACKNOWLEDGES = "## Acknowledged Warnings (Bucket 2 informational)"
_TEXT_SUBHEADER_SKIPS = "## Skipped Cells (excluded from analysis)"
_TEXT_EMPTY_BLOCK = "(no entries)"


def render_cockpit_decisions_text(
    *,
    cockpit_decisions_at_launch: Optional[Mapping[str, Any]],
    derived_run_id: str = "(unspecified)",
    derived_run_number: Optional[int] = None,
) -> str:
    """Render the cockpit decision snapshot block to plain-text audit trail.

    Sprint E.3 fixup +15 (F-BO Boundary 4 + Cat D AC #3 fold).

    Args:
        cockpit_decisions_at_launch: The
            ``cockpit_decisions_at_launch`` block from a derived
            ``PipelineRun.config_snapshot``. Shape per
            :func:`prismpy.packaging.cockpit_snapshot.serialize_decisions_to_config`:
            nested ``[<cell_id>][<check_id>] = record_dict``. ``None``
            or empty dict produces a "no decisions" stub so the file
            still emits and the persona knows the file came from a
            run with no cockpit interventions (vs. a run that never
            wrote the file at all — silent-skip class per
            ``feedback_no_data_cooking.md``).
        derived_run_id: UUID of the derived run, surfaced in the
            preamble so the persona can cross-reference back to the
            audit log entry.
        derived_run_number: Optional sequential run number per the
            project's run roster. Rendered as "Run #N" in the
            preamble when present.

    Returns:
        Multi-line plain-text string. The first line is the
        ``_TEXT_BANNER`` header; subsequent sections list per-decision
        rows under the four sub-headers (value-replacement overrides /
        Cat D documentary overrides / acknowledged warnings / skipped
        cells). Sections with zero entries emit
        ``_TEXT_EMPTY_BLOCK`` so the persona sees the section's
        presence (an empty section signals "we considered this
        category and the persona made no decisions"; a missing
        section would silently lose the discriminator).

    The renderer is pure — no I/O, no global state. The caller
    writes the returned string to disk via
    :class:`pathlib.Path.write_text` on the prismweb-side post-execute
    hook.
    """
    lines: List[str] = []
    lines.append(_TEXT_BANNER)
    lines.append("")
    if derived_run_number is not None:
        lines.append(f"Derived run #{derived_run_number} ({derived_run_id})")
    else:
        lines.append(f"Derived run id: {derived_run_id}")
    lines.append("")

    # Walk the snapshot and partition decisions into four buckets.
    value_replacement_rows: List[str] = []
    cat_d_documentary_rows: List[str] = []
    acknowledge_rows: List[str] = []
    skip_rows: List[str] = []

    if isinstance(cockpit_decisions_at_launch, dict):
        for cell_id in sorted(cockpit_decisions_at_launch):
            inner = cockpit_decisions_at_launch.get(cell_id)
            if not isinstance(inner, dict):
                continue
            for check_id in sorted(inner):
                record_dict = inner.get(check_id)
                if not isinstance(record_dict, dict):
                    continue
                action = record_dict.get("action")
                if action == "document_override":
                    is_cat_d, row = _format_override_row(
                        cell_id, check_id, record_dict,
                    )
                    if is_cat_d:
                        cat_d_documentary_rows.append(row)
                    else:
                        value_replacement_rows.append(row)
                elif action == "acknowledge":
                    row = _format_acknowledge_row(
                        cell_id, check_id, record_dict,
                    )
                    acknowledge_rows.append(row)
                elif action == "skip_from_analysis":
                    row = _format_skip_row(
                        cell_id, check_id, record_dict,
                    )
                    skip_rows.append(row)
                # ``apply_interpolation`` rows fall to a future fixup;
                # not surfaced in v1 methods text per fixup +15 scope.

    lines.append(_TEXT_SUBHEADER_OVERRIDES_VAL)
    if value_replacement_rows:
        lines.extend(value_replacement_rows)
    else:
        lines.append(_TEXT_EMPTY_BLOCK)
    lines.append("")

    lines.append(_TEXT_SUBHEADER_OVERRIDES_DOC)
    if cat_d_documentary_rows:
        lines.extend(cat_d_documentary_rows)
    else:
        lines.append(_TEXT_EMPTY_BLOCK)
    lines.append("")

    lines.append(_TEXT_SUBHEADER_ACKNOWLEDGES)
    if acknowledge_rows:
        lines.extend(acknowledge_rows)
    else:
        lines.append(_TEXT_EMPTY_BLOCK)
    lines.append("")

    lines.append(_TEXT_SUBHEADER_SKIPS)
    if skip_rows:
        lines.extend(skip_rows)
    else:
        lines.append(_TEXT_EMPTY_BLOCK)
    lines.append("")

    return "\n".join(lines)


def _format_override_row(
    cell_id: str,
    check_id: str,
    record_dict: Mapping[str, Any],
) -> tuple[bool, str]:
    """Render a single ``document_override`` decision row.

    Returns ``(is_cat_d, row_text)``. Cat D is discriminated by
    ``override_record.category_d_documentary_basis is not None`` per
    the OverrideRecord validator's invariant.
    """
    decision_id = record_dict.get("decision_id", "(unspecified)")
    timestamp = record_dict.get("timestamp", "(unspecified)")
    method_or_rationale = record_dict.get("method_or_rationale") or ""
    override_record = record_dict.get("override_record") or {}
    if not isinstance(override_record, dict):
        override_record = {}

    cat_d_basis = override_record.get("category_d_documentary_basis")
    is_cat_d = cat_d_basis is not None

    evidence_type = override_record.get("evidence_type") or "(unspecified)"
    evidence_detail = override_record.get("evidence_detail") or "(no detail provided)"

    if is_cat_d:
        cat_d_other = override_record.get("category_d_documentary_basis_other_specify") or ""
        basis_label = cat_d_basis
        if cat_d_basis == "other" and cat_d_other:
            basis_label = f"other — {cat_d_other}"
        row = (
            f"- Cell {cell_id} ({check_id}): documentary basis = "
            f"{basis_label}; evidence_type = {evidence_type}\n"
            f"  Evidence: {evidence_detail}\n"
            f"  Decision: {decision_id} @ {timestamp}"
        )
        return True, row

    # Cat A/B/C value-replacement — emit the override climate / soil
    # values inline so the persona can cross-reference the audit
    # row with the WTH / SOL file content. Per AC #1 + #2 user-snippet
    # bar per durable §25.
    climate_values = override_record.get("override_climate_values") or {}
    soil_values = override_record.get("override_soil_values") or {}
    value_pairs: List[str] = []
    if isinstance(climate_values, dict):
        for vk in sorted(climate_values):
            value_pairs.append(f"{vk}={climate_values[vk]}")
    if isinstance(soil_values, dict):
        for vk in sorted(soil_values):
            value_pairs.append(f"{vk}={soil_values[vk]}")
    values_label = ", ".join(value_pairs) if value_pairs else "(no values)"

    row = (
        f"- Cell {cell_id} ({check_id}): override values = {values_label}; "
        f"evidence_type = {evidence_type}\n"
        f"  Evidence: {evidence_detail}\n"
        f"  Decision: {decision_id} @ {timestamp}\n"
        f"  (Value applied to canonical per-cell files via "
        f"prismpy.translators._shared.cockpit_overrides.apply_override)"
    )
    if method_or_rationale.strip():
        row += f"\n  Rationale: {method_or_rationale}"
    return False, row


def _format_acknowledge_row(
    cell_id: str,
    check_id: str,
    record_dict: Mapping[str, Any],
) -> str:
    """Render a single ``acknowledge`` decision row.

    The Acknowledge decision's evidence-text lives on
    ``method_or_rationale`` per the ``CellDecisionRecord`` schema
    docstring ("acknowledge: optional acknowledgement note"). The
    cell is INCLUDED in the canonical files (acknowledge is not an
    exclusion); the row simply records the persona's audit-trail
    note + decision id for methodology papers (Dr. Kofi persona) +
    accountability cross-references.
    """
    decision_id = record_dict.get("decision_id", "(unspecified)")
    timestamp = record_dict.get("timestamp", "(unspecified)")
    note = record_dict.get("method_or_rationale") or "(no note provided)"
    return (
        f"- Cell {cell_id} ({check_id}): {note}\n"
        f"  Decision: {decision_id} @ {timestamp}\n"
        f"  (Cell INCLUDED in canonical files; acknowledge is non-blocking)"
    )


def _format_skip_row(
    cell_id: str,
    check_id: str,
    record_dict: Mapping[str, Any],
) -> str:
    """Render a single ``skip_from_analysis`` decision row.

    Skipped cells are EXCLUDED from the canonical files (the derived
    run's ``region.exclude_cells`` block carries the cell ids per
    Sprint E.3 AC-E3-6 fold; this row is the audit-trail mirror).
    """
    decision_id = record_dict.get("decision_id", "(unspecified)")
    timestamp = record_dict.get("timestamp", "(unspecified)")
    rationale = record_dict.get("method_or_rationale") or "(no rationale provided)"
    return (
        f"- Cell {cell_id} ({check_id}): {rationale}\n"
        f"  Decision: {decision_id} @ {timestamp}\n"
        f"  (Cell EXCLUDED from canonical files; skip removes from analysis)"
    )


__all__ = [
    "render_cockpit_decisions_text",
]
