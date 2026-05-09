"""Cockpit methods-text generator for IDW-imputed cells.

Sprint E.2 AC-E2-7 + Sprint E.3 AC-E3-11 sub-2 + post-Draft 4 codex
HIGH-2 absorption (consumer-side propagation). Generates the
persona-readable methods paragraph that appears in
``manifest.methods_text`` when a package contains imputed cells.
The paragraph follows Stage-0 §6 State A + §10 honest-signaling
discipline:

* Open with cell-count summary naming the IDW method + Shepard 1968.
* Per-zone caveat clauses appended for cells whose
  ``InterpolatedCellRecord.caveat_codes`` are non-empty.
* Degraded-path caveat (Phrase 1 / Phrase 2 verbatim) when any
  contributing record was imputed with fewer than k neighbours.
* Closing flag note pointing to ``manifest.flags.interpolation_present``.

**Per-record provenance** (post-Sprint E.3 AC-E3-11 sub-2 + codex
HIGH-2 absorbed): the radius / k / weight_power numeric values
come from the per-record ``radius_km`` / ``k`` / ``weight_power``
fields on each :class:`InterpolatedCellRecord` — NOT from module-
level ``IDW_DEFAULT_*`` constants. ACEA records carry
``radius_km=100.0`` per :data:`IDW_RADIUS_BY_PLATFORM`; rendering
"within 15 km" for those cells would emit FALSE method identity
+ violate the audit-trail-integrity contract. Per durable §24
canonical-source-or-pin: per-record fields ARE the canonical
source for the methods text now.

Mixed-radius safety: when records carry different radii (would
indicate a substrate bug since one paragraph is per-platform, but
defensive against a future cross-platform render), the opening
phrase emits the union of distinct radii so the reader sees the
honest ambiguity rather than a single-radius lie.
"""

from __future__ import annotations

from prismpy.config.schema import Platform
from prismpy.models.interpolated_cell import InterpolatedCellRecord
from prismpy.standards.caveat_codes import METHODS_TEXT_CAVEAT_PHRASES


# ── Exact-string phrases per AC-E2-7 sub-criteria ───────────────────


# Phrase 1 (degraded path k=2/3): structural pin asserts presence
# verbatim when any record's degraded flag is True AND
# len(source_cells) > 1. Records carry their own per-record k so
# the rendered "fewer than k=N" matches the actual dispatched k
# rather than the module-level default.
_PHRASE_DEGRADED = (
    "{n} cells imputed with fewer than k={k} neighbors; uncertainty "
    "bounds for these cells are conservative under the normality "
    "assumption"
)

# Phrase 2 (k=1 zero-width): structural pin asserts presence verbatim
# when any record's source_cells is exactly length 1. The radius
# here reads from the per-record radius_km (NOT the module default)
# per Sprint E.3 AC-E3-11 sub-2 + codex HIGH-2 absorption.
_PHRASE_SINGLE_NEIGHBOUR = (
    "{n} cells imputed from a single neighbor within R={radius_km}km; "
    "uncertainty bounds for these cells are uninformative "
    "(zero-width by construction)"
)


# ── Per-platform paragraph framing ──────────────────────────────────


_PLATFORM_NAMES: dict[Platform, str] = {
    Platform.PYTHIA: "PYTHIA",
    Platform.CRAFT: "CRAFT",
    Platform.SARRA_PY: "SARRA-Py",
    Platform.ACEA: "ACEA",
}


def _format_radius_km(value: float) -> str:
    """Format a radius value for the methods text without trailing
    zeros that would suggest false precision (e.g., ``100`` not
    ``100.0``; ``25`` not ``25.0``)."""
    if value == int(value):
        return f"{int(value)}"
    return f"{value:g}"


def _summarise_record_parameters(
    interpolation_records: list[InterpolatedCellRecord],
) -> tuple[str, int]:
    """Collapse the per-record ``radius_km`` / ``k`` values across
    a list of records into a single rendering-ready summary.

    Returns:
        ``(radius_phrase, modal_k)`` — ``radius_phrase`` is the
        opening-phrase fragment for radius (e.g., ``"15 km"`` for
        homogeneous, ``"15/100 km"`` for mixed); ``modal_k`` is
        the most-common ``k`` value across records (used for the
        degraded-path phrase's "fewer than k=N" rendering).

    Mixed-radius / mixed-k packages are a defensive case (one
    paragraph is per-platform; same-platform records SHOULD share
    parameters). The honest-signal floor per
    ``feedback_no_data_cooking.md`` requires us to show the
    distinct values rather than collapse to a misleading single
    value.
    """
    radii = sorted({r.radius_km for r in interpolation_records})
    if len(radii) == 1:
        radius_phrase = f"{_format_radius_km(radii[0])} km"
    else:
        radius_phrase = "/".join(_format_radius_km(r) for r in radii) + " km"

    # Modal k for the degraded-path phrase. The Sprint E.3 v1 case
    # has all records share k; future extensions may differ.
    k_counts: dict[int, int] = {}
    for record in interpolation_records:
        k_counts[record.k] = k_counts.get(record.k, 0) + 1
    modal_k = max(k_counts.items(), key=lambda kv: kv[1])[0]
    return radius_phrase, modal_k


def generate_interpolation_methods_paragraph(
    interpolation_records: list[InterpolatedCellRecord],
    platform: Platform,
) -> str:
    """Return the methods-text paragraph for a package's imputed
    cells.

    Args:
        interpolation_records: All ``InterpolatedCellRecord`` entries
            for cells the cockpit imputed (post-revert active state
            per ``current_decisions`` reader at AC-E2-21). Empty
            list → returns empty string (no paragraph needed).
        platform: Platform whose package this paragraph documents.

    Returns:
        Plain-language paragraph naming method, neighbour count,
        radius, citation, per-zone caveats, degraded-path caveats,
        and the INTERPOLATION-PRESENT flag closing.
    """
    if not interpolation_records:
        return ""

    n_total = len(interpolation_records)
    platform_name = _PLATFORM_NAMES.get(platform, str(platform.value))

    radius_phrase, modal_k = _summarise_record_parameters(interpolation_records)

    parts: list[str] = []

    # Opening: cell-count summary + method anchor. Radius / k come
    # from per-record fields per AC-E3-11 sub-2 (codex HIGH-2
    # absorbed).
    parts.append(
        f"{n_total} {platform_name} cell{'s' if n_total != 1 else ''} "
        f"imputed via inverse-distance-weighted interpolation from up "
        f"to {modal_k} nearest neighbors within {radius_phrase} "
        f"(Shepard 1968)."
    )

    # Per-zone caveat clauses. Collect unique caveat codes from all
    # records + emit each phrase once.
    seen_caveats: set[str] = set()
    caveat_phrases: list[str] = []
    for record in interpolation_records:
        for code in record.caveat_codes:
            if code not in seen_caveats:
                seen_caveats.add(code)
                caveat_phrases.append(METHODS_TEXT_CAVEAT_PHRASES[code])
    parts.extend(caveat_phrases)

    # Degraded-path caveats. The k threshold uses the per-record k
    # (or modal_k for the rendered phrase) — a record's
    # ``len(source_cells) < record.k`` signals degraded.
    n_single = sum(1 for r in interpolation_records if len(r.source_cells) == 1)
    n_degraded_multi = sum(
        1
        for r in interpolation_records
        if 1 < len(r.source_cells) < r.k
    )
    if n_degraded_multi > 0:
        parts.append(_PHRASE_DEGRADED.format(n=n_degraded_multi, k=modal_k))
    if n_single > 0:
        # Per-record radius for the single-neighbour phrase. When
        # records share a single radius, use it directly; mixed-
        # radius case uses the same union phrase as the opening
        # (rare; defensive).
        single_records = [r for r in interpolation_records if len(r.source_cells) == 1]
        single_radii = sorted({r.radius_km for r in single_records})
        if len(single_radii) == 1:
            single_radius_phrase = _format_radius_km(single_radii[0])
        else:
            single_radius_phrase = "/".join(
                _format_radius_km(r) for r in single_radii
            )
        parts.append(_PHRASE_SINGLE_NEIGHBOUR.format(
            n=n_single, radius_km=single_radius_phrase,
        ))

    # Closing flag pointer.
    parts.append(
        "Outputs containing imputed cells are flagged "
        "INTERPOLATION-PRESENT in manifest.flags."
    )

    return " ".join(parts)


__all__ = [
    "generate_interpolation_methods_paragraph",
]
