"""Cockpit methods-text generator for IDW-imputed cells.

Sprint E.2 AC-E2-7. Generates the persona-readable methods paragraph
that appears in ``manifest.methods_text`` when a package contains
imputed cells. The paragraph follows Stage-0 §6 State A + §10
honest-signaling discipline:

* Open with cell-count summary naming the IDW method + Shepard 1968.
* Per-zone caveat clauses appended for cells whose
  ``InterpolatedCellRecord.caveat_codes`` are non-empty.
* Degraded-path caveat (Phrase 1 / Phrase 2 verbatim) when any
  contributing record was imputed with fewer than k=4 neighbours.
* Closing flag note pointing to ``manifest.flags.interpolation_present``.

Per durable §24 canonical-source-or-pin: the IDW parameters quoted
in the paragraph come from ``IDW_DEFAULT_*`` constants; the per-zone
caveat phrases come from ``METHODS_TEXT_CAVEAT_PHRASES``. Drift
between the paragraph text and the canonical sources is caught by
the structural mirror pins those modules carry.
"""

from __future__ import annotations

from prismpy.config.schema import Platform
from prismpy.models.interpolated_cell import InterpolatedCellRecord
from prismpy.standards.caveat_codes import METHODS_TEXT_CAVEAT_PHRASES
from prismpy.standards.idw_methods import (
    IDW_DEFAULT_K,
    IDW_DEFAULT_R,
)


# ── Exact-string phrases per AC-E2-7 sub-criteria ───────────────────


# Phrase 1 (degraded path k=2/3): structural pin asserts presence
# verbatim when any record's degraded flag is True AND
# len(source_cells) > 1.
_PHRASE_DEGRADED = (
    "{n} cells imputed with fewer than k=4 neighbors; uncertainty "
    "bounds for these cells are conservative under the normality "
    "assumption"
)

# Phrase 2 (k=1 zero-width): structural pin asserts presence verbatim
# when any record's source_cells is exactly length 1.
_PHRASE_SINGLE_NEIGHBOUR = (
    "{n} cells imputed from a single neighbor within R=15km; "
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

    parts: list[str] = []

    # Opening: cell-count summary + method anchor.
    parts.append(
        f"{n_total} {platform_name} cell{'s' if n_total != 1 else ''} "
        f"imputed via inverse-distance-weighted interpolation from up "
        f"to {IDW_DEFAULT_K} nearest neighbors within {IDW_DEFAULT_R:.0f} km "
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

    # Degraded-path caveats: count records with single-neighbour
    # vs k=2/3 cases separately.
    n_single = sum(1 for r in interpolation_records if len(r.source_cells) == 1)
    n_degraded_multi = sum(
        1
        for r in interpolation_records
        if 1 < len(r.source_cells) < IDW_DEFAULT_K
    )
    if n_degraded_multi > 0:
        parts.append(_PHRASE_DEGRADED.format(n=n_degraded_multi))
    if n_single > 0:
        parts.append(_PHRASE_SINGLE_NEIGHBOUR.format(n=n_single))

    # Closing flag pointer.
    parts.append(
        "Outputs containing imputed cells are flagged "
        "INTERPOLATION-PRESENT in manifest.flags."
    )

    return " ".join(parts)


__all__ = [
    "generate_interpolation_methods_paragraph",
]
