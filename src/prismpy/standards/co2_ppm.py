"""Canonical atmospheric CO₂ concentration values per ISIMIP3b scenario × period.

Sprint G AC-G-9: every projection climate package's manifest carries
``scenario.co2_ppm`` paired with ``scenario.co2_ppm_provenance``. Both
fields trace to this single canonical source so audit consumers
(Dr. Kofi grep + the validator post-check) can verify the number was
not invented or copy-pasted from a stale source.

Per durable §24 canonical-source-or-pin: this module IS the
single-source-of-truth. Every consumer that needs a scenario × period
CO₂ value imports :func:`get_co2_ppm_with_provenance`. The structural
pins enforce the contract:

* **Layer 1 — Substrate (AST walker)**: no module-level assignment
  whose target name is in :data:`CANONICAL_CO2_CONCENTRATION_IDENTIFIERS`
  with a literal float value, OUTSIDE this module. Walker scope
  ``prismpy/src/prismpy/**/*.py``. See
  ``tests/structural/test_co2_canonical_substrate.py``.

* **Layer 2 — Semantic (Pydantic post-validator)**: ``ScenarioBlock``
  asserts ``math.isclose(co2_ppm, lookup_value, rel_tol=1e-9)`` AND
  exact provenance string match for the ``(rcp_or_ssp, time_slice)``
  tuple. Mismatch raises :exc:`CO2ProvenanceMismatchError`.

* **Layer 3 — Runtime-emit (per-translator AST walker per F-G-2c)**:
  any translator code path emitting CO₂ to a model-input file in
  PROJECTION mode must source from this module's lookup OR declare
  ``manifest.scenario.co2_consumption_path = "not_consumed_by_this_platform"``.
  See ``tests/structural/test_co2_canonical_runtime_emit.py``.

Reference values per AR6 WG1 Annex III + RCMIP, mid-year-of-period
convention. Source URLs in module docstring above each entry.
"""

from __future__ import annotations

import math
from typing import Tuple


# ── Canonical CO₂ values per (scenario, time_slice) tuple ────────────


CO2_PPM_BY_SCENARIO_PERIOD: dict[
    Tuple[str, Tuple[int, int]], Tuple[float, str]
] = {
    # ── SSP2-4.5 (intermediate emissions, ISIMIP3b "middle of the road") ─
    #
    # AR6 WG1 Annex III concentration table; mid-year of the period
    # convention (e.g., 2046–2065 → mean of 2055 +/- internal-decadal
    # smoothing applied by ISIMIP3b atmospheric forcing dataset).
    # https://www.ipcc.ch/report/ar6/wg1/chapter/annex-iii/
    ("SSP245", (2046, 2065)): (
        478.0,
        "AR6 WG1 Annex III + RCMIP, mid-year-of-period convention",
    ),
    ("SSP245", (2086, 2100)): (
        541.0,
        "AR6 WG1 Annex III + RCMIP, mid-year-of-period convention",
    ),
    # ── SSP5-8.5 (high emissions, ISIMIP3b "fossil-fuelled development") ─
    ("SSP585", (2046, 2065)): (
        571.0,
        "AR6 WG1 Annex III + RCMIP, mid-year-of-period convention",
    ),
    ("SSP585", (2086, 2100)): (
        1054.0,
        "AR6 WG1 Annex III + RCMIP, mid-year-of-period convention",
    ),
}
"""Canonical CO₂ ppm + provenance string per (scenario, time_slice).

Tuple value: ``(co2_ppm: float, provenance_string: str)``.

Keys are exactly the ISIMIP3b primary core ensemble (scenario,
time_slice) tuples Sprint G ships. Adding a new scenario requires
extending this table AND the ``isimip_versions.SCENARIO_TIME_SLICES``
roster atomically.
"""


# ── Canonical-source whitelist (Layer 1 walker scope) ────────────────


CANONICAL_CO2_CONCENTRATION_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "co2_ppm",
        "co2_concentration",
        "atmospheric_co2_ppm",
        "atmospheric_co2_concentration",
    }
)
"""Closed whitelist of canonical CO₂-concentration identifier names.

Per warning-auditor pass-2 MEDIUM-Rebase-2 (Variant A explicit
whitelist preferred over Variant B loose-substring): the Layer 1 AST
walker only flags module-level assignments whose target name is in
this set AND whose value is a literal float. Non-canonical names
(e.g., ``CO2_DATA`` ACEA historical table at
``translators/acea/translator.py:2769``) do NOT trigger Layer 1 by
construction; false-positive rate zero, false-negative caught at
Layer 2 if the value reaches ``ScenarioBlock``.

Adding a new identifier name to this whitelist atomically extends
the Layer 1 scope.
"""


# ── Public API ───────────────────────────────────────────────────────


def get_co2_ppm_with_provenance(
    rcp_or_ssp: str,
    time_slice: Tuple[int, int],
) -> Tuple[float, str]:
    """Return the canonical ``(co2_ppm, provenance_string)`` tuple
    for an ISIMIP3b ``(scenario, time_slice)`` pair.

    Args:
        rcp_or_ssp: Scenario identifier. Sprint G primary core
            ensemble: ``"SSP245"`` or ``"SSP585"``. The lookup is
            case-sensitive — pass the exact identifier the
            :data:`prismpy.standards.isimip_versions.SCENARIO_TIME_SLICES`
            roster uses.
        time_slice: ``(start_year, end_year)`` inclusive tuple. Sprint G
            primary core ensemble: ``(2046, 2065)`` near-future or
            ``(2086, 2100)`` end-of-century.

    Returns:
        Tuple of ``(co2_ppm: float, provenance_string: str)``. The
        provenance string is the citation-grade source identifier
        Dr. Kofi greps for in the manifest; do not paraphrase.

    Raises:
        ValueError: If ``(rcp_or_ssp, time_slice)`` is not in
            :data:`CO2_PPM_BY_SCENARIO_PERIOD`. The error message
            enumerates the registered keys so the caller knows which
            scenario × period tuples are supported. Adding a new
            scenario requires extending the table atomically with
            the upstream ISIMIP3b SCENARIO_TIME_SLICES roster.

    Per durable §24 canonical-source-or-pin: every consumer routes
    through this function. The Layer 2 ``ScenarioBlock`` post-validator
    calls it to verify ``co2_ppm`` + ``co2_ppm_provenance`` agree with
    the canonical lookup. The Layer 3 runtime-emit walker (F-G-2c)
    asserts every translator's projection-mode CO₂ emission sources
    from here.

    Case normalisation: ``rcp_or_ssp`` is upper-cased before the
    table lookup so the existing prismpy roster (which uses lowercase
    ``"ssp245"`` / ``"ssp585"`` per
    :data:`prismpy.standards.isimip_versions.SCENARIO_PRODUCT_MAP`)
    routes cleanly through this helper. Codex round 1 boundary 4/7
    P2-1 absorption — without normalisation a translator passing
    ``"ssp585"`` would raise even though ``ScenarioBlock`` accepts
    the same identifier.
    """
    if not isinstance(rcp_or_ssp, str):
        raise ValueError(
            f"rcp_or_ssp must be str, got {type(rcp_or_ssp).__name__}"
        )
    normalised_scenario = rcp_or_ssp.upper()
    key = (normalised_scenario, tuple(time_slice))  # type: ignore[assignment]
    if key not in CO2_PPM_BY_SCENARIO_PERIOD:
        raise ValueError(
            f"No canonical CO₂ ppm registered for "
            f"(rcp_or_ssp={rcp_or_ssp!r}, time_slice={time_slice!r}). "
            f"Registered keys: "
            f"{sorted(CO2_PPM_BY_SCENARIO_PERIOD.keys())}. Extend "
            "prismpy.standards.co2_ppm.CO2_PPM_BY_SCENARIO_PERIOD "
            "atomically with isimip_versions.SCENARIO_TIME_SLICES "
            "when adding a new scenario × period."
        )
    return CO2_PPM_BY_SCENARIO_PERIOD[key]


def is_registered_scenario_period(
    rcp_or_ssp: str,
    time_slice: Tuple[int, int],
) -> bool:
    """Return True iff ``(rcp_or_ssp, time_slice)`` is in the canonical
    table. Mirrors :func:`get_co2_ppm_with_provenance` case-normalisation
    so the predicate and the lookup stay consistent.

    Per codex round 1 boundary 4/7 P2-2 absorption: the
    :func:`prismpy.validators.scenario_set.validate_scenario_set` ship-
    mode pipeline calls this predicate to reject unregistered scenario
    × period tuples that would otherwise carry arbitrary CO₂ values
    through Layer 2's silent-skip path.
    """
    if not isinstance(rcp_or_ssp, str):
        return False
    return (
        rcp_or_ssp.upper(),
        tuple(time_slice),
    ) in CO2_PPM_BY_SCENARIO_PERIOD


# ── Exception type for Layer 2 semantic check ────────────────────────


class CO2ProvenanceMismatchError(ValueError):
    """``ScenarioBlock.co2_ppm`` or ``co2_ppm_provenance`` disagrees
    with the canonical lookup for ``(rcp_or_ssp, time_slice)``.

    Raised by the Layer 2 Pydantic post-validator on
    :class:`ScenarioBlock`. Float comparison uses
    ``math.isclose(rel_tol=1e-9)`` per pass-2 MEDIUM-Rebase-3
    (strict equality is fragile to rounding artifacts; this tolerance
    catches deliberate cooking without flagging legitimate rounding).
    Provenance string remains exact-match — paraphrased citations
    fail loud.

    Attributes:
        scenario: The ``rcp_or_ssp`` identifier on the offending block.
        time_slice: The ``(start_year, end_year)`` tuple.
        observed_co2_ppm: The value carried by the block (may be
            ``None`` if validation fired on the provenance string).
        expected_co2_ppm: The canonical lookup value.
        observed_provenance: The provenance string carried by the
            block (may be ``None``).
        expected_provenance: The canonical lookup provenance string.
    """

    def __init__(
        self,
        message: str,
        *,
        scenario: str,
        time_slice: Tuple[int, int],
        observed_co2_ppm: float | None = None,
        expected_co2_ppm: float | None = None,
        observed_provenance: str | None = None,
        expected_provenance: str | None = None,
    ) -> None:
        super().__init__(message)
        self.scenario = scenario
        self.time_slice = time_slice
        self.observed_co2_ppm = observed_co2_ppm
        self.expected_co2_ppm = expected_co2_ppm
        self.observed_provenance = observed_provenance
        self.expected_provenance = expected_provenance


# ── Convenience: float comparison tolerance for callers ──────────────


CO2_PPM_REL_TOL: float = 1e-9
"""Relative tolerance for ``math.isclose`` on ``co2_ppm`` comparisons.

Per pass-2 MEDIUM-Rebase-3: strict ``==`` on float64 ``co2_ppm`` is
fragile to rounding artifacts (e.g., a JSON round-trip can introduce
sub-ULP drift on values like ``478.0``). ``rel_tol=1e-9`` rejects any
deliberate cooking (the smallest "interesting" alteration would
change the value by far more than 1 part per billion) while
absorbing legitimate float-serialization noise.
"""


def co2_ppm_matches_canonical(
    observed: float,
    expected: float,
    *,
    rel_tol: float = CO2_PPM_REL_TOL,
) -> bool:
    """Thin wrapper over ``math.isclose`` so call sites use the
    same tolerance constant the canonical module declares.

    Per durable §24: do not redeclare the tolerance in callers.
    """
    return math.isclose(observed, expected, rel_tol=rel_tol)


__all__ = [
    "CO2_PPM_BY_SCENARIO_PERIOD",
    "CANONICAL_CO2_CONCENTRATION_IDENTIFIERS",
    "CO2_PPM_REL_TOL",
    "CO2ProvenanceMismatchError",
    "get_co2_ppm_with_provenance",
    "co2_ppm_matches_canonical",
    "is_registered_scenario_period",
]
