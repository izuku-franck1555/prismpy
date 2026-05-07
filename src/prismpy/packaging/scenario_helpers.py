"""Helper functions for constructing scenario manifest blocks.

The ``ScenarioBlock`` Pydantic model at
:mod:`prismpy.models.scenario` is the schema-layer source of truth
for paired baseline+projection metadata. These helpers wrap common
construction patterns so callers (the wizard's translator pipeline,
the prismweb climate-change generator script, downstream UC2
adapters) build scenario blocks with consistent defaults instead of
inlining the field-by-field construction at each site.

Two helpers ship today:

* :func:`build_baseline_scenario_block` — constructs the ``BASE``
  scenario block for an observed-climate baseline package. Required
  per UC2 climate-scenarios consumers that read
  ``manifest.scenario.scenario_role`` from EVERY package in a
  baseline+projection set; without the block on the baseline, the
  pre-flight validator hard-fails.

* :func:`rewrite_pythia_config_for_scenario` — overwrites the
  ``sdate`` / ``pfrst`` / ``plast`` / ``runs[*].startYear`` fields
  in a delivered package's ``pythia_config.json`` to align with a
  scenario's ``time_slice_start`` / ``time_slice_end``. Required for
  cloned-baseline-then-swap-climate flows where the projection's
  config file would otherwise inherit the baseline's year fields and
  cause DSSAT to silently fall through to "weather record not found"
  (zero yields per cell).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from prismpy.models.scenario import (
    BiasCorrectionMethod,
    ScenarioBlock,
    ScenarioRole,
)


# ── Issue 2: baseline scenario block builder ────────────────────────


def build_baseline_scenario_block(
    *,
    scenario_label: str,
    time_slice_start: int,
    time_slice_end: int,
    co2_ppm: float,
    co2_ppm_provenance: str,
    gcm_source: str = "observed_NASA-POWER",
    rcp_or_ssp: str = "historical",
    baseline_reference_label: Optional[str] = None,
) -> ScenarioBlock:
    """Construct the ``BASE`` scenario block for an observed-climate baseline.

    The baseline package in a paired baseline+projection set MUST
    carry a ``manifest.scenario`` block matching the projection's
    schema so the UC2 pre-flight validator routes both packages
    through the same code path. Without the block on the baseline,
    the validator hard-fails and the comparison graph cannot wire.

    Field conventions per ``PRISMPY-SCENARIO-PACKAGE-SPEC.md`` §3:

    - ``scenario_role`` is :attr:`ScenarioRole.BASE` (serializes as
      ``"baseline"`` per the schema's ``use_enum_values=True`` config;
      callers reading the JSON see the literal ``"baseline"`` string,
      not ``"base"``). The ``"baseline"`` form is canonical per
      codex LOW-1 absorption — the longer form is more durable for
      future UI/API consumers that surface the role to researchers.
    - ``baseline_reference_label`` self-references the baseline's own
      ``scenario_label`` when not provided explicitly. The schema
      allows this; AC-G-3 enforces only that the field is present
      and non-empty, not that it differs from ``scenario_label``.
    - ``bias_correction_method`` is :attr:`BiasCorrectionMethod.NONE`
      because observed-climate data is observed; no bias-correction
      algorithm is applied. The schema enforces this is distinct from
      ``"unknown"`` (the legacy / external-source sentinel).
    - ``rcp_or_ssp`` defaults to ``"historical"`` for observed data;
      the schema treats this as a free-form string so it round-trips
      cleanly without the closed-enum tightening reserved for Sprint H+.
    - ``gcm_source`` defaults to ``"observed_NASA-POWER"`` reflecting
      the typical baseline source. Callers using a different observed
      source (e.g., AgERA5, TAMSAT) override this argument.

    Args:
        scenario_label: Unique identifier for the baseline package
            (e.g., ``"OBSERVED_BENOUE_SORGHUM_2013-2015"``). Used as
            the self-reference target by ``baseline_reference_label``
            unless overridden, and read by the pre-flight validator
            to pair the baseline with its projection siblings.
        time_slice_start: Inclusive start year of the observed period.
        time_slice_end: Inclusive end year. Must be >= start.
        co2_ppm: Atmospheric CO₂ concentration in ppm for the baseline
            period. Caller responsibility: the canonical lookup
            (:data:`prismpy.standards.co2_ppm.CO2_PPM_BY_SCENARIO_PERIOD`)
            covers ISIMIP3b PROJECTION periods only; baseline periods
            are caller-provided (e.g., NOAA Mauna Loa observation for
            the period midpoint).
        co2_ppm_provenance: Mandatory citation string per AC-G-10. Empty
            / whitespace / None raises ``MissingProvenanceError`` at
            ScenarioBlock construction.
        gcm_source: Defaults to ``"observed_NASA-POWER"``. Override
            when the baseline uses a different observed source.
        rcp_or_ssp: Defaults to ``"historical"`` for observed-climate
            baselines.
        baseline_reference_label: Optional self-reference override.
            Defaults to ``scenario_label`` (the baseline references
            itself) when not provided.

    Returns:
        A constructed :class:`ScenarioBlock` ready to embed at
        ``manifest.scenario`` via
        :func:`prismpy.packaging.manifest.create_manifest`.

    Raises:
        ValidationError: When any field violates the ScenarioBlock
            schema (e.g., empty ``co2_ppm_provenance``, ``co2_ppm``
            outside [200.0, 2000.0], ``time_slice_end`` before
            ``time_slice_start``).
    """
    return ScenarioBlock(
        scenario_label=scenario_label,
        scenario_role=ScenarioRole.BASE,
        gcm_source=gcm_source,
        rcp_or_ssp=rcp_or_ssp,
        time_slice_start=time_slice_start,
        time_slice_end=time_slice_end,
        baseline_reference_label=(
            baseline_reference_label
            if baseline_reference_label is not None
            else scenario_label
        ),
        bias_correction_method=BiasCorrectionMethod.NONE,
        co2_ppm=co2_ppm,
        co2_ppm_provenance=co2_ppm_provenance,
        # scenario_bias_correction_provenance is exempt for NONE per
        # AC-G-11; defaults to None and the post-validator's NONE-
        # exemption branch keeps construction valid.
    )


# ── Issue 3: pythia_config.json year-field rewriter ─────────────────


def rewrite_pythia_config_for_scenario(
    pythia_config_path: Union[str, Path],
    *,
    time_slice_start: int,
    time_slice_end: int,
    planting_doy: Optional[int] = None,
    planting_window_days: int = 30,
) -> Dict[str, Any]:
    """Overwrite year fields in a delivered ``pythia_config.json``.

    The cloned-baseline-then-swap-climate flow used by the climate
    scenarios use case clones a baseline package, wipes
    ``weather/*.WTH``, and writes projection WTH files in their place.
    The cloned package's ``config/pythia_config.json`` still carries
    the baseline's ``sdate`` / ``pfrst`` / ``plast`` / ``runs[*].startYear``
    fields. DSSAT then requests weather records for baseline years
    (e.g., 2013-001) which are not present in the projection's WTH
    files (e.g., 2046-2048), and the model silently produces zero
    yields per cell with a ``WARNING.OUT: Weather record not found``.

    This helper rewrites the year fields in place to align with the
    scenario's ``time_slice_start`` / ``time_slice_end``. Month-day
    components on planting-window dates stay at the agronomic default
    or the caller-supplied ``planting_doy``; only the year prefix
    changes.

    Fields rewritten (when present in the input JSON):

    - ``default_setup.sdate`` → ``"<time_slice_start>-01-01"``
    - ``default_setup.pfrst`` → ``"<time_slice_start>-MM-DD"`` where
      MM-DD is computed from ``planting_doy`` (or kept from input
      if ``planting_doy`` is None).
    - ``default_setup.plast`` → start_year + planting_window_days.
    - ``runs[i].startYear`` → ``time_slice_start`` for every entry.
    - ``runs[i].nyers`` → ``time_slice_end - time_slice_start + 1``
      for every entry.

    Args:
        pythia_config_path: Path to the package's
            ``config/pythia_config.json``.
        time_slice_start: Inclusive start year of the scenario period.
        time_slice_end: Inclusive end year of the scenario period.
        planting_doy: Optional planting-DOY override. When provided,
            ``pfrst`` is set to the calendar date of this DOY in the
            new start year and ``plast`` is set to
            ``planting_doy + planting_window_days`` (capped at 365).
            When None, the existing ``pfrst`` / ``plast`` month-day
            values are preserved with only the year prefix updated.
        planting_window_days: Width of the planting window in days
            when ``planting_doy`` is provided. Defaults to 30.

    Returns:
        The rewritten config dict. The file at ``pythia_config_path``
        is also written in place with 2-space JSON indent.

    Raises:
        FileNotFoundError: When ``pythia_config_path`` does not exist.
        ValueError: When ``time_slice_end < time_slice_start``.
    """
    if time_slice_end < time_slice_start:
        raise ValueError(
            f"time_slice_end ({time_slice_end}) must be >= "
            f"time_slice_start ({time_slice_start})"
        )
    path = Path(pythia_config_path)
    if not path.exists():
        raise FileNotFoundError(f"pythia_config.json not found at {path}")

    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    nyers = time_slice_end - time_slice_start + 1
    new_sdate = f"{time_slice_start}-01-01"

    # Compute pfrst / plast based on planting_doy or preserve month-day.
    new_pfrst: Optional[str] = None
    new_plast: Optional[str] = None
    if planting_doy is not None:
        from datetime import date, timedelta

        pfrst_date = date(time_slice_start, 1, 1) + timedelta(
            days=planting_doy - 1
        )
        plast_doy_capped = min(planting_doy + planting_window_days, 365)
        plast_date = date(time_slice_start, 1, 1) + timedelta(
            days=plast_doy_capped - 1
        )
        new_pfrst = pfrst_date.isoformat()
        new_plast = plast_date.isoformat()

    setup = config.get("default_setup")
    if isinstance(setup, dict):
        setup["sdate"] = new_sdate
        if new_pfrst is not None:
            setup["pfrst"] = new_pfrst
        elif "pfrst" in setup and isinstance(setup["pfrst"], str):
            setup["pfrst"] = _replace_year_prefix(
                setup["pfrst"], time_slice_start
            )
        if new_plast is not None:
            setup["plast"] = new_plast
        elif "plast" in setup and isinstance(setup["plast"], str):
            setup["plast"] = _replace_year_prefix(
                setup["plast"], time_slice_start
            )

    runs = config.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            run["startYear"] = time_slice_start
            run["nyers"] = nyers

    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return config


def _replace_year_prefix(iso_date: str, new_year: int) -> str:
    """Replace the year component of a YYYY-MM-DD string.

    Preserves month-day. Returns the input unchanged when it is not
    in the expected ``YYYY-MM-DD`` shape (defensive).
    """
    if not isinstance(iso_date, str) or len(iso_date) < 10:
        return iso_date
    if iso_date[4] != "-" or iso_date[7] != "-":
        return iso_date
    return f"{new_year:04d}{iso_date[4:]}"
