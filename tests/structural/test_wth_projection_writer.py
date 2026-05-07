"""Structural pin: AC-G-7a CRAFT+PYTHIA WTH 8-col projection path.

Sprint G AC-G-7a: both DSSAT-format translators (CRAFT, PYTHIA) accept
a ``ClimateKind`` discriminator on ``_generate_weather_files``.

* ``ClimateKind.OBSERVED`` (default): legacy emission path unchanged.
  CRAFT emits 5-column ``YRDOY/SRAD/TMAX/TMIN/RAIN``; PYTHIA emits the
  existing 8-column WTH with MISDAT (-99.0) for missing tdew/hurs/wind.
* ``ClimateKind.PROJECTION``: 8-column WTH on both translators with
  TDEW derived via Tetens (:func:`derive_tdew_for_record_or`) when
  source ``record.tdew`` is None but ``record.rh`` and ``record.tmean``
  are present.

Tests:

* §1 ``ClimateKind`` enum members + values
* §2 CRAFT OBSERVED → 5-col output preserved
* §3 CRAFT PROJECTION → 8-col header + Tetens-derived TDEW
* §4 PYTHIA OBSERVED → 8-col with MISDAT TDEW preserved
* §5 PYTHIA PROJECTION → 8-col with Tetens-derived TDEW
* §6 Saturation invariant on derived TDEW (TDEW ≤ tmean)
* §7 Determinism — same input + kind → byte-identical output
* §8 Sibling-sweep — both translators import from canonical helper
* §9 Default parameter is OBSERVED (backward-compat)
"""

from __future__ import annotations

import datetime as dt
import inspect
from pathlib import Path
from typing import Any, Dict, List

import pytest

from prismpy.harmonize.climate_kind import ClimateKind
from prismpy.harmonize.tetens import derive_tdew, derive_tdew_for_record_or
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries


# ── Fixture builders ─────────────────────────────────────────────────


def _make_records(
    *,
    n_days: int = 5,
    with_tdew: bool = False,
    with_rh: bool = True,
    with_wind: bool = True,
) -> List[ClimateRecord]:
    """Build ``n_days`` ClimateRecord objects with controllable
    optional-field presence."""
    records: List[ClimateRecord] = []
    base_date = dt.date(2046, 6, 1)
    for i in range(n_days):
        rec = ClimateRecord(
            date=base_date + dt.timedelta(days=i),
            tmax=30.0 + i * 0.5,
            tmin=20.0 + i * 0.5,
            precip=2.5 if i % 2 == 0 else 0.0,
            srad=20.0 + i * 0.1,
            wind=3.0 if with_wind else None,
            rh=65.0 if with_rh else None,
            tdew=18.0 if with_tdew else None,
        )
        records.append(rec)
    return records


def _make_ts(records: List[ClimateRecord]) -> ClimateTimeSeries:
    return ClimateTimeSeries(
        location_id=1,
        lat=13.5,
        lon=2.1,
        source="synthetic",
        records=records,
    )


# ── §1 ClimateKind enum ─────────────────────────────────────────────


def test_climate_kind_enum_members() -> None:
    members = {m.name: m.value for m in ClimateKind}
    assert members == {"OBSERVED": "observed", "PROJECTION": "projection"}


# ── §2 CRAFT OBSERVED unchanged ──────────────────────────────────────


def _craft_translator_class():
    from prismpy.translators.craft.translator import CraftTranslator

    return CraftTranslator


def _pythia_translator_class():
    from prismpy.translators.pythia.translator import PythiaTranslator

    return PythiaTranslator


def _instantiate_minimal_craft(tmp_path: Path):
    """Build a minimally-configured CraftTranslator with output_dir
    set to ``tmp_path / 'pkg'`` and ``weather/`` precreated. The
    weather-file generator only needs ``self.output_dir`` + the
    ``_to_craft_cellid`` helper; we don't drive the full translate
    pipeline."""
    cls = _craft_translator_class()
    inst = cls.__new__(cls)
    inst.output_dir = tmp_path / "pkg"
    (inst.output_dir / "weather").mkdir(parents=True, exist_ok=True)
    return inst


def _instantiate_minimal_pythia(tmp_path: Path):
    cls = _pythia_translator_class()
    inst = cls.__new__(cls)
    inst.output_dir = tmp_path / "pkg"
    (inst.output_dir / "weather").mkdir(parents=True, exist_ok=True)
    # PYTHIA's _calculate_tav_amp + downstream helpers reference
    # ``self.provenance`` for record_decision; stub to None so the
    # writer skips the provenance-record branches in this test
    # context. The writer's CORE behavior (header + data rows + TDEW
    # derivation) doesn't depend on provenance.
    inst.provenance = None
    return inst


def test_craft_observed_emits_5_column_header(tmp_path: Path) -> None:
    inst = _instantiate_minimal_craft(tmp_path)
    ts = _make_ts(_make_records())
    files = inst._generate_weather_files(
        {1: ts}, climate_kind=ClimateKind.OBSERVED
    )
    assert len(files) == 1
    header = files[0].read_text(encoding="utf-8").splitlines()[0]
    assert header == "YRDOY\tSRAD\tTMAX\tTMIN\tRAIN"


def test_craft_default_kind_is_observed(tmp_path: Path) -> None:
    """Backward-compat: omitting climate_kind preserves OBSERVED."""
    inst = _instantiate_minimal_craft(tmp_path)
    ts = _make_ts(_make_records())
    files = inst._generate_weather_files({1: ts})
    header = files[0].read_text(encoding="utf-8").splitlines()[0]
    assert header == "YRDOY\tSRAD\tTMAX\tTMIN\tRAIN"


# ── §3 CRAFT PROJECTION — 8-col header + Tetens-derived TDEW ─────────


def test_craft_projection_emits_8_column_header(tmp_path: Path) -> None:
    inst = _instantiate_minimal_craft(tmp_path)
    ts = _make_ts(_make_records(with_tdew=False, with_rh=True))
    files = inst._generate_weather_files(
        {1: ts}, climate_kind=ClimateKind.PROJECTION
    )
    header = files[0].read_text(encoding="utf-8").splitlines()[0]
    assert header == "YRDOY\tSRAD\tTMAX\tTMIN\tRAIN\tTDEW\tRHUM\tWIND"


def test_craft_projection_derives_tdew_via_tetens(tmp_path: Path) -> None:
    """When source has rh + tmean but tdew=None, the projection writer
    must emit a Tetens-derived TDEW column (NOT MISDAT)."""
    inst = _instantiate_minimal_craft(tmp_path)
    records = _make_records(with_tdew=False, with_rh=True)
    ts = _make_ts(records)
    files = inst._generate_weather_files(
        {1: ts}, climate_kind=ClimateKind.PROJECTION
    )
    body_lines = files[0].read_text(encoding="utf-8").splitlines()[1:]
    assert len(body_lines) == len(records)
    # Each row's TDEW column (index 5) must equal Tetens of the row's
    # rh + tmean — not -99.0.
    for line, rec in zip(body_lines, records):
        cols = line.split("\t")
        emitted_tdew = float(cols[5])
        expected = round(derive_tdew(rec.tmean, rec.rh), 1)
        assert abs(emitted_tdew - expected) < 0.05, (
            f"TDEW emission drift: emitted {emitted_tdew} vs "
            f"Tetens({rec.tmean}, {rec.rh}) = {expected}"
        )
        assert emitted_tdew != -99.0


def test_craft_projection_falls_back_to_misdat_when_no_rh(
    tmp_path: Path,
) -> None:
    """If both tdew AND rh are None, the projection path emits
    MISDAT — honest-signal "data genuinely unavailable"."""
    inst = _instantiate_minimal_craft(tmp_path)
    records = _make_records(with_tdew=False, with_rh=False)
    ts = _make_ts(records)
    files = inst._generate_weather_files(
        {1: ts}, climate_kind=ClimateKind.PROJECTION
    )
    body_lines = files[0].read_text(encoding="utf-8").splitlines()[1:]
    for line in body_lines:
        cols = line.split("\t")
        assert float(cols[5]) == -99.0  # TDEW MISDAT


# ── §4 PYTHIA OBSERVED unchanged ─────────────────────────────────────


def test_pythia_observed_emits_misdat_tdew_when_source_is_none(
    tmp_path: Path,
) -> None:
    """Observed path keeps the legacy MISDAT fallback."""
    inst = _instantiate_minimal_pythia(tmp_path)
    records = _make_records(with_tdew=False, with_rh=True)
    ts = _make_ts(records)
    files = inst._generate_weather_files(
        {1: ts}, climate_kind=ClimateKind.OBSERVED
    )
    text = files[0].read_text(encoding="utf-8")
    # Find the data section (after the @  DATE header).
    lines = text.splitlines()
    data_header_idx = next(
        i for i, line in enumerate(lines) if line.startswith("@  DATE")
    )
    data_lines = lines[data_header_idx + 1:]
    for line in data_lines:
        cols = line.split()
        # Format: yrdoy srad tmax tmin rain tdew rhum wind
        # All MISDAT-fallback cells emit -99.0 in the TDEW slot.
        assert float(cols[5]) == -99.0


def test_pythia_default_kind_is_observed(tmp_path: Path) -> None:
    inst = _instantiate_minimal_pythia(tmp_path)
    ts = _make_ts(_make_records(with_tdew=False, with_rh=True))
    files = inst._generate_weather_files({1: ts})
    text = files[0].read_text(encoding="utf-8")
    lines = text.splitlines()
    data_header_idx = next(
        i for i, line in enumerate(lines) if line.startswith("@  DATE")
    )
    cols = lines[data_header_idx + 1].split()
    assert float(cols[5]) == -99.0  # OBSERVED default → MISDAT TDEW


# ── §5 PYTHIA PROJECTION — Tetens-derived TDEW ───────────────────────


def test_pythia_projection_derives_tdew_via_tetens(
    tmp_path: Path,
) -> None:
    inst = _instantiate_minimal_pythia(tmp_path)
    records = _make_records(with_tdew=False, with_rh=True)
    ts = _make_ts(records)
    files = inst._generate_weather_files(
        {1: ts}, climate_kind=ClimateKind.PROJECTION
    )
    text = files[0].read_text(encoding="utf-8")
    lines = text.splitlines()
    data_header_idx = next(
        i for i, line in enumerate(lines) if line.startswith("@  DATE")
    )
    data_lines = lines[data_header_idx + 1:]
    for line, rec in zip(data_lines, records):
        cols = line.split()
        emitted_tdew = float(cols[5])
        expected = round(derive_tdew(rec.tmean, rec.rh), 1)
        assert abs(emitted_tdew - expected) < 0.1, (
            f"PYTHIA projection TDEW drift: {emitted_tdew} vs "
            f"Tetens({rec.tmean}, {rec.rh}) = {expected}"
        )
        assert emitted_tdew != -99.0


# ── §6 Saturation invariant on derived TDEW ──────────────────────────


def test_projection_derived_tdew_below_tmean(tmp_path: Path) -> None:
    """Sub-saturation invariant: derived TDEW < tmean for hurs < 100."""
    inst = _instantiate_minimal_craft(tmp_path)
    records = _make_records(with_tdew=False, with_rh=True)
    ts = _make_ts(records)
    files = inst._generate_weather_files(
        {1: ts}, climate_kind=ClimateKind.PROJECTION
    )
    body_lines = files[0].read_text(encoding="utf-8").splitlines()[1:]
    for line, rec in zip(body_lines, records):
        cols = line.split("\t")
        emitted_tdew = float(cols[5])
        # rh=65% < 100% → tdew < tmean
        assert emitted_tdew < rec.tmean


# ── §7 Determinism — same input + kind → byte-identical output ───────


def test_craft_projection_writer_deterministic(tmp_path: Path) -> None:
    """CC-G-7 + AC-G-13 deliverable hash precondition: same input +
    same kind → byte-identical output across two invocations."""
    inst_a = _instantiate_minimal_craft(tmp_path / "a")
    inst_b = _instantiate_minimal_craft(tmp_path / "b")
    records = _make_records(with_tdew=False, with_rh=True)
    ts = _make_ts(records)
    files_a = inst_a._generate_weather_files(
        {1: ts}, climate_kind=ClimateKind.PROJECTION
    )
    files_b = inst_b._generate_weather_files(
        {1: ts}, climate_kind=ClimateKind.PROJECTION
    )
    assert files_a[0].read_bytes() == files_b[0].read_bytes()


def test_pythia_projection_writer_deterministic(tmp_path: Path) -> None:
    inst_a = _instantiate_minimal_pythia(tmp_path / "a")
    inst_b = _instantiate_minimal_pythia(tmp_path / "b")
    records = _make_records(with_tdew=False, with_rh=True)
    ts = _make_ts(records)
    files_a = inst_a._generate_weather_files(
        {1: ts}, climate_kind=ClimateKind.PROJECTION
    )
    files_b = inst_b._generate_weather_files(
        {1: ts}, climate_kind=ClimateKind.PROJECTION
    )
    assert files_a[0].read_bytes() == files_b[0].read_bytes()


# ── §8 Sibling-sweep — canonical-source pin ──────────────────────────


def test_both_translators_import_canonical_helpers() -> None:
    """Per durable §24: CRAFT + PYTHIA writers MUST import
    ``ClimateKind`` and ``derive_tdew_for_record_or`` from the
    canonical harmonize/ modules. No reimplementation."""
    import ast

    project_root = Path(__file__).resolve().parents[2]
    for translator_path in (
        "src/prismpy/translators/craft/translator.py",
        "src/prismpy/translators/pythia/translator.py",
    ):
        src = (project_root / translator_path).read_text(encoding="utf-8")
        tree = ast.parse(src)
        imports_climate_kind = False
        imports_tetens_helper = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "prismpy.harmonize.climate_kind" and any(
                    a.name == "ClimateKind" for a in node.names
                ):
                    imports_climate_kind = True
                if node.module == "prismpy.harmonize.tetens" and any(
                    a.name == "derive_tdew_for_record_or" for a in node.names
                ):
                    imports_tetens_helper = True
        assert imports_climate_kind, (
            f"{translator_path} must import ClimateKind from "
            "prismpy.harmonize.climate_kind (durable §24 canonical-source)."
        )
        assert imports_tetens_helper, (
            f"{translator_path} must import derive_tdew_for_record_or "
            "from prismpy.harmonize.tetens (durable §24 canonical-source)."
        )


def test_writer_signatures_accept_climate_kind_keyword() -> None:
    """Both translators' ``_generate_weather_files`` must accept the
    ``climate_kind`` keyword. Catches a regression where one
    translator drops the discriminator silently."""
    craft_cls = _craft_translator_class()
    pythia_cls = _pythia_translator_class()
    for cls, name in ((craft_cls, "CRAFT"), (pythia_cls, "PYTHIA")):
        sig = inspect.signature(cls._generate_weather_files)
        assert "climate_kind" in sig.parameters, (
            f"{name}._generate_weather_files missing climate_kind parameter"
        )


# ── §9 derive_tdew_for_record_or — projection-or-fallback chain ──────


def test_derive_tdew_for_record_or_returns_explicit_when_present() -> None:
    """When ``explicit_tdew`` is non-None, return it directly."""
    result = derive_tdew_for_record_or(
        explicit_tdew=15.5,
        tmean_celsius=25.0,
        hurs_pct=60.0,
        fallback=-99.0,
    )
    assert result == 15.5


def test_derive_tdew_for_record_or_derives_when_explicit_is_none() -> None:
    result = derive_tdew_for_record_or(
        explicit_tdew=None,
        tmean_celsius=25.0,
        hurs_pct=60.0,
        fallback=-99.0,
    )
    expected = derive_tdew(25.0, 60.0)
    assert abs(result - expected) < 1e-9


def test_derive_tdew_for_record_or_falls_back_when_rh_missing() -> None:
    result = derive_tdew_for_record_or(
        explicit_tdew=None,
        tmean_celsius=25.0,
        hurs_pct=None,
        fallback=-99.0,
    )
    assert result == -99.0


def test_derive_tdew_for_record_or_falls_back_when_tmean_missing() -> None:
    result = derive_tdew_for_record_or(
        explicit_tdew=None,
        tmean_celsius=None,
        hurs_pct=60.0,
        fallback=-99.0,
    )
    assert result == -99.0


def test_derive_tdew_for_record_or_falls_back_on_out_of_bounds() -> None:
    """Out-of-bound inputs (e.g., kelvin upstream) fall back to the
    sentinel rather than propagating ValueError."""
    result = derive_tdew_for_record_or(
        explicit_tdew=None,
        tmean_celsius=298.15,  # kelvin — out of [-90, 70] bounds
        hurs_pct=60.0,
        fallback=-99.0,
    )
    assert result == -99.0
