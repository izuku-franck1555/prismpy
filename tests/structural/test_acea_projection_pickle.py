"""Structural pin: AC-G-7b ACEA pickle projection-climate path.

Sprint G AC-G-7b: ACEA's per-cell climate pickle writer accepts the
``ClimateKind`` discriminator. Same pickle binary shape on both paths
(tuple of 4 numpy float32 arrays — deterministic by construction);
the projection path additionally writes a sidecar ``.meta.json`` per
cell capturing ``gcm_source`` / ``bias_correction_method`` /
``time_slice`` so downstream consumers can introspect provenance
without re-reading the parent manifest.

Tests:

* §1 ``ProjectionClimateMeta`` Pydantic schema (required fields,
  bounds, time-slice ordering, extra-forbid)
* §2 ACEA OBSERVED path unchanged (no sidecar)
* §3 ACEA PROJECTION path emits pickle + sidecar
* §4 Sidecar contents match ProjectionClimateMeta schema
* §5 PROJECTION path requires projection_meta (raises if None)
* §6 Determinism — same input + same kind → byte-identical pickle
* §7 Sibling-sweep — ACEA writer imports canonical helpers
"""

from __future__ import annotations

import datetime as dt
import inspect
import json
from pathlib import Path
from typing import List

import pytest
from pydantic import ValidationError

from prismpy.harmonize.climate_kind import ClimateKind
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.scenario import (
    BiasCorrectionMethod,
    ProjectionClimateMeta,
)


# ── Fixture builders ─────────────────────────────────────────────────


def _make_records(n_days: int = 5) -> List[ClimateRecord]:
    base_date = dt.date(2046, 6, 1)
    return [
        ClimateRecord(
            date=base_date + dt.timedelta(days=i),
            tmax=30.0 + i * 0.5,
            tmin=20.0 + i * 0.5,
            precip=2.5 if i % 2 == 0 else 0.0,
            srad=20.0 + i * 0.1,
            et0=4.5 + i * 0.05,
        )
        for i in range(n_days)
    ]


def _make_ts(records: List[ClimateRecord]) -> ClimateTimeSeries:
    return ClimateTimeSeries(
        location_id=1,
        lat=13.5,
        lon=2.1,
        source="synthetic",
        records=records,
    )


def _instantiate_minimal_acea(tmp_path: Path):
    from prismpy.translators.acea.translator import AceaTranslator

    inst = AceaTranslator.__new__(AceaTranslator)
    inst.output_dir = tmp_path / "pkg"
    (inst.output_dir / "climate").mkdir(parents=True, exist_ok=True)
    inst.provenance = None
    return inst


def _valid_meta_kwargs() -> dict:
    return {
        "gcm_source": "gfdl-esm4",
        "bias_correction_method": BiasCorrectionMethod.QUANTILE_MAPPING,
        "time_slice_start": 2046,
        "time_slice_end": 2065,
    }


# ── §1 ProjectionClimateMeta Pydantic schema ─────────────────────────


def test_projection_climate_meta_required_fields() -> None:
    """All four core fields (gcm_source, bias_correction_method,
    time_slice_start, time_slice_end) are required."""
    base = _valid_meta_kwargs()
    for missing in (
        "gcm_source",
        "bias_correction_method",
        "time_slice_start",
        "time_slice_end",
    ):
        partial = {k: v for k, v in base.items() if k != missing}
        with pytest.raises(ValidationError):
            ProjectionClimateMeta(**partial)


def test_projection_meta_time_slice_ordering() -> None:
    kwargs = _valid_meta_kwargs()
    kwargs["time_slice_start"] = 2065
    kwargs["time_slice_end"] = 2046
    with pytest.raises(ValidationError):
        ProjectionClimateMeta(**kwargs)


def test_projection_meta_extra_fields_rejected() -> None:
    kwargs = _valid_meta_kwargs()
    kwargs["typo_field"] = "should not be accepted"
    with pytest.raises(ValidationError):
        ProjectionClimateMeta(**kwargs)


def test_projection_meta_cell_id_optional() -> None:
    """``cell_id`` is optional at the schema level (writer enforces
    AC-specific use)."""
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())
    assert meta.cell_id is None


def test_projection_meta_cell_id_must_be_non_negative() -> None:
    kwargs = _valid_meta_kwargs()
    kwargs["cell_id"] = -1
    with pytest.raises(ValidationError):
        ProjectionClimateMeta(**kwargs)


def test_projection_meta_round_trip_through_model_dump() -> None:
    meta_a = ProjectionClimateMeta(**_valid_meta_kwargs(), cell_id=42)
    meta_b = ProjectionClimateMeta.model_validate(meta_a.model_dump())
    assert meta_a.model_dump() == meta_b.model_dump()


# ── §2 ACEA OBSERVED path unchanged ──────────────────────────────────


def test_acea_observed_emits_pickle_only_no_sidecar(tmp_path: Path) -> None:
    inst = _instantiate_minimal_acea(tmp_path)
    ts = _make_ts(_make_records())
    files = inst._generate_climate_pickles(
        {1: ts},
        cell_ids_30arcmin=[1],
        climate_name="climate",
        climate_kind=ClimateKind.OBSERVED,
    )
    pkls = [f for f in files if f.suffix == ".pckl"]
    metas = [f for f in files if f.suffix == ".json"]
    assert len(pkls) == 1
    assert len(metas) == 0  # OBSERVED never writes sidecar


def test_acea_default_kind_is_observed(tmp_path: Path) -> None:
    inst = _instantiate_minimal_acea(tmp_path)
    ts = _make_ts(_make_records())
    files = inst._generate_climate_pickles(
        {1: ts}, cell_ids_30arcmin=[1], climate_name="climate"
    )
    metas = [f for f in files if f.suffix == ".json"]
    assert len(metas) == 0  # No sidecar on default-OBSERVED


# ── §3 + §4 ACEA PROJECTION path emits pickle + sidecar ──────────────


def test_acea_projection_emits_pickle_and_sidecar(tmp_path: Path) -> None:
    inst = _instantiate_minimal_acea(tmp_path)
    ts = _make_ts(_make_records())
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())
    files = inst._generate_climate_pickles(
        {1: ts},
        cell_ids_30arcmin=[1],
        climate_name="climate",
        climate_kind=ClimateKind.PROJECTION,
        projection_meta=meta,
    )
    pkls = [f for f in files if f.suffix == ".pckl"]
    metas = [f for f in files if f.suffix == ".json"]
    assert len(pkls) == 1
    assert len(metas) == 1


def test_acea_projection_sidecar_carries_canonical_fields(
    tmp_path: Path,
) -> None:
    inst = _instantiate_minimal_acea(tmp_path)
    ts = _make_ts(_make_records())
    meta = ProjectionClimateMeta(**_valid_meta_kwargs(), scenario_label="niamey-millet")
    files = inst._generate_climate_pickles(
        {1: ts},
        cell_ids_30arcmin=[1],
        climate_name="climate",
        climate_kind=ClimateKind.PROJECTION,
        projection_meta=meta,
    )
    sidecar = next(f for f in files if f.suffix == ".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["gcm_source"] == "gfdl-esm4"
    assert payload["bias_correction_method"] == "quantile_mapping"
    assert payload["time_slice_start"] == 2046
    assert payload["time_slice_end"] == 2065
    assert payload["cell_id"] == 1  # 30-arcmin id (from cell_ids_30arcmin)
    assert payload["scenario_label"] == "niamey-millet"


def test_acea_projection_sidecar_validates_against_schema(
    tmp_path: Path,
) -> None:
    """Round-trip: write sidecar → re-parse → must validate against
    ProjectionClimateMeta. Catches accidental field shape drift."""
    inst = _instantiate_minimal_acea(tmp_path)
    ts = _make_ts(_make_records())
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())
    files = inst._generate_climate_pickles(
        {1: ts},
        cell_ids_30arcmin=[1],
        climate_name="climate",
        climate_kind=ClimateKind.PROJECTION,
        projection_meta=meta,
    )
    sidecar = next(f for f in files if f.suffix == ".json")
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    # Must validate cleanly against the canonical schema.
    re_validated = ProjectionClimateMeta.model_validate(raw)
    assert re_validated.cell_id == 1


def test_acea_projection_sidecar_per_cell(tmp_path: Path) -> None:
    """One sidecar per cell — multi-cell input emits multi sidecar."""
    inst = _instantiate_minimal_acea(tmp_path)
    ts1 = _make_ts(_make_records())
    ts2 = _make_ts(_make_records())
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())
    files = inst._generate_climate_pickles(
        {1: ts1, 2: ts2},
        cell_ids_30arcmin=[1, 2],
        climate_name="climate",
        climate_kind=ClimateKind.PROJECTION,
        projection_meta=meta,
    )
    metas = [f for f in files if f.suffix == ".json"]
    assert len(metas) == 2
    # Each sidecar carries its own cell_id. Filename shape:
    # ``climate_<cell_id>.meta.json`` so ``.stem`` yields
    # ``climate_<cell_id>.meta``.
    for sidecar in metas:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        # Strip the ``.meta`` infix; the trailing token is the cell id
        cell_id_in_filename = int(
            sidecar.stem.removesuffix(".meta").split("_")[-1]
        )
        assert payload["cell_id"] == cell_id_in_filename


# ── §5 PROJECTION path requires projection_meta ──────────────────────


def test_acea_projection_without_meta_raises(tmp_path: Path) -> None:
    inst = _instantiate_minimal_acea(tmp_path)
    ts = _make_ts(_make_records())
    with pytest.raises(ValueError, match="projection_meta"):
        inst._generate_climate_pickles(
            {1: ts},
            cell_ids_30arcmin=[1],
            climate_name="climate",
            climate_kind=ClimateKind.PROJECTION,
            projection_meta=None,
        )


# ── §6 Determinism — same input + same kind → byte-identical ─────────


def test_acea_pickle_deterministic_observed(tmp_path: Path) -> None:
    """Same input + same kind → byte-identical pickle (CC-G-7
    + AC-G-13 deliverable hash precondition)."""
    inst_a = _instantiate_minimal_acea(tmp_path / "a")
    inst_b = _instantiate_minimal_acea(tmp_path / "b")
    ts = _make_ts(_make_records())
    files_a = inst_a._generate_climate_pickles(
        {1: ts}, cell_ids_30arcmin=[1], climate_name="climate"
    )
    files_b = inst_b._generate_climate_pickles(
        {1: ts}, cell_ids_30arcmin=[1], climate_name="climate"
    )
    pkl_a = next(f for f in files_a if f.suffix == ".pckl")
    pkl_b = next(f for f in files_b if f.suffix == ".pckl")
    assert pkl_a.read_bytes() == pkl_b.read_bytes()


def test_acea_pickle_deterministic_projection(tmp_path: Path) -> None:
    inst_a = _instantiate_minimal_acea(tmp_path / "a")
    inst_b = _instantiate_minimal_acea(tmp_path / "b")
    ts = _make_ts(_make_records())
    meta = ProjectionClimateMeta(**_valid_meta_kwargs())
    files_a = inst_a._generate_climate_pickles(
        {1: ts},
        cell_ids_30arcmin=[1],
        climate_name="climate",
        climate_kind=ClimateKind.PROJECTION,
        projection_meta=meta,
    )
    files_b = inst_b._generate_climate_pickles(
        {1: ts},
        cell_ids_30arcmin=[1],
        climate_name="climate",
        climate_kind=ClimateKind.PROJECTION,
        projection_meta=meta,
    )
    # Both pickle and sidecar must be byte-identical
    pkl_a = next(f for f in files_a if f.suffix == ".pckl")
    pkl_b = next(f for f in files_b if f.suffix == ".pckl")
    assert pkl_a.read_bytes() == pkl_b.read_bytes()
    meta_a = next(f for f in files_a if f.suffix == ".json")
    meta_b = next(f for f in files_b if f.suffix == ".json")
    assert meta_a.read_bytes() == meta_b.read_bytes()


# ── §7 Sibling-sweep + signature pin ─────────────────────────────────


def test_acea_writer_signature_accepts_climate_kind_keyword() -> None:
    """ACEA's ``_generate_climate_pickles`` must accept the
    ``climate_kind`` keyword."""
    from prismpy.translators.acea.translator import AceaTranslator

    sig = inspect.signature(AceaTranslator._generate_climate_pickles)
    assert "climate_kind" in sig.parameters
    assert "projection_meta" in sig.parameters


def test_acea_translator_imports_canonical_helpers() -> None:
    """Per durable §24: ACEA writer must import ClimateKind +
    ProjectionClimateMeta from canonical sources, not redefine them
    locally."""
    import ast

    project_root = Path(__file__).resolve().parents[2]
    src = (
        project_root / "src/prismpy/translators/acea/translator.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = {"ClimateKind": False, "ProjectionClimateMeta": False}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "prismpy.harmonize.climate_kind" and any(
                a.name == "ClimateKind" for a in node.names
            ):
                found["ClimateKind"] = True
            if node.module == "prismpy.models.scenario" and any(
                a.name == "ProjectionClimateMeta" for a in node.names
            ):
                found["ProjectionClimateMeta"] = True
    missing = [k for k, v in found.items() if not v]
    assert not missing, (
        f"ACEA translator must import {missing} from canonical sources "
        "(durable §24 canonical-source-or-pin)."
    )
