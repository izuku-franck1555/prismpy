"""V2-22c-PRE.1.10 (D37) — per-cell `sources` cascade-provenance.

Each cell carries a `sources` block under
`{"climate": {...}, "soil": {...}}` with four fields per entry:

- `name`: source identifier (e.g., "iSDA", "HWSD", "AgERA5", "TAMSAT")
- `version`: source version string (e.g., "S3", "v2", "1.0.0")
- `cascade_rank`: int starting at 1 (1 = primary, 2 = first fallback, ...)
- `fallback_attempts`: list of `{source, reason}` records describing
  the rejected attempts before this rank's success

The cockpit drawer (V2-22c AC-14.3) reads these for the methodology
text — "Climate: AgERA5 v2.0 (rank 1 of 2 — iSDA failed, HWSD fallback
used)" — and Dr. Kofi's audit trail relies on the cascade_rank +
fallback_attempts being deterministic across runs.

D37 elision rule: when no source emitted a profile / ts for the cell,
the `sources` field is absent (the cell's missing-data state surfaces
via PRE.1.9 coverage checks, not via a half-populated `sources`).

This module covers the cell_summary read path. Source-loader
population of `metadata.cascade_rank` + `metadata.version` lands in
companion commits as the cascade orchestrator threads through.
The read path defaults to `cascade_rank=1` + `fallback_attempts=[]`
when the metadata isn't yet populated, so the field shape is stable
even mid-implementation.
"""
from __future__ import annotations

from datetime import date

import pytest

from prismpy.models.region import Region, BoundingBox
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.translators.base import UnifiedData


def _make_pipeline():
    return TranslationPipeline.__new__(TranslationPipeline)


def _make_unified(*, soil=None, climate=None, n_cells=2):
    cells = [
        GridCell(cell_id=i, lat=0.5, lon=0.5,
                 row=0, col=i, resolution="5arcmin")
        for i in range(n_cells)
    ]
    grid = SpatialGrid(
        bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        resolution="5arcmin", cells=cells,
    )
    return UnifiedData(
        region=Region(
            name="t", country="t", country_iso3="TST",
            bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        ),
        grid=grid,
        soil=soil if soil is not None else {},
        climate=climate if climate is not None else {},
    )


def _make_profile(*, source="iSDA", version=None, cascade_rank=None,
                  fallback_attempts=None):
    metadata = {}
    if version is not None:
        metadata["version"] = version
    if cascade_rank is not None:
        metadata["cascade_rank"] = cascade_rank
    if fallback_attempts is not None:
        metadata["fallback_attempts"] = fallback_attempts
    return SoilProfile(
        profile_id="p0", lat=0.5, lon=0.5, source=source,
        layers=[SoilLayer(depth_top=0, depth_bottom=0.2,
                          sand=40, clay=30, silt=30)],
        metadata=metadata,
    )


def _make_ts(*, source="NASA_POWER", version=None, cascade_rank=None,
             fallback_attempts=None, location_id=0):
    metadata = {}
    if version is not None:
        metadata["version"] = version
    if cascade_rank is not None:
        metadata["cascade_rank"] = cascade_rank
    if fallback_attempts is not None:
        metadata["fallback_attempts"] = fallback_attempts
    return ClimateTimeSeries(
        records=[ClimateRecord(
            date=date(2020, 1, 1), tmax=25.0, tmin=15.0,
            precip=0.0, srad=20.0,
        )],
        location_id=str(location_id), lat=0.5, lon=0.5,
        source=source, metadata=metadata,
    )


class TestSoilSourcesProjection:
    """V2-22c-PRE.1.10 — per-cell `sources.soil` reads from
    SoilProfile.metadata."""

    def test_default_cascade_rank_when_metadata_unpopulated(self):
        """Source loaders haven't yet populated cascade_rank — the
        read path defaults to cascade_rank=1 (primary success) so
        the schema shape is stable mid-implementation."""
        pipeline = _make_pipeline()
        soil = {0: _make_profile(source="iSDA")}
        unified = _make_unified(soil=soil, n_cells=1)
        out = pipeline._build_cell_summary(unified)
        cell = out["cells"][0]
        assert "sources" in cell
        assert "soil" in cell["sources"]
        sb = cell["sources"]["soil"]
        assert sb["name"] == "iSDA"
        assert sb["cascade_rank"] == 1
        assert sb["fallback_attempts"] == []
        assert sb["version"] is None

    def test_populated_metadata_surfaces_in_sources(self):
        """When a source loader populates metadata.version +
        cascade_rank, the cockpit-side `sources.soil` shape carries
        the values verbatim."""
        pipeline = _make_pipeline()
        soil = {0: _make_profile(
            source="HWSD", version="v2.0",
            cascade_rank=2,
            fallback_attempts=[{
                "source": "iSDA",
                "reason": "no_data_at_centroid",
            }],
        )}
        unified = _make_unified(soil=soil, n_cells=1)
        out = pipeline._build_cell_summary(unified)
        sb = out["cells"][0]["sources"]["soil"]
        assert sb["name"] == "HWSD"
        assert sb["version"] == "v2.0"
        assert sb["cascade_rank"] == 2
        assert sb["fallback_attempts"] == [
            {"source": "iSDA", "reason": "no_data_at_centroid"},
        ]

    def test_metadata_source_overrides_profile_source_field(self):
        """If metadata explicitly carries a `source` key (the
        canonical name from the cascade-tracked decision), it wins
        over the profile.source attribute. This lets the source
        loader emit a richer name without modifying SoilProfile."""
        pipeline = _make_pipeline()
        profile = SoilProfile(
            profile_id="p0", lat=0.5, lon=0.5,
            source="iSDA",  # set on field
            layers=[SoilLayer(depth_top=0, depth_bottom=0.2,
                              sand=40, clay=30, silt=30)],
            metadata={"source": "iSDA Africa (S3)", "version": "S3"},
        )
        unified = _make_unified(soil={0: profile}, n_cells=1)
        out = pipeline._build_cell_summary(unified)
        sb = out["cells"][0]["sources"]["soil"]
        assert sb["name"] == "iSDA Africa (S3)"
        assert sb["version"] == "S3"


class TestClimateSourcesProjection:
    """V2-22c-PRE.1.10 — per-cell `sources.climate` reads from
    ClimateTimeSeries.metadata."""

    def test_default_cascade_rank_when_metadata_unpopulated(self):
        pipeline = _make_pipeline()
        climate = {0: _make_ts(source="NASA_POWER")}
        unified = _make_unified(climate=climate, n_cells=1)
        out = pipeline._build_cell_summary(unified)
        cell = out["cells"][0]
        sb = cell["sources"]["climate"]
        assert sb["name"] == "NASA_POWER"
        assert sb["cascade_rank"] == 1
        assert sb["fallback_attempts"] == []
        assert sb["version"] is None

    def test_populated_metadata_carries_version(self):
        pipeline = _make_pipeline()
        climate = {0: _make_ts(
            source="AgERA5", version="2.0",
        )}
        unified = _make_unified(climate=climate, n_cells=1)
        out = pipeline._build_cell_summary(unified)
        sb = out["cells"][0]["sources"]["climate"]
        assert sb["name"] == "AgERA5"
        assert sb["version"] == "2.0"


class TestSourcesElisionWhenNoData:
    """V2-22c-PRE.1.10 (D37) — when neither climate nor soil emitted
    for a cell, the `sources` field is absent. The cell's
    missing-data state surfaces via PRE.1.9 coverage checks; a
    half-populated `sources` block would be ambiguous."""

    def test_no_climate_no_soil_elides_sources_field(self):
        pipeline = _make_pipeline()
        unified = _make_unified(n_cells=1)  # empty soil + climate
        out = pipeline._build_cell_summary(unified)
        cell = out["cells"][0]
        assert "sources" not in cell

    def test_only_climate_yields_climate_only_sources(self):
        """A cell with climate but no soil emits `sources.climate`
        but no `sources.soil` — partial population is allowed by
        the spec; only fully-empty triggers elision."""
        pipeline = _make_pipeline()
        climate = {0: _make_ts(source="NASA_POWER")}
        unified = _make_unified(climate=climate, n_cells=1)
        out = pipeline._build_cell_summary(unified)
        sb = out["cells"][0]["sources"]
        assert "climate" in sb
        assert "soil" not in sb

    def test_only_soil_yields_soil_only_sources(self):
        pipeline = _make_pipeline()
        soil = {0: _make_profile(source="iSDA")}
        unified = _make_unified(soil=soil, n_cells=1)
        out = pipeline._build_cell_summary(unified)
        sb = out["cells"][0]["sources"]
        assert "soil" in sb
        assert "climate" not in sb


class TestSourcesShapeAcrossCells:
    """V2-22c-PRE.1.10 — schema invariant across mixed
    populations. Each cell that has data carries the canonical
    four-field shape; cells without data elide the block."""

    def test_mixed_population_emits_field_only_for_cells_with_data(self):
        pipeline = _make_pipeline()
        soil = {0: _make_profile(source="iSDA")}
        climate = {1: _make_ts(source="NASA_POWER")}
        unified = _make_unified(soil=soil, climate=climate, n_cells=3)
        out = pipeline._build_cell_summary(unified)
        cells = out["cells"]
        # Cell 0: soil only
        assert "sources" in cells[0]
        assert "soil" in cells[0]["sources"]
        assert "climate" not in cells[0]["sources"]
        # Cell 1: climate only
        assert "sources" in cells[1]
        assert "climate" in cells[1]["sources"]
        assert "soil" not in cells[1]["sources"]
        # Cell 2: nothing → no sources block
        assert "sources" not in cells[2]

    def test_every_present_block_has_four_canonical_fields(self):
        """Schema discipline — when present, each `sources.<class>`
        carries exactly the canonical fields, no extras and none
        missing."""
        pipeline = _make_pipeline()
        soil = {0: _make_profile(source="iSDA",
                                 version="S3", cascade_rank=1)}
        climate = {0: _make_ts(source="AgERA5",
                                version="2.0")}
        unified = _make_unified(soil=soil, climate=climate, n_cells=1)
        out = pipeline._build_cell_summary(unified)
        sb = out["cells"][0]["sources"]
        expected_keys = {"name", "version", "cascade_rank", "fallback_attempts"}
        assert set(sb["soil"].keys()) == expected_keys
        assert set(sb["climate"].keys()) == expected_keys


class TestSourcesCascadeFallback:
    """V2-22c-PRE.1.10 (D37) — when iSDA falls through and HWSD
    serves the cell, the cockpit drawer reads cascade_rank=2 +
    fallback_attempts=[{source: 'iSDA Africa', reason: ...}]
    from the cell's `sources.soil` block. The orchestrator at
    `executor.py` populates this metadata after the HWSD loader
    returns, so source-loader defaults (cascade_rank=1) get
    overridden for the fallback case.

    This test exercises the read path against a profile whose
    metadata simulates the orchestrator-populated fallback shape;
    the orchestrator path itself is integration-tested at Gate B
    via a real iSDA-fails-HWSD-succeeds run."""

    def test_cascade_rank_2_with_populated_fallback_attempts(self):
        pipeline = _make_pipeline()
        # Profile served by HWSD after iSDA fell through.
        soil = {0: _make_profile(
            source="HWSD", version="v2.0",
            cascade_rank=2,
            fallback_attempts=[{
                "source": "iSDA Africa",
                "reason": "no_data_at_centroid",
            }],
        )}
        unified = _make_unified(soil=soil, n_cells=1)
        out = pipeline._build_cell_summary(unified)
        sb = out["cells"][0]["sources"]["soil"]
        assert sb["cascade_rank"] == 2
        assert len(sb["fallback_attempts"]) == 1
        assert sb["fallback_attempts"][0]["source"] == "iSDA Africa"
        # Reason is a free-form string the cockpit renders verbatim;
        # assert it's at least non-empty.
        assert sb["fallback_attempts"][0]["reason"]

    def test_cascade_rank_1_has_empty_fallback_attempts(self):
        """Loader-default path: when cascade_rank=1 (primary
        success), fallback_attempts is the empty list. Cockpit
        renders "Source: iSDA Africa S3" without any fallback
        annotation."""
        pipeline = _make_pipeline()
        soil = {0: _make_profile(
            source="iSDA", version="S3", cascade_rank=1,
            fallback_attempts=[],
        )}
        unified = _make_unified(soil=soil, n_cells=1)
        out = pipeline._build_cell_summary(unified)
        sb = out["cells"][0]["sources"]["soil"]
        assert sb["cascade_rank"] == 1
        assert sb["fallback_attempts"] == []
