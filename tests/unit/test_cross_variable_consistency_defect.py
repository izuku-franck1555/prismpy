"""cross_variable_consistency emits per-cell PHYSICAL-IMPOSSIBILITY defects.

A fail-tier cross-variable violation (tmax<tmin, negative precip/srad/wind) is a
physical impossibility. The check marks each affected cell as a per-cell DEFECT via
``details.violation_details`` (``cell_id`` + ``defect=True``), so the executor pivot
stamps that cell's ``failed_checks`` with a defect. Downstream (prismweb) a LOCALIZABLE
impossibility then routes to "exclude the N cells and run again" (keep the good cells),
matching the value_range defect family — NOT a package-wide re-scope.

The second test is the load-bearing PREMISE gate: the affected cell_id (from the climate
dict) MATCHES ``cells_by_id`` end-to-end through the real validation + pivot, so the
per-cell defect actually lands on the cell_summary (never silently dropped).
"""
from __future__ import annotations

from datetime import date as _date

from prismpy.models.region import Region, BoundingBox
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.pipeline.executor import TranslationPipeline
from prismpy.translators.base import UnifiedData
from prismpy.validators.scientific import _check_cross_variable_consistency


def _region():
    return Region(name="t", country="t", country_iso3="TST",
                  bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1))


def _grid(n):
    return SpatialGrid(
        bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1), resolution="5arcmin",
        cells=[GridCell(cell_id=i, lat=0.5 + i * 0.01, lon=0.5 + i * 0.01,
                        row=0, col=i, resolution="5arcmin") for i in range(n)])


def _soil(pid):
    return SoilProfile(profile_id=pid, lat=0.5, lon=0.5, source="iSDA",
                       layers=[SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=40.0, clay=30.0)])


def _ts(cid, *, tmax, tmin, precip=2.0, srad=20.0):
    return ClimateTimeSeries(
        location_id=cid, lat=0.5, lon=0.5, source="TEST",
        records=[ClimateRecord(date=_date(2015, 1, 1), tmax=tmax, tmin=tmin,
                               precip=precip, srad=srad)])


def test_cross_variable_defect_is_per_cell_and_emits_violation_details():
    ud = UnifiedData(
        region=_region(),
        climate={0: _ts(0, tmax=10.0, tmin=20.0),    # tmax<tmin (impossible)
                 1: _ts(1, tmax=30.0, tmin=20.0)},   # consistent
        soil={})
    r = _check_cross_variable_consistency(ud)
    assert r["result"] == "fail"
    vds = r["details"]["violation_details"]
    assert vds, "a fail-tier impossibility must emit per-cell violation_details"
    assert all(vd["defect"] for vd in vds)
    assert {vd["cell_id"] for vd in vds} == {0}, "only the tmax<tmin cell is a defect"
    assert r["details"]["defect"] is True


def test_cross_variable_defect_cell_id_flows_through_the_pivot():
    # END-TO-END premise gate: the tmax<tmin cell_id (from climate.items()) MATCHES
    # cells_by_id in the executor pivot, so the per-cell defect lands on the cell_summary.
    ud = UnifiedData(
        region=_region(), grid=_grid(2),
        climate={0: _ts(0, tmax=10.0, tmin=20.0),    # impossible
                 1: _ts(1, tmax=30.0, tmin=20.0)},   # ok
        soil={0: _soil("p0"), 1: _soil("p1")})
    # The validation report the executor pivots — the REAL cross_variable check output.
    report = {"checks": [_check_cross_variable_consistency(ud)]}
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    cs = pipeline._build_cell_summary(ud, validation_report=report)
    by_id = {c["id"]: c for c in cs["cells"]}

    def _has_xvar_defect(cid):
        return any(
            fc.get("defect") and fc.get("check_id") == "cross_variable_consistency"
            for fc in by_id[cid].get("failed_checks", []))

    assert _has_xvar_defect(0), "the tmax<tmin cell must carry a per-cell cross_variable defect"
    assert not _has_xvar_defect(1), "the consistent cell must NOT carry a defect"


def test_post_translate_consistency_defect_count_matches_fail_count():
    # The aggregate post_translate consistency check (no per-cell id) carries
    # defect_count = the real N of impossible values, so a consumer never renders
    # "1 impossible value" for N>1. RED if defect_count is dropped (consumer defaults N=1).
    from prismpy.validators.post_translate import _build_climate_checks
    checks = _build_climate_checks(
        "pythia", {}, {"tmax_le_tmin": 3, "negative_rain": 2, "negative_srad": 0}, 100, 5)
    consistency = next(
        c for c in checks if c["check"] == "post_translate_consistency_pythia")
    assert consistency["result"] == "fail"
    assert consistency["details"]["defect"] is True
    assert consistency["details"]["defect_count"] == 5   # 3 + 2 + 0
