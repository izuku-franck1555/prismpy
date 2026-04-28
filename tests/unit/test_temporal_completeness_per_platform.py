"""F20 — temporal completeness denominator per active translator.

The validator's per-cell expectation must match each translator's
actual fetch range. Pre-F20 the per-cell-records branch hard-coded
``expected_days_with_spinup``, which is correct only for ACEA's
fetch contract (``acea/translator.py:410-413``); PYTHIA's
(``pythia/translator.py:1987``) and CRAFT's
(``craft/translator.py:1395``) fetch from ``start_year-01-01``
without subtracting spinup, same precedent as SARRA-Py file-based
which already used ``expected_days_no_spinup``.

The pre-F13 placeholder sentinel ``{-1: ts(source="placeholder")}``
masked the gap because the placeholder generated full-period fake
records. Post-F13 the helper surfaces the actual translator-fetched
dates onto ``unified_data.climate``; the validator's
silently-wrong denominator then reported 50% completeness on
2-year + spinup=2 PYTHIA / CRAFT projects (the user-evidence
fixture). Aminata's symptom on the all-cells-red Maize Tigania
West run was the writer-faithful surface of this denominator
mismatch.

Persona-walk anchored in the assertions:

* **Aminata (DSSAT MISDAT)** — completeness must be honest about
  what the translator fetched. PYTHIA cells with full-period
  records report 100% complete on the user's ``f7706669`` run,
  not the previously-reported 50%.
* **Moussa (stakeholder)** — the 50% completeness number leaked
  into stakeholder slides while the package was actually
  complete; honest framing matters for trust.
* **Dr. Kofi (audit)** — the per-platform expectation lives in
  the validator's docstring + ``enabled_platforms`` parameter;
  audit-grep can trace which expectation applied per project
  type.
* **Ibrahim (mobile)** — Region Health card binds against the
  honest completeness metric.
"""
from __future__ import annotations

from datetime import date as _date
from types import SimpleNamespace

from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.region import BoundingBox, Region
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.translators.base import UnifiedData
from prismpy.validators.scientific import _check_temporal_completeness


# ---------------------------------------------------------------------------
# Fixture helpers tuned for spinup-axis testing (the existing
# test_validator_short_circuit fixtures use start=end=2020 + spinup=0,
# so they don't exercise the asymmetric-expectation surface).
# ---------------------------------------------------------------------------


def _make_region() -> Region:
    return Region(
        name="t", country="t", country_iso3="TST",
        bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
    )


def _make_grid(n_cells: int = 4) -> SpatialGrid:
    cells = [
        GridCell(
            cell_id=i, lat=0.5, lon=0.5,
            row=0, col=i, resolution="5arcmin",
        )
        for i in range(n_cells)
    ]
    return SpatialGrid(
        bounds=BoundingBox(minx=0, miny=0, maxx=1, maxy=1),
        resolution="5arcmin", cells=cells,
    )


def _make_ts_for_range(
    start: _date, end_inclusive: _date, *, cell_id: int = 0,
) -> ClimateTimeSeries:
    """Build a ClimateTimeSeries with one record per day across
    the inclusive range. Used to simulate "translator fetched the
    full requested period" without the leap-year arithmetic
    pitfalls of the existing ``_make_ts(n_records=N)`` helper
    (which only works for N <= 31)."""
    records = []
    cur = start
    while cur <= end_inclusive:
        records.append(ClimateRecord(
            date=cur, tmax=30.0, tmin=20.0, precip=2.0, srad=20.0,
        ))
        cur = _date.fromordinal(cur.toordinal() + 1)
    return ClimateTimeSeries(
        records=records, location_id=str(cell_id),
        lat=0.5, lon=0.5, source="TEST",
    )


def _make_unified(*, climate=None, n_cells=4) -> UnifiedData:
    return UnifiedData(
        region=_make_region(),
        grid=_make_grid(n_cells),
        climate=climate if climate is not None else {},
        soil={},
    )


class _FakeTemporal:
    """Stand-in for ProjectConfig.temporal that lets each test
    pick its own start/end/spinup without going through the full
    Pydantic ProjectConfig validator chain."""
    def __init__(self, *, start_year: int, end_year: int, spinup_years: int):
        self.start_year = start_year
        self.end_year = end_year
        self.spinup_years = spinup_years
        self._end = _date(end_year, 12, 31)

    def get_climate_end_date(self, crop_cal):
        return self._end


def _make_config(*, start_year: int, end_year: int, spinup_years: int):
    return SimpleNamespace(
        temporal=_FakeTemporal(
            start_year=start_year,
            end_year=end_year,
            spinup_years=spinup_years,
        ),
        crop=SimpleNamespace(calendar=None),
    )


# ---------------------------------------------------------------------------
# F20 — per-platform asymmetric expectation
# ---------------------------------------------------------------------------


class TestPythiaCraftNoSpinupExpectation:
    """PYTHIA + CRAFT translators fetch from ``start_year-01-01``
    (no spinup subtraction). The validator must use the matching
    no-spinup expectation; otherwise a project with ``spinup>0``
    silently reports halved completeness on data that is in fact
    fully fetched.

    Empirical anchor: the user's ``f7706669`` Maize Tigania West
    run (start=2022, end=2023, spinup=2). Translator fetched
    730 days/cell across 4 cells = 2920 actual; pre-F20
    validator expected 1461 days/cell (with spinup) × 4 = 5844;
    reported 50.0% completeness. Post-F20 validator expects 730
    × 4 = 2920; reports 100% completeness."""

    def test_pythia_full_fetch_reports_100pct(self):
        """2-year config + spinup=2; cells have records for the
        full no-spinup range (2022-01-01 .. 2023-12-31 = 730
        days). With ``enabled_platforms=['pythia']`` the
        validator must use ``expected_days_no_spinup`` and
        report 100% completeness, not 50%."""
        start = _date(2022, 1, 1)
        end_inc = _date(2023, 12, 31)
        climate = {
            i: _make_ts_for_range(start, end_inc, cell_id=i)
            for i in range(4)
        }
        config = _make_config(start_year=2022, end_year=2023, spinup_years=2)
        check = _check_temporal_completeness(
            _make_unified(climate=climate),
            config,
            enabled_platforms=["pythia"],
        )
        assert check["result"] == "pass"
        assert check["details"]["completeness_pct"] == 100.0
        # 730 days × 4 cells = 2920 (no spinup expectation)
        assert "2920/2920" in check["summary"]
        # The pre-F20 5844-denominator must NOT appear — that's
        # the user's all-cells-red 50% symptom.
        assert "5844" not in check["summary"]

    def test_craft_full_fetch_reports_100pct(self):
        """1-year config + spinup=2; cells have records for the
        full no-spinup range (2022-01-01 .. 2022-12-31 = 365
        days). User's ``c3cad31b`` CRAFT Maize Bamboutos run
        was the empirical anchor for this fixture shape."""
        start = _date(2022, 1, 1)
        end_inc = _date(2022, 12, 31)
        climate = {
            i: _make_ts_for_range(start, end_inc, cell_id=i)
            for i in range(16)
        }
        config = _make_config(start_year=2022, end_year=2022, spinup_years=2)
        check = _check_temporal_completeness(
            _make_unified(climate=climate, n_cells=16),
            config,
            enabled_platforms=["craft"],
        )
        assert check["result"] == "pass"
        assert check["details"]["completeness_pct"] == 100.0
        # 365 days × 16 cells = 5840
        assert "5840/5840" in check["summary"]

    def test_back_compat_default_no_platform_uses_no_spinup(self):
        """``enabled_platforms=None`` (back-compat default) maps
        to the no-spinup expectation — matching the
        empirically-most-common PYTHIA + CRAFT path. The
        existing ``test_validator_short_circuit`` tests rely on
        this default to keep passing without an opt-in
        parameter sweep."""
        start = _date(2022, 1, 1)
        end_inc = _date(2022, 12, 31)
        climate = {
            i: _make_ts_for_range(start, end_inc, cell_id=i)
            for i in range(2)
        }
        config = _make_config(start_year=2022, end_year=2022, spinup_years=2)
        check = _check_temporal_completeness(
            _make_unified(climate=climate, n_cells=2),
            config,
        )
        assert check["result"] == "pass"
        assert check["details"]["completeness_pct"] == 100.0


class TestAceaWithSpinupExpectation:
    """ACEA's translator fetches from ``date(start_year-spinup,
    1, 1)``; the validator's expectation must match. A blanket
    ``expected_days_no_spinup`` switch (the rejected one-line fix
    that triggered F20's SCOPE CONCERN) would make ACEA cells
    over-claim completeness because actual_days > expected_days
    when the warmup period is fetched.

    The asymmetric expectation preserves ACEA's correctness
    without changing translator behavior. The domain question of
    whether ACEA *should* pre-fetch the warmup window is a
    crop-modeling-specialist call deferred to a follow-on
    sprint."""

    def test_acea_full_fetch_with_spinup_reports_100pct(self):
        """2-year config + spinup=2 + cells have records for the
        full WITH-spinup range (2020-01-01 .. 2023-12-31 = 1461
        days). With ``enabled_platforms=['acea']`` validator must
        use ``expected_days_with_spinup`` and report 100%."""
        start = _date(2020, 1, 1)  # 2022 - spinup=2
        end_inc = _date(2023, 12, 31)
        climate = {
            i: _make_ts_for_range(start, end_inc, cell_id=i)
            for i in range(4)
        }
        config = _make_config(start_year=2022, end_year=2023, spinup_years=2)
        check = _check_temporal_completeness(
            _make_unified(climate=climate),
            config,
            enabled_platforms=["acea"],
        )
        assert check["result"] == "pass"
        assert check["details"]["completeness_pct"] == 100.0
        # 1461 days × 4 cells = 5844 (with spinup expectation)
        # 2020 leap + 2021 + 2022 + 2023 = 366 + 365 + 365 + 365 = 1461
        assert "5844/5844" in check["summary"]

    def test_acea_no_spinup_data_reports_partial(self):
        """If ACEA's translator fetched ONLY the no-spinup range
        (730 days) when the validator expected the full
        with-spinup range (1461), the gap shows up as 50%
        completeness — exactly inverting the user's
        pre-F20 symptom on PYTHIA + CRAFT. This pins that ACEA
        retains the with-spinup expectation regardless of the
        no-spinup default."""
        start = _date(2022, 1, 1)  # only the requested period
        end_inc = _date(2023, 12, 31)
        climate = {
            i: _make_ts_for_range(start, end_inc, cell_id=i)
            for i in range(4)
        }
        config = _make_config(start_year=2022, end_year=2023, spinup_years=2)
        check = _check_temporal_completeness(
            _make_unified(climate=climate),
            config,
            enabled_platforms=["acea"],
        )
        # 730 / 1461 ≈ 49.97% → fail
        assert check["result"] == "fail"
        assert check["details"]["completeness_pct"] < 50.5


class TestExpectedDaysPerCellMetadataMatchesContract:
    """The emit dict's ``details.expected_days_per_cell`` field
    must reflect the active expectation, not silently report the
    with-spinup value while the math used the no-spinup denominator
    (which would split chip text from drawer math the way F16's
    chip↔drawer hierarchy counter-criterion warned about)."""

    def test_pythia_emit_expected_days_per_cell_matches_no_spinup(self):
        start = _date(2022, 1, 1)
        end_inc = _date(2022, 12, 31)
        climate = {
            i: _make_ts_for_range(start, end_inc, cell_id=i)
            for i in range(2)
        }
        config = _make_config(start_year=2022, end_year=2022, spinup_years=2)
        check = _check_temporal_completeness(
            _make_unified(climate=climate, n_cells=2),
            config,
            enabled_platforms=["pythia"],
        )
        assert check["details"]["expected_days_per_cell"] == 365

    def test_acea_emit_expected_days_per_cell_matches_with_spinup(self):
        start = _date(2020, 1, 1)
        end_inc = _date(2023, 12, 31)
        climate = {
            i: _make_ts_for_range(start, end_inc, cell_id=i)
            for i in range(2)
        }
        config = _make_config(start_year=2022, end_year=2023, spinup_years=2)
        check = _check_temporal_completeness(
            _make_unified(climate=climate, n_cells=2),
            config,
            enabled_platforms=["acea"],
        )
        # 4 calendar years 2020-2023 inclusive = 1461 days
        assert check["details"]["expected_days_per_cell"] == 1461
