"""F20 — temporal completeness denominator picked per-cell.

The validator's per-cell expectation matches each cell's actual
fetched range, derived from the records' ``min(date)``. Pre-F20
the validator hard-coded ``expected_days_with_spinup`` for every
per-cell-records cell, silently halving reported completeness on
PYTHIA + CRAFT projects with ``spinup_years > 0``. The first
attempted fix (``"acea" in enabled_platforms``) miscalculated
multi-platform default-target runs because each translator's
fetch is conditional, not unconditional:

* PYTHIA's start has a ``platform_config.pythia.climate_start_date``
  override path (``pythia/translator.py:1984-1987``).
* ACEA only fetches the spinup window when ``n_climate <
  n_cells`` (``acea/translator.py:404``); when CRAFT/PYTHIA have
  already surfaced no-spinup records, ACEA's gate trips false.

The data-driven discriminator below sidesteps every conditional
path: each cell's actual ``min(records.date)`` decides whether it
carries the spinup window. Cells starting before the no-spinup
date get the with-spinup denominator; the rest get the no-spinup
denominator. Multi-platform-safe by construction.

The pre-F13 placeholder sentinel ``{-1: ts(source="placeholder")}``
masked the original denominator gap because the placeholder
generated full-period fake records. Post-F13 the helper surfaces
the actual translator-fetched dates onto ``unified_data.climate``;
the validator's silently-wrong denominator then reported 50%
completeness on 2-year + spinup=2 PYTHIA / CRAFT projects (the
user-evidence fixture: ``f7706669`` Maize Tigania West +
``c3cad31b`` Maize Bamboutos). Aminata's all-cells-red symptom
was the writer-faithful surface of this denominator mismatch.

Persona-walk anchored in the assertions:

* **Aminata (DSSAT MISDAT)** — completeness must be honest about
  what each cell actually carries. PYTHIA cells with full-period
  records report 100% complete on the user's ``f7706669`` run,
  not the previously-reported 50%.
* **Moussa (stakeholder)** — the 50% completeness number leaked
  into stakeholder slides while the package was actually
  complete; honest framing matters for trust.
* **Dr. Kofi (audit)** — the per-cell expectation derivation
  lives in the validator's docstring; audit-grep can trace
  which discriminator applied to which cell from
  ``details.affected_cells`` plus the per-cell record range.
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
    (no spinup subtraction). Cells with records starting at
    ``start_year-01-01`` are picked up by the data-driven
    discriminator's no-spinup branch; otherwise a project with
    ``spinup>0`` silently reports halved completeness on data
    that is in fact fully fetched.

    Empirical anchor: the user's ``f7706669`` Maize Tigania West
    run (start=2022, end=2023, spinup=2). Translator fetched
    730 days/cell across 4 cells = 2920 actual; pre-F20
    validator expected 1461 days/cell (with spinup) × 4 = 5844;
    reported 50.0% completeness. Post-F20 validator picks no-
    spinup per-cell because each cell's ``min(date)`` is
    ``2022-01-01`` (not ``< expected_start_no_spinup``); reports
    100% completeness."""

    def test_pythia_full_fetch_reports_100pct(self):
        """2-year config + spinup=2; cells have records for the
        full no-spinup range (2022-01-01 .. 2023-12-31 = 730
        days). The data-driven discriminator picks the no-spinup
        denominator per-cell because every cell's ``min(date)``
        is exactly ``expected_start_no_spinup`` (NOT less than).
        Validator reports 100% completeness, not 50%."""
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
    1, 1)`` — but only when ``n_climate < n_cells`` is true at
    ``acea/translator.py:404``. Cells whose records actually
    carry the spinup window are picked up by the data-driven
    discriminator's with-spinup branch (``min(date) <
    expected_start_no_spinup``). Cells with preloaded no-spinup
    records (e.g., earlier translators surfaced into
    ``data.climate`` first) get the no-spinup expectation
    instead — honestly reporting density within whatever was
    actually loaded."""

    def test_acea_full_fetch_with_spinup_reports_100pct(self):
        """2-year config + spinup=2 + cells have records for the
        full WITH-spinup range (2020-01-01 .. 2023-12-31 = 1461
        days). The data-driven discriminator picks with-spinup
        per-cell because every cell's ``min(date)`` is
        ``2020-01-01`` (< ``expected_start_no_spinup`` of
        ``2022-01-01``). Validator reports 100% completeness."""
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


class TestMultiPlatformDataDrivenDiscriminator:
    """Q1 HIGH regression pins from F20 codex Gate B. Pre-Path-X
    the per-cell expectation was driven by ``"acea" in
    enabled_platforms``, which silently miscalculated the
    multi-platform default-target case: when ``targets`` defaults
    to all four platforms, CRAFT/PYTHIA run before ACEA and
    surface no-spinup records; ACEA's spinup-fetch gate
    (``acea/translator.py:404``: ``if n_climate < n_cells``)
    trips false because ``data.climate`` is already populated;
    yet ``"acea" in enabled`` still picked the with-spinup
    denominator → false 50% completeness.

    The data-driven per-cell discriminator inspects each cell's
    actual ``min(records.date)`` and matches the expectation to
    the data, not to the platform list — multi-platform-safe by
    construction."""

    def test_multi_platform_default_target_no_false_partial(self):
        """All four platforms enabled; CRAFT/PYTHIA fetched first
        and surfaced no-spinup records; ACEA inherits them via
        the conditional-fetch gate. Validator must report 100%
        complete (not pre-Path-X 50%) because each cell's actual
        range is densely covered by its own records."""
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
            enabled_platforms=["sarra_py", "craft", "pythia", "acea"],
        )
        # Pre-Path-X: 730×4 / 1461×4 = 50%. Post-Path-X: 100%.
        assert check["result"] == "pass"
        assert check["details"]["completeness_pct"] == 100.0
        assert "2920/2920" in check["summary"]
        # The pre-fix 5844 denominator must NOT appear; that was
        # the symptom of the platform-membership miscalculation.
        assert "5844" not in check["summary"]

    def test_per_cell_mixed_range_picks_expectation_per_cell(self):
        """Mixed per-cell ranges in the same ``unified_data.climate``:
        cell 0 carries the full with-spinup window; cell 1 carries
        only the no-spinup window. The data-driven discriminator
        applies the matching expectation per-cell — both report
        100% complete because each is densely covered for its
        OWN range."""
        # Cell 0: ACEA-style — 2020-01-01 to 2023-12-31 = 1461 days
        # Cell 1: PYTHIA-style — 2022-01-01 to 2023-12-31 = 730 days
        climate = {
            0: _make_ts_for_range(
                _date(2020, 1, 1), _date(2023, 12, 31), cell_id=0,
            ),
            1: _make_ts_for_range(
                _date(2022, 1, 1), _date(2023, 12, 31), cell_id=1,
            ),
        }
        config = _make_config(start_year=2022, end_year=2023, spinup_years=2)
        check = _check_temporal_completeness(
            _make_unified(climate=climate, n_cells=2),
            config,
            enabled_platforms=["acea", "pythia"],
        )
        # Cell 0: 1461 actual / 1461 expected (with-spinup)
        # Cell 1: 730 actual / 730 expected (no-spinup)
        # Total: 2191 / 2191 = 100%
        assert check["result"] == "pass"
        assert check["details"]["completeness_pct"] == 100.0
        assert "2191/2191" in check["summary"]
        # No cell appears in the gap accounting (everything is
        # 100% within its own range).
        assert check["details"]["cells_with_gaps"] == 0

    def test_modal_expected_days_per_cell_reports_dominant(self):
        """``details.expected_days_per_cell`` reports the modal
        expectation across cells. With 3 with-spinup cells + 1
        no-spinup cell, the modal is the with-spinup
        denominator. Single-platform runs (the common case)
        keep this exact; mixed runs surface the dominant one."""
        climate = {
            0: _make_ts_for_range(
                _date(2020, 1, 1), _date(2023, 12, 31), cell_id=0,
            ),
            1: _make_ts_for_range(
                _date(2020, 1, 1), _date(2023, 12, 31), cell_id=1,
            ),
            2: _make_ts_for_range(
                _date(2020, 1, 1), _date(2023, 12, 31), cell_id=2,
            ),
            3: _make_ts_for_range(
                _date(2022, 1, 1), _date(2023, 12, 31), cell_id=3,
            ),
        }
        config = _make_config(start_year=2022, end_year=2023, spinup_years=2)
        check = _check_temporal_completeness(
            _make_unified(climate=climate),
            config,
            enabled_platforms=["acea", "pythia"],
        )
        # 3 cells with 1461 expected + 1 cell with 730 expected
        # → modal is with-spinup (1461). Most-common; not mean.
        assert check["details"]["expected_days_per_cell"] == 1461


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
