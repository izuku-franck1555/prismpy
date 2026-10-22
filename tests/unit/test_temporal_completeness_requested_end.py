"""Temporal completeness expects a PYTHIA-only package's days through the end its own download
requested (``platform_config.pythia.climate_end_date``); any other configuration keeps the
calendar end. It still counts dates, never values: a present date with a missing value is the
declared weather end's matter, not this check's."""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from prismpy.config.schema import (
    BoundaryConfig,
    BoundarySource,
    CropCalendarConfig,
    CropConfig,
    ManualBoundsConfig,
    Platform,
    PlatformConfigGroup,
    ProjectConfig,
    ProjectInfo,
    PythiaConfig,
    RegionConfig,
    TemporalConfig,
)
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.validators.scientific import _check_temporal_completeness

MISSED = date(2020, 3, 1)


def _config(end):
    return ProjectConfig(
        project=ProjectInfo(name="requested-end", description="a season in progress"),
        region=RegionConfig(
            name="Niger", country="Nigeria", country_iso3="NGA",
            boundary=BoundaryConfig(source=BoundarySource.MANUAL, manual_bounds=ManualBoundsConfig(
                minx=3.0, miny=9.0, maxx=4.0, maxy=10.0))),
        crop=CropConfig(name="Maize", name_short="mai", variety="Medium-duration",
                        calendar=CropCalendarConfig(planting_doy=166, maturity_doy=285)),
        temporal=TemporalConfig(start_year=2016, end_year=2026, spinup_years=0),
        targets=[Platform.PYTHIA, Platform.CRAFT],
        platform_config=PlatformConfigGroup(pythia=PythiaConfig(climate_end_date=end)),
    )


def _series(cell, skip=(), unvalued=()):
    first, last = date(2016, 1, 1), date(2026, 10, 10)   # 3,936 days a cell
    days = (first + timedelta(days=i) for i in range((last - first).days + 1))
    return ClimateTimeSeries(
        location_id=cell, lat=10.0, lon=3.5, source="nasa_power",
        records=[ClimateRecord(date=day, tmax=30.0, tmin=20.0, srad=20.0,
                               precip=None if day in unvalued else 2.0)  # type: ignore[arg-type]
                 for day in days if day not in skip])


@pytest.mark.parametrize("end,platforms,series,result,summary", [
    ("2026-10-10", ["pythia"], {}, "pass", "7872/7872 days present (100.0%)"),
    ("2026-10-10", ["pythia"], {"skip": {MISSED}}, "warning", "7871/7872 days present"),
    ("2026-10-10", ["pythia"], {"unvalued": {MISSED}}, "pass", "7872/7872 days present (100.0%)"),
    ("2026-10-10", ["pythia", "craft"], {}, "fail", "7872/8036 days present (98.0%)"),
    (None, ["pythia"], {}, "fail", "7872/8036 days present (98.0%)"),
], ids=["PYTHIA alone, complete through its requested end",
        "PYTHIA alone, a date missing before its requested end",
        "PYTHIA alone, a date present with no rain value",
        "PYTHIA with CRAFT keeps the calendar end",
        "PYTHIA alone with no requested end keeps the calendar end"])
def test_completeness_counts_dates_through_a_pythia_only_packages_requested_end(
        end, platforms, series, result, summary):
    check = _check_temporal_completeness(
        SimpleNamespace(climate={1: _series(1, **series), 2: _series(2)}), _config(end),
        enabled_platforms=platforms)
    assert (check["result"], summary in check["summary"]) == (result, True), check["summary"]
