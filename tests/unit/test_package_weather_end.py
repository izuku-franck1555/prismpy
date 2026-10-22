"""A PYTHIA package declares ``temporal.weather_end`` from its own .WTH files: prismpy's real writer
writes them here from constructed series, and any defect on a site's run leaves the end undeclared
(logged with the site and the defect), never cut back to the days before it."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pytest

from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.spatial import BoundingBox, GridCell, SpatialGrid
from prismpy.packaging.manifest import create_manifest
from prismpy.translators.pythia.translator import PythiaTranslator
from tests.package_soil import stamp_package_soil

LOGGER = "prismpy.packaging.weather_declaration"


def _series(cell, first, last, tdew=17.0):
    records = [ClimateRecord(date=first + timedelta(days=i), tmax=31.0 + i % 5, tmin=19.0 + i % 3,
                             precip=float(i % 4), srad=18.0 + i % 6, wind=2.0, rh=60.0, tdew=tdew)
               for i in range((last - first).days + 1)]
    return ClimateTimeSeries(location_id=cell, lat=11.0, lon=3.0, source="test", records=records,
                             elevation=250.0)


def _package(tmp_path, ends, first=date(2025, 1, 1), tdew=17.0):
    """The real writer's weather/: one site per entry of ``ends`` (its last day), and a
    header-only sentinel for each None (a cell without climate)."""
    cells = [GridCell(cell_id=100 + n, lat=11.0 + n / 10, lon=3.0, row=n, col=0)
             for n in range(len(ends))]
    writer = PythiaTranslator.__new__(PythiaTranslator)
    writer.output_dir, writer.provenance, writer.cockpit_override_sidecar = tmp_path, None, None
    (tmp_path / "weather").mkdir()
    writer._generate_weather_files(
        {cell.cell_id: _series(cell.cell_id, first, end, tdew)
         for cell, end in zip(cells, ends) if end is not None},
        grid=SpatialGrid(bounds=BoundingBox(minx=3.0, miny=11.0, maxx=3.1, maxy=12.0),
                         resolution=0.1, cells=cells))
    return tmp_path / "weather"


def _manifest(package_dir, end_year, platform="pythia"):
    stamp_package_soil(package_dir, platform)
    return create_manifest(package_dir, {
        "project_name": "weather-end", "region_name": "Niger", "country": "Nigeria",
        "crop_name": "maize", "start_year": min(2025, end_year), "end_year": end_year,
        "data_sources": {"climate": "NASA POWER"}}, platform=platform)


def _token(day):
    return f"{day.year}{day.timetuple().tm_yday:03d}"


def _rows(site, change):
    """Edit ``site``'s .WTH lines in place: ``change`` gets the lines and the index of a day's row."""
    def edit(weather):
        path = weather / f"{site}.WTH"
        lines = path.read_text().splitlines(keepends=True)
        at = {line.split()[0]: n for n, line in enumerate(lines) if line[:1].isdigit()}
        path.write_text("".join(change(lines, lambda day: at[_token(day)])))
    return edit


def _value(site, day, column, text):
    def change(lines, at):
        tokens = lines[at(day)].split()
        tokens[["DATE", "SRAD", "TMAX", "TMIN", "RAIN", "TDEW", "RHUM", "WIND"].index(column)] = text
        lines[at(day)] = " ".join(tokens) + "\n"
        return lines
    return _rows(site, change)


def _crlf(weather):
    for path in weather.glob("*.WTH"):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))


def _swap_rain_and_tdew(weather):   # a header bound by name reads RAIN wherever it sits
    for path in weather.glob("*.WTH"):
        def swap(line):
            tokens = line.split()
            k = 4 + (tokens[0] == "@")   # RAIN's position; the header's "@" is a token of its own
            return " ".join(tokens[:k] + [tokens[k + 1], tokens[k]] + tokens[k + 2:]) + "\n"
        path.write_text("".join(swap(line) if line[:1].isdigit() or "DATE" in line else line
                                for line in path.read_text().splitlines(keepends=True)))


def _yyddd(weather):
    for path in weather.glob("*.WTH"):
        path.write_text("".join(line[2:] if line[:1].isdigit() else line
                                for line in path.read_text().splitlines(keepends=True)))


MAY1, MAY2 = date(2026, 5, 1), date(2026, 5, 2)
PARTIAL = [date(2026, 10, 10), date(2026, 10, 12), None]
DECLARED = [
    ("partial year, two sites and a sentinel", PARTIAL, 2026, None, {}, "2026-10-10"),
    ("a complete year", [date(2026, 12, 31)] * 2, 2026, None, {}, "2026-12-31"),
    ("a cross-year season", [date(2026, 3, 31)], 2025, None, {}, "2026-03-31"),
    ("a leap year's day 366", [date(2024, 12, 31)], 2024, None, {"first": date(2024, 1, 1)},
     "2024-12-31"),
    ("CRLF line endings", PARTIAL, 2026, _crlf, {}, "2026-10-10"),
    ("RAIN and TDEW in each other's columns", PARTIAL, 2026, _swap_rain_and_tdew, {"tdew": None},
     "2026-10-10"),
    ("-99 in a column that does not count", PARTIAL, 2026, _value(1, MAY1, "TDEW", "-99.0"), {},
     "2026-10-10"),
    ("a gap and a -99 day in the years before the end year", PARTIAL, 2026,
     lambda w: (_value(1, date(2025, 3, 1), "RAIN", "-99.0")(w),
                _rows(2, lambda lines, at: lines[:at(date(2025, 6, 1))]
                      + lines[at(date(2025, 6, 1)) + 1:])(w)), {}, "2026-10-10"),
]


@pytest.mark.parametrize("ends,end_year,edit,kwargs,expected", [row[1:] for row in DECLARED],
                         ids=[row[0] for row in DECLARED])
def test_a_pythia_package_declares_the_earliest_last_recorded_day_of_its_sites(
        tmp_path, ends, end_year, edit, kwargs, expected):
    weather = _package(tmp_path, ends, **kwargs)
    if edit:
        edit(weather)
    assert _manifest(tmp_path, end_year)["temporal"].get("weather_end") == expected


OMITTED = [
    ("a site starting on 2 January", _rows(1, lambda lines, at: lines[:at(date(2026, 1, 1))]
                                           + lines[at(date(2026, 1, 1)) + 1:]),
     "1.WTH", "start on 2026-01-02"),
    ("a duplicate row", _rows(1, lambda lines, at: lines[:at(MAY1) + 1] + lines[at(MAY1):]),
     "1.WTH", "repeats"),
    ("an out-of-order row", _rows(1, lambda lines, at: lines[:at(MAY1) + 1]
                                  + [lines[at(date(2026, 3, 1))]] + lines[at(MAY1) + 1:]),
     "1.WTH", "2026-03-01 comes before 2026-05-01"),
    ("a missing day with later rows", _rows(1, lambda lines, at: lines[:at(MAY1)]
                                            + lines[at(MAY1) + 1:]), "1.WTH", "skips after"),
    *[(f"{text} {column} with later rows", _value(1, MAY1, column, text), "1.WTH", "not recorded")
      for column in ("SRAD", "TMAX", "TMIN", "RAIN") for text in ("-99.0", "nan")],
    ("a non-recorded last row", _value(1, date(2026, 10, 10), "RAIN", "-99.0"), "1.WTH",
     "not recorded"),
    ("an invalid leap-day date", _rows(2, lambda lines, at: lines[:at(MAY1)]
                                       + [lines[at(MAY1)].replace(_token(MAY1), "2026366", 1)]
                                       + lines[at(MAY1) + 1:]), "2.WTH", "names no day of 2026"),
    ("YYDDD dates", _yyddd, "1.WTH", "not 7 digits"),
    ("one bad site among good ones", _rows(2, lambda lines, at: lines[:at(MAY2)]
                                           + lines[at(MAY2) + 1:]), "2.WTH", "skips after"),
]


@pytest.mark.parametrize("edit,site,defect", [row[1:] for row in OMITTED],
                         ids=[row[0] for row in OMITTED])
def test_any_defect_on_a_sites_run_omits_the_weather_end_and_logs_the_site_and_the_defect(
        tmp_path, caplog, edit, site, defect):
    edit(_package(tmp_path, PARTIAL))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        manifest = _manifest(tmp_path, 2026)
    assert "weather_end" not in manifest["temporal"]
    assert any(site in record.getMessage() and defect in record.getMessage()
               for record in caplog.records if record.name == LOGGER), caplog.text


@pytest.mark.parametrize("ends,end_year,defect", [
    ([None, None], 2026, "no site in weather/ carries weather"),
    ([date(2025, 12, 31)], 2026, "no row records 1 January 2026"),
], ids=["every site a sentinel", "the rows end before the end year"])
def test_a_package_without_a_recorded_end_year_omits_the_weather_end(
        tmp_path, caplog, ends, end_year, defect):
    _package(tmp_path, ends)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        manifest = _manifest(tmp_path, end_year)
    assert "weather_end" not in manifest["temporal"]
    assert defect in caplog.text


def test_only_a_pythia_package_declares_a_weather_end(tmp_path):
    pythia, craft = tmp_path / "pythia", tmp_path / "craft"
    for package_dir in (pythia, craft):
        package_dir.mkdir()
        _package(package_dir, PARTIAL)
    assert _manifest(pythia, 2026)["temporal"].get("weather_end") == "2026-10-10"
    assert "weather_end" not in _manifest(craft, 2026, platform="craft")["temporal"]
