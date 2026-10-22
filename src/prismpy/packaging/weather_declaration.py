"""The last day of weather a PYTHIA package records, declared from its own ``.WTH`` files.

A day is RECORDED at a site when SRAD, TMAX, TMIN and RAIN are each finite and not DSSAT's
missing value (-99). From 1 January of the package's end year, each site's rows must run one
recorded day after the other, in order, up to its last row; the package's weather end is the
earliest of those last days. A file with a column header and no data row is a site without
weather and is skipped. Any defect on a site's run leaves the whole package's end undeclared
(and logged): the declaration is never cut back to the days before a defect.
"""
from __future__ import annotations

import calendar
import logging
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional, Union

logger = logging.getLogger(__name__)

WEATHER_DIR = "weather"
RECORDED_COLUMNS = ("SRAD", "TMAX", "TMIN", "RAIN")
DSSAT_MISSING = -99


class WeatherDefect(ValueError):
    """A site's rows that cannot declare a weather end."""


def _day(token: str) -> date:
    if len(token) != 7 or not token.isdigit():
        raise WeatherDefect(f"the date {token!r} is not 7 digits (YYYYDDD)")
    year, doy = int(token[:4]), int(token[4:])
    if year < 1 or not 1 <= doy <= 365 + calendar.isleap(year):
        raise WeatherDefect(f"the date {token!r} names no day of {year}")
    return date(year, 1, 1) + timedelta(days=doy - 1)


def _recorded(values: Iterable[str]) -> bool:
    try:
        numbers = [float(value) for value in values]
    except ValueError:
        return False
    return all(math.isfinite(n) and round(n) != DSSAT_MISSING for n in numbers)


def site_weather_end(path: Path, end_year: int) -> Optional[date]:
    """The last day of the site's clean run from 1 January of ``end_year``; None for a site
    without weather (a header and no data row). Raises :class:`WeatherDefect` when the site's
    rows cannot declare an end."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise WeatherDefect(f"the file cannot be read ({type(exc).__name__})") from None
    names, rows = None, []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if line.lstrip().startswith("@"):
            if names is not None:
                raise WeatherDefect(f"line {number} is a second column header")
            if "DATE" in line.lstrip()[1:].split():
                names = line.lstrip()[1:].split()
            continue
        if names is not None:
            rows.append((number, line.split()))
    if names is None:
        raise WeatherDefect("the file has no DATE column header")
    if any(names.count(name) != 1 for name in ("DATE", *RECORDED_COLUMNS)):
        raise WeatherDefect(f"the header {' '.join(names)!r} does not name DATE, "
                            f"{', '.join(RECORDED_COLUMNS)} once each")
    at, values = names.index("DATE"), [names.index(name) for name in RECORDED_COLUMNS]
    last = None
    for number, tokens in rows:
        if len(tokens) != len(names):
            raise WeatherDefect(f"line {number} has {len(tokens)} fields; "
                                f"the header names {len(names)}")
        day = _day(tokens[at])
        if last is None and day.year < end_year:
            continue   # the years before the end year are not counted
        if last is None and day != date(end_year, 1, 1):
            raise WeatherDefect(f"line {number}: the rows from {end_year} start on {day}, "
                                f"not 1 January")
        if last is not None and day != last + timedelta(days=1):
            what = "repeats" if day == last else "comes before" if day < last else "skips after"
            raise WeatherDefect(f"line {number}: {day} {what} {last}")
        if not _recorded(tokens[index] for index in values):
            raise WeatherDefect(f"line {number}: {day} is not recorded")
        last = day
    if rows and last is None:
        raise WeatherDefect(f"no row records 1 January {end_year}")
    return last


def declared_weather_end(package_dir: Union[str, Path], end_year: object) -> Optional[date]:
    """The package's ``temporal.weather_end``: the earliest last recorded day over its sites
    with weather. None, logged with the site and the defect, when any site's rows cannot
    declare it, or when no site carries weather."""
    if type(end_year) is not int:
        logger.warning("weather_end not declared: the end year %r is not a whole year", end_year)
        return None
    ends = []
    for path in sorted(Path(package_dir, WEATHER_DIR).glob("*.WTH")):
        try:
            end = site_weather_end(path, end_year)
        except WeatherDefect as defect:
            logger.warning("weather_end not declared: %s: %s", path.name, defect)
            return None
        if end is not None:
            ends.append(end)
    if not ends:
        logger.warning("weather_end not declared: no site in %s/ carries weather", WEATHER_DIR)
        return None
    return min(ends)
