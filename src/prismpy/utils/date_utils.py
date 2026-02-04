"""
Date handling utilities for prismpy.

These utilities handle various date formats used by different platforms:
- Day of Year (DOY): 1-366
- Julian Date (YRDOY): e.g., 2015001 for Jan 1, 2015
- ISO format: YYYY-MM-DD
- MMDD format: e.g., 0615 for June 15
"""

from datetime import date, datetime, timedelta
from typing import Optional, Tuple, Union
import re


def doy_to_date(doy: int, year: int) -> date:
    """Convert day of year to date.

    Args:
        doy: Day of year (1-366)
        year: Year

    Returns:
        Date object

    Raises:
        ValueError: If DOY is out of range for the given year
    """
    if doy < 1 or doy > 366:
        raise ValueError(f"DOY must be 1-366, got {doy}")

    # Check if valid for non-leap years
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    max_doy = 366 if is_leap else 365

    if doy > max_doy:
        raise ValueError(f"DOY {doy} is invalid for year {year} (max {max_doy})")

    return date(year, 1, 1) + timedelta(days=doy - 1)


def date_to_doy(d: date) -> int:
    """Convert date to day of year.

    Args:
        d: Date object

    Returns:
        Day of year (1-366)
    """
    return d.timetuple().tm_yday


def yrdoy_to_date(yrdoy: int) -> date:
    """Convert DSSAT YRDOY format to date.

    YRDOY format: YYYYDDD where YYYY is year and DDD is day of year.
    Example: 2015001 = January 1, 2015

    Args:
        yrdoy: Date in YRDOY format

    Returns:
        Date object
    """
    yrdoy_str = str(yrdoy)
    if len(yrdoy_str) != 7:
        raise ValueError(f"YRDOY must be 7 digits (YYYYDDD), got {yrdoy}")

    year = int(yrdoy_str[:4])
    doy = int(yrdoy_str[4:])

    return doy_to_date(doy, year)


def date_to_yrdoy(d: date) -> int:
    """Convert date to DSSAT YRDOY format.

    Args:
        d: Date object

    Returns:
        Date in YRDOY format (e.g., 2015001)
    """
    doy = date_to_doy(d)
    return d.year * 1000 + doy


def mmdd_to_doy(mmdd: str, year: int) -> int:
    """Convert MMDD format to day of year.

    MMDD format is used by CRAFT for planting dates.
    Example: "0615" = June 15

    Args:
        mmdd: Date in MMDD format
        year: Year for calculating DOY (needed for leap years)

    Returns:
        Day of year (1-366)
    """
    if len(mmdd) != 4:
        raise ValueError(f"MMDD must be 4 characters, got '{mmdd}'")

    month = int(mmdd[:2])
    day = int(mmdd[2:])

    try:
        d = date(year, month, day)
        return date_to_doy(d)
    except ValueError as e:
        raise ValueError(f"Invalid MMDD '{mmdd}': {e}")


def doy_to_mmdd(doy: int, year: int) -> str:
    """Convert day of year to MMDD format.

    Args:
        doy: Day of year (1-366)
        year: Year

    Returns:
        Date in MMDD format (e.g., "0615")
    """
    d = doy_to_date(doy, year)
    return f"{d.month:02d}{d.day:02d}"


def parse_date(
    value: Union[str, date, datetime, int],
    default_year: Optional[int] = None,
) -> date:
    """Parse various date formats to a date object.

    Supports:
    - date object: returned as-is
    - datetime object: converted to date
    - ISO string: "YYYY-MM-DD"
    - YAML string: "YYYY-M-D" (e.g., "2016-5-1")
    - YRDOY int: 2015001
    - DOY int with year: requires default_year

    Args:
        value: Date in various formats
        default_year: Default year for DOY-only values

    Returns:
        Date object
    """
    if isinstance(value, date):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, int):
        # Check if it's a YRDOY or just a DOY
        if value > 1000:
            return yrdoy_to_date(value)
        elif default_year:
            return doy_to_date(value, default_year)
        else:
            raise ValueError(f"DOY {value} requires default_year")

    if isinstance(value, str):
        # Try ISO format first
        iso_match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", value)
        if iso_match:
            year = int(iso_match.group(1))
            month = int(iso_match.group(2))
            day = int(iso_match.group(3))
            return date(year, month, day)

        # Try MMDD format
        if len(value) == 4 and value.isdigit():
            if default_year:
                doy = mmdd_to_doy(value, default_year)
                return doy_to_date(doy, default_year)
            else:
                raise ValueError(f"MMDD format '{value}' requires default_year")

        raise ValueError(f"Unable to parse date string: '{value}'")

    raise ValueError(f"Unsupported date type: {type(value)}")


def get_growing_season_dates(
    planting_doy: int,
    maturity_doy: int,
    year: int,
) -> Tuple[date, date]:
    """Get planting and maturity dates for a growing season.

    Handles seasons that span year boundaries.

    Args:
        planting_doy: Planting day of year
        maturity_doy: Maturity day of year
        year: Planting year

    Returns:
        Tuple of (planting_date, maturity_date)
    """
    planting_date = doy_to_date(planting_doy, year)

    if maturity_doy >= planting_doy:
        # Same year
        maturity_date = doy_to_date(maturity_doy, year)
    else:
        # Maturity is in the following year
        maturity_date = doy_to_date(maturity_doy, year + 1)

    return planting_date, maturity_date


def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year.

    Args:
        year: Year to check

    Returns:
        True if leap year
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_in_year(year: int) -> int:
    """Get number of days in a year.

    Args:
        year: Year

    Returns:
        365 or 366
    """
    return 366 if is_leap_year(year) else 365


def date_range(start: date, end: date) -> list[date]:
    """Generate list of dates in a range (inclusive).

    Args:
        start: Start date
        end: End date

    Returns:
        List of dates from start to end (inclusive)
    """
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates
