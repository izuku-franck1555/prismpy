"""Conservative NASA POWER data-availability heuristic.

The single source of "has NASA POWER published this date yet?" for the climate
fetch — the clamp that bounds each request and the coverage check that gates
success both read from here, so they cannot drift.

Kept dependency-light (only the standard ``datetime`` module) so a downstream
web view can import the latest-available date without pulling numpy / pandas /
requests through the full climate-source module.

NASA POWER does not publish its recent days in order. Solar radiation, the
slowest required parameter, trails real time, and a whole span can stay missing
behind its published tail: on 2026-10-22, 2026-10-12..14 had no solar radiation
at every point checked (Abuja, Niger State, Mopti) while later days did. The
fetch fills a short gap of solar radiation or temperature (at most
MAX_INTERP_GAP_DAYS days, never rain) only between two real days, so such a
hole fails the coverage check only while a request ENDS inside it. The 14 days
below were chosen 2026-10-22 against that hole: a default request ends before
it through 2026-10-25, ends inside it from 2026-10-26 to 2026-10-28 (refused
unless NASA fills it), and brackets it from 2026-10-29 (filled, with
provenance); re-probe before 2026-10-26. The follow-up is a cached daily probe
that sets the latest available date to the latest day on which every required
parameter is present and before which every gap is fillable by that rule. The
estimate is deliberately conservative — claiming a date is published later than
it truly is only makes a very recent season unavailable for a few extra days.
The post-fetch coverage check is the real source of truth, so an imperfect lag
can never produce a false success; it counts filled days as covered, so it
proves coverage after the gap fill, not raw NASA completeness.
"""
from datetime import datetime, timedelta, timezone

# Days behind "today" that data is conservatively assumed published.
DEFAULT_LAG_DAYS = 14
# Floor so a caller cannot configure an over-claiming (too-small) lag that
# would request solar radiation NASA POWER has not published yet.
MIN_LAG_DAYS = 14


def nasa_power_latest_available_date(lag_days=DEFAULT_LAG_DAYS, today=None):
    """Latest date NASA POWER daily data is conservatively assumed published.

    Returns ``today (UTC) - lag_days``. ``lag_days`` is floored at
    ``MIN_LAG_DAYS`` so a configured value can only ever make the estimate
    more conservative, never less. ``today`` (a ``date``) is injectable for
    deterministic tests; otherwise the current UTC date is used. No network
    probe — safe to call offline.
    """
    if lag_days is None:
        lag_days = DEFAULT_LAG_DAYS
    lag_days = max(MIN_LAG_DAYS, int(lag_days))
    if today is None:
        today = datetime.now(timezone.utc).date()
    return today - timedelta(days=lag_days)
