"""Conservative NASA POWER data-availability heuristic.

The single source of "has NASA POWER published this date yet?" for the climate
fetch — the clamp that bounds each request and the coverage check that gates
success both read from here, so they cannot drift.

Kept dependency-light (only the standard ``datetime`` module) so a downstream
web view can import the latest-available date without pulling numpy / pandas /
requests through the full climate-source module.

NASA POWER daily data lags real time: solar radiation (the slowest-published
required parameter) runs roughly 5-7 days behind, and the rest a couple of
days. The default lag adds a small margin for processing variance and the
UTC-vs-local-clock skew. The estimate is deliberately conservative — claiming
a date is published a little later than it truly is only makes a very recent
season unavailable for a few extra days; it never reports missing data as
present. The post-fetch coverage check is the real source of truth, so an
imperfect lag can never produce a false success.
"""
from datetime import datetime, timedelta, timezone

# Days behind "today" that data is conservatively assumed published.
DEFAULT_LAG_DAYS = 10
# Floor so a caller cannot configure an over-claiming (too-small) lag that
# would request solar radiation NASA POWER has not published yet.
MIN_LAG_DAYS = 8


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
