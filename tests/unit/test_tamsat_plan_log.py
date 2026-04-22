"""Issue 4 (warning-auditor MEDIUM) — the TAMSAT download path used
to log "N to fetch, M cached, total=T" where N was the dates-needing-
.tif count and M was the .tif-already-exists count. That conflated
two cache tiers and misled users about actual HTTP work: a run that
wiped the marker file but kept the .nc cache reported "1096 to fetch,
0 cached" even though only ~366 HTTP requests actually fired.

This test locks the new log labels in place. Structural: reads the
source of `tamsat.py` and asserts the three-way partition vocabulary
is present on one logger.info call. Mutation-proof: removing any of
the three phrases fails the matching assertion.
"""

import re
import unittest
from pathlib import Path


_TAMSAT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / 'src' / 'prismpy' / 'sources' / 'climate' / 'tamsat.py'
)


class TestTamsatPlanLogHonesty(unittest.TestCase):
    def setUp(self):
        self.source = _TAMSAT_PATH.read_text()

    def test_log_reports_http_count_separately_from_nc_cache(self):
        """The log must name the HTTP-work count and the .nc cache
        count as two distinct quantities. A reader comparing wall
        time to the log should see the work that actually happened,
        not just the work that was theoretically pending."""
        self.assertIn('need HTTP download', self.source)
        self.assertIn('already in .nc cache', self.source)

    def test_log_reports_final_tif_cache_tier(self):
        """The third partition (already-have-final-.tif) is a
        no-op path — no Phase 1 and no Phase 2. The log names it
        explicitly so a user who sees high 'total' with low 'HTTP'
        + low 'nc' immediately understands the work is a no-op."""
        self.assertIn('already have final .tif', self.source)

    def test_log_groups_three_labels_on_one_logger_call(self):
        """All three quantities (HTTP needed, nc cached, final .tif
        cached) must appear inside the SAME `self.logger.info(...)`
        call — otherwise they'd span multiple log records and a grep
        by timestamp would miss the partition picture. Regex matches
        a single logger.info argument list containing all three."""
        # Multi-line logger.info with the three labels anywhere
        # inside the parenthesized call.
        pattern = re.compile(
            r'self\.logger\.info\(\s*(?:f"[^"]*"\s*)*?'
            r'[^)]*need HTTP download'
            r'[^)]*already in .nc cache'
            r'[^)]*already have final \.tif'
            r'[^)]*\)',
            re.DOTALL,
        )
        self.assertIsNotNone(
            pattern.search(self.source),
            'three TAMSAT partition labels are not grouped on one logger.info call',
        )

    def test_old_misleading_label_is_gone(self):
        """Mutation-proof guard against a silent revert — the old
        "TAMSAT download: {X} to fetch, {Y} cached" template must
        no longer appear. The new template uses "TAMSAT plan:"."""
        self.assertNotIn('TAMSAT download: ', self.source)
        # The new header phrase is the replacement contract.
        self.assertIn('TAMSAT plan:', self.source)

    def test_http_needed_is_computed_not_equal_to_dates_to_download(self):
        """Mutation-proof — the HTTP count must be computed as
        `len(dates_to_download) - nc_cached`, not just
        `len(dates_to_download)` (the old bug). Locks in the
        subtraction."""
        self.assertRegex(
            self.source,
            r'http_needed\s*=\s*len\(dates_to_download\)\s*-\s*nc_cached',
        )


class TestTamsatRuntimeLogHonesty(unittest.TestCase):
    """Codex self-check MEDIUM — the plan-log fix at 32a9ada named
    HTTP vs cache partitions honestly at the START of the download,
    but the runtime progress callback and the Phase-1 + completion
    logs still said 'downloaded N files' even when many of those
    files were served from the .nc cache (no HTTP at all).

    This test class locks in the runtime labels: Phase 1 counters
    are split, the progress callback names HTTP fetches separately
    from cache serves, and the completion log reports the three
    partitions that match the plan log."""

    def setUp(self):
        self.source = _TAMSAT_PATH.read_text()

    def test_phase1_counters_are_split_into_http_and_cache(self):
        """Two separate local variables — `dl_http_fetched` and
        `dl_nc_cache_served` — drive the Phase 1 accounting. A single
        `dl_ok` that conflates both results is the exact bug codex
        flagged."""
        self.assertIn('dl_http_fetched', self.source)
        self.assertIn('dl_nc_cache_served', self.source)
        # And the old conflated counter must be gone.
        self.assertNotRegex(
            self.source,
            r'^\s*dl_ok\s*=\s*0\s*$',
        )

    def test_http_vs_cache_dispatch_on_download_result(self):
        """Mutation-proof — the dispatch that splits `"ok"` (HTTP)
        from `"cached"` (nc cache served) must route to the two
        separate counters. Combining them again (the original bug)
        would fail this test."""
        # `"ok"` → http_fetched
        self.assertRegex(
            self.source,
            r'result\s*==\s*"ok"\s*:\s*\n\s+dl_http_fetched\s*\+=\s*1',
        )
        # `"cached"` → nc_cache_served
        self.assertRegex(
            self.source,
            r'result\s*==\s*"cached"\s*:\s*\n\s+dl_nc_cache_served\s*\+=\s*1',
        )

    def test_phase1_completion_log_reports_three_partitions(self):
        """The Phase 1 completion log must name HTTP fetched, nc-cache
        served, and skipped as three distinct quantities. A grep by
        'Phase 1 complete' should reveal the partition immediately."""
        import re
        # f-string parts separate with `"` → allow `.*?` across the
        # boundary. DOTALL lets . match newlines between continued
        # f-string lines.
        pattern = re.compile(
            r'TAMSAT Phase 1 complete.*?'
            r'HTTP fetched.*?'
            r'nc-cache served.*?'
            r'skipped',
            re.DOTALL,
        )
        self.assertRegex(self.source, pattern)

    def test_progress_callback_names_http_and_cache_separately(self):
        """The in-flight progress callback message must NOT say
        'downloading N/T files' (the old conflating phrasing). It
        must show HTTP fetches and cache hits separately so the user
        watching the progress bar can distinguish network activity
        from cache serves."""
        # Old conflating phrasing must be gone from the callback text.
        self.assertNotIn('downloading {processed}', self.source)
        # And the new message must name the split.
        self.assertIn('HTTP fetched', self.source)
        self.assertIn('from cache', self.source)

    def test_final_completion_log_reports_all_three_cache_tiers(self):
        """The final 'TAMSAT download complete' log mirrors the plan
        log's three-tier partition: HTTP fetched, nc-cache served,
        .tif already cached. Without this, operators debugging a
        slow run can't reconstruct where the wall-time went."""
        import re
        pattern = re.compile(
            r'TAMSAT download complete.*?'
            r'HTTP fetched.*?'
            r'nc-cache served.*?'
            r'\.tif already cached',
            re.DOTALL,
        )
        self.assertRegex(self.source, pattern)
