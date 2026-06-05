"""
Unit tests for prismpy utility functions.
"""

import pytest
from datetime import date

from prismpy.utils.date_utils import (
    doy_to_date,
    date_to_doy,
    yrdoy_to_date,
    date_to_yrdoy,
    mmdd_to_doy,
    doy_to_mmdd,
    parse_date,
    get_growing_season_dates,
    is_leap_year,
    days_in_year,
    date_range,
)
from prismpy.utils.sanitization import (
    sanitize_filename,
    sanitize_admin_name,
    sanitize_for_sql,
    remove_accents,
)
from prismpy.utils.gis_utils import (
    haversine_distance,
    bounds_contain_point,
    expand_bounds,
    snap_bounds_outward_to_grid,
    compute_cell_id_global,
    cell_id_to_rowcol,
    latlon_to_rowcol,
    rowcol_to_latlon,
)


# =============================================================================
# Date Utility Tests
# =============================================================================

class TestDoyToDate:
    """Tests for doy_to_date function."""

    def test_january_first(self):
        """Test DOY 1 is January 1."""
        result = doy_to_date(1, 2015)
        assert result == date(2015, 1, 1)

    def test_june_15(self):
        """Test DOY 166 is June 15 (non-leap year)."""
        result = doy_to_date(166, 2015)
        assert result == date(2015, 6, 15)

    def test_december_31_non_leap(self):
        """Test DOY 365 is December 31 (non-leap year)."""
        result = doy_to_date(365, 2015)
        assert result == date(2015, 12, 31)

    def test_december_31_leap(self):
        """Test DOY 366 is December 31 (leap year)."""
        result = doy_to_date(366, 2016)
        assert result == date(2016, 12, 31)

    def test_invalid_doy_zero(self):
        """Test DOY 0 raises error."""
        with pytest.raises(ValueError):
            doy_to_date(0, 2015)

    def test_invalid_doy_367(self):
        """Test DOY 367 raises error."""
        with pytest.raises(ValueError):
            doy_to_date(367, 2015)

    def test_doy_366_non_leap_year(self):
        """Test DOY 366 raises error for non-leap year."""
        with pytest.raises(ValueError):
            doy_to_date(366, 2015)


class TestDateToDoy:
    """Tests for date_to_doy function."""

    def test_january_first(self):
        """Test January 1 is DOY 1."""
        result = date_to_doy(date(2015, 1, 1))
        assert result == 1

    def test_june_15(self):
        """Test June 15 is DOY 166 (non-leap year)."""
        result = date_to_doy(date(2015, 6, 15))
        assert result == 166

    def test_december_31(self):
        """Test December 31 is DOY 365 (non-leap year)."""
        result = date_to_doy(date(2015, 12, 31))
        assert result == 365

    def test_leap_year_march(self):
        """Test March 1 is DOY 61 in leap year."""
        result = date_to_doy(date(2016, 3, 1))
        assert result == 61  # 31 (Jan) + 29 (Feb) + 1

    def test_non_leap_year_march(self):
        """Test March 1 is DOY 60 in non-leap year."""
        result = date_to_doy(date(2015, 3, 1))
        assert result == 60  # 31 (Jan) + 28 (Feb) + 1


class TestYrdoyToDate:
    """Tests for yrdoy_to_date function."""

    def test_basic_conversion(self):
        """Test basic YRDOY conversion."""
        result = yrdoy_to_date(2015001)
        assert result == date(2015, 1, 1)

    def test_mid_year(self):
        """Test mid-year conversion."""
        result = yrdoy_to_date(2015166)
        assert result == date(2015, 6, 15)

    def test_end_year(self):
        """Test end of year conversion."""
        result = yrdoy_to_date(2015365)
        assert result == date(2015, 12, 31)

    def test_invalid_format(self):
        """Test invalid format raises error."""
        with pytest.raises(ValueError):
            yrdoy_to_date(15001)  # Too short


class TestDateToYrdoy:
    """Tests for date_to_yrdoy function."""

    def test_basic_conversion(self):
        """Test basic date to YRDOY conversion."""
        result = date_to_yrdoy(date(2015, 1, 1))
        assert result == 2015001

    def test_mid_year(self):
        """Test mid-year conversion."""
        result = date_to_yrdoy(date(2015, 6, 15))
        assert result == 2015166

    def test_end_year(self):
        """Test end of year conversion."""
        result = date_to_yrdoy(date(2015, 12, 31))
        assert result == 2015365


class TestMmddConversions:
    """Tests for MMDD format conversions."""

    def test_mmdd_to_doy_june(self):
        """Test MMDD to DOY for June 15."""
        result = mmdd_to_doy("0615", 2015)
        assert result == 166

    def test_mmdd_to_doy_january(self):
        """Test MMDD to DOY for January 1."""
        result = mmdd_to_doy("0101", 2015)
        assert result == 1

    def test_doy_to_mmdd_june(self):
        """Test DOY to MMDD for June 15."""
        result = doy_to_mmdd(166, 2015)
        assert result == "0615"

    def test_invalid_mmdd(self):
        """Test invalid MMDD raises error."""
        with pytest.raises(ValueError):
            mmdd_to_doy("1301", 2015)  # Invalid month


class TestParseDate:
    """Tests for parse_date function."""

    def test_date_object(self):
        """Test parsing date object."""
        d = date(2015, 6, 15)
        result = parse_date(d)
        assert result == d

    def test_iso_string(self):
        """Test parsing ISO format string."""
        result = parse_date("2015-06-15")
        assert result == date(2015, 6, 15)

    def test_yaml_string(self):
        """Test parsing YAML-style string (single digits)."""
        result = parse_date("2016-5-1")
        assert result == date(2016, 5, 1)

    def test_yrdoy_int(self):
        """Test parsing YRDOY integer."""
        result = parse_date(2015166)
        assert result == date(2015, 6, 15)

    def test_doy_with_year(self):
        """Test parsing DOY with default year."""
        result = parse_date(166, default_year=2015)
        assert result == date(2015, 6, 15)

    def test_doy_without_year_raises(self):
        """Test DOY without year raises error."""
        with pytest.raises(ValueError):
            parse_date(166)  # No default_year


class TestGrowingSeasonDates:
    """Tests for get_growing_season_dates function."""

    def test_same_year_season(self):
        """Test growing season within same year."""
        planting, maturity = get_growing_season_dates(166, 285, 2015)
        assert planting == date(2015, 6, 15)
        assert maturity == date(2015, 10, 12)

    def test_cross_year_season(self):
        """Test growing season crossing year boundary."""
        planting, maturity = get_growing_season_dates(300, 60, 2015)
        assert planting == date(2015, 10, 27)
        # DOY 60 in 2016 (leap year) is Feb 29
        assert maturity == date(2016, 2, 29)


class TestLeapYear:
    """Tests for leap year functions."""

    def test_leap_year_2016(self):
        """Test 2016 is a leap year."""
        assert is_leap_year(2016) is True

    def test_not_leap_year_2015(self):
        """Test 2015 is not a leap year."""
        assert is_leap_year(2015) is False

    def test_century_not_leap(self):
        """Test 1900 is not a leap year."""
        assert is_leap_year(1900) is False

    def test_400_year_leap(self):
        """Test 2000 is a leap year."""
        assert is_leap_year(2000) is True

    def test_days_in_year(self):
        """Test days_in_year function."""
        assert days_in_year(2015) == 365
        assert days_in_year(2016) == 366


class TestDateRange:
    """Tests for date_range function."""

    def test_basic_range(self):
        """Test basic date range."""
        result = date_range(date(2015, 6, 1), date(2015, 6, 5))
        assert len(result) == 5
        assert result[0] == date(2015, 6, 1)
        assert result[-1] == date(2015, 6, 5)

    def test_single_day(self):
        """Test single day range."""
        result = date_range(date(2015, 6, 1), date(2015, 6, 1))
        assert len(result) == 1


# =============================================================================
# Sanitization Utility Tests
# =============================================================================

class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_basic_name(self):
        """Test basic filename is unchanged."""
        result = sanitize_filename("test_file")
        assert result == "test_file"

    def test_remove_slashes(self):
        """Test slashes are removed."""
        result = sanitize_filename("test/file")
        assert "/" not in result

    def test_remove_special_chars(self):
        """Test special characters are removed."""
        result = sanitize_filename("test<>:file")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result

    def test_spaces_replaced(self):
        """Test spaces are replaced with underscore."""
        result = sanitize_filename("test file name")
        assert " " not in result
        assert "_" in result


class TestSanitizeAdminName:
    """Tests for sanitize_admin_name function."""

    def test_basic_name(self):
        """Test basic name is unchanged."""
        result = sanitize_admin_name("Koutiala")
        assert result == "Koutiala"

    def test_accents_removed(self):
        """Test accents are removed."""
        result = sanitize_admin_name("Côte d'Ivoire")
        assert "ô" not in result
        assert "Cote" in result

    def test_special_chars(self):
        """Test special characters handled."""
        result = sanitize_admin_name("San/Tomé")
        assert "/" not in result or result.replace("/", "_") == result


class TestRemoveAccents:
    """Tests for remove_accents function."""

    def test_french_accents(self):
        """Test French accents are removed."""
        result = remove_accents("Côte d'Ivoire")
        assert result == "Cote d'Ivoire"

    def test_spanish_accents(self):
        """Test Spanish accents are removed."""
        result = remove_accents("España")
        assert result == "Espana"

    def test_no_accents(self):
        """Test string without accents is unchanged."""
        result = remove_accents("Mali")
        assert result == "Mali"


class TestSanitizeForSql:
    """Tests for sanitize_for_sql function."""

    def test_basic_identifier(self):
        """Test basic identifier is unchanged."""
        result = sanitize_for_sql("cell_id")
        assert result == "cell_id"

    def test_remove_special_chars(self):
        """Test special characters are removed."""
        result = sanitize_for_sql("cell-id; DROP TABLE")
        assert ";" not in result
        assert "-" not in result


# =============================================================================
# GIS Utility Tests
# =============================================================================

class TestHaversineDistance:
    """Tests for haversine_distance function."""

    def test_same_point(self):
        """Test distance from point to itself is 0."""
        result = haversine_distance(12.0, -5.5, 12.0, -5.5)
        assert result == 0.0

    def test_known_distance(self):
        """Test known distance between two points."""
        # Approximate distance: 1 degree latitude ≈ 111 km
        result = haversine_distance(12.0, -5.5, 13.0, -5.5)
        assert 110 < result < 112

    def test_equator(self):
        """Test distance calculation at equator."""
        result = haversine_distance(0, 0, 0, 1)
        assert 110 < result < 112


class TestBoundsContainPoint:
    """Tests for bounds_contain_point function."""

    def test_point_inside(self):
        """Test point inside bounds."""
        bounds = (-6.0, 11.5, -5.0, 12.5)
        assert bounds_contain_point(bounds, -5.5, 12.0) is True

    def test_point_outside(self):
        """Test point outside bounds."""
        bounds = (-6.0, 11.5, -5.0, 12.5)
        assert bounds_contain_point(bounds, -7.0, 12.0) is False

    def test_point_on_edge(self):
        """Test point on edge of bounds."""
        bounds = (-6.0, 11.5, -5.0, 12.5)
        assert bounds_contain_point(bounds, -6.0, 12.0) is True


class TestExpandBounds:
    """Tests for expand_bounds function."""

    def test_basic_buffer(self):
        """Test basic buffer expansion."""
        bounds = (-6.0, 11.5, -5.0, 12.5)
        result = expand_bounds(bounds, 0.1)
        assert result[0] < bounds[0]  # minx decreased
        assert result[1] < bounds[1]  # miny decreased
        assert result[2] > bounds[2]  # maxx increased
        assert result[3] > bounds[3]  # maxy increased

    def test_zero_buffer(self):
        """Test zero buffer returns same bounds."""
        bounds = (-6.0, 11.5, -5.0, 12.5)
        result = expand_bounds(bounds, 0.0)
        assert result == bounds


class TestSnapBoundsOutwardToGrid:
    """Climate-fetch widening: snap a raw bbox outward to enclosing native
    pixel edges so the AgMIP outward-snapped cell roster is fully covered.

    The load-bearing case is the real Mopti AOI ``[-4.3, 14.4, -4.15, 14.55]``
    (manual_bounds for the UC1/UC4/UC6 sarra packages), whose 9-cell 5-arcmin
    roster admits perimeter cell centers up to half a sim-cell OUTSIDE the raw
    bbox. The unbuffered fetch inward-snapped to a 2x1 AgERA5 raster and left
    those cells without climate; the widened fetch must reach a 3x3 AgERA5
    grid that encloses all 9 centers.
    """

    # The 9 reproduced 5-arcmin cell centers (lon, lat) for the Mopti AOI —
    # byte-matching cell_summary.json (cell_ids 3911708..3920350). The widen
    # changes ONLY the fetch extent; this roster is unchanged.
    _MOPTI_RAW = (-4.3, 14.4, -4.15, 14.55)
    _MOPTI_CENTERS = [
        (-4.29167, 14.54167), (-4.20833, 14.54167), (-4.125, 14.54167),
        (-4.29167, 14.45833), (-4.20833, 14.45833), (-4.125, 14.45833),
        (-4.29167, 14.375),   (-4.20833, 14.375),   (-4.125, 14.375),
    ]

    def test_agera5_mopti_snaps_to_3x3_enclosing_target(self):
        """AgERA5 (0.1 deg) widen reproduces the 3x3-producing fetch bbox and
        encloses every one of the 9 admitted cell centers."""
        snapped = snap_bounds_outward_to_grid(self._MOPTI_RAW, resolution=0.1)
        minx, miny, maxx, maxy = snapped
        # The geometric target: outward to 0.1 deg pixel edges + tolerance.
        assert snapped == pytest.approx(
            (-4.350001, 14.349999, -4.049999, 14.650001), abs=1e-6
        )
        # 3 native pixels wide and tall (0.3 deg span + 2x tolerance).
        assert (maxx - minx) == pytest.approx(0.3, abs=1e-3)
        assert (maxy - miny) == pytest.approx(0.3, abs=1e-3)
        # Every admitted cell center lies strictly inside the fetch extent.
        for lon, lat in self._MOPTI_CENTERS:
            assert minx < lon < maxx and miny < lat < maxy, (
                f"cell center ({lon}, {lat}) not enclosed by AgERA5 widen"
            )

    def test_tamsat_mopti_extends_south_past_data_bottom(self):
        """TAMSAT (0.0375 deg) widen — lockstep with AgERA5 — must extend the
        crop south below the southmost cell center 14.375 (and below the live
        TAMSAT data bottom 14.4188 that the unbuffered crop stopped at), else
        the southern row keeps dropping and coverage caps at 4/9."""
        snapped = snap_bounds_outward_to_grid(self._MOPTI_RAW, resolution=0.0375)
        minx, miny, maxx, maxy = snapped
        assert miny < 14.375, "TAMSAT crop must reach the southmost cell center"
        assert miny < 14.4188, "TAMSAT crop must extend past the live data bottom"
        for lon, lat in self._MOPTI_CENTERS:
            assert minx < lon < maxx and miny < lat < maxy

    def test_widen_is_outward_only(self):
        """Snapped bounds always enclose the raw bounds (never clip inward) —
        for an arbitrary bbox at both native resolutions."""
        raw = (-6.0, 11.5, -5.0, 12.5)
        for res in (0.1, 0.0375):
            minx, miny, maxx, maxy = snap_bounds_outward_to_grid(raw, resolution=res)
            assert minx <= raw[0] and miny <= raw[1]
            assert maxx >= raw[2] and maxy >= raw[3]

    def test_roster_grid_math_is_independent_of_widen(self):
        """The fetch widen takes a raw-bounds tuple and returns a new tuple;
        it cannot mutate the caller's bounds, so the cell roster derived from
        ``region.bounds`` elsewhere is structurally unaffected."""
        raw = (-4.3, 14.4, -4.15, 14.55)
        snap_bounds_outward_to_grid(raw, resolution=0.1)
        assert raw == (-4.3, 14.4, -4.15, 14.55)


class TestCellIdGlobal:
    """Tests for global grid cell ID functions."""

    def test_compute_cell_id(self):
        """Test cell ID computation."""
        # Test a known point - convert lat/lon to row/col first
        resolution = 5 / 60  # 5 arcmin
        row, col = latlon_to_rowcol(12.0, -5.5, resolution)
        cell_id = compute_cell_id_global(row, col)
        assert isinstance(cell_id, int)
        assert 0 <= cell_id < 4320 * 2160

    def test_roundtrip(self):
        """Test cell ID to lat/lon and back."""
        original_lat, original_lon = 12.0, -5.5
        resolution = 5 / 60  # 5 arcmin
        row, col = latlon_to_rowcol(original_lat, original_lon, resolution)
        cell_id = compute_cell_id_global(row, col)
        recovered_row, recovered_col = cell_id_to_rowcol(cell_id)
        recovered_lat, recovered_lon = rowcol_to_latlon(recovered_row, recovered_col, resolution)

        # Should be within 5 arcmin (0.0833 degrees)
        assert abs(recovered_lat - original_lat) < 0.1
        assert abs(recovered_lon - original_lon) < 0.1

    def test_cell_id_range(self):
        """Test cell IDs are within valid range."""
        # Test various points
        test_points = [
            (0, 0),      # Equator
            (45, 90),    # Northern hemisphere
            (-30, -60),  # Southern hemisphere
        ]
        resolution = 5 / 60  # 5 arcmin

        for lat, lon in test_points:
            row, col = latlon_to_rowcol(lat, lon, resolution)
            cell_id = compute_cell_id_global(row, col)
            assert 0 <= cell_id < 4320 * 2160, f"Invalid cell_id for ({lat}, {lon})"
