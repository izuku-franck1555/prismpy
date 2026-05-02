"""Sprint D.1 AC-2 — PYTHIA writes -99.0 MISDAT for missing rain.

Behavioral pin for the PYTHIA .WTH writer's missing-value
handling. The fixture builds a 365-day climate time series with
3 missing-rain dates, runs the writer against a tempdir, and
asserts the .WTH file emits the DSSAT MISDAT sentinel at the
missing positions instead of a phantom 0.0.

Pre-Sprint-D.1 the writer defaulted missing rain to 0.0 and
then re-clamped any -99 sentinel back to 0.0 — the silent zero
fill the F-X audit caught. The fix preserves -99.0 through the
writer so DSSAT and downstream tools see the missing-value
contract DSSAT documents (Jones 2003).
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from prismpy.config.schema import (
    BoundaryConfig,
    BoundarySource,
    CropCalendarConfig,
    CropConfig,
    ManualBoundsConfig,
    OutputConfig,
    Platform,
    ProjectConfig,
    ProjectInfo,
    RegionConfig,
    TemporalConfig,
)
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries


def _build_pythia_config(output_dir: str) -> ProjectConfig:
    """Minimal ProjectConfig wiring PYTHIA as the target. The
    ``output_dir`` is overridden at translator instantiation so
    the .WTH files land in a tempdir, not the test cwd."""
    return ProjectConfig(
        project=ProjectInfo(
            name="test_misdat",
            description="AC-2 missing-rain MISDAT pin",
        ),
        region=RegionConfig(
            name="Koutiala",
            country="Mali",
            country_iso3="MLI",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(
                    minx=-5.5,
                    miny=12.0,
                    maxx=-5.0,
                    maxy=12.5,
                ),
            ),
        ),
        crop=CropConfig(
            name="Maize",
            name_short="mai",
            variety="Medium-duration",
            calendar=CropCalendarConfig(
                planting_doy=166,
                maturity_doy=285,
            ),
        ),
        temporal=TemporalConfig(
            start_year=2015,
            end_year=2015,
            spinup_years=0,
        ),
        targets=[Platform.PYTHIA],
        output=OutputConfig(base_dir=output_dir, structure="by_platform"),
    )


def _build_synthetic_year(missing_rain_doys: list[int]) -> ClimateTimeSeries:
    """365-day Koutiala-like synthetic year with ``precip=None``
    on the listed day-of-year values and a deterministic shape
    everywhere else. The day-of-year list is 1-indexed (1 ==
    Jan 1, 365 == Dec 31)."""
    records = []
    start = date(2015, 1, 1)
    for i in range(365):
        d = start + timedelta(days=i)
        doy = d.timetuple().tm_yday
        is_missing = doy in missing_rain_doys
        records.append(
            ClimateRecord(
                date=d,
                tmax=32.0 + (i % 5),
                tmin=20.0 + (i % 3),
                precip=None if is_missing else 1.0 + (i % 4),  # type: ignore[arg-type]
                srad=20.0 + (i % 4),
                wind=2.0,
                rh=65.0,
                tdew=18.0,
            )
        )
    return ClimateTimeSeries(
        location_id=1,
        lat=12.4,
        lon=-5.4,
        source="TEST_SYNTH",
        records=records,
        elevation=300.0,
    )


def test_pythia_writes_minus_99_for_missing_rain(tmp_path):
    """A 365-day fixture with 3 missing rain dates produces a
    PYTHIA .WTH file whose RAIN column has exactly 3 ``-99.0``
    values + zero phantom 0.0 entries at those positions."""
    from prismpy.translators.pythia.translator import PythiaTranslator

    config = _build_pythia_config(str(tmp_path))
    translator = PythiaTranslator(config=config, output_dir=str(tmp_path))
    (translator.output_dir / "weather").mkdir(parents=True, exist_ok=True)

    missing_doys = [50, 150, 250]
    ts = _build_synthetic_year(missing_doys)
    output_files = translator._generate_weather_files({0: ts})

    assert len(output_files) == 1
    wth_path = output_files[0]
    content = wth_path.read_text()

    # The .WTH header has 4 leading lines + the data rows. Each
    # data row is "YRDOY SRAD TMAX TMIN RAIN TDEW RHUM WIND".
    # Find the RAIN column by index = 4 (0-indexed) and count
    # entries equal to -99.0.
    minus_99_count = 0
    zero_at_missing_count = 0
    data_rows = [
        line for line in content.splitlines()
        if line.strip() and not line.startswith(("$", "@", " ")) and line[0].isdigit()
    ]
    for line in data_rows:
        cols = line.split()
        if len(cols) < 5:
            continue
        rain_str = cols[4]
        if rain_str == "-99.0":
            minus_99_count += 1
    assert minus_99_count == 3, (
        f"Expected exactly 3 RAIN == -99.0 entries (one per "
        f"missing-rain date), got {minus_99_count}. Sprint D.1 "
        f"AC-2 contract: missing rain must propagate as the "
        f"DSSAT MISDAT sentinel through to the .WTH output."
    )


def test_pythia_does_not_emit_zero_at_missing_dates(tmp_path):
    """Direct counter-pin: the same 3 missing-rain dates must not
    emit RAIN == 0.0 at their positions. Pre-Sprint-D.1 the
    silent-zero behavior would emit 0.0 instead of -99.0 here."""
    from prismpy.translators.pythia.translator import PythiaTranslator

    config = _build_pythia_config(str(tmp_path))
    translator = PythiaTranslator(config=config, output_dir=str(tmp_path))
    (translator.output_dir / "weather").mkdir(parents=True, exist_ok=True)

    missing_doys = [50, 150, 250]
    ts = _build_synthetic_year(missing_doys)

    # Compute the YRDOY for each missing date so the pin can
    # locate the right rows.
    expected_yrdoys = []
    start = date(2015, 1, 1)
    for doy in missing_doys:
        d = start + timedelta(days=doy - 1)
        # YRDOY is 5-digit YYDOY (per DSSAT) or 7-digit YYYYDOY
        # depending on the writer; accept either form by checking
        # both.
        expected_yrdoys.append(int(f"{d.year}{doy:03d}"))
        expected_yrdoys.append(int(f"{d.year % 100:02d}{doy:03d}"))

    output_files = translator._generate_weather_files({0: ts})
    content = output_files[0].read_text()

    for line in content.splitlines():
        if not line.strip() or not line[0].isdigit():
            continue
        cols = line.split()
        if len(cols) < 5:
            continue
        try:
            yrdoy = int(cols[0])
            rain = float(cols[4])
        except ValueError:
            continue
        if yrdoy in expected_yrdoys:
            assert rain == -99.0, (
                f"YRDOY {yrdoy} (a missing-rain date) emitted "
                f"RAIN={rain} instead of -99.0. Silent zero-fill "
                f"regression."
            )


def test_pythia_full_year_present_rain_unchanged(tmp_path):
    """Regression pin: a year with NO missing rain emits zero
    -99.0 entries — the AC-2 fix only affects the missing path."""
    from prismpy.translators.pythia.translator import PythiaTranslator

    config = _build_pythia_config(str(tmp_path))
    translator = PythiaTranslator(config=config, output_dir=str(tmp_path))
    (translator.output_dir / "weather").mkdir(parents=True, exist_ok=True)

    ts = _build_synthetic_year(missing_rain_doys=[])  # all present
    output_files = translator._generate_weather_files({0: ts})
    content = output_files[0].read_text()

    minus_99_in_rain_col = 0
    for line in content.splitlines():
        if not line.strip() or not line[0].isdigit():
            continue
        cols = line.split()
        if len(cols) < 5:
            continue
        if cols[4] == "-99.0":
            minus_99_in_rain_col += 1
    assert minus_99_in_rain_col == 0, (
        f"All-present synthetic year should emit zero RAIN == "
        f"-99.0 entries; got {minus_99_in_rain_col}. The fix "
        f"affects the missing-value path only."
    )


def test_pythia_negative_rain_clamps_to_zero_not_misdat(tmp_path):
    """The clamp logic preserves the missing-value sentinel and
    only clamps negative real-rain values to 0. A ``precip=-2.5``
    record (real value, data error) must clamp to 0; a
    ``precip=None`` record stays at -99.0."""
    from prismpy.translators.pythia.translator import PythiaTranslator

    config = _build_pythia_config(str(tmp_path))
    translator = PythiaTranslator(config=config, output_dir=str(tmp_path))
    (translator.output_dir / "weather").mkdir(parents=True, exist_ok=True)

    records = [
        # day 1: present negative real rain (data error)
        ClimateRecord(
            date=date(2015, 1, 1), tmax=30.0, tmin=20.0, precip=-2.5,
            srad=20.0, wind=2.0, rh=65.0, tdew=18.0,
        ),
        # day 2: missing
        ClimateRecord(
            date=date(2015, 1, 2), tmax=30.0, tmin=20.0,
            precip=None, srad=20.0, wind=2.0, rh=65.0, tdew=18.0,  # type: ignore[arg-type]
        ),
        # day 3: present positive
        ClimateRecord(
            date=date(2015, 1, 3), tmax=30.0, tmin=20.0, precip=5.0,
            srad=20.0, wind=2.0, rh=65.0, tdew=18.0,
        ),
    ]
    ts = ClimateTimeSeries(
        location_id=1, lat=12.4, lon=-5.4, source="TEST_SYNTH",
        records=records, elevation=300.0,
    )

    output_files = translator._generate_weather_files({0: ts})
    content = output_files[0].read_text()

    rain_values = []
    for line in content.splitlines():
        if not line.strip() or not line[0].isdigit():
            continue
        cols = line.split()
        if len(cols) < 5:
            continue
        rain_values.append(cols[4])

    # Day 1 (negative real rain) clamps to 0.0; Day 2 (missing)
    # stays -99.0; Day 3 (5.0) prints as 5.0.
    assert rain_values == ["0.0", "-99.0", "5.0"], (
        f"Expected [0.0 (clamp negative), -99.0 (missing), 5.0 "
        f"(present)], got {rain_values}"
    )
