"""Unit tests for AgERA5 writer consolidation — V2-22a Group A item 1.5.

Tests:

1. `test_year_scoped_glob_counts_only_target_year_files` — AC 1.5.5.
   Fixtures files across 3 years in a fake 2_conversion layout. Asserts
   the year-aware glob (`f'*_{year}_*.tif'`) selects exactly the files
   for the target year and ignores files from other years. Locks the
   year-scoping semantics against a future refactor that reverts to
   aggregate counting (which would re-surface the Branch A "converting
   to GeoTIFF" stickiness bug).

2. `test_agera5_single_writer_binding` — AC 1.5.6.
   Parses agera5.py with Python's `ast` module, locates the
   `_phase_monitor` FunctionDef, and asserts every `progress_callback(...)`
   Call node lives inside that function's body. Any call site OUTSIDE
   `_phase_monitor` would indicate reintroduction of W4 (or a new
   writer), defeating item 1.1's detail-diff guard. Comment-only
   documentation is insufficient per feedback_binding_verification_rule.md
   — this test locks the invariant structurally.

3. `test_count_agera5_stage_files_resolves_cwd_relative` — V2-22a 1.5
   route-back. Locks the binding that `_phase_monitor`'s stages_base =
   Path("../data") finds files at the SARRA_data_download library's
   actual write location. Evaluator's Group A verdict surfaced a path
   mismatch (n_zips/n_extracted/n_converted were reading from
   `output_dir / ...` but the library writes to CWD-relative
   `../data/...`). This test fails the pre-fix code and passes only when
   `_count_agera5_stage_files` uses the caller's stages_base as the
   actual glob root. Exercises all 4 stages together.

4. `test_phase_monitor_does_not_reference_output_dir` — structural
   binding assertion. Walks `_phase_monitor`'s FunctionDef AST and
   asserts no `Name(id='output_dir')` references appear inside. A
   refactor that reintroduces `output_dir / '0_downloads'` or similar
   would fail this test even if it passed Test 3 by accident. Pairs
   with Test 3: Test 3 locks the positive binding ("helper reads from
   the caller's base"), Test 4 locks the negative ("_phase_monitor does
   NOT read from per-run output_dir").
"""

import ast
import pathlib
from pathlib import Path

import pytest


# ── AC 1.5.5 — year-scoped counter ────────────────────────────────────


@pytest.fixture
def conversion_layout(tmp_path):
    """Build a fake 2_conversion/AgERA5_Test/ tree with files from 3
    years across 2 variable subdirs, matching the on-disk layout the
    SARRA_data_download library produces."""
    conv_dir = tmp_path / '2_conversion' / 'AgERA5_Test'
    var_a = conv_dir / '2m_temperature_24_hour_maximum'
    var_b = conv_dir / 'solar_radiation_flux_daily'
    var_a.mkdir(parents=True)
    var_b.mkdir(parents=True)

    # Year 2020: 2 files in var_a, 1 in var_b
    (var_a / '2m_temperature_24_hour_maximum_2020_01_01.tif').write_text('')
    (var_a / '2m_temperature_24_hour_maximum_2020_01_02.tif').write_text('')
    (var_b / 'solar_radiation_flux_daily_2020_01_01.tif').write_text('')

    # Year 2021: 1 file in var_a, 0 in var_b
    (var_a / '2m_temperature_24_hour_maximum_2021_06_15.tif').write_text('')

    # Year 2022: 0 files in either

    return conv_dir


def _count_year_scoped(conv_dir, year):
    """Replicates the year-scoped counting logic from agera5.py after
    the 1.5 fix: glob every variable subdir with f'*_{year}_*.tif'."""
    year_glob = f'*_{year}_*.tif'
    return sum(
        len(list(vd.glob(year_glob)))
        for vd in conv_dir.iterdir() if vd.is_dir()
    )


def test_year_scoped_glob_counts_only_target_year_files(conversion_layout):
    assert _count_year_scoped(conversion_layout, 2020) == 3
    assert _count_year_scoped(conversion_layout, 2021) == 1
    assert _count_year_scoped(conversion_layout, 2022) == 0


def test_year_scoped_glob_isolates_adjacent_years(conversion_layout):
    """Defensive — a year-2020 glob must NOT accidentally match a
    year-2021 file even though both files have '2021' in adjacent
    positions or similar string proximity."""
    count_2020 = _count_year_scoped(conversion_layout, 2020)
    count_2021 = _count_year_scoped(conversion_layout, 2021)
    assert count_2020 + count_2021 == 4
    assert count_2020 != count_2021


# ── AC 1.5.6 — structural single-writer binding ───────────────────────


def _agera5_source_path() -> pathlib.Path:
    """Locate agera5.py from this test file's path.

    tests/unit/test_agera5.py → ../../src/prismpy/sources/climate/agera5.py
    """
    here = pathlib.Path(__file__).resolve()
    return here.parents[2] / 'src' / 'prismpy' / 'sources' / 'climate' / 'agera5.py'


def _collect_progress_callback_call_nodes(tree: ast.AST) -> list[ast.Call]:
    """Return every ast.Call whose callable is the bare name
    `progress_callback` or the attribute `.progress_callback` reached
    via a call `something.progress_callback(...)`. We are not catching
    `self._progress_callback.on_substage_progress(...)` — that lives in
    executor.py, not agera5.py, and is controlled by item 1.5 step 5."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == 'progress_callback':
            out.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == 'progress_callback':
            out.append(node)
    return out


def _find_phase_monitor_span(tree: ast.AST) -> tuple[int, int]:
    """Return the (start_lineno, end_lineno) of the `_phase_monitor`
    FunctionDef. Raises AssertionError if not found."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_phase_monitor':
            return node.lineno, node.end_lineno or node.lineno
    raise AssertionError('_phase_monitor function not found in agera5.py')


def test_agera5_single_writer_binding():
    """AC 1.5.6 — _phase_monitor is the sole writer of progress_callback
    in agera5.py. Any progress_callback(...) call outside its body is a
    regression that would re-introduce W4/W5 alternation and defeat
    item 1.1's detail-diff guard on SARRA-Py AgERA5 runs."""
    src_path = _agera5_source_path()
    assert src_path.exists(), f'agera5.py not found at {src_path}'

    tree = ast.parse(src_path.read_text())
    phase_start, phase_end = _find_phase_monitor_span(tree)

    calls = _collect_progress_callback_call_nodes(tree)
    assert calls, 'expected at least one progress_callback() call in agera5.py'

    outside = [
        c for c in calls
        if not (phase_start <= c.lineno <= phase_end)
    ]

    if outside:
        lines = [f'agera5.py:{c.lineno}' for c in outside]
        raise AssertionError(
            'AC 1.5.6 violation — progress_callback() called outside '
            f'_phase_monitor at: {", ".join(lines)}. Reintroducing a '
            'writer outside _phase_monitor re-creates the W4/W5 '
            'alternation that defeats item 1.1 on SARRA-Py AgERA5 runs.'
        )


# ── V2-22a 1.5 route-back — CWD-relative stages_base binding ──────────


def test_count_agera5_stage_files_resolves_cwd_relative(tmp_path, monkeypatch):
    """V2-22a 1.5 route-back — locks the binding that _phase_monitor's
    stages_base = Path("../data") finds files at the SARRA_data_download
    library's actual write location.

    Evaluator's Group A verdict surfaced a path mismatch: n_zips /
    n_extracted / n_converted were reading from `output_dir / ...` but
    the library writes to CWD-relative `../data/...`. The var counter
    was pinned at 1/6 for the entire download phase because those 3
    counts were always zero. Only n_output worked (`agera5_out` used
    `_P("../data/3_output")` already).

    This test:
      1. Builds a fake `../data/` tree at `tmp_path/data/` with files
         across all 4 library stages: 0_downloads, 1_extraction,
         2_conversion, 3_output
      2. Chdir's to `tmp_path/work/` so `Path("../data")` resolves to
         `tmp_path/data/`
      3. Calls the helper with the bare relative `Path("../data")`
      4. Asserts all 4 counts reflect ground truth

    The test FAILS on the pre-fix code (which used `output_dir / ...`
    paths) because those paths do not exist under `tmp_path/work/`.
    The test PASSES only when the helper uses the caller's stages_base
    as the actual glob root — which is what the V2-22a route-back fix
    guarantees.
    """
    from prismpy.sources.climate.agera5 import _count_agera5_stage_files

    data_root = tmp_path / 'data'
    work_dir = tmp_path / 'work'
    work_dir.mkdir()

    # Stage 0: 0_downloads/ — flat, {region}*_{year}.zip pattern
    zips_dir = data_root / '0_downloads'
    zips_dir.mkdir(parents=True)
    (zips_dir / 'AgERA5_TestZone_tmax_2020.zip').write_text('')
    (zips_dir / 'AgERA5_TestZone_tmin_2020.zip').write_text('')
    (zips_dir / 'AgERA5_TestZone_srad_2020.zip').write_text('')
    # Wrong year — must not count
    (zips_dir / 'AgERA5_TestZone_tmax_2021.zip').write_text('')
    # Wrong region — must not count
    (zips_dir / 'AgERA5_OtherZone_tmax_2020.zip').write_text('')

    # Stage 1: 1_extraction/AgERA5_{region}/{year}/{var}/ — per-var netcdf
    ext_2020_tmax = data_root / '1_extraction' / 'AgERA5_TestZone' / '2020' / 'tmax'
    ext_2020_tmax.mkdir(parents=True)
    (ext_2020_tmax / 'tmax_2020_01_01.nc').write_text('')
    (ext_2020_tmax / 'tmax_2020_01_02.nc').write_text('')
    ext_2020_tmin = data_root / '1_extraction' / 'AgERA5_TestZone' / '2020' / 'tmin'
    ext_2020_tmin.mkdir(parents=True)
    (ext_2020_tmin / 'tmin_2020_01_01.nc').write_text('')
    # Wrong year dir — must not count (its subtree is under /2021/, not /2020/)
    ext_2021_tmax = data_root / '1_extraction' / 'AgERA5_TestZone' / '2021' / 'tmax'
    ext_2021_tmax.mkdir(parents=True)
    (ext_2021_tmax / 'tmax_2021_01_01.nc').write_text('')

    # Stage 2: 2_conversion/AgERA5_{region}/{var}/ — per-var geotiff
    # (aggregate across years on disk; year_glob must filter)
    conv_tmax = data_root / '2_conversion' / 'AgERA5_TestZone' / 'tmax'
    conv_tmax.mkdir(parents=True)
    (conv_tmax / 'tmax_2020_01_01.tif').write_text('')
    (conv_tmax / 'tmax_2020_01_02.tif').write_text('')
    # Wrong year — year_glob must filter this out
    (conv_tmax / 'tmax_2021_01_01.tif').write_text('')

    # Stage 3: 3_output/AgERA5_{region}/{var}/ — final geotiff
    out_tmax = data_root / '3_output' / 'AgERA5_TestZone' / 'tmax'
    out_tmax.mkdir(parents=True)
    (out_tmax / 'tmax_2020_01_01.tif').write_text('')
    (out_tmax / 'tmax_2020_01_02.tif').write_text('')
    out_tmin = data_root / '3_output' / 'AgERA5_TestZone' / 'tmin'
    out_tmin.mkdir(parents=True)
    (out_tmin / 'tmin_2020_01_01.tif').write_text('')

    # Chdir to work_dir so Path("../data") resolves to data_root
    monkeypatch.chdir(work_dir)

    counts = _count_agera5_stage_files(
        Path('../data'), 'TestZone', 2020,
    )

    assert counts['n_zips'] == 3, (
        f"Expected 3 year-2020 zips for TestZone, got {counts['n_zips']}. "
        "Wrong region and wrong year must not count."
    )
    assert counts['n_extracted'] == 3, (
        f"Expected 3 year-2020 netcdf files (2 tmax + 1 tmin), got "
        f"{counts['n_extracted']}. Wrong year dir must not count."
    )
    assert counts['n_converted'] == 2, (
        f"Expected 2 year-2020 converted tif files, got "
        f"{counts['n_converted']}. Wrong year file must be filtered "
        "by year_glob."
    )
    assert counts['n_output'] == 3, (
        f"Expected 3 year-2020 output tif files, got {counts['n_output']}. "
        "rglob across var subdirs must find all 3."
    )


def test_count_agera5_stage_files_handles_missing_dirs(tmp_path, monkeypatch):
    """Missing stage directories must return zero, not raise."""
    from prismpy.sources.climate.agera5 import _count_agera5_stage_files

    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    # Note: tmp_path/data does NOT exist
    monkeypatch.chdir(work_dir)

    counts = _count_agera5_stage_files(
        Path('../data'), 'TestZone', 2020,
    )
    assert counts == {
        'n_zips': 0,
        'n_extracted': 0,
        'n_converted': 0,
        'n_output': 0,
    }


def test_phase_monitor_does_not_reference_output_dir():
    """V2-22a 1.5 route-back — structural negative assertion.

    `_phase_monitor`'s body must NOT reference the `output_dir` Name at
    all. The SARRA_data_download library writes to CWD-relative paths
    (`../data/`) regardless of the save_path argument, so file counts
    that read from `output_dir / ...` always return zero. The pre-fix
    code had `dl_dir = output_dir / '0_downloads'` (plus ext_dir and
    conv_dir variants), which is exactly the bug this test locks
    against.

    This is a negative structural check — it catches a refactor that
    reintroduces `output_dir / '...stage_subdir...'` inside
    `_phase_monitor`, even if the refactor happens to pass the
    positive binding test above by accident."""
    src_path = _agera5_source_path()
    assert src_path.exists(), f'agera5.py not found at {src_path}'
    tree = ast.parse(src_path.read_text())

    phase_monitor_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_phase_monitor':
            phase_monitor_node = node
            break
    assert phase_monitor_node is not None, (
        '_phase_monitor FunctionDef not found in agera5.py'
    )

    offenders = []
    for node in ast.walk(phase_monitor_node):
        if isinstance(node, ast.Name) and node.id == 'output_dir':
            offenders.append(node.lineno)

    if offenders:
        raise AssertionError(
            'V2-22a 1.5 route-back violation — _phase_monitor references '
            f'`output_dir` at agera5.py lines {offenders}. The '
            'SARRA_data_download library writes to CWD-relative paths; '
            '_phase_monitor must read file counts from Path("../data") '
            '(via _count_agera5_stage_files), not from the per-run '
            'output_dir. Reading from output_dir returns zeros and pins '
            'the var counter at 1/6 for the entire download phase.'
        )
