"""Potato on PYTHIA/DSSAT: the SUBSTOR crop profile across the prep package.

DSSAT simulates potato with its dedicated SUBSTOR-Potato module (``PTSUB``), not CERES or
CROPGRO. These pins drive the REAL translator outputs (``pythia_config`` ``default_setup``, the
SNX template rendered through Jinja the way the prism-runner renders it, the cultivar raster, the
manifest) and the prep-side crop x platform admission gate. The runner-rendered SNX and a real
SUBSTOR run are verified runner-side.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import jinja2
import pytest

from prismpy.config.schema import (
    AceaConfig,
    BoundaryConfig,
    BoundarySource,
    CropCalendarConfig,
    CropConfig,
    ManagementConfig,
    ManualBoundsConfig,
    OutputConfig,
    Platform,
    PlatformConfigGroup,
    ProjectConfig,
    ProjectInfo,
    PythiaConfig,
    RegionConfig,
    TemporalConfig,
)
from prismpy.models.climate import ClimateRecord, ClimateTimeSeries
from prismpy.models.crop import CropCalendar
from prismpy.models.region import BoundingBox, Region
from prismpy.translators.base import UnifiedData
from prismpy.translators.pythia.translator import PythiaTranslator
from tests.unit.test_pythia_canonical_substrate_flag import _build_grid_2x3, _build_profiles

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "dssat_genotype"
_PTSUB048_CUL = _FIXTURES / "PTSUB048.CUL"
# DSSAT-CSM 4.8.2.0 release Data/Genotype/PTSUB048.CUL (git blob fef9a232), the runner image tag.
_PTSUB048_SHA256 = "9329951991549ba2dcd61e8e6fe370ea8343654c4c3be4908197e69a2aa17491"

# The runner converts these ISO default_setup dates to DSSAT YYDDD before rendering.
_RUNNER_DATE_FIELDS = ("sdate", "fdate", "pfrst", "plast", "pdate", "hdate", "hlast")


def _cfg(out: Path, *, crop: str = "Potato", short: str = "pot", management=None,
         targets=(Platform.PYTHIA,), planting_doy: int = 166, maturity_doy: int = 285,
         start_year: int = 2015, end_year: int | None = None, pythia: PythiaConfig | None = None,
         acea_enabled: bool = True) -> ProjectConfig:
    return ProjectConfig(
        project=ProjectInfo(name="potato_substor", description="potato SUBSTOR prep package"),
        region=RegionConfig(
            name="Nyandarua", country="Kenya", country_iso3="KEN",
            boundary=BoundaryConfig(
                source=BoundarySource.MANUAL,
                manual_bounds=ManualBoundsConfig(minx=36.2, miny=-0.6, maxx=36.8, maxy=0.0),
            ),
        ),
        crop=CropConfig(
            name=crop, name_short=short,
            calendar=CropCalendarConfig(planting_doy=planting_doy, maturity_doy=maturity_doy),
        ),
        temporal=TemporalConfig(start_year=start_year,
                                end_year=start_year if end_year is None else end_year,
                                spinup_years=0),
        management=management,
        targets=list(targets),
        platform_config=PlatformConfigGroup(pythia=pythia or PythiaConfig(),
                                            acea=AceaConfig(enabled=acea_enabled)),
        output=OutputConfig(base_dir=str(out), structure="by_platform"),
    )


def _translator(out: Path, **kw) -> PythiaTranslator:
    t = PythiaTranslator(config=_cfg(out, **kw), output_dir=str(out))
    (out / "config").mkdir(parents=True, exist_ok=True)
    return t


def _data() -> UnifiedData:
    return UnifiedData(
        region=Region(name="Nyandarua", country="Kenya", country_iso3="KEN",
                      bounds=BoundingBox(minx=36.2, miny=-0.6, maxx=36.8, maxy=0.0)),
        grid=_build_grid_2x3(),
        soil=_build_profiles(),
    )


def _pythia_json(out: Path, **kw) -> dict:
    t = _translator(out, **kw)
    return json.loads(Path(t._generate_pythia_json(_data())).read_text())


def _emitted_template(t: PythiaTranslator) -> str:
    return Path(t._generate_snx_template(_data())).read_text()


def _runner_render(template: str, context: dict) -> str:
    """Render like the prism-runner's PYTHIA path: jinja2 with trim/lstrip blocks; an ISO date
    field becomes DSSAT YYDDD (``%y%j``); every other value passes through, as the runner passes
    the keys it does not register."""
    def runner_value(key, value):
        if key in _RUNNER_DATE_FIELDS and isinstance(value, str) and "::" not in value:
            return datetime.strptime(value, "%Y-%m-%d").strftime("%y%j")
        return value

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template).render({k: runner_value(k, v) for k, v in context.items()})


def _rendered_package_snx(out: Path, **kw) -> str:
    """The package's emitted SNX template rendered with its emitted first-run context — the
    runner's ``{**default_setup, **run}``."""
    t = _translator(out, **kw)
    pythia = json.loads(Path(t._generate_pythia_json(_data())).read_text())
    return _runner_render(_emitted_template(t), {**pythia["default_setup"], **pythia["runs"][0]})


def _header_and_row(snx: str, header_prefix: str) -> tuple:
    lines = snx.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(header_prefix):
            return line, lines[i + 1]
    raise AssertionError(f"{header_prefix!r} header not found in the SNX")


def _row_after(snx: str, header_prefix: str) -> str:
    return _header_and_row(snx, header_prefix)[1]


def _named_columns(snx: str, header_prefix: str) -> dict:
    """Map a whitespace-delimited SNX header's field names to the data row's values."""
    header, row = _header_and_row(snx, header_prefix)
    return dict(zip(header.split()[1:], row.split()[1:]))


def _dssat_planting_row(row: str) -> dict:
    """Read an ``@P`` data row with DSSAT's own fixed-width layout (IPPLNT_Inp FORMAT
    ``(I3,I5,1X,I5,2F6.0,2(5X,A1),8(1X,F5.0))``); a column shift misreads the field."""
    return {
        "PDATE": row[3:8], "PPOP": float(row[14:20]), "PPOE": float(row[20:26]),
        "PLME": row[31], "PLDS": row[37], "PLRS": float(row[39:44]), "PLDP": float(row[51:56]),
        "PLWT": float(row[57:62]), "SPRL": float(row[81:86]),
        "PLWT_FIELD": row[56:62], "SPRL_FIELD": row[80:86],
    }


def _dssat_treatment_mh(snx: str) -> int:
    """The ``*TREATMENTS`` harvest level MH as DSSAT reads it (ipexp FORMAT
    ``(I3,I1,2(1X,I1),1X,A25,14I3)``: MH is columns 68-70)."""
    return int(_row_after(snx, "@N R O C TNAME")[67:70])


def _cul_column_1(cul: Path) -> dict:
    rows = {}
    for raw in cul.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith(("!", "@", "*")):
            continue
        code = stripped.split(None, 1)[0]
        if 5 <= len(code) <= 7 and code.isalnum():
            rows[code] = raw[7:23].strip()
    return rows


def _dssat_harvest_row(row: str) -> dict:
    """Read a ``*HARVEST DETAILS`` data row with DSSAT's own layout (IPHAR FORMAT
    ``(I3,I5,3(1X,A5),2(1X,F5.0))``: HDATE is columns 4-8, HPC columns 28-32)."""
    return {"LEVEL": int(row[0:3]), "HDATE": row[3:8], "HSTG": row[9:14].strip(),
            "HCOM": row[15:20].strip(), "HSIZE": row[21:26].strip(), "HPC": float(row[27:32]),
            "HBPC": float(row[33:38])}


# The prism-runner's UC3 guards, which read the RAW template before any render; verbatim from
# prism-runner 73d9570 src/prism_runner/adapters/pythia.py:191-226 (docstrings dropped).
def _planting_row_consumes_pdate(template_src: str) -> bool:
    lines = template_src.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("@P") and "PDATE" in line:
            # The data row is the next non-blank line.
            for data_line in lines[i + 1:]:
                if data_line.strip():
                    return "{{" in data_line and "pdate" in data_line
            return False
    # No *PLANTING DETAILS @P header at all → cannot consume pdate.
    return False


def _management_row_consumes_plant_mode(template_src: str) -> bool:
    lines = template_src.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("@N MANAGEMENT"):
            for data_line in lines[i + 1:]:
                if data_line.strip():
                    return "{{" in data_line and "plant_mode" in data_line
            return False
    return False


def _site_data(cfg: ProjectConfig, missing=()) -> UnifiedData:
    """A 2x3 grid whose sites all carry a crop calendar and a year of primary climate, except the
    ``missing`` sites, which translate() must fetch from NASA POWER."""
    grid = _build_grid_2x3()
    calendar = cfg.crop.calendar
    first = date(cfg.temporal.start_year, 1, 1)
    climate = {
        cell.cell_id: ClimateTimeSeries(
            location_id=cell.cell_id, lat=cell.lat, lon=cell.lon, source="station",
            records=[ClimateRecord(date=first + timedelta(days=d), tmax=24.0, tmin=10.0,
                                   precip=2.0, srad=18.0) for d in range(365)])
        for cell in grid.cells if cell.cell_id not in missing
    }
    return UnifiedData(
        region=_data().region, grid=grid, soil=_build_profiles(), climate=climate,
        crop_calendar={cell.cell_id: CropCalendar(location_id=cell.cell_id,
                                                  planting_doy=calendar.planting_doy,
                                                  maturity_doy=calendar.maturity_doy)
                       for cell in grid.cells},
    )


def _translate(out: Path, missing=(), **kw):
    """The real ``translate()`` on ``_site_data``; weather fetches carry no delay."""
    kw.setdefault("pythia", PythiaConfig(weather_download_delay=0.0))
    cfg = _cfg(out, **kw)
    return PythiaTranslator(config=cfg, output_dir=str(out)).translate(_site_data(cfg, missing))


def _nasa_fetches(monkeypatch) -> list:
    """Record every NASA POWER request instead of sending it."""
    from prismpy.sources.climate.nasa_power import NASAPowerSource

    fetches = []

    def fetch(self, **request):
        fetches.append(request)
        raise ConnectionError("no NASA POWER request is expected")

    monkeypatch.setattr(NASAPowerSource, "retrieve", fetch)
    return fetches


def _package_provenance(out: Path, **kw) -> dict:
    """The ``provenance.json`` a user downloads with the package, built by the executor's own
    TRANSLATE (with the ``output_pythia`` artifact active) and PACKAGE stages."""
    from prismpy.pipeline.executor import TranslationPipeline
    from prismpy.provenance.tracker import ProvenanceTracker

    cfg = _cfg(out, pythia=PythiaConfig(weather_download_delay=0.0), **kw)
    pipeline = TranslationPipeline(cfg, provenance=ProvenanceTracker(
        enabled=True, output_dir=str(out / "provenance"), project_name="potato_substor"))
    data = _site_data(cfg)
    results = pipeline._execute_translate(data)
    assert results["pythia"].success, results["pythia"].errors
    pipeline._execute_package(data, results)
    return json.loads((out / "pythia" / "provenance.json").read_text())


def _translate_decisions(provenance: dict) -> list:
    return [decision
            for transformation in provenance["artifacts"]["output_pythia"]["transformations"]
            for decision in transformation["decisions"]]


# ── the SUBSTOR simulation module ─────────────────────────────────────────────

def test_potato_uses_the_substor_module_with_no_override(tmp_path):
    t = _translator(tmp_path)
    assert t._get_pythia_config().dssat_smodel is None
    assert t._get_dssat_smodel() == "PTSUB"


def test_rendered_potato_snx_runs_substor_with_the_desiree_cultivar(tmp_path):
    snx = _rendered_package_snx(tmp_path)
    assert _row_after(snx, "@N GENERAL").split()[-1] == "PTSUB"
    # the first run labels its CNAME with the scenario; INGENO selects the coefficients
    assert _row_after(snx, "@C CR INGENO CNAME").split() == [
        "1", "PT", "IB0008", "DESIREE_BASELINE"]


def test_a_padded_potato_name_prepares_the_same_substor_package(tmp_path):
    snx = _rendered_package_snx(tmp_path, crop=" Potato ")
    assert _row_after(snx, "@N GENERAL").split()[-1] == "PTSUB"
    assert _row_after(snx, "@C CR INGENO CNAME").split()[1:3] == ["PT", "IB0008"]
    row = _dssat_planting_row(_row_after(snx, "@P PDATE"))
    assert (row["PLWT"], row["SPRL"], row["PPOP"]) == (444.0, 0.1, 4.4)


def test_potato_is_not_a_nitrogen_fixing_legume_in_the_rendered_options(tmp_path):
    assert _named_columns(_rendered_package_snx(tmp_path), "@N OPTIONS")["SYMBI"] == "N"


# ── one cultivar resolver behind every producer ───────────────────────────────

def test_minimal_config_potato_emits_the_substor_cultivar(tmp_path):
    ds = _pythia_json(tmp_path)
    assert (ds["default_setup"]["ingeno"], ds["default_setup"]["cname"]) == ("IB0008", "DESIREE")
    assert {(r["ingeno"], r["cname"]) for r in ds["runs"]} == {
        ("IB0008", "DESIREE_BASELINE"), ("IB0008", "DESIREE_FERTILIZED")}


def test_managed_potato_emits_the_substor_cultivar(tmp_path):
    management = ManagementConfig(planting_density=44000.0)
    ds = _pythia_json(tmp_path, management=management)
    assert (ds["default_setup"]["ingeno"], ds["default_setup"]["cname"]) == ("IB0008", "DESIREE")
    assert {r["ingeno"] for r in ds["runs"]} == {"IB0008"}
    maturity_class = _translator(tmp_path, management=management)._map_generic_to_cultivar()[
        "maturity_class"]
    assert maturity_class not in {"early", "medium", "late"}


_CROPS = {
    # crop: (name_short, auto-detected SMODEL, default INGENO)
    "Maize": ("mze", "MZCER", "990002"),
    "Sorghum": ("sor", "SGCER", "990002"),
    "Millet": ("mil", "MLCER", "990002"),
    "Rice": ("ric", "RICER", "990002"),
    "Cowpea": ("cpe", "CROPGRO", "II0003"),
    "Groundnut": ("gnt", "CROPGRO", "IB0001"),
    "Beans": ("bns", "CROPGRO", "IB0001"),
    "Potato": ("pot", "PTSUB", "IB0008"),
}

# The minimal config (no management block) and a managed config in each sowing mode, built fresh
# per test so that a write-back to the shared config cannot leak between tests.
_VARIANTS = {
    "minimal-config": lambda: None,
    "opportunistic": lambda: ManagementConfig(planting_density=50000.0, sowing_mode="opportunistic"),
    "fixed_date": lambda: ManagementConfig(planting_density=50000.0, sowing_mode="fixed_date"),
}


@pytest.mark.parametrize("management", [None, ManagementConfig(planting_density=50000.0)],
                         ids=["minimal-config", "managed"])
@pytest.mark.parametrize("crop", sorted(_CROPS))
def test_every_crop_resolves_one_cultivar_on_both_config_paths(tmp_path, crop, management):
    short, smodel, ingeno = _CROPS[crop]
    t = _translator(tmp_path, crop=crop, short=short, management=management)
    ds = json.loads(Path(t._generate_pythia_json(_data())).read_text())
    assert t._get_dssat_smodel() == smodel
    assert ds["default_setup"]["ingeno"] == ingeno == t._map_generic_to_cultivar()["ingeno"]
    assert {r["ingeno"] for r in ds["runs"]} == {ingeno}


@pytest.mark.parametrize("crop,short,smodel,ingeno", [
    ("Beans", "bns", "BNGRO", "IB0001"),
    ("Potato", "pot", "PTSUB", "IB0003"),
])
def test_explicit_platform_overrides_still_win(tmp_path, crop, short, smodel, ingeno):
    t = _translator(tmp_path, crop=crop, short=short,
                    pythia=PythiaConfig(dssat_smodel=smodel, dssat_cultivar_ingeno=ingeno))
    ds = json.loads(Path(t._generate_pythia_json(_data())).read_text())["default_setup"]
    assert (t._get_dssat_smodel(), ds["ingeno"]) == (smodel, ingeno)


def test_cultivar_raster_is_tagged_with_the_substor_cultivar(tmp_path):
    import rasterio

    t = _translator(tmp_path)
    t._generate_management_rasters({}, _build_grid_2x3())
    with rasterio.open(tmp_path / "raster" / "cultivar.tif") as src:
        tags = src.tags()
    assert (tags["cultivar_code"], tags["cultivar_name"]) == ("IB0008", "DESIREE")


def test_cultivar_raster_has_no_silent_maize_fallback(tmp_path, monkeypatch):
    t = _translator(tmp_path)
    monkeypatch.setattr(t, "_map_generic_to_cultivar", lambda: {"cname": "DESIREE"})
    with pytest.raises(KeyError):
        t._generate_management_rasters({}, _build_grid_2x3())


def test_manifest_and_readme_record_the_emitted_cultivar(tmp_path):
    t = _translator(tmp_path)
    data = _data()
    ingeno = json.loads(Path(t._generate_pythia_json(data)).read_text())["default_setup"]["ingeno"]
    manifest = json.loads(Path(t._generate_manifest(data)).read_text())
    assert manifest["crops"][0]["cultivar_id"] == ingeno == "IB0008"
    readme = Path(t._generate_readme(data)).read_text()
    assert "| Cultivar Code | IB0008 |" in readme
    assert "| Cultivar Name | DESIREE |" in readme


def test_package_time_surfaces_reuse_the_translate_time_cultivar(tmp_path, monkeypatch):
    t = _translator(tmp_path)
    data = _data()
    t._generate_pythia_json(data)
    monkeypatch.setattr(t, "_map_generic_to_cultivar", lambda: {
        "ingeno": "XXXXXX", "cname": "SENTINEL", "maturity_class": "module_default",
        "total_gdd": None})
    manifest = Path(t._generate_manifest(data)).read_text()
    readme = Path(t._generate_readme(data)).read_text()
    assert json.loads(manifest)["crops"][0]["cultivar_id"] == "IB0008"
    assert "| Cultivar Code | IB0008 |" in readme
    assert "XXXXXX" not in manifest + readme


def test_desiree_is_a_real_column_1_cultivar_of_the_pinned_ptsub048(tmp_path):
    assert hashlib.sha256(_PTSUB048_CUL.read_bytes()).hexdigest() == _PTSUB048_SHA256
    resolved = _translator(tmp_path)._map_generic_to_cultivar()
    registry = _cul_column_1(_PTSUB048_CUL)
    assert resolved["ingeno"] in registry
    assert registry[resolved["ingeno"]] == resolved["cname"]


# ── SUBSTOR seed-tuber planting material ──────────────────────────────────────

@pytest.mark.parametrize("management", [None, ManagementConfig(planting_density=44000.0,
                                                              row_spacing_cm=75.0)],
                         ids=["minimal-config", "managed"])
def test_rendered_potato_planting_row_carries_the_seed_tuber_values(tmp_path, management):
    row = _dssat_planting_row(_row_after(_rendered_package_snx(tmp_path, management=management),
                                         "@P PDATE"))
    assert (row["PLWT"], row["SPRL"]) == (444.0, 0.1)
    assert (row["PLWT_FIELD"], row["SPRL_FIELD"]) == ("   444", "   0.1")
    assert (row["PPOP"], row["PPOE"], row["PLME"], row["PLRS"], row["PLDP"]) == (
        4.4, 4.4, "S", 75.0, 10.0)


def test_an_explicit_potato_density_beats_the_default(tmp_path):
    snx = _rendered_package_snx(tmp_path, management=ManagementConfig(planting_density=30000.0))
    row = _dssat_planting_row(_row_after(snx, "@P PDATE"))
    assert (row["PPOP"], row["PPOE"], row["PLWT"], row["SPRL"]) == (3.0, 3.0, 444.0, 0.1)


def test_potato_planting_fallback_is_never_the_dssat_fatal_minus_99(tmp_path):
    row = _dssat_planting_row(_row_after(_runner_render(_emitted_template(_translator(tmp_path)),
                                                        {}), "@P PDATE"))
    assert (row["PLWT"], row["SPRL"]) == (444.0, 0.1)


def test_potato_seed_tuber_values_are_literals_of_the_raw_template(tmp_path):
    raw_row = _row_after(_emitted_template(_translator(tmp_path)), "@P PDATE")
    assert raw_row.endswith("   444   -99   -99   -99   0.1                        auto")


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
@pytest.mark.parametrize("crop", sorted(_CROPS))
def test_fixed_dssat_values_have_no_second_source_in_the_run_config(tmp_path, crop, variant):
    t = _translator(tmp_path, crop=crop, short=_CROPS[crop][0], management=_VARIANTS[variant]())
    pythia = json.loads(Path(t._generate_pythia_json(_data())).read_text())
    raw = _emitted_template(t)
    assert [token for token in ("plwt", "sprl", "hdate") if token in raw] == []
    for block in (pythia["default_setup"], *pythia["runs"]):
        assert not {"plwt", "sprl", "hdate"} & set(block)


@pytest.mark.parametrize("crop", sorted(set(_CROPS) - {"Potato"}))
def test_other_crops_keep_no_seed_tuber_material(tmp_path, crop):
    short = _CROPS[crop][0]
    ds = _pythia_json(tmp_path, crop=crop, short=short)["default_setup"]
    assert "plwt" not in ds and "sprl" not in ds
    snx = _rendered_package_snx(tmp_path, crop=crop, short=short)
    row = _dssat_planting_row(_row_after(snx, "@P PDATE"))
    assert (row["PLWT"], row["SPRL"]) == (-99.0, -99.0)


@pytest.mark.parametrize("sowing,plant", [("opportunistic", "A"), ("fixed_date", "R")])
@pytest.mark.parametrize("crop", sorted(set(_CROPS) - {"Potato"}))
def test_other_crops_keep_their_planting_and_maturity_harvest(tmp_path, crop, sowing, plant):
    short = _CROPS[crop][0]
    snx = _rendered_package_snx(tmp_path, crop=crop, short=short, management=ManagementConfig(
        planting_density=50000.0, sowing_mode=sowing))
    management = _named_columns(snx, "@N MANAGEMENT")
    assert (management["PLANT"], management["HARVS"]) == (plant, "M")
    assert _dssat_treatment_mh(snx) == 0
    assert "*HARVEST DETAILS" not in snx


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
@pytest.mark.parametrize("crop", sorted(_CROPS))
def test_every_raw_template_keeps_the_runner_uc3_placeholders(tmp_path, crop, variant):
    raw = _emitted_template(_translator(tmp_path, crop=crop, short=_CROPS[crop][0],
                                        management=_VARIANTS[variant]()))
    assert _planting_row_consumes_pdate(raw)
    assert _management_row_consumes_plant_mode(raw)


# ── potato plants on its reported date: DSSAT cannot plant it automatically ───

@pytest.mark.parametrize("variant", ["minimal-config", "opportunistic", "fixed_date", "fixed"])
def test_potato_plants_on_the_reported_date_in_every_run(tmp_path, variant):
    management = (None if variant == "minimal-config"
                  else ManagementConfig(planting_density=44000.0, sowing_mode=variant))
    requested = management and management.sowing_mode
    t = _translator(tmp_path, management=management)
    pythia = json.loads(Path(t._generate_pythia_json(_data())).read_text())
    template = _emitted_template(t)

    assert pythia["default_setup"]["plant_mode"] == "R"
    assert [run["plant_mode"] for run in pythia["runs"]] == ["R", "R"]
    assert pythia["default_setup"]["pdate"] == "2015-06-15"
    for run in pythia["runs"]:
        snx = _runner_render(template, {**pythia["default_setup"], **run})
        assert _named_columns(snx, "@N MANAGEMENT")["PLANT"] == "R"
        assert _dssat_planting_row(_row_after(snx, "@P PDATE"))["PDATE"] == "15166"
    # the shared config keeps the requested sowing mode; only the DSSAT emit substitutes
    assert (t.config.management and t.config.management.sowing_mode) == requested


def test_potato_management_row_is_a_value_substitution_under_the_runner_placeholder(tmp_path):
    raw = _emitted_template(_translator(tmp_path))
    assert _row_after(raw, "@N MANAGEMENT") == (
        ' 1 MA              {{ plant_mode | default("R") }} {{ irrig }}     D     D     D')


# ── potato is harvested the season length after its actual planting ──────────

@pytest.mark.parametrize("planting,maturity,hdate", [(166, 285, "  119"), (330, 120, "  155")],
                         ids=["same-year", "cross-year"])
@pytest.mark.parametrize("variant", ["minimal-config", "opportunistic"])
def test_potato_is_harvested_the_season_length_after_its_actual_planting(
        tmp_path, variant, planting, maturity, hdate):
    t = _translator(tmp_path, management=_VARIANTS[variant](), planting_doy=planting,
                    maturity_doy=maturity)
    pythia = json.loads(Path(t._generate_pythia_json(_data())).read_text())
    template = _emitted_template(t)
    for run in pythia["runs"]:
        snx = _runner_render(template, {**pythia["default_setup"], **run})
        assert _named_columns(snx, "@N MANAGEMENT")["HARVS"] == "D"
        assert _dssat_treatment_mh(snx) == 1
        assert _dssat_harvest_row(_row_after(snx, "@H HDATE")) == {
            "LEVEL": 1, "HDATE": hdate, "HSTG": "-99", "HCOM": "-99", "HSIZE": "-99",
            "HPC": 100.0, "HBPC": -99.0}


# ── a season DSSAT would refuse or silently cut short fails before any write ──

@pytest.mark.parametrize("planting,maturity,year", [(166, 166, 2015), (366, 1, 2016)],
                         ids=["same-day", "dec-31-to-jan-1"])
def test_a_potato_season_under_one_day_is_refused_at_translate(tmp_path, planting, maturity, year):
    out = tmp_path / "pythia"
    result = _translate(out, planting_doy=planting, maturity_doy=maturity, start_year=year)
    assert not result.success
    assert "growing season of 0 days" in result.errors[0]
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("start,end,non_leap", [(2015, 2016, "2015"), (2016, 2017, "2017")])
@pytest.mark.parametrize("crop,short", [("Potato", "pot"), ("Maize", "mze")])
def test_planting_on_doy_366_is_refused_when_a_season_year_is_not_leap(
        tmp_path, crop, short, start, end, non_leap):
    out = tmp_path / "pythia"
    result = _translate(out, crop=crop, short=short, planting_doy=366, maturity_doy=120,
                        start_year=start, end_year=end)
    assert not result.success
    assert f"{crop} planting DOY 366" in result.errors[0] and non_leap in result.errors[0]
    assert list(out.iterdir()) == []


def test_potato_planting_on_doy_366_of_a_leap_season_is_its_last_day(tmp_path):
    t = _translator(tmp_path, planting_doy=366, maturity_doy=120, start_year=2016)
    t._check_season_dates()
    pythia = json.loads(Path(t._generate_pythia_json(_data())).read_text())
    assert pythia["default_setup"]["pdate"] == "2016-12-31"


@pytest.mark.parametrize("missing", [(0,), ()], ids=["a-site-needs-nasa-power", "none-does"])
def test_a_climate_end_before_the_last_potato_harvest_is_refused_before_any_download(
        tmp_path, monkeypatch, missing):
    fetches = _nasa_fetches(monkeypatch)
    out = tmp_path / "pythia"
    # 166 -> 285 over 2015-2016: the last harvest is 119 days after 2016-06-14, on 2016-10-11
    result = _translate(out, missing=missing, start_year=2015, end_year=2016, pythia=PythiaConfig(
        climate_end_date="2016-10-10", weather_download_delay=0.0))
    assert not result.success
    assert "2016-10-11" in result.errors[0] and "2016-10-10" in result.errors[0]
    assert fetches == []
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("planting,maturity,year,end", [
    (166, 285, 2015, None),          # the default end, 2015-12-31, follows the 2015-10-12 harvest
    (330, 120, 2015, None),          # non-leap planting year: the default end IS the harvest day
    (330, 120, 2016, None),          # leap planting year: harvest 2017-04-29, default end 04-30
    (166, 285, 2015, "2015-10-12"),  # an explicit end on the harvest day itself
], ids=["same-year", "cross-year-non-leap", "cross-year-leap", "explicit-end"])
def test_a_climate_end_on_or_after_the_last_potato_harvest_is_accepted(
        tmp_path, planting, maturity, year, end):
    t = _translator(tmp_path, planting_doy=planting, maturity_doy=maturity, start_year=year,
                    pythia=PythiaConfig(climate_end_date=end))
    t._check_season_dates()


# ── the planting substitution is recorded once, in the package's provenance ──

_SUBSTITUTION = ("Potato sowing: requested_sowing_mode=opportunistic, "
                 "effective_sowing_mode=fixed_date")


@pytest.mark.parametrize("variant", ["minimal-config", "opportunistic"])
def test_an_opportunistic_potato_package_records_its_sowing_substitution_once(tmp_path, variant):
    provenance = _package_provenance(tmp_path, management=_VARIANTS[variant]())
    substitutions = [decision for decision in _translate_decisions(provenance)
                     if decision["decision_type"] == "fallback_substitution"]
    assert [(d["severity"], d["description"]) for d in substitutions] == [
        ("warning", _SUBSTITUTION)]
    assert substitutions[0]["rationale"].startswith(
        "reason=dssat_substor_automatic_planting_unsupported; ")
    assert not provenance.get("unattached_decisions")


@pytest.mark.parametrize("crop,short,sowing", [("Potato", "pot", "fixed_date"),
                                               ("Maize", "mze", "opportunistic")])
def test_a_package_planted_as_requested_records_no_sowing_substitution(
        tmp_path, crop, short, sowing):
    provenance = _package_provenance(tmp_path, crop=crop, short=short, management=ManagementConfig(
        planting_density=44000.0, sowing_mode=sowing))
    assert [decision for decision in _translate_decisions(provenance)
            if decision["decision_type"] == "fallback_substitution"] == []
    assert not provenance.get("unattached_decisions")


# ── crop x platform admission, before any translator is built ─────────────────

def _pipeline(tmp_path, targets, monkeypatch, **cfg):
    from prismpy.pipeline.executor import TranslationPipeline
    from prismpy.provenance.tracker import ProvenanceTracker

    pipeline = TranslationPipeline(
        _cfg(tmp_path, targets=targets, **cfg),
        provenance=ProvenanceTracker(enabled=False, project_name="potato_admission"),
    )
    calls = []

    def spy(platform):
        calls.append(platform)
        return None

    monkeypatch.setattr(pipeline, "_get_translator", spy)
    return pipeline, calls


def test_potato_on_pythia_proceeds_to_translation(tmp_path, monkeypatch):
    pipeline, calls = _pipeline(tmp_path, [Platform.PYTHIA], monkeypatch)
    pipeline._execute_translate(_data())
    assert calls == [Platform.PYTHIA]


@pytest.mark.parametrize("platform", [Platform.ACEA, Platform.CRAFT, Platform.SARRA_PY])
def test_potato_on_an_unsupported_platform_is_refused_before_translation(
        tmp_path, monkeypatch, platform):
    from prismpy.packaging.manifest import UnsupportedCropError

    pipeline, calls = _pipeline(tmp_path, [platform], monkeypatch)
    with pytest.raises(UnsupportedCropError, match=platform.value):
        pipeline._execute_translate(_data())
    assert calls == []
    assert not (tmp_path / platform.value).exists()


@pytest.mark.parametrize("targets", [[Platform.PYTHIA, Platform.ACEA],
                                     [Platform.ACEA, Platform.PYTHIA]],
                         ids=["pythia-first", "acea-first"])
def test_mixed_targets_are_refused_atomically_with_no_partial_output(
        tmp_path, monkeypatch, targets):
    from prismpy.packaging.manifest import UnsupportedCropError

    pipeline, calls = _pipeline(tmp_path, targets, monkeypatch)
    with pytest.raises(UnsupportedCropError, match="acea"):
        pipeline._execute_translate(_data())
    assert calls == []
    assert not (tmp_path / "pythia").exists()


def test_a_disabled_unsupported_target_does_not_block_potato(tmp_path, monkeypatch):
    pipeline, calls = _pipeline(tmp_path, [Platform.PYTHIA, Platform.ACEA], monkeypatch,
                                acea_enabled=False)
    pipeline._execute_translate(_data())
    assert calls == [Platform.PYTHIA]


def test_crop_support_predicate_normalizes_platform_and_crop_spelling():
    from prismpy.packaging.manifest import is_crop_supported

    assert is_crop_supported(Platform.PYTHIA, "Potato")
    assert is_crop_supported("pythia", " potato ")
    assert is_crop_supported("PYTHIA", "POTATO")
    for platform in ("acea", "craft", "sarra_py"):
        assert not is_crop_supported(platform, "Potato")
    assert not is_crop_supported("unknown_engine", "Maize")
    assert not is_crop_supported("pythia", "")
    assert not is_crop_supported("pythia", None)


def test_manifest_support_gate_and_admission_share_one_predicate():
    from prismpy.packaging.manifest import _eval_gate_crop_supported_per_platform

    assert _eval_gate_crop_supported_per_platform({"crop": {"name": "Potato"}}, "pythia")
    assert not _eval_gate_crop_supported_per_platform({"crop": {"name": "Potato"}}, "acea")
    assert _eval_gate_crop_supported_per_platform({"crop": {"name": " POTATO "}}, "pythia")


def test_supported_crop_map_keeps_both_spellings_for_exact_match_readers():
    from prismpy.packaging.manifest import _PLATFORM_SUPPORTED_CROPS

    assert {"potato", "Potato"} <= _PLATFORM_SUPPORTED_CROPS["pythia"]
    for platform in ("acea", "craft", "sarra_py"):
        assert not {"potato", "Potato"} & _PLATFORM_SUPPORTED_CROPS[platform]


# ── ECOCROP climate envelope ──────────────────────────────────────────────────

def test_potato_ecocrop_envelope_is_the_fao_absolute_tolerance():
    from prismpy.koppen.envelopes import load_ecocrop_envelopes

    env = load_ecocrop_envelopes()["potato"]
    assert (env["TMIN"], env["TMAX"], env["RMIN"], env["RMAX"]) == (7.0, 30.0, 250.0, 2000.0)
    assert env["verbatim_source_url"] == (
        "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1971")
    assert env["verbatim_retrieval_date"] == "2026-09-24"


def test_built_wheel_ships_a_loadable_potato_envelope(tmp_path):
    import subprocess
    import sys
    import zipfile

    from prismpy.koppen.envelopes import load_ecocrop_envelopes

    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir",
                    "--wheel-dir", str(tmp_path), str(root)],
                   check=True, capture_output=True, text=True, timeout=240)
    (wheel,) = tmp_path.glob("prismpy-*.whl")
    shipped = tmp_path / "ecocrop_envelopes.json"
    with zipfile.ZipFile(wheel) as zf:
        shipped.write_bytes(zf.read("prismpy/koppen/ecocrop_envelopes.json"))
    env = load_ecocrop_envelopes(shipped)["potato"]
    assert (env["TMIN"], env["TMAX"], env["RMIN"], env["RMAX"]) == (7.0, 30.0, 250.0, 2000.0)
