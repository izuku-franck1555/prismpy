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
from datetime import datetime
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
         start_year: int = 2015, pythia: PythiaConfig | None = None,
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
        temporal=TemporalConfig(start_year=start_year, end_year=start_year, spinup_years=0),
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
    field becomes DSSAT YYDDD (``%y%j``) and an int ``hdate`` a right-justified width-5 field;
    every other value passes through, as the runner passes the keys it does not register."""
    def runner_value(key, value):
        if key in _RUNNER_DATE_FIELDS and isinstance(value, str) and "::" not in value:
            return datetime.strptime(value, "%Y-%m-%d").strftime("%y%j")
        if key == "hdate" and isinstance(value, int) and not isinstance(value, bool):
            return f"{value:>5d}"
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
