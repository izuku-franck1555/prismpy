"""Chemistry-extraction pins for the HWSD v1->v2 column fix (#156).

HWSD v2.0 stores soil chemistry under ORG_CARBON / PH_WATER / BULK, but the
reader used the v1 names (T_OC / T_PH_H2O / T_BULK_DENSITY), so DSSAT-family
runs computed yields on silently-defaulted chemistry (OC 0.5%, pH 6.5, BD
1.4). These pins lock the fix: chemistry resolves from the real columns, a
mis-scaled read is rejected, bulk density reads BULK (not the packed
REF_BULK), and a genuinely-absent value is disclosed to provenance instead
of written as if it were measured.
"""
from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.sources.soil.hwsd import HWSDConfig, HWSDSource
from prismpy.translators._shared.dssat_sol_writer import write_dssat_sol

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hwsd_3smu_fixture_calvados.csv"
)

# Dominant-component (SEQUENCE=1) topsoil (D1) chemistry sampled from the real
# HWSD2.mdb for the Calvados SMUs: organic_carbon, ph, bulk_density.
EXPECTED = {
    9426: (0.628, 5.7, 1.44),
    9445: (3.027, 7.7, 1.23),
    9501: (1.562, 6.4, 1.41),
}
# The packed reference bulk density in the same rows — the value a REF_BULK
# misread would return instead of the moist BULK.
REF_BULK = {9426: 1.2, 9445: 1.81, 9501: 1.71}


def _region() -> Region:
    return Region(
        name="Calvados",
        country="France",
        country_iso3="FRA",
        bounds=BoundingBox(minx=-1.0, miny=48.5, maxx=0.5, maxy=49.5),
    )


def _extract(source: HWSDSource, smu_ids):
    """Drive the real extractor chain with the fixture standing in for the
    MDB export and a fixed SMU-id sample (the BIL raster lookup)."""
    coords = [(48.5 + i * 0.1, -0.5 + i * 0.1) for i in range(len(smu_ids))]
    with patch.object(source, "_sample_bil_raster", return_value=list(smu_ids)), \
            patch.object(source, "_export_mdb_table", return_value=pd.read_csv(FIXTURE)):
        return source._extract_from_bil_mdb(region=_region(), cell_coords=coords)


class ChemistryExtractionPins(TestCase):
    def setUp(self):
        self.source = HWSDSource(
            config=HWSDConfig(mdb_path=Path("unused"), bil_path=Path("unused")),
        )

    def test_p_chem_extract_real_values(self):
        """P-CHEM-EXTRACT: extracted chemistry equals the sampled per-SMU
        values, not the flat 0.5/6.5/1.4 default. RED on pre-fix e484935."""
        smu_ids = [9426, 9445, 9501]
        profiles = _extract(self.source, smu_ids)
        self.assertEqual(len(profiles), 3)
        for i, smu in enumerate(smu_ids):
            oc, ph, bd = EXPECTED[smu]
            layer = profiles[i].layers[0]
            self.assertIsNotNone(layer.organic_carbon, f"SMU {smu} OC is None")
            self.assertAlmostEqual(layer.organic_carbon, oc, delta=1e-3)
            self.assertAlmostEqual(layer.ph, ph, delta=1e-3)
            self.assertAlmostEqual(layer.bulk_density, bd, delta=1e-3)
            # Explicitly not the silent default.
            self.assertNotAlmostEqual(layer.organic_carbon, 0.5, delta=1e-3)

    def test_p_bulk_not_refbulk(self):
        """P-BULK-NOT-REFBULK: bd resolves BULK, never the packed REF_BULK."""
        self.assertIsInstance(HWSDSource.VARIABLES["bd"], list)
        self.assertEqual(HWSDSource.VARIABLES["bd"][0], "BULK")
        self.assertNotIn("REF_BULK", HWSDSource.VARIABLES["bd"])
        smu_ids = [9426, 9445, 9501]
        profiles = _extract(self.source, smu_ids)
        for i, smu in enumerate(smu_ids):
            bd = profiles[i].layers[0].bulk_density
            self.assertAlmostEqual(bd, EXPECTED[smu][2], delta=1e-3)
            self.assertNotAlmostEqual(bd, REF_BULK[smu], delta=1e-2)

    def test_p_units_reject_misscale(self):
        """P-UNITS: an out-of-range (mis-scaled) chemistry value is rejected."""
        # In-range values pass through unchanged.
        self.assertAlmostEqual(
            self.source._read_chem({"ORG_CARBON": 0.628}, "soc"), 0.628, delta=1e-9)
        self.assertAlmostEqual(
            self.source._read_chem({"BULK": 1.44}, "bd"), 1.44, delta=1e-9)
        # x100 OC, /100 BD, x10 pH mis-scales all fall outside the range -> None.
        self.assertIsNone(self.source._read_chem({"ORG_CARBON": 62.8}, "soc"))
        self.assertIsNone(self.source._read_chem({"BULK": 0.0144}, "bd"))
        self.assertIsNone(self.source._read_chem({"PH_WATER": 57.0}, "ph"))

    def test_p_units_none_and_nan_absent(self):
        """A missing (None/NaN) chemistry cell resolves to None (absent)."""
        self.assertIsNone(self.source._read_chem({}, "soc"))
        self.assertIsNone(self.source._read_chem({"ORG_CARBON": float("nan")}, "soc"))

    def test_p_canonical_no_hardcoded_chem_chains(self):
        """P-CANONICAL: Path 1 routes chemistry through _read_chem; no
        hardcoded v1 column chains remain in _create_profile_from_hwsd."""
        src = inspect.getsource(HWSDSource._create_profile_from_hwsd)
        for legacy in ('props.get("T_OC")', 'props.get("T_PH_H2O")',
                       'props.get("T_BULK_DENSITY")'):
            self.assertNotIn(legacy, src)
        for chem in ("soc", "ph", "bd"):
            self.assertIn(f'_read_chem(props, "{chem}")', src)

    def test_p_hwsd_scoped_read_chem_not_shared(self):
        """P-HWSD-SCOPED: _read_chem is on the HWSD source and iSDA neither
        defines nor references it (iSDA scales bd/100, ph/10)."""
        self.assertTrue(hasattr(HWSDSource, "_read_chem"))
        from prismpy.sources.soil import isda as isda_module
        self.assertNotIn("_read_chem", inspect.getsource(isda_module))

    def test_p_both_paths_swept(self):
        """P-BOTH-PATHS: the raw-MDB read-site resolves the v2.0 columns via
        the candidate map (v2 name first); Path 2 reads normalized keys."""
        self.assertEqual(HWSDSource.VARIABLES["soc"][0], "ORG_CARBON")
        self.assertEqual(HWSDSource.VARIABLES["ph"][0], "PH_WATER")
        self.assertEqual(HWSDSource.VARIABLES["bd"][0], "BULK")
        # Path 2 stays on the normalized keys, downstream of the map.
        path2 = inspect.getsource(HWSDSource._create_profile_from_dict)
        self.assertIn('props.get("soc")', path2)


class SolHonestyPin(TestCase):
    def _write(self, layer, log):
        profile = SoilProfile(
            profile_id="FR00000001", lat=49.0, lon=-0.3, source="HWSD",
            total_depth=0.2, layers=[layer],
        )
        with tempfile.TemporaryDirectory() as d:
            write_dssat_sol(
                soil_path=Path(d) / "T.SOL",
                profiles_by_id={1: profile},
                country_code="FR",
                region=_region(),
                chem_default_log=log,
            )

    def test_p_sol_honesty_discloses_default(self):
        """P-SOL-HONESTY: genuinely-absent chemistry is disclosed to the
        default log, not silently written as measured. RED if reverted to
        the silent ``or 0.5`` fill."""
        log = []
        self._write(
            SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=60.0, clay=18.0,
                      silt=22.0, organic_carbon=None, ph=None, bulk_density=None),
            log,
        )
        self.assertEqual({e["field"] for e in log},
                         {"organic_carbon", "ph", "bulk_density"})

    def test_p_sol_honesty_present_value_not_logged(self):
        """A present chemistry value (including a real 0.0) is written as-is
        and never recorded as a default."""
        log = []
        self._write(
            SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=60.0, clay=18.0,
                      silt=22.0, organic_carbon=0.0, ph=6.4, bulk_density=1.41),
            log,
        )
        self.assertEqual(log, [])

    def test_p_sol_honesty_craft_provenance_fail_open(self):
        """The CRAFT chemistry-default provenance record is wrapped fail-open,
        so a tracker failure preserves the already-written .SOL rather than
        crashing soil generation (the HWSD no-coverage note holds the same
        invariant). Unwrap the try/except -> RED."""
        craft_src = (
            Path(__file__).resolve().parents[2]
            / "src" / "prismpy" / "translators" / "craft" / "translator.py"
        ).read_text()
        lines = craft_src.splitlines()
        for i, line in enumerate(lines):
            if "if chem_defaults and self.provenance:" in line:
                window = "\n".join(lines[i:i + 8])
                self.assertIn("try:", window,
                              "chem-default provenance must open a fail-open try")
                self.assertIn("record_decision", window)
                break
        else:
            self.fail("CRAFT chem-default provenance block not found")
        self.assertIn("preserving the generated .SOL", craft_src)
