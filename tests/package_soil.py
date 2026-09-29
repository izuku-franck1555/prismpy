"""A package's soil files as the production writers stamp and bind them, for tests that
build a manifest or README without running a translator."""
from __future__ import annotations

from pathlib import Path

from prismpy.models.region import BoundingBox, Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import GridCell, SpatialGrid
from prismpy.packaging.soil_declaration import DETAIL_FILE, SoilStamp, write_binding
from prismpy.translators._shared.dssat_sol_writer import write_dssat_sol
from prismpy.translators._shared.eghr_substrate import build_eghr_substrate

REGION = Region(name="Test", country="Mali", country_iso3="MLI",
                bounds=BoundingBox(minx=-6.0, miny=12.0, maxx=-5.0, maxy=13.0))


def isda_profile() -> SoilProfile:
    return SoilProfile(profile_id="isda_1", lat=12.5, lon=-5.5, source="iSDA S3 (30m)", layers=[
        SoilLayer(depth_top=0.0, depth_bottom=0.2, sand=50.0, clay=20.0, organic_carbon=0.8,
                  bulk_density=1.4, ph=6.2),
        SoilLayer(depth_top=0.2, depth_bottom=0.5, sand=48.0, clay=24.0, organic_carbon=0.5,
                  bulk_density=1.45, ph=6.3),
    ])


def stamp_package_soil(package_dir, platform) -> Path:
    """Write, stamp and bind the soil files ``platform`` reads (ACEA and SARRA-Py need none)."""
    package_dir = Path(package_dir)
    name = str(getattr(platform, "value", platform)).lower()
    if name == "craft":
        soil = package_dir / "soil"
        soil.mkdir(parents=True, exist_ok=True)
        sol = soil / "ML.SOL"
        names = write_dssat_sol(sol, {90_000_001: isda_profile()}, "ML", REGION,
                                stamp=SoilStamp("profiles"),
                                source_label_for_id=lambda key: f"iSDA profile {key}",
                                profile_cell_counts={90_000_001: 1})
        (soil / "soil_mask.txt").write_text(f"CellID\tSoilProfile\tSharePCT\n1\t{names[90_000_001]}\t1\n")
        write_binding(package_dir, "soil", [sol, soil / "soil_mask.txt", soil / DETAIL_FILE])
    elif name == "pythia":
        grid = SpatialGrid(resolution="5arcmin", cells=[GridCell(cell_id=1, lat=12.5, lon=-5.5, row=0, col=0)])
        build_eghr_substrate(grid, {1: isda_profile()}, "ML", REGION, package_dir)
    return package_dir
