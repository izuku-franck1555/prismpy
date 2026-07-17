"""Generate a realistic SYNTHETIC sorghum N-response trials CSV for the Demo-1
non-AGMIP genericity proof (sorghum, Kano-state Nigeria). AC-7 proves PIPELINE
genericity (a DIFFERENT trials CSV flowing generically), NOT model validation —
so the numbers are plausible-agronomic, deterministic (no RNG), not measured.

Schema (n_response_skill processor): required cell_id, year, scenario_label,
yield_obs_kgha; optional n_level_kg_ha, biomass_obs_kgha, lat, lon."""
from pathlib import Path

OUT = Path("/Users/francktonle/Downloads/DATA-TO-MODEL-TRANSLATION/prismpy/nrisk_demo1_sorghum_nigeria_trials.csv")

# 6 sorghum trial sites SPREAD across the wider Kano bbox (8.0-9.0E, 11.0-12.0N)
# so trials pair to cells in multiple NASA-POWER weather cells + soil zones (Tier-A
# coverage distributed, not corner-clustered).
SITES = [
    ("KNS1", 11.15, 8.15),
    ("KNS2", 11.35, 8.60),
    ("KNS3", 11.55, 8.25),
    ("KNS4", 11.70, 8.80),
    ("KNS5", 11.85, 8.40),
    ("KNS6", 11.25, 8.90),
]
YEARS = [2018, 2019, 2020]
# N treatments: label -> applied kg N/ha
TREATMENTS = [("N0", 0), ("N30", 30), ("N60", 60), ("N90", 90)]

# Plausible sorghum N-response (kg/ha): saturating Mitscherlich-like curve.
#   yield = ymax*(1 - exp(-k*(N + Nsoil))) with site + year multiplicative offsets.
YMAX = 2600.0
K = 0.013
NSOIL = 18.0  # background soil N contribution
SITE_MULT = {"KNS1": 0.92, "KNS2": 1.06, "KNS3": 0.99,
             "KNS4": 1.11, "KNS5": 0.88, "KNS6": 1.02}
YEAR_MULT = {2018: 1.03, 2019: 0.90, 2020: 1.00}  # 2019 a drier year
HARVEST_INDEX = 0.42  # biomass = grain / HI


def yield_kgha(site, year, n):
    import math
    base = YMAX * (1.0 - math.exp(-K * (n + NSOIL)))
    return round(base * SITE_MULT[site] * YEAR_MULT[year], 1)


rows = ["cell_id,year,scenario_label,n_level_kg_ha,yield_obs_kgha,biomass_obs_kgha,lat,lon"]
for site, lat, lon in SITES:
    for year in YEARS:
        for label, n in TREATMENTS:
            y = yield_kgha(site, year, n)
            biomass = round(y / HARVEST_INDEX, 1)
            rows.append(f"{site},{year},{label},{n},{y},{biomass},{lat},{lon}")

OUT.write_text("\n".join(rows) + "\n")
print(f"wrote {OUT} — {len(rows)-1} rows "
      f"({len(SITES)} sites x {len(YEARS)} yr x {len(TREATMENTS)} N-levels)")
print("\n".join(rows[:6]))
