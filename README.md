# prismpy

PRISM: A methodological framework for reproducible data-to-model translation in spatial crop modeling.

---

## Overview

Process-based crop models are increasingly applied to forecast yields and assess climate risks in rainfed smallholder agriculture. However, the prerequisite step of transforming heterogeneous agricultural data into standardized, model-ready inputs remains a major challenge for reproducible and comparable spatial crop modeling. Here we present PRISM (Platform-Ready Inputs for Spatial Modeling), a framework that formalizes data retrieval, quality control, gap-filling, spatial harmonization, and format conversion to produce standardized input packages compatible with multiple crop modeling frameworks. PRISM separates agronomic specifications from model-specific settings, with provenance records documenting preparation decisions.

`prismpy` implements the PRISMA framework. It produces **standardized, self-documenting data packages** for multiple spatial crop modeling platforms from a single configuration.

### Key Features

- **Multi-platform output**: Generate inputs for CRAFT, PYTHIA, ACEA, and SARRA-Py from one workflow
- **ICASA/DOME architecture**: Separate agronomic data from platform-specific settings for interoperability
- **Self-documenting packages**: Each output includes manifest, provenance, and usage instructions
- **Backward compatible**: Legacy single-file configs continue to work

---

## Installation

```bash
git clone https://github.com/izuku-franck1555/prismpy.git
cd prismpy
python -m venv venv
source venv/bin/activate
pip install -e ".[all]"
```

---

## Quick Start

### Option 1: Legacy Format (Single File)

```bash
prismpy translate --config examples/koutiala_craft_config.yaml
```

### Option 2: DOME Format (Base + Platform Overlay)

```bash
# Separate agronomic data from platform-specific settings
prismpy translate \
  --base configs/base/koutiala_maize.yaml \
  --dome configs/domes/craft_dome.yaml
```

The DOME format separates concerns:
- **Base config**: ICASA-compliant agronomic parameters (shareable across platforms)
- **Platform DOME**: Platform-specific settings (DSSAT codes, file paths, schema options)

---

## Configuration

### Base Config (ICASA-Compliant)

```yaml
# configs/base/koutiala_maize.yaml
_meta:
  format: "icasa_ace"

project:
  name: "Koutiala_Maize"

region:
  name: "Koutiala"
  country: "Mali"
  country_iso3: "MLI"
  FL_LAT: 12.4
  FL_LONG: -5.6

crop:
  CRID: "MZ"                    # ICASA crop code
  name: "Maize"                 # Human-readable alias

  phenology:
    P1: 80.0                    # ICASA: GDD to emergence
    emergence_gdd: 80.0         # Alias (both valid)

  physiology:
    TB: 8.0                     # ICASA: Base temperature
    base_temperature: 8.0       # Alias

management:
  PPOP: 5.5                     # ICASA: plants/m²
  planting_density: 55000       # Alias: plants/ha

temporal:
  start_year: 2010
  end_year: 2020
```

### Platform DOME

DOMEs contain **explicit, complete** platform_config sections - no hidden translation logic.

```yaml
# configs/domes/craft_dome.yaml
dome_type: "platform_overlay"
platform: "craft"

platform_config:
  craft:
    enabled: true
    resolution_arcmin: 5
    climate_source: nasa_power
    soil_source: hwsd

    # Cultivar (DSSAT code)
    default_cultivar: "GH0010"

    # Agronomic parameters
    plant_population: 5.5           # plants/m²
    row_spacing_cm: 75
    planting_date_mmdd: "0604"      # June 4

    # Fertilizer
    default_fertilizer_n: 40.0      # kg N/ha
    fertilizer_material_code: "FE005"
    fertilizer_application_code: "AP002"
    fertilizer_app1_dap: 24
    fertilizer_app2_dap: 34

    # Data source paths (update for your system)
    hwsd_bil_path: "/path/to/HWSD2.bil"
    hwsd_mdb_path: "/path/to/HWSD2.mdb"
    spam_raster_path: "/path/to/spam2020_MAIZ.tif"

    # Schema generation
    schema_level: 2
    admin_level1_name: "Mali"
    admin_level2_name: "Koutiala"
    gadm_data_path: "/path/to/GADM/MLI/"
    gadm_country_iso3: "MLI"
```

**Design principle**: What you see in the DOME is exactly what the translator receives.
---

## CLI Commands

```bash
# Translation
prismpy translate --config X.yaml                    # Legacy format
prismpy translate --base X.yaml --dome Y.yaml        # DOME format

# ICASA validation
prismpy validate-icasa --config X.yaml

# AgMIP interoperability
prismpy export-ace --config X.yaml --output X.json   # Export to ACE JSON
prismpy import-ace --ace X.json --output X.yaml      # Import from ACE JSON

# Migration
prismpy migrate --legacy X.yaml --platform craft \
  --output-base base.yaml --output-dome dome.yaml

# Validation
prismpy validate --platform craft --output-dir output/

# Info
prismpy info --config X.yaml
```

---

## Output Packages

Each translation produces a self-documenting package:

```
output/{region}_{platform}/
├── README.md           # Platform-specific usage instructions
├── manifest.json       # File inventory with SHA256 checksums
├── provenance.json     # Data sources and processing decisions
├── schema/             # Grid/site definitions
├── soil/               # Soil profiles and mappings
├── weather/            # Climate data or download specs
├── crop_mask/          # Crop presence fractions
└── management/         # Planting, fertilizer, cultivar files
```

---

## Supported Platforms

All platforms tested with both legacy and DOME formats:

| Platform | Resolution | Climate | Soil | Use Case | DOME Tested |
|----------|------------|---------|------|----------|-------------|
| **CRAFT** | 5-arcmin grid | NASA POWER | HWSD v2.0 | Regional forecasting | ✓ 14 files |
| **PYTHIA** | Point sites | NASA POWER | eGHR | Distributed DSSAT | ✓ 226 files |
| **ACEA** | 5/30-arcmin | NASA POWER | HWSD v2.0 | AquaCrop ensemble | ✓ 77 files |
| **SARRA-Py** | ~4 km | TAMSAT + AgERA5 | iSDA | West Africa monitoring | ✓ 12 files |

---

## Data Sources

| Type | Source | Coverage | Platforms |
|------|--------|----------|-----------|
| **Climate** | NASA POWER | Global | CRAFT, PYTHIA, ACEA |
| **Climate** | TAMSAT, AgERA5 | Africa | SARRA-Py |
| **Soil** | HWSD v2.0 | Global | CRAFT, ACEA |
| **Soil** | eGHR | Global | PYTHIA |
| **Soil** | iSDA | Africa | SARRA-Py |
| **Boundaries** | GADM v4.1 | Global | All |
| **Crop Area** | SPAM 2020 | Global | All |

---

## Architecture

### Translation Pipeline

```
RETRIEVE → HARMONIZE → TRANSLATE → VALIDATE → PACKAGE
    │           │           │           │          │
  Fetch      Align       Generate    Check      Assemble
  from       grids,      platform-   schema,    manifest,
  sources    fill gaps   specific    ranges     provenance
```

### DOME Architecture

```
┌─────────────────────────────────────────────────┐
│            Base Config (ICASA-compliant)         │
│  - Agronomic parameters (phenology, physiology)  │
│  - Human-readable aliases (emergence_gdd = P1)   │
│  - Shareable across platforms, exportable to ACE │
└─────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│          Platform DOME (Explicit Config)         │
│  - Complete platform_config section              │
│  - DSSAT codes, file paths, schema settings      │
│  - What you see = what translator receives       │
└─────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│         Merged Config → Platform Translator      │
│  - Base agronomic data + DOME platform settings  │
│  - No hidden translation logic                   │
└─────────────────────────────────────────────────┘
```

### Module Structure

```
prismpy/
├── config/             # Configuration loading and DOME merging
├── standards/          # ICASA mapping, validation, ACE conversion
├── models/             # Unified data models (grid, climate, soil)
├── sources/            # Data retrieval (HWSD, SPAM, NASA POWER)
├── translators/        # Platform-specific translators
├── packaging/          # Manifest, provenance, README generation
├── configs/            # Example base configs and DOMEs
│   ├── base/
│   └── domes/
└── examples/           # Legacy single-file configs
```

---

## ICASA/AgMIP Interoperability

The framework supports bidirectional naming:

| ICASA Code | Human-Readable | Description |
|------------|----------------|-------------|
| `P1` | `emergence_gdd` | GDD from planting to emergence |
| `P5` | `grain_filling_gdd` | GDD for grain filling |
| `TB` | `base_temperature` | Base temperature (°C) |
| `HI` | `harvest_index` | Harvest index (fraction) |
| `PPOP` | `planting_density` | Plant population |
| `FEAMN` | `fertilizer_n_total` | Total N applied (kg/ha) |

Export to AgMIP ACE JSON format for data exchange:

```bash
prismpy export-ace --config base.yaml --output experiment.json
```

---

## References

- Tonle, F.B.N., Segnon, A.C., Gouroubera, M.W., Zougmore, R.B. 2026. PRISM: A methodological framework for reproducible data-to-model translation in spatial crop modeling. Computers and Electronics in Agriculture 254: 112223. DOI: 10.1016/j.compag.2026.112223
- Porter, C. H., et al. (2014). Harmonization and translation of crop modeling data to ensure interoperability. *Environmental Modelling & Software*, 62, 495-508.
- White, J. W., et al. (2013). Integrated description of agricultural field experiments and production: The ICASA Version 2.0 data standards. *Computers and Electronics in Agriculture*, 96, 1-12.
- ICASA Dictionary: https://github.com/agmip/ICASA-Dictionary

---

## License

MIT License

---

## Citation

Tonle, F.B.N., Segnon, A.C., Gouroubera, M.W., Zougmore, R.B. 2026. PRISM: A methodological framework for reproducible data-to-model translation in spatial crop modeling. Computers and Electronics in Agriculture 254: 112223. DOI: 10.1016/j.compag.2026.112223
