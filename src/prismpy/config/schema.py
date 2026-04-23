"""
Configuration schema for prismpy using Pydantic.

This module defines the complete configuration structure for the
data-to-model translation framework, including region, crop, temporal,
and platform-specific settings.
"""

import re
import unicodedata
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union


_DEFAULT_IGNORABLE_CODE_POINTS = frozenset({
    # Derived from Unicode's Default_Ignorable_Code_Point property.
    # Covers characters that render invisibly but are not category
    # Cc/Cf (so `.isprintable()` returns True). Most Cf characters
    # are handled by the category check below; this set is the
    # Other_Default_Ignorable_Code_Point tail the category check
    # misses.
    0x00AD,                                  # SOFT HYPHEN
    0x034F,                                  # COMBINING GRAPHEME JOINER
    0x061C,                                  # ARABIC LETTER MARK
    0x115F, 0x1160,                          # HANGUL JAMO FILLERS (Lo)
    0x17B4, 0x17B5,                          # KHMER VOWEL INHERENT AQ/AA
    0x16FE4,                                 # KHITAN SMALL SCRIPT FILLER (Mn)
    0x3164,                                  # HANGUL FILLER (Lo)
    0xFFA0,                                  # HALFWIDTH HANGUL FILLER (Lo)
})

# Letter-Other (Lo) code points that render visually blank but
# are NOT in Unicode's Default_Ignorable_Code_Point property —
# Unicode classifies these as regular letters in their script
# despite rendering as whitespace. The whitelist's `Lo` category
# would otherwise accept them; this overlay rejects them
# explicitly. Hand-maintained list (Unicode introduces new
# script blocks periodically; this set captures the known
# invisible Lo codepoints as of Unicode 17).
_INVISIBLE_LO_CODEPOINTS = frozenset({
    0x13441,  # EGYPTIAN HIEROGLYPH FULL BLANK (Unicode 5.2)
    0x13442,  # EGYPTIAN HIEROGLYPH HALF BLANK (Unicode 5.2)
})
_DEFAULT_IGNORABLE_RANGES = (
    (0x180B, 0x180F),      # MONGOLIAN FREE VARIATION SELECTORS
    (0x200B, 0x200F),      # ZERO WIDTH / LEFT-TO-RIGHT MARK family
    (0x202A, 0x202E),      # DIRECTIONAL FORMATTING
    (0x2060, 0x206F),      # WORD JOINER / INVISIBLE OPERATORS family
    (0xFE00, 0xFE0F),      # VARIATION SELECTORS 1-16
    (0xFFF0, 0xFFFB),      # INTERLINEAR ANNOTATION ANCHORS family
    (0x1BCA0, 0x1BCA3),    # SHORTHAND FORMAT CONTROLS
    (0x1D173, 0x1D17A),    # MUSICAL SYMBOL BEGIN/END family
    (0xE0000, 0xE0FFF),    # TAG characters + SUPP. VARIATION SELECTORS
)


def _is_default_ignorable(c: str) -> bool:
    code = ord(c)
    if code in _DEFAULT_IGNORABLE_CODE_POINTS:
        return True
    for lo, hi in _DEFAULT_IGNORABLE_RANGES:
        if lo <= code <= hi:
            return True
    return False


# Unicode general categories accepted in identifier strings
# (region name, country, gadm_filter_value, gadm_filter_field,
# shapefile_path). Includes letters (L*), numbers (N*), and
# combining marks (Mn, Mc) so NFD-decomposed accented Latin
# names like 'Ségou' (e + U+0301) still validate regardless of
# how the caller normalized them.
_IDENTIFIER_CATEGORIES = frozenset({
    "Lu", "Ll", "Lt", "Lm", "Lo",    # letters
    "Nd", "Nl", "No",                 # numbers
    "Mn", "Mc",                       # combining marks
})

# Printable punctuation + separators that appear legitimately in
# region / country / filter / path identifier strings.
_IDENTIFIER_PUNCT = frozenset(" -_.',/()&")


def _is_identifier_char(c: str) -> bool:
    """True if `c` is acceptable in an identifier string.

    Positive-acceptance check: ONLY chars in the identifier-category
    allowlist or the punctuation allowlist pass. Two rejection
    overlays catch invisibles whose Unicode general category would
    otherwise let them through:

    - `_is_default_ignorable(c)` — Hangul Jamo fillers, variation
      selectors, combining grapheme joiner, Khmer inherent vowels,
      Khitan Small Script Filler, SOFT HYPHEN, etc.
    - `_INVISIBLE_LO_CODEPOINTS` — Letter-Other codepoints that
      render visually blank despite their script-letter
      classification (Egyptian Hieroglyph Full/Half Blank). These
      aren't in Unicode's Default_Ignorable property so the
      previous overlay misses them.
    """
    if _is_default_ignorable(c):
        return False
    if ord(c) in _INVISIBLE_LO_CODEPOINTS:
        return False
    if c in _IDENTIFIER_PUNCT:
        return True
    return unicodedata.category(c) in _IDENTIFIER_CATEGORIES


def _contains_invisible_char(s: str) -> bool:
    """True if `s` contains any character that isn't a valid
    identifier char.

    Positive-whitelist shape (V2-22b/P.2 AC-AUDIT-16): earlier
    rounds (R13/R14/R15) each surfaced a new invisible Unicode
    class the blocklist didn't cover — ASCII controls → Cf format
    characters → Other_Default_Ignorable tail (Hangul fillers,
    variation selectors, Khmer inherent vowels). The pattern was
    "blocklist can't enumerate all bad inputs." AC-AUDIT-16
    inverts the check: only Unicode categories defined as
    identifier-acceptable pass, plus a narrow
    `Default_Ignorable_Code_Point` overlay for invisible chars
    whose category overlaps the allowlist (Lo/Mn/Mc).

    The function name stays `_contains_invisible_char` so callers
    don't change; semantically it's now "contains any disallowed
    char" but the original name captures the primary risk we're
    guarding against.
    """
    return any(not _is_identifier_char(c) for c in s)

from pydantic import BaseModel, Field, field_validator, model_validator


class Platform(str, Enum):
    """Supported spatial crop modeling platforms."""
    SARRA_PY = "sarra_py"
    CRAFT = "craft"
    PYTHIA = "pythia"
    ACEA = "acea"


class ClimateSource(str, Enum):
    """Available climate data sources."""
    NASA_POWER = "nasa_power"
    TAMSAT = "tamsat"
    AGERA5 = "agera5"


class SoilSource(str, Enum):
    """Available soil data sources."""
    ISDA = "isda"
    HWSD = "hwsd"
    EGHR = "eghr"
    SOILGRIDS = "soilgrids"


class BoundarySource(str, Enum):
    """Sources for region boundary definition."""
    GADM = "gadm"
    SHAPEFILE = "shapefile"
    MANUAL = "manual"


# =============================================================================
# Region Configuration
# =============================================================================

class ManualBoundsConfig(BaseModel):
    """Manual bounding box specification in standard GIS format."""
    minx: float = Field(..., description="Minimum longitude (western edge)")
    miny: float = Field(..., description="Minimum latitude (southern edge)")
    maxx: float = Field(..., description="Maximum longitude (eastern edge)")
    maxy: float = Field(..., description="Maximum latitude (northern edge)")

    @model_validator(mode="after")
    def validate_bounds(self) -> "ManualBoundsConfig":
        if self.minx >= self.maxx:
            raise ValueError("minx must be less than maxx")
        if self.miny >= self.maxy:
            raise ValueError("miny must be less than maxy")
        if not (-180 <= self.minx <= 180 and -180 <= self.maxx <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (-90 <= self.miny <= 90 and -90 <= self.maxy <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        return self


class BoundaryConfig(BaseModel):
    """Configuration for region boundary extraction."""
    source: BoundarySource = Field(
        default=BoundarySource.GADM,
        description="Source for boundary data"
    )
    gadm_level: Optional[int] = Field(
        default=2,
        ge=0,
        le=5,
        description="GADM administrative level (0=country, 1=region, 2=district)"
    )
    gadm_filter_field: Optional[str] = Field(
        default="NAME_2",
        description="Field name to filter by (e.g., NAME_1, NAME_2)"
    )
    gadm_filter_value: Optional[str] = Field(
        default=None,
        description="Value to filter for (e.g., 'Koutiala')"
    )
    shapefile_path: Optional[Path] = Field(
        default=None,
        description="Path to custom shapefile if source is 'shapefile'"
    )
    manual_bounds: Optional[ManualBoundsConfig] = Field(
        default=None,
        description="Manual bounding box if source is 'manual'"
    )

    @field_validator(
        "gadm_filter_value", "gadm_filter_field", mode="before",
    )
    @classmethod
    def _strip_and_validate_gadm_identifier(cls, value):
        """Universal-invariant rules for GADM identifier strings —
        `gadm_filter_value` and `gadm_filter_field` both flow
        through `if filter_field and filter_value:` guards in
        `GADMSource._extract_bounds`, where a whitespace-only or
        control-char-bearing string is truthy but semantically
        broken. Same rules as `RegionConfig.name` / `country`
        (V2-22b/P.2 AC-AUDIT-10/11).

        `None` stays `None` — both fields are optional at schema
        level, and `validate_source_requirements` handles the
        source-specific None-required checks. Non-None values
        strip surrounding whitespace, reject control characters,
        and reject values that normalize to an empty identifier.
        """
        if value is None:
            return value
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if _contains_invisible_char(stripped):
            raise ValueError(
                f"GADM identifier {value!r} contains disallowed "
                "characters; identifier strings accept letters, "
                "numbers, and common punctuation (space, hyphen, "
                "underscore, dot, apostrophe, slash, parentheses, "
                "comma, ampersand) — non-printable, invisible, or "
                "other symbol characters are rejected"
            )
        from prismpy.utils.sanitization import normalize_region_name
        if not normalize_region_name(stripped):
            raise ValueError(
                f"GADM identifier {value!r} normalizes to an "
                "empty identifier; at least one alphanumeric "
                "character required"
            )
        return stripped

    @field_validator("shapefile_path", mode="before")
    @classmethod
    def _validate_shapefile_path(cls, value):
        """Reject empty, whitespace-only, or empty-path-sentinel
        inputs before or after Pydantic's `Path` coercion. The
        coercion quietly turns `''` and other empty-equivalents
        into `Path('.')` (the current working directory), which
        then passes `Path.exists()` checks downstream and is
        handed to `geopandas.read_file`, surfacing only as a
        runtime DataSourceError instead of a validation error.

        `None` stays `None` (field is optional; the model
        validator requires the path only when
        `source == shapefile`). Both raw-string and PathLike
        inputs run through the same strip + printable-char +
        non-sentinel checks — AC-AUDIT-14 closed the typed
        `Path('')` / `Path('.')` hole codex R13 identified."""
        if value is None:
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if _contains_invisible_char(stripped):
                raise ValueError(
                    f"shapefile_path {value!r} contains disallowed "
                    "characters; path strings accept the same "
                    "letters, numbers, and common punctuation as "
                    "other identifier fields — non-printable, "
                    "invisible, or other symbol characters are "
                    "rejected"
                )
            if not stripped:
                raise ValueError(
                    f"shapefile_path {value!r} is empty or "
                    "whitespace-only; provide a real path or "
                    "omit the field entirely"
                )
            value = stripped
        # PathLike normalization: coerce to Path and reject the
        # empty-sentinel (`Path('')` == `Path('.')`) that would
        # otherwise pass Pydantic's type check and resolve to
        # the current working directory at retrieve time.
        try:
            candidate = Path(value)
        except (TypeError, ValueError):
            return value  # let Pydantic's type validator reject
        if candidate == Path('.') or str(candidate).strip() in ('', '.'):
            raise ValueError(
                f"shapefile_path {value!r} resolves to the current "
                "working directory (`Path('.')`); provide an "
                "explicit path to a shapefile"
            )
        if _contains_invisible_char(str(candidate)):
            raise ValueError(
                f"shapefile_path {value!r} contains non-printable "
                "or invisible characters; only visible, printable "
                "characters are allowed"
            )
        return candidate

    @model_validator(mode="after")
    def validate_source_requirements(self) -> "BoundaryConfig":
        if self.source == BoundarySource.GADM:
            if self.gadm_filter_value is None:
                raise ValueError("gadm_filter_value required when source is 'gadm'")
        elif self.source == BoundarySource.SHAPEFILE:
            if self.shapefile_path is None:
                raise ValueError("shapefile_path required when source is 'shapefile'")
        elif self.source == BoundarySource.MANUAL:
            if self.manual_bounds is None:
                raise ValueError("manual_bounds required when source is 'manual'")
        return self


class RegionConfig(BaseModel):
    """Configuration for the study region."""
    name: str = Field(..., min_length=1, description="Region name (e.g., 'Koutiala')")
    country: str = Field(..., min_length=1, description="Country name (e.g., 'Mali')")
    country_iso3: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="ISO 3166-1 alpha-3 country code (e.g., 'MLI')"
    )
    boundary: BoundaryConfig = Field(
        default_factory=BoundaryConfig,
        description="Boundary extraction configuration"
    )

    @field_validator("country_iso3")
    @classmethod
    def validate_iso3(cls, v: str) -> str:
        """ISO 3166-1 alpha-3: exactly three uppercase ASCII letters.

        Length-only validation (`min_length=3, max_length=3`) accepts
        non-letter codes like `'ML2'` or `'ML!'`, which would flow
        through every downstream consumer that treats the field as a
        short identifier — file paths, logs, provenance records, and
        the CRAFT country-code field. Reject anything outside the
        alpha pattern so an invalid code fails at validate time."""
        stripped = v.strip().upper() if isinstance(v, str) else v
        if not isinstance(stripped, str) or not re.match(r'^[A-Z]{3}$', stripped):
            raise ValueError(
                f"country_iso3 must be exactly three alphabetic "
                f"characters (ISO 3166-1 alpha-3); got {v!r}"
            )
        return stripped

    @field_validator("name", "country", mode="before")
    @classmethod
    def _strip_and_require_normalizable(cls, value):
        """Enforce the universal invariants downstream consumers
        rely on:

        1. `min_length=1` alone accepts whitespace-only (`'   '`),
           punctuation-only (`'!!!'`), or underscore-only (`'___'`)
           inputs that validate upstream but collapse to an empty
           key when passed through `normalize_region_name` /
           `sanitize_admin_name`. Two different malformed inputs
           silently alias onto the same cache path, lock file, or
           filename prefix. REJECT those here.

        2. Internal ASCII control characters (`\\x00-\\x1f`, `\\x7f`)
           are never legitimate in a region name or country name;
           they survive `normalize_region_name` by becoming a `_`
           but corrupt any downstream consumer that writes the raw
           value into a structured format (logs, JSON reports,
           fixed-width records). REJECT them at schema time —
           universal invariant, applies regardless of which
           downstream consumer ultimately reads the value.

        `mode='before'` so the stripped value is stored (no UX
        regression on `'  Koutiala  '`). Import
        `normalize_region_name` lazily to avoid any future
        sanitization → schema cycle."""
        if not isinstance(value, str):
            # Let Pydantic's type validation handle non-strings.
            return value
        stripped = value.strip()
        if _contains_invisible_char(stripped):
            raise ValueError(
                f"{value!r} contains disallowed characters; "
                "identifier strings accept letters, numbers, and "
                "common punctuation (space, hyphen, underscore, "
                "dot, apostrophe, slash, parentheses, comma, "
                "ampersand) — non-printable, invisible, or other "
                "symbol characters are rejected"
            )
        from prismpy.utils.sanitization import normalize_region_name
        if not normalize_region_name(stripped):
            raise ValueError(
                f"{value!r} is not a Latin-script-compatible "
                "identifier; region name / country must contain "
                "at least one alphanumeric character (letters or "
                "digits, possibly with accents or Latin-Extended "
                "diacritics). Non-Latin scripts (Korean, Arabic, "
                "Cyrillic, Devanagari, Chinese, etc.) require "
                "upstream transliteration before reaching "
                "RegionConfig — see PRISMWEB identifier policy "
                "in prismpy/config/schema.py"
            )
        return stripped


# =============================================================================
# Crop Configuration
# =============================================================================

class CropCalendarConfig(BaseModel):
    """Crop planting and maturity calendar."""
    planting_doy: int = Field(
        ...,
        ge=1,
        le=366,
        description="Planting day of year (1-366)"
    )
    maturity_doy: int = Field(
        ...,
        ge=1,
        le=366,
        description="Maturity day of year (1-366)"
    )
    source: str = Field(
        default="literature",
        description="Source of calendar data (e.g., 'literature', 'SAGE', 'survey')"
    )
    reference: Optional[str] = Field(
        default=None,
        description="Citation or reference for the calendar data"
    )

    @property
    def growing_season_days(self) -> int:
        """Calculate growing season length in days."""
        if self.maturity_doy >= self.planting_doy:
            return self.maturity_doy - self.planting_doy
        else:
            # Growing season spans year boundary
            return (365 - self.planting_doy) + self.maturity_doy


# =============================================================================
# Generic Crop Parameters (Platform Agnostic)
# =============================================================================

class PhenologyConfig(BaseModel):
    """Generic phenology in thermal time (GDD) - platform agnostic.

    These parameters describe crop development stages in growing degree days,
    which can be mapped to any crop model's specific format.
    """
    emergence_gdd: float = Field(
        default=90.0,
        ge=0,
        description="GDD from sowing to emergence"
    )
    vegetative_phase_gdd: float = Field(
        default=500.0,
        ge=0,
        description="GDD for vegetative phase (emergence to flowering)"
    )
    reproductive_phase_gdd: float = Field(
        default=400.0,
        ge=0,
        description="GDD for reproductive phase (flowering to grain fill)"
    )
    grain_filling_gdd: float = Field(
        default=700.0,
        ge=0,
        description="GDD for grain filling (to physiological maturity)"
    )
    photoperiod_sensitivity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Photoperiod sensitivity (0=insensitive, 1=very sensitive)"
    )


class PhysiologyConfig(BaseModel):
    """Generic crop physiology parameters - platform agnostic.

    These parameters describe fundamental crop physiological responses
    that can be mapped to any crop model's specific format.
    """
    base_temperature: float = Field(
        default=8.0,
        description="Base temperature for development (°C)"
    )
    optimal_temperature: float = Field(
        default=26.0,
        description="Optimal temperature for growth (°C)"
    )
    max_temperature: float = Field(
        default=44.0,
        description="Maximum/lethal temperature (°C)"
    )
    radiation_use_efficiency: float = Field(
        default=5.8,
        gt=0,
        description="Radiation use efficiency (g DM/MJ)"
    )
    harvest_index: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Fraction of biomass allocated to grain"
    )


class ManagementConfig(BaseModel):
    """Generic crop management parameters - platform agnostic.

    These parameters describe agronomic practices that can be mapped
    to any crop model's specific format.
    """
    planting_density: float = Field(
        ...,
        gt=0,
        description="Plant density (plants/ha)"
    )
    row_spacing_cm: float = Field(
        default=70.0,
        gt=0,
        description="Row spacing in centimeters"
    )
    sowing_mode: Literal["opportunistic", "fixed_date"] = Field(
        default="opportunistic",
        description="Sowing mode: opportunistic (wait for rain) or fixed_date"
    )
    sowing_threshold_mm: float = Field(
        default=10.0,
        ge=0,
        description="Soil moisture threshold for opportunistic sowing (mm)"
    )
    irrigation: bool = Field(
        default=False,
        description="Enable irrigation"
    )
    irrigation_target: float = Field(
        default=0.0,
        ge=0,
        description="Target soil moisture for auto-irrigation (mm)"
    )

    # Fertilizer settings
    fertilizer_n_total: float = Field(
        default=60.0,
        ge=0,
        description="Total nitrogen fertilizer application (kg N/ha)"
    )
    fertilizer_n_splits: List[int] = Field(
        default=[0, 30],
        description="Days after planting for each N application"
    )
    fertilizer_n_fractions: List[float] = Field(
        default=[0.5, 0.5],
        description="Fraction of total N applied at each split (must sum to 1.0)"
    )

    # Residue management
    residue_amount_kg_ha: float = Field(
        default=250.0,
        ge=0,
        description="Crop residue incorporation amount (kg/ha dry weight). Default 250 for typical residue."
    )

    # Cultivar configuration
    default_cultivar: Optional[str] = Field(
        default=None,
        description="Default DSSAT cultivar code (e.g., 'GH0010' for OBATANPA maize). "
                    "Can be overridden per zone via management_zones."
    )


# =============================================================================
# Management Zones (Framework-Agnostic)
# =============================================================================
# Zones allow spatial variability in management parameters.
# Works for ALL platforms (CRAFT, Pythia, ACEA, etc.)
# =============================================================================

class BoundingBoxFilter(BaseModel):
    """Filter cells by geographic bounding box.

    Cells whose centroid falls within the box match this filter.
    """
    type: Literal["bounding_box"] = Field(
        default="bounding_box",
        description="Filter type identifier"
    )
    lat_min: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Southern boundary (decimal degrees)"
    )
    lat_max: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Northern boundary (decimal degrees)"
    )
    lon_min: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Western boundary (decimal degrees)"
    )
    lon_max: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Eastern boundary (decimal degrees)"
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBoxFilter":
        if self.lat_min >= self.lat_max:
            raise ValueError("lat_min must be less than lat_max")
        if self.lon_min >= self.lon_max:
            raise ValueError("lon_min must be less than lon_max")
        return self


class AdminLevelFilter(BaseModel):
    """Filter cells by administrative boundary names.

    Cells matching any of the specified admin names are included.
    Requires schema to have Level{N}Name columns (e.g., Level2Name).
    """
    type: Literal["admin_level"] = Field(
        default="admin_level",
        description="Filter type identifier"
    )
    admin_level: int = Field(
        ...,
        ge=1,
        le=3,
        description="Admin level (1=country, 2=state/region, 3=district)"
    )
    names: List[str] = Field(
        ...,
        min_length=1,
        description="List of admin unit names to include"
    )


# Union type for zone filtering
ZoneFilter = Union[BoundingBoxFilter, AdminLevelFilter]


class ManagementOverrides(BaseModel):
    """Management parameter overrides for a zone.

    Only non-None values override defaults from ManagementConfig.
    All fields are optional - specify only what differs from defaults.
    """
    fertilizer_n_total: Optional[float] = Field(
        default=None,
        ge=0,
        description="Override total N fertilizer (kg N/ha)"
    )
    fertilizer_n_splits: Optional[List[int]] = Field(
        default=None,
        description="Override DAP for each application"
    )
    fertilizer_n_fractions: Optional[List[float]] = Field(
        default=None,
        description="Override fraction at each split (must sum to 1.0)"
    )
    planting_density: Optional[float] = Field(
        default=None,
        gt=0,
        description="Override plant density (plants/ha)"
    )
    row_spacing_cm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Override row spacing (cm)"
    )
    cultivar: Optional[str] = Field(
        default=None,
        description="Override DSSAT cultivar code (e.g., 'GH0010', 'IB0001')"
    )
    planting_date_mmdd: Optional[str] = Field(
        default=None,
        pattern=r"^(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$",
        description="Override planting date in MMDD format (e.g., '0604' for June 4)"
    )


class ManagementZone(BaseModel):
    """A management zone with spatial filter and parameter overrides.

    Zones allow spatial variability in management parameters.
    Cells matching the filter get the specified overrides applied.
    First matching zone wins if zones overlap.
    """
    zone_name: str = Field(
        ...,
        min_length=1,
        description="Descriptive name (e.g., 'South_HighRainfall')"
    )
    filter: ZoneFilter = Field(
        ...,
        description="Spatial filter (bounding_box or admin_level)"
    )
    overrides: ManagementOverrides = Field(
        default_factory=ManagementOverrides,
        description="Management parameters to override"
    )


class GenericSoilConfig(BaseModel):
    """Generic soil configuration - platform agnostic.

    These parameters describe soil properties that can be mapped
    to any crop model's specific format.
    """
    source: SoilSource = Field(
        default=SoilSource.ISDA,
        description="Soil data source"
    )
    default_depth_m: float = Field(
        default=1.5,
        gt=0,
        description="Default soil profile depth (meters)"
    )


class CropConfig(BaseModel):
    """Configuration for the crop to simulate."""
    name: str = Field(..., min_length=1, description="Crop name (e.g., 'Maize')")
    name_short: str = Field(
        ...,
        min_length=2,
        max_length=4,
        description="Short crop code (e.g., 'mai', 'mil', 'sor')"
    )
    variety: str = Field(
        default="default",
        description="Variety or cultivar identifier"
    )
    calendar: CropCalendarConfig = Field(
        ...,
        description="Planting and maturity calendar"
    )

    # Generic crop parameters (platform agnostic)
    phenology: Optional[PhenologyConfig] = Field(
        default=None,
        description="Generic phenology in thermal time (GDD)"
    )
    physiology: Optional[PhysiologyConfig] = Field(
        default=None,
        description="Generic crop physiology parameters"
    )


# =============================================================================
# Temporal Configuration
# =============================================================================

class TemporalConfig(BaseModel):
    """Temporal settings for the simulation."""
    start_year: int = Field(
        ...,
        ge=1980,
        le=2030,
        description="First year of simulation"
    )
    end_year: int = Field(
        ...,
        ge=1980,
        le=2030,
        description="Last year of simulation"
    )
    spinup_years: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Years before start_year for model spinup"
    )

    @model_validator(mode="after")
    def validate_years(self) -> "TemporalConfig":
        if self.end_year < self.start_year:
            raise ValueError("end_year must be >= start_year")
        return self

    @property
    def climate_start_year(self) -> int:
        """First year of required climate data (including spinup)."""
        return self.start_year - self.spinup_years

    @property
    def simulation_years(self) -> int:
        """Number of simulation years."""
        return self.end_year - self.start_year + 1

    def get_climate_end_date(self, crop_calendar: Optional["CropCalendarConfig"] = None) -> date:
        """Climate end date, extended to cover cross-year growing seasons.

        For same-year seasons (maturity_doy >= planting_doy), returns Dec 31
        of end_year. For cross-year seasons (maturity_doy < planting_doy,
        e.g., Nov planting → Mar harvest), returns the maturity date in
        end_year + 1 so climate data covers the final harvest.
        """
        if (
            crop_calendar is not None
            and crop_calendar.maturity_doy < crop_calendar.planting_doy
        ):
            return date(self.end_year + 1, 1, 1) + timedelta(days=crop_calendar.maturity_doy - 1)
        return date(self.end_year, 12, 31)


# =============================================================================
# Platform-Specific Configuration
# =============================================================================

class SarraPyClimateConfig(BaseModel):
    """Climate source configuration for SARRA-Py (uses TAMSAT + AgERA5)."""
    rainfall: ClimateSource = Field(
        default=ClimateSource.TAMSAT,
        description="Rainfall data source"
    )
    temperature: ClimateSource = Field(
        default=ClimateSource.AGERA5,
        description="Temperature data source"
    )
    radiation: ClimateSource = Field(
        default=ClimateSource.AGERA5,
        description="Solar radiation data source"
    )


class SarraPyConfig(BaseModel):
    """Platform-specific configuration for SARRA-Py."""
    enabled: bool = Field(default=True, description="Enable SARRA-Py output")
    resolution: float = Field(
        default=0.0375,
        gt=0,
        description="Spatial resolution in degrees (~4km for SARRA-Py)"
    )
    climate_sources: SarraPyClimateConfig = Field(
        default_factory=SarraPyClimateConfig,
        description="Climate data sources"
    )
    soil_source: SoilSource = Field(
        default=SoilSource.ISDA,
        description="Soil data source"
    )

    # Template configuration
    templates_dir: Optional[Path] = Field(
        default=None,
        description="Path to parameter templates directory containing variety/, itk/, soil/ subdirs"
    )
    variety_template: str = Field(
        default="maize_west_africa",
        description="Variety parameter template name (without .yaml extension)"
    )
    variety_template_file: Optional[Path] = Field(
        default=None,
        description="Explicit path to variety parameter YAML file (overrides templates_dir)"
    )
    itk_template: str = Field(
        default="rainfed_opportunistic",
        description="ITK (cropping practice) template name (without .yaml extension)"
    )
    itk_template_file: Optional[Path] = Field(
        default=None,
        description="Explicit path to ITK parameter YAML file (overrides templates_dir)"
    )
    soil_template: str = Field(
        default="default_soil",
        description="Soil parameter template name (without .yaml extension)"
    )
    soil_template_file: Optional[Path] = Field(
        default=None,
        description="Explicit path to soil parameter YAML file (overrides templates_dir)"
    )

    # Opportunistic sowing configuration
    sowing_search_month: int = Field(
        default=5,
        ge=1,
        le=12,
        description="Month to start searching for sowing conditions (1-12)"
    )
    sowing_search_day: int = Field(
        default=1,
        ge=1,
        le=31,
        description="Day to start searching for sowing conditions (1-31)"
    )

    # Soil defaults
    default_soil_depth_m: float = Field(
        default=1.5,
        gt=0,
        description="Default soil profile depth in meters"
    )


class CraftConfig(BaseModel):
    """Platform-specific configuration for CRAFT.

    All parameters are configurable to support any region/crop combination.
    No hardcoded values - everything comes from user config.
    """
    enabled: bool = Field(default=True, description="Enable CRAFT output")
    resolution_arcmin: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Spatial resolution in arc-minutes"
    )
    climate_source: ClimateSource = Field(
        default=ClimateSource.NASA_POWER,
        description="Climate data source"
    )
    soil_source: SoilSource = Field(
        default=SoilSource.HWSD,
        description="Soil data source"
    )

    # Soil mask generation
    include_soil_mask: bool = Field(
        default=True,
        description="Generate soil mask file (CellID -> SoilProfile mapping)"
    )

    # =========================================================================
    # CULTIVAR CONFIGURATION
    # =========================================================================
    # DSSAT cultivar codes vary by crop. Common examples:
    #   Maize: GH0010 (OBATANPA), IB0001 (PIO3382)
    #   Wheat: IB0001 (NEWTON), IB0002 (KATEPWA)
    #   Sorghum: IB0001 (ATX623 x MARTINZ), IB0002 (DK28E)
    #   Rice: IB0001 (IR 64), IB0002 (IR 72)
    #   Millet: IB0001 (HHB67), IB0002 (WC75)
    # =========================================================================
    default_cultivar: Optional[str] = Field(
        default=None,
        description="DSSAT cultivar code. If not set, uses crop-specific default."
    )

    # =========================================================================
    # AGRONOMIC PARAMETERS (for planting data)
    # =========================================================================
    # These should match crop management practices for the region.
    # If not specified, uses values from ManagementConfig or crop-specific defaults.
    # =========================================================================
    plant_population: Optional[float] = Field(
        default=None,
        ge=0,
        description="Plant population (plants/m²). If not set, derived from planting_density."
    )
    row_spacing_cm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Row spacing in cm. If not set, uses ManagementConfig.row_spacing_cm."
    )
    planting_depth_cm: float = Field(
        default=5.0,
        gt=0,
        description="Planting depth in cm"
    )
    planting_method: str = Field(
        default="S",
        description="Planting method: S=seed, T=transplant"
    )
    plant_distribution: str = Field(
        default="R",
        description="Plant distribution: R=rows, B=broadcast"
    )
    planting_row_direction: int = Field(
        default=0,
        ge=0,
        le=359,
        description="Planting row direction (degrees from North, 0-359). 0=N-S, 90=E-W"
    )

    # =========================================================================
    # FERTILIZER CONFIGURATION
    # =========================================================================
    # Supports split application (2 applications per cell).
    # Application 1: Basal (at/near planting)
    # Application 2: Top-dressing (later in season)
    # =========================================================================
    default_fertilizer_n: float = Field(
        default=40.0,
        ge=0,
        description="Total nitrogen fertilizer (kg N/ha, split across applications)"
    )
    default_fertilizer_p: float = Field(
        default=10.0,
        ge=0,
        description="Phosphorus fertilizer (kg P2O5/ha, applied with first application)"
    )
    default_fertilizer_k: float = Field(
        default=10.0,
        ge=0,
        description="Potassium fertilizer (kg K2O/ha, applied with first application)"
    )
    fertilizer_split_ratio: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Fraction of N applied at first application (rest at second)"
    )
    fertilizer_app1_dap: int = Field(
        default=24,
        ge=0,
        description="Days after planting for first fertilizer application"
    )
    fertilizer_app2_dap: int = Field(
        default=34,
        ge=0,
        description="Days after planting for second fertilizer application"
    )
    fertilizer_material_code: str = Field(
        default="FE005",
        description="DSSAT fertilizer material code (FE005=Urea/NPK)"
    )
    fertilizer_application_code: str = Field(
        default="AP002",
        description="DSSAT application method (AP002=Broadcast incorporated)"
    )
    fertilizer_depth_cm: int = Field(
        default=5,
        ge=0,
        description="Fertilizer application depth in cm"
    )

    # =========================================================================
    # ORGANIC FERTILIZER CONFIGURATION
    # =========================================================================
    # DSSAT residue codes: RE001=crop residue, RE002=green manure, RE003=manure
    # =========================================================================
    organic_fertilizer_enabled: bool = Field(
        default=False,
        description="Enable organic fertilizer (crop residue) application"
    )
    organic_residue_code: str = Field(
        default="RE001",
        description="DSSAT residue type code (RE001=crop residue, RE002=green manure, RE003=manure)"
    )
    organic_amount: float = Field(
        default=0,
        ge=0,
        description="Organic matter amount (kg dry weight/ha)"
    )
    organic_dap: int = Field(
        default=0,
        ge=0,
        description="Days after planting for residue incorporation"
    )

    # =========================================================================
    # PLANTING DATE CONFIGURATION
    # =========================================================================
    planting_date_mmdd: Optional[str] = Field(
        default=None,
        description="Override planting date in MMDD format (e.g., '0604' for June 4)"
    )

    # =========================================================================
    # HWSD PATHS (for self-contained packages)
    # =========================================================================
    hwsd_bil_path: Optional[Path] = Field(
        default=None,
        description="Path to HWSD2.bil raster file"
    )
    hwsd_mdb_path: Optional[Path] = Field(
        default=None,
        description="Path to HWSD2.mdb database file"
    )

    # =========================================================================
    # CRAFT SCHEMA GENERATION
    # =========================================================================
    # CRAFT requires schemas in hierarchical admin levels:
    #   Level1 = Country (e.g., Mali)
    #   Level2 = State/Region (e.g., Sikasso)
    #   Level3 = District (e.g., Koutiala)
    #
    # Output structure: CRAFT_Schema/Level{N}/Schema/5m_{AdminNames}.txt
    # File format: CELLID\tSHAREPERCENT (tab-separated)
    # =========================================================================
    schema_level: int = Field(
        default=2,
        ge=1,
        le=3,
        description="""DEPRECATED: Use craft_level + gadm_level instead.
        Kept for backward compatibility. If craft_level is not set, schema_level
        is used as the CRAFT output folder level AND for GADM level derivation.
        New configs should set craft_level and gadm_level explicitly."""
    )
    craft_level: Optional[int] = Field(
        default=None,
        ge=1,
        le=3,
        description="""CRAFT output folder level (1, 2, or 3). This is PROJECT-RELATIVE:
        Level 1 = your top study area (e.g., Koutiala cercle)
        Level 2 = subdivisions within Level 1 (e.g., communes)
        Level 3 = sub-subdivisions within Level 2
        IMPORTANT: This is independent of GADM admin levels.
        A cercle at GADM Level 2 can be CRAFT Level 1 if it's your top study area.
        If not set, falls back to schema_level for backward compatibility."""
    )
    admin_level1_name: Optional[str] = Field(
        default=None,
        description="Level 1 name (country). Required for CRAFT schema naming. Derived from region.country if not set."
    )
    admin_level2_name: Optional[str] = Field(
        default=None,
        description="Level 2 name (state/region). Required if schema_level >= 2. Derived from region.name if not set."
    )
    admin_level3_name: Optional[str] = Field(
        default=None,
        description="Level 3 name (district). Required if schema_level = 3."
    )
    admin_shapefile_path: Optional[Path] = Field(
        default=None,
        description="Path to admin boundary shapefile for accurate SharePercent calculation. If not set, uses bounding box (100% share for all cells)."
    )
    generate_python_schema: bool = Field(
        default=True,
        description="Also generate Python_Schemas format (CellID, Lat, Lon, Elevation, Area, AdminNames) for internal use"
    )
    schema_decimal_places: int = Field(
        default=2,
        ge=0,
        le=6,
        description="Decimal places for SharePercent values"
    )

    # =========================================================================
    # GADM (Global Administrative Areas) CONFIGURATION
    # =========================================================================
    # GADM provides administrative boundary data for accurate SharePercent
    # and Area calculations. Required for proper CRAFT schema generation.
    #
    # GADM structure:
    #   Level 0 = Country boundary → CRAFT Level 1
    #   Level 1 = State/Province   → CRAFT Level 2
    #   Level 2 = District/County  → CRAFT Level 3
    #
    # File naming: gadm41_{ISO3}_{level}.shp (e.g., gadm41_MLI_1.shp)
    # =========================================================================
    gadm_data_path: Optional[Path] = Field(
        default=None,
        description="Path to directory containing GADM shapefiles (gadm41_{ISO3}_{level}.shp)"
    )
    gadm_country_iso3: Optional[str] = Field(
        default=None,
        description="ISO 3-letter country code for GADM lookup (e.g., 'MLI', 'NGA', 'KEN')"
    )
    gadm_admin_name: Optional[str] = Field(
        default=None,
        description="Admin region name to filter to (e.g., 'Koutiala', 'Kano'). If not set, uses region.name"
    )
    gadm_level: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description="GADM admin level for boundary polygon selection (0=country, 1=state, "
                    "2=district/cercle, 3=commune/ward, 4=village). Controls WHICH GADM boundary "
                    "to use, NOT which CRAFT folder to write to. If not set, defaults to "
                    "schema_level - 1. WARNING: Some admin names exist at multiple GADM levels "
                    "(e.g., 'Koutiala' is both a cercle at L2 and a commune at L3). Always verify "
                    "gadm_level matches the geographic scope you intend."
    )

    # =========================================================================
    # CROP MASK CONFIGURATION
    # =========================================================================
    # Crop mask indicates what fraction of each cell has the target crop.
    # Format: CellId\tPercent (0-1 range, 1.0 = 100%)
    # =========================================================================
    crop_mask_percent: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Default crop coverage percentage (0-1). 1.0 = 100% of cell has crop. For uniform mode."
    )

    # =========================================================================
    # CROP MASK - SPAM RASTER MODE (Optional)
    # =========================================================================
    # SPAM (Spatial Production Allocation Model) provides actual crop harvested
    # area data at ~10km resolution. When configured, crop mask percentages are
    # calculated from real data: Percent = SPAM_hectares / cell_area_ha
    #
    # Download from: https://www.mapspam.info/data/
    # File naming: spam2020_V2r0_global_{TECH}_{CROP}_{SYSTEM}.tif
    #   TECH: A (all technologies), H (high input), etc.
    #   CROP: MAIZ, WHEA, RICE, SORG, MILL, etc.
    #   SYSTEM: R (rainfed), I (irrigated), A (all systems)
    #
    # Without SPAM: Uses uniform crop_mask_percent for all cells.
    # =========================================================================
    spam_raster_path: Optional[Path] = Field(
        default=None,
        description="Path to SPAM harvested area raster (GeoTIFF). If set, calculates crop mask from actual data."
    )
    spam_cap_at_100_percent: bool = Field(
        default=True,
        description="Cap Percent values > 1.0 (can occur due to resolution mismatch)"
    )
    spam_na_to_zero: bool = Field(
        default=True,
        description="Replace NA/NoData values with 0 (no crop)"
    )


class PythiaConfig(BaseModel):
    """Platform-specific configuration for PYTHIA."""
    enabled: bool = Field(default=True, description="Enable PYTHIA output")
    grid_resolution: float = Field(
        default=0.0833,
        gt=0,
        description="Grid resolution in degrees (~10km for 5 arcmin)"
    )
    climate_source: ClimateSource = Field(
        default=ClimateSource.NASA_POWER,
        description="Climate data source"
    )
    soil_source: SoilSource = Field(
        default=SoilSource.EGHR,
        description="Soil data source"
    )
    dssat_version: str = Field(
        default="4.8",
        description="DSSAT version for compatibility"
    )

    # DSSAT executable path (platform-specific)
    dssat_executable: Optional[str] = Field(
        default=None,
        description="Path to DSSAT executable (auto-detected if not set)"
    )

    # Data source paths for full package generation
    eghr_raster_path: Optional[Path] = Field(
        default=None,
        description="Path to global eGHR soil raster (ggcmi_soils_2.tif)"
    )
    eghr_database_path: Optional[Path] = Field(
        default=None,
        description="Path to eGHR SQLite database (GHR.db)"
    )
    eghr_sol_dir: Optional[Path] = Field(
        default=None,
        description="Directory containing country .SOL files"
    )
    spam_raster_dir: Optional[Path] = Field(
        default=None,
        description="Directory containing SPAM 2020 harvest area rasters"
    )
    spam_version: str = Field(
        default="2020",
        description="SPAM data version (2010 or 2020)"
    )
    spam_crop_code: Optional[str] = Field(
        default=None,
        description="SPAM crop code override (e.g., MAIZ, RICE, WHEA). Auto-detected from crop name if not set."
    )

    # Weather download settings
    climate_start_date: Optional[str] = Field(
        default=None,
        description="Start date for climate download (YYYY-MM-DD)"
    )
    climate_end_date: Optional[str] = Field(
        default=None,
        description="End date for climate download (YYYY-MM-DD)"
    )
    weather_download_delay: float = Field(
        default=2.0,
        ge=0,
        description="Delay between NASA POWER API requests (seconds)"
    )

    # Simulation threading
    threads: int = Field(
        default=2,
        ge=1,
        description="Number of threads for PYTHIA simulation"
    )
    cores: int = Field(
        default=4,
        ge=1,
        description="Number of CPU cores for PYTHIA simulation"
    )

    # Planting window
    planting_window_days: int = Field(
        default=30,
        ge=1,
        le=90,
        description="Days after first planting date to search for planting conditions"
    )

    # Initial soil conditions
    initial_soil_water_pct: float = Field(
        default=25.0,
        ge=0,
        le=100,
        description="Initial soil water content as percentage of field capacity"
    )
    initial_inorganic_n: float = Field(
        default=10.0,
        ge=0,
        description="Initial inorganic nitrogen in soil (kg/ha)"
    )
    soil_water_threshold: float = Field(
        default=40.0,
        ge=0,
        le=100,
        description="Soil water threshold for planting (% of field capacity)"
    )
    residue_n_concentration: float = Field(
        default=0.8,
        ge=0,
        le=5,
        description="Residue nitrogen concentration (%)"
    )

    # Cultivar maturity thresholds (GDD)
    short_season_gdd_max: float = Field(
        default=1400.0,
        gt=0,
        description="Maximum GDD for short-season cultivar classification"
    )
    medium_season_gdd_max: float = Field(
        default=1700.0,
        gt=0,
        description="Maximum GDD for medium-season cultivar classification"
    )

    # DSSAT crop model selection (CERES vs CROPGRO)
    dssat_smodel: Optional[str] = Field(
        default=None,
        description=(
            "DSSAT simulation model code (e.g., 'MZCER' for CERES-Maize, "
            "'CPGRO' for CROPGRO-Cowpea). Auto-detected from crop name if not set: "
            "cereals use {crop_code}CER, legumes use CROPGRO."
        )
    )
    dssat_cultivar_ingeno: Optional[str] = Field(
        default=None,
        description=(
            "DSSAT cultivar code (INGENO) override. If set, bypasses the "
            "GDD-based maturity class mapping. E.g., 'II0003' for IT90K-277-2 cowpea."
        )
    )
    dssat_cultivar_cname: Optional[str] = Field(
        default=None,
        description=(
            "DSSAT cultivar name (CNAME) override. Used with dssat_cultivar_ingeno. "
            "E.g., 'IT90K-277-2'."
        )
    )
    dssat_symbiosis: Optional[str] = Field(
        default=None,
        description=(
            "DSSAT symbiotic N fixation switch (Y/N). Auto-detected from crop name "
            "if not set: Y for legumes (cowpea, soybean, groundnut, etc.), N otherwise."
        )
    )


class AceaConfig(BaseModel):
    """Platform-specific configuration for ACEA."""
    enabled: bool = Field(default=True, description="Enable ACEA output")
    resolution: Literal["5arcmin", "30arcmin"] = Field(
        default="5arcmin",
        description="Spatial resolution (5 or 30 arc-minutes)"
    )
    climate_source: ClimateSource = Field(
        default=ClimateSource.NASA_POWER,
        description="Climate data source"
    )
    soil_source: SoilSource = Field(
        default=SoilSource.HWSD,
        description="Soil data source"
    )
    compute_et0: bool = Field(
        default=True,
        description="Compute reference evapotranspiration (FAO-56)"
    )
    download_climate: bool = Field(
        default=True,
        description="Download NASA POWER climate data if missing"
    )
    climate_download_delay: float = Field(
        default=2.0,
        description="Delay between NASA POWER API requests (seconds)"
    )
    climate_name: Optional[str] = Field(
        default=None,
        description="Prefix for climate pickle files (auto-generated if not set)"
    )

    # HWSD soil data paths (for self-contained packages)
    hwsd_bil_path: Optional[Path] = Field(
        default=None,
        description="Path to HWSD2.bil raster file (user-provided)"
    )
    hwsd_mdb_path: Optional[Path] = Field(
        default=None,
        description="Path to HWSD2.mdb database file (user-provided)"
    )
    include_soil_in_package: bool = Field(
        default=True,
        description="Include extracted soil data in output package"
    )

    # SPAM harvested area data (for self-contained packages)
    spam_data_dir: Optional[Path] = Field(
        default=None,
        description="Path to directory containing SPAM raster files"
    )
    include_spam_in_package: bool = Field(
        default=True,
        description="Include SPAM data in output package (required by ACEA)"
    )
    spam_required: bool = Field(
        default=False,
        description="If True, fail when SPAM data not provided. If False, generate dummy files for self-contained packages."
    )

    # GAEZ data (auto-download or user-provided)
    gaez_data_dir: Optional[Path] = Field(
        default=None,
        description="Path to GAEZ data directory (auto-download if not set)"
    )
    gaez_auto_download: bool = Field(
        default=True,
        description="Auto-download GAEZ data from FAO if not cached"
    )
    include_gaez_in_package: bool = Field(
        default=True,
        description="Include clipped GAEZ data in output package"
    )

    # Irrigation and field management
    bunds: bool = Field(
        default=False,
        description=(
            "Enable bunds for paddy rice simulation. When True, AquaCrop "
            "simulates soil water ponding (flooding). Required for irrigated rice."
        )
    )
    bunds_dz: float = Field(
        default=0.3,
        description="Bund height in meters (only used when bunds=True)"
    )
    irr_thresholds: Optional[list] = Field(
        default=None,
        description=(
            "Irrigation depletion thresholds per AquaCrop growth stage (list of 4 integers). "
            "Each value is the % of readily available water depletion that triggers irrigation. "
            "Default [50,50,50,50]. Use [0,0,0,0] for continuous flooding (paddy rice)."
        )
    )
    virtual_irrigation: str = Field(
        default="Lowinput",
        description=(
            "ACEA virtual irrigation mode. Valid values: "
            "'Lowinput' (rainfed-like), 'Lowvirt', 'Highvirt', 'Highinput' (fully irrigated)"
        )
    )

    # GDD overrides (optional — override CROP_GDD_DEFAULTS when provided)
    gdd_maturity: Optional[float] = Field(
        default=None,
        description=(
            "Override GDD from sowing to maturity (°Cd). If not set, uses "
            "CROP_GDD_DEFAULTS for the crop. Set when literature-calibrated "
            "values differ from global defaults (e.g., Sahel rice)."
        )
    )
    gdd_senescence: Optional[float] = Field(
        default=None,
        description="Override GDD from sowing to senescence (°Cd)"
    )
    gdd_max_root: Optional[float] = Field(
        default=None,
        description="Override GDD from sowing to max rooting depth (°Cd)"
    )


class PlatformConfigGroup(BaseModel):
    """Container for all platform-specific configurations."""
    sarra_py: Optional[SarraPyConfig] = Field(
        default_factory=SarraPyConfig,
        description="SARRA-Py configuration"
    )
    craft: Optional[CraftConfig] = Field(
        default_factory=CraftConfig,
        description="CRAFT configuration"
    )
    pythia: Optional[PythiaConfig] = Field(
        default_factory=PythiaConfig,
        description="PYTHIA configuration"
    )
    acea: Optional[AceaConfig] = Field(
        default_factory=AceaConfig,
        description="ACEA configuration"
    )


# =============================================================================
# Data Sources Configuration
# =============================================================================

class GadmSourceConfig(BaseModel):
    """Configuration for GADM data source."""
    version: str = Field(default="4.1", description="GADM version")
    base_path: Path = Field(
        default=Path("data/gadm/"),
        description="Base path for GADM shapefiles"
    )


class NasaPowerSourceConfig(BaseModel):
    """Configuration for NASA POWER API."""
    timeout: int = Field(default=120, ge=10, description="Request timeout in seconds")
    retry_count: int = Field(default=3, ge=1, description="Number of retries")
    request_delay: float = Field(default=1.0, ge=0, description="Delay between requests")


class ClimateSourceConfig(BaseModel):
    """Configuration for climate data sources."""
    rainfall_dir: Optional[Path] = Field(default=None, description="Path to existing TAMSAT rainfall data")
    agera5_dir: Optional[Path] = Field(default=None, description="Path to existing AgERA5 data")
    download_if_missing: bool = Field(default=False, description="Download data if not found locally")


class SoilSourceConfig(BaseModel):
    """Configuration for soil data sources."""
    isda_dir: Optional[Path] = Field(default=None, description="Path to iSDA soil data")
    hwsd_dir: Optional[Path] = Field(default=None, description="Path to HWSD soil data")
    hwsd_bil_path: Optional[Path] = Field(
        default=None,
        description="Path to HWSD2.bil raster file (shared across platforms)"
    )
    hwsd_mdb_path: Optional[Path] = Field(
        default=None,
        description="Path to HWSD2.mdb database file (shared across platforms)"
    )


class DataSourcesConfig(BaseModel):
    """Configuration for all data sources."""
    gadm: GadmSourceConfig = Field(default_factory=GadmSourceConfig)
    nasa_power: NasaPowerSourceConfig = Field(default_factory=NasaPowerSourceConfig)
    climate: ClimateSourceConfig = Field(default_factory=ClimateSourceConfig, description="Climate data sources")
    soil: SoilSourceConfig = Field(default_factory=SoilSourceConfig, description="Soil data sources")
    cache_enabled: bool = Field(default=True, description="Enable data caching")
    cache_dir: Path = Field(default=Path("data/cache/"), description="Cache directory")


# =============================================================================
# Output Configuration
# =============================================================================

class OutputConfig(BaseModel):
    """Configuration for output directory structure."""
    base_dir: Path = Field(
        default=Path("output/"),
        description="Base output directory"
    )
    structure: Literal["by_platform", "by_region"] = Field(
        default="by_platform",
        description="Output directory structure"
    )
    copy_to_simulation_data: bool = Field(
        default=False,
        description="Copy outputs to platform Simulation_Data folders"
    )


# =============================================================================
# Provenance Configuration
# =============================================================================

class ProvenanceConfig(BaseModel):
    """Configuration for provenance tracking and audit trail."""
    enabled: bool = Field(default=True, description="Enable provenance tracking")
    storage: Literal["json", "sqlite", "both"] = Field(
        default="json",
        description="Provenance storage format"
    )
    output_dir: Path = Field(
        default=Path("provenance/"),
        description="Provenance output directory"
    )
    include_hashes: bool = Field(
        default=True,
        description="Include SHA256 hashes of inputs/outputs"
    )
    include_parameters: bool = Field(
        default=True,
        description="Include all parameters in provenance records"
    )


# =============================================================================
# Processing Configuration
# =============================================================================

class ProcessingConfig(BaseModel):
    """Configuration for data processing."""
    parallel: bool = Field(default=False, description="Enable parallel processing")
    max_workers: int = Field(default=4, ge=1, description="Maximum parallel workers")
    verbose: bool = Field(default=True, description="Enable verbose output")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )


# =============================================================================
# Validation Configuration
# =============================================================================

class ClimateValidationConfig(BaseModel):
    """Validation ranges for climate data."""
    tmax_range: tuple[float, float] = Field(default=(-40, 60))
    tmin_range: tuple[float, float] = Field(default=(-50, 50))
    precip_range: tuple[float, float] = Field(default=(0, 500))
    srad_range: tuple[float, float] = Field(default=(0, 40))
    et0_range: tuple[float, float] = Field(default=(0, 20))


class SoilValidationConfig(BaseModel):
    """Validation ranges for soil data."""
    sand_range: tuple[float, float] = Field(default=(0, 100))
    clay_range: tuple[float, float] = Field(default=(0, 100))
    silt_range: tuple[float, float] = Field(default=(0, 100))


class ValidationConfig(BaseModel):
    """Validation configuration for quality control."""
    climate: ClimateValidationConfig = Field(default_factory=ClimateValidationConfig)
    soil: SoilValidationConfig = Field(default_factory=SoilValidationConfig)


# =============================================================================
# Project Configuration
# =============================================================================

class ProjectInfo(BaseModel):
    """Project metadata."""
    name: str = Field(..., min_length=1, description="Project name")
    description: Optional[str] = Field(default=None, description="Project description")
    version: str = Field(default="1.0", description="Project version")
    created: Optional[date] = Field(default=None, description="Creation date")


class ProjectConfig(BaseModel):
    """Complete project configuration schema.

    This is the top-level configuration that combines all settings
    for a data-to-model translation project.
    """
    project: ProjectInfo = Field(..., description="Project metadata")
    region: RegionConfig = Field(..., description="Region configuration")
    crop: CropConfig = Field(..., description="Crop configuration")
    temporal: TemporalConfig = Field(..., description="Temporal settings")
    targets: List[Platform] = Field(
        default=[Platform.SARRA_PY, Platform.CRAFT, Platform.PYTHIA, Platform.ACEA],
        description="Target platforms to generate outputs for"
    )
    platform_config: PlatformConfigGroup = Field(
        default_factory=PlatformConfigGroup,
        description="Platform-specific configurations"
    )
    data_sources: DataSourcesConfig = Field(
        default_factory=DataSourcesConfig,
        description="Data source configurations"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration"
    )
    provenance: ProvenanceConfig = Field(
        default_factory=ProvenanceConfig,
        description="Provenance tracking configuration"
    )
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig,
        description="Processing configuration"
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Validation configuration"
    )

    # Generic parameters (platform agnostic)
    management: Optional[ManagementConfig] = Field(
        default=None,
        description="Generic crop management parameters"
    )
    soil_config: Optional[GenericSoilConfig] = Field(
        default=None,
        description="Generic soil configuration"
    )

    # Management zones for spatial variability (framework-agnostic)
    management_zones: List[ManagementZone] = Field(
        default_factory=list,
        description="Optional management zones for spatial variability. "
                    "Define zones by bounding_box (lat/lon) or admin_level (names). "
                    "Cells not matching any zone use defaults from 'management'. "
                    "First matching zone wins if zones overlap."
    )

    def get_enabled_platforms(self) -> List[Platform]:
        """Get list of enabled target platforms."""
        enabled = []
        for platform in self.targets:
            config = getattr(self.platform_config, platform.value, None)
            if config is not None and config.enabled:
                enabled.append(platform)
        return enabled

    def get_platform_config(self, platform: Platform) -> Union[
        SarraPyConfig, CraftConfig, PythiaConfig, AceaConfig, None
    ]:
        """Get configuration for a specific platform."""
        return getattr(self.platform_config, platform.value, None)
