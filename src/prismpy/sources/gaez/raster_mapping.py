"""Two-vocabulary translator between ACEA's downloader API and the Esri
ImageService res02 HIST baseline attribute schema.

Per durable §27 producer-consumer vocabulary parity:
  - ACEA callsite supplies a 4-tuple ``(cultivar, water_supply, input_level,
    gaez_var)`` in ACEA's downloader vocabulary (e.g., ``Highland_sorghum``,
    ``rf``, ``Low``, ``LUT``).
  - The Esri ImageServer expects the equivalent attribute filter in res02
    HIST's vocabulary (e.g., ``crop='Highland sorghum'``, ``water_supply=
    'Available water content of 200 mm/m (under rain-fed conditions)'``,
    ``input_level='Low'``, ``variable LIKE '%Selected LUT sequence number%'``).
  - ``resolve_esri_query`` is the canonical translator; ``EsriQuerySpec`` is
    the validated 4-axis bundle the client consumes.

The vocabulary values were empirically verified against the live res02
ImageServer (host URL owned by ``esri_client.py``) via
``/query?outFields=*&returnIdsOnly=true`` probes. The ``where`` filters
in this module produce exactly 1 OBJECTID per ACEA 4-tuple — the
``test_exactly_one_raster_per_acea_tuple`` pin asserts this invariant.

LUT pixel range: empirical comparison on 2026-05-11 of the Esri res02 HIST
Sorghum LUT raster (global 4320×2160) returned pixel range ``[-9, 0, 82-88]``
(9 unique values) and the Maize LUT raster returned ``[-9, 0, 39-45]`` (9
unique values) — both match the discrete-sequence scheme the cached old-S3
``LUT_Highland_maize_rf_Low_1981_2010.tif`` file carried (``[-9, 0, 39-45]``,
9 unique). The Esri raster is a value-set match for the old S3 raster: same
``-9`` sentinel, same ``0`` no-data, same crop-family-specific sequence
numbering. ACEA's downstream rasterio consumer reads both rasters with
identical semantics — no value translator required.
"""
from __future__ import annotations

import json
from typing import Dict, Mapping

from pydantic import BaseModel, ConfigDict, Field

from prismpy.sources.gaez.errors import UnknownCombination


# ---------------------------------------------------------------------------
# Vocabulary maps — ACEA downloader vocabulary -> Esri res02 HIST vocabulary.
#
# These are the SOLE definitions of the cultivar / water / level / variable
# mappings. Consumers MUST import these constants rather than redefining the
# mapping inline (durable §24 + WA CA-2). The AST-walker structural pin
# ``test_no_inline_cultivar_map_redefinition`` enforces this contract at the
# translator.py boundary.
# ---------------------------------------------------------------------------

# Cultivar (ACEA 4-letter family) -> Esri ``crop`` attribute value.
# Empirically verified: res02 HIST baseline carries every value below in its
# ``crop`` attribute column with exactly one raster per (crop, water, level,
# variable, year='1981-2010', model='CRUTS32') 5-tuple.
_CULTIVAR_TO_ESRI_CROP: Dict[str, str] = {
    # Maize family
    "Highland_maize": "Highland maize",
    "Lowland_maize": "Lowland maize",
    "Temperate_maize": "Temperate maize",
    "Maize": "Maize",
    # Wheat family
    "Spring_wheat": "Spring wheat",
    "Winter_wheat": "Winter wheat",
    # Rice family
    "Wetland_rice": "Wetland rice",
    "Dryland_rice": "Dryland rice",
    # Sorghum family
    "Highland_sorghum": "Highland sorghum",
    "Lowland_sorghum": "Lowland sorghum",
    # Millet family
    "Pearl_millet": "Pearl millet",
    "Foxtail_millet": "Foxtail millet",
}


# Water supply (ACEA short code) -> Esri ``water_supply`` attribute value.
_WATER_TO_ESRI: Dict[str, str] = {
    "irr": "Available water content of 200 mm/m (under irrigation conditions)",
    "rf": "Available water content of 200 mm/m (under rain-fed conditions)",
}


# Input level (ACEA short code) -> Esri ``input_level`` attribute value.
# NOTE: ``Intermediate`` (a.k.a. 8110I) is NOT carried in the res02 HIST
# baseline — only High + Low. ACEA's default ``input_levels=['High','Low']``
# is therefore the safe-by-default contract; passing ``'Intermediate'`` will
# raise ``UnknownCombination`` to surface the misconfiguration honestly.
_LEVEL_TO_ESRI: Dict[str, str] = {
    "High": "High",
    "Low": "Low",
}


# GAEZ variable code (ACEA's 3-character form) -> Esri ``variable`` LIKE
# pattern. We use SQL ``LIKE`` patterns (``%...%``) for two reasons:
#   1. The F3 attribute value uses Unicode typographic quotes (U+2018/U+2019)
#      around the digit '3'; an exact-match filter is brittle. The ``%3%``
#      pattern matches the digit regardless of quote convention.
#   2. The Esri service accepts ``LIKE`` in the mosaicRule ``where`` clause,
#      so the loose match is wire-level supported without a client-side
#      regex pass.
_VAR_TO_PATTERN: Dict[str, str] = {
    "LUT": "%Selected LUT sequence number%",
    "PotentialYield": "%Agro-climatic potential yield%",
    "F3": "%constraint factor%3%agro-climatic%",
}


class EsriQuerySpec(BaseModel):
    """Pydantic-validated 4-axis query used to build the mosaicRule ``where``.

    Carries the post-translation res02 HIST attribute values. The
    ``year`` and ``model`` fields default to the historical baseline
    contract — F-CP targets the HIST baseline service (res02). Future
    projection-service support (res05) would add an alternate query spec
    with model='IPSL-CM5A-LR' etc. and live alongside this one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    crop: str = Field(description="Esri crop attribute value (e.g., 'Highland sorghum').")
    water_supply: str = Field(
        description="Esri water_supply attribute value (full descriptive string)."
    )
    input_level: str = Field(description="Esri input_level attribute value ('High'|'Low').")
    variable_pattern: str = Field(
        description="SQL LIKE pattern for the 'variable' attribute."
    )
    year: str = Field(default="1981-2010", description="Esri year attribute value.")
    model: str = Field(default="CRUTS32", description="Esri model attribute value.")

    def to_where(self) -> str:
        """Render the Esri attribute filter as a SQL-style ``where`` clause.

        Single-quoted string literals match the Esri ImageServer's SQL
        flavor. The ``variable`` axis uses ``LIKE`` for typographic-quote
        resilience (see ``_VAR_TO_PATTERN`` docstring).
        """
        return (
            f"crop='{_sql_escape(self.crop)}' AND "
            f"water_supply='{_sql_escape(self.water_supply)}' AND "
            f"input_level='{_sql_escape(self.input_level)}' AND "
            f"year='{_sql_escape(self.year)}' AND "
            f"model='{_sql_escape(self.model)}' AND "
            f"variable LIKE '{_sql_escape(self.variable_pattern)}'"
        )

    def to_mosaic_rule_json(self) -> str:
        """Serialize as the ``esriMosaicAttribute`` mosaicRule JSON payload."""
        return json.dumps(
            {
                "mosaicMethod": "esriMosaicAttribute",
                "where": self.to_where(),
            },
            separators=(",", ":"),
        )


def _sql_escape(value: str) -> str:
    """Escape single-quote characters per SQL-92 string-literal escaping.

    Esri's ImageServer accepts SQL-92 quoted strings; the only escape we
    need to honor in attribute-filter values is the embedded single quote.
    All res02 HIST attribute values we ship use plain ASCII or Unicode
    typographic-quote chars (``U+2018``/``U+2019``) which do NOT collide
    with the SQL string delimiter, so doubling-up is defensive belt rather
    than a load-bearing requirement today.
    """
    return value.replace("'", "''")


def resolve_esri_query(
    cultivar: str,
    water_supply: str,
    input_level: str,
    gaez_var: str,
) -> EsriQuerySpec:
    """Canonical 4-tuple -> ``EsriQuerySpec`` translator.

    Accepts the ACEA downloader vocabulary on every axis (cultivar in
    either space-form ``"Highland sorghum"`` or underscore-form
    ``"Highland_sorghum"``; water_supply ``"irr"``/``"rf"``;
    input_level ``"High"``/``"Low"``; gaez_var ``"LUT"``/
    ``"PotentialYield"``/``"F3"``).

    Raises ``UnknownCombination`` if any axis value has no Esri mapping.
    Per durable §27: producer values ⊆ consumer keys; the structural pin
    ``test_full_esri_query_surjectivity`` walks the full Cartesian product
    and asserts every 4-tuple in ACEA's domain resolves.
    """
    # Cultivar — accept both space and underscore forms by normalizing
    # to underscore before lookup. Mirrors download_cultivar's
    # ``cultivar.replace(' ', '_')`` filename normalization.
    cultivar_canonical = cultivar.replace(" ", "_")
    esri_crop = _CULTIVAR_TO_ESRI_CROP.get(cultivar_canonical)
    if esri_crop is None:
        raise UnknownCombination(
            f"No Esri 'crop' mapping for cultivar={cultivar!r} "
            f"(normalized to {cultivar_canonical!r}); "
            f"valid keys: {sorted(_CULTIVAR_TO_ESRI_CROP.keys())!r}"
        )

    esri_water = _WATER_TO_ESRI.get(water_supply)
    if esri_water is None:
        raise UnknownCombination(
            f"No Esri 'water_supply' mapping for water_supply={water_supply!r}; "
            f"valid keys: {sorted(_WATER_TO_ESRI.keys())!r}"
        )

    esri_level = _LEVEL_TO_ESRI.get(input_level)
    if esri_level is None:
        raise UnknownCombination(
            f"No Esri 'input_level' mapping for input_level={input_level!r}; "
            f"valid keys: {sorted(_LEVEL_TO_ESRI.keys())!r}"
        )

    var_pattern = _VAR_TO_PATTERN.get(gaez_var)
    if var_pattern is None:
        raise UnknownCombination(
            f"No Esri 'variable' pattern for gaez_var={gaez_var!r}; "
            f"valid keys: {sorted(_VAR_TO_PATTERN.keys())!r}"
        )

    return EsriQuerySpec(
        crop=esri_crop,
        water_supply=esri_water,
        input_level=esri_level,
        variable_pattern=var_pattern,
    )


# Public helpers exposed for the test layer (read-only views of the maps).
def cultivar_keys() -> Mapping[str, str]:
    return dict(_CULTIVAR_TO_ESRI_CROP)


def water_keys() -> Mapping[str, str]:
    return dict(_WATER_TO_ESRI)


def level_keys() -> Mapping[str, str]:
    return dict(_LEVEL_TO_ESRI)


def variable_keys() -> Mapping[str, str]:
    return dict(_VAR_TO_PATTERN)
