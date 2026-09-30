"""SPAM cropland-vintage registry + fail-loud resolver (PYTHIA-consumed today).

Single airtight choke-point that maps a web-app-SELECTED cropland vintage
``(year, release)`` to the exact provisioned SPAM harvested-area raster, with
NO wildcard and NO silent fallback. If :func:`resolve_spam_raster` returns a
path, that path *is* the selected vintage's raster; otherwise it raises one of
four distinct, actionable errors. This is what makes a completed masked PYTHIA
run guarantee *applied == selected* without needing any provenance record.

Scope note: the resolver :func:`resolve_spam_raster` is consumed by the PYTHIA
translator and by ACEA, which carries the registered layer itself, verified
against its pinned :func:`content_digest`. CRAFT reads a
verbatim raster path and does not resolve — it consumes :func:`identify_vintage`
to derive an honest ``SPAM <year> <release>`` label from that path's basename,
and fails loud when the basename matches no registered vintage.

The per-vintage crop/stratum inventories are LITERAL frozensets generated from
the actually-provisioned MapSPAM files, so ``CropNotInVintageError`` and
``VintageRasterAbsentError`` are distinguishable WITHOUT the raster files being
present (self-contained). Regenerate deterministically with::

    find <vintage_dir> -iname '*.tif' | grep -oE '_H_[A-Z]{4}_' | sed -E 's/_H_|_//g' | sort -u

Provisioned inventory (verified against Franck's files):
  * 2010 / V2r0 : 252 files, 42 crops, 6 strata {A,H,I,L,R,S}, no underscore after ``spam2010``
  * 2020 / V2r2 : 138 rasters (+6 .tif.ovr), 46 crops, 3 strata {A,I,R}, underscore after ``spam2020``
Crop coverage DIFFERS between these two vintages (2010-only: ACOF, SMIL;
2020-only: CITR, COFF, MILL, ONIO, RUBB, TOMA) — so "crop absent in the selected
vintage" is already a real, reachable condition, not hypothetical.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

__all__ = [
    "VintageSpec",
    "AppliedVintage",
    "SPAM_VINTAGES",
    "DEFAULT_SPAM_VINTAGE",
    "SPAM_CONTENT_DIGESTS",
    "SPAM_CONTENT_HEADERS",
    "SpamVintageError",
    "VintageNotRegisteredError",
    "CropNotInVintageError",
    "StratumNotInVintageError",
    "VintageRasterAbsentError",
    "resolve_spam_raster",
    "identify_vintage",
    "code_for_vintage",
    "derive_harvested_area_label",
    "content_digest",
    "acea_canonical_triples",
]


# --- Literal per-vintage crop inventories (from the provisioned files) --------
_CROPS_2010_V2R0 = frozenset({  # 42
    "ACOF", "BANA", "BARL", "BEAN", "CASS", "CHIC", "CNUT", "COCO", "COTT",
    "COWP", "GROU", "LENT", "MAIZ", "OCER", "OFIB", "OILP", "OOIL", "OPUL",
    "ORTS", "PIGE", "PLNT", "PMIL", "POTA", "RAPE", "RCOF", "REST", "RICE",
    "SESA", "SMIL", "SORG", "SOYB", "SUGB", "SUGC", "SUNF", "SWPO", "TEAS",
    "TEMF", "TOBA", "TROF", "VEGE", "WHEA", "YAMS",
})
_CROPS_2020_V2R2 = frozenset({  # 46 (adds CITR,COFF,MILL,ONIO,RUBB,TOMA; drops ACOF,SMIL vs 2010)
    "BANA", "BARL", "BEAN", "CASS", "CHIC", "CITR", "CNUT", "COCO", "COFF",
    "COTT", "COWP", "GROU", "LENT", "MAIZ", "MILL", "OCER", "OFIB", "OILP",
    "ONIO", "OOIL", "OPUL", "ORTS", "PIGE", "PLNT", "PMIL", "POTA", "RAPE",
    "RCOF", "REST", "RICE", "RUBB", "SESA", "SORG", "SOYB", "SUGB", "SUGC",
    "SUNF", "SWPO", "TEAS", "TEMF", "TOBA", "TOMA", "TROF", "VEGE", "WHEA",
    "YAMS",
})


@dataclass(frozen=True)
class VintageSpec:
    """One provisioned cropland vintage. ``pattern`` uses ``{code}``/``{tech}``.

    ``crops`` and ``strata`` are the LITERAL inventories on disk for this
    vintage — used to distinguish crop-not-mapped from stratum-absent from
    not-provisioned before any filesystem access.
    """

    pattern: str
    crops: frozenset
    strata: frozenset


@dataclass(frozen=True)
class AppliedVintage:
    """The single canonical applied-vintage state, produced ONCE per masked run
    and threaded to every emitted surface (never re-derived by parse/glob).

    ``source_filename`` is the resolved global raster's basename;
    ``mask_filename`` is the deterministic clipped-mask basename
    (``harvest_area_{year}_{release}.tif``).
    """

    year: str
    release: str
    source_filename: str
    mask_filename: str

    def to_manifest_dict(self) -> dict:
        """Structured form persisted at ``data_sources.crop_mask_vintage``."""
        return {
            "year": self.year,
            "release": self.release,
            "source_filename": self.source_filename,
            "mask_filename": self.mask_filename,
        }

    @property
    def label(self) -> str:
        """Honest human-readable string persisted at ``data_sources.crop_mask``."""
        return f"SPAM {self.year} {self.release}"


# --- SSOT registry: (year, release) -> exact pattern + inventories ------------
SPAM_VINTAGES: dict = {
    ("2020", "V2r2"): VintageSpec(
        pattern="spam2020_V2r2_global_H_{code}_{tech}.tif",  # underscore after 2020
        crops=_CROPS_2020_V2R2,
        strata=frozenset({"A", "I", "R"}),
    ),
    ("2010", "V2r0"): VintageSpec(
        pattern="spam2010V2r0_global_H_{code}_{tech}.tif",  # no underscore after 2010
        crops=_CROPS_2010_V2R0,
        strata=frozenset({"A", "H", "I", "L", "R", "S"}),
    ),
}

#: The vintage an ACEA package carries when its project selects none.
DEFAULT_SPAM_VINTAGE: Tuple[str, str] = ("2020", "V2r2")

# One crop, spelled differently by the 2010 and 2020 releases.
_CODE_RENAMES = {"ACOF": "COFF", "COFF": "ACOF", "SMIL": "MILL", "MILL": "SMIL"}

_CONTENT_DIGEST_SCHEMA = "spam-content-digest/1"
_CONTENT_DIGESTS_FILE = "spam_content_digests.json"


class SpamVintageError(Exception):
    """Base for all fail-loud cropland-vintage resolution errors."""


class VintageNotRegisteredError(SpamVintageError):
    """The selected ``(year, release)`` is not a known provisioned vintage."""


class CropNotInVintageError(SpamVintageError):
    """The crop is not mapped in the selected vintage (a different vintage may map it)."""


class StratumNotInVintageError(SpamVintageError):
    """The technology stratum is not available in the selected vintage."""


class VintageRasterAbsentError(SpamVintageError):
    """Registered + supported, but the raster is not on disk (not provisioned)."""


def resolve_spam_raster(
    spam_dir: Path,
    year: str,
    release: str,
    crop_code: str,
    tech: str,
) -> Path:
    """Resolve the SPAM raster for a SELECTED vintage — registry-only, fail-loud.

    Resolution order (each failure raises a distinct, actionable error; there is
    no wildcard, no silent substitution, and no ``None`` return):

    1. ``VintageNotRegisteredError`` — ``(year, release)`` not in the registry.
    2. ``CropNotInVintageError`` — ``crop_code`` not mapped in this vintage.
    3. ``StratumNotInVintageError`` — ``tech`` not available in this vintage.
    4. ``VintageRasterAbsentError`` — everything valid but the file is absent.

    Returns the single unambiguous :class:`~pathlib.Path` on success.
    """
    key = (year, release)
    spec = SPAM_VINTAGES.get(key)
    if spec is None:
        known = ", ".join(f"{y}/{r}" for (y, r) in sorted(SPAM_VINTAGES))
        raise VintageNotRegisteredError(
            f"SPAM vintage {year}/{release} is not a registered provisioned vintage "
            f"(known: {known}). A masked run must select a provisioned vintage."
        )

    code = str(crop_code).upper()
    if code not in spec.crops:
        raise CropNotInVintageError(
            f"crop {code!r} is not mapped in SPAM vintage {year}/{release} "
            f"(this vintage maps {len(spec.crops)} crops). The crop may exist in a "
            f"different vintage; never substitute another vintage's raster."
        )

    stratum = str(tech).upper()
    if stratum not in spec.strata:
        raise StratumNotInVintageError(
            f"technology stratum {stratum!r} is not available in SPAM vintage "
            f"{year}/{release} (available: {sorted(spec.strata)})."
        )

    filename = spec.pattern.format(code=code, tech=stratum)
    path = Path(spam_dir) / filename
    if not path.exists():
        raise VintageRasterAbsentError(
            f"SPAM vintage {year}/{release} raster {filename!r} not found in "
            f"{spam_dir} — the selected vintage is not provisioned. A masked run "
            f"fails loud rather than silently applying a different vintage."
        )
    return path


def identify_vintage(filename) -> Optional[Tuple[str, str]]:
    """Reverse-lookup a raster BASENAME to its registered cropland vintage ``(year, release)``.

    CRAFT reads a verbatim ``spam_raster_path`` — it never calls :func:`resolve_spam_raster` —
    so its honest cropland-vintage label must be derived from the ACTUAL file on disk, never a
    separately-declared vintage that could disagree with the basename. This is that single
    source of vintage identity: it matches the basename against every registered vintage's
    ``pattern`` with the ``{code}``/``{tech}`` slots constrained to that vintage's OWN crop and
    stratum inventories, so a 2010 raster can never be misread as 2020 (or vice-versa), and an
    unrecognized / wrong file returns ``None`` (the caller fails loud rather than emit a
    dishonest label).

    Returns the matching ``(year, release)``, or ``None`` when the basename is not a recognized
    provisioned raster of any registered vintage.
    """
    base = Path(filename).name
    for (year, release), spec in SPAM_VINTAGES.items():
        codes = "|".join(sorted(spec.crops))
        techs = "|".join(sorted(spec.strata))
        regex = re.escape(spec.pattern)
        regex = regex.replace(re.escape("{code}"), f"(?:{codes})")
        regex = regex.replace(re.escape("{tech}"), f"(?:{techs})")
        if re.fullmatch(regex, base):
            return (year, release)
    return None


def code_for_vintage(crop_code: str, year: str, release: str) -> str:
    """``crop_code`` as the registered vintage ``(year, release)`` spells it.

    The 2010 and 2020 releases name two crops differently (ACOF/COFF, SMIL/MILL). A code the
    vintage maps under neither spelling is returned unchanged, so resolving it raises
    :class:`CropNotInVintageError`.
    """
    spec = SPAM_VINTAGES.get((year, release))
    if spec is None:
        raise VintageNotRegisteredError(
            f"SPAM vintage {year}/{release} is not a registered provisioned vintage")
    code = str(crop_code).upper()
    if code not in spec.crops and _CODE_RENAMES.get(code) in spec.crops:
        return _CODE_RENAMES[code]
    return code


def derive_harvested_area_label(year: str, release: str, selection: str) -> str:
    """The harvested-area label of a package carrying SPAM ``year`` ``release``, selected or by default."""
    if selection not in ("selected", "default"):
        raise ValueError(f"a SPAM vintage is 'selected' or 'default', not {selection!r}")
    return f"SPAM {year} {release}" + (" (default; no vintage selected)" if selection == "default" else "")


def _content_header(dataset: Any) -> dict:
    if dataset.count != 1:
        raise SpamVintageError(f"a SPAM layer holds one band; this one holds {dataset.count}")
    authority = dataset.crs.to_authority() if dataset.crs else None
    if not authority:
        raise SpamVintageError(f"a SPAM layer's CRS has no authority code: {dataset.crs}")
    nodata = dataset.nodata
    return {
        "schema": _CONTENT_DIGEST_SCHEMA,
        "shape": [dataset.height, dataset.width],
        "count": dataset.count,
        "band_index": 1,
        "dtype": np.dtype(dataset.dtypes[0]).name,
        "byteorder": "<",
        "transform": [repr(float(x)) for x in tuple(dataset.transform)[0:6]],
        "crs": ":".join(authority),
        "nodata": None if nodata is None else repr(float(nodata)),
    }


def content_digest(dataset: Any) -> str:
    """The ``spam-content-digest/1`` of an open single-band raster.

    The sha256 of its canonical header (grid, georeferencing, dtype, nodata) and its band as
    decoded, never normalised: the same layer compressed or tiled differently has the same
    digest, and any change to a value, a NaN payload or the sign of a zero changes it.
    """
    header = json.dumps(_content_header(dataset), sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True, allow_nan=False).encode("utf-8")
    band = np.ascontiguousarray(dataset.read(1).astype("<f4", copy=False))
    return hashlib.sha256(header + b"\n" + band.tobytes()).hexdigest()


def _load_content_digests() -> Tuple[dict, dict]:
    """The pinned digest and header of every registered source, keyed ``(year, release, code, tech)``."""
    data = json.loads(resources.files(__package__).joinpath(_CONTENT_DIGESTS_FILE)
                      .read_text(encoding="utf-8"))
    if data.get("schema") != _CONTENT_DIGEST_SCHEMA:
        raise SpamVintageError(f"{_CONTENT_DIGESTS_FILE} is not {_CONTENT_DIGEST_SCHEMA}")
    digests, headers = {}, {}
    for entry in data["entries"]:
        key = (entry["year"], entry["release"], entry["crop_code"], entry["tech"])
        digests[key], headers[key] = entry["content_digest"], entry["header"]
    return digests, headers


SPAM_CONTENT_DIGESTS, SPAM_CONTENT_HEADERS = _load_content_digests()


def acea_canonical_triples(year: str, release: str) -> Tuple[Tuple[str, int, str], ...]:
    """Every ``(crop name, ACEA FAO code, SPAM code)`` an ACEA package may declare in the vintage."""
    # The maps live in the ACEA translator, which imports this module.
    from prismpy.translators.acea.translator import ACEA_FAO_CODE_MAP, SPAM_CODE_MAP

    return tuple((name, ACEA_FAO_CODE_MAP[name], code_for_vintage(SPAM_CODE_MAP[name], year, release))
                 for name in sorted(ACEA_FAO_CODE_MAP) if name in SPAM_CODE_MAP)
