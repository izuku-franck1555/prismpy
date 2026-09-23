"""SPAM cropland-vintage registry + fail-loud resolver (PYTHIA-consumed today).

Single airtight choke-point that maps a web-app-SELECTED cropland vintage
``(year, release)`` to the exact provisioned SPAM harvested-area raster, with
NO wildcard and NO silent fallback. If :func:`resolve_spam_raster` returns a
path, that path *is* the selected vintage's raster; otherwise it raises one of
four distinct, actionable errors. This is what makes a completed masked PYTHIA
run guarantee *applied == selected* without needing any provenance record.

Scope note: this module is consumed by the PYTHIA translator only. ACEA resolves
harvested areas through its own ``SPAMSource`` / ``_clip_spam_data`` path and is
intentionally NOT wired here. CRAFT reads a verbatim raster path and does not
resolve.

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

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "VintageSpec",
    "AppliedVintage",
    "SPAM_VINTAGES",
    "SpamVintageError",
    "VintageNotRegisteredError",
    "CropNotInVintageError",
    "StratumNotInVintageError",
    "VintageRasterAbsentError",
    "resolve_spam_raster",
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
