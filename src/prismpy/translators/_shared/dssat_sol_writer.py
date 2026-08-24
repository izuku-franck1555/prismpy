"""DSSAT v4.8 .SOL soil-file writer.

Single canonical implementation of the DSSAT v4.8 soil-profile file format
used by every translator that emits .SOL files (CRAFT today, the eGHR
substrate builder in PYTHIA next). Centralising the writer here means
every consumer produces byte-identical headers, surface blocks, and
fixed-width layer tables; format drift is impossible.

The writer follows the column conventions in the DSSAT v4.8 user guide
(Tsuji et al. 1994 plus the v4.8 update) so the file can be loaded
without any post-translation pre-processing by either DSSAT-CSM or the
PYTHIA harness.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from prismpy.models.region import Region
from prismpy.models.soil import SoilProfile
from prismpy.utils.sanitization import sanitize_admin_name


logger = logging.getLogger(__name__)


def _default_source_label(profile_id: int) -> str:
    """Default source-label generator (CRAFT historical wording).

    Returns the string CRAFT has been writing into the per-profile
    "Source" cell since the writer was first introduced. Translators
    that pull from a different upstream substrate can pass their own
    callable to override.
    """
    return f"HWSD v2 SMU {profile_id}"


# ── DSSAT SLTX texture codes (USDA 12-class triangle) ────────────────
# The .SOL profile-header SLTX field is fixed-width. A spelled-out class
# name (e.g. "SandyClayLoam", 13 chars) overflows the column and shifts
# the depth field, so DSSAT's IPSOIL header parser reads the wrong
# column and rejects the profile (Error 5010). Emit the canonical short
# code (<=4 chars) instead — it always fits. Keyed by the class name
# normalised to lower-case with spaces removed, so it matches both the
# spelled-out ("Sandy Clay Loam") and space-stripped ("SandyClayLoam")
# forms the ``SoilProfile.surface_texture`` classifier can produce.
_DSSAT_SLTX_CODES: Dict[str, str] = {
    "sand": "S",
    "loamysand": "LS",
    "sandyloam": "SL",
    "loam": "L",
    "siltloam": "SIL",
    "silt": "SI",
    "sandyclayloam": "SCL",
    "clayloam": "CL",
    "siltyclayloam": "SICL",
    "sandyclay": "SC",
    "siltyclay": "SIC",
    "clay": "C",
}


def _dssat_sltx_code(surface_texture: "str | None") -> str:
    """Map a USDA texture class to its DSSAT SLTX code (fail-loud).

    The .SOL profile-header SLTX field is fixed-width; a spelled-out
    class name overflows it and shifts the depth column, so DSSAT's
    IPSOIL parser rejects the profile (Error 5010). The canonical short
    code (<=4 chars) always fits. Raises on an absent/unrecognised class
    rather than silently emitting an overflowing name.
    """
    key = (surface_texture or "").replace(" ", "").lower()
    try:
        return _DSSAT_SLTX_CODES[key]
    except KeyError:
        raise ValueError(
            f"Unmapped soil texture class {surface_texture!r}: cannot emit a "
            f"DSSAT SLTX code for the .SOL profile header. Expected one of the "
            f"USDA 12-class set {sorted(_DSSAT_SLTX_CODES)} "
            f"(matched case- and space-insensitively)."
        ) from None


def _resolve_chem_default(
    profile_id: int,
    field: str,
    value: Optional[float],
    default: float,
    log: List[Dict[str, Any]],
) -> float:
    """Return ``value`` when present, else the regional ``default``.

    When the default is used the substitution is appended to ``log`` so a
    genuinely-absent chemistry value is disclosed to provenance rather than
    written as if it were a measurement. ``None`` (not falsiness) marks
    absence, so a real ``0.0`` reading is kept instead of being defaulted.
    """
    if value is not None:
        return value
    log.append({"profile_id": profile_id, "field": field, "default": default})
    return default


def write_dssat_sol(
    soil_path: Path,
    profiles_by_id: Mapping[int, SoilProfile],
    country_code: str,
    region: Region,
    file_header_suffix: str = "(HWSD-based)",
    source_label_for_id: Callable[[int], str] = _default_source_label,
    chem_default_log: Optional[List[Dict[str, Any]]] = None,
) -> Dict[int, str]:
    """Write a DSSAT v4.8-spec .SOL file containing the supplied profiles.

    The output file uses ``\\r\\n`` line terminators (the DSSAT-CSM
    convention) and contains, for each profile:

    - a 10-character profile id line beginning with ``*``
    - a ``@SITE`` line carrying the human-readable site name, ISO3
      country code, latitude, and longitude in fixed positions
    - a ``@ SCOM`` surface-properties block (SALB / SLU1 / SLDR /
      SLRO / SLNF / SLPF / SMHB / SMPX / SMKE)
    - a 17-column ``@ SLB`` layer header (SLMH, SRGF, SCEC, and
      SADC included alongside the older 13-column set)
    - one fixed-width layer row per :class:`SoilLayer` on the profile

    Args:
        soil_path: Filesystem path the writer should overwrite.
        profiles_by_id: Mapping of stable profile id (typically an
            HWSD SMU id but any integer is fine) to a fully-populated
            :class:`SoilProfile`. Profiles are written in ascending
            id order so the file content is deterministic.
        country_code: Two-letter country code used as the prefix of
            each 10-character profile name (e.g. ``"NE"`` for Niger,
            ``"CM"`` for Cameroon).
        region: Region carrying the human-readable name plus the ISO3
            country code (used in the per-profile header) and the
            country name (used as a backup for site labelling).
        file_header_suffix: Parenthetical fragment appended to the
            top-of-file ``*SOILS:`` line. CRAFT has historically used
            ``"(HWSD-based)"``. The eGHR substrate builder can
            substitute a different label if it sources profiles from
            another upstream dataset.
        source_label_for_id: Callable that turns a profile id into the
            free-text "Source" cell on the per-profile header line.
            Default reproduces CRAFT's historical wording exactly so
            existing CRAFT packages remain byte-identical.

    Returns:
        Dict mapping profile id (the key from ``profiles_by_id``) to
        the 10-character profile name written into the file. Callers
        that need a cell-to-profile lookup table (CRAFT's
        ``soil_mask.txt``, PYTHIA's ``GHR.db``) build it from this
        return value.
    """
    smu_to_profile_name: Dict[int, str] = {}
    # Chemistry defaults are disclosed, never silent: an internal list
    # always lets the aggregate warning fire, and a caller-supplied log
    # additionally captures each substitution for provenance.
    chem_defaults: List[Dict[str, Any]] = (
        chem_default_log if chem_default_log is not None else []
    )

    with open(soil_path, "w", newline="\r\n") as f:
        f.write(
            f"*SOILS: {region.name} - Generated by prismpy {file_header_suffix}\n\n"
        )

        for smu_id, profile in sorted(profiles_by_id.items()):
            # Profile ID: {CC}{SMU_ID:08d} (10 chars max)
            profile_name = f"{country_code}{smu_id:08d}"[:10]
            smu_to_profile_name[smu_id] = profile_name

            # Profile header aligned to DSSAT FORMAT 5030 (IPSOIL_Inp.for:627):
            #   1X, A10, 2X, A11, 1X, A5, 1X, F5.0, 1X, A50
            #   = *  id(2-11)  SLSOUR(14-24)  SLTX(26-30)  depth(32-36)  SLDESC(38+)
            # SLTX carries the DSSAT texture CODE in the A5 field (cols 26-30);
            # the code (<=4 chars) fits, so DSSAT reads the texture AND the depth
            # at their fixed columns. The prior spelled-out class overflowed the
            # field and shifted the depth column -> IPSOIL Error 5010.
            # SLSOUR (DSSAT "Soil Source") carries the source tag (first token of
            # the label, e.g. "eGHR"/"HWSD"); the full label + country go in the
            # A50 SLDESC (country is also encoded in the PEDON id). Both are
            # DSSAT-descriptive (not parsed for logic).
            sltx_code = _dssat_sltx_code(profile.surface_texture)
            depth_cm = int((profile.total_depth or 0.2) * 100)
            source_desc = source_label_for_id(smu_id)
            slsour = (source_desc.split() or ["-99"])[0][:11]
            sldesc = f"{source_desc} ({region.country_iso3 or 'XXX'})"
            f.write(
                f"*{profile_name:<10}  {slsour:<11} "
                f"{sltx_code:<5} {depth_cm:>5d} {sldesc}\n"
            )

            # Site value row aligned to DSSAT FORMAT 5035 (IPSOIL_Inp.for:628):
            #   2(1X,A11), 2(1X,F8.3), 1X, A50 = SSITE(2-12) SCOUNT(14-24)
            #   SLAT(26-33) SLONG(35-42) TAXON(44+). LAT/LONG are read at those
            # FIXED columns (F8.3), so an off-by-one truncates a coordinate digit.
            f.write("@SITE        COUNTRY          LAT     LONG SCS FAMILY\n")
            site_name = sanitize_admin_name(region.name)[:11]
            texture = profile.surface_texture or "Unknown"
            country_short = (region.country_iso3 or region.country or "XX")[:11]
            f.write(
                f" {site_name:<11} {country_short:<11} "
                f"{profile.lat:8.3f} {profile.lon:8.3f} {texture}\n"
            )

            # Surface properties block (DSSAT required)
            top_sand = profile.layers[0].sand if profile.layers else 60.0
            top_clay = profile.layers[0].clay if profile.layers else 18.0
            if top_sand > 70:
                salb, sldr, slro = 0.13, 0.60, 60.0
            elif top_clay > 40:
                salb, sldr, slro = 0.09, 0.20, 85.0
            else:
                salb, sldr, slro = 0.11, 0.40, 73.0
            f.write(
                "@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE\n"
            )
            f.write(
                f"    -9  {salb:4.2f}  6.00  {sldr:4.2f} {slro:5.2f}"
                f"  1.00  1.00 IB001 IB001 IB001\n"
            )

            # Layer header (with SLMH, SRGF, SCEC, SADC)
            f.write(
                "@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC"
                "  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC\n"
            )

            # Layer data
            for layer in profile.layers:
                if layer.wilting_point is None:
                    layer.estimate_hydraulic_properties()

                slb = int(layer.depth_bottom * 100)
                slll = layer.wilting_point or 0.10
                sdul = layer.field_capacity or 0.25
                ssat = layer.saturated_wc or 0.45
                ssks = 10.0
                sbdm = _resolve_chem_default(
                    smu_id, "bulk_density", layer.bulk_density, 1.4, chem_defaults
                )
                sloc = _resolve_chem_default(
                    smu_id, "organic_carbon", layer.organic_carbon, 0.5, chem_defaults
                )
                slcl = layer.clay or 18.0
                slsi = layer.silt or (100 - (layer.sand or 60) - slcl)
                slcf = 0.0
                slni = 0.0
                slhw = _resolve_chem_default(
                    smu_id, "ph", layer.ph, 6.5, chem_defaults
                )
                slhb = slhw
                srgf = max(
                    0.0,
                    1.0
                    - (layer.depth_bottom / (profile.total_depth or 1.0)) * 0.8,
                )

                # DSSAT fixed-width: every field is exactly 6 characters
                f.write(
                    f"{slb:6d}"
                    f" {'-9':<5s}"
                    f"{slll:6.3f}"
                    f"{sdul:6.3f}"
                    f"{ssat:6.3f}"
                    f"{srgf:6.2f}"
                    f"{ssks:6.2f}"
                    f"{sbdm:6.2f}"
                    f"{sloc:6.2f}"
                    f"{slcl:6.1f}"
                    f"{slsi:6.1f}"
                    f"{slcf:6.1f}"
                    f"{slni:6.2f}"
                    f"{slhw:6.1f}"
                    f"{slhb:6.1f}"
                    f"{-99.0:6.1f}"
                    f"{-99.0:6.1f}"
                    "\n"
                )

            f.write("\n")

    if chem_defaults:
        logger.warning(
            "%d soil-chemistry field(s) across %d profile(s) fell back to a "
            "regional default (absent in HWSD); these are not measured values.",
            len(chem_defaults),
            len({d["profile_id"] for d in chem_defaults}),
        )

    return smu_to_profile_name
