"""Per-package eGHR substrate builder.

The eGHR substrate is a triple of artifacts a PYTHIA-compatible package
needs in order to resolve every cell to a soil profile without depending
on a globally-bundled .SOL library:

- ``raster/soil.tif`` — single-band GeoTIFF whose pixel value at each
  cell carries an integer profile id (``0`` is reserved for nodata).
- ``eGHR/GHR.db`` — SQLite database with one table,
  ``profile_map(id INTEGER PRIMARY KEY, profile TEXT NOT NULL)``,
  where ``id`` is the raster pixel value and ``profile`` is the
  10-character profile name written into the ``.SOL`` file.
- ``eGHR/{CC}.SOL`` — DSSAT v4.8-spec soil-profile file, written via
  the canonical helper at
  :func:`prismpy.translators._shared.dssat_sol_writer.write_dssat_sol`,
  containing one ``*<profile>`` block per row in ``profile_map``.

The builder takes a per-cell mapping of HWSD2/iSDA-derived
:class:`SoilProfile` objects, deduplicates identical profiles into a
shared registry, and emits the three artifacts in the supplied output
directory. The cell-to-profile assignment is computed exactly once
inside :func:`assign_cell_to_profile_id` so the raster, the database,
and the ``.SOL`` cannot drift apart on which cell maps to which profile
(durable lesson §24, canonical-source-or-pin).
"""

from __future__ import annotations

import copy
import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import rasterio
from pydantic import BaseModel, ConfigDict, Field
from rasterio.transform import from_bounds

from prismpy.cockpit.cockpit_overrides_writer import CockpitOverrideSidecar
from prismpy.models.region import Region
from prismpy.models.soil import SoilLayer, SoilProfile
from prismpy.models.spatial import SpatialGrid
from prismpy.translators._shared.dssat_sol_writer import write_dssat_sol


logger = logging.getLogger(__name__)


SHA256_PATTERN = r"^[0-9a-f]{64}$"


class EghrSubstrateResult(BaseModel):
    """Outcome of building a per-package eGHR substrate triple.

    Returned by :func:`build_eghr_substrate`. Field paths point at the
    three artifacts on disk; SHA-256 hashes capture the byte-level state
    so a downstream consumer can detect tampering or stale cache hits.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    soil_raster_path: Path = Field(
        description="GeoTIFF profile-id raster (`raster/soil.tif`).",
    )
    ghr_db_path: Path = Field(
        description="SQLite database with `profile_map(id, profile)` (`eGHR/GHR.db`).",
    )
    sol_path: Path = Field(
        description="DSSAT v4.8 soil-profile file (`eGHR/{CC}.SOL`).",
    )
    cell_count: int = Field(
        ge=0,
        description="Number of grid cells covered by the raster (assigned to a profile).",
    )
    profile_count: int = Field(
        ge=0,
        description="Number of unique profiles in the substrate.",
    )
    soil_raster_sha256: str = Field(
        pattern=SHA256_PATTERN,
        description="SHA-256 hex digest of the raster GeoTIFF file.",
    )
    ghr_db_sha256: str = Field(
        pattern=SHA256_PATTERN,
        description="SHA-256 hex digest of the GHR.db file.",
    )
    sol_sha256: str = Field(
        pattern=SHA256_PATTERN,
        description="SHA-256 hex digest of the .SOL file.",
    )


def _profile_dedup_key(profile: SoilProfile) -> bytes:
    """Deterministic dedup key for a SoilProfile.

    Two profiles with byte-identical layer parameters share a key. The
    key drives the canonical cell-to-profile-id assignment so unrelated
    profile attributes (e.g., ``profile_id`` strings, metadata dicts)
    cannot accidentally produce different pixel ids for the same soil.
    """

    def _layer_tuple(layer: SoilLayer) -> tuple:
        return (
            layer.depth_top,
            layer.depth_bottom,
            layer.sand,
            layer.clay,
            layer.silt,
            layer.organic_carbon,
            layer.bulk_density,
            layer.ph,
            layer.field_capacity,
            layer.wilting_point,
            layer.saturated_wc,
        )

    layers = tuple(_layer_tuple(layer) for layer in profile.layers)
    blob = repr(layers).encode("utf-8")
    return hashlib.sha256(blob).digest()


def assign_cell_to_profile_id(
    grid: SpatialGrid,
    profiles_by_cell: Mapping[int, SoilProfile],
) -> Tuple[Dict[int, int], Dict[int, SoilProfile]]:
    """Compute the canonical cell-to-profile-id assignment.

    This is the canonical-source helper for the cross-boundary invariant
    "every artifact agrees on which cell maps to which profile id"
    (durable lesson §24). All three writers consume the result of this
    one function; no writer ever recomputes the assignment locally.

    Profile ids are 1-based (``0`` is reserved as the GeoTIFF nodata
    value). Cells iterate in ascending ``cell_id`` order so the assigned
    ids are deterministic across reruns; identical input always yields
    the same id mapping.

    Hydraulic properties on every layer are normalized up front (a single
    pre-dedup call to :meth:`SoilLayer.estimate_hydraulic_properties`
    when any of the wilting/field-capacity/saturated-water-content fields
    is missing). Without this step, the canonical SOL writer would
    populate those fields on the representative profile only, leaving
    cell-level duplicates unchanged; subsequent reruns would then
    compute different dedup keys and yield different profile ids — a
    silent break of the idempotency contract that the structural test
    ``test_build_eghr_substrate_is_idempotent_with_unset_hydraulics``
    pins.

    Args:
        grid: Cell roster.
        profiles_by_cell: Mapping ``cell_id`` → :class:`SoilProfile`.
            Cells absent from this mapping are not assigned a profile
            (the raster pixel will be nodata).

    Returns:
        Tuple of ``(cell_id → profile_id, profile_id → SoilProfile)``.
    """
    # Step 1: pre-normalize hydraulic fields so dedup keys are stable
    # across reruns. Touches every layer once; idempotent if already
    # normalized (the model's own short-circuit on populated fields).
    for profile in profiles_by_cell.values():
        for layer in profile.layers:
            if (
                layer.wilting_point is None
                or layer.field_capacity is None
                or layer.saturated_wc is None
            ):
                layer.estimate_hydraulic_properties()

    # Step 2: deterministic dedup pass over cells in ascending id order.
    profile_id_by_dedup_key: Dict[bytes, int] = {}
    profiles_by_id: Dict[int, SoilProfile] = {}
    cell_to_profile_id: Dict[int, int] = {}

    for cell in sorted(grid.cells, key=lambda c: c.cell_id):
        profile = profiles_by_cell.get(cell.cell_id)
        if profile is None:
            continue
        key = _profile_dedup_key(profile)
        if key not in profile_id_by_dedup_key:
            new_id = len(profile_id_by_dedup_key) + 1  # 1-based; 0 == nodata
            profile_id_by_dedup_key[key] = new_id
            profiles_by_id[new_id] = profile
        cell_to_profile_id[cell.cell_id] = profile_id_by_dedup_key[key]

    return cell_to_profile_id, profiles_by_id


# Sprint E.3 fixup +15 (F-BN Boundary 3) — variable_key → SoilLayer field
# mapping. The cockpit registry at
# ``prismpy.standards.override_value_shapes.OVERRIDE_VALUE_SHAPES`` uses
# canonical variable_keys (e.g., ``soil_sand_pct``) that map to attribute
# names on :class:`SoilLayer` (e.g., ``sand``). Override targets the
# top-most layer in the profile (typically the surface 0–30 cm horizon
# DSSAT cares about most for emergence / early-stage water relations).
# Per durable §24 canonical-source-or-pin: the mapping lives here once.
_SOIL_OVERRIDE_VARIABLE_KEYS: Dict[str, str] = {
    "soil_sand_pct": "sand",
    "soil_clay_pct": "clay",
    "soil_organic_carbon_pct": "organic_carbon",
    "soil_ph": "ph",
    "soil_bulk_density_g_cm3": "bulk_density",
}


def _apply_soil_overrides_to_assignment(
    *,
    cell_to_profile_id: Dict[int, int],
    profiles_by_id: Dict[int, SoilProfile],
    sidecar: CockpitOverrideSidecar,
) -> Tuple[Dict[int, int], Dict[int, SoilProfile]]:
    """Synthesize per-cell soil profiles for cells with sidecar overrides.

    Sprint E.3 fixup +15 (F-BN Boundary 3). The cockpit sidecar maps
    ``(cell_id, variable_key)`` to an override value (e.g., persona
    documents ``soil_sand_pct=88.0`` on cell 4374122 based on a
    cited Mathon et al. 2002 field measurement). The eGHR substrate
    builder deduplicates cells with identical raw soil profiles into
    a shared profile id, so an in-place mutation of the shared profile
    would silently affect every other cell using that profile —
    honest-signal floor per ``feedback_no_data_cooking.md``.

    This helper splits the affected cells off into per-cell synthetic
    profiles: each overridden cell gets a deepcopy of its current
    profile with the override applied to the top layer, assigned a
    fresh profile id starting at ``max(existing) + 1``. The original
    profile stays intact for every non-overridden cell that shares
    it.

    Args:
        cell_to_profile_id: Output of :func:`assign_cell_to_profile_id`
            — mapping ``cell_id → profile_id``.
        profiles_by_id: Output of :func:`assign_cell_to_profile_id`
            — registry of deduplicated profiles keyed by id.
        sidecar: Validated cockpit override sidecar carrying
            ``(cell_id, variable_key, value)`` triples per AC-E3-7.

    Returns:
        Tuple ``(cell_to_profile_id, profiles_by_id)`` with the
        overridden cells split into synthetic profiles. Returns the
        inputs unchanged if no sidecar entries match a known soil
        variable_key on a cell present in ``cell_to_profile_id``.

    The helper is PURE — does NOT mutate the input dicts. Caller
    reassigns the result. ``copy.deepcopy`` is used on the
    :class:`SoilProfile` instances so mutating a layer attribute on
    the synthesized copy cannot leak back to the shared registry.
    """
    # Group sidecar entries by cell_id, keep only entries whose
    # variable_key is in the soil registry (climate entries skip this
    # helper). ``cell_id`` arrives as str in the sidecar; the
    # cell_to_profile_id keys are int — defensive int-cast per
    # durable §27 producer-consumer parity.
    soil_overrides_by_cell: Dict[int, Dict[str, float]] = {}
    for entry in sidecar.overrides:
        if entry.variable_key not in _SOIL_OVERRIDE_VARIABLE_KEYS:
            continue
        try:
            cell_id_int = int(entry.cell_id)
        except (TypeError, ValueError):
            continue
        if cell_id_int not in cell_to_profile_id:
            continue
        bucket = soil_overrides_by_cell.setdefault(cell_id_int, {})
        bucket[entry.variable_key] = float(entry.value)

    if not soil_overrides_by_cell:
        return cell_to_profile_id, profiles_by_id

    # Build new dicts so the helper is non-mutating (caller's
    # references stay valid until reassignment).
    new_cell_to_profile_id = dict(cell_to_profile_id)
    new_profiles_by_id = dict(profiles_by_id)
    next_profile_id = (max(new_profiles_by_id) if new_profiles_by_id else 0) + 1

    for cell_id_int, overrides_for_cell in sorted(soil_overrides_by_cell.items()):
        current_profile_id = new_cell_to_profile_id[cell_id_int]
        base_profile = new_profiles_by_id.get(current_profile_id)
        if base_profile is None or not base_profile.layers:
            # Defensive — should never happen given assign_cell_to_profile_id
            # only writes (cell, profile) when the profile exists with
            # at least one layer, but guards prevent a crash if the
            # invariant ever drifts.
            continue
        synthesized = copy.deepcopy(base_profile)
        top_layer = synthesized.layers[0]
        for variable_key, override_value in sorted(overrides_for_cell.items()):
            attr_name = _SOIL_OVERRIDE_VARIABLE_KEYS[variable_key]
            setattr(top_layer, attr_name, override_value)
        # Recompute silt if sand or clay changed so the layer's
        # sand+clay+silt=100 invariant holds (matches the
        # SoilLayer.__post_init__ behavior on construction).
        if "soil_sand_pct" in overrides_for_cell or "soil_clay_pct" in overrides_for_cell:
            top_layer.silt = max(
                0.0,
                100.0 - (top_layer.sand or 0.0) - (top_layer.clay or 0.0),
            )
        # Force re-derivation of hydraulic properties on the
        # overridden layer so wilting_point / field_capacity /
        # saturated_wc track the new texture (the builder later
        # recomputes via estimate_hydraulic_properties if these
        # are None; clearing here triggers that path).
        top_layer.wilting_point = None
        top_layer.field_capacity = None
        top_layer.saturated_wc = None

        new_profiles_by_id[next_profile_id] = synthesized
        new_cell_to_profile_id[cell_id_int] = next_profile_id
        logger.info(
            "Applied cockpit soil override to cell %d: profile %d → "
            "synthetic profile %d (overrides=%s)",
            cell_id_int,
            current_profile_id,
            next_profile_id,
            sorted(overrides_for_cell.keys()),
        )
        next_profile_id += 1

    return new_cell_to_profile_id, new_profiles_by_id


def build_eghr_substrate(
    grid: SpatialGrid,
    profiles_by_cell: Mapping[int, SoilProfile],
    country_code: str,
    region: Region,
    output_dir: Path,
    cockpit_override_sidecar: Optional[CockpitOverrideSidecar] = None,
) -> EghrSubstrateResult:
    """Build the three eGHR substrate artifacts in ``output_dir``.

    Idempotent: re-running with identical inputs regenerates byte-equal
    artifacts. The cell-to-profile-id assignment is the single canonical
    source for all three writers; raster pixel ids, ``profile_map`` rows,
    and ``.SOL`` ``*<profile>`` headers reference the same id space.

    Args:
        grid: Cell roster used to dimension the raster and assign rows.
        profiles_by_cell: Mapping ``cell_id`` → :class:`SoilProfile`. Any
            cell not in this map yields a nodata pixel. Identical
            profiles (per layer-parameter equality) are deduplicated.
        country_code: Two-letter ISO code that prefixes every profile
            name and names the ``.SOL`` file (``{country_code}.SOL``).
        region: Region carrying the human-readable name plus the ISO3
            country code; used by the canonical SOL writer to fill the
            file header and ``@SITE`` lines.
        output_dir: Existing or to-be-created directory. The builder
            creates ``raster/`` and ``eGHR/`` subdirectories inside.

    Returns:
        :class:`EghrSubstrateResult` with paths, counts, and SHA-256
        hashes for each artifact.
    """
    output_dir = Path(output_dir)
    raster_dir = output_dir / "raster"
    eghr_dir = output_dir / "eGHR"
    raster_dir.mkdir(parents=True, exist_ok=True)
    eghr_dir.mkdir(parents=True, exist_ok=True)

    raster_path = raster_dir / "soil.tif"
    db_path = eghr_dir / "GHR.db"
    sol_path = eghr_dir / f"{country_code}.SOL"

    # Step 1: canonical cell-to-profile assignment + deduplicated profile registry.
    cell_to_profile_id, profiles_by_id = assign_cell_to_profile_id(
        grid, profiles_by_cell
    )

    # Step 1.5: Sprint E.3 fixup +15 (F-BN Boundary 3) — apply per-cell
    # soil overrides from the cockpit sidecar. The override semantic is
    # "synthesize a new profile for the overridden cell and re-point its
    # cell_to_profile_id entry at the new profile" — over-broad in-place
    # mutation of a shared profile would silently affect every other cell
    # using that profile (honest-signal floor per
    # ``feedback_no_data_cooking.md``).
    if cockpit_override_sidecar is not None and cockpit_override_sidecar.overrides:
        cell_to_profile_id, profiles_by_id = _apply_soil_overrides_to_assignment(
            cell_to_profile_id=cell_to_profile_id,
            profiles_by_id=profiles_by_id,
            sidecar=cockpit_override_sidecar,
        )

    # Step 2: write the .SOL via the canonical writer. The eGHR substrate
    # uses a different file-header suffix and source label so a manual
    # inspector can tell a per-package substrate apart from a CRAFT
    # HWSD-derived package without grepping for column conventions.
    profile_id_to_name = write_dssat_sol(
        soil_path=sol_path,
        profiles_by_id=profiles_by_id,
        country_code=country_code,
        region=region,
        file_header_suffix="(eGHR per-package substrate)",
        source_label_for_id=lambda pid: f"eGHR profile {pid}",
    )

    # Step 3: GeoTIFF profile-id raster aligned to the grid.
    _write_soil_raster(
        raster_path=raster_path,
        grid=grid,
        cell_to_profile_id=cell_to_profile_id,
    )

    # Step 4: SQLite GHR.db with profile_map(id, profile).
    _write_ghr_db(
        db_path=db_path,
        profile_id_to_name=profile_id_to_name,
    )

    logger.info(
        "Built eGHR substrate at %s: %d cells -> %d unique profiles "
        "(raster=%s db=%s sol=%s)",
        output_dir,
        len(cell_to_profile_id),
        len(profiles_by_id),
        raster_path.name,
        db_path.name,
        sol_path.name,
    )

    return EghrSubstrateResult(
        soil_raster_path=raster_path,
        ghr_db_path=db_path,
        sol_path=sol_path,
        cell_count=len(cell_to_profile_id),
        profile_count=len(profiles_by_id),
        soil_raster_sha256=_sha256(raster_path),
        ghr_db_sha256=_sha256(db_path),
        sol_sha256=_sha256(sol_path),
    )


_RASTER_DTYPE = "uint32"
_RASTER_NODATA = 0
_MAX_PROFILE_ID = 2**32 - 1


def _write_soil_raster(
    raster_path: Path,
    grid: SpatialGrid,
    cell_to_profile_id: Mapping[int, int],
) -> None:
    """Write the cell-to-profile-id assignment as a single-band GeoTIFF.

    The pixel data type is ``uint32`` so the substrate can encode up to
    ~4.3 billion unique profiles without silent overflow (uint16 wraps
    above 65 535 and would split a single profile's coverage between two
    raster ids while the database keeps the larger id intact). Pixel
    value ``0`` means nodata.

    The georeferencing transform is computed from the spatial extent of
    the cells actually present in ``grid.cells``, not from
    ``grid.bounds``. Pipelines routinely filter cells (centroid_strict,
    share-percent, exclude_cells); when that happens, ``grid.cells``
    is the post-filter roster while ``grid.bounds`` still describes the
    original region. Aligning the raster to the cell-extent bounds keeps
    PYTHIA's cell-center sampling pointing at the correct pixel.
    """
    if not grid.cells:
        raise ValueError("Cannot write eGHR raster: grid has no cells.")

    if cell_to_profile_id:
        max_id = max(cell_to_profile_id.values())
        if max_id > _MAX_PROFILE_ID:
            raise ValueError(
                f"eGHR substrate has {max_id} unique profiles; raster dtype "
                f"{_RASTER_DTYPE} can only encode up to {_MAX_PROFILE_ID}."
            )

    n_rows = grid.n_rows
    n_cols = grid.n_cols
    min_row = min(cell.row for cell in grid.cells)
    min_col = min(cell.col for cell in grid.cells)

    data = np.zeros((n_rows, n_cols), dtype=np.uint32)
    for cell in grid.cells:
        profile_id = cell_to_profile_id.get(cell.cell_id)
        if profile_id is None:
            continue
        data[cell.row - min_row, cell.col - min_col] = profile_id

    # Cell-extent bounds (centers ± half-increment). Using these instead
    # of ``grid.bounds`` is what keeps the raster's pixel grid aligned
    # with the cells that actually carry profile data even when the
    # caller filtered the grid down from a larger original region.
    half_inc = grid.increment_deg / 2.0
    min_lat = min(cell.lat for cell in grid.cells) - half_inc
    max_lat = max(cell.lat for cell in grid.cells) + half_inc
    min_lon = min(cell.lon for cell in grid.cells) - half_inc
    max_lon = max(cell.lon for cell in grid.cells) + half_inc

    transform = from_bounds(
        west=min_lon,
        south=min_lat,
        east=max_lon,
        north=max_lat,
        width=n_cols,
        height=n_rows,
    )

    crs = grid.bounds.crs if grid.bounds is not None else "EPSG:4326"

    profile = {
        "driver": "GTiff",
        "height": n_rows,
        "width": n_cols,
        "count": 1,
        "dtype": _RASTER_DTYPE,
        "crs": crs,
        "transform": transform,
        "nodata": _RASTER_NODATA,
        "compress": "deflate",
        "tiled": False,
    }

    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(data, 1)


def _write_ghr_db(
    db_path: Path,
    profile_id_to_name: Mapping[int, str],
) -> None:
    """Write the GHR.db SQLite database with the canonical schema.

    Schema: ``profile_map(id INTEGER PRIMARY KEY, profile TEXT NOT NULL)``.
    Inserts run in ascending ``id`` order so the on-disk page layout is
    deterministic across reruns (combined with a fixed SQLite version
    this gives byte-identical databases for byte-identical inputs).
    """
    if db_path.exists():
        # Recreate so the file's contents reflect only the new inputs.
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE profile_map ("
            "id INTEGER PRIMARY KEY, "
            "profile TEXT NOT NULL"
            ")"
        )
        for profile_id in sorted(profile_id_to_name.keys()):
            cursor.execute(
                "INSERT INTO profile_map (id, profile) VALUES (?, ?)",
                (profile_id, profile_id_to_name[profile_id]),
            )
        conn.commit()
    finally:
        conn.close()


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at ``path``."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
