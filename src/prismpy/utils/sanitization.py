"""
Filename and string sanitization utilities.

These utilities ensure cross-platform compatibility for filenames
and handle special characters in administrative names.
"""

import re
import unicodedata
from typing import Any, Optional


def remove_accents(text: str) -> str:
    """Remove accents and diacritical marks from text.

    Converts accented characters to their ASCII equivalents.
    For example: 'Ségou' -> 'Segou', 'São Paulo' -> 'Sao Paulo'

    Args:
        text: Input text with potential accents

    Returns:
        Text with accents removed
    """
    # Normalize to NFD (decomposed form) to separate base characters from accents
    normalized = unicodedata.normalize("NFD", text)

    # Remove combining diacritical marks (category 'Mn')
    without_accents = "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    )

    return without_accents


def sanitize_filename(
    filename: str,
    replacement: str = "_",
    max_length: Optional[int] = 255,
) -> str:
    """Sanitize a string for use as a filename.

    Removes or replaces characters that are invalid in filenames
    across Windows, macOS, and Linux systems.

    Args:
        filename: Original filename string
        replacement: Character to replace invalid characters with
        max_length: Maximum filename length (None for no limit)

    Returns:
        Sanitized filename safe for all platforms
    """
    # Characters forbidden in Windows filenames
    WINDOWS_FORBIDDEN = r'[<>:"/\\|?*]'

    # Control characters (ASCII 0-31)
    CONTROL_CHARS = r"[\x00-\x1f]"

    # Reserved Windows device names
    RESERVED_NAMES = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }

    # Remove accents first
    sanitized = remove_accents(filename)

    # Replace forbidden characters
    sanitized = re.sub(WINDOWS_FORBIDDEN, replacement, sanitized)
    sanitized = re.sub(CONTROL_CHARS, replacement, sanitized)

    # Replace spaces with underscores (optional but recommended)
    sanitized = sanitized.replace(" ", replacement)

    # Remove leading/trailing periods and spaces
    sanitized = sanitized.strip(". ")

    # Collapse multiple replacement characters
    if replacement:
        sanitized = re.sub(f"{re.escape(replacement)}+", replacement, sanitized)

    # Handle reserved names by adding suffix
    name_upper = sanitized.upper()
    for reserved in RESERVED_NAMES:
        if name_upper == reserved or name_upper.startswith(reserved + "."):
            sanitized = f"_{sanitized}"
            break

    # Truncate if necessary (preserve extension if present)
    if max_length and len(sanitized) > max_length:
        if "." in sanitized:
            name, ext = sanitized.rsplit(".", 1)
            max_name_len = max_length - len(ext) - 1
            sanitized = f"{name[:max_name_len]}.{ext}"
        else:
            sanitized = sanitized[:max_length]

    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed"

    return sanitized


def sanitize_for_sql(text: str) -> str:
    """Sanitize a string for use in SQL identifiers.

    Removes characters that could cause issues in database column names
    or table names.

    Args:
        text: Input text

    Returns:
        Sanitized text safe for SQL identifiers
    """
    # Remove accents
    sanitized = remove_accents(text)

    # Replace non-alphanumeric characters with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", sanitized)

    # Remove leading digits (SQL identifiers can't start with numbers)
    sanitized = re.sub(r"^[0-9]+", "", sanitized)

    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed"

    return sanitized


def sanitize_admin_name(name: str) -> str:
    """Sanitize an administrative region name for consistent use.

    Used for GADM region names to ensure consistency across platforms.

    Args:
        name: Administrative region name (e.g., "Ségou", "São Paulo")

    Returns:
        Sanitized name suitable for filenames and identifiers
    """
    # Remove accents
    sanitized = remove_accents(name)

    # Replace problematic characters but preserve some structure
    sanitized = re.sub(r"[/\\]", "-", sanitized)  # Slashes to hyphens
    sanitized = re.sub(r"[<>:\"|?*]", "", sanitized)  # Remove forbidden chars
    sanitized = sanitized.replace("'", "")  # Remove apostrophes
    sanitized = sanitized.strip()

    return sanitized


def normalize_region_name(name: str) -> str:
    """Normalize a region name to a standard format.

    Creates a lowercase, underscore-separated version suitable for
    use in file paths and identifiers.

    Args:
        name: Original region name

    Returns:
        Normalized region name
    """
    # Sanitize first
    normalized = sanitize_admin_name(name)

    # Convert to lowercase
    normalized = normalized.lower()

    # Replace spaces and hyphens with underscores
    normalized = re.sub(r"[\s-]+", "_", normalized)

    # Remove any remaining non-alphanumeric characters except underscore
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)

    # Collapse multiple underscores
    normalized = re.sub(r"_+", "_", normalized)

    return normalized.strip("_")


def region_cache_key_from_region(region: Any) -> str:
    """Cross-repo region identity contract — POST-RESOLUTION shape.

    Canonical identifier used for cache directories, lock files,
    and output paths inside prismpy. Accepts the `Region`
    dataclass (or any object carrying `.boundary_source: str`,
    `.bounds` with `.minx / .miny / .maxx / .maxy`, and `.name:
    str`). Display name (`region.name`) is metadata only for
    manual regions; it IS the identity for GADM / shapefile /
    unknown-source regions where admin names are already unique.

    Routing:
    - **Manual boundary**: bbox-derived key at 6-decimal
      precision (~11 cm at the equator). `-0.0` components
      canonicalize to `0.0` so a meridian / equator edge doesn't
      produce two different keys for the same region.
    - **Everything else**: `normalize_region_name(region.name)`.

    Strict on every field — a missing `.bounds` under a manual
    source, non-numeric bbox coordinates, or an empty `.name`
    under a non-manual source all raise `ValueError`. No silent
    name-key fallback for malformed manual inputs; no empty-key
    cache path.
    """
    boundary_source = getattr(region, 'boundary_source', None)
    if boundary_source == 'manual':
        bounds = getattr(region, 'bounds', None)
        if bounds is None:
            raise ValueError(
                "region_cache_key_from_region: manual region "
                f"requires .bounds; got {region!r}"
            )
        try:
            miny = _canon_zero(float(bounds.miny))
            maxy = _canon_zero(float(bounds.maxy))
            minx = _canon_zero(float(bounds.minx))
            maxx = _canon_zero(float(bounds.maxx))
        except (AttributeError, TypeError, ValueError) as e:
            raise ValueError(
                "region_cache_key_from_region: manual region "
                ".bounds must have numeric .minx / .miny / .maxx "
                f"/ .maxy; got {bounds!r}"
            ) from e
        return (
            f"manual_"
            f"{miny:.6f}_{maxy:.6f}_"
            f"{minx:.6f}_{maxx:.6f}"
        )
    name = getattr(region, 'name', '') or ''
    key = normalize_region_name(name)
    if not key:
        raise ValueError(
            "region_cache_key_from_region: non-manual region "
            f"requires a non-empty .name; got {region!r}"
        )
    return key


def region_cache_key_from_config(config: "RegionConfig") -> str:
    """Cross-repo region identity contract — PRE-RESOLUTION shape.

    Canonical identifier for the region persisted in project
    configuration. Accepts a validated `RegionConfig` Pydantic
    model — all the strictness this helper used to enforce
    imperatively (non-empty name, enum source, populated
    `manual_bounds` with numeric coords inside geographic ranges,
    `minx < maxx`, `miny < maxy`) is already enforced by
    `RegionConfig` + `BoundaryConfig` + `ManualBoundsConfig`
    validators at `model_validate()` time. This helper assumes a
    validated input and reads fields directly; passing an
    unvalidated dict is a caller-side mistake that Pydantic
    catches at the validate boundary upstream.

    Routing:
    - `boundary.source == BoundarySource.MANUAL`: bbox-derived
      key at 6-decimal precision (~11 cm at the equator) with
      `-0.0 → 0.0` canonicalization.
    - `boundary.source in (GADM, SHAPEFILE)`: `normalize_region_name(name)`.

    Callers should build a `RegionConfig` via
    `RegionConfig.model_validate(region_dict)` at the boundary
    they receive the dict (e.g., prismweb's view layer), then
    hand the validated model to this helper.
    """
    from prismpy.config.schema import BoundarySource

    if config.boundary.source == BoundarySource.MANUAL:
        bounds = config.boundary.manual_bounds
        miny = _canon_zero(bounds.miny)
        maxy = _canon_zero(bounds.maxy)
        minx = _canon_zero(bounds.minx)
        maxx = _canon_zero(bounds.maxx)
        return (
            f"manual_"
            f"{miny:.6f}_{maxy:.6f}_"
            f"{minx:.6f}_{maxx:.6f}"
        )
    return normalize_region_name(config.name)


def _canon_zero(val: float) -> float:
    """Canonicalize -0.0 → 0.0 so `.6f` formatting is stable.
    Python formats `-0.0` as `"-0.000000"` which is text-distinct
    from `"0.000000"`; a bbox edge on the prime meridian /
    equator could produce two different keys for the same
    region without this guard."""
    return 0.0 if val == 0 else float(val)
