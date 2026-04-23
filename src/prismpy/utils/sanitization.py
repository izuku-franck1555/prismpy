"""
Filename and string sanitization utilities.

These utilities ensure cross-platform compatibility for filenames
and handle special characters in administrative names.
"""

import re
import unicodedata
from typing import Optional


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


def region_cache_key(region) -> str:
    """Stable identifier for cache / lock / path construction.

    Independent of the display name, so multiple unnamed-manual
    projects that all carry the shared display name (e.g.,
    "Unnamed study area") each get their own bbox-derived cache
    instead of colliding on the shared key. GADM regions keep
    the normalized-name key because admin names are already
    unique identifiers.

    Routing:
    - GADM / shapefile / missing boundary_source: fall through to
      `normalize_region_name(region.name)` — the display name IS
      the identity key.
    - Manual: `manual_{miny}_{maxy}_{minx}_{maxx}` (4-decimal
      precision, matches the prismweb peer-match key format).

    Accepts two shapes:
    - `Region` (post-resolution): `region.boundary_source` + `region.bounds`
      with `.minx/.miny/.maxx/.maxy`.
    - `RegionConfig` (pre-resolution): `region.boundary.source` +
      `region.boundary.manual_bounds` with the same bbox fields.
    """
    source_value = _region_boundary_source(region)
    if source_value == 'manual':
        bounds = _region_manual_bounds(region)
        if bounds is not None:
            miny, maxy, minx, maxx = bounds
            return (
                f"manual_"
                f"{miny:.4f}_{maxy:.4f}_"
                f"{minx:.4f}_{maxx:.4f}"
            )
        # Malformed manual region — fall through to name-key so
        # caller isn't blocked. Validation upstream would reject
        # this before reaching cache paths.
    return normalize_region_name(getattr(region, 'name', ''))


def _region_boundary_source(region) -> Optional[str]:
    """Extract the boundary source as a string from either shape."""
    # Post-resolution Region with optional `boundary_source`.
    direct = getattr(region, 'boundary_source', None)
    if direct:
        return direct
    # Pre-resolution RegionConfig with `boundary.source` enum.
    boundary = getattr(region, 'boundary', None)
    if boundary is None:
        return None
    source = getattr(boundary, 'source', None)
    if source is None:
        return None
    return source.value if hasattr(source, 'value') else str(source)


def _region_manual_bounds(region) -> Optional[tuple]:
    """Return (miny, maxy, minx, maxx) floats for manual regions.
    Handles both `RegionConfig.boundary.manual_bounds` and the
    post-resolution `Region.bounds` BoundingBox."""
    boundary = getattr(region, 'boundary', None)
    candidates = []
    if boundary is not None:
        manual = getattr(boundary, 'manual_bounds', None)
        if manual is not None:
            candidates.append(manual)
    # Post-resolution Region carries bounds directly; routing by
    # boundary_source ensures we only fall here for manual regions.
    bounds = getattr(region, 'bounds', None)
    if bounds is not None:
        candidates.append(bounds)
    for src in candidates:
        try:
            return (
                float(src.miny), float(src.maxy),
                float(src.minx), float(src.maxx),
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return None
