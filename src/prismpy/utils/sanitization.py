"""
Filename and string sanitization utilities.

These utilities ensure cross-platform compatibility for filenames
and handle special characters in administrative names.
"""

import re
import unicodedata
from typing import Any, Mapping, Optional


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
    """Cross-repo region identity contract.

    Canonical identifier used on both sides of the prismpy/prismweb
    boundary to answer "is this the same region?" — cache directories,
    lock files, output paths (prismpy), and peer-match keys, ETA
    sampling pools, concurrent-run banner triggers (prismweb) all
    derive from this single helper. Display names (`region.name`)
    are purely metadata and never participate in the identity.

    Routing:
    - **Manual boundary**: bbox-derived key at 6-decimal precision
      (~11 cm at the equator). Two unnamed-manual projects that
      happen to share the default display name each get their own
      cache based on their drawn bbox; two named manual projects at
      the same bbox collide by design (legitimate cache reuse —
      same underlying climate data).
    - **GADM / shapefile / missing boundary source**: falls through
      to `normalize_region_name(region.name)`. Admin names are
      already unique identifiers, so the name IS the identity.

    Accepts three input shapes, so callers on either side of the
    boundary can pass what they have:
    - `Region` (post-resolution dataclass) — `region.boundary_source`
      on the object + `region.bounds.{minx,miny,maxx,maxy}`.
    - `RegionConfig` (pre-resolution Pydantic) — `region.boundary.source`
      enum + `region.boundary.manual_bounds.{minx,...,maxy}`.
    - `Mapping` (JSON-parsed dict, as prismweb persists) —
      `region["boundary"]["source"]` string + the same manual_bounds
      dict shape. Prismweb's persisted `config.region` feeds through
      this branch without adapter boilerplate.

    Negative-zero canonicalization: Python formats `-0.0` as
    `"-0.000000"` with `.6f`, which differs textually from
    `"0.000000"` even though the values are mathematically equal.
    For West African users (Ghana/Togo/Benin on the prime meridian)
    and equatorial users, a bbox edge at exactly 0 can surface
    `-0.0` from JS/GeoJSON serialization. Each component is
    normalized with `_canon_zero` before formatting so both variants
    produce the same key.
    """
    source_value = _region_boundary_source(region)
    if source_value == 'manual':
        bounds = _region_manual_bounds(region)
        if bounds is None:
            # Malformed manual region — DO NOT fall through to
            # name-key. Silent fallback would collapse unrelated
            # manual regions (two different bboxes both labeled
            # "Unnamed study area" but with missing / malformed
            # manual_bounds fields) onto the same cache/lock
            # path, recreating exactly the cross-tenant collision
            # the V2-22b/P.2 unification was meant to eliminate.
            # `source == 'manual'` is a hard contract: the caller
            # promised manual bounds, so failing to parse them is
            # a version-skew bug at the call site, not a graceful
            # fallback moment.
            raise ValueError(
                "region_cache_key: manual boundary requires "
                "parseable manual_bounds (miny, maxy, minx, maxx); "
                f"got boundary={_attr_or_key(region, 'boundary', None)!r} "
                f"bounds={_attr_or_key(region, 'bounds', None)!r}"
            )
        miny, maxy, minx, maxx = (_canon_zero(v) for v in bounds)
        return (
            f"manual_"
            f"{miny:.6f}_{maxy:.6f}_"
            f"{minx:.6f}_{maxx:.6f}"
        )
    key = normalize_region_name(_attr_or_key(region, 'name', '') or '')
    if not key:
        # Strict contract: the helper is the CROSS-REPO IDENTITY
        # CONTRACT, so "I don't know what this region is" must
        # fail loudly, not silently produce a degenerate empty-key
        # cache path that would collide with every other broken
        # input. Catches callers that accidentally pass a raw
        # string (old-schema lookup) or a dict missing both
        # `boundary` and `name` fields.
        raise ValueError(
            "region_cache_key requires a Region / RegionConfig / "
            "Mapping with either `boundary.source='manual'` + "
            "`manual_bounds`, or a non-empty `name`; got "
            f"{type(region).__name__}: {region!r}"
        )
    return key


def _canon_zero(val: float) -> float:
    """Canonicalize -0.0 → 0.0 so `.6f` formatting is stable."""
    return 0.0 if val == 0 else float(val)


def _attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` off either an object (getattr) or a Mapping (dict)."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _region_boundary_source(region) -> Optional[str]:
    """Extract the boundary source as a string from any supported shape."""
    # Post-resolution Region with optional `boundary_source` (direct),
    # or a dict that was flattened at persist time.
    direct = _attr_or_key(region, 'boundary_source', None)
    if direct:
        return direct if isinstance(direct, str) else str(direct)
    # Pre-resolution RegionConfig with `boundary.source` enum, or
    # prismweb's persisted dict with nested `boundary.source` string.
    boundary = _attr_or_key(region, 'boundary', None)
    if boundary is None:
        return None
    source = _attr_or_key(boundary, 'source', None)
    if source is None:
        return None
    # Enum → `.value`; string passes through; anything else stringifies.
    return source.value if hasattr(source, 'value') else str(source)


def _region_manual_bounds(region) -> Optional[tuple]:
    """Return (miny, maxy, minx, maxx) floats for manual regions.
    Handles the Pydantic `RegionConfig.boundary.manual_bounds`, the
    post-resolution `Region.bounds` BoundingBox, and the Mapping-
    shaped equivalents used by prismweb's JSON-persisted config."""
    candidates = []
    boundary = _attr_or_key(region, 'boundary', None)
    if boundary is not None:
        manual = _attr_or_key(boundary, 'manual_bounds', None)
        if manual is not None:
            candidates.append(manual)
    # Post-resolution Region / dict carries bounds directly; routing
    # by boundary_source ensures we only fall here for manual regions.
    bounds = _attr_or_key(region, 'bounds', None)
    if bounds is not None:
        candidates.append(bounds)
    for src in candidates:
        try:
            return (
                float(_attr_or_key(src, 'miny')),
                float(_attr_or_key(src, 'maxy')),
                float(_attr_or_key(src, 'minx')),
                float(_attr_or_key(src, 'maxx')),
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return None
