"""
Manifest Generation for prismpy packages.

Creates manifest.json files with file inventory and SHA256 checksums
for reproducibility and integrity verification.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def derive_boundary_label(
    resolved_source: str,
    gadm_level: Optional[int],
) -> Tuple[str, str]:
    """Derive (label, description) for the boundary inclusion field
    on a package manifest and the corresponding README cells.

    The pipeline executor records the RESOLVED boundary source on
    the runtime ``Region`` object after any retrieve-stage fallback
    fires. Manifest writers must read that resolved value and pass
    it here, along with the configured GADM admin level. The level
    is only emitted when the resolved source is GADM; otherwise the
    label / description describe the actual on-disk boundary
    artifact (a manual bounding box, a shapefile, or — when the
    resolved value is the runtime alias ``manual_bounds`` produced
    by a GADM-failed-fallback at retrieve time — the same manual
    label as for an explicit manual configuration).

    Args:
        resolved_source: the runtime-resolved boundary source string.
            Expected values: ``"gadm"``, ``"manual"``,
            ``"manual_bounds"``, ``"shapefile"``. Unknown values
            raise ``ValueError`` so a future ``BoundarySource`` enum
            extension surfaces at sprint-time rather than as a
            silent fallthrough into the manual label.
        gadm_level: the configured GADM admin level. Honored only
            when ``resolved_source == "gadm"``; ignored (and may be
            ``None``) for every other source. ``None`` is also
            tolerated under GADM with a fallback to admin level 2 —
            the same default the BoundaryConfig schema uses.

    Returns:
        A ``(label, description)`` tuple suitable for the manifest's
        ``data_sources.boundaries`` field and the README's boundary
        row.

    Raises:
        ValueError: if ``resolved_source`` is not one of the four
            known values. The message names the offending source so
            the caller can map it to a new branch in this helper.
    """
    if resolved_source == "gadm":
        level = gadm_level if gadm_level is not None else 2
        return (
            f"GADM v4.1 admin level {level}",
            "Official administrative boundaries",
        )
    if resolved_source in ("manual", "manual_bounds"):
        return ("Bounding box", "Manual coordinate bounds")
    if resolved_source == "shapefile":
        return ("Custom shapefile", "User-provided boundary")
    raise ValueError(
        f"Unknown boundary source: {resolved_source!r}. "
        "Update derive_boundary_label() when adding a "
        "BoundarySource enum value."
    )


def compute_sha256(file_path: Union[str, Path]) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        file_path: Path to file

    Returns:
        SHA256 hex digest string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_file_info(
    file_path: Union[str, Path],
    base_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Get file information including size and checksum.

    Args:
        file_path: Path to file
        base_path: Base path for relative path computation

    Returns:
        Dictionary with file info
    """
    file_path = Path(file_path)
    rel_path = str(file_path.relative_to(base_path)) if base_path else str(file_path)

    return {
        "path": rel_path,
        "sha256": compute_sha256(file_path),
        "size_bytes": file_path.stat().st_size,
        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    }


def collect_files_with_checksums(
    directory: Union[str, Path],
    patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Collect all files in directory with their checksums.

    Args:
        directory: Root directory to scan
        patterns: Optional glob patterns to include (default: all files)
        exclude_patterns: Optional patterns to exclude

    Returns:
        List of file info dictionaries
    """
    directory = Path(directory)
    files_info = []

    exclude_patterns = exclude_patterns or [".DS_Store", "*.pyc", "__pycache__"]

    if patterns:
        all_files = []
        for pattern in patterns:
            all_files.extend(directory.rglob(pattern))
    else:
        all_files = [f for f in directory.rglob("*") if f.is_file()]

    for file_path in sorted(all_files):
        skip = False
        for exclude in exclude_patterns:
            if file_path.match(exclude):
                skip = True
                break

        if not skip:
            files_info.append(get_file_info(file_path, directory))

    return files_info


def create_manifest(
    package_dir: Union[str, Path],
    project_config: Dict[str, Any],
    platform: str = "sarra_py",
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a complete manifest for a package.

    Args:
        package_dir: Root directory of the package
        project_config: Project configuration dictionary
        platform: Target platform name
        additional_metadata: Optional additional metadata to include

    Returns:
        Complete manifest dictionary
    """
    package_dir = Path(package_dir)

    # Collect all files
    files = collect_files_with_checksums(package_dir)

    # Compute summary statistics
    total_size = sum(f["size_bytes"] for f in files)

    manifest = {
        "package_version": "1.0",
        "generator": "prismpy",
        "generator_version": "1.0.0",
        "platform": platform,
        "generated_at": datetime.now().isoformat(),

        # Project info from config
        "project_name": project_config.get("project_name", "unknown"),

        "region": {
            "name": project_config.get("region_name", ""),
            "country": project_config.get("country", ""),
            # The default applies only when the translator omits
            # the key entirely; an explicit ``None`` from the
            # translator (the resolved-source-discriminator path
            # for non-GADM sources) is preserved by ``dict.get``
            # because the key is present. The default value 2
            # matches the BoundaryConfig schema default for GADM
            # configs, which is the only path that reaches this
            # branch via the omit semantics.
            "gadm_level": project_config.get("gadm_level", 2),
        },

        "crop": {
            "name": project_config.get("crop_name", ""),
            "planting_doy": project_config.get("planting_doy"),
            "maturity_doy": project_config.get("maturity_doy"),
        },

        "temporal": {
            "start_year": project_config.get("start_year"),
            "end_year": project_config.get("end_year"),
            "spinup_years": project_config.get("spinup_years", 0),
        },

        "data_sources": project_config.get("data_sources", {}),

        "summary": {
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        },

        "files": files,

        "validation_status": "PENDING"
    }

    # Add bounds if available
    if "bounds_sarra_py" in project_config:
        manifest["region"]["bounds_sarra_py"] = project_config["bounds_sarra_py"]
    if "bounds_gis" in project_config:
        manifest["region"]["bounds_gis"] = project_config["bounds_gis"]

    # Merge additional metadata
    if additional_metadata:
        manifest.update(additional_metadata)

    return manifest


def save_manifest(
    manifest: Dict[str, Any],
    output_path: Union[str, Path]
) -> Path:
    """
    Save manifest to JSON file.

    Args:
        manifest: Manifest dictionary
        output_path: Path to save manifest

    Returns:
        Path to saved manifest
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return output_path


def validate_manifest(
    manifest_path: Union[str, Path],
    package_dir: Union[str, Path]
) -> Dict[str, Any]:
    """
    Validate a manifest against the actual package contents.

    Args:
        manifest_path: Path to manifest.json
        package_dir: Path to package directory

    Returns:
        Validation results dictionary
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    package_dir = Path(package_dir)

    results = {
        "valid": True,
        "checked_at": datetime.now().isoformat(),
        "missing_files": [],
        "checksum_mismatches": [],
        "extra_files": [],
    }

    # Track files listed in manifest
    manifest_files = {f["path"] for f in manifest.get("files", [])}

    # Check each file in manifest
    for file_info in manifest.get("files", []):
        file_path = package_dir / file_info["path"]

        if not file_path.exists():
            results["missing_files"].append(file_info["path"])
            results["valid"] = False
        else:
            actual_sha256 = compute_sha256(file_path)
            if actual_sha256 != file_info["sha256"]:
                results["checksum_mismatches"].append({
                    "path": file_info["path"],
                    "expected": file_info["sha256"],
                    "actual": actual_sha256
                })
                results["valid"] = False

    # Check for extra files not in manifest
    actual_files = collect_files_with_checksums(package_dir)
    for file_info in actual_files:
        if file_info["path"] not in manifest_files:
            results["extra_files"].append(file_info["path"])

    return results
