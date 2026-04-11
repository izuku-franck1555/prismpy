"""
Packaging utilities for prismpy.

This module provides tools for creating standardized, self-documenting
data packages with manifest files, provenance tracking, and README generation.
"""

from .manifest import (
    compute_sha256,
    get_file_info,
    collect_files_with_checksums,
    create_manifest,
    save_manifest,
    validate_manifest
)

from .readme_generator import (
    generate_readme,
    get_readme_template
)

# V2-20: packaging/provenance.py (System B) deleted. All provenance
# is now handled by prismpy.provenance.tracker (System A) with
# dual-output (rich + stages-compat). The hybrid save in
# executor._execute_package distributes provenance files to each
# platform's output directory.

__all__ = [
    # Manifest
    'compute_sha256',
    'get_file_info',
    'collect_files_with_checksums',
    'create_manifest',
    'save_manifest',
    'validate_manifest',
    # README
    'generate_readme',
    'get_readme_template',
]
