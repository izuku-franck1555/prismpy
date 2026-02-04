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

from .provenance import (
    ProvenanceTracker,
    create_stage_record,
    create_decision,
    load_provenance,
    DEFAULT_DECISIONS
)

from .readme_generator import (
    generate_readme,
    get_readme_template
)

__all__ = [
    # Manifest
    'compute_sha256',
    'get_file_info',
    'collect_files_with_checksums',
    'create_manifest',
    'save_manifest',
    'validate_manifest',
    # Provenance
    'ProvenanceTracker',
    'create_stage_record',
    'create_decision',
    'load_provenance',
    'DEFAULT_DECISIONS',
    # README
    'generate_readme',
    'get_readme_template',
]
