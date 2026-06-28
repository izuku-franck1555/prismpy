"""
prismpy: A unified data-to-model translation framework for spatial crop modeling.

This framework formalizes the data-to-model translation process for generating
model-ready inputs compatible with multiple spatial crop modeling platforms:
- SARRA-Py (SARRA-H model)
- CRAFT (DSSAT-based regional forecasting)
- PYTHIA (Spatial DSSAT)
- ACEA (AquaCrop)

The framework provides:
1. Formalized translation methodology with documented decision rules
2. Reproducible and transparent workflows with full provenance tracking
3. Model-agnostic design supporting multiple platforms from a single workflow
"""

__version__ = "0.1.0"
__author__ = "Crop Modeling Research Team"

__all__ = [
    "ProjectConfig",
    "TranslationPipeline",
    "__version__",
]


def __getattr__(name: str):
    """Lazy imports — heavy modules load only when accessed."""
    if name == "ProjectConfig":
        from prismpy.config.schema import ProjectConfig
        return ProjectConfig
    if name == "TranslationPipeline":
        from prismpy.pipeline.executor import TranslationPipeline
        return TranslationPipeline
    raise AttributeError(f"module 'prismpy' has no attribute {name!r}")


# Mount the local-first GADM transport at import so the in-process web worker
# (AppConfig.ready) and the standalone CLI both inherit the pinned-dataset
# resilience. Graceful: an unset/missing gpkg falls through to the network and
# never imports the geo stack; an unexpected error never breaks `import prismpy`.
try:
    from prismpy.gadm_local import mount_local_gadm as _mount_local_gadm
    _mount_local_gadm()
except Exception:  # pragma: no cover - defensive import-time guard
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "GADM local-adapter mount skipped (unexpected error).", exc_info=True)
