"""External-data source clients used by prismpy package generators.

Each submodule wraps a single upstream data provider with a thin,
prismpy-shaped facade so the rest of the pipeline talks to a stable
internal API rather than the upstream library's surface.

Currently exposed:

* :mod:`prismpy.data_sources.gadm` — Global Administrative Areas
  boundary downloader.
* :mod:`prismpy.data_sources.isimip3b` — ISIMIP3b bias-adjusted daily
  climate data client (used by the scenario-package generator).
"""
