"""Vendored SARRA_data_download — open-access AgERA5 download library
that supports African crop modeling.

Only the AgERA5 capability is vendored. The upstream sibling module
``get_satellite_rainfall_estimates.py`` is NOT bundled because prismpy
does not import it (TAMSAT serves that role through a direct HTTP
path); future use would require re-vendoring with the satellite
module's own transitive deps (``joblib``) declared in pyproject.toml.

Vendored verbatim from
https://github.com/SARRA-cropmodels/SARRA_data-download with one
operational fix applied locally (AgERA5 dataset version bump from
``1_1`` to ``2_0`` after Copernicus deprecated the older version).

Original copyright: SARRA-cropmodels (2022). See ``LICENSE`` in this
directory for the original MIT license terms.

Why vendored instead of pulled from PyPI / git:
- The upstream repository ships only ``requirements.txt`` and source
  files; there is no ``pyproject.toml`` / ``setup.py``, so
  ``pip install git+https://github.com/SARRA-cropmodels/SARRA_data-download``
  fails with "does not appear to be a Python project".
- A ``file://`` path-dependency in ``pyproject.toml`` would break CI
  portability across runners.
- Vendoring under a clear attribution header preserves the open-source
  contract and gives prismpy a deterministic single-source build.

Replaces the previous ``from SARRA_data_download.X import Y`` pattern,
which depended on an editable install of the local clone at
``/Users/francktonle/Downloads/SARRA-Py-documents/SARRA_data-download/``.
That arrangement silently broke after a venv migration when the
editable install was not re-applied, so AgERA5 retrieval skipped 1/4
climate variables on every fresh run. Vendoring closes that gap.
"""
