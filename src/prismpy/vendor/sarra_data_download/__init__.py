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

DEVIATIONS FROM UPSTREAM (post-vendor 2026-05-06)
=================================================

The vendored ``get_AgERA5_data.py`` carries two corrections from the
upstream SARRA-cropmodels copy. Both are documented at the
modification site with an explanatory comment so a future side-by-
side diff makes the deviations obvious. Authorization for these
modifications was recorded under the project's open-access non-
commercial mission directive (2026-05-06).

1. ``download_AgERA5_year`` now forwards ``save_path`` to every
   nested call (``download_AgERA5_data_alt``, ``extract_AgERA5_data``,
   ``convert_AgERA5_netcdf_to_geotiff``,
   ``calculate_AgERA5_ET0_and_save``). The upstream copy accepted
   ``save_path`` and silently dropped it, so each stage fell back to
   the default ``save_path="../data/"`` and the whole pipeline wrote
   to a CWD-relative tree regardless of where the caller asked.
   Forwarding lets prismpy's per-region cache directory survive the
   handoff so concurrent regions do not contaminate each other.

2. Three bare-except / broad-except sites that swallowed pipeline
   errors now re-raise after their diagnostic ``print`` so the
   caller observes the actual failure honestly:

   - ``download_AgERA5_data_alt`` — the CDS retrieve catch raises
     after printing.
   - ``extract_AgERA5_data`` — the inner zip-extraction catch is
     replaced with the unguarded extraction call; the outer broad
     catch raises after printing.
   - ``convert_AgERA5_netcdf_to_geotiff`` — the outer broad catch
     raises after printing.

   The upstream silent-skip chain produced a misleading
   ``FileNotFoundError`` deep in ``calculate_AgERA5_ET0_and_save``
   five levels removed from the real cause (typically a CDS-side
   error such as a rate limit, malformed bbox, or queue timeout).
   Re-raising at each stage preserves the failure chain so prismpy's
   executor reports the first real error instead of the cascading
   conversion-path symptom.

The canonical un-deviated upstream remains at
https://github.com/SARRA-cropmodels/SARRA_data-download (commit
``e019a35``). Future contributors who want to refresh the vendor from
upstream should diff against the upstream copy and re-apply the two
modifications above by hand; they are deliberate substrate fixes,
not vendor drift.
"""
