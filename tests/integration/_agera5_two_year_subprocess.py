"""F-DP AC-DP-4 / Pin DP-4 subprocess entry-point.

Imports the real vendor ``download_AgERA5_year`` per §Y.3 corrected
spec (VEN-9: 5-arg signature, no ``variables`` parameter — vendor
uses the module-level constant at VEN-8 / line 532). Spawned by
``test_agera5_two_year_conversion_no_sigsegv`` via ``subprocess.run``
so any SIGSEGV in the conversion C-layer surfaces as ``returncode in
(-11, 139)`` to the parent pytest process rather than crashing pytest
itself.

Production caller surface mirrors this exact shape: ``AgERA5Source.
_download_agera5`` wraps a multi-year loop calling
``download_AgERA5_year`` per year. The subprocess inlines the loop
for test simplicity.

Per F-DP contract LOCKED cycle-4 §Z.1 + §X.6 wrapper + §Y.3
corrected subprocess.
"""
from __future__ import annotations

import sys
from pathlib import Path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <tmp_save_path>",
            file=sys.stderr,
        )
        sys.exit(2)

    save_path = Path(sys.argv[1])
    save_path.mkdir(parents=True, exist_ok=True)

    # Imported here (not at module top) so the import cost is paid
    # only when the subprocess actually runs, and so any import-time
    # error becomes a non-zero returncode visible to the parent
    # pytest process instead of a collection-time crash.
    from prismpy.vendor.sarra_data_download.get_AgERA5_data import (
        download_AgERA5_year,
    )

    # Per AGE-10 / VEN-1 production construction: ``area`` is dict
    # ``{region_name: bounds}`` because the vendor indexes
    # ``area[selected_area]`` at line 68. ``bounds`` is the CDS
    # ``[North, West, South, East]`` 4-tuple.
    AREA_DICT = {"maradi": [15.44, 6.28, 12.99, 8.54]}

    # Per VEN-8 / VEN-9: ``download_AgERA5_year`` uses the module-
    # level ``variables`` constant at get_AgERA5_data.py:532; the
    # function signature does NOT accept a ``variables`` parameter,
    # so we do not pass one. The constant enumerates the same six
    # AgERA5 variable tuples SARRA-Py needs.
    for query_year in (2022, 2023):
        download_AgERA5_year(
            query_year=query_year,
            area=AREA_DICT,
            selected_area="maradi",
            save_path=str(save_path),
            version="SARRA-Py",
        )

    sys.exit(0)
