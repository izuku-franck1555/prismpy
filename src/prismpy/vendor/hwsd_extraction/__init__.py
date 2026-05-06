"""Vendored ``hwsd_extraction`` — HWSD v2.0 BIL+MDB soil-property
extraction helper used by the ACEA translator path.

Vendored from the local CIRAD ACEA toolkit at
``DATA-TO-MODEL-TRANSLATION/ACEA/utils/hwsd_extraction.py``. The ACEA
toolkit ships without an upstream ``LICENSE`` file and is not published
on PyPI; it is an internal CIRAD helper distributed as part of the
ACEA crop-modelling pipeline. Vendoring under
``prismpy.vendor.hwsd_extraction`` keeps the ACEA HWSD path runnable
from a fresh ``pip install prismpy`` instead of requiring the ACEA
toolkit to live alongside prismpy at a hard-coded path.

Replaces the previous ``sys.path.insert(0, '../../../ACEA/utils')``
+ ``from hwsd_extraction import extract_hwsd_soil_data`` shim, which
silently fell through to ``return None`` whenever the ACEA toolkit was
not co-located with prismpy. After vendoring, the ACEA path resolves
deterministically across environments.

Re-export ``extract_hwsd_soil_data`` so callers can use a stable
``from prismpy.vendor.hwsd_extraction import extract_hwsd_soil_data``
import path.
"""
from prismpy.vendor.hwsd_extraction._module import extract_hwsd_soil_data  # noqa: F401

__all__ = ["extract_hwsd_soil_data"]
