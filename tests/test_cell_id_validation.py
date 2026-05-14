"""F-DL Pin DL-3 — canonical-helper unit table.

Exercises ``is_real_climate_cell_id`` across every documented
accept / reject case so a future refactor cannot silently widen
or narrow the predicate without tripping a test.

Sibling-file to ``tests/test_cell_summary_schema.py`` (which
covers the other ``prismpy.cells`` canonical module).

Per F-DL contract §D Pin DL-3.
"""
from __future__ import annotations

import pytest

from prismpy.cells.cell_id_validation import is_real_climate_cell_id


# ── Accepts ──────────────────────────────────────────────────────────


def test_zero_is_real_cell_id() -> None:
    """``0`` is a legitimate placeholder cell-id (see
    ``executor.py:1387`` ``_create_placeholder_soil`` and
    ``SpatialGrid.compute_cell_id_5arcmin``)."""
    assert is_real_climate_cell_id(0) is True


def test_positive_int_is_real_cell_id() -> None:
    """Real grid cells in production are positive ints
    (e.g., ``4339573`` for a Sahel cell)."""
    assert is_real_climate_cell_id(4339573) is True


def test_numpy_int64_is_real_cell_id() -> None:
    """Production pandas/xarray code may emit ``numpy.int64`` cell
    IDs. These pass ``isinstance(_, numbers.Integral)`` and so MUST
    be accepted — otherwise the predicate would silently drop
    legitimate cells coming through downstream of NumPy joins."""
    np = pytest.importorskip("numpy")
    assert is_real_climate_cell_id(np.int64(5)) is True
    assert is_real_climate_cell_id(np.int32(5)) is True
    assert is_real_climate_cell_id(np.int64(0)) is True


# ── Rejects ──────────────────────────────────────────────────────────


def test_negative_int_is_rejected() -> None:
    """``-1`` is the harmonize-stage sentinel placeholder; the
    cockpit + translator filters MUST drop it before emit."""
    assert is_real_climate_cell_id(-1) is False
    assert is_real_climate_cell_id(-9999) is False


def test_string_cell_id_is_rejected_even_when_numeric_looking() -> None:
    """Stringified cell-ids (``"3799258"``) come from JSON paths.
    They must be explicitly coerced before reaching this predicate
    — silent acceptance would mix str + int through the same
    ``sorted()`` and crash downstream."""
    assert is_real_climate_cell_id("3799258") is False
    assert is_real_climate_cell_id("0") is False


def test_path_dict_string_keys_are_rejected() -> None:
    """The actual production bug: SARRA-Py ``_load_climate_data``
    emits a path-dict with these string keys. The filter must
    reject them so the union with int-keyed soil keeps the type
    uniform."""
    assert is_real_climate_cell_id("rainfall_dir") is False
    assert is_real_climate_cell_id("agera5_dir") is False
    assert is_real_climate_cell_id("metadata") is False


def test_none_is_rejected() -> None:
    """``None`` from a missing-key fallback must not pass."""
    assert is_real_climate_cell_id(None) is False


def test_float_is_rejected() -> None:
    """``3.14`` is not an Integral; must reject."""
    assert is_real_climate_cell_id(3.14) is False
    assert is_real_climate_cell_id(0.0) is False


def test_bool_is_rejected_explicitly() -> None:
    """``isinstance(True, int)`` returns ``True`` in Python — without
    the explicit guard, ``True`` would pass the Integral + >=0 check.
    The helper rejects both ``True`` and ``False`` so a stray boolean
    in a key collection can't masquerade as cell-id ``1`` or ``0``."""
    assert is_real_climate_cell_id(True) is False
    assert is_real_climate_cell_id(False) is False


def test_collection_types_are_rejected() -> None:
    """list / tuple / bytes / set — none represent a single cell-id."""
    assert is_real_climate_cell_id([1, 2]) is False
    assert is_real_climate_cell_id((1, 2)) is False
    assert is_real_climate_cell_id(b"7") is False
    assert is_real_climate_cell_id({1}) is False
