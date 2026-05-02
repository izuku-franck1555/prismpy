"""Sprint D.1 AC-4 — DEFAULT_SOIL re-route + AC-1.1 source-level retirement.

Pins the contract that the HWSD source loader no longer injects
synthetic Sahel-typical profiles when SMU lookup misses. Cells
without HWSD coverage are recorded on the loader's
``unavailable_cells`` list with cause
``soil_no_hwsd_coverage``; the executor reads the list at
HARMONIZE end and routes each cell to
``data_availability='unavailable'`` with axis ``soil`` and the
recorded cause.

The tests are unit-level (no MDB / BIL files needed); they
exercise the helper plumbing directly.
"""
from __future__ import annotations

import inspect
import re

import pytest

from prismpy.sources.soil.hwsd import HWSDConfig, HWSDSource


# ---------------------------------------------------------------------------
# AC-4 — _create_default_profile returns None
# ---------------------------------------------------------------------------


def test_create_default_profile_returns_none():
    """The synthetic DEFAULT_SOIL injection is retired —
    ``_create_default_profile`` returns None so callers do not
    silently populate a profile dict with synthetic values."""
    src = HWSDSource(config=HWSDConfig())
    result = src._create_default_profile(0, 0.0, 0.0, region=None)
    assert result is None


def test_create_default_profile_docstring_documents_retirement():
    """The function's docstring names the retired contract so a
    future reader does not mis-recover the synthetic injection
    pattern."""
    src = HWSDSource(config=HWSDConfig())
    doc = inspect.getdoc(src._create_default_profile) or ""
    assert "retired" in doc.lower() or "Sprint D.1" in doc
    assert "unavailable" in doc.lower()


# ---------------------------------------------------------------------------
# AC-4 — _record_unavailable populates the loader-level list
# ---------------------------------------------------------------------------


def test_record_unavailable_populates_list_with_default_cause():
    """``_record_unavailable`` records each cell with the default
    cause ``soil_no_hwsd_coverage``."""
    src = HWSDSource(config=HWSDConfig())
    assert src.unavailable_cells == []
    src._record_unavailable(cell_id=0)
    src._record_unavailable(cell_id=1)
    src._record_unavailable(cell_id=2)
    assert src.unavailable_cells == [
        {"cell_id": 0, "cause": "soil_no_hwsd_coverage"},
        {"cell_id": 1, "cause": "soil_no_hwsd_coverage"},
        {"cell_id": 2, "cause": "soil_no_hwsd_coverage"},
    ]


def test_record_unavailable_accepts_explicit_cause():
    """A caller can pass an explicit cause (e.g., a future cause
    string) instead of relying on the default."""
    src = HWSDSource(config=HWSDConfig())
    src._record_unavailable(cell_id=5, cause="soil_texture_invalid")
    assert src.unavailable_cells == [
        {"cell_id": 5, "cause": "soil_texture_invalid"},
    ]


# ---------------------------------------------------------------------------
# AC-4 — 4 historical call sites no longer assign synthetic profiles
# ---------------------------------------------------------------------------


def _read_hwsd_source() -> str:
    import prismpy.sources.soil.hwsd as mod
    return inspect.getsource(mod)


def test_four_call_sites_do_not_assign_synthetic_profile():
    """The four historical injection sites at hwsd.py 177 / 277 /
    298 / 425 must not contain ``profiles[i] =
    self._create_default_profile(...)`` after Sprint D.1.

    The pattern is matched as a regex against the file source so
    a future reintroduction (which would silently re-enable the
    DEFAULT_SOIL injection) surfaces here as a failure."""
    src = _read_hwsd_source()
    # The negative pin: ``profiles[i] = self._create_default_profile``
    # must not appear anywhere in active code paths.
    assert "profiles[i] = self._create_default_profile" not in src, (
        "Sprint D.1 AC-4 contract: synthetic DEFAULT_SOIL "
        "injection is retired. The pattern "
        "`profiles[i] = self._create_default_profile(...)` "
        "indicates a regression — a caller is silently injecting "
        "a synthetic profile instead of recording the cell as "
        "unavailable."
    )


def test_four_call_sites_call_record_unavailable_instead():
    """The replacement pattern is
    ``self._record_unavailable(i, cause='soil_no_hwsd_coverage')``;
    at least four call sites must invoke it."""
    src = _read_hwsd_source()
    # Count call sites that invoke _record_unavailable. The 4
    # historical sites + any executor-side fallbacks add up.
    call_sites = re.findall(r'self\._record_unavailable\(', src)
    assert len(call_sites) >= 4, (
        f"Expected >=4 _record_unavailable() call sites, found "
        f"{len(call_sites)}. Each historical injection site must "
        f"have been replaced with the unavailable-record pattern."
    )


# ---------------------------------------------------------------------------
# AC-1.1 — source-level texture warning retired
# ---------------------------------------------------------------------------


def test_validate_does_not_emit_texture_sum_warning():
    """The previous source-level warning ``Profile {id}: texture
    fractions sum to {total}%`` is retired — texture validation
    happens at the harmonize stage now via
    :func:`prismpy.harmonize.texture_renormalize.texture_action_for`.

    The validate method's source no longer contains the duplicate
    warning string."""
    src = _read_hwsd_source()
    assert "texture fractions sum to" not in src, (
        "Sprint D.1 AC-1.1: the source-level texture-sum warning "
        "is retired. Harmonize-stage renormalization (AC-1) is "
        "the single source of truth for texture validation."
    )


def test_validate_docstring_documents_retirement():
    """The validate method's docstring names the retirement so a
    future reader knows where the texture validation lives."""
    src = HWSDSource(config=HWSDConfig())
    doc = inspect.getdoc(src.validate) or ""
    assert "harmonize" in doc.lower() or "texture_action_for" in doc


# ---------------------------------------------------------------------------
# AC-4 — sibling-sweep grep for DEFAULT_SOIL active references
# ---------------------------------------------------------------------------


def test_no_active_default_soil_profile_construction_in_hwsd():
    """``DEFAULT_SOIL[...]`` references must only appear in
    docstring / comment positions in ``hwsd.py`` after Sprint D.1
    — never inside an active SoilLayer / SoilProfile constructor.

    The negative pin checks for the constructor pattern
    ``=DEFAULT_SOIL["...]`` which is the syntax used by the
    retired ``_create_default_profile`` to assign sand / silt /
    clay from the synthetic dict.
    """
    src = _read_hwsd_source()
    constructor_pattern = re.compile(r'=\s*DEFAULT_SOIL\[')
    matches = constructor_pattern.findall(src)
    assert len(matches) == 0, (
        f"Found {len(matches)} active DEFAULT_SOIL[...] "
        f"constructor references in hwsd.py. Sprint D.1 retires "
        f"the synthetic-fill path; the constant itself may remain "
        f"as historical documentation, but no active code may "
        f"build a SoilLayer from it."
    )


# ---------------------------------------------------------------------------
# AC-4 — DEFAULT_SOIL constant preserved as historical reference
# ---------------------------------------------------------------------------


def test_default_soil_constant_still_importable():
    """The ``DEFAULT_SOIL`` constant remains importable for
    documentation / historical reference purposes — Sprint D.1
    only retires its USE, not the constant itself. Future
    sprints may remove it entirely once no docstring references
    remain."""
    from prismpy.sources.soil.hwsd import DEFAULT_SOIL
    assert isinstance(DEFAULT_SOIL, dict)
    assert "sand" in DEFAULT_SOIL
