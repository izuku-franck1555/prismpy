"""Gen-time required-at-creation gate (MANIFEST_L2_TIERS / create_manifest).

``create_manifest`` hard-fails with ``ManifestError`` when a REQUIRED_AT_CREATION
manifest field (region / crop|crops / platform / temporal / data_sources) is absent
or empty — the creation-surface analog of the run-side required-parameter door.

Authored from the manifest tier-vocabulary contract: field-ABSENT semantics (an empty
value, NOT span-inadequacy — that is the separate base_package_temporal_complete
readiness gate), crop|crops either-satisfies, ManifestError subclasses ValueError, and
zero false-reject on a complete (realistic) native config.
"""
from __future__ import annotations

import pytest

from prismpy.packaging.manifest import (
    MANIFEST_L2_TIERS,
    ManifestError,
    _check_required_at_creation,
    create_manifest,
)


def _complete_config(**overrides):
    base = {
        "project_name": "required-at-creation-test",
        "region_name": "Kano",
        "country": "Nigeria",
        "crop_name": "maize",
        "start_year": 2010,
        "end_year": 2014,
        "data_sources": {"climate": "AgERA5"},
    }
    base.update(overrides)
    return base


def test_manifest_l2_tiers_shape():
    """Every user-facing gen-time field is REQUIRED_AT_CREATION on the creation
    surface with a non-empty rationale (the projection's WHY)."""
    assert set(MANIFEST_L2_TIERS) == {
        "region", "crop", "platform", "temporal", "data_sources",
    }
    for field, spec in MANIFEST_L2_TIERS.items():
        assert spec["tier"] == "REQUIRED_AT_CREATION", field
        assert spec["surface"] == "creation", field
        assert spec["rationale"].strip(), field


def test_manifest_error_is_valueerror():
    """ValueError subclass → existing ``except ValueError`` handlers still catch it."""
    assert issubclass(ManifestError, ValueError)


def test_complete_config_creates_manifest(tmp_path):
    """A complete native config builds without tripping the gate (zero false-reject)."""
    m = create_manifest(tmp_path, _complete_config(), platform="pythia")
    assert m["region"]["name"] == "Kano"
    assert m["crop"]["name"] == "maize"
    assert m["temporal"]["start_year"] == 2010
    assert m["data_sources"] == {"climate": "AgERA5"}


@pytest.mark.parametrize(
    "mutate, needle",
    [
        ({"region_name": ""}, "region"),
        ({"crop_name": ""}, "crop"),        # no crop name anywhere -> rejected
        ({"start_year": None}, "temporal"),
        ({"end_year": None}, "temporal"),
        ({"data_sources": {}}, "data_sources"),
    ],
)
def test_absent_required_field_rejected(tmp_path, mutate, needle):
    """Each REQUIRED_AT_CREATION field, when absent/empty, hard-fails creation."""
    with pytest.raises(ManifestError, match=needle):
        create_manifest(tmp_path, _complete_config(**mutate), platform="pythia")


def test_empty_platform_rejected(tmp_path):
    with pytest.raises(ManifestError, match="platform"):
        create_manifest(tmp_path, _complete_config(), platform="")


def test_crop_or_crops_either_satisfies():
    """crop|crops: EITHER a scalar crop.name OR a crops entry carrying a crop name
    satisfies; only when BOTH are nameless does the gate trip. (Exercised at the
    helper level because create_manifest derives crop.name and crops[*].crop_name
    from the same project_config['crop_name'].)"""
    common = {
        "platform": "pythia",
        "region": {"name": "Kano"},
        "temporal": {"start_year": 2010, "end_year": 2014},
        "data_sources": {"climate": "AgERA5"},
    }
    _check_required_at_creation({**common, "crop": {"name": "maize"}, "crops": []})
    _check_required_at_creation(
        {**common, "crop": {"name": ""}, "crops": [{"crop_name": "maize"}]}
    )
    with pytest.raises(ManifestError, match="crop"):
        _check_required_at_creation(
            {**common, "crop": {"name": ""}, "crops": [{"crop_name": ""}]}
        )
