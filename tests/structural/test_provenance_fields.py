"""Sprint E.0.5 AC-Q2-A1-c — bound-gen provenance fields + writer.

Pins the :class:`BoundGenProvenance` schema, the deposit-
status conjunction validator, the thread-pin literal-1
constraint, the methods-text Reframe (AC-Q2-A1-Reframe), and
the :meth:`ProvenanceTracker.record_bound_gen_provenance`
delegation.

Anti-mutation drills:

- Drop any required field from :class:`BoundGenProvenance` →
  ``test_required_fields_present`` fails.
- Relax the deposit conjunction validator →
  ``test_deposited_status_requires_all_zenodo_fields`` passes
  when it shouldn't.
- Re-introduce the "60+30+90" framing in module docs →
  ``test_methods_reframe_no_legacy_framing`` fails (Reframe
  pin AC-Q2-A1-Reframe).
- Drop the F26 ``blas_backend`` field → introspection test
  fails.
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

from prismpy.bounds import (
    BoundGenProvenance,
    DepositStatus,
    write_bound_gen_provenance,
)
from prismpy.provenance.tracker import ProvenanceTracker


_REPO_ROOT = Path(__file__).resolve().parents[2]


# Required Pydantic field names per AC-Q2-A1-c + AC-Q2-A1-d
# + codex Gate A counter-adds. Anti-mutation: drop any of
# these from the model and the introspection test fails.
_REQUIRED_FIELDS: tuple[str, ...] = (
    # Bound-gen identity
    "bounds_version",
    "regenerated_at",
    # ERA5 archive Zenodo deposit (nullable until deposit)
    "era5_archive_zenodo_doi",
    "era5_archive_zenodo_url",
    "era5_archive_sha256",
    "era5_archive_snapshot_date",
    "era5_archive_deposit_status",
    # AgERA5 cutoff
    "agera5_record_cutoff",
    "agera5_filename_versions_observed",
    # License chain + ECOCROP citation
    "license_chain",
    "ecocrop_citation",
    # Dependency versions
    "python_version",
    "numpy_version",
    "rasterio_version",
    "xarray_version",
    # Thread pin set
    "omp_threads",
    "openblas_threads",
    "mkl_threads",
    "veclib_threads",
    "numexpr_threads",
    # Runtime environment
    "runner_os",
    "runner_image_sha",
    "blas_backend",
    # Quantile method pin
    "quantile_method",
    # Optional subsample seed
    "subsample_seed",
)


def _minimal_pending_provenance(**overrides) -> BoundGenProvenance:
    """Build a minimal-but-valid provenance record with
    deposit_status='pending'. Used as a base for mutation tests."""
    base = dict(
        bounds_version="frozen_v1",
        regenerated_at=datetime(2026, 5, 4, 12, 0, 0),
        era5_archive_deposit_status=DepositStatus.PENDING,
        agera5_record_cutoff=date(2025, 11, 5),  # snapshot - 180d
        agera5_filename_versions_observed=["v1.1.0"],
        license_chain=(
            "Copernicus License (raw AgERA5) -> CC-BY 4.0 "
            "(per-zone aggregated derivative)"
        ),
        ecocrop_citation=(
            "FAO ECOCROP (https://ecocrop.apps.fao.org/), "
            "retrieved 2026-05-03; per-crop derivative under "
            "fair-use educational scope."
        ),
        python_version="3.12.5",
        numpy_version="1.26.4",
        rasterio_version="1.3.10",
        xarray_version="2024.6.0",
        omp_threads=1,
        openblas_threads=1,
        mkl_threads=1,
        veclib_threads=1,
        numexpr_threads=1,
        runner_os="ubuntu-22.04",
        runner_image_sha=None,
        blas_backend="OpenBLAS",
    )
    base.update(overrides)
    return BoundGenProvenance(**base)


class TestBoundGenProvenanceFields(unittest.TestCase):
    """Pin the Pydantic model schema. Drop any required field
    or rename a key and these tests fail."""

    def test_required_fields_present(self):
        model_fields = set(BoundGenProvenance.model_fields.keys())
        for field in _REQUIRED_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, model_fields)

    def test_no_unexpected_fields(self):
        # Tighten: the model should expose exactly the
        # documented field set (no accidental drift).
        model_fields = set(BoundGenProvenance.model_fields.keys())
        unexpected = model_fields - set(_REQUIRED_FIELDS)
        self.assertEqual(
            unexpected, set(),
            f"BoundGenProvenance has unexpected fields not in "
            f"the AC-Q2-A1-c contract: {sorted(unexpected)}",
        )

    def test_quantile_method_default_is_linear(self):
        prov = _minimal_pending_provenance()
        self.assertEqual(prov.quantile_method, "linear")

    def test_subsample_seed_optional(self):
        # Default to None; bound-gen runs without subsampling
        # leave the field unset.
        prov = _minimal_pending_provenance()
        self.assertIsNone(prov.subsample_seed)

    def test_thread_pins_must_be_one(self):
        # Per AC-Q2-B1: each thread pin is constrained to
        # ge=1, le=1, so the only valid value is 1. Try 0 and
        # try 2; both should raise.
        for invalid in (0, 2, 4, 8):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    _minimal_pending_provenance(omp_threads=invalid)


class TestDepositStatusEnum(unittest.TestCase):
    """Pin the enum members used in the deposit-conjunction
    validator."""

    def test_two_members(self):
        self.assertEqual(len(list(DepositStatus)), 2)

    def test_pending_and_deposited(self):
        self.assertEqual(DepositStatus.PENDING.value, "pending")
        self.assertEqual(DepositStatus.DEPOSITED.value, "deposited")


class TestDepositConjunction(unittest.TestCase):
    """Per AC-Q2-A1-c: when deposit_status='deposited', all
    four Zenodo fields MUST be populated. Pending allows null."""

    def test_pending_allows_null_zenodo_fields(self):
        # Sanity: the minimal pending provenance is valid.
        prov = _minimal_pending_provenance()
        self.assertIsNone(prov.era5_archive_zenodo_doi)
        self.assertEqual(
            prov.era5_archive_deposit_status, DepositStatus.PENDING,
        )

    def test_deposited_status_requires_all_zenodo_fields(self):
        # deposit='deposited' but DOI null -> ValueError.
        with self.assertRaises(ValueError) as ctx:
            _minimal_pending_provenance(
                era5_archive_deposit_status=DepositStatus.DEPOSITED,
            )
        self.assertIn("era5_archive_zenodo_doi", str(ctx.exception))
        self.assertIn("DOI retrieval", str(ctx.exception))

    def test_deposited_with_all_fields_passes(self):
        prov = _minimal_pending_provenance(
            era5_archive_deposit_status=DepositStatus.DEPOSITED,
            era5_archive_zenodo_doi="10.5281/zenodo.12345678",
            era5_archive_zenodo_url=(
                "https://zenodo.org/records/12345678"
            ),
            era5_archive_sha256=["abc123def456"],
            era5_archive_snapshot_date=date(2026, 4, 1),
        )
        self.assertEqual(
            prov.era5_archive_deposit_status, DepositStatus.DEPOSITED,
        )

    def test_deposited_partial_fields_rejected(self):
        # DOI populated but URL/SHA256/snapshot still null ->
        # rejected (the conjunction is "all four").
        with self.assertRaises(ValueError):
            _minimal_pending_provenance(
                era5_archive_deposit_status=DepositStatus.DEPOSITED,
                era5_archive_zenodo_doi="10.5281/zenodo.12345678",
                # URL/SHA256/snapshot still None
            )


class TestWriteBoundGenProvenance(unittest.TestCase):
    """Pin the JSON-sidecar serialization round-trip."""

    def test_writes_indented_json(self):
        prov = _minimal_pending_provenance()
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "frozen_v1" / "provenance.json"
            written = write_bound_gen_provenance(prov, target)
            self.assertTrue(written.exists())
            text = written.read_text(encoding="utf-8")
            # Indented (2-space) → contains newlines + indentation.
            self.assertIn("\n  ", text)

    def test_round_trip_via_json(self):
        prov = _minimal_pending_provenance()
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "provenance.json"
            write_bound_gen_provenance(prov, target)
            data = json.loads(target.read_text(encoding="utf-8"))
            for field in _REQUIRED_FIELDS:
                self.assertIn(field, data)
            self.assertEqual(data["bounds_version"], "frozen_v1")
            self.assertEqual(data["era5_archive_deposit_status"], "pending")

    def test_writes_creates_parent_dir(self):
        prov = _minimal_pending_provenance()
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "deep" / "nested" / "provenance.json"
            written = write_bound_gen_provenance(prov, target)
            self.assertTrue(written.exists())


class TestProvenanceTrackerDelegation(unittest.TestCase):
    """Pin :meth:`ProvenanceTracker.record_bound_gen_provenance`
    as a thin delegate to :func:`write_bound_gen_provenance`."""

    def test_delegates_to_writer(self):
        tracker = ProvenanceTracker(enabled=True)
        prov = _minimal_pending_provenance()
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "provenance.json"
            written = tracker.record_bound_gen_provenance(prov, target)
            self.assertTrue(written.exists())
            data = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(data["bounds_version"], "frozen_v1")

    def test_disabled_tracker_returns_path_without_writing(self):
        tracker = ProvenanceTracker(enabled=False)
        prov = _minimal_pending_provenance()
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "provenance.json"
            returned = tracker.record_bound_gen_provenance(prov, target)
            # Path is returned but file is not written.
            self.assertFalse(returned.exists())


class TestMethodsReframePin(unittest.TestCase):
    """Per AC-Q2-A1-Reframe: the cutoff math is framed as
    'up to 120-day AgERA5 lag accommodation; 90+ days margin
    under pessimistic 30-day estimate', NOT the legacy
    '60+30+90 safety margin' framing."""

    @classmethod
    def setUpClass(cls):
        cls.source = (
            _REPO_ROOT
            / "src" / "prismpy" / "bounds" / "provenance.py"
        ).read_text(encoding="utf-8")

    def test_methods_reframe_no_legacy_framing(self):
        # Negative grep: the legacy "60+30+90" framing must
        # not appear anywhere in the bounds-provenance source.
        self.assertNotIn("60+30+90", self.source)

    def test_methods_reframe_120_day_framing_present(self):
        # Positive grep: the canonical Reframe phrasing is
        # documented on the agera5_record_cutoff field.
        self.assertIn("120-day", self.source)
        self.assertIn("90+ days margin", self.source)
        self.assertIn("AgERA5 lag", self.source)


if __name__ == "__main__":
    unittest.main()
