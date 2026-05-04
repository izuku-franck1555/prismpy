"""Bound-generation provenance model + writer.

Per Sprint E.0.5 AC-Q2-A1-c. Records the environment + ERA5
archive metadata + dependency + thread-pin configuration of
a bound-gen run as a sidecar JSON next to the generated
bounds file. The model is Pydantic so re-run reproducibility
(the AC-Q2-B1 within-run determinism + AC-Q2-C2 cross-version
ratchet detector) can validate the provenance against the
present bound-gen environment.

Schema fields:

* ``bounds_version`` + ``regenerated_at`` — sprint-level
  pin for the bounds file.
* ERA5 archive (Zenodo): ``era5_archive_zenodo_doi`` /
  ``_url`` / ``_sha256`` / ``_snapshot_date`` /
  ``era5_archive_deposit_status``. The DOI/URL/SHA256/snapshot
  fields are nullable while ``deposit_status="pending"``
  (deposit lands in Sprint E.1 / E.2). When status flips to
  ``"deposited"``, all four MUST be populated; the model
  validator enforces this conjunction.
* AgERA5 cutoff: ``agera5_record_cutoff`` (= snapshot − 180
  days per AC-Q2-A1-a) + observed filename versions.
* License chain: Copernicus (raw AgERA5) → CC-BY 4.0
  (per-zone aggregated derivative) per the AC-Q2-A1-c
  convergence-pass note.
* ECOCROP citation: URL + access-date + per-crop derivative
  disclosure per fair-use posture.
* Dependency versions: python / numpy / rasterio / xarray
  per AC-Q2-A1-d.
* Thread-pin set: omp / openblas / mkl / veclib / numexpr
  per AC-Q2-B1 (cross-platform reproducibility).
* Runtime environment: ``runner_os`` + ``runner_image_sha``
  + ``blas_backend`` per F26 designated-CI-runner pin.
* ``quantile_method`` pin (np.quantile method='linear' per
  research doc §Q2.X aggregation determinism).
* Optional ``subsample_seed`` if memory-pressure subsampling
  was applied during bound-gen.

The Methods text (Sprint E.0.5 AC-Q2-A1-Reframe) frames
180-day cutoff as "up to 120-day AgERA5 lag accommodation;
90+ days margin under pessimistic 30-day estimate" — see
the field description on ``agera5_record_cutoff``.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# AgERA5 cutoff offset per AC-Q2-A1-a. Bound-gen consumes
# only AgERA5 records dated at least this far before the
# snapshot date. The substrate-construction math is "up to
# 120-day AgERA5 lag accommodation; 90+ days margin under
# pessimistic 30-day estimate" (AC-Q2-A1-Reframe).
_AGERA5_CUTOFF_OFFSET: timedelta = timedelta(days=180)


# SHA256 hex-digest pattern. 64 hex chars (case-insensitive).
_SHA256_HEX_RE: re.Pattern[str] = re.compile(r"^[0-9a-fA-F]{64}$")


# FAO ECOCROP source URL — required substring of any
# ecocrop_citation per fair-use posture (the citation must
# point readers to the canonical source).
_ECOCROP_SOURCE_URL: str = "https://ecocrop.apps.fao.org"


# License chain canonical pattern. The substring "Copernicus"
# (raw AgERA5 license) must be present alongside "CC-BY 4.0"
# (per-zone aggregated derivative). The "->" arrow is
# convention but not strictly enforced; the substring
# presence is.
_LICENSE_COPERNICUS_TOKEN: str = "Copernicus"
_LICENSE_DERIVATIVE_TOKEN: str = "CC-BY 4.0"


class DepositStatus(str, Enum):
    """ERA5-archive Zenodo deposit status.

    ``pending`` — the AgERA5 frozen-snapshot has not yet been
    deposited to Zenodo; DOI/URL/SHA256/snapshot fields are
    nullable. Bound-gen can proceed; the deposit lands in
    Sprint E.1 / E.2.

    ``deposited`` — the deposit landed; all four DOI / URL /
    SHA256 / snapshot fields MUST be populated. Methods text
    references the DOI directly.
    """
    PENDING = "pending"
    DEPOSITED = "deposited"


class BoundGenProvenance(BaseModel):
    """Bound-generation run provenance per AC-Q2-A1-c.

    Pydantic v2 model with strict validation: dependency
    versions, thread pins, BLAS backend, runner OS, AgERA5
    cutoff, and ERA5 archive Zenodo metadata. Round-trip-
    safe via ``.model_dump_json()`` / ``.model_validate_json()``;
    bound-gen sidecar files are written by
    :func:`write_bound_gen_provenance`.

    ``model_config`` sets ``extra='forbid'`` so a typo'd
    field (e.g. ``omp_thread`` instead of ``omp_threads``)
    fails validation rather than silently being ignored.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Bound-gen identity ---
    bounds_version: str = Field(
        ...,
        description=(
            "Semantic version pin for the bounds file (e.g. "
            "'frozen_v1'). Matches the constant exported by "
            "the bounds package; ratchets on a new WMO normals "
            "window, a new Beck KG raster release, an AgERA5 "
            "major-version reprocessing, or numpy/scipy "
            "semantic-change major bumps."
        ),
    )
    regenerated_at: datetime = Field(
        ...,
        description=(
            "UTC timestamp at which the bound-gen run started "
            "and produced this sidecar JSON."
        ),
    )

    # --- ERA5 archive Zenodo deposit (nullable until deposit) ---
    era5_archive_zenodo_doi: Optional[str] = Field(
        default=None,
        description=(
            "Zenodo DOI for the AgERA5 frozen-snapshot archive "
            "(e.g. '10.5281/zenodo.NNNNNNNN'). Null while "
            "deposit_status='pending'."
        ),
    )
    era5_archive_zenodo_url: Optional[str] = Field(
        default=None,
        description=(
            "Zenodo HTTPS URL for the AgERA5 frozen-snapshot "
            "archive. Null while deposit_status='pending'."
        ),
    )
    era5_archive_sha256: Optional[List[str]] = Field(
        default=None,
        description=(
            "SHA256 hashes of per-file AgERA5 archive contents "
            "(one per archived netCDF/Zarr fragment). Null "
            "while deposit_status='pending'."
        ),
    )
    era5_archive_snapshot_date: Optional[date] = Field(
        default=None,
        description=(
            "Calendar date when the AgERA5 archive was sampled "
            "for bound-gen input. Null while "
            "deposit_status='pending'."
        ),
    )
    era5_archive_deposit_status: DepositStatus = Field(
        ...,
        description=(
            "Deposit status. While 'pending', the four DOI/URL/"
            "SHA256/snapshot fields MAY be null. While "
            "'deposited', all four MUST be populated; the model "
            "validator enforces this conjunction. Methods text "
            "MUST NOT claim DOI retrieval until 'deposited'."
        ),
    )

    # --- AgERA5 cutoff (per AC-Q2-A1-a + AC-Q2-A1-Reframe) ---
    agera5_record_cutoff: date = Field(
        ...,
        description=(
            "Cutoff date for AgERA5 records included in bound-"
            "gen. Equals snapshot_date - 180 days per "
            "AC-Q2-A1-a. The 180-day window accommodates up to "
            "120-day AgERA5 lag (4× pessimistic 30-day "
            "estimate) with 90+ days margin (per Methods Reframe "
            "AC-Q2-A1-Reframe)."
        ),
    )
    agera5_filename_versions_observed: List[str] = Field(
        ...,
        description=(
            "AgERA5 filename versions (v1.x.y tags) observed "
            "during bound-gen sampling. Empty list is valid "
            "if no version tag was present in the sampled files."
        ),
    )

    # --- License chain (per AC-Q2-A1-c convergence-pass) ---
    license_chain: str = Field(
        ...,
        description=(
            "License chain string. Raw AgERA5 inputs are under "
            "the Copernicus License; the per-zone aggregated "
            "bounds derivative is CC-BY 4.0. Format: "
            "'Copernicus License (raw AgERA5) -> CC-BY 4.0 "
            "(per-zone aggregated derivative)'."
        ),
    )

    # --- ECOCROP citation chain ---
    ecocrop_citation: str = Field(
        ...,
        description=(
            "FAO ECOCROP citation including the source URL, "
            "the per-crop access date(s), and the derivative-"
            "disclosure note per fair-use posture."
        ),
    )

    # --- Dependency versions (per AC-Q2-A1-d) ---
    python_version: str = Field(
        ...,
        description=(
            "Python interpreter version (e.g. '3.12.5') under "
            "which bound-gen was run. Pinned in pyproject.toml "
            "as '>=3.10,<3.13'."
        ),
    )
    numpy_version: str = Field(..., description="numpy version (e.g. '1.26.4').")
    rasterio_version: str = Field(..., description="rasterio version.")
    xarray_version: str = Field(..., description="xarray version.")

    # --- Thread pins (per AC-Q2-B1 + codex Gate A counter-add #3) ---
    omp_threads: int = Field(
        ..., ge=1, le=1,
        description=(
            "OMP_NUM_THREADS pinned to 1 by the designated CI "
            "runner; recorded for cross-platform reproducibility."
        ),
    )
    openblas_threads: int = Field(..., ge=1, le=1)
    mkl_threads: int = Field(..., ge=1, le=1)
    veclib_threads: int = Field(..., ge=1, le=1)
    numexpr_threads: int = Field(..., ge=1, le=1)

    # --- Runtime environment (per AC-Q2-A1-d codex Gate A counter-add) ---
    runner_os: str = Field(
        ...,
        description=(
            "GitHub Actions runner OS string (e.g. 'ubuntu-22.04'). "
            "F26 requires Linux; the determinism meta-test catches "
            "non-Linux builds."
        ),
    )
    runner_image_sha: Optional[str] = Field(
        default=None,
        description=(
            "GitHub Actions runner image digest (e.g. SHA of the "
            "ubuntu-22.04 image). Optional because local dev runs "
            "have no image SHA; CI runs always populate this."
        ),
    )
    blas_backend: str = Field(
        ...,
        description=(
            "BLAS backend reported by numpy.show_config() (e.g. "
            "'OpenBLAS', 'Accelerate', 'MKL'). F26 runtime guard "
            "rejects non-OpenBLAS at bound-gen time."
        ),
    )

    # --- Numerical method pins ---
    quantile_method: Literal["linear"] = Field(
        ...,
        description=(
            "np.quantile method pinned per substrate determinism "
            "contract (research doc §Q2.X). 'linear' equals the "
            "WMO No. 1203 climatological-normal percentile "
            "convention. Required (no default) so a sidecar "
            "JSON that omits the field cannot silently claim "
            "linear was used."
        ),
    )

    # --- Optional subsample seed ---
    subsample_seed: Optional[int] = Field(
        default=None,
        description=(
            "Fixed RNG seed used when memory-pressure forced "
            "subsampling. Null if no subsampling was applied. "
            "Recorded so re-runs reproduce the same draw."
        ),
    )

    @field_validator("license_chain")
    @classmethod
    def _validate_license_chain(cls, value: str) -> str:
        """The license chain string must reference both the
        raw Copernicus license and the CC-BY 4.0 derivative
        license. A free-text field that omits either token
        defeats the AC-Q2-A1-c license-chain claim."""
        if _LICENSE_COPERNICUS_TOKEN not in value:
            raise ValueError(
                f"license_chain must reference "
                f"{_LICENSE_COPERNICUS_TOKEN!r} (raw AgERA5 "
                f"license); got {value!r}."
            )
        if _LICENSE_DERIVATIVE_TOKEN not in value:
            raise ValueError(
                f"license_chain must reference "
                f"{_LICENSE_DERIVATIVE_TOKEN!r} (per-zone "
                f"aggregated derivative); got {value!r}."
            )
        return value

    @field_validator("ecocrop_citation")
    @classmethod
    def _validate_ecocrop_citation(cls, value: str) -> str:
        """The FAO ECOCROP citation must reference the
        canonical source URL and an ISO 8601 access date.
        Both anchors are AC-Q2-A1-c content pins."""
        if _ECOCROP_SOURCE_URL not in value:
            raise ValueError(
                f"ecocrop_citation must reference the FAO "
                f"ECOCROP source URL {_ECOCROP_SOURCE_URL!r}; "
                f"got {value!r}."
            )
        if not re.search(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(
                f"ecocrop_citation must include an ISO 8601 "
                f"calendar access date (YYYY-MM-DD); got "
                f"{value!r}."
            )
        return value

    @model_validator(mode="after")
    def _validate_deposit_conjunction(self) -> "BoundGenProvenance":
        """Per AC-Q2-A1-c: when deposit_status='deposited', all
        four Zenodo fields MUST be populated AND non-empty.
        A blank DOI string or empty SHA256 list defeats the
        archive-verification point of AC-Q2-A1-c just as
        thoroughly as ``None`` would."""
        if self.era5_archive_deposit_status == DepositStatus.DEPOSITED:
            missing_or_empty: List[str] = []
            if not self.era5_archive_zenodo_doi:  # None or empty string
                missing_or_empty.append("era5_archive_zenodo_doi")
            if not self.era5_archive_zenodo_url:
                missing_or_empty.append("era5_archive_zenodo_url")
            if not self.era5_archive_sha256:  # None or empty list
                missing_or_empty.append("era5_archive_sha256")
            if self.era5_archive_snapshot_date is None:
                missing_or_empty.append("era5_archive_snapshot_date")
            if missing_or_empty:
                raise ValueError(
                    f"BoundGenProvenance: deposit_status='deposited' "
                    f"requires all four Zenodo fields populated "
                    f"and non-empty; missing or blank: "
                    f"{missing_or_empty}. Methods text MUST NOT "
                    f"claim DOI retrieval until the deposit lands "
                    f"and these fields are filled in."
                )
            # SHA256 hashes must be valid 64-char hex digests.
            for i, sha in enumerate(self.era5_archive_sha256 or []):
                if not _SHA256_HEX_RE.fullmatch(sha):
                    raise ValueError(
                        f"BoundGenProvenance: "
                        f"era5_archive_sha256[{i}] = {sha!r} is "
                        f"not a 64-character hex digest. Each "
                        f"hash must match {_SHA256_HEX_RE.pattern!r}."
                    )
        return self

    @model_validator(mode="after")
    def _validate_cutoff_matches_snapshot(self) -> "BoundGenProvenance":
        """Per AC-Q2-A1-a + the field documentation: when the
        snapshot date is populated (i.e. deposited), the cutoff
        MUST equal snapshot − 180 days. Without this check, a
        deposited record can certify the 120-day lag /
        90+-day margin while the actual cutoff is inconsistent
        with the snapshot."""
        if self.era5_archive_snapshot_date is not None:
            expected_cutoff = (
                self.era5_archive_snapshot_date - _AGERA5_CUTOFF_OFFSET
            )
            if self.agera5_record_cutoff != expected_cutoff:
                raise ValueError(
                    f"BoundGenProvenance: agera5_record_cutoff "
                    f"must equal era5_archive_snapshot_date - "
                    f"180 days; got cutoff="
                    f"{self.agera5_record_cutoff!r}, snapshot="
                    f"{self.era5_archive_snapshot_date!r}, "
                    f"expected_cutoff={expected_cutoff!r}."
                )
        return self


def write_bound_gen_provenance(
    provenance: BoundGenProvenance, path: Union[str, Path],
) -> Path:
    """Serialize a :class:`BoundGenProvenance` to JSON on disk.

    Writes via ``model_dump_json(indent=2)`` for human-readable
    review (bound-gen is a low-frequency operation; the size
    overhead is negligible). Returns the absolute path of the
    written file.

    The bound-gen management command calls this once per run,
    placing the JSON next to the bounds file (e.g.,
    ``bounds/frozen_v1/provenance.json``).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        provenance.model_dump_json(indent=2), encoding="utf-8",
    )
    return target.resolve()
