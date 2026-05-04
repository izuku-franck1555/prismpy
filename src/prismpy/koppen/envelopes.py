"""ECOCROP envelope loader + AC-Q3-A-NaN + F28 validation.

Loads the per-crop ECOCROP envelope substrate from the JSON
data file and validates it under the AC-Q3-A-NaN strict
ordering rules (RMIN < RMAX, TMIN < TMAX, all values non-NaN)
and the F28 per-crop provenance contract.

Per AC-Q3-A-d, only four numeric values per crop ship:

* ``TMIN``, ``TMAX`` — annual temperature envelope (°C)
* ``RMIN``, ``RMAX`` — annual rainfall envelope (mm)

Other ECOCROP fields (CLIZ, ALTMX, pH range, photoperiod,
GMIN/GMAX, latitude range) are deliberately excluded per
AC-Q3-A-d (CLIZ has data-completeness gaps and is not load-
bearing) and per probe-1-A scope clarity (the others are
Sprint F / V3 territory).

Per F28, every crop entry MUST also ship a per-crop
provenance block: ``verbatim_source_url`` (HTTPS URL to the
ECOCROP data sheet the values were retrieved from) and
``verbatim_retrieval_date`` (ISO 8601 calendar date of
retrieval). The loader fail-loud rejects any crop missing
either field, or carrying a non-HTTPS URL, or an
unparseable date. F28 protects against future commits
drifting to ship unverified crop values.

The validation is fail-loud: a NaN field, malformed
ordering, or missing/invalid provenance raises
:class:`EnvelopeValidationError` with the crop name and the
violating field. The bound generator + the Stage 1
compatibility classifier consume the validated dict without
re-checking; failing here prevents silent propagation of
bad data into downstream verdicts.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict


# Path to the bundled ECOCROP envelope JSON substrate.
# Co-located with the Köppen-Geiger classifier so the wizard-
# time crop-region compatibility check has both the zone
# raster and the crop envelope under a single namespace.
ECOCROP_ENVELOPE_PATH: Path = (
    Path(__file__).parent / "ecocrop_envelopes.json"
)


# Required envelope fields per AC-Q3-A-d. Stage 1 uses ONLY
# these four; ALTMX / pH / photoperiod / GMIN / GMAX /
# latitude are Sprint F or V3 territory. The F27 AST walker
# enforces this scope discipline at module-code time.
REQUIRED_FIELDS: tuple[str, ...] = ("TMIN", "TMAX", "RMIN", "RMAX")


# Required provenance fields per F28 (per-crop verbatim
# source). Every crop entry must carry both fields; the
# loader fail-loud rejects entries missing either, or
# carrying a non-HTTPS URL, or an unparseable ISO 8601
# date. F28 protects against future commits drifting to
# ship unverified crop values without provenance.
REQUIRED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "verbatim_source_url",
    "verbatim_retrieval_date",
)


# URL prefix required for the verbatim source. HTTPS only;
# http:// is rejected. The structural test pins the FAO
# domain (process discipline at PR-review time); the
# loader enforces only well-formed HTTPS at runtime so a
# future ECOCROP migration to a different domain does not
# require a loader code change.
REQUIRED_URL_PREFIX: str = "https://"


class EnvelopeValidationError(ValueError):
    """Raised when an ECOCROP envelope fails AC-Q3-A-NaN
    validation (NaN value, missing required field, malformed
    RMIN/RMAX or TMIN/TMAX ordering) or F28 provenance
    validation (missing verbatim_source_url /
    verbatim_retrieval_date, non-HTTPS URL, unparseable
    ISO 8601 date).

    The error message names the crop and the violating field
    so the operator sees exactly which envelope record is
    bad without re-running with extra logging.
    """


def load_ecocrop_envelopes(
    path: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Load + validate the ECOCROP envelope substrate.

    Returns a ``{crop_name: {TMIN, TMAX, RMIN, RMAX,
    verbatim_source_url, verbatim_retrieval_date}}`` dict
    keyed by crop name (e.g. ``"maize"``). Each inner dict
    contains the four numeric envelope fields as floats plus
    the two provenance fields as strings (URL + ISO 8601
    date).

    Validation per AC-Q3-A-NaN + F28:

    * Every required envelope field (TMIN/TMAX/RMIN/RMAX) present per crop.
    * Every envelope value is a finite (non-NaN, non-infinite) number.
    * ``RMIN < RMAX`` (strict ordering).
    * ``TMIN < TMAX`` (strict ordering).
    * Every required provenance field (``verbatim_source_url`` + ``verbatim_retrieval_date``) present per crop (F28).
    * ``verbatim_source_url`` is an HTTPS URL.
    * ``verbatim_retrieval_date`` parses as ISO 8601 calendar date.

    Any breach raises :class:`EnvelopeValidationError` with
    the crop name + violating field. The exception is fatal
    by design — propagating bad envelope data into Stage 1
    verdicts would silently classify cells as compatible or
    incompatible based on garbage, undermining the honest-
    signal contract.

    The path argument is optional; defaults to the bundled
    :data:`ECOCROP_ENVELOPE_PATH`. Tests pass a synthetic
    path to exercise validation paths.
    """
    target = path if path is not None else ECOCROP_ENVELOPE_PATH
    with open(target, encoding="utf-8") as fp:
        payload = json.load(fp)
    crops = payload.get("crops")
    if not isinstance(crops, dict) or not crops:
        raise EnvelopeValidationError(
            f"ECOCROP envelope file at {target!r} is missing "
            f"the top-level 'crops' dict or it is empty."
        )
    validated: Dict[str, Dict[str, Any]] = {}
    for crop_name, envelope in crops.items():
        validated[crop_name] = _validate_one_envelope(crop_name, envelope)
    return validated


def _validate_one_envelope(
    crop_name: str, envelope: Any,
) -> Dict[str, Any]:
    """Validate a single per-crop envelope dict.

    Validates per AC-Q3-A-NaN (numeric envelope) + F28
    (per-crop provenance block).
    """
    if not isinstance(envelope, dict):
        raise EnvelopeValidationError(
            f"Envelope for crop {crop_name!r} must be a dict; "
            f"got {type(envelope).__name__}."
        )
    coerced: Dict[str, Any] = {}
    for field in REQUIRED_FIELDS:
        if field not in envelope:
            raise EnvelopeValidationError(
                f"Envelope for crop {crop_name!r} missing "
                f"required field {field!r}. AC-Q3-A-d requires "
                f"all four of TMIN/TMAX/RMIN/RMAX."
            )
        value = envelope[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EnvelopeValidationError(
                f"Envelope for crop {crop_name!r} field "
                f"{field!r} = {value!r} is not numeric."
            )
        fvalue = float(value)
        if math.isnan(fvalue) or math.isinf(fvalue):
            raise EnvelopeValidationError(
                f"Envelope for crop {crop_name!r} field "
                f"{field!r} = {value!r} is NaN or infinite. "
                f"AC-Q3-A-NaN requires finite numeric values."
            )
        coerced[field] = fvalue
    if not coerced["RMIN"] < coerced["RMAX"]:
        raise EnvelopeValidationError(
            f"Envelope for crop {crop_name!r} fails ordering: "
            f"RMIN ({coerced['RMIN']}) must be strictly less "
            f"than RMAX ({coerced['RMAX']}). AC-Q3-A-NaN "
            f"strict-ordering pin."
        )
    if not coerced["TMIN"] < coerced["TMAX"]:
        raise EnvelopeValidationError(
            f"Envelope for crop {crop_name!r} fails ordering: "
            f"TMIN ({coerced['TMIN']}) must be strictly less "
            f"than TMAX ({coerced['TMAX']}). AC-Q3-A-NaN "
            f"strict-ordering pin."
        )

    # F28: per-crop provenance block. Every crop must carry
    # verbatim_source_url + verbatim_retrieval_date so future
    # commits cannot silently ship unverified values.
    for field in REQUIRED_PROVENANCE_FIELDS:
        if field not in envelope:
            raise EnvelopeValidationError(
                f"Envelope for crop {crop_name!r} missing "
                f"required provenance field {field!r}. F28 "
                f"requires verbatim_source_url + "
                f"verbatim_retrieval_date per crop."
            )
        value = envelope[field]
        if not isinstance(value, str):
            raise EnvelopeValidationError(
                f"Envelope for crop {crop_name!r} provenance "
                f"field {field!r} = {value!r} must be a string."
            )
        coerced[field] = value
    if not coerced["verbatim_source_url"].startswith(REQUIRED_URL_PREFIX):
        raise EnvelopeValidationError(
            f"Envelope for crop {crop_name!r} provenance "
            f"verbatim_source_url "
            f"({coerced['verbatim_source_url']!r}) must start "
            f"with {REQUIRED_URL_PREFIX!r}. F28 requires "
            f"HTTPS URLs only."
        )
    try:
        date.fromisoformat(coerced["verbatim_retrieval_date"])
    except ValueError as exc:
        raise EnvelopeValidationError(
            f"Envelope for crop {crop_name!r} provenance "
            f"verbatim_retrieval_date "
            f"({coerced['verbatim_retrieval_date']!r}) is not a "
            f"valid ISO 8601 calendar date (YYYY-MM-DD). F28 "
            f"requires parseable ISO 8601 retrieval dates."
        ) from exc

    return coerced
