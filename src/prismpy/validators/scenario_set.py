"""Scenario-set deliverable validator (Sprint G AC-G-5 + AC-G-6).

Validates a paired baseline+projection set on disk. The validator is
filesystem-shaped because it ships against PACKAGE deliverables —
``manifest.json`` + ``cell_summary.json`` + ``crop_mask/mask.txt`` +
``soil/<COUNTRY>.SOL`` + ``soil/soil_mask.txt`` + every
``management/*.txt`` — not against in-memory ScenarioBlock instances.
The contract Draft 5 §AC-G-5 specifies path-based input.

Public API:

* :class:`ValidationMode` — closed enum ``SHIP / LEGACY`` per Draft 5
  pass-2 MEDIUM-Rebase-5 mode-disambiguation.
* :class:`ScenarioSetValidationError` — base typed error with
  structured trace fields (``package_label``, ``failing_field_path``,
  ``expected``, ``actual``, plus ``__str__`` per spec §4.1 format
  ``"<package_label>: <failing_field_path> mismatch — expected
  <expected>, got <actual>"``).
* :class:`IdentityDriftError` / :class:`PairingRuleError` — typed
  subclasses for the AC-G-5 invariants.
* :class:`BiasCorrectionConflictError` — typed subclass for the AC-G-6
  conflict rule (carries ``method_a`` / ``method_b``).
* :class:`F-G-3-equivalent` ``UnknownBiasCorrectionInShipModeError``
  — fires in mode=SHIP when any package carries
  ``bias_correction_method == "unknown"``.
* :func:`validate_scenario_set` — main entry. Reads manifests, runs
  ScenarioBlock schema validation per package, then layers identity-
  coupling SHA invariants + pairing rule + bias-correction conflict.

Mode disambiguation per Draft 5 + AC-G-6 line:
* ``SHIP`` (default): F-G-3 fires before AC-G-6 unknown-exclusion
  applies. Any projection with ``bias_correction_method == "unknown"``
  is rejected outright.
* ``LEGACY``: F-G-3 not applied. AC-G-6's ``unknown`` exclusion is
  live — a projection with ``unknown`` can coexist with a projection
  using a real method without firing the conflict rule.

CLI wrapper at the bottom: ``python -m prismpy.validators.scenario_set
[--mode=ship|legacy] <baseline_dir> <projection_1_dir> [...]`` exits
0 on PASS, 1 on FAIL with the structured trace on stderr.

Universal-vs-consumer-specific (per durable §6.4):
* Identity-coupling SHA invariants are UNIVERSAL — every consumer of a
  scenario set rejects drift in cells / soil / management.
* Bias-correction conflict rule is CONSUMER-SPECIFIC (manifest
  validation; downstream models read the methods independently and
  could in principle accept a mixed set with explicit declarations).
  The contract still pins it at this validator boundary because the
  Sprint G shipped-package convention forbids the mix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Closed enum: validation mode ─────────────────────────────────────


class ValidationMode(str, Enum):
    """Closed enum for ``validate_scenario_set``'s mode parameter.

    ``SHIP`` is the default and matches the prismpy-generated
    deliverable path. ``LEGACY`` is the CLI user-runtime path for
    validating external-source packages where ``bias_correction_method
    = "unknown"`` is legitimately acceptable.
    """

    SHIP = "ship"
    LEGACY = "legacy"


# ── Typed errors with structured trace fields ────────────────────────


class ScenarioSetValidationError(Exception):
    """Base typed error for any scenario-set invariant violation.

    Carries the four structured-trace fields per Draft 5 pass-2
    MEDIUM-3 absorption. ``__str__`` formats per spec §4.1 so audit
    consumers can grep stderr directly.
    """

    def __init__(
        self,
        *,
        package_label: str,
        failing_field_path: str,
        expected: Any,
        actual: Any,
        message: Optional[str] = None,
    ) -> None:
        self.package_label = package_label
        self.failing_field_path = failing_field_path
        self.expected = expected
        self.actual = actual
        self._message = message
        super().__init__(self._format())

    def _format(self) -> str:
        if self._message:
            return self._message
        return (
            f"{self.package_label}: {self.failing_field_path} "
            f"mismatch — expected {self.expected!r}, got {self.actual!r}"
        )


class IdentityDriftError(ScenarioSetValidationError):
    """The baseline + projection disagree on an identity-coupled file or
    cell-level field. Per AC-G-5 §5.5-5.7."""


class PairingRuleError(ScenarioSetValidationError):
    """A projection's ``baseline_reference_label`` does not match the
    baseline's ``scenario_label``. Per AC-G-5 §5.9 + F-G-7."""


class BiasCorrectionConflictError(ScenarioSetValidationError):
    """Two projections in the same set carry mutually-distinct
    bias_correction_method values (excluding the AC-G-6 mode-aware
    exclusions). Per AC-G-6 §6.1."""

    def __init__(
        self,
        *,
        package_a: str,
        method_a: str,
        package_b: str,
        method_b: str,
    ) -> None:
        self.method_a = method_a
        self.method_b = method_b
        super().__init__(
            package_label=package_b,
            failing_field_path="manifest.scenario.bias_correction_method",
            expected=method_a,
            actual=method_b,
            message=(
                f"Bias-correction method conflict: "
                f"{package_a} uses {method_a!r} but {package_b} uses {method_b!r}. "
                "All projections in a scenario set must share the same method."
            ),
        )


class UnknownBiasCorrectionInShipModeError(ScenarioSetValidationError):
    """F-G-3: ``bias_correction_method == "unknown"`` in a shipped
    package. AC-G-6 mode-disambiguation ensures this only fires in
    ``mode=SHIP``; ``mode=LEGACY`` honors AC-G-6's unknown exclusion.
    """

    def __init__(self, *, package_label: str) -> None:
        super().__init__(
            package_label=package_label,
            failing_field_path="manifest.scenario.bias_correction_method",
            expected="not 'unknown' in ship mode",
            actual="unknown",
            message=(
                f"{package_label}: bias_correction_method='unknown' is "
                "forbidden in ship mode (F-G-3). Use mode=LEGACY for "
                "external-source packages where 'unknown' is acceptable."
            ),
        )


class WeatherSchemaAsymmetricWithoutLimitationError(ScenarioSetValidationError):
    """F-G-8: baseline + projection ship WTH files with different
    column counts AND the projection's manifest.limitations does NOT
    declare ``weather_schema_asymmetric_within_set``.

    Per ``feedback_no_data_cooking.md`` + durable #16 honest-signal
    contract: silent quality loss is forbidden. When a projection's
    WTH (e.g., 8-col AC-G-7a output with TDEW + RH + SRAD + WIND)
    differs in column count from its baseline (5-col observed-mode),
    the asymmetry MUST appear in ``manifest.limitations`` so audit
    consumers (Dr. Kofi's grep on the limitations key) see the
    declaration. Without it, the asymmetry is silent — exactly the
    trust violation Sprint G's ship contract forbids.

    The fix is for the projection generator to populate
    ``manifest.limitations.weather_schema_asymmetric_within_set``
    with a value field carrying the specifics (e.g., "baseline ships
    5-col WTH per existing observed-climate translators; projection
    ships 8-col WTH per AC-G-7"). The general-form key per warning-
    auditor pass-2 MEDIUM-Rebase-1 covers ANY future asymmetry
    dimension; do not introduce per-instance keys.
    """

    def __init__(
        self,
        *,
        package_label: str,
        baseline_columns: int,
        projection_columns: int,
    ) -> None:
        super().__init__(
            package_label=package_label,
            failing_field_path=(
                "manifest.limitations.weather_schema_asymmetric_within_set"
            ),
            expected="present (asymmetric column count declared)",
            actual=(
                f"missing (baseline {baseline_columns}-col / "
                f"projection {projection_columns}-col)"
            ),
            message=(
                f"{package_label}: weather schema asymmetry detected — "
                f"baseline WTH has {baseline_columns} columns, "
                f"projection WTH has {projection_columns} columns. F-G-8 "
                "requires manifest.limitations.weather_schema_asymmetric_"
                "within_set to declare the asymmetry. Otherwise the "
                "quality-loss is silent (per feedback_no_data_cooking.md "
                "+ durable #16 honest-signal contract)."
            ),
        )


class UnregisteredScenarioInShipModeError(ScenarioSetValidationError):
    """Codex round 1 boundary 4/7 P2-2 absorption: a projection
    package's ``(rcp_or_ssp, time_slice)`` tuple is not in the canonical
    :data:`prismpy.standards.co2_ppm.CO2_PPM_BY_SCENARIO_PERIOD` table.

    Without this check, ``ScenarioBlock``'s Layer 2 post-validator
    silently skips for unregistered tuples and ``mode=SHIP`` would
    otherwise let a shipped package carry an arbitrary CO₂ value.
    Sprint G primary core ensemble is closed: SSP245 / SSP585 × two
    time-slices each. Adding a new scenario × period requires
    extending the canonical table atomically with the
    ``isimip_versions.SCENARIO_TIME_SLICES`` roster.

    ``mode=LEGACY`` does NOT raise this error — external-source
    packages can carry non-primary scenarios at the user-runtime CLI.
    """

    def __init__(
        self,
        *,
        package_label: str,
        rcp_or_ssp: str,
        time_slice: Tuple[int, int],
    ) -> None:
        super().__init__(
            package_label=package_label,
            failing_field_path=(
                "manifest.scenario.rcp_or_ssp + time_slice_start/end"
            ),
            expected=(
                "registered in prismpy.standards.co2_ppm."
                "CO2_PPM_BY_SCENARIO_PERIOD"
            ),
            actual=f"({rcp_or_ssp!r}, {time_slice})",
            message=(
                f"{package_label}: scenario × period "
                f"({rcp_or_ssp!r}, {time_slice}) is not in the "
                "canonical CO₂ table. Sprint G primary core ensemble "
                "covers SSP245 / SSP585 × (2046,2065) / (2086,2100); "
                "extend prismpy.standards.co2_ppm + "
                "isimip_versions.SCENARIO_TIME_SLICES atomically OR "
                "use mode=LEGACY for external-source packages."
            ),
        )


# ── Identity-coupling helpers ────────────────────────────────────────


# The set of files whose SHA-256 must match byte-for-byte between
# baseline and every projection in the same set. Per Draft 5 §AC-G-5
# + spec §4.1.
_IDENTITY_FILES = (
    "crop_mask/mask.txt",
    "soil/soil_mask.txt",
)


# Files matched by glob — every match must agree. ``soil/<COUNTRY>.SOL``
# uses a wildcard because the country code varies per region.
_IDENTITY_GLOBS = (
    "soil/*.SOL",
    "management/*.txt",
)


def _compute_sha256(path: Path) -> str:
    """Stream a file through SHA-256 and return the hex digest.

    Streams in 64KiB chunks so large soil profiles or long management
    files don't load fully into memory.
    """
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _read_manifest(package_dir: Path) -> Dict[str, Any]:
    """Read and parse ``manifest.json`` from a package directory.

    Raises :class:`ScenarioSetValidationError` if the manifest is
    missing or unparseable. The structured fields use the package
    label so downstream callers can pinpoint which directory failed.
    """
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        raise ScenarioSetValidationError(
            package_label=package_dir.name,
            failing_field_path="manifest.json",
            expected="present",
            actual="missing",
            message=(
                f"{package_dir.name}: manifest.json not found at "
                f"{manifest_path}"
            ),
        )
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScenarioSetValidationError(
            package_label=package_dir.name,
            failing_field_path="manifest.json",
            expected="valid JSON",
            actual=f"JSONDecodeError: {exc}",
            message=f"{package_dir.name}: manifest.json is unparseable: {exc}",
        ) from exc


def _read_cell_summary(
    package_dir: Path, package_label: str
) -> Dict[str, Any]:
    """Read ``cell_summary.json`` from a package directory.

    Returns the parsed JSON. Raises :class:`ScenarioSetValidationError`
    on missing / unparseable file.
    """
    cs_path = package_dir / "cell_summary.json"
    if not cs_path.exists():
        raise ScenarioSetValidationError(
            package_label=package_label,
            failing_field_path="cell_summary.json",
            expected="present",
            actual="missing",
        )
    try:
        return json.loads(cs_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScenarioSetValidationError(
            package_label=package_label,
            failing_field_path="cell_summary.json",
            expected="valid JSON",
            actual=f"JSONDecodeError: {exc}",
        ) from exc


def _check_cell_identity(
    baseline_pkg: Path,
    projection_pkg: Path,
    projection_label: str,
) -> None:
    """Set-equality on ``cell_summary.cells[].id`` and per-cell lat/lon.

    A mutated cell ID, a missing cell, or a coordinate drift fires
    :class:`IdentityDriftError` with the JSON-path form of the failing
    field.
    """
    base = _read_cell_summary(baseline_pkg, "baseline").get("cells") or []
    proj = _read_cell_summary(projection_pkg, projection_label).get("cells") or []

    base_ids = {cell.get("id") for cell in base}
    proj_ids = {cell.get("id") for cell in proj}
    if base_ids != proj_ids:
        raise IdentityDriftError(
            package_label=projection_label,
            failing_field_path="cell_summary.cells[*].id",
            expected=sorted(str(i) for i in base_ids),
            actual=sorted(str(i) for i in proj_ids),
        )

    base_by_id = {cell.get("id"): cell for cell in base}
    for cell in proj:
        cell_id = cell.get("id")
        base_cell = base_by_id[cell_id]
        for axis in ("lat", "lon"):
            if cell.get(axis) != base_cell.get(axis):
                raise IdentityDriftError(
                    package_label=projection_label,
                    failing_field_path=(
                        f"cell_summary.cells[id={cell_id!r}].{axis}"
                    ),
                    expected=base_cell.get(axis),
                    actual=cell.get(axis),
                )


def _identity_file_paths(package_dir: Path) -> Dict[str, Path]:
    """Resolve every identity-coupled file under ``package_dir``.

    Returns a dict keyed by relative path string so two packages can
    be compared key-for-key.
    """
    paths: Dict[str, Path] = {}
    for rel in _IDENTITY_FILES:
        full = package_dir / rel
        if full.exists():
            paths[rel] = full
    for pattern in _IDENTITY_GLOBS:
        # ``Path.glob`` emits absolute children; record the relative
        # form so baseline + projection comparison works on stable keys.
        for full in package_dir.glob(pattern):
            rel = full.relative_to(package_dir).as_posix()
            paths[rel] = full
    return paths


def _check_identity_files(
    baseline_pkg: Path,
    projection_pkg: Path,
    projection_label: str,
) -> None:
    """SHA-256 byte-equality on every identity-coupled file.

    Drives both:
    * Same key set across baseline + projection (a projection that
      adds or omits a file fails identity coupling).
    * Same SHA per key.

    Per AC-G-5 §5.7 + spec §4.1.
    """
    base_files = _identity_file_paths(baseline_pkg)
    proj_files = _identity_file_paths(projection_pkg)

    base_keys = set(base_files.keys())
    proj_keys = set(proj_files.keys())
    if base_keys != proj_keys:
        raise IdentityDriftError(
            package_label=projection_label,
            failing_field_path="identity_files[*].path",
            expected=sorted(base_keys),
            actual=sorted(proj_keys),
        )

    for rel, base_path in base_files.items():
        proj_path = proj_files[rel]
        base_sha = _compute_sha256(base_path)
        proj_sha = _compute_sha256(proj_path)
        if base_sha != proj_sha:
            raise IdentityDriftError(
                package_label=projection_label,
                failing_field_path=f"identity_files[{rel!r}].sha256",
                expected=base_sha,
                actual=proj_sha,
            )


# ── Pairing + bias-correction rules ──────────────────────────────────


def _wth_column_count(weather_dir: Path) -> Optional[int]:
    """Return the number of whitespace-separated columns in the first
    non-comment, non-empty data row of the first WTH file under
    ``weather_dir``.

    Returns ``None`` if no WTH file is present (out of F-G-8 scope).
    Comment / header lines (starting with ``@``, ``*``, or ``!``) are
    skipped so the count reflects the actual data schema.
    """
    if not weather_dir.exists() or not weather_dir.is_dir():
        return None
    candidates = sorted(
        list(weather_dir.glob("*.WTH"))
        + list(weather_dir.glob("*.wth"))
    )
    if not candidates:
        return None
    first = candidates[0]
    try:
        text = first.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("@", "*", "!", "#")):
            continue
        return len(stripped.split())
    return None


def _check_weather_schema_asymmetry(
    baseline_pkg: Path,
    projection_pkg: Path,
    projection_label: str,
    projection_manifest: Dict[str, Any],
) -> None:
    """F-G-8: detect WTH column-count asymmetry + require
    ``manifest.limitations.weather_schema_asymmetric_within_set``
    declaration when asymmetric.

    Per Sprint G Draft 5 §AC-G-12 drill 6 + drill 6a + warning-auditor
    pass-2 MEDIUM-Rebase-1 (general-form limitation key — covers any
    future asymmetry dimension, not just baseline-5-col-projection-8-col).
    Per ``feedback_no_data_cooking.md`` honest-signal contract.
    """
    base_cols = _wth_column_count(baseline_pkg / "weather")
    proj_cols = _wth_column_count(projection_pkg / "weather")
    if base_cols is None or proj_cols is None:
        return  # Out of scope — no WTH files on at least one side
    if base_cols == proj_cols:
        return  # Symmetric — F-G-8 does not fire
    limitations = projection_manifest.get("limitations") or {}
    if "weather_schema_asymmetric_within_set" in limitations:
        return  # Asymmetry declared — honest signal preserved
    raise WeatherSchemaAsymmetricWithoutLimitationError(
        package_label=projection_label,
        baseline_columns=base_cols,
        projection_columns=proj_cols,
    )


def _check_pairing_rule(
    baseline_manifest: Dict[str, Any],
    projection_manifest: Dict[str, Any],
    projection_label: str,
) -> None:
    """Each projection's ``baseline_reference_label`` MUST equal the
    baseline's ``scenario_label``. Per AC-G-5 §5.9 + F-G-7."""
    base_label = (
        baseline_manifest.get("scenario") or {}
    ).get("scenario_label")
    proj_ref = (
        projection_manifest.get("scenario") or {}
    ).get("baseline_reference_label")
    if base_label != proj_ref:
        raise PairingRuleError(
            package_label=projection_label,
            failing_field_path="manifest.scenario.baseline_reference_label",
            expected=base_label,
            actual=proj_ref,
        )


def _check_bias_correction_methods(
    projections: List[Tuple[str, Dict[str, Any]]],
    *,
    mode: ValidationMode,
) -> None:
    """AC-G-6 conflict rule with mode disambiguation.

    In ``mode=SHIP``: F-G-3 fires first — any projection with
    ``bias_correction_method == "unknown"`` is rejected outright.
    Then the remaining projections must share a single method.

    In ``mode=LEGACY``: F-G-3 not applied. ``unknown`` and ``none``
    are excluded from the conflict-set so a projection with ``unknown``
    can coexist with a projection using a real method.
    """
    # Collect (label, method) tuples, applying the mode-aware filter.
    pairs: List[Tuple[str, str]] = []
    for label, manifest in projections:
        method = (
            (manifest.get("scenario") or {}).get("bias_correction_method")
        )
        if method is None:
            continue
        if mode is ValidationMode.SHIP and method == "unknown":
            raise UnknownBiasCorrectionInShipModeError(package_label=label)
        if mode is ValidationMode.LEGACY and method in {"unknown", "none"}:
            # Excluded from conflict check per AC-G-6
            continue
        pairs.append((label, method))

    if len(pairs) < 2:
        return  # No conflict possible with 0 or 1 method.

    label_a, method_a = pairs[0]
    for label_b, method_b in pairs[1:]:
        if method_a != method_b:
            raise BiasCorrectionConflictError(
                package_a=label_a,
                method_a=method_a,
                package_b=label_b,
                method_b=method_b,
            )


# ── ScenarioBlock schema validation per package ──────────────────────


def _validate_scenario_block(
    manifest: Dict[str, Any], package_label: str
) -> None:
    """Run ScenarioBlock schema validation on the manifest's scenario
    key. The Pydantic model enforces field-level invariants; this
    helper wraps the validation call so a bad scenario block surfaces
    as :class:`ScenarioSetValidationError` rather than the bare
    ValidationError (callers want one uniform error type)."""
    scenario_payload = manifest.get("scenario")
    if scenario_payload is None:
        raise ScenarioSetValidationError(
            package_label=package_label,
            failing_field_path="manifest.scenario",
            expected="present",
            actual="missing",
        )
    # Late import: keeps Pydantic out of every importer's path.
    from pydantic import ValidationError

    from prismpy.models.scenario import ScenarioBlock

    try:
        ScenarioBlock.model_validate(scenario_payload)
    except ValidationError as exc:
        raise ScenarioSetValidationError(
            package_label=package_label,
            failing_field_path="manifest.scenario",
            expected="valid ScenarioBlock",
            actual=f"ValidationError: {exc}",
        ) from exc


# ── Public entry point ──────────────────────────────────────────────


def validate_scenario_set(
    baseline_path: Path,
    projection_paths: Sequence[Path],
    *,
    mode: ValidationMode = ValidationMode.SHIP,
) -> None:
    """Validate a paired baseline+projection scenario set on disk.

    Args:
        baseline_path: Directory containing the baseline package
            (must include ``manifest.json``, ``cell_summary.json``,
            ``crop_mask/mask.txt``, ``soil/soil_mask.txt``,
            ``soil/<COUNTRY>.SOL``, and ``management/*.txt``).
        projection_paths: Iterable of projection package directories,
            each laid out the same way as the baseline.
        mode: Validation mode. ``ValidationMode.SHIP`` (default) is
            the prismpy-generated deliverable path with F-G-3 active;
            ``ValidationMode.LEGACY`` allows ``bias_correction_method
            = "unknown"`` in projections (per AC-G-6 mode disambiguation).

    Raises:
        ScenarioSetValidationError (or one of its subclasses) on any
        invariant violation. The structured-trace fields
        (``package_label``, ``failing_field_path``, ``expected``,
        ``actual``) make the failure point unambiguous in audit logs.
    """
    baseline_path = Path(baseline_path)
    projection_paths = [Path(p) for p in projection_paths]

    baseline_manifest = _read_manifest(baseline_path)
    _validate_scenario_block(baseline_manifest, "baseline")

    if mode is ValidationMode.SHIP:
        baseline_method = (
            baseline_manifest.get("scenario") or {}
        ).get("bias_correction_method")
        if baseline_method == "unknown":
            raise UnknownBiasCorrectionInShipModeError(
                package_label="baseline"
            )

    projection_manifests: List[Tuple[str, Dict[str, Any]]] = []
    for idx, proj_path in enumerate(projection_paths, start=1):
        proj_label = f"projection_{idx}"
        proj_manifest = _read_manifest(proj_path)
        _validate_scenario_block(proj_manifest, proj_label)

        _check_pairing_rule(baseline_manifest, proj_manifest, proj_label)
        _check_cell_identity(baseline_path, proj_path, proj_label)
        _check_identity_files(baseline_path, proj_path, proj_label)
        _check_weather_schema_asymmetry(
            baseline_path, proj_path, proj_label, proj_manifest
        )

        if mode is ValidationMode.SHIP:
            _check_scenario_period_registered(proj_manifest, proj_label)

        projection_manifests.append((proj_label, proj_manifest))

    _check_bias_correction_methods(projection_manifests, mode=mode)


def _check_scenario_period_registered(
    projection_manifest: Dict[str, Any],
    projection_label: str,
) -> None:
    """Codex round 1 boundary 4/7 P2-2 absorption — ship-mode unregistered-
    scenario rejection.

    Without this check, ``ScenarioBlock``'s Layer 2 post-validator
    silently skips for unregistered ``(rcp_or_ssp, time_slice)``
    tuples and ``mode=SHIP`` would otherwise let a shipped package
    carry an arbitrary CO₂ value paired with any provenance string.
    Calls
    :func:`prismpy.standards.co2_ppm.is_registered_scenario_period`
    so the predicate stays consistent with the canonical lookup
    helper (case-normalisation included).
    """
    from prismpy.standards.co2_ppm import is_registered_scenario_period

    scenario = projection_manifest.get("scenario") or {}
    rcp_or_ssp = scenario.get("rcp_or_ssp")
    start = scenario.get("time_slice_start")
    end = scenario.get("time_slice_end")
    if not isinstance(rcp_or_ssp, str) or start is None or end is None:
        # Schema validation already ran upstream — if we reach here
        # the fields exist. Defensive guard against a future schema
        # change that loosens these to optional.
        return
    if not is_registered_scenario_period(rcp_or_ssp, (int(start), int(end))):
        raise UnregisteredScenarioInShipModeError(
            package_label=projection_label,
            rcp_or_ssp=rcp_or_ssp,
            time_slice=(int(start), int(end)),
        )


# ── CLI wrapper ──────────────────────────────────────────────────────


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prismpy.validators.scenario_set",
        description=(
            "Validate a paired baseline+projection scenario set. "
            "Exit 0 on PASS, 1 on FAIL with the structured trace on "
            "stderr."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in ValidationMode],
        default=ValidationMode.SHIP.value,
        help=(
            "ship (default) — F-G-3 active; rejects "
            "bias_correction_method='unknown'. legacy — allows "
            "'unknown' in projections per AC-G-6 mode disambiguation."
        ),
    )
    parser.add_argument(
        "baseline",
        type=Path,
        help="Path to the baseline package directory.",
    )
    parser.add_argument(
        "projections",
        type=Path,
        nargs="+",
        help="One or more projection package directories.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    mode = ValidationMode(args.mode)
    try:
        validate_scenario_set(
            args.baseline,
            args.projections,
            mode=mode,
        )
    except ScenarioSetValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ValidationMode",
    "ScenarioSetValidationError",
    "IdentityDriftError",
    "PairingRuleError",
    "BiasCorrectionConflictError",
    "UnknownBiasCorrectionInShipModeError",
    "validate_scenario_set",
    "UnregisteredScenarioInShipModeError",
    "WeatherSchemaAsymmetricWithoutLimitationError",
    "main",
]
