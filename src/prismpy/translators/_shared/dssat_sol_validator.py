"""DSSAT v4.8 .SOL format validator.

Reads a ``.SOL`` file written by
:func:`prismpy.translators._shared.dssat_sol_writer.write_dssat_sol`
(or any other DSSAT-conformant writer) and verifies the structure
matches the DSSAT v4.8 specification per Tsuji et al. (1994) + the
DSSAT v4.8 user guide §5.2:

- The file opens with a ``*SOILS:`` banner.
- Each profile block opens with a 10-character ``*<id>`` profile-id
  line followed by metadata columns (ISO3 country, texture, depth,
  source).
- The site block (``@SITE  COUNTRY  LAT  LONG  SCS FAMILY``) follows
  the id line; the value row carries lat/long at fixed column
  positions.
- The surface-properties block opens with the ``@ SCOM`` header (10
  columns: SCOM SALB SLU1 SLDR SLRO SLNF SLPF SMHB SMPX SMKE) and
  one value row.
- The layer block opens with the 17-column ``@ SLB`` header (SLB
  SLMH SLLL SDUL SSAT SRGF SSKS SBDM SLOC SLCL SLSI SLCF SLNI
  SLHW SLHB SCEC SADC) and one or more fixed-width data rows.

The validator is the canonical-source format-conformance check per
durable lesson §24: every consumer of an eGHR ``{CC}.SOL`` (CRAFT,
PYTHIA today; future SARRA-Py if applicable) trusts that the file
has been produced or verified by this validator. Any future writer
drift is caught by routing the writer's output through this module
and asserting the validation result has zero errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


# Canonical DSSAT v4.8 layer-row column header. Whitespace-collapsed
# for matching; the writer emits this verbatim per Tsuji et al. 1994
# Table 5.1 + the v4.8 user guide §5.2 layer-property specification.
_LAYER_HEADER_TOKENS: Sequence[str] = (
    "@",
    "SLB",
    "SLMH",
    "SLLL",
    "SDUL",
    "SSAT",
    "SRGF",
    "SSKS",
    "SBDM",
    "SLOC",
    "SLCL",
    "SLSI",
    "SLCF",
    "SLNI",
    "SLHW",
    "SLHB",
    "SCEC",
    "SADC",
)

# Surface-properties block header tokens (DSSAT v4.8 §5.2.3).
_SURFACE_HEADER_TOKENS: Sequence[str] = (
    "@",
    "SCOM",
    "SALB",
    "SLU1",
    "SLDR",
    "SLRO",
    "SLNF",
    "SLPF",
    "SMHB",
    "SMPX",
    "SMKE",
)

# Site block header tokens (DSSAT v4.8 §5.2.2).
_SITE_HEADER_TOKENS: Sequence[str] = (
    "@SITE",
    "COUNTRY",
    "LAT",
    "LONG",
    "SCS",
    "FAMILY",
)

# Each layer row is exactly 17 fields when whitespace-tokenized:
# {SLB} {SLMH} {SLLL} {SDUL} {SSAT} {SRGF} {SSKS} {SBDM} {SLOC}
# {SLCL} {SLSI} {SLCF} {SLNI} {SLHW} {SLHB} {SCEC} {SADC}
_LAYER_ROW_FIELD_COUNT = 17


@dataclass
class DssatSolIssue:
    """A single conformance issue surfaced by the validator."""

    severity: str  # "error" or "warning"
    line_number: int  # 1-based
    message: str
    line_content: Optional[str] = None


@dataclass
class DssatSolValidationResult:
    """Outcome of validating a DSSAT v4.8 .SOL file."""

    sol_path: Path
    profile_count: int
    layer_count: int
    issues: List[DssatSolIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[DssatSolIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[DssatSolIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_dssat_sol(sol_path: Path) -> DssatSolValidationResult:
    """Validate a DSSAT v4.8 ``.SOL`` file's structure and column conformance.

    Reads the file as bytes (DSSAT writes CRLF line endings; the
    validator preserves that convention by splitting on the
    universal-newline boundary so either CRLF or LF input is
    accepted). Walks the line stream once and accumulates a list of
    :class:`DssatSolIssue` records — empty list means the file is
    structurally valid against the v4.8 spec.

    Args:
        sol_path: Filesystem path to the ``.SOL`` file.

    Returns:
        :class:`DssatSolValidationResult` whose ``is_valid`` property
        is ``True`` when no error-severity issues were found.
    """
    sol_path = Path(sol_path)
    issues: List[DssatSolIssue] = []
    profile_count = 0
    layer_count = 0

    if not sol_path.exists():
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=0,
                message=f"SOL file does not exist: {sol_path}",
            )
        )
        return DssatSolValidationResult(
            sol_path=sol_path,
            profile_count=0,
            layer_count=0,
            issues=issues,
        )

    text = sol_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    if not lines:
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=0,
                message="SOL file is empty",
            )
        )
        return DssatSolValidationResult(
            sol_path=sol_path,
            profile_count=0,
            layer_count=0,
            issues=issues,
        )

    # 1) File header: must start with ``*SOILS:``.
    if not lines[0].startswith("*SOILS:"):
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=1,
                message=(
                    "First line must start with '*SOILS:' per DSSAT v4.8 "
                    "spec (Tsuji 1994 §5.2.1); got "
                    f"{lines[0][:40]!r}."
                ),
                line_content=lines[0],
            )
        )

    # 2) Per-profile block walk. Each profile block has the form:
    #    *<10-char-id>  <metadata...>
    #    @SITE        COUNTRY          LAT     LONG SCS FAMILY
    #    <site value row>
    #    @ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE
    #    <surface value row>
    #    @  SLB  SLMH  ... SCEC  SADC
    #    <layer row 1>
    #    [<layer row N>]
    #    [<blank line separating profiles>]
    i = 1  # skip the *SOILS: banner
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        line_no = i + 1

        if not stripped or stripped.startswith("!"):
            # Blank line or DSSAT comment: skip.
            i += 1
            continue

        if line.startswith("*SOILS:"):
            issues.append(
                DssatSolIssue(
                    severity="error",
                    line_number=line_no,
                    message="Duplicate '*SOILS:' banner; expected exactly one at file start.",
                    line_content=line,
                )
            )
            i += 1
            continue

        if line.startswith("*"):
            # New profile block. Validate the id line.
            profile_count += 1
            _validate_profile_id_line(line, line_no, issues)

            # Walk forward through the four expected sub-blocks:
            #   site header + value
            #   surface header + value
            #   layer header + N value rows
            i += 1
            i = _validate_site_block(lines, i, issues)
            i = _validate_surface_block(lines, i, issues)
            i, layers_in_profile = _validate_layer_block(lines, i, issues)
            layer_count += layers_in_profile
            continue

        # An unrecognized non-blank line at top level: surface as a warning so
        # the caller can decide whether it's a benign DSSAT extension or noise.
        issues.append(
            DssatSolIssue(
                severity="warning",
                line_number=line_no,
                message=(
                    "Unrecognized line at file top level (not a profile id, "
                    "not a comment, not blank)."
                ),
                line_content=line,
            )
        )
        i += 1

    if profile_count == 0:
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=0,
                message=(
                    "No profile blocks found — every .SOL file must carry "
                    "at least one '*<10-char-id>' profile block."
                ),
            )
        )

    return DssatSolValidationResult(
        sol_path=sol_path,
        profile_count=profile_count,
        layer_count=layer_count,
        issues=issues,
    )


def _validate_profile_id_line(
    line: str, line_no: int, issues: List[DssatSolIssue]
) -> None:
    """Validate ``*<10-char-id>  <metadata>`` profile-id line."""
    if len(line) < 11:
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=line_no,
                message=(
                    "Profile id line must be at least 11 chars (asterisk + "
                    "10-char id); got "
                    f"length={len(line)}."
                ),
                line_content=line,
            )
        )
        return
    profile_id = line[1:11].rstrip()
    if not profile_id:
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=line_no,
                message="Profile id is empty after the leading asterisk.",
                line_content=line,
            )
        )


def _validate_site_block(
    lines: List[str], i: int, issues: List[DssatSolIssue]
) -> int:
    """Validate the ``@SITE`` header + value row. Returns next line index."""
    if i >= len(lines):
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=i,
                message="Profile block ended before @SITE header.",
            )
        )
        return i

    site_header = lines[i]
    expected = " ".join(_SITE_HEADER_TOKENS)
    if site_header.split() != list(_SITE_HEADER_TOKENS):
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=i + 1,
                message=(
                    f"Expected site header tokens {expected!r}; got "
                    f"{site_header.split()!r}."
                ),
                line_content=site_header,
            )
        )
    i += 1

    # Site value row. Must have at least 4 tokens (site, country, lat, long).
    if i >= len(lines):
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=i,
                message="Profile block ended before @SITE value row.",
            )
        )
        return i

    site_value = lines[i]
    site_tokens = site_value.split()
    if len(site_tokens) < 4:
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=i + 1,
                message=(
                    "Site value row has fewer than 4 tokens (site, country, "
                    f"lat, long required). Got {len(site_tokens)} tokens."
                ),
                line_content=site_value,
            )
        )
    return i + 1


def _validate_surface_block(
    lines: List[str], i: int, issues: List[DssatSolIssue]
) -> int:
    """Validate the ``@ SCOM`` surface-block header + value row.

    Returns next line index.
    """
    if i >= len(lines):
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=i,
                message="Profile block ended before @SCOM surface header.",
            )
        )
        return i

    header = lines[i]
    if header.split() != list(_SURFACE_HEADER_TOKENS):
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=i + 1,
                message=(
                    "Expected surface header tokens "
                    f"{list(_SURFACE_HEADER_TOKENS)!r}; got {header.split()!r}."
                ),
                line_content=header,
            )
        )
    i += 1

    # Surface value row: must have 10 fields matching the 10-column header.
    if i >= len(lines):
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=i,
                message="Profile block ended before surface value row.",
            )
        )
        return i

    surface_value = lines[i]
    surface_tokens = surface_value.split()
    if len(surface_tokens) != 10:
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=i + 1,
                message=(
                    "Surface value row must have exactly 10 fields "
                    "(SCOM SALB SLU1 SLDR SLRO SLNF SLPF SMHB SMPX SMKE); "
                    f"got {len(surface_tokens)}."
                ),
                line_content=surface_value,
            )
        )
    return i + 1


def _validate_layer_block(
    lines: List[str], i: int, issues: List[DssatSolIssue]
) -> tuple[int, int]:
    """Validate the ``@ SLB`` layer header + one or more 17-col data rows.

    Returns (next line index, count of layer rows validated).

    Per Tsuji et al. 1994 §5.2.4 + the DSSAT v4.8 user guide, every
    profile block must carry at least one layer row; a header with
    no data rows is a malformed profile that DSSAT-CSM cannot
    consume. The terminator paths below append a missing-data
    error if the loop exited without reading any rows.
    """
    header_line_number = i + 1  # for the missing-rows error

    if i >= len(lines):
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=i,
                message="Profile block ended before @SLB layer header.",
            )
        )
        return i, 0

    header = lines[i]
    if header.split() != list(_LAYER_HEADER_TOKENS):
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=i + 1,
                message=(
                    "Expected 17-column layer header "
                    f"{list(_LAYER_HEADER_TOKENS)!r}; got {header.split()!r}."
                ),
                line_content=header,
            )
        )
    i += 1

    def _flag_missing_layer_rows() -> None:
        """Append the per-DSSAT-spec error when a profile block ships zero layer rows."""
        issues.append(
            DssatSolIssue(
                severity="error",
                line_number=header_line_number,
                message=(
                    "Profile block has no layer data rows; DSSAT v4.8 spec "
                    "requires at least one row after the @SLB header."
                ),
                line_content=header,
            )
        )

    # Layer data rows: read until a blank line, EOF, or the next profile id.
    layer_count = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            if layer_count == 0:
                _flag_missing_layer_rows()
            return i + 1, layer_count
        if line.startswith("*"):
            # Next profile starts; stop the layer-block walk without
            # consuming the asterisk line.
            if layer_count == 0:
                _flag_missing_layer_rows()
            return i, layer_count

        layer_count += 1
        tokens = line.split()
        if len(tokens) != _LAYER_ROW_FIELD_COUNT:
            issues.append(
                DssatSolIssue(
                    severity="error",
                    line_number=i + 1,
                    message=(
                        f"Layer row must have exactly {_LAYER_ROW_FIELD_COUNT} "
                        f"fields per the @SLB header; got {len(tokens)}."
                    ),
                    line_content=line,
                )
            )
        i += 1

    if layer_count == 0:
        _flag_missing_layer_rows()
    return i, layer_count


# CLI entry point. Run as ``python -m prismpy.translators._shared.dssat_sol_validator <path>``.
if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m prismpy.translators._shared.dssat_sol_validator <path>")
        sys.exit(2)

    result = validate_dssat_sol(Path(sys.argv[1]))
    print(f"File: {result.sol_path}")
    print(f"Profiles: {result.profile_count}")
    print(f"Layers:   {result.layer_count}")
    print(f"Errors:   {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    for issue in result.issues:
        print(f"  [{issue.severity}] line {issue.line_number}: {issue.message}")
    sys.exit(0 if result.is_valid else 1)
