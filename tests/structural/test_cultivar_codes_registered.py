"""Universal substrate pin: every cultivar-shape string literal that
appears in prismpy source code must resolve to a registered DSSAT
cultivar in column 1 of at least one ``Genotype/<MODEL>048.CUL``
file.

The companion test ``test_cultivar_defaults_against_registry.py``
pins the **named dicts** that translators consult for default
cultivars (``generic_cultivars`` in CRAFT, ``_LEGUME_DEFAULT_CULTIVARS``
in PYTHIA). That test catches drift in those two dicts. It does NOT
catch drift in the dozen-plus other hardcoded cultivar literals
scattered across translator return paths, manifest writers, README
generators, and per-cell defaults — and the F-AI(cultivar) failure
mode (unregistered millet/cowpea codes) demonstrated that those
hardcoded sites can carry the same bug shape silently.

This test scans every ``.py`` file under ``prismpy/src/prismpy/`` for
cultivar-shape string literals (``[A-Z]{2}[0-9]{4}`` for the
country-prefix namespace, ``99[0-9]{4}`` for the regional/generic
namespace per durable lesson #23 multi-namespace rule) and for each
one asserts:

1. The literal appears in column 1 of at least one ``.CUL`` file in
   the DSSAT ``Genotype/`` directory (universal substrate guarantee).
2. The set of (file, value) pairs matches the frozen
   ``EXPECTED_LITERAL_PAIRS`` pin (drift detection — adding a NEW
   literal in source without recording it here fails the test).

A literal that resolves to column 4 (ECOTYPE pointer) but not column 1
is rejected — that was precisely the F-AI(cultivar) failure mode for
the prior millet/cowpea defaults. The column-1 vs column-4 distinction
is load-bearing: DSSAT cultivar lookup runs against column 1 only.

Anti-mutation drill: change any in-source cultivar literal to an
unregistered code (e.g., ``"IB9999"``) → both tests fail with the
file:line and the closest column-1 matches in the .CUL files.

Calibration applied: per durable #23 (registry-grep before
classifying), this test reads the actual ``.CUL`` files at run time
when the DSSAT install is reachable; it skips with an explicit
message naming the candidate paths when not.
"""
from __future__ import annotations

import ast
import os
import re
import unittest
from pathlib import Path
from typing import Dict, FrozenSet, Set, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRISMPY_SRC = _REPO_ROOT / "src" / "prismpy"


# Cultivar-code shape per durable lesson #23 multi-namespace rule:
#
# * ``[A-Z]{2}[0-9]{4}`` — country/institution-prefix namespace
#   (e.g., ``IB0001`` IBSNAT, ``GH0010`` Ghana, ``II0003``
#   IBSNAT-international, ``CP0005`` cowpea-prefix, ``IB0149``
#   ICRISAT, etc.)
# * ``99[0-9]{4}`` — regional/generic-fallback namespace used
#   throughout the CERES family for "MEDIUM SEASON / NORTH / SOUTH"
#   varieties (e.g., ``990001``, ``990002``, ``990003``, ``990004``)
#
# The pattern is anchored on both ends so nearly-cultivar-shape
# strings like ``"IB001"`` (5 chars; soil-profile column in
# craft/translator.py) or ``"AB12345"`` (7 chars; not a DSSAT
# cultivar code) do not get scanned.
_CULTIVAR_SHAPE_RE = re.compile(r"^([A-Z]{2}[0-9]{4}|99[0-9]{4})$")


# Frozen pin of every cultivar-shape literal we expect to find under
# ``prismpy/src/prismpy/``. Each entry maps a (relative-file-path,
# literal-value) tuple to the (relative-path-from-prismpy-src,
# why-this-pair-is-here) note. Add a new entry here when introducing
# a new cultivar literal in source; remove an entry when retiring
# one. The drift test ``test_no_unexpected_cultivar_literals`` fails
# when the in-source set diverges from this pin.
#
# The set-shape (rather than list-shape) lets multiple line-number
# occurrences of the same (file, value) collapse into one pin entry,
# so adding a redundant literal at a new line in the same file does
# not require an extra pin entry — the reviewer still sees the new
# line in the diff but the test doesn't false-fail.
EXPECTED_LITERAL_PAIRS: FrozenSet[Tuple[str, str]] = frozenset({
    # CRAFT translator — generic_cultivars dict + hardcoded README example
    ("translators/craft/translator.py", "990002"),  # maize MEDIUM SEASON + ambiguous fallback
    ("translators/craft/translator.py", "IB0001"),  # sorghum / rice / groundnut RIO/IR8/STARR
    ("translators/craft/translator.py", "IB0149"),  # millet Sadore-Local (Sahel)
    ("translators/craft/translator.py", "II0003"),  # cowpea IT90K-277-2 (IITA)
    ("translators/craft/translator.py", "GH0010"),  # maize OBATANPA hardcoded example
    # PYTHIA translator — _LEGUME_DEFAULT_CULTIVARS dict + per-cell GDD codes + fallback
    ("translators/pythia/translator.py", "II0003"),  # cowpea
    ("translators/pythia/translator.py", "IB0001"),  # legumes (groundnut/peanut/soybean/chickpea/bean)
    ("translators/pythia/translator.py", "990001"),  # NORTH VARIETY (CERES-Maize)
    ("translators/pythia/translator.py", "990002"),  # MIDDLE VARIETY (CERES-Maize / CERES-Millet)
    ("translators/pythia/translator.py", "990003"),  # SOUTH VARIETY
    # README generator — package-config defaults
    ("packaging/readme_generator.py", "GH0010"),    # maize OBATANPA
    ("packaging/readme_generator.py", "990002"),    # MEDIUM SEASON
})


# Mapping from cultivar literal to the set of ``.CUL`` filenames in
# which it MUST appear in column 1. Each literal is verified against
# this candidate set when the DSSAT install is reachable; the set
# represents the agronomically-plausible target model files for that
# code's namespace, NOT every file in the registry. Most codes appear
# in only one ``.CUL`` file; ambiguous fallbacks (``990002``) appear
# in several CER-family files.
_CANDIDATE_CUL_FILES: Dict[str, FrozenSet[str]] = {
    # Maize OBATANPA (Ghana-released)
    "GH0010": frozenset({"MZCER048.CUL", "MZIXM048.CUL"}),
    # CERES-Maize regional fallbacks (also in MLCER048 for millet)
    "990001": frozenset({"MZCER048.CUL", "MZIXM048.CUL", "MLCER048.CUL"}),
    "990002": frozenset({"MZCER048.CUL", "MZIXM048.CUL", "MLCER048.CUL"}),
    "990003": frozenset({"MZCER048.CUL", "MZIXM048.CUL", "MLCER048.CUL"}),
    "990004": frozenset({"MZCER048.CUL", "SGCER048.CUL"}),
    # IB0001 — generic IBSNAT cultivar in many model files
    "IB0001": frozenset({
        "SGCER048.CUL",   # sorghum RIO
        "RICER048.CUL",   # rice IR 8
        "PNGRO048.CUL",   # groundnut STARR
        "SBGRO048.CUL",   # soybean BRAGG
        "CHGRO048.CUL",   # chickpea ANNIGERI
        "BNGRO048.CUL",   # drybean Porrillo Sintetico
    }),
    # IB0149 — ICRISAT Sadore-Local pearl millet
    "IB0149": frozenset({"MLCER048.CUL"}),
    # II0003 — IITA cowpea IT90K-277-2
    "II0003": frozenset({"CPGRO048.CUL"}),
}


def _resolve_dssat_genotype_dir() -> Path | None:
    """Locate the DSSAT ``Genotype/`` directory.

    Search order mirrors the F-AI(cultivar) named-dict test:

    1. ``DSSAT_INSTALL`` env var — caller-pinned, highest priority.
    2. ``EXP/dssat-csm-os/Genotype`` sibling-checkout path relative
       to the prismpy + prismweb monorepo layout.

    Returns ``None`` when neither path resolves; the caller skips
    the registry check with an explicit message rather than failing.
    """
    env = os.environ.get("DSSAT_INSTALL")
    if env:
        candidate = Path(env) / "Genotype"
        if candidate.is_dir():
            return candidate
    monorepo_root = _REPO_ROOT.parent
    candidate = monorepo_root / "EXP" / "dssat-csm-os" / "Genotype"
    if candidate.is_dir():
        return candidate
    return None


def _scan_cultivar_literals() -> Set[Tuple[str, str, int]]:
    """Walk every ``.py`` file under ``prismpy/src/prismpy/`` and
    return the set of (relative_path, literal_value, lineno) tuples
    where ``literal_value`` matches the cultivar-code shape.

    The walker only considers ``ast.Constant`` nodes (Python 3.10+
    representation of string literals); f-string components,
    docstring sentences, and concatenated forms are not scanned, so
    a documentation reference like "see GH0010 for example" does not
    trip the pin. The pin is intentionally narrow: a real code emit
    site uses a string literal that the AST captures as a Constant.
    """
    hits: Set[Tuple[str, str, int]] = set()
    for py_file in sorted(_PRISMPY_SRC.rglob("*.py")):
        try:
            tree = ast.parse(
                py_file.read_text(encoding="utf-8"),
                filename=str(py_file),
            )
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if _CULTIVAR_SHAPE_RE.match(node.value):
                    rel = py_file.relative_to(_PRISMPY_SRC).as_posix()
                    hits.add((rel, node.value, node.lineno))
    return hits


def _cul_first_column_codes(cul_path: Path) -> Set[str]:
    """Return the set of column-1 cultivar codes in ``cul_path``.

    Implementation mirrors the helper in
    ``test_cultivar_defaults_against_registry.py``; the duplication
    is deliberate so neither structural test depends on the other
    for substrate.
    """
    codes: Set[str] = set()
    for raw in cul_path.read_text(
        encoding="utf-8", errors="replace",
    ).splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith(("!", "@", "*")):
            continue
        token = stripped.split(None, 1)[0]
        if 5 <= len(token) <= 7 and token.isalnum():
            codes.add(token)
    return codes


class TestCultivarCodesRegistered(unittest.TestCase):
    """Universal substrate pin: every cultivar-shape literal in
    prismpy source must resolve to a registered ``.CUL`` column-1
    cultivar code, AND the set of in-source literals must match the
    EXPECTED_LITERAL_PAIRS frozen pin (drift catch)."""

    @classmethod
    def setUpClass(cls):
        cls.in_source_pairs: Set[Tuple[str, str]] = {
            (rel, value) for rel, value, _ in _scan_cultivar_literals()
        }
        cls.dssat_genotype = _resolve_dssat_genotype_dir()

    def test_no_unexpected_cultivar_literals(self):
        """The set of (file, value) pairs the AST scan finds must
        match EXPECTED_LITERAL_PAIRS exactly. Adding a new literal
        without recording it here surfaces as an "unexpected pair"
        failure with the file path; removing one surfaces as a
        "missing expected pair" failure."""
        unexpected = self.in_source_pairs - EXPECTED_LITERAL_PAIRS
        missing = EXPECTED_LITERAL_PAIRS - self.in_source_pairs
        problems: list[str] = []
        if unexpected:
            problems.append(
                "Unexpected (file, cultivar-literal) pairs in source — "
                "either retire the literal or add it to "
                "EXPECTED_LITERAL_PAIRS with a candidate-CUL-file mapping:\n  "
                + "\n  ".join(f"{f}: {v!r}" for f, v in sorted(unexpected))
            )
        if missing:
            problems.append(
                "Expected (file, cultivar-literal) pairs no longer found "
                "in source — remove from EXPECTED_LITERAL_PAIRS if "
                "intentional:\n  "
                + "\n  ".join(f"{f}: {v!r}" for f, v in sorted(missing))
            )
        self.assertEqual(problems, [], "\n\n".join(problems))

    def test_every_in_source_literal_is_registered_in_at_least_one_cul(self):
        """When the DSSAT install is reachable, assert every (rel,
        value) pair the AST scan finds maps to a column-1 cultivar
        in at least one of its ``_CANDIDATE_CUL_FILES`` entries.
        Skips with an explicit message when DSSAT is not reachable."""
        if self.dssat_genotype is None:
            self.skipTest(
                "DSSAT install not reachable — set DSSAT_INSTALL or "
                "place the dssat-csm-os checkout at the conventional "
                "EXP/dssat-csm-os/ sibling path. The "
                "EXPECTED_LITERAL_PAIRS drift pin still ran."
            )
        # Cache .CUL → codes per file so a literal that maps to many
        # candidate files only reads each file once.
        cul_codes_cache: Dict[str, Set[str]] = {}
        problems: list[str] = []
        for rel, value, _lineno in _scan_cultivar_literals():
            candidates = _CANDIDATE_CUL_FILES.get(value)
            if candidates is None:
                problems.append(
                    f"{rel}: cultivar-shape literal {value!r} has no "
                    "_CANDIDATE_CUL_FILES entry — declare which model "
                    "files it must register in (use frozenset of "
                    "MODEL048.CUL filenames)."
                )
                continue
            registered_in: Set[str] = set()
            for cul_filename in candidates:
                cul_path = self.dssat_genotype / cul_filename
                if not cul_path.is_file():
                    continue
                if cul_filename not in cul_codes_cache:
                    cul_codes_cache[cul_filename] = _cul_first_column_codes(
                        cul_path,
                    )
                if value in cul_codes_cache[cul_filename]:
                    registered_in.add(cul_filename)
            if not registered_in:
                # Find closest matches across the candidate files for
                # an actionable error message.
                hints: list[str] = []
                for cul_filename in sorted(candidates):
                    codes = cul_codes_cache.get(cul_filename, set())
                    near = sorted(c for c in codes if c[:2] == value[:2])[:3]
                    if near:
                        hints.append(
                            f"{cul_filename} has nearby codes: "
                            + ", ".join(near)
                        )
                problems.append(
                    f"{rel}: cultivar literal {value!r} not found in "
                    "column 1 of any candidate .CUL file ("
                    + ", ".join(sorted(candidates))
                    + (
                        "). Hints — " + "; ".join(hints)
                        if hints else ")."
                    )
                )
        self.assertEqual(
            problems, [],
            "Every cultivar-shape literal in prismpy source must "
            "resolve to a registered column-1 cultivar code in at "
            "least one of its declared candidate .CUL files. "
            "Unregistered codes are CRAFT/DSSAT-load-fatal:\n  "
            + "\n  ".join(problems),
        )


if __name__ == "__main__":
    unittest.main()
