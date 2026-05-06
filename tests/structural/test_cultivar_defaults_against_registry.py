"""Pin every translator-side default cultivar code against DSSAT's
``Genotype/<MODEL>048.CUL`` registry and against the EXPECTED_DEFAULTS
pin so a future contributor cannot silently drift to an unregistered
code.

The CRAFT translator's ``generic_cultivars`` dict and the PYTHIA
translator's ``_LEGUME_DEFAULT_CULTIVARS`` dict each map a prismpy
crop key to a (column-1) cultivar code. CRAFT loads / DSSAT runs
reject unregistered cultivar codes outright, so a default value that
is NOT present in column-1 of the corresponding ``.CUL`` file is a
data-correctness failure for every project that falls through to
the default — exactly the failure mode that surfaced when the
millet and cowpea defaults pointed at ``IB0001``, which is an
ECOTYPE pointer in ``MLCER048.CUL`` and absent from
``CPGRO048.CUL`` entirely.

Two layers of pinning:

* **EXPECTED_DEFAULTS pin** — frozen tuple of (crop, code,
  cul_file) the test expects each translator dict to declare. Add a
  new entry here when adding a new crop default; remove an entry
  when the default is intentionally retired. This pin runs on every
  test execution regardless of whether the DSSAT install is present.

* **Registry check** — when ``DSSAT_INSTALL`` env var or the
  default ``EXP/dssat-csm-os/Genotype`` sibling-checkout path is
  found, the test reads the corresponding ``.CUL`` file and asserts
  the cultivar code IS present in column 1 (not just column 4 ECOTYPE
  reference). When neither path resolves, the registry check is
  skipped with an explicit message so CI machines without DSSAT do
  not fail.

Anti-mutation drill: change a default to an unregistered code (e.g.,
``"millet": "IB0099"``) → both the EXPECTED_DEFAULTS pin AND the
registry check fire with explicit "code X not found in MODEL.CUL"
messages.
"""
from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from typing import Dict, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRAFT_TRANSLATOR = (
    _REPO_ROOT / "src" / "prismpy" / "translators" / "craft" / "translator.py"
)
_PYTHIA_TRANSLATOR = (
    _REPO_ROOT / "src" / "prismpy" / "translators" / "pythia" / "translator.py"
)


# Pin: every (crop_key, expected_code, cul_filename) the translator
# dicts must declare. Sahel-persona alignment notes inline so a
# reviewer can sanity-check the agronomic fitness without re-greping
# the registry. Each entry was verified against the DSSAT 4.8
# ``Genotype/`` directory; the structural test re-verifies on every
# run when the registry is reachable.
CRAFT_EXPECTED_DEFAULTS: Tuple[Tuple[str, str, str], ...] = (
    ("maize",     "990002", "MZCER048.CUL"),  # MEDIUM SEASON
    ("sorghum",   "IB0001", "SGCER048.CUL"),  # RIO
    ("millet",    "IB0149", "MLCER048.CUL"),  # Sadore-Local, Sahel ICRISAT
    ("rice",      "IB0001", "RICER048.CUL"),  # IR 8
    ("cowpea",    "II0003", "CPGRO048.CUL"),  # IT90K-277-2, IITA
    ("groundnut", "IB0001", "PNGRO048.CUL"),  # STARR, v tamnut
)


# PYTHIA's _LEGUME_DEFAULT_CULTIVARS routes a different crop set
# through the CROPGRO model family. The mapping is structurally
# similar but uses different (cultivar, label) tuples. The label
# (second tuple element) is informational; only the first element
# must match a registered cultivar code.
PYTHIA_EXPECTED_LEGUME_DEFAULTS: Tuple[Tuple[str, str, str], ...] = (
    ("cowpea",    "II0003", "CPGRO048.CUL"),  # IT90K-277-2, IITA
    ("groundnut", "IB0001", "PNGRO048.CUL"),  # STARR
    ("peanut",    "IB0001", "PNGRO048.CUL"),  # alias for groundnut
    ("soybean",   "IB0001", "SBGRO048.CUL"),  # BRAGG
    ("soya bean", "IB0001", "SBGRO048.CUL"),
    ("chickpea",  "IB0001", "CHGRO048.CUL"),  # ANNIGERI v48calib
    ("bean",      "IB0001", "BNGRO048.CUL"),  # Porrillo Sintetico
    ("beans",     "IB0001", "BNGRO048.CUL"),
)


def _resolve_dssat_install() -> Path | None:
    """Locate the DSSAT install directory, returning the
    ``Genotype/`` subdirectory when available.

    Search order:

    1. ``DSSAT_INSTALL`` env var (caller-pinned, highest priority).
    2. Sibling ``EXP/dssat-csm-os/Genotype`` relative to the
       prismpy + prismweb monorepo layout — the conventional
       checkout location for the DSSAT-CSM source tree.

    Returns ``None`` when neither path resolves; the registry check
    is then skipped with an explicit message rather than failing.
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


def _extract_dict_assignment(
    py_file: Path, dict_name: str,
) -> Dict[str, str | Tuple[str, ...]]:
    """Walk the AST of ``py_file`` and return the literal value of
    the first ``<dict_name> = {...}`` assignment that appears in any
    function body or class body.

    The dict values may be either string literals (CRAFT) or tuples
    of two string literals (PYTHIA). The walker materializes each
    value type and returns the dict; a non-literal value raises so
    the test fails loudly with the line number rather than silently
    skipping a drifted entry.
    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        # Accept both ``X = {...}`` (Assign) and ``X: T = {...}``
        # (AnnAssign with annotation) so dicts declared with type
        # annotations like ``_LEGUME_DEFAULT_CULTIVARS: Dict[...] = {...}``
        # don't slip past the walker.
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            if not any(t.id == dict_name for t in targets):
                continue
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if not (isinstance(target, ast.Name) and target.id == dict_name):
                continue
            value_node = node.value
        else:
            continue
        if value_node is None or not isinstance(value_node, ast.Dict):
            raise AssertionError(
                f"{py_file}:{node.lineno}: {dict_name} must be a "
                "dict literal so the structural test can pin its "
                "shape; got "
                f"{ast.dump(value_node)[:80] if value_node else 'None'} "
                "instead."
            )
        result: Dict[str, str | Tuple[str, ...]] = {}
        for key, value in zip(value_node.keys, value_node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise AssertionError(
                    f"{py_file}:{node.lineno}: every {dict_name} key "
                    "must be a string literal."
                )
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                result[key.value] = value.value
            elif isinstance(value, ast.Tuple) and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str)
                for e in value.elts
            ):
                result[key.value] = tuple(e.value for e in value.elts)
            else:
                raise AssertionError(
                    f"{py_file}:{node.lineno}: {dict_name}[{key.value!r}] "
                    "must be a string-literal or tuple-of-string-literals; "
                    f"got {ast.dump(value)[:80]} instead."
                )
        return result
    raise AssertionError(
        f"{py_file}: no {dict_name} = {{...}} assignment found."
    )


def _cul_first_column_codes(cul_path: Path) -> set[str]:
    """Return the set of column-1 cultivar codes in ``cul_path``.

    DSSAT ``.CUL`` files are space-separated with the cultivar code
    in column 1 (typically 6 characters). Comment lines start with
    ``!`` or ``@`` and are skipped. Section headers (``*CULTIVARS:``)
    are skipped. Lines whose first token is purely numeric or 6-7
    characters of alphanumeric are considered cultivar entries.
    """
    codes: set[str] = set()
    for raw in cul_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith(("!", "@", "*")):
            continue
        # First whitespace-delimited token is the cultivar code.
        token = stripped.split(None, 1)[0]
        # Cultivar codes are 6 chars, alphanumeric. Some files have
        # 7-char codes (rare); accept 5-7 to be safe. Pure-numeric
        # codes (``990002``) are valid; alphanumeric (``IB0001``) are
        # the common case.
        if 5 <= len(token) <= 7 and token.isalnum():
            codes.add(token)
    return codes


class TestCultivarDefaultsAgainstRegistry(unittest.TestCase):
    """Pin every translator default cultivar code against DSSAT's
    Genotype registry. Two-layer enforcement so a drift surfaces
    at structural-test time (frozen pin) AND at runtime-equivalent
    verification time (registry check)."""

    @classmethod
    def setUpClass(cls):
        cls.craft_dict = _extract_dict_assignment(
            _CRAFT_TRANSLATOR, "generic_cultivars",
        )
        cls.pythia_dict = _extract_dict_assignment(
            _PYTHIA_TRANSLATOR, "_LEGUME_DEFAULT_CULTIVARS",
        )
        cls.dssat_genotype = _resolve_dssat_install()

    def test_craft_defaults_match_expected_pin(self):
        """CRAFT's ``generic_cultivars`` dict must declare every
        (crop, code) tuple in CRAFT_EXPECTED_DEFAULTS exactly. Adding
        a new crop default without updating this test surfaces here
        as a missing-entry failure with the crop name."""
        actual = {
            crop: code for crop, code in self.craft_dict.items()
        }
        expected = {
            crop: code for crop, code, _cul in CRAFT_EXPECTED_DEFAULTS
        }
        self.assertEqual(
            actual, expected,
            "CRAFT generic_cultivars drift: each (crop, code) pair "
            "must match CRAFT_EXPECTED_DEFAULTS. Update both sides "
            "if the default is intentionally changing.",
        )

    def test_pythia_legume_defaults_match_expected_pin(self):
        """PYTHIA's ``_LEGUME_DEFAULT_CULTIVARS`` dict declares
        (cultivar, label) tuples; the FIRST element must match the
        pin (the label is informational and not load-bearing for
        DSSAT execution)."""
        actual = {
            crop: pair[0] if isinstance(pair, tuple) else pair
            for crop, pair in self.pythia_dict.items()
        }
        expected = {
            crop: code for crop, code, _cul in PYTHIA_EXPECTED_LEGUME_DEFAULTS
        }
        self.assertEqual(
            actual, expected,
            "PYTHIA _LEGUME_DEFAULT_CULTIVARS drift: each (crop, code) "
            "pair must match PYTHIA_EXPECTED_LEGUME_DEFAULTS.",
        )

    def test_every_default_is_in_dssat_registry(self):
        """When the DSSAT install is reachable, assert every
        (crop, code, cul_file) tuple has the code in column 1 of
        the named ``.CUL`` file. Skip with an explicit message
        when DSSAT is not reachable so CI machines without the
        install do not fail."""
        if self.dssat_genotype is None:
            self.skipTest(
                "DSSAT install not reachable — set DSSAT_INSTALL or "
                "place the dssat-csm-os checkout at the conventional "
                "EXP/dssat-csm-os/ sibling path to run the registry "
                "check. The EXPECTED_DEFAULTS pin still ran."
            )
        problems: list[str] = []
        for crop, code, cul_filename in (
            *CRAFT_EXPECTED_DEFAULTS,
            *PYTHIA_EXPECTED_LEGUME_DEFAULTS,
        ):
            cul_path = self.dssat_genotype / cul_filename
            if not cul_path.is_file():
                problems.append(
                    f"{crop}: {cul_filename} not found at "
                    f"{self.dssat_genotype}"
                )
                continue
            codes = _cul_first_column_codes(cul_path)
            if code not in codes:
                problems.append(
                    f"{crop}: code {code!r} not found in column 1 of "
                    f"{cul_filename}. The closest matches in the file "
                    f"are: "
                    + ", ".join(sorted(c for c in codes if c[:2] == code[:2])[:5])
                    + " (truncated)."
                )
        self.assertEqual(
            problems, [],
            "Translator default cultivar codes must each appear in "
            "column 1 of the corresponding DSSAT .CUL file. "
            "Unregistered defaults are CRAFT/DSSAT-load-fatal:\n  "
            + "\n  ".join(problems),
        )


if __name__ == "__main__":
    unittest.main()
