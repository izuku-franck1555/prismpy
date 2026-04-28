"""F15 — PYTHIA TAV provenance description plain-language translation.

The pre-F15 wording "PYTHIA TAV: unweighted arithmetic mean of
daily mean temperatures" leaked the DSSAT TAV parameter name + the
statistical method into a researcher-facing surface (Methods tab
via the prismweb provenance reader). Per durable lesson #7
(CLI-artifact-leak), human-readable description fields must
describe the user-visible meaning rather than the underlying CLI
parameter name. The new wording — "Average annual temperature
used for soil thermal layer calibration" — describes what the TAV
value DOES for the simulation.

The technical aggregation-method detail (`np.mean(tmeans)`,
unweighted, day-equal-weight) moves to the rationale field below
the description so Dr. Kofi's audit-grep continuity stays intact.
"""

from __future__ import annotations

import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent.parent
PYTHIA_TRANSLATOR = REPO / "src" / "prismpy" / "translators" / "pythia" / "translator.py"


class TestTavProvenanceDescription(unittest.TestCase):
    """The provenance description surfaces in the Methods tab via
    the prismweb provenance reader. Plain-language copy keeps
    Aminata's first-time path readable; the technical method
    stays on the ``rationale`` field below the description for
    Dr. Kofi's audit grep."""

    @classmethod
    def setUpClass(cls):
        cls.src = PYTHIA_TRANSLATOR.read_text(encoding="utf-8")

    def test_old_cli_artifact_phrasing_retired(self):
        """The pre-F15 wording leaked the DSSAT TAV parameter name
        + the statistical method into a researcher-facing
        surface. The exact substring must NOT appear as a
        description value anymore — only the new plain-language
        copy is the canonical render."""
        self.assertNotIn(
            'description=(\n                    "PYTHIA TAV: unweighted',
            self.src,
            "PYTHIA TAV CLI-artifact wording must NOT appear as a "
            "provenance description value (durable lesson #7).",
        )

    def test_plain_language_description_present(self):
        """The Python source carries the description as adjacent
        string literals (split for line-length); the test checks
        the two halves separately so the line-wrap shape doesn't
        force the assertion to track formatting choices."""
        self.assertIn(
            "Average annual temperature used for soil thermal", self.src,
        )
        self.assertIn("layer calibration", self.src)

    def test_technical_rationale_preserved_on_rationale_field(self):
        """Dr. Kofi's audit grep must still find the unweighted-
        arithmetic-mean detail; F15 moves it from the user-facing
        description to the audit-facing rationale, not deletes
        it."""
        # Anchor the trail with two phrases the rationale carries:
        # the method anchor + the DSSAT TAV expectation. Both must
        # appear somewhere in the file for the audit lineage.
        self.assertIn("unweighted arithmetic mean", self.src)
        self.assertIn("DSSAT", self.src)


if __name__ == "__main__":
    unittest.main()
