"""Pin cdsapi as a hard required dependency.

The SARRA-Py climate retrieve flow imports ``cdsapi`` for the
Copernicus Climate Data Store API client used by ``AgERA5Source``.
The library was previously declared only in
``[project.optional-dependencies] agera5``, which let any prismpy
install without the ``[agera5]`` extra silently lose AgERA5
retrieval — a fresh venv would import ``agera5.py`` cleanly thanks
to the ``try/except ImportError`` guard at
``agera5.py:cdsapi_available``, and the silent-skip path would
return ``self._cdsapi_available = False``, ``AgERA5Source.retrieve``
would no-op, and the SARRA-Py climate dict would survive with only
the TAMSAT half (1/4 variables present).

Promoting cdsapi to ``[project] dependencies`` closes that gap at
configuration-time. The structural pin below asserts the
declaration is in place; an anti-mutation drill that reverts the
pyproject change fires this test with a clear "missing required
declaration" diagnostic.

Anti-mutation drill: move ``cdsapi`` back to ``[project.optional-
dependencies]`` → this test fails because the required-deps regex
no longer matches.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


class TestCdsapiDeclaredRequired(unittest.TestCase):
    """``cdsapi`` must be declared in ``[project] dependencies``.

    The library is a hard import inside the SARRA-Py climate retrieve
    flow; an undeclared install can silently drop AgERA5 retrieval
    and emit only TAMSAT data, producing a partial climate dict that
    drives downstream warnings the user can mistake for an
    operational issue rather than a configuration gap.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = _PYPROJECT.read_text(encoding="utf-8")

    def test_cdsapi_in_required_dependencies(self):
        """The ``[project] dependencies`` array must contain a
        cdsapi pin. The regex matches whitespace + version-spec
        flexibility so a future floor bump (``>=0.7.0``) does not
        break the pin without an intentional contract review."""
        # Locate the [project] dependencies = [...] block specifically;
        # avoid false-positives from optional-dependencies that may
        # also have a cdsapi entry (during a future intentional
        # rollback that the test would catch).
        match = re.search(
            r"^dependencies\s*=\s*\[(?P<body>.*?)^\]",
            self.src, re.DOTALL | re.MULTILINE,
        )
        self.assertIsNotNone(
            match,
            "pyproject.toml must contain a top-level "
            "[project] dependencies array.",
        )
        body = match.group("body")
        self.assertRegex(
            body, r'"\s*cdsapi\s*>=',
            "pyproject.toml [project] dependencies must declare "
            "cdsapi (the Copernicus Climate Data Store client used "
            "by AgERA5Source). Promoting cdsapi to a required dep "
            "prevents the silent-skip path that surfaces as a "
            "partial-climate package on every SARRA-Py project.",
        )

    def test_cdsapi_not_in_optional_agera5_extras(self):
        """The ``[project.optional-dependencies] agera5`` extras
        group must NOT carry a cdsapi pin once the migration is
        complete. Leaving the optional declaration in place would
        let pip resolve the optional-extras path and bypass the
        required-dep guarantee on `pip install -e .` invocations
        that happen to ask for the extras group.

        Scope the search to literal extras-group bodies so that
        ``cdsapi`` mentions inside the migration-rationale comment
        on ``[project] dependencies`` (above) and the test's own
        diagnostic strings do not produce a false-positive match.
        """
        # Each extras-group body lives between ``<group> = [`` and the
        # following ``]`` on its own line. Iterate every group body
        # and assert none of them declare cdsapi.
        bodies = re.findall(
            r"(?m)^[A-Za-z0-9_-]+\s*=\s*\[(?P<body>[^\]]*)\]",
            self.src,
        )
        # The ``re.findall`` group above also captures the
        # ``dependencies = [...]`` and ``classifiers = [...]`` arrays,
        # which is fine — cdsapi MUST appear in dependencies and
        # MUST NOT appear anywhere else with the cdsapi-pin shape.
        # Filter to bodies inside ``[project.optional-dependencies]``
        # by re-locating that section once and extracting only the
        # arrays between its header and the next top-level table.
        opt_match = re.search(
            r"\[project\.optional-dependencies\]\s*\n"
            r"(?P<section>.*?)"
            r"(?=^\[|\Z)",
            self.src, re.DOTALL | re.MULTILINE,
        )
        if opt_match is None:
            return
        opt_bodies = re.findall(
            r"(?m)^[A-Za-z0-9_-]+\s*=\s*\[(?P<body>[^\]]*)\]",
            opt_match.group("section"),
        )
        for body in opt_bodies:
            self.assertNotRegex(
                body, r'"\s*cdsapi\s*>=',
                "pyproject.toml [project.optional-dependencies] must "
                "NOT declare cdsapi once it is promoted to required. "
                "Keep the migration clean by removing the agera5 "
                "extras-group entirely (it had cdsapi as its sole "
                "member).",
            )


if __name__ == "__main__":
    unittest.main()
