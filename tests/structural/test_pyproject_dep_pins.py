"""Sprint E.0.5 AC-Q2-A1-d — pyproject.toml dependency pins.

The bound-generation determinism contract requires that the
three scientific deps that drive the bound-generation math —
``numpy``, ``rasterio``, ``xarray`` — carry both a floor and a
major-version upper bound. The bound is paired with the
AC-Q2-B1 CI thread-pin set so cross-platform / cross-version
output stays byte-identical.

The floors preserve existing constraints (do not lower; the
1.24 numpy floor was caught and protected per builder Adj 3
during contract review). The upper bounds prevent silent
adoption of a major numpy/rasterio/xarray bump that would
shift quantile defaults, pixel-sampling semantics, or
reduction-order conventions and require a manual
``bounds_version`` ratchet rather than a free upgrade.

This test pins the constraints structurally so a future
contributor cannot silently drop or relax them. Anti-mutation:
delete an upper bound or lower a floor → assertion fails with
the specific dep name.

The Python version requirement is also pinned (>=3.10,<3.13)
so the floor-bump from 3.9 to 3.10 stays anchored.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


# Each tuple: (dep name, expected version specifier substring).
# The specifier is matched as a substring so trailing comments
# or pip-extras don't break the pin.
EXPECTED_DEP_PINS = (
    ("numpy",    ">=1.24,<2.0"),
    ("rasterio", ">=1.3,<2.0"),
    ("xarray",   ">=2023.1,<2025.0"),
    ("pydantic", ">=2.0,<3.0"),
)


class TestPyprojectDependencyPins(unittest.TestCase):
    """The four scientific deps must carry both a floor AND a
    major-version upper bound so determinism is contractual."""

    @classmethod
    def setUpClass(cls):
        cls.src = _PYPROJECT.read_text(encoding="utf-8")

    def test_python_floor_at_310_with_upper_bound(self):
        """Python 3.9 has been dropped; 3.10-3.12 is the
        supported window. The upper bound is included so a
        Python 3.13 release does not silently change semantics
        on consumers without a contract-review trip."""
        self.assertRegex(
            self.src,
            r'requires-python\s*=\s*"\s*>=\s*3\.10\s*,\s*<\s*3\.13\s*"',
            "pyproject.toml requires-python must be "
            "'>=3.10,<3.13' (drop 3.9; cap below 3.13).",
        )

    def test_each_pin_present(self):
        for dep, spec in EXPECTED_DEP_PINS:
            with self.subTest(dep=dep):
                # Match the dep name + arbitrary whitespace +
                # the expected specifier in the dependencies list.
                # Allow either single or double quotes in the
                # TOML string literal.
                pattern = rf'"\s*{re.escape(dep)}\s*{re.escape(spec)}\s*"'
                self.assertRegex(
                    self.src, pattern,
                    f"pyproject.toml must declare "
                    f"'{dep}{spec}' as a dependency pin. "
                    f"Sprint E.0.5 AC-Q2-A1-d requires the "
                    f"upper bound for byte-identical bound-gen.",
                )

    def test_numpy_floor_preserved_at_124(self):
        """Anti-mutation drill — the 1.24 floor was at risk
        of accidental drift to 1.22 during Draft 1 review (see
        builder Adj 3). Pin the floor explicitly so a future
        contributor cannot lower it without this test
        catching the drop."""
        # The dep line is "numpy>=1.24,<2.0". Refuse anything
        # whose floor digit pattern is "1.X" with X<24.
        self.assertNotRegex(
            self.src,
            r'"\s*numpy\s*>=\s*1\.(0|1|2[0-3])',
            "numpy floor must NOT be lower than 1.24. The "
            "1.24 floor was caught and protected during contract "
            "review; do not lower it.",
        )

    def test_rasterio_floor_preserved_at_13(self):
        self.assertNotRegex(
            self.src,
            r'"\s*rasterio\s*>=\s*1\.[012]',
            "rasterio floor must NOT be lower than 1.3.",
        )

    def test_xarray_floor_preserved_at_20231(self):
        """xarray pre-2023.1 lacks several reduction-order
        guarantees the bound generator depends on. Pin the
        floor so a regression to 2022.x is impossible."""
        self.assertNotRegex(
            self.src,
            r'"\s*xarray\s*>=\s*20(1\d|22)',
            "xarray floor must NOT be lower than 2023.1.",
        )


if __name__ == "__main__":
    unittest.main()
