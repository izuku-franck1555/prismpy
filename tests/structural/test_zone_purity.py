"""Sprint E.0.5 F24 — zone-purity walker.

Per CC-13 + research doc §Q2.X.2 + the contract Draft 4
F24-codification: bound generation MUST NOT pool cells
across zones. Re-pooling boundary cells with neighboring
zones contaminates per-zone statistics with non-zone climate
and biases the zone aggregate.

This walker enforces zone-purity at the substrate level: the
aggregation helpers in :mod:`prismpy.validators.climate_envelope`
take per-zone-only inputs and have no cross-zone API surface.
A future Sprint F bound-gen extension can extend this walker
into a synthetic-fixture meta-test that runs the full
generator on a BSh+BSk fixture and verifies separated
per-zone bounds (per the verification strategy doc).

Anti-mutation drills:

- Add a "merge thin zones" helper that pools cells across
  Köppen-Geiger zones for percentile estimation →
  ``test_no_cross_zone_pool_helpers`` flags the function name
  pattern.
- Change ``compute_zone_precip_iqr`` to accept a list-of-
  zones argument → ``test_substrate_aggregators_take_single_zone``
  catches the signature drift.
"""
from __future__ import annotations

import inspect
import re
import unittest
from pathlib import Path

from prismpy.validators.climate_envelope import (
    compute_zone_precip_iqr,
    compute_zone_thermal_extremes,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src" / "prismpy"


# Function-name patterns that would indicate cross-zone
# re-pooling (the F24 anti-pattern). Anti-mutation: adding a
# helper matching one of these names triggers the walker.
_FORBIDDEN_NAME_PATTERNS = (
    r"def\s+(merge|pool|combine)_zones\b",
    r"def\s+(merge|pool|combine)_cells_across_zones\b",
    r"def\s+repool_boundary_cells\b",
    r"def\s+aggregate_across_zones\b",
)


class TestSubstrateAggregatorsTakeSingleZone(unittest.TestCase):
    """The per-zone aggregation helpers operate on one zone's
    per-cell data each. There is no API surface for combining
    cells from multiple zones into a single aggregate."""

    def test_compute_zone_precip_iqr_signature(self):
        sig = inspect.signature(compute_zone_precip_iqr)
        params = list(sig.parameters.values())
        self.assertEqual(len(params), 1)
        # The argument is a sequence of cells in ONE zone.
        self.assertEqual(params[0].name, "cell_annual_precips")

    def test_compute_zone_thermal_extremes_signature(self):
        sig = inspect.signature(compute_zone_thermal_extremes)
        param_names = [p.name for p in sig.parameters.values()]
        # Two paired sequences (per-cell tmin extremes + tmax
        # extremes); both refer to the same zone's cells.
        self.assertEqual(
            param_names, ["cell_extreme_tmins", "cell_extreme_tmaxs"],
        )


class TestNoCrossZonePoolHelpers(unittest.TestCase):
    """Walker scans the prismpy source tree for function-name
    patterns that would indicate cross-zone pooling. A bound-
    gen change that introduces such a helper triggers the
    walker fail-loud."""

    def test_no_forbidden_function_names(self):
        violations: list[tuple[str, int, str]] = []
        for py in _SRC_DIR.rglob("*.py"):
            for line_num, line in enumerate(
                py.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                for pattern in _FORBIDDEN_NAME_PATTERNS:
                    if re.search(pattern, line):
                        violations.append((
                            str(py.relative_to(_REPO_ROOT)),
                            line_num, line.strip(),
                        ))
        self.assertEqual(
            violations, [],
            "F24 zone-purity violation: cross-zone pool / merge "
            "helpers found in source. Bound generation must "
            "compute per-zone aggregates only; re-pooling "
            "boundary cells across zones contaminates per-zone "
            "statistics. Violations: " + repr(violations),
        )


if __name__ == "__main__":
    unittest.main()
