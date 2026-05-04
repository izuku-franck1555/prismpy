"""Sprint E.0.5 AC-Q3-A-a/b/c — climate-envelope verdict logic.

Pins the four-state Stage 1 verdict matrix:

* ``COMPATIBLE`` — full IQR within precip envelope, OR no
  thermal kill at extremes
* ``MARGINAL_HETEROGENEOUS`` — precip P50 in but IQR
  straddles
* ``MARGINAL_THERMAL_SEASONAL`` — both cold-kill AND
  heat-kill (Sprint F refines via crop seasonal window)
* ``INCOMPATIBLE`` — precip P50 outside, OR cold-kill alone,
  OR heat-kill alone

Anti-mutation drills:

- Revert AC-Q3-A-a IQR to single-point zone-mean → south-
  Sahel BSh fixture flips from MARGINAL to false-COMPATIBLE
  / false-INCOMPATIBLE.
- Revert AC-Q3-A-c aggregation order to zone-mean-of-extremes
  → Cfa-shape fixture (90 warm extremes + 10 cold extremes)
  flips from INCOMPATIBLE to false-COMPATIBLE.
- Introduce any 0.7 / 0.9 / 1.1 / 1.3 multiplier into the
  module → ``test_no_buffer_multipliers_in_source`` fails
  (AC-Q3-A-b structural pin).
"""
from __future__ import annotations

import math
import re
import unittest
from pathlib import Path

import numpy as np

from prismpy.validators.climate_envelope import (
    CompatibilityVerdict,
    aggregate_verdicts,
    compare_precip_iqr,
    compare_thermal_extremes,
    compute_zone_precip_iqr,
    compute_zone_thermal_extremes,
)


# ECOCROP envelope for maize (verbatim from research doc
# §Q3.X.1; matches the bundled ``ecocrop_envelopes.json``).
_MAIZE_TMIN: float = 10.0
_MAIZE_TMAX: float = 47.0
_MAIZE_RMIN: float = 400.0
_MAIZE_RMAX: float = 1800.0


class TestCompatibilityVerdictEnum(unittest.TestCase):
    """The 4-state verdict enum is the canonical Stage 1
    verdict vocabulary."""

    def test_four_verdicts_total(self):
        self.assertEqual(len(list(CompatibilityVerdict)), 4)

    def test_verdict_values_are_strings(self):
        # Subclassing (str, Enum) keeps Python 3.10 compat;
        # values match the canonical lowercase-snake names.
        self.assertEqual(
            CompatibilityVerdict.COMPATIBLE.value, "compatible",
        )
        self.assertEqual(
            CompatibilityVerdict.MARGINAL_HETEROGENEOUS.value,
            "marginal_heterogeneous",
        )
        self.assertEqual(
            CompatibilityVerdict.MARGINAL_THERMAL_SEASONAL.value,
            "marginal_thermal_seasonal",
        )
        self.assertEqual(
            CompatibilityVerdict.INCOMPATIBLE.value, "incompatible",
        )


class TestComparePrecipIQR(unittest.TestCase):
    """Per AC-Q3-A-a + AC-Q3-A-b: three-state verdict using
    inclusive boundary predicates on P50 + IQR endpoints."""

    def test_full_iqr_in_envelope_compatible(self):
        # Maize × tropical-savanna BSh: P25=600, P50=900, P75=1200
        # vs RMIN=400, RMAX=1800 → full IQR in envelope.
        self.assertEqual(
            compare_precip_iqr(
                p25=600.0, p50=900.0, p75=1200.0,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            ),
            CompatibilityVerdict.COMPATIBLE,
        )

    def test_p50_below_rmin_incompatible(self):
        # North-Sahel maize: P25=200, P50=300, P75=380 vs
        # RMIN=400 → median below envelope.
        self.assertEqual(
            compare_precip_iqr(
                p25=200.0, p50=300.0, p75=380.0,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            ),
            CompatibilityVerdict.INCOMPATIBLE,
        )

    def test_p50_above_rmax_incompatible(self):
        self.assertEqual(
            compare_precip_iqr(
                p25=1700.0, p50=2000.0, p75=2300.0,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            ),
            CompatibilityVerdict.INCOMPATIBLE,
        )

    def test_p25_below_rmin_marginal(self):
        # South-Sahel maize: P25=350, P50=500, P75=800 vs
        # RMIN=400 → median in env, P25 below → marginal.
        self.assertEqual(
            compare_precip_iqr(
                p25=350.0, p50=500.0, p75=800.0,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            ),
            CompatibilityVerdict.MARGINAL_HETEROGENEOUS,
        )

    def test_p75_above_rmax_marginal(self):
        # Wet-tropics maize: P25=600, P50=1500, P75=2000 vs
        # RMAX=1800 → median in env, P75 above → marginal.
        self.assertEqual(
            compare_precip_iqr(
                p25=600.0, p50=1500.0, p75=2000.0,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            ),
            CompatibilityVerdict.MARGINAL_HETEROGENEOUS,
        )

    def test_p50_at_rmin_compatible_when_iqr_above(self):
        # P50 == RMIN (boundary inclusive) and P25 == RMIN +
        # P75 == RMIN -> all at boundary, full IQR in env.
        self.assertEqual(
            compare_precip_iqr(
                p25=400.0, p50=400.0, p75=400.0,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            ),
            CompatibilityVerdict.COMPATIBLE,
        )

    def test_p50_at_rmax_compatible_when_iqr_below(self):
        self.assertEqual(
            compare_precip_iqr(
                p25=1700.0, p50=1800.0, p75=1800.0,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            ),
            CompatibilityVerdict.COMPATIBLE,
        )

    def test_p25_at_rmin_compatible(self):
        # Boundary case: P25 == RMIN exactly is inclusive
        # (Option α LEAN); not flagged marginal.
        self.assertEqual(
            compare_precip_iqr(
                p25=400.0, p50=900.0, p75=1500.0,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            ),
            CompatibilityVerdict.COMPATIBLE,
        )

    def test_p75_at_rmax_compatible(self):
        self.assertEqual(
            compare_precip_iqr(
                p25=600.0, p50=1100.0, p75=1800.0,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            ),
            CompatibilityVerdict.COMPATIBLE,
        )

    def test_inverted_iqr_raises(self):
        with self.assertRaises(ValueError):
            compare_precip_iqr(
                p25=900.0, p50=600.0, p75=400.0,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            )

    def test_inverted_envelope_raises(self):
        with self.assertRaises(ValueError):
            compare_precip_iqr(
                p25=600.0, p50=900.0, p75=1200.0,
                rmin=1800.0, rmax=400.0,
            )

    def test_equal_envelope_raises(self):
        # RMIN == RMAX -> degenerate envelope, fail loud.
        with self.assertRaises(ValueError):
            compare_precip_iqr(
                p25=600.0, p50=900.0, p75=1200.0,
                rmin=900.0, rmax=900.0,
            )

    def test_nan_raises(self):
        with self.assertRaises(ValueError):
            compare_precip_iqr(
                p25=float("nan"), p50=900.0, p75=1200.0,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            )

    def test_inf_raises(self):
        with self.assertRaises(ValueError):
            compare_precip_iqr(
                p25=600.0, p50=900.0, p75=math.inf,
                rmin=_MAIZE_RMIN, rmax=_MAIZE_RMAX,
            )


class TestCompareThermalExtremes(unittest.TestCase):
    """Per AC-Q3-A-c: extremes-aware verdict. Cold-kill =
    P10 of per-cell extreme tmins below crop TMIN; heat-kill
    = P90 of per-cell extreme tmaxs above crop TMAX."""

    def test_no_kill_compatible(self):
        # Tropical maize: P10 extreme tmin = 15°C (above 10),
        # P90 extreme tmax = 40°C (below 47).
        self.assertEqual(
            compare_thermal_extremes(
                zone_p10_extreme_tmin=15.0,
                zone_p90_extreme_tmax=40.0,
                crop_tmin=_MAIZE_TMIN, crop_tmax=_MAIZE_TMAX,
            ),
            CompatibilityVerdict.COMPATIBLE,
        )

    def test_cold_kill_only_incompatible(self):
        # Andean maize: P10 extreme tmin = -2°C, P90 extreme
        # tmax = 25°C → cold-kill only → INCOMPATIBLE.
        self.assertEqual(
            compare_thermal_extremes(
                zone_p10_extreme_tmin=-2.0,
                zone_p90_extreme_tmax=25.0,
                crop_tmin=_MAIZE_TMIN, crop_tmax=_MAIZE_TMAX,
            ),
            CompatibilityVerdict.INCOMPATIBLE,
        )

    def test_heat_kill_only_incompatible(self):
        # Hyper-arid maize: P10 extreme tmin = 15°C, P90
        # extreme tmax = 50°C → heat-kill only → INCOMPATIBLE.
        self.assertEqual(
            compare_thermal_extremes(
                zone_p10_extreme_tmin=15.0,
                zone_p90_extreme_tmax=50.0,
                crop_tmin=_MAIZE_TMIN, crop_tmax=_MAIZE_TMAX,
            ),
            CompatibilityVerdict.INCOMPATIBLE,
        )

    def test_both_kills_marginal_thermal_seasonal(self):
        # Continental Sahel: P10 extreme tmin = 5°C (cold-
        # kill), P90 extreme tmax = 50°C (heat-kill) → both
        # extremes → MARGINAL_THERMAL_SEASONAL (Sprint F
        # refines via crop's growing-season window).
        self.assertEqual(
            compare_thermal_extremes(
                zone_p10_extreme_tmin=5.0,
                zone_p90_extreme_tmax=50.0,
                crop_tmin=_MAIZE_TMIN, crop_tmax=_MAIZE_TMAX,
            ),
            CompatibilityVerdict.MARGINAL_THERMAL_SEASONAL,
        )

    def test_p10_at_tmin_compatible(self):
        # Boundary case: P10 == TMIN is inclusive; no cold-kill.
        self.assertEqual(
            compare_thermal_extremes(
                zone_p10_extreme_tmin=10.0,  # == TMIN
                zone_p90_extreme_tmax=40.0,
                crop_tmin=_MAIZE_TMIN, crop_tmax=_MAIZE_TMAX,
            ),
            CompatibilityVerdict.COMPATIBLE,
        )

    def test_p90_at_tmax_compatible(self):
        # Boundary case: P90 == TMAX is inclusive.
        self.assertEqual(
            compare_thermal_extremes(
                zone_p10_extreme_tmin=15.0,
                zone_p90_extreme_tmax=47.0,  # == TMAX
                crop_tmin=_MAIZE_TMIN, crop_tmax=_MAIZE_TMAX,
            ),
            CompatibilityVerdict.COMPATIBLE,
        )

    def test_inverted_thermal_envelope_raises(self):
        with self.assertRaises(ValueError):
            compare_thermal_extremes(
                zone_p10_extreme_tmin=15.0,
                zone_p90_extreme_tmax=40.0,
                crop_tmin=47.0, crop_tmax=10.0,
            )

    def test_nan_raises(self):
        with self.assertRaises(ValueError):
            compare_thermal_extremes(
                zone_p10_extreme_tmin=float("nan"),
                zone_p90_extreme_tmax=40.0,
                crop_tmin=_MAIZE_TMIN, crop_tmax=_MAIZE_TMAX,
            )

    def test_inf_raises(self):
        with self.assertRaises(ValueError):
            compare_thermal_extremes(
                zone_p10_extreme_tmin=15.0,
                zone_p90_extreme_tmax=math.inf,
                crop_tmin=_MAIZE_TMIN, crop_tmax=_MAIZE_TMAX,
            )


class TestComputeZonePrecipIQR(unittest.TestCase):
    """Per AC-Q3-A-a aggregation: zone P25/P50/P75 of per-cell
    annual mean precip across cells."""

    def test_returns_p25_p50_p75_keys(self):
        result = compute_zone_precip_iqr([300.0, 500.0, 700.0])
        self.assertEqual(set(result.keys()), {"p25", "p50", "p75"})

    def test_iqr_ordering_preserved(self):
        result = compute_zone_precip_iqr(
            [200.0, 400.0, 600.0, 800.0, 1000.0],
        )
        self.assertLessEqual(result["p25"], result["p50"])
        self.assertLessEqual(result["p50"], result["p75"])

    def test_known_values_match_numpy_linear(self):
        # Sanity: matches np.quantile method='linear' explicitly.
        cells = [200.0, 400.0, 600.0, 800.0, 1000.0]
        result = compute_zone_precip_iqr(cells)
        self.assertAlmostEqual(
            result["p25"],
            float(np.quantile(cells, 0.25, method="linear")),
        )
        self.assertAlmostEqual(
            result["p50"],
            float(np.quantile(cells, 0.50, method="linear")),
        )
        self.assertAlmostEqual(
            result["p75"],
            float(np.quantile(cells, 0.75, method="linear")),
        )

    def test_empty_sequence_raises(self):
        with self.assertRaises(ValueError):
            compute_zone_precip_iqr([])

    def test_nan_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            compute_zone_precip_iqr([300.0, float("nan"), 700.0])


class TestComputeZoneThermalExtremes(unittest.TestCase):
    """Per AC-Q3-A-c aggregation order: per-cell extremes
    FIRST (caller-pre-aggregated), then zone P10/P90 across
    cells."""

    def test_returns_p10_p90_keys(self):
        result = compute_zone_thermal_extremes(
            cell_extreme_tmins=[-5.0, 0.0, 5.0],
            cell_extreme_tmaxs=[35.0, 40.0, 45.0],
        )
        self.assertEqual(
            set(result.keys()),
            {"p10_extreme_tmin", "p90_extreme_tmax"},
        )

    def test_p10_below_p90(self):
        result = compute_zone_thermal_extremes(
            cell_extreme_tmins=[-22.0, 0.0, 5.0, 10.0, 15.0],
            cell_extreme_tmaxs=[25.0, 30.0, 35.0, 40.0, 50.0],
        )
        self.assertLess(
            result["p10_extreme_tmin"],
            result["p90_extreme_tmax"],
        )

    def test_p10_distinguishes_from_zone_mean_anti_mutation(self):
        # Cfa Corn Belt synthetic: 88 cells with warm extreme
        # tmin (16°C), 12 cells with cold extreme tmin (-22°C).
        # Zone-mean of extreme tmins ≈ 11.4°C → above maize
        # TMIN=10°C → a "zone-mean implementation" (the F22-
        # forbidden pattern) would false-positive "compatible".
        # P10 of extreme tmins = -22°C (≥ 11 cold cells push
        # the P10 cleanly into the cold tail) → correctly
        # INCOMPATIBLE.
        cell_extreme_tmins = [16.0] * 88 + [-22.0] * 12
        cell_extreme_tmaxs = [25.0] * 100  # all moderate
        zone_mean_tmin = sum(cell_extreme_tmins) / len(cell_extreme_tmins)
        # Sanity: mean is above maize TMIN (the false-positive
        # the AC-Q3-A-c aggregation order is designed to avoid).
        self.assertGreater(zone_mean_tmin, _MAIZE_TMIN)
        # Aggregation: P10 of extremes is far below TMIN.
        aggs = compute_zone_thermal_extremes(
            cell_extreme_tmins, cell_extreme_tmaxs,
        )
        self.assertLess(aggs["p10_extreme_tmin"], _MAIZE_TMIN)
        # End-to-end verdict: INCOMPATIBLE (cold-kill alone).
        verdict = compare_thermal_extremes(
            zone_p10_extreme_tmin=aggs["p10_extreme_tmin"],
            zone_p90_extreme_tmax=aggs["p90_extreme_tmax"],
            crop_tmin=_MAIZE_TMIN, crop_tmax=_MAIZE_TMAX,
        )
        self.assertEqual(verdict, CompatibilityVerdict.INCOMPATIBLE)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            compute_zone_thermal_extremes(
                cell_extreme_tmins=[-5.0, 0.0, 5.0],
                cell_extreme_tmaxs=[35.0, 40.0],  # length mismatch
            )

    def test_empty_sequences_raise(self):
        with self.assertRaises(ValueError):
            compute_zone_thermal_extremes(
                cell_extreme_tmins=[],
                cell_extreme_tmaxs=[],
            )

    def test_nan_in_sequence_raises(self):
        with self.assertRaises(ValueError):
            compute_zone_thermal_extremes(
                cell_extreme_tmins=[-5.0, float("nan"), 5.0],
                cell_extreme_tmaxs=[35.0, 40.0, 45.0],
            )


class TestAggregateVerdicts(unittest.TestCase):
    """Worst-case-wins aggregation across variables. Order:
    INCOMPATIBLE > MARGINAL_THERMAL_SEASONAL >
    MARGINAL_HETEROGENEOUS > COMPATIBLE."""

    def test_all_compatible_returns_compatible(self):
        self.assertEqual(
            aggregate_verdicts([
                CompatibilityVerdict.COMPATIBLE,
                CompatibilityVerdict.COMPATIBLE,
            ]),
            CompatibilityVerdict.COMPATIBLE,
        )

    def test_marginal_overrides_compatible(self):
        self.assertEqual(
            aggregate_verdicts([
                CompatibilityVerdict.COMPATIBLE,
                CompatibilityVerdict.MARGINAL_HETEROGENEOUS,
            ]),
            CompatibilityVerdict.MARGINAL_HETEROGENEOUS,
        )

    def test_thermal_seasonal_overrides_heterogeneous(self):
        self.assertEqual(
            aggregate_verdicts([
                CompatibilityVerdict.MARGINAL_HETEROGENEOUS,
                CompatibilityVerdict.MARGINAL_THERMAL_SEASONAL,
            ]),
            CompatibilityVerdict.MARGINAL_THERMAL_SEASONAL,
        )

    def test_incompatible_overrides_all(self):
        self.assertEqual(
            aggregate_verdicts([
                CompatibilityVerdict.MARGINAL_HETEROGENEOUS,
                CompatibilityVerdict.MARGINAL_THERMAL_SEASONAL,
                CompatibilityVerdict.INCOMPATIBLE,
                CompatibilityVerdict.COMPATIBLE,
            ]),
            CompatibilityVerdict.INCOMPATIBLE,
        )

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            aggregate_verdicts([])

    def test_single_verdict_returned_as_is(self):
        for verdict in CompatibilityVerdict:
            with self.subTest(verdict=verdict):
                self.assertEqual(
                    aggregate_verdicts([verdict]), verdict,
                )


class TestNoMultiplierStructural(unittest.TestCase):
    """AC-Q3-A-b: no buffer multipliers in the envelope-
    comparison module. Strict envelope only. Anti-mutation:
    introducing 0.7 / 0.9 / 1.1 / 1.3 floats in multiplication
    context fails this structural pin."""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[2]
        cls.source = (
            repo_root
            / "src" / "prismpy" / "validators" / "climate_envelope.py"
        ).read_text(encoding="utf-8")

    def test_no_buffer_multipliers_in_source(self):
        # Forbidden: multiplication (or division) by 0.7-0.9
        # or 1.1-1.3 of any envelope value (RMIN/RMAX/TMIN/TMAX
        # or any percentile). The pattern looks for the float
        # literal followed by ``*`` (with optional whitespace).
        forbidden_patterns = [
            r"\b0\.7\d*\s*\*",
            r"\b0\.8\d*\s*\*",
            r"\b0\.9\d*\s*\*",
            r"\b1\.1\d*\s*\*",
            r"\b1\.2\d*\s*\*",
            r"\b1\.3\d*\s*\*",
        ]
        for pattern in forbidden_patterns:
            with self.subTest(pattern=pattern):
                self.assertIsNone(
                    re.search(pattern, self.source),
                    f"Forbidden buffer multiplier matching "
                    f"{pattern!r} found in climate_envelope.py "
                    f"source. AC-Q3-A-b prohibits multipliers "
                    f"around envelope endpoints.",
                )

    def test_strict_envelope_predicates_present(self):
        # Sanity-check: the module uses strict comparisons
        # (``< rmin``, ``> rmax``, ``< crop_tmin``, ``> crop_tmax``)
        # rather than buffered comparisons. This pin catches a
        # subtle reversal where someone swaps ``<`` for ``<=``
        # at the boundary; the unit tests above cover the
        # intended boundary semantics.
        self.assertIn("p50 < rmin", self.source)
        self.assertIn("p50 > rmax", self.source)


class TestNumpyQuantileDeterminism(unittest.TestCase):
    """Sanity probe: the substrate's pinned ``method='linear'``
    matches np.quantile's documented behavior. Cross-platform
    reproducibility (Linux + OpenBLAS thread-pinned vs other)
    is exercised at the CI layer; this is the unit-level
    floor."""

    def test_quantile_known_scalar(self):
        result = float(np.quantile(
            [1.0, 2.0, 3.0, 4.0, 5.0], 0.95, method="linear",
        ))
        self.assertAlmostEqual(result, 4.8, places=10)

    def test_quantile_idempotent(self):
        # Same input + same method produces byte-identical
        # output across calls (assuming thread pin holds).
        seq = [10.0, 20.0, 30.0, 40.0, 50.0]
        a = float(np.quantile(seq, 0.25, method="linear"))
        b = float(np.quantile(seq, 0.25, method="linear"))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
