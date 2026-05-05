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

    def test_inverted_zone_aggregates_rejected(self):
        # P10 of cold extremes greater than P90 of hot extremes
        # is physically impossible; without this guard the
        # comparator returned COMPATIBLE silently because both
        # kill predicates evaluate False (cold-tail above TMIN,
        # hot-tail below TMAX). Silent COMPATIBLE on swapped-
        # variable inputs would mask an upstream aggregation
        # bug; reject fail-loud per honest-signal contract.
        with self.assertRaises(ValueError) as ctx:
            compare_thermal_extremes(
                zone_p10_extreme_tmin=50.0,  # warmer than P90
                zone_p90_extreme_tmax=40.0,  # cooler than P10
                crop_tmin=_MAIZE_TMIN, crop_tmax=_MAIZE_TMAX,
            )
        self.assertIn("P10_extreme_tmin", str(ctx.exception))
        self.assertIn("P90_extreme_tmax", str(ctx.exception))


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

    def test_inverted_per_cell_pair_rejected(self):
        # An inverted (tmin > tmax) per-cell pair indicates
        # swapped variables at the caller. Without the guard,
        # the aggregator silently produces a P10 from the
        # "tmin" sequence (actually tmax values) that could
        # fall above P90 from the "tmax" sequence — and the
        # downstream comparator would either reject the
        # aggregate (post-fix) or silently emit COMPATIBLE
        # (pre-fix). Reject at the per-cell layer so the
        # caller's bug surfaces immediately.
        with self.assertRaises(ValueError) as ctx:
            compute_zone_thermal_extremes(
                cell_extreme_tmins=[-5.0, 30.0, 5.0],  # cell 1 inverted
                cell_extreme_tmaxs=[35.0, 10.0, 45.0],  # cell 1: tmin=30 > tmax=10
            )
        self.assertIn("tmin <= tmax", str(ctx.exception))


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


class TestClimateEnvelopeValidator(unittest.TestCase):
    """Sprint E.0.5 commit 8 — :class:`ClimateEnvelopeValidator`
    wraps the function-level verdict logic from commit 6 and
    maps verdicts to canonical :class:`ValidationIssue`
    instances. Pin the EMITS frozenset + the verdict-to-issue
    mapping + the F25-shape discipline (no out-of-EMITS
    categories surface)."""

    @classmethod
    def setUpClass(cls):
        # Local imports to avoid circular import surprises in
        # the rest of this module's tests, which exercise the
        # function-level logic directly.
        from prismpy.validators.climate_envelope import (
            ClimateEnvelopeValidator,
        )
        from prismpy.validators.input_base import (
            InputValidationContext,
        )
        from prismpy.warnings.categories import WarningCategory

        cls.ClimateEnvelopeValidator = ClimateEnvelopeValidator
        cls.InputValidationContext = InputValidationContext
        cls.WarningCategory = WarningCategory

    def _maize_context(self, zone_aggregates):
        from prismpy.validators.input_base import CropEnvelope, ZoneAggregate
        envelope = CropEnvelope(
            TMIN=_MAIZE_TMIN, TMAX=_MAIZE_TMAX,
            RMIN=_MAIZE_RMIN, RMAX=_MAIZE_RMAX,
        )
        typed_aggregates = {
            zone: (
                aggs if isinstance(aggs, ZoneAggregate)
                else ZoneAggregate(**aggs)
            )
            for zone, aggs in zone_aggregates.items()
        }
        return self.InputValidationContext(
            crop_name="maize",
            crop_envelope=envelope,
            zone_aggregates=typed_aggregates,
        )

    def test_emits_frozenset(self):
        self.assertEqual(
            self.ClimateEnvelopeValidator.EMITS,
            frozenset({
                self.WarningCategory.CLIMATE_ENVELOPE_TAIL,
                self.WarningCategory.INSUFFICIENTLY_SAMPLED,
            }),
        )

    def test_compatible_zone_no_issues(self):
        ctx = self._maize_context({
            "BSh": {
                "p25": 600.0, "p50": 900.0, "p75": 1200.0,
                "p10_extreme_tmin": 15.0, "p90_extreme_tmax": 40.0,
                "n_cell_days": 2_000_000,
            },
        })
        result = self.ClimateEnvelopeValidator().validate(ctx)
        self.assertTrue(result.valid)
        self.assertEqual(result.issues, [])

    def test_marginal_heterogeneous_zone_emits_envelope_tail(self):
        ctx = self._maize_context({
            "BSh": {
                "p25": 350.0, "p50": 500.0, "p75": 800.0,  # marginal
                "p10_extreme_tmin": 15.0, "p90_extreme_tmax": 40.0,
                "n_cell_days": 2_000_000,
            },
        })
        result = self.ClimateEnvelopeValidator().validate(ctx)
        precip_issues = [
            i for i in result.issues
            if i.details.get("variable") == "precip"
        ]
        self.assertEqual(len(precip_issues), 1)
        self.assertEqual(
            precip_issues[0].category,
            self.WarningCategory.CLIMATE_ENVELOPE_TAIL.value,
        )
        self.assertEqual(precip_issues[0].severity, "warning")

    def test_marginal_thermal_seasonal_emits_envelope_tail(self):
        ctx = self._maize_context({
            "BSh": {
                "p25": 600.0, "p50": 900.0, "p75": 1200.0,
                "p10_extreme_tmin": 5.0,    # cold-kill
                "p90_extreme_tmax": 50.0,   # heat-kill
                "n_cell_days": 2_000_000,
            },
        })
        result = self.ClimateEnvelopeValidator().validate(ctx)
        thermal_issues = [
            i for i in result.issues
            if i.details.get("variable") == "thermal"
        ]
        self.assertEqual(len(thermal_issues), 1)
        self.assertEqual(
            thermal_issues[0].category,
            self.WarningCategory.CLIMATE_ENVELOPE_TAIL.value,
        )

    def test_incompatible_verdict_emits_no_issue(self):
        # Per the team-lead Decision 2: ClimateEnvelopeValidator
        # does NOT emit CROP_REGION_MISMATCH. INCOMPATIBLE
        # verdicts surface through CropPhysiologicalValidator
        # (Sprint F populates).
        ctx = self._maize_context({
            "BSh": {
                "p25": 200.0, "p50": 300.0, "p75": 380.0,  # P50 < RMIN -> incompat
                "p10_extreme_tmin": 15.0, "p90_extreme_tmax": 40.0,
                "n_cell_days": 2_000_000,
            },
        })
        result = self.ClimateEnvelopeValidator().validate(ctx)
        precip_issues = [
            i for i in result.issues
            if i.details.get("variable") == "precip"
        ]
        # No CLIMATE_ENVELOPE_TAIL for incompatible.
        self.assertEqual(len(precip_issues), 0)

    def test_insufficient_zone_emits_insufficiently_sampled(self):
        ctx = self._maize_context({
            "Csc": {
                "p25": 600.0, "p50": 900.0, "p75": 1200.0,
                "p10_extreme_tmin": 15.0, "p90_extreme_tmax": 40.0,
                "n_cell_days": 50_000,  # below 1M threshold
            },
        })
        result = self.ClimateEnvelopeValidator().validate(ctx)
        self.assertEqual(len(result.issues), 1)
        self.assertEqual(
            result.issues[0].category,
            self.WarningCategory.INSUFFICIENTLY_SAMPLED.value,
        )

    def test_insufficient_zone_skips_verdict(self):
        ctx = self._maize_context({
            "Csc": {
                "p25": 600.0, "p50": 900.0, "p75": 1200.0,
                "p10_extreme_tmin": 15.0, "p90_extreme_tmax": 40.0,
                "n_cell_days": 50_000,
            },
        })
        result = self.ClimateEnvelopeValidator().validate(ctx)
        per_zone = result.metadata["per_zone_verdicts"]
        self.assertEqual(
            per_zone["Csc"]["precip"], "skipped_insufficient_sample",
        )

    def test_emits_discipline_only_emit_categories_in_emits(self):
        # F25-shape walker discipline: every issue's category
        # MUST be a member of EMITS. Test exercises multiple
        # zones to ensure no out-of-EMITS slip.
        ctx = self._maize_context({
            "BSh_marginal": {
                "p25": 350.0, "p50": 500.0, "p75": 800.0,
                "p10_extreme_tmin": 5.0, "p90_extreme_tmax": 50.0,
                "n_cell_days": 2_000_000,
            },
            "BSh_compatible": {
                "p25": 600.0, "p50": 900.0, "p75": 1200.0,
                "p10_extreme_tmin": 15.0, "p90_extreme_tmax": 40.0,
                "n_cell_days": 2_000_000,
            },
            "Csc_insufficient": {
                "p25": 600.0, "p50": 900.0, "p75": 1200.0,
                "p10_extreme_tmin": 15.0, "p90_extreme_tmax": 40.0,
                "n_cell_days": 100,
            },
        })
        result = self.ClimateEnvelopeValidator().validate(ctx)
        emits_values = {
            c.value for c in self.ClimateEnvelopeValidator.EMITS
        }
        for issue in result.issues:
            with self.subTest(category=issue.category):
                self.assertIn(issue.category, emits_values)


# ── Sprint F G-c-4 reason-helper unit tests ──────────────────────────


class TestPrecipVerdictReason(unittest.TestCase):
    """Sprint F AC-F-2: ``precip_verdict_reason`` returns a
    short data-only reason string for INCOMPATIBLE precip
    verdicts and ``None`` otherwise.

    Anti-mutation drill (per AC-F-2): change the helper to skip
    the percentile crossing report → cockpit drawer would show
    only "incompatible" with no number → caller's ECOCROP-URL-
    aware composer has nothing to anchor → audit grep test
    fails."""

    def setUp(self):
        from prismpy.validators.climate_envelope import (
            CompatibilityVerdict,
            precip_verdict_reason,
        )
        self._reason = precip_verdict_reason
        self._Verdict = CompatibilityVerdict

    def test_incompatible_low_side_reports_below_rmin(self):
        # rice (RMIN=1000) × Sahel BSh (P50=280) — the canonical
        # Sprint F G-c-4 inverse case.
        out = self._reason(
            self._Verdict.INCOMPATIBLE,
            p25=200, p50=280, p75=350,
            rmin=1000, rmax=4000,
        )
        self.assertEqual(
            out, "P50 = 280mm/yr below RMIN = 1000mm/yr",
        )

    def test_incompatible_high_side_reports_above_rmax(self):
        # Synthetic high-side branch (no Sahel-zone v1 crop hits
        # this case; sorghum RMAX=700 is the closest).
        out = self._reason(
            self._Verdict.INCOMPATIBLE,
            p25=2000, p50=4500, p75=5200,
            rmin=400, rmax=4000,
        )
        self.assertEqual(
            out, "P50 = 4500mm/yr above RMAX = 4000mm/yr",
        )

    def test_compatible_returns_none(self):
        out = self._reason(
            self._Verdict.COMPATIBLE,
            p25=600, p50=900, p75=1200,
            rmin=400, rmax=1800,
        )
        self.assertIsNone(out)

    def test_marginal_heterogeneous_returns_none(self):
        # MARGINAL_* verdicts NEVER drive a Sprint F emission;
        # ClimateEnvelopeValidator's Bucket 2 INFO already
        # surfaces them. Reason helper returns None so the
        # caller cannot accidentally compose a noisy banner.
        out = self._reason(
            self._Verdict.MARGINAL_HETEROGENEOUS,
            p25=350, p50=500, p75=800,
            rmin=400, rmax=1800,
        )
        self.assertIsNone(out)

    def test_marginal_thermal_seasonal_returns_none(self):
        out = self._reason(
            self._Verdict.MARGINAL_THERMAL_SEASONAL,
            p25=400, p50=600, p75=900,
            rmin=400, rmax=1800,
        )
        self.assertIsNone(out)

    def test_reason_data_only_no_crop_no_url_no_zone(self):
        # AC-F-2 contract: helper omits crop name + zone label +
        # ECOCROP URL. Caller composes those at the wizard banner
        # layer per ≤120-char budget. Pin guards against
        # substrate drift that would re-introduce crop / zone /
        # url into the helper signature.
        out = self._reason(
            self._Verdict.INCOMPATIBLE,
            p25=200, p50=280, p75=350,
            rmin=1000, rmax=4000,
        )
        self.assertNotIn("rice", out.lower())
        self.assertNotIn("BSh", out)
        self.assertNotIn("http", out.lower())
        self.assertNotIn("FAO", out)
        self.assertNotIn("ECOCROP", out)


class TestThermalVerdictReason(unittest.TestCase):
    """Sprint F AC-F-2: ``thermal_verdict_reason`` returns a
    short data-only reason string for INCOMPATIBLE thermal
    verdicts and ``None`` for compatible / marginal_thermal_
    seasonal."""

    def setUp(self):
        from prismpy.validators.climate_envelope import (
            CompatibilityVerdict,
            thermal_verdict_reason,
        )
        self._reason = thermal_verdict_reason
        self._Verdict = CompatibilityVerdict

    def test_incompatible_cold_kill_reports_p10_below_tmin(self):
        # Synthetic: maize TMIN=10 vs zone P10 extreme tmin=-5
        out = self._reason(
            self._Verdict.INCOMPATIBLE,
            zone_p10_extreme_tmin=-5,
            zone_p90_extreme_tmax=40,
            crop_tmin=10, crop_tmax=47,
        )
        self.assertEqual(
            out,
            "P10 extreme tmin = -5°C below crop TMIN = 10°C",
        )

    def test_incompatible_heat_kill_reports_p90_above_tmax(self):
        # Synthetic: rice TMAX=36 vs zone P90 extreme tmax=48
        out = self._reason(
            self._Verdict.INCOMPATIBLE,
            zone_p10_extreme_tmin=12,
            zone_p90_extreme_tmax=48,
            crop_tmin=10, crop_tmax=36,
        )
        self.assertEqual(
            out,
            "P90 extreme tmax = 48°C above crop TMAX = 36°C",
        )

    def test_compatible_returns_none(self):
        out = self._reason(
            self._Verdict.COMPATIBLE,
            zone_p10_extreme_tmin=15,
            zone_p90_extreme_tmax=40,
            crop_tmin=10, crop_tmax=47,
        )
        self.assertIsNone(out)

    def test_marginal_thermal_seasonal_returns_none(self):
        # Both cold-kill AND heat-kill route to
        # MARGINAL_THERMAL_SEASONAL; reason helper returns None
        # because Sprint F does not emit on marginal verdicts.
        out = self._reason(
            self._Verdict.MARGINAL_THERMAL_SEASONAL,
            zone_p10_extreme_tmin=-5,
            zone_p90_extreme_tmax=48,
            crop_tmin=10, crop_tmax=36,
        )
        self.assertIsNone(out)

    def test_reason_under_80_chars_for_caller_budget(self):
        # Banner ≤120-char total budget per AC-F-2; the data-
        # only reason this helper returns must stay well under
        # that to leave room for the caller's "{Crop} requires
        # T per FAO ECOCROP ({url}); {zone} ..." wrapper.
        out = self._reason(
            self._Verdict.INCOMPATIBLE,
            zone_p10_extreme_tmin=-15,
            zone_p90_extreme_tmax=48,
            crop_tmin=10, crop_tmax=36,
        )
        self.assertIsNotNone(out)
        self.assertLessEqual(len(out), 80)


class TestPrecipVerdictExplanation(unittest.TestCase):
    """Sprint F (post-Gate-B ux-expert) — plain-language sibling
    to ``precip_verdict_reason``.

    Anti-mutation drills:

    * Drop the helper or revert to the technical reason → wizard
      banner shows only "P50 = 400mm/yr below RMIN = 1000mm/yr"
      (opaque to Aminata/Moussa/Ibrahim) → Persona-readable
      copy contract broken.
    * Hard-code the explanation copy → substrate ratchet that
      bumps RMIN values would not propagate to the user-facing
      copy → data + UI drift.
    """

    def setUp(self):
        from prismpy.validators.climate_envelope import (
            CompatibilityVerdict,
            precip_verdict_explanation,
        )
        self._explain = precip_verdict_explanation
        self._Verdict = CompatibilityVerdict

    def test_too_dry_explains_water_need_and_zone_average(self):
        # Rice × Sahel BSh — the canonical Sprint F probe.
        out = self._explain(
            self._Verdict.INCOMPATIBLE,
            200, 400, 550,
            1000, 4000,
            "Rice",
            zone_label="Hot semi-arid",
        )
        self.assertIsNotNone(out)
        # Names the crop's water need.
        self.assertIn("1000mm", out)
        # Names the zone's realized precip.
        self.assertIn("400mm", out)
        # Names the zone label.
        self.assertIn("Hot semi-arid", out)
        # Names the crop.
        self.assertIn("Rice", out)
        # Plain-language phrasing — "too dry" vocabulary.
        self.assertIn("too dry", out.lower())

    def test_too_wet_explains_tolerance_and_waterlogging(self):
        # Synthetic high-side branch.
        out = self._explain(
            self._Verdict.INCOMPATIBLE,
            2000, 4500, 5200,
            400, 4000,
            "Maize",
            zone_label="Tropical rainforest",
        )
        self.assertIsNotNone(out)
        self.assertIn("4000mm", out)
        self.assertIn("4500mm", out)
        # Plain-language phrasing — "waterlogging" vocabulary.
        self.assertIn("waterlogging", out.lower())

    def test_compatible_returns_none(self):
        out = self._explain(
            self._Verdict.COMPATIBLE,
            600, 900, 1200,
            400, 1800,
            "Maize",
        )
        self.assertIsNone(out)

    def test_marginal_returns_none(self):
        out = self._explain(
            self._Verdict.MARGINAL_HETEROGENEOUS,
            350, 500, 800,
            400, 1800,
            "Sorghum",
        )
        self.assertIsNone(out)

    def test_falls_back_to_your_region_without_zone_label(self):
        # Codex review #DIM-2 — the prior copy template said
        # "The this region climate zone in your region" with
        # the None fallback. The fixed template uses "Your
        # region averages around..." so the grammar is clean
        # whether or not a zone label is provided.
        out = self._explain(
            self._Verdict.INCOMPATIBLE,
            200, 400, 550,
            1000, 4000,
            "Rice",
        )
        self.assertIsNotNone(out)
        self.assertIn("Your region", out)
        # Must NOT carry the broken double-mention.
        self.assertNotIn("this region climate", out)
        self.assertNotIn("the climate zone", out)

    def test_falls_back_to_your_region_with_empty_zone_label(self):
        # Empty string should fall back the same way None does.
        out = self._explain(
            self._Verdict.INCOMPATIBLE,
            200, 400, 550,
            1000, 4000,
            "Rice",
            zone_label="",
        )
        self.assertIn("Your region", out)

    def test_rejects_empty_crop_name(self):
        # Codex review #DIM-6 — empty crop_name is fail-loud
        # to surface the programmer error rather than rendering
        # "averages around 400mm/year — too dry for  without
        # irrigation." with the empty crop substituted.
        with self.assertRaises(ValueError):
            self._explain(
                self._Verdict.INCOMPATIBLE,
                200, 400, 550,
                1000, 4000,
                "",
                zone_label="BSh",
            )

    def test_crop_name_is_positional(self):
        # Codex review #DIM-3 — crop_name is positional after
        # the substrate args (matches the reason helper
        # signature pattern); zone_label remains keyword-only.
        out = self._explain(
            self._Verdict.INCOMPATIBLE,
            200, 400, 550,
            1000, 4000,
            "Rice",
        )
        self.assertIsNotNone(out)


class TestThermalVerdictExplanation(unittest.TestCase):
    """Sprint F (post-Gate-B ux-expert) — plain-language sibling
    to ``thermal_verdict_reason``."""

    def setUp(self):
        from prismpy.validators.climate_envelope import (
            CompatibilityVerdict,
            thermal_verdict_explanation,
        )
        self._explain = thermal_verdict_explanation
        self._Verdict = CompatibilityVerdict

    def test_heat_kill_explains_tolerance_and_yields(self):
        # Maize × Hot desert — heat-kill scenario.
        out = self._explain(
            self._Verdict.INCOMPATIBLE,
            15, 49,
            10, 47,
            "Maize",
            zone_label="Hot desert",
        )
        self.assertIsNotNone(out)
        # Names the crop's heat tolerance.
        self.assertIn("47°C", out)
        # Names the zone's peak temperature.
        self.assertIn("49°C", out)
        # Plain-language phrasing — "Heat stress" vocabulary.
        self.assertIn("Heat stress", out)
        # Codex review #DIM-2 — copy uses "yields" rather than
        # "grain-fill" because Sprint F's envelope spans
        # non-grain crops (cowpea, groundnut).
        self.assertIn("yields", out)
        self.assertNotIn("grain-fill", out)
        # Names the crop + zone.
        self.assertIn("Maize", out)
        self.assertIn("Hot desert", out)

    def test_cold_kill_explains_minimum_and_cold_damage(self):
        # Rice × Subarctic — cold-kill scenario.
        out = self._explain(
            self._Verdict.INCOMPATIBLE,
            -5, 30,
            10, 36,
            "Rice",
            zone_label="Subarctic",
        )
        self.assertIsNotNone(out)
        self.assertIn("10°C", out)
        self.assertIn("-5°C", out)
        # Plain-language phrasing — "Cold damage" vocabulary.
        self.assertIn("Cold damage", out)
        self.assertIn("Rice", out)

    def test_compatible_returns_none(self):
        out = self._explain(
            self._Verdict.COMPATIBLE,
            15, 40,
            10, 47,
            "Maize",
        )
        self.assertIsNone(out)

    def test_marginal_thermal_seasonal_returns_none(self):
        # Both cold-kill AND heat-kill route to MARGINAL_THERMAL_*
        # → no explanation (already covered by Bucket 2 INFO).
        out = self._explain(
            self._Verdict.MARGINAL_THERMAL_SEASONAL,
            -5, 48,
            10, 36,
            "Rice",
        )
        self.assertIsNone(out)

    def test_falls_back_to_your_region_without_zone_label(self):
        # Codex review #DIM-2 fix — the None fallback now
        # produces "your region" without a broken
        # "the this region climate zone" double-mention.
        out = self._explain(
            self._Verdict.INCOMPATIBLE,
            15, 49,
            10, 47,
            "Maize",
        )
        self.assertIn("your region", out)
        self.assertNotIn("this region climate", out)

    def test_rejects_empty_crop_name(self):
        with self.assertRaises(ValueError):
            self._explain(
                self._Verdict.INCOMPATIBLE,
                15, 49,
                10, 47,
                "",
                zone_label="BSh",
            )


if __name__ == "__main__":
    unittest.main()
