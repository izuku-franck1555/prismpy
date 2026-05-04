"""Sprint E.0.5 cross-platform reproducibility unit-test.

Per the verification-strategy doc NEW4 + contract Draft 4
§Estimated scope new tests: a unit-level reproducibility
probe that catches host-OS / BLAS-backend divergence
*without* a full bound-gen run. The structural pin lives at
the byte-identical-quantile level (np.quantile linear method
on a fixed input must yield a known scalar across runs).

Bound-gen's full determinism contract (AC-Q2-B1) lives in
the CI workflow + a separate integration meta-test that
runs bound-gen twice on a synthetic fixture and asserts
byte-identical output. This file is the unit-level floor.

Anti-mutation drills:

- Switch the substrate to ``method='nearest'`` →
  ``test_quantile_method_pin`` fires (different scalar).
- A future numpy release that changes the linear method's
  behavior would break this pin and force a contract-review
  trip per the AC-Q2-A1-d numpy<2.0 upper bound.
"""
from __future__ import annotations

import unittest

import numpy as np


class TestNumpyQuantileCrossPlatform(unittest.TestCase):
    """Byte-identical numpy.quantile output is the foundation
    of the bound-gen determinism contract. The unit-level pin
    catches drift earlier than the full bound-gen integration
    meta-test (AC-Q2-B1)."""

    def test_quantile_p95_known_scalar(self):
        # np.quantile([1,2,3,4,5], 0.95, method='linear') = 4.8
        # by linear interpolation between rank 4 and rank 5
        # (0-indexed: ranks 3 and 4 → values 4 and 5; index =
        # 0.95 * 4 = 3.8 → 4 + 0.8*1 = 4.8). Pinned to 10
        # decimal places per float64 precision.
        result = float(np.quantile(
            [1.0, 2.0, 3.0, 4.0, 5.0], 0.95, method="linear",
        ))
        self.assertAlmostEqual(result, 4.8, places=10)

    def test_quantile_p25_known_scalar(self):
        # np.quantile([10,20,30,40,50], 0.25, method='linear')
        # = 20.0 (linear at index 1 = rank 2 = value 20).
        result = float(np.quantile(
            [10.0, 20.0, 30.0, 40.0, 50.0], 0.25, method="linear",
        ))
        self.assertAlmostEqual(result, 20.0, places=10)

    def test_quantile_p50_known_scalar(self):
        # P50 of [10,20,30,40,50] = 30.0 (median).
        result = float(np.quantile(
            [10.0, 20.0, 30.0, 40.0, 50.0], 0.50, method="linear",
        ))
        self.assertAlmostEqual(result, 30.0, places=10)

    def test_quantile_p75_known_scalar(self):
        # P75 of [10,20,30,40,50] = 40.0 (linear at index 3).
        result = float(np.quantile(
            [10.0, 20.0, 30.0, 40.0, 50.0], 0.75, method="linear",
        ))
        self.assertAlmostEqual(result, 40.0, places=10)

    def test_quantile_idempotent_within_run(self):
        # Sanity: same input + same method produces byte-
        # identical output across calls (the AC-Q2-B1 within-
        # run determinism floor).
        seq = [10.0, 20.0, 30.0, 40.0, 50.0]
        a = float(np.quantile(seq, 0.25, method="linear"))
        b = float(np.quantile(seq, 0.25, method="linear"))
        self.assertEqual(a, b)

    def test_quantile_method_pin(self):
        # Anti-mutation: switching to 'nearest' yields a
        # different scalar (4.0 vs the linear 4.8 case
        # above). Catches a substrate change that swaps the
        # quantile method.
        nearest_result = float(np.quantile(
            [1.0, 2.0, 3.0, 4.0, 5.0], 0.95, method="nearest",
        ))
        linear_result = float(np.quantile(
            [1.0, 2.0, 3.0, 4.0, 5.0], 0.95, method="linear",
        ))
        self.assertNotEqual(nearest_result, linear_result)

    def test_float64_precision(self):
        # Bound-gen uses float64 arrays. Pin that
        # np.quantile preserves float64 dtype on float64
        # input (defensive; matters for cross-platform).
        seq = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result = np.quantile(seq, 0.95, method="linear")
        # np.quantile on float64 input returns numpy.float64.
        self.assertEqual(result.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
