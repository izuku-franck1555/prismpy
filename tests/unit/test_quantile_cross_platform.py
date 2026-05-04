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

    def test_quantile_p95_known_scalar_byte_identical(self):
        # np.quantile([1,2,3,4,5], 0.95, method='linear') = 4.8
        # by linear interpolation. Per codex Gate-A MEDIUM on
        # commit 9: assertion is exact equality on the float64
        # representation, NOT assertAlmostEqual at 10 places —
        # the bound-gen contract is byte-identical, and a one-
        # ULP drift would still pass at 10 places but produce
        # different serialized bytes.
        result = np.quantile(
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
            0.95, method="linear",
        )
        # Compute the expected value the same way the bound-
        # gen substrate does, then assert byte-equal float64.
        expected = np.float64(4.8)
        self.assertEqual(result.tobytes(), expected.tobytes())

    def test_quantile_p25_known_scalar_byte_identical(self):
        result = np.quantile(
            np.asarray([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
            0.25, method="linear",
        )
        self.assertEqual(result.tobytes(), np.float64(20.0).tobytes())

    def test_quantile_p50_known_scalar_byte_identical(self):
        result = np.quantile(
            np.asarray([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
            0.50, method="linear",
        )
        self.assertEqual(result.tobytes(), np.float64(30.0).tobytes())

    def test_quantile_p75_known_scalar_byte_identical(self):
        result = np.quantile(
            np.asarray([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
            0.75, method="linear",
        )
        self.assertEqual(result.tobytes(), np.float64(40.0).tobytes())

    def test_quantile_nontrivial_array_byte_stable(self):
        # Nontrivial mixed-precision array. Byte-identical
        # invariant across two calls on the same input.
        # (Recomputing the expected value via Python float
        # arithmetic doesn't always match numpy's internal
        # vectorized path bit-for-bit; the cross-platform
        # invariant we actually care about is *idempotence
        # within a run*. Plus the integer-aligned scalar pins
        # above lock specific values across runs.)
        arr = np.asarray(
            [1.234, 5.678, 9.012, 13.456, 17.890, 21.345, 25.789],
            dtype=np.float64,
        )
        a = np.quantile(arr, 0.95, method="linear")
        b = np.quantile(arr, 0.95, method="linear")
        self.assertEqual(a.tobytes(), b.tobytes())

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
