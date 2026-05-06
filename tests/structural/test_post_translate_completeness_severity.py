"""Pin the SARRA-Py partial-climate validator severity at ``fail``.

The post-translate validator
``_validate_sarra_py_geotiffs.post_translate_completeness_sarra_py``
emits a check with ``result == "fail"`` whenever fewer than the
four SARRA-Py climate variables (rainfall + tmean / tmax / tmin +
solar radiation) are present in the translator output. The
``fail`` severity drives the cert--fail honest-signal banner on
the /results/ page so a researcher does not download an unrunnable
package; reverting the severity to ``warning`` lets the cert--
success path render despite SARRA-Py being unable to consume the
package, which is the F-AJ symptom.

The test instantiates a mock platform output tree under a
``tempfile.TemporaryDirectory`` with only the rainfall subdir
populated (the 1/4-variables case the user surfaced empirically),
runs the validator's geotiff path, and asserts the
``post_translate_completeness_sarra_py`` check fires with
``result == "fail"``.

Anti-mutation drill: revert ``post_translate.py:776`` from
``"fail"`` back to ``"warning"`` → this test fails with the
expected severity diff.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prismpy.validators.post_translate import _validate_sarra_py_geotiffs
from prismpy.validators.scientific import _get_check_category


def _write_minimal_tif(path: Path) -> None:
    """Create a tiny 1×1 GeoTIFF the validator can sample. Uses
    rasterio if available; falls back to writing zero bytes
    (validator handles missing/invalid files gracefully via the
    sampling path that returns no data without crashing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        with rasterio.open(
            str(path), "w",
            driver="GTiff", height=1, width=1, count=1,
            dtype="float32",
            transform=from_origin(0, 1, 1, 1), crs="EPSG:4326",
        ) as dst:
            dst.write(np.array([[1.0]], dtype="float32"), 1)
    except Exception:
        # If rasterio is not available, write an empty placeholder.
        # The validator's sampler will skip the file gracefully and
        # still emit the partial-climate check based on the subdir
        # being present but unsampled — same severity contract.
        path.write_bytes(b"")


class TestPostTranslateCompletenessSeverity(unittest.TestCase):
    """SARRA-Py post-translate completeness must report ``fail``
    severity when fewer than the four expected climate variables
    are present. ``warning`` severity would let the cert--success
    path render on a runtime-unusable package."""

    def test_partial_climate_emits_fail_severity(self):
        """One of four variables present (rainfall only) reproduces
        the F-AJ scenario: the validator should report fail, not
        warning, so ``scientific.overall_result`` rolls up to fail
        and the cert--fail banner renders."""
        with tempfile.TemporaryDirectory() as td:
            platform_dir = Path(td)
            # Populate only the rainfall subdir per SARRA_PY_VAR_MAPPING.
            # The other three subdirs (2m_temperature_*, ET0Hargreaves,
            # solar_radiation_flux_daily) are deliberately absent so
            # the validator counts 1/4 variables.
            climate = platform_dir / "data" / "climate" / "rainfall"
            _write_minimal_tif(climate / "rfe_2020_06_01.tif")

            checks = _validate_sarra_py_geotiffs(platform_dir)

        completeness = next(
            (c for c in checks
             if c.get("check") == "post_translate_completeness_sarra_py"),
            None,
        )
        self.assertIsNotNone(
            completeness,
            "validator must emit "
            "post_translate_completeness_sarra_py when fewer than "
            "the expected SARRA-Py climate variables are present "
            "in the output tree",
        )
        self.assertEqual(
            completeness["result"], "fail",
            "Partial-climate completeness check must report fail "
            "severity (not warning) so the /results/ cert renders "
            "the cert--fail banner instead of cert--success. "
            f"got result={completeness['result']!r} for found="
            f"{completeness['details'].get('found_variables')}/"
            f"{completeness['details'].get('expected_variables')}",
        )
        self.assertEqual(
            completeness["details"].get("found_variables"), 1,
            "Test fixture populated only the rainfall subdir; the "
            "validator should report 1 found variable.",
        )
        self.assertEqual(
            completeness["details"].get("expected_variables"), 4,
            "SARRA-Py expects four climate variables; the validator "
            "should report expected=4.",
        )

    def test_partial_climate_routes_to_completeness_category(self):
        """The bumped-severity check must route to the
        ``"completeness"`` category (not the default ``"schema"``)
        so the cockpit's category rollup attributes the failure to
        the chip the researcher associates with "missing data".

        Without the ``"post_translate_completeness_"`` prefix in
        ``_CATEGORY_MAP``, ``_get_check_category`` falls through to
        the ``"schema"`` default and a missing-climate-variables
        failure surfaces on the Schema Conformance chip — the wrong
        cockpit affordance. Codex round 1 caught this; pin the
        routing alongside the severity so a future contributor who
        retires the prefix mapping fires this assertion.
        """
        category = _get_check_category(
            "post_translate_completeness_sarra_py",
        )
        self.assertEqual(
            category, "completeness",
            "post_translate_completeness_sarra_py must route to the "
            "'completeness' category, not the 'schema' fallback. "
            "Add 'post_translate_completeness_' to _CATEGORY_MAP "
            "in scientific.py to fix.",
        )


if __name__ == "__main__":
    unittest.main()
