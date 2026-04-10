"""
Unit tests for TranslationPipeline._ensure_isda_1km_cache (V2-19 B3 v2).

Tests the 3-tier iSDA cache cascade:
  Tier 1: existing cache file → return path without touching rasterio
  Tier 2: download via COG overview pyramid + atomic tmp→rename write
  Tier 3: fallback to None on any exception (caller reads 30m directly)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from prismpy.pipeline.executor import TranslationPipeline


def _make_pipeline_stub() -> TranslationPipeline:
    """Build a minimal TranslationPipeline instance for cache-helper tests.

    The helper only reads ``self.logger``, so we bypass __init__ entirely
    and attach a mock logger. This keeps tests fast and isolated from
    config/provenance/runtime wiring.
    """
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.logger = MagicMock()
    return pipeline


class TestISDACacheCascade:
    """Test the 3-tier iSDA cache cascade (_ensure_isda_1km_cache)."""

    def test_tier_1_hit_returns_existing_cache_without_touching_rasterio(
        self, tmp_path: Path
    ):
        """Tier 1: pre-existing non-empty cache file → return path, no S3 call."""
        pipeline = _make_pipeline_stub()
        target_dir = tmp_path / "isda"
        target_dir.mkdir()
        # Pre-create a non-empty cache file
        prop_name = "sand_content"
        cache_file = target_dir / f"{prop_name}_1km.tif"
        cache_file.write_bytes(b"fake_tif_contents")

        with patch("rasterio.open") as mock_open:
            result = pipeline._ensure_isda_1km_cache(prop_name, target_dir)

        assert result == cache_file
        assert result.exists()
        mock_open.assert_not_called()

    def test_tier_1_empty_file_treated_as_miss(self, tmp_path: Path):
        """Tier 1: zero-byte cache file should NOT be treated as a hit.

        This is the root-cause-fix path: a previous crashed download may
        leave an empty file that would poison subsequent runs.
        """
        pipeline = _make_pipeline_stub()
        target_dir = tmp_path / "isda"
        target_dir.mkdir()
        prop_name = "sand_content"
        cache_file = target_dir / f"{prop_name}_1km.tif"
        cache_file.touch()  # zero bytes

        # Tier 2 should kick in — mock rasterio.open to raise so it falls
        # through to Tier 3 fallback (None), proving Tier 1 didn't accept
        # the empty file.
        with patch("rasterio.open", side_effect=OSError("boom")):
            result = pipeline._ensure_isda_1km_cache(prop_name, target_dir)

        assert result is None

    def test_tier_2_success_writes_cache_atomically(self, tmp_path: Path):
        """Tier 2: mock rasterio, verify tmp→rename produces a real cache file."""
        pipeline = _make_pipeline_stub()
        target_dir = tmp_path / "isda"
        prop_name = "clay_content"
        cache_file = target_dir / f"{prop_name}_1km.tif"

        # Build a fake source COG (EPSG:3857, 30m native, 1000×1000 size).
        # Need source big enough that out_height/out_width pass the >=10 guardrail
        # (scale = 30/1000 = 0.03, 1000*0.03 = 30 ≥ 10 ✓).
        fake_src = MagicMock()
        fake_src.crs = MagicMock()
        fake_src.crs.is_geographic = False  # projected (meters)
        fake_src.transform = MagicMock()
        fake_src.transform.a = 30.0  # native 30m resolution
        fake_src.transform.scale = MagicMock(return_value=MagicMock())
        fake_src.height = 1000
        fake_src.width = 1000
        fake_src.count = 4
        fake_src.dtypes = ["uint8"]
        fake_src.nodata = 0
        fake_src.read.return_value = np.zeros((4, 30, 30), dtype=np.uint8)

        # Track write destinations — the first rasterio.open call is the
        # read (S3 URL), the second is the write (tmp file).
        write_calls = []

        class FakeDst:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def write(self_inner, data):
                # Emulate writing to disk so tmp_file has non-zero size
                # before the rename happens.
                pass

        def fake_open(target, mode="r", **kwargs):
            if mode == "r":
                cm = MagicMock()
                cm.__enter__ = MagicMock(return_value=fake_src)
                cm.__exit__ = MagicMock(return_value=False)
                return cm
            write_calls.append((target, kwargs))
            # Write a few bytes to the tmp path so rename finds a real file.
            Path(target).write_bytes(b"fake_cog_bytes_xxxxx")
            return FakeDst()

        with patch("rasterio.open", side_effect=fake_open):
            result = pipeline._ensure_isda_1km_cache(prop_name, target_dir)

        assert result == cache_file, f"expected {cache_file}, got {result}"
        assert cache_file.exists(), "cache file should exist after rename"
        assert cache_file.stat().st_size > 0, "cache file must be non-empty"

        # Atomic write verification: per-PID tmp file must be gone after replace.
        import os
        tmp_file = cache_file.with_suffix(f".tif.tmp.{os.getpid()}")
        assert not tmp_file.exists(), "per-PID tmp file must be removed after replace"

        # One read call (S3 URL) + one write call (tmp file).
        assert len(write_calls) == 1
        assert ".tif.tmp." in str(write_calls[0][0]), (
            f"write target must be a per-PID tmp file, got {write_calls[0][0]}"
        )

    def test_tier_2_failure_falls_back_to_none_and_cleans_tmp(
        self, tmp_path: Path
    ):
        """Tier 2 failure → return None, no cache file, no leftover tmp file."""
        pipeline = _make_pipeline_stub()
        target_dir = tmp_path / "isda"
        prop_name = "ph"
        cache_file = target_dir / f"{prop_name}_1km.tif"
        tmp_file = cache_file.with_suffix(".tif.tmp")

        # rasterio.open raises on the read call.
        try:
            from rasterio.errors import RasterioIOError
            err = RasterioIOError("S3 unreachable")
        except ImportError:
            err = OSError("S3 unreachable")

        with patch("rasterio.open", side_effect=err):
            result = pipeline._ensure_isda_1km_cache(prop_name, target_dir)

        assert result is None
        assert not cache_file.exists(), "no cache file on failure"
        assert not tmp_file.exists(), "tmp file must be cleaned up"

    def test_tier_2_invalid_native_resolution_aborts(self, tmp_path: Path):
        """Guardrail: native_res <= 0 aborts rather than producing a bad scale."""
        pipeline = _make_pipeline_stub()
        target_dir = tmp_path / "isda"
        prop_name = "silt_content"

        fake_src = MagicMock()
        fake_src.crs = MagicMock()
        fake_src.crs.is_geographic = False
        fake_src.transform = MagicMock()
        fake_src.transform.a = 0.0  # invalid
        fake_src.height = 100
        fake_src.width = 100
        fake_src.count = 4
        fake_src.dtypes = ["uint8"]
        fake_src.nodata = 0

        def fake_open(target, mode="r", **kwargs):
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=fake_src)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        with patch("rasterio.open", side_effect=fake_open):
            result = pipeline._ensure_isda_1km_cache(prop_name, target_dir)

        assert result is None
        cache_file = target_dir / f"{prop_name}_1km.tif"
        assert not cache_file.exists()

    def test_tier_2_too_small_output_aborts(self, tmp_path: Path):
        """Guardrail: computed out_height/out_width < 10 aborts."""
        pipeline = _make_pipeline_stub()
        target_dir = tmp_path / "isda"
        prop_name = "bulk_density"

        # Source is tiny, so scale produces out_height/out_width < 10.
        fake_src = MagicMock()
        fake_src.crs = MagicMock()
        fake_src.crs.is_geographic = False
        fake_src.transform = MagicMock()
        fake_src.transform.a = 30.0
        fake_src.height = 5  # too small
        fake_src.width = 5
        fake_src.count = 4
        fake_src.dtypes = ["uint8"]
        fake_src.nodata = 0

        def fake_open(target, mode="r", **kwargs):
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=fake_src)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        with patch("rasterio.open", side_effect=fake_open):
            result = pipeline._ensure_isda_1km_cache(prop_name, target_dir)

        assert result is None

    def test_tier_2_geographic_crs_uses_degree_resolution(self, tmp_path: Path):
        """CRS-aware: geographic CRS selects 1/111 deg target resolution."""
        pipeline = _make_pipeline_stub()
        target_dir = tmp_path / "isda"
        prop_name = "carbon_organic"

        fake_src = MagicMock()
        fake_src.crs = MagicMock()
        fake_src.crs.is_geographic = True  # degrees
        fake_src.transform = MagicMock()
        fake_src.transform.a = 0.0003  # ~30m at equator in degrees
        fake_src.transform.__mul__ = lambda self_, other: MagicMock()
        fake_src.transform.scale = MagicMock(return_value=MagicMock())
        fake_src.height = 1000
        fake_src.width = 1000
        fake_src.count = 4
        fake_src.dtypes = ["uint8"]
        fake_src.nodata = 0
        # Capture the out_shape to verify scaling used degree-based target.
        captured = {}

        def fake_read(**kwargs):
            captured.update(kwargs)
            return np.zeros((4, 100, 100), dtype=np.uint8)

        fake_src.read.side_effect = fake_read

        class FakeDst:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def write(self_inner, data):
                pass

        def fake_open(target, mode="r", **kwargs):
            if mode == "r":
                cm = MagicMock()
                cm.__enter__ = MagicMock(return_value=fake_src)
                cm.__exit__ = MagicMock(return_value=False)
                return cm
            Path(target).write_bytes(b"fake_cog_bytes")
            return FakeDst()

        with patch("rasterio.open", side_effect=fake_open):
            result = pipeline._ensure_isda_1km_cache(prop_name, target_dir)

        # Confirm Tier 2 succeeded and we called read with a scaled out_shape.
        # scale = native_res / target_res = 0.0003 / (1/111) ≈ 0.0333
        # out_height = max(1, int(1000 * 0.0333)) = 33
        assert result is not None
        assert "out_shape" in captured
        assert captured["out_shape"][1] == 33
        assert captured["out_shape"][2] == 33
