"""
test_data_fetch.py — Phase 14.5 unit tests for data pipeline audit
====================================================================
Tests the pure/deterministic functions added in Phase 14.5:
  - _bbox_key: legacy cache key format
  - _tile_key: quantised tile key stability
  - _cache_fingerprint: content-addressed hashing
  - RasterCache: get/put cycle
  - DEM void-fill: no sentinel cells remain after EDT fill
"""
import os
import sys
import tempfile
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from data_fetch import _bbox_key, _tile_key, _cache_fingerprint, RasterCache


# ── 1. Legacy _bbox_key format ───────────────────────────────────────────────

def test_bbox_key_format():
    """Verify _bbox_key output format and negative coord handling."""
    key = _bbox_key((94.2175, 17.1174, 96.8304, 17.5483))
    assert key.startswith("W"), f"Key should start with W: {key}"
    assert "m" not in key or key.count("m") == 0, f"No negatives, no 'm': {key}"
    assert "." in key, f"Key should contain decimals: {key}"

    # With negative coords (western hemisphere)
    key_neg = _bbox_key((-105.5, -33.12, -104.0, -32.0))
    assert "m" in key_neg, f"Negative coords should use 'm': {key_neg}"
    assert "-" not in key_neg, f"No raw hyphens in key: {key_neg}"


def test_bbox_key_rounding():
    """Two nearly identical bboxes should produce same key with 4dp rounding."""
    k1 = _bbox_key((94.21749999, 17.11740001, 96.83039999, 17.54830001))
    k2 = _bbox_key((94.2175, 17.1174, 96.8304, 17.5483))
    assert k1 == k2, f"Keys should match after rounding: {k1} vs {k2}"


# ── 2. Quantised _tile_key stability ─────────────────────────────────────────

def test_tile_key_stability():
    """Two slightly different bboxes within the same 0.5° tile produce same key."""
    bbox_a = (94.21750, 17.11740, 96.83040, 17.54830)
    bbox_b = (94.21751, 17.11739, 96.83041, 17.54831)
    key_a = _tile_key(bbox_a, tile_size_deg=0.5)
    key_b = _tile_key(bbox_b, tile_size_deg=0.5)
    assert key_a == key_b, f"Same-tile bboxes should match: {key_a} vs {key_b}"


def test_tile_key_different_tiles():
    """Bboxes in different tiles produce different keys."""
    # One degree apart
    bbox_a = (94.0, 17.0, 94.4, 17.4)
    bbox_b = (95.0, 17.0, 95.4, 17.4)
    key_a = _tile_key(bbox_a, tile_size_deg=0.5)
    key_b = _tile_key(bbox_b, tile_size_deg=0.5)
    assert key_a != key_b, f"Different-tile bboxes should differ: {key_a} vs {key_b}"


def test_tile_key_no_special_chars():
    """Tile key should not contain dots, hyphens, or spaces."""
    bbox = (-105.5, -33.12, -104.0, -32.0)
    key = _tile_key(bbox, tile_size_deg=0.5)
    assert "." not in key, f"No dots in tile key: {key}"
    assert "-" not in key, f"No hyphens in tile key: {key}"
    assert " " not in key, f"No spaces in tile key: {key}"


# ── 3. Content-addressed fingerprint ─────────────────────────────────────────

def test_fingerprint_deterministic():
    """Same inputs always produce the same fingerprint."""
    fp1 = _cache_fingerprint(531016, (1737, 9416), 2000, 20, 30, 30)
    fp2 = _cache_fingerprint(531016, (1737, 9416), 2000, 20, 30, 30)
    assert fp1 == fp2, f"Fingerprints should match: {fp1} vs {fp2}"


def test_fingerprint_different_inputs():
    """Different inputs produce different fingerprints."""
    fp1 = _cache_fingerprint(531016, (1737, 9416), 2000, 20, 30, 30)
    fp2 = _cache_fingerprint(531016, (1737, 9416), 3000, 20, 30, 30)  # penalty changed
    assert fp1 != fp2, f"Fingerprints should differ: {fp1} vs {fp2}"


def test_fingerprint_length():
    """Fingerprint should be 12 hex characters."""
    fp = _cache_fingerprint("hello", 42, (100, 200))
    assert len(fp) == 12, f"Fingerprint should be 12 chars: {fp}"
    assert all(c in "0123456789abcdef" for c in fp), f"Should be hex: {fp}"


# ── 4. RasterCache get/put cycle ─────────────────────────────────────────────

def test_raster_cache_roundtrip():
    """Write a raster, read it back, verify values match."""
    from rasterio.transform import from_bounds

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RasterCache(cache_dir=tmpdir)
        arr = np.random.rand(100, 200).astype(np.float32)
        tf = from_bounds(0, 0, 200, 100, 200, 100)
        fp = "test12345678"

        cache.put("test_product", fp, arr, tf, crs_epsg=32646)
        loaded, hit = cache.get("test_product", fp, shape=(100, 200))

        assert hit, "Cache should hit"
        assert loaded is not None, "Loaded array should not be None"
        np.testing.assert_array_almost_equal(loaded, arr, decimal=5)


def test_raster_cache_miss():
    """Cache miss returns (None, False)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RasterCache(cache_dir=tmpdir)
        loaded, hit = cache.get("nonexistent", "deadbeef1234", shape=(10, 10))
        assert not hit
        assert loaded is None


def test_raster_cache_shape_mismatch():
    """Wrong shape causes a miss (forces recompute)."""
    from rasterio.transform import from_bounds

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = RasterCache(cache_dir=tmpdir)
        arr = np.ones((50, 60), dtype=np.float32)
        tf = from_bounds(0, 0, 60, 50, 60, 50)
        fp = "shapemismatch"

        cache.put("test_shape", fp, arr, tf, crs_epsg=32646)
        loaded, hit = cache.get("test_shape", fp, shape=(100, 200))

        assert not hit, "Shape mismatch should cause miss"
        assert loaded is None


# ── 5. DEM void-fill sentinel removal ────────────────────────────────────────

def test_void_fill_no_sentinel_remains():
    """
    Create a synthetic DEM with sentinel voids, apply the EDT fill logic
    from _reproject_to_utm, and assert no sentinels remain.
    """
    from scipy.ndimage import distance_transform_edt
    DEM_NODATA_SENTINEL = -9999.0

    # Synthetic DEM: mostly valid, with sentinel cells at edges
    dem = np.random.uniform(100.0, 500.0, (200, 300)).astype(np.float32)
    # Insert void cells
    dem[0, :] = DEM_NODATA_SENTINEL  # top row
    dem[:, 0] = DEM_NODATA_SENTINEL  # left col
    dem[100:110, 150:160] = DEM_NODATA_SENTINEL  # interior patch

    # Apply the same logic as _reproject_to_utm
    nodata_mask = (dem == DEM_NODATA_SENTINEL)
    nodata_mask |= (dem < -500)
    assert nodata_mask.any(), "Should have void cells"

    _, idx = distance_transform_edt(nodata_mask, return_indices=True)
    dem[nodata_mask] = dem[idx[0][nodata_mask], idx[1][nodata_mask]]

    # Post-fill: no sentinels should remain
    remaining = int(np.sum(dem == DEM_NODATA_SENTINEL))
    assert remaining == 0, f"Void-fill failed: {remaining} sentinel cells remain"

    # All values should be in valid range
    assert dem.min() >= 0, f"Min elevation should be valid: {dem.min()}"


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_bbox_key_format,
        test_bbox_key_rounding,
        test_tile_key_stability,
        test_tile_key_different_tiles,
        test_tile_key_no_special_chars,
        test_fingerprint_deterministic,
        test_fingerprint_different_inputs,
        test_fingerprint_length,
        test_raster_cache_roundtrip,
        test_raster_cache_miss,
        test_raster_cache_shape_mismatch,
        test_void_fill_no_sentinel_remains,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")
        except Exception as e:
            print(f"  ✗ {test.__name__}: UNEXPECTED {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
